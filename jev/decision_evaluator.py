"""Jev 决策质量评估闭环。

从 SQLite 的 ``jev_decisions`` 表读取历史决策，关联决策日之后 N 个交易日的
价格走势，判定每条决策（含被 Jev 否决的决策）的对错，并输出：

  - 总体准确率与六类决策计数（正确/错误 买入/卖出/观望）
  - 按 final_confidence 分桶的校准度统计（高置信桶应更准）
  - 按策略原始信号分组的 Jev 过滤效果（执行数/否决数/执行后正确率）
  - 最近若干条决策的逐笔评估结果

价格数据来源优先级：
  1. 构造时显式传入的 ``price_data``（{symbol: pd.DataFrame}，index 为
     ``pd.Timestamp``，含 ``close`` 列）——web-dashboard 上下文传入
     ``manager.sims`` 中各模拟器的 ``klines``。
  2. 未传入时，惰性调用 ``data.data_fetcher.DataFetcher`` 在线拉取。
  3. 在线拉取失败（无网络/SDK 不可用）时，该标的的决策不纳入评估，不报错。

注意：
  - 未来收益按**交易日**计算：决策日收盘价取决策当日（若非交易日则取最近的
    前一个交易日），N 日后取其后第 N 个交易日的收盘价。
  - 若决策日之后不足 N 个交易日，该决策不计入 ``evaluated_decisions``。
  - ``jev_decisions`` 表本身没有 ``account_id`` 列，该参数仅为接口兼容保留。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from persistence.database import Database

logger = logging.getLogger(__name__)

#: 判定为“正确”的标签集合
CORRECT_LABELS = {"correct_buy", "correct_sell", "correct_hold"}

#: 置信度分桶：左闭右开，最后一桶右端闭合 [0.8, 1.0]
_CONF_BUCKETS = [
    ("0-0.4", 0.0, 0.4),
    ("0.4-0.6", 0.4, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8-1.0", 0.8, 1.0 + 1e-9),
]


class DecisionEvaluator:
    """Jev 决策质量评估器。

    Args:
        db_path: SQLite 数据库路径，默认 ``data/quant_trading.db``。
        price_data: 可选的 {symbol: K线DataFrame} 映射。index 为
            ``pd.Timestamp``，需含 ``close`` 列。传入后优先使用，不再在线拉取。
        hold_threshold: hold/观望 判定阈值（绝对收益率），默认 0.02（2%）。
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
        hold_threshold: float = 0.02,
    ) -> None:
        if db_path is None:
            # 与 persistence.database 默认路径保持一致
            from pathlib import Path

            db_path = str(
                Path(__file__).resolve().parent.parent
                / "data"
                / "quant_trading.db"
            )
        self.db_path = db_path
        self._db: Optional[Database] = None
        self.price_data: Dict[str, pd.DataFrame] = dict(price_data or {})
        self.hold_threshold = float(hold_threshold)
        self._fetcher = None  # 惰性创建

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @property
    def db(self) -> Database:
        """惰性获取 Database 句柄。"""
        if self._db is None:
            self._db = Database(db_path=self.db_path)
        return self._db

    def _resolve_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取某标的的 K线 DataFrame（先查缓存，再尝试在线拉取）。"""
        if symbol in self.price_data:
            return self.price_data[symbol]

        # 容错查找：大小写/后缀不一致时尝试一次归一匹配
        for key, df in self.price_data.items():
            if key.upper() == symbol.upper():
                return df

        # 在线拉取（失败即返回 None，不抛异常）
        try:
            if self._fetcher is None:
                from data.data_fetcher import DataFetcher

                self._fetcher = DataFetcher()
            df = self._fetcher.get_klines(symbol, period="1d", count=500)
            if df is not None and not df.empty:
                self.price_data[symbol] = df
                return df
        except Exception as e:  # noqa: BLE001
            logger.warning("在线拉取 %s K线失败，该标的决策不纳入评估: %s", symbol, e)
        return None

    def _future_return(
        self, df: pd.DataFrame, ts: pd.Timestamp, days: int
    ) -> Optional[float]:
        """计算决策日之后第 ``days`` 个交易日的收益率。

        决策日非交易日时取最近的前一个交易日；不足后续数据返回 None。
        """
        if df is None or df.empty or "close" not in df.columns:
            return None
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(idx)

        # 决策日定位：side='right' - 1 → 命中当日则取当日，否则取前一交易日
        pos = idx.searchsorted(ts, side="right") - 1
        if pos < 0:
            return None
        future_pos = pos + days
        if future_pos >= len(df):
            return None

        close_now = float(df["close"].iloc[pos])
        close_future = float(df["close"].iloc[future_pos])
        if close_now <= 0:
            return None
        return close_future / close_now - 1.0

    def _classify(
        self, action: str, executed: bool, future_return: float
    ) -> str:
        """根据最终动作、是否执行、未来收益判定决策标签。

        规则（见模块 docstring）：
          - buy  执行 且上涨 → correct_buy；buy 执行 且下跌 → wrong_buy
          - sell 执行 且下跌 → correct_sell；sell 执行 且上涨 → wrong_sell
          - hold（含 Jev 否决后实际观望）：|收益| < 阈值 → correct_hold，否则 wrong_hold
          - 未执行的 buy/sell：按“观望”是否正确来给否决打分
        """
        if action == "buy":
            if executed:
                return "correct_buy" if future_return > 0 else "wrong_buy"
            # Jev 否决了买入：上涨=踏空(wrong_hold)，下跌=躲过(correct_hold)
            return "wrong_hold" if future_return > 0 else "correct_hold"

        if action == "sell":
            if executed:
                return "correct_sell" if future_return < 0 else "wrong_sell"
            # Jev 否决了卖出：下跌=没躲过(wrong_hold)，上涨=卖对没卖飞(correct_hold)
            return "wrong_hold" if future_return < 0 else "correct_hold"

        # hold：绝对收益小 → 正确观望
        return (
            "correct_hold" if abs(future_return) < self.hold_threshold else "wrong_hold"
        )

    @staticmethod
    def _conf_bucket(conf: float) -> str:
        """根据 final_confidence 命中分桶名（左闭右开）。"""
        for name, lo, hi in _CONF_BUCKETS:
            if lo <= conf < hi:
                return name
        # conf 恰好为 1.0 或越界，归入最后一桶
        return _CONF_BUCKETS[-1][0]

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def evaluate(
        self,
        days: int = 5,
        account_id: Optional[str] = None,
        limit: int = 1000,
    ) -> Dict[str, Any]:
        """执行评估，返回评估报告 dict。

        Args:
            days: 决策后观察交易日数，默认 5。
            account_id: 账户 ID（保留参数；jev_decisions 表无该列，暂不过滤）。
            limit: 最多读取最近多少条决策。

        Returns:
            评估报告字典，结构见模块 docstring / 任务说明。
        """
        empty: Dict[str, Any] = {
            "summary": {
                "total_decisions": 0,
                "evaluated_decisions": 0,
                "overall_accuracy": 0.0,
                "correct_buy": 0,
                "wrong_buy": 0,
                "correct_sell": 0,
                "wrong_sell": 0,
                "correct_hold": 0,
                "wrong_hold": 0,
            },
            "confidence_buckets": {
                name: {"count": 0, "correct": 0, "accuracy": 0.0}
                for name, _, _ in _CONF_BUCKETS
            },
            "by_strategy": {},
            "recent_decisions": [],
            "params": {
                "days": days,
                "eval_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        try:
            decisions = self.db.get_jev_decisions(limit=limit)
        except Exception as e:  # noqa: BLE001
            logger.exception("读取 jev_decisions 失败")
            empty["summary"]["error"] = str(e)
            return empty

        if not decisions:
            return empty

        counters = {
            "correct_buy": 0,
            "wrong_buy": 0,
            "correct_sell": 0,
            "wrong_sell": 0,
            "correct_hold": 0,
            "wrong_hold": 0,
        }
        buckets: Dict[str, Dict[str, int]] = {
            name: {"count": 0, "correct": 0} for name, _, _ in _CONF_BUCKETS
        }
        by_strategy: Dict[str, Dict[str, Any]] = {}
        evaluated_rows: List[Dict[str, Any]] = []
        evaluated_count = 0
        correct_count = 0

        for d in decisions:
            symbol = d.get("symbol", "")
            action = d.get("final_action", "hold")
            executed = bool(d.get("executed", False))
            conf = float(d.get("final_confidence", 0.0))
            strategy_signal = d.get("strategy_signal") or "unknown"

            # 按策略聚合（无论是否评估成功都计入 total/executed/vetoed）
            slot = by_strategy.setdefault(
                strategy_signal,
                {"total": 0, "executed": 0, "vetoed": 0,
                 "_exec_correct": 0, "_exec_eval": 0},
            )
            slot["total"] += 1
            if executed:
                slot["executed"] += 1
            else:
                slot["vetoed"] += 1

            # 关联未来收益
            df = self._resolve_df(symbol)
            try:
                ts = pd.Timestamp(d.get("timestamp"))
            except Exception:  # noqa: BLE001
                continue
            fut_ret = self._future_return(df, ts, days)
            if fut_ret is None:
                continue  # 数据不足，不计入 evaluated

            label = self._classify(action, executed, fut_ret)
            evaluated_count += 1
            is_correct = label in CORRECT_LABELS
            if is_correct:
                correct_count += 1
            counters[label] += 1

            # 置信度分桶
            b = self._conf_bucket(conf)
            buckets[b]["count"] += 1
            if is_correct:
                buckets[b]["correct"] += 1

            # 按策略：仅统计“被执行且评估成功”的正确率
            if executed:
                slot["_exec_eval"] += 1
                if is_correct:
                    slot["_exec_correct"] += 1

            evaluated_rows.append({
                "id": d.get("id"),
                "timestamp": d.get("timestamp"),
                "symbol": symbol,
                "action": action,
                "confidence": round(conf, 4),
                "future_return": round(fut_ret, 4),
                "correct": is_correct,
                "label": label,
                "reason": d.get("reason", ""),
            })

        # 组装置信度桶报告
        conf_report: Dict[str, Dict[str, Any]] = {}
        for name, _, _ in _CONF_BUCKETS:
            c = buckets[name]["count"]
            ok_n = buckets[name]["correct"]
            conf_report[name] = {
                "count": c,
                "correct": ok_n,
                "accuracy": round(ok_n / c, 4) if c > 0 else 0.0,
            }

        # 组装按策略报告
        strategy_report: Dict[str, Dict[str, Any]] = {}
        for sig, s in by_strategy.items():
            exec_eval = s["_exec_eval"]
            strategy_report[sig] = {
                "total": s["total"],
                "executed": s["executed"],
                "vetoed": s["vetoed"],
                "executed_accuracy": round(s["_exec_correct"] / exec_eval, 4)
                if exec_eval > 0 else 0.0,
            }

        # evaluated_rows 已按 id DESC（DB 返回倒序），取最近 10 条
        recent = [
            {
                "timestamp": r["timestamp"],
                "symbol": r["symbol"],
                "action": r["action"],
                "confidence": r["confidence"],
                "future_return": r["future_return"],
                "correct": r["correct"],
                "reason": r["reason"],
            }
            for r in evaluated_rows[:10]
        ]

        summary = {
            "total_decisions": len(decisions),
            "evaluated_decisions": evaluated_count,
            "overall_accuracy": round(correct_count / evaluated_count, 4)
            if evaluated_count > 0 else 0.0,
            **counters,
        }

        return {
            "summary": summary,
            "confidence_buckets": conf_report,
            "by_strategy": strategy_report,
            "recent_decisions": recent,
            "params": {
                "days": days,
                "eval_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
