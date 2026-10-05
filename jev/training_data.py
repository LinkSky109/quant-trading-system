"""Jev 决策训练数据导出器。

从 SQLite 的 ``jev_decisions`` 表读取历史决策，关联决策日之后 N 个交易日的
价格走势，生成带标签的训练样本（JSONL / CSV），用于 Jev 模型微调。

样本标签（最优动作）由后续走势决定，而非历史动作：
  - 后 N 日收益率 >  hold_threshold → 最优动作 ``buy``（此时买入能赚钱）
  - 后 N 日收益率 < -hold_threshold → 最优动作 ``sell``（此时卖出能避险）
  - |收益率| <= hold_threshold     → 最优动作 ``hold``（无明显波动）

特征取自决策时落库的 ``market_state_json``（扁平 dict），统一转换为与
``jev_engine.MarketState.to_states_list`` 一致的 ``[{feature, value}, ...]`` 格式。

价格数据来源优先级（与 DecisionEvaluator 一致）：
  1. 构造时显式传入的 ``price_data``（{symbol: pd.DataFrame}，含 close 列）。
  2. 未传入时惰性调用 ``data.data_fetcher.DataFetcher`` 在线拉取。
  3. 在线拉取失败（无网络/SDK 不可用）时，该标的的决策静默跳过，不抛异常。

注意：
  - 数据集**按时间顺序切分**（先升序排序再按比例切 train/val/test），
    严禁随机划分，避免未来数据泄露。
  - 决策日之后不足 N 个交易日的样本无法计算标签，直接丢弃。
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from persistence.database import Database

logger = logging.getLogger(__name__)

#: 特征顺序（与 jev_engine.MarketState 字段一致），CSV 展开列时按此顺序输出
FEATURE_NAMES = [
    "price",
    "price_change_5d",
    "ma5_ma20_ratio",
    "volume_ratio",
    "rsi",
    "macd_signal",
    "volatility_20d",
]

#: 后 N 日内跌幅超过该阈值即视为触发止损（用于辅助标签 stop_loss_triggered）
STOP_LOSS_THRESHOLD = 0.03


class TrainingDataExporter:
    """Jev 决策训练数据导出器。

    Args:
        db_path: SQLite 数据库路径，默认 ``data/quant_trading.db``。
        price_data: 可选的 {symbol: K线DataFrame} 映射。index 为
            ``pd.Timestamp``，需含 ``close`` 列。传入后优先使用，不再在线拉取。
        hold_threshold: hold/观望 判定阈值（绝对收益率），默认 0.02（2%）。
        forward_days: 决策后观察的交易日数（用于计算未来收益/标签），默认 5。
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
        hold_threshold: float = 0.02,
        forward_days: int = 5,
    ) -> None:
        if db_path is None:
            db_path = str(
                Path(__file__).resolve().parent.parent
                / "data"
                / "quant_trading.db"
            )
        self.db_path = db_path
        self._db: Optional[Database] = None
        self.price_data: Dict[str, pd.DataFrame] = dict(price_data or {})
        self.hold_threshold = float(hold_threshold)
        self.forward_days = int(forward_days)
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
        """获取某标的的 K线 DataFrame（先查缓存，再尝试在线拉取）。

        在线拉取失败时返回 None，由调用方静默跳过该标的。
        """
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
            logger.warning("在线拉取 %s K线失败，该标的决策不纳入导出: %s", symbol, e)
        return None

    def _future_window(
        self, df: pd.DataFrame, ts: pd.Timestamp, days: int
    ) -> Optional[Dict[str, float]]:
        """计算决策日之后第 ``days`` 个交易日的收益与窗口统计。

        复用 decision_evaluator.DecisionEvaluator._future_return 的定位逻辑：
        决策日用 ``searchsorted(side='right') - 1``（命中当日则取当日，否则取
        前一交易日），未来取其后第 N 个交易日。

        Returns:
            ``{"future_return": 收益率, "min_close_ratio": 窗口最低收盘相对决策日跌幅}``；
            数据不足或收盘价异常时返回 None。
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

        future_return = close_future / close_now - 1.0
        # 后 N 日窗口（不含决策日当日，含第 N 个交易日）的最低收盘价相对跌幅
        window = df["close"].iloc[pos + 1: future_pos + 1]
        min_close = float(window.min()) if len(window) else close_future
        min_close_ratio = min_close / close_now - 1.0
        return {"future_return": future_return, "min_close_ratio": min_close_ratio}

    @staticmethod
    def _extract_states(market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从落库的 market_state 中提取 states 特征数组。

        落库的 market_state 通常是 ``MarketState.to_dict()`` 得到的扁平 dict
        （{price, price_change_5d, ...}），这里统一转换为
        ``[{feature, value}, ...]`` 格式；若本身已是 states 列表则直接返回。
        """
        if not market_state:
            return []
        # 已是 states 列表格式（防御性兼容）
        states = market_state.get("states")
        if isinstance(states, list):
            return states
        # 扁平 dict → states 列表（按 FEATURE_NAMES 顺序）
        out: List[Dict[str, Any]] = []
        for feat in FEATURE_NAMES:
            if feat in market_state:
                out.append({"feature": feat, "value": market_state[feat]})
        return out

    def _label_from_return(self, future_return: float) -> str:
        """根据未来收益率判定最优动作标签。"""
        if future_return > self.hold_threshold:
            return "buy"
        if future_return < -self.hold_threshold:
            return "sell"
        return "hold"

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def load_decisions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """从 jev_decisions 表读取历史决策，自动反序列化 JSON 字段。

        Args:
            start_date: 起始时间（含），按 timestamp 文本过滤，如 "2024-01-01"。
            end_date: 结束时间（含）。
            symbol: 按标的代码精确过滤。
            strategy: 按策略原始信号（strategy_signal）精确过滤。
            limit: 最多读取条数，默认 10000。

        Returns:
            决策 dict 列表，按 timestamp 升序。market_state / probabilities
            已从 JSON 字符串反序列化为 dict，executed 已转为 bool。
        """
        conn = self.db._get_conn()  # noqa: SLF001（复用项目内连接，DB 未暴露该查询）
        query = "SELECT * FROM jev_decisions WHERE 1=1"
        params: list = []
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if strategy:
            query += " AND strategy_signal = ?"
            params.append(strategy)
        query += " ORDER BY timestamp ASC, id ASC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            try:
                d["market_state"] = json.loads(d.pop("market_state_json", "{}") or "{}")
                d["probabilities"] = json.loads(d.pop("probabilities_json", "{}") or "{}")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("解析决策 id=%s 的 JSON 字段失败，跳过: %s", d.get("id"), e)
                continue
            d["executed"] = bool(d.get("executed"))
            result.append(d)
        return result

    def compute_labels(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对每条决策计算后 N 日收益并生成带标签训练样本。

        无法关联 K线或后续数据不足 N 个交易日的决策将被丢弃。

        Args:
            decisions: ``load_decisions`` 返回的决策列表。

        Returns:
            样本列表，每个样本含：id / timestamp / symbol / strategy_signal /
            features(states 数组) / label(最优动作) / future_return /
            max_drawdown / stop_loss_triggered / original_action / original_executed。
        """
        samples: List[Dict[str, Any]] = []
        for d in decisions:
            symbol = d.get("symbol", "")
            df = self._resolve_df(symbol)
            try:
                ts = pd.Timestamp(d.get("timestamp"))
            except Exception:  # noqa: BLE001
                continue

            win = self._future_window(df, ts, self.forward_days)
            if win is None:
                # 后续数据不足或无 K线 → 无法标注，跳过
                continue

            future_return = win["future_return"]
            min_close_ratio = win["min_close_ratio"]
            label = self._label_from_return(future_return)
            features = self._extract_states(d.get("market_state") or {})

            samples.append({
                "id": d.get("id"),
                "timestamp": d.get("timestamp"),
                "symbol": symbol,
                "strategy_signal": d.get("strategy_signal") or "unknown",
                "features": features,
                "label": label,
                "future_return": round(future_return, 6),
                # 后 N 日最大回撤（期间最低收盘相对决策日跌幅，负值）
                "max_drawdown": round(min_close_ratio, 6),
                "stop_loss_triggered": bool(min_close_ratio < -STOP_LOSS_THRESHOLD),
                "original_action": d.get("final_action", "hold"),
                "original_executed": bool(d.get("executed", False)),
            })
        return samples

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def export_jsonl(self, samples: List[Dict[str, Any]], output_path: str) -> str:
        """将样本列表导出为 JSONL（每行一个 JSON 样本）。

        Returns:
            写入的文件路径。
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        return str(out)

    def export_csv(self, samples: List[Dict[str, Any]], output_path: str) -> str:
        """将样本列表导出为 CSV，features 展开为各特征单独列。

        Returns:
            写入的文件路径。
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # 固定列：元信息 + 7 个展开特征列 + 标签/辅助列
        meta_cols = ["id", "timestamp", "symbol", "strategy_signal"]
        feature_cols = FEATURE_NAMES
        label_cols = [
            "label", "future_return", "max_drawdown",
            "stop_loss_triggered", "original_action", "original_executed",
        ]
        fieldnames = meta_cols + feature_cols + label_cols

        # 把 states 数组拍平成 {feature: value}
        def _flatten(features: List[Dict[str, Any]]) -> Dict[str, Any]:
            flat: Dict[str, Any] = {}
            for item in features:
                feat = item.get("feature")
                if feat in FEATURE_NAMES:
                    flat[feat] = item.get("value")
            return flat

        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in samples:
                row = {c: s.get(c, "") for c in meta_cols + label_cols}
                row.update(_flatten(s.get("features") or []))
                writer.writerow(row)
        return str(out)

    # ------------------------------------------------------------------
    # 切分 / 统计
    # ------------------------------------------------------------------

    @staticmethod
    def split_dataset(
        samples: List[Dict[str, Any]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """按时间顺序切分数据集（避免未来数据泄露）。

        先按 timestamp 升序排序，再按比例切分；剩余样本归入 test。

        Returns:
            ``{"train": [...], "val": [...], "test": [...]}``。
        """
        sorted_samples = sorted(
            samples, key=lambda s: pd.Timestamp(s.get("timestamp"))
        )
        n = len(sorted_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = sorted_samples[:n_train]
        val = sorted_samples[n_train: n_train + n_val]
        test = sorted_samples[n_train + n_val:]
        return {"train": train, "val": val, "test": test}

    @staticmethod
    def compute_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """统计样本分布、平均收益、原始动作与标签一致率、各标的样本数。"""
        total = len(samples)
        if total == 0:
            return {
                "total": 0,
                "label_distribution": {"buy": {"count": 0, "ratio": 0.0},
                                       "sell": {"count": 0, "ratio": 0.0},
                                       "hold": {"count": 0, "ratio": 0.0}},
                "avg_future_return": 0.0,
                "accuracy": 0.0,
                "by_symbol": {},
            }

        label_counts = {"buy": 0, "sell": 0, "hold": 0}
        ret_sum = 0.0
        correct = 0
        by_symbol: Dict[str, int] = {}
        for s in samples:
            lab = s.get("label", "hold")
            label_counts[lab] = label_counts.get(lab, 0) + 1
            ret_sum += float(s.get("future_return", 0.0))
            if s.get("original_action") == lab:
                correct += 1
            sym = s.get("symbol", "unknown")
            by_symbol[sym] = by_symbol.get(sym, 0) + 1

        return {
            "total": total,
            "label_distribution": {
                k: {"count": v, "ratio": round(v / total, 4)}
                for k, v in label_counts.items()
            },
            "avg_future_return": round(ret_sum / total, 6),
            "accuracy": round(correct / total, 4),
            "by_symbol": by_symbol,
        }

    # ------------------------------------------------------------------
    # 一站式导出
    # ------------------------------------------------------------------

    def export_all(
        self,
        output_dir: str = "output/training_data",
        **filters: Any,
    ) -> Dict[str, Any]:
        """一站式：加载 → 标注 → 时间切分 → 导出 train/val/test 的 JSONL+CSV → 统计。

        Args:
            output_dir: 导出目录，默认 ``output/training_data``。
            **filters: 透传给 ``load_decisions`` 的过滤条件
                （start_date / end_date / symbol / strategy / limit）。

        Returns:
            ``{"file_paths": {...}, "stats": {...}, "sample_count": N}``。
        """
        decisions = self.load_decisions(**filters)
        samples = self.compute_labels(decisions)
        splits = self.split_dataset(samples)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")

        file_paths: Dict[str, str] = {}
        for part in ("train", "val", "test"):
            part_samples = splits[part]
            jsonl_path = out_dir / f"jev_{part}_{date_str}.jsonl"
            csv_path = out_dir / f"jev_{part}_{date_str}.csv"
            self.export_jsonl(part_samples, str(jsonl_path))
            self.export_csv(part_samples, str(csv_path))
            file_paths[f"{part}_jsonl"] = str(jsonl_path)
            file_paths[f"{part}_csv"] = str(csv_path)

        stats = self.compute_stats(samples)

        # 落盘最近一次导出统计，供 /api/jev/training_stats 读取
        stats_path = out_dir / "last_export_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "forward_days": self.forward_days,
                    "hold_threshold": self.hold_threshold,
                    "file_paths": file_paths,
                    "stats": stats,
                    "sample_count": len(samples),
                },
                f, ensure_ascii=False, indent=2,
            )
        file_paths["stats_json"] = str(stats_path)

        return {
            "file_paths": file_paths,
            "stats": stats,
            "sample_count": len(samples),
        }
