"""绩效归因分析核心模块。

基于回测净值曲线、成交记录与（可选的）组合分标的结果，
输出面向前端可视化的结构化归因报告。

设计约定：
- 所有比率类指标在分母为 0 时返回 ``None``（JSON 安全，前端可识别为 null）。
- 空数据（无净值、无成交）时各维度返回空结构，不抛异常。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult, Trade
from backtest.metrics import (
    calc_annualized_return,
    calc_cumulative_return,
    calc_max_drawdown,
)

logger = logging.getLogger(__name__)


class PerformanceAttribution:
    """绩效归因分析器。

    Args:
        risk_free_rate: 年化无风险利率，用于索提诺比率。
        trading_days: 年交易日数，用于年化与下行波动年化。
    """

    def __init__(self, risk_free_rate: float = 0.02, trading_days: int = 252):
        self.risk_free_rate = float(risk_free_rate)
        self.trading_days = int(trading_days)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def analyze(
        self,
        equity_curve: pd.Series,
        trades: List[Trade],
        symbols: Optional[List[str]] = None,
        symbol_results: Optional[Dict[str, BacktestResult]] = None,
        lookforward_days: int = 5,
        data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """对回测结果进行多维度归因分析。

        Args:
            equity_curve: 组合/标的净值曲线，index 为交易日。
            trades: 全部成交记录（sell 单需带 pnl）。
            symbols: 参与回测的标的列表（可选，仅用于标注）。
            symbol_results: 组合回测时各标的独立结果，用于持仓归因。
            lookforward_days: 信号质量评估的前瞻交易日数。
            data: 各标的原始行情 ``{symbol: df}``（需含 close 列，index 为交易日），
                用于计算买入/卖出信号后的真实价格收益；缺省时策略归因返回空结果。

        Returns:
            归因报告 dict，结构见模块文档与输出样例。
        """
        equity = self._normalize_equity(equity_curve)
        trades = list(trades or [])

        total_return = calc_cumulative_return(equity)
        annualized_return = calc_annualized_return(equity, self.trading_days)
        max_drawdown = calc_max_drawdown(equity)

        return {
            "summary": {
                "total_return": self._f(total_return),
                "annualized_return": self._f(annualized_return),
                "max_drawdown": self._f(max_drawdown),
                "total_trades": len(trades),
            },
            "trade_attribution": self._trade_attribution(trades),
            "time_attribution": self._time_attribution(equity),
            "holding_attribution": self._holding_attribution(symbol_results),
            "strategy_attribution": self._strategy_attribution(
                trades, lookforward_days, data
            ),
            "risk_adjusted": self._risk_adjusted(
                equity, total_return, annualized_return, max_drawdown
            ),
        }

    # ------------------------------------------------------------------
    # 1. 交易归因
    # ------------------------------------------------------------------

    def _trade_attribution(self, trades: List[Trade]) -> Dict[str, Any]:
        """拆解已实现盈亏：盈利 vs 亏损。"""
        closed = [t for t in trades if t.pnl is not None]
        pnls = [float(t.pnl) for t in closed]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        profit_contribution = sum(wins)
        loss_contribution = sum(losses)  # 负值
        total_realized = sum(pnls)

        return {
            "profit_contribution": self._f(profit_contribution),
            "loss_contribution": self._f(loss_contribution),
            "profit_count": len(wins),
            "loss_count": len(losses),
            "max_single_profit": self._f(max(wins)) if wins else 0.0,
            "max_single_loss": self._f(min(losses)) if losses else 0.0,
            "avg_profit": self._f(np.mean(wins)) if wins else 0.0,
            "avg_loss": self._f(np.mean(losses)) if losses else 0.0,
            "total_realized_pnl": self._f(total_realized),
        }

    # ------------------------------------------------------------------
    # 2. 时间归因
    # ------------------------------------------------------------------

    def _time_attribution(self, equity: pd.Series) -> Dict[str, Any]:
        """计算月/周收益率序列，并定位最佳/最差月份。"""
        monthly_returns = self._period_returns(equity, freq="M")
        weekly_returns = self._period_returns(equity, freq="W")

        best_month: Dict[str, Any] = {"month": None, "return": None}
        worst_month: Dict[str, Any] = {"month": None, "return": None}
        if monthly_returns:
            best = max(monthly_returns, key=lambda x: x["return"])
            worst = min(monthly_returns, key=lambda x: x["return"])
            best_month = {"month": best["month"], "return": self._f(best["return"])}
            worst_month = {"month": worst["month"], "return": self._f(worst["return"])}

        return {
            "monthly_returns": monthly_returns,
            "weekly_returns": weekly_returns,
            "best_month": best_month,
            "worst_month": worst_month,
        }

    def _period_returns(self, equity: pd.Series, freq: str) -> List[Dict[str, Any]]:
        """按 period（M=月 / W=周）聚合净值末值并计算区间收益率。

        首月/首周收益相对回测起点净值计算，保证区间首尾可拼接。
        """
        if equity is None or len(equity) < 2:
            return []

        try:
            period = equity.index.to_period(freq)
            grouped = equity.groupby(period).last()
        except Exception as exc:  # pragma: no cover - 防御性
            logger.warning("period 聚合失败: %s", exc)
            return []

        if grouped.empty:
            return []

        # 在最前补一个起点净值，使第一期收益 = 期末/起点 - 1
        anchor = equity.iloc[0]
        values = [float(anchor)] + [float(v) for v in grouped.values]
        labels = [self._period_label(idx, freq) for idx in grouped.index]

        out: List[Dict[str, Any]] = []
        for i, label in enumerate(labels):
            base = values[i]
            cur = values[i + 1]
            if base == 0 or not np.isfinite(base) or not np.isfinite(cur):
                ret = 0.0
            else:
                ret = cur / base - 1.0
            key = "month" if freq == "M" else "week"
            out.append({key: label, "return": self._f(ret)})
        return out

    @staticmethod
    def _period_label(period: pd.Period, freq: str) -> str:
        """将 Period 转为前端友好的字符串标签。"""
        if freq == "M":
            return period.strftime("%Y-%m")
        # 周：ISO 年-周 如 2024-W03
        ts = period.start_time
        iso = ts.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"

    # ------------------------------------------------------------------
    # 3. 持仓（标的）归因
    # ------------------------------------------------------------------

    def _holding_attribution(
        self,
        symbol_results: Optional[Dict[str, BacktestResult]],
    ) -> Dict[str, Any]:
        """基于各标的独立回测结果，统计其已实现盈亏对组合的贡献。"""
        empty = {
            "symbol_contributions": [],
            "top_contributor": {"symbol": None, "contribution": None},
            "top_drag": {"symbol": None, "contribution": None},
        }
        if not symbol_results:
            return empty

        contributions: List[Dict[str, Any]] = []
        raw: Dict[str, float] = {}
        for sym, res in symbol_results.items():
            pnl_sum = 0.0
            if res is not None and res.trades:
                pnl_sum = sum(
                    float(t.pnl) for t in res.trades if t.pnl is not None
                )
            raw[sym] = pnl_sum

        total_net = sum(raw.values())
        for sym, pnl_sum in raw.items():
            pct = (pnl_sum / total_net) if total_net != 0 else 0.0
            contributions.append({
                "symbol": sym,
                "contribution": self._f(pnl_sum),
                "pct": self._f(pct),
            })

        if not contributions:
            return empty

        top = max(contributions, key=lambda x: x["contribution"])
        drag = min(contributions, key=lambda x: x["contribution"])

        return {
            "symbol_contributions": contributions,
            "top_contributor": {
                "symbol": top["symbol"],
                "contribution": top["contribution"],
            },
            "top_drag": {
                "symbol": drag["symbol"],
                "contribution": drag["contribution"],
            },
        }

    # ------------------------------------------------------------------
    # 4. 策略（信号）归因
    # ------------------------------------------------------------------

    def _strategy_attribution(
        self,
        trades: List[Trade],
        lookforward_days: int,
        data: Optional[Dict[str, pd.DataFrame]],
    ) -> Dict[str, Any]:
        """评估买入/卖出信号后的前瞻收益。

        - 买入信号质量：买入后 N 日平均收益为正，说明买点有效。
        - 卖出信号质量：卖出后 N 日平均收益若为正，说明卖早了（价格继续上涨）。
        """
        buy_quality = {"avg_return_after_n_days": None, "sample_count": 0}
        sell_quality = {"avg_return_after_n_days": None, "sample_count": 0}

        if not data or lookforward_days <= 0:
            return {
                "buy_signal_quality": buy_quality,
                "sell_signal_quality": sell_quality,
                "note": "未提供行情数据 data，信号质量维度为空",
            }

        buy_rets: List[float] = []
        sell_rets: List[float] = []
        for t in trades:
            df = data.get(t.symbol)
            if df is None or t.date not in df.index:
                continue
            idx = df.index.get_loc(t.date)
            future_idx = idx + lookforward_days
            if future_idx >= len(df.index):
                continue
            try:
                base_price = float(df["close"].iloc[idx])
                future_price = float(df["close"].iloc[future_idx])
            except (KeyError, IndexError, ValueError):
                continue
            if base_price == 0 or not np.isfinite(base_price):
                continue
            ret = future_price / base_price - 1.0
            if t.action == "buy":
                buy_rets.append(ret)
            elif t.action == "sell":
                sell_rets.append(ret)

        if buy_rets:
            buy_quality = {
                "avg_return_after_n_days": self._f(float(np.mean(buy_rets))),
                "sample_count": len(buy_rets),
            }
        if sell_rets:
            sell_quality = {
                "avg_return_after_n_days": self._f(float(np.mean(sell_rets))),
                "sample_count": len(sell_rets),
            }

        return {
            "buy_signal_quality": buy_quality,
            "sell_signal_quality": sell_quality,
        }

    # ------------------------------------------------------------------
    # 5. 风险调整指标
    # ------------------------------------------------------------------

    def _risk_adjusted(
        self,
        equity: pd.Series,
        total_return: float,
        annualized_return: float,
        max_drawdown: float,
    ) -> Dict[str, Any]:
        """收益回撤比、卡玛比率、索提诺比率。"""
        abs_dd = abs(float(max_drawdown)) if max_drawdown is not None else 0.0

        # 收益/回撤比 = 累计收益 / |最大回撤|
        rd_ratio = self._div(total_return, abs_dd)
        # 卡玛比率 = 年化收益 / |最大回撤|
        calmar = self._div(annualized_return, abs_dd)

        # 索提诺：年化超额收益 / 年化下行标准差
        daily_ret = equity.pct_change().dropna() if len(equity) >= 2 else pd.Series(dtype=float)
        downside = np.minimum(daily_ret.values, 0.0)
        if len(downside) > 0:
            downside_dev = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(self.trading_days))
        else:
            downside_dev = 0.0
        sortino = self._div(annualized_return - self.risk_free_rate, downside_dev)

        return {
            "return_drawdown_ratio": rd_ratio,
            "calmar_ratio": calmar,
            "sortino_ratio": sortino,
        }

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_equity(equity_curve: pd.Series) -> pd.Series:
        """将净值曲线规整为 float Series，空输入返回空 Series。"""
        if equity_curve is None:
            return pd.Series(dtype=float)
        try:
            s = equity_curve.astype(float).dropna()
        except Exception:
            return pd.Series(dtype=float)
        s.name = "equity"
        return s

    @staticmethod
    def _div(numerator: float, denominator: float) -> Optional[float]:
        """安全除法：分母为 0 时返回 None。"""
        if denominator is None or denominator == 0 or not np.isfinite(denominator):
            return None
        return float(numerator) / float(denominator)

    @staticmethod
    def _f(value: Any) -> Optional[float]:
        """转为 JSON 安全的 float；NaN/Inf 转为 None。"""
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(v):
            return None
        return v
