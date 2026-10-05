"""回测绩效指标计算模块。"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def calc_cumulative_return(equity: pd.Series) -> float:
    """累计收益率。"""
    if len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def calc_annualized_return(equity: pd.Series, trading_days: int = 252) -> float:
    """年化收益率（复利）。"""
    if len(equity) < 2:
        return 0.0
    total_days = len(equity)
    total_return = equity.iloc[-1] / equity.iloc[0]
    if total_return <= 0:
        return -1.0
    return float(total_return ** (trading_days / total_days) - 1)


def calc_max_drawdown(equity: pd.Series) -> float:
    """最大回撤。"""
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def calc_sharpe_ratio(
    equity: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days: int = 252,
) -> float:
    """夏普比率。"""
    if len(equity) < 3:
        return 0.0
    daily_returns = equity.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0
    excess = daily_returns - risk_free_rate / trading_days
    return float(np.sqrt(trading_days) * excess.mean() / daily_returns.std())


def calc_win_rate(trades: list[dict]) -> float:
    """胜率：盈利交易数 / 总平仓交易数。"""
    closed = [t for t in trades if t.get("pnl") is not None]
    if not closed:
        return 0.0
    wins = sum(1 for t in closed if t["pnl"] > 0)
    return wins / len(closed)


def calc_profit_loss_ratio(trades: list[dict]) -> float:
    """盈亏比：平均盈利 / 平均亏损的绝对值。"""
    closed = [t for t in trades if t.get("pnl") is not None]
    wins = [t["pnl"] for t in closed if t["pnl"] > 0]
    losses = [t["pnl"] for t in closed if t["pnl"] < 0]
    if not wins or not losses:
        return 0.0
    avg_win = np.mean(wins)
    avg_loss = abs(np.mean(losses))
    if avg_loss == 0:
        return float("inf")
    return float(avg_win / avg_loss)


def calc_total_profit(trades: list[dict]) -> float:
    """总盈利：所有盈利交易（pnl>0）的 pnl 之和。"""
    closed = [t for t in trades if t.get("pnl") is not None]
    return float(sum(t["pnl"] for t in closed if t["pnl"] > 0))


def calc_total_loss(trades: list[dict]) -> float:
    """总亏损：所有亏损交易（pnl<0）的 pnl 绝对值之和。"""
    closed = [t for t in trades if t.get("pnl") is not None]
    return float(abs(sum(t["pnl"] for t in closed if t["pnl"] < 0)))


def calc_order_fill_rate(
    trades: list[dict],
    total_orders: int = 0,
    filled_orders: int = 0,
) -> float:
    """订单成交率：成交订单数 / 总提交订单数。

    优先使用引擎显式传入的 ``total_orders`` / ``filled_orders``；
    若未传入（total_orders=0），则回退到 trades 列表中实际记录的笔数
    （市价单模式下所有信号订单立即成交，成交率为 1.0）。

    Args:
        trades: 交易记录列表（未使用，保留以保持签名一致）。
        total_orders: 引擎累计提交订单数。
        filled_orders: 引擎累计成交订单数。

    Returns:
        成交率，取值 [0, 1]；无订单时返回 0.0。
    """
    if total_orders and total_orders > 0:
        return float(filled_orders / total_orders)
    # 回退：trades 中所有记录视为成交
    if trades:
        return 1.0
    return 0.0


def calc_avg_holding_period(trades: list[dict]) -> float:
    """平均持仓时间（天）：所有平仓交易的持仓天数平均值。

    平仓交易指 ``pnl is not None`` 且 ``entry_date`` 可用的卖出记录。
    无平仓交易时返回 0.0。

    Args:
        trades: 交易记录字典列表，每条需含 ``date``、``entry_date``、``pnl``。

    Returns:
        平均持仓自然日天数。
    """
    holding_days: List[float] = []
    for t in trades:
        if t.get("pnl") is None:
            continue
        entry = t.get("entry_date")
        exit_ = t.get("date")
        if entry is None or exit_ is None:
            continue
        try:
            entry_ts = pd.Timestamp(entry)
            exit_ts = pd.Timestamp(exit_)
        except (TypeError, ValueError):
            continue
        days = (exit_ts - entry_ts).days
        if days >= 0:
            holding_days.append(float(days))
    if not holding_days:
        return 0.0
    return float(np.mean(holding_days))


def calc_all_metrics(
    equity: pd.Series,
    trades: list[dict],
    risk_free_rate: float = 0.02,
    trading_days: int = 252,
    total_orders: int = 0,
    filled_orders: int = 0,
) -> Dict[str, float]:
    """计算全部核心绩效指标。

    Args:
        equity: 净值曲线。
        trades: 交易记录字典列表。
        risk_free_rate: 无风险利率。
        trading_days: 年交易日数。
        total_orders: 引擎累计提交订单数（用于订单成交率）。
        filled_orders: 引擎累计成交订单数（用于订单成交率）。
    """
    return {
        "累计收益率": calc_cumulative_return(equity),
        "年化收益率": calc_annualized_return(equity, trading_days),
        "最大回撤": calc_max_drawdown(equity),
        "夏普比率": calc_sharpe_ratio(equity, risk_free_rate, trading_days),
        "胜率": calc_win_rate(trades),
        "盈亏比": calc_profit_loss_ratio(trades),
        "交易次数": len([t for t in trades if t.get("pnl") is not None]),
        "总盈利": calc_total_profit(trades),
        "总亏损": calc_total_loss(trades),
        "订单成交率": calc_order_fill_rate(trades, total_orders, filled_orders),
        "平均持仓时间": calc_avg_holding_period(trades),
    }


def metrics_to_dataframe(metrics: Dict[str, float]) -> pd.DataFrame:
    """将指标字典转为格式化 DataFrame。"""
    rows = []
    for k, v in metrics.items():
        if k in ("交易次数",):
            display = f"{int(v)}"
        elif k == "夏普比率":
            display = f"{v:.3f}"
        elif k == "盈亏比":
            display = f"{v:.2f}"
        elif k in ("总盈利", "总亏损"):
            display = f"{v:,.2f}"
        elif k == "订单成交率":
            display = f"{v * 100:.1f}%"
        elif k == "平均持仓时间":
            display = f"{v:.1f} 天"
        else:
            display = f"{v * 100:.2f}%"
        rows.append({"指标": k, "数值": display, "原始值": v})
    return pd.DataFrame(rows)
