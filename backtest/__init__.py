"""回测引擎模块。"""
from .engine import BacktestEngine, BacktestResult, Position, Trade
from .metrics import (
    calc_all_metrics,
    calc_annualized_return,
    calc_cumulative_return,
    calc_max_drawdown,
    calc_profit_loss_ratio,
    calc_sharpe_ratio,
    calc_win_rate,
    metrics_to_dataframe,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "Position",
    "Trade",
    "calc_all_metrics",
    "calc_cumulative_return",
    "calc_annualized_return",
    "calc_max_drawdown",
    "calc_sharpe_ratio",
    "calc_win_rate",
    "calc_profit_loss_ratio",
    "metrics_to_dataframe",
]
