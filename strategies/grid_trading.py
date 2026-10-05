"""网格交易策略。

在 [lower_bound, upper_bound] 区间内等距划分 grid_count 条网格线。
价格从上向下跌破网格线 → 买入；从下向上突破网格线 → 卖出。
适合震荡市，单边趋势中会持续逆势加仓/减仓。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class GridTradingStrategy(BaseStrategy):
    """等距网格交易策略。

    Attributes:
        grid_count: 网格区间数量（网格线数 = grid_count + 1）。
        upper_bound: 价格区间上界；None 时自动取数据历史最高价。
        lower_bound: 价格区间下界；None 时自动取数据历史最低价。
        fixed_confidence: 固定置信度，信号产生时使用。
    """

    name = "grid_trading"

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.grid_count = int(self.params.get("grid_count", 10))
        self.upper_bound: Optional[float] = self.params.get("upper_bound")
        self.lower_bound: Optional[float] = self.params.get("lower_bound")
        self.fixed_confidence = float(self.params.get("fixed_confidence", 0.6))

    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = 0
        out["confidence"] = 0.0
        out["grid_index"] = np.nan

        if len(df) == 0 or "close" not in df.columns:
            return out

        try:
            lower = float(self.lower_bound) if self.lower_bound is not None else float(df["close"].min())
            upper = float(self.upper_bound) if self.upper_bound is not None else float(df["close"].max())
        except Exception:
            return out

        if upper <= lower or self.grid_count <= 0:
            # 区间无效或网格数非法，不产生信号
            return out

        step = (upper - lower) / self.grid_count

        # 当日价格落在第几个网格区间（0 ~ grid_count），边界处 clamp
        grid_index = ((out["close"] - lower) / step).apply(np.floor)
        grid_index = grid_index.clip(lower=0, upper=self.grid_count)
        out["grid_index"] = grid_index

        prev_grid = grid_index.shift(1)
        valid = grid_index.notna() & prev_grid.notna()

        # 向下穿越网格线：当日网格编号 < 前一日 → 买入
        buy_cond = valid & (grid_index < prev_grid)
        # 向上穿越网格线：当日网格编号 > 前一日 → 卖出
        sell_cond = valid & (grid_index > prev_grid)

        out.loc[buy_cond, "signal"] = 1
        out.loc[sell_cond, "signal"] = -1

        # 置信度：固定值，也可按穿越的网格层数加权（每层额外 +0.05，clip 到 1.0）
        crossed = (grid_index - prev_grid).abs().where(valid, 0)
        confidence = (self.fixed_confidence + 0.05 * crossed).clip(lower=0.0, upper=1.0)
        out.loc[valid, "confidence"] = confidence[valid]
        out.loc[out["signal"] == 0, "confidence"] = 0.0

        return out
