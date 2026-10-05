"""双均线趋势跟踪策略。

快线 MA5 上穿慢线 MA20 → 金叉买入
快线 MA5 下穿慢线 MA20 → 死叉卖出
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from strategies.base_strategy import BaseStrategy
from utils.indicators import sma


class MACrossStrategy(BaseStrategy):
    """双均线交叉策略。"""

    name = "ma_cross"

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.fast_period = int(self.params.get("fast_period", 5))
        self.slow_period = int(self.params.get("slow_period", 20))

    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ma_fast"] = sma(out["close"], self.fast_period)
        out["ma_slow"] = sma(out["close"], self.slow_period)

        # 金叉/死叉判断
        prev_fast = out["ma_fast"].shift(1)
        prev_slow = out["ma_slow"].shift(1)

        golden_cross = (prev_fast <= prev_slow) & (out["ma_fast"] > out["ma_slow"])
        death_cross = (prev_fast >= prev_slow) & (out["ma_fast"] < out["ma_slow"])

        out["signal"] = 0
        out.loc[golden_cross, "signal"] = 1
        out.loc[death_cross, "signal"] = -1

        # 置信度：均线间距占价格比例越大，置信度越高
        spread = (out["ma_fast"] - out["ma_slow"]).abs() / out["ma_slow"]
        out["confidence"] = (spread / spread.rolling(60).max()).clip(0.3, 1.0)
        out.loc[out["signal"] == 0, "confidence"] = 0.0

        return out
