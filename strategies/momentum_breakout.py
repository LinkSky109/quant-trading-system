"""动量突破策略。

收盘价突破过去 N 日高点 → 买入
收盘价跌破过去 M 日低点 → 卖出
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from strategies.base_strategy import BaseStrategy
from utils.indicators import highest, lowest


class MomentumBreakoutStrategy(BaseStrategy):
    """动量突破策略。"""

    name = "momentum_breakout"

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.breakout_period = int(self.params.get("breakout_period", 20))
        self.breakdown_period = int(self.params.get("breakdown_period", 10))

    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # highest/lowest 已内部 shift(1)，不含当日
        out["prev_high"] = highest(out["high"], self.breakout_period)
        out["prev_low"] = lowest(out["low"], self.breakdown_period)

        buy_cond = out["close"] > out["prev_high"]
        sell_cond = out["close"] < out["prev_low"]

        out["signal"] = 0
        out.loc[buy_cond, "signal"] = 1
        out.loc[sell_cond, "signal"] = -1

        # 置信度：突破幅度
        breakout_strength = (out["close"] - out["prev_high"]) / out["prev_high"]
        breakdown_strength = (out["prev_low"] - out["close"]) / out["prev_low"]
        strength = breakout_strength.where(buy_cond, breakdown_strength)
        out["confidence"] = (strength / 0.05).clip(0.3, 1.0)
        out.loc[out["signal"] == 0, "confidence"] = 0.0

        return out
