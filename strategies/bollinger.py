"""布林带均值回归策略。

价格触及下轨 + 成交量放大 → 买入
价格触及上轨 + 成交量放大 → 卖出
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from strategies.base_strategy import BaseStrategy
from utils.indicators import bollinger_bands, sma


class BollingerStrategy(BaseStrategy):
    """布林带均值回归策略。"""

    name = "bollinger"

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.period = int(self.params.get("period", 20))
        self.num_std = float(self.params.get("num_std", 2.0))
        self.vol_filter_period = int(self.params.get("volume_filter_period", 20))
        self.vol_multiplier = float(self.params.get("volume_multiplier", 1.0))

    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["bb_mid"], out["bb_upper"], out["bb_lower"] = bollinger_bands(
            out["close"], self.period, self.num_std
        )
        out["vol_ma"] = sma(out["volume"], self.vol_filter_period)

        # 触及下轨买入（最低价 <= 下轨），成交量过滤
        buy_cond = (
            (out["low"] <= out["bb_lower"])
            & (out["volume"] > out["vol_ma"] * self.vol_multiplier)
        )
        # 触及上轨卖出（最高价 >= 上轨）
        sell_cond = (
            (out["high"] >= out["bb_upper"])
            & (out["volume"] > out["vol_ma"] * self.vol_multiplier)
        )

        out["signal"] = 0
        out.loc[buy_cond, "signal"] = 1
        out.loc[sell_cond, "signal"] = -1

        # 置信度：价格偏离中轨的程度 + 量比
        deviation = (out["bb_mid"] - out["close"]).abs() / (out["bb_upper"] - out["bb_lower"])
        vol_ratio = out["volume"] / out["vol_ma"].replace(0, pd.NA)
        out["confidence"] = (deviation * 0.6 + (vol_ratio / 3).clip(0, 1) * 0.4).clip(0.3, 1.0)
        out.loc[out["signal"] == 0, "confidence"] = 0.0

        return out
