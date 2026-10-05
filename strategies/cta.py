"""CTA趋势跟踪策略。

包含:
1. ATR通道突破策略: 基于ATR计算动态通道, 价格突破上轨做多, 跌破下轨做空/平仓
2. 唐奇安通道突破策略: 突破N日高点做多, 跌破N日低点平仓
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import numpy as np

from strategies.base_strategy import BaseStrategy


class CTAStrategy(BaseStrategy):
    """CTA趋势跟踪策略。"""

    name = "cta"

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.atr_period = int(self.params.get("atr_period", 14))
        self.channel_period = int(self.params.get("channel_period", 20))
        self.atr_multiplier = float(self.params.get("atr_multiplier", 2.0))
        self.use_donchian = bool(self.params.get("use_donchian", True))
        self.use_atr_channel = bool(self.params.get("use_atr_channel", True))

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """计算ATR (Average True Range)。"""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()

    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # 计算ATR
        out["atr"] = self._atr(out["high"], out["low"], out["close"], self.atr_period)

        # ATR通道
        out["atr_upper"] = out["close"] + out["atr"] * self.atr_multiplier
        out["atr_lower"] = out["close"] - out["atr"] * self.atr_multiplier

        # 唐奇安通道 (不含当日)
        out["donchian_high"] = out["high"].shift(1).rolling(window=self.channel_period, min_periods=1).max()
        out["donchian_low"] = out["low"].shift(1).rolling(window=self.channel_period, min_periods=1).min()

        # 信号合并
        atr_buy = out["close"] > out["atr_upper"].shift(1)
        atr_sell = out["close"] < out["atr_lower"].shift(1)
        dc_buy = out["close"] > out["donchian_high"]
        dc_sell = out["close"] < out["donchian_low"]

        out["signal"] = 0
        if self.use_atr_channel and self.use_donchian:
            # 双通道确认: 至少一个通道给出信号
            out.loc[atr_buy | dc_buy, "signal"] = 1
            out.loc[atr_sell | dc_sell, "signal"] = -1
        elif self.use_atr_channel:
            out.loc[atr_buy, "signal"] = 1
            out.loc[atr_sell, "signal"] = -1
        elif self.use_donchian:
            out.loc[dc_buy, "signal"] = 1
            out.loc[dc_sell, "signal"] = -1

        # 置信度: 基于突破幅度 / ATR
        atr_break_strength = (out["close"] - out["atr_upper"].shift(1)) / out["atr"]
        dc_break_strength = (out["close"] - out["donchian_high"]) / out["atr"]
        strength = pd.concat([atr_break_strength, dc_break_strength], axis=1).max(axis=1)

        out["confidence"] = (strength / 2.0).clip(0.3, 1.0)
        out.loc[out["signal"] == 0, "confidence"] = 0.0

        # 元数据
        out["metadata"] = out.apply(
            lambda row: {
                "atr": round(row["atr"], 4),
                "donchian_high": round(row["donchian_high"], 2),
                "donchian_low": round(row["donchian_low"], 2),
            },
            axis=1,
        )

        return out
