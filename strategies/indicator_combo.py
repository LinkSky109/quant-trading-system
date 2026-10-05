"""多指标组合策略。

融合趋势强度（ADX）、动量交叉（MACD）与超买过滤（RSI）：
  - 买入条件：ADX > adx_strong（趋势强）+ MACD 金叉 + RSI < rsi_overbought（未超买）
  - 卖出条件：ADX < adx_weak（趋势衰竭）+ MACD 死叉
  - 置信度：ADX 强度 × MACD 柱状图绝对值，截断到 [0.3, 1.0]
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from indicators.technical import adx
from strategies.base_strategy import BaseStrategy
from utils.indicators import macd, rsi


class IndicatorComboStrategy(BaseStrategy):
    """多指标组合策略（ADX + MACD + RSI）。"""

    name = "indicator_combo"

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.adx_period = int(self.params.get("adx_period", 14))
        self.adx_strong = float(self.params.get("adx_strong", 25.0))
        self.adx_weak = float(self.params.get("adx_weak", 20.0))
        self.rsi_period = int(self.params.get("rsi_period", 14))
        self.rsi_overbought = float(self.params.get("rsi_overbought", 70.0))

    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- 指标计算 ---
        adx_df = adx(out["high"], out["low"], out["close"], self.adx_period)
        out["adx"] = adx_df["adx"]
        out["plus_di"] = adx_df["plus_di"]
        out["minus_di"] = adx_df["minus_di"]

        dif, dea, hist = macd(out["close"])
        out["macd_dif"] = dif
        out["macd_dea"] = dea
        out["macd_hist"] = hist

        out["rsi"] = rsi(out["close"], self.rsi_period)

        # --- MACD 金叉/死叉 ---
        prev_dif = dif.shift(1)
        prev_dea = dea.shift(1)
        golden = (prev_dif <= prev_dea) & (dif > dea)
        death = (prev_dif >= prev_dea) & (dif < dea)

        # --- 组合信号 ---
        buy = (out["adx"] > self.adx_strong) & golden & (out["rsi"] < self.rsi_overbought)
        sell = (out["adx"] < self.adx_weak) & death

        out["signal"] = 0
        out.loc[buy, "signal"] = 1
        out.loc[sell, "signal"] = -1

        # --- 置信度：ADX 强度越高、MACD 柱越大，置信度越高 ---
        adx_strength = (out["adx"] / 50.0).clip(0.3, 1.0)
        hist_scale = hist.abs() / hist.abs().rolling(60, min_periods=1).max()
        confidence = (adx_strength * (0.5 + hist_scale)).clip(0.3, 1.0).fillna(0.5)

        out["confidence"] = 0.0
        out.loc[out["signal"] != 0, "confidence"] = confidence[out["signal"] != 0]

        # NaN 指标区间不产生信号
        invalid = out["adx"].isna() | out["macd_dif"].isna() | out["rsi"].isna()
        out.loc[invalid, "signal"] = 0
        out.loc[invalid, "confidence"] = 0.0

        return out
