"""RSI 相对强弱均值回归策略。

RSI 低于超卖阈值买入，高于超买阈值卖出。
适合震荡市，趋势市中可能反复逆势（左侧交易）。
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from strategies.base_strategy import BaseStrategy
from utils.indicators import rsi as rsi_indicator


class RSIStrategy(BaseStrategy):
    """RSI 均值回归策略。

    Attributes:
        period: RSI 计算窗口。
        oversold_threshold: 超卖阈值，RSI 低于该值产生买入信号。
        overbought_threshold: 超买阈值，RSI 高于该值产生卖出信号。
    """

    name = "rsi"

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.period = int(self.params.get("period", 14))
        self.oversold_threshold = float(self.params.get("oversold_threshold", 30.0))
        self.overbought_threshold = float(self.params.get("overbought_threshold", 70.0))

    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        try:
            out["rsi"] = rsi_indicator(out["close"], self.period)
        except Exception:
            out["rsi"] = float("nan")

        out["signal"] = 0
        out["confidence"] = 0.0

        # RSI 尚未计算出来的行（前 period 根）为 NaN，不产生信号
        valid = out["rsi"].notna()
        buy_cond = valid & (out["rsi"] < self.oversold_threshold)
        sell_cond = valid & (out["rsi"] > self.overbought_threshold)
        out.loc[buy_cond, "signal"] = 1
        out.loc[sell_cond, "signal"] = -1

        # 置信度：RSI 偏离中轴 50 越远，置信度越高，clip 到 [0.3, 1.0]
        confidence = ((out["rsi"] - 50.0).abs() / 50.0).clip(lower=0.3, upper=1.0)
        out.loc[valid, "confidence"] = confidence[valid]
        out.loc[out["signal"] == 0, "confidence"] = 0.0

        return out
