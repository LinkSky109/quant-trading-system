"""MACD 趋势跟随策略。

DIF 上穿 DEA（金叉）买入，下穿 DEA（死叉）卖出。
配合 MACD 柱状图放量过滤：|hist| 需大于近期均值才确认信号，
避免在零轴附近频繁假交叉。
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy
from utils.indicators import macd as macd_indicator


class MACDStrategy(BaseStrategy):
    """MACD 金叉死叉策略。

    Attributes:
        fast: 快线 EMA 窗口。
        slow: 慢线 EMA 窗口。
        signal: DEA（信号线）EMA 窗口。
        hist_filter_period: 柱状图放量过滤的回看窗口。
    """

    name = "macd"

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.fast = int(self.params.get("fast", 12))
        self.slow = int(self.params.get("slow", 26))
        self.signal = int(self.params.get("signal", 9))
        self.hist_filter_period = int(self.params.get("hist_filter_period", 20))

    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        try:
            out["dif"], out["dea"], out["hist"] = macd_indicator(
                out["close"], self.fast, self.slow, self.signal
            )
        except Exception:
            out["dif"] = float("nan")
            out["dea"] = float("nan")
            out["hist"] = float("nan")

        # 用前一日状态对比当日判断穿越方向（当日内部 shift，基类再统一 shift 一天执行）
        prev_dif = out["dif"].shift(1)
        prev_dea = out["dea"].shift(1)

        # 柱状图放量过滤：|hist| 大于近期（不含当日）均值才确认。
        # 若近期均值为 0/NaN（尚无历史参考），则放行，避免震荡初起被误杀。
        hist_abs = out["hist"].abs()
        hist_mean = hist_abs.rolling(
            window=self.hist_filter_period, min_periods=1
        ).mean().shift(1)
        confirmed = hist_abs > hist_mean
        no_reference = hist_mean.isna() | (hist_mean == 0)
        volume_confirmed = confirmed | no_reference

        # warmup：EWM 从第 1 天即有值，初始 0/0 基线会误判为交叉；
        # 前 slow 根视为指标尚未稳定，不产生信号。
        warmup = self.slow
        row_idx = pd.Series(np.arange(len(out)), index=out.index)
        warmed_up = row_idx >= warmup

        valid = (
            out["dif"].notna()
            & out["dea"].notna()
            & prev_dif.notna()
            & prev_dea.notna()
            & warmed_up
        )

        golden_cross = (
            valid & (prev_dif <= prev_dea) & (out["dif"] > out["dea"]) & volume_confirmed
        )
        death_cross = (
            valid & (prev_dif >= prev_dea) & (out["dif"] < out["dea"]) & volume_confirmed
        )

        out["signal"] = 0
        out.loc[golden_cross, "signal"] = 1
        out.loc[death_cross, "signal"] = -1

        # 置信度：|hist| 相对近期均值的倍数，clip 到 [0.3, 1.0]
        safe_mean = hist_mean.where(hist_mean > 0)
        ratio = hist_abs / safe_mean
        confidence = (ratio / 2.0).clip(lower=0.3, upper=1.0).fillna(0.3)
        out["confidence"] = confidence
        out.loc[out["signal"] == 0, "confidence"] = 0.0

        return out
