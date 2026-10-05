"""波动率交易策略。

基于波动率的交易策略:
1. 波动率锥分析: 当前波动率在历史分位位置
2. 波动率择时: 高波动减仓/低波动加仓
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import numpy as np

from strategies.base_strategy import BaseStrategy


class VolatilityStrategy(BaseStrategy):
    """波动率交易策略。"""

    name = "volatility"

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.lookback_period = int(self.params.get("lookback_period", 252))
        self.vol_period = int(self.params.get("vol_period", 20))
        self.high_vol_threshold = float(self.params.get("high_vol_threshold", 0.8))
        self.low_vol_threshold = float(self.params.get("low_vol_threshold", 0.2))
        self.vol_method = self.params.get("vol_method", "realized")  # realized | garman_klass

    def _realized_vol(self, close: pd.Series, period: int) -> pd.Series:
        """计算实现波动率 (年化)。"""
        log_ret = np.log(close / close.shift(1))
        rolling_vol = log_ret.rolling(window=period, min_periods=1).std() * np.sqrt(252)
        return rolling_vol

    def _garman_klass_vol(self, open_s: pd.Series, high: pd.Series,
                          low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """计算Garman-Klass波动率 (更精确, 使用OHLC)。"""
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_s) ** 2
        var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        var = var.rolling(window=period, min_periods=1).mean()
        return np.sqrt(var) * np.sqrt(252)

    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # 计算当前波动率
        if self.vol_method == "garman_klass" and all(c in out.columns for c in ["open", "high", "low", "close"]):
            out["current_vol"] = self._garman_klass_vol(
                out["open"], out["high"], out["low"], out["close"], self.vol_period
            )
        else:
            out["current_vol"] = self._realized_vol(out["close"], self.vol_period)

        # 历史波动率分位 (波动率锥)
        vol_history = out["current_vol"].shift(1).rolling(window=self.lookback_period, min_periods=30)
        # 使用 searchsorted 替代全窗口排序，提升性能
        vol_arr = out["current_vol"].shift(1).values
        out["vol_percentile"] = np.nan
        for i in range(len(out)):
            window = vol_arr[max(0, i - self.lookback_period + 1):i]
            if len(window) >= 30:
                current = vol_arr[i]
                sorted_vals = np.sort(window)
                idx = np.searchsorted(sorted_vals, current, side="right")
                out.iloc[i, out.columns.get_loc("vol_percentile")] = idx / len(window)

        # 简化计算: 使用quantile
        out["vol_p20"] = vol_history.quantile(0.20)
        out["vol_p80"] = vol_history.quantile(0.80)

        # 信号: 低波动分位 -> 加仓(buy); 高波动分位 -> 减仓(sell)
        buy_cond = out["current_vol"] < out["vol_p20"]
        sell_cond = out["current_vol"] > out["vol_p80"]

        out["signal"] = 0
        out.loc[buy_cond, "signal"] = 1
        out.loc[sell_cond, "signal"] = -1

        # 置信度: 距离分位边界的程度
        vol_range = out["vol_p80"] - out["vol_p20"]
        vol_range = vol_range.replace(0, np.nan)
        dist_low = (out["vol_p20"] - out["current_vol"]) / vol_range
        dist_high = (out["current_vol"] - out["vol_p80"]) / vol_range
        out["confidence"] = pd.concat([dist_low, dist_high], axis=1).max(axis=1).clip(0.3, 1.0)
        out.loc[out["signal"] == 0, "confidence"] = 0.0

        # 元数据
        out["metadata"] = out.apply(
            lambda row: {
                "current_vol": round(row["current_vol"], 4) if pd.notna(row["current_vol"]) else None,
                "vol_percentile": round(row["vol_percentile"], 4) if pd.notna(row["vol_percentile"]) else None,
                "vol_p20": round(row["vol_p20"], 4) if pd.notna(row["vol_p20"]) else None,
                "vol_p80": round(row["vol_p80"], 4) if pd.notna(row["vol_p80"]) else None,
            },
            axis=1,
        )

        return out
