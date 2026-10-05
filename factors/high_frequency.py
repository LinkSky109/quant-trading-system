"""高频因子计算引擎。

基于日K线数据计算7类高频相关因子，捕捉短期价格/成交量微观结构特征：

1. **开盘跳空因子** (overnight_jump) — 隔夜收益率，(open - pre_close) / pre_close
2. **日内振幅比** (intraday_range) — 日内波动占收盘价比率，(high - low) / close
3. **成交量不平衡** (volume_imbalance) — 量比与成交量加速度
4. **价格加速度** (price_acceleration) — 收益率二阶差分
5. **VWAP偏离** (vwap_deviation) — close 与日内 VWAP 的偏离度
6. **连续涨跌天数** (consecutive_days) — 正/负收益率序列计数
7. **波动率聚集** (volatility_clustering) — 收益率绝对值的短期自相关

典型用法::

    engine = HighFrequencyFactorEngine()
    df = engine.calculate_all(klines_df)
    # df 新增 overnight_jump, intraday_range, volume_imbalance, volume_accel,
    #     price_acceleration, vwap_deviation, consecutive_up, consecutive_down,
    #     volatility_clustering 列
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HighFrequencyFactorEngine:
    """高频因子引擎。所有因子基于单标的日K线计算，不依赖外部API。"""

    FACTOR_NAMES: List[str] = [
        "overnight_jump",
        "intraday_range",
        "volume_imbalance",
        "volume_accel",
        "price_acceleration",
        "vwap_deviation",
        "consecutive_up",
        "consecutive_down",
        "volatility_clustering",
    ]

    # ------------------------------------------------------------------ #
    # 批量入口
    # ------------------------------------------------------------------ #
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """一次性计算全部高频因子并追加到原 DataFrame。

        Args:
            df: 日K线 DataFrame，需含 open/high/low/close/volume/amount 列。

        Returns:
            追加因子列后的 DataFrame。
        """
        out = df.copy()
        out = self.overnight_jump(out)
        out = self.intraday_range(out)
        out = self.volume_imbalance(out)
        out = self.price_acceleration(out)
        out = self.vwap_deviation(out)
        out = self.consecutive_days(out)
        out = self.volatility_clustering(out)
        return out

    # ------------------------------------------------------------------ #
    # 1. 开盘跳空因子
    # ------------------------------------------------------------------ #
    @staticmethod
    def overnight_jump(df: pd.DataFrame) -> pd.DataFrame:
        """隔夜跳空收益率 = (open - pre_close) / pre_close。"""
        out = df.copy()
        pre_close = out["close"].shift(1)
        out["overnight_jump"] = (out["open"] - pre_close) / pre_close
        return out

    # ------------------------------------------------------------------ #
    # 2. 日内振幅比
    # ------------------------------------------------------------------ #
    @staticmethod
    def intraday_range(df: pd.DataFrame) -> pd.DataFrame:
        """日内振幅比 = (high - low) / close。"""
        out = df.copy()
        out["intraday_range"] = (out["high"] - out["low"]) / out["close"]
        return out

    # ------------------------------------------------------------------ #
    # 3. 成交量不平衡
    # ------------------------------------------------------------------ #
    @staticmethod
    def volume_imbalance(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """成交量不平衡：量比 + 成交量加速度。

        - volume_imbalance: 当日成交量 / 前 lookback 日均量 - 1
        - volume_accel: 成交量的一阶差分 / 前一日成交量（增速）
        """
        out = df.copy()
        vol_ma = out["volume"].rolling(lookback, min_periods=1).mean()
        out["volume_imbalance"] = out["volume"] / vol_ma - 1.0
        vol_shift = out["volume"].shift(1)
        out["volume_accel"] = (out["volume"] - vol_shift) / vol_shift
        return out

    # ------------------------------------------------------------------ #
    # 4. 价格加速度
    # ------------------------------------------------------------------ #
    @staticmethod
    def price_acceleration(df: pd.DataFrame) -> pd.DataFrame:
        """价格加速度 = 收益率的二阶差分。

        r_t = close_t / close_{t-1} - 1
        acceleration = r_t - r_{t-1}
        """
        out = df.copy()
        ret = out["close"].pct_change()
        out["price_acceleration"] = ret - ret.shift(1)
        return out

    # ------------------------------------------------------------------ #
    # 5. VWAP偏离
    # ------------------------------------------------------------------ #
    @staticmethod
    def vwap_deviation(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
        """VWAP偏离 = (close - VWAP) / VWAP。

        VWAP 用 amount / volume 近似（日内均价）。
        同时计算 lookback 日滚动 VWAP 偏离均值，衡量持续偏离程度。
        """
        out = df.copy()
        vwap = out["amount"] / out["volume"].replace(0, np.nan)
        out["vwap_deviation"] = (out["close"] - vwap) / vwap
        out["vwap_deviation_ma"] = out["vwap_deviation"].rolling(lookback, min_periods=1).mean()
        return out

    # ------------------------------------------------------------------ #
    # 6. 连续涨跌天数
    # ------------------------------------------------------------------ #
    @staticmethod
    def consecutive_days(df: pd.DataFrame) -> pd.DataFrame:
        """连续涨跌天数计数。

        - consecutive_up: 截至当日连续收阳（收益率>0）的天数
        - consecutive_down: 截至当日连续收阴（收益率<0）的天数
        """
        out = df.copy()
        ret = out["close"].pct_change()
        pos = ret > 0
        neg = ret < 0
        out["consecutive_up"] = pos.groupby((pos != pos.shift()).cumsum()).cumcount() + 1
        out["consecutive_up"] = out["consecutive_up"].where(pos, 0)
        out["consecutive_down"] = neg.groupby((neg != neg.shift()).cumsum()).cumcount() + 1
        out["consecutive_down"] = out["consecutive_down"].where(neg, 0)
        return out

    # ------------------------------------------------------------------ #
    # 7. 波动率聚集
    # ------------------------------------------------------------------ #
    @staticmethod
    def volatility_clustering(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
        """波动率聚集：收益率绝对值的短期自相关系数。

        计算 abs(return_t) 与 abs(return_{t-1}) 在 lookback 窗口内的 Pearson 相关系数。
        正值表示波动聚集（大波动后跟着大波动）。
        """
        out = df.copy()
        abs_ret = out["close"].pct_change().abs()
        # 滚动自相关：corr(abs_ret, abs_ret.shift(1))
        shifted = abs_ret.shift(1)
        corr = abs_ret.rolling(lookback, min_periods=3).corr(shifted)
        out["volatility_clustering"] = corr
        return out

    # ------------------------------------------------------------------ #
    # 元数据
    # ------------------------------------------------------------------ #
    def get_factor_list(self) -> List[Dict[str, str]]:
        """返回因子元数据列表。"""
        return [
            {"name": "overnight_jump", "category": "高频", "direction": "-1",
             "description": "隔夜跳空收益率，高开偏空、低开偏多"},
            {"name": "intraday_range", "category": "高频", "direction": "-1",
             "description": "日内振幅比，振幅大可能反转"},
            {"name": "volume_imbalance", "category": "高频", "direction": "+1",
             "description": "成交量量比，放量确认趋势"},
            {"name": "volume_accel", "category": "高频", "direction": "+1",
             "description": "成交量加速度，加速放量"},
            {"name": "price_acceleration", "category": "高频", "direction": "-1",
             "description": "价格加速度，二阶导数捕捉拐点"},
            {"name": "vwap_deviation", "category": "高频", "direction": "-1",
             "description": "VWAP偏离度，偏离大可能回归"},
            {"name": "consecutive_up", "category": "高频", "direction": "-1",
             "description": "连续上涨天数，极值可能反转"},
            {"name": "consecutive_down", "category": "高频", "direction": "+1",
             "description": "连续下跌天数，极值可能反弹"},
            {"name": "volatility_clustering", "category": "高频", "direction": "+1",
             "description": "波动率聚集度，高聚集预示延续"},
        ]
