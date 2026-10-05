"""技术指标计算工具模块。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均。"""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """指数移动平均。"""
    return series.ewm(span=period, adjust=False).mean()


def bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """布林带。

    Returns:
        (中轨, 上轨, 下轨)
    """
    mid = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指标 RSI。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD 指标。

    Returns:
        (DIF, DEA, MACD柱)
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    dif = ema_fast - ema_slow
    dea = ema(dif, signal)
    hist = (dif - dea) * 2
    return dif, dea, hist


def highest(high: pd.Series, period: int) -> pd.Series:
    """过去 N 日最高价（不含当日）。"""
    return high.rolling(window=period, min_periods=period).max().shift(1)


def lowest(low: pd.Series, period: int) -> pd.Series:
    """过去 N 日最低价（不含当日）。"""
    return low.rolling(window=period, min_periods=period).min().shift(1)


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """量比：当日成交量 / 过去N日均量。"""
    avg_vol = volume.rolling(window=period, min_periods=period).mean().shift(1)
    return volume / avg_vol.replace(0, np.nan)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """为行情 DataFrame 批量添加常用技术指标列。

    要求 df 包含 open/high/low/close/volume 列。
    添加列: ma5, ma20, rsi, macd_dif, macd_dea, macd_hist,
            bb_mid, bb_upper, bb_lower, vol_ratio。
    """
    df = df.copy()
    df["ma5"] = sma(df["close"], 5)
    df["ma20"] = sma(df["close"], 20)
    df["rsi"] = rsi(df["close"], 14)
    df["macd_dif"], df["macd_dea"], df["macd_hist"] = macd(df["close"])
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = bollinger_bands(df["close"])
    df["vol_ratio"] = volume_ratio(df["volume"], 20)
    return df
