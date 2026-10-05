"""专业技术指标库包。

导出 :class:`TechnicalIndicators` 门面类与全部指标函数。

Examples:
    >>> from indicators import TechnicalIndicators, atr, adx
    >>> ti = TechnicalIndicators()
"""
from __future__ import annotations

from .technical import (
    INDICATOR_REGISTRY,
    TechnicalIndicators,
    adx,
    atr,
    atr_ratio,
    bollinger_bandwidth,
    cci,
    cmf,
    detect_cross,
    detect_ma_alignment,
    dmi,
    historical_volatility,
    ichimoku,
    kdj,
    mfi,
    mom,
    obv,
    roc,
    sar,
    vwap,
    wr,
)

__all__ = [
    "TechnicalIndicators",
    "INDICATOR_REGISTRY",
    "atr",
    "adx",
    "dmi",
    "ichimoku",
    "sar",
    "kdj",
    "cci",
    "wr",
    "roc",
    "mom",
    "obv",
    "vwap",
    "mfi",
    "cmf",
    "bollinger_bandwidth",
    "atr_ratio",
    "historical_volatility",
    "detect_ma_alignment",
    "detect_cross",
]
