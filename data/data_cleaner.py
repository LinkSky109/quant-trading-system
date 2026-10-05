"""数据清洗模块。"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_klines(df: pd.DataFrame, required_cols: List[str] | None = None) -> pd.DataFrame:
    """清洗 K线数据。

    处理内容:
        1. 去除重复索引
        2. 按日期排序
        3. 去除关键字段为空的行
        4. 修正异常价格（非正、开高低收关系异常）
        5. 前向填充少量缺失

    Args:
        df: 原始 K线 DataFrame。
        required_cols: 必须存在的列，默认 OHLCV。

    Returns:
        清洗后的 DataFrame。
    """
    if required_cols is None:
        required_cols = ["open", "high", "low", "close", "volume"]

    df = df.copy()

    # 确保列存在
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")

    # 去重 & 排序
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # 去除关键字段为空
    df = df.dropna(subset=required_cols)

    # 价格必须为正
    for col in ["open", "high", "low", "close"]:
        df = df[df[col] > 0]

    # high >= max(open, close), low <= min(open, close)
    df = df[df["high"] >= df[["open", "close"]].max(axis=1)]
    df = df[df["low"] <= df[["open", "close"]].min(axis=1)]

    # 成交量非负
    df = df[df["volume"] >= 0]

    # 去除涨跌停异常（单日涨跌幅超过 20% 标记但保留，A 股主板 ±10%）
    pct = df["close"].pct_change()
    abnormal = (pct.abs() > 0.20).sum()
    if abnormal > 0:
        logger.warning("检测到 %d 条单日涨跌幅超 20%% 的记录", abnormal)

    logger.info("数据清洗完成: %d 行, 时间范围 %s ~ %s",
                len(df), df.index[0].date(), df.index[-1].date())
    return df


def fill_missing_dates(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """补齐缺失交易日（仅填充工作日）。

    Args:
        df: 索引为日期的 DataFrame。
        method: 填充方式 ffill / bfill。
    """
    df = df.copy()
    full_idx = pd.bdate_range(start=df.index.min(), end=df.index.max())
    df = df.reindex(full_idx)
    df.index.name = "date"
    if method == "ffill":
        df = df.ffill()
    elif method == "bfill":
        df = df.bfill()
    return df


def remove_outliers(df: pd.DataFrame, column: str = "close", n_std: float = 5.0) -> pd.DataFrame:
    """基于收益率的 z-score 去除极端异常值。"""
    df = df.copy()
    ret = df[column].pct_change()
    z = (ret - ret.mean()) / ret.std()
    mask = z.abs() <= n_std
    mask.iloc[0] = True  # 第一条保留
    removed = (~mask).sum()
    if removed > 0:
        logger.info("去除 %d 条 %s 异常记录", removed, column)
    return df[mask]
