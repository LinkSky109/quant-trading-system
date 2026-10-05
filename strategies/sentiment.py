"""市场情绪/舆情策略。

基于 mock 舆情数据生成交易信号，结合价格动量进行过滤。
情绪数据源为 MockSentimentProvider（确定性哈希生成，便于回测复现）。
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class SentimentData:
    """单标单日的情绪数据。"""
    symbol: str
    date: str
    sentiment_score: float  # -1.0 ~ 1.0，负=悲观，正=乐观
    news_count: int
    social_volume: int
    buzz_score: float  # 0 ~ 1.0，讨论热度
    source: str = "mock"


class MockSentimentProvider:
    """Mock 舆情数据提供者。

    基于 symbol + date 的哈希确定性生成情绪指标，确保回测可复现。
    """

    def fetch(self, symbol: str, date: str) -> SentimentData:
        """获取某日情绪数据。"""
        seed = int(hashlib.md5(f"{symbol}:{date}".encode()).hexdigest(), 16)
        rng = np.random.default_rng(seed)

        # 生成有轻微趋势偏置的情绪分数（模拟某些标的长期偏乐观/悲观）
        symbol_bias = (seed % 100) / 500.0 - 0.1  # -0.1 ~ 0.1
        sentiment_score = float(np.clip(rng.normal(symbol_bias, 0.3), -1.0, 1.0))

        return SentimentData(
            symbol=symbol,
            date=date,
            sentiment_score=sentiment_score,
            news_count=int(rng.integers(0, 50)),
            social_volume=int(rng.integers(100, 10000)),
            buzz_score=float(rng.uniform(0.0, 1.0)),
        )

    def fetch_series(
        self, symbol: str, dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """获取时间序列情绪数据。"""
        records = []
        for d in dates:
            data = self.fetch(symbol, d.strftime("%Y-%m-%d"))
            records.append({
                "sentiment_score": data.sentiment_score,
                "news_count": data.news_count,
                "social_volume": data.social_volume,
                "buzz_score": data.buzz_score,
            })
        df = pd.DataFrame(records, index=dates)
        return df


class SentimentStrategy(BaseStrategy):
    """情绪驱动交易策略。

    信号生成逻辑：
    1. 原始情绪分数经 EWMA 平滑（降低单日噪声）
    2. 情绪 > buy_threshold 且价格动量为正 → 买入
    3. 情绪 < sell_threshold 且价格动量为负 → 卖出
    4. 情绪与价格动量背离（情绪极端但价格反向）→ 反向信号（ contrarian ）

    参数：
        sentiment_window: 情绪平滑窗口天数，默认 5
        buy_threshold: 买入情绪阈值，默认 0.3
        sell_threshold: 卖出情绪阈值，默认 -0.3
        momentum_window: 价格动量窗口，默认 5
        use_contrarian: 是否启用反向逻辑，默认 False
    """

    name = "sentiment"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.sentiment_window: int = int(self.params.get("sentiment_window", 5))
        self.buy_threshold: float = float(self.params.get("buy_threshold", 0.3))
        self.sell_threshold: float = float(self.params.get("sell_threshold", -0.3))
        self.momentum_window: int = int(self.params.get("momentum_window", 5))
        self.use_contrarian: bool = bool(self.params.get("use_contrarian", False))
        self.sentiment_provider = MockSentimentProvider()

    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算情绪信号。"""
        out = df.copy()
        out["signal"] = 0
        out["confidence"] = 0.0

        # 获取情绪序列
        sentiment_df = self.sentiment_provider.fetch_series(
            "", df.index
        )
        out["sentiment_score"] = sentiment_df["sentiment_score"].values
        out["sentiment_ewma"] = (
            out["sentiment_score"]
            .ewm(span=self.sentiment_window, min_periods=1)
            .mean()
        )

        # 价格动量
        out["momentum"] = out["close"].pct_change(self.momentum_window)

        # 信号生成
        if self.use_contrarian:
            # 反向逻辑：极度乐观 → 卖出，极度悲观 → 买入
            buy_mask = (out["sentiment_ewma"] < self.sell_threshold) & (out["momentum"] < 0)
            sell_mask = (out["sentiment_ewma"] > self.buy_threshold) & (out["momentum"] > 0)
        else:
            # 顺势逻辑：乐观 + 动量向上 → 买入
            buy_mask = (out["sentiment_ewma"] > self.buy_threshold) & (out["momentum"] > 0)
            sell_mask = (out["sentiment_ewma"] < self.sell_threshold) & (out["momentum"] < 0)

        out.loc[buy_mask, "signal"] = 1
        out.loc[sell_mask, "signal"] = -1

        # 置信度：情绪绝对值 * 动量绝对值，映射到 [0.3, 1.0]
        sentiment_strength = out["sentiment_ewma"].abs()
        momentum_strength = out["momentum"].abs().fillna(0)
        out["confidence"] = (sentiment_strength * 0.5 + momentum_strength * 5.0).clip(0.3, 1.0)
        out.loc[out["signal"] == 0, "confidence"] = 0.0

        return out
