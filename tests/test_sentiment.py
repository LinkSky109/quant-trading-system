"""市场情绪/舆情策略测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.sentiment import MockSentimentProvider, SentimentStrategy


@pytest.fixture
def sample_df():
    """生成样本周K线数据。"""
    np.random.seed(42)
    n = 100
    base = 100.0
    close = [base]
    for _ in range(1, n):
        close.append(close[-1] * (1 + np.random.normal(0, 0.02)))
    close = np.array(close)
    return pd.DataFrame({
        "open": close * (1 + np.random.normal(0, 0.005, n)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.015, n))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.015, n))),
        "close": close,
        "volume": np.random.randint(100000, 1000000, n),
    }, index=pd.date_range("2024-01-01", periods=n))


# ---------------------------------------------------------------------------
# MockSentimentProvider
# ---------------------------------------------------------------------------

def test_mock_sentiment_deterministic():
    """同一 symbol + date 应返回相同结果。"""
    provider = MockSentimentProvider()
    s1 = provider.fetch("600519.SH", "2024-01-15")
    s2 = provider.fetch("600519.SH", "2024-01-15")
    assert s1.sentiment_score == s2.sentiment_score
    assert s1.news_count == s2.news_count


def test_mock_sentiment_different_symbols():
    """不同 symbol 应大概率返回不同结果。"""
    provider = MockSentimentProvider()
    s1 = provider.fetch("600519.SH", "2024-01-15")
    s2 = provider.fetch("000001.SZ", "2024-01-15")
    assert s1.sentiment_score != s2.sentiment_score or s1.buzz_score != s2.buzz_score


def test_mock_sentiment_range():
    """sentiment_score 应在 [-1, 1] 内。"""
    provider = MockSentimentProvider()
    for i in range(50):
        s = provider.fetch("TEST", f"2024-01-{i+1:02d}")
        assert -1.0 <= s.sentiment_score <= 1.0
        assert 0 <= s.news_count <= 50
        assert 100 <= s.social_volume <= 10000
        assert 0.0 <= s.buzz_score <= 1.0


def test_fetch_series():
    """fetch_series 应返回与日期等长的 DataFrame。"""
    provider = MockSentimentProvider()
    dates = pd.date_range("2024-01-01", periods=30)
    df = provider.fetch_series("TEST", dates)
    assert len(df) == 30
    assert "sentiment_score" in df.columns
    assert "buzz_score" in df.columns


# ---------------------------------------------------------------------------
# SentimentStrategy
# ---------------------------------------------------------------------------

def test_sentiment_strategy_generate_signals(sample_df):
    """策略应生成合法的 Signal 列表。"""
    strategy = SentimentStrategy()
    signals = strategy.generate_signals(sample_df, symbol="600519.SH")
    assert isinstance(signals, list)
    if len(signals) > 0:
        s = signals[0]
        assert s.strategy == "sentiment"
        assert s.action in ("buy", "sell")
        assert 0.0 <= s.confidence <= 1.0


def test_sentiment_contrarian_mode(sample_df):
    """反向模式下信号方向应与顺势模式相反（大致）。"""
    strategy_normal = SentimentStrategy(params={"use_contrarian": False})
    strategy_contra = SentimentStrategy(params={"use_contrarian": True})

    raw_normal = strategy_normal._compute_raw_signals(sample_df)
    raw_contra = strategy_contra._compute_raw_signals(sample_df)

    # 至少某些日期信号方向不同
    diff = (raw_normal["signal"] != raw_contra["signal"]).sum()
    assert diff > 0, "反向模式应与顺势模式产生差异"


def test_sentiment_confidence_range(sample_df):
    """置信度应在合法范围内。"""
    strategy = SentimentStrategy()
    raw = strategy._compute_raw_signals(sample_df)
    non_zero = raw[raw["signal"] != 0]
    if len(non_zero) > 0:
        assert non_zero["confidence"].min() >= 0.3
        assert non_zero["confidence"].max() <= 1.0


def test_sentiment_signal_shift_no_future(sample_df):
    """确认信号已 shift(1)。"""
    strategy = SentimentStrategy()
    raw = strategy._compute_raw_signals(sample_df)
    assert raw["signal"].iloc[0] == 0 or pd.isna(raw["signal"].iloc[0])


def test_sentiment_metadata(sample_df):
    """原始信号输出应包含情绪相关列。"""
    strategy = SentimentStrategy()
    raw = strategy._compute_raw_signals(sample_df)
    assert "sentiment_score" in raw.columns
    assert "sentiment_ewma" in raw.columns
    assert "momentum" in raw.columns
