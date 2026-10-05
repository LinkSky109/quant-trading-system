"""CTA策略测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.cta import CTAStrategy


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
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    return pd.DataFrame({
        "open": close * (1 + np.random.normal(0, 0.005, n)),
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(100000, 1000000, n),
    }, index=pd.date_range("2024-01-01", periods=n))


def test_cta_strategy_generate_signals(sample_df):
    strategy = CTAStrategy(params={"atr_period": 14, "channel_period": 20})
    signals = strategy.generate_signals(sample_df, symbol="600519.SH")
    assert isinstance(signals, list)
    # 信号有shift(1), 前20+1行无信号
    if len(signals) > 0:
        s = signals[0]
        assert s.strategy == "cta"
        assert s.symbol == "600519.SH"
        assert s.action in ("buy", "sell")
        assert 0.0 <= s.confidence <= 1.0


def test_cta_donchian_only(sample_df):
    strategy = CTAStrategy(params={"use_atr_channel": False, "use_donchian": True})
    signals = strategy.generate_signals(sample_df, symbol="TEST")
    assert isinstance(signals, list)


def test_cta_atr_only(sample_df):
    strategy = CTAStrategy(params={"use_atr_channel": True, "use_donchian": False})
    signals = strategy.generate_signals(sample_df, symbol="TEST")
    assert isinstance(signals, list)


def test_cta_no_signal_when_both_disabled(sample_df):
    strategy = CTAStrategy(params={"use_atr_channel": False, "use_donchian": False})
    signals = strategy.generate_signals(sample_df, symbol="TEST")
    assert len(signals) == 0


def test_cta_metadata(sample_df):
    strategy = CTAStrategy()
    raw = strategy._compute_raw_signals(sample_df)
    assert "metadata" in raw.columns
    meta = raw.iloc[-1]["metadata"]
    assert "atr" in meta
    assert "donchian_high" in meta


def test_cta_signal_shift_no_future(sample_df):
    """确认信号已shift(1), 避免未来函数。"""
    strategy = CTAStrategy()
    raw = strategy._compute_raw_signals(sample_df)
    # shift(1)后第一行应为NaN或0
    assert raw["signal"].iloc[0] == 0 or pd.isna(raw["signal"].iloc[0])



def test_cta_sell_confidence_varies(sample_df):
    """验证不同卖出幅度的 confidence 不相等（非恒为 0.3）。"""
    strategy = CTAStrategy()
    raw = strategy._compute_raw_signals(sample_df)
    sell_signals = raw[raw["signal"] == -1]
    if len(sell_signals) > 1:
        # 至少应有不同幅度的卖出信号，confidence 不应全部相等
        unique_conf = sell_signals["confidence"].nunique()
        assert unique_conf > 1 or sell_signals["confidence"].iloc[0] != 0.3
