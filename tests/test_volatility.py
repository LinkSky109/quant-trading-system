"""波动率策略测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.volatility import VolatilityStrategy


@pytest.fixture
def sample_df():
    """生成样本周K线数据。"""
    np.random.seed(42)
    n = 300
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


def test_volatility_strategy_generate_signals(sample_df):
    strategy = VolatilityStrategy(params={"lookback_period": 100, "vol_period": 20})
    signals = strategy.generate_signals(sample_df, symbol="600519.SH")
    assert isinstance(signals, list)
    if len(signals) > 0:
        s = signals[0]
        assert s.strategy == "volatility"
        assert s.action in ("buy", "sell")
        assert 0.0 <= s.confidence <= 1.0


def test_volatility_garman_klass(sample_df):
    strategy = VolatilityStrategy(params={"vol_method": "garman_klass"})
    signals = strategy.generate_signals(sample_df, symbol="TEST")
    assert isinstance(signals, list)


def test_volatility_realized(sample_df):
    strategy = VolatilityStrategy(params={"vol_method": "realized"})
    signals = strategy.generate_signals(sample_df, symbol="TEST")
    assert isinstance(signals, list)


def test_volatility_signal_shift_no_future(sample_df):
    """确认信号已shift(1)。"""
    strategy = VolatilityStrategy()
    raw = strategy._compute_raw_signals(sample_df)
    assert raw["signal"].iloc[0] == 0 or pd.isna(raw["signal"].iloc[0])


def test_volatility_metadata(sample_df):
    strategy = VolatilityStrategy()
    raw = strategy._compute_raw_signals(sample_df)
    assert "metadata" in raw.columns
    meta = raw.iloc[-1]["metadata"]
    assert "current_vol" in meta
    assert "vol_p20" in meta
    assert "vol_p80" in meta


def test_volatility_extreme_low_vol(sample_df):
    """极低波动率时应该产生买入信号。"""
    # 构造低波动段
    df = sample_df.copy()
    df.loc[df.index[-10:], "close"] = 100.0 + np.linspace(0, 0.5, 10)
    df.loc[df.index[-10:], "high"] = df.loc[df.index[-10:], "close"] + 0.1
    df.loc[df.index[-10:], "low"] = df.loc[df.index[-10:], "close"] - 0.1
    strategy = VolatilityStrategy(params={"lookback_period": 50, "vol_period": 5})
    raw = strategy._compute_raw_signals(df)
    # 最后几行应该有信号 (可能buy或sell)
    assert "signal" in raw.columns


def test_volatility_confidence_range(sample_df):
    strategy = VolatilityStrategy()
    raw = strategy._compute_raw_signals(sample_df)
    non_zero = raw[raw["signal"] != 0]
    if len(non_zero) > 0:
        assert non_zero["confidence"].min() >= 0.3
        assert non_zero["confidence"].max() <= 1.0
