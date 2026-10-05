"""高频因子引擎单元测试。

覆盖：
  - 全部 9 个因子计算正确性
  - 批量 calculate_all 入口
  - 因子元数据列表
  - 边界情况（空 DataFrame、数据不足、NaN 处理）

使用确定性随机种子构造模拟 K 线，不依赖外部 API。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.high_frequency import HighFrequencyFactorEngine


# ---------------------------------------------------------------------------
# 构造工具
# ---------------------------------------------------------------------------


def make_klines(n_days: int = 100, seed: int = 42) -> pd.DataFrame:
    """构造含 open/high/low/close/volume/amount 的模拟日 K 线。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rets = rng.normal(0.0005, 0.02, size=n_days)
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = close * (1.0 + rng.normal(0, 0.005, size=n_days))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0, 0.01, size=n_days))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0, 0.01, size=n_days))
    volume = rng.uniform(1e6, 5e6, size=n_days)
    amount = volume * close
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
        },
        index=dates,
    )


@pytest.fixture
def engine() -> HighFrequencyFactorEngine:
    return HighFrequencyFactorEngine()


@pytest.fixture
def klines() -> pd.DataFrame:
    return make_klines(n_days=100, seed=42)


# ---------------------------------------------------------------------------
# 因子正确性
# ---------------------------------------------------------------------------


class TestOvernightJump:
    def test_overnight_jump_formula(self, engine, klines):
        result = engine.overnight_jump(klines)
        expected = (klines["open"] - klines["close"].shift(1)) / klines["close"].shift(1)
        pd.testing.assert_series_equal(
            result["overnight_jump"], expected, check_names=False
        )

    def test_overnight_jump_first_nan(self, engine, klines):
        result = engine.overnight_jump(klines)
        assert pd.isna(result["overnight_jump"].iloc[0])


class TestIntradayRange:
    def test_intraday_range_formula(self, engine, klines):
        result = engine.intraday_range(klines)
        expected = (klines["high"] - klines["low"]) / klines["close"]
        pd.testing.assert_series_equal(
            result["intraday_range"], expected, check_names=False
        )

    def test_intraday_range_positive(self, engine, klines):
        result = engine.intraday_range(klines)
        assert (result["intraday_range"] > 0).all()


class TestVolumeImbalance:
    def test_volume_imbalance_shape(self, engine, klines):
        result = engine.volume_imbalance(klines)
        assert "volume_imbalance" in result.columns
        assert "volume_accel" in result.columns

    def test_volume_imbalance_first_finite(self, engine, klines):
        result = engine.volume_imbalance(klines)
        # 第一个值 volume_ma = 自身，所以 imbalance = 0
        assert result["volume_imbalance"].iloc[0] == pytest.approx(0.0, abs=1e-9)


class TestPriceAcceleration:
    def test_price_acceleration_formula(self, engine, klines):
        result = engine.price_acceleration(klines)
        ret = klines["close"].pct_change()
        expected = ret - ret.shift(1)
        pd.testing.assert_series_equal(
            result["price_acceleration"], expected, check_names=False
        )


class TestVwapDeviation:
    def test_vwap_deviation_formula(self, engine, klines):
        result = engine.vwap_deviation(klines)
        vwap = klines["amount"] / klines["volume"]
        expected = (klines["close"] - vwap) / vwap
        pd.testing.assert_series_equal(
            result["vwap_deviation"], expected, check_names=False
        )

    def test_vwap_deviation_ma_exists(self, engine, klines):
        result = engine.vwap_deviation(klines)
        assert "vwap_deviation_ma" in result.columns


class TestConsecutiveDays:
    def test_consecutive_up_logic(self, engine, klines):
        result = engine.consecutive_days(klines)
        ret = klines["close"].pct_change()
        # 找一个连续上涨的位置验证
        up_counts = result["consecutive_up"]
        for i in range(1, len(up_counts)):
            if ret.iloc[i] > 0 and ret.iloc[i - 1] > 0:
                assert up_counts.iloc[i] == up_counts.iloc[i - 1] + 1
                break

    def test_consecutive_down_logic(self, engine, klines):
        result = engine.consecutive_days(klines)
        ret = klines["close"].pct_change()
        down_counts = result["consecutive_down"]
        for i in range(1, len(down_counts)):
            if ret.iloc[i] < 0 and ret.iloc[i - 1] < 0:
                assert down_counts.iloc[i] == down_counts.iloc[i - 1] + 1
                break

    def test_no_overlap(self, engine, klines):
        result = engine.consecutive_days(klines)
        mask = result["consecutive_up"] > 0
        assert (result.loc[mask, "consecutive_down"] == 0).all()


class TestVolatilityClustering:
    def test_volatility_clustering_range(self, engine, klines):
        result = engine.volatility_clustering(klines)
        vc = result["volatility_clustering"].dropna()
        assert vc.between(-1.0, 1.0).all()

    def test_volatility_clustering_first_nan(self, engine, klines):
        result = engine.volatility_clustering(klines)
        # 前 lookback 个为 NaN
        assert pd.isna(result["volatility_clustering"].iloc[0])


# ---------------------------------------------------------------------------
# 批量入口
# ---------------------------------------------------------------------------


class TestCalculateAll:
    def test_all_factors_present(self, engine, klines):
        result = engine.calculate_all(klines)
        for name in engine.FACTOR_NAMES:
            assert name in result.columns, f"缺少因子: {name}"

    def test_preserve_original_columns(self, engine, klines):
        result = engine.calculate_all(klines)
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            assert col in result.columns

    def test_index_preserved(self, engine, klines):
        result = engine.calculate_all(klines)
        pd.testing.assert_index_equal(result.index, klines.index)


# ---------------------------------------------------------------------------
# 元数据
# ---------------------------------------------------------------------------


def test_get_factor_list(engine):
    meta = engine.get_factor_list()
    names = [m["name"] for m in meta]
    assert set(names) == set(engine.FACTOR_NAMES)
    for m in meta:
        assert "category" in m
        assert "direction" in m
        assert "description" in m


# ---------------------------------------------------------------------------
# 边界
# ---------------------------------------------------------------------------


def test_empty_dataframe(engine):
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "amount"])
    result = engine.calculate_all(empty)
    for name in engine.FACTOR_NAMES:
        assert name in result.columns


def test_single_row(engine):
    single = make_klines(n_days=1, seed=0)
    result = engine.calculate_all(single)
    assert len(result) == 1
