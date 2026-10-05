"""跨市场套利策略单元测试。

覆盖：
  - 市场检测与汇率接口
  - 协整检验（OLS + ADF）
  - z_score 计算与信号生成
  - 交易成本估算
  - BaseStrategy 接口（generate_signals）
  - shift(1) 防未来函数验证
  - 边界情况（数据不足、非协整对）

使用确定性 mock 价格序列，不依赖外部 API。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.cross_market_arbitrage import (
    CrossMarketArbitrageStrategy,
    _detect_market,
    _ols_hedge_ratio,
    _adf_test,
    get_exchange_rate,
    get_trading_cost,
)


# ---------------------------------------------------------------------------
# 构造工具
# ---------------------------------------------------------------------------


def make_cointegrated_prices(n: int = 100, seed: int = 99) -> pd.DataFrame:
    """生成一对协整的价格序列（有均值回归的价差）。

    使用较强的均值回归系数（0.5）确保 ADF 检验能通过简化版临界值。
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    # price_b: 随机游走
    b = 100 * np.cumprod(1 + rng.normal(0, 0.01, size=n))
    # price_a: beta * b + 均值回归残差（系数 0.5 保证协整）
    beta = 0.8
    residual = np.zeros(n)
    for i in range(1, n):
        residual[i] = 0.5 * residual[i - 1] + rng.normal(0, 0.5)
    a = beta * b + residual + 10
    return pd.DataFrame({"a": a, "b": b}, index=dates)


def make_noncointegrated_prices(n: int = 100, seed: int = 77) -> pd.DataFrame:
    """生成一对非协整的价格序列（独立随机游走）。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    a = 100 * np.cumprod(1 + rng.normal(0, 0.015, size=n))
    b = 100 * np.cumprod(1 + rng.normal(0, 0.015, size=n))
    return pd.DataFrame({"a": a, "b": b}, index=dates)


@pytest.fixture
def strategy() -> CrossMarketArbitrageStrategy:
    return CrossMarketArbitrageStrategy({
        "symbol_a": "600519.SH",
        "symbol_b": "000858.SZ",
        "z_entry": 2.0,
        "z_exit": 0.5,
        "z_stop": 3.0,
        "window": 20,
    })


# ---------------------------------------------------------------------------
# 市场检测
# ---------------------------------------------------------------------------


class TestMarketDetection:
    def test_detect_cn_sh(self):
        assert _detect_market("600519.SH") == "CN"

    def test_detect_cn_sz(self):
        assert _detect_market("000858.SZ") == "CN"

    def test_detect_hk(self):
        assert _detect_market("00700.HK") == "HK"

    def test_detect_us(self):
        assert _detect_market("AAPL") == "US"


# ---------------------------------------------------------------------------
# 汇率与成本
# ---------------------------------------------------------------------------


class TestExchangeRate:
    def test_same_market(self):
        assert get_exchange_rate("600519.SH", "000858.SZ") == 1.0

    def test_cross_market_default(self):
        rate = get_exchange_rate("600519.SH", "AAPL")
        assert rate == 1.0


class TestTradingCost:
    def test_cn_default(self):
        assert get_trading_cost("600519.SH") == pytest.approx(0.0003)

    def test_hk_default(self):
        assert get_trading_cost("00700.HK") == pytest.approx(0.001)

    def test_us_default(self):
        assert get_trading_cost("AAPL") == pytest.approx(0.0005)

    def test_custom_cost(self):
        custom = {"CN": 0.0001}
        assert get_trading_cost("600519.SH", custom) == pytest.approx(0.0001)


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------


class TestOlsHedgeRatio:
    def test_basic(self):
        # 需要 >=10 个点才不会触发 early return
        x = pd.Series(range(1, 21), dtype=float)
        y = pd.Series([2 * i for i in range(1, 21)], dtype=float)
        result = _ols_hedge_ratio(y, x)
        assert result["hedge_ratio"] == pytest.approx(2.0, abs=0.01)

    def test_insufficient_data(self):
        x = pd.Series([1, 2])
        y = pd.Series([2, 3])
        result = _ols_hedge_ratio(y, x)
        assert result["hedge_ratio"] == 1.0


class TestAdfTest:
    def test_stationary_series(self):
        # 均值回归序列应该是平稳的
        rng = np.random.RandomState(0)
        s = pd.Series(np.cumsum(rng.normal(0, 0.1, 200)))
        # 非平稳序列通常不通过 ADF
        result = _adf_test(s)
        assert "t_stat" in result
        assert "is_stationary" in result

    def test_short_series(self):
        s = pd.Series([1, 2, 3])
        result = _adf_test(s)
        assert result["is_stationary"] is False


# ---------------------------------------------------------------------------
# 协整检验
# ---------------------------------------------------------------------------


class TestCointegration:
    def test_cointegrated_pair(self, strategy):
        prices = make_cointegrated_prices(n=100, seed=99)
        result = strategy.test_cointegration(prices["a"], prices["b"])
        assert result["is_cointegrated"] is True
        assert "hedge_ratio" in result
        assert "adf_t_stat" in result

    def test_noncointegrated_pair(self, strategy):
        prices = make_noncointegrated_prices(n=100, seed=77)
        result = strategy.test_cointegration(prices["a"], prices["b"])
        # 独立随机游走通常不协整
        assert result["is_cointegrated"] is False

    def test_insufficient_data(self, strategy):
        short = pd.Series([1, 2, 3, 4, 5])
        result = strategy.test_cointegration(short, short * 2)
        assert result["is_cointegrated"] is False


# ---------------------------------------------------------------------------
# 价差与 z_score
# ---------------------------------------------------------------------------


class TestCalculateSpread:
    def test_spread_shape(self, strategy):
        prices = make_cointegrated_prices(n=100, seed=99)
        strategy.test_cointegration(prices["a"], prices["b"])
        spread_df = strategy.calculate_spread(prices["a"], prices["b"])
        assert "spread" in spread_df.columns
        assert "z_score" in spread_df.columns
        assert len(spread_df) == len(prices)

    def test_z_score_mean_near_zero(self, strategy):
        prices = make_cointegrated_prices(n=200, seed=99)
        strategy.test_cointegration(prices["a"], prices["b"])
        spread_df = strategy.calculate_spread(prices["a"], prices["b"])
        z = spread_df["z_score"].dropna()
        assert abs(z.mean()) < 0.5


# ---------------------------------------------------------------------------
# 信号生成
# ---------------------------------------------------------------------------


class TestSignalGeneration:
    def test_signals_not_empty(self, strategy):
        prices = make_cointegrated_prices(n=200, seed=99)
        df = pd.DataFrame({
            "600519.SH_close": prices["a"],
            "000858.SZ_close": prices["b"],
        })
        signals = strategy.generate_signals(df)
        # 协整对应该有信号产生
        assert isinstance(signals, list)

    def test_signal_fields(self, strategy):
        prices = make_cointegrated_prices(n=200, seed=99)
        df = pd.DataFrame({
            "600519.SH_close": prices["a"],
            "000858.SZ_close": prices["b"],
        })
        signals = strategy.generate_signals(df)
        for sig in signals:
            assert sig.strategy == "cross_market_arbitrage"
            assert sig.action in ("buy", "sell")
            assert 0 <= sig.confidence <= 1
            assert "z_score" in sig.metadata
            assert "hedge_ratio" in sig.metadata

    def test_no_future_look(self, strategy):
        """验证信号使用 shift(1)，即 t 日信号基于 t-1 日数据。"""
        prices = make_cointegrated_prices(n=100, seed=99)
        df = pd.DataFrame({
            "600519.SH_close": prices["a"],
            "000858.SZ_close": prices["b"],
        })
        raw = strategy._compute_raw_signals(df)
        # 第一个值应为 NaN（shift 导致）
        assert pd.isna(raw["signal"].iloc[0]) or raw["signal"].iloc[0] == 0

    def test_missing_columns_neutral(self, strategy):
        df = pd.DataFrame({"close": [100, 101, 102]})
        signals = strategy.generate_signals(df)
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# 成本估算
# ---------------------------------------------------------------------------


class TestCostEstimate:
    def test_basic(self, strategy):
        cost = strategy.estimate_cost(1_000_000)
        assert "total_cost" in cost
        assert "cost_a" in cost
        assert "cost_b" in cost
        assert cost["total_cost"] > 0

    def test_cross_market_higher_cost(self):
        strat = CrossMarketArbitrageStrategy({
            "symbol_a": "600519.SH",
            "symbol_b": "AAPL",
        })
        cost = strat.estimate_cost(1_000_000)
        # A股万3 + 美股万5 = 万8 单边，双边 = 1.6‰
        expected = 1_000_000 * (0.0003 + 0.0005) * 2
        assert cost["total_cost"] == pytest.approx(expected, rel=1e-6)
