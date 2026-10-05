"""组合级回测引擎单元测试。

运行:
    cd quant_trading_system
    python -m pytest tests/test_portfolio_engine.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine
from backtest.portfolio_engine import (
    PortfolioBacktestEngine,
    PortfolioBacktestResult,
)
from data.data_fetcher import _generate_mock_klines
from strategies.ma_cross import MACrossStrategy

# 组合级绩效必须包含的 6 项核心指标
CORE_METRICS = [
    "累计收益率", "年化收益率", "最大回撤",
    "夏普比率", "胜率", "盈亏比",
]


# ---------------------------------------------------------------------------
# 测试数据工具
# ---------------------------------------------------------------------------

def make_mock_data(symbols, count: int = 200, start_date: str = "2024-01-02"):
    """生成多标的 mock 行情数据（确定性，不依赖网络）。"""
    return {
        sym: _generate_mock_klines(sym, period="1d", count=count, start_date=start_date)
        for sym in symbols
    }


def make_ohlc_from_close(close: np.ndarray, start_date: str = "2024-01-02") -> pd.DataFrame:
    """由收盘价序列构造完整 OHLCV DataFrame，便于控制波动率。"""
    n = len(close)
    dates = pd.bdate_range(start=start_date, periods=n)
    close = np.asarray(close, dtype=float)
    open_ = close * (1.0 + np.random.normal(0, 0.005, n))
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    volume = np.full(n, 1_000_000)
    df = pd.DataFrame({
        "open": open_.round(2), "high": high.round(2),
        "low": low.round(2), "close": close.round(2),
        "volume": volume, "amount": (close * volume).round(2),
    }, index=dates)
    df.index.name = "date"
    return df


def make_strategy():
    return MACrossStrategy({"fast_period": 5, "slow_period": 20})


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestPortfolioEngine:
    """组合回测引擎测试。"""

    def test_three_symbol_equal_weight_runs(self):
        """1. 3 只标的等权组合回测 -> 返回组合净值曲线和组合绩效。"""
        data = make_mock_data(["600519.SH", "300750.SZ", "002594.SZ"])
        engine = PortfolioBacktestEngine(initial_capital=1_000_000.0)
        result = engine.run(data, make_strategy())

        assert isinstance(result, PortfolioBacktestResult)
        assert len(result.portfolio_equity_curve) > 0
        assert len(result.portfolio_metrics) > 0
        # 三只标的明细都在
        assert set(result.symbol_results.keys()) == set(data.keys())
        assert set(result.symbol_metrics.keys()) == set(data.keys())

    def test_symbol_detail_matches_standalone(self):
        """2. 单标的明细与独立回测结果一致（相同数据、相同分配资金）。"""
        symbols = ["600519.SH", "300750.SZ"]
        data = make_mock_data(symbols)
        total_capital = 1_000_000.0

        engine = PortfolioBacktestEngine(
            initial_capital=total_capital, allocation_method="equal"
        )
        result = engine.run(data, make_strategy())

        # 等权 -> 每只分配 total/2
        alloc_capital = total_capital / 2.0
        standalone = BacktestEngine(initial_capital=alloc_capital).run(
            {"600519.SH": data["600519.SH"]}, make_strategy(), symbol="600519.SH"
        )
        # 组合内该标的的净值曲线应与独立回测完全一致
        eq_in_portfolio = result.symbol_results["600519.SH"].equity_curve
        pd.testing.assert_series_equal(
            eq_in_portfolio, standalone.equity_curve, check_names=False
        )

    def test_equal_weights_are_equal(self):
        """3. 等权分配时每只标的初始资金相等（验证 symbol_weights）。"""
        data = make_mock_data(["600519.SH", "300750.SZ", "002594.SZ"])
        engine = PortfolioBacktestEngine(allocation_method="equal")
        result = engine.run(data, make_strategy())

        weights = result.symbol_weights
        assert len(weights) == 3
        expected = pytest.approx(1.0 / 3.0, abs=1e-9)
        for w in weights.values():
            assert w == expected
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_volatility_inverse_weights(self):
        """4. 波动率倒数分配 -> 波动率低的标的获得更高权重。"""
        np.random.seed(42)
        # 低波动标的：日波动 ~0.2%；高波动标的：日波动 ~3%
        low_vol_ret = np.random.normal(0, 0.002, 200)
        high_vol_ret = np.random.normal(0, 0.03, 200)
        low_vol_price = 100.0 * np.cumprod(1 + low_vol_ret)
        high_vol_price = 100.0 * np.cumprod(1 + high_vol_ret)

        data = {
            "LOW_VOL": make_ohlc_from_close(low_vol_price),
            "HIGH_VOL": make_ohlc_from_close(high_vol_price),
        }
        engine = PortfolioBacktestEngine(allocation_method="volatility_inverse")
        result = engine.run(data, make_strategy())

        w = result.symbol_weights
        assert w["LOW_VOL"] > w["HIGH_VOL"]
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

    def test_custom_weights_normalized(self):
        """5. 自定义权重分配 -> 权重正确归一化和应用。"""
        data = make_mock_data(["AAA.SH", "BBB.SH"])
        engine = PortfolioBacktestEngine(
            allocation_method="custom",
            custom_weights={"AAA.SH": 1.0, "BBB.SH": 3.0},
        )
        result = engine.run(data, make_strategy())

        w = result.symbol_weights
        assert w["AAA.SH"] == pytest.approx(0.25, abs=1e-9)
        assert w["BBB.SH"] == pytest.approx(0.75, abs=1e-9)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

    def test_portfolio_equity_curve_shape(self):
        """6. 组合净值曲线长度 > 0，且初始值 ≈ initial_capital。"""
        data = make_mock_data(["600519.SH", "300750.SZ"])
        capital = 500_000.0
        engine = PortfolioBacktestEngine(initial_capital=capital)
        result = engine.run(data, make_strategy())

        curve = result.portfolio_equity_curve
        assert len(curve) > 0
        # 首日组合净值 = 总初始资金
        assert float(curve.iloc[0]) == pytest.approx(capital, rel=1e-6)

    def test_portfolio_metrics_contain_core_six(self):
        """7. 组合绩效包含所有 6 项核心指标。"""
        data = make_mock_data(["600519.SH", "300750.SZ", "002594.SZ"])
        engine = PortfolioBacktestEngine()
        result = engine.run(data, make_strategy())

        for key in CORE_METRICS:
            assert key in result.portfolio_metrics, f"缺少核心指标: {key}"

    def test_empty_and_single_symbol_do_not_crash(self):
        """8. 空数据或单标的输入不崩溃。"""
        # 空数据
        engine = PortfolioBacktestEngine()
        empty_result = engine.run({}, make_strategy())
        assert len(empty_result.portfolio_equity_curve) == 0
        assert empty_result.symbol_weights == {}

        # 单标的
        data = make_mock_data(["600519.SH"])
        single_result = engine.run(data, make_strategy())
        assert len(single_result.portfolio_equity_curve) > 0
        assert single_result.symbol_weights["600519.SH"] == pytest.approx(1.0)

    def test_import_no_conflict(self):
        """9. 与现有单标的回测不冲突（import 不报错，类可实例化）。"""
        from backtest.engine import BacktestEngine as BE  # noqa: F401
        from backtest.portfolio_engine import PortfolioBacktestEngine as PBE

        # 两种引擎可同时实例化
        be = BE(initial_capital=100_000)
        pe = PBE(initial_capital=1_000_000, allocation_method="equal")
        assert be.initial_capital == 100_000
        assert pe.initial_capital == 1_000_000

    def test_invalid_allocation_method_raises(self):
        """非法分配方式应抛出 ValueError。"""
        with pytest.raises(ValueError):
            PortfolioBacktestEngine(allocation_method="not_a_method")

    def test_custom_without_weights_raises(self):
        """custom 模式但未提供 custom_weights 应抛出 ValueError。"""
        with pytest.raises(ValueError):
            PortfolioBacktestEngine(allocation_method="custom", custom_weights=None)
