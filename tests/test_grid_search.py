"""网格搜索优化器单元测试。

使用构造的 mock 行情数据（无网络依赖），覆盖:
- 组合数与返回字段
- 三种 objective 的排序方向
- max_combos 超限警告
- 空参数网格
- progress_callback 调用次数
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from optimization.grid_search import GridSearchOptimizer
from strategies.ma_cross import MACrossStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_data(n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    """构造带趋势 + 周期波动的 mock 日线数据，确保均线产生交叉。

    Returns:
        含 open/high/low/close/volume 列、DatetimeIndex 的 DataFrame。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2023-01-02", periods=n_days)

    # 上涨趋势叠加正弦波动 + 小幅噪声，制造多次金叉/死叉
    t = np.arange(n_days)
    trend = 100.0 + 0.15 * t
    cycle = 8.0 * np.sin(2 * np.pi * t / 40.0)  # 约40天一个周期
    noise = rng.normal(0, 0.8, size=n_days)
    close = trend + cycle + noise

    open_ = close + rng.normal(0, 0.3, size=n_days)
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.6, size=n_days)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.6, size=n_days)
    volume = rng.uniform(8e5, 2e6, size=n_days)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture
def mock_data() -> pd.DataFrame:
    """mock 行情数据。"""
    return _make_mock_data()


@pytest.fixture
def optimizer() -> GridSearchOptimizer:
    """默认优化器。"""
    return GridSearchOptimizer(max_combos=200)


# ---------------------------------------------------------------------------
# 基本行为
# ---------------------------------------------------------------------------

class TestBasicSearch:
    """基本网格搜索行为。"""

    def test_four_combos_returned(self, optimizer, mock_data):
        """fast=[3,5] × slow=[15,20] 共 4 组，返回 4 组结果。"""
        grid = {"fast_period": [3, 5], "slow_period": [15, 20]}
        results = optimizer.optimize(
            mock_data, MACrossStrategy, grid, symbol="TEST",
            objective="sharpe", top_n=10,
        )
        assert len(results) == 4

    def test_result_has_params_and_metrics(self, optimizer, mock_data):
        """每组结果包含 params 和 metrics 字段。"""
        grid = {"fast_period": [3, 5], "slow_period": [15, 20]}
        results = optimizer.optimize(
            mock_data, MACrossStrategy, grid, symbol="TEST",
        )
        for item in results:
            assert "params" in item
            assert "metrics" in item
            assert "fast_period" in item["params"]
            assert "slow_period" in item["params"]
            # 全量 9 项指标都在
            assert "夏普比率" in item["metrics"]
            assert "累计收益率" in item["metrics"]
            assert "最大回撤" in item["metrics"]
            assert "交易次数" in item["metrics"]

    def test_top_n_truncates(self, optimizer, mock_data):
        """top_n 截断返回数量。"""
        grid = {"fast_period": [3, 5, 8], "slow_period": [15, 20, 30]}  # 9 组
        results = optimizer.optimize(
            mock_data, MACrossStrategy, grid, symbol="TEST", top_n=3,
        )
        assert len(results) == 3


# ---------------------------------------------------------------------------
# 排序
# ---------------------------------------------------------------------------

class TestSorting:
    """各 objective 的排序方向。"""

    def test_sharpe_descending(self, optimizer, mock_data):
        """默认 objective=sharpe 时按夏普比率降序。"""
        grid = {"fast_period": [3, 5, 8], "slow_period": [15, 20, 30]}
        results = optimizer.optimize(
            mock_data, MACrossStrategy, grid, symbol="TEST",
            objective="sharpe",
        )
        sharpes = [r["metrics"]["夏普比率"] for r in results]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_return_descending(self, optimizer, mock_data):
        """objective=return 时按累计收益率降序。"""
        grid = {"fast_period": [3, 5, 8], "slow_period": [15, 20, 30]}
        results = optimizer.optimize(
            mock_data, MACrossStrategy, grid, symbol="TEST",
            objective="return",
        )
        rets = [r["metrics"]["累计收益率"] for r in results]
        assert rets == sorted(rets, reverse=True)

    def test_drawdown_ascending(self, optimizer, mock_data):
        """objective=drawdown 时按最大回撤升序（越接近 0 越好）。"""
        grid = {"fast_period": [3, 5, 8], "slow_period": [15, 20, 30]}
        results = optimizer.optimize(
            mock_data, MACrossStrategy, grid, symbol="TEST",
            objective="drawdown",
        )
        # 最大回撤为负值，升序意味着 -0.02 < -0.10，即回撤小的排前面
        dds = [r["metrics"]["最大回撤"] for r in results]
        assert dds == sorted(dds)

    def test_unsupported_objective_raises(self, optimizer, mock_data):
        """不支持的 objective 抛出 ValueError。"""
        with pytest.raises(ValueError):
            optimizer.optimize(
                mock_data, MACrossStrategy,
                {"fast_period": [5], "slow_period": [20]},
                objective="bogus",
            )


# ---------------------------------------------------------------------------
# max_combos 警告
# ---------------------------------------------------------------------------

class TestMaxCombosWarning:
    """组合数上限保护。"""

    def test_warning_exceeding_max_combos(self, mock_data, caplog):
        """组合数超过 max_combos 时记录 warning。"""
        opt = GridSearchOptimizer(max_combos=2)  # 阈值 2，但网格有 4 组
        grid = {"fast_period": [3, 5], "slow_period": [15, 20]}

        with caplog.at_level(logging.WARNING, logger="optimization.grid_search"):
            results = opt.optimize(
                mock_data, MACrossStrategy, grid, symbol="TEST",
            )

        # 仍执行全部回测
        assert len(results) == 4
        # 出现警告日志
        assert any("超过 max_combos" in rec.message for rec in caplog.records)

    def test_no_warning_below_limit(self, mock_data, caplog):
        """组合数未超限时不报警告。"""
        opt = GridSearchOptimizer(max_combos=100)
        grid = {"fast_period": [3, 5], "slow_period": [15, 20]}

        with caplog.at_level(logging.WARNING, logger="optimization.grid_search"):
            opt.optimize(mock_data, MACrossStrategy, grid, symbol="TEST")

        assert not any("超过 max_combos" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# 边界情况
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """空网格 / 日期过滤 / 进度回调。"""

    def test_empty_param_grid_returns_empty(self, optimizer, mock_data):
        """空 param_grid 返回空列表。"""
        assert optimizer.optimize(
            mock_data, MACrossStrategy, {}, symbol="TEST",
        ) == []

    def test_empty_param_values_returns_empty(self, optimizer, mock_data):
        """参数列表为空时返回空列表。"""
        grid = {"fast_period": [], "slow_period": [20]}
        assert optimizer.optimize(
            mock_data, MACrossStrategy, grid, symbol="TEST",
        ) == []

    def test_date_filter(self, optimizer, mock_data):
        """start_date/end_date 过滤生效（不崩溃即可，数据变短）。"""
        grid = {"fast_period": [5], "slow_period": [20]}
        results = optimizer.optimize(
            mock_data, MACrossStrategy, grid, symbol="TEST",
            start_date="2023-03-01", end_date="2023-09-30",
        )
        assert len(results) == 1

    def test_progress_callback_called_times(self, optimizer, mock_data):
        """progress_callback 被调用次数等于组合数。"""
        grid = {"fast_period": [3, 5], "slow_period": [15, 20]}  # 4 组
        calls: list = []

        def cb(current: int, total: int, item: dict) -> None:
            calls.append((current, total, item))

        optimizer.optimize(
            mock_data, MACrossStrategy, grid, symbol="TEST",
            progress_callback=cb,
        )
        assert len(calls) == 4
        # 进度序号 1..4，total=4
        assert [c[0] for c in calls] == [1, 2, 3, 4]
        assert all(c[1] == 4 for c in calls)
        # 每次回调都带 params
        for _, _, item in calls:
            assert "params" in item
