"""滚动窗口优化（Walk-Forward）单元测试。

使用构造的 mock 行情数据（无网络依赖），覆盖:
- 窗口划分：无重叠 / 无未来函数 / IS-OOS 比例 / 最后 OOS 到末尾
- 每个窗口 IS 优化 + OOS 测试流程
- OOS 净值曲线合并（按时间拼接、无重叠）
- 过拟合检测（OOS/IS 比率 < 0.5 标记）
- 随机搜索采样数量
- 推荐参数（数值中位数 / 类别众数）
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from optimization.walk_forward import WalkForwardOptimizer
from strategies.ma_cross import MACrossStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_data(n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    """构造带趋势 + 周期波动的 mock 日线数据，确保均线产生交叉。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2023-01-02", periods=n_days)

    t = np.arange(n_days)
    trend = 100.0 + 0.15 * t
    cycle = 8.0 * np.sin(2 * np.pi * t / 40.0)
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
    return _make_mock_data(n_days=300)


@pytest.fixture
def optimizer() -> WalkForwardOptimizer:
    return WalkForwardOptimizer(max_combos=500)


@pytest.fixture
def small_grid() -> Dict[str, list]:
    return {"fast_period": [3, 5], "slow_period": [15, 20]}


# ---------------------------------------------------------------------------
# 窗口划分
# ---------------------------------------------------------------------------

class TestSplitWindows:
    """滚动窗口划分正确性。"""

    def test_no_overlap_within_window(self, mock_data):
        """同一窗口内 IS 与 OOS 无重叠、严格时间顺序。"""
        windows = WalkForwardOptimizer.split_windows(mock_data, n_windows=3, is_ratio=0.7)
        assert len(windows) == 3
        for w in windows:
            # IS 结束位置 == OOS 开始位置（首尾相接，无重叠）
            assert w["is_end"] == w["oos_start"]
            # IS 完全在 OOS 之前（无未来函数）
            assert w["is_end"] <= w["oos_start"]
            assert w["is_start"] < w["is_end"] < len(mock_data)

    def test_is_oos_ratio(self, mock_data):
        """IS 占窗口总长比例接近 is_ratio。"""
        windows = WalkForwardOptimizer.split_windows(mock_data, n_windows=3, is_ratio=0.7)
        for w in windows[:-1]:  # 最后一个窗口 OOS 被拉伸，比例不强制
            is_len = w["is_end"] - w["is_start"]
            oos_len = w["oos_end"] - w["oos_start"]
            ratio = is_len / (is_len + oos_len)
            assert ratio == pytest.approx(0.7, abs=0.05)

    def test_last_oos_reaches_end(self, mock_data):
        """最后一个窗口的 OOS 到数据末尾。"""
        n_total = len(mock_data)
        windows = WalkForwardOptimizer.split_windows(mock_data, n_windows=3, is_ratio=0.7)
        assert windows[-1]["oos_end"] == n_total

    def test_oos_segments_no_overlap(self, mock_data):
        """各窗口 OOS 段之间无重叠（按时间拼接）。"""
        windows = WalkForwardOptimizer.split_windows(mock_data, n_windows=3, is_ratio=0.7)
        for prev, cur in zip(windows[:-1], windows[1:]):
            # 上一段 OOS 结束位置 <= 下一段 OOS 开始位置
            assert prev["oos_end"] <= cur["oos_start"]

    def test_example_100_rows(self):
        """复现题目示例：100 条数据、2 窗口、is_ratio=0.7。"""
        df = _make_mock_data(n_days=100)
        windows = WalkForwardOptimizer.split_windows(df, n_windows=2, is_ratio=0.7)
        # 窗口1: IS[0:35] + OOS[35:50]
        assert windows[0]["is_start"] == 0
        assert windows[0]["is_end"] == 35
        assert windows[0]["oos_start"] == 35
        assert windows[0]["oos_end"] == 50

    def test_invalid_params_raise(self, mock_data):
        """非法 n_windows / is_ratio 抛异常。"""
        with pytest.raises(ValueError):
            WalkForwardOptimizer.split_windows(mock_data, n_windows=0, is_ratio=0.7)
        with pytest.raises(ValueError):
            WalkForwardOptimizer.split_windows(mock_data, n_windows=2, is_ratio=1.2)


# ---------------------------------------------------------------------------
# 完整滚动优化流程
# ---------------------------------------------------------------------------

class TestOptimizeFlow:
    """每个窗口 IS 优化 + OOS 测试。"""

    def test_result_structure(self, optimizer, mock_data, small_grid):
        """返回结果包含全部关键字段。"""
        result = optimizer.optimize(
            mock_data, MACrossStrategy, small_grid,
            n_windows=3, is_ratio=0.7, objective="sharpe", symbol="TEST",
        )
        for key in (
            "windows", "combined_oos_equity", "combined_metrics",
            "overfitting_report", "recommended_params", "param_heatmap",
        ):
            assert key in result

    def test_windows_have_best_params_and_metrics(self, optimizer, mock_data, small_grid):
        """每个窗口都有最优参数、IS 绩效、OOS 绩效。"""
        result = optimizer.optimize(
            mock_data, MACrossStrategy, small_grid,
            n_windows=3, is_ratio=0.7, symbol="TEST",
        )
        assert len(result["windows"]) >= 2
        for w in result["windows"]:
            assert "best_params" in w
            assert "fast_period" in w["best_params"]
            assert "slow_period" in w["best_params"]
            assert "is_metrics" in w
            assert "oos_metrics" in w
            assert "夏普比率" in w["is_metrics"]
            assert "夏普比率" in w["oos_metrics"]

    def test_unsupported_objective_raises(self, optimizer, mock_data, small_grid):
        """不支持的 objective 抛 ValueError。"""
        with pytest.raises(ValueError):
            optimizer.optimize(
                mock_data, MACrossStrategy, small_grid, objective="bogus",
            )


# ---------------------------------------------------------------------------
# OOS 合并
# ---------------------------------------------------------------------------

class TestCombinedOOS:
    """合并 OOS 净值曲线。"""

    def test_combined_equity_no_overlap(self, optimizer, mock_data, small_grid):
        """合并后的净值曲线时间索引无重复、单调递增排序。"""
        result = optimizer.optimize(
            mock_data, MACrossStrategy, small_grid,
            n_windows=3, is_ratio=0.7, symbol="TEST",
        )
        eq = result["combined_oos_equity"]
        assert len(eq) > 0
        # 无重复时间索引
        assert eq.index.is_unique
        # 时间按升序
        assert eq.index.is_monotonic_increasing

    def test_combined_equity_chainable(self, optimizer, mock_data, small_grid):
        """合并净值曲线可计算累计收益（至少 2 个点）。"""
        result = optimizer.optimize(
            mock_data, MACrossStrategy, small_grid,
            n_windows=3, is_ratio=0.7, symbol="TEST",
        )
        eq = result["combined_oos_equity"]
        assert len(eq) >= 2
        # 起点归一为 1.0 附近
        assert float(eq.iloc[0]) == pytest.approx(1.0, abs=0.05)

    def test_combined_metrics_present(self, optimizer, mock_data, small_grid):
        """合并 OOS 指标包含核心字段。"""
        result = optimizer.optimize(
            mock_data, MACrossStrategy, small_grid,
            n_windows=3, is_ratio=0.7, symbol="TEST",
        )
        m = result["combined_metrics"]
        assert "累计收益率" in m
        assert "年化收益率" in m
        assert "夏普比率" in m
        assert "卡玛比率" in m


# ---------------------------------------------------------------------------
# 过拟合检测
# ---------------------------------------------------------------------------

class TestOverfitting:
    """过拟合风险评估。"""

    def test_overfitting_report_structure(self, optimizer, mock_data, small_grid):
        """完整跑一遍后报告字段齐全。"""
        result = optimizer.optimize(
            mock_data, MACrossStrategy, small_grid,
            n_windows=3, is_ratio=0.7, symbol="TEST",
        )
        rep = result["overfitting_report"]
        assert rep["risk_level"] in ("low", "medium", "high", "unknown")
        assert "oos_is_ratio" in rep
        assert "param_stability_score" in rep

    def test_high_risk_when_oos_much_worse(self, optimizer):
        """构造 IS 高 / OOS 低的窗口，应标记 high 风险。"""
        windows = [
            {"is_objective": 2.0, "oos_objective": 0.3},   # 比值 0.15 < 0.5
            {"is_objective": 1.5, "oos_objective": 0.2},   # 比值 0.13
        ]
        history = [
            {"fast_period": 5, "slow_period": 20},
            {"fast_period": 8, "slow_period": 25},
        ]
        rep = optimizer._overfitting_report(windows, history)
        assert rep["risk_level"] == "high"
        assert rep["oos_is_ratio"] < 0.5

    def test_low_risk_when_oos_close_to_is(self, optimizer):
        """IS/OOS 绩效接近时标记 low 风险。"""
        windows = [
            {"is_objective": 1.0, "oos_objective": 0.9},
            {"is_objective": 1.2, "oos_objective": 1.0},
        ]
        history = [{"a": 5}, {"a": 5}]
        rep = optimizer._overfitting_report(windows, history)
        assert rep["risk_level"] == "low"


# ---------------------------------------------------------------------------
# 随机搜索
# ---------------------------------------------------------------------------

class TestRandomSearch:
    """随机搜索采样。"""

    def test_random_search_returns_structure(self, optimizer, mock_data):
        """随机搜索正常返回结果结构。"""
        grid = {"fast_period": [3, 5, 8, 10], "slow_period": [15, 20, 30, 40]}
        result = optimizer.random_search(
            mock_data, MACrossStrategy, grid,
            n_samples=5, n_windows=2, is_ratio=0.7, symbol="TEST",
        )
        assert "windows" in result
        assert len(result["windows"]) >= 1

    def test_random_search_samples_n_combos(self, optimizer, mock_data, monkeypatch):
        """随机搜索实际采样数量 == n_samples。"""
        grid = {"fast_period": [3, 5, 8, 10], "slow_period": [15, 20, 30, 40]}
        captured: Dict[str, Any] = {}

        orig = WalkForwardOptimizer._search_best_on_is

        def spy(self, is_data, strategy_class, param_grid, objective,
                symbol, sampled_combos, **kw):
            captured["sampled"] = sampled_combos
            return orig(self, is_data, strategy_class, param_grid, objective,
                        symbol, sampled_combos, **kw)

        monkeypatch.setattr(WalkForwardOptimizer, "_search_best_on_is", spy)
        optimizer.random_search(
            mock_data, MACrossStrategy, grid,
            n_samples=5, n_windows=2, symbol="TEST",
        )
        assert captured["sampled"] is not None
        assert len(captured["sampled"]) == 5


# ---------------------------------------------------------------------------
# 推荐参数
# ---------------------------------------------------------------------------

class TestRecommendedParams:
    """推荐参数（中位数 / 众数）。"""

    def test_numeric_median(self, optimizer):
        """数值参数取中位数。"""
        history = [
            {"fast_period": 3, "slow_period": 15},
            {"fast_period": 5, "slow_period": 20},
            {"fast_period": 8, "slow_period": 30},
        ]
        rec = optimizer._recommend_params(history)
        # fast: 中位数=5; slow: 中位数=20
        assert rec["fast_period"] == 5
        assert rec["slow_period"] == 20

    def test_categorical_mode(self, optimizer):
        """分类参数取众数。"""
        history = [
            {"mode": "conservative"},
            {"mode": "aggressive"},
            {"mode": "conservative"},
        ]
        rec = optimizer._recommend_params(history)
        assert rec["mode"] == "conservative"

    def test_recommended_in_result(self, optimizer, mock_data, small_grid):
        """完整跑一遍后推荐参数落在参数网格内。"""
        result = optimizer.optimize(
            mock_data, MACrossStrategy, small_grid,
            n_windows=3, is_ratio=0.7, symbol="TEST",
        )
        rec = result["recommended_params"]
        assert rec["fast_period"] in small_grid["fast_period"] or \
               rec["fast_period"] in (3, 5)
        assert rec["slow_period"] in small_grid["slow_period"] or \
               rec["slow_period"] in (15, 20)


# ---------------------------------------------------------------------------
# 热力图
# ---------------------------------------------------------------------------

class TestParamHeatmap:
    """参数热力图聚合。"""

    def test_heatmap_structure(self, optimizer, mock_data, small_grid):
        """热力图按 参数名 -> {取值: 平均OOS绩效} 组织。"""
        result = optimizer.optimize(
            mock_data, MACrossStrategy, small_grid,
            n_windows=3, is_ratio=0.7, symbol="TEST",
        )
        hm = result["param_heatmap"]
        assert "fast_period" in hm
        assert "slow_period" in hm
        # 每个取值映射到一个数值
        for pname, val_map in hm.items():
            for value_str, avg_perf in val_map.items():
                assert isinstance(avg_perf, float)
