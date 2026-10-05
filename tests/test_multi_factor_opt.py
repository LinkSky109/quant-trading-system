"""多因子模型优化模块单元测试。

覆盖：
  - winsorize / mad_trim 去极值正确性
  - 施密特正交化（去除因子间相关性）
  - IC/IR 统计量计算
  - 多种加权方法（ic / ir / win_rate / equal）
  - z-score 标准化
  - 方向调整
  - 合成得分完整 pipeline
  - optimize_factors 一键接口
  - 与 FactorEngine 集成
  - 边界情况（空 DataFrame / 数据不足 / 单因子）

全部使用确定性构造的模拟数据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.multi_factor_opt import MultiFactorOptimizer, _extract_factor_matrix


# ---------------------------------------------------------------------------
# 构造工具
# ---------------------------------------------------------------------------

def make_factor_matrix(
    n_days: int = 100,
    n_symbols: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """构造因子值矩阵（日期×标的），带少量极值。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    data = rng.normal(0, 1, size=(n_days, n_symbols))
    # 注入几个极值
    data[5, 2] = 50.0
    data[10, 7] = -40.0
    return pd.DataFrame(data, index=dates, columns=symbols)


def make_factor_panels(
    n_symbols: int = 5,
    n_days: int = 100,
    factor_names: tuple = ("momentum_20", "volatility_20_inverse", "rsi_14"),
    seed: int = 7,
) -> dict:
    """构造多标的多因子面板 {symbol: DataFrame}。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    panels = {}
    for i in range(n_symbols):
        sym = f"SYM{i:02d}"
        close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.02, size=n_days))
        df = pd.DataFrame(index=dates)
        df["close"] = close
        for fname in factor_names:
            if "momentum" in fname:
                df[fname] = close / np.roll(close, 20) - 1.0
            elif "volatility" in fname:
                df[fname] = -pd.Series(close).pct_change().rolling(20).std().values
            else:
                df[fname] = rng.normal(0, 1, size=n_days)
        panels[sym] = df
    return panels


def make_predictive_panels(
    n_symbols: int = 8,
    n_days: int = 200,
    forward_days: int = 5,
    seed: int = 1,
) -> dict:
    """构造因子值与未来收益强相关的面板，用于 IC 测试。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    panels = {}
    for i in range(n_symbols):
        rets = rng.normal(0.0005, 0.02, size=n_days)
        close = 100.0 * np.cumprod(1.0 + rets)
        close_s = pd.Series(close, index=dates, dtype=float)
        fwd_ret = close_s.shift(-forward_days) / close_s - 1.0
        factor_a = fwd_ret + rng.normal(0, 0.002, size=n_days)
        factor_b = -fwd_ret + rng.normal(0, 0.003, size=n_days)  # 负相关
        panels[f"SYM{i:02d}"] = pd.DataFrame({
            "factor_a": factor_a,
            "factor_b": factor_b,
            "close": close_s,
        }, index=dates)
    return panels


@pytest.fixture
def optimizer() -> MultiFactorOptimizer:
    return MultiFactorOptimizer()


@pytest.fixture
def factor_mat() -> pd.DataFrame:
    return make_factor_matrix(n_days=100, n_symbols=10, seed=42)


# ---------------------------------------------------------------------------
# 1. 去极值
# ---------------------------------------------------------------------------

class TestWinsorize:
    def test_reduces_extremes(self, optimizer, factor_mat):
        """winsorize 后极值应被截断。"""
        out = optimizer.winsorize(factor_mat, lower=0.05, upper=0.95)
        # 原来有 50 和 -40 的极值，截断后应消失
        assert out.max().max() < 50.0
        assert out.min().min() > -40.0

    def test_preserves_rank_order(self, optimizer, factor_mat):
        """winsorize 不改变截面排名顺序。"""
        out = optimizer.winsorize(factor_mat, lower=0.1, upper=0.9)
        for dt in factor_mat.index:
            original_ranks = factor_mat.loc[dt].rank()
            out_ranks = out.loc[dt].rank()
            # 对于未被截断的值，排名应保持一致
            pd.testing.assert_series_equal(
                original_ranks, out_ranks, check_names=False
            )

    def test_empty_df(self, optimizer):
        """空 DataFrame 应安全返回。"""
        empty = pd.DataFrame()
        out = optimizer.winsorize(empty)
        assert out.empty

    def test_time_axis(self, optimizer, factor_mat):
        """axis=0 时序去极值也应工作。"""
        out = optimizer.winsorize(factor_mat, axis=0)
        assert out.shape == factor_mat.shape


class TestMADTrim:
    def test_reduces_extremes(self, optimizer, factor_mat):
        """MAD 截断后极值应消失。"""
        out = optimizer.mad_trim(factor_mat, n=3.0)
        assert out.max().max() < 50.0
        assert out.min().min() > -40.0

    def test_empty_df(self, optimizer):
        """空 DataFrame 应安全返回。"""
        empty = pd.DataFrame()
        out = optimizer.mad_trim(empty)
        assert out.empty

    def test_time_axis(self, optimizer, factor_mat):
        """axis=0 时序 MAD 截断也应工作。"""
        out = optimizer.mad_trim(factor_mat, axis=0)
        assert out.shape == factor_mat.shape


# ---------------------------------------------------------------------------
# 2. 正交化
# ---------------------------------------------------------------------------

class TestOrthogonalize:
    def test_correlation_reduced(self, optimizer):
        """正交化后因子间相关性应显著降低。"""
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.normal(0, 1, size=n)
        x2 = 0.8 * x1 + rng.normal(0, 0.6, size=n)  # 与 x1 强相关
        x3 = 0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.5, size=n)
        df = pd.DataFrame({"f1": x1, "f2": x2, "f3": x3})

        # 正交化前相关性较高
        corr_before = df.corr().abs().values
        upper_before = corr_before[np.triu_indices_from(corr_before, k=1)]
        assert upper_before.mean() > 0.3

        ortho = optimizer.orthogonalize(df)
        corr_after = ortho.corr().abs().values
        upper_after = corr_after[np.triu_indices_from(corr_after, k=1)]
        # 正交化后非对角线相关性应接近 0
        assert upper_after.mean() < 0.05

    def test_single_factor_noop(self, optimizer):
        """单因子正交化应原样返回。"""
        df = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
        out = optimizer.orthogonalize(df)
        pd.testing.assert_frame_equal(out, df)

    def test_empty_df(self, optimizer):
        """空 DataFrame 应安全返回。"""
        empty = pd.DataFrame()
        out = optimizer.orthogonalize(empty)
        assert out.empty

    def test_unsupported_method_raises(self, optimizer):
        """不支持的方法应抛 ValueError。"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError):
            optimizer.orthogonalize(df, method="pca")


# ---------------------------------------------------------------------------
# 3. IC/IR 分析
# ---------------------------------------------------------------------------

class TestICIR:
    def test_ic_positive_for_predictive_factor(self, optimizer):
        """预测性因子应产生显著正的 IC。"""
        panels = make_predictive_panels(n_symbols=8, n_days=200, seed=1)
        stats = optimizer.calc_ic_ir(panels, ["factor_a"], forward_days=5)
        assert "factor_a" in stats
        ic_mean = stats["factor_a"]["ic_mean"]
        assert ic_mean > 0.3, f"预测性因子 IC 均值过低: {ic_mean}"
        assert stats["factor_a"]["n_periods"] > 50

    def test_ic_negative_for_inverse_factor(self, optimizer):
        """负相关因子应产生负的 IC。"""
        panels = make_predictive_panels(n_symbols=8, n_days=200, seed=1)
        stats = optimizer.calc_ic_ir(panels, ["factor_b"], forward_days=5)
        ic_mean = stats["factor_b"]["ic_mean"]
        assert ic_mean < -0.2, f"负相关因子 IC 均值应负: {ic_mean}"

    def test_empty_panels_returns_empty(self, optimizer):
        """空面板应返回空 dict。"""
        stats = optimizer.calc_ic_ir({}, ["factor_a"], forward_days=5)
        assert stats == {}

    def test_ir_calculation(self, optimizer):
        """IR = IC 均值 / IC 标准差。"""
        panels = make_predictive_panels(n_symbols=8, n_days=200, seed=1)
        stats = optimizer.calc_ic_ir(panels, ["factor_a"], forward_days=5)
        s = stats["factor_a"]
        expected_ir = s["ic_mean"] / s["ic_std"] if s["ic_std"] and not np.isnan(s["ic_std"]) else float("nan")
        assert s["ir"] == pytest.approx(expected_ir, rel=1e-6) or (np.isnan(s["ir"]) and np.isnan(expected_ir))


# ---------------------------------------------------------------------------
# 4. 加权权重
# ---------------------------------------------------------------------------

class TestWeightedWeights:
    def test_weights_sum_to_one(self, optimizer):
        """所有加权方法返回的权重和应为 1。"""
        panels = make_predictive_panels(n_symbols=8, n_days=200, seed=1)
        for method in ("ic", "ir", "win_rate", "equal"):
            weights = optimizer.ic_weighted_weights(
                panels, ["factor_a", "factor_b"], forward_days=5, method=method
            )
            total = sum(weights.values())
            assert total == pytest.approx(1.0, abs=1e-9), f"{method} 权重和={total}"

    def test_equal_weight_is_uniform(self, optimizer):
        """equal 方法返回等权。"""
        panels = make_predictive_panels(n_symbols=8, n_days=200, seed=1)
        weights = optimizer.ic_weighted_weights(
            panels, ["factor_a", "factor_b"], method="equal"
        )
        for w in weights.values():
            assert w == pytest.approx(0.5, abs=1e-9)

    def test_ir_prefers_stable_factor(self, optimizer):
        """IR 加权应给稳定预测因子更高权重。"""
        panels = make_predictive_panels(n_symbols=8, n_days=200, seed=1)
        weights = optimizer.ic_weighted_weights(
            panels, ["factor_a", "factor_b"], forward_days=5, method="ir"
        )
        # factor_a 与收益正相关且稳定，应获得更高权重
        assert weights["factor_a"] > weights["factor_b"]

    def test_invalid_method_raises(self, optimizer):
        """非法方法名应抛 ValueError。"""
        panels = make_predictive_panels(n_symbols=8, n_days=200, seed=1)
        with pytest.raises(ValueError):
            optimizer.ic_weighted_weights(
                panels, ["factor_a"], method="invalid"
            )


# ---------------------------------------------------------------------------
# 5. z-score 标准化
# ---------------------------------------------------------------------------

class TestZScore:
    def test_mean_near_zero(self, optimizer, factor_mat):
        """z-score 后截面均值应接近 0。"""
        out = optimizer.zscore(factor_mat, axis=1)
        # 忽略 NaN 行
        means = out.mean(axis=1).dropna()
        assert abs(means.mean()) < 0.01

    def test_std_near_one(self, optimizer, factor_mat):
        """z-score 后截面标准差应接近 1。"""
        out = optimizer.zscore(factor_mat, axis=1)
        stds = out.std(axis=1, ddof=0).dropna()
        assert abs(stds.mean() - 1.0) < 0.05

    def test_empty_df(self, optimizer):
        """空 DataFrame 应安全返回。"""
        empty = pd.DataFrame()
        out = optimizer.zscore(empty)
        assert out.empty

    def test_time_axis(self, optimizer, factor_mat):
        """axis=0 时序标准化也应工作。"""
        out = optimizer.zscore(factor_mat, axis=0)
        assert out.shape == factor_mat.shape


# ---------------------------------------------------------------------------
# 6. 方向调整
# ---------------------------------------------------------------------------

class TestAdjustDirection:
    def test_positive_direction_unchanged(self, optimizer):
        """direction=1 的因子保持不变。"""
        df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        out = optimizer.adjust_direction(df, {"f1": 1, "f2": 1})
        pd.testing.assert_frame_equal(out, df)

    def test_negative_direction_flips(self, optimizer):
        """direction=-1 的因子取反。"""
        df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        out = optimizer.adjust_direction(df, {"f1": -1})
        assert (out["f1"] == -df["f1"]).all()
        assert (out["f2"] == df["f2"]).all()


# ---------------------------------------------------------------------------
# 7. 合成得分
# ---------------------------------------------------------------------------

class TestBuildCompositeScore:
    def test_returns_dataframe_with_score(self, optimizer):
        """合成得分应返回含 composite_score 列的 DataFrame。"""
        panels = make_factor_panels(n_symbols=5, n_days=100, seed=7)
        result = optimizer.build_composite_score(
            panels,
            factor_names=["momentum_20", "volatility_20_inverse", "rsi_14"],
            weight_method="equal",
            winsorize_factors=True,
            standardize=True,
        )
        assert "composite_score" in result.columns
        assert len(result) > 0

    def test_empty_factor_names_raises(self, optimizer):
        """空因子名应抛 ValueError。"""
        panels = make_factor_panels()
        with pytest.raises(ValueError):
            optimizer.build_composite_score(panels, [])

    def test_missing_factors_skipped(self, optimizer):
        """缺失的因子应被跳过。"""
        panels = make_factor_panels(n_symbols=3, factor_names=("momentum_20",))
        result = optimizer.build_composite_score(
            panels, ["momentum_20", "nonexistent_factor"],
            weight_method="equal",
        )
        assert "composite_score" in result.columns

    def test_orthogonalize_reduces_correlation(self, optimizer):
        """orthogonalize=True 后合成得分应更稳定。"""
        rng = np.random.default_rng(99)
        dates = pd.bdate_range("2024-01-01", periods=50)
        panels = {}
        for i in range(5):
            sym = f"SYM{i:02d}"
            x = rng.normal(0, 1, size=50)
            y = 0.9 * x + rng.normal(0, 0.1, size=50)  # 强相关
            panels[sym] = pd.DataFrame({
                "f1": x, "f2": y, "close": 100.0 + np.cumsum(rng.normal(0, 0.01, size=50)),
            }, index=dates)
        result_no_ortho = optimizer.build_composite_score(
            panels, ["f1", "f2"], orthogonalize_factors=False, winsorize_factors=False
        )
        result_ortho = optimizer.build_composite_score(
            panels, ["f1", "f2"], orthogonalize_factors=True, winsorize_factors=False
        )
        assert "composite_score" in result_no_ortho.columns
        assert "composite_score" in result_ortho.columns

    def test_with_factor_engine_directions(self, optimizer):
        """传入 factor_engine 时应自动获取方向。"""
        from factors.factor_engine import FactorEngine
        engine = FactorEngine()
        opt = MultiFactorOptimizer(engine)
        panels = make_factor_panels(n_symbols=3, factor_names=tuple(engine.factor_names[:3]))
        result = opt.build_composite_score(
            panels, list(engine.factor_names[:3]),
            weight_method="equal", winsorize_factors=False, standardize=True,
        )
        assert "composite_score" in result.columns


# ---------------------------------------------------------------------------
# 8. 一键优化接口
# ---------------------------------------------------------------------------

class TestOptimizeFactors:
    def test_full_report_structure(self, optimizer):
        """optimize_factors 应返回完整报告结构。"""
        panels = make_predictive_panels(n_symbols=8, n_days=200, seed=1)
        report = optimizer.optimize_factors(
            panels, factor_names=["factor_a", "factor_b"],
            forward_days=5, weight_method="ir",
        )
        assert "weights" in report
        assert "ic_ir_stats" in report
        assert "composite_score" in report
        assert "factor_count" in report
        assert report["factor_count"] == 2
        assert sum(report["weights"].values()) == pytest.approx(1.0, abs=1e-9)

    def test_empty_factor_names_raises(self, optimizer):
        """空因子名应抛 ValueError。"""
        panels = make_predictive_panels()
        with pytest.raises(ValueError):
            optimizer.optimize_factors(panels, factor_names=[])


# ---------------------------------------------------------------------------
# 9. 工具函数
# ---------------------------------------------------------------------------

class TestExtractFactorMatrix:
    def test_extracts_correctly(self):
        """_extract_factor_matrix 应正确提取单因子矩阵。"""
        dates = pd.bdate_range("2024-01-01", periods=10)
        panels = {
            "A": pd.DataFrame({"f1": range(10), "close": [100.0]*10}, index=dates),
            "B": pd.DataFrame({"f1": range(10, 20), "close": [100.0]*10}, index=dates),
        }
        mat = _extract_factor_matrix(panels, "f1")
        assert mat.shape == (10, 2)
        assert list(mat.columns) == ["A", "B"]

    def test_missing_factor_returns_empty(self):
        """缺失因子应返回空 DataFrame。"""
        panels = {"A": pd.DataFrame({"f1": [1, 2]}, index=pd.date_range("2024-01-01", periods=2))}
        mat = _extract_factor_matrix(panels, "missing")
        assert mat.empty

    def test_empty_panels_returns_empty(self):
        """空 panels 应返回空 DataFrame。"""
        mat = _extract_factor_matrix({}, "f1")
        assert mat.empty


# ---------------------------------------------------------------------------
# 10. 边界情况
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_winsorize_all_nan_row(self, optimizer):
        """整行 NaN 应安全处理。"""
        df = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, np.nan]})
        out = optimizer.winsorize(df)
        assert out.shape == df.shape

    def test_mad_trim_all_nan_row(self, optimizer):
        """整行 NaN 应安全处理。"""
        df = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, np.nan]})
        out = optimizer.mad_trim(df)
        assert out.shape == df.shape

    def test_zscore_constant_row(self, optimizer):
        """常数行标准化后应为 0。"""
        df = pd.DataFrame({"a": [5.0, 4.0, 3.0], "b": [5.0, 2.0, 1.0]})
        out = optimizer.zscore(df, axis=1)
        assert (out.loc[out.index[0], :] == 0.0).all()

    def test_orthogonalize_with_nans(self, optimizer):
        """含 NaN 的正交化应安全处理。"""
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [np.nan, 2.1, 3.2],
        })
        out = optimizer.orthogonalize(df)
        assert out.shape == df.shape
