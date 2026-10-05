"""组合权重优化器单元测试。

运行:
    cd quant_trading_system
    python -m pytest tests/test_portfolio_optimizer.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimization.portfolio_optimizer import (
    ALLOWED_METHODS,
    OptimizeResult,
    PortfolioOptimizer,
)


# ---------------------------------------------------------------------------
# 合成数据构造
# ---------------------------------------------------------------------------

def make_synthetic_data(
    n_assets: int = 5,
    n_days: int = 1000,
    seed: int = 42,
    drifts: bool = True,
) -> "tuple[list[str], dict]":
    """构造多只标的的合成价格序列（不同波动率 + 相关性）。

    Returns:
        (symbols, data) —— data 为 {symbol: DataFrame(close)}。
    """
    rng = np.random.default_rng(seed)
    # 每只标的不同年化波动率：8% ~ 30%
    ann_vols = np.linspace(0.08, 0.30, n_assets)
    daily_vols = ann_vols / np.sqrt(252)
    # 不同年化漂移：正收益信号
    daily_drifts = (
        np.linspace(0.0002, 0.0008, n_assets) if drifts else np.zeros(n_assets)
    )

    # 基础独立收益
    rets = rng.normal(0, 1.0, size=(n_days, n_assets)) * daily_vols + daily_drifts
    # 注入公共市场因子，制造相关性
    market = rng.normal(0, 1.0, size=n_days) * 0.01
    for i in range(n_assets):
        rets[:, i] += 0.5 * market  # 与市场因子相关

    dates = pd.bdate_range("2023-01-02", periods=n_days)
    symbols = [f"SYM{i}" for i in range(n_assets)]
    data: dict = {}
    for i, s in enumerate(symbols):
        price = 100.0 * np.cumprod(1.0 + rets[:, i])
        data[s] = pd.DataFrame({"close": price}, index=dates)
    return symbols, data


# ---------------------------------------------------------------------------
# 权重合法性断言工具
# ---------------------------------------------------------------------------

def assert_valid_weights(weights: dict, expected_sum: float = 1.0,
                        max_weight: float = 0.3, tol: float = 1e-6) -> None:
    """校验权重：和≈1、非负、单标的不超上限。"""
    assert len(weights) > 0
    s = sum(weights.values())
    assert s == pytest.approx(expected_sum, abs=1e-6), f"权重和={s}"
    for sym, w in weights.items():
        assert w >= -tol, f"{sym} 权重为负: {w}"
        assert w <= max_weight + tol, f"{sym} 权重超上限: {w} > {max_weight}"


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestPortfolioOptimizer:
    """组合优化器基础行为测试。"""

    def test_all_four_methods_return_valid_weights(self):
        """四种方法都返回有效权重（和≈1、非负、不超上限）。"""
        symbols, data = make_synthetic_data(n_assets=5)
        opt = PortfolioOptimizer(risk_free_rate=0.02, max_weight=0.3)
        for method in ALLOWED_METHODS:
            res = opt.optimize(symbols, data, method=method)
            assert isinstance(res, OptimizeResult)
            assert res.method == method
            assert_valid_weights(res.weights, max_weight=0.3)
            # 各标的都有权重
            assert set(res.weights.keys()) == set(symbols)

    def test_equal_weight_is_uniform(self):
        """等权方法返回 1/N。"""
        symbols, data = make_synthetic_data(n_assets=4)
        opt = PortfolioOptimizer()
        res = opt.optimize(symbols, data, method="equal_weight")
        expected = 1.0 / 4.0
        for w in res.weights.values():
            assert w == pytest.approx(expected, abs=1e-9)

    def test_invalid_method_raises(self):
        """非法方法名抛出 ValueError。"""
        symbols, data = make_synthetic_data()
        opt = PortfolioOptimizer()
        with pytest.raises(ValueError):
            opt.optimize(symbols, data, method="not_a_method")


class TestRiskParity:
    """风险平价专项测试。"""

    def test_risk_contributions_nearly_equal(self):
        """风险平价：各标的风险贡献相对差异 < 5%。

        用宽松上限（不触顶），使等风险贡献可行。
        """
        symbols, data = make_synthetic_data(n_assets=5, seed=7)
        opt = PortfolioOptimizer(risk_free_rate=0.02, max_weight=1.0)
        res = opt.optimize(symbols, data, method="risk_parity")
        assert_valid_weights(res.weights, max_weight=1.0)

        mu, cov = opt._estimate_mu_cov(symbols, data)
        rc = opt.risk_contributions(res.weights, cov, symbols)
        vals = np.array(list(rc.values()))
        spread = (vals.max() - vals.min()) / vals.mean()
        assert spread < 0.05, f"风险贡献差异过大: {spread:.3f}, RC={rc}"


class TestMinVariance:
    """最小方差专项测试。"""

    def test_min_variance_vol_le_equal_weight(self):
        """最小方差组合的年化波动率 ≤ 等权组合波动率。"""
        symbols, data = make_synthetic_data(n_assets=6, seed=11)
        opt = PortfolioOptimizer(risk_free_rate=0.02, max_weight=0.3)
        eq = opt.optimize(symbols, data, method="equal_weight")
        mv = opt.optimize(symbols, data, method="min_variance")
        assert_valid_weights(mv.weights, max_weight=0.3)
        assert mv.expected_volatility <= eq.expected_volatility + 1e-8, (
            f"最小方差波动 {mv.expected_volatility:.4f} > 等权 {eq.expected_volatility:.4f}"
        )


class TestMeanVariance:
    """均值方差专项测试。"""

    def test_mean_variance_sharpe_ge_equal_weight(self):
        """均值方差夏普 ≥ 等权夏普（数据有正向漂移）。"""
        symbols, data = make_synthetic_data(n_assets=6, seed=3, drifts=True)
        opt = PortfolioOptimizer(risk_free_rate=0.02, max_weight=0.3)
        eq = opt.optimize(symbols, data, method="equal_weight")
        mv = opt.optimize(symbols, data, method="mean_variance")
        assert_valid_weights(mv.weights, max_weight=0.3)
        assert mv.sharpe >= eq.sharpe - 1e-6, (
            f"均值方差夏普 {mv.sharpe:.4f} < 等权 {eq.sharpe:.4f}"
        )


class TestConstraints:
    """约束生效测试。"""

    def test_max_weight_respected(self):
        """单标的权重不超过 max_weight（含优化类方法）。"""
        symbols, data = make_synthetic_data(n_assets=5, seed=5)
        cap = 0.3
        opt = PortfolioOptimizer(risk_free_rate=0.02, max_weight=cap)
        for method in ("min_variance", "risk_parity", "mean_variance"):
            res = opt.optimize(symbols, data, method=method)
            for sym, w in res.weights.items():
                assert w <= cap + 1e-6, f"{method}/{sym} 权重 {w} 超过上限 {cap}"

    def test_min_weight_long_only(self):
        """long-only：所有权重非负。"""
        symbols, data = make_synthetic_data(n_assets=5, seed=9)
        opt = PortfolioOptimizer(risk_free_rate=0.02, max_weight=0.3, min_weight=0.0)
        for method in ALLOWED_METHODS:
            res = opt.optimize(symbols, data, method=method)
            for w in res.weights.values():
                assert w >= -1e-9


class TestEdgeCases:
    """边界情况测试。"""

    def test_empty_symbols(self):
        """空输入返回空权重、不崩溃。"""
        opt = PortfolioOptimizer()
        res = opt.optimize([], {}, method="min_variance")
        assert res.weights == {}
        assert res.expected_return == 0.0
        assert res.expected_volatility == 0.0

    def test_single_symbol(self):
        """单标的：权重 1.0，不崩溃。"""
        symbols, data = make_synthetic_data(n_assets=1)
        opt = PortfolioOptimizer(risk_free_rate=0.02, max_weight=0.3)
        for method in ALLOWED_METHODS:
            res = opt.optimize(symbols, data, method=method)
            assert len(res.weights) == 1
            w = list(res.weights.values())[0]
            assert w == pytest.approx(1.0, abs=1e-6)

    def test_result_serializable(self):
        """as_dict 输出可 JSON 序列化（纯 float）。"""
        import json
        symbols, data = make_synthetic_data(n_assets=3)
        opt = PortfolioOptimizer()
        res = opt.optimize(symbols, data, method="risk_parity")
        d = res.as_dict()
        s = json.dumps(d)  # 不抛异常即通过
        assert "weights" in d and "sharpe" in d
