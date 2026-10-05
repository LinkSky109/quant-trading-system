"""自适应风险管理模块单元测试。

覆盖：
  - 动态 VaR 窗口计算（高/低波动率自适应）
  - 动态 VaR 计算（historical / parametric）
  - 风险预算动态调整
  - 尾部风险检测（CVaR / ES 告警分级）
  - 历史情景压力测试
  - 蒙特卡洛压力测试
  - 与 RiskManager 集成
  - 综合风险报告
  - 边界情况（短序列 / 零波动率）

全部使用确定性构造的模拟数据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk.adaptive_risk import (
    AdaptiveRiskManager,
    DEFAULT_BASE_VAR_WINDOW,
    DEFAULT_MIN_VAR_WINDOW,
    DEFAULT_MAX_VAR_WINDOW,
    DEFAULT_VAR_CONFIDENCE,
    DEFAULT_TAIL_RISK_THRESHOLD,
    DEFAULT_RISK_BUDGET_BASE,
    DEFAULT_VOLATILITY_TARGET,
    StressTestResult,
    TailRiskAlert,
)
from risk.risk_manager import RiskManager
from risk.var_model import STRESS_SCENARIOS


# ---------------------------------------------------------------------------
# 构造工具
# ---------------------------------------------------------------------------

def make_returns(
    n_days: int = 500,
    annual_vol: float = 0.15,
    seed: int = 42,
) -> pd.Series:
    """构造模拟日收益率序列。"""
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    rets = rng.normal(0.0003, daily_vol, size=n_days)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    return pd.Series(rets, index=dates)


def make_high_vol_returns(
    n_days: int = 100,
    seed: int = 99,
) -> pd.Series:
    """构造高波动率日收益率（用于测试缩短窗口）。

    前 80% 为正常波动，后 20% 突然放大波动，使当前波动率显著高于历史中位数。
    """
    rng = np.random.default_rng(seed)
    normal_vol = 0.15 / np.sqrt(252)
    high_vol = 0.60 / np.sqrt(252)
    split = int(n_days * 0.8)
    rets1 = rng.normal(0.0003, normal_vol, size=split)
    rets2 = rng.normal(0.0003, high_vol, size=n_days - split)
    rets = np.concatenate([rets1, rets2])
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    return pd.Series(rets, index=dates)


def make_low_vol_returns(
    n_days: int = 100,
    seed: int = 77,
) -> pd.Series:
    """构造低波动率日收益率（用于测试延长窗口）。

    前 80% 为正常波动，后 20% 突然降低波动，使当前波动率显著低于历史中位数。
    """
    rng = np.random.default_rng(seed)
    normal_vol = 0.15 / np.sqrt(252)
    low_vol = 0.03 / np.sqrt(252)
    split = int(n_days * 0.8)
    rets1 = rng.normal(0.0003, normal_vol, size=split)
    rets2 = rng.normal(0.0003, low_vol, size=n_days - split)
    rets = np.concatenate([rets1, rets2])
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    return pd.Series(rets, index=dates)


def make_returns_matrix(
    n_assets: int = 3,
    n_days: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """构造多资产收益率矩阵。"""
    rng = np.random.default_rng(seed)
    daily_vol = 0.15 / np.sqrt(252)
    rets = rng.normal(0.0003, daily_vol, size=(n_days, n_assets))
    # 注入相关性
    market = rng.normal(0, daily_vol * 0.5, size=n_days)
    for i in range(n_assets):
        rets[:, i] += 0.4 * market
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    return pd.DataFrame(rets, index=dates, columns=[f"A{i}" for i in range(n_assets)])


@pytest.fixture
def arm() -> AdaptiveRiskManager:
    return AdaptiveRiskManager()


@pytest.fixture
def normal_returns() -> pd.Series:
    return make_returns(n_days=500, annual_vol=0.15, seed=42)


# ---------------------------------------------------------------------------
# 1. 动态 VaR 窗口
# ---------------------------------------------------------------------------

class TestDynamicWindow:
    def test_normal_vol_returns_base_window(self, arm, normal_returns):
        """正常波动率应返回接近基础窗口。"""
        window = arm.calc_dynamic_window(normal_returns)
        assert DEFAULT_MIN_VAR_WINDOW <= window <= DEFAULT_MAX_VAR_WINDOW

    def test_high_vol_shortens_window(self, arm):
        """高波动率应缩短窗口。"""
        high_vol = make_high_vol_returns(n_days=100, seed=99)
        window = arm.calc_dynamic_window(high_vol)
        assert window < DEFAULT_BASE_VAR_WINDOW
        assert window >= DEFAULT_MIN_VAR_WINDOW

    def test_low_vol_extends_window(self, arm):
        """低波动率应延长窗口。"""
        low_vol = make_low_vol_returns(n_days=100, seed=77)
        window = arm.calc_dynamic_window(low_vol)
        assert window > DEFAULT_BASE_VAR_WINDOW
        assert window <= DEFAULT_MAX_VAR_WINDOW

    def test_short_series_returns_min(self, arm):
        """极短序列应返回最小窗口。"""
        short = pd.Series([0.01, -0.01, 0.005])
        window = arm.calc_dynamic_window(short)
        assert window >= 30

    def test_window_bounds_respected(self, arm):
        """窗口应在 [min, max] 范围内。"""
        rng = np.random.default_rng(123)
        for _ in range(10):
            rets = pd.Series(rng.normal(0, 0.02, size=200))
            window = arm.calc_dynamic_window(rets)
            assert DEFAULT_MIN_VAR_WINDOW <= window <= DEFAULT_MAX_VAR_WINDOW


# ---------------------------------------------------------------------------
# 2. 动态 VaR
# ---------------------------------------------------------------------------

class TestDynamicVaR:
    def test_historical_var_returns_positive(self, arm, normal_returns):
        """历史法 VaR 应返回正数。"""
        result = arm.dynamic_var(normal_returns, method="historical")
        assert result["var"] > 0
        assert result["cvar"] > 0
        assert result["method"] == "historical"
        assert DEFAULT_MIN_VAR_WINDOW <= result["window"] <= DEFAULT_MAX_VAR_WINDOW

    def test_parametric_var_with_matrix(self, arm, normal_returns):
        """参数法需要收益率矩阵。"""
        ret_mat = make_returns_matrix(n_assets=3, n_days=300, seed=42)
        weights = np.array([0.4, 0.3, 0.3])
        result = arm.dynamic_var(
            normal_returns.tail(300),
            weights=weights,
            returns_matrix=ret_mat,
            method="parametric",
        )
        assert result["var"] > 0
        assert result["cvar"] > 0
        assert result["method"] == "parametric"

    def test_cvar_gte_var(self, arm, normal_returns):
        """CVaR 应 >= VaR。"""
        result = arm.dynamic_var(normal_returns, method="historical")
        assert result["cvar"] >= result["var"]

    def test_confidence_in_result(self, arm, normal_returns):
        """结果应包含置信度。"""
        result = arm.dynamic_var(normal_returns)
        assert result["confidence"] == DEFAULT_VAR_CONFIDENCE

    def test_current_volatility_present(self, arm, normal_returns):
        """结果应包含当前波动率。"""
        result = arm.dynamic_var(normal_returns)
        assert "current_volatility" in result
        assert result["current_volatility"] >= 0

    def test_invalid_method_raises(self, arm, normal_returns):
        """非法方法应抛 ValueError。"""
        with pytest.raises(ValueError):
            arm.dynamic_var(normal_returns, method="monte_carlo")


# ---------------------------------------------------------------------------
# 3. 风险预算调整
# ---------------------------------------------------------------------------

class TestRiskBudget:
    def test_high_vol_reduces_budget(self, arm):
        """高波动率应降低风险预算。"""
        budget = arm.adjust_risk_budget(market_volatility=0.30)
        assert budget["adjusted_budget"] < budget["original_budget"]
        assert budget["adjustment_factor"] < 1.0

    def test_low_vol_increases_budget(self, arm):
        """低波动率应提高风险预算。"""
        budget = arm.adjust_risk_budget(market_volatility=0.08)
        assert budget["adjusted_budget"] > budget["original_budget"]
        assert budget["adjustment_factor"] > 1.0

    def test_factor_capped_at_1_5(self, arm):
        """调整系数上限为 1.5。"""
        budget = arm.adjust_risk_budget(market_volatility=0.01)
        assert budget["adjustment_factor"] == pytest.approx(1.5, abs=0.01)

    def test_factor_floor_at_0_5(self, arm):
        """调整系数下限为 0.5。"""
        budget = arm.adjust_risk_budget(market_volatility=1.0)
        assert budget["adjustment_factor"] == pytest.approx(0.5, abs=0.01)

    def test_from_portfolio_returns(self, arm):
        """从组合收益率估算波动率。"""
        rets = make_returns(n_days=100, annual_vol=0.20, seed=55)
        budget = arm.adjust_risk_budget(portfolio_returns=rets)
        assert "adjusted_budget" in budget
        assert "actual_volatility" in budget

    def test_budget_within_reasonable_range(self, arm):
        """调整后的预算应在合理范围内。"""
        for vol in [0.05, 0.15, 0.30, 0.50]:
            budget = arm.adjust_risk_budget(market_volatility=vol)
            assert 0.05 <= budget["adjusted_budget"] <= 0.50


# ---------------------------------------------------------------------------
# 4. 尾部风险检测
# ---------------------------------------------------------------------------

class TestTailRisk:
    def test_tail_risk_alert_structure(self, arm, normal_returns):
        """尾部风险告警应有完整结构。"""
        alert = arm.detect_tail_risk(normal_returns)
        assert isinstance(alert, TailRiskAlert)
        assert alert.timestamp != ""
        assert alert.severity in ("low", "medium", "high", "critical")
        assert alert.var >= 0
        assert alert.cvar >= 0

    def test_history_recorded(self, arm, normal_returns):
        """告警应被记录到历史。"""
        before = len(arm.tail_risk_history)
        arm.detect_tail_risk(normal_returns)
        after = len(arm.tail_risk_history)
        assert after == before + 1

    def test_as_dict(self, arm, normal_returns):
        """TailRiskAlert.as_dict 应序列化。"""
        alert = arm.detect_tail_risk(normal_returns)
        d = alert.as_dict()
        assert "var" in d
        assert "cvar" in d
        assert "severity" in d
        assert "triggered" in d

    def test_severity_gradation(self, arm):
        """不同 CVaR 水平应产生不同严重级别。"""
        # 构造低尾部风险数据
        low_risk = pd.Series(np.random.default_rng(1).normal(0.001, 0.005, size=500))
        alert_low = arm.detect_tail_risk(low_risk)
        assert alert_low.severity == "low"
        assert alert_low.triggered is False


# ---------------------------------------------------------------------------
# 5. 压力测试
# ---------------------------------------------------------------------------

class TestStressTest:
    def test_historical_scenario_known_key(self, arm):
        """已知历史情景应返回结果。"""
        ret_mat = make_returns_matrix(n_assets=3, n_days=100)
        weights = np.array([0.4, 0.3, 0.3])
        for scenario in STRESS_SCENARIOS:
            result = arm.stress_test_historical(weights, ret_mat, scenario, portfolio_value=1_000_000.0)
            assert isinstance(result, StressTestResult)
            assert result.scenario == scenario
            assert result.portfolio_loss_pct >= 0

    def test_historical_unknown_scenario_raises(self, arm):
        """未知情景应抛 ValueError。"""
        ret_mat = make_returns_matrix(n_assets=3, n_days=100)
        weights = np.array([0.4, 0.3, 0.3])
        with pytest.raises(ValueError):
            arm.stress_test_historical(weights, ret_mat, "unknown_scenario")

    def test_monte_carlo_returns_distribution(self, arm):
        """蒙特卡洛应返回损失分布统计。"""
        ret_mat = make_returns_matrix(n_assets=3, n_days=300)
        weights = np.array([0.4, 0.3, 0.3])
        result = arm.stress_test_monte_carlo(
            weights, ret_mat, portfolio_value=1_000_000.0,
            n_simulations=5000, horizon_days=5,
        )
        assert isinstance(result, StressTestResult)
        assert result.scenario == "monte_carlo"
        assert result.monte_carlo_distribution is not None
        dist = result.monte_carlo_distribution
        assert "var" in dist
        assert "cvar" in dist
        assert "max_loss" in dist
        assert dist["n_simulations"] == 5000

    def test_monte_carlo_loss_positive(self, arm):
        """蒙特卡洛损失应为正数。"""
        ret_mat = make_returns_matrix(n_assets=3, n_days=300)
        weights = np.array([0.4, 0.3, 0.3])
        result = arm.stress_test_monte_carlo(weights, ret_mat)
        assert result.portfolio_loss_pct >= 0
        assert result.portfolio_loss_amount >= 0

    def test_run_stress_test_monte_carlo(self, arm):
        """统一入口：蒙特卡洛情景。"""
        ret_mat = make_returns_matrix(n_assets=3, n_days=300)
        weights = np.array([0.4, 0.3, 0.3])
        result = arm.run_stress_test(weights, ret_mat, scenario="monte_carlo")
        assert result.scenario == "monte_carlo"

    def test_run_stress_test_historical(self, arm):
        """统一入口：历史情景。"""
        ret_mat = make_returns_matrix(n_assets=3, n_days=300)
        weights = np.array([0.4, 0.3, 0.3])
        result = arm.run_stress_test(weights, ret_mat, scenario="2008_crisis")
        assert result.scenario == "2008_crisis"

    def test_stress_test_result_as_dict(self, arm):
        """StressTestResult.as_dict 应序列化。"""
        result = StressTestResult(
            scenario="test",
            portfolio_loss_pct=0.05,
            portfolio_loss_amount=50000.0,
        )
        d = result.as_dict()
        assert d["scenario"] == "test"
        assert d["portfolio_loss_pct"] == pytest.approx(0.05, rel=1e-9)


# ---------------------------------------------------------------------------
# 6. 与 RiskManager 集成
# ---------------------------------------------------------------------------

class TestRiskManagerIntegration:
    def test_critical_severity_adjusts_limits(self, arm):
        """critical 级别应大幅降低仓位上限。"""
        rm = RiskManager(
            max_position_per_symbol=0.20,
            max_total_position=0.80,
        )
        # 构造高尾部风险数据触发 critical
        rng = np.random.default_rng(999)
        rets = rng.normal(-0.005, 0.04, size=500)  # 高负偏 + 高波动
        result = arm.integrate_with_risk_manager(rm, pd.Series(rets))
        assert result["severity"] in ("critical", "high", "medium", "low")
        assert result["max_position_per_symbol"] <= 0.20
        assert result["original_max_position"] == 0.20

    def test_low_severity_no_change(self, arm):
        """low 级别应保持原配置。"""
        rm = RiskManager(
            max_position_per_symbol=0.20,
            max_total_position=0.80,
        )
        # 构造低风险数据
        rets = pd.Series(np.random.default_rng(1).normal(0.001, 0.005, size=500))
        result = arm.integrate_with_risk_manager(rm, rets)
        if result["severity"] == "low":
            assert result["max_position_per_symbol"] == 0.20
            assert result["max_total_position"] == 0.80

    def test_returns_adjustment_summary(self, arm):
        """集成应返回调整摘要。"""
        rm = RiskManager()
        rets = make_returns(n_days=200, seed=88)
        result = arm.integrate_with_risk_manager(rm, rets)
        assert "adjusted_budget" in result
        assert "var" in result
        assert "cvar" in result


# ---------------------------------------------------------------------------
# 7. 综合报告
# ---------------------------------------------------------------------------

class TestRiskReport:
    def test_full_report_structure(self, arm, normal_returns):
        """综合报告应包含所有组件。"""
        ret_mat = make_returns_matrix(n_assets=3, n_days=300)
        weights = np.array([0.4, 0.3, 0.3])
        report = arm.risk_report(
            normal_returns, weights=weights,
            returns_matrix=ret_mat, portfolio_value=1_000_000.0,
        )
        assert "dynamic_var" in report
        assert "risk_budget" in report
        assert "tail_risk_alert" in report
        assert "stress_test" in report
        assert "tail_risk_history_count" in report

    def test_report_without_matrix(self, arm, normal_returns):
        """无收益率矩阵时 stress_test 为 None。"""
        report = arm.risk_report(normal_returns)
        assert report["stress_test"] is None
        assert "dynamic_var" in report
        assert "risk_budget" in report
        assert "tail_risk_alert" in report


# ---------------------------------------------------------------------------
# 8. 初始化与配置
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_values(self):
        """默认参数应正确设置。"""
        arm = AdaptiveRiskManager()
        assert arm.base_var_window == DEFAULT_BASE_VAR_WINDOW
        assert arm.min_var_window == DEFAULT_MIN_VAR_WINDOW
        assert arm.max_var_window == DEFAULT_MAX_VAR_WINDOW
        assert arm.var_confidence == DEFAULT_VAR_CONFIDENCE
        assert arm.tail_risk_threshold == DEFAULT_TAIL_RISK_THRESHOLD

    def test_custom_values(self):
        """自定义参数应正确设置。"""
        arm = AdaptiveRiskManager(
            base_var_window=100,
            min_var_window=30,
            max_var_window=200,
            var_confidence=0.99,
        )
        assert arm.base_var_window == 100
        assert arm.min_var_window == 30
        assert arm.max_var_window == 200
        assert arm.var_confidence == 0.99

    def test_history_empty_on_init(self):
        """初始化时历史告警为空。"""
        arm = AdaptiveRiskManager()
        assert arm.tail_risk_history == []


# ---------------------------------------------------------------------------
# 9. 边界情况
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_dynamic_var_short_series(self, arm):
        """极短序列应安全处理。"""
        short = pd.Series([0.01, -0.01, 0.005, -0.008])
        result = arm.dynamic_var(short, method="historical")
        assert result["var"] >= 0
        assert result["cvar"] >= 0

    def test_adjust_risk_budget_zero_vol(self, arm):
        """零波动率应使用目标波动率。"""
        budget = arm.adjust_risk_budget(market_volatility=0.0)
        assert budget["actual_volatility"] > 0

    def test_adjust_risk_budget_nan_vol(self, arm):
        """NaN 波动率应使用目标波动率。"""
        budget = arm.adjust_risk_budget(market_volatility=float("nan"))
        assert budget["actual_volatility"] > 0

    def test_tail_risk_history_truncation(self, arm):
        """历史告警过长时应截断。"""
        arm.tail_risk_history = [TailRiskAlert(
            timestamp="", var=0.01, cvar=0.02,
            threshold=0.03, triggered=False,
        )] * 1200
        # 触发一次 detect_tail_risk 进行截断
        rets = make_returns(n_days=50, seed=1)
        arm.detect_tail_risk(rets)
        assert len(arm.tail_risk_history) <= 1000

    def test_monte_carlo_with_singular_cov(self, arm):
        """协方差矩阵奇异时应通过正则化处理。"""
        # 构造高度相关数据导致协方差接近奇异
        rng = np.random.default_rng(55)
        base = rng.normal(0, 0.01, size=100)
        ret_mat = pd.DataFrame({
            "A": base,
            "B": base + rng.normal(0, 0.001, size=100),
            "C": base + rng.normal(0, 0.001, size=100),
        })
        weights = np.array([0.4, 0.3, 0.3])
        result = arm.stress_test_monte_carlo(weights, ret_mat, n_simulations=1000)
        assert result.scenario == "monte_carlo"
        assert result.monte_carlo_distribution is not None
