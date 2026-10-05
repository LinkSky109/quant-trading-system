"""灰度发布与A/B测试框架测试。"""
from __future__ import annotations

import numpy as np
import pytest

from deployment.ab_testing import (
    ABTestingFramework,
    StrategyVersion,
    ExperimentConfig,
)


@pytest.fixture
def sample_experiment():
    control = StrategyVersion(name="ma_cross_v1", params={"fast": 5, "slow": 20}, code_hash="abc123")
    treatment = StrategyVersion(name="ma_cross_v2", params={"fast": 10, "slow": 30}, code_hash="def456")
    return ExperimentConfig(
        experiment_id="exp_001",
        control_version=control,
        treatment_version=treatment,
        traffic_split=0.5,
        capital_split=0.2,
        min_samples=10,
        rollback_threshold=-0.10,
    )


def test_strategy_version_hash():
    v = StrategyVersion(name="test", params={"a": 1}, code_hash="")
    h = v.compute_hash()
    assert len(h) == 16
    assert h == v.compute_hash()  # 确定性


def test_register_experiment(sample_experiment):
    fw = ABTestingFramework()
    eid = fw.register_experiment(sample_experiment)
    assert eid == "exp_001"
    assert fw.is_active("exp_001")


def test_route_deterministic(sample_experiment):
    fw = ABTestingFramework()
    fw.register_experiment(sample_experiment)
    r1 = fw.route("exp_001", "600519.SH")
    r2 = fw.route("exp_001", "600519.SH")
    assert r1 == r2  # 同一标的路由一致
    assert r1 in ("control", "treatment")


def test_route_distribution(sample_experiment):
    fw = ABTestingFramework()
    fw.register_experiment(sample_experiment)
    routes = [fw.route("exp_001", f"sym_{i}") for i in range(1000)]
    treatment_ratio = routes.count("treatment") / len(routes)
    # 大致50%分配, 允许误差±10%
    assert 0.4 <= treatment_ratio <= 0.6


def test_evaluate_promote(sample_experiment):
    fw = ABTestingFramework()
    fw.register_experiment(sample_experiment)
    # treatment明显优于control
    control_rets = np.random.normal(0.0005, 0.02, 100).tolist()
    treatment_rets = np.random.normal(0.002, 0.02, 100).tolist()
    result = fw.evaluate("exp_001", control_rets, treatment_rets)
    assert result.experiment_id == "exp_001"
    assert result.sample_size == 200
    assert result.recommendation in ("promote", "continue", "rollback")


def test_evaluate_rollback(sample_experiment):
    fw = ABTestingFramework()
    fw.register_experiment(sample_experiment)
    # treatment大幅亏损
    control_rets = np.random.normal(0.001, 0.01, 50).tolist()
    treatment_rets = np.full(50, -0.05).tolist()
    result = fw.evaluate("exp_001", control_rets, treatment_rets)
    assert result.recommendation == "rollback"


def test_evaluate_continue_insufficient_data(sample_experiment):
    fw = ABTestingFramework()
    fw.register_experiment(sample_experiment)
    control_rets = [0.01, -0.01]
    treatment_rets = [0.02, -0.02]
    result = fw.evaluate("exp_001", control_rets, treatment_rets)
    assert result.recommendation == "continue"


def test_stop_experiment(sample_experiment):
    fw = ABTestingFramework()
    fw.register_experiment(sample_experiment)
    assert fw.stop_experiment("exp_001")
    assert not fw.is_active("exp_001")
    # 停止后路由应返回control
    assert fw.route("exp_001", "600519.SH") == "control"


def test_get_report(sample_experiment):
    fw = ABTestingFramework()
    fw.register_experiment(sample_experiment)
    report = fw.get_report("exp_001")
    assert report["experiment_id"] == "exp_001"
    assert report["active"] is True
    assert report["control_version"]["name"] == "ma_cross_v1"
    assert report["treatment_version"]["name"] == "ma_cross_v2"


def test_list_experiments(sample_experiment):
    fw = ABTestingFramework()
    fw.register_experiment(sample_experiment)
    experiments = fw.list_experiments()
    assert len(experiments) == 1
    assert experiments[0]["experiment_id"] == "exp_001"


def test_max_drawdown():
    fw = ABTestingFramework()
    dd = fw._max_drawdown([0.01, 0.02, -0.05, 0.01, 0.02])
    assert dd < 0


def test_max_drawdown_empty():
    fw = ABTestingFramework()
    assert fw._max_drawdown([]) == 0.0
