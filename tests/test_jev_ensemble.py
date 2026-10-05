"""多 Jev 模型 Ensemble 单元测试。"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from jev.ensemble import (
    EWMWeightAdjuster,
    EnsembleMember,
    EnsembleModelConfig,
    JevEnsemble,
    ModelPerformance,
    SoftmaxWeightAdjuster,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ensemble() -> JevEnsemble:
    return JevEnsemble()


class MockModel:
    """模拟 Jev 决策模型。"""

    def __init__(self, action: str = "buy", confidence: float = 0.8) -> None:
        self.action = action
        self.confidence = confidence

    def evaluate(self, symbol: str, market_state: Dict[str, Any], strategy: str = "") -> Dict[str, Any]:
        return {"action": self.action, "confidence": self.confidence}


class FailingModel:
    """总是失败的模型。"""

    def evaluate(self, symbol: str, market_state: Dict[str, Any], strategy: str = "") -> Dict[str, Any]:
        raise RuntimeError("模拟失败")


# ---------------------------------------------------------------------------
# 模型注册
# ---------------------------------------------------------------------------

class TestModelRegistration:
    def test_register_model(self, ensemble):
        model = MockModel("buy", 0.9)
        member = ensemble.register_model("m1", model, name="模型1", markets=["CN"])
        assert member.config.model_id == "m1"
        assert member.config.name == "模型1"
        assert member.config.markets == ["CN"]

    def test_register_duplicate_raises(self, ensemble):
        model = MockModel()
        ensemble.register_model("m1", model)
        with pytest.raises(ValueError, match="模型已注册"):
            ensemble.register_model("m1", model)

    def test_unregister_model(self, ensemble):
        ensemble.register_model("m1", MockModel())
        assert ensemble.unregister_model("m1") is True
        assert ensemble.get_model("m1") is None

    def test_unregister_missing(self, ensemble):
        assert ensemble.unregister_model("missing") is False

    def test_list_models(self, ensemble):
        ensemble.register_model("m1", MockModel())
        ensemble.register_model("m2", MockModel())
        assert len(ensemble.list_models()) == 2

    def test_update_model_weight(self, ensemble):
        ensemble.register_model("m1", MockModel(), initial_weight=1.0)
        member = ensemble.update_model_weight("m1", 2.5)
        assert member is not None
        assert member.weight == 2.5

    def test_update_model_weight_missing(self, ensemble):
        assert ensemble.update_model_weight("missing", 1.0) is None


# ---------------------------------------------------------------------------
# Ensemble 评估
# ---------------------------------------------------------------------------

class TestEnsembleEvaluate:
    def test_evaluate_single_model(self, ensemble):
        ensemble.register_model("m1", MockModel("buy", 0.9))
        result = ensemble.evaluate("600519.SH", {"price": 100})
        assert result["action"] == "buy"
        assert result["confidence"] == pytest.approx(0.9, abs=0.01)
        assert result["models_used"] == 1

    def test_evaluate_multiple_models_voting(self, ensemble):
        ensemble.register_model("m1", MockModel("buy", 0.9), initial_weight=1.0)
        ensemble.register_model("m2", MockModel("sell", 0.6), initial_weight=1.0)
        result = ensemble.evaluate("600519.SH", {"price": 100})
        # buy 得分 = 1.0 * 0.9 = 0.9, sell 得分 = 1.0 * 0.6 = 0.6
        assert result["action"] == "buy"

    def test_evaluate_no_models(self, ensemble):
        result = ensemble.evaluate("600519.SH", {"price": 100})
        assert result["action"] == "hold"
        assert result["models_used"] == 0

    def test_evaluate_with_fallback(self, ensemble):
        ensemble.register_model("m1", FailingModel(), initial_weight=1.0)
        ensemble.register_model("m2", MockModel("buy", 0.8), initial_weight=1.0)
        result = ensemble.evaluate("600519.SH", {"price": 100})
        assert result["action"] == "buy"
        assert result["models_used"] == 1

    def test_evaluate_all_fail(self, ensemble):
        ensemble.register_model("m1", FailingModel())
        result = ensemble.evaluate("600519.SH", {"price": 100})
        assert result["action"] == "hold"
        assert "所有模型调用失败" in result["reason"]

    def test_evaluate_batch(self, ensemble):
        ensemble.register_model("m1", MockModel("buy", 0.8))
        items = [
            {"symbol": "600519.SH", "market_state": {"price": 100}},
            {"symbol": "000001.SZ", "market_state": {"price": 50}},
        ]
        results = ensemble.evaluate_batch(items)
        assert len(results) == 2
        assert all(r["action"] == "buy" for r in results)

    def test_evaluate_routing_by_market(self, ensemble):
        ensemble.register_model("m1", MockModel("buy", 0.9), markets=["CN"])
        ensemble.register_model("m2", MockModel("sell", 0.9), markets=["US"])
        result = ensemble.evaluate("AAPL", {"price": 100}, market="US")
        assert result["action"] == "sell"

    def test_evaluate_routing_by_strategy(self, ensemble):
        ensemble.register_model("m1", MockModel("buy", 0.9), strategies=["ma_cross"])
        ensemble.register_model("m2", MockModel("sell", 0.9), strategies=["bollinger"])
        result = ensemble.evaluate("600519.SH", {"price": 100}, strategy="bollinger")
        assert result["action"] == "sell"


# ---------------------------------------------------------------------------
# 权重管理
# ---------------------------------------------------------------------------

class TestWeightManagement:
    def test_get_weights(self, ensemble):
        ensemble.register_model("m1", MockModel(), initial_weight=1.0)
        ensemble.register_model("m2", MockModel(), initial_weight=2.0)
        weights = ensemble.get_weights()
        assert weights == {"m1": 1.0, "m2": 2.0}

    def test_adjust_weights_softmax(self, ensemble):
        ensemble.register_model("m1", MockModel())
        ensemble.register_model("m2", MockModel())
        ensemble.update_performance("m1", accuracy=0.9)
        ensemble.update_performance("m2", accuracy=0.5)
        weights = ensemble.adjust_weights("softmax")
        assert weights["m1"] > weights["m2"]
        assert pytest.approx(sum(weights.values()), abs=0.001) == 1.0

    def test_adjust_weights_ewm(self, ensemble):
        ensemble.register_model("m1", MockModel())
        ensemble.register_model("m2", MockModel())
        ensemble.update_performance("m1", sharpe=2.0)
        ensemble.update_performance("m2", sharpe=1.0)
        weights = ensemble.adjust_weights("ewm")
        assert weights["m1"] == pytest.approx(2.0 / 3.0, abs=0.01)
        assert weights["m2"] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_adjust_weights_unknown_raises(self, ensemble):
        with pytest.raises(ValueError, match="不支持的权重调整方法"):
            ensemble.adjust_weights("unknown")

    def test_get_adjuster_names(self, ensemble):
        names = ensemble.get_adjuster_names()
        assert "softmax" in names
        assert "ewm" in names


# ---------------------------------------------------------------------------
# 权重调整器
# ---------------------------------------------------------------------------

class TestWeightAdjusters:
    def test_softmax_adjuster(self):
        adjuster = SoftmaxWeightAdjuster(temperature=1.0)
        members = [
            EnsembleMember(
                config=EnsembleModelConfig(model_id="m1", name="M1"),
                model=MockModel(),
                performance=ModelPerformance(model_id="m1", accuracy=0.9),
            ),
            EnsembleMember(
                config=EnsembleModelConfig(model_id="m2", name="M2"),
                model=MockModel(),
                performance=ModelPerformance(model_id="m2", accuracy=0.5),
            ),
        ]
        weights = adjuster.adjust(members)
        assert weights["m1"] > weights["m2"]
        assert pytest.approx(sum(weights.values()), abs=0.001) == 1.0

    def test_softmax_zero_accuracy(self):
        adjuster = SoftmaxWeightAdjuster()
        members = [
            EnsembleMember(
                config=EnsembleModelConfig(model_id="m1", name="M1"),
                model=MockModel(),
                performance=ModelPerformance(model_id="m1", accuracy=0.0),
            ),
        ]
        weights = adjuster.adjust(members)
        assert weights["m1"] == pytest.approx(1.0, abs=0.001)

    def test_ewm_adjuster(self):
        adjuster = EWMWeightAdjuster()
        members = [
            EnsembleMember(
                config=EnsembleModelConfig(model_id="m1", name="M1"),
                model=MockModel(),
                performance=ModelPerformance(model_id="m1", sharpe=2.0),
            ),
            EnsembleMember(
                config=EnsembleModelConfig(model_id="m2", name="M2"),
                model=MockModel(),
                performance=ModelPerformance(model_id="m2", sharpe=1.0),
            ),
        ]
        weights = adjuster.adjust(members)
        assert weights["m1"] == pytest.approx(2.0 / 3.0, abs=0.01)
        assert weights["m2"] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_ewm_zero_sharpe(self):
        adjuster = EWMWeightAdjuster()
        members = [
            EnsembleMember(
                config=EnsembleModelConfig(model_id="m1", name="M1"),
                model=MockModel(),
                performance=ModelPerformance(model_id="m1", sharpe=0.0),
            ),
            EnsembleMember(
                config=EnsembleModelConfig(model_id="m2", name="M2"),
                model=MockModel(),
                performance=ModelPerformance(model_id="m2", sharpe=0.0),
            ),
        ]
        weights = adjuster.adjust(members)
        assert weights["m1"] == pytest.approx(0.5, abs=0.01)
        assert weights["m2"] == pytest.approx(0.5, abs=0.01)

    def test_ewm_empty(self):
        adjuster = EWMWeightAdjuster()
        assert adjuster.adjust([]) == {}


# ---------------------------------------------------------------------------
# 性能追踪
# ---------------------------------------------------------------------------

class TestPerformanceTracking:
    def test_update_performance(self, ensemble):
        ensemble.register_model("m1", MockModel())
        perf = ensemble.update_performance("m1", accuracy=0.85, sharpe=1.5, latency_ms=50)
        assert perf is not None
        assert perf.accuracy == 0.85
        assert perf.sharpe == 1.5
        assert perf.avg_latency_ms == 50.0

    def test_update_performance_ema_latency(self, ensemble):
        ensemble.register_model("m1", MockModel())
        ensemble.update_performance("m1", latency_ms=100)
        ensemble.update_performance("m1", latency_ms=200)
        perf = ensemble.get_performance("m1")
        assert perf["avg_latency_ms"] == pytest.approx(130.0, abs=1.0)  # 0.7*100 + 0.3*200

    def test_get_performance_all(self, ensemble):
        ensemble.register_model("m1", MockModel())
        ensemble.register_model("m2", MockModel())
        all_perf = ensemble.get_performance()
        assert "m1" in all_perf
        assert "m2" in all_perf

    def test_update_performance_missing(self, ensemble):
        assert ensemble.update_performance("missing", accuracy=0.5) is None


# ---------------------------------------------------------------------------
# 降级机制
# ---------------------------------------------------------------------------

class TestFallback:
    def test_disable_model(self, ensemble):
        ensemble.register_model("m1", MockModel())
        assert ensemble.disable_model("m1") is True
        result = ensemble.evaluate("600519.SH", {"price": 100})
        assert result["models_used"] == 0

    def test_disable_model_missing(self, ensemble):
        assert ensemble.disable_model("missing") is False

    def test_enable_model(self, ensemble):
        ensemble.register_model("m1", MockModel())
        ensemble.disable_model("m1")
        assert ensemble.enable_model("m1") is True
        result = ensemble.evaluate("600519.SH", {"price": 100})
        assert result["models_used"] == 1

    def test_set_fallback_enabled(self, ensemble):
        ensemble.set_fallback_enabled(False)
        ensemble.register_model("m1", FailingModel())
        result = ensemble.evaluate("600519.SH", {"price": 100})
        assert result["models_used"] == 0


# ---------------------------------------------------------------------------
# 审计日志
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_audit_log_on_register(self, ensemble):
        ensemble.register_model("m1", MockModel())
        logs = ensemble.get_audit_log()
        assert len(logs) >= 1
        assert logs[0]["action"] == "register_model"

    def test_audit_log_filter(self, ensemble):
        ensemble.register_model("m1", MockModel())
        ensemble.register_model("m2", MockModel())
        logs = ensemble.get_audit_log(action="register_model")
        assert all(l["action"] == "register_model" for l in logs)

    def test_audit_log_limit(self, ensemble):
        for i in range(5):
            ensemble.register_model(f"m{i}", MockModel())
        logs = ensemble.get_audit_log(limit=3)
        assert len(logs) == 3
