"""多 Jev 模型 Ensemble 模块。

实现多模型注册、加权投票、市场/策略路由、动态权重调整和降级机制。
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """模型表现指标。"""
    model_id: str
    total_calls: int = 0
    success_calls: int = 0
    failed_calls: int = 0
    avg_latency_ms: float = 0.0
    accuracy: float = 0.0
    sharpe: float = 0.0
    last_evaluated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnsembleModelConfig:
    """Ensemble 模型配置。"""
    model_id: str
    name: str
    markets: List[str] = field(default_factory=list)
    strategies: List[str] = field(default_factory=list)
    initial_weight: float = 1.0
    mock_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnsembleMember:
    """Ensemble 成员（包装模型实例和配置）。"""
    config: EnsembleModelConfig
    model: Any  # JevDecisionEngine 或其他可调用对象
    weight: float = 1.0
    performance: ModelPerformance = field(default_factory=lambda: ModelPerformance(model_id=""))
    enabled: bool = True

    def __post_init__(self) -> None:
        self.performance.model_id = self.config.model_id


class WeightAdjuster(ABC):
    """权重调整器抽象基类。"""

    @abstractmethod
    def adjust(
        self,
        members: List[EnsembleMember],
    ) -> Dict[str, float]:
        """返回 model_id -> new_weight 映射。"""


class SoftmaxWeightAdjuster(WeightAdjuster):
    """基于准确率的 Softmax 权重调整。"""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def adjust(self, members: List[EnsembleMember]) -> Dict[str, float]:
        if not members:
            return {}
        scores = [m.performance.accuracy for m in members]
        # 防止全0
        if all(s == 0 for s in scores):
            scores = [1.0] * len(members)
        exp_scores = np.exp(np.array(scores) / self.temperature)
        total = exp_scores.sum()
        weights = exp_scores / total
        return {m.config.model_id: float(w) for m, w in zip(members, weights)}


class EWMWeightAdjuster(WeightAdjuster):
    """基于夏普比率的指数加权权重调整。"""

    def adjust(self, members: List[EnsembleMember]) -> Dict[str, float]:
        if not members:
            return {}
        scores = [max(m.performance.sharpe, 0) for m in members]
        total = sum(scores)
        if total <= 0:
            n = len(members)
            return {m.config.model_id: 1.0 / n for m in members}
        return {m.config.model_id: s / total for m, s in zip(members, scores)}


class JevEnsemble:
    """多 Jev 模型 Ensemble。"""

    def __init__(self) -> None:
        self._members: Dict[str, EnsembleMember] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._adjusters: Dict[str, WeightAdjuster] = {
            "softmax": SoftmaxWeightAdjuster(),
            "ewm": EWMWeightAdjuster(),
        }
        self._fallback_enabled: bool = True

    # ------------------------------------------------------------------
    # 模型注册
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_id: str,
        model: Any,
        name: str = "",
        markets: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        initial_weight: float = 1.0,
        mock_mode: bool = False,
    ) -> EnsembleMember:
        if model_id in self._members:
            raise ValueError(f"模型已注册: {model_id}")
        config = EnsembleModelConfig(
            model_id=model_id,
            name=name or model_id,
            markets=markets or [],
            strategies=strategies or [],
            initial_weight=initial_weight,
            mock_mode=mock_mode,
        )
        member = EnsembleMember(
            config=config,
            model=model,
            weight=initial_weight,
        )
        self._members[model_id] = member
        self._log_audit("register_model", {"model_id": model_id, "name": name})
        return member

    def unregister_model(self, model_id: str) -> bool:
        if model_id not in self._members:
            return False
        del self._members[model_id]
        self._log_audit("unregister_model", {"model_id": model_id})
        return True

    def list_models(self) -> List[EnsembleMember]:
        return list(self._members.values())

    def get_model(self, model_id: str) -> Optional[EnsembleMember]:
        return self._members.get(model_id)

    def update_model_weight(self, model_id: str, weight: float) -> Optional[EnsembleMember]:
        member = self._members.get(model_id)
        if not member:
            return None
        member.weight = max(weight, 0.0)
        self._log_audit("update_weight", {"model_id": model_id, "weight": weight})
        return member

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------

    def select_models(
        self,
        market: str = "",
        strategy: str = "",
    ) -> List[EnsembleMember]:
        """根据市场和策略筛选可用模型。"""
        results = []
        for m in self._members.values():
            if not m.enabled:
                continue
            if market and m.config.markets and market not in m.config.markets:
                continue
            if strategy and m.config.strategies and strategy not in m.config.strategies:
                continue
            results.append(m)
        # 按权重排序
        results.sort(key=lambda x: x.weight, reverse=True)
        return results

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def evaluate(
        self,
        symbol: str,
        market_state: Dict[str, Any],
        strategy: str = "",
        market: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """多模型加权投票评估。"""
        start = time.time()
        models = self.select_models(market=market, strategy=strategy)
        if not models:
            return {"action": "hold", "confidence": 0.0, "reason": "无可用模型", "models_used": 0}

        votes: Dict[str, float] = {}
        model_results: List[Dict[str, Any]] = []
        total_weight = 0.0

        for member in models:
            result = self._call_model(member, symbol, market_state, strategy)
            if result is None:
                continue
            action = result.get("action", "hold")
            conf = result.get("confidence", 0.0)
            w = member.weight
            votes[action] = votes.get(action, 0.0) + w * conf
            total_weight += w
            model_results.append({
                "model_id": member.config.model_id,
                "action": action,
                "confidence": conf,
                "weight": w,
            })
            # 更新性能
            member.performance.total_calls += 1
            member.performance.success_calls += 1

        if total_weight == 0:
            return {"action": "hold", "confidence": 0.0, "reason": "所有模型调用失败", "models_used": 0}

        # 加权投票
        best_action = max(votes, key=lambda k: votes[k])
        best_score = votes[best_action]
        confidence = best_score / total_weight if total_weight > 0 else 0.0

        latency = (time.time() - start) * 1000
        result = {
            "action": best_action,
            "confidence": round(confidence, 4),
            "reason": f"加权投票: {best_action} (得分 {best_score:.2f})",
            "models_used": len(model_results),
            "model_details": model_results,
            "latency_ms": round(latency, 2),
        }
        self._log_audit("evaluate", {"symbol": symbol, "result": result})
        return result

    def evaluate_batch(
        self,
        items: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """批量评估。"""
        return [self.evaluate(**item, **kwargs) for item in items]

    def _call_model(
        self,
        member: EnsembleMember,
        symbol: str,
        market_state: Dict[str, Any],
        strategy: str,
    ) -> Optional[Dict[str, Any]]:
        """调用单个模型，失败时返回 None（触发降级）。"""
        try:
            model = member.model
            if hasattr(model, "evaluate"):
                return model.evaluate(symbol, market_state, strategy)
            elif callable(model):
                return model(symbol, market_state, strategy)
            else:
                return {"action": "hold", "confidence": 0.5}
        except Exception as e:
            member.performance.failed_calls += 1
            if self._fallback_enabled:
                logger.warning("模型 %s 调用失败: %s", member.config.model_id, e)
            return None

    # ------------------------------------------------------------------
    # 权重管理
    # ------------------------------------------------------------------

    def adjust_weights(self, method: str = "softmax") -> Dict[str, float]:
        """按指定方法动态调整所有模型权重。"""
        adjuster = self._adjusters.get(method)
        if not adjuster:
            raise ValueError(f"不支持的权重调整方法: {method}")
        members = self.list_models()
        new_weights = adjuster.adjust(members)
        for model_id, w in new_weights.items():
            if model_id in self._members:
                self._members[model_id].weight = w
        self._log_audit("adjust_weights", {"method": method, "weights": new_weights})
        return new_weights

    def get_weights(self) -> Dict[str, float]:
        return {m.config.model_id: m.weight for m in self._members.values()}

    def get_adjuster_names(self) -> List[str]:
        return list(self._adjusters.keys())

    # ------------------------------------------------------------------
    # 性能追踪
    # ------------------------------------------------------------------

    def update_performance(
        self,
        model_id: str,
        accuracy: Optional[float] = None,
        sharpe: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ) -> Optional[ModelPerformance]:
        member = self._members.get(model_id)
        if not member:
            return None
        perf = member.performance
        if accuracy is not None:
            perf.accuracy = accuracy
        if sharpe is not None:
            perf.sharpe = sharpe
        if latency_ms is not None:
            # 指数移动平均
            if perf.avg_latency_ms == 0:
                perf.avg_latency_ms = latency_ms
            else:
                perf.avg_latency_ms = 0.7 * perf.avg_latency_ms + 0.3 * latency_ms
        perf.last_evaluated = datetime.now().isoformat()
        return perf

    def get_performance(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        if model_id:
            m = self._members.get(model_id)
            return m.performance.to_dict() if m else {}
        return {m.config.model_id: m.performance.to_dict() for m in self._members.values()}

    # ------------------------------------------------------------------
    # 降级机制
    # ------------------------------------------------------------------

    def set_fallback_enabled(self, enabled: bool) -> None:
        self._fallback_enabled = enabled

    def disable_model(self, model_id: str) -> bool:
        member = self._members.get(model_id)
        if not member:
            return False
        member.enabled = False
        self._log_audit("disable_model", {"model_id": model_id})
        return True

    def enable_model(self, model_id: str) -> bool:
        member = self._members.get(model_id)
        if not member:
            return False
        member.enabled = True
        self._log_audit("enable_model", {"model_id": model_id})
        return True

    # ------------------------------------------------------------------
    # 审计日志
    # ------------------------------------------------------------------

    def _log_audit(self, action: str, detail: Dict[str, Any]) -> None:
        self._audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "detail": detail,
        })

    def get_audit_log(
        self,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        logs = self._audit_log[::-1]
        if action:
            logs = [l for l in logs if l["action"] == action]
        return logs[:limit]
