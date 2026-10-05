"""多 Jev 模型 Ensemble API 路由。

按 server.py 现有扩展路由模式挂载。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from jev.ensemble import JevEnsemble

logger = logging.getLogger("realtime_server")

# 模块级单例（懒加载）
_default_ensemble: Optional[JevEnsemble] = None


def _get_default_ensemble() -> JevEnsemble:
    """获取模块级单例。"""
    global _default_ensemble
    if _default_ensemble is None:
        _default_ensemble = JevEnsemble()
    return _default_ensemble


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------

class RegisterModelReq(BaseModel):
    model_id: str
    name: str = ""
    markets: List[str] = []
    strategies: List[str] = []
    initial_weight: float = 1.0
    mock_mode: bool = False


class EvaluateReq(BaseModel):
    symbol: str
    market_state: Dict[str, Any]
    strategy: str = ""
    market: str = ""


class EvaluateBatchReq(BaseModel):
    items: List[EvaluateReq]


class UpdateWeightReq(BaseModel):
    weight: float


class AdjustWeightsReq(BaseModel):
    method: str = "softmax"


class UpdatePerformanceReq(BaseModel):
    accuracy: Optional[float] = None
    sharpe: Optional[float] = None
    latency_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------

def register_jev_ensemble_routes(
    app: FastAPI,
    ok: Any,
    err: Any,
    ensemble: Optional[JevEnsemble] = None,
) -> None:
    """注册 Jev Ensemble 相关路由。"""
    e = ensemble or _get_default_ensemble()

    @app.get("/api/jev_ensemble/models")
    async def list_models() -> Any:
        models = e.list_models()
        return ok(data=[
            {
                "config": m.config.to_dict(),
                "weight": m.weight,
                "enabled": m.enabled,
                "performance": m.performance.to_dict(),
            }
            for m in models
        ])

    @app.post("/api/jev_ensemble/models")
    async def register_model(req: RegisterModelReq) -> Any:
        try:
            # 由于无法通过 HTTP 传递实际模型对象，这里注册一个 mock 占位
            # 实际运行时应在服务端初始化 ensemble 并传入真实模型
            member = e.register_model(
                model_id=req.model_id,
                model=lambda s, ms, st: {"action": "hold", "confidence": 0.5},
                name=req.name,
                markets=req.markets,
                strategies=req.strategies,
                initial_weight=req.initial_weight,
                mock_mode=req.mock_mode,
            )
            return ok(data={
                "config": member.config.to_dict(),
                "weight": member.weight,
            })
        except ValueError as ve:
            return err(400, str(ve))

    @app.delete("/api/jev_ensemble/models/{model_id}")
    async def unregister_model(model_id: str) -> Any:
        if e.unregister_model(model_id):
            return ok(message="模型已注销")
        return err(404, "模型不存在")

    @app.post("/api/jev_ensemble/evaluate")
    async def evaluate(req: EvaluateReq) -> Any:
        result = e.evaluate(
            symbol=req.symbol,
            market_state=req.market_state,
            strategy=req.strategy,
            market=req.market,
        )
        return ok(data=result)

    @app.post("/api/jev_ensemble/evaluate_batch")
    async def evaluate_batch(req: EvaluateBatchReq) -> Any:
        items = [
            {
                "symbol": item.symbol,
                "market_state": item.market_state,
                "strategy": item.strategy,
                "market": item.market,
            }
            for item in req.items
        ]
        results = e.evaluate_batch(items)
        return ok(data=results)

    @app.get("/api/jev_ensemble/weights")
    async def get_weights() -> Any:
        return ok(data=e.get_weights())

    @app.post("/api/jev_ensemble/adjust_weights")
    async def adjust_weights(req: AdjustWeightsReq) -> Any:
        try:
            weights = e.adjust_weights(req.method)
            return ok(data=weights)
        except ValueError as ve:
            return err(400, str(ve))

    @app.get("/api/jev_ensemble/adjusters")
    async def list_adjusters() -> Any:
        return ok(data=e.get_adjuster_names())

    @app.get("/api/jev_ensemble/performance")
    async def get_performance(model_id: Optional[str] = None) -> Any:
        return ok(data=e.get_performance(model_id))

    @app.post("/api/jev_ensemble/performance/{model_id}")
    async def update_performance(model_id: str, req: UpdatePerformanceReq) -> Any:
        perf = e.update_performance(
            model_id=model_id,
            accuracy=req.accuracy,
            sharpe=req.sharpe,
            latency_ms=req.latency_ms,
        )
        if not perf:
            return err(404, "模型不存在")
        return ok(data=perf.to_dict())

    @app.post("/api/jev_ensemble/models/{model_id}/enable")
    async def enable_model(model_id: str) -> Any:
        if e.enable_model(model_id):
            return ok(message="模型已启用")
        return err(404, "模型不存在")

    @app.post("/api/jev_ensemble/models/{model_id}/disable")
    async def disable_model(model_id: str) -> Any:
        if e.disable_model(model_id):
            return ok(message="模型已禁用")
        return err(404, "模型不存在")

    @app.get("/api/jev_ensemble/audit")
    async def get_audit_log(action: Optional[str] = None, limit: int = 100) -> Any:
        return ok(data=e.get_audit_log(action=action, limit=limit))
