"""Jev 可解释性 API 路由（扩展模块，REQ-P2-12）。

按 server.py 现有扩展路由模式挂载：

    try:
        from _routes_explain import register_explain_routes
        register_explain_routes(app, ok, err)
    except Exception as e:
        logger.warning("Jev 可解释性路由注册失败: %s", e)

端点：
    POST /api/jev/explain                 —— 对一组特征生成解释报告（贡献度+文本+概率+动作），存历史
    POST /api/jev/feature_importance      —— 单组/多组特征的特征重要性排序
    POST /api/jev/counterfactual         —— 反事实：最小特征变更使决策反转为目标动作
    GET  /api/jev/explain/{decision_id}  —— 取历史解释（不存在返回 404）

解释器以模块级单例维护，历史解释用 ``threading.Lock`` 保护。
Jev 引擎不可用（或离线）时降级到内置 mock predict，保证解释流程完整可演示。
"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

logger = logging.getLogger("realtime_server")

# ---------------------------------------------------------------------------
# 模块级单例与历史缓存（线程安全）
# ---------------------------------------------------------------------------
_EXPLAIN_LOCK = threading.Lock()
_EXPLAINER: Any = None  # JevExplainer 单例
_HISTORY_LOCK = threading.Lock()
_HISTORY: Dict[str, Dict[str, Any]] = {}


def _demo_predict(features: List[Dict[str, Any]]) -> Dict[str, float]:
    """Jev 不可用时的降级演示 predict：一个确定性的线性 softmax。

    保证离线时 POST /api/jev/explain 等端点仍能跑通完整解释流程。
    """
    import math

    d = {str(f["feature"]): float(f["value"]) for f in features}
    logit_buy = 1.0 * d.get("ma5_ma20_ratio", 1.0) + 0.01 * (50.0 - d.get("rsi", 50.0))
    logit_sell = 0.05 * d.get("rsi", 50.0) + 1.0 * max(0.0, 1.0 - d.get("ma5_ma20_ratio", 1.0))
    logit_hold = 1.0
    vals = {"buy": logit_buy, "sell": logit_sell, "hold": logit_hold}
    m = max(vals.values())
    ex = {k: math.exp(v - m) for k, v in vals.items()}
    s = sum(ex.values())
    return {k: v / s for k, v in ex.items()}


def _get_explainer() -> Any:
    """惰性构建 JevExplainer 单例；引擎路径失败时降级到 demo predict。"""
    global _EXPLAINER
    with _EXPLAIN_LOCK:
        if _EXPLAINER is not None:
            return _EXPLAINER
        try:
            from jev.explainability import JevExplainer

            explainer = JevExplainer()  # 默认适配引擎 mock 概率路径（离线可用）
            source = "jev_engine_mock"
        except Exception as e:  # noqa: BLE001
            logger.warning("JevExplainer 初始化失败，降级为 demo predict: %s", e)
            from jev.explainability import JevExplainer

            explainer = JevExplainer(predict_callable=_demo_predict)
            source = "demo_mock"
        explainer._predict_source = source  # type: ignore[attr-defined]
        _EXPLAINER = explainer
        return _EXPLAINER


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class FeatureItem(BaseModel):
    """单个特征项。"""

    feature: str = Field(..., description="特征名，如 rsi/volume_ratio/ma5_ma20_ratio")
    value: float = Field(..., description="特征值")


class ExplainRequest(BaseModel):
    """单次决策解释请求。

    二选一：
        - 提供 ``features`` 原始特征列表，现场计算解释并存历史；
        - 仅提供 ``decision_id`` 时不重复计算（历史由 GET 端点读取，此处仍需
          至少给出 features 才会生成新解释）。
    """

    decision_id: Optional[str] = None
    features: Optional[List[FeatureItem]] = None
    raw_signal: Optional[str] = None  # 仅用于文案/引擎路径先验，可选


class ImportanceRequest(BaseModel):
    """特征重要性请求：单组 ``features`` 或多组 ``samples``。"""

    features: Optional[List[FeatureItem]] = None
    samples: Optional[List[List[FeatureItem]]] = None


class CounterfactualRequest(BaseModel):
    """反事实分析请求。"""

    features: List[FeatureItem]
    target_action: str = Field(..., description="目标动作 buy/sell/hold")


def _to_dicts(items: Optional[List[FeatureItem]]) -> List[Dict[str, Any]]:
    """pydantic 特征列表转引擎特征字典列表。"""
    if not items:
        return []
    return [{"feature": it.feature, "value": it.value} for it in items]


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_explain_routes(app: FastAPI, ok: Any, err: Any) -> None:
    """注册 Jev 可解释性相关路由。

    Args:
        app: FastAPI 实例。
        ok / err: server.py 统一响应封装：
            ``ok(data=None, message="success") -> dict``；
            ``err(code, message, http_status=400) -> JSONResponse``。
    """

    @app.post("/api/jev/explain")
    async def explain_decision(req: ExplainRequest):
        """对一组特征生成解释报告，并写入历史。

        请求体需提供 ``features``；``decision_id`` 可选（缺省自动生成）。
        返回：贡献度明细 + explain_text + 三动作概率 + 最终动作 + decision_id。
        """
        if not req.features:
            return err(40010, "必须提供 features（特征列表）才能生成解释")
        try:
            explainer = _get_explainer()
            features = _to_dicts(req.features)
            result = explainer.explain(features)
            decision_id = req.decision_id or uuid.uuid4().hex[:12]
            record = {
                "decision_id": decision_id,
                "predict_source": getattr(explainer, "_predict_source", "unknown"),
                **result,
            }
            with _HISTORY_LOCK:
                _HISTORY[decision_id] = record
            return ok(record)
        except Exception as e:  # noqa: BLE001
            logger.exception("Jev 解释失败")
            return err(50010, f"Jev 解释失败: {e}", http_status=500)

    @app.post("/api/jev/feature_importance")
    async def feature_importance(req: ImportanceRequest):
        """计算特征重要性。

        提供 ``samples``（多组特征）时返回跨样本平均的全局重要性；
        否则对单组 ``features`` 返回逐特征遮蔽重要性排序。
        """
        try:
            explainer = _get_explainer()
            if req.samples:
                sample_list = [_to_dicts(s) for s in req.samples]
                if not sample_list:
                    return err(40020, "samples 为空")
                result = explainer.global_feature_importance(sample_list)
            elif req.features:
                result = explainer.feature_importance(_to_dicts(req.features))
            else:
                return err(40021, "必须提供 features（单组）或 samples（多组）")
            return ok(result)
        except Exception as e:  # noqa: BLE001
            logger.exception("特征重要性计算失败")
            return err(50020, f"特征重要性计算失败: {e}", http_status=500)

    @app.post("/api/jev/counterfactual")
    async def counterfactual(req: CounterfactualRequest):
        """反事实分析：搜索最小特征变更使决策反转为 target_action。"""
        if req.target_action not in ("buy", "sell", "hold"):
            return err(40030, "target_action 必须为 buy/sell/hold")
        try:
            explainer = _get_explainer()
            result = explainer.counterfactual(_to_dicts(req.features), req.target_action)
            return ok(result)
        except ValueError as e:
            return err(40031, f"反事实参数错误: {e}")
        except Exception as e:  # noqa: BLE001
            logger.exception("反事实分析失败")
            return err(50030, f"反事实分析失败: {e}", http_status=500)

    @app.get("/api/jev/explain/{decision_id}")
    async def get_explain_history(decision_id: str):
        """按 decision_id 取历史解释报告；不存在返回 404。"""
        with _HISTORY_LOCK:
            record = _HISTORY.get(decision_id)
        if record is None:
            return err(40440, f"decision_id 不存在: {decision_id}", http_status=404)
        return ok(record)
