"""Jev 在线学习 API 路由（扩展模块 #27，REQ-P3-05）。

端点：
    POST /api/online_learning/update       —— 增量更新模型（滑动窗口 + 漂移检测）
    GET  /api/online_learning/stats        —— 查询更新历史 / loss 曲线 / 漂移状态
    POST /api/online_learning/drift_check  —— 手动触发概念漂移检查（可确认再训练基线）
    POST /api/online_learning/reset        —— 重置在线学习管线

模块级单例管线与 ``jev.online_learning.OnlineLearningPipeline`` 对接；
``JevDecisionEngine.update_model()`` 也委托到同一实现（jev_engine.py）。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from jev.online_learning import OnlineLearningPipeline

logger = logging.getLogger("realtime_server")

# 模块级单例（懒加载），跨请求保持窗口与模型状态
_pipeline: Optional[OnlineLearningPipeline] = None


def get_pipeline(n_features: int = 8) -> OnlineLearningPipeline:
    """获取模块级在线学习管线单例（首次按 n_features 懒创建）。"""
    global _pipeline
    if _pipeline is None:
        _pipeline = OnlineLearningPipeline(n_features=n_features)
    return _pipeline


def reset_pipeline(**kwargs: Any) -> OnlineLearningPipeline:
    """重置为新的管线实例（测试与运维用）。"""
    global _pipeline
    _pipeline = OnlineLearningPipeline(**kwargs)
    return _pipeline


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class FeedbackRecord(BaseModel):
    """一条反馈样本：特征快照 + 实际 realized_return。"""

    features: Dict[str, float]
    realized_return: float


class OnlineUpdateReq(BaseModel):
    """增量更新请求（支持批量反馈）。"""

    records: List[FeedbackRecord]
    feature_keys: Optional[List[str]] = None


class DriftCheckReq(BaseModel):
    """漂移检查 / 再训练确认请求。"""

    acknowledge_baseline: Optional[float] = None


class OnlineResetReq(BaseModel):
    """重置请求。"""

    n_features: int = 8
    lr: float = 0.01
    window_size: int = 500


def _records_to_arrays(
    records: List[Dict[str, Any]],
    feature_keys: Optional[List[str]],
) -> tuple:
    """反馈记录 → (X, y, feature_keys)，与 update_model 同一转换。"""
    feats = [r["features"] for r in records]
    if feature_keys is None:
        feature_keys = sorted(feats[0].keys())
    X = np.array([[float(f[k]) for k in feature_keys] for f in feats])
    y = np.array([float(r.get("realized_return", 0.0)) for r in records])
    return X, y, feature_keys


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_online_learning_routes(
    app: FastAPI,
    manager: Any,
    symbol_set: List[str],
    ok: Any,
    err: Any,
) -> None:
    """注册 Jev 在线学习路由到 FastAPI app。"""

    @app.post("/api/online_learning/update")
    async def online_learning_update(req: OnlineUpdateReq):
        """增量更新模型（非全量重训）。"""
        try:
            if not req.records:
                return err(400, "records 不能为空")
            if len(req.records) > 5000:
                return err(422, "单次更新 records 上限 5000 条")
            records = [
                {"features": dict(r.features), "realized_return": float(r.realized_return)}
                for r in req.records
            ]
            try:
                X, y, feature_keys = _records_to_arrays(records, req.feature_keys)
            except KeyError as e:
                return err(400, f"特征缺失: {e}")
            pipeline = get_pipeline(n_features=X.shape[1])
            try:
                result = pipeline.update(X, y)
            except ValueError as e:
                return err(400, str(e))
            result["feature_keys"] = feature_keys
            result["server_ts"] = time.time()
            return ok(result)
        except Exception as e:
            logger.exception("在线学习更新失败")
            return err(500, f"更新失败: {e}")

    @app.get("/api/online_learning/stats")
    async def online_learning_stats():
        """查询在线学习状态：更新次数、loss 历史、漂移状态、窗口样本数。"""
        try:
            pipeline = get_pipeline()
            stats = pipeline.get_stats()
            stats["accuracy_proxy"] = pipeline.accuracy_proxy()
            return ok(stats)
        except Exception as e:
            logger.exception("查询在线学习状态失败")
            return err(500, f"查询失败: {e}")

    @app.post("/api/online_learning/drift_check")
    async def online_learning_drift_check(req: DriftCheckReq):
        """手动触发漂移检查；可携带新基线确认再训练完成。"""
        try:
            pipeline = get_pipeline()
            if req.acknowledge_baseline is not None:
                pipeline.acknowledge_retrain(float(req.acknowledge_baseline))
            detector = pipeline.drift_detector
            if detector is None:
                return ok({
                    "drift_detected": False,
                    "retrain_recommended": False,
                    "message": "尚无更新记录，暂无漂移基线",
                })
            # 只读快照：不写入 loss 历史，避免污染漂移检测
            rolling = (
                float(np.mean(detector._losses)) if detector._losses else detector.baseline_loss
            )
            return ok({
                "rolling_loss": rolling,
                "baseline_loss": detector.baseline_loss,
                "threshold": detector.threshold,
                "patience": detector.patience,
                "consecutive_breaches": detector._consecutive_breaches,
                "drift_detected": detector._consecutive_breaches >= detector.patience,
                "retrain_recommended": detector._consecutive_breaches >= detector.patience,
            })
        except Exception as e:
            logger.exception("漂移检查失败")
            return err(500, f"漂移检查失败: {e}")

    @app.post("/api/online_learning/reset")
    async def online_learning_reset(req: OnlineResetReq):
        """重置在线学习管线。"""
        try:
            if req.n_features <= 0 or req.n_features > 512:
                return err(422, "n_features 须在 1~512 之间")
            if req.lr <= 0 or req.lr > 1:
                return err(422, "lr 须在 (0, 1] 之间")
            if req.window_size <= 0 or req.window_size > 100000:
                return err(422, "window_size 须在 1~100000 之间")
            reset_pipeline(
                n_features=req.n_features,
                lr=req.lr,
                window_size=req.window_size,
            )
            return ok({"message": "在线学习管线已重置"})
        except Exception as e:
            logger.exception("重置在线学习管线失败")
            return err(500, f"重置失败: {e}")
