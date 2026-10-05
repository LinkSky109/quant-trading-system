"""Brinson 绩效归因 API 路由（扩展模块）。

按 server.py 现有扩展路由模式挂载：

    try:
        from _routes_brinson import register_brinson_routes
        register_brinson_routes(app, ok, err)
    except Exception as e:
        logger.warning("Brinson 归因路由注册失败: %s", e)

端点：
    POST /api/attribution/brinson            —— 提交归因任务，返回 task_id
    GET  /api/attribution/brinson/{task_id}  —— 按 task_id 取归因结果（含瀑布图数据）

任务采用同步计算 + 模块级字典缓存（threading.Lock 保护）：POST 立即完成计算并落库，
GET 仅做结果读取。
"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from analysis.brinson import BrinsonAttribution

logger = logging.getLogger("realtime_server")

# 任务结果缓存：task_id -> result dict（线程安全）
_RESULTS_LOCK = threading.Lock()
_RESULTS: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class BrinsonGroup(BaseModel):
    """单个分组（行业/风格/市值）的权重与收益。"""

    name: str
    w_p: float = Field(..., description="组合权重")
    w_b: float = Field(..., description="基准权重")
    r_p: float = Field(..., description="组合分组收益")
    r_b: float = Field(..., description="基准分组收益")


class BrinsonRequest(BaseModel):
    """Brinson 归因请求。

    - 单期：填 ``groups``；
    - 多期：填 ``periods``（每期一个 groups 列表），并可指定 ``link_method``。
    """

    groups: List[BrinsonGroup] = []
    periods: Optional[List[List[BrinsonGroup]]] = None
    dimension: str = "industry"
    model: str = "bhb"
    period: Optional[str] = None
    link_method: str = "carino"  # carino / grap


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_brinson_routes(app: FastAPI, ok: Any, err: Any) -> None:
    """注册 Brinson 绩效归因相关路由。

    Args:
        app: FastAPI 实例。
        ok / err: server.py 的统一响应封装：
            ``ok(data=None, message="success") -> dict``；
            ``err(code, message, http_status=400) -> JSONResponse``。
    """

    @app.post("/api/attribution/brinson")
    async def run_brinson(req: BrinsonRequest):
        """提交 Brinson 归因任务。

        请求体:
            groups: 单期分组列表。
            periods: 多期分组列表（可选，提供时走多期链接）。
            dimension: 维度名 industry/style/market_cap。
            model: bhb / fachler，默认 bhb。
            link_method: carino / grap，默认 carino。

        Returns:
            ``{"task_id": "..."}``。
        """
        if req.model not in ("bhb", "fachler"):
            return err(40001, "model 必须为 bhb/fachler")
        if req.link_method not in ("carino", "grap"):
            return err(40002, "link_method 必须为 carino/grap")
        if not req.groups and not req.periods:
            return err(40003, "必须提供 groups（单期）或 periods（多期）")

        try:
            if req.periods:
                period_inputs = [
                    dict(
                        sectors=[g.name for g in p],
                        w_p=[g.w_p for g in p],
                        w_b=[g.w_b for g in p],
                        r_p_sector=[g.r_p for g in p],
                        r_b_sector=[g.r_b for g in p],
                    )
                    for p in req.periods
                ]
                result = BrinsonAttribution.multi_period_attribute(
                    period_inputs,
                    model=req.model,
                    method=req.link_method,
                    dimension=req.dimension,
                )
                result["period"] = req.period
            else:
                result = BrinsonAttribution.attribute(
                    sectors=[g.name for g in req.groups],
                    w_p=[g.w_p for g in req.groups],
                    w_b=[g.w_b for g in req.groups],
                    r_p_sector=[g.r_p for g in req.groups],
                    r_b_sector=[g.r_b for g in req.groups],
                    model=req.model,
                    dimension=req.dimension,
                ).as_dict()
                result["period"] = req.period
        except ValueError as e:
            # 权重不闭合 / 长度不一致 / 模型非法
            return err(40004, f"输入校验失败: {e}")
        except Exception as e:  # noqa: BLE001
            logger.exception("Brinson 归因计算失败")
            return err(50000, f"Brinson 归因计算失败: {e}", http_status=500)

        task_id = uuid.uuid4().hex[:12]
        with _RESULTS_LOCK:
            _RESULTS[task_id] = result
        return ok({"task_id": task_id})

    @app.get("/api/attribution/brinson/{task_id}")
    async def get_brinson(task_id: str):
        """按 task_id 获取归因结果。

        返回三效应明细、闭合残差、瀑布图数据（waterfall）与分组条形图数据。
        task_id 不存在返回 404。
        """
        with _RESULTS_LOCK:
            result = _RESULTS.get(task_id)
        if result is None:
            return err(40404, f"task_id 不存在: {task_id}", http_status=404)
        return ok(result)
