"""算法交易（TWAP/VWAP）API 路由。

按 server.py 现有扩展路由模式挂载：

    try:
        from _routes_algo import register_algo_routes
        register_algo_routes(app, ok, err)
    except Exception as e:
        logger.warning("算法交易路由注册失败: %s", e)

端点：
    POST /api/algo/twap                       —— 创建 TWAP 计划
    POST /api/algo/vwap                        —— 创建 VWAP 计划
    GET  /api/algo/orders/{algo_order_id}      —— 执行状态
    POST /api/algo/orders/{algo_order_id}/pause
    POST /api/algo/orders/{algo_order_id}/resume
    POST /api/algo/orders/{algo_order_id}/stop
    GET  /api/algo/orders/{algo_order_id}/report —— 执行质量报告
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from trading.algorithmic import (
    ALGO_STOPPED,
    BaseAlgoExecutor,
    TWAPExecutor,
    VWAPExecutor,
    AlgoOrderError,
)
from trading.oms import OrderManagementSystem

logger = logging.getLogger("realtime_server")

# 项目根目录与默认 SQLite 路径（锚定文件位置，不受启动工作目录影响）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB_PATH = str(_PROJECT_ROOT / "data" / "quant_trading.db")

# ---------------------------------------------------------------------------
# 模块级注册表（执行器）与共享 OMS 单例
# ---------------------------------------------------------------------------

#: algo_order_id -> 执行器实例
_registry: Dict[str, BaseAlgoExecutor] = {}
_registry_lock = threading.Lock()

_default_oms: Optional[OrderManagementSystem] = None


def _get_default_oms() -> OrderManagementSystem:
    """获取模块级 OMS 单例（首次调用时初始化，db 默认 data/quant_trading.db）。"""
    global _default_oms
    if _default_oms is None:
        _default_oms = OrderManagementSystem(db_path=_DEFAULT_DB_PATH)
    return _default_oms


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class TwapCreateReq(BaseModel):
    """创建 TWAP 母单请求体。"""

    symbol: str
    side: str
    total_quantity: float
    duration_minutes: Optional[float] = None
    num_slices: int
    interval_seconds: Optional[float] = None
    order_type: str = "market"
    limit_price: Optional[float] = None
    auto_start: bool = True


class VwapCreateReq(BaseModel):
    """创建 VWAP 母单请求体。"""

    symbol: str
    side: str
    total_quantity: float
    volume_profile: List[Dict[str, Any]]
    participation_rate: float = 0.10
    order_type: str = "market"
    limit_price: Optional[float] = None
    auto_start: bool = True


# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------


def _executor_status_dict(exe: BaseAlgoExecutor) -> Dict[str, Any]:
    """母单状态 + 逐片状态摘要。"""
    order = exe.update_from_oms()
    return {
        "algo_order_id": order.algo_order_id,
        "symbol": order.symbol,
        "side": order.side,
        "strategy": order.strategy,
        "status": order.status,
        "total_quantity": order.total_quantity,
        "filled_quantity": order.filled_quantity,
        "remaining_quantity": order.remaining_quantity,
        "avg_fill_price": order.avg_fill_price,
        "progress_percent": round(order.progress_percent, 4),
        "leftover_quantity": order.leftover_quantity,
        "params": order.params,
        "slices": [
            {
                "slice_index": s.slice_index,
                "offset_seconds": s.offset_seconds,
                "planned_quantity": s.planned_quantity,
                "filled_quantity": s.filled_quantity,
                "child_order_id": s.child_order_id,
                "status": s.status,
            }
            for s in order.slices
        ],
    }


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_algo_routes(
    app: FastAPI,
    ok: Any,
    err: Any,
    oms: Optional[OrderManagementSystem] = None,
    registry: Optional[Dict[str, BaseAlgoExecutor]] = None,
) -> None:
    """注册算法交易相关路由。

    Args:
        app: FastAPI 实例。
        ok / err: server.py 的统一响应封装。
        oms: 可选注入的 OMS 实例（测试隔离用）；缺省使用模块级单例。
        registry: 可选注入的执行器注册表（测试隔离用）；缺省使用模块级字典。
    """
    manager: OrderManagementSystem = (
        oms if oms is not None else _get_default_oms()
    )
    book: Dict[str, BaseAlgoExecutor] = (
        registry if registry is not None else _registry
    )

    def _get(algo_id: str) -> Optional[BaseAlgoExecutor]:
        with _registry_lock:
            return book.get(algo_id)

    @app.post("/api/algo/twap")
    async def create_twap(req: TwapCreateReq):
        """创建 TWAP 母单，返回 algo_order_id 与 schedule 摘要。"""
        try:
            exe = TWAPExecutor(
                oms=manager,
                symbol=req.symbol,
                side=req.side,
                total_quantity=req.total_quantity,
                num_slices=req.num_slices,
                duration_minutes=req.duration_minutes,
                interval_seconds=req.interval_seconds,
                order_type=req.order_type,
                limit_price=req.limit_price,
            )
        except AlgoOrderError as e:
            return err(400, f"TWAP 参数非法: {e}")
        except Exception as e:  # pragma: no cover
            logger.exception("创建 TWAP 失败")
            return err(500, f"创建 TWAP 失败: {e}", http_status=500)

        with _registry_lock:
            book[exe.order.algo_order_id] = exe

        if req.auto_start:
            try:
                exe.start()
            except AlgoOrderError as e:
                return err(400, f"启动 TWAP 失败: {e}")

        summary = _executor_status_dict(exe)
        return ok(summary)

    @app.post("/api/algo/vwap")
    async def create_vwap(req: VwapCreateReq):
        """创建 VWAP 母单，返回 algo_order_id 与 schedule 摘要。"""
        try:
            exe = VWAPExecutor(
                oms=manager,
                symbol=req.symbol,
                side=req.side,
                total_quantity=req.total_quantity,
                volume_profile=req.volume_profile,
                participation_rate=req.participation_rate,
                order_type=req.order_type,
                limit_price=req.limit_price,
            )
        except AlgoOrderError as e:
            return err(400, f"VWAP 参数非法: {e}")
        except Exception as e:  # pragma: no cover
            logger.exception("创建 VWAP 失败")
            return err(500, f"创建 VWAP 失败: {e}", http_status=500)

        with _registry_lock:
            book[exe.order.algo_order_id] = exe

        if req.auto_start:
            try:
                exe.start()
            except AlgoOrderError as e:
                return err(400, f"启动 VWAP 失败: {e}")

        summary = _executor_status_dict(exe)
        return ok(summary)

    @app.get("/api/algo/orders/{algo_order_id}")
    async def get_algo_order(algo_order_id: str):
        """母单执行状态（状态/已成/剩余/均价/进度%/各 slice 状态）。"""
        exe = _get(algo_order_id)
        if exe is None:
            return err(404, f"算法母单不存在: {algo_order_id}", http_status=404)
        return ok(_executor_status_dict(exe))

    @app.post("/api/algo/orders/{algo_order_id}/pause")
    async def pause_algo(algo_order_id: str):
        """暂停母单。"""
        exe = _get(algo_order_id)
        if exe is None:
            return err(404, f"算法母单不存在: {algo_order_id}", http_status=404)
        try:
            exe.pause()
        except AlgoOrderError as e:
            return err(400, f"暂停失败: {e}")
        return ok(_executor_status_dict(exe))

    @app.post("/api/algo/orders/{algo_order_id}/resume")
    async def resume_algo(algo_order_id: str):
        """恢复母单。"""
        exe = _get(algo_order_id)
        if exe is None:
            return err(404, f"算法母单不存在: {algo_order_id}", http_status=404)
        try:
            exe.resume()
        except AlgoOrderError as e:
            return err(400, f"恢复失败: {e}")
        return ok(_executor_status_dict(exe))

    @app.post("/api/algo/orders/{algo_order_id}/stop")
    async def stop_algo(algo_order_id: str):
        """停止母单（撤活动子单、未提交切片作废）。"""
        exe = _get(algo_order_id)
        if exe is None:
            return err(404, f"算法母单不存在: {algo_order_id}", http_status=404)
        try:
            exe.stop()
        except AlgoOrderError as e:
            return err(400, f"停止失败: {e}")
        return ok(_executor_status_dict(exe))

    @app.get("/api/algo/orders/{algo_order_id}/report")
    async def algo_report(algo_order_id: str):
        """执行质量报告；未完成返回阶段性报告并标注 stages=interim。"""
        exe = _get(algo_order_id)
        if exe is None:
            return err(404, f"算法母单不存在: {algo_order_id}", http_status=404)
        return ok(exe.execution_quality_report())
