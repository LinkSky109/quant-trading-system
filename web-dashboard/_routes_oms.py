"""订单管理系统（OMS）API 路由。

按 server.py 现有扩展路由模式挂载：

    try:
        from _routes_oms import register_oms_routes
        register_oms_routes(app, ok, err)
    except Exception as e:
        logger.warning("OMS 路由注册失败: %s", e)

端点：
    POST   /api/oms/orders                  —— 提交订单
    DELETE /api/oms/orders/{order_id}       —— 撤单
    GET    /api/oms/orders                  —— 查询订单（支持 status/symbol 过滤）
    GET    /api/oms/orders/{order_id}       —— 订单详情
    GET    /api/oms/fills                   —— 成交记录（支持 order_id 过滤）
    POST   /api/oms/orders/{order_id}/reject —— 废单
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

from trading.oms import (
    InvalidOrderStateError,
    Order,
    OrderManagementSystem,
    fill_to_dict,
    order_to_dict,
)

logger = logging.getLogger("realtime_server")

# 项目根目录与默认 SQLite 路径（锚定文件位置，不受启动工作目录影响）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB_PATH = str(_PROJECT_ROOT / "data" / "quant_trading.db")

# 模块级 OMS 单例（懒加载）
_default_oms: Optional[OrderManagementSystem] = None


def _get_default_oms() -> OrderManagementSystem:
    """获取模块级 OMS 单例（首次调用时初始化）。"""
    global _default_oms
    if _default_oms is None:
        _default_oms = OrderManagementSystem(db_path=_DEFAULT_DB_PATH)
    return _default_oms


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class OrderReq(BaseModel):
    """提交订单请求体（与 :class:`Order` 字段对应，order_id 可省略自动生成）。"""

    order_id: Optional[str] = None
    symbol: str
    side: str
    order_type: str
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    strategy_name: str = ""
    account_id: str = ""
    timeout_seconds: int = 300


class CancelReq(BaseModel):
    """撤单请求体（reason 可选，亦可走查询参数）。"""

    reason: str = ""


class RejectReq(BaseModel):
    """废单请求体。"""

    reason: str


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_oms_routes(
    app: FastAPI,
    ok: Any,
    err: Any,
    oms: Optional[OrderManagementSystem] = None,
) -> None:
    """注册 OMS 相关路由。

    Args:
        app: FastAPI 实例。
        ok / err: server.py 的统一响应封装。
        oms: 可选注入的 OMS 实例（测试隔离用）；缺省使用模块级单例。
    """
    manager: OrderManagementSystem = oms if oms is not None else _get_default_oms()

    @app.post("/api/oms/orders")
    async def submit_order(req: OrderReq):
        """提交订单。

        请求体见 :class:`OrderReq`；``order_id`` 缺省时自动生成。
        """
        try:
            order = Order(
                order_id=req.order_id or OrderManagementSystem.generate_order_id(),
                symbol=req.symbol,
                side=req.side,
                order_type=req.order_type,
                quantity=req.quantity,
                limit_price=req.limit_price,
                stop_price=req.stop_price,
                strategy_name=req.strategy_name,
                account_id=req.account_id,
                timeout_seconds=req.timeout_seconds,
            )
            submitted = manager.submit_order(order)
            return ok(order_to_dict(submitted))
        except ValueError as e:
            return err(400, f"订单参数非法: {e}")
        except InvalidOrderStateError as e:
            return err(400, f"订单状态非法: {e}")
        except Exception as e:  # pragma: no cover
            logger.exception("提交订单失败")
            return err(500, f"提交订单失败: {e}", http_status=500)

    @app.delete("/api/oms/orders/{order_id}")
    async def cancel_order(
        order_id: str,
        reason: str = Query(default=""),
        body: Optional[CancelReq] = None,
    ):
        """撤销指定订单，reason 可走查询参数或请求体。"""
        if body is not None and body.reason:
            reason = body.reason
        try:
            order = manager.cancel_order(order_id, reason=reason)
            return ok(order_to_dict(order))
        except KeyError:
            return err(404, f"订单不存在: {order_id}", http_status=404)
        except InvalidOrderStateError as e:
            return err(400, f"撤单失败: {e}")
        except Exception as e:  # pragma: no cover
            logger.exception("撤单失败")
            return err(500, f"撤单失败: {e}", http_status=500)

    @app.get("/api/oms/orders")
    async def list_orders(
        status: Optional[str] = Query(default=None),
        symbol: Optional[str] = Query(default=None),
    ):
        """查询订单列表，支持 status 与 symbol 过滤。"""
        orders = manager.get_all_orders()
        if status:
            orders = [o for o in orders if o.status == status]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return ok([order_to_dict(o) for o in orders])

    @app.get("/api/oms/orders/{order_id}")
    async def get_order(order_id: str):
        """订单详情。"""
        order = manager.get_order(order_id)
        if order is None:
            return err(404, f"订单不存在: {order_id}", http_status=404)
        return ok(order_to_dict(order))

    @app.get("/api/oms/fills")
    async def list_fills(order_id: Optional[str] = Query(default=None)):
        """成交记录，支持 order_id 过滤。"""
        fills = manager.get_fills(order_id=order_id)
        return ok([fill_to_dict(f) for f in fills])

    @app.post("/api/oms/orders/{order_id}/reject")
    async def reject_order(order_id: str, req: RejectReq):
        """废单（请求体含 reason）。"""
        try:
            order = manager.reject_order(order_id, reason=req.reason)
            return ok(order_to_dict(order))
        except KeyError:
            return err(404, f"订单不存在: {order_id}", http_status=404)
        except InvalidOrderStateError as e:
            return err(400, f"废单失败: {e}")
        except ValueError as e:
            return err(400, f"废单参数非法: {e}")
        except Exception as e:  # pragma: no cover
            logger.exception("废单失败")
            return err(500, f"废单失败: {e}", http_status=500)
