"""多账户资金路由 API 路由。

按 server.py 现有扩展路由模式挂载。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from trading.multi_account import MultiAccountRouter

logger = logging.getLogger("realtime_server")

# 模块级单例（懒加载）
_default_router: Optional[MultiAccountRouter] = None


def _get_default_router() -> MultiAccountRouter:
    """获取模块级单例。"""
    global _default_router
    if _default_router is None:
        _default_router = MultiAccountRouter(total_capital=1_000_000.0)
    return _default_router


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------

class CreateAccountReq(BaseModel):
    account_id: str
    name: str
    parent_id: str = "master"
    strategy: str = ""
    risk_level: str = "medium"
    initial_capital: float = 0.0


class UpdateAccountReq(BaseModel):
    name: Optional[str] = None
    strategy: Optional[str] = None
    risk_level: Optional[str] = None


class AllocateReq(BaseModel):
    method: str = "equal"


class RouteReq(BaseModel):
    symbol: str
    strategy: str


class AddRouteRuleReq(BaseModel):
    rule_id: str
    symbol_pattern: str
    strategy: str
    target_account: str
    priority: int = 0


class TransferReq(BaseModel):
    from_account: str
    to_account: str
    amount: float
    reason: str = ""


class PerformanceUpdateReq(BaseModel):
    pnl: float
    used_capital: float


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------

def register_multi_account_routes(
    app: FastAPI,
    ok: Any,
    err: Any,
    router: Optional[MultiAccountRouter] = None,
) -> None:
    """注册多账户资金路由。"""
    r = router or _get_default_router()

    @app.get("/api/multi_account/accounts")
    async def list_accounts() -> Any:
        accounts = r.list_accounts()
        return ok(data=[a.to_dict() for a in accounts])

    @app.post("/api/multi_account/accounts")
    async def create_account(req: CreateAccountReq) -> Any:
        try:
            acc = r.create_account(
                account_id=req.account_id,
                name=req.name,
                parent_id=req.parent_id,
                strategy=req.strategy,
                risk_level=req.risk_level,
                initial_capital=req.initial_capital,
            )
            return ok(data=acc.to_dict())
        except ValueError as e:
            return err(400, str(e))

    @app.get("/api/multi_account/accounts/{account_id}")
    async def get_account(account_id: str) -> Any:
        acc = r.get_account(account_id)
        if not acc:
            return err(404, "账户不存在")
        return ok(data=acc.to_dict())

    @app.patch("/api/multi_account/accounts/{account_id}")
    async def update_account(account_id: str, req: UpdateAccountReq) -> Any:
        kwargs = {k: v for k, v in req.model_dump().items() if v is not None}
        acc = r.update_account(account_id, **kwargs)
        if not acc:
            return err(404, "账户不存在")
        return ok(data=acc.to_dict())

    @app.delete("/api/multi_account/accounts/{account_id}")
    async def delete_account(account_id: str) -> Any:
        if r.delete_account(account_id):
            return ok(message="账户已删除")
        return err(404, "账户不存在")

    @app.post("/api/multi_account/allocate")
    async def allocate_capital(req: AllocateReq) -> Any:
        try:
            allocations = r.allocate_capital(req.method)
            return ok(data=allocations)
        except ValueError as e:
            return err(400, str(e))

    @app.get("/api/multi_account/allocators")
    async def list_allocators() -> Any:
        return ok(data=r.get_allocator_names())

    @app.post("/api/multi_account/route")
    async def route_order(req: RouteReq) -> Any:
        target = r.route_order(req.symbol, req.strategy)
        if target is None:
            return err(404, "无可用账户")
        return ok(data={"target_account": target})

    @app.get("/api/multi_account/route_rules")
    async def list_route_rules() -> Any:
        rules = r.list_route_rules()
        return ok(data=[rule.to_dict() for rule in rules])

    @app.post("/api/multi_account/route_rules")
    async def add_route_rule(req: AddRouteRuleReq) -> Any:
        try:
            rule = r.add_route_rule(
                req.rule_id, req.symbol_pattern, req.strategy,
                req.target_account, req.priority,
            )
            return ok(data=rule.to_dict())
        except ValueError as e:
            return err(400, str(e))

    @app.delete("/api/multi_account/route_rules/{rule_id}")
    async def delete_route_rule(rule_id: str) -> Any:
        if r.delete_route_rule(rule_id):
            return ok(message="规则已删除")
        return err(404, "规则不存在")

    @app.post("/api/multi_account/transfer")
    async def transfer(req: TransferReq) -> Any:
        try:
            record = r.transfer(req.from_account, req.to_account, req.amount, req.reason)
            return ok(data=record.to_dict())
        except ValueError as e:
            return err(400, str(e))

    @app.get("/api/multi_account/transfers")
    async def list_transfers(account_id: Optional[str] = None) -> Any:
        transfers = r.list_transfers(account_id)
        return ok(data=[t.to_dict() for t in transfers])

    @app.get("/api/multi_account/report/consolidated")
    async def consolidated_report() -> Any:
        return ok(data=r.consolidated_report())

    @app.get("/api/multi_account/report/{account_id}")
    async def account_report(account_id: str) -> Any:
        report = r.account_report(account_id)
        if not report:
            return err(404, "账户不存在")
        return ok(data=report)

    @app.post("/api/multi_account/performance/{account_id}")
    async def update_performance(account_id: str, req: PerformanceUpdateReq) -> Any:
        acc = r.update_performance(account_id, req.pnl, req.used_capital)
        if not acc:
            return err(404, "账户不存在")
        return ok(data=acc.to_dict())

    @app.get("/api/multi_account/audit")
    async def get_audit_log(action: Optional[str] = None, limit: int = 100) -> Any:
        return ok(data=r.get_audit_log(action=action, limit=limit))
