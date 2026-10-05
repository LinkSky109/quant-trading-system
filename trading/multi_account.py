"""多账户资金路由模块。

实现账户层级管理、资金分配规则、订单路由和资金调拨。
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SubAccount:
    """子账户信息。"""
    id: str
    name: str
    parent_id: str
    strategy: str = ""
    risk_level: str = "medium"  # low/medium/high
    initial_capital: float = 0.0
    allocated_capital: float = 0.0
    used_capital: float = 0.0
    available_capital: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransferRecord:
    """资金调拨记录。"""
    transfer_id: str
    from_account: str
    to_account: str
    amount: float
    reason: str
    status: str = "pending"  # pending/completed/failed
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RouteRule:
    """订单路由规则。"""
    rule_id: str
    symbol_pattern: str  # 支持通配符，如 "600*.SH"
    strategy: str
    target_account: str
    priority: int = 0
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CapitalAllocator(ABC):
    """资金分配器抽象基类。"""

    @abstractmethod
    def allocate(
        self,
        total_capital: float,
        accounts: List[SubAccount],
        **kwargs: Any,
    ) -> Dict[str, float]:
        """返回 account_id -> allocated_amount 映射。"""


class EqualWeightAllocator(CapitalAllocator):
    """等权分配器。"""

    def allocate(
        self,
        total_capital: float,
        accounts: List[SubAccount],
        **kwargs: Any,
    ) -> Dict[str, float]:
        if not accounts:
            return {}
        per_account = total_capital / len(accounts)
        return {acc.id: per_account for acc in accounts}


class PerformanceBasedAllocator(CapitalAllocator):
    """按策略表现分配（夏普比率加权）。"""

    def allocate(
        self,
        total_capital: float,
        accounts: List[SubAccount],
        performance: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        performance = performance or {}
        if not accounts:
            return {}
        # 收集各账户的夏普值，默认 1.0
        sharps = [performance.get(acc.id, 1.0) for acc in accounts]
        total_sharp = sum(sharps)
        if total_sharp <= 0:
            total_sharp = len(accounts)
            sharps = [1.0] * len(accounts)
        return {
            acc.id: total_capital * s / total_sharp
            for acc, s in zip(accounts, sharps)
        }


class RiskBudgetAllocator(CapitalAllocator):
    """按风险预算分配（风险等级越低，分配越多）。"""

    RISK_WEIGHTS = {"low": 1.5, "medium": 1.0, "high": 0.5}

    def allocate(
        self,
        total_capital: float,
        accounts: List[SubAccount],
        **kwargs: Any,
    ) -> Dict[str, float]:
        if not accounts:
            return {}
        weights = [self.RISK_WEIGHTS.get(acc.risk_level, 1.0) for acc in accounts]
        total_weight = sum(weights)
        return {
            acc.id: total_capital * w / total_weight
            for acc, w in zip(accounts, weights)
        }


class MultiAccountRouter:
    """多账户资金路由器。"""

    def __init__(self, total_capital: float = 1_000_000.0) -> None:
        self.total_capital = total_capital
        self._accounts: Dict[str, SubAccount] = {}
        self._transfers: List[TransferRecord] = []
        self._route_rules: List[RouteRule] = []
        self._audit_log: List[Dict[str, Any]] = []
        self._allocators: Dict[str, CapitalAllocator] = {
            "equal": EqualWeightAllocator(),
            "performance": PerformanceBasedAllocator(),
            "risk_budget": RiskBudgetAllocator(),
        }

    # ------------------------------------------------------------------
    # 账户管理
    # ------------------------------------------------------------------

    def create_account(
        self,
        account_id: str,
        name: str,
        parent_id: str = "master",
        strategy: str = "",
        risk_level: str = "medium",
        initial_capital: float = 0.0,
        **metadata: Any,
    ) -> SubAccount:
        if account_id in self._accounts:
            raise ValueError(f"账户已存在: {account_id}")
        acc = SubAccount(
            id=account_id,
            name=name,
            parent_id=parent_id,
            strategy=strategy,
            risk_level=risk_level,
            initial_capital=initial_capital,
            allocated_capital=initial_capital,
            available_capital=initial_capital,
            metadata=metadata,
        )
        self._accounts[account_id] = acc
        self._log_audit("create_account", {"account_id": account_id, "name": name})
        return acc

    def get_account(self, account_id: str) -> Optional[SubAccount]:
        return self._accounts.get(account_id)

    def list_accounts(self) -> List[SubAccount]:
        return list(self._accounts.values())

    def delete_account(self, account_id: str) -> bool:
        if account_id not in self._accounts:
            return False
        del self._accounts[account_id]
        self._log_audit("delete_account", {"account_id": account_id})
        return True

    def update_account(
        self,
        account_id: str,
        **kwargs: Any,
    ) -> Optional[SubAccount]:
        acc = self._accounts.get(account_id)
        if not acc:
            return None
        for k, v in kwargs.items():
            if hasattr(acc, k):
                setattr(acc, k, v)
        self._log_audit("update_account", {"account_id": account_id, "fields": list(kwargs.keys())})
        return acc

    # ------------------------------------------------------------------
    # 资金分配
    # ------------------------------------------------------------------

    def allocate_capital(
        self,
        method: str = "equal",
        **kwargs: Any,
    ) -> Dict[str, float]:
        allocator = self._allocators.get(method)
        if not allocator:
            raise ValueError(f"不支持的分配方法: {method}")
        accounts = self.list_accounts()
        allocations = allocator.allocate(self.total_capital, accounts, **kwargs)
        for acc_id, amount in allocations.items():
            acc = self._accounts.get(acc_id)
            if acc:
                acc.allocated_capital = amount
                acc.available_capital = amount - acc.used_capital
        self._log_audit("allocate_capital", {"method": method, "allocations": allocations})
        return allocations

    def get_allocator_names(self) -> List[str]:
        return list(self._allocators.keys())

    # ------------------------------------------------------------------
    # 订单路由
    # ------------------------------------------------------------------

    def add_route_rule(
        self,
        rule_id: str,
        symbol_pattern: str,
        strategy: str,
        target_account: str,
        priority: int = 0,
    ) -> RouteRule:
        if target_account not in self._accounts:
            raise ValueError(f"目标账户不存在: {target_account}")
        rule = RouteRule(
            rule_id=rule_id,
            symbol_pattern=symbol_pattern,
            strategy=strategy,
            target_account=target_account,
            priority=priority,
        )
        self._route_rules.append(rule)
        self._route_rules.sort(key=lambda r: r.priority, reverse=True)
        self._log_audit("add_route_rule", rule.to_dict())
        return rule

    def list_route_rules(self) -> List[RouteRule]:
        return list(self._route_rules)

    def delete_route_rule(self, rule_id: str) -> bool:
        for i, r in enumerate(self._route_rules):
            if r.rule_id == rule_id:
                del self._route_rules[i]
                self._log_audit("delete_route_rule", {"rule_id": rule_id})
                return True
        return False

    def route_order(
        self,
        symbol: str,
        strategy: str,
    ) -> Optional[str]:
        """根据标的和策略选择执行账户，返回 account_id。"""
        import fnmatch
        for rule in self._route_rules:
            if not rule.active:
                continue
            if fnmatch.fnmatch(symbol, rule.symbol_pattern) and rule.strategy == strategy:
                self._log_audit("route_order", {"symbol": symbol, "strategy": strategy, "target": rule.target_account})
                return rule.target_account
        # 无匹配规则时返回第一个账户
        if self._accounts:
            default = next(iter(self._accounts))
            self._log_audit("route_order_default", {"symbol": symbol, "strategy": strategy, "target": default})
            return default
        return None

    # ------------------------------------------------------------------
    # 资金调拨
    # ------------------------------------------------------------------

    def transfer(
        self,
        from_account: str,
        to_account: str,
        amount: float,
        reason: str = "",
    ) -> TransferRecord:
        if from_account not in self._accounts or to_account not in self._accounts:
            raise ValueError("账户不存在")
        if from_account == to_account:
            raise ValueError("不能向同一账户调拨")
        from_acc = self._accounts[from_account]
        if from_acc.available_capital < amount:
            raise ValueError("转出账户可用资金不足")

        transfer_id = f"TRF-{int(time.time()*1000)}"
        record = TransferRecord(
            transfer_id=transfer_id,
            from_account=from_account,
            to_account=to_account,
            amount=amount,
            reason=reason,
            status="pending",
        )
        self._transfers.append(record)

        # 执行调拨
        from_acc.available_capital -= amount
        from_acc.allocated_capital -= amount
        to_acc = self._accounts[to_account]
        to_acc.available_capital += amount
        to_acc.allocated_capital += amount

        record.status = "completed"
        record.completed_at = datetime.now().isoformat()
        self._log_audit("transfer", record.to_dict())
        return record

    def list_transfers(
        self,
        account_id: Optional[str] = None,
    ) -> List[TransferRecord]:
        if account_id is None:
            return list(self._transfers)
        return [
            t for t in self._transfers
            if t.from_account == account_id or t.to_account == account_id
        ]

    # ------------------------------------------------------------------
    # 报表
    # ------------------------------------------------------------------

    def consolidated_report(self) -> Dict[str, Any]:
        accounts = self.list_accounts()
        total_allocated = sum(a.allocated_capital for a in accounts)
        total_used = sum(a.used_capital for a in accounts)
        total_pnl = sum(a.pnl for a in accounts)
        return {
            "total_capital": self.total_capital,
            "total_allocated": total_allocated,
            "total_available": total_allocated - total_used,
            "total_used": total_used,
            "total_pnl": total_pnl,
            "total_pnl_pct": (total_pnl / self.total_capital * 100) if self.total_capital else 0,
            "account_count": len(accounts),
            "accounts": [a.to_dict() for a in accounts],
        }

    def account_report(self, account_id: str) -> Optional[Dict[str, Any]]:
        acc = self._accounts.get(account_id)
        if not acc:
            return None
        transfers = self.list_transfers(account_id)
        return {
            "account": acc.to_dict(),
            "transfers": [t.to_dict() for t in transfers],
        }

    # ------------------------------------------------------------------
    # 性能更新
    # ------------------------------------------------------------------

    def update_performance(
        self,
        account_id: str,
        pnl: float,
        used_capital: float,
    ) -> Optional[SubAccount]:
        acc = self._accounts.get(account_id)
        if not acc:
            return None
        acc.pnl = pnl
        acc.pnl_pct = (pnl / acc.initial_capital * 100) if acc.initial_capital else 0
        acc.used_capital = used_capital
        acc.available_capital = acc.allocated_capital - used_capital
        return acc

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
