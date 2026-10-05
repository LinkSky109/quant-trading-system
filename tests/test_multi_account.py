"""多账户资金路由单元测试。"""
from __future__ import annotations

import pytest

from trading.multi_account import (
    EqualWeightAllocator,
    MultiAccountRouter,
    PerformanceBasedAllocator,
    RiskBudgetAllocator,
    RouteRule,
    SubAccount,
    TransferRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def router() -> MultiAccountRouter:
    return MultiAccountRouter(total_capital=1_000_000.0)


@pytest.fixture
def populated_router(router: MultiAccountRouter) -> MultiAccountRouter:
    router.create_account("acc1", "策略A账户", strategy="ma_cross", risk_level="medium", initial_capital=100_000)
    router.create_account("acc2", "策略B账户", strategy="bollinger", risk_level="high", initial_capital=100_000)
    router.create_account("acc3", "策略C账户", strategy="momentum", risk_level="low", initial_capital=100_000)
    return router


# ---------------------------------------------------------------------------
# 子账户管理
# ---------------------------------------------------------------------------

class TestSubAccountManagement:
    def test_create_account(self, router):
        acc = router.create_account("test1", "测试账户", strategy="ma_cross")
        assert acc.id == "test1"
        assert acc.name == "测试账户"
        assert acc.strategy == "ma_cross"

    def test_create_duplicate_raises(self, router):
        router.create_account("test1", "测试账户")
        with pytest.raises(ValueError, match="账户已存在"):
            router.create_account("test1", "重复账户")

    def test_get_account(self, router):
        router.create_account("test1", "测试账户")
        acc = router.get_account("test1")
        assert acc is not None
        assert acc.name == "测试账户"

    def test_get_account_missing(self, router):
        assert router.get_account("missing") is None

    def test_list_accounts(self, populated_router):
        accounts = populated_router.list_accounts()
        assert len(accounts) == 3

    def test_delete_account(self, populated_router):
        assert populated_router.delete_account("acc1") is True
        assert populated_router.get_account("acc1") is None
        assert len(populated_router.list_accounts()) == 2

    def test_delete_account_missing(self, populated_router):
        assert populated_router.delete_account("missing") is False

    def test_update_account(self, populated_router):
        acc = populated_router.update_account("acc1", name="新名称", risk_level="high")
        assert acc is not None
        assert acc.name == "新名称"
        assert acc.risk_level == "high"


# ---------------------------------------------------------------------------
# 资金分配规则
# ---------------------------------------------------------------------------

class TestAllocationRules:
    def test_equal_weight_allocator(self):
        allocator = EqualWeightAllocator()
        accounts = [
            SubAccount(id="a1", name="A", parent_id="m"),
            SubAccount(id="a2", name="B", parent_id="m"),
            SubAccount(id="a3", name="C", parent_id="m"),
        ]
        result = allocator.allocate(300_000, accounts)
        assert result == {"a1": 100_000.0, "a2": 100_000.0, "a3": 100_000.0}

    def test_equal_weight_empty(self):
        allocator = EqualWeightAllocator()
        assert allocator.allocate(100_000, []) == {}

    def test_performance_based_allocator(self):
        allocator = PerformanceBasedAllocator()
        accounts = [
            SubAccount(id="a1", name="A", parent_id="m"),
            SubAccount(id="a2", name="B", parent_id="m"),
        ]
        result = allocator.allocate(300_000, accounts, performance={"a1": 2.0, "a2": 1.0})
        assert result["a1"] == 200_000.0
        assert result["a2"] == 100_000.0

    def test_performance_based_zero_sharpe(self):
        allocator = PerformanceBasedAllocator()
        accounts = [SubAccount(id="a1", name="A", parent_id="m")]
        result = allocator.allocate(100_000, accounts, performance={"a1": 0.0})
        assert result["a1"] == 100_000.0

    def test_risk_budget_allocator(self):
        allocator = RiskBudgetAllocator()
        accounts = [
            SubAccount(id="a1", name="A", parent_id="m", risk_level="low"),
            SubAccount(id="a2", name="B", parent_id="m", risk_level="high"),
        ]
        result = allocator.allocate(300_000, accounts)
        # low=1.5, high=0.5 -> 总权重=2.0
        assert result["a1"] == 225_000.0  # 300k * 1.5/2.0
        assert result["a2"] == 75_000.0   # 300k * 0.5/2.0

    def test_allocate_capital_integration(self, populated_router):
        result = populated_router.allocate_capital("equal")
        assert len(result) == 3
        for v in result.values():
            assert v == pytest.approx(1_000_000.0 / 3, abs=0.01)

    def test_allocate_capital_risk_budget(self, populated_router):
        result = populated_router.allocate_capital("risk_budget")
        assert result["acc3"] > result["acc1"] > result["acc2"]  # low > medium > high

    def test_allocate_unknown_method_raises(self, populated_router):
        with pytest.raises(ValueError, match="不支持的分配方法"):
            populated_router.allocate_capital("unknown")


# ---------------------------------------------------------------------------
# 订单路由
# ---------------------------------------------------------------------------

class TestOrderRouting:
    def test_add_route_rule(self, populated_router):
        rule = populated_router.add_route_rule("r1", "600*.SH", "ma_cross", "acc1")
        assert rule.rule_id == "r1"
        assert rule.target_account == "acc1"

    def test_add_route_rule_invalid_account(self, populated_router):
        with pytest.raises(ValueError, match="目标账户不存在"):
            populated_router.add_route_rule("r1", "*", "ma_cross", "missing")

    def test_list_route_rules(self, populated_router):
        populated_router.add_route_rule("r1", "600*.SH", "ma_cross", "acc1")
        populated_router.add_route_rule("r2", "000*.SZ", "bollinger", "acc2")
        rules = populated_router.list_route_rules()
        assert len(rules) == 2

    def test_delete_route_rule(self, populated_router):
        populated_router.add_route_rule("r1", "*", "ma_cross", "acc1")
        assert populated_router.delete_route_rule("r1") is True
        assert len(populated_router.list_route_rules()) == 0

    def test_delete_route_rule_missing(self, populated_router):
        assert populated_router.delete_route_rule("missing") is False

    def test_route_order_by_rule(self, populated_router):
        populated_router.add_route_rule("r1", "600*.SH", "ma_cross", "acc1")
        target = populated_router.route_order("600519.SH", "ma_cross")
        assert target == "acc1"

    def test_route_order_default(self, populated_router):
        target = populated_router.route_order("UNKNOWN", "unknown")
        assert target in {"acc1", "acc2", "acc3"}

    def test_route_order_priority(self, populated_router):
        populated_router.add_route_rule("r1", "600*.SH", "ma_cross", "acc1", priority=1)
        populated_router.add_route_rule("r2", "6005*.SH", "ma_cross", "acc2", priority=2)
        target = populated_router.route_order("600519.SH", "ma_cross")
        assert target == "acc2"  # 高优先级优先


# ---------------------------------------------------------------------------
# 资金调拨
# ---------------------------------------------------------------------------

class TestCapitalTransfer:
    def test_transfer_success(self, populated_router):
        populated_router.allocate_capital("equal")
        record = populated_router.transfer("acc1", "acc2", 50_000, "测试调拨")
        assert record.status == "completed"
        assert record.from_account == "acc1"
        assert record.to_account == "acc2"
        assert record.amount == 50_000

    def test_transfer_insufficient_funds(self, populated_router):
        populated_router.allocate_capital("equal")
        with pytest.raises(ValueError, match="转出账户可用资金不足"):
            populated_router.transfer("acc1", "acc2", 999_999_999)

    def test_transfer_same_account(self, populated_router):
        with pytest.raises(ValueError, match="不能向同一账户调拨"):
            populated_router.transfer("acc1", "acc1", 1000)

    def test_transfer_invalid_account(self, populated_router):
        with pytest.raises(ValueError, match="账户不存在"):
            populated_router.transfer("acc1", "missing", 1000)

    def test_list_transfers(self, populated_router):
        populated_router.allocate_capital("equal")
        populated_router.transfer("acc1", "acc2", 10_000)
        populated_router.transfer("acc2", "acc3", 5_000)
        all_transfers = populated_router.list_transfers()
        assert len(all_transfers) == 2

    def test_list_transfers_by_account(self, populated_router):
        populated_router.allocate_capital("equal")
        populated_router.transfer("acc1", "acc2", 10_000)
        populated_router.transfer("acc2", "acc3", 5_000)
        acc2_transfers = populated_router.list_transfers("acc2")
        assert len(acc2_transfers) == 2


# ---------------------------------------------------------------------------
# 报表
# ---------------------------------------------------------------------------

class TestReports:
    def test_consolidated_report(self, populated_router):
        populated_router.allocate_capital("equal")
        report = populated_router.consolidated_report()
        assert report["total_capital"] == 1_000_000.0
        assert report["account_count"] == 3
        assert "accounts" in report

    def test_account_report(self, populated_router):
        populated_router.allocate_capital("equal")
        report = populated_router.account_report("acc1")
        assert report is not None
        assert report["account"]["id"] == "acc1"
        assert "transfers" in report

    def test_account_report_missing(self, populated_router):
        assert populated_router.account_report("missing") is None


# ---------------------------------------------------------------------------
# 性能更新
# ---------------------------------------------------------------------------

class TestPerformanceUpdate:
    def test_update_performance(self, populated_router):
        populated_router.allocate_capital("equal")
        acc = populated_router.update_performance("acc1", pnl=5000, used_capital=100_000)
        assert acc is not None
        assert acc.pnl == 5000
        assert acc.used_capital == 100_000

    def test_update_performance_pnl_pct(self, populated_router):
        populated_router.create_account("acc4", "测试", initial_capital=100_000)
        acc = populated_router.update_performance("acc4", pnl=5000, used_capital=50_000)
        assert acc.pnl_pct == 5.0

    def test_update_performance_missing(self, populated_router):
        assert populated_router.update_performance("missing", pnl=100, used_capital=0) is None


# ---------------------------------------------------------------------------
# 审计日志
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_audit_log_created(self, populated_router):
        initial = len(populated_router.get_audit_log())
        populated_router.create_account("audit_test", "审计测试")
        logs = populated_router.get_audit_log()
        assert len(logs) == initial + 1
        assert logs[0]["action"] == "create_account"

    def test_audit_log_filter(self, populated_router):
        populated_router.create_account("a1", "A")
        populated_router.create_account("a2", "B")
        logs = populated_router.get_audit_log(action="create_account")
        assert all(l["action"] == "create_account" for l in logs)

    def test_audit_log_limit(self, populated_router):
        for i in range(5):
            populated_router.create_account(f"limit{i}", f"Limit{i}")
        logs = populated_router.get_audit_log(limit=3)
        assert len(logs) == 3


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------

class TestSubAccountSerialization:
    def test_subaccount_to_dict(self):
        acc = SubAccount(id="a1", name="Test", parent_id="m", initial_capital=100_000)
        d = acc.to_dict()
        assert d["id"] == "a1"
        assert d["name"] == "Test"
        assert d["initial_capital"] == 100_000

    def test_transfer_record_to_dict(self):
        rec = TransferRecord(transfer_id="t1", from_account="a1", to_account="a2", amount=1000, reason="test")
        d = rec.to_dict()
        assert d["transfer_id"] == "t1"
        assert d["status"] == "pending"

    def test_route_rule_to_dict(self):
        rule = RouteRule(rule_id="r1", symbol_pattern="600*.SH", strategy="ma", target_account="a1")
        d = rule.to_dict()
        assert d["rule_id"] == "r1"
        assert d["active"] is True
