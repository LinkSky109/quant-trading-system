"""订单管理系统（OMS）单元测试 + API 集成测试。

覆盖：
- 状态机合法 / 非法转换（非法转换断言抛 InvalidOrderStateError）
- 提交订单 / 撤单 / 废单
- 部分成交（PARTIAL_FILLED、filled_quantity 正确）
- 全部成交（FILLED、filled_at 记录）
- 超时自动撤单（注入 now + 极短 timeout）
- 成交均价计算（多笔不同价成交后 avg_fill_price 正确）
- 持久化（写入后重新实例化能读回订单与成交）
- active / pending / historical 三个订单簿分类正确
- 成交回调被触发
- 审计调用被触发（monkeypatch get_audit_logger）
- 全部 API 端点集成测试（TestClient）
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web-dashboard"))

from trading.oms import (  # noqa: E402
    STATUS_CANCELLED,
    STATUS_FILLED,
    STATUS_PARTIAL_FILLED,
    STATUS_PENDING,
    STATUS_REJECTED,
    STATUS_SUBMITTED,
    Fill,
    InvalidOrderStateError,
    Order,
    OrderManagementSystem,
)
from _routes_oms import register_oms_routes  # noqa: E402


# ---------------------------------------------------------------------------
# 公共夹具 / 工具
# ---------------------------------------------------------------------------


def make_order(
    order_id: str = "o1",
    symbol: str = "AAPL",
    quantity: float = 100.0,
    timeout_seconds: int = 300,
) -> Order:
    """构造一个 PENDING 状态的限价买单。"""
    return Order(
        order_id=order_id,
        symbol=symbol,
        side="buy",
        order_type="limit",
        quantity=quantity,
        limit_price=150.0,
        timeout_seconds=timeout_seconds,
    )


@pytest.fixture
def oms(tmp_path) -> OrderManagementSystem:
    """基于临时 SQLite 的 OMS。"""
    db = tmp_path / "oms_test.db"
    return OrderManagementSystem(db_path=str(db))


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_valid_submit_transition(self, oms):
        o = oms.submit_order(make_order("s1"))
        assert o.status == STATUS_SUBMITTED
        assert o.submitted_at is not None

    def test_pending_to_rejected_valid(self, oms):
        o = make_order("pr1")
        oms._orders[o.order_id] = o  # 直接登记为 PENDING
        oms._rebucket(o)
        result = oms.reject_order(o.order_id, reason="测试废单")
        assert result.status == STATUS_REJECTED
        assert result.reject_reason == "测试废单"

    def test_illegal_transition_pending_to_filled_raises(self, oms):
        o = make_order("bad1")
        with pytest.raises(InvalidOrderStateError):
            oms._transition(o, STATUS_FILLED)

    def test_illegal_transition_from_terminal_raises(self, oms):
        o = oms.submit_order(make_order("term1"))
        oms.cancel_order(o.order_id)  # -> CANCELLED
        with pytest.raises(InvalidOrderStateError):
            oms._transition(o, STATUS_SUBMITTED)

    def test_cannot_submit_non_pending(self, oms):
        o = oms.submit_order(make_order("already"))
        with pytest.raises(InvalidOrderStateError):
            oms.submit_order(o)

    def test_invalid_side_rejected(self, oms):
        bad = Order(order_id="x", symbol="AAPL", side="hold",
                    order_type="limit", quantity=10)
        with pytest.raises(ValueError):
            oms.submit_order(bad)

    def test_invalid_order_type_rejected(self, oms):
        bad = Order(order_id="x", symbol="AAPL", side="buy",
                    order_type="wtf", quantity=10)
        with pytest.raises(ValueError):
            oms.submit_order(bad)


# ---------------------------------------------------------------------------
# 提交 / 撤单 / 废单
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_submit_goes_active(self, oms):
        oms.submit_order(make_order("l1"))
        active = oms.get_active_orders()
        assert len(active) == 1
        assert active[0].status == STATUS_SUBMITTED

    def test_cancel_order(self, oms):
        o = oms.submit_order(make_order("c1"))
        cancelled = oms.cancel_order(o.order_id, reason="风控")
        assert cancelled.status == STATUS_CANCELLED
        assert cancelled.cancelled_at is not None
        assert oms.get_order(o.order_id).status == STATUS_CANCELLED

    def test_cancel_nonexistent_raises(self, oms):
        with pytest.raises(KeyError):
            oms.cancel_order("nope")

    def test_reject_submitted_order(self, oms):
        o = oms.submit_order(make_order("r1"))
        rejected = oms.reject_order(o.order_id, reason="资金不足")
        assert rejected.status == STATUS_REJECTED
        assert rejected.reject_reason == "资金不足"

    def test_reject_empty_reason_raises(self, oms):
        o = oms.submit_order(make_order("r2"))
        with pytest.raises(ValueError):
            oms.reject_order(o.order_id, reason="")


# ---------------------------------------------------------------------------
# 成交：部分成交 / 全成 / 均价
# ---------------------------------------------------------------------------


class TestFills:
    def test_partial_fill(self, oms):
        o = oms.submit_order(make_order("p1", quantity=100))
        fill = oms.process_fill(o.order_id, 40.0, 150.0)
        refreshed = oms.get_order(o.order_id)
        assert refreshed.status == STATUS_PARTIAL_FILLED
        assert refreshed.filled_quantity == 40.0
        assert fill.fill_id == "p1-1"

    def test_full_fill(self, oms):
        o = oms.submit_order(make_order("f1", quantity=100))
        oms.process_fill(o.order_id, 100.0, 150.0)
        refreshed = oms.get_order(o.order_id)
        assert refreshed.status == STATUS_FILLED
        assert refreshed.filled_quantity == 100.0
        assert refreshed.filled_at is not None

    def test_avg_fill_price_weighted(self, oms):
        o = oms.submit_order(make_order("avg1", quantity=100))
        oms.process_fill(o.order_id, 30.0, 100.0)   # 均 100
        oms.process_fill(o.order_id, 70.0, 120.0)   # 均 (30*100+70*120)/100
        refreshed = oms.get_order(o.order_id)
        expected = (30.0 * 100.0 + 70.0 * 120.0) / 100.0
        assert refreshed.status == STATUS_FILLED
        assert refreshed.avg_fill_price == pytest.approx(expected)

    def test_fill_exceeds_quantity_raises(self, oms):
        o = oms.submit_order(make_order("ex1", quantity=100))
        oms.process_fill(o.order_id, 60.0, 150.0)
        with pytest.raises(ValueError):
            oms.process_fill(o.order_id, 50.0, 150.0)  # 累计 110 > 100

    def test_fill_on_terminal_order_raises(self, oms):
        o = oms.submit_order(make_order("termf1", quantity=100))
        oms.process_fill(o.order_id, 100.0, 150.0)  # -> FILLED
        with pytest.raises(InvalidOrderStateError):
            oms.process_fill(o.order_id, 10.0, 150.0)

    def test_fill_ids_increment(self, oms):
        o = oms.submit_order(make_order("seq1", quantity=100))
        f1 = oms.process_fill(o.order_id, 30.0, 150.0)
        f2 = oms.process_fill(o.order_id, 30.0, 151.0)
        assert f1.fill_id == "seq1-1"
        assert f2.fill_id == "seq1-2"


# ---------------------------------------------------------------------------
# 超时自动撤单
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_check_timeouts_cancels_expired(self, oms):
        from datetime import timedelta
        o = oms.submit_order(make_order("t1", quantity=100, timeout_seconds=60))
        # 注入一个远早于现在的时间戳
        expired_now = o.submitted_at + timedelta(seconds=120)
        cancelled = oms.check_timeouts(now=expired_now)
        assert cancelled == ["t1"]
        assert oms.get_order("t1").status == STATUS_CANCELLED

    def test_check_timeouts_keeps_fresh(self, oms):
        from datetime import timedelta
        o = oms.submit_order(make_order("t2", quantity=100, timeout_seconds=600))
        fresh_now = o.submitted_at + timedelta(seconds=10)
        cancelled = oms.check_timeouts(now=fresh_now)
        assert cancelled == []
        assert oms.get_order("t2").status == STATUS_SUBMITTED


# ---------------------------------------------------------------------------
# 订单簿分类
# ---------------------------------------------------------------------------


class TestOrderBooks:
    def test_book_classification(self, oms):
        p = make_order("pend1")           # PENDING，先登记再撤
        oms._orders[p.order_id] = p
        oms._rebucket(p)

        a = oms.submit_order(make_order("act1"))     # SUBMITTED
        h = oms.submit_order(make_order("hist1"))
        oms.process_fill(h.order_id, h.quantity, 150.0)  # -> FILLED

        assert [o.order_id for o in oms.get_pending_orders()] == ["pend1"]
        active_ids = [o.order_id for o in oms.get_active_orders()]
        assert "act1" in active_ids
        hist = oms.get_historical_orders()
        assert h.order_id in [o.order_id for o in hist]
        assert any(o.status == STATUS_FILLED for o in hist)

    def test_get_active_by_symbol(self, oms):
        oms.submit_order(make_order("sym1", symbol="AAPL"))
        oms.submit_order(make_order("sym2", symbol="TSLA"))
        aapl = oms.get_active_orders(symbol="AAPL")
        assert len(aapl) == 1 and aapl[0].symbol == "AAPL"


# ---------------------------------------------------------------------------
# 成交回调
# ---------------------------------------------------------------------------


class TestCallbacks:
    def test_callback_fired(self, oms):
        received = []
        oms.register_fill_callback(lambda f: received.append(f))
        o = oms.submit_order(make_order("cb1", quantity=100))
        oms.process_fill(o.order_id, 50.0, 150.0)
        assert len(received) == 1
        assert isinstance(received[0], Fill)
        assert received[0].order_id == "cb1"

    def test_callback_exception_does_not_break(self, oms):
        def bad(_f):
            raise RuntimeError("boom")
        oms.register_fill_callback(bad)
        o = oms.submit_order(make_order("cb2", quantity=100))
        # 不抛异常即通过
        fill = oms.process_fill(o.order_id, 50.0, 150.0)
        assert fill.order_id == "cb2"


# ---------------------------------------------------------------------------
# 审计
# ---------------------------------------------------------------------------


class TestAudit:
    def test_audit_called_on_submit(self, oms, monkeypatch):
        mock_logger = MagicMock()
        monkeypatch.setattr("trading.oms.get_audit_logger", lambda: mock_logger)
        oms.submit_order(make_order("au1"))
        actions = [c.kwargs.get("action_type") for c in mock_logger.log.call_args_list]
        assert "ORDER_SUBMIT" in actions

    def test_audit_called_on_fill_and_cancel(self, oms, monkeypatch):
        mock_logger = MagicMock()
        monkeypatch.setattr("trading.oms.get_audit_logger", lambda: mock_logger)
        oms.submit_order(make_order("au2", quantity=100))
        mock_logger.reset_mock()
        oms.process_fill("au2", 50.0, 150.0)
        oms.cancel_order("au2", reason="x")
        actions = [c.kwargs.get("action_type") for c in mock_logger.log.call_args_list]
        assert "ORDER_FILL" in actions
        assert "ORDER_CANCEL" in actions


# ---------------------------------------------------------------------------
# 持久化
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_reload_reads_back(self, tmp_path):
        db = tmp_path / "reload.db"
        oms1 = OrderManagementSystem(db_path=str(db))
        o = oms1.submit_order(make_order("rl1", quantity=100))
        oms1.process_fill("rl1", 40.0, 150.0)

        # 重新实例化，从 SQLite 读回
        oms2 = OrderManagementSystem(db_path=str(db))
        restored = oms2.get_order("rl1")
        assert restored is not None
        assert restored.symbol == "AAPL"
        assert restored.filled_quantity == 40.0
        assert restored.status == STATUS_PARTIAL_FILLED
        fills = oms2.get_fills("rl1")
        assert len(fills) == 1
        assert fills[0].price == 150.0
        assert fills[0].quantity == 40.0


# ---------------------------------------------------------------------------
# API 集成测试
# ---------------------------------------------------------------------------


def _ok(data=None, message="success"):
    """与 server.py 一致的成功封装。"""
    return {"code": 0, "message": message, "data": data}


def _err(code, message, http_status=400):
    """与 server.py 一致的错误封装。"""
    return JSONResponse(
        status_code=http_status,
        content={"code": code, "message": message, "data": None},
    )


@pytest.fixture
def client(tmp_path):
    """构造注入临时库的 TestClient。"""
    db = tmp_path / "api_oms.db"
    oms = OrderManagementSystem(db_path=str(db))
    app = FastAPI()
    register_oms_routes(app, _ok, _err, oms=oms)
    return TestClient(app), oms


class TestApi:
    def test_submit_order(self, client):
        c, _ = client
        r = c.post("/api/oms/orders", json={
            "symbol": "AAPL", "side": "buy", "order_type": "limit",
            "quantity": 100, "limit_price": 150.0,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["status"] == STATUS_SUBMITTED
        assert body["data"]["order_id"]  # 自动生成

    def test_submit_invalid_side_400(self, client):
        c, _ = client
        r = c.post("/api/oms/orders", json={
            "symbol": "AAPL", "side": "hold", "order_type": "limit",
            "quantity": 100,
        })
        assert r.status_code == 400
        assert r.json()["code"] == 400

    def test_get_order_detail(self, client):
        c, oms = client
        o = oms.submit_order(make_order("api1"))
        r = c.get(f"/api/oms/orders/{o.order_id}")
        assert r.status_code == 200
        assert r.json()["data"]["order_id"] == "api1"

    def test_get_order_not_found_404(self, client):
        c, _ = client
        r = c.get("/api/oms/orders/nope")
        assert r.status_code == 404

    def test_list_orders_filter(self, client):
        c, oms = client
        oms.submit_order(make_order("list1", symbol="AAPL"))
        oms.submit_order(make_order("list2", symbol="TSLA"))
        r = c.get("/api/oms/orders", params={"symbol": "AAPL"})
        data = r.json()["data"]
        assert len(data) == 1 and data[0]["symbol"] == "AAPL"

    def test_cancel_order(self, client):
        c, oms = client
        o = oms.submit_order(make_order("can1"))
        r = c.delete(f"/api/oms/orders/{o.order_id}", params={"reason": "api"})
        assert r.status_code == 200
        assert r.json()["data"]["status"] == STATUS_CANCELLED

    def test_cancel_not_found_404(self, client):
        c, _ = client
        r = c.delete("/api/oms/orders/nope")
        assert r.status_code == 404

    def test_reject_order(self, client):
        c, oms = client
        o = oms.submit_order(make_order("rej1"))
        r = c.post(f"/api/oms/orders/{o.order_id}/reject", json={"reason": "废"})
        assert r.status_code == 200
        assert r.json()["data"]["status"] == STATUS_REJECTED

    def test_fills_endpoint(self, client):
        c, oms = client
        o = oms.submit_order(make_order("fill1", quantity=100))
        oms.process_fill(o.order_id, 50.0, 150.0)
        r = c.get("/api/oms/fills", params={"order_id": o.order_id})
        data = r.json()["data"]
        assert len(data) == 1
        assert data[0]["quantity"] == 50.0
