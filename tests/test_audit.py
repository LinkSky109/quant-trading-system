"""安全审计日志模块单元测试与 API 集成测试。

覆盖：
  - AuditLogger 初始化（enabled / disabled）
  - 日志写入（SQLite + JSONL 双写验证）
  - 哈希链正确性（连续写入后 prev_hash 链接）
  - 完整性校验（正常通过 / 篡改一条后检测到）
  - 分页查询
  - 操作类型枚举
  - 线程安全（多线程并发写入不丢数据、哈希链不断）
  - API 端点集成测试（TestClient）
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel

from persistence.database import Database
from security.audit import (
    GENESIS_HASH,
    ActionType,
    AuditLogger,
    _compute_hash,
)


# ---------------------------------------------------------------------------
# 构造测试夹具
# ---------------------------------------------------------------------------

@pytest.fixture()
def audit(tmp_path):
    """创建一个启用审计日志的临时 AuditLogger（独立 DB + JSONL）。"""
    db_path = str(tmp_path / "audit_test.db")
    log_file = str(tmp_path / "logs" / "audit.jsonl")
    cfg = {"enabled": True, "log_file": log_file, "retention_days": 90}
    logger = AuditLogger(cfg, db_path=db_path)
    return logger


@pytest.fixture()
def disabled_audit(tmp_path):
    """创建一个 disabled 的 AuditLogger。"""
    db_path = str(tmp_path / "audit_disabled.db")
    cfg = {"enabled": False, "log_file": str(tmp_path / "x.jsonl")}
    return AuditLogger(cfg, db_path=db_path)


# ---------------------------------------------------------------------------
# ActionType 枚举
# ---------------------------------------------------------------------------

class TestActionTypeEnum:
    def test_all_expected_members(self):
        expected = {
            "LOGIN", "LOGOUT", "ORDER_SUBMIT", "ORDER_CANCEL",
            "POSITION_CLOSE", "CONFIG_CHANGE", "STRATEGY_TOGGLE",
            "RISK_TRIGGER", "BACKUP_RESTORE", "API_TOKEN_CREATE",
        }
        actual = {a.value for a in ActionType}
        assert actual == expected

    def test_enum_is_str(self):
        """枚举值可直接当字符串用于 SQLite / JSON。"""
        assert ActionType.LOGIN.value == "LOGIN"
        assert isinstance(ActionType.LOGIN, str)

    def test_genesis_hash_length(self):
        assert len(GENESIS_HASH) == 64
        assert GENESIS_HASH == "0" * 64


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

class TestAuditLoggerInit:
    def test_disabled_by_default(self, tmp_path):
        al = AuditLogger(None, db_path=str(tmp_path / "x.db"))
        assert al.is_enabled() is False

    def test_enabled_flag(self, audit):
        assert audit.is_enabled() is True

    def test_disabled_flag(self, disabled_audit):
        assert disabled_audit.is_enabled() is False

    def test_initial_last_hash_is_genesis(self, audit):
        """空库启动时链尾为创世哈希。"""
        assert audit.last_hash == GENESIS_HASH

    def test_resume_chain_from_db(self, tmp_path):
        """已有数据时重启 AuditLogger 应从 DB 恢复链尾。"""
        db_path = str(tmp_path / "resume.db")
        log_file = str(tmp_path / "a.jsonl")
        cfg = {"enabled": True, "log_file": log_file}

        al1 = AuditLogger(cfg, db_path=db_path)
        rec = al1.log(operator="admin", action_type=ActionType.LOGIN)
        assert rec is not None
        last = al1.last_hash
        assert last != GENESIS_HASH

        # 新建实例，应从 DB 恢复链尾
        al2 = AuditLogger(cfg, db_path=db_path)
        assert al2.last_hash == last

    def test_log_when_disabled_returns_none(self, disabled_audit):
        result = disabled_audit.log(operator="x", action_type=ActionType.LOGIN)
        assert result is None


# ---------------------------------------------------------------------------
# 写入：双写 SQLite + JSONL
# ---------------------------------------------------------------------------

class TestDualWrite:
    def test_sqlite_written(self, audit):
        rec = audit.log(operator="admin", action_type=ActionType.LOGIN,
                        ip="127.0.0.1")
        assert rec is not None
        rows = audit._db.query_audit_logs()
        assert len(rows) == 1
        assert rows[0]["operator"] == "admin"
        assert rows[0]["action_type"] == "LOGIN"

    def test_jsonl_written(self, audit):
        rec = audit.log(operator="admin", action_type=ActionType.LOGIN,
                        ip="127.0.0.1")
        log_path = Path(audit._log_file)
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["operator"] == "admin"
        assert entry["hash"] == rec["hash"]

    def test_append_mode(self, audit):
        audit.log(operator="a", action_type=ActionType.LOGIN)
        audit.log(operator="b", action_type=ActionType.LOGOUT)
        audit.log(operator="c", action_type=ActionType.ORDER_SUBMIT)
        lines = Path(audit._log_file).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_record_fields(self, audit):
        rec = audit.log(
            operator="trader1",
            action_type=ActionType.ORDER_SUBMIT,
            target="600519.SH",
            params={"side": "buy", "qty": 100},
            result="success",
            ip="192.168.1.1",
            request_id="req-123",
        )
        assert rec["operator"] == "trader1"
        assert rec["action_type"] == "ORDER_SUBMIT"
        assert rec["target"] == "600519.SH"
        assert rec["ip"] == "192.168.1.1"
        assert rec["request_id"] == "req-123"
        # params 存储为 JSON 字符串
        db_row = audit._db.query_audit_logs()[0]
        params = json.loads(db_row["params"])
        assert params["side"] == "buy"


# ---------------------------------------------------------------------------
# 哈希链正确性
# ---------------------------------------------------------------------------

class TestHashChain:
    def test_first_prev_hash_is_genesis(self, audit):
        rec = audit.log(operator="admin", action_type=ActionType.LOGIN)
        assert rec["prev_hash"] == GENESIS_HASH

    def test_second_prev_hash_is_first_hash(self, audit):
        r1 = audit.log(operator="a", action_type=ActionType.LOGIN)
        r2 = audit.log(operator="b", action_type=ActionType.LOGOUT)
        assert r2["prev_hash"] == r1["hash"]

    def test_hash_length_64(self, audit):
        rec = audit.log(operator="a", action_type=ActionType.LOGIN)
        assert len(rec["hash"]) == 64

    def test_hash_deterministic(self):
        """相同输入应产生相同 hash。"""
        h1 = _compute_hash(
            timestamp="2026-01-01T10:00:00", operator="admin",
            action_type="LOGIN", target="", params_json="{}",
            result="success", ip="127.0.0.1", request_id="",
            prev_hash=GENESIS_HASH,
        )
        h2 = _compute_hash(
            timestamp="2026-01-01T10:00:00", operator="admin",
            action_type="LOGIN", target="", params_json="{}",
            result="success", ip="127.0.0.1", request_id="",
            prev_hash=GENESIS_HASH,
        )
        assert h1 == h2

    def test_hash_changes_with_field(self):
        """修改任一字段 hash 应不同。"""
        base = dict(
            timestamp="2026-01-01T10:00:00", operator="admin",
            action_type="LOGIN", target="", params_json="{}",
            result="success", ip="127.0.0.1", request_id="",
            prev_hash=GENESIS_HASH,
        )
        h_base = _compute_hash(**base)
        h_modified = _compute_hash(**{**base, "operator": "evil"})
        assert h_base != h_modified


# ---------------------------------------------------------------------------
# 完整性校验
# ---------------------------------------------------------------------------

class TestVerifyIntegrity:
    def test_empty_chain_valid(self, audit):
        report = audit.verify_integrity()
        assert report["valid"] is True
        assert report["total_records"] == 0
        assert report["first_hash"] == ""
        assert report["last_hash"] == GENESIS_HASH

    def test_normal_chain_passes(self, audit):
        for i in range(5):
            audit.log(operator=f"user{i}", action_type=ActionType.LOGIN)
        report = audit.verify_integrity()
        assert report["valid"] is True
        assert report["total_records"] == 5
        assert report["broken_at"] == 0
        assert report["error"] == ""
        assert len(report["first_hash"]) == 64
        assert len(report["last_hash"]) == 64

    def test_tamper_detects_modified_data(self, audit):
        audit.log(operator="alice", action_type=ActionType.LOGIN)
        audit.log(operator="bob", action_type=ActionType.LOGOUT)
        audit.log(operator="carol", action_type=ActionType.ORDER_SUBMIT)

        # 篡改中间一条记录的 operator（不更新 hash）
        conn = audit._db._get_conn()
        conn.execute("UPDATE audit_logs SET operator = 'hacker' WHERE id = 2")
        conn.commit()

        report = audit.verify_integrity()
        assert report["valid"] is False
        assert report["broken_at"] == 2
        assert "hash" in report["error"] or "篡改" in report["error"]

    def test_tamper_detects_broken_prev_hash(self, audit):
        audit.log(operator="a", action_type=ActionType.LOGIN)
        audit.log(operator="b", action_type=ActionType.LOGOUT)
        audit.log(operator="c", action_type=ActionType.ORDER_SUBMIT)

        # 把第二条的 prev_hash 改成错误值
        conn = audit._db._get_conn()
        conn.execute("UPDATE audit_logs SET prev_hash = ? WHERE id = 2",
                     ("f" * 64,))
        conn.commit()

        report = audit.verify_integrity()
        assert report["valid"] is False
        assert report["broken_at"] == 2

    def test_tamper_detects_middle_record_hash_change(self, audit):
        audit.log(operator="a", action_type=ActionType.LOGIN)
        audit.log(operator="b", action_type=ActionType.LOGOUT)
        audit.log(operator="c", action_type=ActionType.ORDER_SUBMIT)

        # 直接修改第二条的 hash 字段
        conn = audit._db._get_conn()
        conn.execute("UPDATE audit_logs SET hash = ? WHERE id = 2",
                     ("e" * 64,))
        conn.commit()

        report = audit.verify_integrity()
        assert report["valid"] is False
        assert report["broken_at"] == 2


# ---------------------------------------------------------------------------
# 分页查询
# ---------------------------------------------------------------------------

class TestQueryLogs:
    def test_pagination(self, audit):
        for i in range(25):
            audit.log(operator=f"op{i % 3}", action_type=ActionType.LOGIN)

        page1 = audit.query_logs(page=1, page_size=10)
        assert page1["total"] == 25
        assert page1["page"] == 1
        assert page1["page_size"] == 10
        assert len(page1["logs"]) == 10

        page3 = audit.query_logs(page=3, page_size=10)
        assert len(page3["logs"]) == 5

    def test_filter_by_action_type(self, audit):
        audit.log(operator="a", action_type=ActionType.LOGIN)
        audit.log(operator="b", action_type=ActionType.LOGOUT)
        audit.log(operator="c", action_type=ActionType.LOGIN)

        logins = audit.query_logs(action_type="LOGIN")
        assert logins["total"] == 2

    def test_filter_by_operator(self, audit):
        audit.log(operator="alice", action_type=ActionType.LOGIN)
        audit.log(operator="bob", action_type=ActionType.LOGIN)
        audit.log(operator="alice", action_type=ActionType.LOGOUT)

        alice = audit.query_logs(operator="alice")
        assert alice["total"] == 2

    def test_filter_by_time_range(self, audit):
        audit.log(operator="a", action_type=ActionType.LOGIN,
                  timestamp="2026-01-01T10:00:00")
        audit.log(operator="b", action_type=ActionType.LOGIN,
                  timestamp="2026-06-01T10:00:00")
        audit.log(operator="c", action_type=ActionType.LOGIN,
                  timestamp="2026-12-01T10:00:00")

        result = audit.query_logs(
            start="2026-03-01T00:00:00", end="2026-09-01T00:00:00")
        assert result["total"] == 1


# ---------------------------------------------------------------------------
# 线程安全
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_writes_no_loss(self, audit):
        """多线程并发写入：总条数正确，哈希链不断。"""
        n_threads = 8
        writes_per_thread = 25

        def worker(tid: int):
            for i in range(writes_per_thread):
                audit.log(
                    operator=f"t{tid}",
                    action_type=ActionType.ORDER_SUBMIT,
                    target=f"sym-{tid}-{i}",
                )

        threads = [threading.Thread(target=worker, args=(t,))
                   for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = n_threads * writes_per_thread
        assert audit._db.count_audit_logs() == expected

        # JSONL 行数也应一致
        lines = Path(audit._log_file).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == expected

        # 哈希链完整
        report = audit.verify_integrity()
        assert report["valid"] is True
        assert report["total_records"] == expected

    def test_chain_not_broken_under_concurrency(self, audit):
        """并发写入后链尾 hash 应等于 DB 中最后一条的 hash。"""
        def worker():
            for _ in range(10):
                audit.log(operator="x", action_type=ActionType.LOGIN)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        report = audit.verify_integrity()
        assert report["valid"] is True
        # 内存中的 last_hash 应与 DB 尾部一致
        assert audit.last_hash == report["last_hash"]


# ---------------------------------------------------------------------------
# API 端点集成测试（最小 FastAPI App，镜像 server.py 契约）
# ---------------------------------------------------------------------------

def _build_api_test_app(audit_logger: AuditLogger) -> FastAPI:
    """构建最小 FastAPI 应用，注册审计查询/校验端点。"""
    app = FastAPI()

    def ok(data=None, message="success"):
        return {"code": 0, "message": message, "data": data}

    def err(code, message, http_status=400):
        return JSONResponse(
            status_code=http_status,
            content={"code": code, "message": message, "data": None},
        )

    @app.get("/api/audit/logs")
    async def api_audit_logs(
        start: str | None = None,
        end: str | None = None,
        action_type: str | None = None,
        operator: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ):
        try:
            data = audit_logger.query_logs(
                start=start, end=end,
                action_type=action_type, operator=operator,
                page=page, page_size=page_size,
            )
            return ok(data)
        except Exception as e:
            return err(50010, f"查询审计日志失败: {e}", http_status=500)

    @app.get("/api/audit/verify")
    async def api_audit_verify():
        try:
            report = audit_logger.verify_integrity()
            return ok(report)
        except Exception as e:
            return err(50011, f"校验失败: {e}", http_status=500)

    return app


class TestAuditAPIEndpoints:
    @pytest.fixture()
    def client(self, audit):
        # 预写几条数据
        audit.log(operator="admin", action_type=ActionType.LOGIN, ip="127.0.0.1")
        audit.log(operator="trader", action_type=ActionType.ORDER_SUBMIT,
                  target="600519.SH")
        audit.log(operator="admin", action_type=ActionType.LOGOUT)
        app = _build_api_test_app(audit)
        return TestClient(app)

    def test_get_logs_returns_paginated(self, client):
        resp = client.get("/api/audit/logs?page=1&page_size=10")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        data = body["data"]
        assert data["total"] == 3
        assert data["page"] == 1
        assert data["page_size"] == 10
        assert len(data["logs"]) == 3

    def test_get_logs_filter_action_type(self, client):
        resp = client.get("/api/audit/logs?action_type=LOGIN")
        body = resp.json()
        assert body["data"]["total"] == 1

    def test_get_logs_filter_operator(self, client):
        resp = client.get("/api/audit/logs?operator=admin")
        body = resp.json()
        assert body["data"]["total"] == 2

    def test_get_logs_pagination(self, client, audit):
        for _ in range(20):
            audit.log(operator="x", action_type=ActionType.LOGIN)
        resp = client.get("/api/audit/logs?page=2&page_size=10")
        body = resp.json()
        assert body["data"]["total"] == 23
        assert len(body["data"]["logs"]) == 10

    def test_verify_returns_valid(self, client):
        resp = client.get("/api/audit/verify")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        data = body["data"]
        assert data["valid"] is True
        assert data["total_records"] == 3
        assert data["broken_at"] == 0
        assert len(data["first_hash"]) == 64
        assert len(data["last_hash"]) == 64

    def test_verify_detects_tamper(self, client, audit):
        # 篡改一条记录
        conn = audit._db._get_conn()
        conn.execute("UPDATE audit_logs SET operator = 'hacked' WHERE id = 2")
        conn.commit()

        resp = client.get("/api/audit/verify")
        body = resp.json()
        data = body["data"]
        assert data["valid"] is False
        assert data["broken_at"] == 2
