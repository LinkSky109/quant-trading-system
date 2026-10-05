"""结构化日志与请求链路 ID 模块测试。

覆盖：
  - StructuredJSONFormatter 输出合法 JSON 且包含全部字段
  - extra 字段 / request_id 进入 JSON
  - RequestIDMiddleware 生成唯一 ID、回写响应头、沿用客户端 ID
  - contextvars 在异步请求中正确传递（端点内 get_request_id 可见）
  - setup_structured_logging 创建 4 个日志文件与 handler
  - get_recent_logs 读取与按级别过滤
  - set_log_level 动态修改根 logger 级别
  - get_log_files 返回正确元数据
  - request_id 实际写入结构化日志文件
  - API 端点集成测试（/api/logs/recent, /api/logs/level, /api/logs/files）
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from monitoring import structured_logging as sl
from monitoring.structured_logging import (
    RequestIDMiddleware,
    StructuredJSONFormatter,
    get_log_files,
    get_recent_logs,
    get_request_id,
    set_log_level,
    setup_structured_logging,
)


class _LevelReqBody(BaseModel):
    """/api/logs/level 请求体（模块级定义，便于 PEP 563 注解解析）。"""

    level: str


# ---------------------------------------------------------------------------
# 夹具：隔离全局 logging 状态
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_logging(tmp_path: Path):
    """保存/恢复 root logger 状态与模块级日志文件注册表。"""
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    old_active = dict(sl._ACTIVE_LOG_FILES)
    # 清空，避免 handler 累积
    sl._ACTIVE_LOG_FILES.clear()
    yield tmp_path
    # 还原
    root.handlers = old_handlers
    root.setLevel(old_level)
    sl._ACTIVE_LOG_FILES.clear()
    sl._ACTIVE_LOG_FILES.update(old_active)


# ---------------------------------------------------------------------------
# StructuredJSONFormatter
# ---------------------------------------------------------------------------

class TestJSONFormatter:
    def _make_record(self, msg="hello", **extra) -> logging.LogRecord:
        logger = logging.getLogger("test_mod")
        return logger.makeRecord(
            "test_mod", logging.INFO, __file__, 42, msg, (),
            None, "test_func", extra,
        )

    def test_outputs_valid_json_with_required_fields(self):
        rec = self._make_record("hello world")
        out = StructuredJSONFormatter().format(rec)
        data = json.loads(out)  # 合法 JSON
        for field in ("timestamp", "level", "module", "function",
                      "line", "message", "request_id"):
            assert field in data, f"缺少字段 {field}"
        assert data["message"] == "hello world"
        assert data["level"] == "INFO"
        assert data["function"] == "test_func"
        assert data["line"] == 42
        assert isinstance(data["timestamp"], str)

    def test_extra_fields_included(self):
        rec = self._make_record("trade", symbol="600519.SH", price=1800.5)
        data = json.loads(StructuredJSONFormatter().format(rec))
        assert data["symbol"] == "600519.SH"
        assert data["price"] == 1800.5

    def test_request_id_from_contextvar(self):
        token = sl._request_id_ctx.set("abc12345")
        try:
            rec = self._make_record("with rid")
            data = json.loads(StructuredJSONFormatter().format(rec))
            assert data["request_id"] == "abc12345"
        finally:
            sl._request_id_ctx.reset(token)

    def test_request_id_empty_outside_request(self):
        rec = self._make_record("no rid")
        data = json.loads(StructuredJSONFormatter().format(rec))
        assert data["request_id"] == ""


# ---------------------------------------------------------------------------
# RequestIDMiddleware
# ---------------------------------------------------------------------------

def build_reqid_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/api/ping")
    async def ping():
        # 端点内应能从 contextvars 拿到 request_id
        return {"request_id": get_request_id()}

    @app.get("/api/log-and-return")
    async def log_and_return():
        logging.getLogger("test_biz").info("handling request")
        return {"request_id": get_request_id()}

    return app


class TestRequestIDMiddleware:
    def test_response_header_and_unique_ids(self):
        client = TestClient(build_reqid_app())
        r1 = client.get("/api/ping")
        r2 = client.get("/api/ping")
        rid1 = r1.headers.get("x-request-id")
        rid2 = r2.headers.get("x-request-id")
        assert rid1, "响应应带 X-Request-ID 头"
        assert rid2, "响应应带 X-Request-ID 头"
        assert rid1 != rid2  # 每次请求唯一
        # 端点内 contextvars 拿到的 id 与响应头一致
        assert r1.json()["request_id"] == rid1

    def test_client_supplied_id_is_honored(self):
        client = TestClient(build_reqid_app())
        r = client.get("/api/ping", headers={"X-Request-ID": "my-custom-id"})
        assert r.headers["x-request-id"] == "my-custom-id"
        assert r.json()["request_id"] == "my-custom-id"

    def test_contextvar_propagates_to_endpoint(self):
        client = TestClient(build_reqid_app())
        r = client.get("/api/ping")
        rid = r.headers["x-request-id"]
        # 端点内 get_request_id() 与中间件生成的一致 => 上下文正确传递
        assert r.json()["request_id"] == rid


# ---------------------------------------------------------------------------
# setup_structured_logging / get_recent_logs / set_log_level / get_log_files
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_creates_four_files(self, isolated_logging: Path):
        paths = setup_structured_logging({
            "level": "INFO",
            "log_dir": str(isolated_logging),
        })
        for name in ("structured", "trades", "alerts", "jev"):
            assert name in paths
            assert Path(paths[name]).exists(), f"{name} 日志文件未创建"

    def test_writes_and_reads_back(self, isolated_logging: Path):
        setup_structured_logging({
            "level": "INFO", "log_dir": str(isolated_logging),
        })
        logging.getLogger("test_biz").info("hello-from-test")
        logging.getLogger("test_biz").error("boom-from-test")

        logs = get_recent_logs()
        messages = [l["message"] for l in logs]
        assert "hello-from-test" in messages
        assert "boom-from-test" in messages

    def test_level_filter(self, isolated_logging: Path):
        setup_structured_logging({
            "level": "DEBUG", "log_dir": str(isolated_logging),
        })
        logging.getLogger("t").info("info-msg")
        logging.getLogger("t").error("error-msg")

        errs = get_recent_logs(level="ERROR")
        assert all(l["level"] == "ERROR" for l in errs)
        msgs = [l["message"] for l in errs]
        assert "error-msg" in msgs
        assert "info-msg" not in msgs

    def test_limit(self, isolated_logging: Path):
        setup_structured_logging({
            "level": "INFO", "log_dir": str(isolated_logging),
        })
        for i in range(10):
            logging.getLogger("t").info(f"msg-{i}")
        logs = get_recent_logs(limit=3)
        assert len(logs) == 3
        assert logs[-1]["message"] == "msg-9"

    def test_set_log_level(self, isolated_logging: Path):
        setup_structured_logging({
            "level": "INFO", "log_dir": str(isolated_logging),
        })
        new_level = set_log_level("DEBUG")
        assert new_level == "DEBUG"
        assert logging.getLogger().level == logging.DEBUG

        with pytest.raises(ValueError):
            set_log_level("NOT_A_LEVEL")

    def test_get_log_files_metadata(self, isolated_logging: Path):
        setup_structured_logging({
            "level": "INFO", "log_dir": str(isolated_logging),
        })
        logging.getLogger("t").warning("write something")
        files = get_log_files()
        names = {f["name"] for f in files}
        assert names == {"structured", "trades", "alerts", "jev"}
        structured = next(f for f in files if f["name"] == "structured")
        assert structured["size_bytes"] > 0
        assert structured["modified"]  # 非空时间戳
        assert Path(structured["path"]).exists()


class TestRequestIdInLogs:
    def test_request_id_written_to_jsonl(self, isolated_logging: Path):
        setup_structured_logging({
            "level": "INFO", "log_dir": str(isolated_logging),
        })
        client = TestClient(build_reqid_app())
        r = client.get("/api/log-and-return")
        rid = r.headers["x-request-id"]

        logs = get_recent_logs()
        biz = [l for l in logs if l["message"] == "handling request"]
        assert biz, "应能读到业务日志"
        assert biz[-1]["request_id"] == rid, "日志中的 request_id 应等于请求 ID"


# ---------------------------------------------------------------------------
# API 端点集成测试
# ---------------------------------------------------------------------------

def build_logs_app() -> FastAPI:
    app = FastAPI()

    def ok(data=None, message="success"):
        return {"code": 0, "message": message, "data": data}

    def err(code, message, http_status=400):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=http_status,
                            content={"code": code, "message": message, "data": None})

    @app.get("/api/logs/recent")
    async def recent(level: str | None = None, limit: int = 100):
        logs = get_recent_logs(level=level, limit=limit)
        return ok({"logs": logs, "total": len(logs)})

    @app.post("/api/logs/level")
    async def set_level(req: _LevelReqBody):
        try:
            new = set_log_level(req.level)
            return ok({"level": new, "changed": True})
        except ValueError as e:
            return err(40010, str(e), http_status=400)

    @app.get("/api/logs/files")
    async def files():
        return ok({"files": get_log_files()})

    return app


class TestLogEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self, isolated_logging: Path):
        setup_structured_logging({
            "level": "INFO", "log_dir": str(isolated_logging),
        })
        self.client = TestClient(build_logs_app())

    def test_recent_endpoint(self):
        logging.getLogger("t").info("ep-log-1")
        logging.getLogger("t").error("ep-log-2")
        r = self.client.get("/api/logs/recent")
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["total"] == len(body["data"]["logs"])
        msgs = [l["message"] for l in body["data"]["logs"]]
        assert "ep-log-1" in msgs and "ep-log-2" in msgs

    def test_recent_with_level_filter(self):
        logging.getLogger("t").info("keep-info")
        logging.getLogger("t").error("keep-error")
        r = self.client.get("/api/logs/recent?level=ERROR")
        logs = r.json()["data"]["logs"]
        assert all(l["level"] == "ERROR" for l in logs)
        assert "keep-error" in [l["message"] for l in logs]

    def test_set_level_endpoint(self):
        r = self.client.post("/api/logs/level", json={"level": "DEBUG"})
        assert r.status_code == 200
        assert r.json()["data"] == {"level": "DEBUG", "changed": True}

    def test_set_level_bad_returns_400(self):
        r = self.client.post("/api/logs/level", json={"level": "BOGUS"})
        assert r.status_code == 400
        assert r.json()["code"] == 40010

    def test_files_endpoint(self):
        r = self.client.get("/api/logs/files")
        assert r.status_code == 200
        names = {f["name"] for f in r.json()["data"]["files"]}
        assert names == {"structured", "trades", "alerts", "jev"}
