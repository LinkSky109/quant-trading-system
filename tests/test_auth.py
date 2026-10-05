"""AuthManager 与 API Token 认证中间件单元测试。

不依赖真实 config.yaml，全部使用构造的 token 配置。
中间件集成测试使用最小 FastAPI App + TestClient，不启动真实 server。
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from security.auth import AuthManager


# ---------------------------------------------------------------------------
# 构造测试配置
# ---------------------------------------------------------------------------

def _make_config(enabled: bool = True) -> dict:
    """构造一组测试用 token 配置。"""
    return {
        "enabled": enabled,
        "tokens": [
            {"name": "admin用户", "token": "admin-token-123",
             "permission": "admin", "expires_at": ""},
            {"name": "交易员", "token": "trade-token-456",
             "permission": "trade", "expires_at": ""},
            {"name": "只读用户", "token": "read-token-789",
             "permission": "read", "expires_at": ""},
            {"name": "已过期用户", "token": "expired-token-000",
             "permission": "admin", "expires_at": "2020-01-01"},
            {"name": "未来到期用户", "token": "future-token-111",
             "permission": "read", "expires_at": "2099-12-31"},
        ],
    }


# ---------------------------------------------------------------------------
# AuthManager 初始化
# ---------------------------------------------------------------------------

class TestAuthManagerInit:
    def test_disabled_by_default_when_config_none(self):
        mgr = AuthManager(None)
        assert mgr.is_enabled() is False
        assert mgr.has_tokens() is False

    def test_enabled_flag(self):
        mgr = AuthManager(_make_config(enabled=True))
        assert mgr.is_enabled() is True
        assert mgr.has_tokens() is True

    def test_disabled_flag(self):
        mgr = AuthManager(_make_config(enabled=False))
        assert mgr.is_enabled() is False

    def test_empty_tokens_list(self):
        mgr = AuthManager({"enabled": True, "tokens": []})
        assert mgr.is_enabled() is True
        assert mgr.has_tokens() is False

    def test_empty_token_entry_skipped(self):
        mgr = AuthManager({
            "enabled": True,
            "tokens": [{"name": "x", "token": "  ", "permission": "read"}],
        })
        assert mgr.has_tokens() is False

    def test_unknown_permission_falls_back_to_read(self):
        mgr = AuthManager({
            "enabled": True,
            "tokens": [{"name": "x", "token": "bad-perm",
                        "permission": "superuser", "expires_at": ""}],
        })
        info = mgr.verify_token("bad-perm")
        assert info is not None
        assert info["permission"] == "read"


# ---------------------------------------------------------------------------
# verify_token
# ---------------------------------------------------------------------------

class TestVerifyToken:
    def test_valid_token_returns_info(self):
        mgr = AuthManager(_make_config())
        info = mgr.verify_token("read-token-789")
        assert info is not None
        assert info["name"] == "只读用户"
        assert info["permission"] == "read"
        assert info["expires_at"] == ""

    def test_invalid_token_returns_none(self):
        mgr = AuthManager(_make_config())
        assert mgr.verify_token("not-exist-token") is None

    def test_empty_token_returns_none(self):
        mgr = AuthManager(_make_config())
        assert mgr.verify_token("") is None
        assert mgr.verify_token(None) is None

    def test_expired_token_returns_none(self):
        mgr = AuthManager(_make_config())
        # 虽然是 admin 权限，但 expires_at 为过去日期
        assert mgr.verify_token("expired-token-000") is None

    def test_future_expiry_token_valid(self):
        mgr = AuthManager(_make_config())
        info = mgr.verify_token("future-token-111")
        assert info is not None
        assert info["permission"] == "read"

    def test_verify_result_is_copy(self):
        """verify_token 返回副本，外部修改不影响内部状态。"""
        mgr = AuthManager(_make_config())
        info = mgr.verify_token("admin-token-123")
        info["permission"] = "read"
        info2 = mgr.verify_token("admin-token-123")
        assert info2["permission"] == "admin"

    def test_verify_works_when_disabled(self):
        """enabled=false 时 AuthManager 本身仍可校验 token（只有中间件依赖 enabled）。"""
        mgr = AuthManager(_make_config(enabled=False))
        assert mgr.is_enabled() is False
        assert mgr.verify_token("admin-token-123") is not None
        assert mgr.verify_token("bad-token") is None


# ---------------------------------------------------------------------------
# 权限层级
# ---------------------------------------------------------------------------

class TestHasPermission:
    @pytest.fixture()
    def mgr(self):
        return AuthManager(_make_config())

    def test_admin_can_access_all(self, mgr):
        admin = mgr.verify_token("admin-token-123")
        assert mgr.has_permission(admin, "read") is True
        assert mgr.has_permission(admin, "trade") is True
        assert mgr.has_permission(admin, "admin") is True

    def test_trade_can_access_read_and_trade(self, mgr):
        trade = mgr.verify_token("trade-token-456")
        assert mgr.has_permission(trade, "read") is True
        assert mgr.has_permission(trade, "trade") is True
        assert mgr.has_permission(trade, "admin") is False

    def test_read_only_read(self, mgr):
        read = mgr.verify_token("read-token-789")
        assert mgr.has_permission(read, "read") is True
        assert mgr.has_permission(read, "trade") is False
        assert mgr.has_permission(read, "admin") is False

    def test_none_token_info_denied(self, mgr):
        assert mgr.has_permission(None, "read") is False


# ---------------------------------------------------------------------------
# generate_token
# ---------------------------------------------------------------------------

class TestGenerateToken:
    def test_generate_token_non_empty_hex(self):
        t1 = AuthManager.generate_token()
        t2 = AuthManager.generate_token()
        assert isinstance(t1, str)
        assert len(t1) == 64  # 32 bytes -> 64 hex chars
        assert t1 != t2  # 随机不重复
        int(t1, 16)  # 是合法 hex


# ---------------------------------------------------------------------------
# 中间件集成测试（最小 FastAPI App，镜像 server.py 的 auth_middleware 契约）
# ---------------------------------------------------------------------------

# 与 server.py 保持一致的路径权限映射
_PUBLIC_EXACT_PATHS = {"/", "/docs", "/openapi.json", "/redoc",
                       "/api/health", "/api/auth/verify"}
_ADMIN_PATH_PREFIXES = ("/api/alerts/trigger", "/api/daily_reports/export")
_TRADE_PATH_PREFIXES = ("/api/trade_toggle",)


def _required_permission(path: str):
    if path in _PUBLIC_EXACT_PATHS:
        return None
    if path.startswith(_ADMIN_PATH_PREFIXES):
        return "admin"
    if path.startswith(_TRADE_PATH_PREFIXES):
        return "trade"
    if path.startswith("/api/") or path.startswith("/training_data/"):
        return "read"
    return None


def build_test_app(auth_manager: AuthManager) -> FastAPI:
    """构建最小 FastAPI 应用，中间件逻辑与 web-dashboard/server.py 一致。"""
    app = FastAPI()

    @app.middleware("http")
    async def _auth_middleware(request, call_next):
        if not auth_manager.is_enabled():
            return await call_next(request)
        required = _required_permission(request.url.path)
        if required is None:
            return await call_next(request)
        auth = request.headers.get("authorization")
        token = None
        if auth and " " in auth:
            scheme, value = auth.split(" ", 1)
            if scheme.lower() == "bearer":
                token = value.strip() or None
        if not token:
            return JSONResponse(
                status_code=401,
                content={"code": 401, "message": "未提供认证token", "data": None},
            )
        info = auth_manager.verify_token(token)
        if info is None:
            return JSONResponse(
                status_code=401,
                content={"code": 401, "message": "token无效或已过期", "data": None},
            )
        if not auth_manager.has_permission(info, required):
            return JSONResponse(
                status_code=403,
                content={"code": 403, "message": "权限不足", "data": None},
            )
        return await call_next(request)

    @app.get("/api/health")
    async def health():
        return {"code": 0, "message": "success", "data": {"status": "ok"}}

    @app.get("/api/symbols")
    async def symbols():
        return {"code": 0, "message": "success", "data": []}

    @app.post("/api/trade_toggle")
    async def trade_toggle():
        return {"code": 0, "message": "success", "data": {"on": True}}

    @app.post("/api/alerts/trigger")
    async def trigger_alert():
        return {"code": 0, "message": "success", "data": {}}

    @app.post("/api/auth/verify")
    async def auth_verify():
        return {"code": 0, "message": "success", "data": {"valid": True}}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        # 与 server.py 的 WebSocket 认证逻辑一致
        if auth_manager.is_enabled():
            ws_token = websocket.query_params.get("token", "")
            info = auth_manager.verify_token(ws_token)
            if info is None or not auth_manager.has_permission(info, "read"):
                await websocket.close(code=4401)
                return
        await websocket.accept()
        await websocket.send_json({"type": "connected"})
        await websocket.close()

    return app


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class TestAuthMiddlewareEnabled:
    @pytest.fixture()
    def client(self):
        mgr = AuthManager(_make_config(enabled=True))
        return TestClient(build_test_app(mgr))

    def test_public_health_no_token_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_public_auth_verify_no_token_200(self, client):
        resp = client.post("/api/auth/verify")
        assert resp.status_code == 200

    def test_protected_no_token_401(self, client):
        resp = client.get("/api/symbols")
        assert resp.status_code == 401
        body = resp.json()
        assert body["code"] == 401
        assert body["data"] is None

    def test_invalid_token_401(self, client):
        resp = client.get("/api/symbols", headers=auth_header("wrong-token"))
        assert resp.status_code == 401
        assert resp.json()["code"] == 401

    def test_expired_token_401(self, client):
        resp = client.get("/api/symbols", headers=auth_header("expired-token-000"))
        assert resp.status_code == 401

    def test_read_token_access_read_api_200(self, client):
        resp = client.get("/api/symbols", headers=auth_header("read-token-789"))
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    def test_read_token_cannot_trade_403(self, client):
        resp = client.post("/api/trade_toggle", headers=auth_header("read-token-789"))
        assert resp.status_code == 403
        assert resp.json()["code"] == 403
        assert resp.json()["message"] == "权限不足"

    def test_trade_token_can_trade_200(self, client):
        resp = client.post("/api/trade_toggle", headers=auth_header("trade-token-456"))
        assert resp.status_code == 200

    def test_read_token_cannot_admin_403(self, client):
        resp = client.post("/api/alerts/trigger", headers=auth_header("read-token-789"))
        assert resp.status_code == 403

    def test_trade_token_cannot_admin_403(self, client):
        resp = client.post("/api/alerts/trigger", headers=auth_header("trade-token-456"))
        assert resp.status_code == 403

    def test_admin_token_admin_api_200(self, client):
        resp = client.post("/api/alerts/trigger", headers=auth_header("admin-token-123"))
        assert resp.status_code == 200

    def test_malformed_authorization_header_401(self, client):
        # 非 Bearer  scheme
        resp = client.get("/api/symbols", headers={"Authorization": "Basic abc"})
        assert resp.status_code == 401


class TestAuthMiddlewareDisabled:
    def test_all_open_when_disabled(self):
        mgr = AuthManager(_make_config(enabled=False))
        client = TestClient(build_test_app(mgr))
        # 无 token 也能访问受保护接口
        assert client.get("/api/symbols").status_code == 200
        assert client.post("/api/trade_toggle").status_code == 200
        assert client.post("/api/alerts/trigger").status_code == 200


class TestWebSocketAuth:
    def test_ws_rejected_without_token_when_enabled(self):
        from starlette.websockets import WebSocketDisconnect
        mgr = AuthManager(_make_config(enabled=True))
        client = TestClient(build_test_app(mgr))
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws"):
                pass
        assert exc_info.value.code == 4401

    def test_ws_rejected_with_invalid_token(self):
        from starlette.websockets import WebSocketDisconnect
        mgr = AuthManager(_make_config(enabled=True))
        client = TestClient(build_test_app(mgr))
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws?token=bogus"):
                pass

    def test_ws_accepted_with_valid_token(self):
        mgr = AuthManager(_make_config(enabled=True))
        client = TestClient(build_test_app(mgr))
        with client.websocket_connect("/ws?token=read-token-789") as ws:
            data = ws.receive_json()
            assert data["type"] == "connected"

    def test_ws_open_when_disabled(self):
        mgr = AuthManager(_make_config(enabled=False))
        client = TestClient(build_test_app(mgr))
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "connected"
