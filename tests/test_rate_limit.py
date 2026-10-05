"""API 限流模块（令牌桶）单元测试与中间件集成测试。

覆盖：
  - TokenBucket 令牌补充（时间推进后令牌恢复、桶满封顶）
  - TokenBucket 超限返回正确 retry_after
  - RateLimiter 分级限流（不同分级不同配额）
  - 白名单 IP 不限流
  - 按 Token 限流 vs 按 IP 限流
  - 429 响应格式（含 Retry-After 头）
  - 中间件集成测试（最小 FastAPI App + TestClient）
  - enabled=false 时全部放行
  - LRU 清理不影响活跃桶
  - get_status 返回正确状态
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from security.rate_limit import DEFAULT_TIERS, RateLimiter, TokenBucket


# ---------------------------------------------------------------------------
# 构造测试配置
# ---------------------------------------------------------------------------

def _cfg(**over) -> dict:
    """构造一组测试用限流配置（小配额，便于触发超限）。"""
    cfg = {
        "enabled": True,
        "tiers": {"read": 5, "trade": 4, "admin": 3, "backtest": 2},
        "whitelist": ["127.0.0.1", "localhost", "::1"],
        "default_tier": "read",
    }
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------

class TestTokenBucket:
    def test_starts_full_and_consumes(self):
        b = TokenBucket(capacity=10, now=1000.0)
        allowed, wait = b.consume(now=1000.0)
        assert allowed is True
        assert wait == 0.0
        assert b.tokens == pytest.approx(9.0)

    def test_refill_recovers_tokens_over_time(self):
        """时间推进后令牌应匀速恢复。"""
        b = TokenBucket(capacity=10, now=1000.0)  # refill_rate = 10/60 ≈ 0.1667/s
        for _ in range(10):
            allowed, _ = b.consume(now=1000.0)
            assert allowed is True
        # 空桶后再消费应被拒
        allowed, _ = b.consume(now=1000.0)
        assert allowed is False
        # 过 6 秒：补充 6 * 0.1667 = 1.0 个令牌，应能再消费 1 次
        allowed, _ = b.consume(now=1006.0)
        assert allowed is True

    def test_refill_capped_at_capacity(self):
        """长时间不消费，令牌补满到容量即止，不超发。"""
        b = TokenBucket(capacity=5, now=1000.0)
        b.consume(now=1000.0)  # 剩 4
        # 直接推进时间补充令牌（不再消费），应补满到容量 5，而非无限堆积
        b._refill(1000.0 + 3600.0)
        assert b.tokens == pytest.approx(5.0)

    def test_retry_after_formula(self):
        """空桶时 retry_after = 缺口 / refill_rate。"""
        # capacity=60 -> refill_rate = 1.0/s
        b = TokenBucket(capacity=60, now=1000.0)
        for _ in range(60):
            b.consume(now=1000.0)
        allowed, wait = b.consume(now=1000.0)
        assert allowed is False
        assert wait == pytest.approx(1.0, abs=0.01)

    def test_retry_after_partial_bucket(self):
        """剩 0.5 个令牌时，再要 1 个需补 0.5 个。"""
        b = TokenBucket(capacity=60, now=1000.0)  # 1/s
        for _ in range(59):
            b.consume(now=1000.0)  # 剩 1
        b.consume(now=1000.0)  # 剩 0
        # 过 0.5s -> 0.5 令牌
        allowed, wait = b.consume(now=1000.5)
        assert allowed is False
        # 缺口 0.5 / 1.0 = 0.5s
        assert wait == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# RateLimiter：分级配额
# ---------------------------------------------------------------------------

class TestTierQuotas:
    def test_different_tiers_have_different_quota(self):
        rl = RateLimiter(_cfg())
        now = 1000.0
        # read 级配额 5
        for _ in range(5):
            allowed, _, info = rl.check_rate_limit("u1", "read", now=now)
            assert allowed is True
        allowed, _, info = rl.check_rate_limit("u1", "read", now=now)
        assert allowed is False
        assert info["limit"] == 5

        # backtest 级配额 2（与 read 独立桶）
        for _ in range(2):
            allowed, _, _ = rl.check_rate_limit("u1", "backtest", now=now)
            assert allowed is True
        allowed, _, info = rl.check_rate_limit("u1", "backtest", now=now)
        assert allowed is False
        assert info["limit"] == 2

    def test_dimensions_are_isolated(self):
        """不同维度（token/IP）互不影响。"""
        rl = RateLimiter(_cfg())
        now = 1000.0
        for _ in range(5):
            allowed, _, _ = rl.check_rate_limit("token-alice", "read", now=now)
            assert allowed is True
        # alice 第 6 次被拒
        allowed, _, _ = rl.check_rate_limit("token-alice", "read", now=now)
        assert allowed is False
        # bob 与另一个 IP 不受影响
        assert rl.check_rate_limit("token-bob", "read", now=now)[0] is True
        assert rl.check_rate_limit("ip-1.2.3.4", "read", now=now)[0] is True

    def test_unknown_tier_falls_back_to_default(self):
        rl = RateLimiter(_cfg())
        now = 1000.0
        allowed, _, info = rl.check_rate_limit("u", "nonexistent-tier", now=now)
        assert allowed is True
        assert info["limit"] == 5  # default_tier=read -> 5

    def test_default_tiers_applied_when_partial(self):
        """配置只给部分分级时，其余分级用默认值补齐。"""
        rl = RateLimiter({"enabled": True, "tiers": {"read": 7}})
        status = rl.get_status()
        assert status["tiers"]["read"] == 7
        assert status["tiers"]["backtest"] == DEFAULT_TIERS["backtest"]


# ---------------------------------------------------------------------------
# 白名单
# ---------------------------------------------------------------------------

class TestWhitelist:
    def test_loopback_whitelisted(self):
        rl = RateLimiter(_cfg())
        assert rl.is_whitelisted("127.0.0.1") is True
        assert rl.is_whitelisted("::1") is True
        assert rl.is_whitelisted("localhost") is True

    def test_external_ip_not_whitelisted(self):
        rl = RateLimiter(_cfg())
        assert rl.is_whitelisted("8.8.8.8") is False
        assert rl.is_whitelisted("") is False

    def test_custom_whitelist(self):
        rl = RateLimiter(_cfg(whitelist=["10.0.0.5"]))
        assert rl.is_whitelisted("10.0.0.5") is True
        assert rl.is_whitelisted("127.0.0.1") is False


# ---------------------------------------------------------------------------
# enabled=false
# ---------------------------------------------------------------------------

class TestDisabled:
    def test_disabled_allows_everything(self):
        rl = RateLimiter(_cfg(enabled=False))
        now = 1000.0
        for _ in range(100):
            allowed, wait, info = rl.check_rate_limit("anyone", "read", now=now)
            assert allowed is True
            assert wait == 0.0
        # disabled 时不消耗、不建桶，remaining 为 None
        assert info["remaining"] is None
        assert rl.get_status()["bucket_count"] == 0

    def test_default_disabled_when_config_none(self):
        rl = RateLimiter(None)
        assert rl.is_enabled() is False
        allowed, _, _ = rl.check_rate_limit("x", "read")
        assert allowed is True


# ---------------------------------------------------------------------------
# LRU 清理
# ---------------------------------------------------------------------------

class TestLRUCleanup:
    def test_cleanup_removes_idle_keeps_active(self):
        rl = RateLimiter(_cfg())
        t0 = 1000.0
        rl.check_rate_limit("stale", "read", now=t0)
        rl.check_rate_limit("fresh", "read", now=t0)
        # fresh 在 t1 又被访问
        t1 = t0 + 1000.0
        rl.check_rate_limit("fresh", "read", now=t1)
        assert rl.get_status()["bucket_count"] == 2

        # t2 = t0+4000：stale 闲置 4000s(>3600)，fresh 闲置 3000s(<3600)
        t2 = t0 + 4000.0
        removed = rl.cleanup_expired(now=t2)
        assert removed == 1
        status = rl.get_status()
        assert status["bucket_count"] == 1
        remaining_dims = list(status["current_limits"]["read"].keys())
        assert remaining_dims == ["fresh"]

    def test_active_bucket_survives_repeated_cleanup(self):
        """活跃桶在多次清理后仍保留且配额不丢。"""
        rl = RateLimiter(_cfg())
        now = 1000.0
        rl.check_rate_limit("busy", "read", now=now)
        rl.check_rate_limit("busy", "read", now=now)  # 用掉 2/5
        # 即使清理时刻很晚，只要 busy 最近被访问就保留
        rl.cleanup_expired(now=now + 10.0)
        assert rl.get_status()["bucket_count"] == 1
        # 剩余 3 个
        allowed, _, info = rl.check_rate_limit("busy", "read", now=now + 10.0)
        assert allowed is True
        assert info["remaining"] == 2  # 又用掉 1 个


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_status_shape_and_remaining(self):
        rl = RateLimiter(_cfg())
        rl.check_rate_limit("alice", "read", now=1000.0)
        rl.check_rate_limit("alice", "read", now=1000.0)
        status = rl.get_status()
        assert status["enabled"] is True
        assert status["tiers"]["read"] == 5
        assert status["tiers"]["backtest"] == 2
        assert "127.0.0.1" in status["whitelist"]
        assert status["default_tier"] == "read"
        alice = status["current_limits"]["read"]["alice"]
        assert alice["limit"] == 5
        assert alice["remaining"] == 3

    def test_status_is_snapshot_copy(self):
        """get_status 返回的 current_limits 不应被后续内部修改影响。"""
        rl = RateLimiter(_cfg())
        rl.check_rate_limit("a", "read", now=1000.0)
        snap = rl.get_status()
        before = snap["current_limits"]["read"]["a"]["remaining"]
        rl.check_rate_limit("a", "read", now=1000.0)
        assert snap["current_limits"]["read"]["a"]["remaining"] == before


# ---------------------------------------------------------------------------
# 中间件集成测试（最小 FastAPI App，镜像 server.py 的限流逻辑）
# ---------------------------------------------------------------------------

# TestClient 的来源 IP 固定为 "testclient"
_TEST_CLIENT_IP = "testclient"


def build_rate_limit_app(rate_limiter: RateLimiter) -> FastAPI:
    """构建最小 FastAPI 应用，中间件逻辑与 web-dashboard/server.py 一致。"""
    app = FastAPI()

    public_paths = {"/api/health"}

    def tier_for(path: str, method: str):
        if not path.startswith("/api/"):
            return None
        if path in public_paths:
            return None
        if method == "POST" and path == "/api/backtest":
            return "backtest"
        if path in {"/api/backtest_compare", "/api/backtest_portfolio",
                    "/api/walkthrough/run"} or path.startswith("/api/optimize/"):
            return "backtest"
        if path.startswith(("/api/alerts/trigger", "/api/daily_reports/export",
                             "/api/backup/")):
            return "admin"
        if method == "POST" and path == "/api/trade_toggle":
            return "trade"
        return "read"

    # 限流中间件先定义（内层）；认证模拟中间件后定义（外层，先执行），
    # 这样 _attach_token 先写入 request.state.token_info，限流中间件再读取。
    @app.middleware("http")
    async def _rate_limit(request: Request, call_next):
        path = request.url.path
        tier = tier_for(path, request.method)
        if tier is None or not rate_limiter.is_enabled():
            return await call_next(request)
        client_ip = request.client.host if request.client else ""
        if rate_limiter.is_whitelisted(client_ip):
            return await call_next(request)
        token_info = getattr(request.state, "token_info", None)
        if token_info:
            key = str(token_info.get("name") or "authenticated")
        else:
            key = client_ip
        allowed, retry_after, _ = rate_limiter.check_rate_limit(key, tier)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"code": 429, "message": "Too Many Requests", "data": None},
                headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
            )
        return await call_next(request)

    # 模拟认证中间件（外层，须在限流之前执行以写入 token_info）
    @app.middleware("http")
    async def _attach_token(request: Request, call_next):
        tok = request.headers.get("x-demo-token")
        if tok:
            request.state.token_info = {"name": tok}
        return await call_next(request)

    @app.get("/api/health")
    async def health():
        return {"code": 0, "message": "success", "data": {"ok": True}}

    @app.get("/api/symbols")
    async def symbols():
        return {"code": 0, "message": "success", "data": []}

    @app.post("/api/backtest")
    async def backtest():
        return {"code": 0, "message": "success", "data": {}}

    @app.post("/api/trade_toggle")
    async def trade_toggle():
        return {"code": 0, "message": "success", "data": {"on": True}}

    @app.get("/api/rate_limit/status")
    async def status():
        return {"code": 0, "message": "success", "data": rate_limiter.get_status()}

    return app


class TestRateLimitMiddleware:
    @pytest.fixture()
    def client(self):
        cfg = {
            "enabled": True,
            "tiers": {"read": 3, "trade": 2, "admin": 1, "backtest": 2},
            "whitelist": ["127.0.0.1"],  # 不含 testclient，便于测 429
        }
        rl = RateLimiter(cfg)
        return TestClient(build_rate_limit_app(rl))

    def test_public_health_never_limited(self, client):
        for _ in range(10):
            assert client.get("/api/health").status_code == 200

    def test_non_api_path_not_limited(self, client):
        # /docs 等非 /api 路径不参与限流（这里没有该路由，验证 tier=None 逻辑）
        # 直接断言限流器对非 /api 路径放行：用一个不存在的 /static 路径
        r = client.get("/static/nope")
        assert r.status_code == 404  # 不限流，直接落到路由 404

    def test_read_tier_returns_429_after_quota(self, client):
        for _ in range(3):
            assert client.get("/api/symbols").status_code == 200
        r = client.get("/api/symbols")
        assert r.status_code == 429

    def test_429_body_and_retry_after_header(self, client):
        for _ in range(3):
            client.get("/api/symbols")
        r = client.get("/api/symbols")
        assert r.status_code == 429
        assert r.json() == {"code": 429, "message": "Too Many Requests", "data": None}
        assert "retry-after" in {k.lower() for k in r.headers}
        assert int(r.headers["retry-after"]) >= 1

    def test_backtest_tier_smaller_quota(self, client):
        for _ in range(2):
            assert client.post("/api/backtest").status_code == 200
        assert client.post("/api/backtest").status_code == 429

    def test_token_scoped_vs_ip_scoped(self, client):
        # token=alice 打满 read=3
        for _ in range(3):
            assert client.get(
                "/api/symbols", headers={"x-demo-token": "alice"}
            ).status_code == 200
        # alice 第 4 次 429
        assert client.get(
            "/api/symbols", headers={"x-demo-token": "alice"}
        ).status_code == 429
        # bob（不同 token）独立桶，不受影响
        assert client.get(
            "/api/symbols", headers={"x-demo-token": "bob"}
        ).status_code == 200
        # 无 token 按 IP（testclient），又是另一个桶
        assert client.get("/api/symbols").status_code == 200

    def test_status_endpoint(self, client):
        client.get("/api/symbols")
        r = client.get("/api/rate_limit/status")
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["enabled"] is True
        assert data["tiers"]["read"] == 3
        assert "read" in data["current_limits"]


class TestWhitelistMiddleware:
    def test_whitelisted_client_bypasses_limit(self):
        cfg = {
            "enabled": True,
            "tiers": {"read": 1},
            "whitelist": [_TEST_CLIENT_IP],  # 把 TestClient 自身加白
        }
        rl = RateLimiter(cfg)
        client = TestClient(build_rate_limit_app(rl))
        for _ in range(10):
            assert client.get("/api/symbols").status_code == 200


class TestMiddlewareDisabled:
    def test_disabled_passes_everything(self):
        cfg = {"enabled": False, "tiers": {"read": 1},
               "whitelist": ["127.0.0.1"]}
        rl = RateLimiter(cfg)
        client = TestClient(build_rate_limit_app(rl))
        for _ in range(10):
            assert client.get("/api/symbols").status_code == 200
        assert client.post("/api/backtest").status_code == 200
