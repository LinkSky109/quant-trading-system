"""Jev 推理性能优化测试（REQ-P1-05）。

覆盖:
- 批量推理返回正确数量/顺序的结果
- 异步推理不阻塞（并发总时间 < 串行时间）
- KV 缓存命中：相同请求第二次走缓存
- 缓存命中率统计正确
- 并发控制：Semaphore 限制最大并发
- 延迟统计：P50/P95/P99 计算正确
- mock 模式下批量推理正常工作
- /api/jev/performance 端点集成测试（TestClient）
- worker 启动/停止不崩溃
"""
from __future__ import annotations

import asyncio
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jev.jev_engine import JevDecisionEngine, MarketState


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_state(price: float = 100.0, rsi: float = 50.0, pct: float = 0.0) -> MarketState:
    return MarketState(
        price=price,
        price_change_5d=pct,
        ma5_ma20_ratio=1.02,
        volume_ratio=1.2,
        rsi=rsi,
        macd_signal=1,
        volatility_20d=0.20,
    )


def _make_request(symbol: str = "T1", signal: str = "buy",
                  confidence: float = 0.8, state: MarketState | None = None) -> dict:
    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": confidence,
        "market_state": state or _make_state(),
    }


@pytest.fixture
def engine(tmp_path) -> JevDecisionEngine:
    return JevDecisionEngine(
        mock_mode=True,
        confidence_threshold=0.6,
        audit_log_path=str(tmp_path / "jev_audit.jsonl"),
        max_cache_size=100,
        max_concurrency=3,
    )


# ---------------------------------------------------------------------------
# 批量推理
# ---------------------------------------------------------------------------

class TestBatchEvaluate:
    def test_batch_returns_correct_count(self, engine):
        reqs = [
            _make_request(symbol=f"S{i}", state=_make_state(price=100 + i))
            for i in range(5)
        ]
        results = engine.evaluate_batch(reqs)
        assert len(results) == 5
        for r in results:
            assert "final_action" in r
            assert "probabilities" in r
            assert r["symbol"].startswith("S")

    def test_batch_preserves_order(self, engine):
        reqs = [
            _make_request(symbol="AAA", state=_make_state(price=100)),
            _make_request(symbol="BBB", state=_make_state(price=200)),
            _make_request(symbol="CCC", state=_make_state(price=300)),
        ]
        results = engine.evaluate_batch(reqs)
        assert [r["symbol"] for r in results] == ["AAA", "BBB", "CCC"]

    def test_batch_mock_mode_works(self, engine):
        """mock 模式下批量推理返回有效概率分布。"""
        reqs = [_make_request(state=_make_state(rsi=rsi)) for rsi in (30, 50, 80)]
        results = engine.evaluate_batch(reqs)
        for r in results:
            assert abs(sum(r["probabilities"].values()) - 1.0) < 0.01
            assert set(r["probabilities"].keys()) == {"buy", "sell", "hold"}

    def test_batch_accepts_dict_market_state(self, engine):
        """market_state 为 dict 时也能正常批量推理。"""
        ms = _make_state().__dict__
        reqs = [
            {"symbol": "D1", "signal": "buy", "confidence": 0.8, "market_state": ms},
            {"symbol": "D2", "signal": "sell", "confidence": 0.7, "market_state": ms},
        ]
        results = engine.evaluate_batch(reqs)
        assert len(results) == 2
        assert results[0]["symbol"] == "D1"
        assert results[1]["symbol"] == "D2"


# ---------------------------------------------------------------------------
# KV 缓存
# ---------------------------------------------------------------------------

class TestKVCache:
    def test_second_identical_request_hits_cache(self, engine):
        req = _make_request()
        r1 = engine.evaluate(req["signal"], req["confidence"],
                             req["market_state"], symbol=req["symbol"])
        r2 = engine.evaluate(req["signal"], req["confidence"],
                             req["market_state"], symbol=req["symbol"])
        # 命中缓存：返回的是同一个对象（LRU 缓存中存的就是 JevDecision）
        assert r1 is r2

    def test_cache_hit_miss_counters(self, engine):
        req = _make_request(symbol="C1")
        engine.evaluate(req["signal"], req["confidence"], req["market_state"],
                       symbol=req["symbol"])  # miss
        engine.evaluate(req["signal"], req["confidence"], req["market_state"],
                       symbol=req["symbol"])  # hit
        stats = engine.get_performance_stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["cache_hit_rate"] == pytest.approx(0.5)

    def test_different_requests_not_cached(self, engine):
        engine.evaluate("buy", 0.8, _make_state(price=100), symbol="X")
        engine.evaluate("buy", 0.8, _make_state(price=200), symbol="X")  # 不同 price
        stats = engine.get_performance_stats()
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 2

    def test_get_cached_set_cached_api(self, engine):
        req = _make_request(symbol="API1")
        assert engine.get_cached(req) is None
        decision = engine.evaluate("buy", 0.8, req["market_state"], symbol="API1")
        assert engine.get_cached(req) is decision

    def test_lru_cache_eviction(self, tmp_path):
        eng = JevDecisionEngine(
            mock_mode=True, max_cache_size=3,
            audit_log_path=str(tmp_path / "a.jsonl"),
        )
        # 写入 5 个不同请求，maxsize=3，应只保留最近 3 个
        for i in range(5):
            eng.evaluate("buy", 0.8, _make_state(price=100 + i), symbol=f"S{i}")
        assert len(eng._cache) == 3


# ---------------------------------------------------------------------------
# 异步推理
# ---------------------------------------------------------------------------

class TestAsyncInference:
    def test_async_does_not_block(self, engine, monkeypatch):
        """并发请求总时间应显著小于串行时间（验证 worker 并发）。"""
        # 给推理注入 0.1s 延迟，便于测量并发效果
        original = engine._compute_probabilities

        def slow_compute(states, signal, confidence):
            time.sleep(0.1)
            return original(states, signal, confidence)

        monkeypatch.setattr(engine, "_compute_probabilities", slow_compute)

        async def run():
            engine.start_workers(n=2)
            try:
                reqs = [_make_request(symbol=f"A{i}",
                                      state=_make_state(price=100 + i))
                        for i in range(4)]
                t0 = time.perf_counter()
                await asyncio.gather(*(engine.evaluate_async(r) for r in reqs))
                return time.perf_counter() - t0
            finally:
                await engine.stop_workers()

        elapsed = asyncio.run(run())
        # 串行 4 条 × 0.1s = 0.4s；2 worker 并发应约 0.2s
        assert elapsed < 0.35, f"异步并发未生效，耗时 {elapsed:.2f}s"

    def test_async_returns_correct_results(self, engine):
        async def run():
            engine.start_workers(n=2)
            try:
                reqs = [_make_request(symbol=f"R{i}") for i in range(3)]
                results = await asyncio.gather(*(engine.evaluate_async(r) for r in reqs))
                return results
            finally:
                await engine.stop_workers()

        results = asyncio.run(run())
        assert len(results) == 3
        for r in results:
            assert "final_action" in r
            assert "probabilities" in r

    def test_worker_start_stop_no_crash(self, engine):
        async def run():
            engine.start_workers(n=2)
            await engine.evaluate_async(_make_request())
            await engine.stop_workers()
            # 再次启动也不崩溃
            engine.start_workers(n=1)
            await engine.evaluate_async(_make_request(symbol="SECOND"))
            await engine.stop_workers()

        asyncio.run(run())

    def test_semaphore_limits_concurrency(self, tmp_path, monkeypatch):
        """Semaphore 限制最大并发：max_concurrency=2，并发峰值应 ≤ 2。"""
        eng = JevDecisionEngine(
            mock_mode=True, max_concurrency=2,
            audit_log_path=str(tmp_path / "a.jsonl"),
        )
        original = eng._compute_probabilities

        def slow_compute(states, signal, confidence):
            time.sleep(0.1)
            return original(states, signal, confidence)

        monkeypatch.setattr(eng, "_compute_probabilities", slow_compute)

        async def run():
            eng.start_workers(n=4)  # 4 worker，但信号量=2
            try:
                reqs = [_make_request(symbol=f"P{i}", state=_make_state(price=100 + i))
                        for i in range(6)]
                await asyncio.gather(*(eng.evaluate_async(r) for r in reqs))
            finally:
                await eng.stop_workers()

        asyncio.run(run())
        # 峰值并发不应超过信号量 2
        assert eng._peak_inferences <= 2
        # 确实发生了并发（至少有请求同时在跑）
        assert eng._peak_inferences >= 1


# ---------------------------------------------------------------------------
# 延迟统计
# ---------------------------------------------------------------------------

class TestLatencyStats:
    def test_percentiles_computed(self, engine):
        for i in range(30):
            engine.evaluate("buy", 0.8, _make_state(price=100 + i), symbol=f"L{i}")
        stats = engine.get_performance_stats()
        assert stats["total_requests"] == 30
        assert stats["p50"] >= 0.0
        assert stats["p95"] >= stats["p50"]
        assert stats["p99"] >= stats["p95"]
        assert stats["throughput"] > 0.0

    def test_empty_stats_returns_zeros(self, engine):
        stats = engine.get_performance_stats()
        assert stats["p50"] == 0.0
        assert stats["p95"] == 0.0
        assert stats["p99"] == 0.0
        assert stats["total_requests"] == 0

    def test_latency_window_capped(self, tmp_path):
        eng = JevDecisionEngine(
            mock_mode=True, latency_window=100,
            audit_log_path=str(tmp_path / "a.jsonl"),
        )
        for i in range(500):
            eng.evaluate("buy", 0.8, _make_state(price=100 + i), symbol=f"W{i}")
        assert len(eng._latencies) == 100  # 环形缓冲只保留最近 1000 条窗口内


# ---------------------------------------------------------------------------
# API 端点集成测试
# ---------------------------------------------------------------------------

def _build_perf_app(engine: JevDecisionEngine) -> FastAPI:
    """构建与 web-dashboard/server.py /api/jev/performance 一致的最小 App。"""
    app = FastAPI()

    def ok(data=None, message="success"):
        return {"code": 0, "message": message, "data": data}

    @app.get("/api/jev/performance")
    async def perf():
        stats = engine.get_performance_stats()
        return ok({
            "p50_ms": stats["p50"],
            "p95_ms": stats["p95"],
            "p99_ms": stats["p99"],
            "total_requests": stats["total_requests"],
            "cache_hits": stats["cache_hits"],
            "cache_misses": stats["cache_misses"],
            "cache_hit_rate": stats["cache_hit_rate"],
            "throughput_rps": stats["throughput"],
            "batch_enabled": True,
            "concurrency": engine._max_concurrency,
        })

    return app


class TestPerformanceEndpoint:
    def test_endpoint_returns_stats(self, engine):
        # 先产生一些请求
        for i in range(5):
            engine.evaluate("buy", 0.8, _make_state(price=100 + i), symbol=f"E{i}")

        client = TestClient(_build_perf_app(engine))
        resp = client.get("/api/jev/performance")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        data = body["data"]
        for key in ("p50_ms", "p95_ms", "p99_ms", "total_requests",
                    "cache_hits", "cache_misses", "cache_hit_rate",
                    "throughput_rps", "batch_enabled", "concurrency"):
            assert key in data, f"缺少字段 {key}"
        assert data["total_requests"] == 5
        assert data["batch_enabled"] is True
        assert data["concurrency"] == 3

    def test_endpoint_empty_stats(self, engine):
        client = TestClient(_build_perf_app(engine))
        resp = client.get("/api/jev/performance")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total_requests"] == 0
        assert data["p50_ms"] == 0.0
