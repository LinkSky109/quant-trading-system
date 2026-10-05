"""扩展路由注册测试（做市 / 期权 / 加密货币，#23-#25）。

使用 FastAPI TestClient + stub manager（mock 日K线），不依赖外部 API。
覆盖：
  - 路由注册成功、端点可达
  - 正常请求返回 code=0
  - 参数校验与数据缺失错误分支
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("fastapi.testclient")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# 路由模块以 _routes_* 平级导入，需把 web-dashboard 加入 path
WEB_DASHBOARD = Path(__file__).resolve().parent.parent / "web-dashboard"
if str(WEB_DASHBOARD) not in sys.path:
    sys.path.insert(0, str(WEB_DASHBOARD))

from _routes_crypto import register_crypto_routes  # noqa: E402
from _routes_market_making import register_market_making_routes  # noqa: E402
from _routes_options import register_options_routes  # noqa: E402


def ok(data=None, message="success"):
    return {"code": 0, "message": message, "data": data}


def err(code, message, http_status=400):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=http_status,
        content={"code": code, "message": message, "data": None},
    )


def make_ohlc(n=120, base=100.0, seed=42):
    rng = np.random.RandomState(seed)
    rets = rng.normal(0.0, 0.01, size=n)
    close = base * np.cumprod(1.0 + rets)
    open_ = close * (1.0 + rng.normal(0, 0.003, size=n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.004, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.004, size=n)))
    volume = rng.uniform(1e5, 1e6, size=n)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class StubManager:
    """stub manager：提供 get_daily_klines。"""

    def get_daily_klines(self, symbol):
        return make_ohlc(120, base=100.0)


class BrokenManager:
    def get_daily_klines(self, symbol):
        raise RuntimeError("no data")


SYMBOLS = ["TEST"]


def make_app(manager=None):
    app = FastAPI()
    mgr = manager or StubManager()
    register_market_making_routes(app, mgr, SYMBOLS, ok, err)
    register_options_routes(app, mgr, SYMBOLS, ok, err)
    register_crypto_routes(app, mgr, SYMBOLS, ok, err)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def client():
    return make_app()


# ---------------------------------------------------------------------------
# 做市路由 (#23)
# ---------------------------------------------------------------------------


class TestMarketMakingRoutes:
    def test_quotes_ok(self, client):
        r = client.post("/api/marketmaking/quotes",
                        json={"symbol": "TEST", "inventory": 20})
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["quote"]["bid"] < body["data"]["quote"]["ask"]

    def test_quotes_empty_symbol(self, client):
        r = client.post("/api/marketmaking/quotes", json={"symbol": " "})
        assert r.status_code == 400

    def test_quotes_missing_klines(self):
        c = make_app(BrokenManager())
        r = c.post("/api/marketmaking/quotes", json={"symbol": "TEST"})
        assert r.status_code == 400
        assert r.json()["code"] == 404

    def test_signals_ok(self, client):
        r = client.post("/api/marketmaking/signals", json={"symbol": "TEST"})
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["signals_count"] >= 0
        assert "final_inventory" in body["data"]

    def test_params_ok(self, client):
        r = client.get("/api/marketmaking/params")
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["params"]["spread_k"] == 0.5


# ---------------------------------------------------------------------------
# 期权路由 (#24)
# ---------------------------------------------------------------------------


class TestOptionsRoutes:
    def test_chain_ok(self, client):
        r = client.post("/api/options/chain",
                        json={"symbol": "TEST", "spot": 100.0,
                              "expiry": "2026-12-31", "ttm_years": 0.5})
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["data_source"] == "mock"
        assert len(body["data"]["calls"]) > 0
        assert len(body["data"]["puts"]) > 0

    def test_chain_bad_params(self, client):
        r = client.post("/api/options/chain",
                        json={"symbol": "TEST", "spot": -1,
                              "expiry": "2026-12-31", "ttm_years": 0.5})
        assert r.status_code == 400
        assert r.json()["code"] == 422

    def test_greeks_ok(self, client):
        r = client.post("/api/options/greeks",
                        json={"spot": 100.0, "strike": 100.0,
                              "ttm_years": 0.5, "vol": 0.25})
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert 0 < body["data"]["greeks"]["delta"] < 1

    def test_greeks_bad_params(self, client):
        r = client.post("/api/options/greeks",
                        json={"spot": 0, "strike": 100.0,
                              "ttm_years": 0.5, "vol": 0.25})
        assert r.status_code == 400
        assert r.json()["code"] == 422

    def test_strategy_covered_call(self, client):
        r = client.post("/api/options/strategy",
                        json={"strategy": "covered_call", "symbol": "TEST",
                              "spot": 100.0, "expiry": "2026-12-31",
                              "ttm_years": 0.5, "strike": 105.0})
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["result"]["name"] == "covered_call"

    def test_strategy_bull_call_spread(self, client):
        r = client.post("/api/options/strategy",
                        json={"strategy": "bull_call_spread", "symbol": "TEST",
                              "spot": 100.0, "expiry": "2026-12-31",
                              "ttm_years": 0.5, "strike_low": 95.0,
                              "strike_high": 105.0})
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["result"]["name"] == "bull_call_spread"

    def test_strategy_straddle_default_strike(self, client):
        r = client.post("/api/options/strategy",
                        json={"strategy": "straddle", "symbol": "TEST",
                              "spot": 100.0, "expiry": "2026-12-31",
                              "ttm_years": 0.5})
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["result"]["name"] == "straddle"

    def test_strategy_missing_strike(self, client):
        r = client.post("/api/options/strategy",
                        json={"strategy": "covered_call", "symbol": "TEST",
                              "spot": 100.0, "expiry": "2026-12-31",
                              "ttm_years": 0.5})
        assert r.status_code == 400
        assert r.json()["code"] == 422

    def test_strategy_strike_not_in_chain(self, client):
        r = client.post("/api/options/strategy",
                        json={"strategy": "covered_call", "symbol": "TEST",
                              "spot": 100.0, "expiry": "2026-12-31",
                              "ttm_years": 0.5, "strike": 999.0})
        assert r.status_code == 400
        assert r.json()["code"] == 404


# ---------------------------------------------------------------------------
# 加密货币路由 (#25)
# ---------------------------------------------------------------------------


class TestCryptoRoutes:
    def test_klines_ok(self, client):
        r = client.post("/api/crypto/klines",
                        json={"symbol": "BTC-USD", "days": 30})
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["count"] == 30
        assert body["data"]["trading_hours"] == "24x7"
        assert body["data"]["timezone"] == "UTC"

    def test_klines_invalid_symbol(self, client):
        r = client.post("/api/crypto/klines",
                        json={"symbol": "600519.SH", "days": 30})
        assert r.status_code == 400
        assert r.json()["code"] == 422

    def test_klines_bad_days(self, client):
        r = client.post("/api/crypto/klines",
                        json={"symbol": "BTC-USD", "days": 0})
        assert r.status_code == 400
        assert r.json()["code"] == 422

    def test_market_status_ok(self, client):
        r = client.get("/api/crypto/market_status")
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["is_open"] is True

    def test_risk_params_ok(self, client):
        r = client.get("/api/crypto/risk_params")
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["risk_params"]["single_stop_loss"] == 0.08
        assert "BTC-USD" in body["data"]["default_symbols"]

    def test_adapt_params_ok(self, client):
        r = client.post("/api/crypto/adapt_params",
                        json={"params": {"single_stop_loss": 0.03}, "symbol": "ETH-USD"})
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["adapted"]["single_stop_loss"] == 0.08

    def test_adapt_params_empty(self, client):
        r = client.post("/api/crypto/adapt_params", json={"params": {}})
        assert r.status_code == 400
        assert r.json()["code"] == 422
