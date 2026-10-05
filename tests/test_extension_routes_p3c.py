"""扩展路由注册测试（另类数据 / 在线学习 / i18n / 主题，#26-#29，REQ-P3-04/05/07/08）。

使用 FastAPI TestClient + stub manager（mock 数据），不依赖外部 API。
另含主题前端 CSS 变量校验（通过 TestClient 请求挂载的看板页面）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("fastapi.testclient")

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

WEB_DASHBOARD = Path(__file__).resolve().parent.parent / "web-dashboard"
if str(WEB_DASHBOARD) not in sys.path:
    sys.path.insert(0, str(WEB_DASHBOARD))

import _routes_theme  # noqa: E402
from _routes_alternative import register_alternative_routes  # noqa: E402
from _routes_i18n import register_i18n_routes  # noqa: E402
from _routes_online_learning import (  # noqa: E402
    register_online_learning_routes,
    reset_pipeline,
)
from _routes_theme import register_theme_routes  # noqa: E402

REPO_ROOT = WEB_DASHBOARD.parent
DASHBOARD_HTML = WEB_DASHBOARD / "quant_dashboard_realtime.html"


def ok(data=None, message="success"):
    return {"code": 0, "message": message, "data": data}


def err(code, message, http_status=400):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=http_status,
        content={"code": code, "message": message, "data": None},
    )


class StubManager:
    pass


SYMBOLS = ["600519.SH"]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # 主题偏好落到临时文件，避免污染仓库
    monkeypatch.setattr(_routes_theme, "THEME_FILE", tmp_path / "theme_pref.json")
    reset_pipeline(n_features=2)  # 在线学习单例复位（与反馈特征数一致）
    app = FastAPI()
    mgr = StubManager()
    register_alternative_routes(app, mgr, SYMBOLS, ok, err)
    register_online_learning_routes(app, mgr, SYMBOLS, ok, err)
    register_i18n_routes(app, mgr, SYMBOLS, ok, err)
    register_theme_routes(app, mgr, SYMBOLS, ok, err)

    # 复刻 server.py 的看板挂载方式，用于 CSS 变量验证
    @app.get("/")
    async def dashboard():
        return FileResponse(DASHBOARD_HTML, media_type="text/html")

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# 另类数据路由 (#26)
# ---------------------------------------------------------------------------


class TestAlternativeRoutes:
    def test_sources(self, client):
        r = client.get("/api/alternative/sources")
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        names = [s["name"] for s in body["data"]["sources"]]
        assert names == ["satellite", "social", "supply_chain", "esg"]
        assert body["data"]["real_source_reserved"] is True
        assert "sentiment_score" in body["data"]["factors"]

    def test_preview(self, client):
        r = client.post("/api/alternative/preview", json={
            "source": "social", "symbol": "600519.SH", "days": 30})
        assert r.json()["code"] == 0
        data = r.json()["data"]
        assert data["count"] == 30
        assert "sentiment_raw" in data["columns"]

    def test_preview_unknown_source(self, client):
        r = client.post("/api/alternative/preview", json={
            "source": "nope", "symbol": "600519.SH"})
        assert r.json()["code"] == 422

    def test_preview_empty_symbol(self, client):
        r = client.post("/api/alternative/preview", json={
            "source": "social", "symbol": "  "})
        assert r.json()["code"] == 400

    def test_preview_bad_days(self, client):
        r = client.post("/api/alternative/preview", json={
            "source": "social", "symbol": "600519.SH", "days": 0})
        assert r.json()["code"] == 422

    def test_factors_shift_lag(self, client):
        r = client.post("/api/alternative/factors", json={
            "symbol": "600519.SH", "days": 120, "lag": 1})
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["lag"] == 1
        assert set(body["data"]["factors"]) >= {"sentiment_score", "esg_score"}

    def test_factors_negative_lag(self, client):
        r = client.post("/api/alternative/factors", json={
            "symbol": "600519.SH", "lag": -1})
        assert r.json()["code"] == 422


# ---------------------------------------------------------------------------
# 在线学习路由 (#27)
# ---------------------------------------------------------------------------


def _fb_records(n, seed=0):
    import random
    rng = random.Random(seed)
    return [
        {"features": {"f1": rng.uniform(-1, 1), "f2": rng.uniform(-1, 1)},
         "realized_return": rng.uniform(-0.02, 0.02)}
        for _ in range(n)
    ]


class TestOnlineLearningRoutes:
    def test_update(self, client):
        r = client.post("/api/online_learning/update", json={
            "records": _fb_records(30)})
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["window_samples"] == 30
        assert "batch_loss" in body["data"]
        assert body["data"]["feature_keys"] == ["f1", "f2"]

    def test_update_empty_records(self, client):
        r = client.post("/api/online_learning/update", json={"records": []})
        assert r.json()["code"] == 400

    def test_update_dim_mismatch(self, client):
        reset_pipeline(n_features=3)
        r = client.post("/api/online_learning/update", json={
            "records": _fb_records(5)})  # 2 特征 vs 管线 3 特征
        assert r.json()["code"] == 400

    def test_stats(self, client):
        client.post("/api/online_learning/update", json={"records": _fb_records(20)})
        r = client.get("/api/online_learning/stats")
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["n_updates"] == 1
        assert "accuracy_proxy" in body["data"]

    def test_drift_check_before_any_update(self, client):
        r = client.post("/api/online_learning/drift_check", json={})
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["drift_detected"] is False

    def test_drift_check_acknowledge(self, client):
        client.post("/api/online_learning/update", json={"records": _fb_records(20)})
        r = client.post("/api/online_learning/drift_check", json={
            "acknowledge_baseline": 0.5})
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["baseline_loss"] == 0.5

    def test_reset(self, client):
        client.post("/api/online_learning/update", json={"records": _fb_records(20)})
        r = client.post("/api/online_learning/reset", json={
            "n_features": 4, "lr": 0.02, "window_size": 100})
        assert r.json()["code"] == 0
        r2 = client.get("/api/online_learning/stats")
        assert r2.json()["data"]["n_updates"] == 0

    def test_reset_invalid_params(self, client):
        r = client.post("/api/online_learning/reset", json={"n_features": 0})
        assert r.json()["code"] == 422
        r = client.post("/api/online_learning/reset", json={"lr": 2.0})
        assert r.json()["code"] == 422


# ---------------------------------------------------------------------------
# i18n 路由 (#28)
# ---------------------------------------------------------------------------


class TestI18nRoutes:
    def test_languages(self, client):
        r = client.get("/api/i18n/languages")
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["supported"] == ["zh", "en"]

    def test_messages_default(self, client):
        r = client.get("/api/i18n/messages")
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["lang"] == "zh"
        assert "api.success" in body["data"]["messages"]

    def test_messages_lang_param(self, client):
        r = client.get("/api/i18n/messages?lang=en")
        assert r.json()["data"]["lang"] == "en"

    def test_messages_accept_language_header(self, client):
        r = client.get("/api/i18n/messages", headers={
            "Accept-Language": "en-US,en;q=0.9"})
        assert r.json()["data"]["lang"] == "en"

    def test_messages_header_beats_default(self, client):
        # lang 参数优先于 header
        r = client.get("/api/i18n/messages?lang=zh", headers={
            "Accept-Language": "en-US"})
        assert r.json()["data"]["lang"] == "zh"

    def test_text_with_placeholder(self, client):
        r = client.get("/api/i18n/text", params={
            "key": "api.symbol_not_found", "lang": "en", "symbol": "AAPL"})
        body = r.json()
        assert body["code"] == 0
        assert "AAPL" in body["data"]["text"]

    def test_text_empty_key(self, client):
        r = client.get("/api/i18n/text", params={"key": " "})
        assert r.json()["code"] == 400

    def test_format_number_and_currency(self, client):
        r = client.post("/api/i18n/format", json={
            "lang": "en",
            "items": [
                {"type": "number", "value": 1234567.891},
                {"type": "currency", "value": 99.5},
                {"type": "percent", "value": 0.1234},
                {"type": "date", "value": "2026-10-05"},
            ]})
        body = r.json()
        assert body["code"] == 0
        texts = [x["text"] for x in body["data"]["results"]]
        assert texts[0] == "1,234,567.89"
        assert texts[1].startswith("$")
        assert texts[2] == "12.34%"
        assert texts[3] == "Oct 05, 2026"

    def test_format_invalid_type(self, client):
        r = client.post("/api/i18n/format", json={
            "items": [{"type": "bogus", "value": 1.0}]})
        assert r.json()["code"] == 422


# ---------------------------------------------------------------------------
# 主题偏好路由 (#29)
# ---------------------------------------------------------------------------


class TestThemeRoutes:
    def test_default_preference(self, client):
        r = client.get("/api/theme/preference")
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["theme"] == "system"
        assert body["data"]["saved"] is False

    def test_save_and_reload(self, client):
        r = client.post("/api/theme/preference", json={"theme": "light"})
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["saved"] is True
        # 重新 GET（走同一临时文件）确认持久化
        r2 = client.get("/api/theme/preference")
        assert r2.json()["data"]["theme"] == "light"
        assert r2.json()["data"]["saved"] is True

    def test_invalid_theme(self, client):
        r = client.post("/api/theme/preference", json={"theme": "blue"})
        assert r.json()["code"] == 422

    def test_valid_themes_listed(self, client):
        r = client.get("/api/theme/preference")
        assert r.json()["data"]["valid_themes"] == ["dark", "light", "system"]


# ---------------------------------------------------------------------------
# 前端主题 CSS 变量验证（通过 TestClient 请求看板页面）
# ---------------------------------------------------------------------------


class TestDashboardThemeFrontend:
    def test_css_variables_present(self, client):
        html = client.get("/").text
        assert "--bg:" in html          # 暗色变量
        assert "--card:" in html
        assert "--up:" in html
        assert '[data-theme="light"]' in html  # 亮色变量块
        assert "theme-transition" in html       # 平滑过渡类
        assert "prefers-color-scheme" in html   # 系统主题适配
        assert "registerTheme('quant-dark'" in html  # ECharts 主题
        assert "registerTheme('quant-light'" in html

    def test_i18n_frontend_present(self, client):
        html = client.get("/").text
        assert 'data-i18n="rt.last"' in html
        assert 'data-i18n-html="ui.title"' in html
        assert 'id="langSelect"' in html
        assert "I18nModule" in html
        assert "Accept-Language" in html

    def test_page_served_200(self, client):
        assert client.get("/").status_code == 200
