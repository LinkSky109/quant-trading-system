"""REQ-P1-09 因子风险暴露监控模块测试。

覆盖：
1. 组合暴露加权计算正确（含缺失 z-score 按 0 处理）
2. 权重不闭合抛 ValueError
3. 中性化限额：超限判定正确且 AlertManager.alert 被调用
4. 时序记录：写入快照后用新实例从 tmp_path 库读回
5. 调仓建议：正超限时建议减持高 z-score 标的
6. 因子收益贡献度手算对照
7. 因子相关性矩阵：对角线=1、完全相关因子 corr=1、矩阵形状
8. 全部 API 端点 TestClient 集成测试（正常路径 + 校验失败 400）
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web-dashboard"))

import pytest  # noqa: E402

from monitoring.alert import AlertManager  # noqa: E402
from risk.factor_exposure import FactorExposureMonitor  # noqa: E402


# ---------------------------------------------------------------------------
# 手工构造数据：3 个标的、4 个因子
# ---------------------------------------------------------------------------

HOLDINGS = [
    {"symbol": "AAA", "weight": 0.5},
    {"symbol": "BBB", "weight": 0.3},
    {"symbol": "CCC", "weight": 0.2},
]

# CCC 完全缺失 factor_scores —— 用于验证缺失标的按 0 处理
FACTOR_SCORES = {
    "AAA": {"value": 1.0, "growth": -1.0, "mom": 2.0, "qual": 0.5},
    "BBB": {"value": -1.0, "growth": 1.0, "mom": 0.0, "qual": -0.5},
}

# 手算期望组合暴露（CCC 按 0）：
#   value : 0.5*1.0 + 0.3*(-1.0) = 0.2
#   growth: 0.5*(-1.0) + 0.3*1.0 = -0.2
#   mom   : 0.5*2.0 + 0.3*0.0 = 1.0
#   qual  : 0.5*0.5 + 0.3*(-0.5) = 0.10
EXPECTED = {"value": 0.2, "growth": -0.2, "mom": 1.0, "qual": 0.10}


class TestExposureCalculation:
    """1+2：组合暴露加权计算与权重校验。"""

    def test_weighted_exposure_handles_missing_symbol(self):
        monitor = FactorExposureMonitor(db_path=":memory:")
        result = monitor.calculate_exposure(HOLDINGS, FACTOR_SCORES)

        for factor, expected in EXPECTED.items():
            assert result.exposures[factor] == pytest.approx(expected, abs=1e-9), \
                f"因子 {factor} 暴露计算错误"
        # CCC 无 z-score，应记录为缺失并按 0 参与计算
        assert "CCC" in result.missing_symbols
        assert result.weight_sum == pytest.approx(1.0)

    def test_non_closed_weights_raise(self):
        monitor = FactorExposureMonitor(db_path=":memory:")
        bad = [{"symbol": "AAA", "weight": 0.5},
               {"symbol": "BBB", "weight": 0.3}]  # 和=0.8，未闭合
        with pytest.raises(ValueError):
            monitor.calculate_exposure(bad, FACTOR_SCORES)

    def test_tolerance_boundary_accepted(self):
        # 和=1.00005 在容差 1e-4 内，应通过
        ok_holdings = [
            {"symbol": "AAA", "weight": 0.5},
            {"symbol": "BBB", "weight": 0.50005},
        ]
        monitor = FactorExposureMonitor(db_path=":memory:")
        result = monitor.calculate_exposure(ok_holdings, {"AAA": {"f": 1.0}})
        assert result.exposures["f"] == pytest.approx(0.5)


class TestLimitCheck:
    """3：中性化限额判定 + 告警触发。"""

    def test_breach_detected_and_alert_fired(self):
        alert_mgr = AlertManager()  # webhook 为空，仅记录 _history
        # 把 mom 限额设为 0.5，mom 暴露=1.0 必超限
        monitor = FactorExposureMonitor(
            db_path=":memory:",
            alert_manager=alert_mgr,
            thresholds={"mom": 0.5},
        )
        result = monitor.calculate_exposure(HOLDINGS, FACTOR_SCORES)
        breaches = monitor.check_limits(result.exposures)

        # 只有 mom 超限（其余因子 |暴露|<=1.0 默认阈值）
        assert len(breaches) == 1
        b = breaches[0]
        assert b.factor == "mom"
        assert b.exposure == pytest.approx(1.0)
        assert b.threshold == pytest.approx(0.5)
        assert b.exceeded_by == pytest.approx(0.5)

        # AlertManager.alert 应被调用，历史中能查到该 category
        alerts = alert_mgr.get_history(category="factor_exposure_limit")
        assert len(alerts) == 1
        assert alerts[0]["current_value"] == pytest.approx(1.0)
        assert alerts[0]["threshold"] == pytest.approx(0.5)
        assert alerts[0]["level"] == "WARNING"

    def test_no_alert_when_manager_none(self):
        # alert_manager=None 时不告警也不报错
        monitor = FactorExposureMonitor(db_path=":memory:", alert_manager=None,
                                        thresholds={"mom": 0.5})
        result = monitor.calculate_exposure(HOLDINGS, FACTOR_SCORES)
        breaches = monitor.check_limits(result.exposures)
        assert [x.factor for x in breaches] == ["mom"]

    def test_within_limit_no_breach(self):
        monitor = FactorExposureMonitor(db_path=":memory:", default_threshold=2.0)
        result = monitor.calculate_exposure(HOLDINGS, FACTOR_SCORES)
        assert monitor.check_limits(result.exposures) == []


class TestHistory:
    """4：时序快照持久化（tmp_path 库，新实例读回）。"""

    def test_snapshot_roundtrip(self, tmp_path):
        db = tmp_path / "exp.db"
        m1 = FactorExposureMonitor(db_path=str(db))
        m1.record_snapshot("2026-10-01", {"value": 0.2, "mom": 1.0})
        m1.record_snapshot("2026-10-02", {"value": 0.3, "mom": 1.2})

        # 新实例从同一库读回
        m2 = FactorExposureMonitor(db_path=str(db))
        rows = m2.get_history()
        assert len(rows) == 4  # 2 天 × 2 因子

        mom_rows = m2.get_history(factor="mom")
        assert [r["date"] for r in mom_rows] == ["2026-10-01", "2026-10-02"]
        assert mom_rows[0]["exposure"] == pytest.approx(1.0)
        assert mom_rows[1]["exposure"] == pytest.approx(1.2)

        # 区间过滤
        sliced = m2.get_history(factor="mom", start="2026-10-02")
        assert len(sliced) == 1 and sliced[0]["date"] == "2026-10-02"

        # 前端图表数据含 threshold
        chart = m2.time_series_chart_data("mom")
        assert chart[0]["date"] == "2026-10-01"
        assert chart[0]["exposure"] == pytest.approx(1.0)
        assert "threshold" in chart[0]


class TestNeutralize:
    """5：调仓建议方向正确。"""

    def test_positive_breach_reduce_high_z(self):
        monitor = FactorExposureMonitor(
            db_path=":memory:", thresholds={"mom": 0.5}
        )
        sugs = monitor.neutralize(HOLDINGS, FACTOR_SCORES)

        by_sym = {s["symbol"]: s for s in sugs}
        # AAA 在 mom 上 z=2.0（最高），正超限应减持
        assert by_sym["AAA"]["action"] == "decrease"
        assert by_sym["AAA"]["target_factor"] == "mom"
        assert by_sym["AAA"]["current_zscore"] == pytest.approx(2.0)
        # BBB 在 mom 上 z=0.0（最低），应增持
        assert by_sym["BBB"]["action"] == "increase"
        # 权重调整量为正且有限
        assert by_sym["AAA"]["weight_delta"] > 0

    def test_target_factor_scoped(self):
        monitor = FactorExposureMonitor(
            db_path=":memory:", thresholds={"mom": 0.5, "value": 0.01}
        )
        # value=0.2 也超限，但只指定 mom
        sugs = monitor.neutralize(HOLDINGS, FACTOR_SCORES, target_factor="mom")
        assert all(s["target_factor"] == "mom" for s in sugs)

    def test_negative_breach_increase_high_z(self):
        # growth=-0.2，把 growth 限额设为 0.1 → 负超限
        monitor = FactorExposureMonitor(
            db_path=":memory:", thresholds={"growth": 0.1}
        )
        sugs = monitor.neutralize(HOLDINGS, FACTOR_SCORES, target_factor="growth")
        by_sym = {s["symbol"]: s for s in sugs}
        # growth 上 AAA=-1.0(最低)，BBB=+1.0(最高)；负超限应增持高 z(BBB)、减持低 z(AAA)
        assert by_sym["BBB"]["action"] == "increase"
        assert by_sym["AAA"]["action"] == "decrease"


class TestContributions:
    """6：因子收益贡献度手算对照。"""

    def test_contributions(self):
        monitor = FactorExposureMonitor(db_path=":memory:")
        exposure = {"value": 0.2, "mom": 1.0}
        factor_returns = {"value": 0.01, "mom": 0.05, "unknown": 0.9}
        out = monitor.factor_contributions(exposure, factor_returns)

        assert out["contributions"]["value"] == pytest.approx(0.2 * 0.01)
        assert out["contributions"]["mom"] == pytest.approx(1.0 * 0.05)
        assert out["total"] == pytest.approx(0.2 * 0.01 + 1.0 * 0.05)


class TestCorrelation:
    """7：因子相关性矩阵。"""

    def test_correlation_matrix(self):
        monitor = FactorExposureMonitor(db_path=":memory:")
        # f2 = 2*f1 完全正相关；f3 为独立波动因子
        scores = {
            "s1": {"f1": 1.0, "f2": 2.0, "f3": 1.0},
            "s2": {"f1": 2.0, "f2": 4.0, "f3": 0.0},
            "s3": {"f1": 3.0, "f2": 6.0, "f3": -1.0},
        }
        mat = monitor.factor_correlation(scores)
        assert mat.shape == (3, 3)
        # 对角线自相关=1
        for f in ("f1", "f2", "f3"):
            assert mat.loc[f, f] == pytest.approx(1.0)
        # f1 与 f2 完全相关
        assert mat.loc["f1", "f2"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 8：API 端点集成测试
# ---------------------------------------------------------------------------


def _build_client(tmp_path):
    """注册因子暴露路由到独立最小 app，monitor/alert 绑定 tmp_path 库。"""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    import _routes_factor_exposure as routes

    alert_mgr = AlertManager()
    monitor = FactorExposureMonitor(
        db_path=str(tmp_path / "api_exp.db"),
        alert_manager=alert_mgr,
        thresholds={"mom": 0.5},  # 制造一个超限
    )
    # 注入模块单例，避免写真实 data/quant_trading.db
    routes._MONITOR = monitor
    routes._ALERT_MANAGER = alert_mgr

    app = FastAPI()

    def ok(data=None, message="success"):
        return {"code": 0, "message": message, "data": data}

    def err(code, message, http_status=400):
        return JSONResponse(status_code=http_status,
                            content={"code": code, "message": message, "data": None})

    routes.register_factor_exposure_routes(app, None, set(), ok, err)
    return TestClient(app), monitor


class TestFactorExposureAPI:
    """全部 API 端点。"""

    def test_post_exposure_happy(self, tmp_path):
        client, monitor = _build_client(tmp_path)
        payload = {"holdings": HOLDINGS, "factor_scores": FACTOR_SCORES}
        r = client.post("/api/risk/factor_exposure", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        data = body["data"]
        assert data["exposures"]["mom"] == pytest.approx(1.0)
        assert "CCC" in data["missing_symbols"]
        # mom 限额 0.5 → 应出现在 breaches
        breached = [b["factor"] for b in data["breaches"]]
        assert "mom" in breached
        # 类别汇总字段存在
        assert isinstance(data["category_exposures"], dict)

    def test_post_exposure_bad_weights_400(self, tmp_path):
        client, _ = _build_client(tmp_path)
        bad = {"holdings": [{"symbol": "A", "weight": 0.5}], "factor_scores": {}}
        r = client.post("/api/risk/factor_exposure", json=bad)
        assert r.status_code == 400
        assert r.json()["code"] != 0

    def test_post_exposure_empty_holdings_400(self, tmp_path):
        client, _ = _build_client(tmp_path)
        r = client.post("/api/risk/factor_exposure", json={"holdings": []})
        assert r.status_code == 400

    def test_history_endpoint(self, tmp_path):
        client, monitor = _build_client(tmp_path)
        monitor.record_snapshot("2026-09-30", {"mom": 0.9})
        r = client.get("/api/risk/factor_exposure/history",
                       params={"factor": "mom"})
        assert r.status_code == 200
        data = r.json()["data"]
        assert any(row["factor"] == "mom" for row in data["history"])
        assert data["chart"][0]["date"] == "2026-09-30"

    def test_alerts_endpoint(self, tmp_path):
        client, monitor = _build_client(tmp_path)
        # 先触发一次超限
        monitor.check_limits({"mom": 1.2})
        r = client.get("/api/risk/factor_exposure/alerts")
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["count"] >= 1
        assert all(a["category"] == "factor_exposure_limit" for a in data["alerts"])

    def test_neutralize_endpoint(self, tmp_path):
        client, _ = _build_client(tmp_path)
        payload = {"holdings": HOLDINGS, "factor_scores": FACTOR_SCORES}
        r = client.post("/api/risk/factor_exposure/neutralize", json=payload)
        assert r.status_code == 200
        suggestions = r.json()["data"]["suggestions"]
        by_sym = {s["symbol"]: s for s in suggestions}
        assert by_sym["AAA"]["action"] == "decrease"
        assert by_sym["BBB"]["action"] == "increase"
