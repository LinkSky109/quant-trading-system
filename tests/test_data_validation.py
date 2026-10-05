"""数据校验与对账（REQ-P2-09）单元测试 + API 集成测试。

核心校验逻辑全部使用手工构造的 DataFrame / dict，不依赖网络；
API 层使用 FastAPI + TestClient（manager 传 None/mock），SQLite 用 tmp_path。

覆盖：
1. 价格交叉验证（超阈值 / 未超阈值，手算偏差率）
2. 价格跳变（美股 +25% 阈值 0.20；A股 +15% 阈值 0.10；美股 +15% 不触发）
3. 成交量异常（5 倍放量 / 1/10 地量）
4. 缺失K线（删掉中间两个工作日）
5. OHLC 逻辑错误（high<low、负价）
6. 血缘记录与状态流转
7. 自动修复（备用数据成功 / 备用也缺保持 open）
8. 质量评分（手算分项与总分）
9. 每日报告结构
10. 全部 API 端点（正常 + 404/400）
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web-dashboard"))

from data.validation import DataReconciler  # noqa: E402


# ---------------------------------------------------------------------------
# 公共夹具 / 工具
# ---------------------------------------------------------------------------


@pytest.fixture()
def rc(tmp_path) -> DataReconciler:
    """基于临时 SQLite 的 DataReconciler（无 fetcher）。"""
    db = tmp_path / "validation_test.db"
    return DataReconciler(db_path=str(db))


def make_klines(
    closes,
    volumes=None,
    start="2024-01-01",
    drop_dates=None,
) -> pd.DataFrame:
    """手工构造合法 OHLCV K线 DataFrame（索引为工作日）。

    Args:
        closes: 收盘价序列（列表）。
        volumes: 成交量序列；缺省全部 1000。
        start: 起始日期。
        drop_dates: 需要从结果索引中删除的日期列表（模拟缺失K线）。
    """
    n = len(closes)
    dates = pd.bdate_range(start=start, periods=n)
    if volumes is None:
        volumes = [1000] * n
    df = pd.DataFrame(
        {
            "open": list(closes),
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": list(closes),
            "volume": list(volumes),
        },
        index=dates,
    )
    df.index.name = "date"
    if drop_dates:
        df = df.drop(pd.to_datetime(drop_dates))
    return df


# ---------------------------------------------------------------------------
# 1. 价格交叉验证
# ---------------------------------------------------------------------------


class TestReconcilePrices:
    def test_under_tolerance_validated(self, rc):
        # p1=100, p2=100.4 -> 偏差率 = 0.4/100 = 0.004 < 0.005
        res = rc.reconcile_prices("600519.SH", {"quantdash": 100.0, "tencent": 100.4})
        assert res["status"] == "validated"
        assert res["deviation"] == pytest.approx(0.004, abs=1e-6)
        assert "anomaly_id" not in res

    def test_over_tolerance_flagged(self, rc):
        # p1=100, p2=101 -> 偏差率 = 1.0/100 = 0.01 > 0.005
        res = rc.reconcile_prices("600519.SH", {"quantdash": 100.0, "tencent": 101.0})
        assert res["status"] == "flagged"
        assert res["deviation"] == pytest.approx(0.01, abs=1e-6)
        anom = next(a for a in rc.get_anomalies() if a["id"] == res["anomaly_id"])
        assert anom["type"] == "price_mismatch"
        assert anom["status"] == "open"

    def test_custom_tolerance(self, rc):
        # 偏差 0.004，阈值 0.002 -> flagged
        res = rc.reconcile_prices(
            "AAPL.US", {"quantdash": 100.0, "tencent": 100.4}, tolerance=0.002
        )
        assert res["status"] == "flagged"

    def test_no_sources_no_fetcher_unavailable(self, rc):
        res = rc.reconcile_prices("AAPL.US")
        assert res["status"] == "unavailable"

    def test_insufficient_sources(self, rc):
        res = rc.reconcile_prices("AAPL.US", {"quantdash": 100.0})
        assert res["status"] == "insufficient"


# ---------------------------------------------------------------------------
# 2. 价格跳变
# ---------------------------------------------------------------------------


class TestJumpAnomaly:
    def test_us_big_jump_flagged(self, rc):
        df = make_klines([100.0, 125.0, 125.0])  # +25%
        anoms = rc.detect_anomalies("AAPL.US", df)
        jumps = [a for a in anoms if a.type == "price_jump"]
        assert len(jumps) == 1
        assert "0.25" in jumps[0].detail or "0.2500" in jumps[0].detail

    def test_a_share_jump_threshold_010(self, rc):
        # A股 +15%，自动阈值 0.10 -> 触发
        df = make_klines([100.0, 115.0, 115.0])
        anoms = rc.detect_anomalies("000001.SZ", df)
        jumps = [a for a in anoms if a.type == "price_jump"]
        assert len(jumps) == 1

    def test_us_15pct_not_triggered(self, rc):
        # 美股阈值 0.20，+15% 不触发
        df = make_klines([100.0, 115.0, 115.0])
        anoms = rc.detect_anomalies("AAPL.US", df)
        jumps = [a for a in anoms if a.type == "price_jump"]
        assert jumps == []

    def test_explicit_jump_threshold_override(self, rc):
        df = make_klines([100.0, 108.0, 108.0])  # +8%
        anoms = rc.detect_anomalies("AAPL.US", df, jump_threshold=0.05)
        jumps = [a for a in anoms if a.type == "price_jump"]
        assert len(jumps) == 1


# ---------------------------------------------------------------------------
# 3. 成交量异常
# ---------------------------------------------------------------------------


class TestVolumeAnomaly:
    def test_surge_and_dry(self, rc):
        # 前 20 日基准量 1000；第 21 日放量到 10000（>5倍均量）；第 22 日地量 50
        volumes = [1000] * 20 + [10000, 50]
        df = make_klines(closes=[100.0] * 22, volumes=volumes)
        anoms = rc.detect_anomalies("600519.SH", df)
        surge = [a for a in anoms if a.type == "volume_surge"]
        dry = [a for a in anoms if a.type == "volume_dry"]
        assert len(surge) == 1
        assert len(dry) == 1
        assert surge[0].severity == "medium"
        assert dry[0].severity == "low"

    def test_normal_volume_no_anomaly(self, rc):
        volumes = [1000] * 22
        df = make_klines(closes=[100.0] * 22, volumes=volumes)
        anoms = rc.detect_anomalies("600519.SH", df)
        vol = [a for a in anoms if a.type in ("volume_surge", "volume_dry")]
        assert vol == []


# ---------------------------------------------------------------------------
# 4. 缺失K线
# ---------------------------------------------------------------------------


class TestMissingKline:
    def test_missing_middle_workdays(self, rc):
        dates = pd.bdate_range(start="2024-01-01", periods=6)
        drop = [str(dates[2].date()), str(dates[4].date())]
        df = make_klines([100.0] * 6, drop_dates=drop)
        anoms = rc.detect_anomalies("600519.SH", df)
        missing = [a for a in anoms if a.type == "missing_kline"]
        missing_dates = {a.date for a in missing}
        assert str(dates[2].date()) in missing_dates
        assert str(dates[4].date()) in missing_dates


# ---------------------------------------------------------------------------
# 5. OHLC 逻辑错误
# ---------------------------------------------------------------------------


class TestOHLCAnomaly:
    def test_high_below_low(self, rc):
        df = make_klines([100.0, 100.0])
        df.iloc[1, df.columns.get_loc("high")] = 90.0  # high<low
        anoms = rc.detect_anomalies("600519.SH", df)
        ohlc = [a for a in anoms if a.type == "ohlc_logic"]
        assert len(ohlc) == 1
        assert "high" in ohlc[0].detail

    def test_negative_price(self, rc):
        df = make_klines([100.0, -100.0])
        anoms = rc.detect_anomalies("600519.SH", df)
        ohlc = [a for a in anoms if a.type == "ohlc_logic"]
        assert any("非正" in a.detail for a in ohlc)


# ---------------------------------------------------------------------------
# 6. 血缘记录
# ---------------------------------------------------------------------------


class TestLineage:
    def test_record_and_query(self, rc):
        rc.record_lineage("600519.SH", source="quantdash", status="validated")
        rc.record_lineage("600519.SH", source="tencent", status="flagged")
        rc.record_lineage("AAPL.US", source="quantdash", status="validated")
        all_recs = rc.get_lineage()
        assert len(all_recs) == 3
        only_sh = rc.get_lineage("600519.SH")
        assert len(only_sh) == 2
        statuses = {r["status"] for r in only_sh}
        assert statuses == {"validated", "flagged"}


# ---------------------------------------------------------------------------
# 7. 自动修复
# ---------------------------------------------------------------------------


class TestFixAnomaly:
    def test_fix_with_backup_data(self, rc):
        rec = rc._add_anomaly(
            symbol="600519.SH", atype="price_mismatch", severity="high",
            date="", detail="偏差超阈值",
        )
        result = rc.fix_anomaly(rec.id, backup_data={"corrected_value": 1688.4})
        assert result["status"] == "fixed"
        # 异常状态流转
        updated = rc._anomalies[rec.id]
        assert updated.status == "fixed"
        assert len(updated.fix_history) == 1
        assert updated.fix_history[0]["after"] == 1688.4
        # 血缘出现 corrected
        corrected = [r for r in rc.get_lineage("600519.SH") if r["status"] == "corrected"]
        assert len(corrected) == 1

    def test_fix_no_backup_stays_open(self, rc):
        # 无 fetcher、无 backup_data -> 无法修复，保持 open
        rec = rc._add_anomaly(
            symbol="600519.SH", atype="missing_kline", severity="medium",
            date="2024-01-03", detail="缺K线",
        )
        result = rc.fix_anomaly(rec.id)
        assert result["status"] == "open"
        assert rc._anomalies[rec.id].status == "open"

    def test_fix_not_found_raises(self, rc):
        with pytest.raises(KeyError):
            rc.fix_anomaly("ANNOTEXIST")


# ---------------------------------------------------------------------------
# 8. 质量评分
# ---------------------------------------------------------------------------


class TestQualityScore:
    def test_hand_calculated_score(self, rc):
        # completeness=0.9, anomaly=2/10, timeliness=0.8, weights 0.4/0.4/0.2
        # 总分 = 100*(0.4*0.9 + 0.4*(1-0.2) + 0.2*0.8)
        #      = 100*(0.36 + 0.32 + 0.16) = 84.0
        res = rc.quality_score(
            "600519.SH",
            completeness=0.9,
            anomaly_count=2,
            total_records=10,
            timely_ratio=0.8,
        )
        assert res["completeness"] == pytest.approx(0.9)
        assert res["accuracy"] == pytest.approx(0.8)
        assert res["timeliness"] == pytest.approx(0.8)
        assert res["total"] == pytest.approx(84.0)
        assert 0 <= res["total"] <= 100

    def test_score_from_df_completeness(self, rc):
        df = make_klines([100.0] * 10)  # 10 个连续工作日，无缺失
        res = rc.quality_score("600519.SH", df=df)
        assert res["completeness"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 9. 每日报告
# ---------------------------------------------------------------------------


class TestDailyReport:
    def test_report_structure(self, rc):
        rc._add_anomaly("600519.SH", "price_jump", "high", "2024-01-02", "跳变")
        report = rc.daily_quality_report(["600519.SH", "AAPL.US"])
        assert set(report.keys()) >= {
            "date", "symbols", "per_symbol", "anomaly_summary", "fix_summary",
        }
        assert report["symbols"] == ["600519.SH", "AAPL.US"]
        assert "600519.SH" in report["per_symbol"]
        assert report["anomaly_summary"]["total_open"] == 1
        assert report["anomaly_summary"]["by_type"]["price_jump"] == 1


# ---------------------------------------------------------------------------
# 10. API 端点（TestClient）
# ---------------------------------------------------------------------------


def _build_client(reconciler: DataReconciler) -> TestClient:
    """用注入的 reconciler 构造测试客户端。"""
    from _routes_data import register_data_routes

    app = FastAPI()

    def ok(data=None, message="success"):
        return {"code": 0, "message": message, "data": data}

    def err(code, message, http_status=400):
        return JSONResponse(status_code=http_status,
                            content={"code": code, "message": message, "data": None})

    register_data_routes(app, None, set(), ok, err, reconciler=reconciler)
    return TestClient(app)


class TestDataAPI:
    def test_reconcile_endpoint_ok(self, tmp_path):
        fetcher = SimpleNamespace(
            get_klines=lambda sym, count=120: make_klines([100.0] * 10),
            get_realtime_quote=lambda sym: {"price": 100.0, "source": "quantdash"},
        )
        db = tmp_path / "api.db"
        rc = DataReconciler(fetcher=fetcher, db_path=str(db))
        client = _build_client(rc)

        r = client.post("/api/data/reconcile", json={
            "symbols": ["AAPL.US"],
            "sources": {"AAPL.US": {"quantdash": 100.0, "tencent": 100.2}},
        })
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        entry = body["data"]["per_symbol"]["AAPL.US"]
        assert entry["price_check"]["status"] == "validated"
        assert "anomalies" in entry

    def test_reconcile_empty_symbols_400(self, tmp_path):
        db = tmp_path / "api.db"
        rc = DataReconciler(db_path=str(db))
        client = _build_client(rc)
        r = client.post("/api/data/reconcile", json={"symbols": []})
        assert r.status_code == 400  # err() 默认 http_status=400
        assert r.json()["code"] == 40001

    def test_anomalies_list_and_filter(self, tmp_path):
        db = tmp_path / "api.db"
        rc = DataReconciler(db_path=str(db))
        rc._add_anomaly("AAPL.US", "price_jump", "high", "2024-01-02", "x")
        client = _build_client(rc)
        r = client.get("/api/data/anomalies")
        assert r.status_code == 200
        assert len(r.json()["data"]) == 1
        r2 = client.get("/api/data/anomalies", params={"status": "open"})
        assert len(r2.json()["data"]) == 1
        r3 = client.get("/api/data/anomalies", params={"status": "fixed"})
        assert r3.json()["data"] == []

    def test_fix_endpoint_ok_and_404(self, tmp_path):
        db = tmp_path / "api.db"
        rc = DataReconciler(db_path=str(db))
        rec = rc._add_anomaly("AAPL.US", "price_mismatch", "high", "", "偏差")
        client = _build_client(rc)

        # 404：不存在的异常
        r404 = client.post("/api/data/anomalies/ANNOEXIST/fix")
        assert r404.status_code == 404
        assert r404.json()["code"] == 40401

        # 400：存在但无法自动修复（无 fetcher、无 body 备用数据）
        r400 = client.post(f"/api/data/anomalies/{rec.id}/fix")
        assert r400.status_code == 400
        assert r400.json()["code"] == 40002

    def test_quality_endpoint(self, tmp_path):
        db = tmp_path / "api.db"
        rc = DataReconciler(db_path=str(db))
        rc._add_anomaly("AAPL.US", "price_jump", "high", "2024-01-02", "x")
        client = _build_client(rc)
        r = client.get("/api/data/quality")
        assert r.status_code == 200
        data = r.json()["data"]
        assert "scores" in data
        assert "report" in data
        assert "AAPL.US" in data["scores"]

    def test_quality_endpoint_with_symbol_param(self, tmp_path):
        db = tmp_path / "api.db"
        rc = DataReconciler(db_path=str(db))
        client = _build_client(rc)
        r = client.get("/api/data/quality", params={"symbol": "AAPL.US"})
        assert r.status_code == 200
        assert r.json()["data"]["symbols"] == ["AAPL.US"]
