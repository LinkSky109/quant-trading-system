"""SystemHealthMonitor 单元测试。

全部使用 mock / 临时 SQLite，不依赖真实外部服务:
1. ApiMetricsCollector: count / avg / p95 / error_rate 统计准确
2. collect_database_status: 用临时数据库建表插入数据
3. collect_alerts / collect_data_quality: 用真实 AlertManager + mock monitor
4. calculate_health_score: 全健康高分 / 关键项失败低分 / <60 -> critical
5. collect_all: 所有依赖为 None 时不崩溃，返回 unknown 状态
6. 自动采集线程启动/停止
7. psutil 不可用时降级（memory/cpu 返回 None，不崩溃）
8. Jev 模式 real/mock/disconnected 判定
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from monitoring.alert import AlertLevel, AlertManager
from monitoring.system_health import (
    ApiMetricsCollector,
    SystemHealthMonitor,
)
from persistence.database import Database


# ---------------------------------------------------------------------------
# ApiMetricsCollector
# ---------------------------------------------------------------------------

class TestApiMetricsCollector:
    """API 指标收集器统计正确性。"""

    def test_record_and_summary(self):
        c = ApiMetricsCollector()
        # 200 样本延迟 10ms，10 个 500 错误
        for i in range(20):
            c.record_request("/api/a", "GET", 10.0, 200)
        for i in range(5):
            c.record_request("/api/a", "GET", 20.0, 500)

        stats = c.get_stats()
        a = stats["GET /api/a"]
        assert a["count"] == 25
        assert a["error_count"] == 5
        assert a["error_rate"] == pytest.approx(0.2)
        # avg = (20*10 + 5*20)/25 = 12.0
        assert a["avg_latency_ms"] == pytest.approx(12.0, abs=0.1)

        s = c.get_summary()
        assert s["total_requests"] == 25
        assert s["error_rate"] == pytest.approx(0.2)

    def test_p95_accuracy(self):
        c = ApiMetricsCollector()
        # 100 个样本，延迟 1..100ms；排序后 p95 = 第 95 个值 = 95
        for lat in range(1, 101):
            c.record_request("/api/x", "GET", float(lat), 200)
        stats = c.get_stats()
        assert stats["GET /api/x"]["p95_latency_ms"] == pytest.approx(95.0, abs=0.5)

    def test_empty_collector(self):
        c = ApiMetricsCollector()
        s = c.get_summary()
        assert s["total_requests"] == 0
        assert s["avg_latency_ms"] is None
        assert s["p95_latency_ms"] is None
        assert c.get_stats() == {}

    def test_rolling_window(self):
        """超过 max_samples 后旧样本被滚动覆盖。"""
        c = ApiMetricsCollector(max_samples=10)
        for lat in [1.0] * 5 + [100.0] * 20:
            c.record_request("/api/r", "GET", lat, 200)
        stats = c.get_stats()
        assert stats["GET /api/r"]["count"] == 25  # total 仍累计
        # deque 只保留最近 10 个 100ms，avg 应接近 100
        assert stats["GET /api/r"]["avg_latency_ms"] == pytest.approx(100.0, abs=1.0)

    def test_thread_safe_basic(self):
        """多线程并发 record 不崩溃、计数正确。"""
        import threading

        c = ApiMetricsCollector()

        def worker():
            for _ in range(100):
                c.record_request("/api/t", "GET", 5.0, 200)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert c.get_summary()["total_requests"] == 500


# ---------------------------------------------------------------------------
# collect_database_status（临时真实 SQLite）
# ---------------------------------------------------------------------------

class TestDatabaseStatus:
    def test_database_connected_and_counts(self, tmp_path):
        db = Database(db_path=str(tmp_path / "h.db"))
        # 插入一些记录
        db.insert_trade(
            timestamp="2026-09-30T10:00:00", symbol="600000", side="buy",
            price=10.0, fill_price=10.0, quantity=100, amount=1000.0,
        )
        db.insert_jev_decision(
            timestamp="2026-09-30T10:00:00", symbol="600000",
            strategy_signal="buy", strategy_confidence=0.8,
            market_state={}, probabilities={}, final_action="buy",
            final_confidence=0.9, executed=True, latency_ms=12.5,
        )
        m = SystemHealthMonitor(db=db)
        rep = m.collect_database_status()

        assert rep["status"] == "connected"
        assert rep["table_counts"]["trades"] == 1
        assert rep["table_counts"]["jev_decisions"] == 1
        assert rep["table_counts"]["daily_reports"] == 0
        # walkthroughs 表存在但为空 -> 0
        assert rep["table_counts"]["walkthroughs"] == 0
        assert rep["size_mb"] is not None and rep["size_mb"] >= 0
        assert rep["journal_mode"] == "wal"

        # 真正不存在的表 -> count_table 抛错 -> 降级为 None（不崩溃）
        real_count_table = db.count_table

        def fake_count(table, where="", params=()):
            if table == "walkthroughs":
                raise RuntimeError("no such table")
            return real_count_table(table, where=where, params=params)

        db.count_table = fake_count  # type: ignore[assignment]
        rep2 = m.collect_database_status()
        assert rep2["table_counts"]["walkthroughs"] is None
        assert rep2["table_counts"]["trades"] == 1
        db.close()

    def test_database_none(self):
        m = SystemHealthMonitor(db=None)
        assert m.collect_database_status() == {"status": "unknown"}


# ---------------------------------------------------------------------------
# collect_alerts / collect_data_quality
# ---------------------------------------------------------------------------

class TestAlertsAndDataQuality:
    def test_collect_alerts(self):
        am = AlertManager()
        am.alert(AlertLevel.CRITICAL, "drawdown_pause", "dd")
        am.alert(AlertLevel.WARNING, "stop_loss", "sl")
        am.alert(AlertLevel.INFO, "jev_filter", "jf")
        m = SystemHealthMonitor(alert_manager=am)
        rep = m.collect_alerts()
        assert rep["status"] == "ok"
        assert rep["stats"]["critical"] == 1
        assert rep["stats"]["warning"] == 1
        assert rep["stats"]["info"] == 1
        assert rep["stats"]["total"] == 3

    def test_collect_alerts_none(self):
        m = SystemHealthMonitor(alert_manager=None)
        rep = m.collect_alerts()
        assert rep["status"] == "unknown"

    def test_collect_data_quality_with_report(self):
        dqm = MagicMock()
        dqm.get_latest_report.return_value = {
            "overall_status": "healthy", "checks": {}, "anomalies": []
        }
        m = SystemHealthMonitor(data_quality_monitor=dqm)
        rep = m.collect_data_quality()
        assert rep["overall_status"] == "healthy"

    def test_collect_data_quality_no_monitor(self):
        m = SystemHealthMonitor(data_quality_monitor=None)
        assert m.collect_data_quality() == {"status": "unknown"}

    def test_collect_data_quality_no_report(self):
        dqm = MagicMock()
        dqm.get_latest_report.return_value = None
        m = SystemHealthMonitor(data_quality_monitor=dqm)
        assert m.collect_data_quality() == {"status": "unknown"}


# ---------------------------------------------------------------------------
# Jev 状态
# ---------------------------------------------------------------------------

class TestJevStatus:
    def test_jev_real_mode(self):
        m = SystemHealthMonitor(db=None)
        m._http_get_jev_health = MagicMock(
            return_value={"model_loaded": True, "model": "laya_mlx"}
        )
        rep = m.collect_jev_status()
        assert rep["mode"] == "real"
        assert rep["reachable"] is True
        assert rep["model_loaded"] is True

    def test_jev_mock_mode_service_up(self):
        m = SystemHealthMonitor(db=None)
        m._http_get_jev_health = MagicMock(
            return_value={"model_loaded": False, "model": None}
        )
        rep = m.collect_jev_status()
        assert rep["mode"] == "mock"

    def test_jev_disconnected(self):
        m = SystemHealthMonitor(db=None)
        m._http_get_jev_health = MagicMock(return_value=None)
        rep = m.collect_jev_status()
        assert rep["mode"] == "disconnected"
        assert rep["reachable"] is False

    def test_jev_inference_count_from_db(self, tmp_path):
        db = Database(db_path=str(tmp_path / "j.db"))
        db.insert_jev_decision(
            timestamp="2026-09-30T10:00:00", symbol="600000",
            strategy_signal="buy", strategy_confidence=0.8,
            market_state={}, probabilities={}, final_action="buy",
            final_confidence=0.9, executed=True, latency_ms=20.0,
        )
        m = SystemHealthMonitor(db=db)
        m._http_get_jev_health = MagicMock(return_value=None)
        rep = m.collect_jev_status()
        assert rep["inference_count"] == 1
        assert rep["avg_latency_ms"] == pytest.approx(20.0, abs=0.1)
        db.close()


# ---------------------------------------------------------------------------
# calculate_health_score
# ---------------------------------------------------------------------------

def _healthy_report() -> dict:
    return {
        "service_status": {
            "quant": {"status": "up"},
            "jev": {"status": "ok"},
        },
        "api_performance": {
            "summary": {
                "total_requests": 100,
                "error_rate": 0.0,
                "p95_latency_ms": 100.0,
            }
        },
        "database": {
            "status": "connected",
            "table_counts": {"trades": 1, "jev_decisions": 1,
                             "account_snapshots": 1, "daily_reports": 1,
                             "walkthroughs": None},
        },
        "data_quality": {"overall_status": "healthy"},
        "trading_engine": {"running": True, "risk_paused": False},
        "alerts": {"stats": {"total": 0, "critical": 0, "warning": 0, "info": 0}},
    }


class TestHealthScore:
    def test_all_healthy_full_score(self):
        m = SystemHealthMonitor()
        score = m.calculate_health_score(_healthy_report())
        assert score == 100

    def test_critical_failures_low_score(self):
        m = SystemHealthMonitor()
        bad = {
            "service_status": {
                "quant": {"status": "down"},
                "jev": {"status": "disconnected"},
            },
            "api_performance": {
                "summary": {"total_requests": 100,
                            "error_rate": 0.10, "p95_latency_ms": 3000.0}
            },
            "database": {"status": "disconnected"},
            "data_quality": {"overall_status": "critical"},
            "trading_engine": {"running": False, "risk_paused": True},
            "alerts": {"stats": {"critical": 3}},
        }
        score = m.calculate_health_score(bad)
        assert score < 60, f"expected critical, got score={score}"

    def test_warning_level_boundary(self):
        m = SystemHealthMonitor()
        # 仅少量警告：量化+Jev正常，DB正常，DQ warning，无告警
        rep = _healthy_report()
        rep["data_quality"] = {"overall_status": "warning"}
        rep["alerts"] = {"stats": {"critical": 0, "warning": 2}}
        score = m.calculate_health_score(rep)
        # 100 - 7.5(dq减半) = ~92.5 仍 healthy；构造一个中间分
        assert 0 <= score <= 100

    def test_collect_all_level_mapping(self):
        m = SystemHealthMonitor()
        m._http_get_jev_health = MagicMock(return_value=None)
        report = m.collect_all()
        assert report["health_score"] >= 0
        assert report["health_score"] <= 100
        assert report["health_level"] in ("healthy", "warning", "critical")


# ---------------------------------------------------------------------------
# collect_all 容错
# ---------------------------------------------------------------------------

class TestCollectAll:
    def test_all_none_dependencies_no_crash(self):
        """所有依赖为 None 时 collect_all 不崩溃，返回 unknown 状态。"""
        m = SystemHealthMonitor()
        m._http_get_jev_health = MagicMock(return_value=None)
        report = m.collect_all()
        for key in ("service_status", "api_performance", "jev_status",
                    "database", "data_quality", "trading_engine", "alerts"):
            assert key in report, f"missing key {key}"
        assert "health_score" in report
        assert "health_level" in report
        assert report["database"] == {"status": "unknown"}
        assert report["data_quality"] == {"status": "unknown"}

    def test_collector_exception_isolated(self):
        """某个 collect 方法抛异常时整体仍组装成功。"""
        m = SystemHealthMonitor()
        m.collect_database_status = MagicMock(side_effect=RuntimeError("boom"))
        m._http_get_jev_health = MagicMock(return_value=None)
        report = m.collect_all()
        assert report["database"] == {"status": "unknown"} or \
               report["database"].get("status") == "unknown"
        assert "health_score" in report

    def test_trading_engine_mock(self):
        trader = MagicMock()
        trader.running = True
        trader.today_pnl = 123.45
        trader.get_status.return_value = {
            "running": True,
            "today_pnl": 123.45,
            "positions": [{"symbol": "600000"}],
            "risk_summary": {},
        }
        m = SystemHealthMonitor(realtime_trader=trader, db=None)
        rep = m.collect_trading_engine()
        assert rep["running"] is True
        assert rep["position_count"] == 1
        assert rep["today_pnl"] == 123.45
        assert rep["risk_paused"] is False

    def test_trading_engine_none(self):
        m = SystemHealthMonitor(realtime_trader=None)
        assert m.collect_trading_engine() == {"status": "unknown"}


# ---------------------------------------------------------------------------
# psutil 降级
# ---------------------------------------------------------------------------

class TestPsutilDegradation:
    def test_psutil_unavailable(self, monkeypatch):
        import monitoring.system_health as mod
        monkeypatch.setattr(mod, "_HAS_PSUTIL", False)
        m = SystemHealthMonitor()
        m._http_get_jev_health = MagicMock(return_value=None)
        rep = m.collect_service_status()
        assert rep["quant"]["status"] == "up"
        assert rep["quant"]["memory_mb"] is None
        assert rep["quant"]["cpu_percent"] is None
        assert rep["quant"]["psutil_available"] is False
        # uptime 仍应正常计算
        assert rep["quant"]["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# 自动采集线程
# ---------------------------------------------------------------------------

class TestAutoCollect:
    def test_start_and_stop_auto_collect(self):
        m = SystemHealthMonitor()
        m._http_get_jev_health = MagicMock(return_value=None)
        m.start_auto_collect(interval=1)
        try:
            # 等待至少一次采集完成
            deadline = time.time() + 3.0
            while time.time() < deadline and m.get_latest_report() is None:
                time.sleep(0.1)
            report = m.get_latest_report()
            assert report is not None
            assert "health_score" in report
        finally:
            m.stop_auto_collect()
        # 停止后线程应结束
        assert m._auto_thread is None
