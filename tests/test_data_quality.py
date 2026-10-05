"""DataQualityMonitor 单元测试。

覆盖场景:
1. 正常数据通过所有检测 -> healthy
2. 连续3次API失败 -> CRITICAL
3. 滞后数据 -> WARNING
4. 缺失K线 -> WARNING
5. 价格跳变15% -> 标记异常但不告警
6. 价格跳变25% -> CRITICAL
7. 成交量为0 -> WARNING
8. 成交量为负 -> WARNING
9. 检测失败不阻断（异常输入）
10. record_fetch_result 成功后重置连续失败计数
11. get_anomaly_history 返回正确记录
12. alert_manager 为 None 时不崩溃
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pandas as pd
import pytest

from monitoring.alert import AlertLevel
from monitoring.data_quality import (
    ALERT_API_DISCONNECT,
    ALERT_LAG,
    ALERT_MISSING_KLINES,
    ALERT_PRICE_JUMP,
    ALERT_VOLUME_ANOMALY,
    DataQualityMonitor,
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def make_quote(
    price: float = 100.0,
    prev_close: float = 100.0,
    volume: float = 1_000_000,
    age_seconds: float = 0.0,
) -> dict:
    """构造一条"正常"行情快照，可按参数制造异常。"""
    ts = pd.Timestamp.now() - pd.Timedelta(seconds=age_seconds)
    return {
        "price": price,
        "last_price": price,
        "prev_close": prev_close,
        "volume": volume,
        "timestamp": ts.isoformat(),
    }


def make_klines(last_date: pd.Timestamp) -> pd.DataFrame:
    """构造日线 DataFrame，最后一行日期为 last_date。"""
    dates = pd.bdate_range(end=last_date, periods=20)
    df = pd.DataFrame(
        {
            "open": [10.0] * len(dates),
            "high": [10.5] * len(dates),
            "low": [9.5] * len(dates),
            "close": [10.2] * len(dates),
            "volume": [100000] * len(dates),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def make_monitor(**cfg) -> tuple[DataQualityMonitor, MagicMock]:
    """构造带 mock alert_manager 的 monitor。"""
    am = MagicMock()
    monitor = DataQualityMonitor(alert_manager=am, config=cfg or None)
    return monitor, am


# ---------------------------------------------------------------------------
# 1. 正常数据 healthy
# ---------------------------------------------------------------------------

class TestHealthy:
    def test_normal_data_healthy(self):
        monitor, am = make_monitor()
        quotes = {
            "600519.SH": make_quote(),
            "300750.SZ": make_quote(price=200.0, prev_close=199.0),
        }
        klines = {"600519.SH": make_klines(pd.Timestamp.now().normalize())}
        report = monitor.check(quotes, klines)

        assert report["overall_status"] == "healthy"
        for name, check in report["checks"].items():
            assert check["status"] == "ok", f"{name} 应为 ok"
        # 正常数据不应触发任何告警
        am.alert.assert_not_called()

    def test_get_latest_report(self):
        monitor, _ = make_monitor()
        assert monitor.get_latest_report() is None
        monitor.check({"600519.SH": make_quote()})
        report = monitor.get_latest_report()
        assert report is not None
        assert "overall_status" in report


# ---------------------------------------------------------------------------
# 2. 连续3次API失败 -> CRITICAL
# ---------------------------------------------------------------------------

class TestApiConnection:
    def test_three_consecutive_failures_critical(self):
        monitor, am = make_monitor(max_consecutive_failures=3)
        for _ in range(3):
            monitor.record_fetch_result("quantdash", success=False, symbol="600519.SH")

        report = monitor.check({})
        assert report["checks"]["api_connection"]["status"] == "critical"
        assert report["overall_status"] == "critical"

        # 应至少发送一次 CRITICAL 告警
        calls = [c for c in am.alert.call_args_list if c.args[0] == AlertLevel.CRITICAL]
        assert calls, "应触发 CRITICAL 告警"
        assert any(c.args[1] == ALERT_API_DISCONNECT for c in calls)

    def test_two_failures_not_critical(self):
        monitor, am = make_monitor(max_consecutive_failures=3)
        monitor.record_fetch_result("quantdash", success=False)
        monitor.record_fetch_result("quantdash", success=False)
        report = monitor.check({})
        # 未达阈值，不应 critical；存在失败 -> warning
        assert report["checks"]["api_connection"]["status"] == "warning"
        am.alert.assert_not_called()

    # 10. 成功后重置计数
    def test_success_resets_consecutive_count(self):
        monitor, am = make_monitor(max_consecutive_failures=3)
        monitor.record_fetch_result("quantdash", success=False)
        monitor.record_fetch_result("quantdash", success=False)
        # 成功一次，重置
        monitor.record_fetch_result("quantdash", success=True)
        # 再失败2次，不应达到阈值
        monitor.record_fetch_result("quantdash", success=False)
        monitor.record_fetch_result("quantdash", success=False)

        report = monitor.check({})
        assert report["checks"]["api_connection"]["status"] == "warning"
        am.alert.assert_not_called()
        assert monitor._consecutive_failures["quantdash"] == 2


# ---------------------------------------------------------------------------
# 3. 滞后数据 -> WARNING
# ---------------------------------------------------------------------------

class TestDataLag:
    def test_stale_quote_triggers_warning(self):
        monitor, am = make_monitor(lag_threshold_seconds=300)
        quotes = {"600519.SH": make_quote(age_seconds=600)}  # 10分钟前
        report = monitor.check(quotes)

        assert report["checks"]["data_lag"]["status"] == "warning"
        assert report["overall_status"] == "warning"
        am.alert.assert_called()
        assert any(
            c.args[1] == ALERT_LAG for c in am.alert.call_args_list
        )

    def test_fresh_quote_ok(self):
        monitor, _ = make_monitor(lag_threshold_seconds=300)
        quotes = {"600519.SH": make_quote(age_seconds=10)}
        report = monitor.check(quotes)
        assert report["checks"]["data_lag"]["status"] == "ok"


# ---------------------------------------------------------------------------
# 4. 缺失K线 -> WARNING
# ---------------------------------------------------------------------------

class TestMissingKlines:
    def test_missing_klines_triggers_warning(self):
        monitor, am = make_monitor()
        # 最后K线日期为 10 个工作日前
        old_date = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=15)[0]
        klines = {"600519.SH": make_klines(old_date)}
        report = monitor.check({}, klines)

        assert report["checks"]["missing_klines"]["status"] == "warning"
        assert report["overall_status"] == "warning"
        assert any(
            c.args[1] == ALERT_MISSING_KLINES for c in am.alert.call_args_list
        )

    def test_fresh_klines_ok(self):
        monitor, _ = make_monitor()
        today_df = make_klines(pd.Timestamp.now().normalize())
        report = monitor.check({}, {"600519.SH": today_df})
        assert report["checks"]["missing_klines"]["status"] == "ok"


# ---------------------------------------------------------------------------
# 5/6. 价格跳变
# ---------------------------------------------------------------------------

class TestPriceJump:
    def test_price_jump_15_flagged_not_alerted(self):
        monitor, am = make_monitor()
        # 15% 跳变，介于 10% 和 20% 之间
        quotes = {"600519.SH": make_quote(price=115.0, prev_close=100.0)}
        report = monitor.check(quotes)

        check = report["checks"]["price_jump"]
        assert check["status"] == "warning"
        # 有异常记录
        assert any(a["type"] == ALERT_PRICE_JUMP for a in report["anomalies"])
        # 但不应发送 critical 告警（可能是真实涨跌停）
        critical_calls = [
            c for c in am.alert.call_args_list if c.args[0] == AlertLevel.CRITICAL
        ]
        assert not critical_calls

    def test_price_jump_25_critical(self):
        monitor, am = make_monitor()
        quotes = {"600519.SH": make_quote(price=125.0, prev_close=100.0)}
        report = monitor.check(quotes)

        assert report["checks"]["price_jump"]["status"] == "critical"
        assert report["overall_status"] == "critical"
        critical_calls = [
            c for c in am.alert.call_args_list
            if c.args[0] == AlertLevel.CRITICAL and c.args[1] == ALERT_PRICE_JUMP
        ]
        assert critical_calls


# ---------------------------------------------------------------------------
# 7/8. 成交量异常
# ---------------------------------------------------------------------------

class TestVolumeAnomaly:
    def test_zero_volume_warning(self):
        monitor, am = make_monitor()
        quotes = {"600519.SH": make_quote(volume=0)}
        report = monitor.check(quotes)

        assert report["checks"]["volume_anomaly"]["status"] == "warning"
        assert report["overall_status"] == "warning"
        assert any(
            c.args[1] == ALERT_VOLUME_ANOMALY for c in am.alert.call_args_list
        )

    def test_negative_volume_warning(self):
        monitor, am = make_monitor()
        quotes = {"600519.SH": make_quote(volume=-100)}
        report = monitor.check(quotes)

        assert report["checks"]["volume_anomaly"]["status"] == "warning"
        assert any(
            c.args[1] == ALERT_VOLUME_ANOMALY for c in am.alert.call_args_list
        )


# ---------------------------------------------------------------------------
# 9. 检测失败不阻断
# ---------------------------------------------------------------------------

class TestResilience:
    def test_malformed_input_does_not_raise(self):
        monitor, _ = make_monitor()
        # 各种异常输入都不应抛异常
        bad_inputs = [
            ({}, {}),
            ({"sym": None}, None),
            ({"sym": {"price": "abc", "prev_close": None}}, None),
            ({"sym": make_quote()}, {"sym": "not-a-dataframe"}),
            ({"sym": {"timestamp": "garbage"}}, None),
            ({"sym": {"price": 100, "prev_close": 0, "volume": "xx"}}, None),
        ]
        for quotes, klines in bad_inputs:
            report = monitor.check(quotes, klines)
            assert report["overall_status"] in ("healthy", "warning", "critical")

    def test_check_with_empty_quotes(self):
        monitor, _ = make_monitor()
        report = monitor.check({}, {})
        assert report["overall_status"] == "healthy"


# ---------------------------------------------------------------------------
# 11. 异常历史
# ---------------------------------------------------------------------------

class TestAnomalyHistory:
    def test_anomaly_history_records(self):
        monitor, _ = make_monitor()
        # 制造两个异常
        monitor.check({"A": make_quote(volume=0)})
        monitor.check({"B": make_quote(price=130.0, prev_close=100.0)})

        history = monitor.get_anomaly_history()
        assert len(history) >= 2
        # 最新在前
        assert history[0]["symbol"] == "B"
        # 倒序返回
        assert history[0]["timestamp"] >= history[-1]["timestamp"]

    def test_anomaly_history_limit(self):
        monitor, _ = make_monitor(max_history=5)
        for _ in range(10):
            monitor.check({"A": make_quote(volume=0)})
        # 内部存储被裁剪
        assert len(monitor._anomaly_history) <= 5


# ---------------------------------------------------------------------------
# 12. alert_manager 为 None
# ---------------------------------------------------------------------------

class TestNoAlertManager:
    def test_none_alert_manager_no_crash(self):
        monitor = DataQualityMonitor(alert_manager=None, config={})
        # 连续失败、各类异常都不应崩溃
        for _ in range(5):
            monitor.record_fetch_result("quantdash", success=False)
        report = monitor.check(
            {
                "A": make_quote(volume=0),
                "B": make_quote(price=130.0, prev_close=100.0),
                "C": make_quote(age_seconds=999),
            }
        )
        assert report["overall_status"] in ("warning", "critical")
        # 异常仍被记录
        assert len(monitor.get_anomaly_history()) > 0

    def test_epoch_timestamp_parsing(self):
        monitor, _ = make_monitor()
        # epoch 秒级时间戳（当前时间）
        quotes = {"A": {"price": 10.0, "prev_close": 10.0, "volume": 100,
                        "timestamp": time.time()}}
        report = monitor.check(quotes)
        assert report["checks"]["data_lag"]["status"] == "ok"
