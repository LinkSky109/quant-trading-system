"""告警管理器单元测试。

覆盖: 告警创建/级别/便捷方法/Webhook推送/冷却去重/历史查询/统计/降级。
使用 mock 替代真实 HTTP 请求。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from monitoring.alert import Alert, AlertLevel, AlertManager


# ---------------------------------------------------------------------------
# Alert 数据类
# ---------------------------------------------------------------------------

class TestAlert:
    def test_create_alert(self):
        a = Alert(level="WARNING", category="stop_loss", symbol="600519.SH", message="止损")
        assert a.level == "WARNING"
        assert a.category == "stop_loss"
        assert a.timestamp > 0

    def test_to_dict(self):
        a = Alert(level="CRITICAL", category="drawdown", symbol=None, message="回撤超标", current_value=0.15)
        d = a.to_dict()
        assert d["level"] == "CRITICAL"
        assert d["current_value"] == 0.15
        assert "timestamp_iso" in d


# ---------------------------------------------------------------------------
# AlertManager 基础
# ---------------------------------------------------------------------------

class TestAlertManagerBasic:
    def test_init_no_webhook(self):
        am = AlertManager()
        assert am.webhook_url == ""
        assert am.stats["webhook_enabled"] is False
        assert am.stats["total"] == 0

    def test_init_with_webhook(self):
        am = AlertManager(webhook_url="http://example.com/hook")
        assert am.webhook_url == "http://example.com/hook"
        assert am.stats["webhook_enabled"] is True

    def test_alert_creates_record(self):
        am = AlertManager()
        a = am.alert(AlertLevel.WARNING, "test", "测试告警")
        assert a.level == "WARNING"
        assert am.stats["total"] == 1
        assert am.stats["warning"] == 1

    def test_alert_history_order(self):
        am = AlertManager()
        am.alert(AlertLevel.INFO, "c1", "第一条")
        am.alert(AlertLevel.WARNING, "c2", "第二条")
        history = am.get_history(limit=10)
        assert len(history) == 2
        assert history[0]["message"] == "第二条"  # 最新在前

    def test_history_limit(self):
        am = AlertManager()
        for i in range(10):
            am.alert(AlertLevel.INFO, f"c{i}", f"告警{i}")
        assert len(am.get_history(limit=3)) == 3

    def test_history_filter_by_level(self):
        am = AlertManager()
        am.alert(AlertLevel.INFO, "c1", "info")
        am.alert(AlertLevel.WARNING, "c2", "warning")
        am.alert(AlertLevel.CRITICAL, "c3", "critical")
        assert len(am.get_history(level="WARNING")) == 1
        assert len(am.get_history(level="CRITICAL")) == 1

    def test_history_filter_by_category(self):
        am = AlertManager()
        am.alert(AlertLevel.INFO, "stop_loss", "a")
        am.alert(AlertLevel.INFO, "take_profit", "b")
        assert len(am.get_history(category="stop_loss")) == 1

    def test_max_history_truncation(self):
        am = AlertManager(max_history=5)
        for i in range(10):
            am.alert(AlertLevel.INFO, f"c{i}", f"告警{i}")
        assert len(am._history) == 5


# ---------------------------------------------------------------------------
# 便捷方法
# ---------------------------------------------------------------------------

class TestConvenienceMethods:
    def test_stop_loss(self):
        am = AlertManager()
        a = am.stop_loss("600519.SH", -0.035, threshold=0.03)
        assert a.category == "stop_loss"
        assert a.level == "WARNING"
        assert "600519.SH" in a.message

    def test_take_profit(self):
        am = AlertManager()
        a = am.take_profit("600519.SH", 0.085, threshold=0.08)
        assert a.category == "take_profit"
        assert a.level == "WARNING"

    def test_drawdown_pause(self):
        am = AlertManager()
        a = am.drawdown_pause(0.12, threshold=0.10)
        assert a.category == "drawdown_pause"
        assert a.level == "CRITICAL"

    def test_daily_loss_limit(self):
        am = AlertManager()
        a = am.daily_loss_limit(0.025, threshold=0.02)
        assert a.category == "daily_loss"
        assert a.level == "CRITICAL"

    def test_position_limit(self):
        am = AlertManager()
        a = am.position_limit("600519.SH", 0.25, threshold=0.20)
        assert a.category == "position_limit"
        assert a.level == "WARNING"

    def test_jev_filtered(self):
        am = AlertManager()
        a = am.jev_filtered("600519.SH", "buy", 0.45, threshold=0.6)
        assert a.category == "jev_filter"
        assert a.level == "INFO"

    def test_consecutive_losses(self):
        am = AlertManager()
        a = am.consecutive_losses(3)
        assert a.category == "consecutive_losses"
        assert a.level == "WARNING"


# ---------------------------------------------------------------------------
# Webhook 推送
# ---------------------------------------------------------------------------

class TestWebhook:
    @patch("monitoring.alert.requests.Session")
    def test_webhook_send_success(self, mock_session_cls):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_session.post.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        am = AlertManager(webhook_url="http://example.com/hook", cooldown_seconds=0)
        am.alert(AlertLevel.WARNING, "test", "测试", force=True)

        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[1]["json"]["alert"]["message"] == "测试"

    @patch("monitoring.alert.requests.Session")
    def test_webhook_failure_degrades(self, mock_session_cls):
        mock_session = MagicMock()
        import requests as req
        mock_session.post.side_effect = req.RequestException("连接失败")
        mock_session_cls.return_value = mock_session

        am = AlertManager(webhook_url="http://example.com/hook", cooldown_seconds=0)
        am.alert(AlertLevel.WARNING, "test", "测试", force=True)

        assert am._webhook_failed is True
        assert am.stats["webhook_available"] is False

    @patch("monitoring.alert.requests.Session")
    def test_webhook_non_2xx(self, mock_session_cls):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_session.post.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        am = AlertManager(webhook_url="http://example.com/hook", cooldown_seconds=0)
        am.alert(AlertLevel.WARNING, "test", "测试", force=True)
        # 500不标记为完全失败（只是warning），但后续仍会尝试
        assert mock_session.post.called

    @patch("monitoring.alert.requests.Session")
    def test_no_webhook_no_session(self, mock_session_cls):
        am = AlertManager()  # 无webhook
        am.alert(AlertLevel.WARNING, "test", "测试")
        mock_session_cls.assert_not_called()

    @patch("monitoring.alert.requests.Session")
    def test_test_webhook_success(self, mock_session_cls):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_session.post.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        am = AlertManager(webhook_url="http://example.com/hook")
        assert am.test_webhook() is True

    @patch("monitoring.alert.requests.Session")
    def test_test_webhook_failure(self, mock_session_cls):
        mock_session = MagicMock()
        import requests as req
        mock_session.post.side_effect = req.RequestException("失败")
        mock_session_cls.return_value = mock_session

        am = AlertManager(webhook_url="http://example.com/hook")
        assert am.test_webhook() is False
        assert am._webhook_failed is True

    def test_test_webhook_no_url(self):
        am = AlertManager()
        assert am.test_webhook() is False

    def test_reset_webhook(self):
        am = AlertManager(webhook_url="http://example.com/hook")
        am._webhook_failed = True
        am.reset_webhook()
        assert am._webhook_failed is False


# ---------------------------------------------------------------------------
# 冷却去重
# ---------------------------------------------------------------------------

class TestCooldown:
    @patch("monitoring.alert.requests.Session")
    def test_cooldown_prevents_duplicate(self, mock_session_cls):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_session.post.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        am = AlertManager(webhook_url="http://example.com/hook", cooldown_seconds=300)
        # 第一次发送
        am.alert(AlertLevel.WARNING, "stop_loss", "止损1", symbol="A")
        # 第二次同类告警应被冷却（不发送webhook，但仍记录历史）
        am.alert(AlertLevel.WARNING, "stop_loss", "止损2", symbol="A")

        assert mock_session.post.call_count == 1  # 只发了一次
        assert am.stats["total"] == 2  # 历史记录了2条

    @patch("monitoring.alert.requests.Session")
    def test_force_bypasses_cooldown(self, mock_session_cls):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_session.post.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        am = AlertManager(webhook_url="http://example.com/hook", cooldown_seconds=300)
        am.alert(AlertLevel.CRITICAL, "drawdown", "回撤", force=True)
        am.alert(AlertLevel.CRITICAL, "drawdown", "回撤2", force=True)
        assert mock_session.post.call_count == 2

    @patch("monitoring.alert.requests.Session")
    def test_different_categories_not_deduped(self, mock_session_cls):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_session.post.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        am = AlertManager(webhook_url="http://example.com/hook", cooldown_seconds=300)
        am.alert(AlertLevel.WARNING, "stop_loss", "止损")
        am.alert(AlertLevel.WARNING, "take_profit", "止盈")
        assert mock_session.post.call_count == 2


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_counts(self):
        am = AlertManager()
        am.alert(AlertLevel.CRITICAL, "c1", "crit")
        am.alert(AlertLevel.CRITICAL, "c2", "crit2")
        am.alert(AlertLevel.WARNING, "c3", "warn")
        am.alert(AlertLevel.INFO, "c4", "info")
        stats = am.stats
        assert stats["total"] == 4
        assert stats["critical"] == 2
        assert stats["warning"] == 1
        assert stats["info"] == 1

    def test_stats_empty(self):
        am = AlertManager()
        stats = am.stats
        assert stats["total"] == 0
        assert stats["critical"] == 0
