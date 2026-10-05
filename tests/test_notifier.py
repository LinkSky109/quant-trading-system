"""多渠道通知系统单元测试。

覆盖:
- 通知模板渲染（变量替换 / 缺失变量保留原样）
- 各渠道适配器 payload 格式（mock 网络 / smtplib）
- 钉钉加签安全验证
- 级别路由 / 事件类型路由（取并集）
- 静默时间（非CRITICAL静默期不推送，CRITICAL除外）
- NotifierManager 集成与 AlertManager 挂载

所有网络请求 / SMTP 均用 unittest.mock 模拟，不实际发送。
"""
from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
from unittest.mock import MagicMock, patch

import pytest

from monitoring.alert import Alert, AlertLevel, AlertManager
from monitoring.notifier import (
    DingTalkNotifier,
    EmailNotifier,
    NotificationTemplate,
    NotifierManager,
    ServerChanNotifier,
    WebhookNotifier,
    WeComNotifier,
    PRESET_TEMPLATES,
)


@pytest.fixture
def mock_channels():
    """把所有渠道适配器的 send 打桩为成功（隔离路由逻辑，不发真实网络）。"""
    with patch.object(EmailNotifier, "send", return_value=True), \
         patch.object(WeComNotifier, "send", return_value=True), \
         patch.object(DingTalkNotifier, "send", return_value=True), \
         patch.object(ServerChanNotifier, "send", return_value=True), \
         patch.object(WebhookNotifier, "send", return_value=True):
        yield


# ---------------------------------------------------------------------------
# 测试用配置（全部渠道 enabled，便于路由测试）
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> dict:
    """构造一份通知配置：总开关开，全部渠道启用，路由规则齐全。"""
    cfg = {
        "enabled": True,
        "channels": {
            "email": {"enabled": True, "smtp_host": "smtp.example.com",
                      "to_addrs": ["a@b.com"]},
            "wecom": {"enabled": True, "webhook_url": "http://wecom/hook"},
            "dingtalk": {"enabled": True, "webhook_url": "http://ding/hook"},
            "serverchan": {"enabled": True, "send_key": "SCT123"},
            "webhook": {"enabled": True, "url": "http://wh/hook"},
        },
        "rules": {
            "level_routing": {
                "CRITICAL": ["email", "wecom"],
                "WARNING": ["wecom"],
                "INFO": [],
            },
            "event_routing": {
                "trade": ["wecom"],
                "risk": ["email", "wecom"],
                "system": ["email"],
                "daily": ["email", "wecom"],
            },
        },
        "silent_hours": {"enabled": False, "start": "23:00", "end": "08:00"},
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# 模板渲染
# ---------------------------------------------------------------------------

class TestTemplate:
    def test_render_substitution(self):
        t = NotificationTemplate(template="标的:{symbol} 方向:{action} 价:{price}")
        out = t.render({"symbol": "600519.SH", "action": "buy", "price": 1800.0})
        assert "600519.SH" in out
        assert "buy" in out
        assert "1800.0" in out

    def test_missing_var_kept_as_is(self):
        t = NotificationTemplate(template="标的:{symbol} 缺失:{not_exist}")
        out = t.render({"symbol": "600519.SH"})
        assert "600519.SH" in out
        assert "{not_exist}" in out  # 缺失变量保留原样

    def test_none_value_becomes_empty(self):
        t = NotificationTemplate(template="标的:{symbol}")
        out = t.render({"symbol": None})
        assert out == "标的:"

    def test_preset_templates_exist(self):
        for name in ("trade", "risk", "system", "daily"):
            assert name in PRESET_TEMPLATES

    def test_preset_render(self):
        t = NotificationTemplate(preset="trade")
        out = t.render({"symbol": "600519.SH", "action": "buy"})
        assert "600519.SH" in out


# ---------------------------------------------------------------------------
# 渠道适配器 payload 格式
# ---------------------------------------------------------------------------

class TestWeCom:
    @patch("monitoring.notifier.requests.post")
    def test_markdown_payload(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"errcode": 0}
        mock_post.return_value = mock_resp

        n = WeComNotifier({"enabled": True, "webhook_url": "http://wecom/hook",
                           "msg_type": "markdown"})
        assert n.send("标题", "正文") is True
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["msgtype"] == "markdown"
        assert "正文" in kwargs["json"]["markdown"]["content"]

    @patch("monitoring.notifier.requests.post")
    def test_text_payload(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"errcode": 0}
        mock_post.return_value = mock_resp

        n = WeComNotifier({"enabled": True, "webhook_url": "http://wecom/hook",
                           "msg_type": "text"})
        n.send("标题", "正文")
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["msgtype"] == "text"
        assert "正文" in kwargs["json"]["text"]["content"]

    def test_disabled_no_send(self):
        n = WeComNotifier({"enabled": False, "webhook_url": "http://x"})
        assert n.send("t", "c") is False


class TestDingTalk:
    def test_sign_calculation(self):
        """已知 secret + timestamp 验证签名计算正确。"""
        secret = "SECtestsecret123"
        timestamp = "1609459200000"  # 固定时间戳
        expected_str = f"{timestamp}\n{secret}"
        expected_hmac = hmac_mod.new(
            secret.encode("utf-8"), expected_str.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        expected_sign = base64.b64encode(expected_hmac)
        # 与实现对比（实现做了 urlencode）
        from urllib.parse import quote_plus
        expected = quote_plus(expected_sign)
        assert DingTalkNotifier._sign(secret, timestamp) == expected

    @patch("monitoring.notifier.requests.post")
    def test_markdown_payload_with_sign(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"errcode": 0}
        mock_post.return_value = mock_resp

        n = DingTalkNotifier({
            "enabled": True,
            "webhook_url": "http://ding/hook?access_token=abc",
            "secret": "SECxxx",
            "msg_type": "markdown",
        })
        assert n.send("标题", "正文") is True
        _, kwargs = mock_post.call_args
        # URL 应带 timestamp 与 sign
        called_url = kwargs.get("url") or mock_post.call_args[0][0]
        assert "timestamp=" in called_url
        assert "sign=" in called_url
        assert kwargs["json"]["msgtype"] == "markdown"
        assert kwargs["json"]["markdown"]["title"] == "标题"

    @patch("monitoring.notifier.requests.post")
    def test_no_secret_no_sign(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"errcode": 0}
        mock_post.return_value = mock_resp

        n = DingTalkNotifier({"enabled": True,
                              "webhook_url": "http://ding/hook?access_token=abc"})
        n.send("标题", "正文")
        called_url = mock_post.call_args[0][0]
        assert "timestamp=" not in called_url


class TestServerChan:
    @patch("monitoring.notifier.requests.post")
    def test_payload(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 0}
        mock_post.return_value = mock_resp

        n = ServerChanNotifier({"enabled": True, "send_key": "SCT123"})
        assert n.send("标题", "正文") is True
        called_url = mock_post.call_args[0][0]
        assert called_url == "https://sctapi.ftqq.com/SCT123.send"
        _, kwargs = mock_post.call_args
        assert kwargs["data"]["title"] == "标题"
        assert kwargs["data"]["desp"] == "正文"


class TestWebhook:
    @patch("monitoring.notifier.requests.post")
    def test_payload(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        n = WebhookNotifier({"enabled": True, "url": "http://wh/hook"})
        assert n.send("标题", "正文") is True
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["title"] == "标题"
        assert kwargs["json"]["content"] == "正文"


class TestEmail:
    @patch("monitoring.notifier.smtplib.SMTP_SSL")
    def test_ssl_send(self, mock_ssl):
        mock_server = MagicMock()
        mock_ssl.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_ssl.return_value.__exit__ = MagicMock(return_value=False)
        # 不用上下文管理器写法，直接 mock 实例
        instance = MagicMock()
        mock_ssl.return_value = instance

        n = EmailNotifier({
            "enabled": True,
            "smtp_host": "smtp.example.com",
            "smtp_port": 465,
            "use_ssl": True,
            "user": "a@b.com",
            "password": "pw",
            "from_addr": "a@b.com",
            "to_addrs": ["admin@b.com"],
        })
        assert n.send("标题", "正文") is True
        instance.login.assert_called_once()
        instance.sendmail.assert_called_once()

    def test_disabled(self):
        n = EmailNotifier({"enabled": False})
        assert n.send("t", "c") is False

    def test_missing_config(self):
        n = EmailNotifier({"enabled": True, "smtp_host": "", "to_addrs": []})
        assert n.send("t", "c") is False


# ---------------------------------------------------------------------------
# 路由规则
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("mock_channels")
class TestRouting:
    def test_critical_routes_email_wecom(self):
        nm = NotifierManager(_make_config())
        # CRITICAL: level_routing=[email,wecom], event=risk -> [email,wecom]
        result = nm.send("CRITICAL", "risk", "回撤超标", {"message": "回撤12%"})
        assert "email" in result
        assert "wecom" in result

    def test_warning_routes_wecom(self):
        nm = NotifierManager(_make_config())
        # WARNING: level_routing=[wecom], event=risk -> [email,wecom]（事件路由并集加email）
        result = nm.send("WARNING", "risk", "止损", {"message": "止损3%"})
        assert "wecom" in result
        # risk 事件路由含 email，所以并集也应有 email
        assert "email" in result

    def test_info_no_level_routing(self):
        nm = NotifierManager(_make_config())
        # INFO: level_routing=[]，event=trade -> [wecom]
        result = nm.send("INFO", "trade", "成交", {"message": "buy"})
        assert "wecom" in result
        # level INFO 无路由，email 不应出现
        assert "email" not in result

    def test_event_routing_trade(self):
        nm = NotifierManager(_make_config())
        result = nm.send("INFO", "trade", "成交", {})
        assert list(result.keys()) == ["wecom"]

    def test_event_routing_system(self):
        nm = NotifierManager(_make_config())
        result = nm.send("CRITICAL", "system", "服务异常", {"message": "db断连"})
        # system 事件 -> [email]，CRITICAL 级别 -> [email,wecom]，并集 [email,wecom]
        assert "email" in result
        assert "wecom" in result

    def test_union_dedup(self):
        """并集去重：同渠道不重复。"""
        nm = NotifierManager(_make_config())
        result = nm.send("CRITICAL", "system", "t", {})
        assert result["email"] is True
        # email 只出现一次
        assert list(result.keys()).count("email") == 1


# ---------------------------------------------------------------------------
# 静默时间
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("mock_channels")
class TestSilentHours:
    def _nm_during_silent(self) -> NotifierManager:
        cfg = _make_config()
        cfg["silent_hours"] = {"enabled": True, "start": "00:00", "end": "23:59"}
        # start=00:00 end=23:59 实际几乎全天静默（用于强制进入静默窗口）
        return NotifierManager(cfg)

    def test_non_critical_suppressed(self):
        nm = self._nm_during_silent()
        result = nm.send("WARNING", "risk", "止损", {"message": "x"})
        assert result == {}  # 静默期非CRITICAL不推送
        # 记录里应有一条 skipped/silent
        status = nm.get_status()
        assert status["records"][0]["status"] == "skipped"

    def test_critical_still_sent(self):
        nm = self._nm_during_silent()
        result = nm.send("CRITICAL", "risk", "回撤", {"message": "x"})
        assert "email" in result  # CRITICAL 不受静默影响

    def test_silent_disabled(self):
        cfg = _make_config()
        cfg["silent_hours"] = {"enabled": False, "start": "23:00", "end": "08:00"}
        nm = NotifierManager(cfg)
        result = nm.send("WARNING", "risk", "止损", {"message": "x"})
        assert "wecom" in result

    def test_in_silent_window_cross_midnight(self):
        """跨午夜区间判断。"""
        nm = NotifierManager(_make_config())
        nm.silent_cfg = {"enabled": True, "start": "23:00", "end": "08:00"}
        # 用固定时间戳：凌晨 02:00 应在静默期
        import time as _t
        ts_am2 = _t.mktime(_t.strptime("2026-10-01 02:00:00", "%Y-%m-%d %H:%M:%S"))
        assert nm._in_silent_window(ts_am2) is True
        # 中午 12:00 不在
        ts_noon = _t.mktime(_t.strptime("2026-10-01 12:00:00", "%Y-%m-%d %H:%M:%S"))
        assert nm._in_silent_window(ts_noon) is False


# ---------------------------------------------------------------------------
# NotifierManager 集成
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("mock_channels")
class TestNotifierManager:
    def test_disabled_master_switch(self):
        cfg = _make_config()
        cfg["enabled"] = False
        nm = NotifierManager(cfg)
        result = nm.send("CRITICAL", "risk", "t", {})
        assert result == {}

    def test_init_default_all_disabled(self):
        nm = NotifierManager({})  # 空配置
        assert nm.enabled is False
        status = nm.get_status()
        for ch in status["channels"].values():
            assert ch["enabled"] is False

    def test_send_trade(self):
        nm = NotifierManager(_make_config())
        result = nm.send_trade({
            "symbol": "600519.SH", "action": "buy", "price": 1800.0,
            "quantity": 100, "pnl": 0.0, "strategy": "ma_cross",
            "confidence": 0.8,
        })
        assert "wecom" in result

    def test_send_risk_alert_with_alert_object(self):
        nm = NotifierManager(_make_config())
        alert = Alert(level="CRITICAL", category="drawdown_pause",
                      symbol=None, message="回撤超标12%",
                      current_value=0.12, threshold=0.10)
        result = nm.send_risk_alert(alert)
        assert "email" in result
        assert "wecom" in result

    def test_send_system_alert(self):
        nm = NotifierManager(_make_config())
        result = nm.send_system_alert("Jev断连", "无法连接决策服务")
        assert "email" in result

    def test_send_daily_report(self):
        nm = NotifierManager(_make_config())
        result = nm.send_daily_report({"message": "今日收益+1.2%"})
        # daily 事件 -> [email,wecom]
        assert "email" in result
        assert "wecom" in result

    def test_test_channel(self):
        nm = NotifierManager(_make_config())
        with patch.object(WeComNotifier, "send", return_value=True) as m:
            assert nm.test_channel("wecom") is True
            m.assert_called_once()

    def test_test_channel_disabled(self):
        cfg = _make_config()
        cfg["channels"]["wecom"]["enabled"] = False
        nm = NotifierManager(cfg)
        assert nm.test_channel("wecom") is False

    def test_test_channel_unknown(self):
        nm = NotifierManager(_make_config())
        assert nm.test_channel("no_such") is False

    def test_get_status_records_order(self):
        nm = NotifierManager(_make_config())
        nm.send("WARNING", "risk", "a", {"message": "1"})
        nm.send("WARNING", "risk", "b", {"message": "2"})
        status = nm.get_status()
        assert len(status["records"]) >= 1
        # 最新在前
        assert status["records"][0]["status"] in ("success", "failed")


# ---------------------------------------------------------------------------
# AlertManager 挂载集成
# ---------------------------------------------------------------------------

class TestAlertManagerIntegration:
    def test_set_notifier_called(self):
        am = AlertManager()
        nm = NotifierManager(_make_config())
        am.set_notifier(nm)
        assert am.notifier is nm

    def test_alert_triggers_notifier(self):
        am = AlertManager()
        nm = MagicMock()
        am.set_notifier(nm)
        am.alert(AlertLevel.CRITICAL, "drawdown_pause", "回撤超标", force=True)
        nm.send_risk_alert.assert_called_once()

    def test_notifier_exception_does_not_break_alert(self):
        am = AlertManager()
        nm = MagicMock()
        nm.send_risk_alert.side_effect = RuntimeError("boom")
        am.set_notifier(nm)
        # 不应抛异常
        a = am.alert(AlertLevel.WARNING, "stop_loss", "止损", symbol="A")
        assert a.category == "stop_loss"
        nm.send_risk_alert.assert_called_once()

    def test_no_notifier_no_crash(self):
        am = AlertManager()  # 未挂载 notifier
        a = am.alert(AlertLevel.WARNING, "stop_loss", "止损")
        assert a.level == "WARNING"
