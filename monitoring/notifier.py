"""多渠道通知告警系统。

在 AlertManager 的 Webhook 推送之外，提供可插拔的多渠道通知能力：

渠道适配器（统一 ``send(title, content)`` 接口）：
- EmailNotifier      : SMTP 发送 HTML 邮件（支持 SSL/TLS）
- WeComNotifier      : 企业微信群机器人 Webhook（text / markdown）
- DingTalkNotifier   : 钉钉群机器人 Webhook（text / markdown，支持加签）
- ServerChanNotifier : Server酱 SendKey 推送（微信接收）
- WebhookNotifier    : 通用 Webhook（POST JSON）

路由规则：
- 按级别路由（CRITICAL / WARNING / INFO）与按事件类型路由（trade / risk /
  system / daily）取**并集**后推送。
- 静默时间（silent_hours）内非 CRITICAL 仅记录日志不推送，CRITICAL 不受影响。

模板系统：
- NotificationTemplate 支持 ``{var}`` 变量替换，缺失变量保留原样。
- 内置 trade / risk / system / daily 四类预设模板。

典型用法::

    cfg = {"notification": {...}}          # 来自 config.yaml 的 notification 节
    nm = NotifierManager(cfg.get("notification", {}))
    nm.send_trade({"symbol": "600519.SH", "action": "buy", ...})
"""
from __future__ import annotations

import base64
import hmac
import logging
import smtplib
import time
from abc import ABC, abstractmethod
from email.mime.text import MIMEText
from email.header import Header
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import requests

try:  # Alert 对象类型注解用；避免循环导入，运行期失败也不影响
    from monitoring.alert import Alert
except Exception:  # pragma: no cover
    Alert = Any  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 通知模板
# ---------------------------------------------------------------------------

class _SafeDict(dict):
    """format_map 用字典：缺失键保留为 ``{key}`` 原样。"""

    def __missing__(self, key: str) -> str:  # noqa: D401
        return "{" + key + "}"


#: 内置通知模板（可用 {symbol}/{action}/{price}/... 占位符）
PRESET_TEMPLATES: Dict[str, str] = {
    "trade": (
        "【交易通知】{level}\n"
        "- 标的: {symbol}\n"
        "- 方向: {action}\n"
        "- 价格: {price}\n"
        "- 数量: {quantity}\n"
        "- 盈亏: {pnl}\n"
        "- 策略: {strategy}\n"
        "- Jev置信度: {confidence}\n"
        "- 时间: {time}"
    ),
    "risk": (
        "【风控告警】{level}\n"
        "- 类别: {category}\n"
        "- 标的: {symbol}\n"
        "- 描述: {message}\n"
        "- 当前值: {current_value}\n"
        "- 阈值: {threshold}\n"
        "- 时间: {time}"
    ),
    "system": (
        "【系统告警】{level}\n"
        "- 标题: {title}\n"
        "- 详情: {message}\n"
        "- 时间: {time}"
    ),
    "daily": (
        "【每日报告】\n"
        "{message}\n"
        "- 时间: {time}"
    ),
}


class NotificationTemplate:
    """通知模板，支持 ``{var}`` 变量替换。

    Args:
        template: 模板字符串，使用 ``{占位符}``。
        preset: 内置模板名（trade/risk/system/daily），与 template 二选一；
            同时传入时 template 优先。
    """

    def __init__(self, template: str = "", preset: str = ""):
        if template:
            self._template = template
        elif preset:
            self._template = PRESET_TEMPLATES.get(preset, preset)
        else:
            self._template = "{message}"

    def render(self, context: Dict[str, Any]) -> str:
        """用 context 渲染模板。

        缺失的占位符保留为 ``{key}`` 原样，不抛异常。

        Args:
            context: 变量键值对。

        Returns:
            渲染后的字符串。
        """
        safe = _SafeDict()
        for k, v in context.items():
            safe[k] = "" if v is None else v
        try:
            return self._template.format_map(safe)
        except Exception as e:  # 模板异常不应阻断通知流程
            logger.warning("模板渲染失败，使用原始文本: %s", e)
            return self._template


# ---------------------------------------------------------------------------
# 渠道适配器基类
# ---------------------------------------------------------------------------

class BaseNotifier(ABC):
    """通知渠道适配器基类。"""

    name: str = "base"

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled: bool = bool(self.config.get("enabled", False))
        self.timeout: float = float(self.config.get("timeout", 5.0))

    @abstractmethod
    def send(self, title: str, content: str) -> bool:
        """发送通知。

        Returns:
            True 表示发送成功。
        """

    def test(self) -> bool:
        """测试渠道连通性，默认复用 send 发送一条测试消息。"""
        try:
            return self.send("连通性测试", "这是一条来自量化交易系统的测试通知")
        except Exception as e:  # pragma: no cover - 依赖网络
            logger.warning("%s 渠道测试失败: %s", self.name, e)
            return False


# ---------------------------------------------------------------------------
# 各渠道适配器
# ---------------------------------------------------------------------------

class EmailNotifier(BaseNotifier):
    """SMTP 邮件通知（HTML 正文）。"""

    name = "email"

    def send(self, title: str, content: str) -> bool:
        if not self.enabled:
            return False
        host = self.config.get("smtp_host", "")
        port = int(self.config.get("smtp_port", 465))
        user = self.config.get("user", "")
        password = self.config.get("password", "")
        from_addr = self.config.get("from_addr", user)
        to_addrs = self.config.get("to_addrs", []) or []
        use_ssl = bool(self.config.get("use_ssl", True))
        if not host or not to_addrs:
            logger.warning("邮件渠道未配置 smtp_host / to_addrs，跳过")
            return False

        msg = MIMEText(content, "plain", "utf-8")
        msg["Subject"] = Header(title, "utf-8")
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)

        try:
            if use_ssl:
                server = smtplib.SMTP_SSL(host, port, timeout=self.timeout)
            else:
                server = smtplib.SMTP(host, port, timeout=self.timeout)
                server.starttls()
            try:
                if user and password:
                    server.login(user, password)
                server.sendmail(from_addr, to_addrs, msg.as_string())
            finally:
                server.quit()
            logger.info("邮件发送成功: %s", title)
            return True
        except Exception as e:
            logger.warning("邮件发送失败: %s", e)
            return False


class WeComNotifier(BaseNotifier):
    """企业微信群机器人 Webhook。"""

    name = "wecom"

    def _build_payload(self, title: str, content: str) -> Dict[str, Any]:
        msg_type = self.config.get("msg_type", "markdown")
        if msg_type == "text":
            return {"msgtype": "text", "text": {"content": f"{title}\n{content}"}}
        # markdown
        return {"msgtype": "markdown", "markdown": {"content": f"### {title}\n{content}"}}

    def send(self, title: str, content: str) -> bool:
        if not self.enabled:
            return False
        url = self.config.get("webhook_url", "")
        if not url:
            logger.warning("企微渠道未配置 webhook_url，跳过")
            return False
        payload = self._build_payload(title, content)
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            ok = resp.status_code < 300 and resp.json().get("errcode", 0) == 0
            if ok:
                logger.info("企微通知发送成功: %s", title)
            else:
                logger.warning("企微通知返回异常: %s %s", resp.status_code, resp.text[:200])
            return ok
        except requests.RequestException as e:
            logger.warning("企微通知发送失败: %s", e)
            return False


class DingTalkNotifier(BaseNotifier):
    """钉钉群机器人 Webhook（可选加签安全验证）。"""

    name = "dingtalk"

    @staticmethod
    def _sign(secret: str, timestamp: str) -> str:
        """计算钉钉加签签名。

        Args:
            secret: 加签密钥（SEC 开头）。
            timestamp: 毫秒时间戳字符串。

        Returns:
            urlencode 后的 sign。
        """
        string_to_sign = f"{timestamp}\n{secret}"
        hmac_code = hmac.new(
            secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            digestmod="sha256",
        ).digest()
        return quote_plus(base64.b64encode(hmac_code))

    def _build_url(self) -> str:
        url = self.config.get("webhook_url", "")
        secret = self.config.get("secret", "")
        if secret:
            timestamp = str(round(time.time() * 1000))
            sign = self._sign(secret, timestamp)
            # webhook_url 已含 ?access_token=...，追加用 &
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}timestamp={timestamp}&sign={sign}"
        return url

    def _build_payload(self, title: str, content: str) -> Dict[str, Any]:
        msg_type = self.config.get("msg_type", "markdown")
        if msg_type == "text":
            return {"msgtype": "text", "text": {"content": f"{title}\n{content}"}}
        return {"msgtype": "markdown", "markdown": {"title": title, "text": content}}

    def send(self, title: str, content: str) -> bool:
        if not self.enabled:
            return False
        url = self.config.get("webhook_url", "")
        if not url:
            logger.warning("钉钉渠道未配置 webhook_url，跳过")
            return False
        url = self._build_url()
        payload = self._build_payload(title, content)
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            ok = resp.status_code < 300 and resp.json().get("errcode", 0) == 0
            if ok:
                logger.info("钉钉通知发送成功: %s", title)
            else:
                logger.warning("钉钉通知返回异常: %s %s", resp.status_code, resp.text[:200])
            return ok
        except requests.RequestException as e:
            logger.warning("钉钉通知发送失败: %s", e)
            return False


class ServerChanNotifier(BaseNotifier):
    """Server酱 SendKey 推送（微信接收）。"""

    name = "serverchan"

    def send(self, title: str, content: str) -> bool:
        if not self.enabled:
            return False
        send_key = self.config.get("send_key", "")
        if not send_key:
            logger.warning("Server酱渠道未配置 send_key，跳过")
            return False
        url = f"https://sctapi.ftqq.com/{send_key}.send"
        try:
            resp = requests.post(
                url,
                data={"title": title, "desp": content},
                timeout=self.timeout,
            )
            ok = resp.status_code < 300 and resp.json().get("code", 0) == 0
            if ok:
                logger.info("Server酱通知发送成功: %s", title)
            else:
                logger.warning("Server酱通知返回异常: %s %s", resp.status_code, resp.text[:200])
            return ok
        except requests.RequestException as e:
            logger.warning("Server酱通知发送失败: %s", e)
            return False


class WebhookNotifier(BaseNotifier):
    """通用 Webhook 通知（POST JSON）。"""

    name = "webhook"

    def send(self, title: str, content: str) -> bool:
        if not self.enabled:
            return False
        url = self.config.get("url", "")
        if not url:
            logger.warning("通用Webhook渠道未配置 url，跳过")
            return False
        payload = {
            "title": title,
            "content": content,
            "level": self.config.get("_level", ""),
            "event_type": self.config.get("_event_type", ""),
            "source": "quant_trading_system/notifier",
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            ok = resp.status_code < 300
            if ok:
                logger.info("通用Webhook通知发送成功: %s", title)
            else:
                logger.warning("通用Webhook通知返回异常: %s", resp.status_code)
            return ok
        except requests.RequestException as e:
            logger.warning("通用Webhook通知发送失败: %s", e)
            return False


#: 渠道名 -> 适配器类
CHANNEL_CLASSES = {
    "email": EmailNotifier,
    "wecom": WeComNotifier,
    "dingtalk": DingTalkNotifier,
    "serverchan": ServerChanNotifier,
    "webhook": WebhookNotifier,
}


# ---------------------------------------------------------------------------
# NotifierManager
# ---------------------------------------------------------------------------

class NotifierManager:
    """多渠道通知管理器。

    Args:
        config: config.yaml 中 ``notification`` 节的字典。为空或
            ``enabled=false`` 时所有渠道不推送（仅日志）。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.enabled: bool = bool(config.get("enabled", False))
        channels_cfg = config.get("channels", {}) or {}
        rules = config.get("rules", {}) or {}
        self.level_routing: Dict[str, List[str]] = dict(
            rules.get("level_routing", {}) or {}
        )
        self.event_routing: Dict[str, List[str]] = dict(
            rules.get("event_routing", {}) or {}
        )
        self.silent_cfg: Dict[str, Any] = config.get("silent_hours", {}) or {}

        # 初始化各渠道适配器（默认全部 disabled）
        self._channels: Dict[str, BaseNotifier] = {}
        for name, cls in CHANNEL_CLASSES.items():
            self._channels[name] = cls(channels_cfg.get(name, {}) or {})

        # 发送记录（内存，最近 max_records 条）
        self.max_records = 50
        self._records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 静默时间判断
    # ------------------------------------------------------------------

    def _in_silent_window(self, now_ts: Optional[float] = None) -> bool:
        """判断当前是否处于静默时间段。

        支持跨午夜区间（如 23:00-08:00）。

        Args:
            now_ts: 指定时间戳（测试用），默认当前时间。

        Returns:
            True 表示处于静默期。
        """
        if not self.silent_cfg.get("enabled", False):
            return False
        start = self.silent_cfg.get("start", "23:00")
        end = self.silent_cfg.get("end", "08:00")

        def _to_minutes(hhmm: str) -> int:
            try:
                h, m = str(hhmm).split(":")
                return int(h) * 60 + int(m)
            except Exception:
                return 0

        now = time.localtime(now_ts) if now_ts else time.localtime()
        cur = now.tm_hour * 60 + now.tm_min
        s = _to_minutes(start)
        e = _to_minutes(end)
        if s == e:
            return False
        if s < e:
            # 不跨午夜，如 01:00-08:00
            return s <= cur < e
        # 跨午夜，如 23:00-08:00
        return cur >= s or cur < e

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------

    def _resolve_channels(self, level: str, event_type: str) -> List[str]:
        """根据级别与事件类型路由规则取并集。"""
        level_channels = self.level_routing.get(level, []) or []
        event_channels = self.event_routing.get(event_type, []) or []
        # 并集（保持顺序）
        merged: List[str] = []
        for ch in list(level_channels) + list(event_channels):
            if ch not in merged:
                merged.append(ch)
        return merged

    # ------------------------------------------------------------------
    # 发送记录
    # ------------------------------------------------------------------

    def _record(self, channel: str, level: str, event_type: str,
                status: str, detail: str = "") -> None:
        rec = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "channel": channel,
            "level": level,
            "event_type": event_type,
            "status": status,
            "detail": detail,
        }
        self._records.append(rec)
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]

    # ------------------------------------------------------------------
    # 核心发送
    # ------------------------------------------------------------------

    def send(
        self,
        level: str,
        event_type: str,
        title: str,
        context: Dict[str, Any],
        template: str = "",
    ) -> Dict[str, bool]:
        """发送通知，自动按规则路由。

        Args:
            level: 级别 CRITICAL / WARNING / INFO。
            event_type: 事件类型 trade / risk / system / daily。
            title: 通知标题。
            context: 模板变量上下文。
            template: 自定义模板字符串（为空时按 event_type 选内置模板）。

        Returns:
            {渠道名: 是否成功} 的字典。
        """
        result: Dict[str, bool] = {}

        if not self.enabled:
            logger.debug("通知总开关关闭，仅记录日志: [%s] %s", level, title)
            self._record("none", level, event_type, "skipped", "总开关关闭")
            return result

        # 渲染正文
        tpl = NotificationTemplate(template=template,
                                   preset=template if template in PRESET_TEMPLATES else "")
        ctx = dict(context or {})
        ctx.setdefault("level", level)
        ctx.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        content = tpl.render(ctx)

        targets = self._resolve_channels(level, event_type)
        if not targets:
            logger.info("通知路由为空，仅记录日志: [%s] %s", level, title)
            self._record("none", level, event_type, "skipped", "无路由渠道")
            return result

        # 静默时间：非 CRITICAL 不推送
        if self._in_silent_window() and level != "CRITICAL":
            logger.info("静默期内非CRITICAL通知仅记录: [%s] %s", level, title)
            self._record("silent", level, event_type, "skipped", "静默时间")
            return result

        for ch_name in targets:
            adapter = self._channels.get(ch_name)
            if adapter is None:
                logger.warning("未知通知渠道: %s", ch_name)
                self._record(ch_name, level, event_type, "failed", "未知渠道")
                result[ch_name] = False
                continue
            # 把级别/事件类型带给通用 webhook
            self.config_hint(adapter, level, event_type)
            ok = False
            try:
                ok = adapter.send(title, content)
            except Exception as e:  # 单渠道异常不影响其他渠道
                logger.warning("渠道 %s 发送异常: %s", ch_name, e)
                ok = False
            self._record(ch_name, level, event_type, "success" if ok else "failed")
            result[ch_name] = ok
        return result

    @staticmethod
    def config_hint(adapter: BaseNotifier, level: str, event_type: str) -> None:
        """给适配器附带级别/事件提示（仅通用 webhook 用到）。"""
        try:
            adapter.config["_level"] = level
            adapter.config["_event_type"] = event_type
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 快捷方法
    # ------------------------------------------------------------------

    def send_trade(self, context: Dict[str, Any]) -> Dict[str, bool]:
        """发送交易成交通知。

        Args:
            context: 含 symbol/action/price/quantity/pnl/strategy/confidence 等。
        """
        symbol = context.get("symbol", "")
        action = context.get("action", "")
        title = f"交易成交: {symbol} {action}"
        return self.send(
            level=context.get("level", "INFO"),
            event_type="trade",
            title=title,
            context=context,
            template="trade",
        )

    def send_risk_alert(self, alert: "Alert") -> Dict[str, bool]:
        """发送风控告警通知（接收 Alert 对象）。

        Args:
            alert: AlertManager 产生的 Alert 对象。
        """
        ctx = {
            "category": alert.category,
            "symbol": alert.symbol or "-",
            "message": alert.message,
            "current_value": alert.current_value if alert.current_value is not None else "-",
            "threshold": alert.threshold if alert.threshold is not None else "-",
        }
        return self.send(
            level=alert.level,
            event_type="risk",
            title=f"风控告警: {alert.category}",
            context=ctx,
            template="risk",
        )

    def send_system_alert(self, title: str, message: str) -> Dict[str, bool]:
        """发送系统告警。

        Args:
            title: 告警标题。
            message: 告警详情。
        """
        return self.send(
            level="CRITICAL",
            event_type="system",
            title=title,
            context={"title": title, "message": message},
            template="system",
        )

    def send_daily_report(self, report_data: Dict[str, Any]) -> Dict[str, bool]:
        """发送每日收盘报告。

        Args:
            report_data: 报告数据，需含 message（摘要文本）或由各字段拼成。
        """
        message = report_data.get("message") or self._format_daily(report_data)
        return self.send(
            level="INFO",
            event_type="daily",
            title="每日交易报告",
            context={"message": message},
            template="daily",
        )

    @staticmethod
    def _format_daily(report_data: Dict[str, Any]) -> str:
        """把报告数据格式化为摘要文本。"""
        lines = []
        for k, v in report_data.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 渠道测试与状态
    # ------------------------------------------------------------------

    def test_channel(self, channel_name: str) -> bool:
        """测试指定渠道连通性。

        Args:
            channel_name: 渠道名 email/wecom/dingtalk/serverchan/webhook。

        Returns:
            True 表示成功。
        """
        adapter = self._channels.get(channel_name)
        if adapter is None:
            logger.warning("未知通知渠道: %s", channel_name)
            return False
        if not adapter.enabled:
            logger.info("渠道 %s 未启用，跳过测试", channel_name)
            return False
        ok = adapter.test()
        self._record(channel_name, "INFO", "test", "success" if ok else "failed")
        return ok

    def get_status(self) -> Dict[str, Any]:
        """返回各渠道配置状态与最近发送记录。"""
        channels_status = {}
        for name, adapter in self._channels.items():
            channels_status[name] = {
                "enabled": adapter.enabled,
                "configured": bool(adapter.enabled),
            }
        return {
            "enabled": self.enabled,
            "channels": channels_status,
            "level_routing": self.level_routing,
            "event_routing": self.event_routing,
            "silent_hours": self.silent_cfg,
            "records": list(reversed(self._records)),  # 最新在前
        }
