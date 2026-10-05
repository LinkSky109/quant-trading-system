"""风控告警管理器。

支持:
- Webhook HTTP POST 推送告警（JSON格式）
- 告警去重（同类型告警冷却期内不重复推送）
- 告警历史记录（内存 + 可选SQLite持久化）
- 三级告警: CRITICAL / WARNING / INFO
- Webhook不可用时自动降级为日志记录，不阻断交易
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """告警级别。"""
    CRITICAL = "CRITICAL"  # 总回撤≥10%暂停、日亏≥2%
    WARNING = "WARNING"    # 单笔止损/止盈、单标的仓位超限
    INFO = "INFO"          # Jev过滤交易、策略信号变化


@dataclass
class Alert:
    """单条告警。"""
    level: str
    category: str          # 告警类别: stop_loss / take_profit / drawdown / daily_loss / position_limit / jev_filter / signal_change
    symbol: Optional[str]  # 相关标的（可为空，如账户级告警）
    message: str           # 告警描述
    current_value: Optional[float] = None  # 当前值
    threshold: Optional[float] = None      # 触发阈值
    account_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)
        )
        return d


class AlertManager:
    """风控告警管理器。

    Args:
        webhook_url: Webhook推送地址（为空则仅日志记录）。
        cooldown_seconds: 同类告警冷却时间（秒），默认300秒。
        max_history: 内存中保留的最大告警条数。
        timeout: Webhook请求超时（秒）。
    """

    def __init__(
        self,
        webhook_url: str = "",
        cooldown_seconds: int = 300,
        max_history: int = 500,
        timeout: float = 5.0,
    ):
        self.webhook_url = webhook_url
        self.cooldown_seconds = cooldown_seconds
        self.max_history = max_history
        self.timeout = timeout
        self._history: List[Alert] = []
        self._last_sent: Dict[str, float] = {}  # category -> last sent timestamp
        self._session = requests.Session() if webhook_url else None
        self._webhook_failed = False  # Webhook是否已标记为不可用
        self.notifier = None  # 可选的 NotifierManager，设置后告警会同步推送多渠道

    # ------------------------------------------------------------------
    # 告警触发
    # ------------------------------------------------------------------

    def alert(
        self,
        level: AlertLevel,
        category: str,
        message: str,
        symbol: Optional[str] = None,
        current_value: Optional[float] = None,
        threshold: Optional[float] = None,
        account_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Alert:
        """触发一条告警。

        Args:
            level: 告警级别。
            category: 告警类别（用于去重）。
            message: 告警描述。
            symbol: 相关标的。
            current_value: 当前触发值。
            threshold: 阈值。
            account_id: 账户ID。
            extra: 额外信息。
            force: 强制发送（跳过冷却去重）。

        Returns:
            创建的Alert对象。
        """
        alert = Alert(
            level=level.value,
            category=category,
            symbol=symbol,
            message=message,
            current_value=current_value,
            threshold=threshold,
            account_id=account_id,
            extra=extra or {},
        )

        # 记录历史
        self._history.append(alert)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

        # 日志记录（所有级别都记日志）
        log_fn = logger.critical if level == AlertLevel.CRITICAL else (
            logger.warning if level == AlertLevel.WARNING else logger.info
        )
        log_fn("[%s] %s%s", level.value, message,
               f" (标的={symbol})" if symbol else "")

        # Webhook推送（检查冷却）
        if self.webhook_url and (force or self._should_send(category)):
            self._send_webhook(alert)

        # 多渠道通知（若已挂载 NotifierManager，异常不阻断交易主流程）
        if self.notifier is not None:
            try:
                self.notifier.send_risk_alert(alert)
            except Exception as e:
                logger.warning("多渠道通知推送失败，已忽略: %s", e)

        return alert

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def stop_loss(self, symbol: str, pnl_ratio: float, threshold: float = 0.03,
                  account_id: Optional[str] = None) -> Alert:
        """单笔止损告警。"""
        return self.alert(
            AlertLevel.WARNING, "stop_loss",
            f"触发止损: {symbol} 亏损 {pnl_ratio*100:.2f}% ≥ 阈值 {threshold*100:.1f}%",
            symbol=symbol, current_value=pnl_ratio, threshold=threshold,
            account_id=account_id,
        )

    def take_profit(self, symbol: str, pnl_ratio: float, threshold: float = 0.08,
                    account_id: Optional[str] = None) -> Alert:
        """单笔止盈告警。"""
        return self.alert(
            AlertLevel.WARNING, "take_profit",
            f"触发止盈: {symbol} 盈利 {pnl_ratio*100:.2f}% ≥ 阈值 {threshold*100:.1f}%",
            symbol=symbol, current_value=pnl_ratio, threshold=threshold,
            account_id=account_id,
        )

    def drawdown_pause(self, drawdown: float, threshold: float = 0.10,
                       account_id: Optional[str] = None) -> Alert:
        """总回撤超标暂停交易告警。"""
        return self.alert(
            AlertLevel.CRITICAL, "drawdown_pause",
            f"总回撤 {drawdown*100:.2f}% ≥ 阈值 {threshold*100:.1f}%，已暂停所有策略",
            current_value=drawdown, threshold=threshold, account_id=account_id,
            force=True,  # CRITICAL强制发送
        )

    def daily_loss_limit(self, daily_loss: float, threshold: float = 0.02,
                         account_id: Optional[str] = None) -> Alert:
        """单日亏损限额告警。"""
        return self.alert(
            AlertLevel.CRITICAL, "daily_loss",
            f"单日亏损 {daily_loss*100:.2f}% ≥ 阈值 {threshold*100:.1f}%",
            current_value=daily_loss, threshold=threshold, account_id=account_id,
            force=True,
        )

    def position_limit(self, symbol: str, position_ratio: float,
                       threshold: float = 0.20, account_id: Optional[str] = None) -> Alert:
        """单标的仓位超限告警。"""
        return self.alert(
            AlertLevel.WARNING, "position_limit",
            f"单标的仓位超限: {symbol} 占比 {position_ratio*100:.1f}% > 上限 {threshold*100:.1f}%",
            symbol=symbol, current_value=position_ratio, threshold=threshold,
            account_id=account_id,
        )

    def jev_filtered(self, symbol: str, action: str, confidence: float,
                     threshold: float = 0.6, account_id: Optional[str] = None) -> Alert:
        """Jev过滤交易告警。"""
        return self.alert(
            AlertLevel.INFO, "jev_filter",
            f"Jev过滤: {symbol} {action} 置信度 {confidence:.2f} < 阈值 {threshold}",
            symbol=symbol, current_value=confidence, threshold=threshold,
            account_id=account_id,
        )

    def consecutive_losses(self, count: int, account_id: Optional[str] = None) -> Alert:
        """连续亏损告警。"""
        return self.alert(
            AlertLevel.WARNING, "consecutive_losses",
            f"连续 {count} 笔亏损，建议暂停检查策略",
            current_value=float(count), account_id=account_id,
        )

    # ------------------------------------------------------------------
    # Webhook
    # ------------------------------------------------------------------

    def _should_send(self, category: str) -> bool:
        """检查是否在冷却期外。"""
        now = time.time()
        last = self._last_sent.get(category, 0)
        if now - last < self.cooldown_seconds:
            logger.debug("告警 %s 在冷却期内，跳过Webhook推送", category)
            return False
        self._last_sent[category] = now
        return True

    def _send_webhook(self, alert: Alert) -> None:
        """发送Webhook告警。"""
        if not self.webhook_url or self._webhook_failed:
            return
        try:
            payload = {
                "alert": alert.to_dict(),
                "source": "quant_trading_system",
            }
            resp = self._session.post(
                self.webhook_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code < 300:
                logger.info("Webhook告警推送成功: %s", alert.category)
            else:
                logger.warning("Webhook返回非2xx: %d %s", resp.status_code, resp.text[:200])
        except requests.RequestException as e:
            logger.warning("Webhook推送失败，降级为日志: %s", e)
            # 连续失败后标记不可用，避免每次都尝试
            self._webhook_failed = True
            logger.warning("Webhook已标记为不可用，后续告警仅日志记录")

    def test_webhook(self) -> bool:
        """测试Webhook连通性。

        Returns:
            True如果Webhook可达。
        """
        if not self.webhook_url:
            return False
        try:
            payload = {
                "alert": {
                    "level": "INFO",
                    "category": "test",
                    "message": "Webhook连通性测试",
                    "timestamp": time.time(),
                },
                "source": "quant_trading_system",
                "type": "test",
            }
            resp = self._session.post(
                self.webhook_url, json=payload, timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            self._webhook_failed = resp.status_code >= 300
            return resp.status_code < 300
        except requests.RequestException:
            self._webhook_failed = True
            return False

    def reset_webhook(self) -> None:
        """重置Webhook失败状态，重新尝试推送。"""
        self._webhook_failed = False

    def set_notifier(self, notifier: Any) -> None:
        """挂载多渠道通知管理器（可选）。

        Args:
            notifier: NotifierManager 实例；设置后触发告警会同步调用
                ``notifier.send_risk_alert()``。传 None 可取消挂载。
        """
        self.notifier = notifier

    # ------------------------------------------------------------------
    # 历史查询
    # ------------------------------------------------------------------

    def get_history(
        self,
        level: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """查询告警历史。

        Args:
            level: 按级别过滤（CRITICAL/WARNING/INFO）。
            category: 按类别过滤。
            limit: 返回条数。

        Returns:
            告警列表（倒序，最新在前）。
        """
        alerts = self._history
        if level:
            alerts = [a for a in alerts if a.level == level]
        if category:
            alerts = [a for a in alerts if a.category == category]
        return [a.to_dict() for a in reversed(alerts[-limit:])]

    @property
    def stats(self) -> Dict[str, int]:
        """告警统计。"""
        return {
            "total": len(self._history),
            "critical": sum(1 for a in self._history if a.level == "CRITICAL"),
            "warning": sum(1 for a in self._history if a.level == "WARNING"),
            "info": sum(1 for a in self._history if a.level == "INFO"),
            "webhook_enabled": bool(self.webhook_url),
            "webhook_available": not self._webhook_failed if self.webhook_url else False,
        }
