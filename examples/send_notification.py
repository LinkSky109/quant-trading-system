#!/usr/bin/env python3
"""多渠道通知系统使用示例。

演示：
  1. 构造 NotifierManager 配置（开启渠道、设置路由规则、静默时间）
  2. 发送交易通知 / 风控告警 / 系统告警 / 每日报告
  3. 测试指定渠道连通性
  4. 查看各渠道状态与最近发送记录

注意：
  - 示例中使用假的 webhook / SMTP 配置，实际发送会失败（仅作演示）。
  - 要真正推送，请填入真实的 webhook_url / smtp 账号 / send_key。
  - 直接运行:  python examples/send_notification.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让脚本可从 examples/ 目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoring.alert import Alert, AlertLevel  # noqa: E402
from monitoring.notifier import NotifierManager  # noqa: E402


def build_demo_config() -> dict:
    """构造一份演示用通知配置（默认全部 disabled，按需打开）。"""
    return {
        "enabled": True,  # 总开关
        "channels": {
            "wecom": {
                "enabled": False,  # 改成 True 并填真实地址后推送
                "webhook_url": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY",
                "msg_type": "markdown",
            },
            "dingtalk": {
                "enabled": False,
                "webhook_url": "https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN",
                "secret": "SEC_YOUR_SECRET",  # 加签密钥
                "msg_type": "markdown",
            },
            "email": {
                "enabled": False,
                "smtp_host": "smtp.exmail.qq.com",
                "smtp_port": 465,
                "use_ssl": True,
                "user": "alert@yourcorp.com",
                "password": "YOUR_MAIL_PASSWORD",
                "from_addr": "alert@yourcorp.com",
                "to_addrs": ["admin@yourcorp.com"],
            },
            "serverchan": {"enabled": False, "send_key": "SCT_YOUR_SENDKEY"},
            "webhook": {"enabled": False, "url": "https://your-server/api/hook"},
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


def main() -> None:
    # 1. 初始化
    nm = NotifierManager(build_demo_config())
    print("== 通知管理器初始化完成 ==")

    # 2. 交易通知（事件类型 trade -> 企微）
    nm.send_trade({
        "symbol": "600519.SH",
        "action": "buy",
        "price": 1800.50,
        "quantity": 100,
        "pnl": 0.0,
        "strategy": "ma_cross",
        "confidence": 0.82,
    })
    print("已发送交易通知")

    # 3. 风控告警（接收 Alert 对象）
    alert = Alert(
        level=AlertLevel.CRITICAL.value,
        category="drawdown_pause",
        symbol=None,
        message="总回撤 12.0% ≥ 阈值 10%，已暂停所有策略",
        current_value=0.12,
        threshold=0.10,
    )
    nm.send_risk_alert(alert)
    print("已发送风控告警")

    # 4. 系统告警
    nm.send_system_alert("Jev决策服务断连", "无法连接 http://localhost:8765，已降级为mock")
    print("已发送系统告警")

    # 5. 每日报告
    nm.send_daily_report({
        "message": "今日收益 +1.2%，胜率 60%，最大回撤 -0.8%",
    })
    print("已发送每日报告")

    # 6. 测试渠道连通性（未启用的渠道直接返回 False）
    for ch in ("email", "wecom", "dingtalk", "serverchan", "webhook"):
        ok = nm.test_channel(ch)
        print(f"测试渠道 {ch}: {'成功' if ok else '失败/未启用'}")

    # 7. 查看状态
    status = nm.get_status()
    print("\n== 各渠道状态 ==")
    for name, st in status["channels"].items():
        print(f"  {name}: enabled={st['enabled']}")
    print(f"== 最近发送记录（{len(status['records'])} 条）==")
    for rec in status["records"][:5]:
        print(f"  [{rec['time']}] {rec['channel']} {rec['level']}/{rec['event_type']} "
              f"-> {rec['status']}")


if __name__ == "__main__":
    main()
