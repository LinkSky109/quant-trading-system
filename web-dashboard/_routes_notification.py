# -*- coding: utf-8 -*-
"""通知管理 API 路由片段（待合并进 web-dashboard/server.py）。

【合并说明】
本文件不直接运行，仅提供可粘贴的路由代码。在 server.py 中：

  1. 顶部 import 区追加：
         from monitoring.notifier import NotifierManager  # noqa: E402

  2. 加载 MAIN_CFG 之后，初始化全局通知管理器：
         notifier_manager = NotifierManager(MAIN_CFG.get("notification", {}))

  3. 将下方所有代码（Req 模型 + 路由函数）复制到 server.py 路由区任意位置。

API 契约保持 {code, message, data}，使用 ok()/err() 辅助函数，与现有路由一致。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------

class NotificationTestReq(BaseModel):
    """发送测试通知请求体。"""
    channel: str = "wecom"          # email / wecom / dingtalk / serverchan / webhook
    message: str = "连通性测试"


class NotificationSendReq(BaseModel):
    """手动发送通知请求体。"""
    level: str = "WARNING"          # CRITICAL / WARNING / INFO
    event_type: str = "system"      # trade / risk / system / daily
    title: str = "手动通知"
    context: Dict[str, Any] = {}
    template: str = ""              # 自定义模板（可选）


# ---------------------------------------------------------------------------
# 路由（依赖 server.py 中已存在的 app / ok / err / notifier_manager）
# ---------------------------------------------------------------------------

# @app.post("/api/notification/test")
async def notification_test(req: NotificationTestReq):
    """发送测试通知到指定渠道。"""
    valid = {"email", "wecom", "dingtalk", "serverchan", "webhook"}
    if req.channel not in valid:
        return err(40001, f"未知渠道: {req.channel}，可选: {sorted(valid)}")
    try:
        ok_flag = notifier_manager.test_channel(req.channel)
        return ok({"channel": req.channel, "success": ok_flag},
                  message="测试通知已发送" if ok_flag else "发送失败或渠道未启用")
    except Exception as e:
        logger.exception("通知渠道测试失败")
        return err(50001, f"渠道测试异常: {e}")


# @app.get("/api/notification/status")
async def notification_status():
    """查询各渠道配置状态与最近发送记录。"""
    try:
        return ok(notifier_manager.get_status())
    except Exception as e:
        logger.exception("查询通知状态失败")
        return err(50002, f"查询状态异常: {e}")


# @app.post("/api/notification/send")
async def notification_send(req: NotificationSendReq):
    """手动发送一条通知（按路由规则自动分发渠道）。"""
    valid_levels = {"CRITICAL", "WARNING", "INFO"}
    if req.level not in valid_levels:
        return err(40002, f"未知级别: {req.level}，可选: {sorted(valid_levels)}")
    try:
        result = notifier_manager.send(
            level=req.level,
            event_type=req.event_type,
            title=req.title,
            context=req.context,
            template=req.template,
        )
        return ok({"result": result}, message="通知已按路由规则分发")
    except Exception as e:
        logger.exception("手动发送通知失败")
        return err(50003, f"发送异常: {e}")


# ---------------------------------------------------------------------------
# 【粘贴后请取消下方装饰器注释，并确认函数名不与现有路由冲突】
# ---------------------------------------------------------------------------
# app.add_api_route("/api/notification/test", notification_test, methods=["POST"])
# app.add_api_route("/api/notification/status", notification_status, methods=["GET"])
# app.add_api_route("/api/notification/send", notification_send, methods=["POST"])
