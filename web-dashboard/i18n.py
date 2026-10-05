"""国际化（i18n）支持（REQ-P3-07）。

组成：
  - ``LANG_PACKS``：后端语言包（至少中文/英文），覆盖 API 错误消息、
    通知与常用日志文案
  - ``t(lang, key, **kwargs)``：按语言取文案（缺 key 回退英文，再回退 key）
  - ``normalize_accept_language(header)``：解析 ``Accept-Language`` header
  - 格式化工具：按地区格式化货币 / 数字 / 百分比 / 日期

前端语言包与切换器见 ``quant_dashboard_realtime.html`` 的 ``I18nModule``。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

SUPPORTED_LANGS: List[str] = ["zh", "en"]
DEFAULT_LANG = "zh"

# --------------------------------------------------------------------------- #
# 语言包（zh / en 完整对齐；新增语言需保持 key 一致）
# --------------------------------------------------------------------------- #

LANG_PACKS: Dict[str, Dict[str, str]] = {
    "zh": {
        # ---- API 消息 ----
        "api.success": "成功",
        "api.bad_request": "请求参数错误",
        "api.not_found": "资源不存在",
        "api.server_error": "服务器内部错误",
        "api.unauthorized": "未授权访问",
        "api.symbol_required": "标的代码不能为空",
        "api.symbol_not_found": "未找到 {symbol} 的数据",
        "api.invalid_date_range": "日期区间不合法",
        "api.rate_limited": "请求过于频繁，请稍后重试",
        # ---- 通知 ----
        "notify.trade_executed": "交易已执行：{symbol} {action} {quantity} 股",
        "notify.risk_alert": "风控告警：{message}",
        "notify.strategy_signal": "策略信号：{strategy} 在 {symbol} 发出 {action} 信号",
        "notify.data_updated": "数据已更新：{symbol}",
        # ---- 日志 ----
        "log.engine_started": "引擎已启动",
        "log.engine_stopped": "引擎已停止",
        "log.backtest_finished": "回测完成，共 {count} 笔交易",
        # ---- 前端主要 UI ----
        "ui.title": "量化交易系统 · 实时看板",
        "ui.subtitle": "多策略并行 · AI 决策 · 实时风控",
        "ui.mode_compare": "策略对比",
        "ui.mode_backtest": "回测",
        "ui.mode_portfolio": "组合回测",
        "ui.mode_optimize": "优化",
        "ui.mode_attribution": "归因",
        "ui.connect_status.connected": "已连接",
        "ui.connect_status.disconnected": "连接断开",
        "ui.language": "语言",
    },
    "en": {
        "api.success": "Success",
        "api.bad_request": "Bad request",
        "api.not_found": "Resource not found",
        "api.server_error": "Internal server error",
        "api.unauthorized": "Unauthorized access",
        "api.symbol_required": "Symbol is required",
        "api.symbol_not_found": "No data found for {symbol}",
        "api.invalid_date_range": "Invalid date range",
        "api.rate_limited": "Too many requests, please retry later",
        "notify.trade_executed": "Trade executed: {symbol} {action} {quantity} shares",
        "notify.risk_alert": "Risk alert: {message}",
        "notify.strategy_signal": "Strategy signal: {strategy} {action} on {symbol}",
        "notify.data_updated": "Data updated: {symbol}",
        "log.engine_started": "Engine started",
        "log.engine_stopped": "Engine stopped",
        "log.backtest_finished": "Backtest finished with {count} trades",
        "ui.title": "Quant Trading System · Realtime Dashboard",
        "ui.subtitle": "Multi-strategy · AI decision · Realtime risk control",
        "ui.mode_compare": "Compare",
        "ui.mode_backtest": "Backtest",
        "ui.mode_portfolio": "Portfolio",
        "ui.mode_optimize": "Optimize",
        "ui.mode_attribution": "Attribution",
        "ui.connect_status.connected": "Connected",
        "ui.connect_status.disconnected": "Disconnected",
        "ui.language": "Language",
    },
}


# --------------------------------------------------------------------------- #
# 文案取用
# --------------------------------------------------------------------------- #


def normalize_lang(lang: Optional[str]) -> str:
    """把任意语言标记归一到支持语言（'zh-CN' -> 'zh'；未知回退默认）。"""
    if not lang:
        return DEFAULT_LANG
    code = lang.strip().lower().split(",")[0].split(";")[0]
    short = code.split("-")[0]
    return short if short in SUPPORTED_LANGS else DEFAULT_LANG


def normalize_accept_language(header: Optional[str]) -> str:
    """解析 Accept-Language header，取权重最高的支持语言。

    例：``"en-US,en;q=0.9,zh-CN;q=0.8"`` -> ``"en"``。
    """
    if not header:
        return DEFAULT_LANG
    parts: List[tuple] = []
    for item in header.split(","):
        seg = item.strip()
        if not seg:
            continue
        if ";" in seg:
            tag, q = seg.split(";", 1)
            try:
                weight = float(q.strip().split("=")[1])
            except (IndexError, ValueError):
                weight = 1.0
        else:
            tag, weight = seg, 1.0
        parts.append((tag.strip(), weight))
    # 权重降序（稳定排序保持同权重时的原始顺序）
    parts.sort(key=lambda p: -p[1])
    for tag, _ in parts:
        lang = normalize_lang(tag)
        # 只有 tag 本身能映射到支持语言时才算命中（避免 en 请求被回退 zh）
        if tag.lower().split("-")[0] in SUPPORTED_LANGS:
            return lang
    return DEFAULT_LANG


def t(lang: Optional[str], key: str, **kwargs: Any) -> str:
    """按语言取文案；缺 key 回退英文，再回退 key 本身。支持 {占位} 填充。"""
    lang = normalize_lang(lang)
    text = LANG_PACKS.get(lang, {}).get(key)
    if text is None:
        text = LANG_PACKS["en"].get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text


# --------------------------------------------------------------------------- #
# 地区格式化
# --------------------------------------------------------------------------- #

# 数字分组：zh/en 均千分位；小数点均 '.'（覆盖中英即可，其他语言走 en）
_LOCALE_FMT = {
    "zh": {"thousands": ",", "decimal": ".", "currency": "¥"},
    "en": {"thousands": ",", "decimal": ".", "currency": "$"},
}


def format_number(value: float, lang: Optional[str] = None, precision: int = 2) -> str:
    """按地区格式化数字（千分位分组 + 固定小数位）。"""
    lang = normalize_lang(lang)
    fmt = _LOCALE_FMT.get(lang, _LOCALE_FMT["en"])
    s = f"{float(value):,.{precision}f}"
    return s  # zh/en 分组符一致；预留其他地区扩展点


def format_currency(value: float, lang: Optional[str] = None, precision: int = 2) -> str:
    """按地区格式化货币金额（货币符号随语言）。"""
    lang = normalize_lang(lang)
    fmt = _LOCALE_FMT.get(lang, _LOCALE_FMT["en"])
    return f"{fmt['currency']}{format_number(value, lang, precision)}"


def format_percent(value: float, lang: Optional[str] = None, precision: int = 2) -> str:
    """格式化百分比（输入为小数比例，0.1234 -> 12.34%）。"""
    return f"{float(value) * 100:.{precision}f}%"


def format_date(dt: Any, lang: Optional[str] = None) -> str:
    """按地区格式化日期：zh -> 2026-01-31；en -> Jan 31, 2026。"""
    lang = normalize_lang(lang)
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00").split(" ")[0])
    if lang == "en":
        return dt.strftime("%b %d, %Y")
    return dt.strftime("%Y-%m-%d")
