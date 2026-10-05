"""国际化 (i18n) API 路由（扩展模块 #28，REQ-P3-07）。

端点：
    GET  /api/i18n/languages —— 支持的语言列表
    GET  /api/i18n/messages  —— 语言包（lang 参数或 Accept-Language header，缺 key 回退默认语言）
    GET  /api/i18n/text      —— 取单条文案
    POST /api/i18n/format    —— 按地区格式化数字 / 货币 / 百分比 / 日期
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from pydantic import BaseModel

from i18n import (
    DEFAULT_LANG,
    LANG_PACKS,
    SUPPORTED_LANGS,
    format_currency,
    format_date,
    format_number,
    format_percent,
    normalize_accept_language,
    normalize_lang,
    t,
)

logger = logging.getLogger("realtime_server")


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class I18nFormatItem(BaseModel):
    """单个格式化任务。"""

    type: str  # number | currency | percent | date
    value: Union[float, str]  # date 类型接受 ISO 字符串
    precision: int = 2


class I18nFormatReq(BaseModel):
    """批量格式化请求。"""

    lang: Optional[str] = None
    items: List[I18nFormatItem]


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_i18n_routes(
    app: FastAPI,
    manager: Any,
    symbol_set: List[str],
    ok: Any,
    err: Any,
) -> None:
    """注册 i18n 路由到 FastAPI app。"""

    @app.get("/api/i18n/languages")
    async def i18n_languages():
        """查询支持的语言列表。"""
        try:
            return ok({
                "supported": SUPPORTED_LANGS,
                "default": DEFAULT_LANG,
                "names": {"zh": "中文", "en": "English"},
            })
        except Exception as e:
            logger.exception("查询语言列表失败")
            return err(500, f"查询失败: {e}")

    @app.get("/api/i18n/messages")
    async def i18n_messages(request: Request, lang: Optional[str] = None):
        """返回语言包；语言优先级：lang 参数 > Accept-Language header > 默认。"""
        try:
            header = request.headers.get("accept-language")
            resolved = normalize_lang(lang) if lang else normalize_accept_language(header)
            # 缺失 key 回退默认语言，保证前端 key 对齐
            merged: Dict[str, str] = dict(LANG_PACKS.get(DEFAULT_LANG, {}))
            merged.update(LANG_PACKS.get(resolved, {}))
            return ok({
                "lang": resolved,
                "accept_language": header,
                "messages": merged,
            })
        except Exception as e:
            logger.exception("查询语言包失败")
            return err(500, f"查询失败: {e}")

    @app.get("/api/i18n/text")
    async def i18n_text(request: Request, key: str, lang: Optional[str] = None):
        """取单条文案（t() 内置 en 回退与 {placeholder} 插值）。

        额外 query 参数作为插值变量，如 ``?key=api.symbol_not_found&symbol=AAPL``。
        """
        try:
            if not key or not key.strip():
                return err(400, "key 不能为空")
            if lang:
                resolved = normalize_lang(lang)
            else:
                resolved = normalize_accept_language(request.headers.get("accept-language"))
            kwargs = {
                k: v for k, v in request.query_params.items()
                if k not in ("key", "lang")
            }
            return ok({
                "lang": resolved,
                "key": key.strip(),
                "text": t(resolved, key.strip(), **kwargs),
            })
        except Exception as e:
            logger.exception("查询文案失败")
            return err(500, f"查询失败: {e}")

    @app.post("/api/i18n/format")
    async def i18n_format(req: I18nFormatReq):
        """按地区格式化数字 / 货币 / 百分比 / 日期。"""
        try:
            lang = normalize_lang(req.lang)
            results: List[Dict[str, Any]] = []
            for i, item in enumerate(req.items):
                try:
                    if item.type == "number":
                        text = format_number(float(item.value), lang, item.precision)
                    elif item.type == "currency":
                        text = format_currency(float(item.value), lang, item.precision)
                    elif item.type == "percent":
                        text = format_percent(float(item.value), lang, item.precision)
                    elif item.type == "date":
                        text = format_date(item.value, lang)
                    else:
                        return err(422, f"items[{i}].type 不支持: {item.type}")
                    results.append({"index": i, "type": item.type, "lang": lang, "text": text})
                except (TypeError, ValueError) as e:
                    return err(422, f"items[{i}] 格式化失败: {e}")
            return ok({"lang": lang, "results": results})
        except Exception as e:
            logger.exception("格式化失败")
            return err(500, f"格式化失败: {e}")
