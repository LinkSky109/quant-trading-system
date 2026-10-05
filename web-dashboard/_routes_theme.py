"""主题偏好 API 路由（扩展模块 #29，REQ-P3-08）。

端点：
    GET  /api/theme/preference —— 读取已保存的主题偏好（无则返回 system）
    POST /api/theme/preference —— 保存主题偏好（dark / light / system）

前端 localStorage 为主存储；本路由提供跨设备同步的后端兜底，
偏好落盘 ``web-dashboard/.theme_preference.json``（重启不丢）。
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("realtime_server")

THEME_FILE = Path(__file__).resolve().parent / ".theme_preference.json"
VALID_THEMES = ["dark", "light", "system"]

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class ThemePreferenceReq(BaseModel):
    """主题偏好保存请求。"""

    theme: str


# ---------------------------------------------------------------------------
# 持久化辅助
# ---------------------------------------------------------------------------


def _read_preference() -> Optional[str]:
    """从磁盘读取偏好；文件不存在或损坏返回 None。"""
    try:
        if THEME_FILE.exists():
            data = json.loads(THEME_FILE.read_text(encoding="utf-8"))
            theme = data.get("theme")
            if theme in VALID_THEMES:
                return str(theme)
    except (json.JSONDecodeError, OSError):
        logger.warning("主题偏好文件读取失败，返回 None")
    return None


def _write_preference(theme: str) -> None:
    """原子写盘保存偏好。"""
    tmp = THEME_FILE.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"theme": theme}, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(THEME_FILE)


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_theme_routes(
    app: FastAPI,
    manager: Any,
    symbol_set: List[str],
    ok: Any,
    err: Any,
) -> None:
    """注册主题偏好路由到 FastAPI app。"""

    @app.get("/api/theme/preference")
    async def theme_get_preference():
        """读取主题偏好；未保存过返回 system（跟随系统）。"""
        try:
            with _lock:
                theme = _read_preference()
            return ok({
                "theme": theme if theme else "system",
                "saved": theme is not None,
                "valid_themes": VALID_THEMES,
            })
        except Exception as e:
            logger.exception("读取主题偏好失败")
            return err(500, f"读取失败: {e}")

    @app.post("/api/theme/preference")
    async def theme_set_preference(req: ThemePreferenceReq):
        """保存主题偏好。"""
        try:
            theme = req.theme.strip().lower()
            if theme not in VALID_THEMES:
                return err(422, f"theme 须为 {'/'.join(VALID_THEMES)} 之一")
            with _lock:
                _write_preference(theme)
            return ok({"theme": theme, "saved": True})
        except OSError as e:
            logger.exception("保存主题偏好失败")
            return err(500, f"保存失败: {e}")
        except Exception as e:
            logger.exception("保存主题偏好失败")
            return err(500, f"保存失败: {e}")
