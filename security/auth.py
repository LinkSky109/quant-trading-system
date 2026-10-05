"""API Token 认证与权限管理模块。

通过 ``config.yaml`` 中的 ``security`` 节配置 token 列表，支持三级权限：

- ``read``：只读，可访问行情/账户/回测等查询类接口
- ``trade``：可交易，包含 read 的全部权限，额外允许开关实盘交易
- ``admin``：全部权限，包含 trade 和 read，额外允许触发告警、导出报表等管理操作

Token 通过 HTTP Header ``Authorization: Bearer <token>`` 传递；
WebSocket 通过 query param ``?token=<token>`` 传递。
"""
from __future__ import annotations

import logging
import secrets
from datetime import date, datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("security.auth")

# 权限层级数值：数值越大权限越高（高层级自动包含低层级）
_PERMISSION_LEVELS: Dict[str, int] = {
    "read": 1,
    "trade": 2,
    "admin": 3,
}

_VALID_PERMISSIONS = set(_PERMISSION_LEVELS.keys())


class AuthManager:
    """管理 API Token 的加载、校验与权限检查。

    Token 配置格式（``config.yaml`` 的 ``security.tokens``）::

        tokens:
          - name: "admin-token"        # token 备注名
            token: "xxxxxxxxxxxxxxxx"   # 实际 token 值（用 AuthManager.generate_token() 生成）
            permission: "admin"        # read / trade / admin
            expires_at: ""             # 空字符串=永不过期，否则 "YYYY-MM-DD"

    注意：
        ``enabled`` 只控制中间件是否强制认证；``verify_token`` 本身始终可用于
        校验任意 token，与 ``enabled`` 无关。
    """

    def __init__(self, config: Optional[Dict[str, Any]]):
        """初始化 AuthManager。

        Args:
            config: ``config.yaml`` 中 ``security`` 节的字典，需包含
                ``enabled``(bool) 与 ``tokens``(list[dict])；为 None 时按空配置处理。
        """
        cfg: Dict[str, Any] = dict(config or {})
        self._enabled: bool = bool(cfg.get("enabled", False))

        raw_tokens: List[Dict[str, Any]] = cfg.get("tokens") or []
        # 以 token 字符串为键建立索引，O(1) 校验
        self._tokens: Dict[str, Dict[str, Any]] = {}
        for item in raw_tokens:
            if not isinstance(item, dict):
                continue
            tok = str(item.get("token") or "").strip()
            if not tok:
                logger.warning("security.tokens 中存在空 token 条目，已跳过")
                continue
            permission = str(item.get("permission", "read"))
            if permission not in _VALID_PERMISSIONS:
                logger.warning(
                    "token %s*** 的未知权限级别 %r，按 read 处理",
                    tok[:4], permission,
                )
                permission = "read"
            self._tokens[tok] = {
                "name": str(item.get("name", "")),
                "permission": permission,
                "expires_at": str(item.get("expires_at") or ""),
            }

    # ------------------------------------------------------------------
    # 基础查询
    # ------------------------------------------------------------------
    def is_enabled(self) -> bool:
        """返回是否启用了强制认证。"""
        return self._enabled

    def has_tokens(self) -> bool:
        """是否已配置至少一个有效 token（用于启动时告警）。"""
        return bool(self._tokens)

    @staticmethod
    def generate_token() -> str:
        """生成一个随机 token（32 字节，64 位 hex 字符串）。"""
        return secrets.token_hex(32)

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def verify_token(self, token: Optional[str]) -> Optional[Dict[str, Any]]:
        """验证 token 是否存在且未过期。

        Args:
            token: 请求头中携带的原始 token 字符串。

        Returns:
            有效时返回 ``{"name", "permission", "expires_at"}`` 的副本；
            token 为空、不存在或已过期时返回 ``None``。
        """
        if not token:
            return None
        info = self._tokens.get(token)
        if info is None:
            return None

        expires_at = info.get("expires_at") or ""
        if expires_at:
            try:
                expire_date = datetime.strptime(expires_at, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                logger.warning(
                    "token %s*** 的 expires_at=%r 无法解析为 YYYY-MM-DD，按永不过期处理",
                    token[:4], expires_at,
                )
                return dict(info)
            if date.today() > expire_date:
                return None
        return dict(info)

    def has_permission(
        self, token_info: Optional[Dict[str, Any]], required: str
    ) -> bool:
        """检查 token 信息是否满足所需权限。

        权限层级：``admin`` > ``trade`` > ``read``。高层级可访问低层级资源。

        Args:
            token_info: ``verify_token`` 返回的 token 信息字典。
            required: 所需权限级别（``read`` / ``trade`` / ``admin``）。

        Returns:
            满足返回 True，否则 False。
        """
        if not token_info:
            return False
        user_level = _PERMISSION_LEVELS.get(token_info.get("permission", ""), 0)
        required_level = _PERMISSION_LEVELS.get(required, 0)
        return required_level > 0 and user_level >= required_level
