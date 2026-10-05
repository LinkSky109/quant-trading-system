"""配置热加载模块。

在不重启进程的前提下，重新读取 ``config/config.yaml`` 并把可热加载的配置节
应用到全局配置字典（``MAIN_CFG``）。

可热加载（修改后无需重启即生效）::

    strategies / risk / notification / stock_pool / alert / rate_limit / audit

不可热加载（需重启进程才生效，热加载时跳过、不更新内存中的旧值）::

    data.api_key / jev.base_url / server port / database 路径 / logging.log_dir

特性：
  - 线程安全：``threading.RLock`` 保护配置读取、重载与回调触发，并发重载不崩溃。
  - 校验前置：重载后先用 :func:`config.validator.validate_config` 校验，
    有错误则不更新内存配置，抛出异常。
  - 变更审计：重载成功后通过 :func:`security.audit.get_audit_logger` 记录
    ``CONFIG_CHANGE`` 审计事件（审计未初始化时容错跳过）。
  - SIGHUP 触发：进程收到 SIGHUP 信号时自动重载一次（仅在主线程注册）。
  - 脱敏：``get_current_config(sanitize=True)`` 时把 api_key/token/password/
    secret 等敏感字段替换为 ``***``，避免通过 API 泄露凭据。
  - 回调：业务模块可通过 :meth:`HotReloadManager.register_callback` 注册变更
    回调，重载成功后被触发，用于让限流器 / 通知器等实例热更新自身配置。
"""
from __future__ import annotations

import copy
import json
import logging
import signal
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from config import load_config
from config.validator import validate_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 可 / 不可热加载的配置节
# ---------------------------------------------------------------------------

#: 可热加载的配置节列表（重载时这些节会被新值覆盖）
HOT_RELOADABLE_SECTIONS: List[str] = [
    "strategies",
    "risk",
    "notification",
    "stock_pool",
    "alert",
    "rate_limit",
    "audit",
]

#: 不可热加载的配置节（仅作说明 / API 展示，重载时不触碰）
NON_HOT_RELOADABLE_SECTIONS: List[str] = [
    "data",
    "jev",
    "logging",
    "security",
    "accounts",
    "backtest",
    "event_strategy",
    "multi_factor",
]

#: 敏感字段名（小写匹配），脱敏时其值替换为 "***"
SENSITIVE_KEY_NAMES = frozenset({
    "api_key",
    "token",
    "password",
    "passwd",
    "secret",
    "send_key",
    "webhook_url",
    "smtp_password",
    "key",
    "private_key",
})


class ConfigReloadError(Exception):
    """配置热加载失败（校验未通过 / 文件读取失败等）。"""


class HotReloadManager:
    """配置热加载管理器。

    用法（在 server.py 配置加载后初始化）::

        hot_reload_manager = HotReloadManager(MAIN_CFG, config_path)

        # 手动触发重载（POST /api/config/reload）
        result = hot_reload_manager.reload_config()

        # 注册回调：限流器热更新自身配置
        hot_reload_manager.register_callback(lambda changes: rate_limiter.reload(...))

    Args:
        config: 全局配置字典引用（通常是 server.py 的 ``MAIN_CFG``）。重载时
            会**原地更新**其中的可热加载节，使其它持引用的模块立即看到新值。
        config_path: ``config.yaml`` 的磁盘路径，重载时重新读取。
        register_signal: 是否注册 SIGHUP 信号处理（默认 True；单元测试可关闭）。
    """

    def __init__(
        self,
        config: Dict[str, Any],
        config_path: str,
        register_signal: bool = True,
    ) -> None:
        self._config: Dict[str, Any] = config
        self._config_path: str = config_path
        self._lock = threading.RLock()
        self._callbacks: List[Callable[[List[Dict[str, Any]]], None]] = []
        self._last_reload_ts: str = ""
        self._signal_handler_registered = False

        if register_signal:
            self._register_signal_handler()

    # ------------------------------------------------------------------
    # 基础查询
    # ------------------------------------------------------------------

    @staticmethod
    def get_hot_reloadable_keys() -> List[str]:
        """返回可热加载的配置节列表。"""
        return list(HOT_RELOADABLE_SECTIONS)

    @staticmethod
    def get_non_hot_reloadable_keys() -> List[str]:
        """返回不可热加载（需重启才生效）的配置节列表。"""
        return list(NON_HOT_RELOADABLE_SECTIONS)

    @property
    def last_reload_timestamp(self) -> str:
        """返回最近一次成功重载的 ISO 时间戳（未重载过为空串）。"""
        return self._last_reload_ts

    # ------------------------------------------------------------------
    # 变更对比
    # ------------------------------------------------------------------

    @staticmethod
    def diff_config(old: Dict[str, Any], new: Dict[str, Any]) -> List[Dict[str, Any]]:
        """对比两个配置字典，返回叶子级变更列表。

        仅对比 dict 嵌套结构：叶子节点（非 dict 值，或 list / 标量）直接做相等
        比较。list 视为整体叶子（不展开逐元素 diff）。

        Args:
            old: 旧配置字典。
            new: 新配置字典。

        Returns:
            变更列表，每项 ``{"section", "key", "old_value", "new_value"}``。
            ``key`` 为点号分隔的路径（如 ``"risk.single_stop_loss"``）。
        """
        changes: List[Dict[str, Any]] = []
        sections = set(old.keys()) | set(new.keys())
        for section in sorted(sections):
            HotReloadManager._diff_node(
                section, "", old.get(section), new.get(section), changes
            )
        return changes

    @staticmethod
    def _diff_node(
        section: str,
        prefix: str,
        old: Any,
        new: Any,
        changes: List[Dict[str, Any]],
    ) -> None:
        """递归 diff 单个节点，把叶子变更追加到 ``changes``。"""
        path = f"{prefix}.{section}" if prefix else section
        # 两侧都是 dict -> 继续递归
        if isinstance(old, dict) and isinstance(new, dict):
            keys = set(old.keys()) | set(new.keys())
            for k in sorted(keys):
                HotReloadManager._diff_node(k, path, old.get(k), new.get(k), changes)
            return
        # 否则视为叶子，直接比较
        if old != new:
            changes.append({
                "section": path.split(".")[0],
                "key": path,
                "old_value": old,
                "new_value": new,
            })

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------

    def register_callback(
        self, callback: Callable[[List[Dict[str, Any]]], None]
    ) -> None:
        """注册配置变更回调函数。

        回调签名 ``callback(changes: list[dict])``，在重载成功后于持锁状态下
        被同步调用；回调内部不应再次调用 ``reload_config``（可重入锁，安全，
        但会重入文件读取）。回调抛异常仅记录日志，不阻断重载。
        """
        with self._lock:
            self._callbacks.append(callback)

    # ------------------------------------------------------------------
    # 脱敏
    # ------------------------------------------------------------------

    def get_current_config(self, sanitize: bool = True) -> Dict[str, Any]:
        """返回当前全局配置的深拷贝。

        Args:
            sanitize: True 时把敏感字段（api_key/token/password/secret 等）
                的值替换为 ``"***"``，用于通过 API 对外展示。

        Returns:
            配置字典的副本（修改返回值不影响内存中的全局配置）。
        """
        with self._lock:
            snapshot = copy.deepcopy(self._config)
        if sanitize:
            self._sanitize(snapshot)
        return snapshot

    def _sanitize(self, node: Any) -> None:
        """原地递归脱敏：敏感键的值替换为 "***"。"""
        if isinstance(node, dict):
            for k, v in list(node.items()):
                if isinstance(k, str) and k.lower() in SENSITIVE_KEY_NAMES:
                    node[k] = "***"
                else:
                    self._sanitize(v)
        elif isinstance(node, list):
            for item in node:
                self._sanitize(item)

    # ------------------------------------------------------------------
    # 重载核心
    # ------------------------------------------------------------------

    def reload_config(self) -> Dict[str, Any]:
        """重新读取 config.yaml、校验并应用可热加载的配置节。

        线程安全：内部加锁，并发调用串行执行。

        Returns:
            ``{"reloaded": True, "changes": [...], "timestamp": iso_str}``。

        Raises:
            ConfigReloadError: 配置校验存在错误，或读取 YAML 失败。内存配置
                保持重载前的旧值不变。
        """
        with self._lock:
            # 1) 记录重载前可热加载节的快照
            old_snapshot = {
                s: copy.deepcopy(self._config.get(s))
                for s in HOT_RELOADABLE_SECTIONS
            }

            # 2) 重新读取 YAML（含环境变量覆盖）
            try:
                new_raw = load_config(self._config_path)
            except Exception as exc:  # 文件损坏 / YAML 解析失败
                logger.exception("热加载读取配置文件失败: %s", self._config_path)
                raise ConfigReloadError(f"读取配置失败: {exc}") from exc

            # 3) 校验（错误则不应用）
            result = validate_config(new_raw)
            if result.has_errors:
                raise ConfigReloadError(
                    f"配置校验未通过: {result.summary()}"
                )

            # 4) 仅应用可热加载节（原地更新共享的 _config）
            for section in HOT_RELOADABLE_SECTIONS:
                if section in new_raw:
                    self._config[section] = copy.deepcopy(new_raw[section])

            # 5) 计算变更（基于可热加载节快照）
            new_snapshot = {
                s: copy.deepcopy(self._config.get(s))
                for s in HOT_RELOADABLE_SECTIONS
            }
            changes = self.diff_config(old_snapshot, new_snapshot)

            ts = datetime.now().isoformat(timespec="seconds")
            self._last_reload_ts = ts

            # 6) 审计（容错：审计未启用 / 未初始化时不影响主流程）
            self._audit_reload(changes)

            # 7) 触发回调（逐个调用，单个失败不影响其它）
            for cb in list(self._callbacks):
                try:
                    cb(changes)
                except Exception:
                    logger.exception("配置变更回调执行失败")

            logger.info(
                "配置热加载完成: %d 项变更, timestamp=%s", len(changes), ts
            )
            return {
                "reloaded": True,
                "changes": changes,
                "timestamp": ts,
            }

    def _audit_reload(self, changes: List[Dict[str, Any]]) -> None:
        """记录 CONFIG_CHANGE 审计事件（容错，审计不可用时静默跳过）。"""
        try:
            from security.audit import ActionType, get_audit_logger
            # changes 里可能含不可 JSON 序列化的值（如 tuple），做一次清洗
            safe_changes = json.loads(
                json.dumps(changes, ensure_ascii=False, default=str)
            )
            get_audit_logger().log(
                operator="system",
                action_type=ActionType.CONFIG_CHANGE,
                target="config.yaml",
                params={"changed_sections": sorted({c["section"] for c in changes}),
                        "changes": safe_changes},
                result="success",
            )
        except Exception:
            logger.exception("记录 CONFIG_CHANGE 审计失败（已忽略）")

    # ------------------------------------------------------------------
    # SIGHUP 信号
    # ------------------------------------------------------------------

    def _register_signal_handler(self) -> None:
        """注册 SIGHUP 信号处理（仅在主线程，且支持 POSIX 信号时）。"""
        try:
            if not hasattr(signal, "SIGHUP"):
                return
            if threading.current_thread() is not threading.main_thread():
                return
            signal.signal(signal.SIGHUP, self._on_sighup)
            self._signal_handler_registered = True
            logger.info("已注册 SIGHUP 信号处理：收到 SIGHUP 自动热加载配置")
        except Exception:
            # 非主线程 / Windows / 测试环境无信号支持时静默跳过
            logger.debug("SIGHUP 信号注册失败（已忽略）")

    def _on_sighup(self, signum: int, frame: Any) -> None:
        """SIGHUP 信号处理回调：触发一次热加载。"""
        logger.info("收到 SIGHUP 信号，开始热加载配置...")
        try:
            self.reload_config()
        except Exception:
            logger.exception("SIGHUP 触发的热加载失败")
