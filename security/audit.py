"""安全审计日志模块。

记录所有交易操作 / 配置变更 / 登录行为的不可篡改审计日志，支持哈希链
完整性校验。

特性：
  - 双写：同时写入 SQLite ``audit_logs`` 表 与 JSONL 文件（``logs/audit.jsonl``，
    追加模式），任一存储被篡改都可通过交叉比对发现。
  - 哈希链：每条记录的 ``hash`` 覆盖前一条记录的 ``hash``，首条记录的
    ``prev_hash`` 为 64 个零（创世哈希）。任何对历史记录的篡改都会破坏后续
    整条链。
  - 线程安全：``threading.Lock`` 保护哈希链状态与文件写入，多线程并发写入
    不丢数据、不断链。
  - 完整性校验：``verify_integrity()`` 从头遍历逐条重算 hash 并比对 prev_hash。

哈希计算::

    hash = SHA-256(timestamp | operator | action_type | target | params_json
                    | result | ip | request_id | prev_hash)
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 创世哈希：链中第一条记录的 prev_hash 固定为 64 个零
GENESIS_HASH = "0" * 64

# 字段分隔符（单元分隔符 \x1f，避免与正常文本冲突）
_FIELD_SEP = "\x1f"


class ActionType(str, Enum):
    """审计操作类型枚举。

    继承 ``str`` 以便直接序列化为 JSON / 存入 SQLite TEXT 列。
    """

    LOGIN = "LOGIN"                      # 登录认证
    LOGOUT = "LOGOUT"                    # 登出
    ORDER_SUBMIT = "ORDER_SUBMIT"        # 委托下单
    ORDER_CANCEL = "ORDER_CANCEL"        # 委托撤单
    POSITION_CLOSE = "POSITION_CLOSE"    # 平仓
    CONFIG_CHANGE = "CONFIG_CHANGE"      # 配置变更
    STRATEGY_TOGGLE = "STRATEGY_TOGGLE"  # 策略开关
    RISK_TRIGGER = "RISK_TRIGGER"        # 风控触发
    BACKUP_RESTORE = "BACKUP_RESTORE"    # 备份恢复
    API_TOKEN_CREATE = "API_TOKEN_CREATE"  # API Token 创建


def _compute_hash(
    timestamp: str,
    operator: str,
    action_type: str,
    target: str,
    params_json: str,
    result: str,
    ip: str,
    request_id: str,
    prev_hash: str,
) -> str:
    """计算单条审计记录的 SHA-256 哈希。

    Args:
        timestamp: 记录时间 ISO 字符串。
        operator: 操作人名称。
        action_type: 操作类型（ActionType 的 value）。
        target: 操作目标（如标的代码 / 配置项名）。
        params_json: 参数字典的 JSON 字符串（已序列化）。
        result: 操作结果描述（success / failure / 错误信息）。
        ip: 来源 IP 地址。
        request_id: 请求唯一标识。
        prev_hash: 前一条记录的 hash。

    Returns:
        64 位 hex 摘要字符串。
    """
    payload = _FIELD_SEP.join([
        timestamp, operator, action_type, target, params_json,
        result, ip, request_id, prev_hash,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLogger:
    """安全审计日志记录器（双写 SQLite + JSONL，哈希链防篡改）。

    用法::

        logger = AuditLogger(config_dict, db_path="data/quant_trading.db")
        logger.log(
            operator="admin",
            action_type=ActionType.LOGIN,
            ip="127.0.0.1",
        )
        report = logger.verify_integrity()
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]],
        db_path: Optional[str] = None,
    ) -> None:
        """初始化审计日志记录器。

        Args:
            config: ``config.yaml`` 中 ``audit`` 节的字典，支持字段：
                ``enabled``(bool)、``log_file``(str)、``retention_days``(int)。
                为 None 时按空配置处理（默认 disabled）。
            db_path: SQLite 数据库文件路径；为 None 时不写 SQLite，仅写 JSONL。
        """
        cfg: Dict[str, Any] = dict(config or {})
        self._enabled: bool = bool(cfg.get("enabled", False))
        self._log_file: str = str(cfg.get("log_file", "./logs/audit.jsonl"))
        self._retention_days: int = int(cfg.get("retention_days", 90))

        # 延迟导入避免循环依赖（persistence.database 不依赖 security.audit）
        self._db = None
        if db_path:
            from persistence.database import Database
            self._db = Database(db_path=db_path)

        # 哈希链状态：最近一条记录的 hash（首条为创世哈希）
        self._last_hash: str = GENESIS_HASH
        self._lock = threading.Lock()

        if self._enabled:
            # 启动时从数据库恢复哈希链尾部，避免重启后断链
            if self._db is not None:
                try:
                    last = self._db.get_last_audit_hash()
                    if last:
                        self._last_hash = last
                except Exception:
                    logger.exception("加载审计哈希链尾部失败，从创世哈希重建")
            # 确保日志目录存在
            Path(self._log_file).parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                "审计日志已启用: db=%s, jsonl=%s, 链尾=%s",
                db_path or "无", self._log_file,
                self._last_hash[:16] + "..." if self._last_hash != GENESIS_HASH else "创世",
            )
        else:
            logger.info("审计日志未启用（audit.enabled=false）")

    # ------------------------------------------------------------------
    # 基础属性
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """返回是否启用审计记录。"""
        return self._enabled

    @property
    def last_hash(self) -> str:
        """返回当前哈希链尾部 hash（线程安全读取）。"""
        with self._lock:
            return self._last_hash

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def log(
        self,
        operator: str,
        action_type: ActionType | str,
        target: str = "",
        params: Optional[Dict[str, Any]] = None,
        result: str = "success",
        ip: str = "",
        request_id: str = "",
        timestamp: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """写入一条审计记录（双写 SQLite + JSONL，线程安全）。

        Args:
            operator: 操作人名称（token name / 系统标识）。
            action_type: 操作类型枚举值或其字符串。
            target: 操作目标（标的代码 / 配置项名 / 备份文件名等）。
            params: 操作参数字典（自动序列化为 JSON）。
            result: 操作结果描述（"success" / "failure: xxx"）。
            ip: 请求来源 IP。
            request_id: 请求唯一标识（可用于链路追踪）。
            timestamp: 记录时间 ISO 字符串，默认当前时间。

        Returns:
            写入的记录字典（含 hash）；审计未启用时返回 None。
        """
        if not self._enabled:
            return None

        if isinstance(action_type, ActionType):
            action_str = action_type.value
        else:
            action_str = str(action_type)

        ts = timestamp or datetime.now().isoformat(timespec="seconds")
        params_json = json.dumps(params or {}, ensure_ascii=False, sort_keys=True)

        with self._lock:
            prev_hash = self._last_hash
            entry_hash = _compute_hash(
                timestamp=ts,
                operator=operator,
                action_type=action_str,
                target=target,
                params_json=params_json,
                result=result,
                ip=ip,
                request_id=request_id,
                prev_hash=prev_hash,
            )

            record: Dict[str, Any] = {
                "timestamp": ts,
                "operator": operator,
                "action_type": action_str,
                "target": target,
                "params": params_json,
                "result": result,
                "ip": ip,
                "request_id": request_id,
                "prev_hash": prev_hash,
                "hash": entry_hash,
            }

            # 1) 写 SQLite
            if self._db is not None:
                try:
                    self._db.insert_audit_log(
                        timestamp=ts,
                        operator=operator,
                        action_type=action_str,
                        target=target,
                        params=params_json,
                        result=result,
                        ip=ip,
                        request_id=request_id,
                        prev_hash=prev_hash,
                        entry_hash=entry_hash,
                    )
                except Exception:
                    logger.exception("写入 SQLite 审计日志失败")

            # 2) 写 JSONL（追加一行）
            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                logger.exception("写入 JSONL 审计日志失败: %s", self._log_file)

            self._last_hash = entry_hash

        return record

    # ------------------------------------------------------------------
    # 完整性校验
    # ------------------------------------------------------------------

    def verify_integrity(self) -> Dict[str, Any]:
        """从头遍历审计日志，逐条验证 hash 与 prev_hash 链。

        Returns:
            校验结果字典::

                {
                    "valid": bool,           # 整条链是否完整
                    "total_records": int,    # 总记录数
                    "first_hash": str,       # 首条记录 hash（空表为空串）
                    "last_hash": str,        # 末条记录 hash（空表为创世哈希）
                    "broken_at": int,        # 断裂处记录 id（未断裂为 0）
                    "error": str,            # 断裂描述（无错误为空串）
                }
        """
        result: Dict[str, Any] = {
            "valid": True,
            "total_records": 0,
            "first_hash": "",
            "last_hash": GENESIS_HASH,
            "broken_at": 0,
            "error": "",
        }

        if self._db is None:
            result["error"] = "未配置数据库，无法校验"
            result["valid"] = False
            return result

        try:
            rows = self._db.query_audit_logs(limit=100000)
        except Exception:
            logger.exception("查询审计日志失败")
            result["valid"] = False
            result["error"] = "查询审计日志失败"
            return result

        result["total_records"] = len(rows)
        if not rows:
            return result

        result["first_hash"] = rows[0]["hash"]
        expected_prev = GENESIS_HASH

        for row in rows:
            # 重算 hash
            recomputed = _compute_hash(
                timestamp=row["timestamp"],
                operator=row["operator"],
                action_type=row["action_type"],
                target=row.get("target", ""),
                params_json=row.get("params", "{}"),
                result=row.get("result", ""),
                ip=row.get("ip", ""),
                request_id=row.get("request_id", ""),
                prev_hash=row.get("prev_hash", ""),
            )
            # 检查 prev_hash 链接
            if row.get("prev_hash", "") != expected_prev:
                result["valid"] = False
                result["broken_at"] = row["id"]
                result["error"] = (
                    f"prev_hash 不匹配: 记录 id={row['id']} "
                    f"期望 prev_hash={expected_prev[:16]}... "
                    f"实际={row.get('prev_hash', '')[:16]}..."
                )
                result["last_hash"] = row["hash"]
                return result
            # 检查本条 hash
            if recomputed != row["hash"]:
                result["valid"] = False
                result["broken_at"] = row["id"]
                result["error"] = (
                    f"hash 不匹配: 记录 id={row['id']} 被篡改 "
                    f"(重算={recomputed[:16]}..., 存储={row['hash'][:16]}...)"
                )
                result["last_hash"] = row["hash"]
                return result
            expected_prev = row["hash"]

        result["last_hash"] = rows[-1]["hash"]
        return result

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def query_logs(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        action_type: Optional[str] = None,
        operator: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        """分页查询审计日志。

        Args:
            start: 起始时间（ISO 字符串，含），可选。
            end: 结束时间（ISO 字符串，含），可选。
            action_type: 按操作类型精确过滤，可选。
            operator: 按操作人精确过滤，可选。
            page: 页码（从 1 开始）。
            page_size: 每页条数。

        Returns:
            ``{"total": int, "page": int, "page_size": int, "logs": list}``。
        """
        page = max(1, int(page))
        page_size = max(1, min(500, int(page_size)))
        offset = (page - 1) * page_size

        if self._db is None:
            return {"total": 0, "page": page, "page_size": page_size, "logs": []}

        total = self._db.count_audit_logs(
            start=start, end=end,
            action_type=action_type, operator=operator,
        )
        logs = self._db.query_audit_logs(
            start=start, end=end,
            action_type=action_type, operator=operator,
            limit=page_size, offset=offset,
        )
        return {"total": total, "page": page, "page_size": page_size, "logs": logs}

    # ------------------------------------------------------------------
    # 日志保留清理（预留）
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """清理超过 retention_days 天的旧审计记录（SQLite 侧）。

        Returns:
            清理的记录数。JSONL 文件按追加模式保留，不截断。
        """
        if self._db is None or self._retention_days <= 0:
            return 0
        cutoff = (datetime.now() - timedelta(days=self._retention_days)).isoformat(
            timespec="seconds"
        )
        try:
            conn = self._db._get_conn()
            cursor = conn.execute(
                "DELETE FROM audit_logs WHERE timestamp < ?", (cutoff,)
            )
            conn.commit()
            removed = cursor.rowcount
            if removed:
                logger.info("清理过期审计记录 %d 条（cutoff=%s）", removed, cutoff)
            return removed
        except Exception:
            logger.exception("清理过期审计记录失败")
            return 0


# ---------------------------------------------------------------------------
# 模块级单例
# ---------------------------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """获取模块级 AuditLogger 单例。

    首次调用前需先调用 :func:`init_audit_logger` 完成初始化（通常在
    server.py 启动时）；未初始化时返回一个 disabled 的空实例。
    """
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = AuditLogger(None)
        return _singleton


def init_audit_logger(
    config: Optional[Dict[str, Any]],
    db_path: Optional[str] = None,
) -> AuditLogger:
    """初始化模块级单例（server.py 启动时调用）。

    Args:
        config: ``audit`` 节配置字典。
        db_path: SQLite 数据库路径。

    Returns:
        初始化后的 AuditLogger 实例。
    """
    global _singleton
    with _singleton_lock:
        _singleton = AuditLogger(config=config, db_path=db_path)
        return _singleton
