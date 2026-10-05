"""结构化 JSON 日志与请求链路 ID 模块。

提供：
  - :class:`StructuredJSONFormatter`：把 ``logging.LogRecord`` 序列化为单行 JSON，
    字段含 ``timestamp``(ISO8601) / ``level`` / ``module`` / ``function`` /
    ``line`` / ``message`` / ``request_id`` 及业务 extra 字段。
  - :class:`RequestIDMiddleware`：纯 ASGI 中间件，为每个 HTTP 请求生成唯一
    ``request_id``（uuid4 hex 前 8 位），写入 contextvars 与响应头
    ``X-Request-ID``。必须作为**最外层中间件**注册，使后续所有中间件 / 路由 /
    日志都能拿到同一个 request_id。
  - :func:`setup_structured_logging`：为根 logger 追加 JSON 格式的滚动文件
    handler（``logs/structured.jsonl``），并为 trades / alerts / jev 三个
    独立 logger 各建一个 JSON 文件；控制台保持现有人类可读格式不动。
  - :func:`get_recent_logs` / :func:`set_log_level` / :func:`get_log_files`：
    供 ``/api/logs/*`` 端点查询与动态调级。

请求 ID 在异步链路中的传递：本中间件采用**纯 ASGI** 实现（而非 BaseHTTP
Middleware），``await self.app(...)`` 与端点运行在同一任务/上下文里，因此
在调用前 ``contextvars.set`` 的 request_id 能被端点内的日志记录器读到。
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# request_id contextvar
# ---------------------------------------------------------------------------

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """获取当前请求的 request_id（无请求上下文时返回空串）。"""
    return _request_id_ctx.get()


# LogRecord 内置属性（输出 JSON 时不作为 extra 字段重复输出）
_RESERVED_RECORD_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
})


class StructuredJSONFormatter(logging.Formatter):
    """把日志记录格式化为单行 JSON。

    输出字段：
      - ``timestamp``: ISO8601（含秒）
      - ``level``: DEBUG/INFO/WARNING/ERROR/CRITICAL
      - ``module`` / ``function`` / ``line``: 代码位置
      - ``message``: 格式化后的日志文本
      - ``request_id``: 来自 contextvars（无请求时为空串）
      - 其它业务 extra 字段（``logger.info("...", extra={"k": v})``）
      - ``exc_info``: 异常堆栈（如有）
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(
                timespec="seconds"
            ),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }

        # 合并业务 extra 字段（跳过 LogRecord 内置属性）
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_ATTRS:
                continue
            payload[key] = _safe_json_value(value)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def _safe_json_value(value: Any) -> Any:
    """尝试 JSON 序列化，失败时转字符串。"""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# RequestIDMiddleware（纯 ASGI）
# ---------------------------------------------------------------------------

class RequestIDMiddleware:
    """为每个 HTTP 请求生成并透传 request_id 的纯 ASGI 中间件。

    - 若客户端已带 ``X-Request-ID`` 请求头则沿用，否则生成 uuid4 hex 前 8 位。
    - 把 request_id 写入 contextvars（供日志 formatter 读取）。
    - 把 ``X-Request-ID`` 注入请求 scope 头（供下游中间件/路由读取
      ``request.headers``），并在响应头回写。
    - WebSocket / 其它 scope 类型直接放行，不做处理。
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # 读取请求头中已有的 X-Request-ID
        incoming = ""
        for k, v in scope.get("headers", []):
            if k == b"x-request-id":
                incoming = v.decode("latin-1")
                break
        rid = incoming or uuid.uuid4().hex[:8]

        # 注入到请求头 scope，下游 request.headers 可见
        scope.setdefault("headers", []).append((b"x-request-id", rid.encode("latin-1")))

        token = _request_id_ctx.set(rid)
        try:
            async def send_wrapper(message: Dict[str, Any]) -> None:
                if message["type"] == "http.response.start":
                    headers = message.setdefault("headers", [])
                    headers.append((b"x-request-id", rid.encode("latin-1")))
                await send(message)

            await self.app(scope, receive, send_wrapper)
        finally:
            _request_id_ctx.reset(token)


# ---------------------------------------------------------------------------
# 日志文件路径注册（供 get_recent_logs / get_log_files 查询）
# ---------------------------------------------------------------------------

#: 已注册的日志文件：{逻辑名: 绝对路径}
_ACTIVE_LOG_FILES: Dict[str, str] = {}


def _resolve_log_dir(config: Dict[str, Any]) -> Path:
    """从 logging 节解析日志目录，默认 ./logs。"""
    log_dir = str(config.get("log_dir", "./logs") or "./logs")
    return Path(log_dir)


def setup_structured_logging(config: Dict[str, Any]) -> Dict[str, str]:
    """配置结构化日志系统。

    - 根 logger 追加 JSON 格式的 ``RotatingFileHandler`` ->
      ``<log_dir>/structured.jsonl``（默认 50MB × 10 个备份）。
    - ``trades`` logger -> ``<log_dir>/trades.jsonl``
    - ``alerts`` logger -> ``<log_dir>/alerts.jsonl``
    - ``jev`` logger -> ``<log_dir>/jev.jsonl``
    - 控制台保持现有人类可读格式（不删除 / 不替换已有 handler）。

    Args:
        config: ``config.yaml`` 中 ``logging`` 节的字典。

    Returns:
        ``{逻辑名: 绝对路径}`` 映射。
    """
    cfg: Dict[str, Any] = dict(config or {})
    log_dir = _resolve_log_dir(cfg)
    log_dir.mkdir(parents=True, exist_ok=True)

    level_name = str(cfg.get("level", "INFO")).upper()
    max_bytes = int(cfg.get("max_bytes", 50 * 1024 * 1024))  # 默认 50MB
    backup_count = int(cfg.get("backup_count", 10))

    fmt = StructuredJSONFormatter()
    root = logging.getLogger()

    paths: Dict[str, str] = {}

    def _add_rotating(target_logger: logging.Logger, filename: str) -> str:
        path = log_dir / filename
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        handler.setFormatter(fmt)
        handler.setLevel(logging.DEBUG)  # 文件记录全部级别，由 logger 级别控制
        target_logger.addHandler(handler)
        return str(path)

    # 根 logger：全量结构化日志
    paths["structured"] = _add_rotating(root, "structured.jsonl")

    # 业务独立 logger（propagate=False 避免重复写根文件）
    for name, filename in (
        ("trades", "trades.jsonl"),
        ("alerts", "alerts.jsonl"),
        ("jev", "jev.jsonl"),
    ):
        lg = logging.getLogger(name)
        paths[name] = _add_rotating(lg, filename)
        lg.propagate = False

    # 设置根 logger 级别
    numeric_level = getattr(logging, level_name, logging.INFO)
    root.setLevel(numeric_level)

    # 注册到模块级，供查询 API 使用
    _ACTIVE_LOG_FILES.clear()
    _ACTIVE_LOG_FILES.update(paths)

    logger.info(
        "结构化日志已就绪: dir=%s, level=%s, files=%s",
        log_dir, level_name, list(paths.keys()),
    )
    return paths


# ---------------------------------------------------------------------------
# 查询 / 动态调级
# ---------------------------------------------------------------------------

def get_recent_logs(
    level: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """从 ``structured.jsonl`` 读取最近 N 条结构化日志。

    Args:
        level: 可选，按级别过滤（如 "ERROR"）；None 表示不过滤。
        limit: 返回条数（取文件末尾最近的 N 条）。

    Returns:
        日志字典列表，按文件顺序（旧 -> 新）。文件不存在时返回空列表。
    """
    path = _ACTIVE_LOG_FILES.get("structured")
    if not path or not os.path.exists(path):
        return []

    limit = max(1, min(1000, int(limit)))
    want_level = level.upper() if level else None

    # 逐行读取，保留最近满足条件的记录
    collected: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if want_level and rec.get("level") != want_level:
                    continue
                collected.append(rec)
    except OSError:
        logger.exception("读取结构化日志失败: %s", path)
        return []

    return collected[-limit:]


def set_log_level(level: str) -> str:
    """动态设置根 logger 级别。

    Args:
        level: 级别名（DEBUG/INFO/WARNING/ERROR/CRITICAL）。

    Returns:
        实际生效的级别名（非法级别抛 ValueError）。
    """
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"未知日志级别: {level!r}")
    logging.getLogger().setLevel(numeric)
    return level.upper()


def get_log_files() -> List[Dict[str, Any]]:
    """返回已注册日志文件的元数据列表。

    Returns:
        ``[{name, path, size_bytes, modified}]``，按注册顺序。
    """
    result: List[Dict[str, Any]] = []
    for name, path in _ACTIVE_LOG_FILES.items():
        entry: Dict[str, Any] = {
            "name": name,
            "path": path,
            "size_bytes": 0,
            "modified": "",
        }
        try:
            st = os.stat(path)
            entry["size_bytes"] = st.st_size
            entry["modified"] = datetime.fromtimestamp(st.st_mtime).isoformat(
                timespec="seconds"
            )
        except OSError:
            pass
        result.append(entry)
    return result
