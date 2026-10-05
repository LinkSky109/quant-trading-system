#!/usr/bin/env python3
"""量化交易系统实时数据服务（FastAPI）v3.1。

支持：
  - 多标的真实行情（腾讯财经接口，含五档盘口，后台每 3 秒刷新）
  - 回测按钮（POST /api/backtest，可附加 Jev 信号过滤）
  - 多账户（3 个模拟账户，独立资金/持仓/策略）
  - 策略切换（ma_cross / bollinger / momentum_breakout）
  - Jev 决策展示（真实 laya_mlx subprocess，失败降级 mock）

启动:
    python server.py
    访问 http://localhost:8766
"""
from __future__ import annotations

import asyncio
import json
import logging
import select
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config, load_stock_pool  # noqa: E402
from config.hot_reload import HotReloadManager  # noqa: E402
from config.validator import validate_and_exit_on_error  # noqa: E402
from data.data_fetcher import DataFetcher, normalize_symbol  # noqa: E402
from utils.indicators import add_indicators  # noqa: E402
from monitoring.alert import AlertManager  # noqa: E402
from monitoring.data_quality import DataQualityMonitor  # noqa: E402
from monitoring.daily_report_generator import DailyReportGenerator  # noqa: E402
from monitoring.notifier import NotifierManager  # noqa: E402
from monitoring.structured_logging import (  # noqa: E402
    RequestIDMiddleware,
    get_log_files,
    get_recent_logs,
    set_log_level,
    setup_structured_logging,
)
from security.auth import AuthManager  # noqa: E402
from security.audit import AuditLogger, ActionType, get_audit_logger, init_audit_logger  # noqa: E402
from security.rate_limit import RateLimiter  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("realtime_server")

app = FastAPI(title="量化交易系统实时数据服务", version="3.0.0")
START_TIME = time.time()

# ---------------------------------------------------------------------------
# API 请求性能统计中间件（系统健康监控用）
# ---------------------------------------------------------------------------
from monitoring.system_health import ApiMetricsCollector  # noqa: E402

api_metrics = ApiMetricsCollector()


@app.middleware("http")
async def api_stats_middleware(request, call_next):
    """统计每个 /api/ 端点的调用次数、延迟、错误率。"""
    start = time.time()
    response = await call_next(request)
    latency = (time.time() - start) * 1000  # 毫秒
    endpoint = request.url.path
    if endpoint.startswith("/api/"):
        api_metrics.record_request(
            endpoint, request.method, latency, response.status_code
        )
    return response


# ---------------------------------------------------------------------------
# API Token 认证中间件
# ---------------------------------------------------------------------------
# 公开路径（无需认证）：精确匹配
_PUBLIC_EXACT_PATHS = {
    "/",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/api/health",
    "/api/auth/verify",
    "/healthz",
}

# 需要 admin 权限的路径前缀（前缀匹配）
_ADMIN_PATH_PREFIXES = (
    "/api/alerts/trigger",
    "/api/daily_reports/export",
)

# 需要 trade 权限的路径前缀（前缀匹配）
_TRADE_PATH_PREFIXES = (
    "/api/trade_toggle",
)


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """从 ``Authorization: Bearer <token>`` 头中提取 token。"""
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip() or None
    return None


def _required_permission_for_path(path: str) -> Optional[str]:
    """根据请求路径返回所需权限级别。

    Returns:
        None 表示公开路径（无需认证）；否则返回 "read" / "trade" / "admin"。
    """
    if path in _PUBLIC_EXACT_PATHS:
        return None
    if path.startswith(_ADMIN_PATH_PREFIXES):
        return "admin"
    if path.startswith(_TRADE_PATH_PREFIXES):
        return "trade"
    # 其余 API 与训练数据接口按只读处理
    if path.startswith("/api/") or path.startswith("/training_data/"):
        return "read"
    # 非 API 路径（如页面静态资源）默认公开
    return None


# ---------------------------------------------------------------------------
# API 限流中间件（令牌桶，按 Token / IP 维度 + 分级配额）
# ---------------------------------------------------------------------------
# Starlette/FastAPI 中：后定义（后注册）的中间件为外层，请求先经过。
# 本中间件刻意定义在 auth_middleware 之前，使 auth_middleware 成为外层、
# 先执行并写入 request.state.token_info，再由本中间件读取以 token 名称作为
# 限流维度。最终请求顺序（外->内）：auth_middleware -> rate_limit_middleware ->
# 路由（api_stats_middleware 在最前定义，为最内层，仅用于计时）。

# 重计算 / 高成本接口（backtest 级，5 次/分钟）的精确路径与前缀
_BACKTEST_EXACT_PATHS = frozenset({
    "/api/backtest_compare",
    "/api/backtest_portfolio",
    "/api/walkthrough/run",
})
_BACKTEST_PATH_PREFIXES = ("/api/optimize/",)

# 管理操作（admin 级，10 次/分钟）的路径前缀
_ADMIN_RL_PREFIXES = (
    "/api/alerts/trigger",
    "/api/daily_reports/export",
    "/api/backup/",
)


def _rate_limit_tier_for_path(path: str, method: str) -> Optional[str]:
    """根据请求路径与方法返回限流分级。

    Returns:
        ``"backtest"`` / ``"trade"`` / ``"admin"`` / ``"read"``；
        返回 None 表示不限流（公开路径或非 /api 路径）。
    """
    # 非 API 路径（静态页面、文档等）不限流
    if not path.startswith("/api/"):
        return None
    # 公开路径（健康检查、认证校验等）不限流
    if path in _PUBLIC_EXACT_PATHS:
        return None

    # backtest 级：POST /api/backtest、回测对比/组合、walkthrough 运行、优化接口
    if method == "POST" and path == "/api/backtest":
        return "backtest"
    if path in _BACKTEST_EXACT_PATHS:
        return "backtest"
    if path.startswith(_BACKTEST_PATH_PREFIXES):
        return "backtest"

    # admin 级：告警触发、报表导出、备份管理
    if path.startswith(_ADMIN_RL_PREFIXES):
        return "admin"

    # trade 级：POST 交易开关
    if method == "POST" and path == "/api/trade_toggle":
        return "trade"

    # 其余所有 /api/* 按只读配额
    return "read"


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """API 限流中间件（令牌桶）。

    - 非 /api 路径、公开路径、白名单 IP、``rate_limit.enabled=false`` 时直接放行。
    - 限流 key：认证用户用 token 名称；未认证用来源 IP。
    - 超限时返回 429 + ``Retry-After`` 头，并记录 ``RATE_LIMIT_BLOCKED`` 审计事件。
    """
    path = request.url.path
    method = request.method

    tier = _rate_limit_tier_for_path(path, method)
    if tier is None:
        return await call_next(request)

    if not rate_limiter.is_enabled():
        return await call_next(request)

    client_ip = request.client.host if request.client else ""
    if rate_limiter.is_whitelisted(client_ip):
        return await call_next(request)

    # 限流维度：认证用户优先用 token 名称，否则按 IP
    token_info = getattr(request.state, "token_info", None)
    if token_info:
        key = str(token_info.get("name") or "authenticated")
        key_type = "token"
    else:
        key = client_ip
        key_type = "ip"

    allowed, retry_after, info = rate_limiter.check_rate_limit(key, tier)
    if not allowed:
        retry_after_int = max(1, int(retry_after + 0.999))  # 向上取整，避免 0
        try:
            get_audit_logger().log(
                operator=key,
                action_type="RATE_LIMIT_BLOCKED",
                target=path,
                params={
                    "method": method,
                    "tier": tier,
                    "key_type": key_type,
                    "limit": info.get("limit"),
                },
                result=f"blocked: retry_after={retry_after_int}s",
                ip=client_ip,
                request_id=request.headers.get("x-request-id", ""),
            )
        except Exception:
            logger.exception("记录 RATE_LIMIT_BLOCKED 审计失败")
        return JSONResponse(
            status_code=429,
            content={"code": 429, "message": "Too Many Requests", "data": None},
            headers={"Retry-After": str(retry_after_int)},
        )

    return await call_next(request)


@app.middleware("http")
async def auth_middleware(request, call_next):
    """API Token 认证中间件。

    - ``security.enabled=false`` 时直接放行，不影响现有功能。
    - 公开路径（见 ``_PUBLIC_EXACT_PATHS``）无需 token。
    - 其余接口按 ``_required_permission_for_path`` 映射所需权限：
      无 token -> 401；token 无效/过期 -> 401；权限不足 -> 403。
    - 校验通过后将 token 信息挂到 ``request.state.token_info`` 供下游使用。
    """
    if not auth_manager.is_enabled():
        return await call_next(request)

    path = request.url.path
    required = _required_permission_for_path(path)
    if required is None:
        return await call_next(request)

    token = _extract_bearer_token(request.headers.get("authorization"))
    if not token:
        return JSONResponse(
            status_code=401,
            content={"code": 401, "message": "未提供认证token", "data": None},
        )
    token_info = auth_manager.verify_token(token)
    if token_info is None:
        return JSONResponse(
            status_code=401,
            content={"code": 401, "message": "token无效或已过期", "data": None},
        )
    if not auth_manager.has_permission(token_info, required):
        return JSONResponse(
            status_code=403,
            content={"code": 403, "message": "权限不足", "data": None},
        )

    request.state.token_info = token_info

    # 记录登录审计（认证成功即记录，ip 取自客户端连接）
    try:
        audit_logger.log(
            operator=str(token_info.get("name", "")),
            action_type=ActionType.LOGIN,
            target=path,
            params={"method": request.method},
            result="success",
            ip=request.client.host if request.client else "",
            request_id=request.headers.get("x-request-id", ""),
        )
    except Exception:
        logger.exception("记录 LOGIN 审计失败")

    return await call_next(request)


# ---------------------------------------------------------------------------
# 请求链路 ID 中间件（最外层，最先执行）
# ---------------------------------------------------------------------------
# 必须在 auth / rate_limit / stats 等所有中间件**之后**通过 add_middleware 注册，
# Starlette 中后注册的中间件位于更外层、请求先经过它，从而把 request_id 写入
# contextvars 与请求头 scope，供后续中间件（如审计日志读取 x-request-id）使用。
app.add_middleware(RequestIDMiddleware)


def require_auth(permission: str = "read"):
    """FastAPI 依赖：对单个路由做精细 token 校验。

    全局 ``auth_middleware`` 已覆盖绝大多数路由，此依赖主要供未来单路由
    级别的精细控制使用（``security.enabled=false`` 时直接放行，返回 None）。

    Usage::

        @app.get("/api/sensitive", dependencies=[Depends(require_auth("admin"))])
        async def sensitive(): ...
    """

    async def _dependency(
        authorization: Optional[str] = Header(None),
    ) -> Optional[Dict[str, Any]]:
        if not auth_manager.is_enabled():
            return None
        token = _extract_bearer_token(authorization)
        if not token:
            raise HTTPException(status_code=401, detail="未提供认证token")
        token_info = auth_manager.verify_token(token)
        if token_info is None:
            raise HTTPException(status_code=401, detail="token无效或已过期")
        if not auth_manager.has_permission(token_info, permission):
            raise HTTPException(status_code=403, detail="权限不足")
        return token_info

    return _dependency


# ---------------------------------------------------------------------------
# 统一 API 响应封装 & 错误处理
# ---------------------------------------------------------------------------

def ok(data: Any = None, message: str = "success") -> Dict[str, Any]:
    """成功响应统一封装。"""
    return {"code": 0, "message": message, "data": data}


def err(code: int, message: str, http_status: int = 400) -> JSONResponse:
    """错误响应统一封装（不暴露堆栈）。"""
    return JSONResponse(
        status_code=http_status,
        content={"code": code, "message": message, "data": None},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    """捕获所有未处理异常，返回统一 JSON 错误格式，不暴露堆栈。"""
    logger.exception("未处理异常 %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": f"服务器内部错误: {exc}", "data": None},
    )


# ---------------------------------------------------------------------------
# 配置加载：股票池 / 策略 / 账户 / Jev 阈值（全部来自外部 YAML）
# ---------------------------------------------------------------------------

# 主配置
MAIN_CFG = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))

# 认证管理器：enabled=false 时所有接口放行，不影响现有功能
auth_manager = AuthManager(MAIN_CFG.get("security", {}))
if auth_manager.is_enabled() and not auth_manager.has_tokens():
    logger.warning(
        "security.enabled=true 但未配置任何 token，所有受保护接口将返回 401，"
        "请在 config.yaml 的 security.tokens 中添加 token"
    )

# 安全审计日志：双写 SQLite + JSONL，哈希链完整性校验
audit_logger = init_audit_logger(
    MAIN_CFG.get("audit", {}),
    db_path=str(PROJECT_ROOT / "data" / "quant_trading.db"),
)

# API 限流：令牌桶，按 Token/IP 维度 + 分级配额（默认关闭，config 中开启）
rate_limiter = RateLimiter(MAIN_CFG.get("rate_limit", {}))

# 配置热加载管理器：原地更新 MAIN_CFG 中的可热加载节，支持 SIGHUP / API 触发
hot_reload_manager = HotReloadManager(
    MAIN_CFG,
    config_path=str(PROJECT_ROOT / "config" / "config.yaml"),
)

# 结构化 JSON 日志：追加滚动文件 handler（控制台保持人类可读格式不变）
setup_structured_logging(MAIN_CFG.get("logging", {}))

# 股票池（来自 config/stock_pool.yaml）
_POOL = load_stock_pool(str(PROJECT_ROOT / "config" / "stock_pool.yaml"))
STOCK_POOL_META = _POOL["pool_meta"]
STOCK_POOL_STOCKS = _POOL["stocks"]

# 配置启动校验：不合规直接退出，不带着错误配置运行
_ = validate_and_exit_on_error(MAIN_CFG, _POOL)

SYMBOLS: List[Dict[str, Any]] = [
    {
        "symbol": s["symbol"],
        "name": s["name"],
        "market": s.get("market", "A股"),
        "base_price": float(s["base_price"]),
        # 附带元数据，供 /api/stock_pool 使用
        "industry": s.get("industry", ""),
        "market_cap_yi": s.get("market_cap_yi"),
        "daily_amount_yi": s.get("daily_amount_yi"),
        "pe_ttm": s.get("pe_ttm"),
        "volatility_annual": s.get("volatility_annual"),
        "trend_score": s.get("trend_score", ""),
    }
    for s in STOCK_POOL_STOCKS
]
SYM_INFO = {s["symbol"]: s for s in SYMBOLS}
SYMBOL_SET = set(SYM_INFO.keys())

# 策略元数据（从 config.yaml strategies 节读取 label/description）
_strat_cfg = MAIN_CFG.get("strategies", {})
STRATEGIES_META: List[Dict[str, str]] = []
for _name, _cfg in _strat_cfg.items():
    if not isinstance(_cfg, dict):
        continue
    if _cfg.get("enabled", True) is False:
        continue
    STRATEGIES_META.append({
        "name": _name,
        "label": _cfg.get("label", _name),
        "description": _cfg.get("description", ""),
    })
STRATEGY_NAMES = [s["name"] for s in STRATEGIES_META]

# 账户元数据（从 config.yaml accounts 节读取）
ACCOUNTS_META: List[Dict[str, Any]] = []
ACCOUNT_SEEDS: Dict[str, List[tuple]] = {}
for _a in MAIN_CFG.get("accounts", []):
    ACCOUNTS_META.append({
        "account_id": _a["account_id"],
        "name": _a["name"],
        "initial_capital": float(_a["initial_capital"]),
        "strategy": _a["strategy"],
        "description": _a.get("description", ""),
        "account_type": _a.get("account_type", "simulated"),
    })
    # seed_positions: [[symbol, frac], ...] -> [(symbol, frac), ...]
    ACCOUNT_SEEDS[_a["account_id"]] = [
        (str(sp[0]), float(sp[1])) for sp in (_a.get("seed_positions") or [])
    ]

# Jev 阈值（从 config.yaml jev.confidence_threshold 读取）
JEV_THRESHOLD = float(MAIN_CFG.get("jev", {}).get("confidence_threshold", 0.6))

JEV_PYTHON = "/Users/link/myApp/ai/jev/.venv/bin/python"
JEV_RUNNER = str(Path(__file__).parent / "jev_runner.py")
JEV_PREDICT_TIMEOUT = 30.0  # 秒



# ---------------------------------------------------------------------------
# 真实行情获取（腾讯财经免费接口，含五档盘口）
# ---------------------------------------------------------------------------

import requests as _requests

_REAL_QUOTE_CACHE: Dict[str, Dict[str, Any]] = {}
_REAL_QUOTE_TIME = 0.0
_REAL_QUOTE_LOCK = threading.Lock()


def _tencent_symbol(symbol: str) -> str:
    """将 600519.SH 转为 sh600519 供腾讯接口使用。"""
    norm = normalize_symbol(symbol)
    code, market = norm.split(".")
    prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj", "HK": "hk", "US": "us"}.get(market, "sh")
    return f"{prefix}{code}"


def fetch_real_quotes(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """批量从腾讯财经获取真实行情和五档盘口。

    Returns:
        {symbol: {price, prev_close, open, high, low, volume, amount, bids[], asks[]}}
    """
    global _REAL_QUOTE_CACHE, _REAL_QUOTE_TIME
    with _REAL_QUOTE_LOCK:
        # 缓存1.5秒，避免频繁请求
        if time.time() - _REAL_QUOTE_TIME < 1.5 and _REAL_QUOTE_CACHE:
            return dict(_REAL_QUOTE_CACHE)

    tencent_codes = [_tencent_symbol(s) for s in symbols]
    url = f"http://qt.gtimg.cn/q={','.join(tencent_codes)}"
    try:
        r = _requests.get(url, headers={"Referer": "http://finance.qq.com"}, timeout=5)
        r.encoding = "gbk"
        result = {}
        for line in r.text.strip().split(";"):
            if "=" not in line or "~" not in line:
                continue
            raw = line.split("=", 1)[1].strip().strip('"')
            p = raw.split("~")
            if len(p) < 45:
                continue
            code = p[2]
            market = "SH" if code.startswith("6") else "SZ"
            sym = f"{code}.{market}"
            try:
                price = float(p[3]) if p[3] else 0.0
                prev_close = float(p[4]) if p[4] else 0.0
                open_p = float(p[5]) if p[5] else 0.0
                volume = int(float(p[6])) * 100 if p[6] else 0  # 手→股
                amount = float(p[37]) * 10000 if p[37] else 0.0  # 万→元
                high = float(p[33]) if p[33] else price
                low = float(p[34]) if p[34] else price
                bids = []
                asks = []
                for i in range(5):
                    bp = float(p[9 + i*2]) if p[9 + i*2] else price
                    bv = int(float(p[10 + i*2])) * 100 if p[10 + i*2] else 0
                    bids.append({"price": round(bp, 2), "volume": bv})
                    ap = float(p[19 + i*2]) if p[19 + i*2] else price
                    av = int(float(p[20 + i*2])) * 100 if p[20 + i*2] else 0
                    asks.append({"price": round(ap, 2), "volume": av})
                result[sym] = {
                    "price": price, "prev_close": prev_close, "open": open_p,
                    "high": high, "low": low, "volume": volume, "amount": amount,
                    "bids": bids, "asks": asks,
                }
            except (ValueError, IndexError):
                continue
        with _REAL_QUOTE_LOCK:
            _REAL_QUOTE_CACHE = result
            _REAL_QUOTE_TIME = time.time()
        return result
    except Exception as e:
        logger.warning("真实行情获取失败: %s", e)
        return dict(_REAL_QUOTE_CACHE)


# ---------------------------------------------------------------------------
# 实时模拟器（单标的）
# ---------------------------------------------------------------------------

class RealTimeSimulator:
    """单标的实时行情模拟器：基于历史 mock K线，tick 随机游走。"""

    def __init__(self, info: Dict[str, Any], fetcher: DataFetcher):
        self.symbol = normalize_symbol(info["symbol"])
        self.name = info["name"]
        self.base_price = float(info["base_price"])
        self.fetcher = fetcher
        self._lock = asyncio.Lock()
        self.tick_count = 0
        self.last_update = time.time()
        self._init_data()

    def _init_data(self) -> None:
        df = self.fetcher.get_klines(
            self.symbol, period="1d", count=250,
            start_date="2024-01-02", use_cache=False,
        )
        # 按标的 base_price 缩放，使不同股票价格量级真实
        factor = self.base_price / float(df["close"].iloc[0])
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] * factor
        df["amount"] = df["close"] * df["volume"]
        self.klines = df

        self.current_price = float(df["close"].iloc[-1])
        self.prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else self.current_price
        self.day_open = float(df["open"].iloc[-1])
        self.day_high = float(df["high"].iloc[-1])
        self.day_low = float(df["low"].iloc[-1])
        self.volume = int(df["volume"].iloc[-1])
        self.amount = float(df["amount"].iloc[-1])

    async def tick(self) -> None:
        async with self._lock:
            self.tick_count += 1
            # 从腾讯财经获取真实行情
            quotes = fetch_real_quotes([self.symbol])
            q = quotes.get(self.symbol)
            if q and q["price"] > 0:
                self.current_price = q["price"]
                self.prev_close = q["prev_close"] or self.prev_close
                self.day_open = q["open"] or self.day_open
                self.day_high = q["high"] or self.day_high
                self.day_low = q["low"] or self.day_low
                self.volume = q["volume"] or self.volume
                self.amount = q["amount"] or self.amount
                self._real_orderbook = {"bids": q["bids"], "asks": q["asks"]}
            else:
                # 非交易时段或获取失败，保持最后价格
                logger.debug("真实行情不可用，保持上次价格: %s", self.symbol)
            self.last_update = time.time()

    def get_quote(self) -> Dict[str, Any]:
        change = self.current_price - self.prev_close
        change_pct = change / self.prev_close * 100 if self.prev_close else 0.0
        return {
            "symbol": self.symbol,
            "name": self.name,
            "price": round(self.current_price, 2),
            "prev_close": round(self.prev_close, 2),
            "open": round(self.day_open, 2),
            "high": round(self.day_high, 2),
            "low": round(self.day_low, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": self.volume,
            "amount": round(self.amount, 2),
            "timestamp": pd.Timestamp.now().isoformat(),
            "tick": self.tick_count,
        }

    def get_orderbook(self) -> Dict[str, Any]:
        # 优先返回真实五档盘口，无真实数据时降级为模拟
        if hasattr(self, "_real_orderbook") and self._real_orderbook.get("bids"):
            ob = self._real_orderbook
            return {
                "symbol": self.symbol,
                "bids": ob["bids"], "asks": ob["asks"],
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        price = self.current_price
        bids, asks = [], []
        for i in range(1, 6):
            bids.append({"price": round(price * (1 - 0.0002 * i), 2), "volume": 1000})
            asks.append({"price": round(price * (1 + 0.0002 * i), 2), "volume": 1000})
        return {"symbol": self.symbol, "bids": bids, "asks": asks,
                "timestamp": pd.Timestamp.now().isoformat()}

    def get_klines_realtime(self, count: int = 120) -> List[List[Any]]:
        df = self.klines.tail(count).copy()
        if not df.empty:
            df.iloc[-1, df.columns.get_loc("close")] = self.current_price
            df.iloc[-1, df.columns.get_loc("high")] = self.day_high
            df.iloc[-1, df.columns.get_loc("low")] = self.day_low
            df.iloc[-1, df.columns.get_loc("volume")] = self.volume
        result = []
        for idx, row in df.iterrows():
            result.append([
                idx.strftime("%Y-%m-%d"),
                round(float(row["open"]), 2),
                round(float(row["close"]), 2),
                round(float(row["low"]), 2),
                round(float(row["high"]), 2),
            ])
        return result

    def get_realtime_df(self) -> pd.DataFrame:
        """返回含实时价格的最新 K线 DataFrame（最后一根 close/high/low 更新）。"""
        df = self.klines.copy()
        df.iloc[-1, df.columns.get_loc("close")] = self.current_price
        df.iloc[-1, df.columns.get_loc("high")] = self.day_high
        df.iloc[-1, df.columns.get_loc("low")] = self.day_low
        return df


# ---------------------------------------------------------------------------
# 多标的管理器
# ---------------------------------------------------------------------------

class MultiSymbolManager:
    """维护 6 只标的的实时模拟器，后台统一 tick。"""

    def __init__(self):
        self.cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
        self.fetcher = DataFetcher(
            api_key=self.cfg["data"].get("api_key", ""),
            cache_dir=str(PROJECT_ROOT / "cache"),
            use_mock=False,  # 使用 QuantDash 真实K线数据
        )
        self.sims: Dict[str, RealTimeSimulator] = {}
        for info in SYMBOLS:
            sym = info["symbol"]
            try:
                self.sims[sym] = RealTimeSimulator(info, self.fetcher)
                logger.info("已初始化模拟器: %s %s @ %.2f", sym, info["name"],
                            self.sims[sym].current_price)
            except Exception as e:
                logger.error("初始化 %s 失败: %s", sym, e)
        self.current_symbol = SYMBOLS[0]["symbol"]

    def get(self, symbol: Optional[str] = None) -> RealTimeSimulator:
        sym = normalize_symbol(symbol) if symbol else self.current_symbol
        if sym not in self.sims:
            sym = self.current_symbol
        return self.sims[sym]

    async def tick_all(self) -> None:
        # 批量获取所有标的的真实行情（一次HTTP请求）
        all_syms = list(self.sims.keys())
        fetch_real_quotes(all_syms)
        for sim in self.sims.values():
            try:
                await sim.tick()
            except Exception as e:
                logger.warning("tick %s 失败: %s", sim.symbol, e)

    @property
    def total_tick(self) -> int:
        return max((s.tick_count for s in self.sims.values()), default=0)


manager = MultiSymbolManager()


# ---------------------------------------------------------------------------
# WebSocket 连接管理器：按标的订阅推送行情/交易/告警/账户/Jev 决策
# ---------------------------------------------------------------------------

class ConnectionManager:
    """管理所有活跃 WebSocket 连接，支持按标的订阅。

    每个连接维护一个 subscribed_symbols 集合；连接建立时默认订阅全部标的。
    """

    def __init__(self) -> None:
        self.active: Dict[WebSocket, set] = {}

    async def connect(self, websocket: WebSocket) -> int:
        await websocket.accept()
        # 默认订阅全部标的
        self.active[websocket] = set(SYMBOL_SET)
        return len(self.active)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active.pop(websocket, None)

    def subscribe(self, websocket: WebSocket, symbol: str) -> None:
        if websocket in self.active and symbol:
            self.active[websocket].add(normalize_symbol(symbol))

    def unsubscribe(self, websocket: WebSocket, symbol: str) -> None:
        if websocket in self.active and symbol:
            self.active[websocket].discard(normalize_symbol(symbol))

    async def send_to_subscribers(self, symbol: str, message: Dict[str, Any]) -> None:
        """向所有订阅了该标的的连接推送消息；发送失败的连接自动清理。"""
        dead = []
        for ws, subs in list(self.active.items()):
            if symbol not in subs:
                continue
            try:
                await ws.send_text(json.dumps(message, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """向所有连接广播消息（交易/告警/账户/Jev 等不区分标的的事件）。"""
        dead = []
        for ws in list(self.active.keys()):
            try:
                await ws.send_text(json.dumps(message, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# 策略管理
# ---------------------------------------------------------------------------

def _strategy_instance(name: str):
    """根据名称实例化策略。"""
    from strategies.ma_cross import MACrossStrategy
    from strategies.bollinger import BollingerStrategy
    from strategies.momentum_breakout import MomentumBreakoutStrategy
    from strategies.rsi import RSIStrategy
    from strategies.macd import MACDStrategy
    from strategies.grid_trading import GridTradingStrategy

    mapping = {
        "ma_cross": MACrossStrategy,
        "bollinger": BollingerStrategy,
        "momentum_breakout": MomentumBreakoutStrategy,
        "rsi": RSIStrategy,
        "macd": MACDStrategy,
        "grid_trading": GridTradingStrategy,
    }
    cls = mapping.get(name)
    if cls is None:
        raise ValueError(f"unknown strategy: {name}")
    return cls()


current_strategy: str = "ma_cross"


# ---------------------------------------------------------------------------
# 多账户
# ---------------------------------------------------------------------------

class SimAccount:
    """单个模拟账户：独立现金与持仓。"""

    def __init__(self, meta: Dict[str, Any]):
        self.account_id = meta["account_id"]
        self.name = meta["name"]
        self.initial_capital = float(meta["initial_capital"])
        self.strategy = meta["strategy"]
        self.description = meta["description"]
        self.account_type = meta.get("account_type", "simulated")
        self.cash = self.initial_capital
        self.positions: Dict[str, Dict[str, float]] = {}

    def seed_positions(self) -> None:
        for sym, frac in ACCOUNT_SEEDS.get(self.account_id, []):
            sim = manager.sims.get(sym)
            if sim is None:
                continue
            price = float(sim.klines["close"].iloc[-5]) if len(sim.klines) > 5 else sim.current_price
            target_value = self.initial_capital * frac
            shares = int(target_value / price / 100) * 100
            if shares > 0:
                self.positions[sym] = {"shares": float(shares), "avg_cost": price}
                self.cash -= shares * price

    def snapshot(self) -> Dict[str, Any]:
        market_value = 0.0
        positions_list = []
        for sym, pos in self.positions.items():
            sim = manager.sims.get(sym)
            if sim is None:
                continue
            cur = sim.current_price
            shares = pos["shares"]
            val = shares * cur
            market_value += val
            pnl = (cur - pos["avg_cost"]) * shares
            pnl_pct = (cur / pos["avg_cost"] - 1) * 100 if pos["avg_cost"] else 0.0
            positions_list.append({
                "symbol": sym,
                "name": SYM_INFO.get(sym, {}).get("name", sym),
                "shares": int(shares),
                "avg_cost": round(pos["avg_cost"], 2),
                "current_price": round(cur, 2),
                "market_value": round(val, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
            })
        total_asset = self.cash + market_value
        total_pnl = total_asset - self.initial_capital
        return {
            "account_id": self.account_id,
            "name": self.name,
            "strategy": self.strategy,
            "initial_capital": self.initial_capital,
            "total_asset": round(total_asset, 2),
            "available_cash": round(self.cash, 2),
            "market_value": round(market_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / self.initial_capital * 100, 2) if self.initial_capital else 0.0,
            "position_ratio": round(market_value / total_asset * 100, 2) if total_asset else 0.0,
            "positions": positions_list,
            "timestamp": pd.Timestamp.now().isoformat(),
        }


accounts: Dict[str, SimAccount] = {}
for _meta in ACCOUNTS_META:
    acc = SimAccount(_meta)
    acc.seed_positions()
    accounts[acc.account_id] = acc
current_account_id = "acc_1"


# ---------------------------------------------------------------------------
# Jev 真实子进程客户端
# ---------------------------------------------------------------------------

JEV_QUESTIONS = {
    "decision": {
        "type": "choice",
        "instructions": "基于当前市场状态，应该买入、卖出还是持有？",
        "criteria": {
            "buy": "多头趋势明确，技术指标支撑上涨，建议买入",
            "sell": "空头趋势明确，技术指标支撑下跌，建议卖出",
            "hold": "趋势不明或震荡，建议观望等待",
        },
    }
}


class JevRealClient:
    """通过 HTTP 调用本地 Jev 服务 (localhost:8765)，使用 Session 保持长连接。"""

    def __init__(self, base_url: str = "http://localhost:8765"):
        self.base_url = base_url
        self._session = None
        self.available = False
        self._checked = False
        self._last_check = 0.0

    def _get_session(self):
        if self._session is None:
            import requests as _req
            self._session = _req.Session()
            adapter = _req.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8)
            self._session.mount("http://", adapter)
        return self._session

    def _ensure_available(self) -> bool:
        # 已确认可用则直接返回；失败后每 30 秒重试一次
        import time as _time
        now = _time.time()
        if self.available:
            return True
        if self._checked and now - self._last_check < 30:
            return False
        self._checked = True
        self._last_check = now
        try:
            r = self._get_session().get(f"{self.base_url}/api/health", timeout=5)
            if r.status_code == 200 and r.json().get("model_loaded"):
                self.available = True
                logger.info("Jev HTTP 服务已连接（real mode，长连接）: %s", self.base_url)
                return True
        except Exception as e:
            logger.warning("Jev HTTP 服务不可用，降级 mock（30s后重试）: %s", e)
        self.available = False
        return False

    def predict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """同步调用 Jev HTTP API，返回 {"probabilities":..., "latency_ms":...}。失败抛异常。"""
        if not self._ensure_available():
            raise RuntimeError("jev_unavailable")
        states_list = [{"feature": k, "value": v} for k, v in state.items()]
        payload = {"states": states_list, "candidate_actions": ["buy", "sell", "hold"]}
        r = self._get_session().post(
            f"{self.base_url}/api/evaluate",
            json=payload,
            timeout=JEV_PREDICT_TIMEOUT,
        )
        if r.status_code != 200:
            raise RuntimeError(f"jev_http_{r.status_code}")
        data = r.json()
        return {
            "probabilities": data["probabilities"],
            "latency_ms": data.get("latency_ms", 0.0),
        }

    def predict_batch(self, states_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量调用，states_list 为多个 state dict，返回对应结果列表。"""
        if not self._ensure_available():
            raise RuntimeError("jev_unavailable")
        payload = {
            "batch": [
                {"states": [{"feature": k, "value": v} for k, v in s.items()],
                 "candidate_actions": ["buy", "sell", "hold"]}
                for s in states_list
            ]
        }
        r = self._get_session().post(
            f"{self.base_url}/api/evaluate_batch",
            json=payload,
            timeout=JEV_PREDICT_TIMEOUT * len(states_list),
        )
        if r.status_code != 200:
            raise RuntimeError(f"jev_http_batch_{r.status_code}")
        return r.json()["results"]

    def shutdown(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None


jev_client = JevRealClient()


# ---------------------------------------------------------------------------
# 后台 tick 任务
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    async def tick_loop():
        dq_tick_counter = 0
        jev_prev_ok = True  # 追踪Jev状态变化，仅在断连时告警一次
        while True:
            await manager.tick_all()
            # 数据质量检测（每10轮执行一次，约30秒，避免开销）
            dq_tick_counter += 1
            if dq_tick_counter % 10 == 0:
                try:
                    _quotes = {sym: sim.get_quote() for sym, sim in manager.sims.items()}
                    _klines = {sym: sim.klines for sym, sim in manager.sims.items()}
                    data_quality_monitor.check(quotes=_quotes, klines=_klines)
                except Exception:
                    logger.exception("数据质量检测失败（已忽略，不影响交易）")
            # Jev 健康检测（每20轮执行一次，约60秒）
            if dq_tick_counter % 20 == 0:
                try:
                    jev_status = _jev_health_cached()
                    jev_ok = jev_status.get("status") == "ok"
                    if jev_ok and not jev_prev_ok:
                        logger.info("Jev 服务恢复连接，切换为 real 模式")
                        jev_prev_ok = True
                    elif not jev_ok and jev_prev_ok:
                        logger.warning(
                            "Jev 服务断连（%s），已自动降级为 mock 模式，实时交易继续运行",
                            jev_status.get("status", "unknown"),
                        )
                        jev_prev_ok = False
                except Exception:
                    logger.exception("Jev 健康检测失败（已忽略）")
            # 收盘后 15:30 自动生成当日报告（内部幂等，未到时间/已生成直接返回 []）
            try:
                report_generator.auto_generate()
            except Exception:
                logger.exception("自动生成每日报告失败（已忽略）")
            # 每日收盘后自动备份数据库（幂等，到点且当天未备份才执行）
            if _backup_enabled:
                try:
                    if backup_manager.should_auto_backup(auto_time=_backup_auto_time):
                        bk_path = backup_manager.backup(tag="auto")
                        logger.info("自动备份完成: %s", bk_path)
                except Exception:
                    logger.exception("自动备份失败（已忽略）")
            await asyncio.sleep(3)

    asyncio.create_task(tick_loop())
    asyncio.create_task(ws_push_loop())
    logger.info("实时数据服务已启动（多标的模式），tick 间隔 3 秒，WS 推送间隔 1 秒")


@app.on_event("shutdown")
async def shutdown_event():
    try:
        health_monitor.stop_auto_collect()
    except Exception:
        pass
    jev_client.shutdown()


# ---------------------------------------------------------------------------
# 工具：策略信号 & 市场状态
# ---------------------------------------------------------------------------

def _latest_strategy_signal(sim: RealTimeSimulator) -> tuple:
    """返回 (strategy_signal, confidence)。"""
    df = add_indicators(sim.get_realtime_df())
    if len(df) < 25:
        return "hold", 0.0
    strat = _strategy_instance(current_strategy)
    sig_df = strat.get_signal_dataframe(df, sim.symbol)
    row = sig_df.iloc[-1]
    sig = row.get("signal", 0)
    conf = row.get("confidence", 0.0)
    if pd.isna(sig):
        sig = 0
    if pd.isna(conf):
        conf = 0.0
    sig = float(sig)
    if sig > 0:
        return "buy", float(conf)
    if sig < 0:
        return "sell", float(conf)
    return "hold", 0.0


def _build_market_state_dict(sim: RealTimeSimulator) -> Optional[Dict[str, Any]]:
    from jev.jev_engine import JevDecisionEngine
    df = add_indicators(sim.get_realtime_df())
    engine = JevDecisionEngine(mock_mode=True)
    ms = engine.build_market_state(df, len(df) - 1)
    if ms is None:
        return None
    return asdict(ms)


# ---------------------------------------------------------------------------
# 实时模拟交易引擎
# ---------------------------------------------------------------------------

from realtime_trader import RealtimeTrader  # noqa: E402
from persistence.database import Database  # noqa: E402
from persistence.backup import BackupManager  # noqa: E402

# 初始化持久化数据库（SQLite，自动建表）
_db_path = str(PROJECT_ROOT / "data" / "quant_trading.db")
db = Database(db_path=_db_path)

# 初始化告警管理器（Webhook推送 + 冷却去重 + 历史记录）
_alert_cfg = MAIN_CFG.get("alert", {})
alert_manager = AlertManager(
    webhook_url=_alert_cfg.get("webhook_url", ""),
    cooldown_seconds=int(_alert_cfg.get("cooldown_seconds", 300)),
    max_history=int(_alert_cfg.get("max_history", 500)),
    timeout=float(_alert_cfg.get("timeout", 5.0)),
)
logger.info("告警管理器已初始化（webhook=%s）", "已配置" if _alert_cfg.get("webhook_url") else "未配置，仅日志")

# 初始化多渠道通知管理器（邮件/企微/钉钉/Server酱/Webhook，默认全 disabled）
_notification_cfg = MAIN_CFG.get("notification", {})
notifier_manager = NotifierManager(_notification_cfg)
alert_manager.set_notifier(notifier_manager)
logger.info("多渠道通知管理器已初始化（总开关=%s）", _notification_cfg.get("enabled", False))

# 初始化数据质量监控器
_dq_cfg = MAIN_CFG.get("data_quality", {})
data_quality_monitor = DataQualityMonitor(
    alert_manager=alert_manager,
    config={
        "lag_threshold_seconds": _dq_cfg.get("lag_threshold_seconds", 300),
        "price_jump_warning": _dq_cfg.get("price_jump_warning", 0.10),
        "price_jump_critical": _dq_cfg.get("price_jump_critical", 0.20),
        "max_consecutive_failures": _dq_cfg.get("max_consecutive_failures", 3),
        "max_history": _dq_cfg.get("anomaly_history_limit", 200),
    },
)
logger.info("数据质量监控器已初始化（enabled=%s）", _dq_cfg.get("enabled", True))

# 每日报告自动生成器（15:30 后幂等生成 acc_1/2/3 报告）
report_generator = DailyReportGenerator(db=db, alert_manager=alert_manager)
logger.info("每日报告生成器已初始化")

# 数据库备份管理器（在线备份/恢复/清理，默认 data/backups/）
_backup_cfg = MAIN_CFG.get("backup", {})
backup_manager = BackupManager(
    db_path=_db_path,
    backup_dir=str(PROJECT_ROOT / _backup_cfg.get("backup_dir", "data/backups")),
    keep_days=int(_backup_cfg.get("keep_days", 30)),
)
_backup_auto_time = _backup_cfg.get("auto_backup_time", "15:30")
_backup_enabled = _backup_cfg.get("enabled", True)
logger.info("数据库备份管理器已初始化（enabled=%s, auto_time=%s, keep_days=%d）",
            _backup_enabled, _backup_auto_time, int(_backup_cfg.get("keep_days", 30)))

# 策略排行榜单例（懒加载，复用 manager.fetcher 与全局配置）
_leaderboard: Optional[Any] = None


def _get_leaderboard() -> Any:
    global _leaderboard
    if _leaderboard is None:
        from strategies.strategy_leaderboard import StrategyLeaderboard
        _leaderboard = StrategyLeaderboard(data_fetcher=manager.fetcher, config=MAIN_CFG)
    return _leaderboard

_risk_cfg = MAIN_CFG.get("risk", {})
_realtime_cfg = MAIN_CFG.get("realtime_trading", {})
trader = RealtimeTrader(
    manager=manager,
    accounts=accounts,
    jev_client=jev_client,
    risk_config=_risk_cfg,
    get_strategy=lambda: current_strategy,
    get_account_id=lambda: current_account_id,
    strategy_signal_fn=_latest_strategy_signal,
    market_state_fn=_build_market_state_dict,
    interval=5.0,
    jev_threshold=JEV_THRESHOLD,
    db=db,
    alert_manager=alert_manager,
    realtime_config=_realtime_cfg,
)

# ---- 走查缓存（内存，最多保留50次） ----
from backtest.walkthrough import BacktestWalkthrough  # noqa: E402

_WALKTHROUGH_CACHE: Dict[str, BacktestWalkthrough] = {}
_WALKTHROUGH_MAX = 50


def _walkthrough_put(wt: BacktestWalkthrough) -> None:
    if wt.walkthrough_id is None:
        return
    _WALKTHROUGH_CACHE[wt.walkthrough_id] = wt
    while len(_WALKTHROUGH_CACHE) > _WALKTHROUGH_MAX:
        _WALKTHROUGH_CACHE.pop(next(iter(_WALKTHROUGH_CACHE)))


def _walkthrough_get(walkthrough_id: str) -> Optional[BacktestWalkthrough]:
    return _WALKTHROUGH_CACHE.get(walkthrough_id)


# ---- Jev 训练数据导出目录 ----
TRAINING_DATA_DIR = PROJECT_ROOT / "output" / "training_data"
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---- 系统健康监控 ----
from monitoring.system_health import SystemHealthMonitor  # noqa: E402

health_monitor = SystemHealthMonitor(
    db=db,
    data_quality_monitor=data_quality_monitor,
    alert_manager=alert_manager,
    realtime_trader=trader,
    api_metrics=api_metrics,
    jev_base_url=jev_client.base_url,
    quant_port=8766,
    start_time=START_TIME,
)
health_monitor.start_auto_collect(interval=30)


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    html_path = Path(__file__).parent / "quant_dashboard_realtime.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return JSONResponse({"code": 404, "message": "dashboard not found", "data": None}, status_code=404)


@app.get("/api/health")
async def health():
    # Jev 服务状态（非阻塞快速检测，缓存30秒）
    jev_status = _jev_health_cached()
    return ok({
        "status": "ok",
        "version": app.version,
        "tick": manager.total_tick,
        "uptime": round(time.time() - START_TIME, 1),
        "stock_pool_count": len(SYMBOLS),
        "strategies": STRATEGY_NAMES,
        "accounts": [a.account_id for a in accounts.values()],
        "jev": jev_status,
    })


@app.get("/healthz")
async def healthz():
    """极简健康检查端点（用于负载均衡 / Docker HEALTHCHECK / k8s livenessProbe）。

    不依赖任何业务状态，不查 Jev、不查策略，只返回进程存活信号。
    无需认证，公开路径。
    """
    return {"status": "ok"}


class AuthVerifyRequest(BaseModel):
    """/api/auth/verify 请求体。"""

    token: str


@app.post("/api/auth/verify")
async def verify_api_token(body: AuthVerifyRequest):
    """校验客户端 token 是否有效（公开端点，无需认证）。

    用于客户端在调用受保护接口前确认 token 可用性与权限级别。
    """
    info = auth_manager.verify_token(body.token)
    if info is None:
        return ok({"valid": False, "permission": None, "name": None, "expires_at": None})
    return ok({
        "valid": True,
        "permission": info["permission"],
        "name": info["name"],
        # 永不过期时返回空字符串，与配置格式一致
        "expires_at": info["expires_at"],
    })


# ---- Jev 健康监控 ---------------------------------------------------------

_jev_health_cache: Dict[str, Any] = {"status": "unknown", "mode": "mock", "last_check": 0.0}


def _jev_health_cached() -> Dict[str, Any]:
    """获取 Jev 健康状态（30秒缓存，避免每次健康检查都发请求）。"""
    global _jev_health_cache
    now = time.time()
    if now - _jev_health_cache["last_check"] < 30:
        return _jev_health_cache
    try:
        r = jev_client._get_session().get(
            f"{jev_client.base_url}/api/health", timeout=3
        )
        if r.status_code == 200:
            data = r.json()
            model_loaded = data.get("model_loaded", False)
            _jev_health_cache = {
                "status": "ok" if model_loaded else "loading",
                "mode": "real" if model_loaded else "mock",
                "model_loaded": model_loaded,
                "model": data.get("model", "unknown"),
                "last_check": now,
            }
        else:
            _jev_health_cache = {
                "status": "error", "mode": "mock",
                "http_status": r.status_code, "last_check": now,
            }
    except Exception as e:
        _jev_health_cache = {
            "status": "unavailable", "mode": "mock",
            "error": str(e)[:100], "last_check": now,
        }
    return _jev_health_cache


@app.get("/api/jev_health")
async def jev_health():
    """Jev 决策服务健康状态端点。

    返回 Jev 服务是否可用、当前模式(real/mock)、模型加载状态。
    Jev 不可用时量化服务自动降级为 mock 模式，不影响运行。
    """
    # 强制刷新（绕过缓存）
    global _jev_health_cache
    _jev_health_cache["last_check"] = 0.0
    status = _jev_health_cached()
    if status["status"] != "ok":
        logger.warning("Jev 健康检查异常: %s（当前运行在 %s 模式）", status, status.get("mode", "mock"))
    return ok(status)


# ---- 行情类 ---------------------------------------------------------------

@app.get("/api/quote")
async def get_quote(symbol: str = Query(None)):
    sym = normalize_symbol(symbol) if symbol else manager.current_symbol
    if sym not in manager.sims:
        return err(40001, f"标的不在股票池内: {sym}")
    return ok(manager.sims[sym].get_quote())


@app.get("/api/orderbook")
async def get_orderbook(symbol: str = Query(None)):
    sym = normalize_symbol(symbol) if symbol else manager.current_symbol
    if sym not in manager.sims:
        return err(40001, f"标的不在股票池内: {sym}")
    return ok(manager.sims[sym].get_orderbook())


@app.get("/api/klines")
async def get_klines(
    symbol: str = Query(None),
    count: int = Query(120, ge=10, le=500),
):
    sim = manager.get(symbol)
    return ok({"symbol": sim.symbol, "count": count, "data": sim.get_klines_realtime(count)})


@app.get("/api/all")
async def get_all(
    symbol: str = Query(None),
    account_id: str = Query("acc_1"),
):
    sim = manager.get(symbol)
    acc = accounts.get(account_id, accounts[current_account_id])
    return ok({
        "quote": sim.get_quote(),
        "orderbook": sim.get_orderbook(),
        "account": acc.snapshot(),
        "klines": sim.get_klines_realtime(120),
    })


# ---- 股票池 ---------------------------------------------------------------

@app.get("/api/symbols")
async def get_symbols():
    from trading.market_config import get_currency
    return ok({
        "symbols": [
            {"symbol": s["symbol"], "name": s["name"],
             "market": s["market"], "currency": get_currency(s["symbol"])}
            for s in SYMBOLS
        ],
        "current": manager.current_symbol,
    })


@app.get("/api/stock_pool")
async def get_stock_pool():
    """返回完整股票池信息（含行业、市值、波动率等元数据）。"""
    return ok({
        "meta": STOCK_POOL_META,
        "count": len(SYMBOLS),
        "stocks": STOCK_POOL_STOCKS,
    })


# ---- 策略 -----------------------------------------------------------------

@app.get("/api/strategies")
async def get_strategies():
    return ok({"strategies": STRATEGIES_META, "current": current_strategy})


@app.get("/api/strategy")
async def set_strategy(name: str = Query(...)):
    global current_strategy
    if name not in STRATEGY_NAMES:
        return err(40002, f"未知策略: {name}，可选: {STRATEGY_NAMES}")
    current_strategy = name
    return ok({"current": current_strategy, "strategies": STRATEGIES_META})


# ---- 账户 -----------------------------------------------------------------

@app.get("/api/accounts")
async def list_accounts():
    return ok({
        "accounts": [
            {
                "account_id": a.account_id,
                "name": a.name,
                "initial_capital": a.initial_capital,
                "strategy": a.strategy,
                "description": a.description,
                "account_type": a.account_type,
            }
            for a in accounts.values()
        ],
        "current": current_account_id,
    })


@app.get("/api/account")
async def get_account(account_id: Optional[str] = Query(None)):
    acc = accounts.get(account_id or current_account_id, accounts[current_account_id])
    return ok(acc.snapshot())


# ---- 回测 -----------------------------------------------------------------

class BacktestReq(BaseModel):
    symbol: str = "600519.SH"
    strategy: str = "ma_cross"
    start_date: str = "2024-01-02"
    end_date: str = "2025-12-31"
    use_jev: bool = False
    order_type: str = "market"  # market / limit / stop / trailing_stop


class WalkthroughReq(BaseModel):
    """运行一次回测走查的请求体。"""
    symbol: str = "600519.SH"
    strategy: str = "ma_cross"
    start_date: str = "2024-01-02"
    end_date: str = "2025-12-31"
    initial_capital: float = 1_000_000.0
    use_jev: bool = False


class TrainingExportReq(BaseModel):
    """Jev 训练数据导出请求体。"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    forward_days: int = 5
    hold_threshold: float = 0.02


class BacktestCompareReq(BaseModel):
    symbol: str = "600519.SH"
    strategies: List[str] = []
    start_date: str = "2024-01-02"
    end_date: str = "2025-12-31"


class PortfolioBacktestReq(BaseModel):
    symbols: List[str] = ["600519.SH", "300750.SZ", "002594.SZ"]
    strategy: str = "ma_cross"
    start_date: str = "2024-01-02"
    end_date: str = "2025-12-31"
    allocation_method: str = "equal"
    custom_weights: Optional[Dict[str, float]] = None


class PortfolioOptimizeReq(BaseModel):
    """组合权重优化请求体。"""
    symbols: List[str] = ["600519.SH", "300750.SZ", "002594.SZ"]
    method: str = "equal_weight"  # equal_weight/min_variance/risk_parity/mean_variance
    start_date: str = "2024-01-02"
    end_date: str = "2025-12-31"
    max_weight: float = 0.3
    risk_free_rate: float = 0.02


class StrategyPortfolioReq(BaseModel):
    """多策略组合回测请求体（同一标的、多策略资金分配）。"""
    symbol: str = "600519.SH"
    strategies: List[str] = []
    weights: Optional[List[float]] = None
    start_date: str = "2024-01-02"
    end_date: str = "2025-12-31"


class BackupRestoreReq(BaseModel):
    """从备份恢复数据库的请求体。"""
    backup_file: str = ""


class LeaderboardReq(BaseModel):
    """策略排行榜请求体。"""
    symbol: str = "600519.SH"
    start_date: str = "2024-01-02"
    end_date: str = "2025-12-31"


class AggregateSignalReq(BaseModel):
    """多策略聚合信号请求体。"""
    symbol: str = "600519.SH"
    threshold: float = 0.2
    start_date: str = "2024-01-02"


_EMPTY_METRICS = {
    "累计收益率": 0.0, "年化收益率": 0.0, "最大回撤": 0.0,
    "夏普比率": 0.0, "胜率": 0.0, "盈亏比": 0.0,
    "交易次数": 0, "总盈利": 0.0, "总亏损": 0.0,
    "订单成交率": 0.0, "平均持仓时间": 0.0,
}


def _run_single_backtest(symbol: str, strategy_name: str,
                         start_date: str, end_date: str,
                         use_jev: bool = False,
                         order_type: str = "market") -> Dict[str, Any]:
    """运行单个策略回测，返回 params/metrics/equity_curve/benchmark_curve/trades。"""
    from backtest.engine import BacktestEngine
    from jev.jev_engine import JevDecisionEngine

    sim = manager.get(symbol)
    df = sim.klines.copy()
    df = df.loc[(df.index >= pd.Timestamp(start_date)) &
                (df.index <= pd.Timestamp(end_date))]
    if len(df) < 30:
        return {
            "params": {"symbol": sim.symbol, "strategy": strategy_name,
                       "start_date": start_date, "end_date": end_date, "use_jev": use_jev},
            "metrics": dict(_EMPTY_METRICS),
            "equity_curve": [],
            "benchmark_curve": [],
            "trades": [],
        }

    strategy = _strategy_instance(strategy_name)
    jev_engine = JevDecisionEngine(mock_mode=True) if use_jev else None

    bt_cfg = manager.cfg.get("backtest", {})
    engine = BacktestEngine(
        initial_capital=float(bt_cfg.get("initial_capital", 1_000_000.0)),
        commission_rate=float(bt_cfg.get("commission_rate", 0.00025)),
        stamp_tax_rate=float(bt_cfg.get("stamp_tax_rate", 0.0005)),
        slippage_rate=float(bt_cfg.get("slippage_rate", 0.001)),
        risk_free_rate=float(bt_cfg.get("risk_free_rate", 0.02)),
        trading_days=int(bt_cfg.get("trading_days_per_year", 252)),
        jev_engine=jev_engine,
        order_type=order_type,
    )
    result = engine.run(df, strategy, symbol=sim.symbol)

    equity_curve = [
        [d.strftime("%Y-%m-%d"), round(float(v), 2)]
        for d, v in result.equity_curve.items()
    ]
    benchmark_curve = [
        [d.strftime("%Y-%m-%d"), round(float(v), 2)]
        for d, v in result.benchmark_curve.items()
    ]
    trades = [
        {
            "date": t.date.strftime("%Y-%m-%d"),
            "action": t.action,
            "price": round(float(t.price), 2),
            "shares": int(t.shares),
            "amount": round(float(t.amount), 2),
            "pnl": round(float(t.pnl), 2) if t.pnl is not None else None,
            "reason": t.reason,
        }
        for t in result.trades
    ]
    return {
        "params": {
            "symbol": sim.symbol, "strategy": strategy_name,
            "start_date": start_date, "end_date": end_date, "use_jev": use_jev,
        },
        "metrics": result.metrics,
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
        "trades": trades,
    }


def _run_backtest_to_result(symbol: str, strategy_name: str,
                            start_date: str, end_date: str,
                            use_jev: bool = False):
    """运行回测并返回原始 BacktestResult（供 HTML 报告生成使用）。

    与 :func:`_run_single_backtest` 使用相同的引擎配置，但保留完整的
    BacktestResult 对象（含未序列化的交易明细、净值序列），便于 ReportGenerator 渲染。
    数据不足 30 条时返回空结果（不崩溃）。
    """
    from backtest.engine import BacktestEngine, BacktestResult
    from backtest.metrics import metrics_to_dataframe
    from jev.jev_engine import JevDecisionEngine

    sim = manager.get(symbol)
    df = sim.klines.copy()
    df = df.loc[(df.index >= pd.Timestamp(start_date)) &
                (df.index <= pd.Timestamp(end_date))]
    if len(df) < 30:
        # 数据不足，返回空 BacktestResult
        empty_series = pd.Series(dtype=float)
        return BacktestResult(
            equity_curve=empty_series,
            benchmark_curve=empty_series,
            trades=[],
            metrics=dict(_EMPTY_METRICS),
            metrics_df=metrics_to_dataframe(dict(_EMPTY_METRICS)),
            daily_returns=empty_series,
            positions_history=[],
        )

    strategy = _strategy_instance(strategy_name)
    jev_engine = JevDecisionEngine(mock_mode=True) if use_jev else None

    bt_cfg = manager.cfg.get("backtest", {})
    engine = BacktestEngine(
        initial_capital=float(bt_cfg.get("initial_capital", 1_000_000.0)),
        commission_rate=float(bt_cfg.get("commission_rate", 0.00025)),
        stamp_tax_rate=float(bt_cfg.get("stamp_tax_rate", 0.0005)),
        slippage_rate=float(bt_cfg.get("slippage_rate", 0.001)),
        risk_free_rate=float(bt_cfg.get("risk_free_rate", 0.02)),
        trading_days=int(bt_cfg.get("trading_days_per_year", 252)),
        jev_engine=jev_engine,
    )
    return engine.run(df, strategy, symbol=sim.symbol)


@app.get("/api/backtest/report")
async def backtest_report(
    symbol: str = Query("600519.SH", description="标的代码"),
    strategy: str = Query("ma_cross", description="策略名称"),
    start_date: str = Query("2024-01-02", description="开始日期 YYYY-MM-DD"),
    end_date: str = Query("2025-12-31", description="结束日期 YYYY-MM-DD"),
    use_jev: bool = Query(False, description="是否启用 Jev 信号过滤"),
):
    """运行回测并返回自包含 HTML 报告（Content-Type: text/html）。

    报告同时缓存到 output/reports/<symbol>_<strategy>_<timestamp>.html。
    """
    from datetime import datetime

    sym = normalize_symbol(symbol)
    if sym not in SYMBOL_SET:
        return err(40001, f"标的不在股票池内: {symbol}")
    if strategy not in STRATEGY_NAMES:
        return err(40002, f"未知策略: {strategy}，可选: {STRATEGY_NAMES}")
    try:
        result = _run_backtest_to_result(sym, strategy, start_date, end_date, use_jev)
    except Exception as e:
        logger.exception("回测报告生成失败")
        return err(50001, f"回测失败: {e}", http_status=500)

    from backtest.report_generator import ReportGenerator

    reports_dir = PROJECT_ROOT / "output" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = reports_dir / f"{sym}_{strategy}_{ts}.html"

    title = f"回测报告 - {sym} - {strategy}"
    params = {
        "标的": sym,
        "策略": strategy,
        "开始日期": start_date,
        "结束日期": end_date,
        "Jev信号过滤": "是" if use_jev else "否",
    }
    try:
        html_path = ReportGenerator().generate_html_report(
            result, str(out_path), title=title, params=params,
        )
    except Exception as e:
        logger.exception("HTML 报告渲染失败")
        return err(50002, f"报告渲染失败: {e}", http_status=500)

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    logger.info("回测报告已导出: %s", html_path)
    return HTMLResponse(content=html)


@app.get("/api/backtest")
async def get_backtest_static():
    """保留旧接口：返回静态 dashboard_data.json。"""
    data_path = Path(__file__).parent / "dashboard_data.json"
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            return ok(json.load(f))
    return ok({})


@app.post("/api/backtest")
async def run_backtest(req: BacktestReq):
    # 参数校验：标的必须在股票池内
    sym = normalize_symbol(req.symbol)
    if sym not in SYMBOL_SET:
        return err(40001, f"标的不在股票池内: {req.symbol}")
    # 参数校验：策略必须有效
    if req.strategy not in STRATEGY_NAMES:
        return err(40002, f"未知策略: {req.strategy}，可选: {STRATEGY_NAMES}")
    try:
        result = _run_single_backtest(sym, req.strategy, req.start_date,
                                      req.end_date, req.use_jev,
                                      order_type=req.order_type)
        return ok(result)
    except Exception as e:
        logger.exception("回测失败")
        return err(50001, f"回测失败: {e}", http_status=500)


@app.post("/api/backtest_compare")
async def backtest_compare(req: BacktestCompareReq):
    """多策略批量回测对比：返回每个策略的 metrics 与 equity_curve。"""
    sym = normalize_symbol(req.symbol)
    if sym not in SYMBOL_SET:
        return err(40001, f"标的不在股票池内: {req.symbol}")
    strategies = [s for s in (req.strategies or []) if s]
    if not strategies:
        return err(40003, "至少选择 1 个策略")
    invalid = [s for s in strategies if s not in STRATEGY_NAMES]
    if invalid:
        return err(40002, f"未知策略: {invalid}，可选: {STRATEGY_NAMES}")

    results = []
    benchmark_curve = []
    for sname in strategies:
        try:
            r = _run_single_backtest(sym, sname, req.start_date, req.end_date,
                                     use_jev=False)
            results.append({
                "strategy": sname,
                "label": next((m["label"] for m in STRATEGIES_META if m["name"] == sname), sname),
                "metrics": r["metrics"],
                "equity_curve": r["equity_curve"],
                "trades": r["trades"],
            })
            if not benchmark_curve and r["benchmark_curve"]:
                benchmark_curve = r["benchmark_curve"]
        except Exception as e:
            logger.exception("策略 %s 回测失败", sname)
            results.append({
                "strategy": sname,
                "label": next((m["label"] for m in STRATEGIES_META if m["name"] == sname), sname),
                "metrics": dict(_EMPTY_METRICS),
                "equity_curve": [],
                "trades": [],
                "error": str(e),
            })
    return ok({
        "symbol": sym,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "benchmark_curve": benchmark_curve,
        "results": results,
    })


@app.post("/api/backtest_portfolio")
async def backtest_portfolio(req: PortfolioBacktestReq):
    """多标的组合回测：等权/波动率倒数/自定义权重分配资金，逐标的独立回测后合并。"""
    symbols = [normalize_symbol(s) for s in (req.symbols or []) if s]
    if len(symbols) < 1:
        return err(40003, "至少选择 1 只标的")
    invalid_syms = [s for s in symbols if s not in SYMBOL_SET]
    if invalid_syms:
        return err(40001, f"标的不在股票池内: {invalid_syms}")
    if req.strategy not in STRATEGY_NAMES:
        return err(40002, f"未知策略: {req.strategy}，可选: {STRATEGY_NAMES}")
    if req.allocation_method not in ("equal", "volatility_inverse", "custom"):
        return err(40004, "allocation_method 必须为 equal/volatility_inverse/custom")
    if req.allocation_method == "custom" and not req.custom_weights:
        return err(40005, "allocation_method=custom 时必须提供 custom_weights")

    try:
        from backtest.portfolio_engine import PortfolioBacktestEngine

        data: Dict[str, Any] = {}
        for sym in symbols:
            sim = manager.get(sym)
            df = sim.klines.copy()
            df = df.loc[(df.index >= pd.Timestamp(req.start_date)) &
                        (df.index <= pd.Timestamp(req.end_date))]
            if len(df) >= 30:
                data[sym] = df

        if not data:
            return err(40006, "所选标的在日期区间内有效数据不足(<30 根K线)")

        strategy = _strategy_instance(req.strategy)
        bt_cfg = manager.cfg.get("backtest", {})
        engine = PortfolioBacktestEngine(
            initial_capital=float(bt_cfg.get("initial_capital", 1_000_000.0)),
            commission_rate=float(bt_cfg.get("commission_rate", 0.00025)),
            stamp_tax_rate=float(bt_cfg.get("stamp_tax_rate", 0.0005)),
            slippage_rate=float(bt_cfg.get("slippage_rate", 0.001)),
            risk_free_rate=float(bt_cfg.get("risk_free_rate", 0.02)),
            trading_days=int(bt_cfg.get("trading_days_per_year", 252)),
            allocation_method=req.allocation_method,
            custom_weights=req.custom_weights,
        )
        result = engine.run(data, strategy)

        portfolio_curve = [
            [d.strftime("%Y-%m-%d"), round(float(v), 2)]
            for d, v in result.portfolio_equity_curve.items()
        ]
        benchmark_curve = [
            [d.strftime("%Y-%m-%d"), round(float(v), 2)]
            for d, v in result.benchmark_curve.items()
        ]
        symbol_details = {}
        for sym, res in result.symbol_results.items():
            symbol_details[sym] = {
                "metrics": result.symbol_metrics[sym],
                "equity_curve": [
                    [d.strftime("%Y-%m-%d"), round(float(v), 2)]
                    for d, v in res.equity_curve.items()
                ],
                "trades": [
                    {
                        "date": t.date.strftime("%Y-%m-%d"),
                        "action": t.action, "price": round(float(t.price), 2),
                        "shares": int(t.shares), "amount": round(float(t.amount), 2),
                        "pnl": round(float(t.pnl), 2) if t.pnl is not None else None,
                        "reason": t.reason,
                    }
                    for t in res.trades
                ],
            }

        return ok({
            "params": {
                "symbols": symbols, "strategy": req.strategy,
                "start_date": req.start_date, "end_date": req.end_date,
                "allocation_method": req.allocation_method,
                "symbol_weights": result.symbol_weights,
            },
            "metrics": result.portfolio_metrics,
            "portfolio_equity_curve": portfolio_curve,
            "benchmark_curve": benchmark_curve,
            "trades_count": len(result.all_trades),
            "symbol_details": symbol_details,
        })
    except Exception as e:
        logger.exception("组合回测失败")
        return err(50001, f"组合回测失败: {e}", http_status=500)


@app.post("/api/strategy_portfolio/backtest")
async def run_strategy_portfolio_backtest(req: StrategyPortfolioReq):
    """多策略组合回测：同一标的上叠加多个策略，按权重分配资金，各自独立回测后合并净值。"""
    sym = normalize_symbol(req.symbol)
    if sym not in SYMBOL_SET:
        return err(40001, f"标的不在股票池内: {req.symbol}")

    strategies = [s for s in (req.strategies or []) if s]
    if not strategies:
        return err(40003, "至少选择 1 个策略")
    invalid = [s for s in strategies if s not in STRATEGY_NAMES]
    if invalid:
        return err(40002, f"未知策略: {invalid}，可选: {STRATEGY_NAMES}")

    if req.weights is not None and len(req.weights) != len(strategies):
        return err(40004, f"weights 长度({len(req.weights)})必须与 strategies 长度({len(strategies)})一致")

    try:
        from strategies.strategy_portfolio import StrategyPortfolio

        sim = manager.get(sym)
        df = sim.klines.copy()
        df = df.loc[(df.index >= pd.Timestamp(req.start_date)) &
                    (df.index <= pd.Timestamp(req.end_date))]
        if len(df) < 30:
            return err(40006, "所选标的在日期区间内有效数据不足(<30 根K线)")

        bt_cfg = manager.cfg.get("backtest", {})
        sp = StrategyPortfolio(
            strategies=strategies,
            weights=req.weights,
            initial_capital=float(bt_cfg.get("initial_capital", 1_000_000.0)),
            commission_rate=float(bt_cfg.get("commission_rate", 0.00025)),
            stamp_tax_rate=float(bt_cfg.get("stamp_tax_rate", 0.0005)),
            slippage_rate=float(bt_cfg.get("slippage_rate", 0.001)),
            risk_free_rate=float(bt_cfg.get("risk_free_rate", 0.02)),
            trading_days=int(bt_cfg.get("trading_days_per_year", 252)),
        )
        result = sp.run(df, symbol=sim.symbol)

        portfolio_curve = [
            [d.strftime("%Y-%m-%d"), round(float(v), 2)]
            for d, v in result.portfolio_equity_curve.items()
        ]
        first_result = next(iter(result.strategy_results.values()), None)
        benchmark_curve = [
            [d.strftime("%Y-%m-%d"), round(float(v), 2)]
            for d, v in (first_result.benchmark_curve.items() if first_result is not None else [])
        ]

        strategy_details: Dict[str, Any] = {}
        for name, res in result.strategy_results.items():
            strategy_details[name] = {
                "metrics": res.metrics,
                "equity_curve": [
                    [d.strftime("%Y-%m-%d"), round(float(v), 2)]
                    for d, v in res.equity_curve.items()
                ],
                "trades": [
                    {
                        "date": t.date.strftime("%Y-%m-%d"),
                        "action": t.action,
                        "price": round(float(t.price), 2),
                        "shares": int(t.shares),
                        "amount": round(float(t.amount), 2),
                        "pnl": round(float(t.pnl), 2) if t.pnl is not None else None,
                        "reason": t.reason,
                    }
                    for t in res.trades
                ],
            }

        return ok({
            "params": {
                "symbol": sim.symbol,
                "strategies": strategies,
                "start_date": req.start_date,
                "end_date": req.end_date,
                "strategy_weights": result.strategy_weights,
            },
            "portfolio_metrics": result.portfolio_metrics,
            "portfolio_equity_curve": portfolio_curve,
            "benchmark_curve": benchmark_curve,
            "strategy_details": strategy_details,
            "contribution": result.contribution,
            "conflict_log": result.conflict_log,
        })
    except Exception as e:
        logger.exception("多策略组合回测失败")
        return err(50001, f"多策略组合回测失败: {e}", http_status=500)


# ---- 参数优化（P2-1） ------------------------------------------------------

class GridSearchReq(BaseModel):
    """网格搜索优化请求体。"""
    symbol: str = "600519.SH"
    strategy: str = "ma_cross"
    param_grid: Dict[str, list]
    start_date: str = "2024-01-02"
    end_date: str = "2025-12-31"
    objective: str = "sharpe"  # sharpe / return / drawdown
    top_n: int = 10
    max_combos: int = 200


_STRATEGY_CLASS_MAP: Dict[str, type] = {}


def _get_strategy_class(name: str) -> type:
    """根据策略名称获取策略类（延迟导入，避免循环依赖）。"""
    if not _STRATEGY_CLASS_MAP:
        from strategies.ma_cross import MACrossStrategy
        from strategies.bollinger import BollingerStrategy
        from strategies.momentum_breakout import MomentumBreakoutStrategy
        from strategies.indicator_combo import IndicatorComboStrategy
        _STRATEGY_CLASS_MAP.update({
            "ma_cross": MACrossStrategy,
            "bollinger": BollingerStrategy,
            "momentum_breakout": MomentumBreakoutStrategy,
            "indicator_combo": IndicatorComboStrategy,
        })
    cls = _STRATEGY_CLASS_MAP.get(name)
    if cls is None:
        raise ValueError(f"未知策略: {name}")
    return cls


@app.post("/api/optimize/grid")
async def optimize_grid(req: GridSearchReq):
    """策略参数网格搜索：遍历参数组合，按优化目标排序返回 Top N。"""
    sym = normalize_symbol(req.symbol)
    if sym not in SYMBOL_SET:
        return err(40001, f"标的不在股票池内: {req.symbol}")
    if req.strategy not in STRATEGY_NAMES:
        return err(40002, f"未知策略: {req.strategy}，可选: {STRATEGY_NAMES}")
    if req.objective not in ("sharpe", "return", "drawdown"):
        return err(40007, f"objective 必须为 sharpe/return/drawdown，收到: {req.objective}")
    if not req.param_grid or any(len(v) == 0 for v in req.param_grid.values()):
        return err(40003, "param_grid 不能为空或包含空列表")

    try:
        from optimization.grid_search import GridSearchOptimizer

        sim = manager.get(sym)
        df = sim.klines.copy()
        df = df.loc[(df.index >= pd.Timestamp(req.start_date)) &
                    (df.index <= pd.Timestamp(req.end_date))]
        if len(df) < 30:
            return err(40006, "所选标的在日期区间内有效数据不足(<30 根K线)")

        strategy_cls = _get_strategy_class(req.strategy)
        bt_cfg = manager.cfg.get("backtest", {})
        optimizer = GridSearchOptimizer(max_combos=req.max_combos)
        results = optimizer.optimize(
            data=df,
            strategy_class=strategy_cls,
            param_grid=req.param_grid,
            symbol=sym,
            objective=req.objective,
            top_n=req.top_n,
            initial_capital=float(bt_cfg.get("initial_capital", 1_000_000.0)),
            commission_rate=float(bt_cfg.get("commission_rate", 0.00025)),
            stamp_tax_rate=float(bt_cfg.get("stamp_tax_rate", 0.0005)),
            slippage_rate=float(bt_cfg.get("slippage_rate", 0.001)),
            risk_free_rate=float(bt_cfg.get("risk_free_rate", 0.02)),
            trading_days=int(bt_cfg.get("trading_days_per_year", 252)),
        )

        # 计算总组合数
        total_combos = 1
        for v in req.param_grid.values():
            total_combos *= len(v)

        # 添加排名
        ranked = [{"rank": i, **r} for i, r in enumerate(results, 1)]

        return ok({
            "strategy": req.strategy,
            "symbol": sym,
            "objective": req.objective,
            "total_combos": total_combos,
            "top_n": req.top_n,
            "results": ranked,
        })
    except Exception as e:
        logger.exception("网格搜索优化失败")
        return err(50001, f"网格搜索优化失败: {e}", http_status=500)


@app.post("/api/optimize/portfolio")
async def optimize_portfolio(req: PortfolioOptimizeReq):
    """组合权重优化：等权/最小方差/风险平价/均值方差，返回最优权重组合。"""
    from optimization.portfolio_optimizer import ALLOWED_METHODS, PortfolioOptimizer

    symbols = [normalize_symbol(s) for s in (req.symbols or []) if s]
    if len(symbols) < 1:
        return err(40003, "至少选择 1 只标的")
    invalid_syms = [s for s in symbols if s not in SYMBOL_SET]
    if invalid_syms:
        return err(40001, f"标的不在股票池内: {invalid_syms}")
    if req.method not in ALLOWED_METHODS:
        return err(40004, f"method 必须为 {list(ALLOWED_METHODS)} 之一，收到: {req.method}")

    try:
        bt_cfg = manager.cfg.get("backtest", {})
        trading_days = int(bt_cfg.get("trading_days_per_year", 252))

        data: Dict[str, Any] = {}
        for sym in symbols:
            sim = manager.get(sym)
            df = sim.klines.copy()
            df = df.loc[(df.index >= pd.Timestamp(req.start_date)) &
                        (df.index <= pd.Timestamp(req.end_date))]
            if len(df) >= 30:
                data[sym] = df

        if not data:
            return err(40006, "所选标的在日期区间内有效数据不足(<30 根K线)")

        optimizer = PortfolioOptimizer(
            risk_free_rate=req.risk_free_rate,
            max_weight=req.max_weight,
            trading_days=trading_days,
        )
        result = optimizer.optimize(list(data.keys()), data, method=req.method)

        return ok({
            "weights": result.weights,
            "expected_return": round(float(result.expected_return), 6),
            "expected_volatility": round(float(result.expected_volatility), 6),
            "sharpe": round(float(result.sharpe), 6),
            "method": result.method,
        })
    except Exception as e:
        logger.exception("组合权重优化失败")
        return err(50001, f"组合权重优化失败: {e}", http_status=500)


# ---- 绩效归因（P2-2） ------------------------------------------------------

class AttributionReq(BaseModel):
    """绩效归因请求体（简化版：传入 symbol+strategy+日期，内部运行回测后归因）。"""
    symbol: str = "600519.SH"
    strategy: str = "ma_cross"
    start_date: str = "2024-01-02"
    end_date: str = "2025-12-31"
    lookforward_days: int = 5


@app.post("/api/attribution")
async def attribution(req: AttributionReq):
    """绩效归因分析：内部运行回测后输出交易/时间/持仓/策略/风险调整五维归因报告。"""
    sym = normalize_symbol(req.symbol)
    if sym not in SYMBOL_SET:
        return err(40001, f"标的不在股票池内: {req.symbol}")
    if req.strategy not in STRATEGY_NAMES:
        return err(40002, f"未知策略: {req.strategy}，可选: {STRATEGY_NAMES}")

    try:
        from analysis.attribution import PerformanceAttribution
        from backtest.engine import BacktestEngine

        sim = manager.get(sym)
        df = sim.klines.copy()
        df = df.loc[(df.index >= pd.Timestamp(req.start_date)) &
                    (df.index <= pd.Timestamp(req.end_date))]
        if len(df) < 30:
            return err(40006, "所选标的在日期区间内有效数据不足(<30 根K线)")

        strategy = _strategy_instance(req.strategy)
        bt_cfg = manager.cfg.get("backtest", {})
        engine = BacktestEngine(
            initial_capital=float(bt_cfg.get("initial_capital", 1_000_000.0)),
            commission_rate=float(bt_cfg.get("commission_rate", 0.00025)),
            stamp_tax_rate=float(bt_cfg.get("stamp_tax_rate", 0.0005)),
            slippage_rate=float(bt_cfg.get("slippage_rate", 0.001)),
            risk_free_rate=float(bt_cfg.get("risk_free_rate", 0.02)),
            trading_days=int(bt_cfg.get("trading_days_per_year", 252)),
        )
        result = engine.run(df, strategy, symbol=sym)

        analyzer = PerformanceAttribution(
            risk_free_rate=float(bt_cfg.get("risk_free_rate", 0.02)),
            trading_days=int(bt_cfg.get("trading_days_per_year", 252)),
        )
        report = analyzer.analyze(
            equity_curve=result.equity_curve,
            trades=result.trades,
            symbols=[sym],
            lookforward_days=req.lookforward_days,
            data={sym: df},
        )

        return ok({
            "params": {
                "symbol": sym,
                "strategy": req.strategy,
                "start_date": req.start_date,
                "end_date": req.end_date,
            },
            "metrics": result.metrics,
            "attribution": report,
        })
    except Exception as e:
        logger.exception("绩效归因失败")
        return err(50001, f"绩效归因失败: {e}", http_status=500)


# ---- Jev 决策 -------------------------------------------------------------

@app.get("/api/jev_decision")
async def jev_decision(symbol: str = Query(None)):
    from jev.jev_engine import JevDecisionEngine

    try:
        sim = manager.get(symbol)
        strategy_signal, strategy_conf = _latest_strategy_signal(sim)
        market_state = _build_market_state_dict(sim)
        if market_state is None:
            return err(50002, "insufficient data for jev decision", http_status=500)

        mode = "mock"
        latency_ms = 0.0
        probabilities = {"buy": 0.0, "sell": 0.0, "hold": 1.0}

        # 优先尝试真实 laya_mlx 子进程
        try:
            real = await asyncio.to_thread(jev_client.predict, market_state)
            probabilities = real["probabilities"]
            latency_ms = real.get("latency_ms", 0.0)
            mode = "real"
        except Exception as e:
            logger.info("Jev real 推理失败，降级 mock: %s", e)
            mock_engine = JevDecisionEngine(mock_mode=True)
            states_list = [
                {"feature": k, "value": v} for k, v in market_state.items()
            ]
            probabilities = mock_engine._mock_evaluate(states_list, strategy_signal, strategy_conf)
            latency_ms = 0.0
            mode = "mock"

        # 决策逻辑（与 JevDecisionEngine.evaluate 一致）
        final_action = max(probabilities, key=probabilities.get)
        final_confidence = float(probabilities[final_action])

        if final_action == "hold":
            reason = "Jev 建议观望"
            executed = False
        elif final_action != strategy_signal and strategy_signal != "hold":
            reason = f"Jev 方向({final_action})与策略方向({strategy_signal})冲突"
            executed = False
        elif final_confidence < JEV_THRESHOLD:
            reason = f"置信度 {final_confidence:.3f} 低于阈值 {JEV_THRESHOLD}"
            executed = False
        else:
            executed = True
            reason = "通过 Jev 过滤，执行交易"

        return ok({
            "symbol": sim.symbol,
            "strategy": current_strategy,
            "strategy_signal": strategy_signal,
            "strategy_confidence": round(strategy_conf, 4),
            "market_state": {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
                              for k, v in market_state.items()},
            "probabilities": {k: round(float(v), 4) for k, v in probabilities.items()},
            "final_action": final_action,
            "final_confidence": round(final_confidence, 4),
            "threshold": JEV_THRESHOLD,
            "executed": executed,
            "reason": reason,
            "mode": mode,
            "latency_ms": round(float(latency_ms), 1),
        })
    except Exception as e:
        logger.exception("Jev 决策失败")
        return err(50003, f"jev decision failed: {e}", http_status=500)


# ---- 实时交易 -------------------------------------------------------------

@app.get("/api/trade_log")
async def get_trade_log(limit: int = Query(50, ge=1, le=200)):
    """返回最近的实时交易决策日志（内存 + SQLite 历史合并）。"""
    # 内存中的当前会话日志
    mem_logs = trader.get_logs(limit)
    # DB 中的历史成交记录
    try:
        db_trades = db.get_trades(limit=limit)
    except Exception:
        logger.exception("读取DB交易记录失败")
        db_trades = []
    return ok({
        "logs": mem_logs,
        "total": len(trader.trade_log),
        "history_trades": db_trades,
        "history_count": len(db_trades),
    })


@app.get("/api/trade_status")
async def get_trade_status():
    """返回当前实时交易状态。"""
    return ok(trader.get_status())


class TradeToggleReq(BaseModel):
    """实时交易开关请求体。"""
    running: bool


@app.post("/api/trade_toggle")
async def trade_toggle(req: TradeToggleReq, http_req: Request):
    """开启/关闭实时交易。"""
    token_info = getattr(http_req.state, "token_info", None) or {}
    operator = str(token_info.get("name", ""))
    if req.running and not trader.running:
        trader.start()
        audit_logger.log(
            operator=operator,
            action_type=ActionType.STRATEGY_TOGGLE,
            target="realtime_trading",
            params={"running": True},
            result="started",
            ip=http_req.client.host if http_req.client else "",
        )
        return ok({"running": True, "message": "实时交易已开启"})
    elif not req.running and trader.running:
        trader.stop()
        audit_logger.log(
            operator=operator,
            action_type=ActionType.STRATEGY_TOGGLE,
            target="realtime_trading",
            params={"running": False},
            result="stopped",
            ip=http_req.client.host if http_req.client else "",
        )
        return ok({"running": False, "message": "实时交易已关闭"})
    else:
        return ok({"running": trader.running, "message": "状态未变更"})


# ---- 持久化查询 -----------------------------------------------------------

@app.get("/api/jev_decisions")
async def get_jev_decisions(
    limit: int = Query(50, ge=1, le=200),
    symbol: Optional[str] = None,
):
    """查询历史 Jev 决策审计记录。"""
    try:
        decisions = db.get_jev_decisions(limit=limit, symbol=symbol)
        total = db.count_table("jev_decisions", "symbol = ?" if symbol else "", (symbol,) if symbol else ())
        return ok({"decisions": decisions, "total": total})
    except Exception as e:
        logger.exception("查询Jev决策失败")
        return ok({"decisions": [], "total": 0, "error": str(e)})


@app.get("/api/jev/evaluation")
async def get_jev_evaluation(
    days: int = Query(5, ge=1, le=60),
    account_id: str = Query("acc_1"),
    limit: int = Query(500, ge=1, le=5000),
):
    """Jev 历史决策质量评估报告。

    关联决策日之后 N 个交易日的收盘价，统计买入/卖出/观望的对错、
    置信度分桶校准度、以及按策略信号的 Jev 过滤效果。
    无决策数据时返回空报告（不报错）。
    """
    try:
        from jev.decision_evaluator import DecisionEvaluator

        price_data: Dict[str, Any] = {}
        for sym, sim in getattr(manager, "sims", {}).items():
            kl = getattr(sim, "klines", None)
            if kl is not None and not kl.empty and "close" in kl.columns:
                price_data[sym] = kl

        evaluator = DecisionEvaluator(
            db_path=db.db_path,
            price_data=price_data or None,
        )
        report = evaluator.evaluate(days=days, account_id=account_id, limit=limit)
        return ok(report)
    except Exception as e:
        logger.exception("Jev 决策质量评估失败")
        empty = {
            "summary": {
                "total_decisions": 0, "evaluated_decisions": 0,
                "overall_accuracy": 0.0,
                "correct_buy": 0, "wrong_buy": 0,
                "correct_sell": 0, "wrong_sell": 0,
                "correct_hold": 0, "wrong_hold": 0,
            },
            "confidence_buckets": {
                "0-0.4": {"count": 0, "correct": 0, "accuracy": 0.0},
                "0.4-0.6": {"count": 0, "correct": 0, "accuracy": 0.0},
                "0.6-0.8": {"count": 0, "correct": 0, "accuracy": 0.0},
                "0.8-1.0": {"count": 0, "correct": 0, "accuracy": 0.0},
            },
            "by_strategy": {},
            "recent_decisions": [],
            "params": {"days": days, "eval_date": "", "error": str(e)},
        }
        return ok(empty)


@app.get("/api/account_snapshots")
async def get_account_snapshots(
    account_id: str = Query("balanced"),
    limit: int = Query(100, ge=1, le=500),
):
    """查询账户历史快照（用于净值曲线）。"""
    try:
        snapshots = db.get_account_snapshots(account_id=account_id, limit=limit)
        total = db.count_table("account_snapshots", "account_id = ?", (account_id,))
        return ok({"snapshots": list(reversed(snapshots)), "total": total})
    except Exception as e:
        logger.exception("查询账户快照失败")
        return ok({"snapshots": [], "total": 0, "error": str(e)})


@app.get("/api/daily_reports")
async def get_daily_reports(
    account_id: Optional[str] = None,
    limit: int = Query(30, ge=1, le=365),
):
    """查询每日绩效报告。"""
    try:
        reports = db.get_daily_reports(account_id=account_id, limit=limit)
        return ok({"reports": reports, "total": len(reports)})
    except Exception as e:
        logger.exception("查询每日报告失败")
        return ok({"reports": [], "total": 0, "error": str(e)})


@app.get("/api/daily_reports/export")
async def export_daily_reports(
    account_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """导出指定账户每日报告为 CSV 文件下载。"""
    try:
        csv_text = report_generator.export_csv(
            account_id=account_id,
            start_date=start_date,
            end_date=end_date,
        )
        filename = f"daily_reports_{account_id}.csv"
        return Response(
            content=csv_text,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.exception("导出每日报告 CSV 失败")
        return ok({"error": str(e)})


@app.get("/api/trade_stats")
async def get_trade_stats(account_id: Optional[str] = None):
    """获取交易统计概览（总交易数/总盈亏/胜率）。"""
    try:
        stats = db.get_trade_stats(account_id=account_id)
        return ok(stats)
    except Exception as e:
        logger.exception("查询交易统计失败")
        return ok({"total_trades": 0, "total_pnl": 0, "win_rate": 0, "error": str(e)})


# ---- 告警系统 -------------------------------------------------------------

@app.get("/api/alerts")
async def get_alerts(
    level: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
):
    """查询告警历史。"""
    alerts = alert_manager.get_history(level=level, category=category, limit=limit)
    return ok({"alerts": alerts, "stats": alert_manager.stats})


@app.post("/api/alerts/test")
async def test_alert_webhook():
    """测试Webhook连通性。"""
    if not alert_manager.webhook_url:
        return ok({"success": False, "message": "未配置Webhook URL"})
    alert_manager.reset_webhook()
    success = alert_manager.test_webhook()
    return ok({
        "success": success,
        "webhook_url": alert_manager.webhook_url,
        "message": "Webhook连通" if success else "Webhook不可达，已降级为日志",
    })


@app.post("/api/alerts/trigger")
async def trigger_test_alert():
    """触发一条测试告警（用于验证Webhook和前端展示）。"""
    from monitoring.alert import AlertLevel
    alert = alert_manager.alert(
        AlertLevel.INFO, "test", "测试告警 - 系统正常",
        symbol="600519.SH", force=True,
    )
    return ok({"alert": alert.to_dict(), "stats": alert_manager.stats})


# ---- 多渠道通知管理 ---------------------------------------------------------

class NotificationTestReq(BaseModel):
    """发送测试通知请求体。"""
    channel: str = "wecom"
    message: str = "连通性测试"


class NotificationSendReq(BaseModel):
    """手动发送通知请求体。"""
    level: str = "WARNING"
    event_type: str = "system"
    title: str = "手动通知"
    context: Dict[str, Any] = {}
    template: str = ""


@app.post("/api/notification/test")
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


@app.get("/api/notification/status")
async def notification_status():
    """查询各渠道配置状态与最近发送记录。"""
    try:
        return ok(notifier_manager.get_status())
    except Exception as e:
        logger.exception("查询通知状态失败")
        return err(50002, f"查询状态异常: {e}")


@app.post("/api/notification/send")
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


# ---- 配置热加载 -------------------------------------------------------------

@app.post("/api/config/reload")
async def config_reload():
    """手动触发配置热加载（重新读取 config.yaml 并应用可热加载节）。

    不可热加载的节（data.api_key / jev.base_url / 端口 / 数据库路径 /
    logging.log_dir）本次重载不生效，需重启进程。重载成功会记录 CONFIG_CHANGE
    审计事件。
    """
    try:
        result = hot_reload_manager.reload_config()
        return ok(result, message="配置热加载完成")
    except Exception as e:
        logger.exception("配置热加载失败")
        return err(50010, f"配置热加载失败: {e}", http_status=400)


@app.get("/api/config/current")
async def config_current():
    """返回当前全局配置（敏感字段已脱敏为 ***）。"""
    cfg = hot_reload_manager.get_current_config(sanitize=True)
    return ok({"config": cfg})


@app.get("/api/config/hot_reloadable")
async def config_hot_reloadable():
    """返回可热加载 / 不可热加载的配置节列表。"""
    return ok({
        "keys": hot_reload_manager.get_hot_reloadable_keys(),
        "non_reloadable": hot_reload_manager.get_non_hot_reloadable_keys(),
    })


# ---- 结构化日志查询 / 动态调级 ---------------------------------------------

class LogLevelReq(BaseModel):
    """/api/logs/level 请求体。"""

    level: str


@app.get("/api/logs/recent")
async def logs_recent(
    level: Optional[str] = Query(None),
    limit: int = Query(100),
):
    """返回最近的结构化日志（可按级别过滤）。"""
    try:
        logs = get_recent_logs(level=level, limit=limit)
        return ok({"logs": logs, "total": len(logs)})
    except Exception as e:
        logger.exception("查询最近日志失败")
        return err(50020, f"查询日志失败: {e}")


@app.post("/api/logs/level")
async def logs_set_level(req: LogLevelReq):
    """动态设置根 logger 级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）。"""
    try:
        new_level = set_log_level(req.level)
        return ok({"level": new_level, "changed": True})
    except ValueError as e:
        return err(40010, str(e), http_status=400)
    except Exception as e:
        logger.exception("设置日志级别失败")
        return err(50021, f"设置日志级别失败: {e}")


@app.get("/api/logs/files")
async def logs_files():
    """返回结构化日志文件列表与元数据。"""
    return ok({"files": get_log_files()})


# ---- WebSocket 实时推送 -----------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """实时推送端点：行情/交易日志/告警/账户变化/Jev 决策。

    客户端消息：
      {"action": "subscribe", "symbol": "600519.SH"}
      {"action": "unsubscribe", "symbol": "600519.SH"}
      {"action": "ping"}  -> 服务端回 {"type": "pong"}

    认证（security.enabled=true 时）：
      连接时需通过 query param 携带 token，否则拒绝连接：
      ws://host:8766/ws?token=<token>  （至少需要 read 权限）
    """
    # token 认证（启用时）：query param ?token=<token>，失败直接拒绝连接
    if auth_manager.is_enabled():
        ws_token = websocket.query_params.get("token", "")
        ws_token_info = auth_manager.verify_token(ws_token)
        if ws_token_info is None or not auth_manager.has_permission(ws_token_info, "read"):
            await websocket.close(code=4401)  # 4401 = 未授权
            return

    client_id = f"client-{id(websocket) & 0xFFFFFF:x}"
    n = await ws_manager.connect(websocket)
    logger.info("WS 客户端 %s 已连接（当前连接数 %d）", client_id, n)
    try:
        await websocket.send_json({
            "type": "connected",
            "data": {
                "client_id": client_id,
                "symbols": list(SYMBOL_SET),
                "subscribed": sorted(ws_manager.active.get(websocket, set())),
            },
        })
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                continue
            action = msg.get("action")
            if action == "subscribe":
                ws_manager.subscribe(websocket, str(msg.get("symbol", "")))
                await websocket.send_json({"type": "subscribed",
                                           "data": {"symbol": msg.get("symbol")}})
            elif action == "unsubscribe":
                ws_manager.unsubscribe(websocket, str(msg.get("symbol", "")))
                await websocket.send_json({"type": "unsubscribed",
                                           "data": {"symbol": msg.get("symbol")}})
            elif action == "ping":
                await websocket.send_json({"type": "pong", "data": {"ts": time.time()}})
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WS 连接 %s 异常断开", client_id)
    finally:
        ws_manager.disconnect(websocket)
        logger.info("WS 客户端 %s 断开（剩余 %d）", client_id, len(ws_manager.active))


async def ws_push_loop() -> None:
    """后台推送任务：每 1 秒检查一次，向 WebSocket 客户端推送实时数据。

    - quote: 每轮推送所有已订阅标的的行情
    - trade: trader.trade_log 新增记录时推送新条目
    - alert: alert_manager 新增告警时广播
    - account: 当前账户总资产变动时广播快照
    - jev: jev_decisions 表新增记录时广播最新决策
    - 每 30 秒广播一次应用层心跳 ping
    """
    # 基线：避免启动时把历史日志/告警全量推给新连接（新连接建立后客户端可自行 REST 拉历史）
    _last_log_len = len(trader.trade_log)
    _last_alert_len = len(alert_manager._history)
    try:
        _last_jev_count = db.count_table("jev_decisions")
    except Exception:
        _last_jev_count = 0
    _last_account_asset: Optional[float] = None
    tick = 0
    while True:
        try:
            tick += 1
            # 1) 行情 tick：每轮向订阅者推送各标的 quote
            for sym, sim in manager.sims.items():
                await ws_manager.send_to_subscribers(
                    sym, {"type": "quote", "data": sim.get_quote()}
                )

            # 2) 新交易日志（窗口滑动时直接重置基线，不重推）
            cur_log_len = len(trader.trade_log)
            if cur_log_len > _last_log_len:
                for entry in trader.trade_log[_last_log_len:]:
                    await ws_manager.send_to_subscribers(
                        entry.get("symbol", ""),
                        {"type": "trade", "data": entry},
                    )
            _last_log_len = cur_log_len

            # 3) 新告警（广播，不区分标的）
            cur_alert_len = len(alert_manager._history)
            if cur_alert_len > _last_alert_len:
                for alert_obj in alert_manager._history[_last_alert_len:]:
                    await ws_manager.broadcast(
                        {"type": "alert", "data": alert_obj.to_dict()}
                    )
            _last_alert_len = cur_alert_len

            # 4) 账户变化（总资产变动超过 0.01 元才推送）
            try:
                acc = accounts.get(current_account_id,
                                   next(iter(accounts.values())))
                snap = acc.snapshot()
                asset = float(snap["total_asset"])
                if (_last_account_asset is None
                        or abs(asset - _last_account_asset) > 0.01):
                    await ws_manager.broadcast({"type": "account", "data": snap})
                    _last_account_asset = asset
            except Exception:
                logger.exception("账户快照推送失败（已忽略）")

            # 5) 新 Jev 决策（轮询 DB 行数变化）
            try:
                jev_cnt = db.count_table("jev_decisions")
                if jev_cnt > _last_jev_count:
                    n_new = min(jev_cnt - _last_jev_count, 50)
                    for decision in db.get_jev_decisions(limit=n_new):
                        await ws_manager.broadcast({"type": "jev", "data": decision})
                _last_jev_count = jev_cnt
            except Exception:
                logger.exception("Jev 决策推送失败（已忽略）")

            # 6) 应用层心跳：每 30 秒广播一次，便于客户端检测断线
            if tick % 30 == 0:
                await ws_manager.broadcast({"type": "ping", "data": {"ts": time.time()}})
        except Exception:
            logger.exception("ws_push_loop 迭代异常（已忽略）")
        await asyncio.sleep(1)


# ---- 数据质量监控 ---------------------------------------------------------

@app.get("/api/data_quality")
async def get_data_quality(
    include_anomalies: bool = Query(True, description="是否附带异常历史"),
    anomaly_limit: int = Query(50, ge=1, le=200),
):
    """返回最近一次数据质量报告 + 异常历史。"""
    report = data_quality_monitor.get_latest_report()
    anomalies = (
        data_quality_monitor.get_anomaly_history(limit=anomaly_limit)
        if include_anomalies else []
    )
    return ok({
        "report": report,
        "anomaly_history": anomalies,
    })


@app.get("/api/data_quality/anomalies")
async def get_data_quality_anomalies(
    limit: int = Query(50, ge=1, le=200),
):
    """仅返回异常历史列表。"""
    return ok({"anomalies": data_quality_monitor.get_anomaly_history(limit=limit)})


# ---------------------------------------------------------------------------
# 回测走查（Walkthrough）
# ---------------------------------------------------------------------------

@app.post("/api/walkthrough/run")
async def walkthrough_run(req: WalkthroughReq):
    """运行走查回测，逐日记录信号/Jev/操作/持仓/盈亏，返回 walkthrough_id。"""
    if req.strategy not in STRATEGY_NAMES:
        return err(40002, f"未知策略: {req.strategy}，可选: {STRATEGY_NAMES}")

    sym = normalize_symbol(req.symbol)
    sim = manager.get(sym)
    if sim is None or sim.klines is None or sim.klines.empty:
        return err(40404, f"无行情数据: {req.symbol}")

    df = sim.klines.copy()
    df = df.loc[(df.index >= pd.Timestamp(req.start_date)) &
                (df.index <= pd.Timestamp(req.end_date))]
    if len(df) < 30:
        return err(40010, "区间内K线不足30根，无法走查")

    try:
        wt = BacktestWalkthrough(
            symbol=sym,
            strategy_name=req.strategy,
            start_date=req.start_date,
            end_date=req.end_date,
            initial_capital=req.initial_capital,
            use_jev=req.use_jev,
        )
        walkthrough_id = wt.run(df)
        _walkthrough_put(wt)
    except Exception as e:
        logger.exception("走查回测失败")
        return err(50000, f"走查回测失败: {e}")

    return ok({
        "walkthrough_id": walkthrough_id,
        "summary": wt.to_dict(),
    })


@app.get("/api/walkthrough/{walkthrough_id}/day")
async def walkthrough_day(
    walkthrough_id: str,
    date: str = Query(..., description="YYYY-MM-DD"),
):
    """查询走查中某日的完整快照。"""
    wt = _walkthrough_get(walkthrough_id)
    if wt is None:
        return err(40404, f"走查不存在或已过期: {walkthrough_id}")
    day = wt.get_day(date)
    if day is None:
        return err(40404, f"该日无快照: {date}")
    return ok(day)


@app.get("/api/walkthrough/{walkthrough_id}/range")
async def walkthrough_range(
    walkthrough_id: str,
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
):
    """查询走查中某区间的所有日快照。"""
    wt = _walkthrough_get(walkthrough_id)
    if wt is None:
        return err(40404, f"走查不存在或已过期: {walkthrough_id}")
    snapshots = wt.get_range(start, end)
    return ok({
        "start": start, "end": end,
        "count": len(snapshots),
        "snapshots": snapshots,
    })


@app.get("/api/walkthrough/{walkthrough_id}/trades")
async def walkthrough_trades(walkthrough_id: str):
    """查询走查中所有交易事件及信号vs操作对比。"""
    wt = _walkthrough_get(walkthrough_id)
    if wt is None:
        return err(40404, f"走查不存在或已过期: {walkthrough_id}")
    return ok({
        "walkthrough_id": walkthrough_id,
        "trades": wt.get_trades(),
        "signal_vs_action": wt.get_signal_vs_action(),
    })


# ---------------------------------------------------------------------------
# Jev 训练数据导出
# ---------------------------------------------------------------------------

@app.post("/api/jev/export_training")
async def export_jev_training(req: TrainingExportReq):
    """导出带标签的 Jev 训练数据（JSONL + CSV，按时间切分 train/val/test）。"""
    from jev.training_data import TrainingDataExporter

    try:
        exporter = TrainingDataExporter(
            hold_threshold=req.hold_threshold,
            forward_days=req.forward_days,
        )
        result = exporter.export_all(
            output_dir=str(TRAINING_DATA_DIR),
            start_date=req.start_date,
            end_date=req.end_date,
            symbol=req.symbol,
            strategy=req.strategy,
        )
        download_urls = {
            k: f"/training_data/{Path(v).name}"
            for k, v in result["file_paths"].items()
        }
        return ok({
            "sample_count": result["sample_count"],
            "stats": result["stats"],
            "files": result["file_paths"],
            "download_urls": download_urls,
        })
    except Exception as e:
        logger.exception("导出 Jev 训练数据失败")
        return err(50010, f"导出失败: {e}", http_status=500)


@app.get("/api/jev/training_stats")
async def get_jev_training_stats():
    """返回最近一次训练数据导出的统计信息。"""
    stats_file = TRAINING_DATA_DIR / "last_export_stats.json"
    if not stats_file.exists():
        return ok({"available": False, "message": "尚未导出训练数据"})
    try:
        import json as _json
        data = _json.loads(stats_file.read_text(encoding="utf-8"))
        return ok({"available": True, **data})
    except Exception as e:
        logger.exception("读取训练数据统计失败")
        return err(50011, f"读取统计失败: {e}", http_status=500)


# ---- Jev 推理性能统计（REQ-P1-05） ---------------------------------------

# 共享的性能观测引擎单例：用于采集延迟/缓存/吞吐等指标。
# 懒加载创建，避免在 import 阶段产生副作用。
_jev_perf_engine = None


def _get_jev_perf_engine():
    """获取共享的 Jev 性能观测引擎（mock 模式，仅做指标采集，不影响真实推理路径）。"""
    global _jev_perf_engine
    if _jev_perf_engine is None:
        from jev.jev_engine import JevDecisionEngine
        _jev_perf_engine = JevDecisionEngine(
            mock_mode=True,
            audit_log_path=str(Path(__file__).parent / "logs" / "jev_perf_audit.jsonl"),
        )
    return _jev_perf_engine


@app.get("/api/jev/performance")
async def jev_performance():
    """返回 Jev 推理引擎的性能指标。

    包括延迟分位（P50/P95/P99 ms）、请求总数、缓存命中/未命中与命中率、
    吞吐量（req/s）、是否启用批量推理、当前并发上限。
    """
    try:
        engine = _get_jev_perf_engine()
        stats = engine.get_performance_stats()
        return ok({
            "p50_ms": stats["p50"],
            "p95_ms": stats["p95"],
            "p99_ms": stats["p99"],
            "total_requests": stats["total_requests"],
            "cache_hits": stats["cache_hits"],
            "cache_misses": stats["cache_misses"],
            "cache_hit_rate": stats["cache_hit_rate"],
            "throughput_rps": stats["throughput"],
            "batch_enabled": True,
            "concurrency": getattr(engine, "_max_concurrency", 3),
            "cache_size": stats["cache_size"],
            "cache_maxsize": stats["cache_maxsize"],
            "peak_concurrency": stats["peak_concurrency"],
        })
    except Exception as e:
        logger.exception("读取 Jev 性能统计失败")
        return err(50012, f"读取 Jev 性能统计失败: {e}", http_status=500)


@app.get("/training_data/{name}")
async def download_training_file(name: str):
    """下载导出的训练数据文件（防目录穿越）。"""
    safe_name = Path(name).name
    file_path = TRAINING_DATA_DIR / safe_name
    if not file_path.exists():
        return err(40404, "文件不存在", http_status=404)
    return FileResponse(
        str(file_path), filename=safe_name,
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# 数据库备份与恢复
# ---------------------------------------------------------------------------

@app.post("/api/backup/create")
async def backup_create(tag: str = ""):
    """手动创建一次数据库在线备份。"""
    try:
        path = backup_manager.backup(tag=tag)
        return ok({"path": path, "filename": Path(path).name})
    except Exception as e:
        logger.exception("手动备份失败")
        return err(50001, f"备份失败: {e}", http_status=500)


@app.get("/api/backup/list")
async def backup_list():
    """列出所有备份文件（时间倒序）。"""
    try:
        return ok({"backups": backup_manager.list_backups()})
    except Exception as e:
        logger.exception("列出备份失败")
        return err(50002, f"列出备份失败: {e}", http_status=500)


@app.post("/api/backup/restore")
async def backup_restore(req: BackupRestoreReq, http_req: Request):
    """从指定备份恢复数据库（恢复前自动兜底备份当前库）。"""
    token_info = getattr(http_req.state, "token_info", None) or {}
    operator = str(token_info.get("name", ""))
    if not req.backup_file:
        return err(40001, "backup_file 不能为空")
    cand = Path(req.backup_file)
    if not cand.is_absolute():
        cand = Path(backup_manager.backup_dir) / req.backup_file
    if not cand.exists():
        return err(40404, f"备份文件不存在: {req.backup_file}", http_status=404)
    try:
        if not backup_manager.restore(str(cand)):
            audit_logger.log(
                operator=operator,
                action_type=ActionType.BACKUP_RESTORE,
                target=req.backup_file,
                result="failure",
                ip=http_req.client.host if http_req.client else "",
            )
            return err(50003, "恢复失败，请查看服务日志", http_status=500)
        audit_logger.log(
            operator=operator,
            action_type=ActionType.BACKUP_RESTORE,
            target=req.backup_file,
            result="success",
            ip=http_req.client.host if http_req.client else "",
        )
        return ok({"restored_from": req.backup_file})
    except Exception as e:
        logger.exception("数据库恢复失败")
        audit_logger.log(
            operator=operator,
            action_type=ActionType.BACKUP_RESTORE,
            target=req.backup_file,
            result=f"failure: {e}",
            ip=http_req.client.host if http_req.client else "",
        )
        return err(50004, f"恢复失败: {e}", http_status=500)


@app.post("/api/backup/cleanup")
async def backup_cleanup(keep_days: int = 30):
    """清理超过保留天数的旧备份。"""
    try:
        removed = backup_manager.cleanup(keep_days=keep_days)
        return ok({"removed": removed})
    except Exception as e:
        logger.exception("清理备份失败")
        return err(50005, f"清理备份失败: {e}", http_status=500)


# ---------------------------------------------------------------------------
# 安全审计日志查询与完整性校验
# ---------------------------------------------------------------------------

@app.get("/api/audit/logs")
async def api_audit_logs(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    action_type: Optional[str] = Query(None),
    operator: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    """分页查询安全审计日志。"""
    try:
        data = audit_logger.query_logs(
            start=start, end=end,
            action_type=action_type, operator=operator,
            page=page, page_size=page_size,
        )
        return ok(data)
    except Exception as e:
        logger.exception("查询审计日志失败")
        return err(50010, f"查询审计日志失败: {e}", http_status=500)


@app.get("/api/audit/verify")
async def api_audit_verify():
    """校验审计日志哈希链完整性。"""
    try:
        report = audit_logger.verify_integrity()
        return ok(report)
    except Exception as e:
        logger.exception("校验审计日志完整性失败")
        return err(50011, f"校验失败: {e}", http_status=500)


@app.get("/api/rate_limit/status")
async def api_rate_limit_status():
    """返回 API 限流配置与各维度当前剩余配额。"""
    try:
        return ok(rate_limiter.get_status())
    except Exception as e:
        logger.exception("查询限流状态失败")
        return err(50012, f"查询限流状态失败: {e}", http_status=500)


# ---------------------------------------------------------------------------
# 策略排行榜与信号聚合
# ---------------------------------------------------------------------------

@app.post("/api/strategy/leaderboard")
async def strategy_leaderboard(req: LeaderboardReq):
    """批量回测全部策略并按综合得分排名，返回策略排行榜。"""
    sym = normalize_symbol(req.symbol)
    if sym not in SYMBOL_SET:
        return err(40001, f"标的不在股票池内: {req.symbol}")
    try:
        lb = _get_leaderboard()
        board = await asyncio.to_thread(
            lb.get_leaderboard, sym, req.start_date, req.end_date
        )
        rows = [{
            "rank": r["rank"],
            "strategy": r["strategy"],
            "label": r["label"],
            "score": r["score"],
            "sharpe": r["sharpe"],
            "total_return": r["total_return"],
            "max_drawdown": r["max_drawdown"],
            "win_rate": r["win_rate"],
            "metrics": r["metrics"],
        } for r in board]
        return ok({"symbol": sym, "start_date": req.start_date,
                   "end_date": req.end_date, "leaderboard": rows})
    except Exception as e:
        logger.exception("策略排行榜失败")
        return err(50010, f"策略排行榜失败: {e}", http_status=500)


@app.post("/api/strategy/aggregate_signal")
async def strategy_aggregate_signal(req: AggregateSignalReq):
    """聚合多策略最新信号（按排名加权 + 近期表现动态调整），返回综合交易建议。"""
    sym = normalize_symbol(req.symbol)
    if sym not in SYMBOL_SET:
        return err(40001, f"标的不在股票池内: {req.symbol}")
    try:
        lb = _get_leaderboard()
        # 先跑一次排行榜建立排名权重，再聚合最新信号
        await asyncio.to_thread(
            lb.get_leaderboard, sym, req.start_date, "2099-12-31"
        )
        agg = await asyncio.to_thread(lb.aggregate_signals, sym, None, req.threshold)
        return ok({"symbol": sym, **agg})
    except Exception as e:
        logger.exception("聚合信号失败")
        return err(50011, f"聚合信号失败: {e}", http_status=500)


# ---------------------------------------------------------------------------
# 系统健康监控
# ---------------------------------------------------------------------------

@app.get("/api/system/health")
async def system_health(
    refresh: bool = Query(False, description="是否强制实时采集（默认读缓存）"),
):
    """全链路系统健康报告：服务/API/Jev/数据库/数据质量/交易引擎/告警 + 0-100评分。"""
    if refresh or health_monitor.get_latest_report() is None:
        report = await asyncio.to_thread(health_monitor.collect_all)
    else:
        report = health_monitor.get_latest_report()
    return ok(report)


@app.get("/api/system/health/metrics")
async def system_health_metrics():
    """仅返回 API 性能指标（各端点延迟/错误率）。"""
    return ok({
        "summary": api_metrics.get_summary(),
        "endpoints": api_metrics.get_stats(),
    })


# ---------------------------------------------------------------------------
# 扩展模块路由注册（因子分析 / 事件驱动 / 报告生成）
# ---------------------------------------------------------------------------

def _register_extension_routes() -> None:
    """注册扩展模块的 API 路由。

    - 因子分析引擎: POST /api/factors/{calculate,ic_analysis,layered_backtest}, GET /api/factors/list
    - 事件驱动策略: POST /api/event/{detect,study,backtest}
    - 报告生成: POST /api/backtest/report, GET /api/reports/list, GET /api/reports/{id}/download
    - 多因子选股: POST /api/multifactor/{score,backtest}, GET /api/multifactor/config
    - 机器学习预测: POST /api/ml/{train,predict,backtest}, GET /api/ml/models
    - 多因子模型优化: POST /api/factors/optimize, /api/factors/orthogonalize, /api/factors/winsorize
    - 智能组合再平衡: POST /api/rebalance/run, /api/rebalance/drift, /api/rebalance/cost, /api/rebalance/tax_estimate, /api/rebalance/tax_loss_harvest
    - 自适应风险管理: POST /api/risk/adaptive/var, /api/risk/adaptive/budget, /api/risk/adaptive/tail_risk, /api/risk/adaptive/stress_test, /api/risk/adaptive/report
    """
    # 1. 因子分析路由
    try:
        from factors.server_routes import register_factor_routes
        register_factor_routes(app, manager, SYMBOL_SET, ok, err)
        logger.info("因子分析路由已注册: /api/factors/*")
    except Exception as e:
        logger.warning("因子分析路由注册失败: %s", e)

    # 2. 事件驱动策略路由
    try:
        from strategies.server_routes import register_event_routes
        register_event_routes(app)
        logger.info("事件驱动策略路由已注册: /api/event/*")
    except Exception as e:
        logger.warning("事件驱动策略路由注册失败: %s", e)

    # 3. 报告生成路由（POST /api/backtest/report 与现有 GET 端点共存）
    try:
        from backtest.report_routes import ReportRouteDeps, register_report_routes
        deps = ReportRouteDeps(
            project_root=PROJECT_ROOT,
            reports_dir=PROJECT_ROOT / "output" / "reports",
            symbol_set=SYMBOL_SET,
            strategy_names=STRATEGY_NAMES,
            normalize_symbol=normalize_symbol,
            run_backtest=_run_backtest_to_result,
            ok=ok,
            err=err,
        )
        register_report_routes(app, deps)
        logger.info("报告生成路由已注册: /api/backtest/report, /api/reports/*")
    except Exception as e:
        logger.warning("报告生成路由注册失败: %s", e)

    # 4. 多因子选股策略路由
    try:
        from multifactor_routes import register_multifactor_routes
        register_multifactor_routes(app, manager, SYMBOL_SET, ok, err)
        logger.info("多因子选股路由已注册: /api/multifactor/*")
    except Exception as e:
        logger.warning("多因子选股路由注册失败: %s", e)

    # 5. 机器学习预测策略路由
    try:
        from ml_routes import register_ml_routes
        register_ml_routes(app, manager, SYMBOL_SET, ok, err)
        logger.info("机器学习预测路由已注册: /api/ml/*")
    except Exception as e:
        logger.warning("机器学习预测路由注册失败: %s", e)

    # 6. 专业技术指标路由
    try:
        from _routes_indicators import register_indicator_routes
        register_indicator_routes(app, manager, SYMBOL_SET, ok, err)
        logger.info("技术指标路由已注册: /api/indicators/*")
    except Exception as e:
        logger.warning("技术指标路由注册失败: %s", e)

    # 7. 滚动窗口优化（Walk-Forward）路由
    try:
        from _routes_walkforward import register_walkforward_routes
        register_walkforward_routes(
            app, manager, normalize_symbol, SYMBOL_SET, STRATEGY_NAMES,
            ok, err, _get_strategy_class,
        )
        logger.info("滚动优化路由已注册: /api/optimize/walk_forward, /api/optimize/random_search")
    except Exception as e:
        logger.warning("滚动优化路由注册失败: %s", e)

    # 8. 风险模型路由（VaR / CVaR / 压力测试）
    try:
        from _routes_risk import register_risk_routes
        register_risk_routes(app, manager, SYMBOL_SET, ok, err)
        logger.info("风险模型路由已注册: /api/risk/*")
    except Exception as e:
        logger.warning("风险模型路由注册失败: %s", e)

    # 9. 配对交易路由（标的对筛选 / 回测）
    try:
        from _routes_pairs import register_pairs_routes
        register_pairs_routes(app, manager, SYMBOL_SET, ok, err)
        logger.info("配对交易路由已注册: /api/pairs/*")
    except Exception as e:
        logger.warning("配对交易路由注册失败: %s", e)

    # 10. 订单管理系统（OMS）路由
    try:
        from _routes_oms import register_oms_routes
        register_oms_routes(app, ok, err)
        logger.info("OMS 路由已注册: /api/oms/*")
    except Exception as e:
        logger.warning("OMS 路由注册失败: %s", e)

    # 11. Brinson 绩效归因路由
    try:
        from _routes_brinson import register_brinson_routes
        register_brinson_routes(app, ok, err)
        logger.info("Brinson 归因路由已注册: /api/attribution/brinson/*")
    except Exception as e:
        logger.warning("Brinson 归因路由注册失败: %s", e)

    # 12. 因子风险暴露监控路由
    try:
        from _routes_factor_exposure import register_factor_exposure_routes
        register_factor_exposure_routes(app, manager, SYMBOL_SET, ok, err)
        logger.info("因子风险暴露路由已注册: /api/risk/factor_exposure/*")
    except Exception as e:
        logger.warning("因子风险暴露路由注册失败: %s", e)

    # 13. 算法交易（TWAP/VWAP）路由
    try:
        from _routes_algo import register_algo_routes
        register_algo_routes(app, ok, err)
        logger.info("算法交易路由已注册: /api/algo/*")
    except Exception as e:
        logger.warning("算法交易路由注册失败: %s", e)

    # 14. 数据校验与对账路由
    try:
        from _routes_data import register_data_routes
        register_data_routes(app, manager, SYMBOL_SET, ok, err)
        logger.info("数据校验对账路由已注册: /api/data/*")
    except Exception as e:
        logger.warning("数据校验对账路由注册失败: %s", e)

    # 15. Jev 可解释性路由
    try:
        from _routes_explain import register_explain_routes
        register_explain_routes(app, ok, err)
        logger.info("Jev 可解释性路由已注册: /api/jev/explain/*")
    except Exception as e:
        logger.warning("Jev 可解释性路由注册失败: %s", e)

    # 16. 多账户资金路由
    try:
        from _routes_multi_account import register_multi_account_routes
        register_multi_account_routes(app, ok, err)
        logger.info("多账户资金路由已注册: /api/multi_account/*")
    except Exception as e:
        logger.warning("多账户资金路由注册失败: %s", e)

    # 17. 多 Jev 模型 Ensemble 路由
    try:
        from _routes_jev_ensemble import register_jev_ensemble_routes
        register_jev_ensemble_routes(app, ok, err)
        logger.info("Jev Ensemble 路由已注册: /api/jev_ensemble/*")
    except Exception as e:
        logger.warning("Jev Ensemble 路由注册失败: %s", e)

    # 18. 多因子模型优化路由
    try:
        from _routes_multi_factor_opt import register_multi_factor_opt_routes
        register_multi_factor_opt_routes(app, manager, SYMBOL_SET, ok, err)
        logger.info("多因子优化路由已注册: /api/factors/optimize/*")
    except Exception as e:
        logger.warning("多因子优化路由注册失败: %s", e)

    # 19. 智能组合再平衡路由
    try:
        from _routes_smart_rebalance import register_smart_rebalance_routes
        register_smart_rebalance_routes(app, ok, err)
        logger.info("智能再平衡路由已注册: /api/rebalance/*")
    except Exception as e:
        logger.warning("智能再平衡路由注册失败: %s", e)

    # 20. 自适应风险管理路由
    try:
        from _routes_adaptive_risk import register_adaptive_risk_routes
        register_adaptive_risk_routes(app, manager, SYMBOL_SET, ok, err)
        logger.info("自适应风险管理路由已注册: /api/risk/adaptive/*")
    except Exception as e:
        logger.warning("自适应风险管理路由注册失败: %s", e)


_register_extension_routes()


# ---------------------------------------------------------------------------
# PWA 静态资源（manifest / service worker / 图标）
# ---------------------------------------------------------------------------
_WEB_DIR = Path(__file__).resolve().parent


@app.get("/manifest.json")
async def pwa_manifest():
    """PWA manifest（application/manifest+json）。"""
    p = _WEB_DIR / "manifest.json"
    if not p.exists():
        return err(404, "manifest.json 不存在", http_status=404)
    return FileResponse(str(p), media_type="application/manifest+json")


@app.get("/sw.js")
async def pwa_service_worker():
    """Service Worker 脚本（Service-Worker-Allowed 允许根作用域）。"""
    p = _WEB_DIR / "sw.js"
    if not p.exists():
        return err(404, "sw.js 不存在", http_status=404)
    headers = {"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"}
    return FileResponse(str(p), media_type="application/javascript", headers=headers)


@app.get("/icons/{name}")
async def pwa_icon(name: str):
    """PWA 图标（仅允许 icons 目录下文件，防路径穿越）。"""
    if "/" in name or "\\" in name or ".." in name:
        return err(400, "非法图标名")
    p = _WEB_DIR / "icons" / name
    if not p.exists():
        return err(404, f"图标不存在: {name}", http_status=404)
    media = "image/png" if name.endswith(".png") else "image/svg+xml"
    return FileResponse(str(p), media_type=media)


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 50)
    logger.info("量化交易系统实时数据服务 v3（股票池驱动）")
    logger.info("访问 http://localhost:8766")
    logger.info("API 文档 http://localhost:8766/docs")
    logger.info("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8766, log_level="info")
