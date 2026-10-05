"""配对交易 API 路由（扩展模块）。

按 server.py 现有扩展路由模式挂载（见 server.py 中 _register_extension_routes）。

端点：
    POST /api/pairs/screen     —— 在股票池中筛选高相关且协整的标的对
    POST /api/pairs/backtest   —— 对指定标的对做配对策略回测
    GET  /api/pairs/list       —— 预计算/缓存的可交易标的对列表
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from strategies.pairs_trading import run_pairs_backtest, screen_pairs

logger = logging.getLogger("realtime_server")


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class PairsScreenReq(BaseModel):
    """标的对筛选请求。"""

    symbols: List[str] = []
    min_correlation: float = 0.7
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class PairsBacktestReq(BaseModel):
    """配对策略回测请求。"""

    symbol_a: str
    symbol_b: str
    hedge_ratio: float = 1.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 3.0
    window: int = 20
    initial_capital: float = 1_000_000.0


# 模块级缓存：screen 结果可复用（避免每次 list 都重算协整）
_CACHED_PAIRS: List[Dict[str, Any]] = []


def _extract_close(
    manager: Any,
    symbols: List[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> Dict[str, pd.Series]:
    """从 manager 提取各标的收盘价序列（按日期索引）。"""
    result: Dict[str, pd.Series] = {}
    for sym in symbols:
        sim = manager.get(sym)
        df = getattr(sim, "klines", None)
        if df is None or df.empty:
            raise ValueError(f"标的 {sym} 暂无K线数据")
        df = df.copy()
        if start_date:
            df = df.loc[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df.loc[df.index <= pd.Timestamp(end_date)]
        if len(df) < 30:
            raise ValueError(f"标的 {sym} 有效K线不足(<30)")
        result[sym] = df["close"].astype(float)
    return result


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_pairs_routes(
    app: FastAPI,
    manager: Any,
    symbol_set: set,
    ok: Any,
    err: Any,
) -> None:
    """注册配对交易相关路由。"""
    from data.data_fetcher import normalize_symbol

    @app.post("/api/pairs/screen")
    async def pairs_screen(req: PairsScreenReq):
        """在给定标的集合中筛选高相关且协整的可交易标的对。"""
        symbols = [normalize_symbol(s) for s in (req.symbols or []) if s]
        if len(symbols) < 2:
            return err(40001, "至少选择 2 只标的才能配对")
        invalid = [s for s in symbols if s not in symbol_set]
        if invalid:
            return err(40002, f"标的不在股票池内: {invalid}")
        try:
            prices = _extract_close(manager, symbols, req.start_date, req.end_date)
            pairs = screen_pairs(prices, min_correlation=req.min_correlation)
            # 更新缓存（仅保留协整对作为"可交易对"）
            _CACHED_PAIRS.clear()
            _CACHED_PAIRS.extend(p for p in pairs if p["is_cointegrated"])
            return ok({"pairs": pairs, "total": len(pairs)})
        except ValueError as e:
            return err(40003, str(e))
        except Exception as e:
            logger.exception("配对筛选失败")
            return err(50000, f"配对筛选失败: {e}", http_status=500)

    @app.post("/api/pairs/backtest")
    async def pairs_backtest(req: PairsBacktestReq):
        """对指定标的对运行配对策略回测。"""
        sym_a = normalize_symbol(req.symbol_a)
        sym_b = normalize_symbol(req.symbol_b)
        for s in (sym_a, sym_b):
            if s not in symbol_set:
                return err(40002, f"标的不在股票池内: {s}")
        if sym_a == sym_b:
            return err(40003, "两个标的不能相同")
        try:
            prices = _extract_close(
                manager, [sym_a, sym_b], req.start_date, req.end_date
            )
            result = run_pairs_backtest(
                prices[sym_a], prices[sym_b],
                hedge_ratio=req.hedge_ratio,
                z_entry=req.z_entry, z_exit=req.z_exit, z_stop=req.z_stop,
                window=req.window, initial_capital=req.initial_capital,
            )
            result["symbol_a"] = sym_a
            result["symbol_b"] = sym_b
            result["hedge_ratio"] = req.hedge_ratio
            return ok(result)
        except ValueError as e:
            return err(40004, str(e))
        except Exception as e:
            logger.exception("配对回测失败")
            return err(50000, f"配对回测失败: {e}", http_status=500)

    @app.get("/api/pairs/list")
    async def pairs_list():
        """返回预计算的可交易标的对列表（最近一次 /screen 的协整对缓存）。"""
        return ok({"pairs": list(_CACHED_PAIRS), "count": len(_CACHED_PAIRS)})
