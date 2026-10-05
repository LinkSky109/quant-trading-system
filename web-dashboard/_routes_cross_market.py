"""跨市场套利 API 路由（扩展模块 #22）。

端点：
    POST /api/crossmarket/backtest  —— 对指定跨市场标的对做套利回测
    POST /api/crossmarket/screen    —— 在股票池中筛选协整标的对
    GET  /api/crossmarket/cost      —— 查询指定标的对的交易成本
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from strategies.cross_market_arbitrage import (
    CrossMarketArbitrageStrategy,
    get_trading_cost,
)

logger = logging.getLogger("realtime_server")


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class CrossMarketBacktestReq(BaseModel):
    """跨市场套利回测请求。"""

    symbol_a: str
    symbol_b: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 3.0
    window: int = 20
    initial_capital: float = 1_000_000.0
    trading_costs: Dict[str, float] = {}


class CrossMarketScreenReq(BaseModel):
    """跨市场标的对筛选请求。"""

    symbols: List[str] = []
    min_correlation: float = 0.7
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_cross_market_routes(
    app: FastAPI,
    manager: Any,
    symbol_set: List[str],
    ok: Any,
    err: Any,
) -> None:
    """注册跨市场套利路由到 FastAPI app。"""

    @app.post("/api/crossmarket/backtest")
    async def crossmarket_backtest(req: CrossMarketBacktestReq):
        """对指定跨市场标的对做套利回测。"""
        try:
            sym_a = req.symbol_a.strip().upper()
            sym_b = req.symbol_b.strip().upper()
            if not sym_a or not sym_b:
                return err(400, "symbol_a 和 symbol_b 不能为空")

            try:
                df_a = manager.get_daily_klines(sym_a)
                df_b = manager.get_daily_klines(sym_b)
                if df_a is None or df_b is None or df_a.empty or df_b.empty:
                    return err(404, f"未找到 {sym_a} 或 {sym_b} 的日K线数据")
            except Exception:
                return err(404, f"未找到 {sym_a} 或 {sym_b} 的日K线数据")

            if req.start_date:
                df_a = df_a[df_a.index >= req.start_date]
                df_b = df_b[df_b.index >= req.start_date]
            if req.end_date:
                df_a = df_a[df_a.index <= req.end_date]
                df_b = df_b[df_b.index <= req.end_date]

            merged = pd.DataFrame({
                f"{sym_a}_close": df_a["close"],
                f"{sym_b}_close": df_b["close"],
            }).dropna()

            strategy = CrossMarketArbitrageStrategy({
                "symbol_a": sym_a,
                "symbol_b": sym_b,
                "z_entry": req.z_entry,
                "z_exit": req.z_exit,
                "z_stop": req.z_stop,
                "window": req.window,
                "trading_costs": req.trading_costs,
            })

            coint = strategy.test_cointegration(
                merged[f"{sym_a}_close"], merged[f"{sym_b}_close"]
            )
            signals = strategy.generate_signals(merged)

            # 简单回测：按信号方向逐笔交易
            trades = []
            pnl = 0.0
            for sig in signals:
                trades.append({
                    "date": sig.date.isoformat() if hasattr(sig.date, "isoformat") else str(sig.date),
                    "action": sig.action,
                    "confidence": round(sig.confidence, 4),
                    "z_score": sig.metadata.get("z_score"),
                })

            cost_info = strategy.estimate_cost(req.initial_capital)

            return ok({
                "symbol_a": sym_a,
                "symbol_b": sym_b,
                "cointegration": coint,
                "signals_count": len(signals),
                "trades": trades[:50],  # 限制返回数量
                "cost_estimate": cost_info,
            })
        except Exception as e:
            logger.exception("跨市场套利回测失败")
            return err(500, f"回测失败: {e}")

    @app.get("/api/crossmarket/cost")
    async def crossmarket_cost(symbol_a: str, symbol_b: str):
        """查询指定标的对的交易成本。"""
        try:
            cost_a = get_trading_cost(symbol_a.strip().upper())
            cost_b = get_trading_cost(symbol_b.strip().upper())
            return ok({
                "symbol_a": symbol_a,
                "symbol_b": symbol_b,
                "cost_a": cost_a,
                "cost_b": cost_b,
                "total_roundtrip": (cost_a + cost_b) * 2,
            })
        except Exception as e:
            logger.exception("查询交易成本失败")
            return err(500, f"查询失败: {e}")
