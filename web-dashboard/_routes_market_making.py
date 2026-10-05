"""做市策略 API 路由（扩展模块 #23）。

端点：
    POST /api/marketmaking/quotes    —— 对指定标的生成双边报价（ATR + 库存偏斜）
    POST /api/marketmaking/signals   —— 基于日K线的做市降级模拟（信号 + 库存演进）
    GET  /api/marketmaking/params    —— 查询做市策略默认参数
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from strategies.market_making import MarketMakingStrategy

logger = logging.getLogger("realtime_server")


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class MarketMakingQuotesReq(BaseModel):
    """双边报价请求。"""

    symbol: str
    inventory: int = 0
    atr_period: int = 14
    spread_k: float = 0.5
    inventory_skew: float = 0.5
    max_inventory: int = 100
    quote_size: int = 10


class MarketMakingSignalsReq(BaseModel):
    """做市降级模拟（日K线）请求。"""

    symbol: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    spread_k: float = 0.5
    inventory_skew: float = 0.5
    max_inventory: int = 100
    quote_size: int = 10
    atr_period: int = 14
    vol_period: int = 20


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_market_making_routes(
    app: FastAPI,
    manager: Any,
    symbol_set: List[str],
    ok: Any,
    err: Any,
) -> None:
    """注册做市策略路由到 FastAPI app。"""

    @app.post("/api/marketmaking/quotes")
    async def marketmaking_quotes(req: MarketMakingQuotesReq):
        """对指定标的生成双边报价。"""
        try:
            symbol = req.symbol.strip().upper()
            if not symbol:
                return err(400, "symbol 不能为空")

            try:
                df = manager.get_daily_klines(symbol)
            except Exception:
                df = None
            if df is None or getattr(df, "empty", True) or "close" not in df.columns:
                return err(404, f"未找到 {symbol} 的日K线数据")

            atr_series = MarketMakingStrategy.compute_atr(df, period=req.atr_period)
            atr_value = float(atr_series.iloc[-1])
            if atr_value != atr_value or atr_value <= 0:  # NaN 或非正
                return err(422, f"{symbol} 无法计算有效 ATR（数据不足或异常）")

            mid_price = float(df["close"].iloc[-1])
            strategy = MarketMakingStrategy({
                "spread_k": req.spread_k,
                "inventory_skew": req.inventory_skew,
                "max_inventory": req.max_inventory,
                "quote_size": req.quote_size,
                "atr_period": req.atr_period,
            })
            quote_result = strategy.quote(mid_price, atr_value, req.inventory)

            return ok({
                "symbol": symbol,
                "inventory": req.inventory,
                "atr": round(atr_value, 4),
                "quote": quote_result,
            })
        except Exception as e:
            logger.exception("生成双边报价失败")
            return err(500, f"报价失败: {e}")

    @app.post("/api/marketmaking/signals")
    async def marketmaking_signals(req: MarketMakingSignalsReq):
        """基于日K线做市降级模拟（信号 + 库存演进 + 价差捕捉）。"""
        try:
            symbol = req.symbol.strip().upper()
            if not symbol:
                return err(400, "symbol 不能为空")

            try:
                df = manager.get_daily_klines(symbol)
            except Exception:
                df = None
            if df is None or getattr(df, "empty", True):
                return err(404, f"未找到 {symbol} 的日K线数据")

            if req.start_date:
                df = df[df.index >= req.start_date]
            if req.end_date:
                df = df[df.index <= req.end_date]
            if df.empty:
                return err(422, f"{symbol} 在指定日期区间内无数据")

            strategy = MarketMakingStrategy({
                "spread_k": req.spread_k,
                "inventory_skew": req.inventory_skew,
                "max_inventory": req.max_inventory,
                "quote_size": req.quote_size,
                "atr_period": req.atr_period,
                "vol_period": req.vol_period,
            })
            signals = strategy.generate_signals(df, symbol=symbol)

            trades = []
            for sig in signals:
                trades.append({
                    "date": sig.date.isoformat() if hasattr(sig.date, "isoformat") else str(sig.date),
                    "action": sig.action,
                    "confidence": round(sig.confidence, 4),
                    "bid": sig.metadata.get("bid"),
                    "ask": sig.metadata.get("ask"),
                    "spread": sig.metadata.get("spread"),
                    "inventory": sig.metadata.get("inventory"),
                })

            final_inventory = trades[-1]["inventory"] if trades else 0
            return ok({
                "symbol": symbol,
                "signals_count": len(signals),
                "final_inventory": final_inventory,
                "trades": trades[:50],  # 限制返回数量
            })
        except Exception as e:
            logger.exception("做市降级模拟失败")
            return err(500, f"模拟失败: {e}")

    @app.get("/api/marketmaking/params")
    async def marketmaking_params():
        """查询做市策略默认参数。"""
        try:
            strategy = MarketMakingStrategy()
            return ok({
                "name": strategy.name,
                "params": {
                    "spread_k": strategy.spread_k,
                    "inventory_skew": strategy.inventory_skew,
                    "max_inventory": strategy.max_inventory,
                    "quote_size": strategy.quote_size,
                    "atr_period": strategy.atr_period,
                    "vol_period": strategy.vol_period,
                },
            })
        except Exception as e:
            logger.exception("查询做市参数失败")
            return err(500, f"查询失败: {e}")
