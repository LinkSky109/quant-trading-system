"""加密货币数据 API 路由（扩展模块 #25）。

端点：
    POST /api/crypto/klines        —— 获取加密货币日K线（24x7、UTC 日切、8位精度）
    GET  /api/crypto/market_status —— 查询加密货币市场状态（24小时交易）
    GET  /api/crypto/risk_params   —— 查询加密货币差异化风控参数
    POST /api/crypto/adapt_params  —— 股票策略参数适配为加密货币参数
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from data.crypto import (
    CRYPTO_MARKET_CONFIG,
    CryptoDataProvider,
    is_crypto_symbol,
    round_crypto,
)

logger = logging.getLogger("realtime_server")


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class CryptoKlinesReq(BaseModel):
    """加密货币K线请求。"""

    symbol: str = "BTC-USD"
    days: int = 250


class CryptoAdaptParamsReq(BaseModel):
    """策略参数适配请求。"""

    params: dict = {}
    symbol: str = "BTC-USD"


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_crypto_routes(
    app: FastAPI,
    manager: Any,
    symbol_set: List[str],
    ok: Any,
    err: Any,
) -> None:
    """注册加密货币路由到 FastAPI app。"""
    provider = CryptoDataProvider()

    @app.post("/api/crypto/klines")
    async def crypto_klines(req: CryptoKlinesReq):
        """获取加密货币日K线（24x7、UTC 日切、8 位小数精度）。"""
        try:
            symbol = req.symbol.strip().upper()
            if not symbol:
                return err(400, "symbol 不能为空")
            if not is_crypto_symbol(symbol):
                return err(422, f"{symbol} 不是合法的加密货币代码（示例: BTC-USD, ETH-USD）")
            if req.days <= 0 or req.days > 1500:
                return err(422, "days 须在 1~1500 之间")

            df = provider.fetch_klines(symbol, days=req.days)
            if df is None or df.empty:
                return err(404, f"未获取到 {symbol} 的K线数据")

            records = []
            for idx, row in df.iterrows():
                records.append({
                    "date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                    "open": round_crypto(float(row["open"])),
                    "high": round_crypto(float(row["high"])),
                    "low": round_crypto(float(row["low"])),
                    "close": round_crypto(float(row["close"])),
                    "volume": round_crypto(float(row["volume"])),
                })

            return ok({
                "symbol": symbol,
                "days": req.days,
                "count": len(records),
                "trading_hours": CRYPTO_MARKET_CONFIG["trading_hours"],
                "timezone": CRYPTO_MARKET_CONFIG["timezone"],
                "data_source": "mock" if provider.is_mock else "real",
                "klines": records[:500],  # 限制返回数量
            })
        except Exception as e:
            logger.exception("获取加密货币K线失败")
            return err(500, f"获取失败: {e}")

    @app.get("/api/crypto/market_status")
    async def crypto_market_status():
        """查询加密货币市场状态（24 小时无休）。"""
        try:
            return ok(provider.get_market_status().as_dict())
        except Exception as e:
            logger.exception("查询市场状态失败")
            return err(500, f"查询失败: {e}")

    @app.get("/api/crypto/risk_params")
    async def crypto_risk_params():
        """查询加密货币差异化风控参数。"""
        try:
            return ok({
                "market_type": CRYPTO_MARKET_CONFIG["market_type"],
                "risk_params": provider.get_risk_params(),
                "symbol_params": CRYPTO_MARKET_CONFIG["symbol_params"],
                "default_symbols": CRYPTO_MARKET_CONFIG["default_symbols"],
            })
        except Exception as e:
            logger.exception("查询风控参数失败")
            return err(500, f"查询失败: {e}")

    @app.post("/api/crypto/adapt_params")
    async def crypto_adapt_params(req: CryptoAdaptParamsReq):
        """把股票市场策略参数适配为加密货币参数。"""
        try:
            if not req.params:
                return err(422, "params 不能为空")
            symbol = req.symbol.strip().upper()
            if not is_crypto_symbol(symbol):
                return err(422, f"{symbol} 不是合法的加密货币代码")
            adapted = provider.adapt_strategy_params(dict(req.params))
            return ok({
                "symbol": symbol,
                "original": req.params,
                "adapted": adapted,
            })
        except Exception as e:
            logger.exception("适配策略参数失败")
            return err(500, f"适配失败: {e}")
