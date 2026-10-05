"""期权支持 API 路由（扩展模块 #24）。

端点：
    POST /api/options/chain      —— 获取指定标的的期权链（mock 降级，预留真实源）
    POST /api/options/greeks     —— 计算 Black-Scholes 价格与 Greeks
    POST /api/options/strategy   —— 生成简单期权组合模板（备兑/牛市价差/跨式）
"""
from __future__ import annotations

import logging
from typing import Any, List, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from data.options import (
    OptionDataProvider,
    OptionQuote,
    OptionStrategyBuilder,
    black_scholes_price,
    calculate_greeks,
)

logger = logging.getLogger("realtime_server")


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class OptionChainReq(BaseModel):
    """期权链请求。"""

    symbol: str
    spot: float
    expiry: str
    ttm_years: float
    num_strikes: int = 5


class OptionGreeksReq(BaseModel):
    """Greeks 计算请求。"""

    spot: float
    strike: float
    ttm_years: float
    vol: float
    risk_free_rate: float = 0.03
    option_type: Literal["call", "put"] = "call"


class OptionStrategyReq(BaseModel):
    """期权组合模板请求。"""

    strategy: Literal["covered_call", "bull_call_spread", "straddle"]
    symbol: str
    spot: float
    expiry: str
    ttm_years: float
    strike: Optional[float] = None          # covered_call / straddle 用
    strike_low: Optional[float] = None      # bull_call_spread 用
    strike_high: Optional[float] = None     # bull_call_spread 用
    quantity: int = 1


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _quotes_to_dicts(quotes: List[OptionQuote]) -> List[dict]:
    return [q.as_dict() for q in quotes]


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_options_routes(
    app: FastAPI,
    manager: Any = None,
    symbol_set: Optional[list] = None,
    ok: Any = None,
    err: Any = None,
) -> None:
    """注册期权支持路由到 FastAPI app。"""
    provider = OptionDataProvider()

    @app.post("/api/options/chain")
    async def options_chain(req: OptionChainReq):
        """获取指定标的的期权链（当前为 mock 生成，预留真实数据源接口）。"""
        try:
            if req.spot <= 0 or req.ttm_years <= 0:
                return err(422, "spot 与 ttm_years 必须为正数")
            chain = provider.fetch_chain(
                req.symbol.strip().upper(), req.spot, req.expiry, req.ttm_years
            )
            return ok({
                "symbol": req.symbol.strip().upper(),
                "spot": req.spot,
                "expiry": req.expiry,
                "ttm_years": req.ttm_years,
                "data_source": "mock",
                "calls": _quotes_to_dicts(chain.get("calls", [])),
                "puts": _quotes_to_dicts(chain.get("puts", [])),
            })
        except Exception as e:
            logger.exception("获取期权链失败")
            return err(500, f"获取期权链失败: {e}")

    @app.post("/api/options/greeks")
    async def options_greeks(req: OptionGreeksReq):
        """计算 Black-Scholes 价格与 Greeks。"""
        try:
            if req.spot <= 0 or req.strike <= 0 or req.ttm_years < 0 or req.vol < 0:
                return err(422, "价格参数不合法（spot/strike 须为正，ttm/vol 须非负）")
            price = black_scholes_price(
                req.spot, req.strike, req.ttm_years, req.vol,
                req.risk_free_rate, req.option_type,
            )
            greeks = calculate_greeks(
                req.spot, req.strike, req.ttm_years, req.vol,
                req.risk_free_rate, req.option_type,
            )
            return ok({
                "option_type": req.option_type,
                "price": round(price, 6),
                "greeks": {k: round(v, 6) for k, v in greeks.items()},
            })
        except Exception as e:
            logger.exception("计算 Greeks 失败")
            return err(500, f"计算失败: {e}")

    @app.post("/api/options/strategy")
    async def options_strategy(req: OptionStrategyReq):
        """生成简单期权组合模板。"""
        try:
            if req.spot <= 0 or req.ttm_years <= 0:
                return err(422, "spot 与 ttm_years 必须为正数")
            chain = provider.fetch_chain(
                req.symbol.strip().upper(), req.spot, req.expiry, req.ttm_years
            )
            builder = OptionStrategyBuilder(chain)

            if req.strategy == "covered_call":
                strike = req.strike
                if strike is None:
                    return err(422, "covered_call 需要指定 strike")
                result = builder.covered_call(req.spot, strike, req.quantity)
            elif req.strategy == "bull_call_spread":
                if req.strike_low is None or req.strike_high is None:
                    return err(422, "bull_call_spread 需要指定 strike_low 与 strike_high")
                result = builder.bull_call_spread(
                    req.strike_low, req.strike_high, req.quantity
                )
            else:  # straddle
                strike = req.strike if req.strike is not None else req.spot
                result = builder.straddle(strike, req.quantity)

            if result is None:
                return err(404, "期权链中找不到所需的行权价，请调整参数")

            return ok({
                "strategy": req.strategy,
                "symbol": req.symbol.strip().upper(),
                "data_source": "mock",
                "result": result.as_dict(),
            })
        except Exception as e:
            logger.exception("生成期权组合失败")
            return err(500, f"生成组合失败: {e}")
