"""智能组合再平衡 API 路由（扩展模块）。

按 server.py 现有扩展路由模式挂载：

    try:
        from _routes_smart_rebalance import register_smart_rebalance_routes
        register_smart_rebalance_routes(app, ok, err)
    except Exception as e:
        logger.warning("智能再平衡路由注册失败: %s", e)

端点：
    POST /api/rebalance/run          —— 执行智能再平衡
    POST /api/rebalance/drift        —— 计算当前漂移
    POST /api/rebalance/cost         —— 估算交易成本
    POST /api/rebalance/tax_estimate —— 税务估算
    GET  /api/rebalance/status       —— 再平衡参数说明
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from optimization.smart_rebalance import SmartRebalancer, TaxLot

logger = logging.getLogger("realtime_server")


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class RebalanceRunReq(BaseModel):
    """执行再平衡请求。"""

    current_weights: Dict[str, float] = {}
    target_weights: Dict[str, float] = {}
    prices: Dict[str, float] = {}
    portfolio_value: float = 1_000_000.0
    positions: Optional[Dict[str, Dict[str, float]]] = None
    volumes: Optional[Dict[str, float]] = None
    tax_lots: Optional[Dict[str, List[Dict[str, Any]]]] = None
    min_trade_value: float = 0.0
    allow_tax_loss_harvest: bool = True
    avoid_short_term: bool = True
    drift_threshold: float = 0.05


class DriftReq(BaseModel):
    """漂移计算请求。"""

    current_weights: Dict[str, float] = {}
    target_weights: Dict[str, float] = {}


class CostReq(BaseModel):
    """交易成本估算请求。"""

    symbol: str = ""
    action: str = "buy"
    shares: float = 100.0
    price: float = 50.0
    daily_volume: float = 0.0
    commission_rate: float = 0.00025
    stamp_tax_rate: float = 0.0005
    slippage_rate: float = 0.001
    impact_cost_coeff: float = 0.1


class TaxEstimateReq(BaseModel):
    """税务估算请求。"""

    action: str = "sell"
    shares: float = 100.0
    price: float = 60.0
    avg_cost: float = 50.0
    holding_days: int = 30
    short_term_days: int = 365
    short_term_tax_rate: float = 0.20
    long_term_tax_rate: float = 0.10


class TaxLossHarvestReq(BaseModel):
    """亏损收割候选识别请求。"""

    positions: Dict[str, Dict[str, float]] = {}
    prices: Dict[str, float] = {}


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_smart_rebalance_routes(
    app: FastAPI,
    ok: Any,
    err: Any,
) -> None:
    """注册智能再平衡相关路由。

    Args:
        app: FastAPI 实例。
        ok / err: server.py 的统一响应封装。
    """

    @app.post("/api/rebalance/run")
    async def rebalance_run(req: RebalanceRunReq):
        """执行智能再平衡，返回交易清单与成本估算。"""
        if not req.current_weights or not req.target_weights:
            return err(40001, "current_weights 与 target_weights 不能为空")
        if not req.prices:
            return err(40002, "prices 不能为空")

        try:
            rb = SmartRebalancer(drift_threshold=req.drift_threshold)
            # 构造 TaxLot
            tax_lots = None
            if req.tax_lots:
                tax_lots = {}
                for sym, lots in req.tax_lots.items():
                    tax_lots[sym] = [
                        TaxLot(
                            symbol=lot["symbol"],
                            shares=float(lot["shares"]),
                            avg_cost=float(lot["avg_cost"]),
                            entry_date=pd.Timestamp(lot["entry_date"]),
                        )
                        for lot in lots
                    ]
            result = rb.rebalance(
                current_weights=req.current_weights,
                target_weights=req.target_weights,
                prices=req.prices,
                portfolio_value=req.portfolio_value,
                positions=req.positions,
                volumes=req.volumes,
                tax_lots=tax_lots,
                min_trade_value=req.min_trade_value,
                allow_tax_loss_harvest=req.allow_tax_loss_harvest,
                avoid_short_term=req.avoid_short_term,
            )
            return ok(result.as_dict())
        except Exception as e:
            logger.exception("再平衡失败")
            return err(50001, f"再平衡失败: {e}", http_status=500)

    @app.post("/api/rebalance/drift")
    async def rebalance_drift(req: DriftReq):
        """计算当前权重与目标权重的漂移。"""
        if not req.current_weights or not req.target_weights:
            return err(40001, "current_weights 与 target_weights 不能为空")
        try:
            rb = SmartRebalancer()
            drift = rb.calc_drift(req.current_weights, req.target_weights)
            triggered = rb.should_rebalance(req.current_weights, req.target_weights)
            return ok({
                "drift": round(drift, 6),
                "triggered": triggered,
                "threshold": rb.drift_threshold,
            })
        except Exception as e:
            logger.exception("漂移计算失败")
            return err(50001, f"漂移计算失败: {e}", http_status=500)

    @app.post("/api/rebalance/cost")
    async def rebalance_cost(req: CostReq):
        """估算单笔交易成本。"""
        try:
            rb = SmartRebalancer(
                commission_rate=req.commission_rate,
                stamp_tax_rate=req.stamp_tax_rate,
                slippage_rate=req.slippage_rate,
                impact_cost_coeff=req.impact_cost_coeff,
            )
            cost = rb.transaction_cost(
                req.symbol, req.action, req.shares, req.price, req.daily_volume,
            )
            amount = req.shares * req.price
            return ok({
                "symbol": req.symbol,
                "action": req.action,
                "shares": req.shares,
                "price": req.price,
                "amount": round(amount, 2),
                "estimated_cost": round(cost, 4),
                "cost_pct": round(cost / amount, 6) if amount > 0 else 0.0,
            })
        except Exception as e:
            logger.exception("成本估算失败")
            return err(50001, f"成本估算失败: {e}", http_status=500)

    @app.post("/api/rebalance/tax_estimate")
    async def rebalance_tax_estimate(req: TaxEstimateReq):
        """估算资本利得税。"""
        try:
            rb = SmartRebalancer(
                short_term_days=req.short_term_days,
                short_term_tax_rate=req.short_term_tax_rate,
                long_term_tax_rate=req.long_term_tax_rate,
            )
            tax = rb.estimate_tax(
                req.action, req.shares, req.price, req.avg_cost, req.holding_days,
            )
            pnl = (req.price - req.avg_cost) * req.shares
            return ok({
                "action": req.action,
                "shares": req.shares,
                "price": req.price,
                "avg_cost": req.avg_cost,
                "holding_days": req.holding_days,
                "unrealized_pnl": round(pnl, 2),
                "estimated_tax": round(tax, 2),
                "is_short_term": req.holding_days < req.short_term_days,
            })
        except Exception as e:
            logger.exception("税务估算失败")
            return err(50001, f"税务估算失败: {e}", http_status=500)

    @app.post("/api/rebalance/tax_loss_harvest")
    async def rebalance_tax_loss_harvest(req: TaxLossHarvestReq):
        """识别亏损收割候选。"""
        try:
            rb = SmartRebalancer()
            candidates = rb.tax_loss_harvest_candidates(req.positions, req.prices)
            return ok({
                "candidates": candidates,
                "total_candidates": len(candidates),
                "total_unrealized_loss": round(sum(c["unrealized_loss"] for c in candidates), 2),
            })
        except Exception as e:
            logger.exception("亏损收割识别失败")
            return err(50001, f"亏损收割识别失败: {e}", http_status=500)

    @app.get("/api/rebalance/status")
    async def rebalance_status():
        """返回再平衡模块支持的参数与默认配置。"""
        rb = SmartRebalancer()
        return ok({
            "drift_threshold": rb.drift_threshold,
            "commission_rate": rb.commission_rate,
            "stamp_tax_rate": rb.stamp_tax_rate,
            "slippage_rate": rb.slippage_rate,
            "short_term_days": rb.short_term_days,
            "short_term_tax_rate": rb.short_term_tax_rate,
            "long_term_tax_rate": rb.long_term_tax_rate,
            "impact_cost_coeff": rb.impact_cost_coeff,
        })
