"""风险模型 API 路由（扩展模块）。

按 server.py 现有扩展路由模式挂载：

    try:
        from _routes_risk import register_risk_routes
        register_risk_routes(app, manager, SYMBOL_SET, ok, err)
    except Exception as e:
        logger.warning("风险模型路由注册失败: %s", e)

端点：
    POST /api/risk/var           —— 组合 VaR/CVaR 计算（历史法/参数法）
    POST /api/risk/stress_test   —— 压力测试
    GET  /api/risk/var_status    —— 可用方法/置信度/场景清单
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from risk.var_model import STRESS_SCENARIOS, VaRModel

logger = logging.getLogger("realtime_server")

_MIN_OBS = 30  # 参与 VaR 计算的最少日收益率观测数


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class VaRReq(BaseModel):
    """组合 VaR 计算请求。"""

    symbols: List[str] = []
    weights: List[float] = []
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    method: str = "historical"  # historical / parametric
    confidence: float = 0.95


class StressTestReq(BaseModel):
    """压力测试请求。"""

    symbols: List[str] = []
    weights: List[float] = []
    scenario: str = "2008_crisis"
    portfolio_value: float = 1_000_000.0


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _extract_returns_matrix(
    manager: Any,
    symbols: List[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    """从 manager 中提取各标的收盘价并对齐成收益率矩阵。

    Returns:
        行=日期，列=标的 的日收益率 DataFrame。

    Raises:
        ValueError: 标的缺失数据或有效观测不足。
    """
    from data.data_fetcher import normalize_symbol

    series: Dict[str, pd.Series] = {}
    for sym in symbols:
        norm = normalize_symbol(sym)
        sim = manager.get(norm)
        df = getattr(sim, "klines", None)
        if df is None or df.empty:
            raise ValueError(f"标的 {norm} 暂无K线数据")
        df = df.copy()
        if start_date:
            df = df.loc[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df.loc[df.index <= pd.Timestamp(end_date)]
        if len(df) < 2:
            raise ValueError(f"标的 {norm} 有效K线不足")
        series[norm] = df["close"].astype(float)

    prices = pd.DataFrame(series).sort_index().ffill().dropna()
    if len(prices) < _MIN_OBS:
        raise ValueError(f"对齐后有效收益率观测不足({len(prices)}<{_MIN_OBS})")
    returns = prices.pct_change().dropna()
    return returns


def _normalize_weights(symbols: List[str], weights: Optional[List[float]]) -> List[float]:
    """权重归一化：未提供时等权；提供时按和归一化。"""
    n = len(symbols)
    if n == 0:
        return []
    if not weights or len(weights) != n:
        return [1.0 / n] * n
    arr = np.asarray(weights, dtype=float)
    total = arr.sum()
    if total == 0:
        return [1.0 / n] * n
    return (arr / total).tolist()


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_risk_routes(
    app: FastAPI,
    manager: Any,
    symbol_set: set,
    ok: Any,
    err: Any,
) -> None:
    """注册风险模型相关路由。

    Args:
        app: FastAPI 实例。
        manager: 实时数据管理器（manager.get(symbol).klines）。
        symbol_set: 股票池标的集合。
        ok / err: server.py 的统一响应封装。
    """
    from data.data_fetcher import normalize_symbol

    @app.post("/api/risk/var")
    async def calc_var(req: VaRReq):
        """计算组合 VaR/CVaR。

        请求体:
            symbols: 标的列表。
            weights: 权重（可空，等权）。
            method: historical / parametric。
            confidence: 0.95 / 0.99。
        """
        symbols = [normalize_symbol(s) for s in (req.symbols or []) if s]
        if len(symbols) < 1:
            return err(40001, "至少选择 1 只标的")
        invalid = [s for s in symbols if s not in symbol_set]
        if invalid:
            return err(40002, f"标的不在股票池内: {invalid}")
        if req.method not in ("historical", "parametric"):
            return err(40003, "method 必须为 historical/parametric")
        if not 0.5 < req.confidence < 1.0:
            return err(40004, "confidence 必须在 (0.5, 1.0) 之间")

        try:
            returns = _extract_returns_matrix(
                manager, symbols, req.start_date, req.end_date
            )
            weights = _normalize_weights(symbols, req.weights)
        except ValueError as e:
            return err(40005, str(e))
        except Exception as e:
            logger.exception("VaR 数据准备失败")
            return err(50000, f"VaR 计算失败: {e}")

        try:
            if req.method == "historical":
                # 历史法在组合层面：先按权重合成组合日收益率
                w = np.asarray(weights)
                port_returns = (returns * w).sum(axis=1)
                var = VaRModel.historical_var(port_returns, req.confidence)
                cvar = VaRModel.historical_cvar(port_returns, req.confidence)
                port_vol = float(port_returns.std(ddof=1))
                decomp = VaRModel.component_var(weights, returns, req.confidence)
            else:
                var = VaRModel.parametric_var(weights, returns, req.confidence)
                cvar = VaRModel.parametric_cvar(weights, returns, req.confidence)
                decomp = VaRModel.component_var(weights, returns, req.confidence)
                port_vol = decomp["portfolio_volatility"]

            stats = VaRModel.get_stats((returns * np.asarray(weights)).sum(axis=1))
            return ok({
                "var": float(var),
                "cvar": float(cvar),
                "method": req.method,
                "confidence": req.confidence,
                "portfolio_volatility": port_vol,
                "component_var": decomp["items"],
                "stats": stats,
                "symbols": symbols,
                "weights": weights,
            })
        except Exception as e:
            logger.exception("VaR 计算失败")
            return err(50001, f"VaR 计算失败: {e}", http_status=500)

    @app.post("/api/risk/stress_test")
    async def stress_test(req: StressTestReq):
        """对组合施加预设压力情景。"""
        symbols = [normalize_symbol(s) for s in (req.symbols or []) if s]
        if len(symbols) < 1:
            return err(40001, "至少选择 1 只标的")
        if req.scenario not in STRESS_SCENARIOS:
            return err(40002, f"未知情景: {req.scenario}，可选: {list(STRESS_SCENARIOS)}")

        try:
            # 取最新价作为基准价
            prices: Dict[str, float] = {}
            for sym in symbols:
                sim = manager.get(sym)
                df = getattr(sim, "klines", None)
                if df is None or df.empty:
                    return err(40003, f"标的 {sym} 暂无K线数据")
                prices[sym] = float(df["close"].iloc[-1])
            weights = _normalize_weights(symbols, req.weights)
            result = VaRModel.stress_test(
                weights, prices, req.scenario, req.portfolio_value
            )
            result["description"] = STRESS_SCENARIOS[req.scenario]["description"]
            return ok(result)
        except Exception as e:
            logger.exception("压力测试失败")
            return err(50000, f"压力测试失败: {e}", http_status=500)

    @app.get("/api/risk/var_status")
    async def var_status():
        """返回风险模型支持的方法、置信度与预设压力场景。"""
        return ok({
            "methods": ["historical", "parametric"],
            "confidences": [0.95, 0.99],
            "scenarios": VaRModel.list_scenarios(),
        })
