"""自适应风险管理 API 路由（扩展模块）。

按 server.py 现有扩展路由模式挂载：

    try:
        from _routes_adaptive_risk import register_adaptive_risk_routes
        register_adaptive_risk_routes(app, manager, SYMBOL_SET, ok, err)
    except Exception as e:
        logger.warning("自适应风险管理路由注册失败: %s", e)

端点：
    POST /api/risk/adaptive/var           —— 动态窗口 VaR/CVaR 计算
    POST /api/risk/adaptive/budget        —— 风险预算动态调整
    POST /api/risk/adaptive/tail_risk     —— 尾部风险检测
    POST /api/risk/adaptive/stress_test   —— 压力测试（历史/蒙特卡洛）
    POST /api/risk/adaptive/report        —— 综合风险报告
    GET  /api/risk/adaptive/status        —— 参数与配置说明
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from risk.adaptive_risk import AdaptiveRiskManager
from risk.risk_manager import RiskManager

logger = logging.getLogger("realtime_server")

_MIN_OBS = 30


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class AdaptiveVaRReq(BaseModel):
    """动态 VaR 计算请求。"""

    symbols: List[str] = []
    weights: List[float] = []
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    method: str = "historical"  # historical / parametric
    confidence: float = 0.95
    base_var_window: int = 252
    min_var_window: int = 63
    max_var_window: int = 504


class RiskBudgetReq(BaseModel):
    """风险预算调整请求。"""

    market_volatility: Optional[float] = None
    portfolio_returns: Optional[List[float]] = None
    risk_budget_base: float = 0.20
    volatility_target: float = 0.15


class TailRiskReq(BaseModel):
    """尾部风险检测请求。"""

    symbols: List[str] = []
    weights: List[float] = []
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    method: str = "historical"
    confidence: float = 0.95
    tail_risk_threshold: float = 0.03


class StressTestReq(BaseModel):
    """压力测试请求。"""

    symbols: List[str] = []
    weights: List[float] = []
    scenario: str = "monte_carlo"
    portfolio_value: float = 1_000_000.0
    n_simulations: int = 10000
    horizon_days: int = 1
    confidence: float = 0.95


class RiskReportReq(BaseModel):
    """综合风险报告请求。"""

    symbols: List[str] = []
    weights: List[float] = []
    portfolio_value: float = 1_000_000.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _extract_returns_matrix(
    manager: Any,
    symbols: List[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    """从 manager 中提取各标的收盘价并对齐成收益率矩阵。"""
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


def register_adaptive_risk_routes(
    app: FastAPI,
    manager: Any,
    symbol_set: set,
    ok: Any,
    err: Any,
) -> None:
    """注册自适应风险管理相关路由。

    Args:
        app: FastAPI 实例。
        manager: 实时数据管理器。
        symbol_set: 股票池标的集合。
        ok / err: server.py 的统一响应封装。
    """
    from data.data_fetcher import normalize_symbol
    from risk.var_model import STRESS_SCENARIOS

    @app.post("/api/risk/adaptive/var")
    async def adaptive_var(req: AdaptiveVaRReq):
        """动态窗口 VaR/CVaR 计算。"""
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
                manager, symbols, req.start_date, req.end_date,
            )
            weights = _normalize_weights(symbols, req.weights)
        except ValueError as e:
            return err(40005, str(e))
        except Exception as e:
            logger.exception("VaR 数据准备失败")
            return err(50000, f"VaR 计算失败: {e}")

        try:
            arm = AdaptiveRiskManager(
                base_var_window=req.base_var_window,
                min_var_window=req.min_var_window,
                max_var_window=req.max_var_window,
                var_confidence=req.confidence,
            )
            w_arr = np.asarray(weights)
            port_returns = (returns * w_arr).sum(axis=1)
            result = arm.dynamic_var(
                port_returns,
                weights=w_arr if req.method == "parametric" else None,
                returns_matrix=returns if req.method == "parametric" else None,
                method=req.method,
            )
            return ok(result)
        except Exception as e:
            logger.exception("动态 VaR 计算失败")
            return err(50001, f"动态 VaR 计算失败: {e}", http_status=500)

    @app.post("/api/risk/adaptive/budget")
    async def adaptive_budget(req: RiskBudgetReq):
        """基于市场波动率动态调整风险预算。"""
        try:
            arm = AdaptiveRiskManager(
                risk_budget_base=req.risk_budget_base,
                volatility_target=req.volatility_target,
            )
            if req.portfolio_returns is not None:
                rets = pd.Series(req.portfolio_returns)
                result = arm.adjust_risk_budget(portfolio_returns=rets)
            else:
                result = arm.adjust_risk_budget(market_volatility=req.market_volatility)
            return ok(result)
        except Exception as e:
            logger.exception("风险预算调整失败")
            return err(50001, f"风险预算调整失败: {e}", http_status=500)

    @app.post("/api/risk/adaptive/tail_risk")
    async def adaptive_tail_risk(req: TailRiskReq):
        """尾部风险检测。"""
        symbols = [normalize_symbol(s) for s in (req.symbols or []) if s]
        if len(symbols) < 1:
            return err(40001, "至少选择 1 只标的")
        invalid = [s for s in symbols if s not in symbol_set]
        if invalid:
            return err(40002, f"标的不在股票池内: {invalid}")

        try:
            returns = _extract_returns_matrix(
                manager, symbols, req.start_date, req.end_date,
            )
            weights = _normalize_weights(symbols, req.weights)
        except ValueError as e:
            return err(40005, str(e))
        except Exception as e:
            logger.exception("尾部风险数据准备失败")
            return err(50000, f"尾部风险数据准备失败: {e}")

        try:
            arm = AdaptiveRiskManager(
                var_confidence=req.confidence,
                tail_risk_threshold=req.tail_risk_threshold,
            )
            w_arr = np.asarray(weights)
            port_returns = (returns * w_arr).sum(axis=1)
            alert = arm.detect_tail_risk(
                port_returns,
                weights=w_arr if req.method == "parametric" else None,
                returns_matrix=returns if req.method == "parametric" else None,
                method=req.method,
            )
            return ok(alert.as_dict())
        except Exception as e:
            logger.exception("尾部风险检测失败")
            return err(50001, f"尾部风险检测失败: {e}", http_status=500)

    @app.post("/api/risk/adaptive/stress_test")
    async def adaptive_stress_test(req: StressTestReq):
        """压力测试（历史情景或蒙特卡洛）。"""
        symbols = [normalize_symbol(s) for s in (req.symbols or []) if s]
        if len(symbols) < 1:
            return err(40001, "至少选择 1 只标的")
        invalid = [s for s in symbols if s not in symbol_set]
        if invalid:
            return err(40002, f"标的不在股票池内: {invalid}")
        if req.scenario != "monte_carlo" and req.scenario not in STRESS_SCENARIOS:
            return err(40003, f"未知情景: {req.scenario}")

        try:
            returns = _extract_returns_matrix(
                manager, symbols, None, None,
            )
            weights = _normalize_weights(symbols, req.weights)
        except ValueError as e:
            return err(40005, str(e))
        except Exception as e:
            logger.exception("压力测试数据准备失败")
            return err(50000, f"压力测试数据准备失败: {e}")

        try:
            arm = AdaptiveRiskManager()
            w_arr = np.asarray(weights)
            result = arm.run_stress_test(
                weights=w_arr,
                returns_matrix=returns,
                scenario=req.scenario,
                portfolio_value=req.portfolio_value,
                n_simulations=req.n_simulations,
                horizon_days=req.horizon_days,
                confidence=req.confidence,
            )
            return ok(result.as_dict())
        except Exception as e:
            logger.exception("压力测试失败")
            return err(50001, f"压力测试失败: {e}", http_status=500)

    @app.post("/api/risk/adaptive/report")
    async def adaptive_risk_report(req: RiskReportReq):
        """生成综合风险报告。"""
        symbols = [normalize_symbol(s) for s in (req.symbols or []) if s]
        if len(symbols) < 1:
            return err(40001, "至少选择 1 只标的")
        invalid = [s for s in symbols if s not in symbol_set]
        if invalid:
            return err(40002, f"标的不在股票池内: {invalid}")

        try:
            returns = _extract_returns_matrix(
                manager, symbols, req.start_date, req.end_date,
            )
            weights = _normalize_weights(symbols, req.weights)
        except ValueError as e:
            return err(40005, str(e))
        except Exception as e:
            logger.exception("风险报告数据准备失败")
            return err(50000, f"风险报告数据准备失败: {e}")

        try:
            arm = AdaptiveRiskManager()
            w_arr = np.asarray(weights)
            port_returns = (returns * w_arr).sum(axis=1)
            report = arm.risk_report(
                portfolio_returns=port_returns,
                weights=w_arr,
                returns_matrix=returns,
                portfolio_value=req.portfolio_value,
            )
            # 清理不可序列化的对象
            if report.get("stress_test") is None:
                report["stress_test"] = None
            return ok(report)
        except Exception as e:
            logger.exception("风险报告生成失败")
            return err(50001, f"风险报告生成失败: {e}", http_status=500)

    @app.get("/api/risk/adaptive/status")
    async def adaptive_risk_status():
        """返回自适应风险管理支持的参数与配置。"""
        arm = AdaptiveRiskManager()
        return ok({
            "base_var_window": arm.base_var_window,
            "min_var_window": arm.min_var_window,
            "max_var_window": arm.max_var_window,
            "var_confidence": arm.var_confidence,
            "tail_risk_threshold": arm.tail_risk_threshold,
            "risk_budget_base": arm.risk_budget_base,
            "volatility_target": arm.volatility_target,
            "var_methods": ["historical", "parametric"],
            "stress_scenarios": list(STRESS_SCENARIOS.keys()) + ["monte_carlo"],
            "tail_risk_levels": ["low", "medium", "high", "critical"],
        })
