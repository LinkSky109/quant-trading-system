"""因子风险暴露监控 API 路由（REQ-P1-09 扩展模块）。

按 server.py 现有扩展路由模式挂载::

    try:
        from _routes_factor_exposure import register_factor_exposure_routes
        register_factor_exposure_routes(app, manager, SYMBOL_SET, ok, err)
    except Exception as e:
        logger.warning("因子暴露路由注册失败: %s", e)

端点：
    POST /api/risk/factor_exposure           —— 计算当前组合暴露 + 超限 + 类别汇总
    GET  /api/risk/factor_exposure/history   —— 暴露时序历史
    GET  /api/risk/factor_exposure/alerts    —— 因子超限告警列表
    POST /api/risk/factor_exposure/neutralize—— 中性化调仓建议
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

from risk.factor_exposure import FactorExposureMonitor

logger = logging.getLogger("realtime_server")

#: 默认 SQLite 库：项目根目录/data/quant_trading.db
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_PROJECT_ROOT, "data", "quant_trading.db")

# 模块级单例：告警管理器 + 因子暴露监控器（惰性创建，便于测试 monkeypatch）
_ALERT_MANAGER: Any = None
_MONITOR: Any = None


def _get_alert_manager() -> Any:
    """返回模块级告警管理器单例（懒加载）。"""
    global _ALERT_MANAGER
    if _ALERT_MANAGER is None:
        from monitoring.alert import AlertManager

        _ALERT_MANAGER = AlertManager()
    return _ALERT_MANAGER


def _get_monitor() -> FactorExposureMonitor:
    """返回模块级 FactorExposureMonitor 单例（懒加载）。"""
    global _MONITOR
    if _MONITOR is None:
        _MONITOR = FactorExposureMonitor(
            db_path=DB_PATH, alert_manager=_get_alert_manager()
        )
    return _MONITOR


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class ExposureRequest(BaseModel):
    """组合暴露计算请求。"""

    holdings: List[Dict[str, Any]] = []           # [{symbol, weight}]
    factor_scores: Optional[Dict[str, Dict[str, float]]] = None  # {symbol: {factor: z}}


class NeutralizeRequest(BaseModel):
    """中性化调仓建议请求。"""

    holdings: List[Dict[str, Any]] = []
    factor_scores: Optional[Dict[str, Dict[str, float]]] = None
    target_factor: Optional[str] = None


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _derive_factor_scores(
    manager: Any, holdings: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """当请求未提供 factor_scores 时，用 FactorEngine 对持仓标的取最新 z-score。

    任何取不到 K 线或计算失败的标的都会被跳过（防御性 try/except），
    不影响其余标的。

    Args:
        manager: 实时数据管理器（manager.get(symbol).klines）。
        holdings: 持仓列表 ``[{symbol, weight}]``。

    Returns:
        ``{symbol: {factor: z_score}}``。
    """
    scores: Dict[str, Dict[str, float]] = {}
    if manager is None:
        return scores
    try:
        from factors.factor_engine import FactorEngine
        from data.data_fetcher import normalize_symbol

        engine = FactorEngine()
    except Exception as exc:  # pragma: no cover - 引擎不可用
        logger.warning("FactorEngine 初始化失败，无法自动计算因子分: %s", exc)
        return scores

    for h in holdings:
        sym = h.get("symbol")
        if not sym:
            continue
        try:
            norm = normalize_symbol(sym)
            sim = manager.get(norm)
            df = getattr(sim, "klines", None)
            if df is None or df.empty:
                logger.warning("标的 %s 无K线，跳过因子分计算", norm)
                continue
            scores[norm] = engine.factor_exposure(df, symbol=norm)
        except Exception as exc:  # 防御：单票失败不影响整体
            logger.warning("标的 %s 因子分计算失败，已跳过: %s", sym, exc)
            continue
    return scores


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_factor_exposure_routes(
    app: FastAPI,
    manager: Any,
    symbol_set: set,
    ok: Any,
    err: Any,
) -> None:
    """注册因子风险暴露监控相关路由。

    Args:
        app: FastAPI 实例。
        manager: 实时数据管理器（用于缺省 factor_scores 时取 K 线）。
        symbol_set: 股票池标的集合（保留参数，与其它扩展路由一致）。
        ok / err: server.py 的统一响应封装。
    """
    monitor = _get_monitor()
    alert_mgr = _get_alert_manager()

    @app.post("/api/risk/factor_exposure")
    async def calc_factor_exposure(req: ExposureRequest):
        """计算当前组合因子暴露、超限列表与类别汇总。

        请求体:
            holdings: ``[{symbol, weight}]``，权重和≈1。
            factor_scores: 可选 ``{symbol: {factor: z}}``；缺省时用 FactorEngine
                对持仓标的取 K 线计算（失败标的跳过）。
        """
        holdings = req.holdings or []
        if not holdings:
            return err(40001, "holdings 不能为空")

        factor_scores = req.factor_scores
        if not factor_scores:
            factor_scores = _derive_factor_scores(manager, holdings)

        try:
            result = monitor.calculate_exposure(holdings, factor_scores or {})
        except ValueError as e:
            return err(40002, str(e))
        except Exception as e:  # pragma: no cover
            logger.exception("因子暴露计算失败")
            return err(50000, f"因子暴露计算失败: {e}", http_status=500)

        breaches = monitor.check_limits(result.exposures)
        return ok({
            "exposures": result.exposures,
            "category_exposures": result.category_exposures,
            "factor_categories": result.factor_categories,
            "missing_symbols": result.missing_symbols,
            "weight_sum": result.weight_sum,
            "breaches": [b.to_dict() for b in breaches],
        })

    @app.get("/api/risk/factor_exposure/history")
    async def factor_exposure_history(
        factor: Optional[str] = Query(default=None),
        start: Optional[str] = Query(default=None),
        end: Optional[str] = Query(default=None),
    ):
        """查询因子暴露历史，可选按因子/日期区间过滤。"""
        try:
            rows = monitor.get_history(factor=factor, start=start, end=end)
            chart = (
                monitor.time_series_chart_data(factor) if factor else None
            )
            return ok({"history": rows, "chart": chart})
        except Exception as e:  # pragma: no cover
            logger.exception("查询因子暴露历史失败")
            return err(50000, f"查询历史失败: {e}", http_status=500)

    @app.get("/api/risk/factor_exposure/alerts")
    async def factor_exposure_alerts():
        """返回因子超限告警列表（category=factor_exposure_limit）。"""
        try:
            rows = alert_mgr.get_history(category="factor_exposure_limit")
            return ok({"alerts": rows, "count": len(rows)})
        except Exception as e:  # pragma: no cover
            logger.exception("查询因子告警失败")
            return err(50000, f"查询告警失败: {e}", http_status=500)

    @app.post("/api/risk/factor_exposure/neutralize")
    async def neutralize_exposure(req: NeutralizeRequest):
        """返回把超限因子暴露拉回阈值内的调仓建议。

        请求体:
            holdings: ``[{symbol, weight}]``。
            factor_scores: ``{symbol: {factor: z}}``。
            target_factor: 可选，只针对该因子给建议。
        """
        holdings = req.holdings or []
        if not holdings:
            return err(40001, "holdings 不能为空")
        factor_scores = req.factor_scores or {}
        if not factor_scores:
            factor_scores = _derive_factor_scores(manager, holdings)

        try:
            suggestions = monitor.neutralize(
                holdings, factor_scores or {}, target_factor=req.target_factor
            )
        except ValueError as e:
            return err(40002, str(e))
        except Exception as e:  # pragma: no cover
            logger.exception("生成中性化建议失败")
            return err(50000, f"生成中性化建议失败: {e}", http_status=500)

        return ok({"suggestions": suggestions, "target_factor": req.target_factor})
