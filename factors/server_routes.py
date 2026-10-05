"""多因子分析 API 路由集成代码片段。

本文件**不直接注册路由**，而是导出 :func:`register_factor_routes`，
由 ``web-dashboard/server.py`` 在创建 ``app`` 后调用一次即可：

.. code-block:: python

    from factors.server_routes import register_factor_routes
    register_factor_routes(app, manager, SYMBOL_SET, ok, err)

这样避免本文件反向 import server.py 造成循环依赖。

提供的接口：
  - POST /api/factors/calculate         计算单票因子值
  - POST /api/factors/ic_analysis      横截面 IC 分析
  - POST /api/factors/layered_backtest 分层回测
  - GET  /api/factors/list             因子列表
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic 请求体模型
# ---------------------------------------------------------------------------
class FactorCalculateReq(BaseModel):
    """计算单只股票因子值请求体。"""

    symbol: str = Field(..., description="标的代码，如 600519.SH")
    factor_names: Optional[List[str]] = Field(
        default=None, description="只返回指定因子；为空则返回全部因子"
    )


class ICAnalysisReq(BaseModel):
    """IC 分析请求体。"""

    factor_name: str = Field(..., description="因子名，见 GET /api/factors/list")
    forward_days: int = Field(default=5, ge=1, le=60, description="未来收益天数")
    symbols: Optional[List[str]] = Field(
        default=None, description="横截面标的子集；为空则使用整个股票池"
    )


class LayeredBacktestReq(BaseModel):
    """分层回测请求体。"""

    factor_name: str = Field(..., description="因子名")
    n_layers: int = Field(default=5, ge=2, le=10, description="分层数")
    forward_days: int = Field(default=5, ge=1, le=60, description="持有期天数")
    symbols: Optional[List[str]] = Field(default=None, description="标的子集")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _get_symbol_klines(manager: Any, symbol: str) -> Optional[pd.DataFrame]:
    """从数据管理器取某只标的的 K 线 DataFrame，失败返回 None。"""
    try:
        sim = manager.get(symbol)
    except Exception:  # pragma: no cover
        return None
    if sim is None or getattr(sim, "klines", None) is None:
        return None
    df = sim.klines
    return df if isinstance(df, pd.DataFrame) and not df.empty else None


def _build_panel_dict(
    manager: Any, engine: Any, symbols: List[str]
) -> Dict[str, pd.DataFrame]:
    """从 manager 拉取多只标的的因子+收盘价面板。"""
    panel: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = _get_symbol_klines(manager, sym)
        if df is None:
            continue
        try:
            computed = engine.calculate_factors(df, symbol=sym)
        except Exception as exc:  # pragma: no cover
            logger.warning("计算 %s 因子失败: %s", sym, exc)
            continue
        panel[sym] = computed
    return panel


# ---------------------------------------------------------------------------
# 路由注册入口
# ---------------------------------------------------------------------------
def register_factor_routes(
    app: Any,
    manager: Any,
    symbol_set: Set[str],
    ok: Any,
    err: Any,
) -> None:
    """把多因子分析路由挂载到 FastAPI app。

    Args:
        app: FastAPI 实例。
        manager: MultiSymbolManager，``manager.get(symbol).klines`` 为 K 线。
        symbol_set: 股票池集合（server.py 中的 SYMBOL_SET）。
        ok: 成功响应封装 ``ok(data)``。
        err: 错误响应封装 ``err(code, message, http_status)``。
    """
    # 延迟 import，避免本模块加载时拉起整个引擎依赖
    from factors.factor_engine import FactorEngine

    engine = FactorEngine()

    # ------------------------------------------------------------------
    # GET /api/factors/list
    # ------------------------------------------------------------------
    @app.get("/api/factors/list")
    async def list_factors() -> Dict[str, Any]:
        """返回全部可用因子的元数据列表。"""
        factors = engine.get_factor_list()
        return ok({"count": len(factors), "factors": factors})

    # ------------------------------------------------------------------
    # POST /api/factors/calculate
    # ------------------------------------------------------------------
    @app.post("/api/factors/calculate")
    async def calculate_factors(req: FactorCalculateReq) -> Any:
        """计算单只股票最新一日及历史因子值。"""
        from data.data_fetcher import normalize_symbol

        sym = normalize_symbol(req.symbol)
        df = _get_symbol_klines(manager, sym)
        if df is None:
            return err(40404, f"无行情数据: {req.symbol}", http_status=404)

        try:
            computed = engine.calculate_factors(df, symbol=sym)
        except Exception as exc:
            logger.exception("因子计算失败")
            return err(50000, f"因子计算失败: {exc}")

        names = req.factor_names or engine.factor_names
        invalid = [n for n in names if n not in engine.factor_names]
        if invalid:
            return err(40002, f"未知因子: {invalid}，"
                              f"可选: {engine.factor_names}")

        history = computed[names].dropna(how="all")
        latest = history.iloc[-1].to_dict() if not history.empty else {}
        history_records = {
            str(idx.date() if hasattr(idx, "date") else idx): row.dropna().to_dict()
            for idx, row in history.iterrows()
        }
        return ok({
            "symbol": sym,
            "latest": latest,
            "history": history_records,
            "exposure": engine.factor_exposure(df, symbol=sym),
        })

    # ------------------------------------------------------------------
    # POST /api/factors/ic_analysis
    # ------------------------------------------------------------------
    @app.post("/api/factors/ic_analysis")
    async def ic_analysis(req: ICAnalysisReq) -> Any:
        """对股票池（或子集）做因子 IC 分析。"""
        if req.factor_name not in engine.factor_names:
            return err(40002, f"未知因子: {req.factor_name}，"
                              f"可选: {engine.factor_names}")

        symbols = req.symbols or sorted(symbol_set)
        panel = _build_panel_dict(manager, engine, symbols)
        if len(panel) < 3:
            return err(40010, f"有效标的不足 3 只（当前 {len(panel)}），"
                              f"无法做横截面分析")

        try:
            result = engine.factor_ic_analysis(
                panel, req.factor_name, forward_days=req.forward_days
            )
        except Exception as exc:
            logger.exception("IC 分析失败")
            return err(50000, f"IC 分析失败: {exc}")

        ic_series = result.pop("ic_series")
        result["ic_series"] = {
            str(idx.date() if hasattr(idx, "date") else idx): float(v)
            for idx, v in ic_series.items()
        }
        result["symbols_used"] = sorted(panel.keys())
        return ok(result)

    # ------------------------------------------------------------------
    # POST /api/factors/layered_backtest
    # ------------------------------------------------------------------
    @app.post("/api/factors/layered_backtest")
    async def layered_backtest(req: LayeredBacktestReq) -> Any:
        """对股票池做因子分层回测。"""
        if req.factor_name not in engine.factor_names:
            return err(40002, f"未知因子: {req.factor_name}，"
                              f"可选: {engine.factor_names}")

        symbols = req.symbols or sorted(symbol_set)
        panel = _build_panel_dict(manager, engine, symbols)
        if len(panel) < req.n_layers:
            return err(40010, f"有效标的 {len(panel)} 只少于分层数 "
                              f"{req.n_layers}，无法分层")

        try:
            result = engine.layered_backtest(
                panel, req.factor_name,
                n_layers=req.n_layers, forward_days=req.forward_days,
            )
        except Exception as exc:
            logger.exception("分层回测失败")
            return err(50000, f"分层回测失败: {exc}")

        ls = result.pop("long_short_series")
        result["long_short_series"] = [float(v) for v in ls]
        result["symbols_used"] = sorted(panel.keys())
        return ok(result)
