"""多因子模型优化 API 路由（扩展模块）。

按 server.py 现有扩展路由模式挂载：

    try:
        from _routes_multi_factor_opt import register_multi_factor_opt_routes
        register_multi_factor_opt_routes(app, manager, SYMBOL_SET, ok, err)
    except Exception as e:
        logger.warning("多因子优化路由注册失败: %s", e)

端点：
    POST /api/factors/optimize        —— 多因子优化（IC/IR 加权 + 合成得分）
    POST /api/factors/orthogonalize   —— 因子正交化
    POST /api/factors/winsorize       —— 去极值处理
    GET  /api/factors/opt_status      —— 支持的加权方法与参数说明
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from factors.multi_factor_opt import MultiFactorOptimizer

logger = logging.getLogger("realtime_server")

_MIN_OBS = 30


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class FactorOptimizeReq(BaseModel):
    """多因子优化请求。"""

    symbols: List[str] = []
    factor_names: List[str] = []
    weight_method: str = "ir"  # ic / ir / win_rate / equal
    forward_days: int = 5
    orthogonalize: bool = False
    winsorize: bool = True
    standardize: bool = True


class OrthogonalizeReq(BaseModel):
    """因子正交化请求。"""

    symbols: List[str] = []
    factor_names: List[str] = []
    method: str = "schmidt"


class WinsorizeReq(BaseModel):
    """去极值请求。"""

    symbols: List[str] = []
    factor_names: List[str] = []
    lower: float = 0.01
    upper: float = 0.99
    use_mad: bool = False
    mad_n: float = 3.0


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _extract_factor_panels(
    manager: Any,
    symbols: List[str],
    factor_names: List[str],
) -> Dict[str, pd.DataFrame]:
    """从 manager 中提取各标的 K 线并计算因子面板。

    Returns:
        {symbol: DataFrame(含因子列与 close 列)}。
    """
    from data.data_fetcher import normalize_symbol

    panels: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        norm = normalize_symbol(sym)
        sim = manager.get(norm)
        df = getattr(sim, "klines", None)
        if df is None or df.empty or len(df) < _MIN_OBS:
            continue
        from factors.factor_engine import FactorEngine
        engine = FactorEngine()
        factors = engine.calculate_factors(df.copy(), symbol=norm)
        # 只保留请求的因子 + close
        cols = [c for c in factor_names if c in factors.columns] + ["close"]
        if "close" not in factors.columns:
            factors["close"] = df["close"]
        panels[norm] = factors[[c for c in cols if c in factors.columns]].copy()
    return panels


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_multi_factor_opt_routes(
    app: FastAPI,
    manager: Any,
    symbol_set: set,
    ok: Any,
    err: Any,
) -> None:
    """注册多因子优化相关路由。

    Args:
        app: FastAPI 实例。
        manager: 实时数据管理器。
        symbol_set: 股票池标的集合。
        ok / err: server.py 的统一响应封装。
    """
    from data.data_fetcher import normalize_symbol

    @app.post("/api/factors/optimize")
    async def factor_optimize(req: FactorOptimizeReq):
        """多因子优化：计算 IC/IR、加权权重、合成得分。"""
        symbols = [normalize_symbol(s) for s in (req.symbols or []) if s]
        if len(symbols) < 1:
            return err(40001, "至少选择 1 只标的")
        invalid = [s for s in symbols if s not in symbol_set]
        if invalid:
            return err(40002, f"标的不在股票池内: {invalid}")
        if req.weight_method not in ("ic", "ir", "win_rate", "equal"):
            return err(40003, "weight_method 必须为 ic/ir/win_rate/equal")

        try:
            panels = _extract_factor_panels(manager, symbols, req.factor_names)
            if not panels:
                return err(40005, "所选标的因子数据不足")
        except Exception as e:
            logger.exception("因子面板提取失败")
            return err(50000, f"因子面板提取失败: {e}")

        try:
            opt = MultiFactorOptimizer()
            result = opt.optimize_factors(
                factor_panels=panels,
                factor_names=req.factor_names or list(
                    set(c for p in panels.values() for c in p.columns if c != "close")
                ),
                forward_days=req.forward_days,
                weight_method=req.weight_method,
            )
            # DataFrame 转 dict 便于 JSON 序列化
            score_df = result.get("composite_score")
            score_dict = {}
            if score_df is not None and not score_df.empty:
                score_dict = {
                    "index": score_df.index.strftime("%Y-%m-%d").tolist(),
                    "columns": list(score_df.columns),
                    "values": score_df.values.tolist(),
                }
            return ok({
                "weights": result["weights"],
                "ic_ir_stats": result["ic_ir_stats"],
                "factor_count": result["factor_count"],
                "composite_score": score_dict,
            })
        except Exception as e:
            logger.exception("多因子优化失败")
            return err(50001, f"多因子优化失败: {e}", http_status=500)

    @app.post("/api/factors/orthogonalize")
    async def factor_orthogonalize(req: OrthogonalizeReq):
        """对多标的多因子面板做正交化处理。"""
        symbols = [normalize_symbol(s) for s in (req.symbols or []) if s]
        if len(symbols) < 1:
            return err(40001, "至少选择 1 只标的")
        invalid = [s for s in symbols if s not in symbol_set]
        if invalid:
            return err(40002, f"标的不在股票池内: {invalid}")

        try:
            panels = _extract_factor_panels(manager, symbols, req.factor_names)
            if not panels:
                return err(40005, "所选标的因子数据不足")
        except Exception as e:
            logger.exception("因子面板提取失败")
            return err(50000, f"因子面板提取失败: {e}")

        try:
            opt = MultiFactorOptimizer()
            from factors.multi_factor_opt import _extract_factor_matrix
            # 对每个因子做正交化（跨标的截面）
            result = {}
            for fname in req.factor_names:
                mat = _extract_factor_matrix(panels, fname)
                if mat.empty or mat.shape[1] <= 1:
                    result[fname] = {"skipped": True, "reason": "标的不足或数据缺失"}
                    continue
                ortho = opt.orthogonalize(mat.T).T  # 标的×日期 -> 正交化 -> 转回
                result[fname] = {
                    "shape": list(ortho.shape),
                    "correlation_before": float(mat.T.corr().abs().values[np.triu_indices_from(np.ones((mat.shape[1], mat.shape[1])), k=1)].mean()) if mat.shape[1] > 1 else 0.0,
                    "correlation_after": float(ortho.T.corr().abs().values[np.triu_indices_from(np.ones((ortho.shape[1], ortho.shape[1])), k=1)].mean()) if ortho.shape[1] > 1 else 0.0,
                }
            return ok({"factors": result})
        except Exception as e:
            logger.exception("正交化失败")
            return err(50001, f"正交化失败: {e}", http_status=500)

    @app.post("/api/factors/winsorize")
    async def factor_winsorize(req: WinsorizeReq):
        """对多标的多因子面板做去极值处理。"""
        symbols = [normalize_symbol(s) for s in (req.symbols or []) if s]
        if len(symbols) < 1:
            return err(40001, "至少选择 1 只标的")
        invalid = [s for s in symbols if s not in symbol_set]
        if invalid:
            return err(40002, f"标的不在股票池内: {invalid}")

        try:
            panels = _extract_factor_panels(manager, symbols, req.factor_names)
            if not panels:
                return err(40005, "所选标的因子数据不足")
        except Exception as e:
            logger.exception("因子面板提取失败")
            return err(50000, f"因子面板提取失败: {e}")

        try:
            opt = MultiFactorOptimizer()
            from factors.multi_factor_opt import _extract_factor_matrix
            result = {}
            for fname in req.factor_names:
                mat = _extract_factor_matrix(panels, fname)
                if mat.empty:
                    result[fname] = {"skipped": True}
                    continue
                if req.use_mad:
                    trimmed = opt.mad_trim(mat, n=req.mad_n)
                else:
                    trimmed = opt.winsorize(mat, lower=req.lower, upper=req.upper)
                result[fname] = {
                    "shape": list(trimmed.shape),
                    "max_before": float(mat.max().max()),
                    "min_before": float(mat.min().min()),
                    "max_after": float(trimmed.max().max()),
                    "min_after": float(trimmed.min().min()),
                }
            return ok({"factors": result})
        except Exception as e:
            logger.exception("去极值失败")
            return err(50001, f"去极值失败: {e}", http_status=500)

    @app.get("/api/factors/opt_status")
    async def factor_opt_status():
        """返回多因子优化支持的参数与方法。"""
        return ok({
            "weight_methods": ["ic", "ir", "win_rate", "equal"],
            "outlier_methods": ["winsorize", "mad_trim"],
            "orthogonalize_methods": ["schmidt"],
            "standardize": True,
        })
