"""多因子选股策略 API 路由模块。

本文件**不直接注册路由**，而是导出 :func:`register_multifactor_routes`，
由 ``web-dashboard/server.py`` 在 ``_register_extension_routes()`` 中调用：

.. code-block:: python

    from web-dashboard.multifactor_routes import register_multifactor_routes
    register_multifactor_routes(app, manager, SYMBOL_SET, ok, err)

提供的接口：
  - POST /api/multifactor/score     计算股票池最新横截面因子得分与排名
  - POST /api/multifactor/backtest  多因子策略多标的回测
  - GET  /api/multifactor/config    获取当前因子配置
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
class FactorWeight(BaseModel):
    """单个因子配置。"""

    name: str = Field(..., description="因子名，如 momentum_20")
    weight: float = Field(..., description="权重")
    direction: int = Field(default=1, description="+1 正向 / -1 反向")


class MultiFactorScoreReq(BaseModel):
    """多因子得分请求体。"""

    symbols: Optional[List[str]] = Field(
        default=None, description="标的子集；为空则使用整个股票池"
    )
    factors: Optional[List[FactorWeight]] = Field(
        default=None, description="因子配置；为空则使用 config 中的 multi_factor"
    )


class MultiFactorBacktestReq(BaseModel):
    """多因子策略回测请求体。"""

    start_date: str = Field(..., description="回测开始日期 YYYY-MM-DD")
    end_date: str = Field(..., description="回测结束日期 YYYY-MM-DD")
    symbols: Optional[List[str]] = Field(default=None, description="标的子集")
    factors: Optional[List[FactorWeight]] = Field(default=None, description="因子配置")
    rebalance_days: Optional[int] = Field(default=None, ge=1, le=60)
    top_n: Optional[int] = Field(default=None, ge=1, le=30)
    initial_capital: Optional[float] = Field(default=None, gt=0)


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


def _load_strategy_params(
    manager: Any,
    req_factors: Optional[List[FactorWeight]] = None,
    rebalance_days: Optional[int] = None,
    top_n: Optional[int] = None,
) -> Dict[str, Any]:
    """合并 config 中的 multi_factor 配置与请求覆盖参数。"""
    cfg: Dict[str, Any] = {}
    try:
        cfg = (manager.cfg or {}).get("multi_factor", {}) or {}
    except Exception:  # pragma: no cover
        cfg = {}

    params: Dict[str, Any] = {
        "rebalance_days": rebalance_days or cfg.get("rebalance_days", 5),
        "top_n": top_n or cfg.get("top_n", 5),
        "standardization": cfg.get("standardization", "zscore"),
        "missing_handling": cfg.get("missing_handling", "median"),
    }
    if req_factors:
        params["factors"] = [f.model_dump() for f in req_factors]
    elif cfg.get("factors"):
        params["factors"] = cfg["factors"]
    return params


def _serialize_curve(s: pd.Series) -> List[Dict[str, Any]]:
    """把净值序列化为 [{date, value}] 列表。"""
    return [
        {"date": pd.Timestamp(idx).strftime("%Y-%m-%d"), "value": float(v)}
        for idx, v in s.items()
    ]


def _serialize_trades(trades: List[Any]) -> List[Dict[str, Any]]:
    """把回测交易记录序列化为可 JSON 化的字典列表。"""
    out: List[Dict[str, Any]] = []
    for t in trades:
        out.append({
            "date": pd.Timestamp(t.date).strftime("%Y-%m-%d"),
            "symbol": t.symbol,
            "action": t.action,
            "price": float(t.price),
            "shares": int(t.shares),
            "amount": float(t.amount),
            "pnl": float(t.pnl) if t.pnl is not None else None,
            "reason": t.reason,
        })
    return out


# ---------------------------------------------------------------------------
# 路由注册入口
# ---------------------------------------------------------------------------
def register_multifactor_routes(
    app: Any,
    manager: Any,
    symbol_set: Set[str],
    ok: Any,
    err: Any,
) -> None:
    """把多因子选股路由挂载到 FastAPI app。

    Args:
        app: FastAPI 实例。
        manager: MultiSymbolManager，``manager.get(symbol).klines`` 为 K 线，
            ``manager.cfg`` 为加载后的全局配置字典。
        symbol_set: 股票池集合。
        ok: 成功响应封装 ``ok(data)``。
        err: 错误响应封装 ``err(code, message, http_status)``。
    """
    # 延迟 import，避免模块加载时拉起整个引擎依赖
    from strategies.multi_factor import MultiFactorStrategy

    # ------------------------------------------------------------------
    # GET /api/multifactor/config
    # ------------------------------------------------------------------
    @app.get("/api/multifactor/config")
    async def get_multifactor_config() -> Any:
        """返回当前多因子策略配置。"""
        try:
            cfg = (manager.cfg or {}).get("multi_factor", {}) or {}
            return ok({
                "factors": cfg.get("factors", []),
                "rebalance_days": cfg.get("rebalance_days", 5),
                "top_n": cfg.get("top_n", 5),
                "standardization": cfg.get("standardization", "zscore"),
                "missing_handling": cfg.get("missing_handling", "median"),
            })
        except Exception as exc:
            logger.exception("获取多因子配置失败")
            return err(50001, str(exc), http_status=500)

    # ------------------------------------------------------------------
    # POST /api/multifactor/score
    # ------------------------------------------------------------------
    @app.post("/api/multifactor/score")
    async def multifactor_score(req: MultiFactorScoreReq) -> Any:
        """计算股票池所有标的最新横截面因子得分与排名。"""
        try:
            symbols = req.symbols or sorted(symbol_set)
            if not symbols:
                return err(40010, "股票池为空", http_status=400)

            # 拉取 K 线
            panel: Dict[str, pd.DataFrame] = {}
            for sym in symbols:
                df = _get_symbol_klines(manager, sym)
                if df is not None:
                    panel[sym] = df
            if len(panel) < 2:
                return err(40010, f"有效标的不足 2 只（当前 {len(panel)}），"
                                  f"无法做横截面打分")

            params = _load_strategy_params(manager, req.factors)
            strategy = MultiFactorStrategy(params)
            scores_df = strategy.compute_cross_sectional_scores(panel)
            if scores_df.empty:
                return err(50002, "因子得分为空，请检查行情数据长度")

            latest_date = scores_df.dropna(how="all").index.max()
            row = scores_df.loc[latest_date].dropna().sort_values(ascending=False)

            # 计算各标的最新一日的单因子暴露值
            from factors.factor_engine import FactorEngine
            engine = FactorEngine()
            factor_values: Dict[str, Dict[str, float]] = {}
            for sym, df in panel.items():
                try:
                    computed = engine.calculate_factors(df, symbol=sym)
                    fnames = [f["name"] for f in strategy.factors]
                    last = computed[fnames].iloc[-1]
                    factor_values[sym] = {
                        k: (float(v) if pd.notna(v) else None)
                        for k, v in last.items()
                    }
                except Exception:  # pragma: no cover
                    factor_values[sym] = {}

            scores_list = [
                {
                    "symbol": sym,
                    "score": float(score),
                    "rank": rank,
                    "factor_values": factor_values.get(sym, {}),
                }
                for rank, (sym, score) in enumerate(row.items(), start=1)
            ]

            return ok({
                "as_of_date": pd.Timestamp(latest_date).strftime("%Y-%m-%d"),
                "scores": scores_list,
                "factor_config": [
                    {k: v for k, v in f.items()} for f in strategy.factors
                ],
                "n_symbols": len(panel),
            })
        except Exception as exc:
            logger.exception("多因子得分计算失败")
            return err(50001, str(exc), http_status=500)

    # ------------------------------------------------------------------
    # POST /api/multifactor/backtest
    # ------------------------------------------------------------------
    @app.post("/api/multifactor/backtest")
    async def multifactor_backtest(req: MultiFactorBacktestReq) -> Any:
        """多因子策略多标的回测。"""
        try:
            from backtest.engine import BacktestEngine

            symbols = req.symbols or sorted(symbol_set)
            if len(symbols) < 2:
                return err(40010, "标的数不足 2 只，无法做横截面选股")

            # 拉取并按日期切片
            panel: Dict[str, pd.DataFrame] = {}
            for sym in symbols:
                df = _get_symbol_klines(manager, sym)
                if df is None or df.empty:
                    continue
                mask = (df.index >= pd.Timestamp(req.start_date)) & \
                       (df.index <= pd.Timestamp(req.end_date))
                sliced = df.loc[mask]
                if not sliced.empty:
                    panel[sym] = sliced
            if len(panel) < 2:
                return err(40010, f"回测区间内有效标的不足 2 只（{len(panel)}）")

            params = _load_strategy_params(
                manager, req.factors, req.rebalance_days, req.top_n
            )
            strategy = MultiFactorStrategy(params)
            strategy.set_cross_section_data(panel)

            # 回测引擎参数：优先请求体，其次 config.backtest
            bt_cfg = (manager.cfg or {}).get("backtest", {}) or {}
            initial_capital = req.initial_capital or bt_cfg.get(
                "initial_capital", 1_000_000.0
            )
            engine = BacktestEngine(
                initial_capital=float(initial_capital),
                commission_rate=float(bt_cfg.get("commission_rate", 0.00025)),
                stamp_tax_rate=float(bt_cfg.get("stamp_tax_rate", 0.0005)),
                slippage_rate=float(bt_cfg.get("slippage_rate", 0.001)),
                risk_free_rate=float(bt_cfg.get("risk_free_rate", 0.02)),
                trading_days=int(bt_cfg.get("trading_days_per_year", 252)),
            )
            result = engine.run(panel, strategy, symbol="")

            return ok({
                "metrics": {k: (float(v) if isinstance(v, (int, float))
                                and pd.notna(v) else v)
                            for k, v in result.metrics.items()},
                "equity_curve": _serialize_curve(result.equity_curve),
                "benchmark_curve": _serialize_curve(result.benchmark_curve),
                "trades": _serialize_trades(result.trades),
                "rebalance_history": strategy.get_rebalance_history(),
                "n_symbols": len(panel),
            })
        except Exception as exc:
            logger.exception("多因子回测失败")
            return err(50001, str(exc), http_status=500)
