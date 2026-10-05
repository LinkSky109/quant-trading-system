"""滚动窗口优化（Walk-Forward）API 路由。

本文件为独立路由模块，**不直接修改 server.py**。在 server.py 末尾挂载：

    from _routes_walkforward import register_walkforward_routes
    register_walkforward_routes(app)

提供三个接口：
- ``POST /api/optimize/walk_forward``  运行滚动优化
- ``POST /api/optimize/random_search``  随机搜索滚动优化
- ``GET  /api/optimize/results/{result_id}`` 查询优化结果（内存缓存）

结果缓存：模块级全局 dict ``_WF_RESULTS``，optimize 完成后生成
``result_id``（时间戳 + 随机串）存入，GET 接口按 id 读取。
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Dict, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("realtime_server")

# 结果缓存：{result_id: result_dict}
_WF_RESULTS: Dict[str, dict] = {}

# 单次结果缓存上限（防止内存无限增长）
_MAX_CACHED_RESULTS = 50


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------

class WalkForwardReq(BaseModel):
    """滚动窗口优化请求体。"""
    symbol: str = "600519.SH"
    strategy: str = "ma_cross"
    param_grid: Dict[str, list]
    n_windows: int = 3
    is_ratio: float = 0.7
    objective: str = "sharpe"  # sharpe / return / calmar / sortino
    start_date: str = "2024-01-02"
    end_date: str = "2025-12-31"
    max_combos: int = 500


class RandomSearchReq(BaseModel):
    """随机搜索滚动优化请求体。"""
    symbol: str = "600519.SH"
    strategy: str = "ma_cross"
    param_grid: Dict[str, list]
    n_samples: int = 50
    n_windows: int = 3
    is_ratio: float = 0.7
    objective: str = "sharpe"
    start_date: str = "2024-01-02"
    end_date: str = "2025-12-31"


_ALLOWED_OBJECTIVES = ("sharpe", "return", "calmar", "sortino")


def _gen_result_id() -> str:
    """生成唯一结果 ID：时间戳 + 随机串。"""
    return f"wf_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def _cache_result(result_id: str, payload: dict) -> None:
    """把结果写入内存缓存，超出上限时淘汰最早写入的条目。"""
    _WF_RESULTS[result_id] = payload
    while len(_WF_RESULTS) > _MAX_CACHED_RESULTS:
        oldest = next(iter(_WF_RESULTS))
        _WF_RESULTS.pop(oldest, None)


def _serialize_result(result: dict, result_id: str) -> dict:
    """把 WalkForwardOptimizer 结果转为可 JSON 序列化的字典。

    combined_oos_equity（pd.Series）转为 [date, value] 列表；
    其余字段原样保留。
    """
    equity = result.get("combined_oos_equity")
    equity_list: list = []
    if isinstance(equity, pd.Series) and len(equity) > 0:
        equity_list = [
            [d.strftime("%Y-%m-%d"), round(float(v), 4)]
            for d, v in equity.items()
        ]

    windows = []
    for w in result.get("windows", []):
        windows.append({
            "window_index": w["window_index"],
            "is_range": w["is_range"],
            "oos_range": w["oos_range"],
            "best_params": w["best_params"],
            "is_metrics": w["is_metrics"],
            "oos_metrics": w["oos_metrics"],
            "is_objective": round(float(w["is_objective"]), 4),
            "oos_objective": round(float(w["oos_objective"]), 4),
        })

    return {
        "result_id": result_id,
        "objective": result.get("objective"),
        "n_windows_actual": result.get("n_windows_actual"),
        "total_combos": result.get("total_combos"),
        "windows": windows,
        "combined_oos_equity": equity_list,
        "combined_metrics": result.get("combined_metrics", {}),
        "overfitting_report": result.get("overfitting_report", {}),
        "recommended_params": result.get("recommended_params", {}),
        "param_heatmap": result.get("param_heatmap", {}),
    }


def register_walkforward_routes(
    app: FastAPI,
    manager: Any,
    normalize_symbol: Callable[[str], str],
    SYMBOL_SET: set,
    STRATEGY_NAMES: list,
    ok: Callable[..., dict],
    err: Callable[..., Any],
    get_strategy_class: Callable[[str], type],
) -> None:
    """把滚动优化路由挂载到 FastAPI app。

    所有依赖（manager / ok / err 等）由 server.py 注入，避免循环导入。
    """

    def _prepare_data(sym: str, start_date: str, end_date: str):
        """按日期区间取行情数据，返回 (df, 错误响应或None)。"""
        sim = manager.get(sym)
        df = sim.klines.copy()
        df = df.loc[(df.index >= pd.Timestamp(start_date)) &
                    (df.index <= pd.Timestamp(end_date))]
        if len(df) < 60:
            return None, err(40006, "所选标的在日期区间内有效数据不足(<60 根K线)")
        return df, None

    def _engine_kwargs() -> dict:
        """从全局配置读取回测引擎参数。"""
        bt_cfg = manager.cfg.get("backtest", {})
        return dict(
            initial_capital=float(bt_cfg.get("initial_capital", 1_000_000.0)),
            commission_rate=float(bt_cfg.get("commission_rate", 0.00025)),
            stamp_tax_rate=float(bt_cfg.get("stamp_tax_rate", 0.0005)),
            slippage_rate=float(bt_cfg.get("slippage_rate", 0.001)),
            risk_free_rate=float(bt_cfg.get("risk_free_rate", 0.02)),
            trading_days=int(bt_cfg.get("trading_days_per_year", 252)),
        )

    @app.post("/api/optimize/walk_forward")
    async def optimize_walk_forward(req: WalkForwardReq):
        """滚动窗口优化：IS 内寻优、OOS 检验，返回样本外合并结果与过拟合报告。"""
        sym = normalize_symbol(req.symbol)
        if sym not in SYMBOL_SET:
            return err(40001, f"标的不在股票池内: {req.symbol}")
        if req.strategy not in STRATEGY_NAMES:
            return err(40002, f"未知策略: {req.strategy}，可选: {STRATEGY_NAMES}")
        if req.objective not in _ALLOWED_OBJECTIVES:
            return err(40007, f"objective 必须为 {_ALLOWED_OBJECTIVES}，收到: {req.objective}")
        if not req.param_grid or any(len(v) == 0 for v in req.param_grid.values()):
            return err(40003, "param_grid 不能为空或包含空列表")

        try:
            from optimization.walk_forward import WalkForwardOptimizer

            df, err_resp = _prepare_data(sym, req.start_date, req.end_date)
            if err_resp is not None:
                return err_resp

            strategy_cls = get_strategy_class(req.strategy)
            optimizer = WalkForwardOptimizer(max_combos=req.max_combos)
            result = optimizer.optimize(
                data=df,
                strategy_class=strategy_cls,
                param_grid=req.param_grid,
                n_windows=req.n_windows,
                is_ratio=req.is_ratio,
                objective=req.objective,
                symbol=sym,
                **_engine_kwargs(),
            )

            result_id = _gen_result_id()
            payload = _serialize_result(result, result_id)
            _cache_result(result_id, payload)

            return ok({
                "result_id": result_id,
                "summary": {
                    "strategy": req.strategy,
                    "symbol": sym,
                    "objective": req.objective,
                    "n_windows": result.get("n_windows_actual"),
                    "overfitting": result.get("overfitting_report", {}),
                    "recommended_params": result.get("recommended_params", {}),
                },
            })
        except Exception as e:  # noqa: BLE001
            logger.exception("滚动窗口优化失败")
            return err(50001, f"滚动窗口优化失败: {e}", http_status=500)

    @app.post("/api/optimize/random_search")
    async def optimize_random_search(req: RandomSearchReq):
        """随机搜索滚动优化：从参数空间随机采样 n_samples 组做滚动寻优。"""
        sym = normalize_symbol(req.symbol)
        if sym not in SYMBOL_SET:
            return err(40001, f"标的不在股票池内: {req.symbol}")
        if req.strategy not in STRATEGY_NAMES:
            return err(40002, f"未知策略: {req.strategy}，可选: {STRATEGY_NAMES}")
        if req.objective not in _ALLOWED_OBJECTIVES:
            return err(40007, f"objective 必须为 {_ALLOWED_OBJECTIVES}，收到: {req.objective}")
        if not req.param_grid or any(len(v) == 0 for v in req.param_grid.values()):
            return err(40003, "param_grid 不能为空或包含空列表")

        try:
            from optimization.walk_forward import WalkForwardOptimizer

            df, err_resp = _prepare_data(sym, req.start_date, req.end_date)
            if err_resp is not None:
                return err_resp

            strategy_cls = get_strategy_class(req.strategy)
            optimizer = WalkForwardOptimizer()
            result = optimizer.random_search(
                data=df,
                strategy_class=strategy_cls,
                param_grid=req.param_grid,
                n_samples=req.n_samples,
                n_windows=req.n_windows,
                is_ratio=req.is_ratio,
                objective=req.objective,
                symbol=sym,
                **_engine_kwargs(),
            )

            result_id = _gen_result_id()
            payload = _serialize_result(result, result_id)
            _cache_result(result_id, payload)

            return ok({
                "result_id": result_id,
                "summary": {
                    "strategy": req.strategy,
                    "symbol": sym,
                    "objective": req.objective,
                    "n_samples": req.n_samples,
                    "n_windows": result.get("n_windows_actual"),
                    "overfitting": result.get("overfitting_report", {}),
                    "recommended_params": result.get("recommended_params", {}),
                },
            })
        except Exception as e:  # noqa: BLE001
            logger.exception("随机搜索滚动优化失败")
            return err(50001, f"随机搜索滚动优化失败: {e}", http_status=500)

    @app.get("/api/optimize/results/{result_id}")
    async def get_optimize_result(result_id: str):
        """按 result_id 查询滚动优化结果（含合并净值曲线、过拟合报告等）。"""
        payload = _WF_RESULTS.get(result_id)
        if payload is None:
            return err(40404, f"优化结果不存在或已过期: {result_id}", http_status=404)
        return ok(payload)
