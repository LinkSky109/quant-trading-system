"""策略参数网格搜索优化模块。

遍历参数网格中的所有组合，每组运行一次回测，
按指定目标（夏普/收益/回撤）排序后返回 Top N。

与 BacktestEngine.parameter_sweep 的区别：
- 支持目标指标排序与 Top N 截断
- 支持进度回调，便于 Web 端实时展示
- 单组回测失败不影响整体流程
- 组合数上限保护（超限时警告）
"""
from __future__ import annotations

import itertools
import logging
import math
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from backtest.engine import BacktestEngine

logger = logging.getLogger(__name__)

# 目标指标 → metrics dict 中的键
_OBJECTIVE_KEYS: Dict[str, str] = {
    "sharpe": "夏普比率",
    "return": "累计收益率",
    "drawdown": "最大回撤",
}

# 排序方向：True 表示降序，False 表示升序
_OBJECTIVE_DESC: Dict[str, bool] = {
    "sharpe": True,
    "return": True,
    "drawdown": False,
}


class GridSearchOptimizer:
    """网格搜索优化器。

    对给定策略类和参数网格进行穷举回测，按目标指标排序。

    Attributes:
        max_combos: 参数组合数上限保护，超过时仅记录警告、仍继续执行。
    """

    def __init__(self, max_combos: int = 200):
        """初始化优化器。

        Args:
            max_combos: 参数组合数上限保护。当组合数超过该值时，
                通过 logging.warning 记录警告，但仍会执行全部回测。
        """
        self.max_combos = max_combos

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def optimize(
        self,
        data: pd.DataFrame,
        strategy_class: type,
        param_grid: Dict[str, list],
        symbol: str = "",
        start_date: str = "",
        end_date: str = "",
        objective: str = "sharpe",
        top_n: int = 10,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
        **engine_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """执行网格搜索。

        Args:
            data: 行情数据，需含 open/high/low/close/volume 列，DatetimeIndex。
            strategy_class: 策略类（如 MACrossStrategy），构造函数接受 params 字典。
            param_grid: 参数网格，如 {"fast_period": [3, 5, 8], "slow_period": [15, 20]}。
            symbol: 标的代码，透传给回测引擎。
            start_date: 回测起始日期（含），如 "2024-01-01"；为空则不过滤。
            end_date: 回测结束日期（含），如 "2024-12-31"；为空则不过滤。
            objective: 优化目标，可选 sharpe / return / drawdown。
                - sharpe   → 按夏普比率降序
                - return   → 按累计收益率降序
                - drawdown → 按最大回撤升序（回撤为负值，越接近 0 越好）
            top_n: 返回排名前 N 的组合。
            progress_callback: 进度回调，签名 callback(current_index, total_count, current_result)。
                current_result 为当前已完成组合的结果字典（含 params/metrics），
                若当前组合回测失败则为 {"params": ..., "error": ...}。
            **engine_kwargs: 透传给 BacktestEngine 的参数（initial_capital 等）。

        Returns:
            排序后的结果列表，每项形如
            {"params": {...}, "metrics": {...全量指标...}}。
            若 param_grid 为空则返回空列表。
        """
        if objective not in _OBJECTIVE_KEYS:
            raise ValueError(
                f"不支持的 objective={objective!r}，"
                f"可选: {list(_OBJECTIVE_KEYS.keys())}"
            )

        # 空网格直接返回
        if not param_grid or any(len(v) == 0 for v in param_grid.values()):
            logger.info("param_grid 为空，跳过网格搜索")
            return []

        # 按日期过滤数据
        work_data = self._filter_by_date(data, start_date, end_date)

        # 生成参数组合
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))
        total = len(combos)

        if total > self.max_combos:
            logger.warning(
                "参数组合数 %d 超过 max_combos=%d，仍将继续执行全部回测",
                total, self.max_combos,
            )

        results: List[Dict[str, Any]] = []
        for idx, combo in enumerate(combos, start=1):
            params = dict(zip(keys, combo))
            item = self._run_one(
                work_data, strategy_class, params, symbol, engine_kwargs,
            )
            results.append(item)

            if progress_callback is not None:
                try:
                    progress_callback(idx, total, item)
                except Exception:  # noqa: BLE001 - 回调异常不应中断搜索
                    logger.exception("progress_callback 抛出异常（已忽略）")

        # 按目标排序并截断
        results = self._sort_results(results, objective)
        return results[:top_n]

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_by_date(
        data: pd.DataFrame, start_date: str, end_date: str,
    ) -> pd.DataFrame:
        """按 start/end 日期过滤数据（含端点）。"""
        if not start_date and not end_date:
            return data
        mask = pd.Series(True, index=data.index)
        if start_date:
            mask &= data.index >= pd.Timestamp(start_date)
        if end_date:
            mask &= data.index <= pd.Timestamp(end_date)
        return data.loc[mask]

    def _run_one(
        self,
        data: pd.DataFrame,
        strategy_class: type,
        params: Dict[str, Any],
        symbol: str,
        engine_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """运行单组参数回测，失败时返回带 error 的条目而不抛出。"""
        try:
            strategy = strategy_class(params)
            engine = BacktestEngine(**engine_kwargs)
            result = engine.run(data, strategy, symbol=symbol)
            return {"params": dict(params), "metrics": dict(result.metrics)}
        except Exception as exc:  # noqa: BLE001 - 单组失败不影响其他组
            logger.exception("参数组合回测失败: params=%s", params)
            return {
                "params": dict(params),
                "metrics": {},
                "error": f"{type(exc).__name__}: {exc}",
            }

    @staticmethod
    def _sort_results(
        results: List[Dict[str, Any]], objective: str,
    ) -> List[Dict[str, Any]]:
        """按目标指标对结果排序。失败/无指标的条目排到末尾。"""
        metric_key = _OBJECTIVE_KEYS[objective]
        descending = _OBJECTIVE_DESC[objective]

        def sort_key(item: Dict[str, Any]) -> float:
            metrics = item.get("metrics") or {}
            value = metrics.get(metric_key)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                # 缺失值排到末尾
                return float("-inf") if descending else float("inf")
            return float(value)

        return sorted(results, key=sort_key, reverse=descending)
