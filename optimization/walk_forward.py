"""策略参数滚动优化（Walk-Forward Optimization）模块。

将历史数据按时间切分为多个滚动窗口，每个窗口 =
IS（in-sample，训练段）+ OOS（out-of-sample，测试段）：

1. 在 IS 窗口内用 :class:`~optimization.grid_search.GridSearchOptimizer`
   做参数搜索，选出最优参数；
2. 用该最优参数在紧随其后、**无重叠**的 OOS 窗口上回测，得到样本外绩效；
3. 滚动到下一个窗口，重复上述过程；
4. 合并所有 OOS 段净值曲线作为样本外真实表现，并做过拟合检测。

严格避免未来函数：每个窗口的 OOS 全部位于其 IS 之后，
IS 内选出的参数只用于之后的数据，绝不回看未来。

与 :class:`~optimization.grid_search.GridSearchOptimizer` 的区别：
- 网格搜索在全段数据上一次性寻优（容易过拟合）；
- 滚动优化把训练与测试按时间切分，用样本外绩效检验参数稳健性。
"""
from __future__ import annotations

import itertools
import logging
import math
import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.metrics import (
    calc_all_metrics,
    calc_annualized_return,
    calc_cumulative_return,
    calc_max_drawdown,
    calc_sharpe_ratio,
)
from optimization.grid_search import GridSearchOptimizer

logger = logging.getLogger(__name__)

# 优化目标 → 是否越大越好（True 降序）
_OBJECTIVE_HIGHER_BETTER: Dict[str, bool] = {
    "sharpe": True,
    "return": True,
    "calmar": True,
    "sortino": True,
}


class WalkForwardOptimizer:
    """滚动窗口优化器。

    Attributes:
        max_combos: 单个 IS 窗口内参数组合数上限保护，超过时仅记录警告。
    """

    def __init__(self, max_combos: int = 500):
        """初始化优化器。

        Args:
            max_combos: 单个 IS 窗口内参数组合数上限保护，
                超过该值时记录 warning 但仍继续执行。
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
        n_windows: int = 3,
        is_ratio: float = 0.7,
        objective: str = "sharpe",
        symbol: str = "",
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
        **engine_kwargs: Any,
    ) -> Dict[str, Any]:
        """执行滚动优化。

        Args:
            data: 行情数据，需含 open/high/low/close/volume 列，DatetimeIndex。
            strategy_class: 策略类（如 MACrossStrategy），构造函数接受 params 字典。
            param_grid: 参数网格，如 {"fast_period": [3, 5, 8], "slow_period": [15, 20]}。
            n_windows: 滚动窗口数量。
            is_ratio: 每个窗口中 IS 段占比（0~1），默认 0.7。
            objective: 优化目标，可选
                sharpe（夏普比率，默认）/ return（累计收益）/
                calmar（卡玛比率）/ sortino（索提诺比率）。
            symbol: 标的代码，透传给回测引擎。
            progress_callback: 进度回调，签名 callback(window_idx, n_windows, window_info)。
            **engine_kwargs: 透传给 BacktestEngine 的参数（initial_capital 等）。

        Returns:
            完整结果字典，包含 windows / combined_oos_equity / combined_metrics /
            overfitting_report / recommended_params / param_heatmap。
        """
        self._validate_objective(objective)
        self._validate_grid(param_grid)

        return self._run_walk_forward(
            data=data,
            strategy_class=strategy_class,
            param_grid=param_grid,
            n_windows=n_windows,
            is_ratio=is_ratio,
            objective=objective,
            symbol=symbol,
            progress_callback=progress_callback,
            sampled_combos=None,
            **engine_kwargs,
        )

    def random_search(
        self,
        data: pd.DataFrame,
        strategy_class: type,
        param_grid: Dict[str, list],
        n_samples: int = 50,
        n_windows: int = 3,
        is_ratio: float = 0.7,
        objective: str = "sharpe",
        symbol: str = "",
        **engine_kwargs: Any,
    ) -> Dict[str, Any]:
        """随机搜索优化：从参数空间随机采样 n_samples 组进行滚动优化。

        适用于参数空间过大、穷举成本过高的场景。接口与 :meth:`optimize` 类似，
        额外增加 ``n_samples`` 参数。

        Args:
            data: 行情数据。
            strategy_class: 策略类。
            param_grid: 参数网格。
            n_samples: 从完整参数空间随机采样的组合数量。
            n_windows: 滚动窗口数量。
            is_ratio: IS 占比。
            objective: 优化目标。
            symbol: 标的代码。
            **engine_kwargs: 透传给 BacktestEngine。

        Returns:
            与 :meth:`optimize` 相同结构的结果字典。
        """
        self._validate_objective(objective)
        self._validate_grid(param_grid)

        all_combos = self._build_combos(param_grid)
        if len(all_combos) <= n_samples:
            sampled = all_combos
        else:
            sampled = random.sample(all_combos, n_samples)
        logger.info("随机搜索: 完整空间 %d 组，采样 %d 组", len(all_combos), len(sampled))

        return self._run_walk_forward(
            data=data,
            strategy_class=strategy_class,
            param_grid=param_grid,
            n_windows=n_windows,
            is_ratio=is_ratio,
            objective=objective,
            symbol=symbol,
            progress_callback=None,
            sampled_combos=sampled,
            **engine_kwargs,
        )

    @staticmethod
    def split_windows(
        data: pd.DataFrame, n_windows: int, is_ratio: float,
    ) -> List[Dict[str, Any]]:
        """划分滚动窗口。

        规则：
        - 窗口总长 ``L ≈ N / n_windows``，其中 IS 占 ``is_ratio``，OOS 占剩余；
        - 每个窗口向前滚动一个 OOS 长度（step = OOS 长度）；
        - 同一窗口内 IS 与 OOS **无重叠**、严格时间顺序；
        - 最后一个窗口的 OOS 向后延伸到数据末尾。

        Args:
            data: 行情数据（DatetimeIndex）。
            n_windows: 窗口数量。
            is_ratio: IS 占比。

        Returns:
            窗口列表，每项含 is_data / oos_data / is_start / is_end /
            oos_start / oos_end（整数位置）以及起止日期字符串。
        """
        if n_windows < 1:
            raise ValueError("n_windows 必须 >= 1")
        if not 0.0 < is_ratio < 1.0:
            raise ValueError("is_ratio 必须在 (0, 1) 之间")

        n_total = len(data)
        # 至少要能切出一个完整窗口
        if n_total < n_windows + 2:
            raise ValueError(
                f"数据量 {n_total} 不足以切分 {n_windows} 个窗口"
            )

        base_len = n_total / n_windows
        is_len = max(2, int(round(base_len * is_ratio)))
        oos_len = max(1, int(round(base_len - is_len)))
        step = oos_len  # 默认步长 = 一个 OOS 长度

        windows: List[Dict[str, Any]] = []
        for i in range(n_windows):
            is_start = i * step
            is_end = is_start + is_len
            if is_end >= n_total:
                # 数据不足以再切一个完整 IS
                break
            oos_end = is_end + oos_len
            if i == n_windows - 1:
                oos_end = n_total  # 最后一个窗口 OOS 到数据末尾
            oos_end = min(oos_end, n_total)
            if oos_end <= is_end:
                break

            is_data = data.iloc[is_start:is_end]
            oos_data = data.iloc[is_end:oos_end]
            windows.append({
                "window_index": i,
                "is_data": is_data,
                "oos_data": oos_data,
                "is_start": is_start,
                "is_end": is_end,
                "oos_start": is_end,
                "oos_end": oos_end,
                "is_start_date": data.index[is_start],
                "is_end_date": data.index[is_end - 1],
                "oos_start_date": data.index[is_end],
                "oos_end_date": data.index[oos_end - 1],
            })
        return windows

    # ------------------------------------------------------------------
    # 内部：滚动优化主流程
    # ------------------------------------------------------------------

    def _run_walk_forward(
        self,
        data: pd.DataFrame,
        strategy_class: type,
        param_grid: Dict[str, list],
        n_windows: int,
        is_ratio: float,
        objective: str,
        symbol: str,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]],
        sampled_combos: Optional[List[Dict[str, Any]]],
        **engine_kwargs: Any,
    ) -> Dict[str, Any]:
        """滚动优化核心流程。sampled_combos 非空时在 IS 内只评估采样组合。"""
        windows = self.split_windows(data, n_windows, is_ratio)
        total_combos = len(self._build_combos(param_grid))

        window_results: List[Dict[str, Any]] = []
        oos_equity_parts: List[pd.Series] = []
        # 热力图累加：{param_name: {value: [oos_objective, ...]}}
        heatmap_acc: Dict[str, Dict[Any, List[float]]] = {}
        best_params_history: List[Dict[str, Any]] = []

        for win in windows:
            is_data = win["is_data"]
            oos_data = win["oos_data"]

            # 1. IS 窗口内参数搜索
            best = self._search_best_on_is(
                is_data=is_data,
                strategy_class=strategy_class,
                param_grid=param_grid,
                objective=objective,
                symbol=symbol,
                sampled_combos=sampled_combos,
                **engine_kwargs,
            )

            if best is None:
                logger.warning("窗口 %d 在 IS 内无有效参数组合，跳过", win["window_index"])
                continue

            best_params = best["params"]
            is_metrics = best["metrics"]

            # 2. OOS 窗口回测（用 IS 选出的最优参数）
            oos_metrics, oos_equity = self._backtest_params(
                data=oos_data,
                strategy_class=strategy_class,
                params=best_params,
                symbol=symbol,
                **engine_kwargs,
            )

            # 3. 记录
            oos_obj = self._objective_value(oos_metrics, objective, equity=oos_equity)
            is_obj = self._objective_value(is_metrics, objective)

            window_results.append({
                "window_index": win["window_index"],
                "is_range": [
                    pd.Timestamp(win["is_start_date"]).strftime("%Y-%m-%d"),
                    pd.Timestamp(win["is_end_date"]).strftime("%Y-%m-%d"),
                ],
                "oos_range": [
                    pd.Timestamp(win["oos_start_date"]).strftime("%Y-%m-%d"),
                    pd.Timestamp(win["oos_end_date"]).strftime("%Y-%m-%d"),
                ],
                "best_params": best_params,
                "is_metrics": is_metrics,
                "oos_metrics": oos_metrics,
                "is_objective": is_obj,
                "oos_objective": oos_obj,
            })
            best_params_history.append(best_params)

            # 热力图：把该窗口 OOS 表现记到最优参数的每个取值上
            for pname, pval in best_params.items():
                heatmap_acc.setdefault(pname, {}).setdefault(pval, []).append(oos_obj)

            if oos_equity is not None and len(oos_equity) >= 1:
                oos_equity_parts.append(oos_equity)

            if progress_callback is not None:
                try:
                    progress_callback(
                        win["window_index"] + 1, len(windows),
                        window_results[-1],
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("progress_callback 抛出异常（已忽略）")

        # 4. 合并 OOS 净值曲线
        combined_equity = self._combine_oos_equity(oos_equity_parts)
        rf = float(engine_kwargs.get("risk_free_rate", 0.02))
        td = int(engine_kwargs.get("trading_days", 252))
        combined_metrics = self._metrics_from_equity(combined_equity, rf, td)

        # 5. 过拟合检测
        overfitting_report = self._overfitting_report(
            window_results, best_params_history,
        )

        # 6. 推荐参数：数值取中位数，类别取众数
        recommended_params = self._recommend_params(best_params_history)

        # 7. 热力图聚合（平均值）
        param_heatmap = {
            pname: {
                str(pval): float(np.mean(vals))
                for pval, vals in val_map.items()
            }
            for pname, val_map in heatmap_acc.items()
        }

        return {
            "n_windows_actual": len(window_results),
            "n_windows_requested": n_windows,
            "objective": objective,
            "total_combos": total_combos,
            "windows": window_results,
            "combined_oos_equity": combined_equity,
            "combined_metrics": combined_metrics,
            "overfitting_report": overfitting_report,
            "recommended_params": recommended_params,
            "param_heatmap": param_heatmap,
        }

    # ------------------------------------------------------------------
    # IS 搜索 / OOS 回测
    # ------------------------------------------------------------------

    def _search_best_on_is(
        self,
        is_data: pd.DataFrame,
        strategy_class: type,
        param_grid: Dict[str, list],
        objective: str,
        symbol: str,
        sampled_combos: Optional[List[Dict[str, Any]]],
        **engine_kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """在 IS 段上搜索最优参数。

        复用 GridSearchOptimizer 跑完全部（或采样）组合，再按本优化器的
        objective 重新排序，返回最优一项；无有效组合时返回 None。
        """
        gs = GridSearchOptimizer(max_combos=self.max_combos)

        if sampled_combos is not None:
            # 随机搜索：逐组评估采样组合
            results: List[Dict[str, Any]] = []
            for params in sampled_combos:
                item = self._run_one_backtest(
                    is_data, strategy_class, params, symbol, engine_kwargs,
                )
                results.append(item)
        else:
            total_combos = len(self._build_combos(param_grid))
            # top_n 取全部，便于按 objective 重排
            results = gs.optimize(
                data=is_data,
                strategy_class=strategy_class,
                param_grid=param_grid,
                symbol=symbol,
                objective="sharpe",
                top_n=max(total_combos, 1),
                **engine_kwargs,
            )

        # 过滤失败项
        valid = [r for r in results if r.get("metrics")]
        if not valid:
            return None

        # 按 objective 排序（越大越好）
        valid.sort(
            key=lambda r: self._objective_value(r["metrics"], objective),
            reverse=True,
        )
        return valid[0]

    def _backtest_params(
        self,
        data: pd.DataFrame,
        strategy_class: type,
        params: Dict[str, Any],
        symbol: str,
        **engine_kwargs: Any,
    ) -> tuple[Dict[str, float], Optional[pd.Series]]:
        """用指定参数在一段数据上回测，返回 (metrics, equity_curve)。"""
        try:
            strategy = strategy_class(params)
            engine = BacktestEngine(**engine_kwargs)
            result = engine.run(data, strategy, symbol=symbol)
            return dict(result.metrics), result.equity_curve
        except Exception as exc:  # noqa: BLE001
            logger.exception("OOS 回测失败: params=%s", params)
            return {"错误": f"{type(exc).__name__}: {exc}"}, None

    @staticmethod
    def _run_one_backtest(
        data: pd.DataFrame,
        strategy_class: type,
        params: Dict[str, Any],
        symbol: str,
        engine_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """运行单组回测（随机搜索用），失败时返回带 error 的条目。"""
        try:
            strategy = strategy_class(params)
            engine = BacktestEngine(**engine_kwargs)
            result = engine.run(data, strategy, symbol=symbol)
            return {"params": dict(params), "metrics": dict(result.metrics)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("参数组合回测失败: params=%s", params)
            return {
                "params": dict(params), "metrics": {},
                "error": f"{type(exc).__name__}: {exc}",
            }

    # ------------------------------------------------------------------
    # 目标指标提取
    # ------------------------------------------------------------------

    def _objective_value(
        self,
        metrics: Dict[str, float],
        objective: str,
        equity: Optional[pd.Series] = None,
    ) -> float:
        """从 metrics（可选 equity）中提取优化目标值。"""
        try:
            if objective == "sharpe":
                return float(metrics.get("夏普比率", 0.0) or 0.0)
            if objective == "return":
                return float(metrics.get("累计收益率", 0.0) or 0.0)
            if objective == "calmar":
                ann = float(metrics.get("年化收益率", 0.0) or 0.0)
                dd = abs(float(metrics.get("最大回撤", 0.0) or 0.0))
                return ann / dd if dd > 1e-9 else 0.0
            if objective == "sortino":
                # metrics 内无下行标准差；若有 equity 则精确计算，
                # 否则退化为夏普比率（IS 段无逐组 equity 时的近似）。
                if equity is not None and len(equity) >= 3:
                    rf = 0.02
                    td = 252
                    return self._calc_sortino(equity, rf, td)
                return float(metrics.get("夏普比率", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return 0.0

    @staticmethod
    def _calc_sortino(equity: pd.Series, rf: float, td: int) -> float:
        """索提诺比率 = (年化收益 - 无风险利率) / 下行标准差(年化)。"""
        rets = equity.pct_change().dropna()
        if len(rets) < 2:
            return 0.0
        downside = rets[rets < 0]
        dd_std = downside.std()
        if dd_std is None or dd_std == 0 or pd.isna(dd_std):
            return 0.0
        ann = calc_annualized_return(equity, td)
        return float((ann - rf) / (dd_std * math.sqrt(td)))

    # ------------------------------------------------------------------
    # 结果聚合
    # ------------------------------------------------------------------

    @staticmethod
    def _combine_oos_equity(parts: List[pd.Series]) -> pd.Series:
        """把各 OOS 段净值曲线按时间首尾相接、复利拼接成一条连续净值曲线。

        每段先归一化为起点=1，再乘以上一段末端净值，从而段间无重叠、
        无未来函数，且绩效按复利衔接。
        """
        valid = [p for p in parts if p is not None and len(p) >= 1]
        if not valid:
            return pd.Series(dtype=float, name="combined_oos_equity")

        chained: List[pd.Series] = []
        running = 1.0
        for seg in valid:
            seg = seg.dropna()
            if len(seg) == 0:
                continue
            norm = seg / float(seg.iloc[0])
            norm = norm * running
            chained.append(norm)
            running = float(norm.iloc[-1])

        if not chained:
            return pd.Series(dtype=float, name="combined_oos_equity")
        combined = pd.concat(chained)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined.name = "combined_oos_equity"
        return combined

    @staticmethod
    def _metrics_from_equity(
        equity: pd.Series, rf: float, td: int,
    ) -> Dict[str, float]:
        """由合并后的 OOS 净值曲线计算绩效指标。"""
        if equity is None or len(equity) < 2:
            return {}
        base = calc_all_metrics(equity, [], rf, td)
        ann = base.get("年化收益率", 0.0)
        dd = abs(base.get("最大回撤", 0.0))
        base["卡玛比率"] = ann / dd if dd > 1e-9 else 0.0
        base["索提诺比率"] = WalkForwardOptimizer._calc_sortino(equity, rf, td)
        return base

    # ------------------------------------------------------------------
    # 过拟合检测 / 推荐参数
    # ------------------------------------------------------------------

    def _overfitting_report(
        self,
        window_results: List[Dict[str, Any]],
        best_params_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """生成过拟合风险评估报告。"""
        if not window_results:
            return {
                "risk_level": "unknown",
                "oos_is_ratio": None,
                "param_stability_score": None,
                "details": "无有效窗口结果",
            }

        # OOS/IS 目标绩效比（逐窗口取比值，再平均）
        ratios = []
        for w in window_results:
            is_obj = w["is_objective"]
            oos_obj = w["oos_objective"]
            if is_obj and abs(is_obj) > 1e-9:
                ratios.append(oos_obj / is_obj)
        ois_ratio = float(np.mean(ratios)) if ratios else None

        # 参数稳定性：相邻窗口最优参数的变化程度
        stability = self._param_stability(best_params_history)

        # 风险分级
        if ois_ratio is None:
            risk_level = "unknown"
        elif ois_ratio < 0.5:
            risk_level = "high"
        elif ois_ratio < 0.7:
            risk_level = "medium"
        else:
            risk_level = "low"

        details = (
            f"共 {len(window_results)} 个有效窗口；"
            f"OOS/IS 平均绩效比 = {ois_ratio:.3f}（<0.5 标记过拟合风险）；"
            f"参数稳定性得分 = {stability:.3f}（越接近 1 越稳定）。"
        )
        return {
            "risk_level": risk_level,
            "oos_is_ratio": round(ois_ratio, 4) if ois_ratio is not None else None,
            "param_stability_score": round(stability, 4),
            "details": details,
        }

    @staticmethod
    def _param_stability(history: List[Dict[str, Any]]) -> float:
        """相邻窗口最优参数的稳定性得分（0~1，越接近 1 越稳定）。

        数值参数：按相对变化度量稳定性（变化越小分越高）；
        分类参数：相等得 1，变化得 0。
        """
        if len(history) < 2:
            return 1.0
        scores: List[float] = []
        for prev, cur in zip(history[:-1], history[1:]):
            for key, cur_val in cur.items():
                if key not in prev:
                    continue
                prev_val = prev[key]
                if isinstance(prev_val, (int, float)) and isinstance(cur_val, (int, float)):
                    if prev_val == 0:
                        scores.append(1.0 if cur_val == 0 else 0.0)
                    else:
                        rel = abs(cur_val - prev_val) / (abs(prev_val) + 1e-9)
                        scores.append(max(0.0, 1.0 - rel))
                else:
                    scores.append(1.0 if prev_val == cur_val else 0.0)
        return float(np.mean(scores)) if scores else 1.0

    @staticmethod
    def _recommend_params(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总推荐参数：数值参数取中位数，分类参数取众数。"""
        if not history:
            return {}
        keys = history[0].keys()
        recommended: Dict[str, Any] = {}
        for key in keys:
            values = [h[key] for h in history if key in h]
            if not values:
                continue
            numeric = [v for v in values if isinstance(v, (int, float))]
            if len(numeric) == len(values):
                recommended[key] = float(np.median(numeric))
                # 均线周期等取整更直观
                if all(isinstance(v, int) for v in values):
                    recommended[key] = int(round(recommended[key]))
            else:
                # 众数
                counts: Dict[Any, int] = {}
                for v in values:
                    counts[v] = counts.get(v, 0) + 1
                recommended[key] = max(counts, key=counts.get)
        return recommended

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _build_combos(param_grid: Dict[str, list]) -> List[Dict[str, Any]]:
        """生成全部参数组合列表。"""
        if not param_grid or any(len(v) == 0 for v in param_grid.values()):
            return []
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    @staticmethod
    def _validate_objective(objective: str) -> None:
        if objective not in _OBJECTIVE_HIGHER_BETTER:
            raise ValueError(
                f"不支持的 objective={objective!r}，"
                f"可选: {list(_OBJECTIVE_HIGHER_BETTER.keys())}"
            )

    @staticmethod
    def _validate_grid(param_grid: Dict[str, list]) -> None:
        if not param_grid or any(len(v) == 0 for v in param_grid.values()):
            raise ValueError("param_grid 不能为空或包含空列表")
