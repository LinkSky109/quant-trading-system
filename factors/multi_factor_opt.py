"""多因子模型优化模块。

实现 IC/IR 加权、因子正交化、去极值与合成得分计算，与 :mod:`factors.factor_engine`
无缝集成。

核心能力：
1. **去极值**：winsorize（分位数截断）与 MAD（median absolute deviation）两种方法。
2. **正交化**：施密特正交化（Gram-Schmidt）去除因子间线性相关性。
3. **IC/IR 加权**：基于历史 IC 序列计算因子权重，支持 IC 均值、IR、IC>0 胜率多种目标。
4. **合成得分**：将多个单因子按权重合成为统一评分，便于选股排序。

典型用法::

    from factors.factor_engine import FactorEngine
    from factors.multi_factor_opt import MultiFactorOptimizer

    engine = FactorEngine()
    opt = MultiFactorOptimizer(engine)

    # 构造因子面板 {symbol: DataFrame(含多个因子列)}
    panels = {sym: engine.calculate_factors(df, sym) for sym, df in klines.items()}

    # 合成得分（含去极值+正交化+IC加权）
    score = opt.build_composite_score(
        panels, forward_returns,
        orthogonalize=True, winsorize=True, weight_method="ir"
    )
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _extract_factor_matrix(
    factor_panels: Dict[str, pd.DataFrame],
    factor_name: str,
) -> pd.DataFrame:
    """从多标的因子面板中提取单因子横截面矩阵（日期×标的）。"""
    series: Dict[str, pd.Series] = {}
    for sym, df in factor_panels.items():
        if df is None or df.empty or factor_name not in df.columns:
            continue
        s = df[factor_name].copy()
        s.name = sym
        series[sym] = s
    if not series:
        return pd.DataFrame()
    mat = pd.concat(series.values(), axis=1)
    mat.columns = list(series.keys())
    return mat.sort_index()


def _extract_close_matrix(
    factor_panels: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """从多标的因子面板中提取收盘价矩阵（日期×标的）。"""
    series: Dict[str, pd.Series] = {}
    for sym, df in factor_panels.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        s = df["close"].copy()
        s.name = sym
        series[sym] = s
    if not series:
        return pd.DataFrame()
    mat = pd.concat(series.values(), axis=1)
    mat.columns = list(series.keys())
    return mat.sort_index()


class MultiFactorOptimizer:
    """多因子模型优化器。

    支持去极值、正交化、IC/IR 加权与合成得分计算。
    所有方法均为纯函数式（接收 DataFrame 返回 DataFrame），便于在 pipeline 中链式调用。

    Args:
        factor_engine: 可选的 FactorEngine 实例，用于获取因子列表与元数据。
    """

    def __init__(self, factor_engine: Optional[Any] = None) -> None:
        self.factor_engine = factor_engine

    # ------------------------------------------------------------------ #
    # 去极值
    # ------------------------------------------------------------------ #
    @staticmethod
    def winsorize(
        df: pd.DataFrame,
        lower: float = 0.01,
        upper: float = 0.99,
        axis: int = 1,
    ) -> pd.DataFrame:
        """分位数去极值（截面 winsorize）。

        对每一行（每个交易日），将因子值截断到 ``[q_lower, q_upper]`` 分位数之间。

        Args:
            df: 因子值矩阵（index=日期，columns=标的）。
            lower: 下分位数，默认 1%。
            upper: 上分位数，默认 99%。
            axis: 0 为时序去极值（按列），1 为截面去极值（按行）；默认 1（截面）。

        Returns:
            截断后的 DataFrame。
        """
        if df.empty:
            return df.copy()
        out = df.copy()
        if axis == 1:
            for dt in out.index:
                row = out.loc[dt]
                mask = row.notna()
                if int(mask.sum()) < 3:
                    continue
                valid = row[mask]
                lo = float(valid.quantile(lower))
                hi = float(valid.quantile(upper))
                out.loc[dt] = row.clip(lo, hi)
        else:
            for col in out.columns:
                lo = float(out[col].quantile(lower))
                hi = float(out[col].quantile(upper))
                out[col] = out[col].clip(lo, hi)
        return out

    @staticmethod
    def mad_trim(
        df: pd.DataFrame,
        n: float = 3.0,
        axis: int = 1,
    ) -> pd.DataFrame:
        """MAD（Median Absolute Deviation）去极值。

        对每一行截面数据：
            median = median(row)
            mad = median(|row - median|)
            截断范围 = [median - n * 1.4826 * mad, median + n * 1.4826 * mad]

        1.4826 是常数因子，使 MAD 与标准差在正态分布下一致。

        Args:
            df: 因子值矩阵。
            n: MAD 倍数，默认 3。
            axis: 1 为截面（默认），0 为时序。

        Returns:
            截断后的 DataFrame。
        """
        if df.empty:
            return df.copy()
        out = df.copy()
        if axis == 1:
            for dt in out.index:
                row = out.loc[dt]
                mask = row.notna()
                if int(mask.sum()) < 3:
                    continue
                valid = row[mask].astype(float)
                median = float(valid.median())
                mad = float(np.median(np.abs(valid - median)))
                cutoff = n * 1.4826 * mad
                lo, hi = median - cutoff, median + cutoff
                out.loc[dt] = row.clip(lo, hi)
        else:
            for col in out.columns:
                median = float(out[col].median())
                mad = float(np.median(np.abs(out[col] - median)))
                cutoff = n * 1.4826 * mad
                out[col] = out[col].clip(median - cutoff, median + cutoff)
        return out

    # ------------------------------------------------------------------ #
    # 正交化
    # ------------------------------------------------------------------ #
    @staticmethod
    def orthogonalize(
        factor_df: pd.DataFrame,
        method: str = "schmidt",
    ) -> pd.DataFrame:
        """因子正交化：去除因子间相关性。

        当前支持 **施密特正交化（Gram-Schmidt）**：按列顺序依次将每个因子对其前面
        所有因子做线性回归，取残差作为正交化后的因子值。

        Args:
            factor_df: 因子值矩阵（index=日期，columns=因子名）。
            method: ``"schmidt"`` 为施密特正交化（当前唯一实现）。

        Returns:
            正交化后的因子矩阵（列名不变）。
        """
        if factor_df.empty or factor_df.shape[1] <= 1:
            return factor_df.copy()
        if method != "schmidt":
            raise ValueError(f"正交化方法只支持 schmidt，收到: {method!r}")

        out = pd.DataFrame(index=factor_df.index, columns=factor_df.columns, dtype=float)
        cols = list(factor_df.columns)
        for i, col in enumerate(cols):
            y = factor_df[col].astype(float)
            if i == 0:
                out[col] = y
                continue
            # 用前面已正交化的因子对当前因子做回归
            X = out[cols[:i]].dropna()
            if X.empty or X.shape[0] < 3:
                out[col] = y
                continue
            common_idx = y.dropna().index.intersection(X.index)
            if len(common_idx) < 3:
                out[col] = y
                continue
            yc = y.loc[common_idx].astype(float).values
            Xc = X.loc[common_idx].astype(float).values
            # 加入常数项
            Xc = np.column_stack([np.ones(Xc.shape[0]), Xc])
            # 最小二乘
            beta, _, _, _ = np.linalg.lstsq(Xc, yc, rcond=None)
            pred = Xc @ beta
            resid = pd.Series(np.nan, index=factor_df.index)
            resid.loc[common_idx] = yc - pred
            out[col] = resid
        return out

    # ------------------------------------------------------------------ #
    # IC/IR 分析
    # ------------------------------------------------------------------ #
    @staticmethod
    def _calc_ic_series(
        factor_mat: pd.DataFrame,
        forward_ret: pd.DataFrame,
    ) -> pd.Series:
        """计算单个因子横截面 IC（Spearman 秩相关）时间序列。"""
        ic_values: List[float] = []
        ic_dates: List[Any] = []
        common_idx = factor_mat.index.intersection(forward_ret.index)
        for dt in common_idx:
            f_row = factor_mat.loc[dt]
            r_row = forward_ret.loc[dt]
            mask = f_row.notna() & r_row.notna()
            if int(mask.sum()) < 3:
                continue
            fv = f_row[mask].values.astype(float)
            rv = r_row[mask].values.astype(float)
            if np.std(fv) == 0 or np.std(rv) == 0:
                continue
            ic, _ = stats.spearmanr(fv, rv)
            if np.isnan(ic):
                continue
            ic_values.append(float(ic))
            ic_dates.append(dt)
        return pd.Series(ic_values, index=pd.Index(ic_dates, name="date"), name="ic")

    def calc_ic_ir(
        self,
        factor_panels: Dict[str, pd.DataFrame],
        factor_names: List[str],
        forward_days: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """计算多个因子的 IC/IR 统计量。

        Args:
            factor_panels: ``{symbol: DataFrame}``，每个 DataFrame 含因子列与 close 列。
            factor_names: 要分析的因子名列表。
            forward_days: 未来收益天数，默认 5。

        Returns:
            ``{factor_name: {"ic_mean": ..., "ic_std": ..., "ir": ..., "ic_win_rate": ...}}``
        """
        closes = _extract_close_matrix(factor_panels)
        if closes.empty:
            return {}
        fwd = closes.shift(-forward_days) / closes - 1.0

        results: Dict[str, Dict[str, float]] = {}
        for name in factor_names:
            fmat = _extract_factor_matrix(factor_panels, name)
            ic_series = self._calc_ic_series(fmat, fwd)
            if ic_series.empty:
                results[name] = {
                    "ic_mean": float("nan"), "ic_std": float("nan"),
                    "ir": float("nan"), "ic_win_rate": float("nan"),
                    "n_periods": 0,
                }
                continue
            ic_mean = float(ic_series.mean())
            ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else float("nan")
            ir = ic_mean / ic_std if ic_std and not np.isnan(ic_std) else float("nan")
            win_rate = float((ic_series > 0).mean())
            results[name] = {
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ir": ir,
                "ic_win_rate": win_rate,
                "n_periods": int(len(ic_series)),
            }
        return results

    def ic_weighted_weights(
        self,
        factor_panels: Dict[str, pd.DataFrame],
        factor_names: List[str],
        forward_days: int = 5,
        method: str = "ir",
    ) -> Dict[str, float]:
        """基于 IC/IR 计算因子加权权重。

        Args:
            method:
                - ``"ic"``: 按 IC 均值绝对值加权。
                - ``"ir"``: 按 IR（IC 均值 / IC 标准差）加权（默认）。
                - ``"win_rate"``: 按 IC>0 胜率加权。
                - ``"equal"``: 等权。

        Returns:
            ``{factor_name: weight}``，权重和为 1。
        """
        stats_dict = self.calc_ic_ir(factor_panels, factor_names, forward_days)
        if not stats_dict:
            return {name: 1.0 / len(factor_names) for name in factor_names}

        scores: Dict[str, float] = {}
        for name in factor_names:
            s = stats_dict.get(name, {})
            if method == "ic":
                scores[name] = abs(s.get("ic_mean", 0.0))
            elif method == "ir":
                scores[name] = max(0.0, s.get("ir", 0.0))
            elif method == "win_rate":
                scores[name] = max(0.0, s.get("ic_win_rate", 0.0) - 0.5)
            elif method == "equal":
                scores[name] = 1.0
            else:
                raise ValueError(f"未知加权方法: {method!r}")

        total = sum(scores.values())
        if total == 0:
            return {name: 1.0 / len(factor_names) for name in factor_names}
        return {name: scores[name] / total for name in factor_names}

    # ------------------------------------------------------------------ #
    # z-score 标准化
    # ------------------------------------------------------------------ #
    @staticmethod
    def zscore(df: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
        """截面 z-score 标准化（默认按行：每个交易日所有标的标准化）。

        Returns:
            标准化后的 DataFrame（均值≈0，标准差≈1）。
        """
        if df.empty:
            return df.copy()
        out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        if axis == 1:
            for dt in df.index:
                row = df.loc[dt].dropna().astype(float)
                if len(row) < 2:
                    out.loc[dt] = df.loc[dt]
                    continue
                mu = float(row.mean())
                sigma = float(row.std(ddof=0))
                if sigma == 0 or np.isnan(sigma):
                    out.loc[dt] = 0.0
                else:
                    out.loc[dt] = (df.loc[dt] - mu) / sigma
        else:
            for col in df.columns:
                s = df[col].dropna().astype(float)
                if len(s) < 2:
                    out[col] = df[col]
                    continue
                mu = float(s.mean())
                sigma = float(s.std(ddof=0))
                if sigma == 0 or np.isnan(sigma):
                    out[col] = 0.0
                else:
                    out[col] = (df[col] - mu) / sigma
        return out

    # ------------------------------------------------------------------ #
    # 方向调整（与 factor_engine 的 direction 对齐）
    # ------------------------------------------------------------------ #
    def adjust_direction(
        self,
        factor_df: pd.DataFrame,
        directions: Dict[str, int],
    ) -> pd.DataFrame:
        """按因子方向调整符号（direction=-1 的因子取反）。

        Args:
            factor_df: 因子值矩阵（columns=因子名）。
            directions: ``{factor_name: +1/-1}``。

        Returns:
            调整后的 DataFrame。
        """
        out = factor_df.copy()
        for col in out.columns:
            if directions.get(col, 1) == -1:
                out[col] = -out[col]
        return out

    # ------------------------------------------------------------------ #
    # 合成得分
    # ------------------------------------------------------------------ #
    def build_composite_score(
        self,
        factor_panels: Dict[str, pd.DataFrame],
        factor_names: List[str],
        weights: Optional[Dict[str, float]] = None,
        forward_days: int = 5,
        weight_method: str = "ir",
        orthogonalize_factors: bool = False,
        winsorize_factors: bool = True,
        mad_trim_factors: bool = False,
        standardize: bool = True,
        directions: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """构建多因子合成得分。

        Pipeline:
            1. 提取各因子横截面矩阵。
            2. （可选）去极值（winsorize / mad_trim）。
            3. （可选）正交化。
            4. （可选）按 direction 调整符号。
            5. （可选）截面 z-score 标准化。
            6. 按权重加权求和。

        Args:
            factor_panels: ``{symbol: DataFrame}``。
            factor_names: 参与合成的因子名。
            weights: 手动指定权重；为 None 时按 ``weight_method`` 自动计算。
            forward_days: 用于 IC/IR 加权的未来收益天数。
            weight_method: ``"ic"`` / ``"ir"`` / ``"win_rate"`` / ``"equal"``。
            orthogonalize_factors: 是否先做正交化。
            winsorize_factors: 是否先做 winsorize 去极值。
            mad_trim_factors: 是否先做 MAD 去极值（与 winsorize 互斥时优先 winsorize）。
            standardize: 是否做截面 z-score。
            directions: 因子方向字典；为 None 时尝试从 factor_engine 获取。

        Returns:
            DataFrame，index=日期，columns=[各因子列 + "composite_score"]。
        """
        if not factor_names:
            raise ValueError("factor_names 不能为空")

        # 提取各因子矩阵
        mats: Dict[str, pd.DataFrame] = {}
        for name in factor_names:
            mat = _extract_factor_matrix(factor_panels, name)
            if mat.empty:
                logger.warning("因子 %s 无法提取有效矩阵，跳过", name)
                continue
            mats[name] = mat

        valid_names = list(mats.keys())
        if not valid_names:
            return pd.DataFrame()

        # 去极值（每个因子分别处理）
        for name in valid_names:
            if winsorize_factors:
                mats[name] = self.winsorize(mats[name])
            elif mad_trim_factors:
                mats[name] = self.mad_trim(mats[name])

        # 合并成 日期×(因子×标的) 长格式不方便；这里对每个因子标准化后保留
        # 正交化需要在同一截面上对多个因子值做，因此先把每个因子拉成 日期×标的
        # 然后对每个日期，把各因子值拼成向量再做正交化
        if orthogonalize_factors and len(valid_names) > 1:
            # 对齐日期与标的
            common_idx = mats[valid_names[0]].index
            common_cols = set(mats[valid_names[0]].columns)
            for name in valid_names[1:]:
                common_idx = common_idx.intersection(mats[name].index)
                common_cols = common_cols.intersection(set(mats[name].columns))
            common_cols = sorted(common_cols)
            if len(common_idx) < 3 or len(common_cols) < 2:
                logger.warning("正交化条件不足，跳过")
            else:
                aligned = {name: mats[name].loc[common_idx, common_cols] for name in valid_names}
                for dt in common_idx:
                    row_dict = {name: aligned[name].loc[dt].values.astype(float) for name in valid_names}
                    fdf = pd.DataFrame(row_dict)
                    ortho = self.orthogonalize(fdf)
                    for name in valid_names:
                        mats[name].loc[dt, common_cols] = ortho[name].values

        # 方向调整
        if directions is None and self.factor_engine is not None:
            try:
                registry = self.factor_engine._factor_registry
                directions = {f["name"]: f.get("direction", 1) for f in registry}
            except Exception:
                directions = {}
        if directions:
            for name in valid_names:
                if directions.get(name, 1) == -1:
                    mats[name] = -mats[name]

        # 标准化
        if standardize:
            for name in valid_names:
                mats[name] = self.zscore(mats[name])

        # 权重
        if weights is None:
            weights = self.ic_weighted_weights(
                factor_panels, valid_names, forward_days, weight_method
            )
        # 归一化
        wsum = sum(weights.get(n, 0.0) for n in valid_names)
        if wsum == 0:
            weights = {n: 1.0 / len(valid_names) for n in valid_names}
        else:
            weights = {n: weights.get(n, 0.0) / wsum for n in valid_names}

        # 加权合成：所有因子矩阵对齐后逐元素加权求和
        common_idx = mats[valid_names[0]].index
        common_cols = set(mats[valid_names[0]].columns)
        for name in valid_names[1:]:
            common_idx = common_idx.intersection(mats[name].index)
            common_cols = common_cols.intersection(set(mats[name].columns))
        common_cols = sorted(common_cols)

        composite = pd.DataFrame(0.0, index=common_idx, columns=common_cols)
        for name in valid_names:
            sub = mats[name].loc[common_idx, common_cols]
            composite += sub * weights[name]

        # 合并输出
        result = pd.DataFrame(index=common_idx)
        for name in valid_names:
            for col in common_cols:
                result[f"{name}_{col}"] = mats[name].loc[common_idx, col]
        result["composite_score"] = composite.mean(axis=1)
        return result

    # ------------------------------------------------------------------ #
    # 与 factor_engine 集成：便捷接口
    # ------------------------------------------------------------------ #
    def optimize_factors(
        self,
        factor_panels: Dict[str, pd.DataFrame],
        factor_names: Optional[List[str]] = None,
        forward_days: int = 5,
        weight_method: str = "ir",
    ) -> Dict[str, Any]:
        """一键运行多因子优化全流程，返回完整报告。

        Returns:
            {
                "weights": {factor: weight},
                "ic_ir_stats": {factor: {ic_mean, ic_std, ir, ic_win_rate}},
                "composite_score": DataFrame,
                "factor_count": int,
            }
        """
        if factor_names is None and self.factor_engine is not None:
            factor_names = self.factor_engine.factor_names
        if not factor_names:
            raise ValueError("factor_names 不能为空")

        ic_stats = self.calc_ic_ir(factor_panels, factor_names, forward_days)
        weights = self.ic_weighted_weights(
            factor_panels, factor_names, forward_days, weight_method
        )
        score = self.build_composite_score(
            factor_panels,
            factor_names,
            weights=weights,
            forward_days=forward_days,
            weight_method=weight_method,
            winsorize_factors=True,
            standardize=True,
        )
        return {
            "weights": weights,
            "ic_ir_stats": ic_stats,
            "composite_score": score,
            "factor_count": len(factor_names),
        }
