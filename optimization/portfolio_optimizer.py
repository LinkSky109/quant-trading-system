"""专业组合权重优化器。

在给定一组标的的历史行情数据上，按指定目标求解最优资金权重：

- ``equal_weight``   等权（基准），1/N。
- ``min_variance``   最小方差，最小化组合方差 ``w^T Σ w``。
- ``risk_parity``    风险平价，使各标的对组合总波动的风险贡献相等。
- ``mean_variance``  均值方差，最大化夏普比率 ``(μ^T w - rf) / sqrt(w^T Σ w)``。

约束：long-only（权重非负），单标的权重上限 ``max_weight``，权重和为 1。

收益率口径：用收盘价日收益率 ``pct_change()`` 估计；
预期收益为日收益均值年化（×252），协方差为日收益协方差年化（×252）。

优先使用 :func:`scipy.optimize.minimize`（SLSQP）求解；
当 scipy 不可用时自动降级为纯 numpy 迭代/解析方法，并记录 warning。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:  # pragma: no cover - 取决于运行环境
    from scipy.optimize import minimize as _sp_minimize
    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001 - scipy 缺失时降级
    _sp_minimize = None
    _HAVE_SCIPY = False

#: 支持的优化方法
ALLOWED_METHODS = ("equal_weight", "min_variance", "risk_parity", "mean_variance")

#: 默认年交易日数
DEFAULT_TRADING_DAYS = 252


@dataclass
class OptimizeResult:
    """组合权重优化结果。

    Attributes:
        weights: 各标的权重 ``{symbol: weight}``，和为 1。
        expected_return: 组合预期年化收益。
        expected_volatility: 组合预期年化波动率（标准差）。
        sharpe: 夏普比率 ``(expected_return - rf) / expected_volatility``。
        method: 使用的优化方法名。
    """

    weights: Dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe: float = 0.0
    method: str = "equal_weight"

    def as_dict(self) -> Dict[str, float]:
        """序列化为普通 dict（便于 JSON 输出）。"""
        return {
            "weights": dict(self.weights),
            "expected_return": float(self.expected_return),
            "expected_volatility": float(self.expected_volatility),
            "sharpe": float(self.sharpe),
            "method": self.method,
        }


class PortfolioOptimizer:
    """专业组合权重优化器。

    Args:
        risk_free_rate: 无风险利率（年化），用于夏普比率计算，默认 0.02。
        max_weight: 单标的权重上限（long-only），默认 0.3。
            若 ``N * max_weight < 1``（问题不可行），会自动放宽到 ``1/N`` 并记录 warning。
        min_weight: 单标的权重下限，默认 0.0（long-only）。
        trading_days: 年交易日数，用于年化，默认 252。
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        max_weight: float = 0.3,
        min_weight: float = 0.0,
        trading_days: int = DEFAULT_TRADING_DAYS,
    ):
        self.risk_free_rate = float(risk_free_rate)
        self.max_weight = float(max_weight)
        self.min_weight = float(min_weight)
        self.trading_days = int(trading_days)

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def optimize(
        self,
        symbols: List[str],
        data: Dict[str, pd.DataFrame],
        method: str = "equal_weight",
    ) -> OptimizeResult:
        """按指定方法求解组合权重。

        Args:
            symbols: 参与优化的标的列表。
            data: 行情数据 ``{symbol: df}``，每个 df 需含 ``close`` 列（DatetimeIndex）。
            method: 优化方法，见 :data:`ALLOWED_METHODS`。

        Returns:
            :class:`OptimizeResult`。空输入或单标的也会安全返回。

        Raises:
            ValueError: ``method`` 非法。
        """
        if method not in ALLOWED_METHODS:
            raise ValueError(
                f"method 必须为 {ALLOWED_METHODS} 之一，收到: {method!r}"
            )

        # 过滤出有效标的
        valid_syms = [
            s for s in (symbols or [])
            if s in data and data[s] is not None and len(data[s]) > 1
        ]

        # 空输入
        if not valid_syms:
            return OptimizeResult(weights={}, method=method)

        # 估计年化收益向量 μ 与年化协方差 Σ
        mu, cov = self._estimate_mu_cov(valid_syms, data)

        # 单标的：权重 1.0（受 max_weight 约束时下方会做可行性处理）
        n = len(valid_syms)
        # 可行性：N*max_weight 必须 >= 1，否则放宽上限到 1/N
        eff_max = max(self.max_weight, 1.0 / n)
        eff_min = min(self.min_weight, eff_max)

        if n == 1 or method == "equal_weight":
            w_arr = np.full(n, 1.0 / n)
        elif method == "min_variance":
            w_arr = self._solve_min_variance(mu, cov, eff_min, eff_max)
        elif method == "risk_parity":
            w_arr = self._solve_risk_parity(mu, cov, eff_min, eff_max)
        elif method == "mean_variance":
            w_arr = self._solve_mean_variance(mu, cov, eff_min, eff_max)
        else:  # pragma: no cover - 上面已校验
            raise ValueError(f"未知 method: {method!r}")

        # 数值清洗：裁剪 + 归一化，保证和为 1、非负
        w_arr = np.clip(w_arr, eff_min, eff_max)
        w_sum = w_arr.sum()
        if w_sum <= 0:
            w_arr = np.full(n, 1.0 / n)
        else:
            w_arr = w_arr / w_arr.sum()
            # 归一化后可能略超上限，再裁一次并归一化
            w_arr = np.clip(w_arr, 0.0, self.max_weight)
            w_arr = w_arr / w_arr.sum()

        weights = {s: float(w_arr[i]) for i, s in enumerate(valid_syms)}
        port_ret = float(w_arr @ mu)
        port_var = float(w_arr @ cov @ w_arr)
        port_vol = float(np.sqrt(max(port_var, 0.0)))
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 1e-12 else 0.0

        return OptimizeResult(
            weights=weights,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe=float(sharpe),
            method=method,
        )

    # ------------------------------------------------------------------
    # 数据估计
    # ------------------------------------------------------------------

    def _estimate_mu_cov(
        self,
        symbols: List[str],
        data: Dict[str, pd.DataFrame],
    ) -> "tuple[np.ndarray, np.ndarray]":
        """由收盘价估计年化收益向量 μ 与年化协方差矩阵 Σ。

        各标的按日期取交集后计算日收益率，避免错位。
        """
        closes = {}
        for s in symbols:
            close = pd.to_numeric(data[s]["close"], errors="coerce").dropna()
            closes[s] = close
        price_df = pd.concat(closes, axis=1).dropna()
        if price_df.shape[0] < 2:
            # 数据不足：退化为零收益、单位方差（避免除零）
            n = len(symbols)
            return np.zeros(n), np.eye(n) * 1e-4

        daily_ret = price_df.pct_change().dropna()
        if len(daily_ret) < 2:
            n = len(symbols)
            return np.zeros(n), np.eye(n) * 1e-4

        mu = daily_ret.mean().to_numpy() * self.trading_days
        cov = daily_ret.cov().to_numpy() * self.trading_days
        # 保证对称正定（数值噪声）
        cov = (cov + cov.T) / 2.0
        return mu, cov

    # ------------------------------------------------------------------
    # 各优化方法
    # ------------------------------------------------------------------

    def _solve_min_variance(
        self, mu: np.ndarray, cov: np.ndarray,
        w_min: float, w_max: float,
    ) -> np.ndarray:
        """最小方差：min w'Σw s.t. sum w=1, w_min<=w<=w_max。"""
        n = len(mu)
        x0 = np.full(n, 1.0 / n)

        if _HAVE_SCIPY:
            return self._slsqp(
                objective=lambda w: float(w @ cov @ w),
                n=n, w_min=w_min, w_max=w_max, x0=x0,
            )

        # 降级：解析解（无约束最小方差）+ 裁剪
        logger.warning("scipy 不可用，min_variance 降级为解析解+裁剪")
        try:
            ones = np.ones(n)
            inv_cov = np.linalg.pinv(cov)
            w = inv_cov @ ones
            w = w / w.sum()
        except np.linalg.LinAlgError:
            w = x0
        return w

    def _solve_risk_parity(
        self, mu: np.ndarray, cov: np.ndarray,
        w_min: float, w_max: float,
    ) -> np.ndarray:
        """风险平价：最小化各标的风险贡献与均值的平方差之和。"""
        n = len(mu)

        def objective(w: np.ndarray) -> float:
            port_var = float(w @ cov @ w)
            sigma = np.sqrt(max(port_var, 1e-18))
            mrc = cov @ w / sigma           # 边际风险贡献
            rc = w * mrc                    # 风险贡献
            target = rc.mean()
            return float(np.sum((rc - target) ** 2))

        x0 = np.full(n, 1.0 / n)

        if _HAVE_SCIPY:
            return self._slsqp(objective=objective, n=n, w_min=w_min, w_max=w_max, x0=x0)

        # 降级：循环坐标下降（Spinu 算法的简化实现）
        logger.warning("scipy 不可用，risk_parity 降级为循环坐标下降")
        return self._risk_parity_cd(cov, w_min, w_max)

    def _solve_mean_variance(
        self, mu: np.ndarray, cov: np.ndarray,
        w_min: float, w_max: float,
    ) -> np.ndarray:
        """均值方差：最大化夏普 (μ'w - rf)/sqrt(w'Σw)。"""
        n = len(mu)

        def neg_sharpe(w: np.ndarray) -> float:
            ret = float(mu @ w) - self.risk_free_rate
            vol = float(np.sqrt(max(w @ cov @ w, 1e-18)))
            return -ret / vol

        # 多个初始点避免局部最优
        starts = [np.full(n, 1.0 / n)]
        # 波动率倒数起点
        inv_vol = 1.0 / np.sqrt(np.diag(cov))
        starts.append(inv_vol / inv_vol.sum())

        if _HAVE_SCIPY:
            best_w = starts[0]
            best_val = neg_sharpe(best_w)
            for x0 in starts:
                w = self._slsqp(objective=neg_sharpe, n=n,
                                w_min=w_min, w_max=w_max, x0=x0)
                val = neg_sharpe(w)
                if val < best_val:
                    best_val, best_w = val, w
            return best_w

        # 降级：网格搜索（随机 simplex 采样）
        logger.warning("scipy 不可用，mean_variance 降级为随机 simplex 搜索")
        rng = np.random.default_rng(42)
        best_w, best_val = starts[0], neg_sharpe(starts[0])
        for _ in range(20000):
            w = rng.dirichlet(np.ones(n))
            if w.max() > w_max or w.min() < w_min:
                continue
            val = neg_sharpe(w)
            if val < best_val:
                best_val, best_w = val, w
        return best_w

    # ------------------------------------------------------------------
    # 数值工具
    # ------------------------------------------------------------------

    def _slsqp(
        self,
        objective,
        n: int,
        w_min: float,
        w_max: float,
        x0: np.ndarray,
    ) -> np.ndarray:
        """用 scipy SLSQP 求解带 sum(w)=1 与 box 约束的优化问题。"""
        bounds = [(w_min, w_max)] * n
        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        res = _sp_minimize(
            objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10, "disp": False},
        )
        if not res.success:
            # 退化为等权，避免坏解
            logger.warning("SLSQP 未收敛: %s，退化为等权", res.message)
            return np.full(n, 1.0 / n)
        return np.asarray(res.x, dtype=float)

    @staticmethod
    def _risk_parity_cd(
        cov: np.ndarray, w_min: float, w_max: float,
        iters: int = 5000, tol: float = 1e-9,
    ) -> np.ndarray:
        """循环坐标下降求风险平价权重（无 scipy 降级路径）。

        固定其他权重，对单个标的沿风险贡献相等方向做牛顿步，迭代至收敛。
        """
        n = cov.shape[0]
        w = np.full(n, 1.0 / n)
        for _ in range(iters):
            prev = w.copy()
            for i in range(n):
                # RC_i = w_i * (Σw)_i / σ ; 目标 RC_i = RC_j
                # 固定其他 w，解析求 w_i 使 RC_i 与平均风险贡献匹配
                sigma = np.sqrt(max(w @ cov @ w, 1e-18))
                # 其他标的当前风险贡献之和 = σ - RC_i
                others_sum = sigma - w[i] * (cov[i] @ w) / sigma
                # 目标：RC_i = others_sum/(n-1)
                target_rc = others_sum / max(n - 1, 1)
                # RC_i = w_i * (cov[i,i]*w_i + cov[i,~i]·w~i)/sigma
                # 解二次方程: a w_i^2 + b w_i - target_rc*sigma = 0
                a = cov[i, i] / sigma
                b = (cov[i] @ w - cov[i, i] * w[i]) / sigma
                c = -target_rc * sigma
                disc = b * b - 4 * a * c
                if disc > 0 and abs(a) > 1e-18:
                    w_i_new = (-b + np.sqrt(disc)) / (2 * a)
                    w[i] = float(np.clip(w_i_new, w_min, w_max))
            if np.max(np.abs(w - prev)) < tol:
                break
        s = w.sum()
        return w / s if s > 0 else np.full(n, 1.0 / n)

    # ------------------------------------------------------------------
    # 风险贡献（供测试/诊断）
    # ------------------------------------------------------------------

    @staticmethod
    def risk_contributions(
        weights: Dict[str, float], cov: np.ndarray, symbols: List[str],
    ) -> Dict[str, float]:
        """计算各标的对组合总波动的风险贡献 RC_i。

        ``RC_i = w_i * (Σw)_i / sqrt(w'Σw)``。
        """
        w = np.array([weights[s] for s in symbols], dtype=float)
        sigma = np.sqrt(max(float(w @ cov @ w), 1e-18))
        rc = w * (cov @ w) / sigma
        return {s: float(rc[i]) for i, s in enumerate(symbols)}
