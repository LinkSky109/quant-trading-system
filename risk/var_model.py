"""风险模型：VaR / CVaR / 压力测试 / 组合VaR分解。

实现四种核心风险度量：

1. **历史模拟法 VaR/CVaR**：基于历史收益率经验分布，直接取分位数与尾部均值。
2. **方差-协方差法（参数法）VaR/CVaR**：假设收益率正态，用协方差矩阵与 z 分位数
   推导组合波动率，再乘以 z 得分（CVaR 用标准正态的尾部条件期望）。
3. **压力测试**：把预设极端情景（2008 金融危机、2015 股灾、2020 疫情、
   单日涨跌）直接施加到组合持仓上，测算损失。
4. **组合 VaR 分解（Euler 分解）**：把组合 VaR 拆成各标的的边际贡献，
   便于识别组合的主要风险来源。

约定：
- 收益率使用简单收益率 ``r_t = p_t / p_{t-1} - 1``。
- VaR/CVaR 一律返回**正数**，表示“损失金额的相对比例”（如 0.03 表示 3% 损失）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# 预设压力测试情景：key -> (描述, 组合整体冲击比例)
# 负号表示下跌。累计型情景直接把组合市值一次性乘以 (1 + shock)。
STRESS_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "2008_crisis": {
        "description": "2008 全球金融危机：组合累计下跌 50%",
        "shock": -0.50,
        "horizon": "cumulative",
    },
    "2015_crash": {
        "description": "2015 A股股灾：组合累计下跌 45%",
        "shock": -0.45,
        "horizon": "cumulative",
    },
    "2020_covid": {
        "description": "2020 新冠疫情冲击：单日下跌 10%",
        "shock": -0.10,
        "horizon": "single_day",
    },
    "single_day_drop_10": {
        "description": "极端单日下跌 10%",
        "shock": -0.10,
        "horizon": "single_day",
    },
    "single_day_rise_5": {
        "description": "极端单日上涨 5%（正向情景）",
        "shock": 0.05,
        "horizon": "single_day",
    },
}


class VaRModel:
    """VaR / CVaR / 压力测试模型。

    所有方法均为无状态纯函数式实现，可在 API 层与回测层复用。
    """

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """从价格序列计算日简单收益率。

        Args:
            prices: 价格序列（按时间升序，index 为日期）。

        Returns:
            收益率序列，长度比 prices 少 1（首值为 NaN 已丢弃）。
        """
        s = pd.Series(prices).astype(float)
        rets = s.pct_change().dropna()
        return rets

    @staticmethod
    def get_stats(returns: pd.Series) -> Dict[str, float]:
        """计算收益率序列的统计特征。

        Returns:
            含 mean / volatility / skewness / kurtosis / max_drawdown 的字典。
        """
        r = pd.Series(returns).dropna().astype(float)
        if len(r) == 0:
            return {
                "mean": 0.0, "volatility": 0.0, "skewness": 0.0,
                "kurtosis": 0.0, "max_drawdown": 0.0, "count": 0,
            }
        # 累计净值曲线用于最大回撤
        equity = (1.0 + r).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return {
            "mean": float(r.mean()),
            "volatility": float(r.std(ddof=1)),
            "skewness": float(r.skew()),
            "kurtosis": float(r.kurt()),
            "max_drawdown": float(drawdown.min()),  # 负值
            "count": int(len(r)),
        }

    # ------------------------------------------------------------------
    # 历史模拟法
    # ------------------------------------------------------------------
    @staticmethod
    def historical_var(returns: pd.Series | np.ndarray, confidence: float = 0.95) -> float:
        """历史模拟法 VaR。

        取收益率分布的 ``(1 - confidence)`` 分位数作为损失阈值，
        返回正数表示“在 confidence 置信度下，单日最大损失不超过该比例”。

        Args:
            returns: 历史日收益率序列。
            confidence: 置信水平，如 0.95 / 0.99。

        Returns:
            VaR（正数，损失比例）。
        """
        r = np.asarray(pd.Series(returns).dropna(), dtype=float)
        if len(r) == 0:
            return 0.0
        tail_level = (1.0 - confidence) * 100.0
        worst_quantile = float(np.percentile(r, tail_level))
        # worst_quantile 通常为负（如 -0.03），VaR 取相反数变成正损失
        return max(0.0, -worst_quantile)

    @classmethod
    def historical_cvar(
        cls, returns: pd.Series | np.ndarray, confidence: float = 0.95
    ) -> float:
        """历史模拟法 CVaR（条件 VaR / Expected Shortfall）。

        定义：损失超过 VaR 的尾部情形下的平均损失。
        数学上 ``CVaR >= VaR``（尾部均值必然不差于尾部最差点）。
        """
        r = np.asarray(pd.Series(returns).dropna(), dtype=float)
        if len(r) == 0:
            return 0.0
        tail_level = (1.0 - confidence) * 100.0
        var_threshold = float(np.percentile(r, tail_level))
        tail = r[r <= var_threshold]
        if len(tail) == 0:
            return cls.historical_var(r, confidence)
        return max(0.0, -float(tail.mean()))

    # ------------------------------------------------------------------
    # 方差-协方差法（参数法）
    # ------------------------------------------------------------------
    @staticmethod
    def _portfolio_metrics(
        weights: Sequence[float], returns_matrix: pd.DataFrame
    ) -> tuple[float, float]:
        """计算组合期望收益与波动率。

        Args:
            weights: 各标的权重。
            returns_matrix: 行=日期，列=标的 的收益率矩阵。

        Returns:
            (portfolio_mean, portfolio_volatility)。
        """
        w = np.asarray(weights, dtype=float)
        w = w / w.sum() if w.sum() != 0 else w
        mu = returns_matrix.mean().to_numpy(dtype=float)
        cov = returns_matrix.cov().to_numpy(dtype=float)
        port_mean = float(w @ mu)
        port_vol = float(np.sqrt(w @ cov @ w))
        return port_mean, port_vol

    @classmethod
    def parametric_var(
        cls,
        weights: Sequence[float],
        returns_matrix: pd.DataFrame,
        confidence: float = 0.95,
    ) -> float:
        """参数法 VaR：组合波动率 × z 分位数 - 期望收益。

        VaR_loss = z_{conf} * sigma_p - mu_p
        """
        port_mean, port_vol = cls._portfolio_metrics(weights, returns_matrix)
        z = float(sp_stats.norm.ppf(confidence))
        var_loss = z * port_vol - port_mean
        return max(0.0, var_loss)

    @classmethod
    def parametric_cvar(
        cls,
        weights: Sequence[float],
        returns_matrix: pd.DataFrame,
        confidence: float = 0.95,
    ) -> float:
        """参数法 CVaR：正态分布下的尾部条件期望损失。

        CVaR = sigma_p * phi(z_{conf}) / (1 - confidence) - mu_p
        其中 phi 为标准正态概率密度函数。
        """
        port_mean, port_vol = cls._portfolio_metrics(weights, returns_matrix)
        z = float(sp_stats.norm.ppf(confidence))
        tail_prob = 1.0 - confidence
        cvar_loss = port_vol * float(sp_stats.norm.pdf(z)) / tail_prob - port_mean
        return max(0.0, cvar_loss)

    # ------------------------------------------------------------------
    # 组合 VaR 分解（Euler 分解）
    # ------------------------------------------------------------------
    @classmethod
    def component_var(
        cls,
        weights: Sequence[float],
        returns_matrix: pd.DataFrame,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """组合 VaR 的 Euler 分解。

        边际 VaR（mVaR）= 组合 VaR 对该标的权重的偏导：
            mVaR_i = z * (Σ w)_i / sigma_p - mu_i
        成分 VaR（component VaR）= w_i * mVaR_i
        满足：sum(component_var) == 组合 VaR（欧拉齐次定理）。

        Returns:
            {
              "total_var": 组合 VaR,
              "items": [{symbol, weight, marginal_var, component_var,
                         contribution_pct}, ...]
            }
        """
        w = np.asarray(weights, dtype=float)
        w = w / w.sum() if w.sum() != 0 else w
        mu = returns_matrix.mean().to_numpy(dtype=float)
        cov = returns_matrix.cov().to_numpy(dtype=float)
        port_mean = float(w @ mu)
        port_vol = float(np.sqrt(w @ cov @ w))
        z = float(sp_stats.norm.ppf(confidence))

        symbols = list(returns_matrix.columns)
        # 边际 VaR：组合 VaR 对 w_i 的偏导
        if port_vol > 0:
            marginal = z * (cov @ w) / port_vol - mu
        else:
            marginal = np.zeros_like(mu)
        component = w * marginal
        total_var = float(component.sum())
        total_var = max(0.0, total_var)

        items: List[Dict[str, Any]] = []
        abs_sum = float(np.sum(np.abs(component)))
        for sym, wi, mvar, cvar in zip(symbols, w, marginal, component):
            items.append({
                "symbol": sym,
                "weight": float(wi),
                "marginal_var": float(mvar),
                "component_var": float(cvar),
                "contribution_pct": (float(abs(cvar)) / abs_sum) if abs_sum > 0 else 0.0,
            })
        return {
            "total_var": total_var,
            "portfolio_mean": port_mean,
            "portfolio_volatility": port_vol,
            "items": items,
        }

    # ------------------------------------------------------------------
    # 压力测试
    # ------------------------------------------------------------------
    @staticmethod
    def list_scenarios() -> List[Dict[str, Any]]:
        """返回所有预设压力测试场景。"""
        return [
            {"key": key, **meta}
            for key, meta in STRESS_SCENARIOS.items()
        ]

    @classmethod
    def stress_test(
        cls,
        weights: Sequence[float],
        prices: Dict[str, float] | pd.Series,
        scenario: str,
        portfolio_value: float = 1.0,
    ) -> Dict[str, Any]:
        """对组合施加预设压力情景。

        Args:
            weights: 各标的权重（与 prices 的 key 对齐）。
            prices: {symbol: 当前价格} 或 symbol->price 的 Series。
            scenario: 情景 key，见 :meth:`list_scenarios`。
            portfolio_value: 组合总市值（用于把损失比例换算成金额）。

        Returns:
            {scenario, description, portfolio_loss, loss_pct, positions_impact}。
        """
        if scenario not in STRESS_SCENARIOS:
            raise ValueError(
                f"未知压力情景: {scenario}，可选: {list(STRESS_SCENARIOS)}"
            )
        meta = STRESS_SCENARIOS[scenario]
        shock = float(meta["shock"])

        symbols = list(prices.keys()) if hasattr(prices, "keys") else list(prices.index)
        w = np.asarray(weights, dtype=float)
        if w.sum() != 0:
            w = w / w.sum()
        prices_arr = np.asarray([float(prices[s]) for s in symbols], dtype=float)

        positions_impact: List[Dict[str, Any]] = []
        for sym, wi, px in zip(symbols, w, prices_arr):
            after = px * (1.0 + shock)
            pnl = (after - px) * wi * portfolio_value
            positions_impact.append({
                "symbol": sym,
                "weight": float(wi),
                "price_before": float(px),
                "price_after": float(after),
                "pnl": float(pnl),
            })

        loss_pct = shock  # 等权统一冲击下组合整体涨跌比例
        portfolio_loss = loss_pct * portfolio_value
        return {
            "scenario": scenario,
            "description": meta["description"],
            "horizon": meta["horizon"],
            "shock": shock,
            "portfolio_loss": float(portfolio_loss),
            "loss_pct": float(loss_pct),
            "positions_impact": positions_impact,
        }

    # ------------------------------------------------------------------
    # 实时监控
    # ------------------------------------------------------------------
    @staticmethod
    def check_var_threshold(current_var: float, threshold: float) -> bool:
        """检查当前 VaR 是否超过阈值。

        Returns:
            True 表示已超限（应触发告警）。
        """
        return float(current_var) > float(threshold)
