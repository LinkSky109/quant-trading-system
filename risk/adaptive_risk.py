"""自适应风险管理模块。

实现动态 VaR 窗口、基于波动率的风险预算调整、尾部风险检测与压力测试，与
:mod:`risk.risk_manager` 和 :mod:`risk.var_model` 无缝集成。

核心能力：
1. **动态 VaR 窗口**：根据市场波动率自适应调整历史窗口长度（波动率高时缩短窗口，
   波动率低时延长窗口）。
2. **风险预算动态调整**：基于当前波动率水平自动缩放风险预算上限。
3. **尾部风险检测**：CVaR / Expected Shortfall 实时监测，触发阈值告警。
4. **压力测试**：历史回溯 + 蒙特卡洛模拟，评估极端情景下的组合损失。

典型用法::

    from risk.adaptive_risk import AdaptiveRiskManager
    from risk.var_model import VaRModel

    arm = AdaptiveRiskManager(
        base_var_window=252,
        var_confidence=0.95,
        tail_risk_threshold=0.03,
    )

    # 动态 VaR
    var = arm.dynamic_var(portfolio_returns)

    # 风险预算调整
    budget = arm.adjust_risk_budget(market_volatility=0.15)

    # 尾部风险检测
    alert = arm.detect_tail_risk(portfolio_returns, weights, returns_matrix)

    # 压力测试
    stress = arm.run_stress_test(weights, prices, scenario="monte_carlo")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from risk.risk_manager import RiskManager
from risk.var_model import STRESS_SCENARIOS, VaRModel

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_BASE_VAR_WINDOW = 252
DEFAULT_MIN_VAR_WINDOW = 63
DEFAULT_MAX_VAR_WINDOW = 504
DEFAULT_VAR_CONFIDENCE = 0.95
DEFAULT_TAIL_RISK_THRESHOLD = 0.03
DEFAULT_RISK_BUDGET_BASE = 0.20
DEFAULT_VOLATILITY_TARGET = 0.15


@dataclass
class TailRiskAlert:
    """尾部风险告警记录。"""

    timestamp: str
    var: float
    cvar: float
    threshold: float
    triggered: bool
    severity: str = ""  # low / medium / high / critical
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "var": float(self.var),
            "cvar": float(self.cvar),
            "threshold": float(self.threshold),
            "triggered": bool(self.triggered),
            "severity": self.severity,
            "details": dict(self.details),
        }


@dataclass
class StressTestResult:
    """压力测试结果。"""

    scenario: str
    portfolio_loss_pct: float
    portfolio_loss_amount: float
    positions_impact: List[Dict[str, Any]] = field(default_factory=list)
    monte_carlo_distribution: Optional[Dict[str, float]] = None
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "portfolio_loss_pct": float(self.portfolio_loss_pct),
            "portfolio_loss_amount": float(self.portfolio_loss_amount),
            "positions_impact": list(self.positions_impact),
            "monte_carlo_distribution": self.monte_carlo_distribution,
            "description": self.description,
        }


class AdaptiveRiskManager:
    """自适应风险管理器。

    Args:
        base_var_window: 基础 VaR 历史窗口（默认 252 日）。
        min_var_window: 最小窗口（波动率高时）。
        max_var_window: 最大窗口（波动率低时）。
        var_confidence: VaR 置信水平。
        tail_risk_threshold: 尾部风险触发阈值（CVaR 比例）。
        risk_budget_base: 基础风险预算（如单标的上限）。
        volatility_target: 目标年化波动率。
        risk_manager: 可选的 RiskManager 实例，用于集成风控规则。
    """

    def __init__(
        self,
        base_var_window: int = DEFAULT_BASE_VAR_WINDOW,
        min_var_window: int = DEFAULT_MIN_VAR_WINDOW,
        max_var_window: int = DEFAULT_MAX_VAR_WINDOW,
        var_confidence: float = DEFAULT_VAR_CONFIDENCE,
        tail_risk_threshold: float = DEFAULT_TAIL_RISK_THRESHOLD,
        risk_budget_base: float = DEFAULT_RISK_BUDGET_BASE,
        volatility_target: float = DEFAULT_VOLATILITY_TARGET,
        risk_manager: Optional[RiskManager] = None,
    ):
        self.base_var_window = int(base_var_window)
        self.min_var_window = int(min_var_window)
        self.max_var_window = int(max_var_window)
        self.var_confidence = float(var_confidence)
        self.tail_risk_threshold = float(tail_risk_threshold)
        self.risk_budget_base = float(risk_budget_base)
        self.volatility_target = float(volatility_target)
        self.risk_manager = risk_manager

        # 历史告警记录
        self.tail_risk_history: List[TailRiskAlert] = []

    # ------------------------------------------------------------------ #
    # 动态 VaR 窗口
    # ------------------------------------------------------------------ #
    def calc_dynamic_window(self, returns: pd.Series | np.ndarray) -> int:
        """根据近期波动率自适应调整 VaR 历史窗口。

        逻辑：
            - 计算最近 ``base_var_window`` 期的滚动波动率（20 日标准差年化）。
            - 若当前波动率 > 历史中位数 × 1.5，认为市场处于高波动，缩短窗口。
            - 若当前波动率 < 历史中位数 × 0.7，认为市场处于低波动，延长窗口。

        Args:
            returns: 日收益率序列。

        Returns:
            建议的历史窗口长度。
        """
        r = pd.Series(returns).dropna().astype(float)
        if len(r) < self.min_var_window:
            return max(len(r), 30)

        # 20 日滚动标准差年化
        rolling_vol = r.rolling(window=20, min_periods=10).std() * np.sqrt(252)
        valid_vol = rolling_vol.dropna()
        if len(valid_vol) < 10:
            return self.base_var_window

        current_vol = float(valid_vol.iloc[-1])
        median_vol = float(valid_vol.median())

        if median_vol == 0 or np.isnan(median_vol):
            return self.base_var_window

        ratio = current_vol / median_vol
        if ratio > 1.5:
            # 高波动 -> 缩短窗口
            window = int(self.base_var_window / ratio)
        elif ratio < 0.7:
            # 低波动 -> 延长窗口
            window = int(self.base_var_window / ratio)
        else:
            window = self.base_var_window

        window = max(self.min_var_window, min(self.max_var_window, window))
        return int(window)

    def dynamic_var(
        self,
        returns: pd.Series | np.ndarray,
        weights: Optional[np.ndarray] = None,
        returns_matrix: Optional[pd.DataFrame] = None,
        method: str = "historical",
    ) -> Dict[str, Any]:
        """动态窗口 VaR 计算。

        Args:
            returns: 组合日收益率序列（单序列）。
            weights: 组合权重（与 returns_matrix 配合用于参数法）。
            returns_matrix: 各标的日收益率矩阵（行=日期，列=标的）。
            method: ``"historical"`` 或 ``"parametric"``。

        Returns:
            {
                "var": float,
                "cvar": float,
                "window": int,
                "method": str,
                "current_volatility": float,
                "confidence": float,
            }
        """
        r = pd.Series(returns).dropna().astype(float)
        window = self.calc_dynamic_window(r)
        # 取最近 window 期
        recent = r.tail(window)
        if len(recent) < 30:
            logger.warning("动态 VaR 有效样本不足: %d < 30", len(recent))

        current_vol = float(r.tail(20).std() * np.sqrt(252)) if len(r) >= 20 else 0.0

        if method == "historical":
            var_val = VaRModel.historical_var(recent, self.var_confidence)
            cvar_val = VaRModel.historical_cvar(recent, self.var_confidence)
        elif method == "parametric":
            if returns_matrix is not None and weights is not None:
                recent_mat = returns_matrix.tail(window)
                var_val = VaRModel.parametric_var(weights, recent_mat, self.var_confidence)
                cvar_val = VaRModel.parametric_cvar(weights, recent_mat, self.var_confidence)
            else:
                # 单序列退化为历史法
                var_val = VaRModel.historical_var(recent, self.var_confidence)
                cvar_val = VaRModel.historical_cvar(recent, self.var_confidence)
        else:
            raise ValueError(f"method 只支持 historical/parametric，收到: {method!r}")

        return {
            "var": float(var_val),
            "cvar": float(cvar_val),
            "window": int(window),
            "method": method,
            "current_volatility": round(current_vol, 6),
            "confidence": self.var_confidence,
            "sample_size": int(len(recent)),
        }

    # ------------------------------------------------------------------ #
    # 风险预算动态调整
    # ------------------------------------------------------------------ #
    def adjust_risk_budget(
        self,
        market_volatility: Optional[float] = None,
        portfolio_returns: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """基于市场波动率动态调整风险预算。

        逻辑：
            - 当实际波动率 > 目标波动率时，收缩风险预算（降低仓位上限）。
            - 当实际波动率 < 目标波动率时，适度扩张风险预算（提高仓位上限）。
            - 调整系数 = volatility_target / actual_volatility（带上下限）。

        Args:
            market_volatility: 当前市场年化波动率（直接传入优先）。
            portfolio_returns: 组合日收益率序列（用于估算波动率）。

        Returns:
            {
                "original_budget": float,
                "adjusted_budget": float,
                "volatility_target": float,
                "actual_volatility": float,
                "adjustment_factor": float,
            }
        """
        if market_volatility is not None:
            actual_vol = float(market_volatility)
        elif portfolio_returns is not None:
            r = pd.Series(portfolio_returns).dropna().astype(float)
            actual_vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) >= 20 else self.volatility_target
        else:
            actual_vol = self.volatility_target

        if actual_vol <= 0 or np.isnan(actual_vol):
            actual_vol = self.volatility_target

        # 调整系数：目标/实际，限制在 [0.5, 1.5]
        factor = self.volatility_target / actual_vol
        factor = max(0.5, min(1.5, factor))

        adjusted = self.risk_budget_base * factor
        # 进一步限制在合理范围
        adjusted = max(0.05, min(0.50, adjusted))

        return {
            "original_budget": round(self.risk_budget_base, 4),
            "adjusted_budget": round(adjusted, 4),
            "volatility_target": round(self.volatility_target, 4),
            "actual_volatility": round(actual_vol, 4),
            "adjustment_factor": round(factor, 4),
        }

    # ------------------------------------------------------------------ #
    # 尾部风险检测
    # ------------------------------------------------------------------ #
    def detect_tail_risk(
        self,
        portfolio_returns: pd.Series | np.ndarray,
        weights: Optional[np.ndarray] = None,
        returns_matrix: Optional[pd.DataFrame] = None,
        method: str = "historical",
    ) -> TailRiskAlert:
        """检测尾部风险是否超过阈值。

        基于 CVaR / ES（Expected Shortfall）进行判断：
            - CVaR >= threshold × 2   -> critical
            - CVaR >= threshold × 1.5 -> high
            - CVaR >= threshold       -> medium
            - 否则 -> low

        Args:
            portfolio_returns: 组合日收益率序列。
            weights: 权重（参数法需要）。
            returns_matrix: 收益率矩阵（参数法需要）。
            method: ``"historical"`` 或 ``"parametric"``。

        Returns:
            :class:`TailRiskAlert`。
        """
        var_info = self.dynamic_var(portfolio_returns, weights, returns_matrix, method)
        var_val = var_info["var"]
        cvar_val = var_info["cvar"]
        triggered = cvar_val >= self.tail_risk_threshold

        if triggered:
            ratio = cvar_val / self.tail_risk_threshold if self.tail_risk_threshold > 0 else 0
            if ratio >= 2.0:
                severity = "critical"
            elif ratio >= 1.5:
                severity = "high"
            else:
                severity = "medium"
        else:
            severity = "low"

        alert = TailRiskAlert(
            timestamp=pd.Timestamp.now().isoformat(),
            var=var_val,
            cvar=cvar_val,
            threshold=self.tail_risk_threshold,
            triggered=triggered,
            severity=severity,
            details={
                "window": var_info["window"],
                "method": method,
                "current_volatility": var_info.get("current_volatility"),
            },
        )
        self.tail_risk_history.append(alert)
        # 保持历史长度
        if len(self.tail_risk_history) > 1000:
            self.tail_risk_history = self.tail_risk_history[-500:]

        if triggered:
            logger.warning(
                "尾部风险告警 [%s]: VaR=%.4f CVaR=%.4f threshold=%.4f",
                severity, var_val, cvar_val, self.tail_risk_threshold,
            )

        return alert

    # ------------------------------------------------------------------ #
    # 压力测试
    # ------------------------------------------------------------------ #
    def stress_test_historical(
        self,
        weights: np.ndarray,
        returns_matrix: pd.DataFrame,
        scenario: str = "2008_crisis",
        portfolio_value: float = 1.0,
    ) -> StressTestResult:
        """历史情景压力测试。

        使用 :mod:`risk.var_model` 的预设情景直接施加冲击。

        Args:
            weights: 组合权重。
            returns_matrix: 收益率矩阵。
            scenario: 情景 key。
            portfolio_value: 组合市值。

        Returns:
            :class:`StressTestResult`。
        """
        if scenario not in STRESS_SCENARIOS:
            raise ValueError(f"未知压力情景: {scenario}")

        # 取最新价格（用最近一日净值近似）
        prices = {}
        for col in returns_matrix.columns:
            last_price = 100.0  # 基准价
            prices[col] = last_price

        result = VaRModel.stress_test(weights, prices, scenario, portfolio_value)
        loss_pct = abs(float(result["loss_pct"]))
        loss_amount = abs(float(result["portfolio_loss"]))
        return StressTestResult(
            scenario=scenario,
            portfolio_loss_pct=loss_pct,
            portfolio_loss_amount=loss_amount,
            positions_impact=result.get("positions_impact", []),
            description=STRESS_SCENARIOS[scenario]["description"],
        )

    def stress_test_monte_carlo(
        self,
        weights: np.ndarray,
        returns_matrix: pd.DataFrame,
        portfolio_value: float = 1.0,
        n_simulations: int = 10000,
        horizon_days: int = 1,
        confidence: float = 0.95,
    ) -> StressTestResult:
        """蒙特卡洛压力测试。

        基于历史收益率的均值与协方差，用多元正态分布模拟未来 ``horizon_days`` 天的
        组合收益，统计极端分位的损失分布。

        Args:
            weights: 组合权重。
            returns_matrix: 历史日收益率矩阵。
            portfolio_value: 组合市值。
            n_simulations: 模拟次数。
            horizon_days: 持有期天数。
            confidence: 置信水平。

        Returns:
            :class:`StressTestResult`，附带 ``monte_carlo_distribution`` 字段。
        """
        w = np.asarray(weights, dtype=float)
        w = w / w.sum() if w.sum() != 0 else w

        mu = returns_matrix.mean().to_numpy(dtype=float)
        cov = returns_matrix.cov().to_numpy(dtype=float)
        # 保证半正定
        cov = (cov + cov.T) / 2.0
        # 添加微小正则化
        eigvals = np.linalg.eigvalsh(cov)
        if np.min(eigvals) < 1e-12:
            cov += np.eye(cov.shape[0]) * 1e-8

        rng = np.random.default_rng(42)
        # 模拟 horizon_days 天的多资产收益
        daily_sims = rng.multivariate_normal(mu, cov, size=(n_simulations, horizon_days))
        # 组合日收益 = 权重 · 资产日收益
        # daily_sims shape: (n_simulations, horizon_days, n_assets)
        # w shape: (n_assets,)
        # result shape: (n_simulations, horizon_days)
        port_daily = np.einsum("nji,i->nj", daily_sims, w)
        # 累计收益（简单加总，近似）
        port_cumulative = port_daily.sum(axis=1)
        # 组合损失比例（负收益）
        losses = -port_cumulative

        var_mc = float(np.percentile(losses, confidence * 100))
        cvar_mc = float(losses[losses >= var_mc].mean()) if np.any(losses >= var_mc) else var_mc
        max_loss = float(losses.max())
        median_loss = float(np.median(losses))

        # 换算为金额（损失不低于 0）
        loss_pct = max(0.0, float(np.percentile(losses, confidence * 100)))
        loss_amount = loss_pct * portfolio_value

        return StressTestResult(
            scenario="monte_carlo",
            portfolio_loss_pct=loss_pct,
            portfolio_loss_amount=loss_amount,
            monte_carlo_distribution={
                "n_simulations": n_simulations,
                "horizon_days": horizon_days,
                "confidence": confidence,
                "var": var_mc,
                "cvar": cvar_mc,
                "max_loss": max_loss,
                "median_loss": median_loss,
                "mean_loss": float(losses.mean()),
                "std_loss": float(losses.std(ddof=1)),
            },
            description=f"蒙特卡洛模拟 {n_simulations} 次，{horizon_days} 日持有期",
        )

    def run_stress_test(
        self,
        weights: np.ndarray,
        returns_matrix: pd.DataFrame,
        scenario: str = "monte_carlo",
        portfolio_value: float = 1.0,
        n_simulations: int = 10000,
        horizon_days: int = 1,
        confidence: float = 0.95,
    ) -> StressTestResult:
        """统一入口：运行压力测试（历史情景或蒙特卡洛）。

        Args:
            scenario: ``"monte_carlo"`` 或 STRESS_SCENARIOS 中的 key。
            其余参数见上述方法。

        Returns:
            :class:`StressTestResult`。
        """
        if scenario == "monte_carlo":
            return self.stress_test_monte_carlo(
                weights, returns_matrix, portfolio_value,
                n_simulations, horizon_days, confidence,
            )
        return self.stress_test_historical(
            weights, returns_matrix, scenario, portfolio_value,
        )

    # ------------------------------------------------------------------ #
    # 与 RiskManager 集成
    # ------------------------------------------------------------------ #
    def integrate_with_risk_manager(
        self,
        risk_manager: RiskManager,
        portfolio_returns: pd.Series,
        weights: Optional[np.ndarray] = None,
        returns_matrix: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """将自适应风险结果同步到 RiskManager。

        根据当前尾部风险状态，动态调整 RiskManager 的仓位上限与暂停阈值：
            - critical: 暂停交易 + 仓位上限降至 10%。
            - high: 仓位上限降至 15%。
            - medium: 仓位上限降至 20%。
            - low: 保持原配置。

        Args:
            risk_manager: 要更新的 RiskManager 实例。
            portfolio_returns: 组合日收益率。
            weights: 权重。
            returns_matrix: 收益率矩阵。

        Returns:
            调整摘要 dict。
        """
        alert = self.detect_tail_risk(portfolio_returns, weights, returns_matrix)
        budget = self.adjust_risk_budget(portfolio_returns=portfolio_returns)

        original_max_position = risk_manager.max_position_per_symbol
        original_max_total = risk_manager.max_total_position

        if alert.severity == "critical":
            risk_manager.max_position_per_symbol = 0.10
            risk_manager.max_total_position = 0.30
            if not risk_manager.is_paused():
                risk_manager._paused = True
                from risk.risk_manager import RiskEvent
                risk_manager.events.append(
                    RiskEvent(
                        timestamp=pd.Timestamp.now().isoformat(),
                        event_type="adaptive_risk_pause",
                        symbol="ALL",
                        detail=f"自适应风险管理触发暂停: CVaR={alert.cvar:.4f}",
                    )
                )
        elif alert.severity == "high":
            risk_manager.max_position_per_symbol = 0.15
            risk_manager.max_total_position = 0.40
        elif alert.severity == "medium":
            risk_manager.max_position_per_symbol = 0.20
            risk_manager.max_total_position = 0.50
        # low: 不调整

        return {
            "severity": alert.severity,
            "var": alert.var,
            "cvar": alert.cvar,
            "adjusted_budget": budget["adjusted_budget"],
            "max_position_per_symbol": risk_manager.max_position_per_symbol,
            "max_total_position": risk_manager.max_total_position,
            "original_max_position": original_max_position,
            "original_max_total": original_max_total,
            "paused": risk_manager.is_paused(),
        }

    # ------------------------------------------------------------------ #
    # 综合报告
    # ------------------------------------------------------------------ #
    def risk_report(
        self,
        portfolio_returns: pd.Series,
        weights: Optional[np.ndarray] = None,
        returns_matrix: Optional[pd.DataFrame] = None,
        portfolio_value: float = 1.0,
    ) -> Dict[str, Any]:
        """生成综合风险报告。

        Returns:
            {
                "dynamic_var": {...},
                "risk_budget": {...},
                "tail_risk_alert": {...},
                "stress_test": {...},
                "tail_risk_history_count": int,
            }
        """
        var_info = self.dynamic_var(portfolio_returns, weights, returns_matrix)
        budget = self.adjust_risk_budget(portfolio_returns=portfolio_returns)
        alert = self.detect_tail_risk(portfolio_returns, weights, returns_matrix)

        # 蒙特卡洛压力测试
        if returns_matrix is not None and weights is not None:
            stress = self.stress_test_monte_carlo(
                weights, returns_matrix, portfolio_value,
            )
            stress_dict = stress.as_dict()
        else:
            stress_dict = None

        return {
            "dynamic_var": var_info,
            "risk_budget": budget,
            "tail_risk_alert": alert.as_dict(),
            "stress_test": stress_dict,
            "tail_risk_history_count": len(self.tail_risk_history),
        }
