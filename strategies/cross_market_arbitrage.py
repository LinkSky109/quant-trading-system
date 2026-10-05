"""跨市场套利策略（Cross-Market Arbitrage）。

基于配对交易逻辑扩展，支持多市场标的配对：

- **A股** (`.SH` / `.SZ`)
- **港股** (`.HK`)
- **美股** (无后缀，如 `AAPL`)

核心逻辑：
1. 对两只跨市场标的做协整检验（复用 pairs_trading.py 的 OLS + ADF）
2. 计算标准化价差 z_score（考虑汇率预留接口，当前 1:1）
3. 信号规则：
   - z_score >= z_entry (默认 2.0)：开仓（卖高买低）
   - |z_score| <= z_exit (默认 0.5)：平仓
   - |z_score| >= z_stop (默认 3.0)：止损
4. 不同市场差异化交易成本处理

继承 BaseStrategy，所有信号统一 shift(1) 防未来函数。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)

# ADF 简化版临界值（与 pairs_trading.py 保持一致）
_ADF_CRITICALS = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}

# 市场默认交易成本映射（单边，比例）
_DEFAULT_MARKET_COSTS: Dict[str, float] = {
    "CN": 0.0003,   # A股 万3
    "HK": 0.001,    # 港股 千1
    "US": 0.0005,   # 美股 万5
}


def _detect_market(symbol: str) -> str:
    """从标的代码推断所属市场。"""
    if symbol.endswith(".SH") or symbol.endswith(".SZ"):
        return "CN"
    if symbol.endswith(".HK"):
        return "HK"
    return "US"


def _ols_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> Dict[str, float]:
    """OLS 回归 price_a = alpha + beta * price_b（复用 pairs_trading 逻辑）。"""
    x = np.asarray(price_b, dtype=float)
    y = np.asarray(price_a, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return {"hedge_ratio": 1.0, "intercept": 0.0}
    x_mean, y_mean = x.mean(), y.mean()
    beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    alpha = y_mean - beta * x_mean
    return {"hedge_ratio": float(beta), "intercept": float(alpha)}


def _adf_test(series: pd.Series) -> Dict[str, Any]:
    """简化版 ADF 检验（复用 pairs_trading 逻辑）。"""
    s = np.asarray(series.dropna(), dtype=float)
    if len(s) < 30:
        return {"t_stat": -999.0, "p_value": 1.0, "is_stationary": False, "critical": None}
    # Δy_t = α + β*y_{t-1} + ε_t，只看 β 的 t 统计量
    y_lag = s[:-1]
    dy = np.diff(s)
    y_mean = y_lag.mean()
    dy_mean = dy.mean()
    num = np.sum((y_lag - y_mean) * (dy - dy_mean))
    den = np.sum((y_lag - y_mean) ** 2)
    if den == 0:
        return {"t_stat": -999.0, "p_value": 1.0, "is_stationary": False, "critical": None}
    beta = num / den
    residuals = dy - beta * (y_lag - y_mean)
    se = np.sqrt(np.sum(residuals ** 2) / (len(dy) - 1)) / np.sqrt(den)
    if se == 0:
        return {"t_stat": -999.0, "p_value": 1.0, "is_stationary": False, "critical": None}
    t_stat = beta / se
    is_stationary = t_stat < _ADF_CRITICALS[0.05]
    return {
        "t_stat": float(t_stat),
        "p_value": 1.0,
        "is_stationary": bool(is_stationary),
        "critical": _ADF_CRITICALS[0.05],
    }


def get_exchange_rate(symbol_a: str, symbol_b: str) -> float:
    """获取汇率转换因子（预留接口）。

    当前默认 1:1，未来接入实时汇率 API 后可按市场组合返回实际汇率。
    """
    market_a = _detect_market(symbol_a)
    market_b = _detect_market(symbol_b)
    if market_a == market_b:
        return 1.0
    # 跨市场时预留：未来从外部服务获取实时汇率
    logger.debug("跨市场汇率暂未接入，使用 1:1: %s vs %s", symbol_a, symbol_b)
    return 1.0


def get_trading_cost(symbol: str, custom_costs: Optional[Dict[str, float]] = None) -> float:
    """获取标的所在市场的单边交易成本。"""
    costs = {**_DEFAULT_MARKET_COSTS, **(custom_costs or {})}
    return costs.get(_detect_market(symbol), 0.001)


# ---------------------------------------------------------------------------
# 策略实现
# ---------------------------------------------------------------------------

class CrossMarketArbitrageStrategy(BaseStrategy):
    """跨市场套利策略。

    Args:
        params: 策略参数字典，可含：
            - symbol_a, symbol_b: 标的对
            - z_entry: 开仓阈值（默认 2.0）
            - z_exit: 平仓阈值（默认 0.5）
            - z_stop: 止损阈值（默认 3.0）
            - window: z_score 滚动窗口（默认 20）
            - trading_costs: 自定义交易成本 {market: cost_ratio}
    """

    name = "cross_market_arbitrage"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.symbol_a = self.params.get("symbol_a", "")
        self.symbol_b = self.params.get("symbol_b", "")
        self.z_entry = self.params.get("z_entry", 2.0)
        self.z_exit = self.params.get("z_exit", 0.5)
        self.z_stop = self.params.get("z_stop", 3.0)
        self.window = self.params.get("window", 20)
        self.trading_costs: Dict[str, float] = self.params.get("trading_costs", {})

        # 运行时状态
        self._hedge_ratio: float = 1.0
        self._intercept: float = 0.0
        self._cointegrated: bool = False
        self._position: int = 0  # 0=空仓, +1=价差多头(A低B高), -1=价差空头

    # ------------------------------------------------------------------ #
    # 协整检验
    # ------------------------------------------------------------------ #
    def test_cointegration(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
    ) -> Dict[str, Any]:
        """对两只标的做协整检验。

        Returns:
            dict 含 hedge_ratio, intercept, is_cointegrated, adf_t_stat。
        """
        joined = pd.DataFrame({"a": prices_a, "b": prices_b}).dropna()
        if len(joined) < 30:
            logger.warning("数据不足，无法做协整检验")
            return {
                "hedge_ratio": 1.0,
                "intercept": 0.0,
                "is_cointegrated": False,
                "adf_t_stat": -999.0,
            }
        fit = _ols_hedge_ratio(joined["a"], joined["b"])
        residual = joined["a"] - (fit["intercept"] + fit["hedge_ratio"] * joined["b"])
        adf = _adf_test(residual)
        self._hedge_ratio = fit["hedge_ratio"]
        self._intercept = fit["intercept"]
        self._cointegrated = adf["is_stationary"]
        return {
            "hedge_ratio": self._hedge_ratio,
            "intercept": self._intercept,
            "is_cointegrated": self._cointegrated,
            "adf_t_stat": adf["t_stat"],
        }

    # ------------------------------------------------------------------ #
    # 价差与 z_score
    # ------------------------------------------------------------------ #
    def calculate_spread(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
    ) -> pd.DataFrame:
        """计算价差与 z_score 序列。

        Returns:
            DataFrame 含 spread, spread_mean, spread_std, z_score 列。
        """
        joined = pd.DataFrame({"a": prices_a, "b": prices_b}).dropna()
        if len(joined) < self.window:
            return pd.DataFrame(index=joined.index)

        fx = get_exchange_rate(self.symbol_a, self.symbol_b)
        spread = joined["a"] - (self._intercept + self._hedge_ratio * joined["b"] * fx)
        spread_mean = spread.rolling(self.window, min_periods=self.window).mean()
        spread_std = spread.rolling(self.window, min_periods=self.window).std()
        z_score = (spread - spread_mean) / spread_std.replace(0, np.nan)

        return pd.DataFrame({
            "spread": spread,
            "spread_mean": spread_mean,
            "spread_std": spread_std,
            "z_score": z_score,
        })

    # ------------------------------------------------------------------ #
    # BaseStrategy 接口
    # ------------------------------------------------------------------ #
    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成跨市场套利原始信号。

        输入 df 需含两列：{symbol_a}_close 和 {symbol_b}_close，
        或单标的时退化为仅输出 neutral。
        """
        out = df.copy()
        col_a = f"{self.symbol_a}_close" if self.symbol_a else "close_a"
        col_b = f"{self.symbol_b}_close" if self.symbol_b else "close_b"

        if col_a not in out.columns or col_b not in out.columns:
            # 尝试默认列名
            if "close_a" in out.columns and "close_b" in out.columns:
                col_a, col_b = "close_a", "close_b"
            else:
                out["signal"] = 0
                out["confidence"] = 0.0
                return out

        # 协整检验（用全部历史数据）
        self.test_cointegration(out[col_a], out[col_b])
        if not self._cointegrated:
            logger.info("标的对未通过协整检验，输出 neutral")

        spread_df = self.calculate_spread(out[col_a], out[col_b])
        out = pd.concat([out, spread_df], axis=1)

        z = out["z_score"]
        signal = pd.Series(0, index=out.index)
        confidence = pd.Series(0.5, index=out.index)

        # 开仓信号
        signal[z >= self.z_entry] = -1   # 价差过高 → 卖A买B（价差空头）
        signal[z <= -self.z_entry] = 1   # 价差过低 → 买A卖B（价差多头）

        # 平仓/止损
        flat_mask = z.abs() <= self.z_exit
        stop_mask = z.abs() >= self.z_stop
        signal[flat_mask | stop_mask] = 0

        # confidence = min(|z| / z_entry, 1.0)
        confidence = (z.abs() / self.z_entry).clip(0.0, 1.0)
        confidence[z.isna()] = 0.0

        out["signal"] = signal
        out["confidence"] = confidence
        return out

    def generate_signals(self, df: pd.DataFrame, symbol: str = "") -> List[Signal]:
        """生成 Signal 列表，shift(1) 防未来函数。"""
        raw = self._compute_raw_signals(df)
        raw["signal"] = raw["signal"].shift(1)
        raw["confidence"] = raw["confidence"].shift(1)

        signals: List[Signal] = []
        for idx, row in raw.iterrows():
            sig_val = row.get("signal", 0)
            if pd.isna(sig_val) or sig_val == 0:
                continue
            action = "buy" if sig_val > 0 else "sell"
            pair_label = f"{self.symbol_a}/{self.symbol_b}" if self.symbol_a and self.symbol_b else symbol
            signals.append(Signal(
                date=idx,
                symbol=pair_label,
                strategy=self.name,
                action=action,
                confidence=float(row.get("confidence", 0.5)),
                price=float(row.get("close", 0)) if "close" in row else 0.0,
                metadata={
                    "z_score": _safe_float(row.get("z_score")),
                    "hedge_ratio": self._hedge_ratio,
                    "spread": _safe_float(row.get("spread")),
                    "cost_a": get_trading_cost(self.symbol_a, self.trading_costs),
                    "cost_b": get_trading_cost(self.symbol_b, self.trading_costs),
                },
            ))
        return signals

    # ------------------------------------------------------------------ #
    # 成本估算
    # ------------------------------------------------------------------ #
    def estimate_cost(self, notional: float = 1_000_000.0) -> Dict[str, float]:
        """估算双边交易成本。

        Returns:
            {"total_cost": ..., "cost_a": ..., "cost_b": ...}
        """
        cost_a = get_trading_cost(self.symbol_a, self.trading_costs)
        cost_b = get_trading_cost(self.symbol_b, self.trading_costs)
        # 双边各一次开仓 + 一次平仓 = 4 笔单边
        total = notional * (cost_a + cost_b) * 2
        return {
            "total_cost": float(total),
            "cost_a": float(cost_a),
            "cost_b": float(cost_b),
        }


def _safe_float(v: Any) -> Optional[float]:
    """安全转 float（NaN/Inf -> None）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f
