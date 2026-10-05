"""配对交易策略（Pairs Trading）。

核心思想：找两只长期协整（价格走势存在稳定线性关系）的标的，
当价差（spread）短期偏离均衡时下注其回归：

- z_score >= z_entry (默认 2)：价差过高 → 卖 A 买 B
- z_score <= -z_entry：价差过低 → 买 A 卖 B
- |z_score| <= z_exit (默认 0.5)：价差回归中性 → 平仓
- |z_score| >= z_stop (默认 3)：价差进一步走阔 → 止损平仓

协整检验采用 Engle-Granger 两步法：
1. OLS 回归 price_a = α + β * price_b，得对冲比率 β；
2. 对残差做 ADF 平稳性检验，平稳则认为协整。

说明：本环境未安装 statsmodels，ADF 检验使用 numpy 自实现的简化版
（Δe_t ~ e_{t-1} 回归的 t 统计量，对比近似临界值）；若后续安装了
statsmodels，可直接替换 :func._adf_test 为
``statsmodels.tsa.stattools.adfuller``。
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

# ADF 简化版近似临界值（无常数项漂移、样本量 n>100 的常用 MacKinnon 近似值）
_ADF_CRITICALS = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}


def _safe_float(v: Any) -> Optional[float]:
    """把值转成 JSON 安全的 float（NaN/Inf -> None）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------


def _ols_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> Dict[str, float]:
    """OLS 回归 price_a = alpha + beta * price_b。

    Returns:
        {"hedge_ratio": beta, "intercept": alpha}。
    """
    x = np.asarray(price_b, dtype=float)
    y = np.asarray(price_a, dtype=float)
    if len(x) < 2:
        return {"hedge_ratio": 1.0, "intercept": 0.0}
    x_mean = x.mean()
    y_mean = y.mean()
    cov = np.mean((x - x_mean) * (y - y_mean))
    var_x = np.var(x, ddof=0)
    beta = cov / var_x if var_x > 0 else 1.0
    alpha = y_mean - beta * x_mean
    return {"hedge_ratio": float(beta), "intercept": float(alpha)}


def _adf_test(series: pd.Series) -> Dict[str, float]:
    """简化版 ADF 单位根检验（无 statsmodels 依赖时的降级实现）。

    回归 Δe_t = gamma * e_{t-1} + eps，取 gamma 的 t 统计量；
    越负说明越倾向平稳（拒绝单位根）。

    Returns:
        {"adf_statistic", "p_value", "is_stationary"}。
    """
    e = pd.Series(series).dropna().astype(float).to_numpy()
    if len(e) < 10:
        return {"adf_statistic": 0.0, "p_value": 1.0, "is_stationary": False}
    e_lag = e[:-1]
    delta_e = np.diff(e)
    # OLS: delta_e = gamma * e_lag
    var_x = np.var(e_lag, ddof=0)
    if var_x == 0:
        return {"adf_statistic": 0.0, "p_value": 1.0, "is_stationary": False}
    gamma = np.mean((e_lag - e_lag.mean()) * (delta_e - delta_e.mean())) / var_x
    resid = delta_e - gamma * (e_lag - e_lag.mean()) - delta_e.mean()
    se = np.sqrt(np.sum(resid ** 2) / max(len(e_lag) - 2, 1)) / (
        np.sqrt(np.sum((e_lag - e_lag.mean()) ** 2)) + 1e-12
    )
    t_stat = gamma / se if se > 0 else 0.0

    # 近似 p 值：按临界值分档
    if t_stat <= _ADF_CRITICALS[0.01]:
        p_value = 0.01
    elif t_stat <= _ADF_CRITICALS[0.05]:
        p_value = 0.05
    elif t_stat <= _ADF_CRITICALS[0.10]:
        p_value = 0.10
    else:
        p_value = 0.5  # 不平稳
    return {
        "adf_statistic": float(t_stat),
        "p_value": float(p_value),
        "is_stationary": bool(t_stat <= _ADF_CRITICALS[0.05]),
    }


def engle_granger_test(price_a: pd.Series, price_b: pd.Series) -> Dict[str, Any]:
    """Engle-Granger 两步法协整检验。

    Returns:
        {hedge_ratio, intercept, adf_statistic, p_value,
         is_cointegrated, spread_std, n}。
    """
    a = pd.Series(price_a).dropna().astype(float)
    b = pd.Series(price_b).dropna().astype(float)
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    joined.columns = ["a", "b"]
    if len(joined) < 30:
        return {"hedge_ratio": 1.0, "intercept": 0.0, "adf_statistic": 0.0,
                "p_value": 1.0, "is_cointegrated": False,
                "spread_std": 0.0, "n": int(len(joined))}

    fit = _ols_hedge_ratio(joined["a"], joined["b"])
    residual = joined["a"] - (fit["intercept"] + fit["hedge_ratio"] * joined["b"])
    adf = _adf_test(residual)
    return {
        "hedge_ratio": fit["hedge_ratio"],
        "intercept": fit["intercept"],
        "adf_statistic": adf["adf_statistic"],
        "p_value": adf["p_value"],
        "is_cointegrated": adf["is_stationary"],
        "spread_std": float(residual.std(ddof=1)),
        "n": int(len(joined)),
    }


# ---------------------------------------------------------------------------
# 价差 / z-score
# ---------------------------------------------------------------------------


def calculate_spread(price_a: pd.Series, price_b: pd.Series,
                     hedge_ratio: float) -> pd.Series:
    """计算价差 spread = price_a - hedge_ratio * price_b。"""
    return pd.Series(price_a).astype(float) - float(hedge_ratio) * pd.Series(price_b).astype(float)


def calculate_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    """滚动 z-score：(spread - rolling_mean) / rolling_std。"""
    s = pd.Series(spread).astype(float)
    rm = s.rolling(window, min_periods=max(2, window // 2)).mean()
    rstd = s.rolling(window, min_periods=max(2, window // 2)).std(ddof=1)
    return (s - rm) / rstd.replace(0, np.nan)


# ---------------------------------------------------------------------------
# 标的对筛选
# ---------------------------------------------------------------------------


def screen_pairs(
    price_data: Dict[str, pd.Series],
    min_correlation: float = 0.7,
) -> List[Dict[str, Any]]:
    """在股票池中筛选可交易标的对。

    流程：
    1. 计算所有标的对的收益率相关系数；
    2. 对 |corr| >= min_correlation 的对做 Engle-Granger 协整检验；
    3. 按 |corr| 降序返回。

    Args:
        price_data: {symbol: 价格序列}。
        min_correlation: 最小相关系数阈值。

    Returns:
        标的对列表，含 correlation / hedge_ratio / adf_pvalue /
        is_cointegrated / spread_std 等字段。
    """
    # 对齐所有价格序列并计算收益率
    prices = pd.DataFrame({k: pd.Series(v, dtype=float) for k, v in price_data.items()})
    rets = prices.pct_change().dropna()
    if rets.shape[1] < 2:
        return []

    results: List[Dict[str, Any]] = []
    for sym_a, sym_b in combinations(prices.columns, 2):
        corr = float(rets[sym_a].corr(rets[sym_b]))
        if not np.isfinite(corr) or abs(corr) < min_correlation:
            continue
        eg = engle_granger_test(prices[sym_a], prices[sym_b])
        results.append({
            "symbol_a": sym_a,
            "symbol_b": sym_b,
            "correlation": corr,
            "hedge_ratio": eg["hedge_ratio"],
            "intercept": eg["intercept"],
            "adf_pvalue": eg["p_value"],
            "adf_statistic": eg["adf_statistic"],
            "is_cointegrated": eg["is_cointegrated"],
            "spread_std": eg["spread_std"],
        })
    results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# 策略类
# ---------------------------------------------------------------------------


class PairsTradingStrategy(BaseStrategy):
    """配对交易策略（继承 BaseStrategy，可被 BacktestEngine 调用）。

    与单标的策略不同，本策略需要两条腿的价格。使用约定：
    传给 :meth:`generate_signals` 的 ``df`` 需同时包含
    ``close``（腿 A 价格）与 ``price_b``（腿 B 价格）两列。
    """

    name = "pairs_trading"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.symbol_a: str = str(self.params.get("symbol_a", "A"))
        self.symbol_b: str = str(self.params.get("symbol_b", "B"))
        self.hedge_ratio: float = float(self.params.get("hedge_ratio", 1.0))
        self.z_entry: float = float(self.params.get("z_entry", 2.0))
        self.z_exit: float = float(self.params.get("z_exit", 0.5))
        self.z_stop: float = float(self.params.get("z_stop", 3.0))
        self.window: int = int(self.params.get("window", 20))

    # ------------------------------------------------------------------
    # 核心信号（两腿价格直接传入）
    # ------------------------------------------------------------------
    def generate_pair_signals(
        self, price_a: pd.Series, price_b: pd.Series
    ) -> pd.DataFrame:
        """根据两腿价格生成价差、z-score 与目标持仓。

        Returns:
            DataFrame，列：spread, zscore, position
            （+1=买A卖B/做多价差，-1=卖A买B/做空价差，0=空仓）。
        """
        spread = calculate_spread(price_a, price_b, self.hedge_ratio)
        z = calculate_zscore(spread, self.window)

        position = np.zeros(len(z), dtype=int)
        pos = 0
        zv = z.to_numpy()
        for i in range(len(zv)):
            zi = zv[i]
            if not np.isfinite(zi):
                position[i] = pos
                continue
            if pos == 0:
                if zi >= self.z_entry:
                    pos = -1   # 价差过高 → 卖A买B（做空价差）
                elif zi <= -self.z_entry:
                    pos = 1    # 价差过低 → 买A卖B（做多价差）
            elif pos == 1:     # 做多价差，等 z 回归 0
                if zi <= -self.z_stop:
                    pos = 0  # 反向止损
                elif zi >= -self.z_exit:
                    pos = 0  # 回归中性平仓
            elif pos == -1:    # 做空价差
                if zi >= self.z_stop:
                    pos = 0  # 反向止损
                elif zi <= self.z_exit:
                    pos = 0  # 回归中性平仓
            position[i] = pos

        out = pd.DataFrame({
            "spread": spread.values,
            "zscore": z.values,
            "position": position,
        }, index=z.index)
        return out

    # ------------------------------------------------------------------
    # BaseStrategy 接口
    # ------------------------------------------------------------------
    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """在 df 上计算 signal/confidence 列。

        df 需包含 ``close``（腿 A）与 ``price_b``（腿 B）列。
        signal 编码为腿 A 的仓位变动：+1 买 A / -1 卖 A / 0 无操作。
        """
        if "price_b" not in df.columns:
            raise ValueError("PairsTradingStrategy 需要 df 同时包含 close 与 price_b 列")
        out = df.copy()
        sig = self.generate_pair_signals(out["close"], out["price_b"])
        out["spread"] = sig["spread"]
        out["zscore"] = sig["zscore"]
        out["position"] = sig["position"]
        # 腿 A 信号 = 仓位差分（开仓/平仓动作）
        out["signal"] = sig["position"].diff().fillna(sig["position"]).clip(-1, 1)
        out["confidence"] = sig["zscore"].abs().fillna(0).clip(0, 1)
        return out


# ---------------------------------------------------------------------------
# 配对策略回测（自包含，供 API 层调用）
# ---------------------------------------------------------------------------


def run_pairs_backtest(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 3.0,
    window: int = 20,
    initial_capital: float = 1_000_000.0,
) -> Dict[str, Any]:
    """配对策略简易回测：模拟价差持仓的逐日 PnL。

    持仓约定（position_t）：
        +1：买 1 股 A、卖 hedge_ratio 股 B
        -1：卖 1 股 A、买 hedge_ratio 股 B
    当日 PnL = position_{t-1} * (Δa_t - hedge_ratio * Δb_t)

    Returns:
        {metrics, equity_curve, trades, spread_series}。
    """
    strat = PairsTradingStrategy({
        "hedge_ratio": hedge_ratio, "z_entry": z_entry,
        "z_exit": z_exit, "z_stop": z_stop, "window": window,
    })
    sig = strat.generate_pair_signals(price_a, price_b)

    # 用 values 直接对齐，避免索引不一致引入 NaN
    a = pd.Series(np.asarray(price_a, dtype=float), index=sig.index)
    b = pd.Series(np.asarray(price_b, dtype=float), index=sig.index)

    # 用昨日持仓计算今日 PnL（避免未来函数）
    pos_prev = sig["position"].shift(1).fillna(0)
    pnl = pos_prev * (a.diff().fillna(0) - hedge_ratio * b.diff().fillna(0))
    # 归一化为相对初始资金的日收益率
    capital_per_unit = float(a.iloc[0]) * (1.0 + abs(hedge_ratio))
    daily_ret = pnl / capital_per_unit if capital_per_unit > 0 else pnl * 0
    equity = initial_capital * (1.0 + daily_ret).cumprod()

    # 交易记录：position 变化处
    trades: List[Dict[str, Any]] = []
    pos_arr = sig["position"].to_numpy()
    idx = sig.index
    for i in range(1, len(pos_arr)):
        if pos_arr[i] != pos_arr[i - 1]:
            action = {1: "buy_spread", -1: "sell_spread", 0: "close"}[int(pos_arr[i])]
            trades.append({
                "date": idx[i].strftime("%Y-%m-%d") if hasattr(idx[i], "strftime") else str(idx[i]),
                "position_after": int(pos_arr[i]),
                "action": action,
                "zscore": float(sig["zscore"].iloc[i]) if pd.notna(sig["zscore"].iloc[i]) else 0.0,
                "price_a": float(a.iloc[i]),
                "price_b": float(b.iloc[i]),
            })

    # 绩效指标
    total_ret = float(equity.iloc[-1] / initial_capital - 1.0)
    daily = equity.pct_change().dropna()
    ann_vol = float(daily.std(ddof=1) * np.sqrt(252)) if len(daily) > 1 else 0.0
    ann_ret = float((1.0 + total_ret) ** (252.0 / max(len(daily), 1)) - 1.0)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    peak = equity.cummax()
    max_dd = float(((equity - peak) / peak).min())

    metrics = {
        "total_return": total_ret,
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "num_trades": len(trades),
        "final_equity": float(equity.iloc[-1]),
    }
    equity_curve = [
        {"date": idx[i].strftime("%Y-%m-%d") if hasattr(idx[i], "strftime") else str(idx[i]),
         "equity": _safe_float(equity.iloc[i])}
        for i in range(len(equity))
    ]
    spread_series = [
        {"date": idx[i].strftime("%Y-%m-%d") if hasattr(idx[i], "strftime") else str(idx[i]),
         "spread": _safe_float(sig["spread"].iloc[i]),
         "zscore": _safe_float(sig["zscore"].iloc[i]) if pd.notna(sig["zscore"].iloc[i]) else None,
         "position": int(sig["position"].iloc[i])}
        for i in range(len(sig))
    ]
    return {
        "metrics": metrics,
        "equity_curve": equity_curve,
        "trades": trades,
        "spread_series": spread_series,
    }
