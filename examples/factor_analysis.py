#!/usr/bin/env python3
"""多因子分析引擎演示脚本。

演示内容：
  1. 计算单只股票全部因子值（取最新一日）
  2. 用多只模拟股票构建横截面面板做 IC 分析
  3. 分层回测（5 层多空）
  4. 单只股票因子暴露度（z-score）分析

运行方式:
    cd quant_trading_system
    python examples/factor_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from factors.factor_engine import FactorEngine


def make_klines(n_days: int = 300, seed: int = 0,
                start_price: float = 100.0) -> pd.DataFrame:
    """构造模拟日 K 线（与测试中一致）。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rets = rng.normal(0.0005, 0.02, size=n_days)
    close = start_price * np.cumprod(1.0 + rets)
    volume = rng.uniform(1e6, 5e6, size=n_days)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": volume, "amount": volume * close},
        index=dates,
    )


def main() -> None:
    engine = FactorEngine()

    # ------------------------------------------------------------------
    print("=" * 70)
    print("1) 因子库列表")
    print("=" * 70)
    factors = engine.get_factor_list()
    print(f"共 {len(factors)} 个因子：")
    for f in factors:
        print(f"  [{f['category']:<4}] {f['name']:<24} "
              f"direction={f['direction']:+d}  {f['description']}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2) 单只股票因子计算（模拟 600519.SH，300 日 K 线）")
    print("=" * 70)
    df = make_klines(300, seed=42)
    computed = engine.calculate_factors(df, symbol="600519.SH")
    latest = computed[engine.factor_names].iloc[-1]
    print("最新交易日因子值：")
    print(latest.round(4).to_string())

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3) IC 分析（8 只模拟股票，构造与未来收益相关的动量因子）")
    print("=" * 70)
    n_symbols, n_days, fwd = 8, 300, 5
    rng = np.random.RandomState(7)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    panel: dict = {}
    for i in range(n_symbols):
        rets = rng.normal(0.0005, 0.02, size=n_days)
        close = pd.Series(100.0 * np.cumprod(1.0 + rets), index=dates)
        fwd_ret = close.shift(-fwd) / close - 1.0
        factor = fwd_ret + rng.normal(0, 0.002, size=n_days)
        panel[f"SYM{i:02d}"] = pd.DataFrame(
            {"momentum_20": factor, "close": close}, index=dates
        )
    ic = engine.factor_ic_analysis(panel, "momentum_20", forward_days=fwd)
    print(f"IC 均值   : {ic['ic_mean']:.4f}")
    print(f"IC 标准差 : {ic['ic_std']:.4f}")
    print(f"IC IR     : {ic['ic_ir']:.4f}")
    print(f"IC 胜率   : {ic['ic_win_rate']:.2%}")
    print(f"有效期数  : {ic['n_periods']}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4) 分层回测（5 层等权，持有 5 日）")
    print("=" * 70)
    bt = engine.layered_backtest(panel, "momentum_20",
                                 n_layers=5, forward_days=fwd)
    for q, ret in bt["layer_returns"].items():
        print(f"  {q} 层平均未来收益: {ret:.4%}")
    print(f"  多空(Q5-Q1) 平均收益: {bt['long_short_return']:.4%}")
    print(f"  单调性 Spearman 相关: {bt['monotonicity']:.4f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5) 单只股票因子暴露度（z-score，最新一日）")
    print("=" * 70)
    exposure = engine.factor_exposure(df, symbol="600519.SH")
    for name, val in exposure.items():
        print(f"  {name:<24} {val:+.3f}")


if __name__ == "__main__":
    main()
