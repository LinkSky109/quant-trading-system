#!/usr/bin/env python3
"""组合权重优化示例：四种方法权重对比。

使用合成行情数据（无网络依赖），对比：
- equal_weight   等权
- min_variance   最小方差
- risk_parity    风险平价
- mean_variance  均值方差（最大夏普）

打印每种方法的权重、预期年化收益、波动率与夏普比率。

运行方式:
    cd quant_trading_system
    python examples/run_portfolio_optimization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from optimization.portfolio_optimizer import (
    ALLOWED_METHODS,
    PortfolioOptimizer,
)


def make_synthetic_pool(
    n_assets: int = 6,
    n_days: int = 1000,
    seed: int = 42,
) -> "tuple[list[str], dict]":
    """构造一个含不同波动率/收益/相关性的股票池（合成数据）。"""
    rng = np.random.default_rng(seed)
    ann_vols = np.linspace(0.08, 0.28, n_assets)
    daily_vols = ann_vols / np.sqrt(252)
    daily_drifts = np.linspace(0.0002, 0.0009, n_assets)

    rets = rng.normal(0, 1.0, size=(n_days, n_assets)) * daily_vols + daily_drifts
    # 公共市场因子 -> 标的间相关性
    market = rng.normal(0, 1.0, size=n_days) * 0.01
    for i in range(n_assets):
        rets[:, i] += 0.5 * market

    dates = pd.bdate_range("2023-01-02", periods=n_days)
    symbols = [f"STOCK{i:02d}" for i in range(n_assets)]
    data = {}
    for i, s in enumerate(symbols):
        price = 100.0 * np.cumprod(1.0 + rets[:, i])
        data[s] = pd.DataFrame({"close": price}, index=dates)
    return symbols, data


def main() -> None:
    print("=" * 72)
    print("组合权重优化：四种方法对比")
    print("=" * 72)

    symbols, data = make_synthetic_pool()
    print(f"股票池: {len(symbols)} 只标的, {len(data[symbols[0]])} 个交易日")
    print(f"区间: {data[symbols[0]].index[0].date()} ~ "
          f"{data[symbols[0]].index[-1].date()}")
    print(f"约束: long-only, 单标的上限 0.30, 无风险利率 0.02\n")

    optimizer = PortfolioOptimizer(risk_free_rate=0.02, max_weight=0.30)

    rows = []
    weight_table = {}
    for method in ALLOWED_METHODS:
        res = optimizer.optimize(symbols, data, method=method)
        weight_table[method] = res.weights
        rows.append({
            "method": method,
            "预期年化收益": f"{res.expected_return * 100:.2f}%",
            "年化波动": f"{res.expected_volatility * 100:.2f}%",
            "夏普比率": f"{res.sharpe:.3f}",
            "最大权重": f"{max(res.weights.values()):.3f}",
        })

    # 1. 指标对比表
    print("-" * 72)
    print("组合指标对比")
    print("-" * 72)
    print(pd.DataFrame(rows).to_string(index=False))

    # 2. 权重对比表
    print("\n" + "-" * 72)
    print("权重对比（行=标的，列=方法）")
    print("-" * 72)
    wdf = pd.DataFrame(weight_table).loc[symbols]
    print(wdf.round(4).to_string())

    print("\n说明: 结果基于合成数据，仅供方法演示，不构成投资建议。")


if __name__ == "__main__":
    main()
