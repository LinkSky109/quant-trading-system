#!/usr/bin/env python3
"""网格搜索示例：双均线策略参数寻优。

使用 mock 行情数据（无网络依赖），对 fast_period × slow_period 网格
进行穷举回测，按夏普比率排序后打印 Top 5。

运行方式:
    cd quant_trading_system
    python examples/run_grid_search.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from optimization.grid_search import GridSearchOptimizer
from strategies.ma_cross import MACrossStrategy


def make_mock_data(n_days: int = 400, seed: int = 42) -> pd.DataFrame:
    """构造带趋势 + 周期波动的 mock 日线数据。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2023-01-02", periods=n_days)

    t = np.arange(n_days)
    trend = 100.0 + 0.12 * t
    cycle = 10.0 * np.sin(2 * np.pi * t / 45.0)
    noise = rng.normal(0, 0.9, size=n_days)
    close = trend + cycle + noise

    open_ = close + rng.normal(0, 0.3, size=n_days)
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.6, size=n_days)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.6, size=n_days)
    volume = rng.uniform(8e5, 2e6, size=n_days)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)
    df.index.name = "date"
    return df


def main() -> None:
    """主函数。"""
    print("=" * 64)
    print("双均线策略参数网格搜索示例")
    print("=" * 64)

    # 1. 构造 mock 数据
    df = make_mock_data()
    print(f"mock 数据: {len(df)} 个交易日, "
          f"{df.index[0].date()} ~ {df.index[-1].date()}")

    # 2. 定义参数网格
    param_grid = {
        "fast_period": [3, 5, 8],
        "slow_period": [15, 20, 30],
    }
    total = len(param_grid["fast_period"]) * len(param_grid["slow_period"])
    print(f"参数组合数: {total} (fast×slow)")
    print(f"优化目标: 夏普比率（越大越好）\n")

    # 3. 运行网格搜索
    optimizer = GridSearchOptimizer(max_combos=200)

    def progress(current: int, total_count: int, item: dict) -> None:
        p = item.get("params", {})
        m = item.get("metrics", {})
        sharpe = m.get("夏普比率", float("nan"))
        print(f"  进度 {current}/{total_count}: "
              f"fast={p.get('fast_period')}, slow={p.get('slow_period')} "
              f"→ 夏普={sharpe:.3f}, 交易次数={m.get('交易次数', 0)}")

    results = optimizer.optimize(
        data=df,
        strategy_class=MACrossStrategy,
        param_grid=param_grid,
        symbol="MOCK",
        objective="sharpe",
        top_n=5,
        progress_callback=progress,
        initial_capital=1_000_000.0,
        commission_rate=0.00025,
    )

    # 4. 打印 Top 5 表格
    print("\n" + "=" * 64)
    print("Top 5 参数组合（按夏普比率降序）")
    print("=" * 64)

    rows = []
    for rank, item in enumerate(results, start=1):
        m = item["metrics"]
        rows.append({
            "排名": rank,
            "fast": item["params"]["fast_period"],
            "slow": item["params"]["slow_period"],
            "夏普比率": f"{m.get('夏普比率', 0):.3f}",
            "累计收益率": f"{m.get('累计收益率', 0) * 100:.2f}%",
            "年化收益率": f"{m.get('年化收益率', 0) * 100:.2f}%",
            "最大回撤": f"{m.get('最大回撤', 0) * 100:.2f}%",
            "胜率": f"{m.get('胜率', 0) * 100:.1f}%",
            "交易次数": int(m.get("交易次数", 0)),
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print("=" * 64)

    if results:
        best = results[0]
        print(f"\n最优参数: fast_period={best['params']['fast_period']}, "
              f"slow_period={best['params']['slow_period']}")
        print("注意: 以上结果基于 mock 数据，仅供演示，不构成投资建议。")


if __name__ == "__main__":
    main()
