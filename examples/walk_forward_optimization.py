#!/usr/bin/env python3
"""滚动窗口优化（Walk-Forward Optimization）示例。

使用 mock 行情数据（无网络依赖），对双均线策略参数做滚动寻优：
- 切分多个 IS/OOS 窗口，IS 内选优、OOS 上检验；
- 合并所有 OOS 段净值作为样本外真实表现；
- 输出过拟合风险报告与推荐参数。

运行方式:
    cd quant_trading_system
    python examples/walk_forward_optimization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from optimization.walk_forward import WalkForwardOptimizer
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
    print("双均线策略参数滚动优化（Walk-Forward）示例")
    print("=" * 64)

    # 1. 构造 mock 数据
    df = make_mock_data(n_days=400)
    print(f"mock 数据: {len(df)} 个交易日, "
          f"{df.index[0].date()} ~ {df.index[-1].date()}")

    # 2. 定义参数空间（连续周期用离散点）
    param_grid = {
        "fast_period": [3, 5, 8, 10],
        "slow_period": [15, 20, 30, 40],
    }
    total = 1
    for v in param_grid.values():
        total *= len(v)
    print(f"参数组合数: {total}")
    print(f"窗口数: 3, IS 占比: 0.7, 优化目标: 夏普比率\n")

    # 3. 运行滚动优化
    optimizer = WalkForwardOptimizer(max_combos=500)

    def progress(current: int, total_windows: int, info: dict) -> None:
        p = info.get("best_params", {})
        print(f"  窗口 {current}/{total_windows} 完成: "
              f"fast={p.get('fast_period')}, slow={p.get('slow_period')} "
              f"→ IS夏普={info.get('is_objective', 0):.3f}, "
              f"OOS夏普={info.get('oos_objective', 0):.3f}")

    result = optimizer.optimize(
        data=df,
        strategy_class=MACrossStrategy,
        param_grid=param_grid,
        n_windows=3,
        is_ratio=0.7,
        objective="sharpe",
        symbol="MOCK",
        progress_callback=progress,
        initial_capital=1_000_000.0,
        commission_rate=0.00025,
    )

    # 4. 打印各窗口详情
    print("\n" + "=" * 64)
    print("各窗口详情（IS 选优 / OOS 检验）")
    print("=" * 64)
    rows = []
    for w in result["windows"]:
        rows.append({
            "窗口": w["window_index"],
            "IS区间": f"{w['is_range'][0]}~{w['is_range'][1]}",
            "OOS区间": f"{w['oos_range'][0]}~{w['oos_range'][1]}",
            "fast": w["best_params"]["fast_period"],
            "slow": w["best_params"]["slow_period"],
            "IS夏普": f"{w['is_objective']:.3f}",
            "OOS夏普": f"{w['oos_objective']:.3f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # 5. 合并 OOS 绩效
    print("\n" + "=" * 64)
    print("合并 OOS 样本外绩效")
    print("=" * 64)
    cm = result["combined_metrics"]
    print(f"  累计收益率: {cm.get('累计收益率', 0) * 100:.2f}%")
    print(f"  年化收益率: {cm.get('年化收益率', 0) * 100:.2f}%")
    print(f"  夏普比率:   {cm.get('夏普比率', 0):.3f}")
    print(f"  最大回撤:   {cm.get('最大回撤', 0) * 100:.2f}%")
    print(f"  卡玛比率:   {cm.get('卡玛比率', 0):.3f}")
    print(f"  索提诺比率: {cm.get('索提诺比率', 0):.3f}")

    # 6. 过拟合报告
    print("\n" + "=" * 64)
    print("过拟合风险报告")
    print("=" * 64)
    rep = result["overfitting_report"]
    print(f"  风险等级:     {rep['risk_level']}")
    print(f"  OOS/IS绩效比: {rep['oos_is_ratio']}")
    print(f"  参数稳定性:   {rep['param_stability_score']}")
    print(f"  说明:         {rep['details']}")

    # 7. 推荐参数
    print("\n" + "=" * 64)
    print("推荐参数（各窗口最优参数的中位数/众数）")
    print("=" * 64)
    for k, v in result["recommended_params"].items():
        print(f"  {k} = {v}")

    print("\n注意: 以上结果基于 mock 数据，仅供演示，不构成投资建议。")


if __name__ == "__main__":
    main()
