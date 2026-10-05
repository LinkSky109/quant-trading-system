#!/usr/bin/env python3
"""多策略组合回测示例：同一标的(600519.SH)上 3 个策略等权叠加。

演示 :class:`strategies.strategy_portfolio.StrategyPortfolio`：
  - 3 个策略（ma_cross / bollinger / momentum_breakout）等权分配资金
  - 打印组合级绩效 + 单策略明细 + 策略贡献度分析 + 信号冲突日志

使用 mock 行情数据（无需网络 / API Key）。

运行方式:
    cd quant_trading_system
    python examples/run_strategy_portfolio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from data.data_fetcher import _generate_mock_klines
from monitoring.monitor import setup_logging
from strategies.strategy_portfolio import StrategyPortfolio

SYMBOL = "600519.SH"
STRATEGIES = ["ma_cross", "bollinger", "momentum_breakout"]
INITIAL_CAPITAL = 1_000_000.0
# mock 基准价：高价蓝筹等权拆分资金后单笔 20% 仓位不足 1 手，
# 用亲民价格演示，确保策略信号能真实成交。
MOCK_BASE_PRICE = 25.0


def fetch_mock_data() -> pd.DataFrame:
    """获取 600519.SH 的 mock 日线数据（确定性，无需网络）。"""
    df = _generate_mock_klines(
        SYMBOL, period="1d", count=250,
        start_date="2024-01-02", base_price=MOCK_BASE_PRICE,
    )
    print(f"  {SYMBOL}: {len(df)} 条, "
          f"{df.index[0].date()} ~ {df.index[-1].date()}, "
          f"收盘 {df['close'].iloc[0]:.2f} -> {df['close'].iloc[-1]:.2f}")
    return df


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def print_portfolio_metrics(metrics: dict) -> None:
    """打印组合级绩效。"""
    print("-" * 60)
    print("组合级绩效:")
    rows = [
        ("累计收益率", _fmt_pct(metrics["累计收益率"])),
        ("年化收益率", _fmt_pct(metrics["年化收益率"])),
        ("最大回撤", _fmt_pct(metrics["最大回撤"])),
        ("夏普比率", f"{metrics['夏普比率']:.3f}"),
        ("胜率", _fmt_pct(metrics["胜率"])),
        ("盈亏比", f"{metrics['盈亏比']:.2f}"),
        ("平仓交易次数", f"{int(metrics['交易次数'])}"),
    ]
    for k, v in rows:
        print(f"  {k:<12}: {v}")


def print_strategy_details(result) -> None:
    """打印每个策略的独立回测明细。"""
    print("-" * 60)
    print("单策略明细:")
    header = f"  {'策略':<20}{'权重':>8}{'累计收益':>12}{'最大回撤':>12}{'夏普':>8}{'交易':>6}"
    print(header)
    for name, res in result.strategy_results.items():
        m = res.metrics
        w = result.strategy_weights[name]
        print(
            f"  {name:<20}{w*100:>7.1f}%"
            f"{_fmt_pct(m['累计收益率']):>12}"
            f"{_fmt_pct(m['最大回撤']):>12}"
            f"{m['夏普比率']:>8.3f}"
            f"{int(m['交易次数']):>6d}"
        )


def print_contribution(result) -> None:
    """打印策略贡献度分析。"""
    print("-" * 60)
    print("策略贡献度分析:")
    print(f"  {'策略':<20}{'盈亏(元)':>14}{'收益率':>12}{'贡献占比':>12}")
    for name, info in result.contribution.items():
        print(
            f"  {name:<20}{info['pnl']:>14,.2f}"
            f"{_fmt_pct(info['return_pct']):>12}"
            f"{info['contribution_pct']*100:>11.2f}%"
        )


def print_conflict_log(result, limit: int = 5) -> None:
    """打印信号冲突日志（仅前 limit 条）。"""
    print("-" * 60)
    print(f"信号冲突日志: 共 {len(result.conflict_log)} 个分歧日"
          f"（展示前 {min(limit, len(result.conflict_log))} 条）")
    for entry in result.conflict_log[:limit]:
        dirs = ", ".join(
            f"{n}:{'买' if d == 1 else '卖' if d == -1 else '观'}"
            for n, d in entry["directions"].items()
        )
        print(f"  {entry['date']}  net_vote={entry['net_vote']:+.3f}  "
              f"参考结论={entry['decision']:<4}  [{dirs}]")


def main() -> None:
    setup_logging(log_dir=str(PROJECT_ROOT / "logs"), level="WARNING")

    print("=" * 60)
    print("多策略组合回测示例（等权 / 3 策略 / mock 数据）")
    print("=" * 60)

    # 1. 获取数据
    print("\n[1/3] 获取 600519.SH mock 行情数据...")
    data = fetch_mock_data()

    # 2. 运行多策略组合回测
    print("\n[2/3] 运行等权多策略组合回测...")
    sp = StrategyPortfolio(STRATEGIES, initial_capital=INITIAL_CAPITAL)
    print(f"  策略: {STRATEGIES}")
    print(f"  权重: 等权 {[round(w, 3) for w in sp.strategy_weights.values()]}")
    result = sp.run(data, symbol=SYMBOL)

    # 3. 打印结果
    print("\n[3/3] 回测结果:")
    print_portfolio_metrics(result.portfolio_metrics)
    print()
    print_strategy_details(result)
    print()
    print_contribution(result)
    print()
    print_conflict_log(result)

    print("\n完成！注意: 回测基于 mock 数据，不代表真实表现。")


if __name__ == "__main__":
    main()
