#!/usr/bin/env python3
"""组合级回表示例：3 只 A 股蓝筹等权组合，双均线策略。

使用 mock 行情数据（无需网络 / API Key），运行组合回测并输出：
  - 组合级绩效 + 单标的明细对比表格（打印 + CSV）
  - 组合净值曲线 vs 各标的净值曲线对比图（保存到 output/）

运行方式:
    cd quant_trading_system
    python examples/run_portfolio_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境
import matplotlib.pyplot as plt
import pandas as pd

from backtest.portfolio_engine import PortfolioBacktestEngine
from data.data_fetcher import _generate_mock_klines
from monitoring.monitor import setup_logging
from strategies.ma_cross import MACrossStrategy

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


SYMBOLS = ["600519.SH", "300750.SZ", "002594.SZ"]
INITIAL_CAPITAL = 1_000_000.0
# mock 基准价：A 股高价蓝筹（如茅台 ~1600 元）在等权拆分资金后，
# 单笔 20% 仓位不足买 1 手（100 股），会导致 0 交易、净值走平。
# 这里用亲民价格演示，确保策略信号能真实成交。
MOCK_BASE_PRICE = 25.0


def fetch_mock_data() -> dict:
    """获取 3 只标的的 mock 行情数据（确定性，无需网络 / API Key）。"""
    data = {}
    for sym in SYMBOLS:
        df = _generate_mock_klines(
            sym, period="1d", count=250,
            start_date="2024-01-02", base_price=MOCK_BASE_PRICE,
        )
        data[sym] = df
        print(f"  {sym}: {len(df)} 条, "
              f"{df.index[0].date()} ~ {df.index[-1].date()}, "
              f"收盘 {df['close'].iloc[0]:.2f} -> {df['close'].iloc[-1]:.2f}")
    return data


def build_comparison_table(result) -> pd.DataFrame:
    """构造组合 vs 单标的绩效对比表。"""
    rows = {}
    # 组合行
    rows["组合(等权)"] = result.portfolio_metrics
    # 各标的行
    for sym, metrics in result.symbol_metrics.items():
        rows[sym] = metrics
    df = pd.DataFrame(rows).T
    # 格式化展示列
    display = df.copy()
    for col in ["累计收益率", "年化收益率", "最大回撤", "胜率"]:
        if col in display.columns:
            display[col] = (display[col] * 100).map(lambda x: f"{x:.2f}%")
    for col in ["夏普比率", "盈亏比"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.3f}" if col == "夏普比率" else f"{x:.2f}")
    if "交易次数" in display.columns:
        display["交易次数"] = display["交易次数"].map(lambda x: f"{int(x)}")
    return display


def plot_curves(result, output_path: str) -> None:
    """绘制组合净值 + 各标的净值 + 基准对比图（归一化为起始=100）。"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # 组合净值
    eq = result.portfolio_equity_curve
    ax.plot(eq.index, eq / eq.iloc[0] * 100,
            label="组合(等权)", color="#2C3E50", linewidth=2.6)

    # 基准
    bench = result.benchmark_curve
    if len(bench) > 0:
        ax.plot(bench.index, bench / bench.iloc[0] * 100,
                label="基准(等权买入持有)", color="#7F8C8D",
                linewidth=1.4, linestyle="--")

    # 各标的净值
    palette = ["#E74C3C", "#2E86C1", "#27AE60", "#F39C12"]
    for i, (sym, res) in enumerate(result.symbol_results.items()):
        s = res.equity_curve
        ax.plot(s.index, s / s.iloc[0] * 100,
                label=f"{sym}", color=palette[i % len(palette)],
                linewidth=1.3, alpha=0.85)

    ax.set_title("组合回测净值曲线对比 (归一化, 起始=100)", fontsize=15, fontweight="bold")
    ax.set_ylabel("净值")
    ax.set_xlabel("日期")
    ax.legend(loc="upper left", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color="black", linewidth=0.5, linestyle=":")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"净值曲线图已保存: {output_path}")


def main() -> None:
    setup_logging(log_dir=str(PROJECT_ROOT / "logs"), level="WARNING")
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("组合级回测示例（等权分配 / 双均线策略 / mock 数据）")
    print("=" * 60)

    # 1. 获取数据
    print("\n[1/4] 获取 3 只标的 mock 行情数据...")
    data = fetch_mock_data()

    # 2. 运行组合回测
    print("\n[2/4] 运行等权组合回测...")
    strategy = MACrossStrategy({"fast_period": 5, "slow_period": 20})
    engine = PortfolioBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        allocation_method="equal",
    )
    result = engine.run(data, strategy)

    # 3. 打印组合绩效 + 单标的对比
    print("\n[3/4] 组合绩效与单标的对比:")
    print("-" * 60)
    print(f"实际资金分配权重: "
          + ", ".join(f"{k}={v:.1%}" for k, v in result.symbol_weights.items()))
    print(f"组合总交易笔数: {len(result.all_trades)}")
    table = build_comparison_table(result)
    print("\n绩效对比表:")
    print(table.to_string())
    table.to_csv(output_dir / "portfolio_metrics_comparison.csv", encoding="utf-8-sig")
    print(f"\n对比表已保存: {output_dir / 'portfolio_metrics_comparison.csv'}")

    # 4. 绘图
    print("\n[4/4] 绘制净值曲线对比图...")
    plot_curves(result, str(output_dir / "portfolio_equity_curve.png"))

    print("\n完成！结果保存在 output/ 目录下。")
    print("注意: 回测基于 mock 数据，不代表真实表现。")


if __name__ == "__main__":
    main()
