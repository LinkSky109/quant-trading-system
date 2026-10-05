#!/usr/bin/env python3
"""策略排行榜 + 多策略信号聚合示例。

演示 :class:`strategies.strategy_leaderboard.StrategyLeaderboard`：
  1. 对同一标的批量回测 6 个策略，按综合得分排名
  2. 基于排名权重聚合各策略最新信号，给出 buy / sell / hold 综合建议

使用 mock 行情数据（无需网络 / API Key）。

运行方式:
    cd quant_trading_system
    python examples/strategy_leaderboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.data_fetcher import _generate_mock_klines
from monitoring.monitor import setup_logging
from strategies.strategy_leaderboard import StrategyLeaderboard

SYMBOL = "600519.SH"
START_DATE = "2024-01-02"
END_DATE = "2024-12-31"
INITIAL_CAPITAL = 1_000_000.0
# mock 基准价：低价确保策略信号能真实成交
MOCK_BASE_PRICE = 25.0


class MockFetcher:
    """不依赖网络的数据获取器。"""

    def get_klines(self, symbol, count=250, start_date=None, end_date=None):
        df = _generate_mock_klines(
            symbol, period="1d", count=400,
            start_date="2024-01-02", base_price=MOCK_BASE_PRICE,
        )
        if start_date:
            df = df[df.index >= __import__("pandas").Timestamp(start_date)]
        if end_date:
            df = df[df.index <= __import__("pandas").Timestamp(end_date)]
        return df


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def print_leaderboard(board) -> None:
    """打印策略排行榜。"""
    print("-" * 78)
    print(f"{'排名':<4}{'策略':<20}{'综合分':>8}{'夏普':>8}{'累计收益':>10}"
          f"{'最大回撤':>10}{'胜率':>8}")
    for r in board:
        print(
            f"{r['rank']:<4}{r['strategy']:<20}{r['score']:>8.3f}"
            f"{r['sharpe']:>8.2f}{_fmt_pct(r['total_return']):>10}"
            f"{_fmt_pct(r['max_drawdown']):>10}{_fmt_pct(r['win_rate']):>8}"
        )


def print_aggregate(agg: dict) -> None:
    """打印聚合信号。"""
    print("-" * 78)
    print(f"综合信号: {agg['aggregate_signal'].upper()}  "
          f"(net_score={agg['net_score']:+.2f}, threshold={agg['threshold']})")
    print(f"  加权买入 {agg['weighted_buy']:.2f} / 加权卖出 {agg['weighted_sell']:.2f}")
    print(f"  {'策略':<20}{'信号':>6}{'置信度':>8}{'权重':>8}{'近20日收益':>12}")
    for d in agg["strategy_details"]:
        print(
            f"  {d['strategy']:<20}{d['signal']:>6}{d['confidence']:>8.2f}"
            f"{d['weight']:>8.2f}{_fmt_pct(d['recent_return']):>12}"
        )


def main() -> None:
    setup_logging(log_dir=str(PROJECT_ROOT / "logs"), level="WARNING")

    print("=" * 78)
    print("策略排行榜 + 多策略信号聚合（mock 数据）")
    print("=" * 78)

    fetcher = MockFetcher()
    lb = StrategyLeaderboard(data_fetcher=fetcher)

    # 1. 回测 + 排名
    print(f"\n[1/2] 回测 {SYMBOL} ({START_DATE} ~ {END_DATE}) 的 6 个策略...")
    board = lb.get_leaderboard(SYMBOL, START_DATE, END_DATE)
    print_leaderboard(board)

    # 2. 信号聚合（复用同一份数据）
    print(f"\n[2/2] 聚合各策略最新信号（按排名加权 + 近期表现动态调整）...")
    df = fetcher.get_klines(SYMBOL, start_date=START_DATE, end_date=END_DATE)
    agg = lb.aggregate_signals(SYMBOL, df=df, threshold=0.2)
    print_aggregate(agg)

    print("\n完成！注意: 回测基于 mock 数据，不代表真实表现。")


if __name__ == "__main__":
    main()
