#!/usr/bin/env python3
"""专业版回测报告 HTML 生成示例。

构造一个**离线模拟**的 :class:`BacktestResult`（含净值曲线、基准、日收益率、
平仓交易记录），调用 :class:`backtest.report_generator.ReportGenerator` 生成
包含以下全部章节的自包含深色主题 HTML 报告：

- 封面（标的 / 策略 / 区间 / 初始资金 / 生成时间）
- 绩效指标卡片（含索提诺 / 卡玛比率）
- 净值曲线、回撤曲线（ECharts）
- 月度收益热力图（ECharts heatmap）
- 持仓分析（时长分桶柱状图 + 最大/平均盈亏）
- 风险指标（VaR(95%) / CVaR(95%) / 最大连续亏损天数）
- 信号分析（reason 分布 + 各信号胜率）
- 交易明细表、参数说明

运行方式::

    cd quant_trading_system
    python3 examples/generate_backtest_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import BacktestResult, Trade  # noqa: E402
from backtest.report_generator import ReportGenerator  # noqa: E402


# ---------------------------------------------------------------------------
# 模拟数据构造
# ---------------------------------------------------------------------------
def _build_mock_equity(n_days: int = 504, seed: int = 42) -> pd.Series:
    """构造一条带漂移与回撤的模拟净值曲线。

    Args:
        n_days: 交易日数（约 2 年）。
        seed: 随机种子，保证可复现。

    Returns:
        pd.Series，index 为工作日日期，值为账户净值（初始 1,000,000）。
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    # 日收益：正漂移 0.04% + 波动 1.2%
    daily_ret = rng.normal(0.0004, 0.012, n_days)
    # 人为在中段插入一段下跌（模拟 3 个月的回撤期）
    crash_start = n_days // 3
    crash_end = crash_start + 60
    daily_ret[crash_start:crash_end] -= 0.006
    equity = 1_000_000.0 * np.cumprod(1.0 + daily_ret)
    return pd.Series(equity, index=dates, name="equity")


def _build_mock_trades(dates: pd.DatetimeIndex) -> list[Trade]:
    """构造一组模拟平仓交易（含开仓 entry_date，便于持仓分析）。

    Returns:
        Trade 列表，含若干买/卖对，reason 覆盖多种信号类型。
    """
    trades: list[Trade] = []
    # (buy_idx, sell_idx, reason_buy, reason_sell, pnl)
    plan = [
        (10, 25, "金叉买入", "止盈(5%)", 5200.0),
        (30, 38, "突破买入", "止损(-3%)", -1800.0),
        (50, 90, "金叉买入", "死叉卖出", 3100.0),
        (120, 125, "突破买入", "止损(-3%)", -900.0),
        (150, 200, "金叉买入", "止盈(5%)", 8700.0),
        (250, 260, "突破买入", "死叉卖出", -1200.0),
        (300, 360, "金叉买入", "止盈(5%)", 12400.0),
        (400, 410, "突破买入", "止损(-3%)", -2100.0),
    ]
    for buy_i, sell_i, r_buy, r_sell, pnl in plan:
        entry = dates[buy_i]
        exit_ = dates[sell_i]
        trades.append(Trade(
            date=entry, symbol="600519.SH", action="buy",
            price=1700.0, shares=100, amount=170000.0,
            commission=42.5, stamp_tax=0.0, slippage_cost=170.0,
            pnl=None, reason=r_buy, entry_date=entry,
        ))
        # 卖价按盈亏反推一个合理值
        sell_price = 1700.0 + pnl / 100.0
        trades.append(Trade(
            date=exit_, symbol="600519.SH", action="sell",
            price=sell_price, shares=100, amount=sell_price * 100,
            commission=44.5, stamp_tax=sell_price * 100 * 0.0005,
            slippage_cost=178.0, pnl=pnl, reason=r_sell, entry_date=entry,
        ))
    return trades


def _build_mock_result() -> BacktestResult:
    """组装一个完整的模拟 BacktestResult。"""
    equity = _build_mock_equity()
    benchmark = equity * 0.97 + np.linspace(0, 30000, len(equity))  # 基准略弱
    trades = _build_mock_trades(equity.index)

    # 手工计算一组核心指标（仅用于卡片展示，报告内部会再算 Sortino/Calmar）
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    peak = equity.cummax()
    mdd = float(((equity - peak) / peak).min())
    closed_pnls = [t.pnl for t in trades if t.pnl is not None]
    wins = [p for p in closed_pnls if p > 0]
    losses = [p for p in closed_pnls if p < 0]

    metrics = {
        "累计收益率": total_return,
        "年化收益率": float((1 + total_return) ** (252 / len(equity)) - 1),
        "最大回撤": mdd,
        "夏普比率": 1.15,
        "胜率": len(wins) / len(closed_pnls) if closed_pnls else 0.0,
        "盈亏比": (np.mean(wins) / abs(np.mean(losses))) if wins and losses else 0.0,
        "交易次数": len(closed_pnls),
        "总盈利": float(sum(wins)),
        "总亏损": float(abs(sum(losses))),
    }

    return BacktestResult(
        equity_curve=equity,
        benchmark_curve=pd.Series(benchmark.values, index=equity.index, name="benchmark"),
        trades=trades,
        metrics=metrics,
        metrics_df=pd.DataFrame(),
        daily_returns=equity.pct_change().fillna(0.0),
        positions_history=[],
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main() -> None:
    """构造模拟结果并生成完整 HTML 报告到 output/reports/。"""
    result = _build_mock_result()

    output_dir = PROJECT_ROOT / "output" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "example_pro_report.html"

    params = {
        "标的": "600519.SH",
        "策略": "ma_cross (双均线) + 模拟数据",
        "开始日期": str(result.equity_curve.index[0].date()),
        "结束日期": str(result.equity_curve.index[-1].date()),
        "初始资金": "1,000,000",
        "手续费率": "0.025%",
        "滑点率": "0.1%",
        "数据来源": "离线模拟（np.random）",
    }

    report_path = ReportGenerator().generate_html_report(
        result,
        str(output_path),
        title="专业回测报告 - 600519.SH 双均线策略",
        params=params,
    )

    print(f"模拟回测完成: {len(result.trades)} 笔交易, "
          f"累计收益={result.metrics['累计收益率']:.2%}, "
          f"最大回撤={result.metrics['最大回撤']:.2%}")
    print(f"\n专业 HTML 报告已生成: {report_path}")
    print("用浏览器打开该文件即可查看（含月度热力图/持仓/风险/信号分析）。")


if __name__ == "__main__":
    main()
