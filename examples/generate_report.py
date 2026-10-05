#!/usr/bin/env python3
"""回测报告 HTML 导出示例。

使用离线 mock 行情数据运行一次双均线策略回测，
并调用 :class:`backtest.report_generator.ReportGenerator` 生成自包含深色主题 HTML 报告。

运行方式:
    cd quant_trading_system
    python3 examples/generate_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import BacktestEngine  # noqa: E402
from backtest.report_generator import ReportGenerator  # noqa: E402
from data.data_fetcher import DataFetcher  # noqa: E402
from strategies.ma_cross import MACrossStrategy  # noqa: E402


def build_mock_data(symbol: str = "600519.SH"):
    """构造离线 mock 日线行情数据。

    Returns:
        pd.DataFrame: 含 open/high/low/close/volume 列、日期索引的 K 线。
    """
    fetcher = DataFetcher(use_mock=True, cache_dir=str(PROJECT_ROOT / "cache"))
    df = fetcher.get_klines(
        symbol, period="1d", count=250,
        start_date="2024-01-02", use_cache=False,
    )
    print(f"mock 行情: {len(df)} 条, {df.index[0].date()} ~ {df.index[-1].date()}")
    return df


def main() -> None:
    """主函数：运行回测并生成 HTML 报告。"""
    symbol = "600519.SH"
    df = build_mock_data(symbol)

    # 双均线策略（使用默认参数）
    strategy = MACrossStrategy({"fast_period": 5, "slow_period": 20})

    # 回测引擎
    engine = BacktestEngine(
        initial_capital=1_000_000.0,
        commission_rate=0.00025,
        stamp_tax_rate=0.0005,
        slippage_rate=0.001,
        risk_free_rate=0.02,
        trading_days=252,
    )
    print("正在运行回测...")
    result = engine.run(df, strategy, symbol=symbol)
    print(f"回测完成: {len(result.trades)} 笔交易, "
          f"累计收益率={result.metrics.get('累计收益率', 0):.2%}")

    # 生成 HTML 报告
    output_path = PROJECT_ROOT / "output" / "reports" / "example_report.html"
    params = {
        "标的": symbol,
        "策略": "ma_cross (双均线)",
        "初始资金": "1,000,000",
        "手续费率": "0.025%",
        "滑点率": "0.1%",
        "回测区间": f"{df.index[0].date()} ~ {df.index[-1].date()}",
    }
    report_path = ReportGenerator().generate_html_report(
        result, str(output_path),
        title="回测报告 - 600519.SH 双均线策略",
        params=params,
    )

    print(f"\nHTML 报告已生成: {report_path}")
    print("用浏览器打开该文件即可查看。")


if __name__ == "__main__":
    main()
