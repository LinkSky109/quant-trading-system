#!/usr/bin/env python3
"""最小可运行回测示例：贵州茅台 600519.SH，2024年数据。

对比"有 Jev 过滤"和"无 Jev 过滤"的绩效差异，
输出绩效对比表格和净值曲线图。

运行方式:
    cd quant_trading_system
    python examples/run_backtest.py
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

from backtest.engine import BacktestEngine
from config import load_config
from data.data_cleaner import clean_klines
from data.data_fetcher import DataFetcher, normalize_symbol
from jev.jev_engine import JevDecisionEngine
from monitoring.monitor import setup_logging
from risk.risk_manager import RiskManager
from strategies.ma_cross import MACrossStrategy

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


def fetch_maotai_data() -> pd.DataFrame:
    """获取贵州茅台 2024 年日线数据。

    优先使用 akshare 获取真实数据，失败则降级为 mock。
    """
    symbol = "600519.SH"
    print(f"正在获取 {symbol} 2024年日线数据...")

    # 尝试 akshare 真实数据
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(
            symbol="600519",
            period="daily",
            start_date="20240101",
            end_date="20241231",
            adjust="qfq",
        )
        # 统一列名
        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount",
        }
        df = df.rename(columns=col_map)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df[["open", "high", "low", "close", "volume", "amount"]]
        print(f"  akshare 真实数据: {len(df)} 条, {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"  价格范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
        return df
    except Exception as e:
        print(f"  akshare 获取失败 ({e})，使用 mock 数据")

    # 降级 mock
    fetcher = DataFetcher(use_mock=True, cache_dir=str(PROJECT_ROOT / "cache"))
    df = fetcher.get_klines(
        symbol, period="1d", count=250,
        start_date="2024-01-02", use_cache=False,
    )
    print(f"  mock 数据: {len(df)} 条")
    return df


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    use_jev: bool,
    cfg: dict,
) -> tuple:
    """运行单次回测。

    Returns:
        (BacktestResult, RiskManager)
    """
    # 风控
    risk_cfg = cfg["risk"]
    risk_mgr = RiskManager(
        single_stop_loss=risk_cfg["single_stop_loss"],
        single_take_profit=risk_cfg["single_take_profit"],
        max_drawdown_pause=risk_cfg["max_drawdown_pause"],
        max_position_per_symbol=risk_cfg["max_position_per_symbol"],
        max_total_position=risk_cfg["max_total_position"],
        daily_loss_limit=risk_cfg["daily_loss_limit"],
        initial_capital=cfg["backtest"]["initial_capital"],
    )

    # Jev 引擎
    jev_engine = None
    if use_jev:
        jev_cfg = cfg["jev"]
        jev_engine = JevDecisionEngine(
            base_url=jev_cfg["base_url"],
            confidence_threshold=jev_cfg["confidence_threshold"],
            mock_mode=jev_cfg["mock_mode"],
            audit_log_path=str(PROJECT_ROOT / "logs" / "jev_audit.jsonl"),
        )

    # 策略：双均线
    strat_cfg = cfg["strategies"]["ma_cross"]
    strategy = MACrossStrategy({
        "fast_period": strat_cfg["fast_period"],
        "slow_period": strat_cfg["slow_period"],
    })

    # 回测引擎
    bt_cfg = cfg["backtest"]
    engine = BacktestEngine(
        initial_capital=bt_cfg["initial_capital"],
        commission_rate=bt_cfg["commission_rate"],
        stamp_tax_rate=bt_cfg["stamp_tax_rate"],
        slippage_rate=bt_cfg["slippage_rate"],
        risk_free_rate=bt_cfg["risk_free_rate"],
        trading_days=bt_cfg["trading_days_per_year"],
        risk_manager=risk_mgr,
        jev_engine=jev_engine,
    )

    result = engine.run(df, strategy, symbol=symbol)
    return result, risk_mgr


def plot_equity_curves(
    result_no_jev,
    result_with_jev,
    symbol: str,
    output_path: str,
) -> None:
    """绘制净值曲线对比图。"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # 净值曲线
    ax = axes[0]
    # 归一化为起始资金=100
    eq_no = result_no_jev.equity_curve / result_no_jev.equity_curve.iloc[0] * 100
    eq_with = result_with_jev.equity_curve / result_with_jev.equity_curve.iloc[0] * 100
    bench = result_no_jev.benchmark_curve / result_no_jev.benchmark_curve.iloc[0] * 100

    ax.plot(eq_no.index, eq_no.values, label="无Jev过滤", color="#E74C3C", linewidth=1.8)
    ax.plot(eq_with.index, eq_with.values, label="有Jev过滤", color="#2E86C1", linewidth=1.8)
    ax.plot(bench.index, bench.values, label="基准(买入持有)", color="#7F8C8D", linewidth=1.2, linestyle="--")

    ax.set_title(f"{symbol} 回测净值曲线对比 (初始资金=100)", fontsize=14, fontweight="bold")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color="black", linewidth=0.5, linestyle=":")

    # 回撤曲线
    ax2 = axes[1]
    dd_no = (eq_no / eq_no.cummax() - 1) * 100
    dd_with = (eq_with / eq_with.cummax() - 1) * 100
    ax2.fill_between(dd_no.index, dd_no.values, 0, color="#E74C3C", alpha=0.2, label="无Jev回撤")
    ax2.fill_between(dd_with.index, dd_with.values, 0, color="#2E86C1", alpha=0.2, label="有Jev回撤")
    ax2.set_ylabel("回撤 (%)")
    ax2.set_xlabel("日期")
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"净值曲线图已保存: {output_path}")


def print_comparison(result_no_jev, result_with_jev) -> pd.DataFrame:
    """打印并返回绩效对比表格。"""
    comparison = pd.DataFrame({
        "无Jev过滤": result_no_jev.metrics_df.set_index("指标")["数值"],
        "有Jev过滤": result_with_jev.metrics_df.set_index("指标")["数值"],
    })

    print("\n" + "=" * 60)
    print("绩效对比")
    print("=" * 60)
    print(comparison.to_string())
    print("=" * 60)

    # 交易统计
    trades_no = [t for t in result_no_jev.trades if t.pnl is not None]
    trades_with = [t for t in result_with_jev.trades if t.pnl is not None]
    print(f"\n交易统计:")
    print(f"  无Jev: {len(trades_no)} 笔平仓, "
          f"盈利 {sum(1 for t in trades_no if t.pnl > 0)} 笔, "
          f"亏损 {sum(1 for t in trades_no if t.pnl < 0)} 笔")
    print(f"  有Jev: {len(trades_with)} 笔平仓, "
          f"盈利 {sum(1 for t in trades_with if t.pnl > 0)} 笔, "
          f"亏损 {sum(1 for t in trades_with if t.pnl < 0)} 笔")

    return comparison


def main():
    """主函数。"""
    # 初始化日志
    setup_logging(log_dir=str(PROJECT_ROOT / "logs"), level="WARNING")

    # 加载配置
    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    symbol = "600519.SH"

    # 获取数据
    df = fetch_maotai_data()
    df = clean_klines(df)

    # 输出目录
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    # 回测1：无 Jev 过滤
    print("\n--- 回测1: 无 Jev 过滤 ---")
    result_no_jev, risk_no = run_backtest(df, symbol, use_jev=False, cfg=cfg)

    # 回测2：有 Jev 过滤
    print("\n--- 回测2: 有 Jev 过滤 ---")
    result_with_jev, risk_with = run_backtest(df, symbol, use_jev=True, cfg=cfg)

    # 绩效对比
    comparison = print_comparison(result_no_jev, result_with_jev)

    # 保存对比表格
    comparison.to_csv(output_dir / "performance_comparison.csv", encoding="utf-8-sig")
    print(f"\n对比表格已保存: {output_dir / 'performance_comparison.csv'}")

    # 保存交易明细
    trades_df = pd.DataFrame([
        {**t.__dict__, "mode": "无Jev"} for t in result_no_jev.trades
    ] + [
        {**t.__dict__, "mode": "有Jev"} for t in result_with_jev.trades
    ])
    trades_df.to_csv(output_dir / "trade_details.csv", index=False, encoding="utf-8-sig")
    print(f"交易明细已保存: {output_dir / 'trade_details.csv'}")

    # 绘制净值曲线
    plot_equity_curves(
        result_no_jev, result_with_jev, symbol,
        str(output_dir / "equity_curve_comparison.png"),
    )

    print("\n回测完成！所有结果保存在 output/ 目录下。")
    print("\n注意: 回测结果不代表未来表现，实盘前需进行样本外测试和模拟盘验证。")


if __name__ == "__main__":
    main()
