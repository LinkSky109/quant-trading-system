#!/usr/bin/env python3
"""多周期回测：6种时间窗口 × 2种K线频率。

时间窗口: 半年 / 3个月 / 1个月 / t-7 / t-3 / t-1
K线频率: 日线 / 5分钟线
策略: 双均线 MA5/MA20（信号shift(1)防未来函数）
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from config import load_config
from data.data_cleaner import clean_klines
from risk.risk_manager import RiskManager
from strategies.ma_cross import MACrossStrategy
from utils.indicators import add_indicators

logging.basicConfig(level=logging.WARNING)
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 数据获取（含重试 + 分钟线mock修复）
# ---------------------------------------------------------------------------

def _retry_fetch(func, retries=3, delay=4):
    """带重试的数据获取。"""
    import time
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            if i < retries - 1:
                print(f"    第{i+1}次失败({e})，{delay}s后重试...")
                time.sleep(delay)
            else:
                raise


def _generate_minute_mock(symbol: str, count: int = 3000, period_min: int = 5) -> pd.DataFrame:
    """生成模拟分钟线数据（5分钟间隔，A股交易时段，含震荡交叉）。"""
    np.random.seed(hash(symbol) % (2**31))
    timestamps = []
    current = datetime.now() - timedelta(days=60)
    current = current.replace(hour=9, minute=30, second=0, microsecond=0)
    while len(timestamps) < count:
        is_morning = (current.hour == 9 and current.minute >= 30) or current.hour == 10 or (current.hour == 11 and current.minute <= 30)
        is_afternoon = current.hour >= 13 and (current.hour < 15 or (current.hour == 15 and current.minute == 0))
        if is_morning or is_afternoon:
            timestamps.append(current)
        current += timedelta(minutes=period_min)
        if current.hour > 15 or (current.hour == 11 and current.minute > 30):
            if current.hour >= 15:
                current += timedelta(days=1)
                current = current.replace(hour=9, minute=30)
            elif current.hour == 11:
                current = current.replace(hour=13, minute=0)
        if current.weekday() >= 5:
            current += timedelta(days=7 - current.weekday())
            current = current.replace(hour=9, minute=30)

    timestamps = timestamps[:count]
    daily_vol = 0.25 / np.sqrt(252)
    bar_vol = daily_vol / np.sqrt(48)
    returns = np.random.normal(0, bar_vol, count)
    returns += 0.0003 * np.sin(np.linspace(0, 25 * np.pi, count))  # 震荡项增加交叉
    close = 1700.0 * np.cumprod(1 + returns)
    open_ = close * (1 + np.random.normal(0, bar_vol * 0.3, count))
    upper = np.maximum(open_, close)
    lower = np.minimum(open_, close)
    high = upper * (1 + np.abs(np.random.normal(0, bar_vol * 0.5, count)))
    low = lower * (1 - np.abs(np.random.normal(0, bar_vol * 0.5, count)))
    volume = np.random.lognormal(12, 0.6, count).astype(int)

    return pd.DataFrame({
        "open": open_.round(2), "high": high.round(2),
        "low": low.round(2), "close": close.round(2),
        "volume": volume, "amount": (close * volume).round(2),
    }, index=pd.DatetimeIndex(timestamps, name="date"))


def _standardize_akshare(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """标准化 akshare 返回的列名。"""
    col_map = {date_col: "date", "开盘": "open", "收盘": "close",
               "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"}
    df = df.rename(columns=col_map)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df[["open", "high", "low", "close", "volume", "amount"]]


def fetch_daily(symbol: str = "600519") -> pd.DataFrame:
    """获取日线数据（QuantDash 真实数据，失败降级mock）。"""
    from data.data_fetcher import DataFetcher
    try:
        fetcher = DataFetcher(api_key="sk_292e39cbd4884df0bc9bac0d05d3686d", use_mock=False)
        df = fetcher.get_klines(f"{symbol}.SH", period="1d", count=300, use_cache=False)
        days_old = (datetime.now() - df.index[-1].to_pydatetime()).days
        if days_old > 7:
            print(f"  日线数据过期({days_old}天)，使用mock")
            raise ValueError(f"data stale: {days_old} days old")
        print(f"  日线(QuantDash): {len(df)} 条, {df.index[0].date()} ~ {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"  日线使用mock ({e})")
        from data.data_fetcher import _generate_mock_klines
        df = _generate_mock_klines(f"{symbol}.SH", count=300,
                                   start_date=(datetime.now()-timedelta(days=420)).strftime("%Y-%m-%d"))
        print(f"  日线(mock): {len(df)} 条, {df.index[0].date()} ~ {df.index[-1].date()}")
        return df


def fetch_minute(symbol: str = "600519", period: str = "5") -> pd.DataFrame:
    """获取分钟线数据（含重试）。"""
    def _fetch():
        import akshare as ak
        df = ak.stock_zh_a_hist_min_em(
            symbol=symbol, period=period,
            start_date=(datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d 09:30:00"),
            end_date=datetime.now().strftime("%Y-%m-%d 15:00:00"), adjust="qfq")
        return _standardize_akshare(df, "时间")
    try:
        df = _retry_fetch(_fetch)
        print(f"  {period}分钟线(真实): {len(df)} 条, {df.index[0]} ~ {df.index[-1]}")
        return df
    except Exception as e:
        print(f"  分钟线获取失败({e})，使用mock")
        df = _generate_minute_mock(symbol, count=3000, period_min=int(period))
        print(f"  {period}分钟线(mock): {len(df)} 条, {df.index[0]} ~ {df.index[-1]}")
        return df


# ---------------------------------------------------------------------------
# 时间窗口定义
# ---------------------------------------------------------------------------

def get_windows() -> list[dict]:
    """定义6种时间窗口。"""
    now = datetime.now()
    return [
        {"name": "半年", "days": 180, "label": "6M"},
        {"name": "3个月", "days": 90, "label": "3M"},
        {"name": "1个月", "days": 30, "label": "1M"},
        {"name": "t-7", "days": 7, "label": "7D"},
        {"name": "t-3", "days": 3, "label": "3D"},
        {"name": "t-1", "days": 1, "label": "1D"},
    ]


def slice_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """按最近N天截取数据。自动识别日线/分钟线。"""
    if df.empty or len(df) < 2:
        return df
    # 判断频率：平均间隔 > 12小时 视为日线
    avg_interval = (df.index[-1] - df.index[0]) / len(df)
    is_daily = avg_interval > pd.Timedelta(hours=12)

    if is_daily:
        # 日线：取最后N个交易日
        return df.tail(max(days, 5))
    else:
        # 分钟线：按自然日截取
        cutoff = df.index[-1] - pd.Timedelta(days=days)
        return df[df.index >= cutoff]


# ---------------------------------------------------------------------------
# 单次回测
# ---------------------------------------------------------------------------

def run_single_backtest(df: pd.DataFrame, symbol: str, cfg: dict) -> dict | None:
    """运行单次回测，返回指标字典或None（数据不足）。"""
    min_bars = 25  # MA20 + shift 需要至少21根K线
    if len(df) < min_bars:
        return None

    try:
        df = clean_klines(df)
    except Exception:
        pass

    if len(df) < min_bars:
        return None

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

    strategy = MACrossStrategy({
        "fast_period": cfg["strategies"]["ma_cross"]["fast_period"],
        "slow_period": cfg["strategies"]["ma_cross"]["slow_period"],
    })

    bt_cfg = cfg["backtest"]
    engine = BacktestEngine(
        initial_capital=bt_cfg["initial_capital"],
        commission_rate=bt_cfg["commission_rate"],
        stamp_tax_rate=bt_cfg["stamp_tax_rate"],
        slippage_rate=bt_cfg["slippage_rate"],
        risk_free_rate=bt_cfg["risk_free_rate"],
        trading_days=bt_cfg["trading_days_per_year"],
        risk_manager=risk_mgr,
        jev_engine=None,
    )

    result = engine.run(df, strategy, symbol=symbol)
    m = result.metrics
    return {
        "累计收益率": m["累计收益率"],
        "年化收益率": m["年化收益率"],
        "最大回撤": m["最大回撤"],
        "夏普比率": m["夏普比率"],
        "胜率": m["胜率"],
        "盈亏比": m["盈亏比"],
        "交易次数": int(m["交易次数"]),
        "K线数": len(df),
        "起始日": str(df.index[0].date()),
        "结束日": str(df.index[-1].date()),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    symbol = "600519"
    windows = get_windows()
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("多周期回测：6种时间窗口 × 日线/5分钟线")
    print("=" * 60)

    # 获取数据
    print("\n[1/3] 获取数据...")
    df_daily = fetch_daily(symbol)
    df_minute = fetch_minute(symbol, "5")

    # 运行所有组合
    print("\n[2/3] 运行回测...")
    results = []
    for win in windows:
        for freq_name, df in [("日线", df_daily), ("5分钟线", df_minute)]:
            sliced = slice_window(df, win["days"])
            print(f"  {win['name']:4s} × {freq_name:5s}: {len(sliced):4d}根K线 ", end="")
            r = run_single_backtest(sliced, f"{symbol}.SH", cfg)
            if r is None:
                print("→ 数据不足，跳过")
                results.append({
                    "时间窗口": win["name"], "K线频率": freq_name,
                    "累计收益率": None, "年化收益率": None, "最大回撤": None,
                    "夏普比率": None, "胜率": None, "盈亏比": None,
                    "交易次数": 0, "K线数": len(sliced),
                    "起始日": "-", "结束日": "-", "状态": "数据不足",
                })
            else:
                print(f"→ 收益 {r['累计收益率']*100:.2f}%, 交易 {r['交易次数']}次")
                results.append({
                    "时间窗口": win["name"], "K线频率": freq_name,
                    **r, "状态": "正常",
                })

    # 结果表格
    df_result = pd.DataFrame(results)
    csv_path = output_dir / "multi_period_backtest.csv"
    df_result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  结果已保存: {csv_path}")

    # 打印格式化表格
    print("\n" + "=" * 80)
    print("回测结果汇总")
    print("=" * 80)
    display_cols = ["时间窗口", "K线频率", "K线数", "累计收益率", "最大回撤", "夏普比率", "胜率", "交易次数", "状态"]
    fmt_df = df_result[display_cols].copy()
    for col in ["累计收益率", "最大回撤", "胜率"]:
        fmt_df[col] = fmt_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-")
    fmt_df["夏普比率"] = fmt_df["夏普比率"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    print(fmt_df.to_string(index=False))
    print("=" * 80)

    # 可视化
    print("\n[3/3] 生成图表...")
    plot_results(df_result, output_dir)

    # 保存JSON
    json_path = output_dir / "multi_period_backtest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"  JSON已保存: {json_path}")

    print("\n完成！")


def plot_results(df_result: pd.DataFrame, output_dir: Path):
    """生成对比图表。"""
    valid = df_result[df_result["状态"] == "正常"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("多周期回测绩效对比（双均线 MA5/MA20）", fontsize=15, fontweight="bold")

    windows_order = ["半年", "3个月", "1个月", "t-7", "t-3", "t-1"]
    freqs = ["日线", "5分钟线"]
    colors = {"日线": "#4da6ff", "5分钟线": "#00d4aa"}

    # 1. 累计收益率
    ax = axes[0, 0]
    x = np.arange(len(windows_order))
    width = 0.35
    for i, freq in enumerate(freqs):
        vals = []
        for w in windows_order:
            row = valid[(valid["时间窗口"] == w) & (valid["K线频率"] == freq)]
            vals.append(row["累计收益率"].values[0] * 100 if len(row) else 0)
        bars = ax.bar(x + i * width - width/2, vals, width, label=freq, color=colors[freq], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v != 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.1 if v >= 0 else -0.4),
                        f"{v:.1f}%", ha="center", fontsize=8, color=colors[freq])
    ax.set_xticks(x)
    ax.set_xticklabels(windows_order)
    ax.set_ylabel("累计收益率 (%)")
    ax.set_title("累计收益率对比")
    ax.legend()
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    # 2. 最大回撤
    ax = axes[0, 1]
    for i, freq in enumerate(freqs):
        vals = []
        for w in windows_order:
            row = valid[(valid["时间窗口"] == w) & (valid["K线频率"] == freq)]
            vals.append(abs(row["最大回撤"].values[0]) * 100 if len(row) else 0)
        bars = ax.bar(x + i * width - width/2, vals, width, label=freq, color=colors[freq], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(windows_order)
    ax.set_ylabel("最大回撤 (%)")
    ax.set_title("最大回撤对比（越小越好）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 3. 夏普比率
    ax = axes[1, 0]
    for i, freq in enumerate(freqs):
        vals = []
        for w in windows_order:
            row = valid[(valid["时间窗口"] == w) & (valid["K线频率"] == freq)]
            vals.append(row["夏普比率"].values[0] if len(row) else 0)
        bars = ax.bar(x + i * width - width/2, vals, width, label=freq, color=colors[freq], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v != 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.02 if v >= 0 else -0.08),
                        f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(windows_order)
    ax.set_ylabel("夏普比率")
    ax.set_title("夏普比率对比")
    ax.legend()
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    # 4. 胜率 & 交易次数
    ax = axes[1, 1]
    ax2 = ax.twinx()
    for i, freq in enumerate(freqs):
        win_rates = []
        trade_counts = []
        for w in windows_order:
            row = valid[(valid["时间窗口"] == w) & (valid["K线频率"] == freq)]
            win_rates.append(row["胜率"].values[0] * 100 if len(row) else 0)
            trade_counts.append(row["交易次数"].values[0] if len(row) else 0)
        ax.bar(x + i * width - width/2, win_rates, width, label=f"{freq}胜率", color=colors[freq], alpha=0.7)
        ax2.plot(x + i * width - width/2, trade_counts, "o-", color=colors[freq],
                label=f"{freq}交易次数", linewidth=1.5, markersize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(windows_order)
    ax.set_ylabel("胜率 (%)")
    ax2.set_ylabel("交易次数")
    ax.set_title("胜率与交易次数")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart_path = output_dir / "multi_period_comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表已保存: {chart_path}")


if __name__ == "__main__":
    main()
