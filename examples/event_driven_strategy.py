#!/usr/bin/env python3
"""事件驱动策略可运行示例。

演示三步：
    1. 从 K线检测市场事件（涨停 / 成交量异常 / 价格跳空）
    2. 对某类事件做事件研究（CAR 曲线 + t 检验）
    3. 用事件驱动策略模式跑回测

运行方式:
    cd quant_trading_system
    python examples/event_driven_strategy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtest.engine import BacktestEngine
from config import load_config
from data.data_fetcher import DataFetcher
from monitoring.monitor import setup_logging
from strategies.event_driven import EventDrivenStrategy

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


def fetch_data(symbol: str, count: int = 500) -> pd.DataFrame:
    """获取日线数据（mock 降级）。"""
    fetcher = DataFetcher(use_mock=True,
                          cache_dir=str(PROJECT_ROOT / "cache"))
    df = fetcher.get_klines(symbol, period="1d", count=count,
                            adjust="qfq", use_cache=False)
    print(f"获取 {symbol} 数据 {len(df)} 条: "
          f"{df.index[0].date()} ~ {df.index[-1].date()}")
    return df


def step1_detect_events(df: pd.DataFrame, symbol: str) -> list:
    """第一步：检测事件。"""
    print("\n" + "=" * 60)
    print("步骤 1: 事件检测")
    print("=" * 60)
    strat = EventDrivenStrategy()
    events = strat.detect_events(df, symbol=symbol)
    by_type: dict = {}
    for e in events:
        by_type.setdefault(e["event_type"], []).append(e)
    for etype, lst in by_type.items():
        print(f"  {etype}: {len(lst)} 次")
        for e in lst[:3]:
            print(f"    - {pd.Timestamp(e['date']).date()}  "
                  f"metadata={e['metadata']}")
    if len(by_type.get("limit_up", [])) > 3:
        print(f"    ... 共 {len(by_type['limit_up'])} 次")
    return events


def step2_event_study(df: pd.DataFrame, events: list,
                      event_type: str = "limit_up") -> dict:
    """第二步：事件研究。"""
    print("\n" + "=" * 60)
    print(f"步骤 2: 事件研究（事件类型 = {event_type}）")
    print("=" * 60)
    strat = EventDrivenStrategy()
    sub_events = [e for e in events if e["event_type"] == event_type]
    result = strat.event_study(df, sub_events, window=20)
    print(f"  有效事件数: {result['event_count']}")
    print(f"  t 统计量: {result['t_statistic']:.4f}")
    print(f"  p 值:     {result['p_value']:.4f}")
    if not result["car_series"].empty:
        s = result["car_series"]
        print(f"  CAR[-20]={s.iloc[0]:.4f}  "
              f"CAR[0]={s.loc[0]:.4f}  "
              f"CAR[+20]={s.iloc[-1]:.4f}")
        # 绘图
        out_dir = PROJECT_ROOT / "output"
        out_dir.mkdir(exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(s.index, s.values, marker="o", linewidth=1.8,
                 label="平均 CAR")
        plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
        plt.axvline(0, color="red", linewidth=0.8, linestyle=":")
        plt.title(f"事件研究: {event_type} 累计异常收益 (n={result['event_count']})")
        plt.xlabel("相对事件日偏移 (交易日)")
        plt.ylabel("累计异常收益")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path = out_dir / f"event_study_{event_type}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  CAR 曲线已保存: {out_path}")
    return result


def step3_backtest(df: pd.DataFrame, symbol: str, mode: str) -> None:
    """第三步：回测。"""
    print("\n" + "=" * 60)
    print(f"步骤 3: 事件驱动策略回测 (mode={mode})")
    print("=" * 60)
    strat = EventDrivenStrategy({"mode": mode, "hold_days": 5})
    engine = BacktestEngine(initial_capital=1_000_000.0)
    result = engine.run(df, strat, symbol=symbol)
    print(f"  交易次数:   {result.metrics.get('交易次数', 0)}")
    print(f"  累计收益率: {result.metrics.get('累计收益率', 0) * 100:.2f}%")
    print(f"  最大回撤:   {result.metrics.get('最大回撤', 0) * 100:.2f}%")
    print(f"  夏普比率:   {result.metrics.get('夏普比率', 0):.3f}")


def main() -> None:
    setup_logging(log_dir=str(PROJECT_ROOT / "logs"), level="WARNING")
    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    symbol = "600519.SH"

    df = fetch_data(symbol, count=500)
    events = step1_detect_events(df, symbol)

    # 选择出现次数最多的事件类型做研究
    counts = {}
    for e in events:
        counts[e["event_type"]] = counts.get(e["event_type"], 0) + 1
    study_type = max(counts, key=lambda k: counts[k]) if counts else "limit_up"
    step2_event_study(df, events, event_type=study_type)

    # 用 PEAD 模式回测
    step3_backtest(df, symbol, mode=cfg.get("event_strategy", {})
                   .get("default_mode", "pead"))

    print("\n示例运行完成。")


if __name__ == "__main__":
    main()
