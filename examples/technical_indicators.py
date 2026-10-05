#!/usr/bin/env python3
"""专业技术指标库演示脚本。

演示内容：
  1. 生成合成 OHLCV 行情（上涨段 + 下跌段）
  2. 按分类计算各类指标（趋势/震荡/成交量/波动率/形态）
  3. 通过 TechnicalIndicators 门面批量计算并合并
  4. 使用 IndicatorComboStrategy 进行回测

运行：
    cd quant_trading_system
    python examples/technical_indicators.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 允许从项目根目录直接导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from indicators import TechnicalIndicators  # noqa: E402
from strategies.indicator_combo import IndicatorComboStrategy  # noqa: E402


def make_demo_data(n_up: int = 80, n_down: int = 80, seed: int = 7) -> pd.DataFrame:
    """构造“先涨后跌”的合成行情。"""
    rng = np.random.default_rng(seed)
    up = 100.0 + np.cumsum(rng.standard_normal(n_up) * 0.4) + np.linspace(0, 15, n_up)
    down = up[-1] + np.cumsum(rng.standard_normal(n_down) * 0.4) - np.linspace(0, 12, n_down)
    close = np.concatenate([up, down])
    high = close + np.abs(rng.standard_normal(len(close))) * 0.3
    low = close - np.abs(rng.standard_normal(len(close))) * 0.3
    open_ = (high + low) / 2.0
    volume = rng.integers(2000, 12000, size=len(close)).astype(float)
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    df = make_demo_data()
    print(f"合成行情: {len(df)} 根K线, 区间 {df.index[0].date()} ~ {df.index[-1].date()}")

    ti = TechnicalIndicators()

    # 1. 可用指标列表
    section("1. 可用指标列表")
    for item in ti.list_indicators():
        print(f"  [{item['category']:>4}] {item['name']:<22} "
              f"参数={item['params']}")

    # 2. 分类计算演示
    section("2. 趋势类指标（ADX / 一目均衡表 / SAR 末值）")
    adx_df = ti.calculate(df, "adx")
    print(f"  ADX 末值: {adx_df['adx'].iloc[-1]:.2f} "
          f"(+DI={adx_df['plus_di'].iloc[-1]:.2f}, -DI={adx_df['minus_di'].iloc[-1]:.2f})")
    ichi = ti.calculate(df, "ichimoku")
    print(f"  一目均衡表: 转换线={ichi['tenkan_sen'].iloc[-1]:.2f}, "
          f"基准线={ichi['kijun_sen'].iloc[-1]:.2f}")
    print(f"  SAR 末值: {ti.calculate(df, 'sar').iloc[-1]:.2f} "
          f"(close={df['close'].iloc[-1]:.2f})")

    section("3. 震荡类指标（KDJ / CCI / WR / ROC / MOM 末值）")
    kdj_df = ti.calculate(df, "kdj")
    print(f"  KDJ: K={kdj_df['k'].iloc[-1]:.1f}, D={kdj_df['d'].iloc[-1]:.1f}, "
          f"J={kdj_df['j'].iloc[-1]:.1f}")
    print(f"  CCI={ti.calculate(df, 'cci').iloc[-1]:.1f}, "
          f"WR={ti.calculate(df, 'wr').iloc[-1]:.1f}, "
          f"ROC={ti.calculate(df, 'roc').iloc[-1]:.2f}%, "
          f"MOM={ti.calculate(df, 'mom').iloc[-1]:.2f}")

    section("4. 成交量类指标（OBV / VWAP / MFI / CMF 末值）")
    print(f"  OBV={ti.calculate(df, 'obv').iloc[-1]:.0f}, "
          f"VWAP={ti.calculate(df, 'vwap').iloc[-1]:.2f}")
    print(f"  MFI={ti.calculate(df, 'mfi').iloc[-1]:.1f}, "
          f"CMF={ti.calculate(df, 'cmf').iloc[-1]:.3f}")

    section("5. 波动率类 & 形态类")
    print(f"  布林带宽={ti.calculate(df, 'bollinger_bandwidth').iloc[-1]:.4f}, "
          f"ATR比率={ti.calculate(df, 'atr_ratio').iloc[-1]:.4f}, "
          f"年化波动率={ti.calculate(df, 'historical_volatility').iloc[-1]:.3f}")
    align = ti.calculate(df, "detect_ma_alignment")
    print(f"  均线排列末值: {align.iloc[-1]:.0f} "
          f"(1=多头, -1=空头, 0=无)")

    # 3. 批量合并演示
    section("6. 批量计算全部指标（calculate_all）")
    all_df = ti.calculate_all(df)
    new_cols = [c for c in all_df.columns if c not in df.columns]
    print(f"  原始列 {len(df.columns)} 个 -> 合并后 {len(all_df.columns)} 个，"
          f"新增指标列 {len(new_cols)} 个")

    # 4. IndicatorComboStrategy 回测
    section("7. IndicatorComboStrategy 回测")
    strategy = IndicatorComboStrategy()
    signals = strategy.generate_signals(df, symbol="DEMO")
    print(f"  策略信号数: {len(signals)}")
    for sig in signals[:5]:
        print(f"    {sig.date.date()} {sig.action:<4} "
              f"@{sig.price:.2f} 置信度={sig.confidence:.2f}")

    try:
        from backtest.engine import BacktestEngine
        engine = BacktestEngine(initial_capital=1_000_000.0)
        result = engine.run(df, strategy, symbol="DEMO")
        m = result.metrics
        print(f"  回测交易次数: {m.get('交易次数', 0)}")
        print(f"  累计收益率: {m.get('累计收益率', 0)*100:.2f}%")
        print(f"  最大回撤: {m.get('最大回撤', 0)*100:.2f}%")
    except Exception as exc:  # 回测引擎依赖较重时给出提示而非崩溃
        print(f"  （回测引擎未完整运行: {exc}）")


if __name__ == "__main__":
    main()
