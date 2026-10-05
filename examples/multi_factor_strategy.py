#!/usr/bin/env python3
"""多因子选股策略演示脚本。

演示内容：
  1. 加载股票池配置
  2. 构造模拟 K 线（优先尝试 DataFetcher 真实数据，失败降级为 mock）
  3. 创建 MultiFactorStrategy 并计算横截面因子得分排名
  4. 运行多标的回测，打印关键指标
  5. 打印调仓历史

运行方式:
    cd quant_trading_system
    python examples/multi_factor_strategy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from config import load_config, load_stock_pool
from strategies.multi_factor import MultiFactorStrategy


def make_klines(n_days: int = 300, seed: int = 0,
                start_price: float = 100.0) -> pd.DataFrame:
    """构造模拟日 K 线（确定性随机种子，可复现）。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rets = rng.normal(0.0005, 0.02, size=n_days)
    close = start_price * np.cumprod(1.0 + rets)
    open_ = close * (1.0 + rng.normal(0, 0.005, size=n_days))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0, 0.01, size=n_days))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0, 0.01, size=n_days))
    volume = rng.uniform(1e6, 5e6, size=n_days)
    amount = volume * close
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": volume, "amount": amount},
        index=dates,
    )


def build_panel(pool: list) -> dict:
    """根据股票池构造 {symbol: K线DataFrame}。

    优先尝试 DataFetcher 获取真实数据，任何异常都降级为 mock 数据。
    """
    panel: dict = {}
    try:
        from data.data_fetcher import DataFetcher
        fetcher = DataFetcher(use_mock=False, cache_dir=str(PROJECT_ROOT / "cache"))
        for i, stock in enumerate(pool):
            sym = stock["symbol"]
            try:
                df = fetcher.get_klines(sym, period="daily", count=300)
                if df is not None and not df.empty:
                    panel[sym] = df
                    continue
            except Exception:
                pass
            # 降级 mock
            panel[sym] = make_klines(300, seed=i,
                                     start_price=float(stock.get("base_price", 100)))
    except Exception:
        # DataFetcher 不可用，全部 mock
        for i, stock in enumerate(pool):
            panel[stock["symbol"]] = make_klines(
                300, seed=i, start_price=float(stock.get("base_price", 100))
            )
    return panel


def main() -> None:
    # ------------------------------------------------------------------
    print("=" * 70)
    print("多因子选股策略演示")
    print("=" * 70)

    # 1. 加载配置
    cfg = load_config()
    mf_cfg = cfg.get("multi_factor", {})
    pool = load_stock_pool().get("stocks", [])
    # 演示用前 10 只标的，加速运行
    pool = pool[:10]
    symbols = [s["symbol"] for s in pool]
    print(f"股票池: {len(symbols)} 只 -> {symbols[:5]} ...")

    # 2. 构造行情面板
    panel = build_panel(pool)
    print(f"行情面板: {len(panel)} 只标的")

    # 3. 创建策略（使用 config/multi_factor 节的参数）
    params = {
        "factors": mf_cfg.get("factors"),
        "rebalance_days": mf_cfg.get("rebalance_days", 5),
        "top_n": mf_cfg.get("top_n", 5),
        "standardization": mf_cfg.get("standardization", "zscore"),
        "missing_handling": mf_cfg.get("missing_handling", "median"),
    }
    strategy = MultiFactorStrategy(params)
    strategy.set_cross_section_data(panel)

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1) 最新交易日因子得分排名")
    print("=" * 70)
    scores = strategy.scores_df
    latest_date = scores.dropna(how="all").index.max()
    row = scores.loc[latest_date].dropna().sort_values(ascending=False)
    print(f"日期: {pd.Timestamp(latest_date).date()}")
    for rank, (sym, sc) in enumerate(row.items(), 1):
        name = next((s.get("name", "") for s in pool if s["symbol"] == sym), "")
        marker = " <- 持仓" if rank <= strategy.top_n else ""
        print(f"  #{rank:<2} {sym:<12} {name:<8} score={sc:+.4f}{marker}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2) 多标的回测")
    print("=" * 70)
    bt_cfg = cfg.get("backtest", {})
    engine = BacktestEngine(
        initial_capital=float(bt_cfg.get("initial_capital", 1_000_000)),
        commission_rate=float(bt_cfg.get("commission_rate", 0.00025)),
        stamp_tax_rate=float(bt_cfg.get("stamp_tax_rate", 0.0005)),
        slippage_rate=float(bt_cfg.get("slippage_rate", 0.001)),
    )
    result = engine.run(panel, strategy, symbol="")

    m = result.metrics
    print(f"  累计收益率 : {m.get('累计收益率', 0):.2%}")
    print(f"  年化收益率 : {m.get('年化收益率', 0):.2%}")
    print(f"  最大回撤   : {m.get('最大回撤', 0):.2%}")
    print(f"  夏普比率   : {m.get('夏普比率', 0):.2f}")
    print(f"  交易次数   : {m.get('交易次数', 0)}")
    print(f"  最终权益   : {result.equity_curve.iloc[-1]:,.0f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3) 调仓历史（前 10 次）")
    print("=" * 70)
    for h in strategy.get_rebalance_history()[:10]:
        added = ",".join(h["added"]) or "-"
        removed = ",".join(h["removed"]) or "-"
        print(f"  {h['date']}  买入[{added}]  卖出[{removed}]  "
              f"持仓{len(h['holdings'])}只")


if __name__ == "__main__":
    main()
