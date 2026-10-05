#!/usr/bin/env python3
"""导出看板所需数据为 JSON。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from backtest.engine import BacktestEngine
from config import load_config
from data.data_cleaner import clean_klines
from jev.jev_engine import JevDecisionEngine
from risk.risk_manager import RiskManager
from strategies.ma_cross import MACrossStrategy


def fetch_data() -> pd.DataFrame:
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(
            symbol="600519", period="daily",
            start_date="20240101", end_date="20241231", adjust="qfq",
        )
        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
        }
        df = df.rename(columns=col_map)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df[["open", "high", "low", "close", "volume", "amount"]]
    except Exception:
        from data.data_fetcher import DataFetcher
        fetcher = DataFetcher(use_mock=True)
        return fetcher.get_klines("600519.SH", count=250, start_date="2024-01-02", use_cache=False)


def run_bt(df, use_jev, cfg):
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
    jev_engine = None
    if use_jev:
        jev_cfg = cfg["jev"]
        jev_engine = JevDecisionEngine(
            base_url=jev_cfg["base_url"],
            confidence_threshold=jev_cfg["confidence_threshold"],
            mock_mode=jev_cfg["mock_mode"],
            audit_log_path=str(PROJECT_ROOT / "logs" / "jev_audit.jsonl"),
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
        jev_engine=jev_engine,
    )
    return engine.run(df, strategy, symbol="600519.SH")


def main():
    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    df = clean_klines(fetch_data())

    result_no = run_bt(df, False, cfg)
    result_yes = run_bt(df, True, cfg)

    # K线数据
    kline = []
    for idx, row in df.iterrows():
        kline.append([
            idx.strftime("%Y-%m-%d"),
            round(float(row["open"]), 2),
            round(float(row["close"]), 2),
            round(float(row["low"]), 2),
            round(float(row["high"]), 2),
        ])

    # 净值曲线（归一化到100）
    def norm_eq(series):
        base = series.iloc[0]
        return [[d.strftime("%Y-%m-%d"), round(v / base * 100, 4)] for d, v in series.items()]

    eq_no = norm_eq(result_no.equity_curve)
    eq_yes = norm_eq(result_yes.equity_curve)
    bench = norm_eq(result_no.benchmark_curve)

    # 回撤
    def drawdown(series):
        peak = series.cummax()
        dd = (series / peak - 1) * 100
        return [[d.strftime("%Y-%m-%d"), round(v, 4)] for d, v in dd.items()]

    dd_no = drawdown(result_no.equity_curve)
    dd_yes = drawdown(result_yes.equity_curve)

    # 交易记录
    def trades_to_list(trades):
        out = []
        for t in trades:
            out.append({
                "date": t.date.strftime("%Y-%m-%d"),
                "symbol": t.symbol,
                "action": t.action,
                "price": round(t.price, 2),
                "shares": t.shares,
                "amount": round(t.amount, 2),
                "commission": round(t.commission, 2),
                "stamp_tax": round(t.stamp_tax, 2),
                "pnl": round(t.pnl, 2) if t.pnl is not None else None,
                "reason": t.reason,
            })
        return out

    # 买卖点标记（用于K线图）
    def mark_points(trades):
        buys = []
        sells = []
        for t in trades:
            if t.action == "buy":
                buys.append([t.date.strftime("%Y-%m-%d"), round(t.price, 2)])
            else:
                sells.append([t.date.strftime("%Y-%m-%d"), round(t.price, 2)])
        return buys, sells

    buys_no, sells_no = mark_points(result_no.trades)
    buys_yes, sells_yes = mark_points(result_yes.trades)

    data = {
        "symbol": "600519.SH",
        "symbol_name": "贵州茅台",
        "period": "2024-01-02 ~ 2024-12-31",
        "initial_capital": cfg["backtest"]["initial_capital"],
        "kline": kline,
        "equity": {"no_jev": eq_no, "with_jev": eq_yes, "benchmark": bench},
        "drawdown": {"no_jev": dd_no, "with_jev": dd_yes},
        "metrics": {
            "no_jev": {k: round(v, 6) for k, v in result_no.metrics.items()},
            "with_jev": {k: round(v, 6) for k, v in result_yes.metrics.items()},
        },
        "trades": {
            "no_jev": trades_to_list(result_no.trades),
            "with_jev": trades_to_list(result_yes.trades),
        },
        "markers": {
            "no_jev": {"buys": buys_no, "sells": sells_no},
            "with_jev": {"buys": buys_yes, "sells": sells_yes},
        },
    }

    out_path = PROJECT_ROOT / "web-dashboard" / "dashboard_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"数据已导出: {out_path}")
    print(f"K线: {len(kline)} 条, 无Jev交易: {len(result_no.trades)} 笔, 有Jev交易: {len(result_yes.trades)} 笔")


if __name__ == "__main__":
    main()
