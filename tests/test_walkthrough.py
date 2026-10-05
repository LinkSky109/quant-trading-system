"""回测走查器（BacktestWalkthrough）单元测试。

覆盖：
- 逐日快照包含 信号 / Jev决策 / 实际操作 / 持仓 / 盈亏
- get_day / get_range / get_trades 查询正确
- 信号 vs 操作对比能识别 Jev 过滤 / 风控拦截
- 无 Jev 时走查正常
- daily_pnl 计算正确
- SQLite walkthroughs 表持久化与查询

全部使用构造的 mock K 线数据，不依赖网络。

运行:
    cd quant_trading_system
    python -m pytest tests/test_walkthrough.py -v
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine
from backtest.walkthrough import BacktestWalkthrough
from jev.jev_engine import JevDecisionEngine
from persistence.database import Database


# ---------------------------------------------------------------------------
# 测试数据构造
# ---------------------------------------------------------------------------

def make_zigzag_data(n: int = 120, seed: int = 7) -> pd.DataFrame:
    """构造一段有明确金叉/死叉的震荡行情（离线、确定性）。

    四段涨跌（涨→跌→涨→跌），配合 ma_cross(5,20) 产生明确买卖点。
    """
    rng = np.random.default_rng(seed)
    seg1 = np.linspace(100, 130, 30)
    seg2 = np.linspace(130, 90, 30)
    seg3 = np.linspace(90, 125, 30)
    seg4 = np.linspace(125, 100, n - 90)
    close = np.concatenate([seg1, seg2, seg3, seg4]) + rng.normal(0, 0.4, n)
    dates = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame({
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full(n, 1_000_000),
    }, index=dates)
    return df


class _ForceRejectJev:
    """强制否决所有买入信号的 Jev mock（包装真实引擎以复用 build_market_state）。"""

    def __init__(self, inner: JevDecisionEngine):
        self._inner = inner

    def build_market_state(self, df, idx):
        return self._inner.build_market_state(df, idx)

    def evaluate(self, raw_signal, raw_confidence, market_state, symbol=""):
        decision = self._inner.evaluate(
            raw_signal=raw_signal,
            raw_confidence=raw_confidence,
            market_state=market_state,
            symbol=symbol,
        )
        if raw_signal == "buy":
            decision.executed = False
            decision.final_action = "hold"
            decision.reason = "测试:强制否决买入"
        return decision


class _BlockingRiskManager:
    """总是拦截新开仓的风控桩，用于验证 risk_blocked 记录。"""

    single_stop_loss = 0.03
    single_take_profit = 0.08

    def reset(self, *a: Any, **k: Any) -> None:
        pass

    def update_equity(self, *a: Any, **k: Any) -> None:
        pass

    def is_paused(self) -> bool:
        return False

    def check_trade_allowed(self, *a: Any, **k: Any) -> bool:
        return False  # 拦截一切新开仓

    def calc_position_size(self, *a: Any, **k: Any) -> float:
        return 0.0

    def record_trade(self, *a: Any, **k: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# 基础走查（无 Jev）
# ---------------------------------------------------------------------------

class TestWalkthroughBasic:
    def test_run_returns_id_and_snapshots(self):
        df = make_zigzag_data()
        wt = BacktestWalkthrough("TEST", "ma_cross", "2024-01-02", "2024-06-30")
        wid = wt.run(df)
        assert isinstance(wid, str) and len(wid) >= 8
        # 每个交易日一条快照
        assert len(wt.snapshots) == len(df)

    def test_snapshot_has_required_fields(self):
        df = make_zigzag_data()
        wt = BacktestWalkthrough("TEST", "ma_cross", "2024-01-02", "2024-06-30")
        wt.run(df)
        snap = wt.snapshots[0]
        for key in ("date", "close", "signals", "positions",
                    "cash", "total_equity", "daily_pnl", "cumulative_pnl"):
            assert key in snap, f"快照缺少字段 {key}"

    def test_get_day(self):
        df = make_zigzag_data()
        wt = BacktestWalkthrough("TEST", "ma_cross", "2024-01-02", "2024-06-30")
        wt.run(df)
        target = wt.snapshots[10]["date"]
        day = wt.get_day(target)
        assert day is not None
        assert day["date"] == target
        assert wt.get_day("1999-01-01") is None

    def test_get_range(self):
        df = make_zigzag_data()
        wt = BacktestWalkthrough("TEST", "ma_cross", "2024-01-02", "2024-06-30")
        wt.run(df)
        start = wt.snapshots[5]["date"]
        end = wt.snapshots[15]["date"]
        rng = wt.get_range(start, end)
        assert len(rng) == 11  # 闭区间含两端
        assert rng[0]["date"] == start
        assert rng[-1]["date"] == end

    def test_get_trades(self):
        df = make_zigzag_data()
        wt = BacktestWalkthrough("TEST", "ma_cross", "2024-01-02", "2024-06-30")
        wt.run(df)
        trades = wt.get_trades()
        # 震荡行情应至少产生一次买、一次卖
        actions = [t["action"] for t in trades]
        assert "buy" in actions
        assert "sell" in actions
        for t in trades:
            for k in ("date", "symbol", "action", "price", "shares", "reason"):
                assert k in t

    def test_daily_pnl_correct(self):
        df = make_zigzag_data()
        wt = BacktestWalkthrough("TEST", "ma_cross", "2024-01-02", "2024-06-30")
        wt.run(df)
        snaps = wt.snapshots
        # 首日 daily_pnl = 0（无前序权益）
        assert snaps[0]["daily_pnl"] == pytest.approx(0.0)
        # 后续每日 daily_pnl = 当日权益 - 前日权益
        for i in range(1, len(snaps)):
            expected = snaps[i]["total_equity"] - snaps[i - 1]["total_equity"]
            assert snaps[i]["daily_pnl"] == pytest.approx(expected, rel=1e-9)
            # 累计盈亏 = 总权益 - 初始资金
            assert snaps[i]["cumulative_pnl"] == pytest.approx(
                snaps[i]["total_equity"] - wt.initial_capital, rel=1e-9
            )

    def test_to_dict(self):
        df = make_zigzag_data()
        wt = BacktestWalkthrough("TEST", "ma_cross", "2024-01-02", "2024-06-30")
        wt.run(df)
        d = wt.to_dict()
        assert d["snapshots_count"] == len(wt.snapshots)
        assert d["trades_count"] == len(wt.trades)
        assert d["symbol"] == "TEST" and d["strategy"] == "ma_cross"


# ---------------------------------------------------------------------------
# Jev 过滤识别
# ---------------------------------------------------------------------------

class TestWalkthroughJev:
    def test_jev_filtered_signals_identified(self, monkeypatch):
        df = make_zigzag_data()

        # 用 monkeypatch 让 JevDecisionEngine 强制否决买入
        inner_evaluate = JevDecisionEngine.evaluate

        def force_reject(self, raw_signal, raw_confidence, market_state, symbol=""):
            d = inner_evaluate(self, raw_signal, raw_confidence,
                               market_state, symbol)
            if raw_signal == "buy":
                d.executed = False
                d.final_action = "hold"
                d.reason = "测试:强制否决买入"
            return d

        monkeypatch.setattr(JevDecisionEngine, "evaluate", force_reject)

        wt = BacktestWalkthrough("TEST", "ma_cross", "2024-01-02",
                                 "2024-06-30", use_jev=True)
        wt.run(df)

        rows = wt.get_signal_vs_action()
        # 存在买入信号被 Jev 否决
        buy_rows = [r for r in rows if r["signal"] == "buy"]
        assert len(buy_rows) > 0
        filtered = [r for r in buy_rows if r["jev_filtered"]]
        assert len(filtered) > 0, "应至少识别到一条被 Jev 过滤的买入信号"
        for r in filtered:
            assert r["executed"] is False
            assert r["jev_final_action"] == "hold"
            assert r["fill_price"] is None
            assert r["shares"] == 0

        # 由于所有买入被否决，不应有任何成交
        assert len(wt.get_trades()) == 0

    def test_signal_vs_action_structure(self):
        df = make_zigzag_data()
        wt = BacktestWalkthrough("TEST", "ma_cross", "2024-01-02",
                                 "2024-06-30", use_jev=False)
        wt.run(df)
        rows = wt.get_signal_vs_action()
        assert isinstance(rows, list)
        if rows:
            for k in ("date", "signal", "executed", "jev_filtered",
                      "risk_blocked", "block_reason", "fill_price", "shares"):
                assert k in rows[0]


# ---------------------------------------------------------------------------
# 风控拦截识别（直接在引擎层注入拦截风控）
# ---------------------------------------------------------------------------

class TestWalkthroughRiskBlock:
    def test_risk_blocked_signals_recorded(self):
        df = make_zigzag_data()
        snapshots: List[Dict[str, Any]] = []
        from strategies.ma_cross import MACrossStrategy

        engine = BacktestEngine(
            initial_capital=1_000_000.0,
            risk_manager=_BlockingRiskManager(),
            walkthrough_snapshots=snapshots,
        )
        engine.run(df, MACrossStrategy({"fast_period": 5, "slow_period": 20}),
                   symbol="TEST")

        # 收集所有被风控拦截的买入信号
        blocked = []
        executed_buys = []
        for snap in snapshots:
            for sig in snap["signals"]:
                if sig["signal"] == "buy":
                    if sig["risk_blocked"]:
                        blocked.append(sig)
                    if sig["executed"]:
                        executed_buys.append(sig)

        assert len(blocked) > 0, "应至少有一条买入信号被风控拦截"
        for b in blocked:
            assert b["risk_blocked"] is True
            assert b["executed"] is False
            assert b["fill_price"] is None
        # 风控全程拦截，不应有买入成交
        assert len(executed_buys) == 0


# ---------------------------------------------------------------------------
# 无走查时引擎行为不变（回归保护）
# ---------------------------------------------------------------------------

class TestNoWalkthroughNoBehaviorChange:
    def test_engine_without_walkthrough_runs(self):
        df = make_zigzag_data()
        from strategies.ma_cross import MACrossStrategy
        engine = BacktestEngine()  # walkthrough_snapshots 默认 None
        result = engine.run(
            df, MACrossStrategy({"fast_period": 5, "slow_period": 20}),
            symbol="TEST",
        )
        # 正常产出净值与交易
        assert len(result.equity_curve) == len(df)
        assert isinstance(result.trades, list)


# ---------------------------------------------------------------------------
# SQLite 持久化
# ---------------------------------------------------------------------------

class TestWalkthroughPersistence:
    def test_persist_and_readback(self, tmp_path):
        db_path = str(tmp_path / "wt.db")
        df = make_zigzag_data()
        wt = BacktestWalkthrough("TEST", "ma_cross", "2024-01-02",
                                 "2024-06-30", db_path=db_path)
        wid = wt.run(df)

        db = Database(db_path)
        rec = db.get_walkthrough(wid)
        assert rec is not None
        assert rec["id"] == wid
        assert rec["symbol"] == "TEST"
        assert rec["strategy"] == "ma_cross"
        # JSON 自动反序列化
        assert isinstance(rec["snapshots"], list)
        assert len(rec["snapshots"]) == len(wt.snapshots)
        assert isinstance(rec["trades"], list)
        assert rec["trades"] == wt.trades

        # 列表查询
        listing = db.list_walkthroughs()
        assert any(row["id"] == wid for row in listing)

        # 不存在的 id 返回 None
        assert db.get_walkthrough("nope") is None
