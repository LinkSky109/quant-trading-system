"""RealtimeTrader 多标的并行扫描单元测试。

覆盖：
  1. scan_all_symbols          — scan_all=True 时返回全部标的
  2. enabled_symbols_filter    — enabled_symbols 指定子集时只返回子集
  3. scan_all_false_only_current — scan_all=False 时只返回 current_symbol
  4. multi_symbol_cycle_generates_decisions — 一轮后 trade_log 含多只标的
  5. max_positions_blocks_new_buy   — 持仓数达上限时新标的买入被拒
  6. max_positions_allows_sell      — 持仓数达上限时已有持仓卖出仍允许
  7. position_limit_enforced        — 单标的20%/总仓80%限制在多标的下生效
  8. stop_loss_per_symbol           — 每个持仓标的独立触发止损
  9. persistence_per_symbol        — 每只标的的 Jev 决策和成交写入 DB
 10. account_snapshot_once_per_cycle — 一轮只写一次账户快照
 11. single_symbol_failure_doesnt_block_others — 单标的异常不影响其它
 12. get_status_multi_symbol       — get_status 含多标的新字段

全部使用 mock 对象，不依赖真实行情/数据库/Jev 服务/网络。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest

# 让 tests 可以 import web-dashboard/realtime_trader.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WEB_DASHBOARD = _PROJECT_ROOT / "web-dashboard"
if str(_WEB_DASHBOARD) not in sys.path:
    sys.path.insert(0, str(_WEB_DASHBOARD))

from realtime_trader import RealtimeTrader  # noqa: E402


# ---------------------------------------------------------------------------
# Mock 对象
# ---------------------------------------------------------------------------

class MockSim:
    """模拟行情模拟器。"""

    def __init__(self, symbol: str, price: float, name: str = "") -> None:
        self.symbol = symbol
        self.current_price = price
        self.name = name or symbol
        self.tick_count = 100

    def get_realtime_df(self) -> pd.DataFrame:
        """返回一个最小合法 DataFrame（供 market_state_fn 使用）。"""
        return pd.DataFrame({
            "open": [self.current_price] * 30,
            "high": [self.current_price * 1.01] * 30,
            "low": [self.current_price * 0.99] * 30,
            "close": [self.current_price] * 30,
            "volume": [1000] * 30,
        })


class MockManager:
    """模拟 MultiSymbolManager。"""

    def __init__(self, sims: Dict[str, MockSim], current_symbol: str) -> None:
        self.sims = sims
        self.current_symbol = current_symbol
        self.total_tick = 1

    def get(self, symbol: str) -> MockSim:
        return self.sims[symbol]


class MockPosition:
    """模拟持仓。"""

    def __init__(self, shares: float, avg_cost: float) -> None:
        self.shares = shares
        self.avg_cost = avg_cost


class MockAccount:
    """模拟 SimAccount。"""

    def __init__(
        self,
        cash: float = 1_000_000.0,
        positions: Optional[Dict[str, Dict[str, float]]] = None,
        initial_capital: float = 1_000_000.0,
    ) -> None:
        self.cash = cash
        self.positions: Dict[str, Dict[str, float]] = dict(positions or {})
        self.initial_capital = initial_capital

    def snapshot(self) -> Dict[str, Any]:
        return {
            "cash": self.cash,
            "positions": [
                {"symbol": sym, "shares": p["shares"], "avg_cost": p["avg_cost"]}
                for sym, p in self.positions.items()
            ],
        }


class MockJevClient:
    """模拟 Jev 客户端，predict 返回固定概率。"""

    def __init__(
        self,
        probabilities: Optional[Dict[str, float]] = None,
        raise_exc: bool = False,
    ) -> None:
        self.probabilities = probabilities or {"buy": 0.9, "sell": 0.05, "hold": 0.05}
        self.raise_exc = raise_exc
        self.calls: List[Dict[str, Any]] = []

    def predict(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(market_state)
        if self.raise_exc:
            raise RuntimeError("jev service unavailable")
        return {
            "probabilities": dict(self.probabilities),
            "latency_ms": 1.0,
        }


class MockDB:
    """模拟持久化层，记录所有调用。"""

    def __init__(self) -> None:
        self.jev_decisions: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.snapshots: List[Dict[str, Any]] = []

    def insert_jev_decision(self, **kwargs: Any) -> None:
        self.jev_decisions.append(kwargs)

    def insert_trade(self, **kwargs: Any) -> None:
        self.trades.append(kwargs)

    def insert_account_snapshot(self, **kwargs: Any) -> None:
        self.snapshots.append(kwargs)


class MockAlertManager:
    """模拟告警管理器（无操作）。"""

    def __init__(self) -> None:
        self.events: List[str] = []

    def stop_loss(self, symbol: str, pnl_pct: float, **kwargs: Any) -> None:
        self.events.append(f"stop_loss:{symbol}")

    def take_profit(self, symbol: str, pnl_pct: float, **kwargs: Any) -> None:
        self.events.append(f"take_profit:{symbol}")

    def jev_filtered(self, symbol: str, action: str, conf: float, **kwargs: Any) -> None:
        self.events.append(f"jev_filtered:{symbol}")


# ---------------------------------------------------------------------------
# 测试辅助：构建 RealtimeTrader
# ---------------------------------------------------------------------------

def _build_trader(
    symbols: Optional[List[str]] = None,
    prices: Optional[Dict[str, float]] = None,
    positions: Optional[Dict[str, Dict[str, float]]] = None,
    cash: float = 1_000_000.0,
    initial_capital: float = 1_000_000.0,
    strategy_signal: str = "buy",
    strategy_conf: float = 0.9,
    market_state_ok: bool = True,
    market_state_raise_for: Optional[set] = None,
    jev_probabilities: Optional[Dict[str, float]] = None,
    realtime_config: Optional[Dict[str, Any]] = None,
    jev_threshold: float = 0.6,
):
    """构建一个带 mock 依赖的 RealtimeTrader 实例。

    Args:
        symbols: 股票池标的代码列表。
        prices: {symbol: price} 覆盖默认价格。
        positions: 初始持仓 {symbol: {shares, avg_cost}}。
        cash: 初始现金。
        initial_capital: 初始资金。
        strategy_signal: 策略信号 (buy/sell/hold)。
        strategy_conf: 策略置信度。
        market_state_ok: False 时 market_state_fn 返回 None。
        market_state_raise_for: 这些 symbol 会让 market_state_fn 抛异常。
        jev_probabilities: Jev 概率分布。
        realtime_config: 传给 RealtimeTrader 的 realtime_trading 配置。
        jev_threshold: Jev 置信度阈值。

    Returns:
        (trader, manager, account, jev_client, db, alert_manager)
    """
    symbols = symbols or ["AAA", "BBB", "CCC", "DDD", "EEE"]
    prices = prices or {}
    sims: Dict[str, MockSim] = {}
    for i, sym in enumerate(symbols):
        price = prices.get(sym, 100.0 + i)
        sims[sym] = MockSim(sym, price=price, name=f"标的{sym}")

    manager = MockManager(sims=sims, current_symbol=symbols[0])
    account = MockAccount(cash=cash, positions=positions, initial_capital=initial_capital)
    jev_client = MockJevClient(probabilities=jev_probabilities)
    db = MockDB()
    alert_manager = MockAlertManager()

    def _strategy_signal_fn(sim: MockSim) -> tuple:
        return strategy_signal, strategy_conf

    raise_for = market_state_raise_for or set()

    def _market_state_fn(sim: MockSim) -> Optional[Dict[str, Any]]:
        if sim.symbol in raise_for:
            raise RuntimeError(f"market_state boom for {sim.symbol}")
        if not market_state_ok:
            return None
        return {"close": sim.current_price, "volume": 1000}

    trader = RealtimeTrader(
        manager=manager,
        accounts={"acc_1": account},
        jev_client=jev_client,
        risk_config={
            "single_stop_loss": 0.03,
            "single_take_profit": 0.08,
            "max_drawdown_pause": 0.10,
            "max_position_per_symbol": 0.20,
            "max_total_position": 0.80,
            "daily_loss_limit": 0.02,
            "jev_confidence_threshold": 0.6,
        },
        get_strategy=lambda: "ma_cross",
        get_account_id=lambda: "acc_1",
        strategy_signal_fn=_strategy_signal_fn,
        market_state_fn=_market_state_fn,
        interval=5.0,
        jev_threshold=jev_threshold,
        db=db,
        alert_manager=alert_manager,
        realtime_config=realtime_config,
    )
    return trader, manager, account, jev_client, db, alert_manager


def _run_cycle(trader: RealtimeTrader) -> None:
    """同步驱动一轮 async 交易循环。"""
    asyncio.run(trader._run_one_cycle())


# ---------------------------------------------------------------------------
# 1. scan_all_symbols
# ---------------------------------------------------------------------------

def test_scan_all_symbols() -> None:
    """scan_all=True 时 _get_scan_symbols 应返回全部 5 只标的。"""
    trader, *_ = _build_trader(
        symbols=["A1", "A2", "A3", "A4", "A5"],
        realtime_config={"scan_all": True, "enabled_symbols": []},
    )
    result = trader._get_scan_symbols()
    assert result == ["A1", "A2", "A3", "A4", "A5"]


# ---------------------------------------------------------------------------
# 2. enabled_symbols_filter
# ---------------------------------------------------------------------------

def test_enabled_symbols_filter() -> None:
    """enabled_symbols 指定子集时只返回在 sims 中存在的子集。"""
    trader, manager, *_ = _build_trader(
        symbols=["A1", "A2", "A3", "A4", "A5"],
        realtime_config={
            "scan_all": True,
            "enabled_symbols": ["A2", "A4", "NOT_EXIST"],
        },
    )
    result = trader._get_scan_symbols()
    assert result == ["A2", "A4"]


# ---------------------------------------------------------------------------
# 3. scan_all_false_only_current
# ---------------------------------------------------------------------------

def test_scan_all_false_only_current() -> None:
    """scan_all=False 时只返回 current_symbol。"""
    trader, manager, *_ = _build_trader(
        symbols=["A1", "A2", "A3"],
        realtime_config={"scan_all": False, "enabled_symbols": []},
    )
    result = trader._get_scan_symbols()
    assert result == [manager.current_symbol]
    assert result == ["A1"]


# ---------------------------------------------------------------------------
# 4. multi_symbol_cycle_generates_decisions
# ---------------------------------------------------------------------------

def test_multi_symbol_cycle_generates_decisions() -> None:
    """运行一轮后 trade_log 中应包含多只标的的记录。"""
    trader, manager, *_ = _build_trader(
        symbols=["A1", "A2", "A3"],
        prices={s: 100.0 for s in ["A1", "A2", "A3"]},
        realtime_config={"scan_all": True, "enabled_symbols": []},
        jev_probabilities={"buy": 0.9, "sell": 0.05, "hold": 0.05},
    )
    _run_cycle(trader)

    logged_symbols = {entry["symbol"] for entry in trader.trade_log}
    assert logged_symbols == {"A1", "A2", "A3"}
    assert len(trader.trade_log) == 3


# ---------------------------------------------------------------------------
# 5. max_positions_blocks_new_buy
# ---------------------------------------------------------------------------

def test_max_positions_blocks_new_buy() -> None:
    """持仓数已达 max_positions 时，新标的买入应被拒绝。"""
    # 已有 2 只持仓，max_positions=2
    positions = {
        "OLD1": {"shares": 1000, "avg_cost": 100.0},
        "OLD2": {"shares": 1000, "avg_cost": 100.0},
    }
    trader, manager, account, *_ = _build_trader(
        symbols=["OLD1", "OLD2", "NEW1"],
        prices={s: 100.0 for s in ["OLD1", "OLD2", "NEW1"]},
        positions=positions,
        cash=800_000.0,
        realtime_config={"scan_all": True, "max_positions": 2, "enabled_symbols": []},
        jev_probabilities={"buy": 0.9, "sell": 0.05, "hold": 0.05},
    )
    _run_cycle(trader)

    # NEW1 应被 max_positions 拦截
    new1_log = [e for e in trader.trade_log if e["symbol"] == "NEW1"]
    assert len(new1_log) == 1
    assert new1_log[0]["order_executed"] is False
    assert "持仓标的数已达上限" in new1_log[0]["risk_reason"]
    # NEW1 不应出现在持仓中
    assert "NEW1" not in account.positions


# ---------------------------------------------------------------------------
# 6. max_positions_allows_sell
# ---------------------------------------------------------------------------

def test_max_positions_allows_sell() -> None:
    """持仓数达上限时，已有持仓标的的卖出仍应允许。"""
    positions = {
        "OLD1": {"shares": 2000, "avg_cost": 100.0},
        "OLD2": {"shares": 2000, "avg_cost": 100.0},
    }
    trader, manager, account, *_ = _build_trader(
        symbols=["OLD1", "OLD2"],
        prices={s: 100.0 for s in ["OLD1", "OLD2"]},
        positions=positions,
        cash=600_000.0,
        strategy_signal="sell",
        realtime_config={"scan_all": True, "max_positions": 2, "enabled_symbols": []},
        jev_probabilities={"sell": 0.9, "buy": 0.05, "hold": 0.05},
    )
    _run_cycle(trader)

    # 两只都应卖出成功
    sold = [e for e in trader.trade_log if e["order_executed"]]
    assert len(sold) == 2
    for e in sold:
        assert e["jev_decision"] == "sell"


# ---------------------------------------------------------------------------
# 7. position_limit_enforced
# ---------------------------------------------------------------------------

def test_position_limit_enforced() -> None:
    """总仓位 80% 限制在多标的场景下仍应生效（新买入被拒）。"""
    # 已有持仓价值 800,000（80% 权益），新买入应被风控拒绝
    positions = {
        "BIG1": {"shares": 8000, "avg_cost": 100.0},
    }
    trader, manager, account, *_ = _build_trader(
        symbols=["BIG1", "NEW1"],
        prices={s: 100.0 for s in ["BIG1", "NEW1"]},
        positions=positions,
        cash=200_000.0,
        realtime_config={"scan_all": True, "max_positions": 10, "enabled_symbols": []},
        jev_probabilities={"buy": 0.9, "sell": 0.05, "hold": 0.05},
    )
    _run_cycle(trader)

    new1_log = [e for e in trader.trade_log if e["symbol"] == "NEW1"]
    assert len(new1_log) == 1
    assert new1_log[0]["order_executed"] is False
    assert "风控检查未通过" in new1_log[0]["risk_reason"]


# ---------------------------------------------------------------------------
# 8. stop_loss_per_symbol
# ---------------------------------------------------------------------------

def test_stop_loss_per_symbol() -> None:
    """每个持仓标的应独立触发止损并卖出。"""
    # 两只持仓，成本 100，现价 95（-5%，超过 -3% 止损线）
    positions = {
        "LOSS1": {"shares": 2000, "avg_cost": 100.0},
        "LOSS2": {"shares": 2000, "avg_cost": 100.0},
    }
    trader, manager, account, jev_client, db, alert = _build_trader(
        symbols=["LOSS1", "LOSS2"],
        prices={s: 95.0 for s in ["LOSS1", "LOSS2"]},
        positions=positions,
        cash=800_000.0,
        realtime_config={"scan_all": True, "max_positions": 10, "enabled_symbols": []},
        # 即使 Jev 说 hold，止损也应强制卖出
        jev_probabilities={"hold": 0.9, "buy": 0.05, "sell": 0.05},
    )
    _run_cycle(trader)

    sold = [e for e in trader.trade_log if e["order_executed"]]
    assert len(sold) == 2, f"期望 2 笔止损卖出，实际 {len(sold)}"
    for e in sold:
        # 止损强制卖出 → jev_decision 被置为 sell
        assert e["jev_decision"] == "sell"
        # 成交价低于成本价 → 已实现盈亏为负
        assert e["realized_pnl"] < 0
    # 告警管理器应记录两条 stop_loss 事件
    assert "stop_loss:LOSS1" in alert.events
    assert "stop_loss:LOSS2" in alert.events
    # 止损卖出后持仓应清空
    assert len(account.positions) == 0


# ---------------------------------------------------------------------------
# 9. persistence_per_symbol
# ---------------------------------------------------------------------------

def test_persistence_per_symbol() -> None:
    """每只标的的 Jev 决策都应写入 DB；成交的标的还应写 trades。"""
    trader, manager, account, jev_client, db, alert = _build_trader(
        symbols=["A1", "A2", "A3"],
        prices={s: 100.0 for s in ["A1", "A2", "A3"]},
        realtime_config={"scan_all": True, "max_positions": 10, "enabled_symbols": []},
        jev_probabilities={"buy": 0.9, "sell": 0.05, "hold": 0.05},
    )
    _run_cycle(trader)

    # 3 只标的各写一次 Jev 决策
    assert len(db.jev_decisions) == 3
    decision_symbols = {d["symbol"] for d in db.jev_decisions}
    assert decision_symbols == {"A1", "A2", "A3"}

    # 3 只都应成交（现金足够）
    assert len(db.trades) == 3
    trade_symbols = {t["symbol"] for t in db.trades}
    assert trade_symbols == {"A1", "A2", "A3"}


# ---------------------------------------------------------------------------
# 10. account_snapshot_once_per_cycle
# ---------------------------------------------------------------------------

def test_account_snapshot_once_per_cycle() -> None:
    """一轮多标的扫描后，账户快照应只写一次。"""
    trader, manager, account, jev_client, db, alert = _build_trader(
        symbols=["A1", "A2", "A3", "A4", "A5"],
        prices={s: 100.0 for s in ["A1", "A2", "A3", "A4", "A5"]},
        realtime_config={"scan_all": True, "max_positions": 10, "enabled_symbols": []},
        jev_probabilities={"buy": 0.9, "sell": 0.05, "hold": 0.05},
    )
    _run_cycle(trader)

    # 一轮只写一次快照
    assert len(db.snapshots) == 1
    # 但 Jev 决策应是 5 次
    assert len(db.jev_decisions) == 5


# ---------------------------------------------------------------------------
# 11. single_symbol_failure_doesnt_block_others
# ---------------------------------------------------------------------------

def test_single_symbol_failure_doesnt_block_others() -> None:
    """某只标的处理异常时，不应影响同轮其它标的。"""
    trader, manager, account, jev_client, db, alert = _build_trader(
        symbols=["GOOD1", "BAD", "GOOD2"],
        prices={s: 100.0 for s in ["GOOD1", "BAD", "GOOD2"]},
        market_state_raise_for={"BAD"},
        realtime_config={"scan_all": True, "max_positions": 10, "enabled_symbols": []},
        jev_probabilities={"buy": 0.9, "sell": 0.05, "hold": 0.05},
    )
    _run_cycle(trader)

    logged_symbols = {e["symbol"] for e in trader.trade_log}
    # GOOD1 和 GOOD2 应被处理；BAD 因异常被跳过
    assert "GOOD1" in logged_symbols
    assert "GOOD2" in logged_symbols
    assert "BAD" not in logged_symbols


# ---------------------------------------------------------------------------
# 12. get_status_multi_symbol
# ---------------------------------------------------------------------------

def test_get_status_multi_symbol() -> None:
    """get_status 应包含 scan_symbols/symbol_count/max_positions/symbol_decisions。"""
    trader, manager, account, jev_client, db, alert = _build_trader(
        symbols=["A1", "A2", "A3"],
        prices={s: 100.0 for s in ["A1", "A2", "A3"]},
        realtime_config={"scan_all": True, "max_positions": 4, "enabled_symbols": []},
        jev_probabilities={"buy": 0.9, "sell": 0.05, "hold": 0.05},
    )
    _run_cycle(trader)

    status = trader.get_status()

    # 旧字段保留
    for key in ("running", "interval", "current_symbol", "current_account_id",
                "today_pnl", "log_count", "positions", "risk_summary"):
        assert key in status, f"缺少旧字段 {key}"

    # 新字段
    assert status["scan_symbols"] == ["A1", "A2", "A3"]
    assert status["symbol_count"] == 3
    assert status["max_positions"] == 4
    assert status["scan_all"] is True
    assert isinstance(status["symbol_decisions"], dict)
    assert set(status["symbol_decisions"].keys()) == {"A1", "A2", "A3"}
    for sym, dec in status["symbol_decisions"].items():
        assert dec["symbol"] == sym
        assert "action" in dec
        assert "confidence" in dec
        assert "reason" in dec
        assert "executed" in dec
        assert "timestamp" in dec
