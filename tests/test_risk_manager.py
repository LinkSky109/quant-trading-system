"""风控管理器单元测试。

覆盖: 最大回撤暂停 / 单标的仓位上限 / 总仓位上限 / 单日亏损限额 / 仓位计算 / 卖出豁免
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict

import pandas as pd
import pytest

from risk.risk_manager import RiskEvent, RiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_positions(shares_map: Dict[str, float]) -> Dict:
    """构造持仓字典，值为含shares属性的对象。"""
    return {sym: SimpleNamespace(shares=s) for sym, s in shares_map.items()}


@pytest.fixture
def rm() -> RiskManager:
    return RiskManager(
        single_stop_loss=0.03,
        single_take_profit=0.08,
        max_drawdown_pause=0.10,
        max_position_per_symbol=0.20,
        max_total_position=0.80,
        daily_loss_limit=0.02,
        jev_confidence_threshold=0.6,
        initial_capital=1_000_000.0,
    )


# ---------------------------------------------------------------------------
# 初始化与重置
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_values(self):
        rm = RiskManager()
        assert rm.single_stop_loss == 0.03
        assert rm.single_take_profit == 0.08
        assert rm.max_drawdown_pause == 0.10
        assert rm.max_position_per_symbol == 0.20
        assert rm.max_total_position == 0.80
        assert rm.daily_loss_limit == 0.02
        assert rm.jev_confidence_threshold == 0.6
        assert rm.initial_capital == 1_000_000.0

    def test_initial_state_not_paused(self, rm):
        assert rm.is_paused() is False
        assert rm.current_equity == 1_000_000.0
        assert rm.peak_equity == 1_000_000.0

    def test_reset_clears_state(self, rm):
        rm.update_equity(800_000)  # 触发暂停
        assert rm.is_paused() is True
        rm.reset(2_000_000.0)
        assert rm.is_paused() is False
        assert rm.initial_capital == 2_000_000.0
        assert rm.peak_equity == 2_000_000.0
        assert rm.events == []


# ---------------------------------------------------------------------------
# 最大回撤暂停
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_drawdown_below_threshold_no_pause(self, rm):
        rm.update_equity(950_000)  # 回撤5% < 10%
        assert rm.is_paused() is False

    def test_drawdown_at_threshold_pauses(self, rm):
        rm.update_equity(900_000)  # 回撤10% >= 10%
        assert rm.is_paused() is True

    def test_drawdown_above_threshold_pauses(self, rm):
        rm.update_equity(800_000)  # 回撤20%
        assert rm.is_paused() is True

    def test_pause_records_event(self, rm):
        rm.update_equity(850_000)
        assert len(rm.events) == 1
        assert rm.events[0].event_type == "drawdown"
        assert rm.events[0].symbol == "ALL"

    def test_peak_updates_on_new_high(self, rm):
        rm.update_equity(1_200_000)
        assert rm.peak_equity == 1_200_000
        rm.update_equity(1_100_000)  # 从新高回撤8.3%，不暂停
        assert rm.is_paused() is False

    def test_paused_blocks_buy(self, rm):
        rm.update_equity(800_000)  # 暂停
        allowed = rm.check_trade_allowed(
            symbol="600519.SH", action="buy", price=100.0,
            equity=800_000, cash=800_000, positions={},
        )
        assert allowed is False

    def test_paused_blocks_all_trades_including_sell(self, rm):
        """当前设计：暂停时所有交易均被阻止（含卖出），需reset后恢复。"""
        rm.update_equity(800_000)  # 暂停
        allowed = rm.check_trade_allowed(
            symbol="600519.SH", action="sell", price=100.0,
            equity=800_000, cash=0, positions=_make_positions({"600519.SH": 100}),
        )
        assert allowed is False


# ---------------------------------------------------------------------------
# 仓位限制
# ---------------------------------------------------------------------------

class TestPositionLimits:
    def test_buy_allowed_when_under_limits(self, rm):
        allowed = rm.check_trade_allowed(
            symbol="600519.SH", action="buy", price=100.0,
            equity=1_000_000, cash=500_000, positions={},
        )
        assert allowed is True

    def test_total_position_limit_blocks_buy(self, rm):
        # 总仓位80万 = 80%，达到上限
        positions = _make_positions({
            "600519.SH": 4000,  # 4000*100=40万
            "300750.SZ": 4000,  # 4000*100=40万
        })
        allowed = rm.check_trade_allowed(
            symbol="002594.SZ", action="buy", price=100.0,
            equity=1_000_000, cash=200_000, positions=positions,
        )
        assert allowed is False

    def test_single_symbol_limit_blocks_buy(self, rm):
        # 单标的20万 = 20%，达到上限
        positions = _make_positions({"600519.SH": 2000})  # 2000*100=20万
        allowed = rm.check_trade_allowed(
            symbol="600519.SH", action="buy", price=100.0,
            equity=1_000_000, cash=800_000, positions=positions,
        )
        assert allowed is False

    def test_single_symbol_under_limit_allows(self, rm):
        # 单标的10万 = 10% < 20%
        positions = _make_positions({"600519.SH": 1000})  # 1000*100=10万
        allowed = rm.check_trade_allowed(
            symbol="600519.SH", action="buy", price=100.0,
            equity=1_000_000, cash=800_000, positions=positions,
        )
        assert allowed is True

    def test_sell_always_allowed(self, rm):
        # 即使仓位超限，卖出也允许
        positions = _make_positions({"600519.SH": 5000})
        allowed = rm.check_trade_allowed(
            symbol="600519.SH", action="sell", price=100.0,
            equity=1_000_000, cash=0, positions=positions,
        )
        assert allowed is True


# ---------------------------------------------------------------------------
# 单日亏损限额
# ---------------------------------------------------------------------------

class TestDailyLoss:
    def test_daily_loss_under_limit_allows_buy(self, rm):
        rm.set_date(pd.Timestamp("2024-01-02"))
        rm.record_trade(-10_000)  # 亏1万 < 2万限额
        allowed = rm.check_trade_allowed(
            symbol="600519.SH", action="buy", price=100.0,
            equity=990_000, cash=500_000, positions={},
        )
        assert allowed is True

    def test_daily_loss_exceeds_limit_blocks_buy(self, rm):
        rm.set_date(pd.Timestamp("2024-01-02"))
        rm.record_trade(-20_001)  # 亏20001 > 2万限额（严格小于）
        allowed = rm.check_trade_allowed(
            symbol="600519.SH", action="buy", price=100.0,
            equity=979_999, cash=500_000, positions={},
        )
        assert allowed is False

    def test_daily_loss_at_exact_limit_allows(self, rm):
        """日亏检查用严格小于，恰好等于限额时仍允许。"""
        rm.set_date(pd.Timestamp("2024-01-02"))
        rm.record_trade(-20_000)  # 恰好等于限额
        allowed = rm.check_trade_allowed(
            symbol="600519.SH", action="buy", price=100.0,
            equity=980_000, cash=500_000, positions={},
        )
        assert allowed is True

    def test_daily_loss_records_event(self, rm):
        rm.set_date(pd.Timestamp("2024-01-02"))
        rm.record_trade(-25_000)
        assert len(rm.events) >= 1
        daily_events = [e for e in rm.events if e.event_type == "daily_loss"]
        assert len(daily_events) >= 1

    def test_daily_loss_does_not_block_sell(self, rm):
        rm.set_date(pd.Timestamp("2024-01-02"))
        rm.record_trade(-25_000)  # 超限
        allowed = rm.check_trade_allowed(
            symbol="600519.SH", action="sell", price=100.0,
            equity=975_000, cash=0, positions=_make_positions({"600519.SH": 100}),
        )
        assert allowed is True

    def test_profit_does_not_trigger_daily_loss(self, rm):
        rm.set_date(pd.Timestamp("2024-01-02"))
        rm.record_trade(50_000)  # 盈利
        assert len([e for e in rm.events if e.event_type == "daily_loss"]) == 0


# ---------------------------------------------------------------------------
# 仓位计算
# ---------------------------------------------------------------------------

class TestPositionSizing:
    def test_high_confidence_larger_position(self, rm):
        size_high = rm.calc_position_size(
            symbol="600519.SH", price=100.0, equity=1_000_000,
            cash=500_000, confidence=0.95, positions={},
        )
        size_low = rm.calc_position_size(
            symbol="600519.SH", price=100.0, equity=1_000_000,
            cash=500_000, confidence=0.6, positions={},
        )
        assert size_high > size_low

    def test_position_capped_by_per_symbol_limit(self, rm):
        size = rm.calc_position_size(
            symbol="600519.SH", price=100.0, equity=1_000_000,
            cash=900_000, confidence=1.0, positions={},
        )
        # 单标的上限20% = 20万，置信度1.0时conf_factor=1.0
        assert size <= 200_000

    def test_position_capped_by_cash(self, rm):
        size = rm.calc_position_size(
            symbol="600519.SH", price=100.0, equity=1_000_000,
            cash=10_000, confidence=1.0, positions={},
        )
        # 现金1万，95%可用 = 9500
        assert size <= 9_500

    def test_position_capped_by_total_position(self, rm):
        # 已有总仓位70万，总上限80万，剩余10万
        positions = _make_positions({"A": 3500, "B": 3500})  # 各35万
        size = rm.calc_position_size(
            symbol="C", price=100.0, equity=1_000_000,
            cash=500_000, confidence=1.0, positions=positions,
        )
        assert size <= 100_000

    def test_position_never_negative(self, rm):
        # 总仓位已满，剩余空间为负
        positions = _make_positions({"A": 5000, "B": 5000})  # 各50万=100%
        size = rm.calc_position_size(
            symbol="C", price=100.0, equity=1_000_000,
            cash=0, confidence=1.0, positions=positions,
        )
        assert size >= 0.0


# ---------------------------------------------------------------------------
# 状态摘要
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_initial(self, rm):
        s = rm.get_summary()
        assert s["current_equity"] == 1_000_000.0
        assert s["peak_equity"] == 1_000_000.0
        assert s["current_drawdown"] == 0.0
        assert s["paused"] is False
        assert s["risk_events"] == 0

    def test_summary_after_drawdown(self, rm):
        rm.update_equity(850_000)
        s = rm.get_summary()
        assert s["current_drawdown"] == pytest.approx(0.15)
        assert s["paused"] is True
        assert s["risk_events"] == 1


# ---------------------------------------------------------------------------
# RiskEvent
# ---------------------------------------------------------------------------

class TestRiskEvent:
    def test_event_creation(self):
        event = RiskEvent(
            timestamp="2024-01-01T00:00:00",
            event_type="stop_loss",
            symbol="600519.SH",
            detail="亏损3%触发止损",
        )
        assert event.event_type == "stop_loss"
        assert event.symbol == "600519.SH"
