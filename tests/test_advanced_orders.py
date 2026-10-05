"""高级订单类型单元测试。

覆盖:
  - 限价单（limit）：达到限价成交 / 未达到保持 pending
  - 止损单（stop）：价格穿破止损价后触发成交
  - 止损限价单（stop_limit）：先激活再限价成交
  - 移动止盈（trailing_stop）：high_water_mark 跟随最高价，回撤触发卖出
  - 撤单：cancel_order 对 pending 订单生效
  - 当日有效（expiry="day"）跨日自动撤销
  - 回测引擎 order_type 参数生效（market/limit/stop/trailing_stop 产生不同结果）
  - 订单成交率、平均持仓时间指标计算正确
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine, Trade
from backtest.metrics import calc_all_metrics
from trading.broker import Order, SimulatedBroker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def broker() -> SimulatedBroker:
    return SimulatedBroker(initial_capital=1_000_000.0)


def _make_kline(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    start: str = "2024-01-02",
) -> pd.DataFrame:
    """构造合成 K 线 DataFrame。"""
    dates = pd.bdate_range(start=start, periods=len(opens))
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1_000_000] * len(opens),
        },
        index=dates,
    )


class ConstantSignalStrategy:
    """固定在指定交易日产生 buy/sell 信号的策略。

    signal_dates: {date_str: +1/-1}
    """

    def __init__(self, signal_dates: Dict[str, int]):
        self.signal_dates = signal_dates

    def get_signal_dataframe(
        self, df: pd.DataFrame, symbol: str = ""
    ) -> pd.DataFrame:
        sig = pd.Series(0, index=df.index, dtype=int)
        conf = pd.Series(0.0, index=df.index)
        for ds, val in self.signal_dates.items():
            ts = pd.Timestamp(ds)
            if ts in sig.index:
                sig.loc[ts] = val
                conf.loc[ts] = 0.8
        return pd.DataFrame({"signal": sig, "confidence": conf}, index=df.index)


# ===========================================================================
# 1. 限价单
# ===========================================================================

class TestLimitOrder:
    def test_limit_buy_fills_when_price_reaches(self, broker):
        """限价买单：low 触及 limit_price 时成交。"""
        # 先买入底仓，确保有现金
        order = broker.submit_order(
            "X", "buy", 100, order_type="limit", limit_price=95.0,
        )
        assert order.status == "pending"
        assert len(broker.get_pending_orders()) == 1

        # 当日 low=94 <= 95，应成交
        filled = broker.check_pending_orders(
            current_price=96.0, high=97.0, low=94.0,
            date=pd.Timestamp("2024-01-02"),
        )
        assert len(filled) == 1
        assert order.status == "filled"
        # 成交价 = limit_price * (1+slippage) = 95 * 1.001 = 95.095
        assert order.filled_price == pytest.approx(95.0 * 1.001, abs=0.01)
        assert len(broker.get_pending_orders()) == 0

    def test_limit_buy_stays_pending_when_not_reached(self, broker):
        """限价买单：low 高于 limit_price 时保持 pending。"""
        order = broker.submit_order(
            "X", "buy", 100, order_type="limit", limit_price=90.0,
        )
        filled = broker.check_pending_orders(
            current_price=100.0, high=102.0, low=98.0,
            date=pd.Timestamp("2024-01-02"),
        )
        assert filled == []
        assert order.status == "pending"
        assert len(broker.get_pending_orders()) == 1

    def test_limit_sell_fills_when_price_reaches(self, broker):
        """限价卖单：high 触及 limit_price 时成交。"""
        # 先建底仓
        broker.submit_order("X", "buy", 200, price=100.0)
        order = broker.submit_order(
            "X", "sell", 100, order_type="limit", limit_price=105.0,
        )
        assert order.status == "pending"

        filled = broker.check_pending_orders(
            current_price=106.0, high=106.0, low=104.0,
            date=pd.Timestamp("2024-01-02"),
        )
        assert len(filled) == 1
        assert order.status == "filled"
        # 成交价 = 105 * (1-0.001) = 104.895
        assert order.filled_price == pytest.approx(105.0 * 0.999, abs=0.01)


# ===========================================================================
# 2. 止损单
# ===========================================================================

class TestStopOrder:
    def test_stop_sell_triggers_on_break(self, broker):
        """止损卖单：low 跌破 stop_price 时触发成交。"""
        broker.submit_order("X", "buy", 200, price=100.0)
        order = broker.submit_order(
            "X", "sell", 100, order_type="stop", stop_price=95.0,
        )
        assert order.status == "pending"

        # 当日 low=94 <= 95，触发
        filled = broker.check_pending_orders(
            current_price=94.5, high=96.0, low=94.0,
            date=pd.Timestamp("2024-01-02"),
        )
        assert len(filled) == 1
        assert order.status == "filled"
        assert order.triggered is True
        # 成交价 = 95 * 0.999
        assert order.filled_price == pytest.approx(95.0 * 0.999, abs=0.01)

    def test_stop_sell_no_trigger_above(self, broker):
        """止损卖单：low 高于 stop_price 时不触发。"""
        broker.submit_order("X", "buy", 200, price=100.0)
        order = broker.submit_order(
            "X", "sell", 100, order_type="stop", stop_price=95.0,
        )
        filled = broker.check_pending_orders(
            current_price=100.0, high=101.0, low=99.0,
            date=pd.Timestamp("2024-01-02"),
        )
        assert filled == []
        assert order.status == "pending"

    def test_stop_limit_two_phase(self, broker):
        """止损限价单：先穿破 stop_price 激活，再等待 limit_price 成交。"""
        broker.submit_order("X", "buy", 200, price=100.0)
        # stop_price=95（触发），limit_price=93（激活后限价卖出线）
        order = broker.submit_order(
            "X", "sell", 100, order_type="stop_limit",
            stop_price=95.0, limit_price=93.0,
        )

        # 第一天：low=92.5 穿破 95 激活 stop；high=92.8 < 93 未到限价，保持 pending
        broker.check_pending_orders(
            current_price=92.6, high=92.8, low=92.5,
            date=pd.Timestamp("2024-01-02"),
        )
        assert order.triggered is True
        assert order.status == "pending"

        # 第二天：high=93.5 >= 93，限价成交
        filled = broker.check_pending_orders(
            current_price=93.2, high=93.5, low=92.0,
            date=pd.Timestamp("2024-01-03"),
        )
        assert len(filled) == 1
        assert order.status == "filled"


# ===========================================================================
# 3. 移动止盈
# ===========================================================================

class TestTrailingStop:
    def test_high_water_mark_updates(self, broker):
        """high_water_mark 跟随每日 high 上移。"""
        broker.submit_order("X", "buy", 200, price=100.0)
        order = broker.submit_order(
            "X", "sell", 100, order_type="trailing_stop",
            trailing_pct=0.05,
        )
        assert order.high_water_mark == 0.0  # 初始

        # Day1 high=110, low=106（>触发线 110*0.95=104.5，不触发卖出）
        broker.check_pending_orders(
            current_price=108.0, high=110.0, low=106.0,
            date=pd.Timestamp("2024-01-02"),
        )
        assert order.high_water_mark == 110.0
        assert order.status == "pending"

        # Day2 high=115, low=112（>触发线 115*0.95=109.25，不触发卖出）
        broker.check_pending_orders(
            current_price=113.0, high=115.0, low=112.0,
            date=pd.Timestamp("2024-01-03"),
        )
        assert order.high_water_mark == 115.0
        assert order.status == "pending"

    def test_trailing_stop_sells_on_drawdown(self, broker):
        """从最高点回撤 trailing_pct 后触发卖出。"""
        broker.submit_order("X", "buy", 200, price=100.0)
        order = broker.submit_order(
            "X", "sell", 100, order_type="trailing_stop",
            trailing_pct=0.05,
        )

        # Day1: high=120, 触发线 = 120*0.95 = 114
        broker.check_pending_orders(
            current_price=118.0, high=120.0, low=117.0,
            date=pd.Timestamp("2024-01-02"),
        )
        assert order.status == "pending"

        # Day2: low=113 <= 114，触发卖出
        filled = broker.check_pending_orders(
            current_price=113.5, high=118.0, low=113.0,
            date=pd.Timestamp("2024-01-03"),
        )
        assert len(filled) == 1
        assert order.status == "filled"
        assert order.triggered is True


# ===========================================================================
# 4. 撤单
# ===========================================================================

class TestCancelPending:
    def test_cancel_limit_order(self, broker):
        """cancel_order 对 pending 限价单生效。"""
        order = broker.submit_order(
            "X", "buy", 100, order_type="limit", limit_price=90.0,
        )
        assert len(broker.get_pending_orders()) == 1
        ok = broker.cancel_order(order.order_id)
        assert ok is True
        assert order.status == "cancelled"
        assert broker.get_pending_orders() == []

    def test_cancel_filled_order_fails(self, broker):
        """已成交订单不能撤。"""
        order = broker.submit_order("X", "buy", 100, price=100.0)
        assert broker.cancel_order(order.order_id) is False


# ===========================================================================
# 5. 当日有效（GTC vs Day）
# ===========================================================================

class TestExpiry:
    def test_day_order_expires_on_date_change(self, broker):
        """expiry='day' 订单在跨日时自动撤销。"""
        # 手动构造一个 created_at 为昨天的 day 订单
        order = Order(
            order_id="day1", symbol="X", action="buy",
            order_type="limit", limit_price=90.0, quantity=100,
            status="pending",
            created_at=(pd.Timestamp("2024-01-02") - pd.Timedelta(days=1))
            .isoformat(),
            expiry="day",
        )
        broker.orders["day1"] = order
        broker.pending_orders.append(order)

        # 传入不同日期，应自动撤销
        filled = broker.check_pending_orders(
            current_price=100.0, high=101.0, low=99.0,
            date=pd.Timestamp("2024-01-03"),
        )
        assert filled == []
        assert order.status == "cancelled"
        assert broker.get_pending_orders() == []

    def test_gtc_order_survives_days(self, broker):
        """GTC 订单（expiry=''）跨日不撤销。"""
        order = broker.submit_order(
            "X", "buy", 100, order_type="limit", limit_price=90.0,
        )
        # 跨日但价格未到，仍 pending
        broker.check_pending_orders(
            current_price=100.0, high=101.0, low=99.0,
            date=pd.Timestamp("2024-01-03"),
        )
        assert order.status == "pending"
        assert len(broker.get_pending_orders()) == 1


# ===========================================================================
# 6. 回测引擎 order_type 参数
# ===========================================================================

def _build_engine_with_data(
    order_type: str, signal_dates: Dict[str, int]
) -> tuple[BacktestEngine, Any]:
    """构造一个用合成数据的回测引擎。"""
    # 价格路径：前几天平稳，第 5 天买入后先涨后跌
    n = 20
    opens = [100.0] * n
    highs = [101.0] * n
    lows = [99.0] * n
    closes = [100.5] * n
    # 第 6..10 天上涨到 115
    for i in range(5, 10):
        opens[i] = 100.0 + i
        highs[i] = opens[i] + 1.5
        lows[i] = opens[i] - 0.5
        closes[i] = opens[i] + 0.5
    # 第 11..15 天回落
    for i in range(10, 15):
        opens[i] = 110.0 - (i - 10) * 2
        highs[i] = opens[i] + 1.0
        lows[i] = opens[i] - 1.5
        closes[i] = opens[i]
    df = _make_kline(opens, highs, lows, closes)
    strat = ConstantSignalStrategy(signal_dates)
    engine = BacktestEngine(order_type=order_type)
    return engine, df, strat


class TestEngineOrderType:
    def test_default_is_market(self):
        """默认 order_type='market'，向后兼容。"""
        engine = BacktestEngine()
        assert engine.order_type == "market"
        assert engine.pending_orders == []

    def test_limit_mode_produces_pending_orders(self):
        """limit 模式下买入信号先挂单，不一定当日成交。"""
        engine, df, strat = _build_engine_with_data(
            "limit", {"2024-01-08": 1, "2024-01-12": -1}
        )
        result = engine.run(df, strat, symbol="X")
        # 至少有挂单被提交过
        assert engine._orders_submitted >= 1
        # 限价买单（开盘价*0.995）在后续 low 范围内应能成交
        buy_trades = [t for t in result.trades if t.action == "buy"]
        assert len(buy_trades) >= 1
        # 买单 order_type 应为 limit（挂单成交）或 market（兼容）
        assert all(t.order_type in ("limit", "market") for t in buy_trades)

    def test_stop_mode_attaches_stop_order(self):
        """stop 模式下买入后自动附带止损卖单。"""
        engine, df, strat = _build_engine_with_data(
            "stop", {"2024-01-08": 1, "2024-01-12": -1}
        )
        engine.run(df, strat, symbol="X")
        # 买入是市价单，止损卖单挂入 pending
        # submitted = 1(market buy) + 1(stop sell) >= 2
        assert engine._orders_submitted >= 2

    def test_trailing_stop_mode_attaches_trailing_order(self):
        """trailing_stop 模式下买入后自动附带移动止盈卖单。"""
        engine, df, strat = _build_engine_with_data(
            "trailing_stop", {"2024-01-08": 1}
        )
        engine.run(df, strat, symbol="X")
        # 至少提交了 1 个 market buy + 1 个 trailing_stop sell
        assert engine._orders_submitted >= 2

    def test_market_mode_no_pending_orders(self):
        """market 模式下不产生 pending_orders。"""
        engine, df, strat = _build_engine_with_data(
            "market", {"2024-01-08": 1, "2024-01-12": -1}
        )
        engine.run(df, strat, symbol="X")
        # market 模式下所有订单立即成交，pending 队列在期末为空
        assert engine.pending_orders == []


# ===========================================================================
# 7. 指标：订单成交率 & 平均持仓时间
# ===========================================================================

class TestFillRateMetric:
    def test_fill_rate_market_mode_is_100pct(self):
        """market 模式下提交的订单全部成交。"""
        engine, df, strat = _build_engine_with_data(
            "market", {"2024-01-08": 1, "2024-01-12": -1}
        )
        result = engine.run(df, strat, symbol="X")
        assert result.metrics["订单成交率"] == pytest.approx(1.0)

    def test_fill_rate_uses_passed_counters(self):
        """calc_all_metrics 使用传入的 total/filled 计算成交率。"""
        equity = pd.Series([1.0, 1.02, 1.05])
        metrics = calc_all_metrics(equity, [], total_orders=10, filled_orders=8)
        assert metrics["订单成交率"] == pytest.approx(0.8)

    def test_fill_rate_no_orders_returns_zero(self):
        equity = pd.Series([1.0, 1.02])
        metrics = calc_all_metrics(equity, [], total_orders=0, filled_orders=0)
        assert metrics["订单成交率"] == 0.0


class TestAvgHoldingPeriod:
    def test_avg_holding_period_computed(self):
        """平仓交易的 (exit - entry).days 平均值。"""
        entry = pd.Timestamp("2024-01-02")
        exit1 = pd.Timestamp("2024-01-10")  # 8 天
        exit2 = pd.Timestamp("2024-01-12")  # 10 天
        trades = [
            {"pnl": 100.0, "date": exit1, "entry_date": entry},
            {"pnl": -50.0, "date": exit2, "entry_date": entry},
        ]
        equity = pd.Series([1.0, 1.02, 1.05])
        metrics = calc_all_metrics(equity, trades)
        # 平均 = (8 + 10) / 2 = 9
        assert metrics["平均持仓时间"] == pytest.approx(9.0)

    def test_avg_holding_period_no_closed_trades(self):
        trades = [{"pnl": None, "date": pd.Timestamp("2024-01-02"),
                   "entry_date": None}]
        equity = pd.Series([1.0, 1.02])
        metrics = calc_all_metrics(equity, trades)
        assert metrics["平均持仓时间"] == 0.0

    def test_engine_records_entry_date(self):
        """回测引擎 Trade 记录了 entry_date。"""
        engine, df, strat = _build_engine_with_data(
            "market", {"2024-01-08": 1, "2024-01-12": -1}
        )
        result = engine.run(df, strat, symbol="X")
        sell_trades = [t for t in result.trades if t.action == "sell"]
        assert len(sell_trades) >= 1
        for t in sell_trades:
            assert t.entry_date is not None
            assert (t.date - t.entry_date).days >= 0


# ===========================================================================
# 8. Trade dataclass 新字段
# ===========================================================================

class TestTradeDataclass:
    def test_trade_default_order_type_market(self):
        t = Trade(
            date=pd.Timestamp("2024-01-02"), symbol="X", action="buy",
            price=100.0, shares=100, amount=10000.0,
            commission=5.0, stamp_tax=0.0, slippage_cost=0.0,
        )
        assert t.order_type == "market"
        assert t.entry_date is None

    def test_trade_with_entry_date(self):
        entry = pd.Timestamp("2024-01-02")
        t = Trade(
            date=pd.Timestamp("2024-01-10"), symbol="X", action="sell",
            price=110.0, shares=100, amount=11000.0,
            commission=5.0, stamp_tax=5.5, slippage_cost=0.0,
            pnl=1000.0, order_type="trailing_stop", entry_date=entry,
        )
        assert t.order_type == "trailing_stop"
        assert (t.date - t.entry_date).days == 8
