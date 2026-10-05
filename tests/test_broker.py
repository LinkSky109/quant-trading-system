"""经纪商（模拟盘）单元测试。

覆盖: 买入成交/卖出成交/手续费最低5元/滑点/印花税卖出单边/资金不足/持仓不足/平均成本/撤单/账户查询/RealBroker预留
"""
from __future__ import annotations

import pytest

from trading.broker import (
    AccountInfo,
    BaseBroker,
    BrokerType,
    Order,
    RealBroker,
    SimulatedBroker,
    create_broker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def broker() -> SimulatedBroker:
    return SimulatedBroker(initial_capital=1_000_000.0)


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

class TestInit:
    def test_initial_capital(self, broker):
        assert broker.cash == 1_000_000.0
        assert broker.initial_capital == 1_000_000.0

    def test_empty_positions(self, broker):
        assert broker.get_positions() == []

    def test_default_rates(self, broker):
        assert broker.commission_rate == 0.00025
        assert broker.stamp_tax_rate == 0.0005
        assert broker.slippage_rate == 0.001

    def test_factory_creates_simulated(self):
        b = create_broker("simulated", initial_capital=500_000)
        assert isinstance(b, SimulatedBroker)
        assert b.cash == 500_000.0

    def test_factory_unknown_type_raises(self):
        with pytest.raises(ValueError, match="不支持的券商类型"):
            create_broker("unknown_broker")


# ---------------------------------------------------------------------------
# 买入
# ---------------------------------------------------------------------------

class TestBuy:
    def test_buy_fills_and_updates_cash(self, broker):
        order = broker.submit_order("600519.SH", "buy", 100, price=1000.0)
        assert order.status == "filled"
        # 滑点: 1000 * 1.001 = 1001
        # 金额: 1001 * 100 = 100100
        # 手续费: max(100100*0.00025, 5) = max(25.025, 5) = 25.025
        # 总成本: 100100 + 25.025 = 100125.025
        expected_cost = 1001 * 100 + max(1001 * 100 * 0.00025, 5.0)
        assert broker.cash == pytest.approx(1_000_000 - expected_cost, abs=0.01)

    def test_buy_updates_position(self, broker):
        broker.submit_order("600519.SH", "buy", 100, price=1000.0)
        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "600519.SH"
        assert positions[0]["shares"] == 100
        # 平均成本 = 滑点后价格 = 1001
        assert positions[0]["avg_cost"] == pytest.approx(1001.0, abs=0.01)

    def test_buy_slippage_increases_price(self, broker):
        order = broker.submit_order("600519.SH", "buy", 100, price=1000.0)
        assert order.filled_price == pytest.approx(1001.0, abs=0.01)

    def test_buy_insufficient_cash_rejected(self, broker):
        # 10000股 * 1000 = 1000万 > 100万
        order = broker.submit_order("600519.SH", "buy", 10000, price=1000.0)
        assert order.status == "rejected"
        assert broker.cash == 1_000_000.0  # 现金不变
        assert broker.get_positions() == []

    def test_buy_commission_minimum_5_yuan(self, broker):
        # 小额交易: 10股 * 10元 = 100元, 手续费=0.025 < 5, 按5元收
        order = broker.submit_order("TEST", "buy", 10, price=10.0)
        assert order.status == "filled"
        # 滑点价=10.01, 金额=100.1, 手续费=max(0.025, 5)=5
        expected_cash = 1_000_000 - 100.1 - 5.0
        assert broker.cash == pytest.approx(expected_cash, abs=0.01)

    def test_buy_average_cost_when_adding(self, broker):
        # 第一次买入: 100股 @1001(滑点后)
        broker.submit_order("600519.SH", "buy", 100, price=1000.0)
        # 第二次买入: 100股 @1101.1(滑点后)
        broker.submit_order("600519.SH", "buy", 100, price=1100.0)
        pos = broker.get_positions()[0]
        assert pos["shares"] == 200
        # 平均成本 = (1001*100 + 1101.1*100) / 200 = 1051.05
        assert pos["avg_cost"] == pytest.approx(1051.05, abs=0.01)

    def test_buy_order_recorded(self, broker):
        order = broker.submit_order("600519.SH", "buy", 100, price=1000.0)
        retrieved = broker.get_order_status(order.order_id)
        assert retrieved.order_id == order.order_id
        assert retrieved.status == "filled"


# ---------------------------------------------------------------------------
# 卖出
# ---------------------------------------------------------------------------

class TestSell:
    def test_sell_fills_and_adds_cash(self, broker):
        # 先买入
        broker.submit_order("600519.SH", "buy", 100, price=1000.0)
        cash_after_buy = broker.cash
        # 卖出
        order = broker.submit_order("600519.SH", "sell", 100, price=1100.0)
        assert order.status == "filled"
        # 滑点: 1100 * 0.999 = 1098.9
        # 金额: 1098.9 * 100 = 109890
        # 手续费: max(109890*0.00025, 5) = 27.47
        # 印花税: 109890 * 0.0005 = 54.945
        # 净收入: 109890 - 27.47 - 54.945 = 109807.585
        net = 1098.9 * 100 - max(1098.9 * 100 * 0.00025, 5) - 1098.9 * 100 * 0.0005
        assert broker.cash == pytest.approx(cash_after_buy + net, abs=0.01)

    def test_sell_slippage_decreases_price(self, broker):
        broker.submit_order("600519.SH", "buy", 100, price=1000.0)
        order = broker.submit_order("600519.SH", "sell", 100, price=1000.0)
        assert order.filled_price == pytest.approx(999.0, abs=0.01)

    def test_sell_has_stamp_tax(self, broker):
        """卖出有印花税，买入没有。"""
        broker.submit_order("600519.SH", "buy", 100, price=1000.0)
        cash_before_sell = broker.cash
        broker.submit_order("600519.SH", "sell", 100, price=1000.0)
        # 如果没有印花税，净收入 = 金额 - 手续费
        # 有印花税，净收入更少
        amount = 999.0 * 100
        commission = max(amount * 0.00025, 5)
        stamp_tax = amount * 0.0005
        expected = cash_before_sell + amount - commission - stamp_tax
        assert broker.cash == pytest.approx(expected, abs=0.01)

    def test_sell_insufficient_position_rejected(self, broker):
        # 没有持仓就卖出
        order = broker.submit_order("600519.SH", "sell", 100, price=1000.0)
        assert order.status == "rejected"

    def test_sell_partial_position(self, broker):
        broker.submit_order("600519.SH", "buy", 200, price=1000.0)
        order = broker.submit_order("600519.SH", "sell", 100, price=1100.0)
        assert order.status == "filled"
        pos = broker.get_positions()[0]
        assert pos["shares"] == 100  # 剩100股

    def test_sell_all_resets_avg_cost(self, broker):
        broker.submit_order("600519.SH", "buy", 100, price=1000.0)
        broker.submit_order("600519.SH", "sell", 100, price=1100.0)
        # 清仓后持仓列表为空
        assert broker.get_positions() == []

    def test_sell_more_than_holding_rejected(self, broker):
        broker.submit_order("600519.SH", "buy", 100, price=1000.0)
        order = broker.submit_order("600519.SH", "sell", 200, price=1100.0)
        assert order.status == "rejected"


# ---------------------------------------------------------------------------
# 撤单
# ---------------------------------------------------------------------------

class TestCancel:
    def test_cancel_pending_order(self, broker):
        # 限价单不会立即成交（当前实现只有市价单立即成交）
        # 模拟一个pending订单
        order = Order(order_id="test123", symbol="X", action="buy",
                      order_type="limit", price=100, quantity=10, status="pending")
        broker.orders["test123"] = order
        result = broker.cancel_order("test123")
        assert result is True
        assert broker.orders["test123"].status == "cancelled"

    def test_cancel_filled_order_fails(self, broker):
        order = broker.submit_order("600519.SH", "buy", 100, price=1000.0)
        result = broker.cancel_order(order.order_id)
        assert result is False  # 已成交不能撤

    def test_cancel_nonexistent_order_fails(self, broker):
        assert broker.cancel_order("nonexistent") is False


# ---------------------------------------------------------------------------
# 账户查询
# ---------------------------------------------------------------------------

class TestAccount:
    def test_account_info_initial(self, broker):
        info = broker.get_account()
        assert isinstance(info, AccountInfo)
        assert info.total_asset == 1_000_000.0
        assert info.available_cash == 1_000_000.0
        assert info.market_value == 0.0
        assert info.frozen_cash == 0.0

    def test_account_info_after_trade(self, broker):
        broker.submit_order("600519.SH", "buy", 100, price=1000.0)
        info = broker.get_account()
        # 市值按平均成本计算
        assert info.market_value == pytest.approx(1001.0 * 100, abs=0.01)
        assert info.total_asset == pytest.approx(info.available_cash + info.market_value, abs=0.01)

    def test_get_order_status_nonexistent(self, broker):
        order = broker.get_order_status("nonexistent")
        assert order.order_id == "nonexistent"
        assert order.symbol == ""


# ---------------------------------------------------------------------------
# Order / AccountInfo 数据类
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_order_defaults(self):
        order = Order(order_id="1", symbol="X", action="buy", order_type="market")
        assert order.status == "pending"
        assert order.filled_price == 0.0
        assert order.filled_quantity == 0

    def test_account_info_positions_default(self):
        info = AccountInfo(total_asset=100, available_cash=100,
                           frozen_cash=0, market_value=0)
        assert info.positions == []


# ---------------------------------------------------------------------------
# BrokerType 枚举
# ---------------------------------------------------------------------------

class TestBrokerType:
    def test_enum_values(self):
        assert BrokerType.XTP.value == "xtp"
        assert BrokerType.QMT.value == "qmt"
        assert BrokerType.THS.value == "ths"
        assert BrokerType.CTP.value == "ctp"

    def test_enum_has_four_types(self):
        assert len(BrokerType) == 4


# ---------------------------------------------------------------------------
# RealBroker 预留接口
# ---------------------------------------------------------------------------

class TestRealBroker:
    def test_cannot_instantiate_directly(self):
        """RealBroker有抽象方法，不能直接实例化。"""
        with pytest.raises(TypeError):
            RealBroker()

    def test_cancel_order_raises_not_implemented(self):
        """RealBroker.cancel_order 预留为 NotImplementedError。"""
        # 创建一个最小子类来测试
        class MinimalReal(RealBroker):
            def connect(self, **kw): pass
            def get_account_info(self): return AccountInfo(0,0,0,0)
            def get_positions(self): return []
            def place_order(self, **kw): return Order(order_id="",symbol="",action="",order_type="")
            def get_order_status(self, oid): return Order(order_id=oid,symbol="",action="",order_type="")
            def get_trade_history(self, s, e): return []

        broker = MinimalReal()
        with pytest.raises(NotImplementedError, match="尚未实现"):
            broker.cancel_order("any")

    def test_submit_order_delegates_to_place_order(self):
        """RealBroker.submit_order 委托给 place_order。"""
        call_log = []
        class TestReal(RealBroker):
            def connect(self, **kw): pass
            def get_account_info(self): return AccountInfo(0,0,0,0)
            def get_positions(self): return []
            def place_order(self, symbol, side, quantity, price=0, order_type="market"):
                call_log.append((symbol, side, quantity))
                return Order(order_id="x", symbol=symbol, action=side, order_type=order_type)
            def get_order_status(self, oid): return Order(order_id=oid,symbol="",action="",order_type="")
            def get_trade_history(self, s, e): return []

        broker = TestReal()
        broker.submit_order("600519.SH", "buy", 100, price=1000.0)
        assert call_log == [("600519.SH", "buy", 100)]

    def test_get_account_delegates_to_get_account_info(self):
        """RealBroker.get_account 委托给 get_account_info。"""
        class TestReal(RealBroker):
            def connect(self, **kw): pass
            def get_account_info(self): return AccountInfo(999, 888, 0, 111)
            def get_positions(self): return []
            def place_order(self, **kw): return Order(order_id="",symbol="",action="",order_type="")
            def get_order_status(self, oid): return Order(order_id=oid,symbol="",action="",order_type="")
            def get_trade_history(self, s, e): return []

        broker = TestReal()
        info = broker.get_account()
        assert info.total_asset == 999
        assert info.available_cash == 888


# ---------------------------------------------------------------------------
# BaseBroker 抽象
# ---------------------------------------------------------------------------

class TestBaseBroker:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseBroker()
