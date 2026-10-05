"""实盘交易接口预留模块。

提供统一的 Broker 抽象基类，以及模拟盘实现。
实盘接入时只需继承 BaseBroker 并实现各方法即可。
"""
from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from .market_config import (
    calc_commission,
    calc_sec_fee,
    calc_stamp_tax,
    calc_trading_fee,
    get_market_key,
)

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """订单对象。

    支持的订单类型（``order_type``）:
        - ``market``: 市价单，立即以 ``price`` 成交（含滑点）。
        - ``limit``: 限价单，买入时 ``low <= limit_price`` 成交，
          卖出时 ``high >= limit_price`` 成交。
        - ``stop``: 止损单，价格穿破 ``stop_price`` 后触发市价成交。
        - ``stop_limit``: 止损限价单，先触达 ``stop_price`` 激活，
          再等待 ``limit_price`` 成交。
        - ``trailing_stop``: 移动止盈单，跟踪持仓期间最高价
          ``high_water_mark``，从最高点回撤 ``trailing_pct``（或回撤
          ``trailing_amount``）后触发卖出。
    """

    order_id: str
    symbol: str
    action: str  # buy / sell
    order_type: str  # market / limit / stop / stop_limit / trailing_stop
    price: float = 0.0
    quantity: int = 0
    status: str = "pending"  # pending / filled / cancelled / rejected
    filled_price: float = 0.0
    filled_quantity: int = 0
    created_at: str = ""
    updated_at: str = ""
    # ---- 高级订单扩展字段（非市价单使用） ----
    stop_price: float = 0.0
    """止损触发价（stop / stop_limit 用）。"""
    limit_price: float = 0.0
    """限价（limit / stop_limit 用）。"""
    trailing_pct: float = 0.0
    """移动止盈回撤比例，如 0.05 表示从最高点回撤 5% 卖出。"""
    trailing_amount: float = 0.0
    """移动止盈回撤金额（与 trailing_pct 二选一，pct 优先）。"""
    expiry: str = ""
    """有效期：空串=GTC 永久有效；``"day"``=当日有效，跨日自动撤销。"""
    high_water_mark: float = 0.0
    """移动止盈跟踪的最高价（内部维护）。"""
    triggered: bool = False
    """止损/止损限价单是否已触发（内部状态）。"""


@dataclass
class AccountInfo:
    """账户信息。"""
    total_asset: float
    available_cash: float
    frozen_cash: float
    market_value: float
    positions: List[Dict[str, Any]] = field(default_factory=list)


class BaseBroker(ABC):
    """券商接口抽象基类。

    实盘接入时继承此类并实现以下方法。
    """

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "market",
        price: float = 0.0,
    ) -> Order:
        """提交订单。"""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤销订单。"""
        ...

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """查询持仓。"""
        ...

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """查询资金账户。"""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """查询订单状态。"""
        ...


class SimulatedBroker(BaseBroker):
    """模拟盘券商实现。

    用于实盘前的模拟测试，维护虚拟资金和持仓。
    """

    def __init__(self, initial_capital: float = 1_000_000.0):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Order] = {}
        # 待成交（挂单）订单队列：限价/止损/止损限价/移动止盈单进入此列表，
        # 每根 K 线由 check_pending_orders() 驱动检查触发条件。
        self.pending_orders: List[Order] = []
        self.commission_rate = 0.00025
        self.stamp_tax_rate = 0.0005
        self.slippage_rate = 0.001

    def submit_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "market",
        price: float = 0.0,
        *,
        stop_price: float = 0.0,
        limit_price: float = 0.0,
        trailing_pct: float = 0.0,
        trailing_amount: float = 0.0,
        expiry: str = "",
    ) -> Order:
        """提交模拟订单。

        Args:
            symbol: 标的代码。
            action: ``"buy"`` 或 ``"sell"``。
            quantity: 委托股数。
            order_type: ``market`` / ``limit`` / ``stop`` / ``stop_limit`` /
                ``trailing_stop``。
            price: 委托参考价。市价单以此价成交（含滑点）；限价单若未显式
                传 ``limit_price`` 则以 ``price`` 作为限价；止损单若未显式传
                ``stop_price`` 则以 ``price`` 作为止损价。
            stop_price: 止损触发价（stop / stop_limit 用）。
            limit_price: 限价（limit / stop_limit 用）。
            trailing_pct: 移动止盈回撤比例（如 0.05 = 5%）。
            trailing_amount: 移动止盈回撤金额（与 trailing_pct 二选一）。
            expiry: ``""``=GTC；``"day"``=当日有效，跨日自动撤销。

        Returns:
            已创建的 :class:`Order`。市价单立即成交；非市价单进入 pending。
        """
        order_id = str(uuid.uuid4())[:8]
        now = pd.Timestamp.now().isoformat()

        # 限价/止损价的回退：未显式传入时用 price
        effective_limit = limit_price or price
        effective_stop = stop_price or price

        order = Order(
            order_id=order_id,
            symbol=symbol,
            action=action,
            order_type=order_type,
            price=price,
            quantity=quantity,
            created_at=now,
            updated_at=now,
            stop_price=effective_stop,
            limit_price=effective_limit,
            trailing_pct=trailing_pct,
            trailing_amount=trailing_amount,
            expiry=expiry,
            high_water_mark=price if order_type == "trailing_stop" else 0.0,
        )

        # 模拟市价单立即成交
        if order_type == "market":
            self._fill_market_order(order, price)
        else:
            # 非市价单进入待成交队列（不冻结资金，成交时再扣减）
            order.status = "pending"
            self.pending_orders.append(order)
            logger.info("挂单: %s %s %d股 type=%s limit=%.2f stop=%.2f",
                        action, symbol, quantity, order_type,
                        effective_limit, effective_stop)

        self.orders[order_id] = order
        return order

    # ------------------------------------------------------------------
    # 成交内部逻辑
    # ------------------------------------------------------------------

    def _fill_market_order(self, order: Order, ref_price: float) -> None:
        """按市价单逻辑立即成交（含滑点、手续费、印花税）。"""
        fill_price = ref_price * (1 + self.slippage_rate) if order.action == "buy" \
            else ref_price * (1 - self.slippage_rate)
        self._settle_fill(order, fill_price)

    def _calc_fees(
        self, symbol: str, amount: float, shares: int, action: str
    ) -> tuple[float, float, float]:
        """按标的所在市场计算费用。

        A股沿用 broker 级 ``commission_rate`` / ``stamp_tax_rate``（向后兼容）；
        美股/港股走 :mod:`trading.market_config` 的多费率结构。

        Returns:
            (commission, stamp_tax, other_fees)。
            other_fees = 美股 SEC 费 + 港股交易费。
        """
        if get_market_key(symbol) == "CN":
            commission = max(amount * self.commission_rate, 5.0)
            stamp_tax = amount * self.stamp_tax_rate if action == "sell" else 0.0
            return commission, stamp_tax, 0.0

        commission = calc_commission(symbol, amount, shares, action)
        stamp_tax = calc_stamp_tax(symbol, amount, action)
        other_fees = calc_sec_fee(symbol, amount, action) + calc_trading_fee(
            symbol, amount, action
        )
        return commission, stamp_tax, other_fees

    def _settle_fill(self, order: Order, fill_price: float) -> None:
        """按给定成交价结算订单：更新现金/持仓，写回 order 成交字段。

        资金不足（买入）或持仓不足（卖出）时把订单标记为 rejected，
        与原市价单行为保持一致。
        """
        amount = fill_price * order.quantity
        commission, stamp_tax, other_fees = self._calc_fees(
            order.symbol, amount, order.quantity, order.action
        )

        if order.action == "buy":
            total_cost = amount + commission + other_fees
            if total_cost > self.cash:
                order.status = "rejected"
                logger.warning("模拟单拒绝: 资金不足")
                return
            self.cash -= total_cost
            pos = self.positions.get(order.symbol, {"shares": 0, "avg_cost": 0.0})
            new_shares = pos["shares"] + order.quantity
            pos["avg_cost"] = (pos["avg_cost"] * pos["shares"]
                               + fill_price * order.quantity) / new_shares
            pos["shares"] = new_shares
            self.positions[order.symbol] = pos
        else:  # sell
            pos = self.positions.get(order.symbol)
            if pos is None or pos["shares"] < order.quantity:
                order.status = "rejected"
                logger.warning("模拟单拒绝: 持仓不足")
                return
            net = amount - commission - stamp_tax - other_fees
            self.cash += net
            pos["shares"] -= order.quantity
            if pos["shares"] == 0:
                pos["avg_cost"] = 0.0

        order.status = "filled"
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.updated_at = pd.Timestamp.now().isoformat()
        logger.info("模拟单成交: %s %s %d股 @ %.2f",
                    order.action, order.symbol, order.quantity, fill_price)

    # ------------------------------------------------------------------
    # 待成交订单驱动
    # ------------------------------------------------------------------

    def check_pending_orders(
        self,
        current_price: float,
        high: float,
        low: float,
        date: pd.Timestamp,
    ) -> List[Order]:
        """每根 K 线调用一次，检查待成交订单是否触发并成交。

        成交规则：
            - ``limit buy``: ``low <= limit_price`` → 以 limit_price 成交（买入加滑点）。
            - ``limit sell``: ``high >= limit_price`` → 以 limit_price 成交（卖出减滑点）。
            - ``stop sell``: ``low <= stop_price`` → 触发，按 stop_price 卖出。
            - ``stop buy``: ``high >= stop_price`` → 触发，按 stop_price 买入。
            - ``stop_limit``: 先穿破 stop_price 激活，再等待 limit_price 成交。
            - ``trailing_stop sell``: 每日更新 high_water_mark=max(..., high)，
              当 ``low <= high_water_mark*(1-trailing_pct)``
              （或 ``low <= high_water_mark - trailing_amount``）时卖出。
            - ``expiry="day"`` 的订单在日期变更时自动撤销。

        Args:
            current_price: 当前价（收盘价或参考价）。
            high: 当日最高价。
            low: 当日最低价。
            date: 当前交易日。

        Returns:
            本次检查中成交的订单列表。
        """
        filled: List[Order] = []
        expired: List[Order] = []

        for order in list(self.pending_orders):
            if order.status != "pending":
                self.pending_orders.remove(order)
                continue

            # 当日有效订单跨日撤销
            if order.expiry == "day":
                try:
                    created_date = pd.Timestamp(order.created_at).normalize()
                except (ValueError, TypeError):
                    created_date = pd.Timestamp(order.created_at)
                if pd.Timestamp(date).normalize() != created_date:
                    order.status = "cancelled"
                    order.updated_at = pd.Timestamp.now().isoformat()
                    expired.append(order)
                    self.pending_orders.remove(order)
                    continue

            fill_price = self._maybe_trigger(order, current_price, high, low)
            if fill_price is None:
                continue

            # 触发成交（含滑点与手续费/印花税结算）
            self._settle_fill(order, fill_price)
            if order.status == "filled":
                filled.append(order)
            self.pending_orders.remove(order)

        if expired:
            logger.info("当日有效订单自动撤销 %d 笔", len(expired))
        return filled

    def _maybe_trigger(
        self,
        order: Order,
        current_price: float,
        high: float,
        low: float,
    ) -> Optional[float]:
        """根据订单类型与当日 high/low 判断是否触发，返回成交价（未触发返回 None）。"""
        otype = order.order_type

        # ---- 限价单 ----
        if otype == "limit":
            if order.action == "buy" and low <= order.limit_price:
                return order.limit_price * (1 + self.slippage_rate)
            if order.action == "sell" and high >= order.limit_price:
                return order.limit_price * (1 - self.slippage_rate)
            return None

        # ---- 止损单（触发后市价成交） ----
        if otype == "stop":
            if order.action == "sell" and low <= order.stop_price:
                order.triggered = True
                return order.stop_price * (1 - self.slippage_rate)
            if order.action == "buy" and high >= order.stop_price:
                order.triggered = True
                return order.stop_price * (1 + self.slippage_rate)
            return None

        # ---- 止损限价单 ----
        if otype == "stop_limit":
            if not order.triggered:
                # 阶段一：等待穿破 stop_price
                if order.action == "sell" and low <= order.stop_price:
                    order.triggered = True
                elif order.action == "buy" and high >= order.stop_price:
                    order.triggered = True
                else:
                    return None
            # 阶段二：等待 limit_price 成交
            if order.action == "sell" and high >= order.limit_price:
                return order.limit_price * (1 - self.slippage_rate)
            if order.action == "buy" and low <= order.limit_price:
                return order.limit_price * (1 + self.slippage_rate)
            return None

        # ---- 移动止盈（默认卖出方向） ----
        if otype == "trailing_stop":
            # 跟踪最高价
            order.high_water_mark = max(order.high_water_mark, high)
            if order.action == "sell":
                if order.trailing_pct > 0:
                    trigger_price = order.high_water_mark * (1 - order.trailing_pct)
                elif order.trailing_amount > 0:
                    trigger_price = order.high_water_mark - order.trailing_amount
                else:
                    return None
                if low <= trigger_price:
                    order.triggered = True
                    return trigger_price * (1 - self.slippage_rate)
            else:  # trailing buy（跟踪最低价，向上突破买入，少见但支持）
                if order.trailing_pct > 0:
                    trigger_price = order.high_water_mark * (1 + order.trailing_pct)
                elif order.trailing_amount > 0:
                    trigger_price = order.high_water_mark + order.trailing_amount
                else:
                    return None
                if high >= trigger_price:
                    order.triggered = True
                    return trigger_price * (1 + self.slippage_rate)
            return None

        return None

    def get_pending_orders(self) -> List[Order]:
        """返回所有待成交订单。"""
        return list(self.pending_orders)

    def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if order and order.status == "pending":
            order.status = "cancelled"
            order.updated_at = pd.Timestamp.now().isoformat()
            # 同步从待成交队列移除
            self.pending_orders = [o for o in self.pending_orders
                                   if o.order_id != order_id]
            return True
        return False

    def get_positions(self) -> List[Dict[str, Any]]:
        return [
            {"symbol": sym, "shares": p["shares"], "avg_cost": p["avg_cost"]}
            for sym, p in self.positions.items() if p["shares"] > 0
        ]

    def get_account(self) -> AccountInfo:
        market_value = sum(p["shares"] * p["avg_cost"] for p in self.positions.values())
        return AccountInfo(
            total_asset=self.cash + market_value,
            available_cash=self.cash,
            frozen_cash=0.0,
            market_value=market_value,
            positions=self.get_positions(),
        )

    def get_order_status(self, order_id: str) -> Order:
        return self.orders.get(order_id, Order(order_id=order_id, symbol="", action="", order_type=""))


class BrokerType(Enum):
    """券商接入类型枚举。"""

    XTP = "xtp"      # 中泰 XTP
    QMT = "qmt"      # 迅投 QMT
    THS = "ths"      # 同花顺
    CTP = "ctp"      # 期货 CTP


class RealBroker(BaseBroker):
    """真实券商接入抽象基类。

    继承 BaseBroker，定义实盘交易所需的完整接口。
    当前为预留定义，所有方法均未实现。

    使用前需用户提供：
      1. 券商类型（XTP/QMT/THS/CTP）
      2. 券商 API 密钥（api_key / api_secret）
      3. 资金账号（account_id）

    实现时需继承本类并实现所有抽象方法，
    然后在 create_broker() 工厂中注册对应 broker_type。
    """

    def __init__(self) -> None:
        """初始化真实券商连接状态。"""
        self._connected: bool = False

    @abstractmethod
    def connect(
        self,
        account_id: str,
        api_key: str,
        api_secret: str,
        broker_type: BrokerType = BrokerType.XTP,
    ) -> None:
        """连接券商 API，建立会话。

        Args:
            account_id: 资金账号。
            api_key: 券商 API Key。
            api_secret: 券商 API Secret。
            broker_type: 券商类型，默认 XTP。
        """
        ...

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """获取真实账户资金信息（总资产/可用/冻结/市值）。"""
        ...

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:  # type: ignore[override]
        """获取真实持仓列表。"""
        ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float = 0.0,
        order_type: str = "market",
    ) -> Order:
        """真实下单接口。

        Args:
            symbol: 标的代码。
            side: 买卖方向，buy / sell。
            quantity: 委托数量。
            price: 委托价格，市价单可为 0。
            order_type: market / limit。
        """
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:  # type: ignore[override]
        """查询订单状态。"""
        ...

    @abstractmethod
    def get_trade_history(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """查询成交历史。

        Args:
            start_date: 开始日期（YYYY-MM-DD）。
            end_date: 结束日期（YYYY-MM-DD）。
        """
        ...

    # ---- 委托 BaseBroker 抽象方法到真实接口 ----

    def submit_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "market",
        price: float = 0.0,
    ) -> Order:
        """委托给 place_order，保持与 BaseBroker 接口兼容。"""
        return self.place_order(
            symbol=symbol,
            side=action,
            quantity=quantity,
            price=price,
            order_type=order_type,
        )

    def cancel_order(self, order_id: str) -> bool:
        """撤销订单。"""
        raise NotImplementedError(
            "真实券商撤单尚未实现，请在子类中覆盖 cancel_order 方法"
        )

    def get_account(self) -> AccountInfo:
        """委托给 get_account_info，保持与 BaseBroker 接口兼容。"""
        return self.get_account_info()


def create_broker(broker_type: str = "simulated", **kwargs: Any) -> BaseBroker:
    """工厂方法创建券商实例。

    Args:
        broker_type: 券商类型，当前仅支持 "simulated"（模拟盘）。
            实盘接入需继承 RealBroker 并在此注册对应类型，
            可选的真实券商类型见 BrokerType 枚举（xtp / qmt / ths / ctp）。
    """
    if broker_type == "simulated":
        return SimulatedBroker(**kwargs)
    raise ValueError(
        f"不支持的券商类型: {broker_type}，实盘请继承 RealBroker 实现"
    )
