"""回测引擎核心模块。

事件驱动的日频回测，严格避免未来函数：
- 策略信号已 shift(1)，在 T 日产生的信号于 T+1 日开盘执行
- 所有指标计算仅使用当日及之前的数据
- 成交价格 = 开盘价 × (1 ± 滑点)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.metrics import calc_all_metrics, metrics_to_dataframe
from jev.jev_engine import JevDecisionEngine
from risk.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy
from strategies.strategy_engine import StrategyEngine
from trading.market_config import (
    calc_commission,
    calc_sec_fee,
    calc_stamp_tax,
    calc_trading_fee,
    get_market_key,
    lot_size_of,
)
from utils.indicators import add_indicators

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """持仓记录。"""
    symbol: str
    shares: int = 0
    avg_cost: float = 0.0
    entry_date: Optional[pd.Timestamp] = None

    @property
    def market_value(self) -> float:
        return self.shares * self.avg_cost  # 需外部更新最新价

    def update_cost(self, price: float, shares: int, commission: float) -> None:
        """加仓时更新平均成本。"""
        total_cost = self.avg_cost * self.shares + price * shares + commission
        total_shares = self.shares + shares
        self.avg_cost = total_cost / total_shares if total_shares > 0 else 0
        self.shares = total_shares


@dataclass
class Trade:
    """交易记录。"""
    date: pd.Timestamp
    symbol: str
    action: str  # buy / sell
    price: float
    shares: int
    amount: float
    commission: float
    stamp_tax: float
    slippage_cost: float
    pnl: Optional[float] = None  # 平仓时记录
    reason: str = ""
    order_type: str = "market"
    """产生该笔成交的订单类型（market/limit/stop/stop_limit/trailing_stop）。"""
    entry_date: Optional[pd.Timestamp] = None
    """开仓日期（平仓卖出时回填，用于平均持仓时间统计）。"""


@dataclass
class BacktestResult:
    """回测结果容器。"""
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    trades: List[Trade]
    metrics: Dict[str, float]
    metrics_df: pd.DataFrame
    daily_returns: pd.Series
    positions_history: List[Dict[str, Any]] = field(default_factory=list)


class BacktestEngine:
    """日频回测引擎。

    支持单标的/多标的、Jev 信号过滤、风控规则、参数扫描。
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.0005,
        slippage_rate: float = 0.001,
        risk_free_rate: float = 0.02,
        trading_days: int = 252,
        risk_manager: Optional[RiskManager] = None,
        jev_engine: Optional[JevDecisionEngine] = None,
        walkthrough_snapshots: Optional[List[Dict[str, Any]]] = None,
        order_type: str = "market",
    ):
        """初始化回测引擎。

        Args:
            order_type: 订单模式，默认 ``"market"`` 保持向后兼容。
                - ``"market"``: 信号产生后以开盘价立即市价成交。
                - ``"limit"``: 买入信号挂限价买单（开盘价×0.995），
                  卖出信号挂限价卖单（开盘价×1.005），当日 high/low 触发成交。
                - ``"stop"``: 买入后自动附带止损卖单
                  （止损价=买入价×(1-止损比例)，止损比例取
                  ``risk_manager.single_stop_loss`` 或默认 0.03）。
                - ``"trailing_stop"``: 买入后自动附带移动止盈卖单
                  （trailing_pct=0.05）。
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage_rate = slippage_rate
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.risk_manager = risk_manager
        self.jev_engine = jev_engine
        self.order_type = order_type

        # 走查（walkthrough）记录钩子：传入 list 时逐日记录完整快照；
        # 为 None 时回测行为与原来完全一致（零开销、零副作用）。
        self._walkthrough_snapshots = walkthrough_snapshots
        # 当前交易日的信号/操作明细，由 _daily_step 开头重置，
        # 供 _check_risk_controls（止损/止盈）与信号处理循环追加。
        self._current_day_signals: List[Dict[str, Any]] = []

        # 运行时状态
        self.cash: float = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[pd.Timestamp, float]] = []
        self.positions_history: List[Dict[str, Any]] = []

        # ---- 高级订单状态 ----
        # 待成交订单队列（复用 trading.broker.Order 作为数据结构，
        # 但成交结算逻辑由本引擎内聚处理，不依赖 SimulatedBroker）。
        from trading.broker import Order as _Order
        self._Order = _Order
        self.pending_orders: List[_Order] = []
        # 订单统计：提交总数 / 成交数（用于订单成交率指标）
        self._orders_submitted: int = 0
        self._orders_filled: int = 0

    # ------------------------------------------------------------------
    # 主回测入口
    # ------------------------------------------------------------------

    def run(
        self,
        data: Dict[str, pd.DataFrame] | pd.DataFrame,
        strategy: BaseStrategy | StrategyEngine,
        symbol: str = "",
    ) -> BacktestResult:
        """执行回测。

        Args:
            data: 单标的 DataFrame 或多标的 {symbol: df} 字典。
            strategy: 策略实例或策略引擎。
            symbol: 单标的模式下的标的代码。

        Returns:
            BacktestResult 回测结果。
        """
        # 统一为多标的格式
        if isinstance(data, pd.DataFrame):
            data = {symbol or "UNKNOWN": data}

        # 预处理：添加技术指标、对齐日期
        processed = {}
        for sym, df in data.items():
            df = add_indicators(df.copy())
            processed[sym] = df

        # 获取所有交易日并集
        all_dates = sorted(set().union(*[set(df.index) for df in processed.values()]))

        # 预计算每个标的的信号序列
        signal_map: Dict[str, pd.DataFrame] = {}
        for sym, df in processed.items():
            if isinstance(strategy, StrategyEngine):
                sig_df = strategy.get_combined_signals(df, sym)
            else:
                sig_df = strategy.get_signal_dataframe(df, sym)
            signal_map[sym] = sig_df

        # 重置状态
        self._reset()

        # 逐日回测
        for date in all_dates:
            self._daily_step(date, processed, signal_map)

        # 期末平仓（按最后一天收盘价）
        if all_dates:
            self._close_all_positions(all_dates[-1], processed, "期末平仓")

        # 构建结果
        equity = pd.Series(
            [v for _, v in self.equity_history],
            index=[d for d, _ in self.equity_history],
            name="equity",
        )
        # 基准：等权买入持有
        benchmark = self._build_benchmark(processed, all_dates)
        daily_returns = equity.pct_change().fillna(0)
        metrics = calc_all_metrics(
            equity, [t.__dict__ for t in self.trades],
            self.risk_free_rate, self.trading_days,
            total_orders=self._orders_submitted,
            filled_orders=self._orders_filled,
        )
        metrics_df = metrics_to_dataframe(metrics)

        logger.info(
            "回测完成: %d 笔交易, 累计收益 %.2f%%, 最大回撤 %.2f%%",
            metrics["交易次数"],
            metrics["累计收益率"] * 100,
            metrics["最大回撤"] * 100,
        )

        return BacktestResult(
            equity_curve=equity,
            benchmark_curve=benchmark,
            trades=self.trades,
            metrics=metrics,
            metrics_df=metrics_df,
            daily_returns=daily_returns,
            positions_history=self.positions_history,
        )

    # ------------------------------------------------------------------
    # 单日回测逻辑
    # ------------------------------------------------------------------

    def _daily_step(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
        signal_map: Dict[str, pd.DataFrame],
    ) -> None:
        """执行单日回测逻辑。"""
        # 走查模式：重置当日信号/操作明细（供止损止盈与信号循环追加）
        if self._walkthrough_snapshots is not None:
            self._current_day_signals = []

        # 1. 先检查持仓的止损/止盈（以当日开盘价判断）
        self._check_risk_controls(date, data)

        # 1.5 处理待成交高级订单（限价/止损/移动止盈），用当日 high/low 触发。
        # 必须在策略信号之前处理：先让已有挂单成交，再根据最新持仓状态响应新信号。
        self._process_pending_orders(date, data)

        # 2. 处理信号（信号已 shift，代表前一日产生，今日执行）
        for sym, sig_df in signal_map.items():
            if date not in sig_df.index:
                continue
            if date not in data[sym].index:
                continue

            sig_row = sig_df.loc[date]
            signal = sig_row.get("signal", 0)
            confidence = sig_row.get("confidence", 0.0)

            if pd.isna(signal) or signal == 0:
                continue

            action = "buy" if signal > 0 else "sell"

            # 走查：为该信号建立记录骨架（无论后续是否被拦截都要记录原因）
            rec: Optional[Dict[str, Any]] = None
            if self._walkthrough_snapshots is not None:
                rec = {
                    "symbol": sym,
                    "signal": action,
                    "confidence": float(confidence) if not pd.isna(confidence) else 0.0,
                    "jev_filtered": False,
                    "jev_decision": None,
                    "risk_blocked": False,
                    "executed": False,
                    "action": action,
                    "fill_price": None,
                    "shares": 0,
                    "reason": "",
                }

            # Jev 过滤
            if self.jev_engine is not None:
                df = data[sym]
                idx = df.index.get_loc(date)
                state = self.jev_engine.build_market_state(df, idx)
                if state is not None:
                    decision = self.jev_engine.evaluate(
                        raw_signal=action,
                        raw_confidence=float(confidence),
                        market_state=state,
                        symbol=sym,
                    )
                    if rec is not None:
                        rec["jev_decision"] = self._jev_decision_to_dict(decision)
                    if not decision.executed:
                        if rec is not None:
                            # 记录信号被 Jev 拦截/过滤的原因
                            rec["jev_filtered"] = True
                            rec["reason"] = decision.reason
                            self._current_day_signals.append(rec)
                        continue
                    confidence = decision.final_confidence

            # 风控检查
            if self.risk_manager is not None:
                if not self.risk_manager.check_trade_allowed(
                    symbol=sym,
                    action=action,
                    price=float(data[sym].loc[date, "open"]),
                    equity=self._current_equity(date, data),
                    cash=self.cash,
                    positions=self.positions,
                ):
                    if rec is not None:
                        rec["risk_blocked"] = True
                        rec["reason"] = "风控拦截（仓位/回撤/交易频率限制）"
                        self._current_day_signals.append(rec)
                    continue

            # 执行交易
            n_trades_before = len(self.trades)
            if self.order_type == "limit":
                # 限价模式：信号产生挂单，等待后续 high/low 触发成交
                open_price = float(data[sym].loc[date, "open"])
                if action == "buy":
                    limit_price = open_price * 0.995
                    self._submit_pending_order(
                        symbol=sym, action="buy", order_type="limit",
                        limit_price=limit_price, reason="限价买入",
                    )
                else:
                    limit_price = open_price * 1.005
                    self._submit_pending_order(
                        symbol=sym, action="sell", order_type="limit",
                        limit_price=limit_price, reason="限价卖出",
                    )
            elif action == "buy":
                self._execute_buy(date, sym, data[sym], confidence)
            elif action == "sell":
                self._execute_sell(date, sym, data[sym], "策略卖出")

            if rec is not None:
                if len(self.trades) > n_trades_before:
                    t = self.trades[-1]
                    rec["executed"] = True
                    rec["fill_price"] = float(t.price)
                    rec["shares"] = int(t.shares)
                    rec["reason"] = t.reason
                else:
                    rec["reason"] = rec["reason"] or "未成交（已持仓/资金不足/无持仓可卖）"
                self._current_day_signals.append(rec)

        # 3. 记录当日净值
        equity = self._current_equity(date, data)
        self.equity_history.append((date, equity))

        # 记录持仓快照
        snapshot = {"date": date, "equity": equity, "cash": self.cash}
        for sym, pos in self.positions.items():
            if pos.shares > 0 and date in data[sym].index:
                snapshot[f"{sym}_shares"] = pos.shares
                snapshot[f"{sym}_price"] = float(data[sym].loc[date, "close"])
        self.positions_history.append(snapshot)

        # 走查：追加日末完整快照（信号/Jev/操作/持仓/盈亏）
        if self._walkthrough_snapshots is not None:
            self._record_walkthrough_day(date, data, equity)

    # ------------------------------------------------------------------
    # 交易执行
    # ------------------------------------------------------------------

    def _execute_buy(
        self, date: pd.Timestamp, symbol: str, df: pd.DataFrame, confidence: float
    ) -> None:
        """执行买入。"""
        if symbol in self.positions and self.positions[symbol].shares > 0:
            return  # 已持仓则不重复买入

        open_price = float(df.loc[date, "open"])
        fill_price = open_price * (1 + self.slippage_rate)

        # 仓位计算
        equity = self._current_equity(date, {symbol: df})
        if self.risk_manager:
            target_value = self.risk_manager.calc_position_size(
                symbol=symbol,
                price=fill_price,
                equity=equity,
                cash=self.cash,
                confidence=confidence,
                positions=self.positions,
            )
        else:
            target_value = equity * 0.2  # 默认单标的20%

        shares = self._round_lot(symbol, target_value / fill_price)  # 按市场取整一手
        if shares <= 0:
            return

        amount = fill_price * shares
        commission, stamp_tax, other_fees = self._calc_trade_fees(symbol, amount, shares, "buy")
        slippage_cost = (fill_price - open_price) * shares
        total_cost = amount + commission + other_fees

        if total_cost > self.cash:
            # 资金不足，调整
            shares = self._round_lot(symbol, self.cash * 0.99 / fill_price)
            if shares <= 0:
                return
            amount = fill_price * shares
            commission, stamp_tax, other_fees = self._calc_trade_fees(symbol, amount, shares, "buy")
            slippage_cost = (fill_price - open_price) * shares
            total_cost = amount + commission + other_fees

        self.cash -= total_cost

        pos = self.positions.get(symbol, Position(symbol=symbol))
        pos.shares += shares
        pos.avg_cost = fill_price  # 新建仓
        pos.entry_date = date
        self.positions[symbol] = pos

        self.trades.append(Trade(
            date=date, symbol=symbol, action="buy",
            price=fill_price, shares=shares, amount=amount,
            commission=commission + other_fees, stamp_tax=0.0,
            slippage_cost=slippage_cost, reason="策略买入",
            order_type="market", entry_date=date,
        ))
        self._orders_submitted += 1
        self._orders_filled += 1
        logger.debug("买入 %s %d股 @ %.2f", symbol, shares, fill_price)

        # stop / trailing_stop 模式：买入成功后自动附带退出卖单
        if self.order_type == "stop":
            stop_pct = (self.risk_manager.single_stop_loss
                        if self.risk_manager is not None else 0.03)
            stop_price = fill_price * (1 - stop_pct)
            self._submit_pending_order(
                symbol=symbol, action="sell", order_type="stop",
                stop_price=stop_price, reason=f"止损单({stop_pct*100:.1f}%)",
                shares=shares,
            )
        elif self.order_type == "trailing_stop":
            self._submit_pending_order(
                symbol=symbol, action="sell", order_type="trailing_stop",
                trailing_pct=0.05, reason="移动止盈(5%)",
                shares=shares, high_water_mark=fill_price,
            )

    def _execute_sell(
        self, date: pd.Timestamp, symbol: str, df: pd.DataFrame, reason: str
    ) -> None:
        """执行卖出（全部平仓）。"""
        pos = self.positions.get(symbol)
        if pos is None or pos.shares <= 0:
            return

        open_price = float(df.loc[date, "open"])
        fill_price = open_price * (1 - self.slippage_rate)
        shares = pos.shares

        amount = fill_price * shares
        commission, stamp_tax, other_fees = self._calc_trade_fees(symbol, amount, shares, "sell")
        slippage_cost = (open_price - fill_price) * shares
        net_proceeds = amount - commission - stamp_tax - other_fees

        # 计算盈亏
        cost_basis = pos.avg_cost * shares
        pnl = net_proceeds - cost_basis
        entry_date = pos.entry_date  # 平仓前记录开仓日期

        self.cash += net_proceeds
        pos.shares = 0
        pos.avg_cost = 0.0
        pos.entry_date = None

        self.trades.append(Trade(
            date=date, symbol=symbol, action="sell",
            price=fill_price, shares=shares, amount=amount,
            commission=commission + other_fees, stamp_tax=stamp_tax,
            slippage_cost=slippage_cost, pnl=pnl, reason=reason,
            order_type="market", entry_date=entry_date,
        ))
        self._orders_submitted += 1
        self._orders_filled += 1

        # 平仓后取消该标的剩余挂单（防止重复平仓）
        self.pending_orders = [
            o for o in self.pending_orders
            if not (o.symbol == symbol and o.action == "sell")
        ]

        # 更新风控状态
        if self.risk_manager:
            self.risk_manager.record_trade(pnl)

        logger.debug("卖出 %s %d股 @ %.2f, 盈亏 %.2f", symbol, shares, fill_price, pnl)

    def _close_all_positions(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame], reason: str
    ) -> None:
        """平掉所有持仓。"""
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            if pos.shares > 0 and sym in data and date in data[sym].index:
                self._execute_sell(date, sym, data[sym], reason)
        # 期末取消所有未成交挂单
        self._cancel_all_pending_orders()

    # ------------------------------------------------------------------
    # 高级订单队列（限价/止损/移动止盈）
    # ------------------------------------------------------------------

    def _submit_pending_order(
        self,
        symbol: str,
        action: str,
        order_type: str,
        *,
        limit_price: float = 0.0,
        stop_price: float = 0.0,
        trailing_pct: float = 0.0,
        reason: str = "",
        shares: int = 0,
        high_water_mark: float = 0.0,
    ) -> Any:
        """创建并登记一笔待成交订单。

        资金/持仓不立即冻结，成交时在 :meth:`_fill_pending_order` 中结算。
        """
        order = self._Order(
            order_id=f"ord_{self._orders_submitted:06d}",
            symbol=symbol,
            action=action,
            order_type=order_type,
            price=limit_price or stop_price,
            quantity=shares,
            limit_price=limit_price,
            stop_price=stop_price,
            trailing_pct=trailing_pct,
            high_water_mark=high_water_mark,
            status="pending",
            created_at=pd.Timestamp.now().isoformat(),
        )
        order.reason = reason  # type: ignore[attr-defined]
        self.pending_orders.append(order)
        self._orders_submitted += 1
        return order

    def _cancel_all_pending_orders(self) -> None:
        """取消所有待成交订单（期末/换仓时调用）。"""
        for o in self.pending_orders:
            o.status = "cancelled"
        self.pending_orders.clear()

    def _process_pending_orders(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> None:
        """逐笔检查待成交订单，用当日 high/low 判断是否触发成交。

        执行顺序：
            1. 更新移动止盈单的 high_water_mark（用当日 high）。
            2. 根据订单类型与 high/low 判断触发。
            3. 买入类挂单成交前过风控 check_trade_allowed；
               卖出类挂单视为平仓，直接成交并 record_trade。
        """
        if not self.pending_orders:
            return

        still_pending: List[Any] = []
        for order in list(self.pending_orders):
            if order.status != "pending":
                continue
            sym = order.symbol
            if sym not in data or date not in data[sym].index:
                still_pending.append(order)
                continue

            df = data[sym]
            day_high = float(df.loc[date, "high"])
            day_low = float(df.loc[date, "low"])
            day_close = float(df.loc[date, "close"])

            # 移动止盈：先更新跟踪最高价
            if order.order_type == "trailing_stop":
                order.high_water_mark = max(order.high_water_mark, day_high)

            fill_price = self._maybe_trigger_price(order, day_high, day_low)
            if fill_price is None:
                still_pending.append(order)
                continue

            # ---- 风控约束 ----
            # 买入挂单成交前过开仓风控
            if order.action == "buy" and self.risk_manager is not None:
                if not self.risk_manager.check_trade_allowed(
                    symbol=sym, action="buy", price=fill_price,
                    equity=self._current_equity(date, data),
                    cash=self.cash, positions=self.positions,
                ):
                    # 风控拦截：挂单保留，次日再试
                    still_pending.append(order)
                    continue

            # 执行成交
            filled = self._fill_pending_order(date, df, order, fill_price,
                                             day_close=day_close)
            if not filled:
                # 成交失败（资金不足/无持仓），保留挂单次日再试
                still_pending.append(order)

        self.pending_orders = still_pending

    @staticmethod
    def _maybe_trigger_price(
        order: Any, day_high: float, day_low: float
    ) -> Optional[float]:
        """根据订单类型与当日 high/low 计算成交价，未触发返回 None。"""
        otype = order.order_type

        if otype == "limit":
            if order.action == "buy" and day_low <= order.limit_price:
                return order.limit_price  # 由 _fill_pending_order 加滑点
            if order.action == "sell" and day_high >= order.limit_price:
                return order.limit_price
            return None

        if otype == "stop":
            if order.action == "sell" and day_low <= order.stop_price:
                order.triggered = True
                return order.stop_price
            if order.action == "buy" and day_high >= order.stop_price:
                order.triggered = True
                return order.stop_price
            return None

        if otype == "stop_limit":
            if not order.triggered:
                if order.action == "sell" and day_low <= order.stop_price:
                    order.triggered = True
                elif order.action == "buy" and day_high >= order.stop_price:
                    order.triggered = True
                else:
                    return None
            if order.action == "sell" and day_high >= order.limit_price:
                return order.limit_price
            if order.action == "buy" and day_low <= order.limit_price:
                return order.limit_price
            return None

        if otype == "trailing_stop" and order.action == "sell":
            if order.trailing_pct > 0:
                trigger = order.high_water_mark * (1 - order.trailing_pct)
            elif order.trailing_amount > 0:
                trigger = order.high_water_mark - order.trailing_amount
            else:
                return None
            if day_low <= trigger:
                order.triggered = True
                return trigger
            return None

        return None

    def _fill_pending_order(
        self,
        date: pd.Timestamp,
        df: pd.DataFrame,
        order: Any,
        ref_price: float,
        day_close: float,
    ) -> bool:
        """按给定参考价结算一笔已触发的待成交订单。

        Returns:
            True 表示成交成功；False 表示资金/持仓不足等原因未成交。
        """
        sym = order.symbol

        if order.action == "buy":
            # 买入：按 ref_price 加滑点成交，仓位按当前权益动态计算
            fill_price = ref_price * (1 + self.slippage_rate)
            equity = self._current_equity(date, {sym: df})
            if self.risk_manager:
                target_value = self.risk_manager.calc_position_size(
                    symbol=sym, price=fill_price, equity=equity,
                    cash=self.cash, confidence=1.0, positions=self.positions,
                )
            else:
                target_value = equity * 0.2
            shares = self._round_lot(sym, target_value / fill_price)
            if shares <= 0:
                return False

            amount = fill_price * shares
            commission, stamp_tax, other_fees = self._calc_trade_fees(sym, amount, shares, "buy")
            slippage_cost = (fill_price - ref_price) * shares
            total_cost = amount + commission + other_fees
            if total_cost > self.cash:
                shares = self._round_lot(sym, self.cash * 0.99 / fill_price)
                if shares <= 0:
                    return False
                amount = fill_price * shares
                commission, stamp_tax, other_fees = self._calc_trade_fees(sym, amount, shares, "buy")
                slippage_cost = (fill_price - ref_price) * shares
                total_cost = amount + commission + other_fees

            self.cash -= total_cost
            pos = self.positions.get(sym, Position(symbol=sym))
            pos.shares += shares
            pos.avg_cost = fill_price
            pos.entry_date = date
            self.positions[sym] = pos

            reason = getattr(order, "reason", "") or "挂单买入"
            self.trades.append(Trade(
                date=date, symbol=sym, action="buy",
                price=fill_price, shares=shares, amount=amount,
                commission=commission + other_fees, stamp_tax=0.0,
                slippage_cost=slippage_cost, reason=reason,
                order_type=order.order_type, entry_date=date,
            ))
            self._orders_filled += 1
            return True

        # ---- 卖出（平仓） ----
        pos = self.positions.get(sym)
        if pos is None or pos.shares <= 0:
            return False

        fill_price = ref_price * (1 - self.slippage_rate)
        shares = order.quantity if order.quantity > 0 else pos.shares
        shares = min(shares, pos.shares)

        amount = fill_price * shares
        commission, stamp_tax, other_fees = self._calc_trade_fees(sym, amount, shares, "sell")
        slippage_cost = (ref_price - fill_price) * shares
        net_proceeds = amount - commission - stamp_tax - other_fees
        cost_basis = pos.avg_cost * shares
        pnl = net_proceeds - cost_basis
        entry_date = pos.entry_date

        self.cash += net_proceeds
        pos.shares -= shares
        if pos.shares == 0:
            pos.avg_cost = 0.0
            pos.entry_date = None

        reason = getattr(order, "reason", "") or "挂单卖出"
        self.trades.append(Trade(
            date=date, symbol=sym, action="sell",
            price=fill_price, shares=shares, amount=amount,
            commission=commission + other_fees, stamp_tax=stamp_tax,
            slippage_cost=slippage_cost, pnl=pnl, reason=reason,
            order_type=order.order_type, entry_date=entry_date,
        ))
        self._orders_filled += 1

        # 平仓后取消该标的剩余挂单（防止重复平仓）
        self.pending_orders = [
            o for o in self.pending_orders
            if not (o.symbol == sym and o.action == "sell")
        ]

        # 风控记录（平仓盈亏）
        if self.risk_manager:
            self.risk_manager.record_trade(pnl)
        return True

    # ------------------------------------------------------------------
    # 风控检查
    # ------------------------------------------------------------------

    def _check_risk_controls(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> None:
        """检查止损/止盈等风控规则。"""
        if self.risk_manager is None:
            return

        for sym, pos in list(self.positions.items()):
            if pos.shares <= 0 or sym not in data or date not in data[sym].index:
                continue

            open_price = float(data[sym].loc[date, "open"])
            pnl_pct = (open_price - pos.avg_cost) / pos.avg_cost

            # 止损
            if pnl_pct <= -self.risk_manager.single_stop_loss:
                n_before = len(self.trades)
                self._execute_sell(date, sym, data[sym], f"止损({pnl_pct*100:.1f}%)")
                self._record_risk_sell(sym, n_before, reason=f"止损({pnl_pct*100:.1f}%)")
                continue

            # 止盈评估
            if pnl_pct >= self.risk_manager.single_take_profit:
                # 简单止盈：达到目标直接平仓
                n_before = len(self.trades)
                self._execute_sell(date, sym, data[sym], f"止盈({pnl_pct*100:.1f}%)")
                self._record_risk_sell(sym, n_before, reason=f"止盈({pnl_pct*100:.1f}%)")
                continue

        # 总回撤检查
        equity = self._current_equity(date, data)
        self.risk_manager.update_equity(equity)
        if self.risk_manager.is_paused():
            logger.warning("账户回撤超限，暂停开新仓")

    # ------------------------------------------------------------------
    # 走查（walkthrough）记录钩子
    # ------------------------------------------------------------------

    @staticmethod
    def _jev_decision_to_dict(decision: Any) -> Dict[str, Any]:
        """将 JevDecision 序列化为可 JSON 化的字典。"""
        return {
            "probabilities": decision.probabilities,
            "final_action": decision.final_action,
            "final_confidence": float(decision.final_confidence),
            "executed": bool(decision.executed),
            "reason": decision.reason,
        }

    def _record_risk_sell(
        self, sym: str, n_trades_before: int, reason: str
    ) -> None:
        """把风控触发的止损/止盈卖出追加到当日信号明细。"""
        if self._walkthrough_snapshots is None:
            return
        rec: Dict[str, Any] = {
            "symbol": sym,
            "signal": "sell",
            "confidence": 0.0,
            "jev_filtered": False,
            "jev_decision": None,
            "risk_blocked": False,   # 这是风控主动平仓，不是拦截新开仓
            "executed": False,
            "action": "sell",
            "fill_price": None,
            "shares": 0,
            "reason": reason,
        }
        if len(self.trades) > n_trades_before:
            t = self.trades[-1]
            rec["executed"] = True
            rec["fill_price"] = float(t.price)
            rec["shares"] = int(t.shares)
        self._current_day_signals.append(rec)

    def _record_walkthrough_day(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
        equity: float,
    ) -> None:
        """追加日末走查快照：收盘价 / 信号明细 / 持仓 / 现金 / 权益 / 盈亏。"""
        # 当日各标的收盘价
        close_map: Dict[str, float] = {}
        for sym, df in data.items():
            if date in df.index:
                close_map[sym] = float(df.loc[date, "close"])

        # 日末持仓（当日交易已完成，反映收盘后状态）
        positions_snap: Dict[str, Dict[str, Any]] = {}
        for sym, pos in self.positions.items():
            if pos.shares > 0:
                positions_snap[sym] = {
                    "shares": int(pos.shares),
                    "avg_cost": float(pos.avg_cost),
                }

        prev_equity = (
            self.equity_history[-2][1]
            if len(self.equity_history) >= 2
            else self.initial_capital
        )

        snap = {
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "close": close_map,
            "signals": list(self._current_day_signals),
            "positions": positions_snap,
            "cash": float(self.cash),
            "total_equity": float(equity),
            "daily_pnl": float(equity - prev_equity),
            "cumulative_pnl": float(equity - self.initial_capital),
        }
        self._walkthrough_snapshots.append(snap)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _round_lot(self, symbol: str, raw_shares: float) -> int:
        """按标的所在市场的每手股数向下取整。

        A股/港股 100 股一手，美股 1 股起。未识别标的默认 100 股一手。
        """
        lot = lot_size_of(symbol)
        return int(raw_shares // lot) * lot

    def _calc_trade_fees(
        self, symbol: str, amount: float, shares: int, action: str
    ) -> Tuple[float, float, float]:
        """按标的所在市场计算交易费用。

        A股沿用引擎级 ``commission_rate`` / ``stamp_tax_rate``（保持向后兼容）；
        美股/港股走 :mod:`trading.market_config` 的多费率结构。

        Returns:
            (commission, stamp_tax, other_fees)。
            other_fees 包含美股 SEC 费与港股交易费。
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

    def _current_equity(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> float:
        """计算当前总权益 = 现金 + 持仓市值。"""
        equity = self.cash
        for sym, pos in self.positions.items():
            if pos.shares > 0 and sym in data and date in data[sym].index:
                price = float(data[sym].loc[date, "close"])
                equity += pos.shares * price
            elif pos.shares > 0:
                equity += pos.shares * pos.avg_cost
        return equity

    def _build_benchmark(
        self, data: Dict[str, pd.DataFrame], dates: List[pd.Timestamp]
    ) -> pd.Series:
        """构建基准曲线（等权买入持有）。"""
        if not data:
            return pd.Series(dtype=float)
        # 取第一个标的的收盘价作为基准
        first_sym = list(data.keys())[0]
        df = data[first_sym]
        benchmark = df["close"].reindex(dates).ffill()
        benchmark = benchmark / benchmark.iloc[0] * self.initial_capital
        benchmark.name = "benchmark"
        return benchmark

    def _reset(self) -> None:
        """重置回测状态。"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_history = []
        self.positions_history = []
        self.pending_orders = []
        self._orders_submitted = 0
        self._orders_filled = 0
        if self.risk_manager:
            self.risk_manager.reset(self.initial_capital)

    # ------------------------------------------------------------------
    # 参数扫描
    # ------------------------------------------------------------------

    @staticmethod
    def parameter_sweep(
        data: pd.DataFrame,
        strategy_class: type,
        param_grid: Dict[str, list],
        symbol: str = "",
        **engine_kwargs: Any,
    ) -> pd.DataFrame:
        """参数扫描：遍历参数组合，返回各组合的绩效指标。

        Args:
            data: 行情数据。
            strategy_class: 策略类。
            param_grid: 参数网格，如 {"fast_period": [3,5,10], "slow_period": [20,30]}。
            symbol: 标的代码。
            **engine_kwargs: 回测引擎参数。

        Returns:
            参数组合与绩效指标的 DataFrame。
        """
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        results = []

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            strategy = strategy_class(params)
            engine = BacktestEngine(**engine_kwargs)
            result = engine.run(data, strategy, symbol=symbol)
            row = dict(params)
            row.update(result.metrics)
            results.append(row)

        return pd.DataFrame(results)
