#!/usr/bin/env python3
"""实时模拟交易引擎（多标的并行扫描版）。

每 interval 秒遍历股票池内所有启用标的，对每只独立执行完整交易流程：
  1. 策略生成信号（buy/sell/hold + 置信度）
  2. 构建市场状态特征（7个特征）
  3. 调用本地 Jev 获取概率分布（失败降级 mock）
  4. 风控检查（止损/止盈/仓位/日亏/Jev 置信度/持仓标的数上限）
  5. 模拟下单（通过风控则执行，含滑点/佣金/印花税）
  6. 更新持仓和账户

全流程日志记录每一步的详细信息，供前端实时展示。
单只标的处理异常不会影响同轮其它标的。
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("realtime_trader")


class RealtimeTrader:
    """实时模拟交易引擎（多标的并行扫描）。

    每 interval 秒对股票池内所有启用标的执行完整交易流程：
      1. 策略生成信号（buy/sell/hold + 置信度）
      2. 构建市场状态特征（7个特征）
      3. 调用本地Jev获取概率分布
      4. 风控检查（止损/止盈/仓位/日亏/Jev置信度/持仓数上限）
      5. 模拟下单（通过风控则执行）
      6. 更新持仓和账户

    全流程日志记录每一步的详细信息。
    """

    #: 日志最大保留条数
    MAX_LOGS: int = 500

    def __init__(
        self,
        manager: Any,
        accounts: Dict[str, Any],
        jev_client: Any,
        risk_config: Dict[str, Any],
        get_strategy: Callable[[], str],
        get_account_id: Callable[[], str],
        strategy_signal_fn: Callable[[Any], tuple],
        market_state_fn: Callable[[Any], Optional[Dict[str, Any]]],
        interval: float = 5.0,
        jev_threshold: float = 0.6,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.0005,
        slippage_rate: float = 0.001,
        db: Any = None,
        alert_manager: Any = None,
        realtime_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """初始化实时交易引擎。

        Args:
            manager: MultiSymbolManager 实例，提供当前标的与行情模拟器。
            accounts: {account_id: SimAccount} 字典。
            jev_client: JevRealClient 实例，同步调用 .predict(state_dict)。
            risk_config: 风控参数字典（来自 config.yaml risk 节）。
            get_strategy: 返回当前策略名的 callable。
            get_account_id: 返回当前账户 ID 的 callable。
            strategy_signal_fn: 输入 sim，返回 (signal, confidence) 的 callable。
            market_state_fn: 输入 sim，返回市场状态 dict 或 None 的 callable。
            interval: 交易循环间隔（秒）。
            jev_threshold: Jev 置信度阈值。
            commission_rate: 佣金费率（双边）。
            stamp_tax_rate: 印花税率（仅卖出）。
            slippage_rate: 滑点比例。
            db: Database 持久化实例（可选），传入后自动写入交易/Jev/快照。
            alert_manager: AlertManager 告警管理器（可选），风控触发时推送告警。
            realtime_config: config.yaml realtime_trading 节，支持
                enabled_symbols / max_positions / scan_all 三个字段。
        """
        self.manager = manager
        self.accounts = accounts
        self.jev_client = jev_client
        self.risk_config = risk_config
        self.get_strategy = get_strategy
        self.get_account_id = get_account_id
        self.strategy_signal_fn = strategy_signal_fn
        self.market_state_fn = market_state_fn
        self.interval = interval
        self.jev_threshold = jev_threshold
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage_rate = slippage_rate
        self.db = db
        self.alert_manager = alert_manager

        # ---- 多标的扫描配置 ----
        rt_cfg: Dict[str, Any] = dict(realtime_config or {})
        self.enabled_symbols: List[str] = list(rt_cfg.get("enabled_symbols") or [])
        self.max_positions: int = int(rt_cfg.get("max_positions", 5))
        self.scan_all: bool = bool(rt_cfg.get("scan_all", True))
        #: 每只标的最近一次决策状态缓存，供前端展示
        self.symbol_status: Dict[str, Dict[str, Any]] = {}

        self.running: bool = False
        self.trade_log: List[Dict[str, Any]] = []
        self.risk_managers: Dict[str, Any] = {}
        self.today_pnl: float = 0.0
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # 启动 / 停止
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动实时交易循环（在事件循环中调用）。"""
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("实时交易引擎已启动，间隔 %.1f 秒", self.interval)

    def stop(self) -> None:
        """停止实时交易循环。"""
        self.running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None
        logger.info("实时交易引擎已停止")

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """后台交易循环：每 interval 秒执行一次完整交易流程。"""
        while self.running:
            try:
                await self._run_one_cycle()
            except Exception:
                logger.exception("交易循环异常")
            await asyncio.sleep(self.interval)

    # ------------------------------------------------------------------
    # 扫描标的选择
    # ------------------------------------------------------------------

    def _get_scan_symbols(self) -> List[str]:
        """返回本轮需要扫描的标的列表。

        优先级：
          1. enabled_symbols 非空 → 取其中在 manager.sims 中存在的标的；
          2. scan_all=True → 返回 manager.sims 的全部标的；
          3. 否则 → 仅返回当前选中标的 manager.current_symbol。

        Returns:
            待扫描标的代码列表。
        """
        sims: Dict[str, Any] = getattr(self.manager, "sims", {}) or {}

        if self.enabled_symbols:
            return [sym for sym in self.enabled_symbols if sym in sims]

        if self.scan_all:
            return list(sims.keys())

        return [self.manager.current_symbol]

    # ------------------------------------------------------------------
    # 核心：单轮交易流程（多标的）
    # ------------------------------------------------------------------

    async def _run_one_cycle(self) -> None:
        """执行一轮完整的多标的扫描：策略→Jev→风控→下单。

        每轮开始时刷新一次账户权益；逐只处理标的，单只异常不影响其它；
        全部处理完后写入一次账户快照。
        """
        account_id = self.get_account_id()
        acc = self.accounts[account_id]

        # 获取或创建该账户的风控管理器（每轮复用）
        rm = self._get_risk_manager(account_id, acc.initial_capital)
        rm.set_date(pd.Timestamp.now().normalize())

        # 每轮开始时更新一次账户权益
        start_equity = acc.cash + self._calc_market_value(acc)
        rm.update_equity(start_equity)

        scan_symbols = self._get_scan_symbols()
        cycle_ts = pd.Timestamp.now().isoformat()

        for symbol in scan_symbols:
            try:
                await self._process_symbol(symbol, acc, account_id, rm, cycle_ts)
            except Exception:
                logger.exception("标的 %s 本轮处理异常，跳过", symbol)

        # 所有标的处理完后，写入一次账户快照
        if self.db is not None:
            try:
                self._persist_account_snapshot(acc, account_id, cycle_ts)
            except Exception:
                logger.exception("账户快照写入失败（不影响交易）")

    async def _process_symbol(
        self,
        symbol: str,
        acc: Any,
        account_id: str,
        rm: Any,
        cycle_ts: str,
    ) -> None:
        """对单只标的执行完整交易流程：信号→Jev→风控→下单→记录。

        Args:
            symbol: 标的代码。
            acc: SimAccount 实例（同账户共享，本方法内会修改现金/持仓）。
            account_id: 账户 ID。
            rm: 该账户的 RiskManager 实例。
            cycle_ts: 本轮时间戳（ISO 格式），用于日志与持久化对齐。
        """
        sim = self.manager.get(symbol)
        price = sim.current_price
        name = getattr(sim, "name", symbol)

        # ---- 步骤2: 策略信号 ----
        strategy_signal, strategy_conf = self.strategy_signal_fn(sim)

        # ---- 步骤3: 构建市场状态 ----
        market_state = self.market_state_fn(sim)
        if market_state is None:
            logger.debug("市场状态数据不足，跳过本轮: %s", symbol)
            return

        # ---- 步骤4: 调用 Jev（真实服务优先，失败降级 mock）----
        mode = "mock"
        latency_ms = 0.0
        probabilities: Dict[str, float] = {"buy": 0.0, "sell": 0.0, "hold": 1.0}

        try:
            real = await asyncio.to_thread(self.jev_client.predict, market_state)
            probabilities = real["probabilities"]
            latency_ms = real.get("latency_ms", 0.0)
            mode = "real"
        except Exception as e:
            logger.info("Jev real 推理失败，降级 mock: %s", e)
            from jev.jev_engine import JevDecisionEngine
            mock_engine = JevDecisionEngine(mock_mode=True)
            states_list = [
                {"feature": k, "value": v} for k, v in market_state.items()
            ]
            probabilities = mock_engine._mock_evaluate(
                states_list, strategy_signal, strategy_conf
            )
            latency_ms = 0.0
            mode = "mock"

        jev_action = max(probabilities, key=probabilities.get)
        jev_confidence = float(probabilities[jev_action])

        # ---- 步骤5: 风控检查 ----
        # 重新计算当前权益（前面标的可能已成交改变了账户）
        current_equity = acc.cash + self._calc_market_value(acc)
        # 持续更新峰值和回撤
        rm.update_equity(current_equity)

        # 止损/止盈检查（优先于 Jev 决策）
        stop_reason = self._check_stop_loss_take_profit(acc, symbol, price)

        final_action: str
        final_confidence: float
        executed: bool = False
        reason: str = ""
        quantity: int = 0
        order_details: Optional[Dict[str, Any]] = None
        realized_pnl: float = 0.0

        if stop_reason is not None:
            # 触发止损/止盈 → 强制卖出
            final_action = "sell"
            final_confidence = 1.0
            reason = stop_reason
            # 告警推送
            if self.alert_manager:
                pos = acc.positions.get(symbol)
                if pos and pos.get("avg_cost"):
                    pnl_pct = (price - pos["avg_cost"]) / pos["avg_cost"]
                    if "stop_loss" in stop_reason:
                        self.alert_manager.stop_loss(
                            symbol, pnl_pct,
                            threshold=float(self.risk_config.get("single_stop_loss", 0.03)),
                            account_id=account_id,
                        )
                    elif "take_profit" in stop_reason:
                        self.alert_manager.take_profit(
                            symbol, pnl_pct,
                            threshold=float(self.risk_config.get("single_take_profit", 0.08)),
                            account_id=account_id,
                        )
            # 检查是否有持仓可卖
            pos = acc.positions.get(symbol)
            if pos is None or pos["shares"] < 100:
                executed = False
                reason = f"{stop_reason}（无足够持仓可卖）"
            else:
                executed = True
                quantity = int(pos["shares"] // 100) * 100
        else:
            # 使用 Jev 决策结果
            final_action = jev_action
            final_confidence = jev_confidence

            # Jev 决策过滤（与 /api/jev_decision 一致）
            if final_action == "hold":
                executed = False
                reason = "Jev建议观望"
            elif final_action != strategy_signal and strategy_signal != "hold":
                executed = False
                reason = f"Jev方向({final_action})与策略方向({strategy_signal})冲突"
            elif final_confidence < self.jev_threshold:
                executed = False
                reason = f"置信度{final_confidence:.3f}低于阈值{self.jev_threshold}"
                if self.alert_manager:
                    self.alert_manager.jev_filtered(
                        symbol, final_action, final_confidence,
                        threshold=self.jev_threshold, account_id=account_id,
                    )
            else:
                executed = True
                reason = "通过Jev过滤"

            # 买入：持仓标的数上限检查
            if executed and final_action == "buy":
                if symbol not in acc.positions and len(acc.positions) >= self.max_positions:
                    executed = False
                    reason = f"持仓标的数已达上限{self.max_positions}"

            # 买入：风控仓位检查 + 计算仓位
            if executed and final_action == "buy":
                positions_obj = {
                    sym: SimpleNamespace(**pos) for sym, pos in acc.positions.items()
                }
                if not rm.check_trade_allowed(symbol, "buy", price, current_equity,
                                              acc.cash, positions_obj):
                    executed = False
                    reason = "风控检查未通过（仓位/回撤/日亏限制）"
                else:
                    target_value = rm.calc_position_size(
                        symbol, price, current_equity, acc.cash,
                        final_confidence, positions_obj,
                    )
                    quantity = int(target_value / price / 100) * 100
                    if quantity < 100:
                        executed = False
                        reason = "计算仓位不足1手"

            # 卖出：检查持仓
            if executed and final_action == "sell":
                pos = acc.positions.get(symbol)
                if pos is None or pos["shares"] < 100:
                    executed = False
                    reason = "无持仓可卖"
                else:
                    quantity = int(pos["shares"] // 100) * 100

        # ---- 步骤6: 模拟下单 ----
        if executed and quantity > 0:
            result = self._execute_order(
                acc, symbol, final_action, price, quantity, rm,
            )
            executed = result["executed"]
            reason = result.get("reason", reason)
            order_details = result.get("order_details")
            realized_pnl = result.get("realized_pnl", 0.0)
            quantity = result.get("quantity", quantity)

        # ---- 记录日志 ----
        market_value_after = self._calc_market_value(acc)
        total_asset_after = acc.cash + market_value_after

        log_entry: Dict[str, Any] = {
            "timestamp": cycle_ts,
            "symbol": symbol,
            "name": name,
            "account_id": account_id,
            "strategy": self.get_strategy(),
            "strategy_signal": strategy_signal,
            "strategy_confidence": round(strategy_conf, 4),
            "market_state": market_state,
            "jev_probabilities": {k: round(float(v), 4) for k, v in probabilities.items()},
            "jev_decision": final_action,
            "jev_confidence": round(final_confidence, 4),
            "jev_mode": mode,
            "jev_latency_ms": round(float(latency_ms), 1),
            "risk_passed": executed,
            "risk_reason": reason,
            "order_executed": executed and quantity > 0,
            "order_details": order_details,
            "realized_pnl": round(realized_pnl, 2) if realized_pnl else 0.0,
            "account_after": {
                "cash": round(acc.cash, 2),
                "total_asset": round(total_asset_after, 2),
            },
        }
        self.trade_log.append(log_entry)
        if len(self.trade_log) > self.MAX_LOGS:
            self.trade_log.pop(0)

        # 更新每只标的最近一次决策状态缓存
        self.symbol_status[symbol] = {
            "symbol": symbol,
            "name": name,
            "action": final_action,
            "confidence": round(final_confidence, 4),
            "reason": reason,
            "executed": bool(log_entry["order_executed"]),
            "timestamp": cycle_ts,
        }

        # ---- 持久化该标的的 Jev 决策与成交记录（不写账户快照）----
        if self.db is not None:
            try:
                self._persist_symbol_result(
                    log_entry, order_details, realized_pnl,
                    acc, account_id, symbol, name, price,
                )
            except Exception:
                logger.exception("标的 %s 持久化写入失败（不影响交易）", symbol)

    # ------------------------------------------------------------------
    # 持久化（拆分为：单标的结果 + 账户快照）
    # ------------------------------------------------------------------

    def _persist_symbol_result(
        self,
        log_entry: Dict[str, Any],
        order_details: Optional[Dict[str, Any]],
        realized_pnl: float,
        acc: Any,
        account_id: str,
        symbol: str,
        name: str,
        price: float,
    ) -> None:
        """将单只标的本轮的 Jev 决策与成交记录写入 SQLite。

        不写账户快照（账户快照由 _persist_account_snapshot 每轮写一次）。
        任何写入失败仅记日志，不影响交易流程。

        Args:
            log_entry: 本轮单标的日志条目。
            order_details: 成交明细，未成交为 None。
            realized_pnl: 已实现盈亏。
            acc: SimAccount 实例。
            account_id: 账户 ID。
            symbol: 标的代码。
            name: 标的名称。
            price: 当前价格。
        """
        ts = log_entry["timestamp"]
        # 1. Jev 决策审计（每只标的每轮都写）
        self.db.insert_jev_decision(
            timestamp=ts,
            symbol=symbol,
            strategy_signal=log_entry.get("strategy_signal", ""),
            strategy_confidence=log_entry.get("strategy_confidence", 0),
            market_state=log_entry.get("market_state", {}),
            probabilities=log_entry.get("jev_probabilities", {}),
            final_action=log_entry.get("jev_decision", "hold"),
            final_confidence=log_entry.get("jev_confidence", 0),
            executed=bool(log_entry.get("order_executed", False)),
            reason=log_entry.get("risk_reason", ""),
            mode=log_entry.get("jev_mode", "mock"),
            latency_ms=log_entry.get("jev_latency_ms", 0),
        )

        # 2. 成交记录（仅实际下单时写）
        if order_details is not None:
            self.db.insert_trade(
                timestamp=ts,
                symbol=symbol,
                name=name,
                side=order_details.get("action", log_entry.get("jev_decision", "")),
                price=price,
                fill_price=order_details.get("fill_price", price),
                quantity=order_details.get("quantity", 0),
                amount=order_details.get("amount", 0),
                commission=order_details.get("commission", 0),
                stamp_tax=order_details.get("stamp_tax", 0),
                realized_pnl=realized_pnl,
                reason=log_entry.get("risk_reason", ""),
                account_id=account_id,
                strategy=log_entry.get("strategy", ""),
                jev_confidence=log_entry.get("jev_confidence", 0),
            )

    def _persist_account_snapshot(
        self,
        acc: Any,
        account_id: str,
        timestamp: str,
    ) -> None:
        """写入一次账户快照（每轮仅调用一次）。

        Args:
            acc: SimAccount 实例。
            account_id: 账户 ID。
            timestamp: 本轮时间戳（ISO 格式）。
        """
        positions_list = []
        for sym, pos in acc.positions.items():
            sim = self.manager.sims.get(sym)
            cur_price = sim.current_price if sim else pos.get("avg_cost", 0)
            positions_list.append({
                "symbol": sym,
                "shares": pos["shares"],
                "avg_cost": pos["avg_cost"],
                "market_value": round(pos["shares"] * cur_price, 2),
            })
        market_value = sum(p["market_value"] for p in positions_list)
        self.db.insert_account_snapshot(
            timestamp=timestamp,
            account_id=account_id,
            total_asset=round(acc.cash + market_value, 2),
            cash=round(acc.cash, 2),
            position_value=round(market_value, 2),
            daily_pnl=round(self.today_pnl, 2),
            positions=positions_list,
        )

    def _persist_cycle(
        self,
        log_entry: Dict[str, Any],
        order_details: Optional[Dict[str, Any]],
        realized_pnl: float,
        acc: Any,
        account_id: str,
        symbol: str,
        name: str,
        price: float,
    ) -> None:
        """兼容旧接口：单轮持久化（单标的结果 + 账户快照）。

        新代码应分别调用 _persist_symbol_result 与 _persist_account_snapshot。
        保留此方法仅为向后兼容。

        Args:
            log_entry: 本轮日志条目。
            order_details: 成交明细。
            realized_pnl: 已实现盈亏。
            acc: SimAccount 实例。
            account_id: 账户 ID。
            symbol: 标的代码。
            name: 标的名称。
            price: 当前价格。
        """
        try:
            self._persist_symbol_result(
                log_entry, order_details, realized_pnl,
                acc, account_id, symbol, name, price,
            )
            self._persist_account_snapshot(acc, account_id, log_entry["timestamp"])
        except Exception:
            logger.exception("持久化写入失败（不影响交易）")

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _get_risk_manager(self, account_id: str, initial_capital: float) -> Any:
        """懒加载每个账户的 RiskManager 实例。

        Args:
            account_id: 账户 ID。
            initial_capital: 账户初始资金。

        Returns:
            对应账户的 RiskManager 实例。
        """
        if account_id not in self.risk_managers:
            from risk.risk_manager import RiskManager
            cfg = self.risk_config
            self.risk_managers[account_id] = RiskManager(
                single_stop_loss=float(cfg.get("single_stop_loss", 0.03)),
                single_take_profit=float(cfg.get("single_take_profit", 0.08)),
                max_drawdown_pause=float(cfg.get("max_drawdown_pause", 0.10)),
                max_position_per_symbol=float(cfg.get("max_position_per_symbol", 0.20)),
                max_total_position=float(cfg.get("max_total_position", 0.80)),
                daily_loss_limit=float(cfg.get("daily_loss_limit", 0.02)),
                jev_confidence_threshold=float(
                    cfg.get("jev_confidence_threshold", self.jev_threshold)
                ),
                initial_capital=initial_capital,
            )
        return self.risk_managers[account_id]

    def _calc_market_value(self, acc: Any) -> float:
        """计算账户当前持仓市值。

        Args:
            acc: SimAccount 实例。

        Returns:
            持仓总市值。
        """
        mv = 0.0
        for sym, pos in acc.positions.items():
            sim = self.manager.sims.get(sym)
            if sim is not None:
                mv += pos["shares"] * sim.current_price
        return mv

    def _check_stop_loss_take_profit(
        self,
        acc: Any,
        symbol: str,
        price: float,
    ) -> Optional[str]:
        """检查当前标的持仓是否触发止损/止盈。

        Args:
            acc: SimAccount 实例。
            symbol: 当前标的代码。
            price: 当前价格。

        Returns:
            触发原因字符串（"stop_loss" / "take_profit"），未触发返回 None。
        """
        pos = acc.positions.get(symbol)
        if pos is None or pos["shares"] <= 0:
            return None

        avg_cost = pos["avg_cost"]
        if avg_cost <= 0:
            return None

        pnl_pct = (price - avg_cost) / avg_cost
        stop_loss_pct = float(self.risk_config.get("single_stop_loss", 0.03))
        take_profit_pct = float(self.risk_config.get("single_take_profit", 0.08))

        if pnl_pct <= -stop_loss_pct:
            return f"stop_loss({pnl_pct*100:.1f}%)"
        if pnl_pct >= take_profit_pct:
            return f"take_profit({pnl_pct*100:.1f}%)"
        return None

    def _execute_order(
        self,
        acc: Any,
        symbol: str,
        action: str,
        price: float,
        quantity: int,
        risk_manager: Any,
    ) -> Dict[str, Any]:
        """执行模拟下单，更新账户现金与持仓。

        Args:
            acc: SimAccount 实例。
            symbol: 标的代码。
            action: "buy" 或 "sell"。
            price: 当前市价。
            quantity: 委托数量（股）。
            risk_manager: 对应账户的 RiskManager。

        Returns:
            {"executed": bool, "reason": str, "order_details": dict|None,
             "realized_pnl": float, "quantity": int}
        """
        # 滑点：买入成交价更高，卖出更低
        if action == "buy":
            fill_price = price * (1 + self.slippage_rate)
        else:
            fill_price = price * (1 - self.slippage_rate)

        amount = fill_price * quantity
        commission = max(amount * self.commission_rate, 5.0)

        if action == "buy":
            total_cost = amount + commission
            if total_cost > acc.cash:
                return {
                    "executed": False,
                    "reason": "资金不足",
                    "order_details": None,
                    "realized_pnl": 0.0,
                    "quantity": 0,
                }
            # 扣款
            acc.cash -= total_cost
            # 更新持仓（加仓：新平均成本）
            existing = acc.positions.get(symbol, {"shares": 0.0, "avg_cost": 0.0})
            old_shares = existing["shares"]
            new_shares = old_shares + quantity
            existing["avg_cost"] = (
                (existing["avg_cost"] * old_shares + fill_price * quantity) / new_shares
            )
            existing["shares"] = new_shares
            acc.positions[symbol] = existing

            order_details = {
                "action": "buy",
                "fill_price": round(fill_price, 4),
                "quantity": quantity,
                "commission": round(commission, 2),
                "amount": round(amount, 2),
            }
            logger.info("模拟买入 %s %d股 @ %.2f", symbol, quantity, fill_price)
            return {
                "executed": True,
                "reason": "买入成交",
                "order_details": order_details,
                "realized_pnl": 0.0,
                "quantity": quantity,
            }

        else:  # sell
            stamp_tax = amount * self.stamp_tax_rate
            net = amount - commission - stamp_tax
            existing = acc.positions.get(symbol)
            if existing is None or existing["shares"] < quantity:
                return {
                    "executed": False,
                    "reason": "持仓不足",
                    "order_details": None,
                    "realized_pnl": 0.0,
                    "quantity": 0,
                }
            avg_cost = existing["avg_cost"]
            realized_pnl = (fill_price - avg_cost) * quantity - commission - stamp_tax
            # 回款
            acc.cash += net
            # 更新持仓
            existing["shares"] -= quantity
            if existing["shares"] <= 0:
                del acc.positions[symbol]
            # 记录风控盈亏
            risk_manager.record_trade(realized_pnl)
            self.today_pnl += realized_pnl

            order_details = {
                "action": "sell",
                "fill_price": round(fill_price, 4),
                "quantity": quantity,
                "commission": round(commission, 2),
                "amount": round(amount, 2),
                "stamp_tax": round(stamp_tax, 2),
            }
            logger.info(
                "模拟卖出 %s %d股 @ %.2f 已实现盈亏 %.2f",
                symbol, quantity, fill_price, realized_pnl,
            )
            return {
                "executed": True,
                "reason": "卖出成交",
                "order_details": order_details,
                "realized_pnl": realized_pnl,
                "quantity": quantity,
            }

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """返回当前实时交易状态摘要。

        Returns:
            包含运行状态、扫描标的列表、当前账户、今日盈亏、日志条数、
            持仓快照、风控摘要、每只标的最近一次决策。
        """
        account_id = self.get_account_id()
        acc = self.accounts.get(account_id)
        positions_snapshot: List[Dict[str, Any]] = []
        risk_summary: Dict[str, Any] = {}
        if acc is not None:
            positions_snapshot = acc.snapshot().get("positions", [])
            rm = self.risk_managers.get(account_id)
            if rm is not None:
                risk_summary = rm.get_summary()

        scan_symbols = self._get_scan_symbols()

        return {
            "running": self.running,
            "interval": self.interval,
            "current_symbol": self.manager.current_symbol,
            "current_account_id": account_id,
            "today_pnl": round(self.today_pnl, 2),
            "log_count": len(self.trade_log),
            "positions": positions_snapshot,
            "risk_summary": risk_summary,
            # ---- 多标的扫描新增字段 ----
            "scan_symbols": scan_symbols,
            "symbol_count": len(scan_symbols),
            "max_positions": self.max_positions,
            "scan_all": self.scan_all,
            "symbol_decisions": dict(self.symbol_status),
        }

    def get_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """返回最近 limit 条交易日志（倒序，最新在前）。

        Args:
            limit: 返回条数上限。

        Returns:
            日志列表，最新的在前。
        """
        return list(reversed(self.trade_log[-limit:]))
