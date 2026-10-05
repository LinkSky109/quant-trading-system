"""智能组合再平衡模块。

实现漂移阈值触发、交易成本优化、税务 aware 的再平衡路径规划，与
:mod:`optimization.portfolio_optimizer` 无缝集成。

核心能力：
1. **漂移检测**：监控实际权重与目标权重的偏离，超过阈值时触发再平衡。
2. **交易成本模型**：包含固定佣金 + 滑点 + 冲击成本（与成交量/市值相关）。
3. **税务 aware**：实现亏损收割（tax-loss harvesting）与短期资本利得规避。
4. **路径优化**：在再平衡时选择最小总成本的交易顺序与金额。

典型用法::

    from optimization.portfolio_optimizer import PortfolioOptimizer
    from optimization.smart_rebalance import SmartRebalancer

    opt = PortfolioOptimizer()
    target = opt.optimize(symbols, data, method="risk_parity")

    rebalancer = SmartRebalancer(drift_threshold=0.05)
    result = rebalancer.rebalance(
        current_weights=current,
        target_weights=target.weights,
        prices=prices,
        positions=positions,
        volumes=volumes,
        tax_lots=tax_lots,
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from optimization.portfolio_optimizer import OptimizeResult, PortfolioOptimizer

logger = logging.getLogger(__name__)

# 中国 A 股默认交易成本参数（可覆盖）
DEFAULT_COMMISSION_RATE = 0.00025  # 佣金率
DEFAULT_STAMP_TAX_RATE = 0.0005  # 印花税（卖出）
DEFAULT_SLIPPAGE_RATE = 0.001  # 滑点
DEFAULT_SHORT_TERM_DAYS = 365  # 短期持有天数阈值
DEFAULT_SHORT_TERM_TAX_RATE = 0.20  # 短期资本利得税率（示意）
DEFAULT_LONG_TERM_TAX_RATE = 0.10  # 长期资本利得税率（示意）


@dataclass
class RebalanceResult:
    """再平衡结果。

    Attributes:
        trades: 交易清单 ``[{symbol, action, shares, estimated_cost, reason}]``。
        total_cost: 预估总交易成本。
        total_tax: 预估总税费。
        drift_before: 再平衡前最大漂移。
        drift_after: 执行交易后的预期漂移。
        triggered: 是否因漂移超限而触发。
        tax_harvested: 亏损收割产生的税务抵扣金额。
    """

    trades: List[Dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0
    total_tax: float = 0.0
    drift_before: float = 0.0
    drift_after: float = 0.0
    triggered: bool = False
    tax_harvested: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        """序列化为普通 dict。"""
        return {
            "trades": list(self.trades),
            "total_cost": float(self.total_cost),
            "total_tax": float(self.total_tax),
            "drift_before": float(self.drift_before),
            "drift_after": float(self.drift_after),
            "triggered": bool(self.triggered),
            "tax_harvested": float(self.tax_harvested),
        }


@dataclass
class TaxLot:
    """税务批次记录（用于税务 aware 再平衡）。"""

    symbol: str
    shares: float
    avg_cost: float
    entry_date: pd.Timestamp

    def is_short_term(self, as_of: Optional[pd.Timestamp] = None) -> bool:
        """是否短期持有（默认 1 年内）。"""
        if as_of is None:
            as_of = pd.Timestamp.now()
        return (as_of - self.entry_date).days < DEFAULT_SHORT_TERM_DAYS

    def unrealized_pnl(self, price: float) -> float:
        """未实现盈亏。"""
        return (price - self.avg_cost) * self.shares


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """权重归一化和为 1。"""
    total = sum(weights.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in weights.items()}


class SmartRebalancer:
    """智能组合再平衡器。

    Args:
        drift_threshold: 漂移触发阈值（如 0.05 表示权重偏离 5% 触发再平衡）。
        commission_rate: 佣金率。
        stamp_tax_rate: 印花税率（仅卖出）。
        slippage_rate: 滑点率。
        short_term_days: 短期持有天数阈值。
        short_term_tax_rate: 短期资本利得税率。
        long_term_tax_rate: 长期资本利得税率。
        impact_cost_coeff: 冲击成本系数（与成交量占比相关）。
    """

    def __init__(
        self,
        drift_threshold: float = 0.05,
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        stamp_tax_rate: float = DEFAULT_STAMP_TAX_RATE,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
        short_term_days: int = DEFAULT_SHORT_TERM_DAYS,
        short_term_tax_rate: float = DEFAULT_SHORT_TERM_TAX_RATE,
        long_term_tax_rate: float = DEFAULT_LONG_TERM_TAX_RATE,
        impact_cost_coeff: float = 0.1,
    ):
        self.drift_threshold = float(drift_threshold)
        self.commission_rate = float(commission_rate)
        self.stamp_tax_rate = float(stamp_tax_rate)
        self.slippage_rate = float(slippage_rate)
        self.short_term_days = int(short_term_days)
        self.short_term_tax_rate = float(short_term_tax_rate)
        self.long_term_tax_rate = float(long_term_tax_rate)
        self.impact_cost_coeff = float(impact_cost_coeff)

    # ------------------------------------------------------------------ #
    # 漂移检测
    # ------------------------------------------------------------------ #
    def calc_drift(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> float:
        """计算当前权重与目标权重的最大绝对漂移。

        Returns:
            最大绝对偏离值。
        """
        cw = _normalize_weights(current_weights)
        tw = _normalize_weights(target_weights)
        all_syms = set(cw.keys()) | set(tw.keys())
        max_drift = 0.0
        for sym in all_syms:
            drift = abs(cw.get(sym, 0.0) - tw.get(sym, 0.0))
            if drift > max_drift:
                max_drift = drift
        return max_drift

    def should_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> bool:
        """漂移是否超过阈值。"""
        return self.calc_drift(current_weights, target_weights) >= self.drift_threshold

    # ------------------------------------------------------------------ #
    # 交易成本模型
    # ------------------------------------------------------------------ #
    def transaction_cost(
        self,
        symbol: str,
        action: str,
        shares: float,
        price: float,
        daily_volume: float = 0.0,
    ) -> float:
        """估算单笔交易成本。

        成本组成：
            1. 佣金 = 成交金额 × commission_rate（买卖双向）。
            2. 印花税 = 成交金额 × stamp_tax_rate（仅卖出）。
            3. 滑点 = 成交金额 × slippage_rate。
            4. 冲击成本 = 成交金额 × impact_cost_coeff × (|shares| / daily_volume)。

        Args:
            daily_volume: 当日成交量（股），用于估算冲击成本。

        Returns:
            总交易成本（正数）。
        """
        amount = abs(shares) * price
        commission = amount * self.commission_rate
        stamp_tax = amount * self.stamp_tax_rate if action == "sell" else 0.0
        slippage = amount * self.slippage_rate
        impact = 0.0
        if daily_volume > 0:
            impact = amount * self.impact_cost_coeff * (abs(shares) / daily_volume)
        return commission + stamp_tax + slippage + impact

    # ------------------------------------------------------------------ #
    # 税务 aware
    # ------------------------------------------------------------------ #
    def estimate_tax(
        self,
        action: str,
        shares: float,
        price: float,
        avg_cost: float,
        holding_days: int,
    ) -> float:
        """估算资本利得税。

        卖出盈利时：短期税率高于长期税率。
        卖出亏损时：产生税务抵扣（负税）。

        Returns:
            正数表示应缴税额，负数表示可抵扣额度。
        """
        if action != "sell" or shares <= 0:
            return 0.0
        pnl = (price - avg_cost) * shares
        if pnl <= 0:
            return pnl  # 亏损可抵扣
        rate = self.short_term_tax_rate if holding_days < self.short_term_days else self.long_term_tax_rate
        return pnl * rate

    def tax_loss_harvest_candidates(
        self,
        positions: Dict[str, Any],
        prices: Dict[str, float],
        tax_lots: Optional[Dict[str, List[TaxLot]]] = None,
    ) -> List[Dict[str, Any]]:
        """识别适合亏损收割的持仓。

        返回亏损且为短期的持仓（优先收割短期亏损，可立即抵税）。

        Returns:
            ``[{symbol, unrealized_loss, is_short_term, lot}]`` 列表，按亏损金额降序。
        """
        candidates: List[Dict[str, Any]] = []
        if tax_lots is not None:
            for sym, lots in tax_lots.items():
                price = prices.get(sym, 0.0)
                for lot in lots:
                    pnl = lot.unrealized_pnl(price)
                    if pnl < 0:
                        candidates.append({
                            "symbol": sym,
                            "unrealized_loss": abs(pnl),
                            "is_short_term": lot.is_short_term(),
                            "lot": lot,
                        })
        else:
            # 无 tax_lots 时从 positions 估算
            for sym, pos in positions.items():
                price = prices.get(sym, 0.0)
                shares = getattr(pos, "shares", pos.get("shares", 0)) if isinstance(pos, dict) else getattr(pos, "shares", 0)
                avg_cost = getattr(pos, "avg_cost", pos.get("avg_cost", 0)) if isinstance(pos, dict) else getattr(pos, "avg_cost", 0)
                pnl = (price - avg_cost) * shares
                if pnl < 0:
                    candidates.append({
                        "symbol": sym,
                        "unrealized_loss": abs(pnl),
                        "is_short_term": True,
                        "lot": None,
                    })
        candidates.sort(key=lambda x: x["unrealized_loss"], reverse=True)
        return candidates

    def avoid_short_term_gains(
        self,
        trades: List[Dict[str, Any]],
        tax_lots: Optional[Dict[str, List[TaxLot]]] = None,
        as_of: Optional[pd.Timestamp] = None,
    ) -> List[Dict[str, Any]]:
        """对交易列表进行短期资本利得规避调整。

        对盈利且为短期的持仓，将卖出量减少或推迟。

        Returns:
            调整后的交易列表。
        """
        if tax_lots is None:
            return trades
        if as_of is None:
            as_of = pd.Timestamp.now()
        adjusted: List[Dict[str, Any]] = []
        for t in trades:
            sym = t["symbol"]
            action = t["action"]
            if action != "sell" or sym not in tax_lots:
                adjusted.append(t)
                continue
            # 检查是否全部为短期盈利
            total_short_term_profit = 0.0
            for lot in tax_lots[sym]:
                if lot.is_short_term(as_of) and lot.avg_cost < t.get("price", 0.0):
                    total_short_term_profit += (t["price"] - lot.avg_cost) * lot.shares
            if total_short_term_profit > 0:
                # 减少卖出量至 50% 以规避短期税
                t_adj = dict(t)
                t_adj["shares"] = t["shares"] * 0.5
                t_adj["reason"] = (t.get("reason", "") + " [短期利得规避: 卖出量折半]").strip()
                adjusted.append(t_adj)
            else:
                adjusted.append(t)
        return adjusted

    # ------------------------------------------------------------------ #
    # 再平衡路径优化
    # ------------------------------------------------------------------ #
    def rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float,
        positions: Optional[Dict[str, Any]] = None,
        volumes: Optional[Dict[str, float]] = None,
        tax_lots: Optional[Dict[str, List[TaxLot]]] = None,
        min_trade_value: float = 0.0,
        allow_tax_loss_harvest: bool = True,
        avoid_short_term: bool = True,
    ) -> RebalanceResult:
        """执行智能再平衡。

        Pipeline:
            1. 计算漂移，判断是否需要再平衡。
            2. 计算目标持仓金额与当前持仓金额的差额。
            3. 生成初步交易清单（买入/卖出）。
            4. 税务 aware 调整（亏损收割 + 短期利得规避）。
            5. 估算交易成本与税费。
            6. 输出最终交易路径。

        Args:
            current_weights: 当前权重。
            target_weights: 目标权重。
            prices: ``{symbol: price}`` 当前价格。
            portfolio_value: 组合总市值。
            positions: 当前持仓 ``{symbol: Position-like}``。
            volumes: ``{symbol: daily_volume}`` 日成交量。
            tax_lots: 税务批次 ``{symbol: [TaxLot]}``。
            min_trade_value: 最小交易金额（低于此值的交易忽略）。
            allow_tax_loss_harvest: 是否启用亏损收割。
            avoid_short_term: 是否规避短期资本利得。

        Returns:
            :class:`RebalanceResult`。
        """
        cw = _normalize_weights(current_weights)
        tw = _normalize_weights(target_weights)
        drift = self.calc_drift(cw, tw)
        triggered = drift >= self.drift_threshold

        result = RebalanceResult(
            drift_before=drift,
            triggered=triggered,
        )

        if not triggered:
            return result

        volumes = volumes or {}
        positions = positions or {}
        if tax_lots is None and positions:
            # 从 positions 构造简单 tax_lots
            tax_lots = {}
            for sym, pos in positions.items():
                if isinstance(pos, dict):
                    shares = pos.get("shares", 0)
                    avg_cost = pos.get("avg_cost", 0.0)
                else:
                    shares = getattr(pos, "shares", 0)
                    avg_cost = getattr(pos, "avg_cost", 0.0)
                if shares > 0:
                    tax_lots[sym] = [TaxLot(
                        symbol=sym,
                        shares=float(shares),
                        avg_cost=float(avg_cost),
                        entry_date=pd.Timestamp.now() - pd.Timedelta(days=30),
                    )]

        # 1. 亏损收割（优先卖出亏损持仓）
        harvested = 0.0
        if allow_tax_loss_harvest:
            candidates = self.tax_loss_harvest_candidates(positions, prices, tax_lots)
            for cand in candidates:
                sym = cand["symbol"]
                lot = cand.get("lot")
                if lot is None:
                    continue
                # 全部卖出该亏损 lot
                shares = lot.shares
                price = prices.get(sym, 0.0)
                if shares > 0 and price > 0:
                    cost = self.transaction_cost(sym, "sell", shares, price, volumes.get(sym, 0.0))
                    tax = self.estimate_tax("sell", shares, price, lot.avg_cost, (pd.Timestamp.now() - lot.entry_date).days)
                    result.trades.append({
                        "symbol": sym,
                        "action": "sell",
                        "shares": float(shares),
                        "price": float(price),
                        "estimated_cost": round(cost, 2),
                        "estimated_tax": round(tax, 2),
                        "reason": "tax-loss harvesting",
                    })
                    result.total_cost += cost
                    result.total_tax += tax
                    if tax < 0:
                        harvested += abs(tax)
                    # 更新 current_weights（已卖出）
                    cw[sym] = max(0.0, cw.get(sym, 0.0) - (shares * price) / portfolio_value)

        result.tax_harvested = harvested

        # 2. 计算目标与当前差异，生成再平衡交易
        all_syms = sorted(set(cw.keys()) | set(tw.keys()) | set(prices.keys()))
        trades_raw: List[Dict[str, Any]] = []
        for sym in all_syms:
            price = prices.get(sym, 0.0)
            if price <= 0:
                continue
            target_val = tw.get(sym, 0.0) * portfolio_value
            current_val = cw.get(sym, 0.0) * portfolio_value
            delta_val = target_val - current_val
            if abs(delta_val) < min_trade_value:
                continue
            action = "buy" if delta_val > 0 else "sell"
            shares = abs(delta_val) / price
            # 卖出时不能超过当前持仓
            if action == "sell":
                pos_val = current_val
                max_sell_val = pos_val
                if abs(delta_val) > max_sell_val:
                    shares = max_sell_val / price if max_sell_val > 0 else 0
                    delta_val = -max_sell_val
                if shares <= 0:
                    continue
            vol = volumes.get(sym, 0.0)
            cost = self.transaction_cost(sym, action, shares, price, vol)
            trades_raw.append({
                "symbol": sym,
                "action": action,
                "shares": float(shares),
                "price": float(price),
                "delta_value": float(delta_val),
                "estimated_cost": round(cost, 2),
                "reason": "rebalance",
            })

        # 3. 短期利得规避
        if avoid_short_term and tax_lots is not None:
            trades_raw = self.avoid_short_term_gains(trades_raw, tax_lots)

        # 4. 按成本从低到高排序交易（路径优化：先做低成本交易）
        trades_sorted = sorted(trades_raw, key=lambda t: t["estimated_cost"])

        # 5. 累积成本并生成最终交易清单
        new_weights = dict(cw)
        for t in trades_sorted:
            sym = t["symbol"]
            action = t["action"]
            shares = t["shares"]
            price = t["price"]
            vol = volumes.get(sym, 0.0)
            cost = self.transaction_cost(sym, action, shares, price, vol)
            # 税费估算
            tax = 0.0
            if action == "sell" and tax_lots and sym in tax_lots:
                for lot in tax_lots[sym]:
                    days = (pd.Timestamp.now() - lot.entry_date).days
                    tax += self.estimate_tax(action, min(shares, lot.shares), price, lot.avg_cost, days)
                    shares -= lot.shares
                    if shares <= 0:
                        break

            result.trades.append({
                "symbol": sym,
                "action": action,
                "shares": round(float(t["shares"]), 2),
                "price": float(price),
                "estimated_cost": round(cost, 2),
                "estimated_tax": round(tax, 2),
                "reason": t.get("reason", "rebalance"),
            })
            result.total_cost += cost
            result.total_tax += tax
            # 更新 new_weights
            delta = (t["shares"] * price) / portfolio_value
            if action == "buy":
                new_weights[sym] = new_weights.get(sym, 0.0) + delta
            else:
                new_weights[sym] = max(0.0, new_weights.get(sym, 0.0) - delta)

        result.drift_after = self.calc_drift(new_weights, tw)
        return result

    # ------------------------------------------------------------------ #
    # 与 portfolio_optimizer 集成
    # ------------------------------------------------------------------ #
    def rebalance_from_optimizer(
        self,
        optimizer_result: OptimizeResult,
        current_weights: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float,
        **kwargs: Any,
    ) -> RebalanceResult:
        """直接基于 :class:`OptimizeResult` 执行再平衡。

        Args:
            optimizer_result: PortfolioOptimizer.optimize() 的返回结果。
            current_weights: 当前权重。
            prices: 当前价格。
            portfolio_value: 组合总市值。
            **kwargs: 透传给 :meth:`rebalance`。

        Returns:
            :class:`RebalanceResult`。
        """
        return self.rebalance(
            current_weights=current_weights,
            target_weights=optimizer_result.weights,
            prices=prices,
            portfolio_value=portfolio_value,
            **kwargs,
        )
