"""智能组合再平衡模块单元测试。

覆盖：
  - 漂移检测（calc_drift / should_rebalance）
  - 交易成本模型（commission + stamp_tax + slippage + impact）
  - 税务 aware（estimate_tax / tax_loss_harvest / avoid_short_term_gains）
  - 再平衡路径优化（最小化总成本排序）
  - 与 PortfolioOptimizer 集成（rebalance_from_optimizer）
  - 边界情况（空持仓 / 零漂移 / 无触发）

全部使用确定性构造的模拟数据。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from optimization.smart_rebalance import (
    RebalanceResult,
    SmartRebalancer,
    TaxLot,
    _normalize_weights,
)
from optimization.portfolio_optimizer import OptimizeResult


# ---------------------------------------------------------------------------
# 构造工具
# ---------------------------------------------------------------------------

def make_rebalancer(**kwargs) -> SmartRebalancer:
    """构造默认再平衡器。"""
    defaults = {
        "drift_threshold": 0.05,
        "commission_rate": 0.00025,
        "stamp_tax_rate": 0.0005,
        "slippage_rate": 0.001,
        "short_term_days": 365,
        "short_term_tax_rate": 0.20,
        "long_term_tax_rate": 0.10,
        "impact_cost_coeff": 0.1,
    }
    defaults.update(kwargs)
    return SmartRebalancer(**defaults)


def make_simple_positions() -> Dict[str, Dict[str, float]]:
    return {
        "A": {"shares": 1000.0, "avg_cost": 50.0},
        "B": {"shares": 500.0, "avg_cost": 100.0},
        "C": {"shares": 2000.0, "avg_cost": 25.0},
    }


# ---------------------------------------------------------------------------
# 1. 漂移检测
# ---------------------------------------------------------------------------

class TestDriftDetection:
    def test_equal_weights_zero_drift(self):
        """相同权重漂移应为 0。"""
        rb = make_rebalancer()
        w = {"A": 0.5, "B": 0.5}
        assert rb.calc_drift(w, w) == 0.0

    def test_drift_calculation(self):
        """漂移应取最大绝对偏离。"""
        rb = make_rebalancer()
        current = {"A": 0.6, "B": 0.4}
        target = {"A": 0.5, "B": 0.5}
        assert rb.calc_drift(current, target) == pytest.approx(0.1)

    def test_drift_with_extra_symbol(self):
        """目标含新标的时漂移应正确计算。"""
        rb = make_rebalancer()
        current = {"A": 1.0}
        target = {"A": 0.5, "B": 0.5}
        assert rb.calc_drift(current, target) == pytest.approx(0.5)

    def test_should_rebalance_at_threshold(self):
        """漂移等于阈值时应触发。"""
        rb = make_rebalancer(drift_threshold=0.05)
        current = {"A": 0.55, "B": 0.45}
        target = {"A": 0.5, "B": 0.5}
        assert rb.should_rebalance(current, target) is True

    def test_should_not_rebalance_below_threshold(self):
        """漂移低于阈值时不触发。"""
        rb = make_rebalancer(drift_threshold=0.05)
        current = {"A": 0.52, "B": 0.48}
        target = {"A": 0.5, "B": 0.5}
        assert rb.should_rebalance(current, target) is False

    def test_normalize_weights(self):
        """权重归一化后和为 1。"""
        w = {"A": 30.0, "B": 70.0}
        nw = _normalize_weights(w)
        assert sum(nw.values()) == pytest.approx(1.0)
        assert nw["A"] == pytest.approx(0.3)

    def test_normalize_zero_weights(self):
        """零权重归一化返回空。"""
        assert _normalize_weights({"A": 0.0}) == {}


# ---------------------------------------------------------------------------
# 2. 交易成本模型
# ---------------------------------------------------------------------------

class TestTransactionCost:
    def test_buy_cost_components(self):
        """买入成本 = 佣金 + 滑点 + 冲击成本。"""
        rb = make_rebalancer()
        cost = rb.transaction_cost("A", "buy", 1000, 50.0, daily_volume=1e6)
        amount = 1000 * 50.0
        expected_commission = amount * 0.00025
        expected_slippage = amount * 0.001
        expected_impact = amount * 0.1 * (1000 / 1e6)
        # 买入无印花税
        expected = expected_commission + expected_slippage + expected_impact
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_sell_cost_includes_stamp_tax(self):
        """卖出成本含印花税 + 冲击成本。"""
        rb = make_rebalancer()
        cost = rb.transaction_cost("A", "sell", 1000, 50.0, daily_volume=1e6)
        amount = 1000 * 50.0
        expected_commission = amount * 0.00025
        expected_stamp = amount * 0.0005
        expected_slippage = amount * 0.001
        expected_impact = amount * 0.1 * (1000 / 1e6)
        expected = expected_commission + expected_stamp + expected_slippage + expected_impact
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_impact_cost(self):
        """冲击成本与成交量占比相关。"""
        rb = make_rebalancer()
        vol = 1e6
        shares = 0.1 * vol  # 占 10% 成交量
        price = 50.0
        cost_with_impact = rb.transaction_cost("A", "buy", shares, price, daily_volume=vol)
        cost_without_impact = rb.transaction_cost("A", "buy", shares, price, daily_volume=0.0)
        assert cost_with_impact > cost_without_impact

    def test_zero_volume_no_impact(self):
        """成交量为 0 时冲击成本为 0。"""
        rb = make_rebalancer()
        cost = rb.transaction_cost("A", "buy", 100, 50.0, daily_volume=0.0)
        amount = 100 * 50.0
        expected = amount * (0.00025 + 0.001)
        assert cost == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. 税务 aware
# ---------------------------------------------------------------------------

class TestTaxAware:
    def test_estimate_tax_profit_short_term(self):
        """短期盈利卖出应缴税。"""
        rb = make_rebalancer()
        tax = rb.estimate_tax("sell", 100, 60.0, 50.0, holding_days=30)
        pnl = (60.0 - 50.0) * 100
        assert tax == pytest.approx(pnl * 0.20, rel=1e-9)

    def test_estimate_tax_profit_long_term(self):
        """长期盈利卖出税率更低。"""
        rb = make_rebalancer()
        tax = rb.estimate_tax("sell", 100, 60.0, 50.0, holding_days=500)
        pnl = (60.0 - 50.0) * 100
        assert tax == pytest.approx(pnl * 0.10, rel=1e-9)

    def test_estimate_tax_loss(self):
        """亏损卖出产生税务抵扣（负数）。"""
        rb = make_rebalancer()
        tax = rb.estimate_tax("sell", 100, 40.0, 50.0, holding_days=30)
        assert tax < 0

    def test_estimate_tax_buy_is_zero(self):
        """买入不应产生税费。"""
        rb = make_rebalancer()
        tax = rb.estimate_tax("buy", 100, 50.0, 50.0, holding_days=30)
        assert tax == 0.0

    def test_tax_loss_harvest_candidates(self):
        """亏损持仓应被识别为收割候选。"""
        rb = make_rebalancer()
        positions = {
            "A": {"shares": 100, "avg_cost": 50.0},
            "B": {"shares": 100, "avg_cost": 100.0},
        }
        prices = {"A": 40.0, "B": 120.0}
        candidates = rb.tax_loss_harvest_candidates(positions, prices)
        assert len(candidates) == 1
        assert candidates[0]["symbol"] == "A"
        assert candidates[0]["unrealized_loss"] == pytest.approx(1000.0, rel=1e-9)

    def test_tax_loss_harvest_with_tax_lots(self):
        """使用 TaxLot 的亏损收割。"""
        rb = make_rebalancer()
        tax_lots = {
            "A": [
                TaxLot("A", 100, 50.0, pd.Timestamp.now() - pd.Timedelta(days=30)),
            ],
        }
        prices = {"A": 40.0}
        candidates = rb.tax_loss_harvest_candidates({}, prices, tax_lots)
        assert len(candidates) == 1
        assert candidates[0]["symbol"] == "A"

    def test_avoid_short_term_gains(self):
        """短期盈利卖出应减少卖出量。"""
        rb = make_rebalancer()
        tax_lots = {
            "A": [
                TaxLot("A", 100, 50.0, pd.Timestamp.now() - pd.Timedelta(days=30)),
            ],
        }
        trades = [{"symbol": "A", "action": "sell", "shares": 100, "price": 60.0}]
        adjusted = rb.avoid_short_term_gains(trades, tax_lots)
        assert adjusted[0]["shares"] == pytest.approx(50.0, rel=1e-9)
        assert "短期利得规避" in adjusted[0]["reason"]

    def test_avoid_short_term_no_tax_lots(self):
        """无 tax_lots 时不调整。"""
        rb = make_rebalancer()
        trades = [{"symbol": "A", "action": "sell", "shares": 100, "price": 60.0}]
        adjusted = rb.avoid_short_term_gains(trades, None)
        assert adjusted == trades

    def test_tax_lot_is_short_term(self):
        """TaxLot 短期判断。"""
        lot = TaxLot("A", 100, 50.0, pd.Timestamp.now() - pd.Timedelta(days=30))
        assert lot.is_short_term() is True

    def test_tax_lot_is_long_term(self):
        """TaxLot 长期判断。"""
        lot = TaxLot("A", 100, 50.0, pd.Timestamp.now() - pd.Timedelta(days=500))
        assert lot.is_short_term() is False

    def test_tax_lot_unrealized_pnl(self):
        """TaxLot 未实现盈亏计算。"""
        lot = TaxLot("A", 100, 50.0, pd.Timestamp.now())
        assert lot.unrealized_pnl(60.0) == pytest.approx(1000.0, rel=1e-9)
        assert lot.unrealized_pnl(40.0) == pytest.approx(-1000.0, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. 再平衡
# ---------------------------------------------------------------------------

class TestRebalance:
    def test_no_trigger_below_threshold(self):
        """漂移低于阈值时不产生交易。"""
        rb = make_rebalancer(drift_threshold=0.1)
        current = {"A": 0.51, "B": 0.49}
        target = {"A": 0.50, "B": 0.50}
        prices = {"A": 50.0, "B": 100.0}
        result = rb.rebalance(current, target, prices, portfolio_value=100_000.0)
        assert result.triggered is False
        assert len(result.trades) == 0

    def test_trigger_above_threshold(self):
        """漂移超过阈值时产生交易。"""
        rb = make_rebalancer(drift_threshold=0.05)
        current = {"A": 0.7, "B": 0.3}
        target = {"A": 0.5, "B": 0.5}
        prices = {"A": 50.0, "B": 100.0}
        result = rb.rebalance(current, target, prices, portfolio_value=100_000.0)
        assert result.triggered is True
        assert len(result.trades) > 0

    def test_drift_after_less_than_before(self):
        """再平衡后漂移应减小。"""
        rb = make_rebalancer(drift_threshold=0.05)
        current = {"A": 0.7, "B": 0.3}
        target = {"A": 0.5, "B": 0.5}
        prices = {"A": 50.0, "B": 100.0}
        result = rb.rebalance(current, target, prices, portfolio_value=100_000.0)
        assert result.drift_after < result.drift_before

    def test_trades_sorted_by_cost(self):
        """交易应按成本从低到高排序。"""
        rb = make_rebalancer(drift_threshold=0.05)
        current = {"A": 0.8, "B": 0.1, "C": 0.1}
        target = {"A": 0.4, "B": 0.3, "C": 0.3}
        prices = {"A": 100.0, "B": 10.0, "C": 10.0}
        volumes = {"A": 1e8, "B": 1e8, "C": 1e8}
        result = rb.rebalance(
            current, target, prices, portfolio_value=100_000.0,
            volumes=volumes,
        )
        costs = [t["estimated_cost"] for t in result.trades]
        assert costs == sorted(costs)

    def test_min_trade_value_filter(self):
        """min_trade_value 应过滤小额交易。"""
        rb = make_rebalancer(drift_threshold=0.05)
        current = {"A": 0.51, "B": 0.49}
        target = {"A": 0.50, "B": 0.50}
        prices = {"A": 50.0, "B": 100.0}
        result = rb.rebalance(
            current, target, prices, portfolio_value=100_000.0,
            min_trade_value=10_000.0,
        )
        # 差额只有 1000 元，应被过滤
        assert len(result.trades) == 0

    def test_tax_loss_harvest_in_rebalance(self):
        """启用亏损收割时优先卖出亏损持仓。"""
        rb = make_rebalancer(drift_threshold=0.05)
        current = {"A": 0.5, "B": 0.5}
        target = {"A": 0.3, "B": 0.7}
        prices = {"A": 40.0, "B": 100.0}
        positions = {
            "A": {"shares": 1250.0, "avg_cost": 50.0},  # 亏损
            "B": {"shares": 500.0, "avg_cost": 100.0},
        }
        result = rb.rebalance(
            current, target, prices, portfolio_value=100_000.0,
            positions=positions, allow_tax_loss_harvest=True,
        )
        assert result.triggered is True
        # 应有亏损收割交易
        harvest_trades = [t for t in result.trades if "tax-loss harvesting" in t.get("reason", "")]
        assert len(harvest_trades) > 0

    def test_rebalance_result_as_dict(self):
        """RebalanceResult.as_dict 应序列化。"""
        result = RebalanceResult(
            trades=[{"symbol": "A", "action": "buy"}],
            total_cost=10.0,
            drift_before=0.1,
            triggered=True,
        )
        d = result.as_dict()
        assert d["total_cost"] == 10.0
        assert d["triggered"] is True
        assert len(d["trades"]) == 1


# ---------------------------------------------------------------------------
# 5. 与 PortfolioOptimizer 集成
# ---------------------------------------------------------------------------

class TestOptimizerIntegration:
    def test_rebalance_from_optimizer(self):
        """rebalance_from_optimizer 应正确使用 OptimizeResult 的权重。"""
        rb = make_rebalancer(drift_threshold=0.05)
        opt_result = OptimizeResult(
            weights={"A": 0.5, "B": 0.5},
            expected_return=0.1,
            expected_volatility=0.15,
            sharpe=0.6,
            method="equal_weight",
        )
        current = {"A": 0.7, "B": 0.3}
        prices = {"A": 50.0, "B": 100.0}
        result = rb.rebalance_from_optimizer(
            opt_result, current, prices, portfolio_value=100_000.0
        )
        assert result.triggered is True
        # 验证交易方向正确（卖出 A，买入 B）
        actions = {t["symbol"]: t["action"] for t in result.trades}
        assert actions.get("A") == "sell"
        assert actions.get("B") == "buy"


# ---------------------------------------------------------------------------
# 6. 边界情况
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_current_weights(self):
        """空当前权重应能处理。"""
        rb = make_rebalancer(drift_threshold=0.05)
        current = {}
        target = {"A": 1.0}
        prices = {"A": 50.0}
        result = rb.rebalance(current, target, prices, portfolio_value=100_000.0)
        assert result.triggered is True

    def test_zero_price_skipped(self):
        """价格为 0 的标的应被跳过。"""
        rb = make_rebalancer(drift_threshold=0.05)
        current = {"A": 0.5, "B": 0.5}
        target = {"A": 0.7, "B": 0.3}
        prices = {"A": 50.0, "B": 0.0}
        result = rb.rebalance(current, target, prices, portfolio_value=100_000.0)
        # B 价格无效，不应产生交易
        b_trades = [t for t in result.trades if t["symbol"] == "B"]
        assert len(b_trades) == 0

    def test_all_same_weights_no_trigger(self):
        """权重完全一致不应触发。"""
        rb = make_rebalancer(drift_threshold=0.01)
        w = {"A": 0.5, "B": 0.5}
        prices = {"A": 50.0, "B": 100.0}
        result = rb.rebalance(w, w, prices, portfolio_value=100_000.0)
        assert result.triggered is False

    def test_rebalance_result_defaults(self):
        """RebalanceResult 默认值测试。"""
        result = RebalanceResult()
        assert result.trades == []
        assert result.total_cost == 0.0
        assert result.total_tax == 0.0
        assert result.triggered is False
        assert result.tax_harvested == 0.0
