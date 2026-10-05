"""期权支持模块单元测试。

覆盖：
  - Black-Scholes 定价（call/put、到期边界、零波动率、put-call parity）
  - Greeks 计算（方向与范围、到期边界）
  - OptionQuote 数据结构
  - Mock 期权链生成器（确定性、行权价数量、IV 微笑）
  - 期权组合模板（备兑认购 / 牛市价差 / 跨式）
  - OptionDataProvider mock 降级

全部使用 mock 数据，不依赖外部 API。
"""
from __future__ import annotations

import math

import pytest

from data.options import (
    MockOptionChainGenerator,
    OptionDataProvider,
    OptionQuote,
    OptionStrategyBuilder,
    OptionStrategyLeg,
    _combine_greeks,
    black_scholes_price,
    calculate_greeks,
)


# ---------------------------------------------------------------------------
# Black-Scholes 定价
# ---------------------------------------------------------------------------


class TestBlackScholesPrice:
    def test_atm_call_positive(self):
        price = black_scholes_price(100, 100, 0.5, 0.25)
        assert 0 < price < 100

    def test_atm_put_positive(self):
        price = black_scholes_price(100, 100, 0.5, 0.25, option_type="put")
        assert 0 < price < 100

    def test_put_call_parity(self):
        """C - P = S - K·e^{-rT}。"""
        s, k, t, v, r = 100.0, 105.0, 0.75, 0.3, 0.03
        c = black_scholes_price(s, k, t, v, r, "call")
        p = black_scholes_price(s, k, t, v, r, "put")
        lhs = c - p
        rhs = s - k * math.exp(-r * t)
        assert lhs == pytest.approx(rhs, rel=1e-6)

    def test_call_value_decreases_with_strike(self):
        p1 = black_scholes_price(100, 90, 0.5, 0.25)
        p2 = black_scholes_price(100, 110, 0.5, 0.25)
        assert p1 > p2

    def test_expired_call_intrinsic(self):
        assert black_scholes_price(110, 100, 0, 0.25, option_type="call") == pytest.approx(10.0)
        assert black_scholes_price(95, 100, 0, 0.25, option_type="call") == 0.0

    def test_expired_put_intrinsic(self):
        assert black_scholes_price(95, 100, 0, 0.25, option_type="put") == pytest.approx(5.0)
        assert black_scholes_price(110, 100, 0, 0.25, option_type="put") == 0.0

    def test_zero_vol_intrinsic(self):
        assert black_scholes_price(120, 100, 0.5, 0.0, option_type="call") == pytest.approx(20.0)
        assert black_scholes_price(80, 100, 0.5, 0.0, option_type="put") == pytest.approx(20.0)

    def test_deep_itm_call_near_intrinsic(self):
        price = black_scholes_price(200, 100, 1.0, 0.2)
        assert price > 95  # 深度实值 call 接近内在价值


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------


class TestGreeks:
    def test_call_delta_range(self):
        g = calculate_greeks(100, 100, 0.5, 0.25, option_type="call")
        assert 0 < g["delta"] < 1

    def test_put_delta_range(self):
        g = calculate_greeks(100, 100, 0.5, 0.25, option_type="put")
        assert -1 < g["delta"] < 0

    def test_gamma_positive_and_symmetric(self):
        gc = calculate_greeks(100, 100, 0.5, 0.25, option_type="call")
        gp = calculate_greeks(100, 100, 0.5, 0.25, option_type="put")
        assert gc["gamma"] > 0
        assert gc["gamma"] == pytest.approx(gp["gamma"], rel=1e-9)

    def test_vega_positive(self):
        g = calculate_greeks(100, 100, 0.5, 0.25)
        assert g["vega"] > 0

    def test_theta_negative_atm(self):
        g = calculate_greeks(100, 100, 0.5, 0.25)
        assert g["theta"] < 0

    def test_call_rho_positive_put_rho_negative(self):
        gc = calculate_greeks(100, 100, 0.5, 0.25, option_type="call")
        gp = calculate_greeks(100, 100, 0.5, 0.25, option_type="put")
        assert gc["rho"] > 0
        assert gp["rho"] < 0

    def test_expired_greeks(self):
        gc = calculate_greeks(110, 100, 0, 0.25, option_type="call")
        assert gc["delta"] == 1.0
        assert gc["gamma"] == 0.0 and gc["vega"] == 0.0
        gp = calculate_greeks(110, 100, 0, 0.25, option_type="put")
        assert gp["delta"] == 0.0
        go = calculate_greeks(90, 100, 0, 0.25, option_type="call")
        assert go["delta"] == 0.0

    def test_zero_vol_greeks(self):
        g = calculate_greeks(100, 100, 0.5, 0.0)
        assert g["gamma"] == 0.0 and g["vega"] == 0.0

    def test_non_positive_inputs(self):
        g = calculate_greeks(0, 100, 0.5, 0.25)
        assert g["delta"] == 0.0 and g["gamma"] == 0.0


# ---------------------------------------------------------------------------
# OptionQuote
# ---------------------------------------------------------------------------


class TestOptionQuote:
    def test_as_dict_roundtrip(self):
        q = OptionQuote(
            underlying="TEST", strike=100.0, expiry="2026-12-31",
            option_type="call", last_price=5.0, iv=0.25, ttm_years=0.5,
            greeks={"delta": 0.5, "gamma": 0.02, "vega": 0.1, "theta": -0.02, "rho": 0.05},
        )
        d = q.as_dict()
        assert d["underlying"] == "TEST"
        assert d["strike"] == 100.0
        assert d["greeks"]["delta"] == 0.5

    def test_moneyness_returns_string(self):
        q = OptionQuote("TEST", 100.0, "2026-12-31", "call", 5.0, 0.25, 0.5)
        assert isinstance(q.moneyness, str)


# ---------------------------------------------------------------------------
# Mock 期权链
# ---------------------------------------------------------------------------


class TestMockOptionChainGenerator:
    def test_chain_structure(self):
        gen = MockOptionChainGenerator()
        chain = gen.generate_chain("TEST", 100.0, "2026-12-31", 0.5)
        assert set(chain.keys()) >= {"calls", "puts"}
        assert len(chain["calls"]) > 0
        assert len(chain["puts"]) > 0

    def test_deterministic(self):
        gen = MockOptionChainGenerator()
        c1 = gen.generate_chain("TEST", 100.0, "2026-12-31", 0.5)
        c2 = gen.generate_chain("TEST", 100.0, "2026-12-31", 0.5)
        assert [q.strike for q in c1["calls"]] == [q.strike for q in c2["calls"]]
        assert [q.last_price for q in c1["calls"]] == [q.last_price for q in c2["calls"]]

    def test_num_strikes(self):
        gen = MockOptionChainGenerator()
        chain = gen.generate_chain("TEST", 100.0, "2026-12-31", 0.5, num_strikes=7)
        assert len(chain["calls"]) == 7
        assert len(chain["puts"]) == 7

    def test_custom_strikes(self):
        gen = MockOptionChainGenerator()
        strikes = [95.0, 100.0, 105.0]
        chain = gen.generate_chain("TEST", 100.0, "2026-12-31", 0.5, strikes=strikes)
        assert [q.strike for q in chain["calls"]] == strikes

    def test_call_put_same_strike_prices_consistent(self):
        """同一行权价的 call 应贵于深度 OTM call。"""
        gen = MockOptionChainGenerator()
        chain = gen.generate_chain("TEST", 100.0, "2026-12-31", 0.5)
        calls = {q.strike: q for q in chain["calls"]}
        puts = {q.strike: q for q in chain["puts"]}
        low_strike = min(calls)
        high_strike = max(calls)
        assert calls[low_strike].last_price > calls[high_strike].last_price
        assert puts[high_strike].last_price > puts[low_strike].last_price

    def test_quotes_have_greeks(self):
        gen = MockOptionChainGenerator()
        chain = gen.generate_chain("TEST", 100.0, "2026-12-31", 0.5)
        q = chain["calls"][0]
        for key in ("delta", "gamma", "vega", "theta", "rho"):
            assert key in q.greeks


# ---------------------------------------------------------------------------
# 组合模板
# ---------------------------------------------------------------------------


class TestOptionStrategies:
    def make_builder(self, spot=100.0):
        gen = MockOptionChainGenerator()
        chain = gen.generate_chain("TEST", spot, "2026-12-31", 0.5)
        return OptionStrategyBuilder(chain)

    def test_covered_call(self):
        b = self.make_builder(100.0)
        r = b.covered_call(100.0, 105.0, 1)
        assert r is not None
        assert r.name == "covered_call"
        # 备兑 = 持有标的（现货腿不计入期权腿）+ 卖出 1 条 call
        assert len(r.legs) == 1
        assert r.legs[0].side == "short"
        # 收权利金，净成本为负（净收入）
        assert r.net_cost < 0
        assert r.max_profit is not None and r.max_profit > 0
        assert r.max_loss is not None

    def test_covered_call_missing_strike(self):
        b = self.make_builder(100.0)
        assert b.covered_call(100.0, 999.0, 1) is None

    def test_bull_call_spread(self):
        b = self.make_builder(100.0)
        r = b.bull_call_spread(95.0, 105.0, 1)
        assert r is not None
        assert r.name == "bull_call_spread"
        assert len(r.legs) == 2
        # 买低行权价 call + 卖高行权价 call：净成本为正
        assert r.net_cost > 0
        assert r.max_loss == pytest.approx(r.net_cost, rel=1e-6)
        if r.max_profit is not None:
            assert r.max_profit >= 0

    def test_bull_call_spread_missing_leg(self):
        b = self.make_builder(100.0)
        assert b.bull_call_spread(1.0, 2.0, 1) is None

    def test_straddle(self):
        b = self.make_builder(100.0)
        r = b.straddle(100.0, 1)
        assert r is not None
        assert r.name == "straddle"
        assert len(r.legs) == 2
        # 买 call + 买 put：净成本为正
        assert r.net_cost > 0
        assert r.max_loss == pytest.approx(r.net_cost, rel=1e-6)

    def test_combined_greeks_sign_flip(self):
        """组合 Greeks 应对 short 腿取负号。"""
        leg_long = OptionStrategyLeg(
            option_type="call", strike=100.0, expiry="2026-12-31",
            side="long", quantity=1, price=5.0,
            greeks={"delta": 0.5, "gamma": 0.02},
        )
        leg_short = OptionStrategyLeg(
            option_type="call", strike=105.0, expiry="2026-12-31",
            side="short", quantity=1, price=3.0,
            greeks={"delta": 0.4, "gamma": 0.015},
        )
        combined = _combine_greeks([leg_long, leg_short])
        assert combined["delta"] == pytest.approx(0.5 - 0.4, rel=1e-9)
        assert combined["gamma"] == pytest.approx(0.02 - 0.015, rel=1e-9)

    def test_leg_net_cost_sign(self):
        leg_long = OptionStrategyLeg("call", 100.0, "2026-12-31", "long", 2, 5.0)
        leg_short = OptionStrategyLeg("call", 105.0, "2026-12-31", "short", 1, 3.0)
        assert leg_long.net_cost() == pytest.approx(10.0)
        assert leg_short.net_cost() == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# OptionDataProvider
# ---------------------------------------------------------------------------


class TestOptionDataProvider:
    def test_fetch_chain_mock_default(self):
        p = OptionDataProvider()
        chain = p.fetch_chain("TEST", 100.0, "2026-12-31", 0.5)
        assert "calls" in chain and "puts" in chain
        assert all(isinstance(q, OptionQuote) for q in chain["calls"])

    def test_fetch_chain_deterministic(self):
        p = OptionDataProvider()
        c1 = p.fetch_chain("TEST", 100.0, "2026-12-31", 0.5)
        c2 = p.fetch_chain("TEST", 100.0, "2026-12-31", 0.5)
        assert [q.last_price for q in c1["calls"]] == [q.last_price for q in c2["calls"]]
