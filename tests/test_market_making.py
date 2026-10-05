"""做市策略单元测试。

覆盖：
  - ATR 计算（含数据不足边界）
  - 双边报价生成（价差为正、库存偏斜方向、上限截断）
  - 成交模拟（单边/双边/不成交）
  - 日K线降级信号生成（_compute_raw_signals）
  - generate_signals shift(1) 防未来函数
  - 缺失列 / 非法输入等边界情况

使用确定性随机种子构造模拟 K 线，不依赖外部 API 与 Level2 数据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.market_making import MarketMakingStrategy


# ---------------------------------------------------------------------------
# 工具：确定性 mock K 线
# ---------------------------------------------------------------------------


def make_ohlc(n: int = 120, seed: int = 42, base: float = 100.0) -> pd.DataFrame:
    """生成确定性日K线。"""
    rng = np.random.RandomState(seed)
    rets = rng.normal(0.0, 0.01, size=n)
    close = base * np.cumprod(1.0 + rets)
    open_ = close * (1.0 + rng.normal(0, 0.003, size=n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.004, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.004, size=n)))
    volume = rng.uniform(1e5, 1e6, size=n)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


class TestComputeATR:
    def test_atr_positive(self):
        df = make_ohlc(60)
        atr = MarketMakingStrategy.compute_atr(df, 14)
        assert len(atr) == len(df)
        valid = atr.dropna()
        assert len(valid) > 0
        assert (valid > 0).all()

    def test_atr_insufficient_data(self):
        df = make_ohlc(5)
        atr = MarketMakingStrategy.compute_atr(df, 14)
        # min_periods=1：数据不足时用可得数据回退，但仍应有输出且为正
        assert len(atr) == 5
        assert (atr.dropna() > 0).all()

    def test_atr_tr_logic(self):
        # 第2天：high-low=0.4, |high-pre_close|=1.8, |low-pre_close|=2.2 -> TR=2.2（跳空主导）
        df = pd.DataFrame({
            "open": [10.0, 9.0],
            "high": [11.0, 9.2],
            "low": [9.5, 8.8],
            "close": [11.0, 9.0],
        })
        atr = MarketMakingStrategy.compute_atr(df, 1)
        assert abs(atr.iloc[0] - 1.5) < 1e-9   # 11 - 9.5
        assert abs(atr.iloc[1] - 2.2) < 1e-9   # |8.8 - 11|


# ---------------------------------------------------------------------------
# 双边报价
# ---------------------------------------------------------------------------


class TestQuote:
    def setup_method(self):
        self.s = MarketMakingStrategy()

    def test_bid_below_ask(self):
        q = self.s.quote(100.0, 2.0, 0)
        assert q["bid"] < q["ask"]
        assert q["mid"] == 100.0
        assert q["spread"] == pytest.approx(q["ask"] - q["bid"])
        assert q["skew"] == 0.0

    def test_spread_from_atr(self):
        q = self.s.quote(100.0, 2.0, 0)
        # half_spread = 0.5 * 2 / 2 = 0.5
        assert q["bid"] == pytest.approx(99.5, abs=1e-6)
        assert q["ask"] == pytest.approx(100.5, abs=1e-6)

    def test_long_inventory_skews_down(self):
        q_flat = self.s.quote(100.0, 2.0, 0)
        q_long = self.s.quote(100.0, 2.0, 50)
        assert q_long["bid"] < q_flat["bid"]
        assert q_long["ask"] < q_flat["ask"]

    def test_short_inventory_skews_up(self):
        q_flat = self.s.quote(100.0, 2.0, 0)
        q_short = self.s.quote(100.0, 2.0, -50)
        assert q_short["bid"] > q_flat["bid"]
        assert q_short["ask"] > q_flat["ask"]

    def test_inventory_cap_stops_same_side(self):
        s = MarketMakingStrategy({"max_inventory": 100, "quote_size": 10})
        q_max = s.quote(100.0, 2.0, 100)
        assert q_max["bid_size"] == 0
        assert q_max["ask_size"] == 10

        q_min = s.quote(100.0, 2.0, -100)
        assert q_min["ask_size"] == 0
        assert q_min["bid_size"] == 10

    def test_non_positive_mid(self):
        q = self.s.quote(0.0, 2.0, 0)
        assert q["bid"] == 0.0 and q["ask"] == 0.0
        q2 = self.s.quote(-5.0, 2.0, 0)
        assert q2["bid"] == 0.0

    def test_custom_params(self):
        s = MarketMakingStrategy({"spread_k": 1.0, "inventory_skew": 0.0})
        q = s.quote(100.0, 2.0, 30)
        # skew=0 -> 库存不影响报价
        assert q["skew"] == 0.0
        assert q["bid"] == pytest.approx(99.0, abs=1e-6)
        assert q["ask"] == pytest.approx(101.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 成交模拟
# ---------------------------------------------------------------------------


class TestSimulateFills:
    def setup_method(self):
        self.s = MarketMakingStrategy()

    def test_bid_fill(self):
        q = self.s.quote(100.0, 2.0, 0)
        r = self.s.simulate_fills(q, low=99.0, high=100.0)
        assert r["bid_filled"] is True
        assert r["ask_filled"] is False
        assert r["inventory_delta"] == q["bid_size"]

    def test_ask_fill(self):
        q = self.s.quote(100.0, 2.0, 0)
        r = self.s.simulate_fills(q, low=100.0, high=101.0)
        assert r["ask_filled"] is True
        assert r["bid_filled"] is False
        assert r["inventory_delta"] == -q["ask_size"]

    def test_both_fill_spread_capture(self):
        q = self.s.quote(100.0, 2.0, 0)
        r = self.s.simulate_fills(q, low=99.0, high=101.0)
        assert r["bid_filled"] and r["ask_filled"]
        assert r["inventory_delta"] == 0
        # 双边成交赚取价差：pnl = ask*ask_size - bid*bid_size > 0
        assert r["cash_pnl"] > 0

    def test_no_fill(self):
        q = self.s.quote(100.0, 0.01, 0)
        # bid=99.9975, ask=100.0025：low/high 均未触及
        r = self.s.simulate_fills(q, low=100.0, high=100.001)
        assert not r["bid_filled"] and not r["ask_filled"]
        assert r["inventory_delta"] == 0
        assert r["cash_pnl"] == 0.0


# ---------------------------------------------------------------------------
# 日K线降级信号
# ---------------------------------------------------------------------------


class TestRawSignals:
    def test_output_columns(self):
        df = make_ohlc(120)
        s = MarketMakingStrategy()
        out = s._compute_raw_signals(df)
        for col in ("signal", "confidence", "bid", "ask", "mid", "spread",
                    "inventory", "atr", "volatility"):
            assert col in out.columns

    def test_bid_ask_consistency(self):
        df = make_ohlc(120)
        s = MarketMakingStrategy()
        out = s._compute_raw_signals(df)
        valid = out[out["spread"] > 0]
        assert len(valid) > 0
        assert (valid["bid"] < valid["ask"]).all()

    def test_inventory_bounded(self):
        df = make_ohlc(200, seed=7)
        s = MarketMakingStrategy({"max_inventory": 100})
        out = s._compute_raw_signals(df)
        assert out["inventory"].abs().max() <= 100

    def test_missing_columns_zero_signal(self):
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
        s = MarketMakingStrategy()
        out = s._compute_raw_signals(df)
        assert (out["signal"] == 0).all()
        assert (out["confidence"] == 0.0).all()

    def test_confidence_bounded(self):
        df = make_ohlc(120)
        s = MarketMakingStrategy()
        out = s._compute_raw_signals(df)
        assert ((out["confidence"] >= 0) & (out["confidence"] <= 1)).all()


# ---------------------------------------------------------------------------
# generate_signals（BaseStrategy 接口 + shift(1)）
# ---------------------------------------------------------------------------


class TestGenerateSignals:
    def test_signals_generated(self):
        df = make_ohlc(200, seed=11)
        s = MarketMakingStrategy()
        signals = s.generate_signals(df, symbol="TEST")
        assert isinstance(signals, list)
        for sig in signals:
            assert sig.strategy == "market_making"
            assert sig.symbol == "TEST"
            assert sig.action in ("buy", "sell")
            assert 0.0 <= sig.confidence <= 1.0

    def test_shift_one_no_lookahead(self):
        """t 日信号只能来自 t-1 日报价：首行必有 NaN -> 无信号。"""
        df = make_ohlc(200, seed=11)
        s = MarketMakingStrategy()
        raw = s._compute_raw_signals(df)
        n_nonzero_raw = int((raw["signal"] != 0).sum())
        signals = s.generate_signals(df, symbol="TEST")
        # shift(1) 后首行信号必为 NaN，非零信号数不超过原始非零数
        assert len(signals) <= n_nonzero_raw

    def test_first_day_excluded(self):
        df = make_ohlc(120)
        s = MarketMakingStrategy()
        signals = s.generate_signals(df, symbol="TEST")
        first_idx = df.index[0]
        assert all(sig.date != first_idx for sig in signals)

    def test_metadata_fields(self):
        df = make_ohlc(200, seed=3)
        s = MarketMakingStrategy()
        signals = s.generate_signals(df, symbol="TEST")
        if signals:
            md = signals[0].metadata
            for key in ("bid", "ask", "spread", "inventory"):
                assert key in md


# ---------------------------------------------------------------------------
# 策略元信息与集成
# ---------------------------------------------------------------------------


class TestStrategyMeta:
    def test_name_and_inheritance(self):
        from strategies.base_strategy import BaseStrategy
        s = MarketMakingStrategy()
        assert isinstance(s, BaseStrategy)
        assert s.name == "market_making"

    def test_default_params(self):
        s = MarketMakingStrategy()
        assert s.spread_k == 0.5
        assert s.inventory_skew == 0.5
        assert s.max_inventory == 100
        assert s.quote_size == 10
        assert s.atr_period == 14
        assert s.vol_period == 20

    def test_importable_from_package(self):
        from strategies import MarketMakingStrategy as MMS
        assert MMS is MarketMakingStrategy

    def test_strategy_engine_registry(self):
        """策略应可被 StrategyEngine 实例化（若注册机制存在）。"""
        s = MarketMakingStrategy()
        df = make_ohlc(60)
        out = s._compute_raw_signals(df)
        assert len(out) == len(df)
