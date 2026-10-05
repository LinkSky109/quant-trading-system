"""策略引擎单元测试。

覆盖: 基类shift(1)防未来函数 / 双均线金叉死叉 / 布林带触及轨道+量过滤 / 动量突破
关键验证: 信号产生日与执行日分离，杜绝未来函数
"""
from __future__ import annotations

import pandas as pd
import pytest

from strategies.base_strategy import BaseStrategy, Signal
from strategies.bollinger import BollingerStrategy
from strategies.grid_trading import GridTradingStrategy
from strategies.ma_cross import MACrossStrategy
from strategies.macd import MACDStrategy
from strategies.momentum_breakout import MomentumBreakoutStrategy
from strategies.rsi import RSIStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(prices: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    """构造OHLCV测试数据，high=close*1.01, low=close*0.99。"""
    n = len(prices)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    vol = volumes if volumes is not None else [1_000_000] * n
    return pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": vol,
    }, index=dates)


# ---------------------------------------------------------------------------
# BaseStrategy / Signal
# ---------------------------------------------------------------------------

class TestSignal:
    def test_signal_dataclass_fields(self):
        sig = Signal(
            date=pd.Timestamp("2024-01-01"),
            symbol="600519.SH",
            strategy="test",
            action="buy",
            confidence=0.8,
            price=100.0,
        )
        assert sig.action == "buy"
        assert sig.confidence == 0.8
        assert sig.metadata == {}

    def test_signal_action_values(self):
        for action in ("buy", "sell", "hold"):
            sig = Signal(date=pd.Timestamp.now(), symbol="X", strategy="t",
                         action=action, confidence=0.5, price=1.0)
            assert sig.action == action


class TestBaseStrategyShift:
    """验证 shift(1) 杜绝未来函数：T日产生的信号，T+1日才可见。"""

    def test_signal_is_shifted(self):
        """构造一个在第5天产生金叉的序列，验证信号出现在第6天而非第5天。"""
        # 前4天横盘，第5天快速上涨触发金叉(fast=2, slow=4)
        prices = [100, 100, 100, 100, 110, 110]
        df = _make_df(prices)
        strategy = MACrossStrategy({"fast_period": 2, "slow_period": 4})
        signals = strategy.generate_signals(df, symbol="TEST")

        # 金叉在第5天(索引4)产生，shift后信号出现在第6天(索引5)
        assert len(signals) >= 1
        assert signals[0].date == df.index[5]

    def test_first_row_never_has_signal(self):
        """第一行永远不会有信号（shift后为NaN）。"""
        prices = list(range(100, 120))
        df = _make_df(prices)
        strategy = MACrossStrategy({"fast_period": 2, "slow_period": 5})
        signals = strategy.generate_signals(df, symbol="TEST")
        for sig in signals:
            assert sig.date != df.index[0]

    def test_hold_not_in_signals(self):
        """signal=0的行不出现在信号列表中。"""
        prices = [100] * 30  # 完全横盘，无金叉死叉
        df = _make_df(prices)
        strategy = MACrossStrategy({"fast_period": 2, "slow_period": 5})
        signals = strategy.generate_signals(df, symbol="TEST")
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# MACrossStrategy
# ---------------------------------------------------------------------------

class TestMACross:
    def test_golden_cross_generates_buy(self):
        # 下跌后上涨，fast从下方穿越slow
        prices = [110, 108, 106, 104, 102, 100, 102, 105, 108, 112]
        df = _make_df(prices)
        strategy = MACrossStrategy({"fast_period": 2, "slow_period": 5})
        signals = strategy.generate_signals(df, symbol="TEST")
        buy_signals = [s for s in signals if s.action == "buy"]
        assert len(buy_signals) >= 1

    def test_death_cross_generates_sell(self):
        # 上涨后下跌，fast从上方穿越slow
        prices = [100, 102, 104, 106, 108, 110, 108, 105, 102, 98]
        df = _make_df(prices)
        strategy = MACrossStrategy({"fast_period": 2, "slow_period": 5})
        signals = strategy.generate_signals(df, symbol="TEST")
        sell_signals = [s for s in signals if s.action == "sell"]
        assert len(sell_signals) >= 1

    def test_custom_parameters(self):
        strategy = MACrossStrategy({"fast_period": 10, "slow_period": 30})
        assert strategy.fast_period == 10
        assert strategy.slow_period == 30

    def test_default_parameters(self):
        strategy = MACrossStrategy()
        assert strategy.fast_period == 5
        assert strategy.slow_period == 20

    def test_confidence_between_0_and_1(self):
        # 需要60+数据点使rolling(60).max()有效
        import numpy as np
        base = np.linspace(100, 120, 70).tolist()
        df = _make_df(base)
        strategy = MACrossStrategy({"fast_period": 5, "slow_period": 20})
        signals = strategy.generate_signals(df, symbol="TEST")
        for sig in signals:
            assert 0.0 <= sig.confidence <= 1.0 or pd.isna(sig.confidence)

    def test_insufficient_data_no_signals(self):
        """数据少于slow_period时不产生信号。"""
        df = _make_df([100, 101, 102])
        strategy = MACrossStrategy({"fast_period": 5, "slow_period": 20})
        signals = strategy.generate_signals(df, symbol="TEST")
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# BollingerStrategy
# ---------------------------------------------------------------------------

class TestBollinger:
    def test_touch_lower_band_buy(self):
        """价格大跌触及下轨+放量→买入。"""
        # 横盘后暴跌
        prices = [100] * 10 + [85, 86]
        volumes = [1_000_000] * 10 + [5_000_000, 5_000_000]
        df = _make_df(prices, volumes)
        strategy = BollingerStrategy({"period": 5, "num_std": 1.5,
                                      "volume_filter_period": 5, "vol_multiplier": 1.0})
        signals = strategy.generate_signals(df, symbol="TEST")
        buy_signals = [s for s in signals if s.action == "buy"]
        assert len(buy_signals) >= 1

    def test_touch_upper_band_sell(self):
        """价格大涨触及上轨+放量→卖出。"""
        prices = [100] * 10 + [115, 116]
        volumes = [1_000_000] * 10 + [5_000_000, 5_000_000]
        df = _make_df(prices, volumes)
        strategy = BollingerStrategy({"period": 5, "num_std": 1.5,
                                      "volume_filter_period": 5, "vol_multiplier": 1.0})
        signals = strategy.generate_signals(df, symbol="TEST")
        sell_signals = [s for s in signals if s.action == "sell"]
        assert len(sell_signals) >= 1

    def test_volume_filter_blocks_signal(self):
        """触及轨道但成交量不足→不产生信号。"""
        prices = [100] * 10 + [85, 86]
        volumes = [1_000_000] * 12  # 放量日也只有100万，不满足>均量
        df = _make_df(prices, volumes)
        strategy = BollingerStrategy({"period": 5, "num_std": 1.5,
                                      "volume_filter_period": 5, "vol_multiplier": 2.0})
        signals = strategy.generate_signals(df, symbol="TEST")
        # 量比=1.0 < multiplier=2.0，不应有信号
        assert len(signals) == 0

    def test_default_parameters(self):
        strategy = BollingerStrategy()
        assert strategy.period == 20
        assert strategy.num_std == 2.0

    def test_signal_shifted(self):
        """布林带信号也经过shift(1)。"""
        prices = [100] * 10 + [85, 86]
        volumes = [1_000_000] * 10 + [5_000_000, 5_000_000]
        df = _make_df(prices, volumes)
        strategy = BollingerStrategy({"period": 5, "num_std": 1.5})
        signals = strategy.generate_signals(df, symbol="TEST")
        for sig in signals:
            assert sig.date != df.index[0]


# ---------------------------------------------------------------------------
# MomentumBreakoutStrategy
# ---------------------------------------------------------------------------

class TestMomentumBreakout:
    def test_breakout_high_buy(self):
        """收盘价突破过去N日高点→买入。"""
        # 前5天横盘100，第6天突破
        prices = [100, 100, 100, 100, 100, 105, 106]
        df = _make_df(prices)
        strategy = MomentumBreakoutStrategy({"breakout_period": 3, "breakdown_period": 2})
        signals = strategy.generate_signals(df, symbol="TEST")
        buy_signals = [s for s in signals if s.action == "buy"]
        assert len(buy_signals) >= 1

    def test_breakdown_low_sell(self):
        """收盘价跌破过去M日低点→卖出。"""
        prices = [100, 100, 100, 100, 100, 95, 94]
        df = _make_df(prices)
        strategy = MomentumBreakoutStrategy({"breakout_period": 3, "breakdown_period": 2})
        signals = strategy.generate_signals(df, symbol="TEST")
        sell_signals = [s for s in signals if s.action == "sell"]
        assert len(sell_signals) >= 1

    def test_no_lookahead_in_highest(self):
        """highest内部已shift(1)，不含当日高点。"""
        # 第6天close=105，prev_high是前3天最高=100，突破
        # 如果含当日，prev_high会包含105就不会突破
        prices = [100, 100, 100, 100, 100, 105]
        df = _make_df(prices)
        strategy = MomentumBreakoutStrategy({"breakout_period": 3, "breakdown_period": 2})
        raw = strategy._compute_raw_signals(df)
        # 第6天(索引5)的prev_high应该=101（前3天high=100*1.01），不含当日
        assert raw["prev_high"].iloc[5] == pytest.approx(101.0)

    def test_default_parameters(self):
        strategy = MomentumBreakoutStrategy()
        assert strategy.breakout_period == 20
        assert strategy.breakdown_period == 10

    def test_confidence_scales_with_breakout(self):
        """突破幅度越大，置信度越高。"""
        prices_small = [100] * 5 + [101]  # 小突破
        prices_large = [100] * 5 + [110]  # 大突破
        strat = MomentumBreakoutStrategy({"breakout_period": 3, "breakdown_period": 2})

        raw_small = strat._compute_raw_signals(_make_df(prices_small))
        raw_large = strat._compute_raw_signals(_make_df(prices_large))

        conf_small = raw_small["confidence"].iloc[-1]
        conf_large = raw_large["confidence"].iloc[-1]
        assert conf_large >= conf_small


# ---------------------------------------------------------------------------
# get_signal_dataframe
# ---------------------------------------------------------------------------

class TestGetSignalDataframe:
    def test_returns_dataframe_with_signal_columns(self):
        df = _make_df([100, 100, 100, 100, 100, 105, 106])
        strategy = MomentumBreakoutStrategy({"breakout_period": 3, "breakdown_period": 2})
        result = strategy.get_signal_dataframe(df, symbol="TEST")
        assert "signal" in result.columns
        assert "confidence" in result.columns
        assert len(result) == len(df)

    def test_signal_column_is_shifted(self):
        df = _make_df([100] * 5 + [105, 106])
        strategy = MomentumBreakoutStrategy({"breakout_period": 3, "breakdown_period": 2})
        result = strategy.get_signal_dataframe(df)
        # 第一行signal应为NaN（shift后）
        assert pd.isna(result["signal"].iloc[0])


# ---------------------------------------------------------------------------
# RSIStrategy
# ---------------------------------------------------------------------------

class TestRSI:
    def test_oversold_generates_buy(self):
        """持续下跌后 RSI 跌破超卖阈值 → 买入。"""
        prices = [100] * 15 + [95, 90, 85, 80]
        df = _make_df(prices)
        strategy = RSIStrategy({"period": 2, "oversold_threshold": 30,
                                "overbought_threshold": 70})
        signals = strategy.generate_signals(df, symbol="TEST")
        buy = [s for s in signals if s.action == "buy"]
        assert len(buy) >= 1

    def test_overbought_generates_sell(self):
        """强势上涨中 RSI 突破超买阈值 → 卖出。"""
        # 上涨途中夹杂极小回撤，使 avg_loss 非零以避免 RSI 为 NaN
        prices = [100, 101, 102, 101, 103, 104, 105, 106, 107, 108,
                  107, 109, 110, 111, 112, 113, 114, 115]
        df = _make_df(prices)
        strategy = RSIStrategy({"period": 2, "oversold_threshold": 30,
                                "overbought_threshold": 70})
        signals = strategy.generate_signals(df, symbol="TEST")
        sell = [s for s in signals if s.action == "sell"]
        assert len(sell) >= 1

    def test_signal_shifted_no_lookahead(self):
        """信号经基类 shift(1)，第一行不会有信号。"""
        prices = [100] * 15 + [90, 85, 80]
        df = _make_df(prices)
        strategy = RSIStrategy({"period": 2})
        signals = strategy.generate_signals(df, symbol="TEST")
        for sig in signals:
            assert sig.date != df.index[0]

    def test_insufficient_data_no_signals(self):
        """数据少于 period 时 RSI 为 NaN，不产生信号。"""
        df = _make_df([100, 99, 98])
        strategy = RSIStrategy({"period": 14})
        signals = strategy.generate_signals(df, symbol="TEST")
        assert len(signals) == 0

    def test_default_parameters(self):
        strategy = RSIStrategy()
        assert strategy.period == 14
        assert strategy.oversold_threshold == 30.0
        assert strategy.overbought_threshold == 70.0

    def test_confidence_in_range(self):
        prices = [100] * 15 + [90, 85, 80]
        df = _make_df(prices)
        strategy = RSIStrategy({"period": 2})
        signals = strategy.generate_signals(df, symbol="TEST")
        for sig in signals:
            assert 0.0 <= sig.confidence <= 1.0


# ---------------------------------------------------------------------------
# MACDStrategy
# ---------------------------------------------------------------------------

class TestMACD:
    def test_golden_cross_generates_buy(self):
        """横盘后急涨，DIF 上穿 DEA → 买入。"""
        prices = [100] * 9 + [105, 110, 115, 120]
        df = _make_df(prices)
        strategy = MACDStrategy({"fast": 3, "slow": 8, "signal": 3,
                                 "hist_filter_period": 3})
        signals = strategy.generate_signals(df, symbol="TEST")
        buy = [s for s in signals if s.action == "buy"]
        assert len(buy) >= 1

    def test_death_cross_generates_sell(self):
        """横盘后急跌，DIF 下穿 DEA → 卖出。"""
        prices = [100] * 9 + [95, 90, 85, 80]
        df = _make_df(prices)
        strategy = MACDStrategy({"fast": 3, "slow": 8, "signal": 3,
                                 "hist_filter_period": 3})
        signals = strategy.generate_signals(df, symbol="TEST")
        sell = [s for s in signals if s.action == "sell"]
        assert len(sell) >= 1

    def test_signal_shifted_no_lookahead(self):
        """信号经基类 shift(1)，第一行不会有信号。"""
        prices = [100] * 9 + [105, 110, 115, 120]
        df = _make_df(prices)
        strategy = MACDStrategy({"fast": 3, "slow": 8, "signal": 3})
        signals = strategy.generate_signals(df, symbol="TEST")
        for sig in signals:
            assert sig.date != df.index[0]

    def test_insufficient_data_no_signals(self):
        """数据不足 slow 窗口时不产生交叉信号。"""
        df = _make_df([100, 101, 102])
        strategy = MACDStrategy({"fast": 12, "slow": 26, "signal": 9})
        signals = strategy.generate_signals(df, symbol="TEST")
        assert len(signals) == 0

    def test_default_parameters(self):
        strategy = MACDStrategy()
        assert strategy.fast == 12
        assert strategy.slow == 26
        assert strategy.signal == 9


# ---------------------------------------------------------------------------
# GridTradingStrategy
# ---------------------------------------------------------------------------

class TestGridTrading:
    def test_cross_down_buy_cross_up_sell(self):
        """价格震荡穿越网格线：向下破网格买入，向上突破卖出。"""
        # min=98, max=104, grid_count=10 → step=0.6
        prices = [100, 102, 104, 102, 100, 98, 100, 102, 104]
        df = _make_df(prices)
        strategy = GridTradingStrategy({"grid_count": 10})
        signals = strategy.generate_signals(df, symbol="TEST")
        buy = [s for s in signals if s.action == "buy"]
        sell = [s for s in signals if s.action == "sell"]
        assert len(buy) >= 1
        assert len(sell) >= 1

    def test_signal_shifted_no_lookahead(self):
        """第一行无前一日网格信息，shift 后更不会有信号。"""
        prices = [100, 102, 100, 98, 100, 102]
        df = _make_df(prices)
        strategy = GridTradingStrategy({"grid_count": 5})
        signals = strategy.generate_signals(df, symbol="TEST")
        for sig in signals:
            assert sig.date != df.index[0]

    def test_empty_df_no_signals(self):
        """空数据不崩溃、不产生信号。"""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        strategy = GridTradingStrategy({"grid_count": 10})
        signals = strategy.generate_signals(df, symbol="TEST")
        assert len(signals) == 0

    def test_invalid_bounds_no_signals(self):
        """上界<=下界时不产生信号。"""
        df = _make_df([100, 102, 100, 98])
        strategy = GridTradingStrategy({"grid_count": 10,
                                        "upper_bound": 100.0,
                                        "lower_bound": 100.0})
        signals = strategy.generate_signals(df, symbol="TEST")
        assert len(signals) == 0

    def test_default_parameters(self):
        strategy = GridTradingStrategy()
        assert strategy.grid_count == 10
        assert strategy.upper_bound is None
        assert strategy.lower_bound is None
        assert strategy.fixed_confidence == 0.6
