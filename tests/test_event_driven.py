"""事件驱动策略单元测试。

覆盖：
    - 三种 K线事件检测（涨停 / 成交量异常 / 价格跳空）
    - 四种策略模式信号生成（pead / limit_up_reversal /
      limit_up_continuation / volume_spike）
    - 事件研究（CAR 曲线长度、t 统计量、p 值）
    - 与 BaseStrategy / BacktestEngine 插拔集成
    - 边界情况（空数据、窗口不足、外部事件预留接口）
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine
from strategies.base_strategy import BaseStrategy
from strategies.event_driven import (
    EXTERNAL_EVENT_TYPES,
    KLINE_EVENT_TYPES,
    SUPPORTED_MODES,
    EventDrivenStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 60, base_close: float = 100.0,
             base_vol: float = 1000.0) -> pd.DataFrame:
    """构造 n 个交易日的平整 OHLCV 数据（open=close=base_close）。"""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": [base_close] * n,
        "high": [base_close * 1.005] * n,
        "low": [base_close * 0.995] * n,
        "close": [base_close] * n,
        "volume": [base_vol] * n,
        "amount": [base_close * base_vol] * n,
    }, index=dates)


# ---------------------------------------------------------------------------
# 事件检测
# ---------------------------------------------------------------------------

class TestEventDetection:
    def test_detect_limit_up(self):
        """构造涨停日（收盘涨幅 10%），应被检测为 limit_up。"""
        df = _make_df(n=30)
        # 第 10 天涨停：前收 100，收 110
        df.iloc[10, df.columns.get_loc("close")] = 110.0
        df.iloc[10, df.columns.get_loc("high")] = 110.0

        strat = EventDrivenStrategy({"mode": "pead"})
        events = strat.detect_events(df, symbol="600519.SH")
        lu = [e for e in events if e["event_type"] == "limit_up"]
        assert len(lu) == 1
        assert lu[0]["date"] == df.index[10]
        assert lu[0]["symbol"] == "600519.SH"
        assert lu[0]["metadata"]["pct_change"] == pytest.approx(0.10, abs=1e-6)

    def test_detect_no_limit_up_below_threshold(self):
        """涨幅 5% 不应触发涨停（阈值 9.8%）。"""
        df = _make_df(n=30)
        df.iloc[10, df.columns.get_loc("close")] = 105.0  # +5%
        strat = EventDrivenStrategy()
        events = strat.detect_events(df)
        assert not any(e["event_type"] == "limit_up" for e in events)

    def test_detect_volume_spike(self):
        """构造成交量突增（5倍均量），应检测为 volume_spike。"""
        df = _make_df(n=40, base_vol=1000.0)
        df.iloc[25, df.columns.get_loc("volume")] = 6000.0  # 6倍
        strat = EventDrivenStrategy({"volume_spike_multiplier": 2.0})
        events = strat.detect_events(df)
        vs = [e for e in events if e["event_type"] == "volume_spike"]
        assert len(vs) >= 1
        assert vs[0]["date"] == df.index[25]
        assert vs[0]["metadata"]["volume_ratio"] > 2.0

    def test_detect_price_gap(self):
        """构造跳空高开 3%，应检测为 price_gap。"""
        df = _make_df(n=30)
        # 第 10 天：前收 100，开 103（高开后收平 100，避免次日连锁跳空）
        df.iloc[10, df.columns.get_loc("open")] = 103.0
        strat = EventDrivenStrategy({"price_gap_threshold": 0.02})
        events = strat.detect_events(df)
        pg = [e for e in events if e["event_type"] == "price_gap"]
        assert len(pg) == 1
        assert pg[0]["date"] == df.index[10]
        assert pg[0]["metadata"]["gap"] == pytest.approx(0.03, abs=1e-6)

    def test_detect_three_event_types(self):
        """同一份数据上应能检测出三种 K线事件。"""
        df = _make_df(n=40)
        # 涨停
        df.iloc[10, df.columns.get_loc("close")] = 110.0
        # 放量
        df.iloc[20, df.columns.get_loc("volume")] = 6000.0
        # 跳空
        df.iloc[30, df.columns.get_loc("open")] = 103.0
        df.iloc[30, df.columns.get_loc("close")] = 103.0

        strat = EventDrivenStrategy()
        events = strat.detect_events(df)
        types = {e["event_type"] for e in events}
        assert "limit_up" in types
        assert "volume_spike" in types
        assert "price_gap" in types
        # 至少 3 个事件
        assert len(events) >= 3
        # 事件按日期排序
        dates = [e["date"] for e in events]
        assert dates == sorted(dates)

    def test_kline_event_types_constant(self):
        """K线可检测事件类型常量完整。"""
        assert set(KLINE_EVENT_TYPES) == {"limit_up", "volume_spike",
                                          "price_gap"}
        assert set(EXTERNAL_EVENT_TYPES) == {"earnings", "dividend",
                                              "stock_split", "index_rebalance"}


# ---------------------------------------------------------------------------
# 策略模式信号
# ---------------------------------------------------------------------------

class TestStrategyModes:
    def test_pead_gap_up_buy(self):
        """PEAD：跳空上涨（gap>2% 且 close>open）应产生买入信号。"""
        df = _make_df(n=40)
        # 第 15 天跳空高开且收阳
        df.iloc[15, df.columns.get_loc("open")] = 103.0
        df.iloc[15, df.columns.get_loc("close")] = 104.0
        df.iloc[15, df.columns.get_loc("high")] = 104.5

        strat = EventDrivenStrategy({"mode": "pead", "hold_days": 5})
        raw = strat._compute_raw_signals(df)
        assert raw.iloc[15]["signal"] == 1
        assert raw.iloc[15]["confidence"] > 0
        # 5 个交易日后应产生卖出信号
        exit_pos = 15 + 5
        assert raw.iloc[exit_pos]["signal"] == -1

    def test_pead_no_signal_on_flat(self):
        """平整数据上 PEAD 不应产生信号。"""
        df = _make_df(n=40)
        strat = EventDrivenStrategy({"mode": "pead"})
        raw = strat._compute_raw_signals(df)
        assert (raw["signal"] == 0).all()

    def test_limit_up_reversal_sell(self):
        """涨停反转：涨停日应产生卖出信号 (-1)。"""
        df = _make_df(n=30)
        df.iloc[10, df.columns.get_loc("close")] = 110.0
        strat = EventDrivenStrategy({"mode": "limit_up_reversal"})
        raw = strat._compute_raw_signals(df)
        assert raw.iloc[10]["signal"] == -1
        assert raw.iloc[10]["confidence"] > 0

    def test_limit_up_continuation_buy(self):
        """涨停延续：涨停日应产生买入信号并在 hold_days 后卖出。"""
        df = _make_df(n=40)
        df.iloc[10, df.columns.get_loc("close")] = 110.0
        strat = EventDrivenStrategy({"mode": "limit_up_continuation",
                                    "hold_days": 3})
        raw = strat._compute_raw_signals(df)
        assert raw.iloc[10]["signal"] == 1
        assert raw.iloc[10 + 3]["signal"] == -1

    def test_volume_spike_buy_and_sell(self):
        """放量上涨买入，放量下跌卖出。"""
        df = _make_df(n=40, base_vol=1000.0)
        # 第 15 天：放量收阳
        df.iloc[15, df.columns.get_loc("volume")] = 6000.0
        df.iloc[15, df.columns.get_loc("close")] = 101.0
        df.iloc[15, df.columns.get_loc("open")] = 100.0
        # 第 25 天：放量收阴
        df.iloc[25, df.columns.get_loc("volume")] = 6000.0
        df.iloc[25, df.columns.get_loc("close")] = 99.0
        df.iloc[25, df.columns.get_loc("open")] = 100.0

        strat = EventDrivenStrategy({"mode": "volume_spike"})
        raw = strat._compute_raw_signals(df)
        assert raw.iloc[15]["signal"] == 1
        assert raw.iloc[25]["signal"] == -1

    def test_all_modes_generate_signals(self):
        """四种模式均可实例化并在事件数据上产生信号。"""
        df = _make_df(n=40)
        df.iloc[10, df.columns.get_loc("close")] = 110.0  # 涨停
        df.iloc[15, df.columns.get_loc("open")] = 103.0
        df.iloc[15, df.columns.get_loc("close")] = 104.0  # 跳空涨
        df.iloc[20, df.columns.get_loc("volume")] = 6000.0
        df.iloc[20, df.columns.get_loc("close")] = 101.0  # 放量收阳

        for mode in SUPPORTED_MODES:
            strat = EventDrivenStrategy({"mode": mode})
            raw = strat._compute_raw_signals(df)
            assert "signal" in raw.columns
            assert "confidence" in raw.columns
            # 至少有一个非零信号
            assert (raw["signal"] != 0).any(), f"模式 {mode} 未产生信号"

    def test_invalid_mode_raises(self):
        """未知模式应抛出 ValueError。"""
        with pytest.raises(ValueError):
            EventDrivenStrategy({"mode": "not_a_mode"})


# ---------------------------------------------------------------------------
# 事件研究
# ---------------------------------------------------------------------------

class TestEventStudy:
    def _make_event_df(self, n: int = 100, event_pos: int = 50,
                       drift: float = 0.01) -> pd.DataFrame:
        """构造在 event_pos 处有正漂移的数据。"""
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = np.full(n, 100.0)
        # 事件后持续上涨 drift
        for i in range(event_pos + 1, n):
            close[i] = close[i - 1] * (1 + drift)
        return pd.DataFrame({
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": [1000.0] * n,
            "amount": close * 1000.0,
        }, index=dates)

    def test_event_study_returns_car_and_tstat(self):
        """事件研究应返回 car_series / t_statistic / p_value。"""
        df = self._make_event_df()
        events = [{"date": df.index[50], "event_type": "limit_up",
                   "symbol": "X", "metadata": {}}]
        strat = EventDrivenStrategy()
        result = strat.event_study(df, events, window=20)

        assert result["event_count"] == 1
        assert isinstance(result["car_series"], pd.Series)
        assert not result["car_series"].empty
        # 单事件时 std 为 0，t 为 nan
        assert np.isnan(result["t_statistic"])
        assert np.isnan(result["p_value"])
        assert len(result["individual_cars"]) == 1

    def test_car_series_length(self):
        """CAR 曲线长度应为 2*window+1。"""
        df = self._make_event_df()
        events = [{"date": df.index[50], "event_type": "x"}]
        strat = EventDrivenStrategy()
        window = 10
        result = strat.event_study(df, events, window=window)
        assert len(result["car_series"]) == 2 * window + 1
        # index 偏移从 -window 到 +window
        assert list(result["car_series"].index) == \
            list(range(-window, window + 1))

    def test_event_study_multiple_events_tstat(self):
        """多事件时应计算出有限的 t 统计量。"""
        df = self._make_event_df(n=200, event_pos=50)
        # 多个事件点
        events = [
            {"date": df.index[50], "event_type": "x"},
            {"date": df.index[80], "event_type": "x"},
            {"date": df.index[110], "event_type": "x"},
        ]
        strat = EventDrivenStrategy()
        result = strat.event_study(df, events, window=10)
        assert result["event_count"] == 3
        assert len(result["car_series"]) == 21
        # 多事件且有方差时 t 统计量应为有限数
        assert not np.isnan(result["t_statistic"])
        assert not np.isnan(result["p_value"])

    def test_event_study_empty_events(self):
        """空事件列表应返回空结果而非异常。"""
        df = _make_df(n=30)
        strat = EventDrivenStrategy()
        result = strat.event_study(df, [], window=5)
        assert result["event_count"] == 0
        assert result["car_series"].empty

    def test_event_study_insufficient_window(self):
        """靠近数据边缘、窗口不足的事件应被剔除。"""
        df = self._make_event_df(n=60)
        # event_pos=2，前后 window=20 无法满足
        events = [{"date": df.index[2], "event_type": "x"}]
        strat = EventDrivenStrategy()
        result = strat.event_study(df, events, window=20)
        assert result["event_count"] == 0


# ---------------------------------------------------------------------------
# 基类集成 / 插拔
# ---------------------------------------------------------------------------

class TestBaseIntegration:
    def test_inherits_base_strategy(self):
        """EventDrivenStrategy 应是 BaseStrategy 子类。"""
        strat = EventDrivenStrategy()
        assert isinstance(strat, BaseStrategy)

    def test_shift_prevents_future_function(self):
        """generate_signals 应将信号 shift(1)，事件日信号次日才出现。"""
        df = _make_df(n=30)
        df.iloc[10, df.columns.get_loc("close")] = 110.0  # 涨停
        strat = EventDrivenStrategy({"mode": "limit_up_continuation"})
        signals = strat.generate_signals(df, symbol="600519.SH")
        # 事件日 index[10] 产生原始买入信号，shift 后出现在 index[11]
        buy_dates = [s.date for s in signals if s.action == "buy"]
        assert df.index[11] in buy_dates
        # 事件当天不应有信号（已 shift）
        assert df.index[10] not in buy_dates

    def test_backtest_engine_runs(self):
        """策略应可被 BacktestEngine 直接运行回测。"""
        df = _make_df(n=60)
        # 注入若干跳空事件
        for pos in (15, 30, 45):
            df.iloc[pos, df.columns.get_loc("open")] = 103.0
            df.iloc[pos, df.columns.get_loc("close")] = 104.0
            df.iloc[pos, df.columns.get_loc("high")] = 104.5
        strat = EventDrivenStrategy({"mode": "pead", "hold_days": 5})
        engine = BacktestEngine(initial_capital=1_000_000.0)
        result = engine.run(df, strat, symbol="600519.SH")
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
        assert "累计收益率" in result.metrics


# ---------------------------------------------------------------------------
# 外部事件 / 边界
# ---------------------------------------------------------------------------

class TestExternalAndEdge:
    def test_load_external_events_returns_list(self):
        """外部事件预留接口应返回列表（当前为空）。"""
        strat = EventDrivenStrategy()
        events = strat.load_external_events("600519.SH")
        assert isinstance(events, list)
        assert events == []

    def test_detect_events_empty_df(self):
        """空 DataFrame 应返回空事件列表。"""
        strat = EventDrivenStrategy()
        empty = pd.DataFrame(columns=["open", "high", "low", "close",
                                      "volume"])
        assert strat.detect_events(empty) == []
        assert strat.detect_events(pd.DataFrame()) == []

    def test_compute_signals_empty_df(self):
        """空 DataFrame 上 _compute_raw_signals 应返回带 signal 列的空表。"""
        strat = EventDrivenStrategy()
        empty = pd.DataFrame(columns=["open", "high", "low", "close",
                                     "volume"])
        out = strat._compute_raw_signals(empty)
        assert "signal" in out.columns
        assert "confidence" in out.columns
