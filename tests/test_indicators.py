"""专业技术指标库单元测试。

用确定性合成 OHLCV 数据验证各指标的计算正确性、NaN 处理与策略可用性。
运行：python -m pytest tests/test_indicators.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators import (
    TechnicalIndicators,
    adx,
    atr,
    atr_ratio,
    bollinger_bandwidth,
    cci,
    cmf,
    detect_cross,
    detect_ma_alignment,
    historical_volatility,
    ichimoku,
    kdj,
    mfi,
    mom,
    obv,
    roc,
    sar,
    vwap,
    wr,
)
from strategies.indicator_combo import IndicatorComboStrategy


# ---------------------------------------------------------------------------
#  fixtures
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 80, seed: int = 42, trend: float = 0.0) -> pd.DataFrame:
    """生成确定性合成 OHLCV 数据。

    Args:
        n: K 线数量。
        seed: 随机种子。
        trend: 线性漂移（正=上涨趋势，负=下跌趋势）。
    """
    rng = np.random.default_rng(seed)
    close = 100.0 + trend * np.arange(n) + np.cumsum(rng.standard_normal(n) * 0.5)
    high = close + np.abs(rng.standard_normal(n)) * 0.3 + 0.05
    low = close - np.abs(rng.standard_normal(n)) * 0.3 - 0.05
    open_ = (high + low) / 2.0
    volume = rng.integers(1000, 10000, size=n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def df() -> pd.DataFrame:
    return _make_ohlcv()


# ---------------------------------------------------------------------------
# 趋势类
# ---------------------------------------------------------------------------


class TestTrendIndicators:
    def test_atr_nan_warmup_and_positive(self, df: pd.DataFrame):
        result = atr(df["high"], df["low"], df["close"], period=14)
        # Wilder 平滑 min_periods=14：前 13 个为 NaN
        assert result.iloc[:13].isna().all()
        assert result.iloc[13:].notna().all()
        assert (result.dropna() > 0).all()
        # 与手算第一根平滑值对比：第 13 根（index=13）的 ATR = mean(TR[0..13])
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        expected_first = tr.iloc[:14].mean()  # 首值为简单均值（ewm 初值即首值平均）
        assert result.iloc[13] == pytest.approx(expected_first, rel=1e-6)

    def test_adx_columns_and_range(self, df: pd.DataFrame):
        result = adx(df["high"], df["low"], df["close"], period=14)
        assert list(result.columns) == ["adx", "plus_di", "minus_di"]
        tail = result.dropna()
        assert len(tail) > 10
        assert (tail["adx"] >= 0).all() and (tail["adx"] <= 100).all()
        assert (tail["plus_di"] >= 0).all() and (tail["minus_di"] >= 0).all()

    def test_dmi_shares_adx_result(self, df: pd.DataFrame):
        a = adx(df["high"], df["low"], df["close"], period=14)
        from indicators.technical import dmi
        d = dmi(df["high"], df["low"], df["close"], period=14)
        pd.testing.assert_series_equal(a["adx"], d["adx"])
        pd.testing.assert_series_equal(a["plus_di"], d["plus_di"])
        pd.testing.assert_series_equal(a["minus_di"], d["minus_di"])

    def test_ichimoku_columns_and_shifts(self, df: pd.DataFrame):
        result = ichimoku(df["high"], df["low"], df["close"])
        expected_cols = {"tenkan_sen", "kijun_sen", "senkou_a", "senkou_b", "chikou"}
        assert expected_cols.issubset(set(result.columns))
        # 先行带向前投影 26 期（历史对齐口径）：头部 26 个为 NaN
        assert result["senkou_a"].iloc[:26].isna().all()
        # 迟行带向后平移 26 期：末尾 26 个为 NaN
        assert result["chikou"].iloc[-26:].isna().all()

    def test_sar_finite(self, df: pd.DataFrame):
        result = sar(df["high"], df["low"])
        assert result.iloc[1:].notna().all()  # index 0 之后全部有值
        assert np.isfinite(result.iloc[1:].values).all()


# ---------------------------------------------------------------------------
# 震荡类
# ---------------------------------------------------------------------------


class TestOscillatorIndicators:
    def test_kdj_columns_and_range(self, df: pd.DataFrame):
        result = kdj(df["high"], df["low"], df["close"])
        assert list(result.columns) == ["k", "d", "j"]
        tail = result.dropna()
        assert len(tail) > 10
        assert (tail["k"] >= 0).all() and (tail["k"] <= 100).all()
        assert (tail["d"] >= 0).all() and (tail["d"] <= 100).all()

    def test_cci_known_value(self, df: pd.DataFrame):
        period = 20
        result = cci(df["high"], df["low"], df["close"], period=period)
        assert result.iloc[: period - 1].isna().all()
        # 手算最后一根
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        window = tp.iloc[-period:]
        ma = window.mean()
        mad = (window - ma).abs().mean()
        expected = (tp.iloc[-1] - ma) / (0.015 * mad)
        assert result.iloc[-1] == pytest.approx(expected, rel=1e-6)

    def test_wr_range(self, df: pd.DataFrame):
        result = wr(df["high"], df["low"], df["close"], period=14)
        tail = result.dropna()
        assert (tail >= -100).all() and (tail <= 0).all()

    def test_roc_known_value(self, df: pd.DataFrame):
        period = 12
        result = roc(df["close"], period=period)
        assert result.iloc[:period].isna().all()
        i = 30
        expected = (df["close"].iloc[i] - df["close"].iloc[i - period]) / df["close"].iloc[i - period] * 100
        assert result.iloc[i] == pytest.approx(expected, rel=1e-9)

    def test_mom_known_value(self, df: pd.DataFrame):
        period = 10
        result = mom(df["close"], period=period)
        assert result.iloc[:period].isna().all()
        i = 25
        expected = df["close"].iloc[i] - df["close"].iloc[i - period]
        assert result.iloc[i] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# 成交量类
# ---------------------------------------------------------------------------


class TestVolumeIndicators:
    def test_obv_manual_cumsum(self, df: pd.DataFrame):
        result = obv(df["close"], df["volume"])
        direction = np.sign(df["close"].diff()).fillna(0.0)
        expected = (direction * df["volume"]).cumsum()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_vwap_first_equals_tp(self, df: pd.DataFrame):
        result = vwap(df["high"], df["low"], df["close"], df["volume"])
        tp0 = (df["high"].iloc[0] + df["low"].iloc[0] + df["close"].iloc[0]) / 3.0
        # 累计 VWAP 第一根 = 当日 TP
        assert result.iloc[0] == pytest.approx(tp0, rel=1e-9)
        # 后续值有限且非负
        assert np.isfinite(result.iloc[1:].values).all()

    def test_mfi_range(self, df: pd.DataFrame):
        result = mfi(df["high"], df["low"], df["close"], df["volume"], period=14)
        tail = result.dropna()
        assert (tail >= 0).all() and (tail <= 100).all()

    def test_cmf_range(self, df: pd.DataFrame):
        result = cmf(df["high"], df["low"], df["close"], df["volume"], period=20)
        tail = result.dropna()
        assert (tail >= -1).all() and (tail <= 1).all()


# ---------------------------------------------------------------------------
# 波动率类
# ---------------------------------------------------------------------------


class TestVolatilityIndicators:
    def test_bollinger_bandwidth_positive(self, df: pd.DataFrame):
        result = bollinger_bandwidth(df["close"], period=20, num_std=2.0)
        tail = result.dropna()
        assert (tail > 0).all()

    def test_atr_ratio_equals_atr_over_close(self, df: pd.DataFrame):
        period = 14
        expected = atr(df["high"], df["low"], df["close"], period) / df["close"]
        result = atr_ratio(df["high"], df["low"], df["close"], period=period)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_historical_volatility_positive(self, df: pd.DataFrame):
        result = historical_volatility(df["close"], period=20)
        tail = result.dropna()
        assert (tail > 0).all()
        # 年化值 ≈ 日波动 * sqrt(252)
        daily_std = df["close"].pct_change().rolling(20).std().iloc[-1]
        assert result.iloc[-1] == pytest.approx(daily_std * np.sqrt(252), rel=1e-6)


# ---------------------------------------------------------------------------
# 形态类
# ---------------------------------------------------------------------------


class TestPatternIndicators:
    def test_detect_cross_known_series(self):
        fast = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0, 2.0])
        slow = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        result = detect_cross(fast, slow)
        assert result.iloc[0] == 0          # 初始无交叉
        assert result.iloc[2] == 1           # 金叉（上穿）
        assert result.iloc[4] == -1         # 死叉（下穿）
        assert result.iloc[[1, 3, 5]].eq(0).all()

    def test_detect_ma_alignment_bull(self):
        n = 80
        close = pd.Series(np.arange(10, 10 + n, dtype=float))  # 单调上涨
        result = detect_ma_alignment(close, periods=[5, 10, 20, 60])
        assert result.iloc[-1] == 1        # 多头排列
        assert result.iloc[:59].isna().any()  #  warmup 期存在 NaN

    def test_detect_ma_alignment_bear(self):
        n = 80
        close = pd.Series(np.arange(70, 70 - n, -1.0))  # 单调下跌
        result = detect_ma_alignment(close, periods=[5, 10, 20, 60])
        assert result.iloc[-1] == -1       # 空头排列


# ---------------------------------------------------------------------------
# 门面类
# ---------------------------------------------------------------------------


class TestTechnicalIndicatorsFacade:
    def test_list_indicators_at_least_15(self):
        ti = TechnicalIndicators()
        items = ti.list_indicators()
        assert len(items) >= 15
        names = {i["name"] for i in items}
        for required in ("atr", "adx", "kdj", "obv", "vwap", "cmf", "detect_cross"):
            assert required in names
        # 每条目包含分类与参数说明
        for item in items:
            assert {"name", "category", "description", "params", "output"} <= set(item.keys())

    def test_calculate_unknown_raises(self, df: pd.DataFrame):
        ti = TechnicalIndicators()
        with pytest.raises(KeyError):
            ti.calculate(df, "no_such_indicator")

    def test_calculate_dispatch(self, df: pd.DataFrame):
        ti = TechnicalIndicators()
        series_result = ti.calculate(df, "atr")
        assert isinstance(series_result, pd.Series)
        df_result = ti.calculate(df, "adx")
        assert isinstance(df_result, pd.DataFrame)
        assert "adx" in df_result.columns

    def test_calculate_with_override_params(self, df: pd.DataFrame):
        ti = TechnicalIndicators()
        result = ti.calculate(df, "roc", period=6)
        assert result.iloc[:6].isna().all()

    def test_calculate_all_merges_columns(self, df: pd.DataFrame):
        ti = TechnicalIndicators()
        out = ti.calculate_all(df)
        # 原列保留
        for col in ("open", "high", "low", "close", "volume"):
            assert col in out.columns
        # 多列指标以 <name>_<col> 前缀合并
        assert "adx_adx" in out.columns
        assert "kdj_k" in out.columns
        assert "obv" in out.columns


# ---------------------------------------------------------------------------
# 策略
# ---------------------------------------------------------------------------


class TestIndicatorComboStrategy:
    def _make_trending_df(self) -> pd.DataFrame:
        # 构造明显趋势 + 震荡交替的数据，确保信号可产生
        up = _make_ohlcv(n=60, seed=1, trend=0.5)
        down = _make_ohlcv(n=60, seed=2, trend=-0.5)
        df = pd.concat([up, down])
        df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")
        return df

    def test_strategy_signal_columns(self):
        strat = IndicatorComboStrategy()
        df = self._make_trending_df()
        sig_df = strat.get_signal_dataframe(df, symbol="TEST")
        assert "signal" in sig_df.columns
        assert "confidence" in sig_df.columns
        assert "adx" in sig_df.columns
        assert "macd_dif" in sig_df.columns
        assert "rsi" in sig_df.columns
        # signal ∈ {-1, 0, 1}（NaN 除外）
        valid = sig_df["signal"].dropna()
        assert set(unique_vals := set(valid.unique())) <= {-1, 0, 1}

    def test_strategy_generates_signal_objects(self):
        strat = IndicatorComboStrategy()
        df = self._make_trending_df()
        signals = strat.generate_signals(df, symbol="TEST")
        # 不要求一定有交易，但方法必须正常返回 Signal 列表
        assert isinstance(signals, list)

    def test_strategy_runs_backtest_engine(self):
        from backtest.engine import BacktestEngine
        strat = IndicatorComboStrategy()
        df = self._make_trending_df()
        engine = BacktestEngine(initial_capital=1_000_000.0)
        result = engine.run(df, strat, symbol="TEST")
        assert result.equity_curve is not None
        assert len(result.equity_curve) == len(df)
