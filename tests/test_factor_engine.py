"""多因子分析引擎单元测试。

覆盖：
  - 因子库数量与分类（>=15 个因子、7 大类）
  - 因子计算正确性（动量/波动率/技术/流动性）
  - mock 基本面数据稳定性
  - IC 分析统计量完整性
  - 分层回测结构与单调性
  - 因子暴露度 z-score 标准化
  - 边界情况（空 DataFrame / 数据不足）

全部使用确定性随机种子构造的模拟 K 线，不依赖网络与真实行情。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.factor_engine import FactorEngine


# ---------------------------------------------------------------------------
# 构造工具
# ---------------------------------------------------------------------------
def make_klines(
    n_days: int = 250,
    seed: int = 0,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """构造带 open/high/low/close/volume/amount 列的模拟日 K 线。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rets = rng.normal(0.0005, 0.02, size=n_days)
    close = start_price * np.cumprod(1.0 + rets)
    open_ = close * (1.0 + rng.normal(0, 0.005, size=n_days))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0, 0.01, size=n_days))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0, 0.01, size=n_days))
    volume = rng.uniform(1e6, 5e6, size=n_days)
    amount = volume * close
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "amount": amount,
        },
        index=dates,
    )


@pytest.fixture(scope="module")
def engine() -> FactorEngine:
    return FactorEngine()


@pytest.fixture(scope="module")
def klines() -> pd.DataFrame:
    return make_klines(n_days=250, seed=42)


# ---------------------------------------------------------------------------
# 1. 因子库元数据
# ---------------------------------------------------------------------------
def test_factor_list_count(engine: FactorEngine) -> None:
    """因子库至少 15 个因子，且分属 8 大类（含高频）。"""
    fl = engine.get_factor_list()
    assert len(fl) >= 15, f"因子数不足: {len(fl)}"
    categories = {f["category"] for f in fl}
    assert len(categories) == 8, f"分类数 != 8: {categories}"
    for f in fl:
        assert {"name", "category", "description", "direction"} <= set(f)


def test_factor_names_unique(engine: FactorEngine) -> None:
    names = engine.factor_names
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# 2. 因子计算
# ---------------------------------------------------------------------------
def test_calculate_factors_columns(engine: FactorEngine, klines: pd.DataFrame) -> None:
    """所有因子列都应被添加到输出 DataFrame。"""
    out = engine.calculate_factors(klines, symbol="600519.SH")
    for name in engine.factor_names:
        assert name in out.columns, f"缺少因子列: {name}"


def test_at_least_ten_factors_non_nan(engine: FactorEngine, klines: pd.DataFrame) -> None:
    """最新交易日至少 10 个因子有非 NaN 值。"""
    out = engine.calculate_factors(klines, symbol="600519.SH")
    latest = out[engine.factor_names].iloc[-1]
    valid = latest.dropna()
    assert len(valid) >= 10, f"非 NaN 因子仅 {len(valid)}: {valid.to_dict()}"


def test_momentum_20_manual(engine: FactorEngine, klines: pd.DataFrame) -> None:
    """momentum_20 应等于 close[t]/close[t-20]-1。"""
    out = engine.calculate_factors(klines, symbol="600519.SH")
    manual = klines["close"] / klines["close"].shift(20) - 1.0
    pd.testing.assert_series_equal(
        out["momentum_20"], manual, check_names=False
    )


def test_momentum_120_excl5_logic(engine: FactorEngine, klines: pd.DataFrame) -> None:
    """momentum_120_excl5 应等于 close[t-5]/close[t-125]-1。"""
    out = engine.calculate_factors(klines, symbol="600519.SH")
    manual = klines["close"].shift(5) / klines["close"].shift(125) - 1.0
    pd.testing.assert_series_equal(
        out["momentum_120_excl5"], manual, check_names=False
    )


def test_volatility_inverse_sign(engine: FactorEngine, klines: pd.DataFrame) -> None:
    """波动率倒数因子应与真实波动率负相关（波动越大因子越小）。"""
    out = engine.calculate_factors(klines, symbol="600519.SH")
    real_vol = klines["close"].pct_change().rolling(20).std()
    valid = out["volatility_20_inverse"].notna() & real_vol.notna()
    corr = np.corrcoef(out.loc[valid, "volatility_20_inverse"],
                       real_vol[valid])[0, 1]
    assert corr < -0.9, f"波动率倒数与波动率相关性应为负, got {corr}"


def test_rsi_computable(engine: FactorEngine, klines: pd.DataFrame) -> None:
    out = engine.calculate_factors(klines, symbol="600519.SH")
    rsi = out["rsi_14"].dropna()
    assert len(rsi) > 50
    assert ((rsi > 0) & (rsi < 100)).all()


def test_macd_hist_computable(engine: FactorEngine, klines: pd.DataFrame) -> None:
    out = engine.calculate_factors(klines, symbol="600519.SH")
    hist = out["macd_hist"].dropna()
    assert len(hist) > 100
    assert np.isfinite(hist.iloc[-1])


def test_bollinger_position_range(engine: FactorEngine, klines: pd.DataFrame) -> None:
    """布林带位置应落在 0~1 区间附近。"""
    out = engine.calculate_factors(klines, symbol="600519.SH")
    pos = out["bollinger_position"].dropna()
    assert len(pos) > 50
    assert pos.min() > -1.0 and pos.max() < 2.0  # 允许少量越界


def test_amount_log_manual(engine: FactorEngine, klines: pd.DataFrame) -> None:
    out = engine.calculate_factors(klines, symbol="600519.SH")
    manual = np.log(klines["amount"])
    pd.testing.assert_series_equal(out["amount_log"], manual, check_names=False)


# ---------------------------------------------------------------------------
# 3. mock 基本面数据
# ---------------------------------------------------------------------------
def test_mock_fundamentals_stable(engine: FactorEngine) -> None:
    """同一 symbol 两次生成的 mock 基本面应完全一致。"""
    a = engine._generate_mock_fundamentals("600519.SH")
    b = engine._generate_mock_fundamentals("600519.SH")
    assert a == b
    # 关键字段存在
    for k in ("pe_ttm", "pb", "ps", "roe", "gross_margin", "debt_ratio"):
        assert k in a and np.isfinite(a[k])


def test_mock_fundamentals_differ_across_symbols(engine: FactorEngine) -> None:
    """不同 symbol 的 mock 基本面应不同。"""
    a = engine._generate_mock_fundamentals("600519.SH")
    b = engine._generate_mock_fundamentals("AAPL.US")
    assert a != b


# ---------------------------------------------------------------------------
# 4. IC 分析
# ---------------------------------------------------------------------------
def _make_predictive_panel(
    engine: FactorEngine,
    n_symbols: int = 8,
    n_days: int = 300,
    forward_days: int = 5,
    seed: int = 1,
) -> dict:
    """构造因子值与未来收益强相关的面板（dict 形式）。

    每只股票一条随机游走收盘价；因子列 = 未来 forward_days 期收益 + 小噪声。
    """
    rng = np.random.RandomState(seed)
    panel: dict = {}
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    for i in range(n_symbols):
        rets = rng.normal(0.0005, 0.02, size=n_days)
        close = 100.0 * np.cumprod(1.0 + rets)
        close_s = pd.Series(close, index=dates, dtype=float)
        fwd_ret = close_s.shift(-forward_days) / close_s - 1.0
        # 因子 = 未来收益 + 小噪声，保证 IC 显著为正
        factor = fwd_ret + rng.normal(0, 0.002, size=n_days)
        df = pd.DataFrame({"momentum_20": factor, "close": close_s}, index=dates)
        panel[f"SYM{i:02d}"] = df
    return panel


def test_ic_analysis_keys(engine: FactorEngine) -> None:
    """IC 分析必须返回 ic_mean/ic_std/ic_ir/ic_win_rate/ic_series。"""
    panel = _make_predictive_panel(engine)
    result = engine.factor_ic_analysis(panel, "momentum_20", forward_days=5)
    for key in ("ic_mean", "ic_std", "ic_ir", "ic_win_rate", "ic_series"):
        assert key in result, f"缺少 IC 统计量: {key}"
    assert isinstance(result["ic_series"], pd.Series)


def test_ic_positive_when_factor_predictive(engine: FactorEngine) -> None:
    """因子与未来收益强相关时 IC 应为显著正值。"""
    panel = _make_predictive_panel(engine, seed=7)
    result = engine.factor_ic_analysis(panel, "momentum_20", forward_days=5)
    assert result["n_periods"] > 50
    assert result["ic_mean"] > 0.3, f"IC 均值过低: {result['ic_mean']}"
    assert result["ic_win_rate"] > 0.6


# ---------------------------------------------------------------------------
# 5. 分层回测
# ---------------------------------------------------------------------------
def test_layered_backtest_structure(engine: FactorEngine) -> None:
    """分层回测应返回 5 层收益 + 多空收益 + 单调性。"""
    panel = _make_predictive_panel(engine, n_symbols=10, seed=3)
    result = engine.layered_backtest(panel, "momentum_20",
                                     n_layers=5, forward_days=5)
    assert len(result["layer_returns"]) == 5
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        assert q in result["layer_returns"]
    assert "long_short_return" in result
    assert "monotonicity" in result


def test_layered_backtest_monotonic(engine: FactorEngine) -> None:
    """预测性因子下，Q5（高分位）收益应高于 Q1，多空收益为正。"""
    panel = _make_predictive_panel(engine, n_symbols=10, seed=11)
    result = engine.layered_backtest(panel, "momentum_20",
                                     n_layers=5, forward_days=5)
    q1 = result["layer_returns"]["Q1"]
    q5 = result["layer_returns"]["Q5"]
    assert q5 > q1, f"Q5={q5} 应高于 Q1={q1}"
    assert result["long_short_return"] > 0


# ---------------------------------------------------------------------------
# 6. 因子暴露度 z-score
# ---------------------------------------------------------------------------
def test_factor_exposure_zscore(engine: FactorEngine, klines: pd.DataFrame) -> None:
    """时序 z-score 标准化后均值≈0、标准差≈1；暴露度返回最新值。"""
    exposure = engine.factor_exposure(klines, symbol="600519.SH")
    assert len(exposure) == len(engine.factor_names)
    # 对动量因子手工验证 z-score 性质
    out = engine.calculate_factors(klines, symbol="600519.SH")
    z = engine._zscore(out["momentum_20"].dropna())
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=0) - 1.0) < 1e-9
    # 最新暴露度应是有限值
    assert np.isfinite(exposure["momentum_20"])


# ---------------------------------------------------------------------------
# 7. 面板构建
# ---------------------------------------------------------------------------
def test_build_factor_panel_shape(engine: FactorEngine) -> None:
    symbol_data = {
        "A": make_klines(200, seed=1),
        "B": make_klines(200, seed=2),
        "C": make_klines(200, seed=3),
    }
    panel = engine.build_factor_panel(symbol_data, "momentum_20")
    assert list(panel.columns) == ["A", "B", "C"]
    assert len(panel) == 200
    assert panel.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# 8. 边界情况
# ---------------------------------------------------------------------------
def test_empty_dataframe(engine: FactorEngine) -> None:
    """空 DataFrame 不应报错，应原样返回。"""
    empty = pd.DataFrame(columns=["open", "high", "low", "close",
                                  "volume", "amount"])
    out = engine.calculate_factors(empty, symbol="X.SH")
    assert out.empty
    exposure = engine.factor_exposure(empty, symbol="X.SH")
    assert len(exposure) == len(engine.factor_names)


def test_insufficient_data_no_crash(engine: FactorEngine) -> None:
    """K 线不足 30 根时不应抛异常，价量因子为 NaN 即可。"""
    short = make_klines(n_days=10, seed=0)
    out = engine.calculate_factors(short, symbol="X.SH")
    assert "close" in out.columns
    # 基本面 mock 列仍应有值
    assert np.isfinite(out["pe_inverse"].iloc[-1])


def test_ic_analysis_empty_panel(engine: FactorEngine) -> None:
    """空面板 IC 分析返回 NaN 统计量而非异常。"""
    result = engine.factor_ic_analysis({}, "momentum_20")
    assert result["n_periods"] == 0
    assert np.isnan(result["ic_mean"])


def test_layered_backtest_empty_panel(engine: FactorEngine) -> None:
    result = engine.layered_backtest({}, "momentum_20", n_layers=5)
    assert result["n_periods"] == 0
    assert len(result["layer_returns"]) == 5
