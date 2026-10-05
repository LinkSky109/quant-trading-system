"""另类数据模块测试（REQ-P3-04）。

覆盖：
  - Mock 数据源确定性（同参数同结果）与值域约束
  - 真实数据源预留接口行为
  - 另类因子计算（含 shift(1) 防未来函数验证）
  - 与 FactorEngine 注册表 / calculate_factors 集成
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.alternative import (
    fetch_all,
    ALTERNATIVE_FACTOR_NAMES,
    DEFAULT_SOURCES,
    AlternativeFactorEngine,
    ESGScoreGenerator,
    RealAlternativeDataSource,
    SatelliteDataGenerator,
    SocialSentimentGenerator,
    SupplyChainGenerator,
    get_source,
)


# ---------------------------------------------------------------------------
# Mock 数据源：确定性与值域
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,gen", [
    ("satellite", SatelliteDataGenerator()),
    ("social", SocialSentimentGenerator()),
    ("supply_chain", SupplyChainGenerator()),
    ("esg", ESGScoreGenerator()),
])
class TestMockGenerators:
    def test_deterministic(self, name, gen):
        a = gen.fetch("600519.SH", days=120)
        b = gen.fetch("600519.SH", days=120)
        pd.testing.assert_frame_equal(a, b)

    def test_different_symbol_different_data(self, name, gen):
        a = gen.fetch("600519.SH", days=120)
        b = gen.fetch("000001.SZ", days=120)
        assert not np.allclose(a.iloc[:, 0].values, b.iloc[:, 0].values)

    def test_index_and_range(self, name, gen):
        df = gen.fetch("600519.SH", days=120)
        assert len(df) == 120
        assert isinstance(df.index, pd.DatetimeIndex)
        for col in df.columns:
            v = df[col].dropna()
            assert v.notna().all()

    def test_meta(self, name, gen):
        meta = gen.meta()
        assert set(meta) >= {"name", "description"}


def test_satellite_range():
    df = SatelliteDataGenerator().fetch("600519.SH", days=200)
    assert df["parking_lot_saturation"].between(0, 1).all()
    assert df["night_light_index"].between(0, 1).all()


def test_social_range():
    df = SocialSentimentGenerator().fetch("600519.SH", days=200)
    assert df["sentiment_raw"].between(-1, 1).all()
    assert (df["mentions_volume"] >= 0).all()


def test_supply_chain_range():
    df = SupplyChainGenerator().fetch("600519.SH", days=200)
    assert df["supplier_delivery_rate"].between(0, 1).all()
    assert df["inventory_pressure"].between(0, 1).all()


def test_esg_range():
    df = ESGScoreGenerator().fetch("600519.SH", days=200)
    assert df["esg_score"].between(0, 100).all()
    for sub in ("esg_e", "esg_s", "esg_g"):
        assert df[sub].between(0, 100).all()
    # 分项平均 ≈ 总分（允许舍入误差）
    mean_sub = df[["esg_e", "esg_s", "esg_g"]].mean(axis=1)
    assert np.allclose(mean_sub, df["esg_score"], atol=1.5)


# ---------------------------------------------------------------------------
# 数据源注册与真实接口预留
# ---------------------------------------------------------------------------


def test_default_sources_registered():
    assert set(DEFAULT_SOURCES) == {"satellite", "social", "supply_chain", "esg"}


def test_get_source_valid():
    assert get_source("satellite") is DEFAULT_SOURCES["satellite"]


def test_get_source_invalid():
    with pytest.raises(KeyError):
        get_source("nonexistent")


def test_real_source_reserved():
    real = RealAlternativeDataSource(api_key="test-key")
    meta = real.meta()
    assert meta["name"] == "real"
    assert "description" in meta
    with pytest.raises(NotImplementedError):
        real.fetch("600519.SH", days=10)
    with pytest.raises(NotImplementedError):
        RealAlternativeDataSource.convert({"payload": 1})


def test_fetch_all_merge():
    df = fetch_all("600519.SH", days=100)
    assert len(df) == 100
    assert "sentiment_raw" in df.columns
    assert "esg_score" in df.columns


# ---------------------------------------------------------------------------
# 另类因子：shift(1) 防未来函数
# ---------------------------------------------------------------------------


def test_factor_names_complete():
    assert ALTERNATIVE_FACTOR_NAMES == [
        "sentiment_score", "sentiment_momentum", "supply_chain_health",
        "esg_score", "esg_momentum", "satellite_activity",
        "alternative_composite",
    ]


def test_shift_lag_default_1():
    """lag=1：t 日因子值 = t-1 日原始值，杜绝未来函数。"""
    eng = AlternativeFactorEngine(lag=1)
    factors = eng.calculate_all("600519.SH", days=150)
    raw = eng.compute_raw(fetch_all("600519.SH", days=150))
    for col in ALTERNATIVE_FACTOR_NAMES:
        shifted = raw[col].shift(1)
        a = factors[col].iloc[2:100].to_numpy()
        b = shifted.iloc[2:100].to_numpy()
        mask = ~(np.isnan(a) | np.isnan(b))
        assert np.allclose(a[mask], b[mask]), col


def test_shift_lag_zero_equals_raw():
    """lag=0 时（仅测试用）因子等于原始值（首行之后无 NaN 的列）。"""
    eng = AlternativeFactorEngine(lag=0)
    factors = eng.calculate_all("600519.SH", days=150)
    raw = eng.compute_raw(fetch_all("600519.SH", days=150))
    col = "esg_score"  # 除以 100，无滚动窗口，无前置 NaN
    assert np.allclose(factors[col].to_numpy(), raw[col].to_numpy())


def test_factor_meta_registry_format():
    metas = AlternativeFactorEngine.factor_meta()
    assert len(metas) == len(ALTERNATIVE_FACTOR_NAMES)
    for m in metas:
        assert set(m) >= {"name", "category", "description", "direction"}
        assert m["category"] == "另类"
        assert m["direction"] == 1


def test_attach_to_klines_alignment():
    eng = AlternativeFactorEngine()
    idx = pd.date_range("2026-01-01", periods=80, freq="D")
    df = pd.DataFrame(
        {"close": np.linspace(100, 110, 80), "volume": np.full(80, 1e6)},
        index=idx,
    )
    out = eng.attach_to_klines(df, "600519.SH")
    assert len(out) == 80
    for col in ALTERNATIVE_FACTOR_NAMES:
        assert col in out.columns


# ---------------------------------------------------------------------------
# 与 FactorEngine 集成
# ---------------------------------------------------------------------------


def test_factor_engine_registry_contains_alternative():
    from factors.factor_engine import FactorEngine

    fe = FactorEngine()
    for name in ALTERNATIVE_FACTOR_NAMES:
        assert name in fe.factor_names
    entries = [f for f in fe.get_factor_list() if f["category"] == "另类"]
    assert len(entries) == len(ALTERNATIVE_FACTOR_NAMES)


def test_factor_engine_calculate_factors_includes_alternative():
    from factors.factor_engine import FactorEngine

    fe = FactorEngine()
    idx = pd.date_range("2026-01-01", periods=120, freq="D")
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 105, 120),
            "high": np.linspace(101, 106, 120),
            "low": np.linspace(99, 104, 120),
            "close": np.linspace(100, 105, 120),
            "volume": np.full(120, 1e6),
            "amount": np.full(120, 1e8),
        },
        index=idx,
    )
    out = fe.calculate_factors(df, symbol="600519.SH")
    for name in ALTERNATIVE_FACTOR_NAMES:
        assert name in out.columns
    # 不带 symbol（空字符串）时不计算另类因子（防御分支）
    out2 = fe.calculate_factors(df)
    assert "sentiment_score" not in out2.columns
