"""财务数据模块测试。"""
from __future__ import annotations

import pytest

from data.fundamental import (
    FundamentalData,
    FundamentalDataService,
    MockFundamentalProvider,
    CachedFundamentalProvider,
)


def test_mock_provider_returns_data():
    provider = MockFundamentalProvider()
    data = provider.fetch("600519.SH")
    assert data.symbol == "600519.SH"
    assert data.pe_ttm is not None
    assert data.pb is not None
    assert data.roe is not None
    assert data.source == "mock"


def test_mock_provider_deterministic():
    """同一标的应返回一致数据。"""
    provider = MockFundamentalProvider()
    d1 = provider.fetch("600519.SH")
    d2 = provider.fetch("600519.SH")
    assert d1.pe_ttm == d2.pe_ttm
    assert d1.pb == d2.pb


def test_mock_provider_different_symbols():
    """不同标的数据应不同。"""
    provider = MockFundamentalProvider()
    d1 = provider.fetch("600519.SH")
    d2 = provider.fetch("000001.SZ")
    # 大概率不同 (不是绝对保证)
    assert d1.pe_ttm != d2.pe_ttm or d1.pb != d2.pb


def test_mock_provider_batch():
    provider = MockFundamentalProvider()
    batch = provider.fetch_batch(["600519.SH", "000001.SZ", "AAPL.US"])
    assert len(batch) == 3
    assert all(isinstance(v, FundamentalData) for v in batch.values())


def test_cached_provider():
    inner = MockFundamentalProvider()
    cached = CachedFundamentalProvider(inner, cache_ttl_hours=1)
    d1 = cached.fetch("600519.SH")
    d2 = cached.fetch("600519.SH")
    assert d1.pe_ttm == d2.pe_ttm


def test_fundamental_service():
    service = FundamentalDataService()
    data = service.get("600519.SH")
    assert isinstance(data, FundamentalData)
    assert data.pe_ttm is not None


def test_fundamental_service_batch():
    service = FundamentalDataService()
    batch = service.get_batch(["600519.SH", "000001.SZ"])
    assert len(batch) == 2


def test_fundamental_service_to_dataframe():
    service = FundamentalDataService()
    df = service.to_dataframe(["600519.SH", "000001.SZ"])
    assert len(df) == 2
    assert "pe_ttm" in df.columns
    assert "pb" in df.columns
    assert "roe" in df.columns


def test_fundamental_data_to_dict():
    data = FundamentalData(symbol="TEST", date="2024-01-01", pe_ttm=15.0)
    d = data.to_dict()
    assert d["symbol"] == "TEST"
    assert d["pe_ttm"] == 15.0


def test_different_markets():
    provider = MockFundamentalProvider()
    cn = provider.fetch("600519.SH")
    us = provider.fetch("AAPL.US")
    hk = provider.fetch("00700.HK")
    # 不同市场应有不同基准特征
    assert cn.pe_ttm is not None
    assert us.pe_ttm is not None
    assert hk.pe_ttm is not None



def test_mock_provider_all_fields():
    """断言 MockFundamentalProvider 返回的全部 18 个字段均不为 None。"""
    provider = MockFundamentalProvider()
    data = provider.fetch("TEST.SH")
    fields = [
        "symbol", "date", "pe_ttm", "pb", "ps_ttm", "dividend_yield",
        "ev_ebitda", "roe", "roa", "gross_margin", "net_margin",
        "revenue_growth_yoy", "profit_growth_yoy", "debt_to_asset",
        "current_ratio", "inventory_turnover", "receivable_turnover", "source",
    ]
    for field in fields:
        val = getattr(data, field)
        assert val is not None, f"Field {field} is None"


def test_cache_path_traversal():
    """验证 _cache_path 对路径遍历字符的正确清理。"""
    provider = CachedFundamentalProvider(MockFundamentalProvider())
    # 正常 symbol
    p1 = provider._cache_path("600519.SH")
    assert ".." not in str(p1)
    # 含路径遍历字符
    p2 = provider._cache_path("../etc/passwd")
    assert ".." not in str(p2)
    assert "passwd" in str(p2)
