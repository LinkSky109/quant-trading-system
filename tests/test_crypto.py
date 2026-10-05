"""加密货币模块单元测试。

覆盖：
  - 标的识别（is_crypto_symbol）
  - 小数精度处理（round_crypto 8 位）
  - 市场配置扩展（CRYPTO_MARKET_CONFIG：24x7 / UTC / 差异化风控参数）
  - Mock K 线生成（确定性、UTC 日切、精度、自定义参数）
  - CryptoDataProvider（mock 降级、真实源预留接口、参数适配）

全部使用 mock 数据，不依赖外部 API。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.crypto import (
    CRYPTO_MARKET_CONFIG,
    CryptoDataProvider,
    CryptoMarketStatus,
    MockCryptoProvider,
    is_crypto_symbol,
    round_crypto,
)


# ---------------------------------------------------------------------------
# 标的识别
# ---------------------------------------------------------------------------


class TestIsCryptoSymbol:
    @pytest.mark.parametrize("symbol", ["BTC-USD", "ETH-USD", "btc-usd",
                                        "SOL-USDT", "DOGE-BTC", " ETH-USD "])
    def test_valid(self, symbol):
        assert is_crypto_symbol(symbol) is True

    @pytest.mark.parametrize("symbol", ["", "BTC", "600519.SH", "AAPL",
                                        "BTC-EUR", "BTC-USDX"])
    def test_invalid(self, symbol):
        assert is_crypto_symbol(symbol) is False


# ---------------------------------------------------------------------------
# 精度
# ---------------------------------------------------------------------------


class TestRoundCrypto:
    def test_8_decimal_default(self):
        assert round_crypto(1.123456789123) == 1.12345679

    def test_no_float_noise(self):
        v = round_crypto(60000.1234567854321)
        s = f"{v:.10f}".rstrip("0")
        assert len(s.split(".")[1].rstrip("0")) <= 8

    def test_custom_precision(self):
        assert round_crypto(1.23456789, precision=2) == 1.23

    def test_negative_and_zero(self):
        assert round_crypto(0.0) == 0.0
        assert round_crypto(-1.123456789) == -1.12345679


# ---------------------------------------------------------------------------
# 市场配置
# ---------------------------------------------------------------------------


class TestMarketConfig:
    def test_market_type(self):
        assert CRYPTO_MARKET_CONFIG["market_type"] == "crypto"

    def test_24x7_trading(self):
        assert CRYPTO_MARKET_CONFIG["trading_hours"] == "24x7"
        assert CRYPTO_MARKET_CONFIG["timezone"] == "UTC"

    def test_precision(self):
        assert CRYPTO_MARKET_CONFIG["price_precision"] == 8
        assert CRYPTO_MARKET_CONFIG["quantity_precision"] == 8

    def test_default_symbols(self):
        assert "BTC-USD" in CRYPTO_MARKET_CONFIG["default_symbols"]
        assert "ETH-USD" in CRYPTO_MARKET_CONFIG["default_symbols"]

    def test_high_vol_risk_params(self):
        rp = CRYPTO_MARKET_CONFIG["risk_params"]
        # 加密货币止损比股票更宽
        assert rp["single_stop_loss"] == pytest.approx(0.08)
        assert rp["single_take_profit"] == pytest.approx(0.25)
        assert rp["max_position_per_symbol"] == pytest.approx(0.10)
        assert rp["max_total_position"] == pytest.approx(0.60)
        assert rp["daily_loss_limit"] == pytest.approx(0.05)

    def test_symbol_params(self):
        sp = CRYPTO_MARKET_CONFIG["symbol_params"]
        assert sp["BTC-USD"]["base_price"] == 60000.0
        assert sp["ETH-USD"]["base_price"] == 3000.0
        assert sp["BTC-USD"]["annual_vol"] == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# 市场状态
# ---------------------------------------------------------------------------


class TestMarketStatus:
    def test_always_open(self):
        st = CryptoMarketStatus()
        assert st.is_open is True
        assert st.trading_hours == "24x7"
        d = st.as_dict()
        assert d["market_type"] == "crypto"
        assert d["is_open"] is True


# ---------------------------------------------------------------------------
# Mock K 线生成
# ---------------------------------------------------------------------------


class TestMockCryptoProvider:
    def test_basic_structure(self):
        p = MockCryptoProvider()
        df = p.fetch_klines("BTC-USD", days=100)
        assert len(df) == 100
        for col in ("open", "high", "low", "close", "volume", "amount"):
            assert col in df.columns

    def test_utc_daily_index(self):
        p = MockCryptoProvider()
        df = p.fetch_klines("BTC-USD", days=50)
        assert str(df.index.tz) == "UTC"
        diffs = df.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(days=1)).all()

    def test_deterministic(self):
        p = MockCryptoProvider()
        df1 = p.fetch_klines("BTC-USD", days=100)
        df2 = p.fetch_klines("BTC-USD", days=100)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_symbols_differ(self):
        p = MockCryptoProvider()
        btc = p.fetch_klines("BTC-USD", days=50)
        eth = p.fetch_klines("ETH-USD", days=50)
        assert not np.allclose(btc["close"].values, eth["close"].values)

    def test_8_decimal_precision(self):
        p = MockCryptoProvider()
        df = p.fetch_klines("BTC-USD", days=60)
        for col in ("open", "high", "low", "close", "volume", "amount"):
            for v in df[col]:
                # round(v, 8) 后不应再变化（浮点二进制表示误差忽略不计）
                assert v == pytest.approx(round(v, 8), abs=1e-12)

    def test_ohlc_sanity(self):
        p = MockCryptoProvider()
        df = p.fetch_klines("BTC-USD", days=100)
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-6).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-6).all()
        assert (df["close"] > 0).all()

    def test_custom_params(self):
        p = MockCryptoProvider()
        df = p.fetch_klines("BTC-USD", days=30, base_price=100.0, annual_vol=0.1,
                            annual_drift=0.0)
        assert df["close"].mean() < 200  # 远低于 BTC 默认 6 万量级

    def test_unknown_symbol_fallback(self):
        p = MockCryptoProvider()
        df = p.fetch_klines("XYZ-USD", days=20)
        assert len(df) == 20
        assert df["close"].mean() > 0  # 走默认 base_price=10000


# ---------------------------------------------------------------------------
# CryptoDataProvider
# ---------------------------------------------------------------------------


class TestCryptoDataProvider:
    def test_is_mock_without_provider(self):
        p = CryptoDataProvider()
        assert p.is_mock is True

    def test_fetch_klines_mock(self):
        p = CryptoDataProvider()
        df = p.fetch_klines("BTC-USD", days=30)
        assert len(df) == 30

    def test_get_market_status(self):
        p = CryptoDataProvider()
        st = p.get_market_status()
        assert isinstance(st, CryptoMarketStatus)
        assert st.is_open is True

    def test_get_risk_params(self):
        p = CryptoDataProvider()
        rp = p.get_risk_params()
        assert rp == CRYPTO_MARKET_CONFIG["risk_params"]
        # 返回副本，修改不影响配置
        rp["single_stop_loss"] = 999
        assert CRYPTO_MARKET_CONFIG["risk_params"]["single_stop_loss"] == 0.08

    def test_adapt_strategy_params(self):
        p = CryptoDataProvider()
        stock_params = {
            "single_stop_loss": 0.03,
            "single_take_profit": 0.10,
            "max_position_per_symbol": 0.30,
            "max_total_position": 0.95,
            "daily_loss_limit": 0.02,
            "custom_field": 1.0,
        }
        adapted = p.adapt_strategy_params(stock_params)
        assert adapted["single_stop_loss"] == 0.08
        assert adapted["single_take_profit"] == 0.25
        assert adapted["max_position_per_symbol"] == 0.10
        assert adapted["custom_field"] == 1.0  # 非风控字段保留
        assert stock_params["single_stop_loss"] == 0.03  # 原参数不被修改

    def test_real_provider_preferred(self):
        class FakeReal:
            def fetch_klines(self, symbol, days):
                return pd.DataFrame({"close": [1.0] * days})

        p = CryptoDataProvider(provider=FakeReal())
        assert p.is_mock is False
        df = p.fetch_klines("BTC-USD", days=5)
        assert len(df) == 5

    def test_real_provider_fallback_on_error(self):
        class BrokenReal:
            def fetch_klines(self, symbol, days):
                raise RuntimeError("network down")

        p = CryptoDataProvider(provider=BrokenReal())
        df = p.fetch_klines("BTC-USD", days=10)
        # 真实源异常 -> 降级 mock
        assert len(df) == 10
        assert "close" in df.columns
