"""多市场（A股 / 美股 / 港股）适配单元测试。

覆盖：
    - 标的代码标准化 / 腾讯格式转换 / 货币判断
    - 各市场交易成本（佣金、印花税、SEC 费、港股交易费）
    - 美股按股计费、港股多费率
    - 股票池包含三个市场共 25 只标的
    - 用 mock 数据跑通美股 / 港股回测

运行:
    cd quant_trading_system
    python -m pytest tests/test_multi_market.py -v
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from backtest.engine import BacktestEngine
from data.data_fetcher import (
    _default_mock_base_price,
    get_currency,
    normalize_symbol,
    to_tencent_format,
)
from strategies.ma_cross import MACrossStrategy
from trading.market_config import (
    MARKET_COSTS,
    calc_commission,
    calc_sec_fee,
    calc_stamp_tax,
    calc_trading_fee,
    get_market_cost,
    get_market_key,
    lot_size_of,
)

POOL_PATH = Path(__file__).resolve().parent.parent / "config" / "stock_pool.yaml"


# ---------------------------------------------------------------------------
# 标的代码格式
# ---------------------------------------------------------------------------

class TestNormalizeSymbol:
    """normalize_symbol 各种输入格式转换。"""

    def test_a_share_with_suffix(self):
        assert normalize_symbol("600519.SH") == "600519.SH"
        assert normalize_symbol("000001.SZ") == "000001.SZ"

    def test_a_share_bare_digits(self):
        assert normalize_symbol("600519") == "600519.SH"
        assert normalize_symbol("000001") == "000001.SZ"

    def test_tencent_prefix(self):
        assert normalize_symbol("sh600519") == "600519.SH"
        assert normalize_symbol("sz000001") == "000001.SZ"

    def test_us(self):
        assert normalize_symbol("AAPL.US") == "AAPL.US"
        assert normalize_symbol("aapl") == "AAPL.US"

    def test_hk(self):
        assert normalize_symbol("00700.HK") == "00700.HK"


class TestToTencentFormat:
    """标准化代码 -> 腾讯财经接口格式。"""

    def test_a_share(self):
        assert to_tencent_format("600519.SH") == "sh600519"
        assert to_tencent_format("000001.SZ") == "sz000001"

    def test_us(self):
        assert to_tencent_format("AAPL.US") == "usAAPL"
        assert to_tencent_format("MSFT.US") == "usMSFT"

    def test_hk_pads_to_5_digits(self):
        assert to_tencent_format("00700.HK") == "hk00700"
        assert to_tencent_format("09988.HK") == "hk09988"


class TestGetCurrency:
    """货币判断。"""

    def test_cny(self):
        assert get_currency("600519.SH") == "CNY"
        assert get_currency("000001.SZ") == "CNY"

    def test_usd(self):
        assert get_currency("AAPL.US") == "USD"
        assert get_currency("NVDA.US") == "USD"

    def test_hkd(self):
        assert get_currency("00700.HK") == "HKD"
        assert get_currency("09618.HK") == "HKD"


# ---------------------------------------------------------------------------
# 交易成本
# ---------------------------------------------------------------------------

class TestMarketCostCalculation:
    """各市场佣金 / 印花税计算。"""

    def test_a_share_commission_min(self):
        # 小额 A股：低于最低 5 元按 5 元收
        fee = calc_commission("600519.SH", amount=1000.0, shares=1, action="buy")
        assert fee == pytest.approx(5.0)

    def test_a_share_commission_rate(self):
        fee = calc_commission("600519.SH", amount=1_000_000.0, shares=1000, action="buy")
        assert fee == pytest.approx(1_000_000.0 * 0.00025)

    def test_a_share_stamp_tax_sell_only(self):
        assert calc_stamp_tax("600519.SH", 100_000.0, "buy") == 0.0
        assert calc_stamp_tax("600519.SH", 100_000.0, "sell") == pytest.approx(50.0)

    def test_market_key_and_config(self):
        assert get_market_key("AAPL.US") == "US"
        assert get_market_key("00700.HK") == "HK"
        assert get_market_key("600519.SH") == "CN"
        assert get_market_cost("AAPL.US").currency == "USD"
        assert lot_size_of("AAPL.US") == 1
        assert lot_size_of("00700.HK") == 100
        assert lot_size_of("600519.SH") == 100

    def test_configs_registered(self):
        assert set(MARKET_COSTS) == {"CN", "US", "HK"}


class TestUSCommissionPerShare:
    """美股按股计费：max(shares * 0.005, 1.0)。"""

    def test_small_order_floored_to_minimum(self):
        # 10 股：10*0.005 = 0.05 -> 最低 1 美元
        assert calc_commission("AAPL.US", amount=1800.0, shares=10, action="buy") == pytest.approx(1.0)

    def test_large_order_per_share(self):
        # 1000 股：1000*0.005 = 5.0
        fee = calc_commission("AAPL.US", amount=180_000.0, shares=1000, action="buy")
        assert fee == pytest.approx(5.0)

    def test_no_stamp_tax_but_sec_fee_on_sell(self):
        assert calc_stamp_tax("AAPL.US", 100_000.0, "sell") == 0.0
        # SEC 费 0.000008 on sell
        assert calc_sec_fee("AAPL.US", 100_000.0, "sell") == pytest.approx(0.8)
        assert calc_sec_fee("AAPL.US", 100_000.0, "buy") == 0.0


class TestHKMultipleFees:
    """港股多费率：佣金 0.03% 最低 3、印花税 0.1%、交易费 0.005%。"""

    def test_commission(self):
        # amount=38000 -> 38000*0.0003 = 11.4 > 3
        assert calc_commission("00700.HK", 38000.0, 100, "buy") == pytest.approx(11.4)

    def test_commission_minimum(self):
        # 小额：max(amount*0.0003, 3.0) = 3.0
        assert calc_commission("00700.HK", 1000.0, 10, "buy") == pytest.approx(3.0)

    def test_stamp_tax_sell(self):
        assert calc_stamp_tax("00700.HK", 38000.0, "sell") == pytest.approx(38.0)
        assert calc_stamp_tax("00700.HK", 38000.0, "buy") == 0.0

    def test_trading_fee_both_directions(self):
        # 港股交易费双向收取
        assert calc_trading_fee("00700.HK", 38000.0, "buy") == pytest.approx(1.9)
        assert calc_trading_fee("00700.HK", 38000.0, "sell") == pytest.approx(1.9)


# ---------------------------------------------------------------------------
# 股票池
# ---------------------------------------------------------------------------

class TestStockPool:
    """股票池包含 3 个市场共 25 只标的。"""

    def test_pool_loads(self):
        assert POOL_PATH.exists(), "stock_pool.yaml 应存在"
        data = yaml.safe_load(open(POOL_PATH, encoding="utf-8"))
        stocks = data["stocks"]
        assert len(stocks) == 25
        assert data["pool_meta"]["count"] == 25
        markets = {s["market"] for s in stocks}
        assert markets == {"A股", "美股", "港股"}

    def test_us_hk_symbols_present(self):
        data = yaml.safe_load(open(POOL_PATH, encoding="utf-8"))
        symbols = {s["symbol"] for s in data["stocks"]}
        for must in ["AAPL.US", "MSFT.US", "GOOGL.US", "AMZN.US", "NVDA.US",
                     "00700.HK", "09988.HK", "03690.HK", "01810.HK", "09618.HK"]:
            assert must in symbols

    def test_every_symbol_has_currency(self):
        data = yaml.safe_load(open(POOL_PATH, encoding="utf-8"))
        for s in data["stocks"]:
            assert "currency" in s and s["currency"] in {"CNY", "USD", "HKD"}
            assert "market" in s and "exchange" in s


# ---------------------------------------------------------------------------
# 回测集成（mock 数据）
# ---------------------------------------------------------------------------

def _make_trend_data(symbol: str, base: float, n: int = 150) -> pd.DataFrame:
    """构造 先横盘偏弱 -> 上涨 -> 下跌 的确定性行情，确保双均线产生金叉/死叉。"""
    dates = pd.bdate_range("2024-01-02", periods=n)
    # 前 25 天横盘偏弱（MA5 位于 MA20 下方），随后 75 天上涨（金叉），最后下跌（死叉）
    flat = np.linspace(base, base * 0.97, 25)
    up = np.linspace(base * 0.97, base * 1.4, 75)
    down = np.linspace(base * 1.4, base * 1.05, n - 25 - 75)
    close = np.concatenate([flat, up, down])
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
            "amount": close * 1_000_000.0,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


class TestMultiMarketBacktest:
    """美股 / 港股回测能正常运行。"""

    def test_mock_base_price_per_market(self):
        assert _default_mock_base_price("AAPL.US") == 180.0
        assert _default_mock_base_price("00700.HK") == 380.0
        assert _default_mock_base_price("600519.SH") == 1680.0

    def test_backtest_us_stock(self):
        """用 mock 数据跑 AAPL.US 回测，美股按股计费。"""
        df = _make_trend_data("AAPL.US", base=180.0)
        engine = BacktestEngine(initial_capital=1_000_000.0)
        result = engine.run({"AAPL.US": df},
                            MACrossStrategy({"fast_period": 5, "slow_period": 20}),
                            symbol="AAPL.US")
        assert len(result.equity_curve) == len(df)
        buys = [t for t in result.trades if t.action == "buy"]
        sells = [t for t in result.trades if t.action == "sell"]
        assert buys, "应至少产生一笔买入"
        # 美股买入佣金 = max(shares*0.005, 1.0)，无印花税
        for t in buys:
            assert t.stamp_tax == 0.0
            assert t.commission == pytest.approx(max(t.shares * 0.005, 1.0))
        # 卖出不收印花税（stamp_tax=0），但引擎仍正常平仓
        for t in sells:
            assert t.stamp_tax == 0.0

    def test_backtest_hk_stock(self):
        """用 mock 数据跑 00700.HK 回测，港股多费率。"""
        df = _make_trend_data("00700.HK", base=380.0)
        engine = BacktestEngine(initial_capital=1_000_000.0)
        result = engine.run({"00700.HK": df},
                            MACrossStrategy({"fast_period": 5, "slow_period": 20}),
                            symbol="00700.HK")
        assert len(result.equity_curve) == len(df)
        sells = [t for t in result.trades if t.action == "sell"]
        buys = [t for t in result.trades if t.action == "buy"]
        assert buys, "应至少产生一笔买入"
        # 港股卖出印花税 = amount * 0.001 > 0
        for t in sells:
            assert t.stamp_tax == pytest.approx(t.amount * 0.001, rel=1e-6)
        # 港股按 100 股一手，股数应为 100 的整数倍
        for t in result.trades:
            assert t.shares % 100 == 0
