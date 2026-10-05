"""加密货币数据模块（Crypto）。

提供 BTC/ETH 等加密货币的数据获取与特殊市场规则处理：

1. **数据获取**：MockProvider 确定性生成 K 线 + 真实数据源接口预留
   （复用 data_fetcher 架构，provider 可切换）。
2. **24 小时交易**：无开盘/收盘概念，日切用 UTC 00:00；`is_market_open` 恒真。
3. **高波动率适配**：默认波动率参数高于股票市场，提供风控参数差异化建议
   （更宽的止损、更小的仓位上限）。
4. **小数精度**：价格/数量统一 8 位小数（satoshi 精度），round 到位。
5. **市场配置扩展**：新增 crypto 市场类型，symbols 形如 "BTC-USD", "ETH-USD"。
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 市场配置扩展
# --------------------------------------------------------------------------- #

CRYPTO_MARKET_CONFIG: Dict[str, Any] = {
    "market_type": "crypto",
    "description": "加密货币市场（24小时交易）",
    "trading_hours": "24x7",
    "timezone": "UTC",
    "price_precision": 8,       # satoshi 精度
    "quantity_precision": 8,
    "default_symbols": ["BTC-USD", "ETH-USD"],
    # 高波动率风控参数（相对股票市场放宽）
    "risk_params": {
        "single_stop_loss": 0.08,        # 8%（股票默认 3%）
        "single_take_profit": 0.25,      # 25%
        "max_position_per_symbol": 0.10, # 单币种仓位更低（波动大）
        "max_total_position": 0.60,      # 总仓位上限更低
        "daily_loss_limit": 0.05,        # 5%
    },
    # 示例标的的波动率与漂移参数（mock 数据生成用）
    "symbol_params": {
        "BTC-USD": {"annual_vol": 0.60, "annual_drift": 0.40, "base_price": 60000.0},
        "ETH-USD": {"annual_vol": 0.75, "annual_drift": 0.30, "base_price": 3000.0},
    },
}


def is_crypto_symbol(symbol: str) -> bool:
    """判断是否加密货币标的（形如 BTC-USD / ETH-USDT）。"""
    s = symbol.strip().upper()
    parts = s.split("-")
    return len(parts) == 2 and parts[1] in ("USD", "USDT", "BTC", "ETH")


def round_crypto(value: float, precision: int = 8) -> float:
    """加密货币小数精度处理：统一 round 到 8 位小数。"""
    return round(float(value), precision)


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #

@dataclass
class CryptoMarketStatus:
    """加密货币市场状态。"""

    market_type: str = "crypto"
    is_open: bool = True           # 恒为 True（24x7）
    trading_hours: str = "24x7"
    timezone: str = "UTC"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "market_type": self.market_type,
            "is_open": self.is_open,
            "trading_hours": self.trading_hours,
            "timezone": self.timezone,
        }


# --------------------------------------------------------------------------- #
# Mock 数据生成
# --------------------------------------------------------------------------- #

def _symbol_seed(symbol: str) -> int:
    return int(hashlib.md5(symbol.upper().encode()).hexdigest()[:8], 16) % 10_000


class MockCryptoProvider:
    """确定性加密货币 K 线 mock 生成器。

    同一 symbol + 相同参数每次生成相同序列，保证回测可复现。
    """

    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate

    def fetch_klines(
        self,
        symbol: str,
        days: int = 250,
        base_price: Optional[float] = None,
        annual_vol: Optional[float] = None,
        annual_drift: Optional[float] = None,
    ) -> pd.DataFrame:
        """生成日级 K 线（按 UTC 日切）。

        Args:
            symbol: 形如 "BTC-USD"。
            days: 生成天数。
            base_price: 基准价；None 则取配置或 symbol 哈希推导。
            annual_vol: 年化波动率；None 则取配置。
            annual_drift: 年化漂移；None 则取配置。

        Returns:
            含 open/high/low/close/volume/amount 列的 DataFrame（index 为 UTC 日期）。
        """
        params = CRYPTO_MARKET_CONFIG["symbol_params"].get(symbol.upper(), {})
        base_price = base_price or params.get("base_price") or 10000.0
        annual_vol = annual_vol or params.get("annual_vol") or 0.60
        annual_drift = annual_drift or params.get("annual_drift") or 0.20

        rng = np.random.RandomState(_symbol_seed(symbol))
        daily_vol = annual_vol / math_sqrt(365)
        daily_drift = annual_drift / 365.0

        rets = rng.normal(daily_drift, daily_vol, size=days)
        close = base_price * np.cumprod(1.0 + rets)
        open_ = close * (1.0 + rng.normal(0, daily_vol / 4.0, size=days))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, daily_vol / 3.0, size=days)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, daily_vol / 3.0, size=days)))
        volume = rng.uniform(100, 5000, size=days)  # 币本数量级
        amount = volume * close

        index = pd.date_range("2026-01-01", periods=days, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [round_crypto(v) for v in open_],
                "high": [round_crypto(v) for v in high],
                "low": [round_crypto(v) for v in low],
                "close": [round_crypto(v) for v in close],
                "volume": [round_crypto(v) for v in volume],
                "amount": [round_crypto(v) for v in amount],
            },
            index=index,
        )
        return df


def math_sqrt(x: float) -> float:
    """math.sqrt 的模块内别名（便于测试 mock）。"""
    import math
    return math.sqrt(x)


# --------------------------------------------------------------------------- #
# 数据提供者（真实源接口预留）
# --------------------------------------------------------------------------- #

class CryptoDataProvider:
    """加密货币数据提供者。

    真实数据源（交易所 API，如 Binance/Coinbase）就绪后，替换 provider 即可；
    上层 24 小时规则 / 精度处理 / 风控参数无需变动。
    """

    def __init__(self, provider: Optional[Any] = None):
        self._provider = provider  # 真实源预留
        self._mock = MockCryptoProvider()

    @property
    def is_mock(self) -> bool:
        """当前是否使用 mock 数据源。"""
        return self._provider is None

    def get_market_status(self) -> CryptoMarketStatus:
        """获取市场状态（24x7 恒开）。"""
        return CryptoMarketStatus()

    def fetch_klines(self, symbol: str, days: int = 250) -> pd.DataFrame:
        """获取 K 线数据；真实源优先，失败/未配置时降级 mock。"""
        if self._provider is not None:
            try:
                return self._provider.fetch_klines(symbol, days)
            except Exception as e:  # noqa: BLE001
                logger.warning("真实加密货币数据源获取失败，降级 mock: %s", e)
        return self._mock.fetch_klines(symbol, days=days)

    def get_risk_params(self) -> Dict[str, float]:
        """获取加密货币市场差异化风控参数。"""
        return dict(CRYPTO_MARKET_CONFIG["risk_params"])

    def adapt_strategy_params(self, stock_params: Dict[str, float]) -> Dict[str, float]:
        """把股票市场策略参数适配为加密货币参数（放宽止损/止盈、缩小仓位）。"""
        rp = CRYPTO_MARKET_CONFIG["risk_params"]
        adapted = dict(stock_params)
        adapted["single_stop_loss"] = rp["single_stop_loss"]
        adapted["single_take_profit"] = rp["single_take_profit"]
        adapted["max_position_per_symbol"] = rp["max_position_per_symbol"]
        adapted["max_total_position"] = rp["max_total_position"]
        adapted["daily_loss_limit"] = rp["daily_loss_limit"]
        return adapted
