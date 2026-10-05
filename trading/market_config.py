"""多市场交易成本配置模块。

为 A股 / 美股 / 港股 提供统一的交易成本结构与计算函数，
回测引擎（:mod:`backtest.engine`）与模拟券商（:mod:`trading.broker`）
均通过本模块根据标的所在市场自动选择费率，避免在业务代码中散落硬编码。

市场键（market key）:
    - ``CN``:  A股（上交所 SH / 深交所 SZ / 北交所 BJ）
    - ``US``:  美股（纳斯达克 / 纽交所）
    - ``HK``:  港股（港交所）

注意：
    - 美股佣金不按金额比例，而是 ``0.005 美元/股``，单笔最低 1 美元；
      卖出时另收 SEC 费（0.0008%，即 0.000008）。
    - 港股佣金按金额 0.03%（最低 3 港元），卖出另收印花税 0.1% 与交易费 0.005%。
    - A股佣金按金额 0.025%（最低 5 元），卖出印花税 0.05%。
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MarketCostConfig",
    "MARKET_COSTS",
    "get_market_key",
    "get_market_cost",
    "get_currency",
    "calc_commission",
    "calc_stamp_tax",
    "calc_sec_fee",
    "calc_trading_fee",
    "lot_size_of",
]


@dataclass(frozen=True)
class MarketCostConfig:
    """单个市场的交易成本与交易单位配置。

    Attributes:
        commission_rate: 佣金率（按成交金额比例）。美股为占位 0.0，实际按股计费。
        commission_min: 单笔最低佣金（本币）。
        stamp_tax_rate: 印花税率（仅卖出收取）。
        sec_fee_rate: SEC 费费率（仅美股卖出收取，0.0008% = 0.000008）。
        trading_fee_rate: 交易费费率（港股 0.005% = 0.00005）。
        slippage_rate: 默认滑点率。
        currency: 计价货币代码，CNY / USD / HKD。
        lot_size: 每手股数（A股 100，美股 1，港股 100）。
    """

    commission_rate: float
    commission_min: float
    stamp_tax_rate: float
    sec_fee_rate: float
    trading_fee_rate: float
    slippage_rate: float
    currency: str
    lot_size: int


#: 各市场默认成本配置。
MARKET_COSTS: dict[str, MarketCostConfig] = {
    "CN": MarketCostConfig(
        commission_rate=0.00025,
        commission_min=5.0,
        stamp_tax_rate=0.0005,
        sec_fee_rate=0.0,
        trading_fee_rate=0.0,
        slippage_rate=0.001,
        currency="CNY",
        lot_size=100,
    ),
    "US": MarketCostConfig(
        # 佣金特殊：0.005 美元/股，最低 1 美元（见 calc_commission）
        commission_rate=0.0,
        commission_min=1.0,
        stamp_tax_rate=0.0,
        sec_fee_rate=0.000008,
        trading_fee_rate=0.0,
        slippage_rate=0.0005,
        currency="USD",
        lot_size=1,
    ),
    "HK": MarketCostConfig(
        commission_rate=0.0003,
        commission_min=3.0,
        stamp_tax_rate=0.001,
        sec_fee_rate=0.0,
        trading_fee_rate=0.00005,
        slippage_rate=0.001,
        currency="HKD",
        lot_size=100,
    ),
}

#: 美股按股计费的佣金单价（美元/股）。
_US_COMMISSION_PER_SHARE = 0.005


def _exchange_suffix(symbol: str) -> str:
    """提取标的代码对应的交易所后缀（大写），不依赖 data 模块。

    支持 ``600519.SH`` / ``AAPL.US`` / ``00700.HK`` 以及腾讯前缀
    （sh600519 / usAAPL / hk00700）等写法。

    无后缀的裸字母代码（如回测里的占位符 ``TEST``）保守归为 A股，
    以保持对既有回测行为的向后兼容；真实美股请使用 ``AAPL.US`` 写法。
    """
    s = symbol.strip().upper()
    if not s:
        return "CN"
    if "." in s:
        return s.split(".")[-1]
    # 腾讯式前缀 sh600519 / usAAPL / hk00700
    for prefix in ("SH", "SZ", "BJ", "US", "HK"):
        if s.startswith(prefix) and len(s) > 2:
            return prefix
    # 纯数字
    if s.isdigit():
        if len(s) == 5:
            return "HK"
        if len(s) == 6:
            if s.startswith(("6", "9")):
                return "SH"
            return "SZ"
    # 裸字母 / 其他写法：保守归为 A股（向后兼容）
    return "CN"


def get_market_key(symbol: str) -> str:
    """根据标的代码返回市场键 ``"CN"`` / ``"US"`` / ``"HK"``。

    Args:
        symbol: 标的代码，支持多种写法。

    Returns:
        市场键字符串。无法识别时默认归为 ``"CN"``（保持向后兼容）。
    """
    suffix = _exchange_suffix(symbol)
    if suffix in ("SH", "SZ", "BJ"):
        return "CN"
    if suffix == "US":
        return "US"
    if suffix == "HK":
        return "HK"
    return "CN"


def get_market_cost(symbol: str) -> MarketCostConfig:
    """返回标的所在市场的成本配置。

    Args:
        symbol: 标的代码。

    Returns:
        对应的 :class:`MarketCostConfig`。
    """
    return MARKET_COSTS[get_market_key(symbol)]


def get_currency(symbol: str) -> str:
    """返回标的计价货币代码 ``CNY`` / ``USD`` / ``HKD``。"""
    return get_market_cost(symbol).currency


def lot_size_of(symbol: str) -> int:
    """返回标的所在市场每手股数。"""
    return get_market_cost(symbol).lot_size


def calc_commission(symbol: str, amount: float, shares: int, action: str) -> float:
    """计算单笔成交佣金。

    - 美股：``max(shares * 0.005, 1.0)``（按股计费，与买卖方向无关）。
    - 其他市场：``max(amount * commission_rate, commission_min)``。

    Args:
        symbol: 标的代码。
        amount: 成交金额 = 成交价 * 股数。
        shares: 成交股数（美股按股计费需要）。
        action: ``"buy"`` / ``"sell"``（当前佣金与方向无关，保留参数以统一签名）。

    Returns:
        佣金金额（本币）。
    """
    cfg = get_market_cost(symbol)
    if get_market_key(symbol) == "US":
        return max(shares * _US_COMMISSION_PER_SHARE, cfg.commission_min)
    return max(amount * cfg.commission_rate, cfg.commission_min)


def calc_stamp_tax(symbol: str, amount: float, action: str) -> float:
    """计算印花税（仅卖出收取）。

    A股 0.05%、港股 0.1%、美股 0。
    """
    if action != "sell":
        return 0.0
    return amount * get_market_cost(symbol).stamp_tax_rate


def calc_sec_fee(symbol: str, amount: float, action: str) -> float:
    """计算 SEC 费（仅美股卖出收取，0.0008%）。"""
    if action != "sell":
        return 0.0
    return amount * get_market_cost(symbol).sec_fee_rate


def calc_trading_fee(symbol: str, amount: float, action: str) -> float:
    """计算交易费（港股双向收取 0.005%，其他市场为 0）。"""
    return amount * get_market_cost(symbol).trading_fee_rate
