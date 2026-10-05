"""期权支持模块（Option Support）。

提供期权行情数据结构、Black-Scholes 简化版 Greeks 计算、常用期权策略模板
与 Mock 期权链生成器。当前无真实期权数据源，全部用 mock 实现，
预留真实数据源接入接口（REQ-P2-04 数据源就绪后切换 provider 即可）。

包含：
1. **OptionQuote** — 期权行情数据结构（标的/行权价/到期日/类型/价格/Greeks）
2. **Greeks 计算** — Delta/Gamma/Vega/Theta/Rho（Black-Scholes 欧式期权解析解）
3. **策略模板** — 备兑认购（Covered Call）、牛市价差（Bull Call Spread）、
   跨式组合（Straddle）
4. **MockOptionChainGenerator** — 基于标的价格推导期权链
5. **真实数据源接口预留** — OptionDataProvider 基类
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Black-Scholes 基础
# --------------------------------------------------------------------------- #

def _norm_cdf(x: float) -> float:
    """标准正态分布累积函数。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """标准正态分布概率密度函数。"""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_price(
    spot: float,
    strike: float,
    ttm_years: float,
    vol: float,
    risk_free_rate: float = 0.03,
    option_type: str = "call",
) -> float:
    """Black-Scholes 欧式期权理论价格。

    Args:
        spot: 标的现价。
        strike: 行权价。
        ttm_years: 剩余到期时间（年）。
        vol: 年化波动率。
        risk_free_rate: 无风险利率。
        option_type: "call" 或 "put"。

    Returns:
        理论价格；ttm <= 0 或 vol <= 0 时返回内在价值。
    """
    if ttm_years <= 0 or vol <= 0:
        intrinsic = spot - strike if option_type == "call" else strike - spot
        return max(intrinsic, 0.0)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * vol * vol) * ttm_years) / (
        vol * math.sqrt(ttm_years)
    )
    d2 = d1 - vol * math.sqrt(ttm_years)
    if option_type == "call":
        return spot * _norm_cdf(d1) - strike * math.exp(-risk_free_rate * ttm_years) * _norm_cdf(d2)
    return strike * math.exp(-risk_free_rate * ttm_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def calculate_greeks(
    spot: float,
    strike: float,
    ttm_years: float,
    vol: float,
    risk_free_rate: float = 0.03,
    option_type: str = "call",
) -> Dict[str, float]:
    """计算 Black-Scholes Greeks（Delta/Gamma/Vega/Theta/Rho）。

    Returns:
        {"delta", "gamma", "vega", "theta", "rho"}；到期时仅返回内在价值方向。
    """
    if ttm_years <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        if option_type == "call":
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    sqrt_t = math.sqrt(ttm_years)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * vol * vol) * ttm_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    pdf_d1 = _norm_pdf(d1)
    disc = math.exp(-risk_free_rate * ttm_years)

    gamma = pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t / 100.0  # 每 1% 波动率变化

    if option_type == "call":
        delta = _norm_cdf(d1)
        theta = (
            -spot * pdf_d1 * vol / (2.0 * sqrt_t)
            - risk_free_rate * strike * disc * _norm_cdf(d2)
        ) / 365.0  # 每自然日
        rho = strike * ttm_years * disc * _norm_cdf(d2) / 100.0
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = (
            -spot * pdf_d1 * vol / (2.0 * sqrt_t)
            + risk_free_rate * strike * disc * _norm_cdf(-d2)
        ) / 365.0
        rho = -strike * ttm_years * disc * _norm_cdf(-d2) / 100.0

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #

@dataclass
class OptionQuote:
    """单条期权行情。

    Attributes:
        underlying: 标的代码。
        strike: 行权价。
        expiry: 到期日（ISO 日期字符串，如 "2026-12-31"）。
        option_type: "call" / "put"。
        last_price: 最新价。
        iv: 隐含波动率（年化）。
        ttm_years: 剩余到期时间（年）。
        greeks: Delta/Gamma/Vega/Theta/Rho。
    """

    underlying: str
    strike: float
    expiry: str
    option_type: str  # call / put
    last_price: float
    iv: float
    ttm_years: float
    greeks: Dict[str, float] = field(default_factory=dict)

    @property
    def moneyness(self) -> str:
        """ITM / ATM / OTM。"""
        spot_ref = self.greeks.get("_spot", 0.0)
        return "UNKNOWN"

    def as_dict(self) -> Dict[str, Any]:
        """序列化为 dict。"""
        return asdict(self)


# --------------------------------------------------------------------------- #
# Mock 期权链生成器
# --------------------------------------------------------------------------- #

class MockOptionChainGenerator:
    """基于标的价格确定性推导期权链（无外部依赖）。

    同一 (symbol, spot, expiry) 组合每次生成相同结果（hash 种子）。
    """

    def __init__(self, base_iv: float = 0.25, risk_free_rate: float = 0.03):
        self.base_iv = base_iv
        self.risk_free_rate = risk_free_rate

    def generate_chain(
        self,
        symbol: str,
        spot: float,
        expiry: str,
        ttm_years: float,
        strikes: Optional[List[float]] = None,
        num_strikes: int = 5,
        strike_step: Optional[float] = None,
    ) -> Dict[str, List[OptionQuote]]:
        """生成期权链。

        Args:
            symbol: 标的代码。
            spot: 标的现价。
            expiry: 到期日 ISO 字符串。
            ttm_years: 剩余到期时间（年）。
            strikes: 指定行权价列表；None 则围绕 spot 生成。
            num_strikes: 每个 strike_step 间隔下生成的行权价数量（默认 5）。
            strike_step: 行权价间隔；None 则取 spot 的 5% 并取整。

        Returns:
            {"calls": [OptionQuote...], "puts": [OptionQuote...]}
        """
        if spot <= 0:
            return {"calls": [], "puts": []}

        if strike_step is None:
            strike_step = max(round(spot * 0.05, 2), 0.01)
        if strikes is None:
            center = round(spot / strike_step) * strike_step
            half = num_strikes // 2
            strikes = [
                round(center + (i - half) * strike_step, 4)
                for i in range(num_strikes)
            ]

        # 波动率微笑：行权价偏离 spot 越远 IV 越高（确定性 skew）
        seed = int(hashlib.md5(f"{symbol}_{expiry}".encode()).hexdigest()[:8], 16) % 1000
        rng = np.random.RandomState(seed)

        calls: List[OptionQuote] = []
        puts: List[OptionQuote] = []
        for strike in strikes:
            moneyness = math.log(strike / spot) if spot > 0 else 0.0
            iv = max(self.base_iv + 0.8 * moneyness * moneyness + rng.uniform(-0.01, 0.01), 0.01)
            for opt_type in ("call", "put"):
                price = black_scholes_price(
                    spot, strike, ttm_years, iv, self.risk_free_rate, opt_type
                )
                greeks = calculate_greeks(
                    spot, strike, ttm_years, iv, self.risk_free_rate, opt_type
                )
                quote = OptionQuote(
                    underlying=symbol,
                    strike=float(strike),
                    expiry=expiry,
                    option_type=opt_type,
                    last_price=round(max(price, 0.0), 4),
                    iv=round(iv, 4),
                    ttm_years=ttm_years,
                    greeks={k: round(v, 6) for k, v in greeks.items()},
                )
                (calls if opt_type == "call" else puts).append(quote)
        return {"calls": calls, "puts": puts}


# --------------------------------------------------------------------------- #
# 期权策略模板
# --------------------------------------------------------------------------- #

@dataclass
class OptionStrategyLeg:
    """期权策略单腿。"""

    option_type: str      # call / put
    strike: float
    expiry: str
    side: str             # long / short
    quantity: int = 1
    price: float = 0.0
    greeks: Dict[str, float] = field(default_factory=dict)

    def net_cost(self) -> float:
        """该腿净成本（long 为支出为正，short 收入为负）。"""
        sign = 1.0 if self.side == "long" else -1.0
        return sign * self.price * self.quantity


@dataclass
class OptionStrategyResult:
    """期权策略构建结果。"""

    name: str
    legs: List[OptionStrategyLeg]
    net_cost: float
    max_profit: Optional[float]
    max_loss: Optional[float]
    combined_greeks: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "legs": [asdict(l) for l in self.legs],
            "net_cost": self.net_cost,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "combined_greeks": self.combined_greeks,
        }


def _combine_greeks(legs: List[OptionStrategyLeg]) -> Dict[str, float]:
    """汇总各腿 Greeks（short 腿取反）。"""
    keys = ("delta", "gamma", "vega", "theta", "rho")
    combined = {k: 0.0 for k in keys}
    for leg in legs:
        sign = 1.0 if leg.side == "long" else -1.0
        for k in keys:
            combined[k] += sign * leg.greeks.get(k, 0.0) * leg.quantity
    return {k: round(v, 6) for k, v in combined.items()}


class OptionStrategyBuilder:
    """常用期权策略模板构建器。"""

    def __init__(self, chain: Dict[str, List[OptionQuote]]):
        self.chain = chain

    def _find_quote(self, option_type: str, strike: float) -> Optional[OptionQuote]:
        for q in self.chain.get("calls" if option_type == "call" else "puts", []):
            if q.strike == strike:
                return q
        return None

    def _make_leg(self, quote: OptionQuote, side: str, quantity: int = 1) -> OptionStrategyLeg:
        return OptionStrategyLeg(
            option_type=quote.option_type,
            strike=quote.strike,
            expiry=quote.expiry,
            side=side,
            quantity=quantity,
            price=quote.last_price,
            greeks=quote.greeks,
        )

    def covered_call(self, spot: float, strike: float, quantity: int = 1) -> Optional[OptionStrategyResult]:
        """备兑认购：持有标的 + 卖出认购。

        max_profit = strike - spot + premium；max_loss = spot - premium（标的归零）。
        """
        q = self._find_quote("call", strike)
        if q is None:
            return None
        leg = self._make_leg(q, "short", quantity)
        premium = leg.net_cost() * -1.0  # short 收取权利金（正数）
        return OptionStrategyResult(
            name="covered_call",
            legs=[leg],
            net_cost=-premium,  # 净成本为负（净收入）
            max_profit=strike - spot + premium,
            max_loss=spot - premium,
            combined_greeks=_combine_greeks([leg]),
        )

    def bull_call_spread(
        self, low_strike: float, high_strike: float, quantity: int = 1
    ) -> Optional[OptionStrategyResult]:
        """牛市价差：买低行权价认购 + 卖高行权价认购。

        max_profit = (high - low) - net_cost；max_loss = net_cost。
        """
        ql = self._find_quote("call", low_strike)
        qs = self._find_quote("call", high_strike)
        if ql is None or qs is None:
            return None
        legs = [self._make_leg(ql, "long", quantity), self._make_leg(qs, "short", quantity)]
        net_cost = sum(l.net_cost() for l in legs)
        width = high_strike - low_strike
        return OptionStrategyResult(
            name="bull_call_spread",
            legs=legs,
            net_cost=net_cost,
            max_profit=width - net_cost,
            max_loss=net_cost,
            combined_greeks=_combine_greeks(legs),
        )

    def straddle(self, strike: float, quantity: int = 1) -> Optional[OptionStrategyResult]:
        """跨式组合：同行权价买认购 + 买认沽。

        max_loss = net_cost；理论上 max_profit 无上限。
        """
        qc = self._find_quote("call", strike)
        qp = self._find_quote("put", strike)
        if qc is None or qp is None:
            return None
        legs = [self._make_leg(qc, "long", quantity), self._make_leg(qp, "long", quantity)]
        net_cost = sum(l.net_cost() for l in legs)
        return OptionStrategyResult(
            name="straddle",
            legs=legs,
            net_cost=net_cost,
            max_profit=None,  # 理论无限
            max_loss=net_cost,
            combined_greeks=_combine_greeks(legs),
        )


# --------------------------------------------------------------------------- #
# 真实数据源接口预留
# --------------------------------------------------------------------------- #

class OptionDataProvider:
    """期权数据源接口基类（预留）。

    真实数据源（如交易所期权行情 API）就绪后，继承并实现 fetch_chain 即可，
    上层 Greeks / 策略模板逻辑无需变动。
    """

    def fetch_chain(
        self, symbol: str, spot: float, expiry: str, ttm_years: float
    ) -> Dict[str, List[OptionQuote]]:
        """获取指定标的与到期日的期权链。默认降级到 mock 生成器。"""
        gen = MockOptionChainGenerator()
        return gen.generate_chain(symbol, spot, expiry, ttm_years)
