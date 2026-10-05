"""做市策略（Market Making）— 基于日K线的降级模拟实现。

当前系统无 Level2 盘口数据，本模块用日K线（ATR + 近期波动率）模拟做市逻辑：

1. **双边报价生成**：以当日 close 为中间价，基于 ATR 与近期波动率
   计算买卖报价 spread， spread = k * ATR / close（随波动率放大）。
2. **库存管理**：跟踪净持仓 inventory，偏离零时调整报价偏移 skew，
   使报价整体下移（多头时更愿卖出）或上移（空头时更愿买入），
   倾向于把库存拉回零。
3. **价差捕捉**：bid < ask，假设双边各成交一单位即赚取 spread。
4. **信号输出**：bid/ask price + size + 库存状态，作为 Signal.metadata。

继承 BaseStrategy；信号统一 shift(1) 防未来函数。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class MarketMakingStrategy(BaseStrategy):
    """日K线降级做市策略。

    Args:
        params: 可含：
            - spread_k: 价差系数（spread = k * ATR/close），默认 0.5
            - inventory_skew: 库存偏斜系数，默认 0.5
            - max_inventory: 最大库存（单位与持仓一致），默认 100
            - quote_size: 单边报价数量，默认 10
            - atr_period: ATR 周期，默认 14
            - vol_period: 波动率窗口，默认 20
    """

    name = "market_making"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.spread_k = self.params.get("spread_k", 0.5)
        self.inventory_skew = self.params.get("inventory_skew", 0.5)
        self.max_inventory = self.params.get("max_inventory", 100)
        self.quote_size = self.params.get("quote_size", 10)
        self.atr_period = self.params.get("atr_period", 14)
        self.vol_period = self.params.get("vol_period", 20)

        # 运行时库存（逐日演进，generate_signals 内部重放）
        self.inventory: int = 0
        self.realized_spread_pnl: float = 0.0

    # ------------------------------------------------------------------ #
    # 指标计算
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算 ATR（Average True Range）。

        TR = max(high - low, |high - pre_close|, |low - pre_close|)
        ATR = TR 的 period 日滚动均值。
        """
        high, low, close = df["high"], df["low"], df["close"]
        pre_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - pre_close).abs(), (low - pre_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()

    # ------------------------------------------------------------------ #
    # 单日报价
    # ------------------------------------------------------------------ #
    def quote(
        self,
        mid_price: float,
        atr: float,
        inventory: int,
    ) -> Dict[str, float]:
        """基于中间价、ATR 与库存生成双边报价。

        Returns:
            {"bid", "ask", "mid", "spread", "skew", "bid_size", "ask_size"}
        """
        if mid_price <= 0:
            return {
                "bid": 0.0, "ask": 0.0, "mid": 0.0, "spread": 0.0,
                "skew": 0.0, "bid_size": 0, "ask_size": 0,
            }

        half_spread = self.spread_k * atr / 2.0
        # 库存偏斜：多头（inventory>0）时报价整体下移，鼓励卖出减仓
        inv_ratio = inventory / self.max_inventory if self.max_inventory > 0 else 0.0
        inv_ratio = float(np.clip(inv_ratio, -1.0, 1.0))
        skew = -self.inventory_skew * half_spread * 2.0 * inv_ratio

        bid = mid_price - half_spread + skew
        ask = mid_price + half_spread + skew

        # 库存达到上限时停止同方向加仓
        bid_size = 0 if inventory >= self.max_inventory else self.quote_size
        ask_size = 0 if inventory <= -self.max_inventory else self.quote_size

        return {
            "bid": round(bid, 4),
            "ask": round(ask, 4),
            "mid": round(mid_price, 4),
            "spread": round(ask - bid, 4),
            "skew": round(skew, 4),
            "bid_size": bid_size,
            "ask_size": ask_size,
        }

    # ------------------------------------------------------------------ #
    # 成交模拟（库存演进）
    # ------------------------------------------------------------------ #
    def simulate_fills(self, quote_result: Dict[str, float], low: float, high: float) -> Dict[str, Any]:
        """用当日 low/high 模拟双边成交。

        - low <= bid：买单成交，库存 +size
        - high >= ask：卖单成交，库存 -size
        - 双边同时成交时赚取价差
        """
        bid_filled = low <= quote_result["bid"] and quote_result["bid_size"] > 0
        ask_filled = high >= quote_result["ask"] and quote_result["ask_size"] > 0

        delta = 0
        pnl = 0.0
        if bid_filled:
            delta += quote_result["bid_size"]
            pnl -= quote_result["bid"] * quote_result["bid_size"]
        if ask_filled:
            delta -= quote_result["ask_size"]
            pnl += quote_result["ask"] * quote_result["ask_size"]

        return {"inventory_delta": delta, "cash_pnl": pnl,
                "bid_filled": bid_filled, "ask_filled": ask_filled}

    # ------------------------------------------------------------------ #
    # BaseStrategy 接口
    # ------------------------------------------------------------------ #
    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """逐日生成报价信号。

        输出列：
            signal: 双边均成交（赚价差）= 0；仅买成交 = 1（库存增加，倾向后续卖出）；
                    仅卖成交 = -1。
            confidence: 基于 spread / mid 的比例（越大越高，封顶 1）。
            bid/ask/mid/spread/skew/inventory 列供 metadata 使用。
        """
        out = df.copy()
        if not all(c in out.columns for c in ("open", "high", "low", "close")):
            out["signal"] = 0
            out["confidence"] = 0.0
            return out

        out["atr"] = self.compute_atr(out, self.atr_period)
        out["volatility"] = out["close"].pct_change().rolling(
            self.vol_period, min_periods=1).std()

        self.inventory = 0
        self.realized_spread_pnl = 0.0

        signals = np.zeros(len(out))
        confidences = np.zeros(len(out))
        bids = np.zeros(len(out))
        asks = np.zeros(len(out))
        mids = np.zeros(len(out))
        spreads = np.zeros(len(out))
        inventories = np.zeros(len(out))

        for i, (idx, row) in enumerate(out.iterrows()):
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                inventories[i] = self.inventory
                continue

            q = self.quote(row["close"], atr, self.inventory)
            fill = self.simulate_fills(q, row["low"], row["high"])
            self.inventory += fill["inventory_delta"]
            self.realized_spread_pnl += fill["cash_pnl"]

            signals[i] = float(np.sign(fill["inventory_delta"]))
            # 置信度：价差占中间价比例
            if q["mid"] > 0:
                confidences[i] = min(q["spread"] / q["mid"] * 10.0, 1.0)
            bids[i] = q["bid"]
            asks[i] = q["ask"]
            mids[i] = q["mid"]
            spreads[i] = q["spread"]
            inventories[i] = self.inventory

        out["signal"] = signals
        out["confidence"] = confidences
        out["bid"] = bids
        out["ask"] = asks
        out["mid"] = mids
        out["spread"] = spreads
        out["inventory"] = inventories
        return out

    def generate_signals(self, df: pd.DataFrame, symbol: str = "") -> List[Signal]:
        """生成 Signal 列表，shift(1) 防未来函数。"""
        raw = self._compute_raw_signals(df)
        raw["signal"] = raw["signal"].shift(1)
        raw["confidence"] = raw["confidence"].shift(1)
        # 报价列也 shift(1)：t 日展示的是 t-1 日的报价
        for col in ("bid", "ask", "mid", "spread"):
            raw[col] = raw[col].shift(1)

        signals: List[Signal] = []
        for idx, row in raw.iterrows():
            sig_val = row.get("signal", 0)
            if pd.isna(sig_val) or sig_val == 0:
                continue
            action = "buy" if sig_val > 0 else "sell"
            signals.append(Signal(
                date=idx,
                symbol=symbol,
                strategy=self.name,
                action=action,
                confidence=float(row.get("confidence", 0.0)),
                price=float(row.get("mid", 0)),
                metadata={
                    "bid": _sf(row.get("bid")),
                    "ask": _sf(row.get("ask")),
                    "spread": _sf(row.get("spread")),
                    "inventory": int(row.get("inventory", 0)),
                },
            ))
        return signals


def _sf(v: Any) -> Optional[float]:
    """安全转 float（NaN/Inf -> None）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f
