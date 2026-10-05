"""策略基类定义。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class Signal:
    """单条策略信号。

    Attributes:
        date: 信号产生日期。
        symbol: 标的代码。
        strategy: 策略名称。
        action: 动作类型 buy / sell / hold。
        confidence: 置信度 0~1。
        price: 产生信号时的参考价格。
        metadata: 附加信息（指标值等）。
    """
    date: pd.Timestamp
    symbol: str
    strategy: str
    action: str  # buy / sell / hold
    confidence: float
    price: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """策略抽象基类。

    所有子类必须实现 generate_signals 方法。
    信号计算完成后统一 shift(1)，避免未来函数。
    """

    name: str = "base"

    def __init__(self, params: Dict[str, Any] | None = None):
        self.params = params or {}

    @abstractmethod
    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """在 df 上计算原始信号列（signal, confidence），不做 shift。

        子类实现，返回包含 signal(1/-1/0) 和 confidence 列的 DataFrame。
        """
        ...

    def generate_signals(self, df: pd.DataFrame, symbol: str = "") -> List[Signal]:
        """生成信号列表，自动 shift(1) 避免未来函数。

        Args:
            df: 行情数据，需包含 close 等列。
            symbol: 标的代码。

        Returns:
            Signal 对象列表。
        """
        raw = self._compute_raw_signals(df)
        # 关键：信号延迟一天执行，避免未来函数
        raw["signal"] = raw["signal"].shift(1)
        raw["confidence"] = raw["confidence"].shift(1)

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
                confidence=float(row.get("confidence", 0.5)),
                price=float(row.get("close", 0)),
                metadata={k: row[k] for k in raw.columns if k not in ("signal", "confidence")},
            ))
        return signals

    def get_signal_dataframe(self, df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        """返回带 signal / confidence 列的 DataFrame（已 shift）。"""
        raw = self._compute_raw_signals(df)
        raw["signal"] = raw["signal"].shift(1)
        raw["confidence"] = raw["confidence"].shift(1)
        return raw
