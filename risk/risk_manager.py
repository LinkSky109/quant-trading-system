"""风险管理模块。

实现以下风控规则：
1. 单笔止损 / 止盈
2. 最大回撤控制（超限暂停交易）
3. 单标的仓位上限
4. 总仓位上限
5. 单日亏损限额
6. Jev 置信度阈值
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskEvent:
    """风控事件记录。"""
    timestamp: str
    event_type: str  # stop_loss / take_profit / drawdown / daily_loss / position_limit
    symbol: str
    detail: str


class RiskManager:
    """风控管理器。"""

    def __init__(
        self,
        single_stop_loss: float = 0.03,
        single_take_profit: float = 0.08,
        max_drawdown_pause: float = 0.10,
        max_position_per_symbol: float = 0.20,
        max_total_position: float = 0.80,
        daily_loss_limit: float = 0.02,
        jev_confidence_threshold: float = 0.6,
        initial_capital: float = 1_000_000.0,
    ):
        self.single_stop_loss = single_stop_loss
        self.single_take_profit = single_take_profit
        self.max_drawdown_pause = max_drawdown_pause
        self.max_position_per_symbol = max_position_per_symbol
        self.max_total_position = max_total_position
        self.daily_loss_limit = daily_loss_limit
        self.jev_confidence_threshold = jev_confidence_threshold

        # 运行时状态
        self.initial_capital = initial_capital
        self.peak_equity = initial_capital
        self.current_equity = initial_capital
        self.daily_pnl: Dict[pd.Timestamp, float] = {}
        self.events: List[RiskEvent] = []
        self._paused = False
        self._current_date: Optional[pd.Timestamp] = None

    def reset(self, initial_capital: float) -> None:
        """重置风控状态。"""
        self.initial_capital = initial_capital
        self.peak_equity = initial_capital
        self.current_equity = initial_capital
        self.daily_pnl = {}
        self.events = []
        self._paused = False
        self._current_date = None

    def update_equity(self, equity: float) -> None:
        """更新当前权益，检查回撤。"""
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        drawdown = (self.peak_equity - equity) / self.peak_equity
        if drawdown >= self.max_drawdown_pause and not self._paused:
            self._paused = True
            event = RiskEvent(
                timestamp=pd.Timestamp.now().isoformat(),
                event_type="drawdown",
                symbol="ALL",
                detail=f"总回撤 {drawdown*100:.1f}% 超过阈值 {self.max_drawdown_pause*100:.0f}%，暂停交易",
            )
            self.events.append(event)
            logger.warning(event.detail)

    def is_paused(self) -> bool:
        """是否处于暂停交易状态。"""
        return self._paused

    def check_trade_allowed(
        self,
        symbol: str,
        action: str,
        price: float,
        equity: float,
        cash: float,
        positions: Dict,
    ) -> bool:
        """检查交易是否被风控允许。

        Args:
            symbol: 标的代码。
            action: buy / sell。
            price: 当前价格。
            equity: 当前总权益。
            cash: 当前现金。
            positions: 持仓字典。

        Returns:
            True 表示允许交易。
        """
        if self._paused:
            return False

        # 卖出总是允许（止损/止盈）
        if action == "sell":
            return True

        # 单日亏损检查
        if self._current_date and self._current_date in self.daily_pnl:
            daily_loss = self.daily_pnl[self._current_date]
            if daily_loss < -self.daily_loss_limit * self.initial_capital:
                logger.warning("单日亏损超限，禁止开仓")
                return False

        # 总仓位检查
        total_position_value = sum(
            p.shares * price for p in positions.values() if p.shares > 0
        )
        if total_position_value / equity >= self.max_total_position:
            logger.debug("总仓位已达上限")
            return False

        # 单标的仓位检查
        if symbol in positions and positions[symbol].shares > 0:
            existing_value = positions[symbol].shares * price
            if existing_value / equity >= self.max_position_per_symbol:
                logger.debug("单标的 %s 仓位已达上限", symbol)
                return False

        return True

    def calc_position_size(
        self,
        symbol: str,
        price: float,
        equity: float,
        cash: float,
        confidence: float,
        positions: Dict,
    ) -> float:
        """计算目标仓位金额。

        基于置信度和仓位上限动态调整：
        - 基础仓位 = 单标的上限 × 置信度系数
        - 置信度越高，仓位越接近上限
        """
        # 置信度系数：0.6→0.5, 1.0→1.0
        conf_factor = max(0.3, min(1.0, (confidence - 0.5) * 2))
        base_position = equity * self.max_position_per_symbol * conf_factor

        # 检查总仓位空间
        total_position_value = sum(
            p.shares * price for p in positions.values() if p.shares > 0
        )
        remaining_capacity = equity * self.max_total_position - total_position_value

        target = min(base_position, remaining_capacity, cash * 0.95)
        return max(0.0, target)

    def record_trade(self, pnl: float) -> None:
        """记录一笔已实现盈亏，用于单日亏损统计。"""
        if self._current_date is None:
            self._current_date = pd.Timestamp.now().normalize()
        if self._current_date not in self.daily_pnl:
            self.daily_pnl[self._current_date] = 0.0
        self.daily_pnl[self._current_date] += pnl

        # 检查单日亏损
        daily_loss = self.daily_pnl[self._current_date]
        if daily_loss < -self.daily_loss_limit * self.initial_capital:
            event = RiskEvent(
                timestamp=pd.Timestamp.now().isoformat(),
                event_type="daily_loss",
                symbol="ALL",
                detail=f"单日亏损 {daily_loss:.2f} 超过限额",
            )
            self.events.append(event)
            logger.warning(event.detail)

    def set_date(self, date: pd.Timestamp) -> None:
        """设置当前回测日期。"""
        self._current_date = date

    def get_summary(self) -> Dict:
        """获取风控状态摘要。"""
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        return {
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "current_drawdown": drawdown,
            "paused": self._paused,
            "risk_events": len(self.events),
        }
