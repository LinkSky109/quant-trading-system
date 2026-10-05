"""多策略组合引擎（同一标的、多策略资金分配 + 组合回测）。

与 :mod:`backtest.portfolio_engine`（多标的、单策略）互补，本模块面向
**同一标的上叠加多个交易策略**的场景：

- 按权重把组合初始资金切分为若干「子账户」，每个子账户分配给一个策略；
- 每个策略在自己的子账户内独立运行一次 :class:`BacktestEngine`（资金 =
  总资金 × 权重），互不干扰；
- 组合净值 = 各子账户净值按日对齐后求和（首日 = 组合初始资金）；
- 组合级绩效基于合并净值 + 全部成交记录计算（复用
  :func:`backtest.metrics.calc_all_metrics`）。

信号冲突处理说明
----------------
由于资金已按权重隔离到各策略子账户，**各策略独立交易，天然不会在执行层面
发生冲突**（一个策略买、另一个策略卖只是各自子账户的独立决策，不存在同一笔
资金的争抢）。本类同时提供 ``conflict_log``：逐日扫描各策略的方向投票
（buy=+1 / sell=-1 / hold=0），当某日同时存在买入与卖出信号时记录下来，
并按 ``net_vote = Σ(weight_i · direction_i)`` 计算一个「中心化投票」参考结论
（净投票 > threshold 视为买入、< -threshold 视为卖出、否则观望）。该日志仅作
分析参考，不改变各子账户的实际交易。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.engine import BacktestEngine, BacktestResult
from backtest.metrics import calc_all_metrics
from utils.indicators import add_indicators

logger = logging.getLogger(__name__)

# buy=+1, sell=-1, hold=0 的方向投票取值
_DIR_BUY = 1
_DIR_SELL = -1
_DIR_HOLD = 0


@dataclass
class StrategyPortfolioResult:
    """多策略组合回测结果容器。

    Attributes:
        portfolio_equity_curve: 组合级净值曲线（pd.Series，索引为交易日），
            首日 = 组合初始资金。
        portfolio_metrics: 组合级绩效指标（键与 ``calc_all_metrics`` 一致）。
        strategy_results: 每个策略独立回测的 :class:`BacktestResult`，
            ``{strategy_name: BacktestResult}``。
        strategy_weights: 实际资金分配权重 ``{strategy_name: weight}``，和为 1。
        contribution: 策略贡献度分析
            ``{strategy_name: {pnl, return_pct, contribution_pct}}``。
        conflict_log: 信号冲突日志（List[Dict]），每条记录一个冲突日的
            方向投票与中心化参考结论。
    """

    portfolio_equity_curve: pd.Series
    portfolio_metrics: Dict[str, float]
    strategy_results: Dict[str, BacktestResult]
    strategy_weights: Dict[str, float]
    contribution: Dict[str, Dict[str, float]] = field(default_factory=dict)
    conflict_log: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 策略名 -> 策略类映射（与 web-dashboard/server.py 中 _strategy_instance 一致）
# ---------------------------------------------------------------------------

_STRATEGY_CLASS_MAP: Dict[str, type] = {}


def _get_strategy_class(name: str) -> type:
    """根据策略名称获取策略类（延迟导入，避免循环依赖）。"""
    if not _STRATEGY_CLASS_MAP:
        from strategies.ma_cross import MACrossStrategy
        from strategies.bollinger import BollingerStrategy
        from strategies.momentum_breakout import MomentumBreakoutStrategy
        from strategies.rsi import RSIStrategy
        from strategies.macd import MACDStrategy
        from strategies.grid_trading import GridTradingStrategy

        _STRATEGY_CLASS_MAP.update({
            "ma_cross": MACrossStrategy,
            "bollinger": BollingerStrategy,
            "momentum_breakout": MomentumBreakoutStrategy,
            "rsi": RSIStrategy,
            "macd": MACDStrategy,
            "grid_trading": GridTradingStrategy,
        })
    cls = _STRATEGY_CLASS_MAP.get(name)
    if cls is None:
        raise ValueError(f"未知策略: {name}，可选: {sorted(_STRATEGY_CLASS_MAP)}")
    return cls


class StrategyPortfolio:
    """多策略资金分配 + 组合回测引擎（同一标的）。

    Args:
        strategies: 策略名称列表，如 ``["ma_cross", "bollinger", "momentum_breakout"]``。
        weights: 各策略资金分配权重，长度须与 ``strategies`` 一致；
            为 ``None`` 时等权（各 1/N）。提供后会自动归一化使权重和为 1。
        initial_capital: 组合总初始资金。
        conflict_threshold: 信号冲突投票阈值，净投票 > 阈值参考为买入、
            < -阈值参考为卖出，否则观望。
        **engine_kwargs: 透传给 :class:`BacktestEngine` 的引擎参数
            （commission_rate / stamp_tax_rate / slippage_rate /
            risk_free_rate / trading_days 等）。

    Raises:
        ValueError: 策略列表为空、权重长度不匹配或权重全非正。
    """

    def __init__(
        self,
        strategies: List[str],
        weights: Optional[List[float]] = None,
        initial_capital: float = 1_000_000.0,
        conflict_threshold: float = 0.1,
        **engine_kwargs: Any,
    ):
        if not strategies:
            raise ValueError("strategies 不能为空列表，至少需要 1 个策略")
        # 去重保序
        self.strategies: List[str] = list(dict.fromkeys(strategies))
        if len(self.strategies) != len(strategies):
            logger.warning("strategies 中存在重复策略，已去重: %s", self.strategies)

        if weights is not None and len(weights) != len(strategies):
            raise ValueError(
                f"weights 长度({len(weights)})必须与 strategies 长度({len(strategies)})一致"
            )

        # 计算归一化权重
        self.strategy_weights: Dict[str, float] = self._normalize_weights(
            self.strategies, weights
        )

        self.initial_capital = float(initial_capital)
        self.conflict_threshold = float(conflict_threshold)
        self.engine_kwargs: Dict[str, Any] = dict(engine_kwargs)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self, data: pd.DataFrame, symbol: str = "") -> StrategyPortfolioResult:
        """执行多策略组合回测。

        Args:
            data: 标的日线数据，DataFrame，index 为 pd.Timestamp，
                需含 open/high/low/close/volume 列。
            symbol: 标的代码（仅用于日志/信号元数据）。

        Returns:
            StrategyPortfolioResult 组合回测结果。
        """
        # 空数据：返回空结果，不崩溃
        if data is None or len(data) == 0:
            empty_idx = pd.DatetimeIndex([], name="date")
            return StrategyPortfolioResult(
                portfolio_equity_curve=pd.Series(
                    dtype=float, index=empty_idx, name="portfolio_equity"
                ),
                portfolio_metrics=calc_all_metrics(
                    pd.Series(dtype=float), [],
                    self.engine_kwargs.get("risk_free_rate", 0.02),
                    self.engine_kwargs.get("trading_days", 252),
                ),
                strategy_results={},
                strategy_weights=self.strategy_weights,
                contribution={},
                conflict_log=[],
            )

        # 预处理指标（与 BacktestEngine 内部一致，供冲突扫描取信号）
        processed = add_indicators(data.copy())

        # 1. 逐策略分配资金并独立回测
        strategy_results: Dict[str, BacktestResult] = {}
        signal_map: Dict[str, pd.DataFrame] = {}
        for name in self.strategies:
            alloc_capital = self.initial_capital * self.strategy_weights[name]
            strategy = _get_strategy_class(name)()  # 默认参数实例化
            engine = BacktestEngine(
                initial_capital=alloc_capital,
                **self.engine_kwargs,
            )
            result = engine.run(data, strategy, symbol=symbol)
            strategy_results[name] = result
            # 取（已 shift 的）信号序列用于冲突扫描
            signal_map[name] = strategy.get_signal_dataframe(processed, symbol)

        # 2. 合并各子账户净值曲线 -> 组合净值
        portfolio_equity = self._merge_equity_curves(strategy_results)

        # 3. 组合级绩效：合并全部成交记录
        all_trades: List[Dict[str, Any]] = []
        for name in self.strategies:
            all_trades.extend(t.__dict__ for t in strategy_results[name].trades)
        portfolio_metrics = calc_all_metrics(
            portfolio_equity,
            all_trades,
            self.engine_kwargs.get("risk_free_rate", 0.02),
            self.engine_kwargs.get("trading_days", 252),
        )

        # 4. 信号冲突日志
        conflict_log = self._build_conflict_log(signal_map)

        # 5. 策略贡献度
        contribution = self._build_contribution(strategy_results)

        logger.info(
            "多策略组合回测完成: %d 个策略 %s, 权重=%s, 累计收益 %.2f%%, 最大回撤 %.2f%%, "
            "冲突日 %d 个",
            len(self.strategies), symbol,
            {k: round(v, 4) for k, v in self.strategy_weights.items()},
            portfolio_metrics["累计收益率"] * 100,
            portfolio_metrics["最大回撤"] * 100,
            len(conflict_log),
        )

        return StrategyPortfolioResult(
            portfolio_equity_curve=portfolio_equity,
            portfolio_metrics=portfolio_metrics,
            strategy_results=strategy_results,
            strategy_weights=self.strategy_weights,
            contribution=contribution,
            conflict_log=conflict_log,
        )

    # ------------------------------------------------------------------
    # 权重
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_weights(
        strategies: List[str], weights: Optional[List[float]]
    ) -> Dict[str, float]:
        """计算归一化权重（和为 1）。"""
        n = len(strategies)
        if weights is None:
            w = 1.0 / n
            return {s: w for s in strategies}

        raw = {s: float(w) for s, w in zip(strategies, weights)}
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("weights 归一化前总和必须大于 0")
        return {s: raw[s] / total for s in strategies}

    # ------------------------------------------------------------------
    # 净值合并
    # ------------------------------------------------------------------

    def _merge_equity_curves(
        self, strategy_results: Dict[str, BacktestResult]
    ) -> pd.Series:
        """按日对齐各子账户净值曲线后求和。

        各子账户初始资金 = 组合总资金 × 权重，故求和后首日 = 组合初始资金。
        同一标的数据下各策略交易日一致；这里仍做并集对齐 + ffill/bill 兜底。
        """
        equity_map = {
            name: res.equity_curve.astype(float)
            for name, res in strategy_results.items()
        }
        union_index = sorted(set().union(*[set(s.index) for s in equity_map.values()]))
        if not union_index:
            return pd.Series(dtype=float, name="portfolio_equity")

        aligned = pd.DataFrame(index=pd.DatetimeIndex(union_index, name="date"))
        for name, eq in equity_map.items():
            aligned[name] = eq.reindex(aligned.index).ffill().bfill()

        portfolio = aligned.sum(axis=1)
        portfolio.name = "portfolio_equity"
        return portfolio

    # ------------------------------------------------------------------
    # 冲突日志
    # ------------------------------------------------------------------

    def _build_conflict_log(self, signal_map: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """逐日扫描各策略方向投票，记录存在买/卖分歧的日期。

        direction: buy=+1, sell=-1, hold=0。净投票 = Σ(weight_i · direction_i)。
        """
        # 汇总所有交易日
        union_index = sorted(set().union(*[set(df.index) for df in signal_map.values()]))
        if not union_index:
            return []

        # 构造方向矩阵：行=日期，列=策略，值=+1/-1/0
        dir_matrix = pd.DataFrame(index=pd.DatetimeIndex(union_index, name="date"))
        for name, sig_df in signal_map.items():
            sig = sig_df["signal"].reindex(dir_matrix.index).fillna(0.0)
            dir_matrix[name] = pd.Series(
                [
                    _DIR_BUY if v > 0 else (_DIR_SELL if v < 0 else _DIR_HOLD)
                    for v in sig
                ],
                index=dir_matrix.index,
            )

        conflict_log: List[Dict[str, Any]] = []
        for date, row in dir_matrix.iterrows():
            directions = {name: int(row[name]) for name in dir_matrix.columns}
            has_buy = any(d == _DIR_BUY for d in directions.values())
            has_sell = any(d == _DIR_SELL for d in directions.values())
            # 仅记录同时存在买入与卖出信号的分歧日
            if not (has_buy and has_sell):
                continue

            net_vote = sum(
                self.strategy_weights[name] * directions[name]
                for name in directions
            )
            if net_vote > self.conflict_threshold:
                decision = "buy"
            elif net_vote < -self.conflict_threshold:
                decision = "sell"
            else:
                decision = "hold"

            conflict_log.append({
                "date": date.strftime("%Y-%m-%d"),
                "directions": directions,
                "weights": dict(self.strategy_weights),
                "net_vote": round(float(net_vote), 4),
                "decision": decision,
                "note": "信号分歧日：各子账户独立交易，此为中心化投票参考结论",
            })
        return conflict_log

    # ------------------------------------------------------------------
    # 贡献度
    # ------------------------------------------------------------------

    def _build_contribution(
        self, strategy_results: Dict[str, BacktestResult]
    ) -> Dict[str, Dict[str, float]]:
        """计算每个策略的累计盈亏、收益率、对组合总收益的贡献占比。"""
        contribution: Dict[str, Dict[str, float]] = {}
        total_pnl = 0.0
        pnls: Dict[str, float] = {}
        for name, res in strategy_results.items():
            eq = res.equity_curve
            if len(eq) < 2:
                pnls[name] = 0.0
                continue
            alloc = float(eq.iloc[0])
            pnl = float(eq.iloc[-1] - eq.iloc[0])
            pnls[name] = pnl
            total_pnl += pnl

        for name, pnl in pnls.items():
            res = strategy_results[name]
            eq = res.equity_curve
            alloc = float(eq.iloc[0]) if len(eq) >= 1 else 0.0
            return_pct = (pnl / alloc) if alloc > 0 else 0.0
            contribution_pct = (pnl / total_pnl) if total_pnl != 0 else 0.0
            contribution[name] = {
                "pnl": round(pnl, 2),
                "return_pct": round(return_pct, 6),
                "contribution_pct": round(contribution_pct, 6),
            }
        return contribution
