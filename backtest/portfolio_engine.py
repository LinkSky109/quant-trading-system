"""组合级回测引擎。

在单标的 :class:`BacktestEngine` 之上构建多标的组合回测：
- 每只标的独立分配初始资金、独立运行事件驱动回测（复用 BacktestEngine，不重写回测逻辑）
- 支持等权 / 波动率倒数 / 自定义权重三种资金分配方式
- 按日期对齐各标的净值曲线后资金加权求和，得到组合级净值
- 组合级绩效基于合并后的净值曲线与全部成交记录计算
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine, BacktestResult, Trade
from backtest.metrics import calc_all_metrics
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class PortfolioBacktestResult:
    """组合回测结果容器。

    Attributes:
        portfolio_equity_curve: 组合级净值曲线（pd.Series，索引为交易日）。
        portfolio_metrics: 组合级绩效指标（键与 ``calc_all_metrics`` 一致）。
        symbol_results: 每只标的的独立 :class:`BacktestResult`。
        symbol_metrics: 每只标的的绩效明细 {symbol: metrics}。
        symbol_weights: 实际仓位（资金）分配权重 {symbol: weight}，和为 1。
        benchmark_curve: 基准曲线（等权买入持有，规模 = 初始资金）。
        all_trades: 所有标的成交记录合并列表。
    """

    portfolio_equity_curve: pd.Series
    portfolio_metrics: Dict[str, float]
    symbol_results: Dict[str, BacktestResult]
    symbol_metrics: Dict[str, Dict[str, float]]
    symbol_weights: Dict[str, float]
    benchmark_curve: pd.Series
    all_trades: List[Trade] = field(default_factory=list)


class PortfolioBacktestEngine:
    """多标的组合回测引擎。

    对每只标的分配独立资金并各自运行一次 :class:`BacktestEngine`，
    再将各标的净值曲线按交易日对齐后求和，得到组合净值。

    Args:
        initial_capital: 组合总初始资金。
        commission_rate: 佣金费率。
        stamp_tax_rate: 印花税率（卖出收取）。
        slippage_rate: 滑点费率。
        risk_free_rate: 无风险利率（年化）。
        trading_days: 年交易日数。
        allocation_method: 资金分配方式，
            ``equal`` 等权 / ``volatility_inverse`` 波动率倒数 / ``custom`` 自定义权重 /
            ``min_variance`` 最小方差 / ``risk_parity`` 风险平价 /
            ``mean_variance`` 均值方差（最大夏普）。
            后三者由 :class:`~optimization.portfolio_optimizer.PortfolioOptimizer` 求解，
            long-only 且单标的上限 0.3。
        custom_weights: 自定义权重 ``{symbol: weight}``，``allocation_method="custom"``
            时必填；内部会自动归一化使权重和为 1。

    Raises:
        ValueError: ``allocation_method`` 非法，或 ``custom`` 模式下未提供权重。
    """

    _ALLOWED_METHODS = (
        "equal", "volatility_inverse", "custom",
        "min_variance", "risk_parity", "mean_variance",
    )

    #: 优化类方法（交给 PortfolioOptimizer 求解）
    _OPTIM_METHODS = ("min_variance", "risk_parity", "mean_variance")

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.0005,
        slippage_rate: float = 0.001,
        risk_free_rate: float = 0.02,
        trading_days: int = 252,
        allocation_method: str = "equal",
        custom_weights: Optional[Dict[str, float]] = None,
    ):
        if allocation_method not in self._ALLOWED_METHODS:
            raise ValueError(
                f"allocation_method 必须为 {self._ALLOWED_METHODS} 之一，"
                f"收到: {allocation_method}"
            )
        if allocation_method == "custom" and not custom_weights:
            raise ValueError("allocation_method='custom' 时必须提供 custom_weights")

        self.initial_capital = float(initial_capital)
        self.commission_rate = float(commission_rate)
        self.stamp_tax_rate = float(stamp_tax_rate)
        self.slippage_rate = float(slippage_rate)
        self.risk_free_rate = float(risk_free_rate)
        self.trading_days = int(trading_days)
        self.allocation_method = allocation_method
        self.custom_weights = custom_weights

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        strategy: BaseStrategy,
    ) -> PortfolioBacktestResult:
        """执行组合回测。

        Args:
            data: 多标的行情数据 ``{symbol: df}``，每个 df 需含 open/high/low/close。
            strategy: 策略实例，所有标的共用（策略本身无跨标的可变状态）。

        Returns:
            组合回测结果 :class:`PortfolioBacktestResult`。
        """
        symbols = [s for s, df in (data or {}).items() if df is not None and len(df) > 0]

        # 空输入：返回空结果，不崩溃
        if not symbols:
            empty_idx = pd.DatetimeIndex([], name="date")
            return PortfolioBacktestResult(
                portfolio_equity_curve=pd.Series(dtype=float, index=empty_idx, name="portfolio_equity"),
                portfolio_metrics=calc_all_metrics(
                    pd.Series(dtype=float), [], self.risk_free_rate, self.trading_days
                ),
                symbol_results={},
                symbol_metrics={},
                symbol_weights={},
                benchmark_curve=pd.Series(dtype=float, index=empty_idx, name="benchmark"),
                all_trades=[],
            )

        # 1. 计算每只标的的资金分配权重
        weights = self._compute_weights({s: data[s] for s in symbols})

        # 2. 逐标的独立运行回测（分配独立初始资金）
        symbol_results: Dict[str, BacktestResult] = {}
        symbol_metrics: Dict[str, Dict[str, float]] = {}
        for sym in symbols:
            alloc_capital = self.initial_capital * weights[sym]
            engine = BacktestEngine(
                initial_capital=alloc_capital,
                commission_rate=self.commission_rate,
                stamp_tax_rate=self.stamp_tax_rate,
                slippage_rate=self.slippage_rate,
                risk_free_rate=self.risk_free_rate,
                trading_days=self.trading_days,
            )
            result = engine.run({sym: data[sym]}, strategy, symbol=sym)
            symbol_results[sym] = result
            symbol_metrics[sym] = result.metrics

        # 3. 合并各标的净值曲线 -> 组合净值
        portfolio_equity = self._merge_equity_curves(symbol_results, weights)

        # 4. 合并全部成交记录，计算组合级绩效
        all_trades: List[Trade] = []
        for sym in symbols:
            all_trades.extend(symbol_results[sym].trades)
        portfolio_metrics = calc_all_metrics(
            portfolio_equity,
            [t.__dict__ for t in all_trades],
            self.risk_free_rate,
            self.trading_days,
        )

        # 5. 基准：等权买入持有所有标的
        benchmark = self._build_benchmark({s: data[s] for s in symbols})

        logger.info(
            "组合回测完成: %d 只标的, 权重=%s, 累计收益 %.2f%%, 最大回撤 %.2f%%",
            len(symbols),
            {k: round(v, 4) for k, v in weights.items()},
            portfolio_metrics["累计收益率"] * 100,
            portfolio_metrics["最大回撤"] * 100,
        )

        return PortfolioBacktestResult(
            portfolio_equity_curve=portfolio_equity,
            portfolio_metrics=portfolio_metrics,
            symbol_results=symbol_results,
            symbol_metrics=symbol_metrics,
            symbol_weights=weights,
            benchmark_curve=benchmark,
            all_trades=all_trades,
        )

    # ------------------------------------------------------------------
    # 资金分配
    # ------------------------------------------------------------------

    def _compute_weights(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """根据 allocation_method 计算各标的权重（和为 1）。"""
        symbols = list(data.keys())

        if self.allocation_method == "equal":
            w = 1.0 / len(symbols)
            return {s: w for s in symbols}

        if self.allocation_method in self._OPTIM_METHODS:
            # 交给 PortfolioOptimizer：最小方差 / 风险平价 / 均值方差
            from optimization.portfolio_optimizer import PortfolioOptimizer
            optimizer = PortfolioOptimizer(
                risk_free_rate=self.risk_free_rate,
                max_weight=0.3,
                trading_days=self.trading_days,
            )
            result = optimizer.optimize(symbols, data, method=self.allocation_method)
            logger.info(
                "组合优化[%s]: 预期收益 %.2f%%, 波动 %.2f%%, 夏普 %.3f",
                self.allocation_method,
                result.expected_return * 100,
                result.expected_volatility * 100,
                result.sharpe,
            )
            return result.weights

        if self.allocation_method == "volatility_inverse":
            inv_vol: Dict[str, float] = {}
            for s in symbols:
                vol = self._estimate_volatility(data[s])
                # 波动率为 0 时给一个极小值，避免除零
                inv_vol[s] = 1.0 / max(vol, 1e-8)
            total = sum(inv_vol.values())
            return {s: inv_vol[s] / total for s in symbols}

        # custom：使用外部权重并归一化
        raw = {s: float(self.custom_weights.get(s, 0.0)) for s in symbols}
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("custom_weights 归一化前总和必须大于 0")
        return {s: raw[s] / total for s in symbols}

    @staticmethod
    def _estimate_volatility(df: pd.DataFrame) -> float:
        """用收盘价日收益率标准差作为波动率估计。"""
        close = df["close"].astype(float)
        daily_ret = close.pct_change().dropna()
        if len(daily_ret) < 2:
            return 0.0
        return float(daily_ret.std())

    # ------------------------------------------------------------------
    # 组合净值与基准
    # ------------------------------------------------------------------

    def _merge_equity_curves(
        self,
        symbol_results: Dict[str, BacktestResult],
        weights: Dict[str, float],
    ) -> pd.Series:
        """按日期对齐各标的净值曲线后求和。

        - 以所有标的交易日并集为时间轴；
        - 某标的在其首日之前尚未运行，按分配到的初始资金（= 其净值首日值）填充；
        - 某标的在其最后一日之后不再交易，保持最后净值（ffill）。
        组合净值 = 各标的净值之和，首日 = 组合初始资金。
        """
        # 全部交易日并集
        union_index = sorted(set().union(*[
            set(r.equity_curve.index) for r in symbol_results.values()
        ]))
        if not union_index:
            return pd.Series(dtype=float, name="portfolio_equity")

        aligned = pd.DataFrame(index=pd.DatetimeIndex(union_index, name="date"))
        for sym, result in symbol_results.items():
            eq = result.equity_curve.astype(float)
            # reindex 到并集，ffill 填充尾部空缺，bfill 填充首日之前的空缺（持有分配资金）
            aligned[sym] = eq.reindex(aligned.index).ffill().bfill()

        portfolio = aligned.sum(axis=1)
        portfolio.name = "portfolio_equity"
        return portfolio

    def _build_benchmark(self, data: Dict[str, pd.DataFrame]) -> pd.Series:
        """构建组合基准：所有标的等权买入持有。

        每只标的收盘价归一化为首日=1，等权平均后按组合初始资金缩放。
        """
        # 全部交易日并集
        union_index = sorted(set().union(*[set(df.index) for df in data.values()]))
        if not union_index:
            return pd.Series(dtype=float, name="benchmark")

        normalized: List[pd.Series] = []
        for df in data.values():
            close = df["close"].astype(float).reindex(union_index).ffill().bfill()
            base = close.iloc[0]
            if base and not np.isnan(base):
                normalized.append(close / base)

        if not normalized:
            return pd.Series(dtype=float, index=pd.DatetimeIndex(union_index), name="benchmark")

        avg = pd.concat(normalized, axis=1).mean(axis=1)
        benchmark = avg * self.initial_capital
        benchmark.name = "benchmark"
        return benchmark
