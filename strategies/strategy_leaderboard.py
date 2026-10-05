"""策略排行榜与多策略信号聚合。

对同一标的批量回测全部策略，按「风险调整后收益 + 收益 + 回撤 + 胜率」
的加权综合得分排名；并基于排名权重对各策略最新信号做加权投票，
生成综合交易建议（buy / sell / hold）。

综合得分公式::

    score = 0.3 * 归一化(夏普)
          + 0.3 * 归一化(累计收益率)
          + 0.2 * 归一化(1 - |最大回撤|)
          + 0.2 * 归一化(胜率)

其中归一化采用 min-max 缩放，全相等时取 0.5。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.engine import BacktestEngine
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

# 策略名 -> 策略类（与 web-dashboard/server.py 中保持一致）
_STRATEGY_CLASS_MAP: Dict[str, type] = {}


def _get_strategy_class(name: str) -> type:
    """根据策略名获取策略类（延迟导入，避免循环依赖）。"""
    if not _STRATEGY_CLASS_MAP:
        from strategies.ma_cross import MACrossStrategy
        from strategies.bollinger import BollingerStrategy
        from strategies.momentum_breakout import MomentumBreakoutStrategy
        from strategies.rsi import RSIStrategy
        from strategies.macd import MACDStrategy
        from strategies.grid_trading import GridTradingStrategy
        from strategies.indicator_combo import IndicatorComboStrategy

        _STRATEGY_CLASS_MAP.update({
            "ma_cross": MACrossStrategy,
            "bollinger": BollingerStrategy,
            "momentum_breakout": MomentumBreakoutStrategy,
            "rsi": RSIStrategy,
            "macd": MACDStrategy,
            "grid_trading": GridTradingStrategy,
            "indicator_combo": IndicatorComboStrategy,
        })
    cls = _STRATEGY_CLASS_MAP.get(name)
    if cls is None:
        raise ValueError(f"未知策略: {name}，可选: {sorted(_STRATEGY_CLASS_MAP)}")
    return cls


# 综合得分权重
W_SHARPE = 0.3
W_RETURN = 0.3
W_DRAWDOWN = 0.2
W_WINRATE = 0.2

# 排名 -> 投票基础权重（第1名最高，第6名最低）
RANK_WEIGHTS: Dict[int, float] = {
    1: 3.0,
    2: 2.5,
    3: 2.0,
    4: 1.5,
    5: 1.0,
    6: 0.5,
}
_DEFAULT_WEIGHT = 1.0  # 无排行榜数据时的等权

# 动态权重调整：近期表现好/差 的乘子
_DYNAMIC_BETTER = 1.2
_DYNAMIC_WORSE = 0.8


class StrategyLeaderboard:
    """策略表现排行榜 + 多策略信号聚合。

    Args:
        data_fetcher: 数据获取器，需实现 ``get_klines(symbol, start_date, end_date)``，
            返回带 open/high/low/close/volume 列、日期索引的 DataFrame。
            测试时可传入 mock 对象；为 None 时回测需自行提供数据（见 run_backtests）。
        config: 系统配置字典；为 None 时从 ``config/config.yaml`` 加载。
    """

    def __init__(self, data_fetcher: Any = None, config: Dict | None = None):
        if config is None:
            from config import load_config
            config = load_config()
        self.config: Dict[str, Any] = config or {}
        self.data_fetcher: Any = data_fetcher

        # 解析策略配置并实例化
        strategies_cfg = self.config.get("strategies", {}) or {}
        self._strategy_names: List[str] = []
        self._labels: Dict[str, str] = {}
        self._strategies: Dict[str, BaseStrategy] = {}
        for name, cfg in strategies_cfg.items():
            cfg = cfg or {}
            if not cfg.get("enabled", True):
                continue
            params = {k: v for k, v in cfg.items()
                      if k not in ("enabled", "label", "description")}
            self._strategies[name] = _get_strategy_class(name)(params)
            self._strategy_names.append(name)
            self._labels[name] = cfg.get("label", name)

        # 回测缓存：key = (symbol, start_date, end_date)
        self._bt_cache: Dict[tuple, List[Dict[str, Any]]] = {}
        # 最近一次排名结果（按综合得分降序），供信号聚合加权使用
        self._ranking: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 批量回测
    # ------------------------------------------------------------------

    def run_backtests(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000.0,
    ) -> List[Dict[str, Any]]:
        """对指定标的批量回测所有策略。

        Args:
            symbol: 标的代码。
            start_date: 起始日期 YYYY-MM-DD。
            end_date: 结束日期 YYYY-MM-DD。
            initial_capital: 每个策略独立回测的初始资金。

        Returns:
            每个策略一项的指标列表，字段：
            ``strategy`` / ``label`` / ``metrics``(完整指标字典) /
            ``sharpe`` / ``total_return`` / ``max_drawdown`` / ``win_rate``。
            相同 ``(symbol, start_date, end_date)`` 直接命中缓存。
        """
        cache_key = (symbol, start_date, end_date)
        if cache_key in self._bt_cache:
            logger.debug("回测缓存命中: %s", cache_key)
            return self._bt_cache[cache_key]

        if self.data_fetcher is None:
            raise ValueError(
                "data_fetcher 未配置，无法获取K线数据；请在构造时传入 DataFetcher 或 mock 对象"
            )
        df = self.data_fetcher.get_klines(
            symbol, start_date=start_date, end_date=end_date,
        )
        if df is None or len(df) < 30:
            raise ValueError(f"{symbol} 在 {start_date}~{end_date} 区间数据不足（<30 根K线）")

        bt_cfg = self.config.get("backtest", {}) or {}
        results: List[Dict[str, Any]] = []
        for name in self._strategy_names:
            strategy = self._strategies[name]
            engine = BacktestEngine(
                initial_capital=initial_capital,
                commission_rate=float(bt_cfg.get("commission_rate", 0.00025)),
                stamp_tax_rate=float(bt_cfg.get("stamp_tax_rate", 0.0005)),
                slippage_rate=float(bt_cfg.get("slippage_rate", 0.001)),
                risk_free_rate=float(bt_cfg.get("risk_free_rate", 0.02)),
                trading_days=int(bt_cfg.get("trading_days_per_year", 252)),
            )
            try:
                res = engine.run(df, strategy, symbol=symbol)
                metrics = res.metrics
            except Exception as e:  # 单个策略失败不影响整体排行
                logger.exception("策略 %s 回测失败", name)
                metrics = {}
            results.append({
                "strategy": name,
                "label": self._labels.get(name, name),
                "metrics": metrics,
                "sharpe": float(metrics.get("夏普比率", 0.0)),
                "total_return": float(metrics.get("累计收益率", 0.0)),
                # 统一为正的回撤幅度（越大越差），便于归一化比较
                "max_drawdown": abs(float(metrics.get("最大回撤", 0.0))),
                "win_rate": float(metrics.get("胜率", 0.0)),
            })

        self._bt_cache[cache_key] = results
        return results

    # ------------------------------------------------------------------
    # 排名
    # ------------------------------------------------------------------

    def rank_strategies(self, backtest_results: List[Dict]) -> List[Dict[str, Any]]:
        """按综合得分对回测结果排名。

        综合得分 = 0.3*夏普归一化 + 0.3*收益归一化
                 + 0.2*(1-回撤)归一化 + 0.2*胜率归一化。

        Returns:
            按综合得分降序排列的列表，每项增加 ``score`` 与 ``rank`` 字段。
        """
        if not backtest_results:
            self._ranking = []
            return []

        sharpes = [float(r.get("sharpe", 0.0)) for r in backtest_results]
        returns = [float(r.get("total_return", 0.0)) for r in backtest_results]
        # 回撤质量：回撤越小越好，取 1 - |回撤|
        dd_quality = [1.0 - abs(float(r.get("max_drawdown", 0.0))) for r in backtest_results]
        winrates = [float(r.get("win_rate", 0.0)) for r in backtest_results]

        n_sharpe = self._normalize(sharpes)
        n_return = self._normalize(returns)
        n_dd = self._normalize(dd_quality)
        n_win = self._normalize(winrates)

        scored: List[Dict[str, Any]] = []
        for i, r in enumerate(backtest_results):
            score = (
                W_SHARPE * n_sharpe[i]
                + W_RETURN * n_return[i]
                + W_DRAWDOWN * n_dd[i]
                + W_WINRATE * n_win[i]
            )
            item = dict(r)
            item["score"] = round(float(score), 6)
            scored.append(item)

        scored.sort(key=lambda x: x["score"], reverse=True)
        for rank, item in enumerate(scored, start=1):
            item["rank"] = rank

        self._ranking = scored
        return scored

    def get_leaderboard(
        self, symbol: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """一站式：回测 + 排名，返回按综合得分降序的排行榜。"""
        results = self.run_backtests(symbol, start_date, end_date)
        return self.rank_strategies(results)

    # ------------------------------------------------------------------
    # 信号聚合
    # ------------------------------------------------------------------

    def aggregate_signals(
        self,
        symbol: str,
        df: pd.DataFrame | None = None,
        threshold: float = 0.2,
    ) -> Dict[str, Any]:
        """多策略信号加权投票聚合。

        1. 取各策略最新一根K线上的信号（买入/卖出/观望）。
        2. 按排行榜名次赋基础权重：第1名=3.0 ... 第6名=0.5；无排行榜时等权 1.0。
        3. 动态调整：近 20 日收益高于中位数的策略权重 ×1.2，低于中位数 ×0.8。
        4. 净信号 = 买入权重和 - 卖出权重和。
        5. 净信号 > threshold → buy；< -threshold → sell；否则 hold。

        Args:
            symbol: 标的代码。
            df: 行情数据；为 None 时通过 data_fetcher 拉取最新数据。
            threshold: 净信号触发阈值。

        Returns:
            聚合结果字典，含 ``aggregate_signal`` / ``net_score`` / ``threshold`` /
            ``strategy_details`` / ``weighted_buy`` / ``weighted_sell``。
        """
        if df is None:
            if self.data_fetcher is None:
                raise ValueError("df 与 data_fetcher 均为空，无法获取行情数据")
            df = self.data_fetcher.get_klines(symbol, count=250)

        # 排名 -> 基础权重映射
        base_weight_by_strategy: Dict[str, float] = {}
        if self._ranking:
            for item in self._ranking:
                rank = int(item.get("rank", 1))
                base_weight_by_strategy[item["strategy"]] = RANK_WEIGHTS.get(rank, _DEFAULT_WEIGHT)
        else:
            base_weight_by_strategy = {n: _DEFAULT_WEIGHT for n in self._strategy_names}

        recent_returns = self._calc_recent_returns(df, periods=20)
        # 动态权重：以近 N 日收益中位数为界
        ret_values = list(recent_returns.values())
        median_ret = pd.Series(ret_values).median() if ret_values else 0.0

        strategy_details: List[Dict[str, Any]] = []
        weighted_buy = 0.0
        weighted_sell = 0.0

        for name in self._strategy_names:
            strategy = self._strategies[name]
            sig_df = strategy.get_signal_dataframe(df, symbol)
            last = sig_df.iloc[-1]
            sig_val = last.get("signal", 0)
            conf = last.get("confidence", 0.0)
            if pd.isna(sig_val):
                sig_val = 0.0
            if pd.isna(conf):
                conf = 0.0

            if sig_val > 0:
                action = "buy"
            elif sig_val < 0:
                action = "sell"
            else:
                action = "hold"

            recent_ret = float(recent_returns.get(name, 0.0))
            base_w = float(base_weight_by_strategy.get(name, _DEFAULT_WEIGHT))
            if ret_values:
                if recent_ret > median_ret:
                    dyn_factor = _DYNAMIC_BETTER
                elif recent_ret < median_ret:
                    dyn_factor = _DYNAMIC_WORSE
                else:
                    dyn_factor = 1.0
            else:
                dyn_factor = 1.0
            final_weight = base_w * dyn_factor

            if action == "buy":
                weighted_buy += final_weight
            elif action == "sell":
                weighted_sell += final_weight

            strategy_details.append({
                "strategy": name,
                "label": self._labels.get(name, name),
                "signal": action,
                "confidence": round(float(conf), 4),
                "weight": round(final_weight, 4),
                "recent_return": round(recent_ret, 6),
            })

        net_score = weighted_buy - weighted_sell
        if net_score > threshold:
            aggregate_signal = "buy"
        elif net_score < -threshold:
            aggregate_signal = "sell"
        else:
            aggregate_signal = "hold"

        return {
            "aggregate_signal": aggregate_signal,
            "net_score": round(float(net_score), 4),
            "threshold": threshold,
            "weighted_buy": round(float(weighted_buy), 4),
            "weighted_sell": round(float(weighted_sell), 4),
            "strategy_details": strategy_details,
        }

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        """min-max 归一化；当所有值相等时返回 0.5。"""
        if not values:
            return []
        vmin = min(values)
        vmax = max(values)
        rng = vmax - vmin
        if rng == 0:
            return [0.5 for _ in values]
        return [(v - vmin) / rng for v in values]

    def _calc_recent_returns(
        self, df: pd.DataFrame, periods: int = 20
    ) -> Dict[str, float]:
        """用各策略信号模拟近 N 日表现。

        以信号方向作为仓位（+1 做多 / -1 做空 / 0 空仓），信号 shift(1) 后
        与次日收盘收益相乘，截取最近 ``periods`` 个交易日复利。

        Args:
            df: 行情数据，需含 close 列。
            periods: 统计窗口（交易日）。

        Returns:
            ``{strategy_name: 近N日收益率}``。
        """
        close = df["close"].astype(float)
        daily_ret = close.pct_change().fillna(0.0)
        out: Dict[str, float] = {}
        for name in self._strategy_names:
            try:
                sig_df = self._strategies[name].get_signal_dataframe(df, "")
                position = sig_df["signal"].ffill().fillna(0.0)
                # 信号已在 get_signal_dataframe 内 shift(1)，此处直接对齐
                strat_ret = position.reindex(daily_ret.index).fillna(0.0) * daily_ret
                window = strat_ret.tail(periods)
                out[name] = float((1.0 + window).prod() - 1.0)
            except Exception as e:  # 单个策略计算失败不影响聚合
                logger.warning("策略 %s 近期收益计算失败: %s", name, e)
                out[name] = 0.0
        return out
