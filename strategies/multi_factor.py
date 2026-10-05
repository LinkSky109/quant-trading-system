"""多因子选股策略。

通过横截面标准化 + 方向调整 + 加权合成，对股票池内所有标的进行打分，
每 ``rebalance_days`` 个交易日调仓一次，等权持有得分最高的 ``top_n`` 只。

典型用法::

    strategy = MultiFactorStrategy(params)
    strategy.set_cross_section_data(symbol_data)   # 预加载 {symbol: K线DataFrame}
    engine = BacktestEngine(initial_capital=1_000_000)
    result = engine.run(symbol_data, strategy, symbol="")

信号在 T 日收盘后基于当日因子得分产生，经基类统一 shift(1) 后于 T+1 日开盘执行，
严格避免未来函数。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from factors.factor_engine import FactorEngine
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


# 默认因子配置：name / weight / direction（+1 正向，-1 反向）
_DEFAULT_FACTORS: List[Dict[str, Any]] = [
    {"name": "momentum_20", "weight": 0.30, "direction": 1},
    {"name": "volatility_20_inverse", "weight": 0.20, "direction": 1},
    {"name": "rsi_14", "weight": 0.20, "direction": -1},
    {"name": "amount_log", "weight": 0.15, "direction": 1},
    {"name": "macd_hist", "weight": 0.15, "direction": 1},
]


class MultiFactorStrategy(BaseStrategy):
    """多因子选股策略。

    因子合成流程（每个调仓日横截面进行一次）::

        原始因子值 -> 缺失值处理 -> 横截面标准化(zscore/rank)
                  -> 方向调整(乘 direction) -> 按 weight 加权求和 -> 综合得分

    选股逻辑：取综合得分最高的 top_n 只等权持有；新进入者产生买入信号，
    掉出者产生卖出信号，其余持有不动。
    """

    name = "multi_factor"

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)

        # ---- 策略参数 ----
        self.factors: List[Dict[str, Any]] = list(
            self.params.get("factors") or _DEFAULT_FACTORS
        )
        self.rebalance_days: int = int(self.params.get("rebalance_days", 5))
        self.top_n: int = int(self.params.get("top_n", 5))
        self.standardization: str = str(
            self.params.get("standardization", "zscore")
        ).lower()
        self.missing_handling: str = str(
            self.params.get("missing_handling", "median")
        ).lower()

        # 权重归一化（容错：外部传入权重和不为 1 时自动归一）
        total_w = sum(float(f["weight"]) for f in self.factors)
        if total_w > 0:
            for f in self.factors:
                f["weight"] = float(f["weight"]) / total_w

        # ---- 运行时缓存 ----
        self._factor_engine = FactorEngine()
        # scores_df: 日期 × 标的 的综合得分矩阵
        self._scores_df: Optional[pd.DataFrame] = None
        # signal_map: {symbol: 带 signal/confidence 列的 DataFrame（未 shift）}
        self._signal_map: Dict[str, pd.DataFrame] = {}
        self.rebalance_history: List[Dict[str, Any]] = []
        # 是否已加载横截面数据
        self._cross_section_loaded: bool = False
        # BacktestEngine 当前正在询问的标的（由 get_signal_dataframe 注入）
        self._current_symbol: str = ""

    # ------------------------------------------------------------------ #
    # 横截面数据加载与得分计算
    # ------------------------------------------------------------------ #
    def set_cross_section_data(
        self, symbol_data: Dict[str, pd.DataFrame]
    ) -> None:
        """预加载股票池行情数据，触发横截面得分计算并缓存信号。

        Args:
            symbol_data: ``{symbol: K线DataFrame}``，index 为日期。
        """
        self._cross_section_loaded = True
        self._scores_df = self.compute_cross_sectional_scores(symbol_data)
        self._signal_map = self.generate_rebalance_signals(self._scores_df)
        logger.info(
            "多因子策略横截面数据已加载: %d 只标的, %d 个调仓日",
            len(self._signal_map),
            len(self.rebalance_history),
        )

    def compute_cross_sectional_scores(
        self, symbol_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """计算所有标的在所有日期的多因子综合得分。

        Args:
            symbol_data: ``{symbol: K线DataFrame}``。

        Returns:
            DataFrame，index 为日期，columns 为标的代码，值为综合得分；
            输入为空时返回空 DataFrame。
        """
        # 1. 对每只标的计算全部因子
        factor_dfs: Dict[str, pd.DataFrame] = {}
        for sym, df in symbol_data.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            try:
                factor_dfs[sym] = self._factor_engine.calculate_factors(
                    df, symbol=sym
                )
            except Exception as exc:  # pragma: no cover - 单票失败不影响整体
                logger.warning("计算 %s 因子失败: %s", sym, exc)

        if not factor_dfs:
            return pd.DataFrame()

        # 2. 对每个配置因子构建 日期×标的 的原始因子面板
        score: Optional[pd.DataFrame] = None
        for fac in self.factors:
            name = str(fac["name"])
            weight = float(fac["weight"])
            direction = int(fac.get("direction", 1))

            cols: Dict[str, pd.Series] = {}
            for sym, computed in factor_dfs.items():
                if name in computed.columns:
                    s = computed[name].copy()
                    s.name = sym
                    cols[sym] = s
            if not cols:
                logger.warning("因子 %s 在所有标的上均缺失，跳过", name)
                continue

            panel = pd.concat(cols.values(), axis=1).sort_index()
            panel.columns = list(cols.keys())

            # 3. 缺失值处理
            if self.missing_handling == "median":
                # 每行（每个调仓日）用该因子横截面中位数填充
                panel = panel.T.fillna(panel.median(axis=1)).T
            # "drop" 模式：保留 NaN，后续打分时该标的当日不可入选

            # 4. 横截面标准化（按行，即每个日期在截面上做）
            if self.standardization == "rank":
                normed = panel.rank(axis=1, pct=True) - 0.5  # 映射到 [-0.5, 0.5]
            else:  # zscore（默认）
                mu = panel.mean(axis=1)
                sd = panel.std(axis=1, ddof=0).replace(0, np.nan)
                normed = panel.sub(mu, axis=0).div(sd, axis=0)

            # 5. 方向调整 + 加权
            weighted = normed * direction * weight

            score = weighted if score is None else score.add(weighted, fill_value=0.0)

        if score is None:
            return pd.DataFrame()
        return score.sort_index()

    # ------------------------------------------------------------------ #
    # 调仓信号生成
    # ------------------------------------------------------------------ #
    def generate_rebalance_signals(
        self, scores_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """根据得分矩阵生成每个标的的逐日交易信号（未 shift）。

        Args:
            scores_df: :meth:`compute_cross_sectional_scores` 返回的得分矩阵。

        Returns:
            ``{symbol: DataFrame}``，每个 DataFrame 含 signal(1/-1/0) 与
            confidence 列，index 为日期。
        """
        self.rebalance_history = []

        if scores_df is None or scores_df.empty:
            return {}

        symbols = list(scores_df.columns)
        # 初始化所有标的的信号表（默认持有现金、无信号）
        sig_map: Dict[str, pd.DataFrame] = {
            sym: pd.DataFrame(
                0.0, index=scores_df.index, columns=["signal", "confidence"]
            )
            for sym in symbols
        }

        # 有效调仓日：剔除整行全 NaN 的日期（因子预热期）
        valid_dates = scores_df.dropna(how="all").index.tolist()
        if not valid_dates:
            return sig_map

        # 每 rebalance_days 个交易日调仓一次
        rebal_dates = set(
            valid_dates[i] for i in range(0, len(valid_dates), self.rebalance_days)
        )

        holdings: set = set()
        for dt in scores_df.index:
            if dt not in rebal_dates:
                continue

            row = scores_df.loc[dt].dropna()
            if row.empty:
                continue

            n_pick = min(self.top_n, len(row))
            top_set = set(row.nlargest(n_pick).index.tolist())

            added = top_set - holdings
            removed = holdings - top_set

            for sym in added:
                sig_map[sym].loc[dt, "signal"] = 1
                sig_map[sym].loc[dt, "confidence"] = 0.8
            for sym in removed:
                sig_map[sym].loc[dt, "signal"] = -1
                sig_map[sym].loc[dt, "confidence"] = 1.0

            self.rebalance_history.append({
                "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "added": sorted(added),
                "removed": sorted(removed),
                "holdings": sorted(top_set),
                "scores": {sym: float(v) for sym, v in row.sort_values(ascending=False).items()},
            })
            holdings = top_set

        return sig_map

    # ------------------------------------------------------------------ #
    # BaseStrategy 接口
    # ------------------------------------------------------------------ #
    def get_signal_dataframe(
        self, df: pd.DataFrame, symbol: str = ""
    ) -> pd.DataFrame:
        """返回带 signal / confidence 列的 DataFrame（已 shift）。

        重写基类方法以把 ``symbol`` 透传给 :meth:`_compute_raw_signals`，
        从而精确定位该标的在横截面信号表中的预计算信号。
        """
        self._current_symbol = symbol
        return super().get_signal_dataframe(df, symbol)

    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """返回单只标的的原始信号列（未 shift，由基类统一 shift）。

        若已通过 :meth:`set_cross_section_data` 加载横截面数据，则返回该标的
        预计算好的调仓信号；否则退化为单标的中性模式（始终持币，仅作参考）。
        """
        out = df.copy()
        out["signal"] = 0
        out["confidence"] = 0.0

        if not self._cross_section_loaded or not self._signal_map:
            return out

        sym = getattr(self, "_current_symbol", "")
        sig_df = self._signal_map.get(sym)
        if sig_df is None:
            # 兜底：取与当前 index 交集最长的信号表
            best_sym = max(
                self._signal_map.keys(),
                key=lambda s: len(out.index.intersection(self._signal_map[s].index)),
                default=None,
            )
            sig_df = self._signal_map.get(best_sym) if best_sym else None
        if sig_df is None:
            return out

        sig = sig_df.reindex(out.index)
        out["signal"] = sig["signal"].fillna(0.0)
        out["confidence"] = sig["confidence"].fillna(0.0)
        return out

    # ------------------------------------------------------------------ #
    # 访问器
    # ------------------------------------------------------------------ #
    def get_rebalance_history(self) -> List[Dict[str, Any]]:
        """返回调仓历史记录列表。

        每条记录含 ``date / added / removed / holdings / scores`` 字段。
        """
        return self.rebalance_history

    @property
    def scores_df(self) -> Optional[pd.DataFrame]:
        """最近一次 :meth:`set_cross_section_data` 计算出的得分矩阵。"""
        return self._scores_df
