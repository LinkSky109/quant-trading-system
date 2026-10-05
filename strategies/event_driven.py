"""事件驱动策略模块。

支持基于市场事件的策略信号生成与事件研究（Event Study）。

可从K线直接检测的事件：
    - ``limit_up``        涨停板（涨幅 ≥ 阈值，默认 9.8%）
    - ``volume_spike``    成交量异常（volume > multiplier × 20日均量）
    - ``price_gap``       价格跳空（|open/prev_close - 1| > threshold）

需外部数据源接入的事件（预留接口，当前返回空列表）：
    - ``earnings``        财报事件
    - ``dividend``        分红事件
    - ``stock_split``     拆股事件
    - ``index_rebalance`` 指数调整

策略模式（通过 ``params["mode"]`` 选择）：
    - ``pead``                     PEAD 盈余动量：跳空上涨后买入持有 N 日
    - ``limit_up_reversal``        涨停反转：涨停次日卖出（均值回归）
    - ``limit_up_continuation``    涨停延续：涨停次日买入（动量延续）
    - ``volume_spike``             量价齐升买入 / 放量下跌卖出

信号遵循基类 ``shift(1)`` 机制：事件日产生的原始信号在次日开盘执行，
杜绝未来函数。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

#: 支持的策略模式
SUPPORTED_MODES: Tuple[str, ...] = (
    "pead",
    "limit_up_reversal",
    "limit_up_continuation",
    "volume_spike",
)

#: 可从K线检测的事件类型
KLINE_EVENT_TYPES: Tuple[str, ...] = (
    "limit_up",
    "volume_spike",
    "price_gap",
)

#: 需外部数据源的事件类型
EXTERNAL_EVENT_TYPES: Tuple[str, ...] = (
    "earnings",
    "dividend",
    "stock_split",
    "index_rebalance",
)


class EventDrivenStrategy(BaseStrategy):
    """事件驱动策略。

    继承 :class:`~strategies.base_strategy.BaseStrategy`，可被
    :class:`~backtest.engine.BacktestEngine` 直接插拔使用。

    Args:
        params: 策略参数字典，支持以下键：
            - ``mode``: 策略模式，默认 ``"pead"``。
            - ``hold_days``: 持有期（交易日），默认 5。
            - ``limit_up_threshold``: 涨停涨幅阈值，默认 0.098。
            - ``volume_spike_multiplier``: 成交量异常倍数，默认 2.0。
            - ``volume_spike_window``: 均量窗口，默认 20。
            - ``price_gap_threshold``: 跳空阈值，默认 0.02。
    """

    name = "event_driven"

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self.mode: str = str(self.params.get("mode", "pead"))
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(
                f"未知策略模式: {self.mode}，可选: {SUPPORTED_MODES}"
            )
        self.hold_days: int = int(self.params.get("hold_days", 5))
        self.limit_up_threshold: float = float(
            self.params.get("limit_up_threshold", 0.098)
        )
        self.volume_spike_multiplier: float = float(
            self.params.get("volume_spike_multiplier", 2.0)
        )
        self.volume_spike_window: int = int(
            self.params.get("volume_spike_window", 20)
        )
        self.price_gap_threshold: float = float(
            self.params.get("price_gap_threshold", 0.02)
        )

    # ------------------------------------------------------------------
    # 事件检测
    # ------------------------------------------------------------------

    def _detect_limit_up(self, df: pd.DataFrame) -> pd.Series:
        """检测涨停板。

        涨停定义：当日收盘价相对前收盘价涨幅 ≥ ``limit_up_threshold``。

        Args:
            df: 含 ``close`` 列的 K线 DataFrame。

        Returns:
            与 df 等长的布尔 Series，True 表示当日涨停。
        """
        prev_close = df["close"].shift(1)
        pct = df["close"] / prev_close - 1.0
        return (pct >= self.limit_up_threshold).fillna(False)

    def _detect_volume_spike(self, df: pd.DataFrame,
                             multiplier: float | None = None) -> pd.Series:
        """检测成交量异常放大。

        定义：当日 volume > multiplier × 过去 N 日均量（不含当日）。

        Args:
            df: 含 ``volume`` 列的 K线 DataFrame。
            multiplier: 倍数，默认使用 ``self.volume_spike_multiplier``。

        Returns:
            布尔 Series。
        """
        mult = (self.volume_spike_multiplier
                if multiplier is None else float(multiplier))
        vol = df["volume"]
        avg_vol = vol.shift(1).rolling(self.volume_spike_window,
                                       min_periods=1).mean()
        return (vol > mult * avg_vol).fillna(False)

    def _detect_price_gap(self, df: pd.DataFrame,
                          threshold: float | None = None) -> pd.Series:
        """检测价格跳空。

        定义：|open / prev_close - 1| > threshold。

        Args:
            df: 含 ``open`` / ``close`` 列的 K线 DataFrame。
            threshold: 跳空阈值，默认使用 ``self.price_gap_threshold``。

        Returns:
            布尔 Series。
        """
        thr = (self.price_gap_threshold
               if threshold is None else float(threshold))
        prev_close = df["close"].shift(1)
        gap = df["open"] / prev_close - 1.0
        return (gap.abs() > thr).fillna(False)

    def detect_events(self, df: pd.DataFrame,
                     symbol: str = "") -> List[Dict[str, Any]]:
        """从 K线检测所有可计算事件（涨停 / 量异常 / 跳空）。

        Args:
            df: K线 DataFrame，index 为日期。
            symbol: 标的代码，写入事件元数据。

        Returns:
            事件字典列表，每项含 ``date`` / ``event_type`` / ``symbol`` /
            ``metadata``。
        """
        events: List[Dict[str, Any]] = []
        if df is None or df.empty:
            return events

        lu = self._detect_limit_up(df)
        vs = self._detect_volume_spike(df)
        pg = self._detect_price_gap(df)

        prev_close = df["close"].shift(1)
        pct = (df["close"] / prev_close - 1.0).fillna(0.0)
        gap = (df["open"] / prev_close - 1.0).fillna(0.0)
        avg_vol = (df["volume"].shift(1)
                   .rolling(self.volume_spike_window, min_periods=1).mean())

        for idx in df.index[lu]:
            events.append({
                "date": idx,
                "event_type": "limit_up",
                "symbol": symbol,
                "metadata": {"pct_change": float(pct.loc[idx])},
            })
        for idx in df.index[vs]:
            ratio = float(df.loc[idx, "volume"] / avg_vol.loc[idx]) \
                if pd.notna(avg_vol.loc[idx]) and avg_vol.loc[idx] > 0 else 0.0
            events.append({
                "date": idx,
                "event_type": "volume_spike",
                "symbol": symbol,
                "metadata": {"volume_ratio": ratio},
            })
        for idx in df.index[pg]:
            events.append({
                "date": idx,
                "event_type": "price_gap",
                "symbol": symbol,
                "metadata": {"gap": float(gap.loc[idx])},
            })

        events.sort(key=lambda e: e["date"])
        return events

    def load_external_events(self, symbol: str) -> List[Dict[str, Any]]:
        """预留接口：从外部数据源获取财报 / 分红 / 拆股 / 指数调整事件。

        当前未接入外部数据源，返回空列表。子类或后续版本可在此对接
        akshare / Wind / 财报日历等数据源。

        Args:
            symbol: 标的代码。

        Returns:
            事件字典列表（当前为空）。
        """
        logger.debug("load_external_events 预留接口，当前返回空列表: %s",
                     symbol)
        return []

    # ------------------------------------------------------------------
    # 信号生成（基类接口）
    # ------------------------------------------------------------------

    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """根据当前 mode 计算原始信号（signal / confidence）。

        注意：本方法返回的信号**未** shift，由基类统一 shift(1)。

        Args:
            df: K线 DataFrame。

        Returns:
            含 ``signal`` (1/-1/0) 与 ``confidence`` 列的 DataFrame。
        """
        out = df.copy()
        out["signal"] = 0
        out["confidence"] = 0.0

        if out.empty:
            return out

        if self.mode == "pead":
            self._apply_pead(out)
        elif self.mode == "limit_up_reversal":
            self._apply_limit_up_reversal(out)
        elif self.mode == "limit_up_continuation":
            self._apply_limit_up_continuation(out)
        elif self.mode == "volume_spike":
            self._apply_volume_spike(out)

        return out

    # -- 各模式信号逻辑 -------------------------------------------------

    def _mark_exit_after_hold(self, out: pd.DataFrame,
                              entry_indices: List[pd.Timestamp]) -> None:
        """在每个入场点后第 hold_days 个交易日标记卖出信号 (-1)。

        若多个入场点的退出日重叠，保留 -1 即可。
        """
        if not entry_indices:
            return
        idx = out.index
        for entry in entry_indices:
            if entry not in idx:
                continue
            pos = idx.get_loc(entry)
            exit_pos = pos + self.hold_days
            if exit_pos < len(idx):
                out.iloc[exit_pos, out.columns.get_loc("signal")] = -1
                out.iloc[exit_pos, out.columns.get_loc("confidence")] = 0.7

    def _apply_pead(self, out: pd.DataFrame) -> None:
        """PEAD：跳空上涨（gap>阈值 且 close>open）后买入持有 N 日。"""
        prev_close = out["close"].shift(1)
        gap = out["open"] / prev_close - 1.0
        gap_up = (gap > self.price_gap_threshold) & (out["close"] > out["open"])
        gap_up = gap_up.fillna(False)

        entry_dates = list(out.index[gap_up])
        out.loc[gap_up, "signal"] = 1
        out.loc[gap_up, "confidence"] = 0.75
        self._mark_exit_after_hold(out, entry_dates)

    def _apply_limit_up_reversal(self, out: pd.DataFrame) -> None:
        """涨停反转：涨停日产生卖出信号（次日开盘执行，均值回归）。"""
        lu = self._detect_limit_up(out)
        out.loc[lu, "signal"] = -1
        out.loc[lu, "confidence"] = 0.8

    def _apply_limit_up_continuation(self, out: pd.DataFrame) -> None:
        """涨停延续：涨停日产生买入信号（次日开盘追涨），持有 N 日。"""
        lu = self._detect_limit_up(out)
        entry_dates = list(out.index[lu])
        out.loc[lu, "signal"] = 1
        out.loc[lu, "confidence"] = 0.7
        self._mark_exit_after_hold(out, entry_dates)

    def _apply_volume_spike(self, out: pd.DataFrame) -> None:
        """量价信号：放量上涨买入，放量下跌卖出，持有 N 日。"""
        vs = self._detect_volume_spike(out)
        up = vs & (out["close"] > out["open"])
        down = vs & (out["close"] < out["open"])
        up = up.fillna(False)
        down = down.fillna(False)

        entry_dates = list(out.index[up])
        out.loc[up, "signal"] = 1
        out.loc[up, "confidence"] = 0.7
        out.loc[down, "signal"] = -1
        out.loc[down, "confidence"] = 0.75
        self._mark_exit_after_hold(out, entry_dates)

    # ------------------------------------------------------------------
    # 事件研究
    # ------------------------------------------------------------------

    def event_study(
        self,
        df: pd.DataFrame,
        events: List[Dict[str, Any]],
        window: int = 20,
    ) -> Dict[str, Any]:
        """事件研究：计算事件窗口内的平均累计异常收益（CAR）。

        异常收益定义：个股日收益 - 全期日均收益（均值模型）。
        对每个事件提取 ``[-window, +window]`` 窗口内的异常收益并累加，
        再对所有事件取平均得到 CAR 曲线。

        Args:
            df: K线 DataFrame，index 为日期。
            events: 事件列表，每项需含 ``date`` 键。
            window: 事件窗口（前后各 N 个交易日），默认 20。

        Returns:
            字典含：
                - ``car_series``: 平均 CAR 曲线（长度 2*window+1），index 为
                  相对事件日的偏移 (-window..window)。
                - ``car_std``: 各偏移处 CAR 的横截面标准差。
                - ``t_statistic``: 事件日（offset=0）CAR 横截面单样本 t 统计量。
                - ``p_value``: 对应 p 值。
                - ``event_count``: 有效事件数。
                - ``individual_cars``: 各事件的 CAR 数组列表。
        """
        empty_result: Dict[str, Any] = {
            "car_series": pd.Series(dtype=float),
            "car_std": pd.Series(dtype=float),
            "t_statistic": float("nan"),
            "p_value": float("nan"),
            "event_count": 0,
            "individual_cars": [],
        }
        if df is None or df.empty or not events or window <= 0:
            return empty_result

        # 个股日收益
        rets = df["close"].pct_change().fillna(0.0)
        # 均值模型基准：全期日均收益
        mu = float(rets.mean())
        abnormal = rets - mu

        offsets = np.arange(-window, window + 1)
        idx = df.index
        pos_map = {d: i for i, d in enumerate(idx)}

        car_rows: List[np.ndarray] = []
        for ev in events:
            ev_date = ev.get("date")
            if ev_date not in pos_map:
                continue
            center = pos_map[ev_date]
            start = center - window
            end = center + window
            if start < 0 or end >= len(idx):
                continue
            window_ar = abnormal.iloc[start:end + 1].values
            car = np.cumsum(window_ar)
            car_rows.append(car)

        if not car_rows:
            return empty_result

        car_matrix = np.vstack(car_rows)  # shape: (n_events, 2*window+1)
        mean_car = car_matrix.mean(axis=0)
        std_car = car_matrix.std(axis=0, ddof=1) if car_matrix.shape[0] > 1 \
            else np.zeros_like(mean_car)

        car_series = pd.Series(mean_car, index=offsets, name="avg_car")
        car_std = pd.Series(std_car, index=offsets, name="car_std")

        # t 检验：事件日（offset=0，即索引 window）的 CAR 横截面
        event_day_cars = car_matrix[:, window]
        if len(event_day_cars) >= 2 and np.std(event_day_cars) > 0:
            t_res = stats.ttest_1samp(event_day_cars, popmean=0.0)
            t_stat = float(t_res.statistic)
            p_val = float(t_res.pvalue)
        else:
            t_stat = float("nan")
            p_val = float("nan")

        return {
            "car_series": car_series,
            "car_std": car_std,
            "t_statistic": t_stat,
            "p_value": p_val,
            "event_count": int(car_matrix.shape[0]),
            "individual_cars": [list(row) for row in car_rows],
        }
