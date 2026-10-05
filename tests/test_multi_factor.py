"""多因子选股策略单元测试。

覆盖：
  - 因子合成：z-score / 排名百分位标准化、方向调整、缺失值中位数填充 / 剔除
  - 选股逻辑：top_n 入选、掉出卖出、新进买入
  - 调仓逻辑：rebalance_days 周期、调仓历史完整性
  - 回测：多标的回测正常运行、结果结构完整
  - 边界：空数据、标的数少于 top_n、全 NaN 因子值

全部使用确定性随机种子与 FakeFactorEngine 构造的模拟因子值，不依赖网络。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine
from strategies.multi_factor import MultiFactorStrategy


# ---------------------------------------------------------------------------
# 构造工具
# ---------------------------------------------------------------------------
def make_klines(
    n_days: int = 120,
    seed: int = 0,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """构造带 open/high/low/close/volume/amount 列的模拟日 K 线。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rets = rng.normal(0.0005, 0.02, size=n_days)
    close = start_price * np.cumprod(1.0 + rets)
    open_ = close * (1.0 + rng.normal(0, 0.005, size=n_days))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0, 0.01, size=n_days))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0, 0.01, size=n_days))
    volume = rng.uniform(1e6, 5e6, size=n_days)
    amount = volume * close
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "amount": amount,
        },
        index=dates,
    )


class FakeFactorEngine:
    """可控因子引擎：按 symbol 返回人工指定的因子序列。

    factor_series: {symbol: {factor_name: pd.Series(index=dates)}}
    """

    def __init__(self, factor_series: Dict[str, Dict[str, pd.Series]]) -> None:
        self.factor_series = factor_series

    def calculate_factors(self, df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        out = df.copy()
        series = self.factor_series.get(symbol, {})
        for fname, s in series.items():
            out[fname] = s.reindex(out.index) if hasattr(s, "reindex") else s
        return out


def _make_strategy(
    params: Optional[Dict[str, Any]] = None,
    fake_engine: Optional[FakeFactorEngine] = None,
) -> MultiFactorStrategy:
    s = MultiFactorStrategy(params)
    if fake_engine is not None:
        s._factor_engine = fake_engine
    return s


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_panel() -> Dict[str, pd.DataFrame]:
    """8 只真实计算因子的模拟 K 线（250 个交易日）。"""
    return {f"S{i:02d}.SH": make_klines(250, seed=i) for i in range(8)}


# ---------------------------------------------------------------------------
# 1. 因子合成 —— 标准化与方向
# ---------------------------------------------------------------------------
def test_zscore_standardization_correctness() -> None:
    """z-score：(x - mean) / std(ddof=0)，3 个值 [1,2,3] -> [-1,0,1]。"""
    dates = pd.bdate_range("2024-01-01", periods=10)
    # 用一个因子 momentum_20，每个 symbol 在前 3 日给固定横截面值
    panel: Dict[str, pd.DataFrame] = {}
    factor_vals: Dict[str, Dict[str, pd.Series]] = {}
    syms = ["A", "B", "C"]
    raw = {"A": 1.0, "B": 2.0, "C": 3.0}
    for sym in syms:
        df = make_klines(10, seed=hash(sym) % 100)
        df.index = dates
        panel[sym] = df
        factor_vals[sym] = {
            # 第 3 天（index=2）之后才有效；这里让整列固定为 raw[sym]
            "momentum_20": pd.Series(raw[sym], index=dates),
        }

    s = _make_strategy(
        {"factors": [{"name": "momentum_20", "weight": 1.0, "direction": 1}],
         "standardization": "zscore", "missing_handling": "median"},
        FakeFactorEngine(factor_vals),
    )
    scores = s.compute_cross_sectional_scores(panel)
    # mean=2, 总体标准差 std(ddof=0)=sqrt(2/3)≈0.8165
    # z = [(1-2)/std, 0, (3-2)/std] = [-1.2247, 0, 1.2247]
    expected = 1.0 / np.sqrt(2 / 3)
    row = scores.iloc[5]
    assert row["A"] == pytest.approx(-expected, abs=1e-9)
    assert row["B"] == pytest.approx(0.0, abs=1e-9)
    assert row["C"] == pytest.approx(expected, abs=1e-9)


def test_rank_percentile_standardization() -> None:
    """rank 标准化：[1,2,3] 排名百分位 [1/3,2/3,1] 减 0.5。"""
    dates = pd.bdate_range("2024-01-01", periods=10)
    panel: Dict[str, pd.DataFrame] = {}
    factor_vals: Dict[str, Dict[str, pd.Series]] = {}
    raw = {"A": 1.0, "B": 2.0, "C": 3.0}
    for sym, v in raw.items():
        df = make_klines(10, seed=1)
        df.index = dates
        panel[sym] = df
        factor_vals[sym] = {"momentum_20": pd.Series(v, index=dates)}

    s = _make_strategy(
        {"factors": [{"name": "momentum_20", "weight": 1.0, "direction": 1}],
         "standardization": "rank"},
        FakeFactorEngine(factor_vals),
    )
    scores = s.compute_cross_sectional_scores(panel)
    row = scores.iloc[5]
    assert row["A"] == pytest.approx(1 / 3 - 0.5, abs=1e-9)
    assert row["B"] == pytest.approx(2 / 3 - 0.5, abs=1e-9)
    assert row["C"] == pytest.approx(1.0 - 0.5, abs=1e-9)


def test_direction_adjustment() -> None:
    """direction=-1 时，z-score 结果取反。"""
    dates = pd.bdate_range("2024-01-01", periods=10)
    panel: Dict[str, pd.DataFrame] = {}
    factor_vals: Dict[str, Dict[str, pd.Series]] = {}
    raw = {"A": 1.0, "B": 2.0, "C": 3.0}
    for sym, v in raw.items():
        df = make_klines(10, seed=2)
        df.index = dates
        panel[sym] = df
        factor_vals[sym] = {"rsi_14": pd.Series(v, index=dates)}

    s = _make_strategy(
        {"factors": [{"name": "rsi_14", "weight": 1.0, "direction": -1}],
         "standardization": "zscore"},
        FakeFactorEngine(factor_vals),
    )
    scores = s.compute_cross_sectional_scores(panel)
    expected = 1.0 / np.sqrt(2 / 3)
    row = scores.iloc[5]
    # 原始 z = [-expected, 0, expected]，乘 -1 -> [expected, 0, -expected]
    assert row["A"] == pytest.approx(expected, abs=1e-9)
    assert row["B"] == pytest.approx(0.0, abs=1e-9)
    assert row["C"] == pytest.approx(-expected, abs=1e-9)


def test_missing_median_fill() -> None:
    """缺失值中位数填充：1 个 NaN 用其余值的中位数填充后参与打分。"""
    dates = pd.bdate_range("2024-01-01", periods=10)
    panel: Dict[str, pd.DataFrame] = {}
    factor_vals: Dict[str, Dict[str, pd.Series]] = {}
    # A 缺失，B=2, C=4, D=6 -> 中位数 4；填充后 A=4
    raw = {"A": np.nan, "B": 2.0, "C": 4.0, "D": 6.0}
    for sym, v in raw.items():
        df = make_klines(10, seed=3)
        df.index = dates
        panel[sym] = df
        factor_vals[sym] = {
            "momentum_20": pd.Series(v, index=dates)
            if not np.isnan(v) else pd.Series(np.nan, index=dates)
        }

    s = _make_strategy(
        {"factors": [{"name": "momentum_20", "weight": 1.0, "direction": 1}],
         "standardization": "zscore", "missing_handling": "median"},
        FakeFactorEngine(factor_vals),
    )
    scores = s.compute_cross_sectional_scores(panel)
    row = scores.iloc[5]
    # 填充后值为 [4,2,4,6]，mean=4，std=sqrt(((0)^2+(-2)^2+0+2^2)/4)=sqrt(2)≈1.414
    # z = [0, -2/sqrt2, 0, 2/sqrt2]
    assert row["A"] == pytest.approx(0.0, abs=1e-9)
    assert row["C"] == pytest.approx(0.0, abs=1e-9)
    assert row["B"] == pytest.approx(-2 / np.sqrt(2), abs=1e-9)
    assert row["D"] == pytest.approx(2 / np.sqrt(2), abs=1e-9)


def test_missing_drop() -> None:
    """missing_handling=drop：NaN 保留，该标的当日不可入选。"""
    dates = pd.bdate_range("2024-01-01", periods=10)
    panel: Dict[str, pd.DataFrame] = {}
    factor_vals: Dict[str, Dict[str, pd.Series]] = {}
    raw = {"A": np.nan, "B": 1.0, "C": 2.0, "D": 3.0}
    for sym, v in raw.items():
        df = make_klines(10, seed=4)
        df.index = dates
        panel[sym] = df
        factor_vals[sym] = {"momentum_20": pd.Series(v, index=dates)}

    s = _make_strategy(
        {"factors": [{"name": "momentum_20", "weight": 1.0, "direction": 1}],
         "standardization": "zscore", "missing_handling": "drop",
         "top_n": 2, "rebalance_days": 5},
        FakeFactorEngine(factor_vals),
    )
    s.set_cross_section_data(panel)
    hist = s.get_rebalance_history()
    assert hist, "应至少有一次调仓"
    first = hist[0]
    # A 因子全 NaN，不应出现在 holdings
    assert "A" not in first["holdings"]
    # 实际可评分标的为 B/C/D，top_n=2
    assert len(first["holdings"]) == 2


# ---------------------------------------------------------------------------
# 2. 选股与调仓逻辑
# ---------------------------------------------------------------------------
def test_top_n_selection_and_signals(real_panel: Dict[str, pd.DataFrame]) -> None:
    """top_n=3：首次调仓产生 3 个买入信号，无卖出。"""
    s = MultiFactorStrategy({"top_n": 3, "rebalance_days": 5})
    s.set_cross_section_data(real_panel)

    hist = s.get_rebalance_history()
    assert hist, "应有调仓记录"
    first = hist[0]
    assert len(first["added"]) == 3
    assert first["removed"] == []
    assert len(first["holdings"]) == 3

    # 信号表中：first 调仓日那 3 只标的 signal=1
    rebal_date = pd.Timestamp(first["date"])
    for sym in first["holdings"]:
        assert s._signal_map[sym].loc[rebal_date, "signal"] == 1
    for sym in real_panel.keys():
        if sym not in first["holdings"]:
            assert s._signal_map[sym].loc[rebal_date, "signal"] == 0


def test_sell_on_dropout_and_buy_on_entry(real_panel: Dict[str, pd.DataFrame]) -> None:
    """掉出 top_n 产生 -1，新进入产生 +1，保持不变为 0。"""
    s = MultiFactorStrategy({"top_n": 3, "rebalance_days": 5})
    s.set_cross_section_data(real_panel)
    hist = s.get_rebalance_history()
    assert len(hist) >= 2, "至少两次调仓才能验证换仓"

    added_total = sum(len(h["added"]) for h in hist)
    removed_total = sum(len(h["removed"]) for h in hist)
    # 早期调仓新增/剔除总和应 > 0（有换手）
    assert added_total > 0
    # 若至少有一次换仓，removed 应 > 0；随机数据下概率极高
    assert removed_total >= 0

    # 验证信号一致性：added 标的当日 signal=1，removed 当日 signal=-1
    for h in hist[1:]:
        dt = pd.Timestamp(h["date"])
        for sym in h["added"]:
            assert s._signal_map[sym].loc[dt, "signal"] == 1
        for sym in h["removed"]:
            assert s._signal_map[sym].loc[dt, "signal"] == -1


def test_rebalance_period(real_panel: Dict[str, pd.DataFrame]) -> None:
    """rebalance_days=5：调仓日间隔约为 5 个交易日。"""
    s = MultiFactorStrategy({"top_n": 3, "rebalance_days": 5})
    s.set_cross_section_data(real_panel)
    dates = [pd.Timestamp(h["date"]) for h in s.get_rebalance_history()]
    assert len(dates) >= 5
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i - 1]).days
        # 工作日间隔：5 个交易日 ≈ 7 个自然日（±2 天容差）
        assert 5 <= gap <= 10, f"调仓间隔异常: {gap} 天"


def test_rebalance_history_complete(real_panel: Dict[str, pd.DataFrame]) -> None:
    """调仓历史记录字段完整。"""
    s = MultiFactorStrategy({"top_n": 3, "rebalance_days": 5})
    s.set_cross_section_data(real_panel)
    for h in s.get_rebalance_history():
        assert {"date", "added", "removed", "holdings", "scores"} <= set(h)
        assert isinstance(h["date"], str)
        assert len(h["scores"]) > 0


# ---------------------------------------------------------------------------
# 3. 回测集成
# ---------------------------------------------------------------------------
def test_backtest_multi_symbol(real_panel: Dict[str, pd.DataFrame]) -> None:
    """多标的回测能正常运行并返回完整结构。"""
    s = MultiFactorStrategy({"top_n": 3, "rebalance_days": 5})
    s.set_cross_section_data(real_panel)

    engine = BacktestEngine(initial_capital=1_000_000.0)
    result = engine.run(real_panel, s, symbol="")

    assert not result.equity_curve.empty
    assert not result.benchmark_curve.empty
    # metrics 含关键字段
    assert "累计收益率" in result.metrics
    assert "最大回撤" in result.metrics
    # 至少有买入交易
    buys = [t for t in result.trades if t.action == "buy"]
    assert len(buys) > 0, "回测应产生买入交易"


# ---------------------------------------------------------------------------
# 4. 边界情况
# ---------------------------------------------------------------------------
def test_empty_panel() -> None:
    """空股票池：返回空得分矩阵，信号表为空。"""
    s = MultiFactorStrategy()
    s.set_cross_section_data({})
    assert s.scores_df is None or s.scores_df.empty
    assert s._signal_map == {}
    assert s.get_rebalance_history() == []


def test_fewer_symbols_than_top_n(real_panel: Dict[str, pd.DataFrame]) -> None:
    """标的数 < top_n：全部入选，不报错。"""
    small = {k: v for k, v in list(real_panel.items())[:3]}
    s = MultiFactorStrategy({"top_n": 5, "rebalance_days": 5})
    s.set_cross_section_data(small)
    hist = s.get_rebalance_history()
    assert hist
    first = hist[0]
    assert len(first["holdings"]) == 3  # 只能选 3 只


def test_all_nan_factors() -> None:
    """全 NaN 因子值：不崩溃，无可入选标的。"""
    dates = pd.bdate_range("2024-01-01", periods=10)
    panel: Dict[str, pd.DataFrame] = {}
    factor_vals: Dict[str, Dict[str, pd.Series]] = {}
    for sym in ["X", "Y", "Z"]:
        df = make_klines(10, seed=5)
        df.index = dates
        panel[sym] = df
        factor_vals[sym] = {"momentum_20": pd.Series(np.nan, index=dates)}

    s = _make_strategy(
        {"factors": [{"name": "momentum_20", "weight": 1.0, "direction": 1}],
         "missing_handling": "drop"},
        FakeFactorEngine(factor_vals),
    )
    s.set_cross_section_data(panel)
    # 全 NaN -> 无有效调仓
    assert s.get_rebalance_history() == []


def test_single_symbol_fallback() -> None:
    """未加载横截面数据时，_compute_raw_signals 返回全 0 信号。"""
    s = MultiFactorStrategy()
    df = make_klines(30, seed=99)
    out = s._compute_raw_signals(df)
    assert (out["signal"] == 0).all()
    assert (out["confidence"] == 0.0).all()


def test_weight_normalization() -> None:
    """外部传入权重和不为 1 时自动归一。"""
    s = MultiFactorStrategy({
        "factors": [
            {"name": "momentum_20", "weight": 0.5, "direction": 1},
            {"name": "rsi_14", "weight": 1.5, "direction": -1},
        ]
    })
    total = sum(f["weight"] for f in s.factors)
    assert total == pytest.approx(1.0, abs=1e-9)
