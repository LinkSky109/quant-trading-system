"""策略排行榜与信号聚合单元测试。

运行:
    cd quant_trading_system
    python -m pytest tests/test_strategy_leaderboard.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.data_fetcher import _generate_mock_klines
from strategies.strategy_leaderboard import StrategyLeaderboard

# mock 基准价：低价确保信号能真实成交（与 examples 一致）
MOCK_BASE_PRICE = 25.0


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------

def _make_df(count: int = 120, symbol: str = "600519.SH") -> pd.DataFrame:
    """生成确定性 mock 日线数据。"""
    return _generate_mock_klines(
        symbol, period="1d", count=count,
        start_date="2024-01-02", base_price=MOCK_BASE_PRICE,
    )


class MockFetcher:
    """不依赖网络的数据获取器：返回固定 mock K线。"""

    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df if df is not None else _make_df()
        self.call_count = 0

    def get_klines(self, symbol, count=250, start_date=None, end_date=None):
        self.call_count += 1
        df = self.df
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        return df.copy()


class FakeStrategy:
    """可控信号的假策略：最后一根K线给出指定信号。"""

    def __init__(self, signal: float, confidence: float = 0.8):
        self._signal = signal
        self._confidence = confidence

    def get_signal_dataframe(self, df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        out["signal"] = [0.0] * (n - 1) + [self._signal]
        out["confidence"] = [0.0] * (n - 1) + [self._confidence]
        return out


def _make_leaderboard(strategy_names, signal_map, ranking=None):
    """构造一个用 FakeStrategy 替换真实策略的排行榜实例。"""
    cfg = {"strategies": {}}  # 空配置，__init__ 不加载任何真实策略
    lb = StrategyLeaderboard(data_fetcher=MockFetcher(), config=cfg)
    lb._strategy_names = list(strategy_names)
    lb._strategies = {
        n: FakeStrategy(signal_map[n]) for n in strategy_names
    }
    lb._labels = {n: n for n in strategy_names}
    if ranking is not None:
        lb._ranking = ranking
    return lb


# ---------------------------------------------------------------------------
# 归一化 / 打分
# ---------------------------------------------------------------------------

def test_normalize():
    vals = [1.0, 2.0, 3.0]
    out = StrategyLeaderboard._normalize(vals)
    assert out == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_all_equal():
    out = StrategyLeaderboard._normalize([0.5, 0.5, 0.5])
    assert out == [0.5, 0.5, 0.5]


def test_normalize_empty():
    assert StrategyLeaderboard._normalize([]) == []


def test_score_formula():
    """验证综合得分 = 0.3*夏普 + 0.3*收益 + 0.2*(1-回撤) + 0.2*胜率。"""
    results = [
        # s1 仅夏普更高，其余指标与 s2 相同
        {"strategy": "s1", "sharpe": 2.0, "total_return": 0.2,
         "max_drawdown": 0.1, "win_rate": 0.5},
        {"strategy": "s2", "sharpe": 0.0, "total_return": 0.2,
         "max_drawdown": 0.1, "win_rate": 0.5},
    ]
    lb = StrategyLeaderboard(data_fetcher=MockFetcher(), config={"strategies": {}})
    ranked = lb.rank_strategies(results)
    # 收益/回撤/胜率三项全相等 -> 归一化均为 0.5；夏普归一化为 [1, 0]
    # s1 = 0.3*1 + 0.3*0.5 + 0.2*0.5 + 0.2*0.5 = 0.65
    # s2 = 0.3*0 + 0.3*0.5 + 0.2*0.5 + 0.2*0.5 = 0.35
    by = {r["strategy"]: r for r in ranked}
    assert by["s1"]["score"] == pytest.approx(0.65)
    assert by["s2"]["score"] == pytest.approx(0.35)


def test_rank_strategies():
    """综合得分高的排前面，rank 从 1 开始。"""
    results = [
        {"strategy": "bad", "sharpe": -1.0, "total_return": -0.2,
         "max_drawdown": 0.5, "win_rate": 0.1},
        {"strategy": "good", "sharpe": 1.5, "total_return": 0.4,
         "max_drawdown": 0.05, "win_rate": 0.8},
    ]
    lb = StrategyLeaderboard(data_fetcher=MockFetcher(), config={"strategies": {}})
    ranked = lb.rank_strategies(results)
    assert ranked[0]["strategy"] == "good"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["strategy"] == "bad"
    assert ranked[1]["rank"] == 2
    assert ranked[0]["score"] > ranked[1]["score"]


# ---------------------------------------------------------------------------
# 信号聚合
# ---------------------------------------------------------------------------

def test_aggregate_signals_buy():
    """多数高权重策略买入 -> buy。"""
    names = ["s1", "s2", "s3"]
    # s1/s2 买入，s3 卖出；排名靠前买入方权重占优
    ranking = [
        {"strategy": "s1", "rank": 1},
        {"strategy": "s2", "rank": 2},
        {"strategy": "s3", "rank": 3},
    ]
    lb = _make_leaderboard(
        names, {"s1": 1, "s2": 1, "s3": -1}, ranking=ranking,
    )
    # 近期收益全相等 -> 动态因子=1.0，权重等于排名基础权重
    lb._calc_recent_returns = lambda df, periods=20: {n: 0.0 for n in names}  # type: ignore
    out = lb.aggregate_signals("600519.SH", threshold=2.0)
    # 买入权重 3.0+2.5=5.5, 卖出权重 2.0 -> 净=3.5 > 2.0
    assert out["aggregate_signal"] == "buy"
    assert out["net_score"] > 0


def test_aggregate_signals_sell():
    """高权重策略卖出 -> sell。"""
    names = ["s1", "s2", "s3"]
    ranking = [
        {"strategy": "s1", "rank": 1},
        {"strategy": "s2", "rank": 2},
        {"strategy": "s3", "rank": 3},
    ]
    lb = _make_leaderboard(
        names, {"s1": -1, "s2": -1, "s3": 1}, ranking=ranking,
    )
    lb._calc_recent_returns = lambda df, periods=20: {n: 0.0 for n in names}  # type: ignore
    out = lb.aggregate_signals("600519.SH", threshold=2.0)
    assert out["aggregate_signal"] == "sell"
    assert out["net_score"] < 0


def test_aggregate_signals_hold():
    """买卖权重均衡 -> hold。"""
    names = ["s1", "s2"]
    ranking = [
        {"strategy": "s1", "rank": 1},
        {"strategy": "s2", "rank": 2},
    ]
    lb = _make_leaderboard(
        names, {"s1": 1, "s2": -1}, ranking=ranking,
    )
    lb._calc_recent_returns = lambda df, periods=20: {n: 0.0 for n in names}  # type: ignore
    # 买入 3.0, 卖出 2.5 -> 净 0.5，阈值 1.0 -> hold
    out = lb.aggregate_signals("600519.SH", threshold=1.0)
    assert out["aggregate_signal"] == "hold"
    assert -1.0 <= out["net_score"] <= 1.0


def test_weighted_voting():
    """排名高的策略权重更大（第1名=3.0 > 最后一名=0.5）。"""
    names = ["top", "bottom"]
    ranking = [
        {"strategy": "top", "rank": 1},
        {"strategy": "bottom", "rank": 2},
    ]
    lb = _make_leaderboard(
        names, {"top": 0, "bottom": 0}, ranking=ranking,
    )
    lb._calc_recent_returns = lambda df, periods=20: {n: 0.0 for n in names}  # type: ignore
    out = lb.aggregate_signals("600519.SH")
    details = {d["strategy"]: d for d in out["strategy_details"]}
    # 无动态调整干扰（近期收益相等 -> factor=1.0）
    assert details["top"]["weight"] == pytest.approx(3.0)
    assert details["bottom"]["weight"] == pytest.approx(2.5)


def test_dynamic_weights():
    """近期表现好的策略权重 ×1.2，差的 ×0.8。"""
    names = ["good", "bad"]
    ranking = [
        {"strategy": "good", "rank": 1},
        {"strategy": "bad", "rank": 2},
    ]
    lb = _make_leaderboard(
        names, {"good": 0, "bad": 0}, ranking=ranking,
    )
    # 覆盖近期收益计算：good 明显好于 bad
    lb._calc_recent_returns = lambda df, periods=20: {"good": 0.15, "bad": -0.10}  # type: ignore
    out = lb.aggregate_signals("600519.SH")
    details = {d["strategy"]: d for d in out["strategy_details"]}
    # good: base 3.0 * 1.2 = 3.6 ; bad: base 2.5 * 0.8 = 2.0
    assert details["good"]["weight"] == pytest.approx(3.6)
    assert details["bad"]["weight"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 缓存 / 端到端
# ---------------------------------------------------------------------------

def test_cache():
    """相同 (symbol, start, end) 第二次回测命中缓存，不重复取数。"""
    fetcher = MockFetcher()
    lb = StrategyLeaderboard(data_fetcher=fetcher)
    lb.get_leaderboard("600519.SH", "2024-01-02", "2024-06-30")
    calls_after_first = fetcher.call_count
    assert calls_after_first >= 1
    # 再次回测同一区间
    lb.run_backtests("600519.SH", "2024-01-02", "2024-06-30")
    assert fetcher.call_count == calls_after_first  # 未增加 -> 命中缓存


def test_leaderboard_with_mock_data():
    """用真实策略 + mock 数据完整跑排行榜。"""
    fetcher = MockFetcher()
    lb = StrategyLeaderboard(data_fetcher=fetcher)
    board = lb.get_leaderboard("600519.SH", "2024-01-02", "2024-12-31")

    # 所有启用策略全部参与排名
    assert len(board) >= 6
    # 按综合得分降序
    scores = [r["score"] for r in board]
    assert scores == sorted(scores, reverse=True)
    # rank 从 1 递增
    assert [r["rank"] for r in board] == list(range(1, len(board) + 1))
    # 关键字段齐全
    for r in board:
        assert {"strategy", "label", "metrics", "sharpe",
                "total_return", "max_drawdown", "win_rate",
                "score", "rank"} <= set(r.keys())

    # 信号聚合可正常返回
    agg = lb.aggregate_signals("600519.SH")
    assert agg["aggregate_signal"] in ("buy", "sell", "hold")
    assert len(agg["strategy_details"]) == len(board)
