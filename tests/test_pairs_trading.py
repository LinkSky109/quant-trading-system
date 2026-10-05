"""配对交易策略单元测试 + API 集成测试。

覆盖：
- OLS 对冲比率计算正确性（已知线性关系）
- 价差 / z-score 计算正确性
- 信号规则：z>=2 卖A买B，z<=-2 买A卖B，|z|<=0.5 平仓
- 止损：|z|>=3 触发
- 协整检验：构造协整序列返回 True；独立随机游走返回 False
- 标的对筛选：高相关对被选出，低相关对被过滤
- 回测返回正确结构
- 三个 API 端点集成测试（mock manager）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web-dashboard"))

from strategies.pairs_trading import (  # noqa: E402
    PairsTradingStrategy,
    calculate_spread,
    calculate_zscore,
    engle_granger_test,
    run_pairs_backtest,
    screen_pairs,
)


# ---------------------------------------------------------------------------
# 夹具：构造已知数据
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cointegrated_pair() -> tuple[pd.Series, pd.Series]:
    """构造协整序列：a = 2*b + 平稳噪声。"""
    rng = np.random.default_rng(42)
    n = 400
    b = pd.Series(100.0 + np.cumsum(rng.normal(0.05, 1.0, n)))
    a = 2.0 * b + pd.Series(rng.normal(0.0, 0.3, n))
    return a, b


@pytest.fixture(scope="module")
def independent_walks() -> tuple[pd.Series, pd.Series]:
    """两条独立随机游走（不应协整）。"""
    rng = np.random.default_rng(99)
    n = 400
    x = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 1.0, n)))
    y = pd.Series(50.0 + np.cumsum(rng.normal(0.0, 1.2, n)))
    return x, y


# ---------------------------------------------------------------------------
# 对冲比率 / 价差 / z-score
# ---------------------------------------------------------------------------


class TestHedgeRatio:
    def test_ols_recovers_known_beta(self, cointegrated_pair):
        a, b = cointegrated_pair
        eg = engle_granger_test(a, b)
        # a = 2*b + noise，应恢复 beta ≈ 2
        assert eg["hedge_ratio"] == pytest.approx(2.0, abs=0.05)
        assert eg["n"] == 400

    def test_spread_calculation(self):
        a = pd.Series([10.0, 11.0, 12.0])
        b = pd.Series([4.0, 5.0, 6.0])
        sp = calculate_spread(a, b, hedge_ratio=2.0)
        # spread = a - 2*b = [2, 1, 0]
        assert list(sp) == pytest.approx([2.0, 1.0, 0.0])

    def test_zscore(self):
        # 构造序列：前 40 个平稳为 0，末尾一个巨大跳升
        s = pd.Series([0.0] * 40 + [100.0])
        z = calculate_zscore(s, window=20)
        # 跳升点的 z-score 应显著为正
        assert z.iloc[-1] > 2.0


# ---------------------------------------------------------------------------
# 协整检验
# ---------------------------------------------------------------------------


class TestCointegration:
    def test_cointegrated_pair_detected(self, cointegrated_pair):
        a, b = cointegrated_pair
        eg = engle_granger_test(a, b)
        assert eg["is_cointegrated"] is True
        assert eg["p_value"] <= 0.05

    def test_independent_walks_not_cointegrated(self, independent_walks):
        x, y = independent_walks
        eg = engle_granger_test(x, y)
        assert eg["is_cointegrated"] is False


# ---------------------------------------------------------------------------
# 信号规则
# ---------------------------------------------------------------------------


class TestSignalRules:
    def _make_strategy(self) -> PairsTradingStrategy:
        return PairsTradingStrategy({
            "hedge_ratio": 1.0, "z_entry": 2.0,
            "z_exit": 0.5, "z_stop": 3.0, "window": 20,
        })

    def test_entry_short_when_z_high(self):
        """价差阶跃走高 → z>=2，应进入做空价差(position=-1)。"""
        rng = np.random.default_rng(1)
        n = 80
        b = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.2, n)))
        # 前 40 天平稳，第 40 天起价差阶跃到 +20 并维持
        spread_offset = np.concatenate([np.zeros(40), np.full(40, 20.0)])
        a = b + pd.Series(spread_offset)
        strat = self._make_strategy()
        sig = strat.generate_pair_signals(a, b)
        assert -1 in set(sig["position"].tolist())

    def test_entry_long_when_z_low(self):
        """价差阶跃走低 → z<=-2，应进入做多价差(position=+1)。"""
        rng = np.random.default_rng(2)
        n = 80
        b = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.2, n)))
        spread_offset = np.concatenate([np.zeros(40), np.full(40, -20.0)])
        a = b + pd.Series(spread_offset)
        strat = self._make_strategy()
        sig = strat.generate_pair_signals(a, b)
        assert 1 in set(sig["position"].tolist())

    def test_stop_loss_at_3(self):
        """z 从 2 继续走阔到 3 以上，应止损平仓回到 0。"""
        rng = np.random.default_rng(3)
        n = 100
        b = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.15, n)))
        # 三段：平稳 → 阶跃到 +8（进入 -1）→ 再阶跃到 +40（触发止损）
        spread_offset = np.concatenate([
            np.zeros(30), np.full(35, 8.0), np.full(35, 40.0),
        ])
        a = b + pd.Series(spread_offset)
        strat = self._make_strategy()
        sig = strat.generate_pair_signals(a, b)
        positions = sig["position"].tolist()
        assert -1 in positions  # 先开空价差仓
        # 最后一段应止损平仓（z>=3）
        assert positions[-1] == 0

    def test_signal_dataframe_columns(self, cointegrated_pair):
        a, b = cointegrated_pair
        strat = self._make_strategy()
        sig = strat.generate_pair_signals(a, b)
        assert {"spread", "zscore", "position"} <= set(sig.columns)
        assert set(sig["position"].unique()) <= {-1, 0, 1}


# ---------------------------------------------------------------------------
# 标的对筛选
# ---------------------------------------------------------------------------


class TestScreenPairs:
    def test_high_corr_selected_low_corr_filtered(self):
        rng = np.random.default_rng(5)
        n = 300
        # A/B 高度相关
        b = pd.Series(100 + np.cumsum(rng.normal(0, 1.0, n)))
        a = b + pd.Series(rng.normal(0, 0.3, n))
        # C/D 与 A/B 几乎不相关
        c = pd.Series(200 + np.cumsum(rng.normal(0, 1.5, n)))
        d = pd.Series(50 + np.cumsum(rng.normal(0, 1.2, n)))
        result = screen_pairs({"A": a, "B": b, "C": c, "D": d},
                              min_correlation=0.7)
        pair_names = {(p["symbol_a"], p["symbol_b"]) for p in result}
        # A-B 高相关必被选出
        assert ("A", "B") in pair_names
        # 与 C/D 的跨类对相关性低，应被过滤
        for p in result:
            assert abs(p["correlation"]) >= 0.7

    def test_screen_returns_fields(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = screen_pairs({"A": a, "B": b}, min_correlation=0.5)
        assert len(result) >= 1
        top = result[0]
        assert {"symbol_a", "symbol_b", "correlation", "hedge_ratio",
                "adf_pvalue", "is_cointegrated", "spread_std"} <= set(top)


# ---------------------------------------------------------------------------
# 回测
# ---------------------------------------------------------------------------


class TestBacktest:
    def test_backtest_structure(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = run_pairs_backtest(a, b, hedge_ratio=2.0)
        assert {"metrics", "equity_curve", "trades", "spread_series"} <= set(result)
        m = result["metrics"]
        assert {"total_return", "sharpe", "max_drawdown",
                "num_trades", "final_equity"} <= set(m)
        # 序列长度对齐
        assert len(result["equity_curve"]) == len(a)
        assert len(result["spread_series"]) == len(a)
        for eq in result["equity_curve"][:3]:
            assert {"date", "equity"} <= set(eq)

    def test_backtest_profitable_on_cointegrated(self, cointegrated_pair):
        """协整且价差均值回复的序列，回测应产生交易且权益非零。"""
        a, b = cointegrated_pair
        result = run_pairs_backtest(a, b, hedge_ratio=2.0)
        assert result["metrics"]["num_trades"] >= 1
        assert result["metrics"]["final_equity"] > 0


# ---------------------------------------------------------------------------
# BaseStrategy 接口兼容
# ---------------------------------------------------------------------------


class TestBaseStrategyInterface:
    def test_compute_raw_signals(self, cointegrated_pair):
        a, b = cointegrated_pair
        df = pd.DataFrame({"close": a.values, "price_b": b.values}, index=a.index)
        strat = PairsTradingStrategy({"hedge_ratio": 2.0})
        raw = strat._compute_raw_signals(df)
        assert {"signal", "confidence", "position", "zscore"} <= set(raw.columns)
        # BaseStrategy.generate_signals 应能跑通
        sigs = strat.generate_signals(df, symbol="TEST")
        # 不抛异常即可（可能为 0~多条信号）
        assert isinstance(sigs, list)


# ---------------------------------------------------------------------------
# API 集成测试
# ---------------------------------------------------------------------------


class _MockSim:
    def __init__(self, symbol: str, df: pd.DataFrame):
        self.symbol = symbol
        self.klines = df


class _MockManager:
    def __init__(self, sims: dict):
        self._sims = sims

    def get(self, symbol: str) -> _MockSim:
        return self._sims[symbol]


def _build_client(manager: _MockManager, symbol_set: set) -> TestClient:
    from _routes_pairs import register_pairs_routes

    app = FastAPI()

    def ok(data=None, message="success"):
        return {"code": 0, "message": message, "data": data}

    def err(code, message, http_status=400):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=http_status,
                            content={"code": code, "message": message, "data": None})

    register_pairs_routes(app, manager, symbol_set, ok, err)
    return TestClient(app)


@pytest.fixture(scope="module")
def api_client() -> TestClient:
    from data.data_fetcher import normalize_symbol

    rng = np.random.default_rng(11)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    # 构造两对协整资产 + 一个独立资产
    b = pd.Series(100 + np.cumsum(rng.normal(0.05, 1.0, n)))
    a = 2.0 * b + pd.Series(rng.normal(0, 0.3, n))
    c = pd.Series(50 + np.cumsum(rng.normal(0.03, 0.8, n)))
    d = c * 1.5 + pd.Series(rng.normal(0, 0.2, n))  # 与 c 协整
    e = pd.Series(200 + np.cumsum(rng.normal(0.0, 1.5, n)))  # 独立

    sims = {}
    symbol_set = set()
    for sym, series in [("AAA", a), ("BBB", b), ("CCC", c), ("DDD", d), ("EEE", e)]:
        norm = normalize_symbol(sym)
        symbol_set.add(norm)
        df = pd.DataFrame({"open": series, "high": series, "low": series,
                           "close": series, "volume": 1000}, index=dates)
        sims[norm] = _MockSim(norm, df)
    return _build_client(_MockManager(sims), symbol_set)


class TestPairsAPI:
    def test_screen_endpoint(self, api_client):
        r = api_client.post("/api/pairs/screen", json={
            "symbols": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "min_correlation": 0.7,
        })
        body = r.json()
        assert body["code"] == 0
        d = body["data"]
        assert "pairs" in d and "total" in d
        assert d["total"] == len(d["pairs"])
        for p in d["pairs"]:
            assert {"symbol_a", "symbol_b", "correlation",
                    "hedge_ratio", "adf_pvalue", "is_cointegrated"} <= set(p)

    def test_backtest_endpoint(self, api_client):
        from data.data_fetcher import normalize_symbol
        sa = normalize_symbol("AAA")
        sb = normalize_symbol("BBB")
        r = api_client.post("/api/pairs/backtest", json={
            "symbol_a": sa, "symbol_b": sb, "hedge_ratio": 2.0,
            "z_entry": 2.0, "z_exit": 0.5, "z_stop": 3.0,
        })
        body = r.json()
        assert body["code"] == 0
        d = body["data"]
        assert {"metrics", "equity_curve", "trades", "spread_series"} <= set(d)

    def test_list_endpoint_after_screen(self, api_client):
        # 先触发一次 screen 填充缓存
        api_client.post("/api/pairs/screen", json={
            "symbols": ["AAA", "BBB", "CCC", "DDD"], "min_correlation": 0.7,
        })
        r = api_client.get("/api/pairs/list")
        body = r.json()
        assert body["code"] == 0
        assert "pairs" in body["data"]
        assert "count" in body["data"]

    def test_invalid_symbol_rejected(self, api_client):
        r = api_client.post("/api/pairs/backtest", json={
            "symbol_a": "NOPE", "symbol_b": "AAA",
        })
        assert r.json()["code"] != 0

    def test_screen_needs_two_symbols(self, api_client):
        r = api_client.post("/api/pairs/screen", json={"symbols": ["AAA"]})
        assert r.json()["code"] != 0
