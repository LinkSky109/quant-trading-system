"""VaR/CVaR/压力测试模型单元测试 + API 集成测试。

覆盖：
- 历史模拟法 VaR/CVaR 计算正确性
- 参数法与历史法的合理性对比
- CVaR >= VaR；99% VaR > 95% VaR
- 压力测试各场景返回结构
- 组合 VaR 分解：各 component 之和 ≈ 组合 VaR
- 等权组合 vs 单标的 VaR 分散化效果
- 收益率/统计量计算正确性
- 三个 API 端点集成测试（构造 mock manager）
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

from risk.var_model import STRESS_SCENARIOS, VaRModel  # noqa: E402


# ---------------------------------------------------------------------------
# 夹具：构造已知收益率序列
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simple_returns() -> pd.Series:
    """构造已知分布的收益率序列：含几个明确的尾部损失。"""
    rng = np.random.default_rng(42)
    normal = rng.normal(0.0005, 0.01, size=200)
    tail = np.array([-0.08, -0.05, -0.03])  # 明确的尾部
    return pd.Series(np.concatenate([normal, tail]))


@pytest.fixture(scope="module")
def two_asset_returns() -> pd.DataFrame:
    """两只相关标的的收益率矩阵。"""
    rng = np.random.default_rng(7)
    a = rng.normal(0.0005, 0.015, size=400)
    b = 0.6 * rng.normal(0.0003, 0.012, size=400) + 0.4 * a
    return pd.DataFrame({"AAA": a, "BBB": b})


# ---------------------------------------------------------------------------
# 工具方法
# ---------------------------------------------------------------------------


class TestUtils:
    def test_calculate_returns(self):
        prices = pd.Series([100.0, 110.0, 121.0])
        rets = VaRModel.calculate_returns(prices)
        assert len(rets) == 2
        assert rets.iloc[0] == pytest.approx(0.10)
        assert rets.iloc[1] == pytest.approx(0.10)

    def test_get_stats(self, simple_returns):
        stats = VaRModel.get_stats(simple_returns)
        assert set(stats) >= {"mean", "volatility", "skewness", "kurtosis", "max_drawdown"}
        assert stats["count"] == len(simple_returns)
        assert stats["volatility"] > 0
        assert stats["max_drawdown"] <= 0

    def test_check_var_threshold(self):
        assert VaRModel.check_var_threshold(0.05, 0.03) is True
        assert VaRModel.check_var_threshold(0.01, 0.03) is False


# ---------------------------------------------------------------------------
# 历史模拟法
# ---------------------------------------------------------------------------


class TestHistoricalVaR:
    def test_var_matches_percentile(self, simple_returns):
        """VaR 应等于 -5% 分位数（独立用 numpy 复算）。"""
        var = VaRModel.historical_var(simple_returns, 0.95)
        expected = max(0.0, -float(np.percentile(simple_returns.values, 5)))
        assert var == pytest.approx(expected, rel=1e-9)

    def test_cvar_ge_var(self, simple_returns):
        var = VaRModel.historical_var(simple_returns, 0.95)
        cvar = VaRModel.historical_cvar(simple_returns, 0.95)
        assert cvar >= var - 1e-12

    def test_99_var_gt_95_var(self, simple_returns):
        v95 = VaRModel.historical_var(simple_returns, 0.95)
        v99 = VaRModel.historical_var(simple_returns, 0.99)
        assert v99 > v95

    def test_empty_returns(self):
        assert VaRModel.historical_var(pd.Series(dtype=float)) == 0.0
        assert VaRModel.historical_cvar(pd.Series(dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# 参数法
# ---------------------------------------------------------------------------


class TestParametricVaR:
    def test_param_var_positive(self, two_asset_returns):
        var = VaRModel.parametric_var([0.5, 0.5], two_asset_returns, 0.95)
        cvar = VaRModel.parametric_cvar([0.5, 0.5], two_asset_returns, 0.95)
        assert var > 0
        assert cvar >= var - 1e-9

    def test_param_vs_historical_reasonable(self, two_asset_returns):
        """参数法与历史法应在同一数量级（同符号、不出现离谱偏差）。"""
        w = np.array([0.5, 0.5])
        port = (two_asset_returns * w).sum(axis=1)
        hist_var = VaRModel.historical_var(port, 0.95)
        para_var = VaRModel.parametric_var([0.5, 0.5], two_asset_returns, 0.95)
        # 两者差距不应超过 3 倍（都是日度 95% VaR，数量级应一致）
        assert 0.2 * hist_var <= para_var <= 3.0 * hist_var


# ---------------------------------------------------------------------------
# 组合 VaR 分解
# ---------------------------------------------------------------------------


class TestComponentVaR:
    def test_components_sum_to_total(self, two_asset_returns):
        decomp = VaRModel.component_var([0.5, 0.5], two_asset_returns, 0.95)
        total = decomp["total_var"]
        comp_sum = sum(item["component_var"] for item in decomp["items"])
        assert comp_sum == pytest.approx(total, rel=1e-9, abs=1e-12)

    def test_component_items_structure(self, two_asset_returns):
        decomp = VaRModel.component_var([0.5, 0.5], two_asset_returns, 0.95)
        assert len(decomp["items"]) == 2
        for item in decomp["items"]:
            assert {"symbol", "weight", "marginal_var", "component_var",
                    "contribution_pct"} <= set(item)

    def test_equal_portfolio_lower_than_single(self, two_asset_returns):
        """分散化：等权组合 VaR 应小于单标的 standalone VaR 之和。"""
        port_var = VaRModel.parametric_var([0.5, 0.5], two_asset_returns, 0.95)
        single_a = VaRModel.parametric_var([1.0, 0.0], two_asset_returns, 0.95)
        single_b = VaRModel.parametric_var([0.0, 1.0], two_asset_returns, 0.95)
        # 组合波动因相关性 < 1 而被稀释
        assert port_var < 0.5 * (single_a + single_b) + 1e-9


# ---------------------------------------------------------------------------
# 压力测试
# ---------------------------------------------------------------------------


class TestStressTest:
    def test_list_scenarios(self):
        items = VaRModel.list_scenarios()
        keys = {i["key"] for i in items}
        assert keys == set(STRESS_SCENARIOS.keys())
        for it in items:
            assert "description" in it

    def test_2008_crisis_structure(self):
        result = VaRModel.stress_test(
            [0.5, 0.5], {"AAA": 100.0, "BBB": 50.0}, "2008_crisis", 1_000_000.0
        )
        assert result["scenario"] == "2008_crisis"
        assert result["loss_pct"] == pytest.approx(-0.50)
        assert result["portfolio_loss"] == pytest.approx(-500_000.0)
        assert len(result["positions_impact"]) == 2
        for pos in result["positions_impact"]:
            assert {"symbol", "weight", "price_before", "price_after", "pnl"} <= set(pos)

    def test_single_day_rise(self):
        result = VaRModel.stress_test(
            [1.0], {"AAA": 100.0}, "single_day_rise_5", 100_000.0
        )
        assert result["loss_pct"] == pytest.approx(0.05)
        assert result["portfolio_loss"] == pytest.approx(5_000.0)

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError):
            VaRModel.stress_test([1.0], {"AAA": 100.0}, "no_such_scenario")


# ---------------------------------------------------------------------------
# API 集成测试（mock manager）
# ---------------------------------------------------------------------------


class _MockSim:
    """模拟 RealTimeSimulator：携带一段构造好的 klines。"""

    def __init__(self, symbol: str, df: pd.DataFrame):
        self.symbol = symbol
        self.klines = df


class _MockManager:
    """模拟 MultiSymbolManager。"""

    def __init__(self, sims: dict):
        self._sims = sims

    def get(self, symbol: str) -> _MockSim:
        return self._sims[symbol]


def _build_client(manager: _MockManager, symbol_set: set) -> TestClient:
    """注册风险路由到一个独立最小 app，避免导入 server.py 触发网络/后台线程。"""
    from _routes_risk import register_risk_routes

    app = FastAPI()

    def ok(data=None, message="success"):
        return {"code": 0, "message": message, "data": data}

    def err(code, message, http_status=400):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=http_status,
                            content={"code": code, "message": message, "data": None})

    register_risk_routes(app, manager, symbol_set, ok, err)
    return TestClient(app)


@pytest.fixture(scope="module")
def api_client() -> TestClient:
    from data.data_fetcher import normalize_symbol

    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=250, freq="B")
    sims = {}
    symbol_set = set()
    for sym, vol in (("AAA", 0.015), ("BBB", 0.012)):
        norm = normalize_symbol(sym)
        symbol_set.add(norm)
        rets = rng.normal(0.0005, vol, size=len(dates))
        close = 100.0 * np.cumprod(1 + rets)
        df = pd.DataFrame({"open": close, "high": close, "low": close,
                           "close": close, "volume": 1000}, index=dates)
        sims[norm] = _MockSim(norm, df)
    return _build_client(_MockManager(sims), symbol_set)


class TestRiskAPI:
    def test_var_status(self, api_client):
        r = api_client.get("/api/risk/var_status")
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["methods"] == ["historical", "parametric"]
        assert 0.95 in body["data"]["confidences"]
        assert len(body["data"]["scenarios"]) == len(STRESS_SCENARIOS)

    def test_var_historical(self, api_client):
        r = api_client.post("/api/risk/var", json={
            "symbols": ["AAA", "BBB"], "weights": [0.5, 0.5],
            "method": "historical", "confidence": 0.95,
        })
        assert r.status_code == 200
        d = r.json()["data"]
        assert d["method"] == "historical"
        assert d["var"] > 0
        assert d["cvar"] >= d["var"]
        assert len(d["component_var"]) == 2
        assert "portfolio_volatility" in d
        assert d["stats"]["count"] > 0

    def test_var_parametric(self, api_client):
        r = api_client.post("/api/risk/var", json={
            "symbols": ["AAA", "BBB"], "method": "parametric", "confidence": 0.99,
        })
        assert r.status_code == 200
        d = r.json()["data"]
        assert d["method"] == "parametric"
        assert d["confidence"] == 0.99
        assert d["var"] > 0

    def test_var_invalid_symbol(self, api_client):
        r = api_client.post("/api/risk/var", json={
            "symbols": ["NOPE"], "method": "historical",
        })
        assert r.json()["code"] != 0

    def test_var_bad_method(self, api_client):
        r = api_client.post("/api/risk/var", json={
            "symbols": ["AAA"], "method": "monte_carlo",
        })
        assert r.json()["code"] != 0

    def test_stress_test_api(self, api_client):
        r = api_client.post("/api/risk/stress_test", json={
            "symbols": ["AAA", "BBB"], "weights": [0.5, 0.5],
            "scenario": "2015_crash", "portfolio_value": 1_000_000,
        })
        body = r.json()
        assert body["code"] == 0
        d = body["data"]
        assert d["scenario"] == "2015_crash"
        assert d["loss_pct"] == pytest.approx(-0.45)
        assert len(d["positions_impact"]) == 2
