"""Brinson 绩效归因模块单元测试 + API 集成测试。

覆盖：
1. 两组/三组行业手算算例验证 AR/SR/IR 单组值；
2. 总效应闭合：Σ(AR+SR+IR) ≈ r_p - r_b（|closure_residual| < 1e-6）；
3. Brinson-Fachler 变体与 BHB 结果关系正确（权重闭合时逐组 AR 一致）；
4. 权重不闭合 / 长度不一致抛 ValueError；
5. 多期 Cariño 链接：两期算例链接总效应≈复利超额收益（<1e-4）；GRAP smoke；
6. 瀑布图数据：首尾分别为 r_b 和 r_p，各段变化等于三效应；
7. 多维度（dimension 参数）smoke + mock 分组标注存在；
8. 两个 API 端点集成测试（提交→取结果→闭合；task_id 不存在返回 404）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web-dashboard"))

from analysis.brinson import BrinsonAttribution  # noqa: E402


# ---------------------------------------------------------------------------
# 手算算例
# ---------------------------------------------------------------------------

# 两组：金融/科技
TWO = dict(
    sectors=["金融", "科技"],
    w_p=[0.6, 0.4],
    w_b=[0.5, 0.5],
    r_p_sector=[0.10, 0.20],
    r_b_sector=[0.05, 0.10],
)
# 手算：r_p=0.6*.1+.4*.2=.14；r_b=.5*.05+.5*.1=.075；超额=.065
# 金融: dw=.1,dr=.05 -> AR=.1*(.05-.075)=-.0025; SR=.5*.05=.025; IR=.1*.05=.005
# 科技: dw=-.1,dr=.10 -> AR=-.1*(.10-.075)=-.0025; SR=.5*.10=.05; IR=-.1*.10=-.01

# 三组：A/B/C
THREE = dict(
    sectors=["A", "B", "C"],
    w_p=[0.5, 0.3, 0.2],
    w_b=[0.4, 0.4, 0.2],
    r_p_sector=[0.08, 0.06, 0.10],
    r_b_sector=[0.06, 0.05, 0.08],
)
# 手算：r_p=.5*.08+.3*.06+.2*.1=.078；r_b=.4*.06+.4*.05+.2*.08=.06；超额=.018
# A: dw=.1,dr=.02 -> AR=.1*(.06-.06)=0; SR=.4*.02=.008; IR=.1*.02=.002
# B: dw=-.1,dr=.01 -> AR=-.1*(.05-.06)=.001; SR=.4*.01=.004; IR=-.1*.01=-.001
# C: dw=0,dr=.02 -> AR=0; SR=.2*.02=.004; IR=0


class TestHandCalculation:
    def test_two_group_values(self):
        res = BrinsonAttribution.attribute(**TWO, model="bhb")
        g = {x.sector: x for x in res.groups}
        assert g["金融"].ar == pytest.approx(-0.0025, abs=1e-9)
        assert g["金融"].sr == pytest.approx(0.025, abs=1e-9)
        assert g["金融"].ir == pytest.approx(0.005, abs=1e-9)
        assert g["科技"].ar == pytest.approx(-0.0025, abs=1e-9)
        assert g["科技"].sr == pytest.approx(0.05, abs=1e-9)
        assert g["科技"].ir == pytest.approx(-0.01, abs=1e-9)

    def test_three_group_values(self):
        res = BrinsonAttribution.attribute(**THREE, model="bhb")
        g = {x.sector: x for x in res.groups}
        assert g["A"].ar == pytest.approx(0.0, abs=1e-12)
        assert g["A"].sr == pytest.approx(0.008, abs=1e-9)
        assert g["A"].ir == pytest.approx(0.002, abs=1e-9)
        assert g["B"].ar == pytest.approx(0.001, abs=1e-9)
        assert g["B"].sr == pytest.approx(0.004, abs=1e-9)
        assert g["B"].ir == pytest.approx(-0.001, abs=1e-9)
        assert g["C"].sr == pytest.approx(0.004, abs=1e-9)
        assert g["C"].ir == pytest.approx(0.0, abs=1e-12)

    def test_total_effects_match_hand(self):
        res = BrinsonAttribution.attribute(**THREE, model="bhb")
        assert res.allocation_effect == pytest.approx(0.001, abs=1e-9)
        assert res.selection_effect == pytest.approx(0.016, abs=1e-9)
        assert res.interaction_effect == pytest.approx(0.001, abs=1e-9)
        assert res.portfolio_return == pytest.approx(0.078, abs=1e-9)
        assert res.benchmark_return == pytest.approx(0.06, abs=1e-9)


class TestClosure:
    @pytest.mark.parametrize("case", [TWO, THREE])
    def test_closure_residual_tiny(self, case):
        res = BrinsonAttribution.attribute(**case, model="bhb")
        expected_active = res.portfolio_return - res.benchmark_return
        assert res.total_active_return == pytest.approx(expected_active, abs=1e-9)
        assert abs(res.closure_residual) < 1e-6

    def test_closure_randomized(self):
        import numpy as np
        rng = np.random.default_rng(7)
        for _ in range(50):
            n = rng.integers(2, 6)
            wp = rng.dirichlet(np.ones(n))
            wb = rng.dirichlet(np.ones(n))
            rp = rng.normal(0.05, 0.05, n)
            rb = rng.normal(0.05, 0.05, n)
            res = BrinsonAttribution.attribute(
                [f"s{i}" for i in range(n)], wp, wb, rp, rb, model="bhb"
            )
            assert abs(res.closure_residual) < 1e-9


class TestFachler:
    def test_fachler_vs_bhb_relationship(self):
        bhb = BrinsonAttribution.attribute(**TWO, model="bhb")
        fach = BrinsonAttribution.attribute(**TWO, model="fachler")
        # 契约：r_b == r_b_total，权重闭合时逐组配置效应一致
        for gb, gf in zip(bhb.groups, fach.groups):
            assert gb.ar == pytest.approx(gf.ar, abs=1e-12)
            assert gb.sr == pytest.approx(gf.sr, abs=1e-12)
            assert gb.ir == pytest.approx(gf.ir, abs=1e-12)
        assert bhb.allocation_effect == pytest.approx(fach.allocation_effect, abs=1e-12)
        # 两者均闭合
        assert abs(fach.closure_residual) < 1e-9
        assert abs(bhb.closure_residual) < 1e-9

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            BrinsonAttribution.attribute(**TWO, model="sharp")


class TestValidation:
    def test_weights_not_closed_raises(self):
        with pytest.raises(ValueError):
            BrinsonAttribution.attribute(
                ["A", "B"], [0.5, 0.3], [0.5, 0.5], [0.1, 0.1], [0.1, 0.1]
            )

    def test_benchmark_weights_not_closed_raises(self):
        with pytest.raises(ValueError):
            BrinsonAttribution.attribute(
                ["A", "B"], [0.5, 0.5], [0.5, 0.4], [0.1, 0.1], [0.1, 0.1]
            )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            BrinsonAttribution.attribute(
                ["A", "B", "C"], [0.4, 0.3, 0.3], [0.5, 0.5], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]
            )


class TestMultiPeriod:
    def _periods(self):
        return [
            dict(sectors=["A", "B"], w_p=[0.5, 0.5], w_b=[0.5, 0.5],
                 r_p_sector=[0.02, 0.02], r_b_sector=[0.01, 0.01]),
            dict(sectors=["A", "B"], w_p=[0.5, 0.5], w_b=[0.5, 0.5],
                 r_p_sector=[0.015, 0.015], r_b_sector=[0.005, 0.005]),
        ]

    def test_carino_links_to_compounded_active(self):
        mp = BrinsonAttribution.multi_period_attribute(self._periods(), method="carino")
        assert mp["method"] == "carino"
        # 链接后总效应 ≈ 复利超额收益（容差 1e-4）
        assert abs(mp["link_residual"]) < 1e-4
        assert mp["linked"]["total_active_return"] == pytest.approx(
            mp["compounded_active_return"], abs=1e-4
        )

    def test_grap_smoke(self):
        mp = BrinsonAttribution.multi_period_attribute(self._periods(), method="grap")
        assert mp["method"] == "grap"
        # GRAP 算术链接：两期主动收益 .01+.01=.02
        assert mp["linked"]["total_active_return"] == pytest.approx(0.02, abs=1e-9)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            BrinsonAttribution.multi_period_attribute(self._periods(), method="magic")


class TestWaterfall:
    def test_waterfall_endpoints_and_steps(self):
        res = BrinsonAttribution.attribute(**TWO, model="bhb")
        bars = res.waterfall_data()
        assert bars[0]["name"] == "benchmark_return"
        assert bars[0]["end"] == pytest.approx(res.benchmark_return, abs=1e-12)
        assert bars[-1]["name"] == "portfolio_return"
        assert bars[-1]["end"] == pytest.approx(res.portfolio_return, abs=1e-12)
        # 中间三段变化 = 三效应
        assert bars[1]["change"] == pytest.approx(res.allocation_effect, abs=1e-12)
        assert bars[2]["change"] == pytest.approx(res.selection_effect, abs=1e-12)
        assert bars[3]["change"] == pytest.approx(res.interaction_effect, abs=1e-12)
        # 末段组合收益 = r_b + 三效应
        assert bars[-1]["end"] == pytest.approx(
            res.benchmark_return
            + res.allocation_effect
            + res.selection_effect
            + res.interaction_effect,
            abs=1e-12,
        )

    def test_sector_bar_data(self):
        res = BrinsonAttribution.attribute(**TWO, model="bhb")
        bars = res.sector_bar_data()
        assert {b["sector"] for b in bars} == {"金融", "科技"}
        for b in bars:
            assert set(b) >= {"sector", "ar", "sr", "ir", "total"}


class TestMultiDimension:
    def test_dimension_smoke(self):
        res = BrinsonAttribution.attribute(**TWO, model="bhb", dimension="style")
        assert res.dimension == "style"
        assert abs(res.closure_residual) < 1e-9

    def test_mock_grouping_annotation(self):
        holdings = [
            {"symbol": "AAA", "weight": 0.5},
            {"symbol": "BBB", "weight": 0.3},
            {"symbol": "CCC", "weight": 0.2},
        ]
        symbol_returns = {"AAA": 0.05, "BBB": 0.03, "CCC": 0.10}
        res = BrinsonAttribution.attribute_holdings(
            holdings=holdings,
            symbol_returns=symbol_returns,
        )
        assert res.sector_source == "mock_simplified"
        assert "sector_classification_warning" in res.meta
        assert "mock" in res.meta["sector_classification_warning"]

    def test_mock_sector_stable(self):
        s1 = BrinsonAttribution.mock_sector("600519")
        s2 = BrinsonAttribution.mock_sector("600519")
        assert s1 == s2  # 稳定哈希


# ---------------------------------------------------------------------------
# API 集成测试
# ---------------------------------------------------------------------------


def _build_client() -> TestClient:
    from _routes_brinson import register_brinson_routes

    app = FastAPI()

    def ok(data=None, message="success"):
        return {"code": 0, "message": message, "data": data}

    def err(code, message, http_status=400):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=http_status,
                            content={"code": code, "message": message, "data": None})

    register_brinson_routes(app, ok, err)
    return TestClient(app)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return _build_client()


class TestBrinsonAPI:
    def test_post_then_get_closure(self, client):
        payload = {
            "groups": [
                {"name": "金融", "w_p": 0.6, "w_b": 0.5, "r_p": 0.10, "r_b": 0.05},
                {"name": "科技", "w_p": 0.4, "w_b": 0.5, "r_p": 0.20, "r_b": 0.10},
            ],
            "model": "bhb",
            "dimension": "industry",
        }
        r = client.post("/api/attribution/brinson", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        task_id = body["data"]["task_id"]
        assert task_id

        r2 = client.get(f"/api/attribution/brinson/{task_id}")
        assert r2.status_code == 200
        data = r2.json()["data"]
        assert abs(data["closure_residual"]) < 1e-6
        assert data["total_active_return"] == pytest.approx(0.065, abs=1e-9)
        # 瀑布图已附带
        assert data["waterfall"][0]["name"] == "benchmark_return"
        assert data["waterfall"][-1]["name"] == "portfolio_return"

    def test_task_id_not_found_404(self, client):
        r = client.get("/api/attribution/brinson/nonexistent123")
        assert r.status_code == 404

    def test_validation_error_400(self, client):
        # 权重不闭合
        payload = {
            "groups": [
                {"name": "A", "w_p": 0.6, "w_b": 0.5, "r_p": 0.1, "r_b": 0.05},
                {"name": "B", "w_p": 0.2, "w_b": 0.5, "r_p": 0.2, "r_b": 0.1},
            ]
        }
        r = client.post("/api/attribution/brinson", json=payload)
        assert r.status_code == 400

    def test_invalid_model_400(self, client):
        payload = {"groups": [], "model": "foo"}
        r = client.post("/api/attribution/brinson", json=payload)
        assert r.status_code == 400

    def test_multi_period_endpoint(self, client):
        payload = {
            "periods": [
                [
                    {"name": "A", "w_p": 0.5, "w_b": 0.5, "r_p": 0.02, "r_b": 0.01},
                    {"name": "B", "w_p": 0.5, "w_b": 0.5, "r_p": 0.02, "r_b": 0.01},
                ],
                [
                    {"name": "A", "w_p": 0.5, "w_b": 0.5, "r_p": 0.015, "r_b": 0.005},
                    {"name": "B", "w_p": 0.5, "w_b": 0.5, "r_p": 0.015, "r_b": 0.005},
                ],
            ],
            "link_method": "carino",
        }
        r = client.post("/api/attribution/brinson", json=payload)
        assert r.status_code == 200
        task_id = r.json()["data"]["task_id"]
        data = client.get(f"/api/attribution/brinson/{task_id}").json()["data"]
        assert data["method"] == "carino"
        assert abs(data["link_residual"]) < 1e-4
