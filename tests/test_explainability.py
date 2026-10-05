"""Jev 可解释性模块单元测试 + API 集成测试（REQ-P2-12）。

全部用注入的**确定性 mock predict**（特征线性映射到概率），不依赖网络/8765 服务。
覆盖：
1. 特征重要性：遮蔽关键特征后概率显著变化、排序正确；
2. 单次解释：贡献度符号正确（支持/反对）、explain_text 含真实数值且覆盖
   RSI 超买与放量两条语义规则；
3. 反事实：可反转场景断言最小变更方向/from/to；不可反转场景断言 infeasible；
4. SHAP 近似：返回结构正确、shap_available 布尔标注；
5. 决策路径数据结构完整（首尾分别为输入特征与最终决策）；
6. 全局重要性多样本平均 smoke；
7. 全部 API 端点 TestClient（正常 + 404 + 校验失败 400）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web-dashboard"))

from jev.explainability import JevExplainer  # noqa: E402


# ---------------------------------------------------------------------------
# 确定性 mock predict：logits = 线性 + softmax，可手算
#   logit_buy  = 1.0 * ma5_ma20_ratio
#   logit_sell = 0.05 * rsi
#   logit_hold = 1.0
# 其余特征（price/volume_ratio/price_change_5d/macd_signal/volatility_20d）
# 不进入 predict，仅用于解释文本的语义规则演示。
# ---------------------------------------------------------------------------

def linear_predict(features):
    d = {str(f["feature"]): float(f["value"]) for f in features}
    vals = {
        "buy": 1.0 * d.get("ma5_ma20_ratio", 1.0),
        "sell": 0.05 * d.get("rsi", 50.0),
        "hold": 1.0,
    }
    m = max(vals.values())
    ex = {k: math.exp(v - m) for k, v in vals.items()}
    s = sum(ex.values())
    return {k: v / s for k, v in ex.items()}


# 基准特征：rsi=80（超买）→ logit_sell=4.0 占优，最终动作=sell
FEATURES = [
    {"feature": "price", "value": 100.0},
    {"feature": "price_change_5d", "value": 0.02},
    {"feature": "ma5_ma20_ratio", "value": 1.10},
    {"feature": "volume_ratio", "value": 1.5},
    {"feature": "rsi", "value": 80.0},
    {"feature": "macd_signal", "value": -1.0},
    {"feature": "volatility_20d", "value": 0.20},
]


def _make_explainer(predict_fn=linear_predict) -> JevExplainer:
    return JevExplainer(predict_callable=predict_fn)


class TestFeatureImportance:
    def test_masking_key_feature_changes_probs_and_order(self):
        explainer = _make_explainer()
        fi = explainer.feature_importance(FEATURES)

        # rsi 是主导特征（logit_sell 由它决定），应为整体/卖出维度第一
        assert fi["overall"][0]["feature"] == "rsi"
        assert fi["by_action"]["sell"][0]["feature"] == "rsi"

        # 遮蔽 rsi（→基线50）后 sell 概率显著下降
        rsi_row = next(r for r in fi["overall"] if r["feature"] == "rsi")
        assert rsi_row["masked_probs"]["sell"] < fi["base_probs"]["sell"]
        assert rsi_row["importance"] > 0.01

        # ma5_ma20 从 1.10 遮蔽到 1.0，影响很小，重要性低于 rsi
        ma_row = next(r for r in fi["overall"] if r["feature"] == "ma5_ma20_ratio")
        assert ma_row["importance"] < rsi_row["importance"]

    def test_global_importance_average(self):
        explainer = _make_explainer()
        samples = [
            FEATURES,
            [{**f, "value": (f["value"] * 1.05 if f["feature"] == "rsi" else f["value"])}
             for f in FEATURES],
        ]
        g = explainer.global_feature_importance(samples)
        assert g["n_samples"] == 2
        assert g["overall"][0]["feature"] == "rsi"
        assert all("delta" in row for row in g["overall"])


class TestExplain:
    def test_contribution_sign_and_direction(self):
        explainer = _make_explainer()
        exp = explainer.explain(FEATURES)
        assert exp["final_action"] == "sell"

        rsi_c = next(c for c in exp["contributions"] if c["feature"] == "rsi")
        # 遮蔽 rsi → sell 概率下降 → contribution_to_sell 为正 → 支持卖出
        assert rsi_c["contribution_to_sell"] > 0
        assert rsi_c["direction"] == "support"

    def test_explain_text_has_real_numbers_and_rules(self):
        explainer = _make_explainer()
        exp = explainer.explain(FEATURES)
        text = exp["explain_text"]

        # 真实数值填入
        assert "RSI=80.0" in text
        assert "1.50" in text
        assert "100.00" in text
        # 两条关键语义规则：RSI 超买 + 放量
        assert "超买" in text
        assert "放量" in text
        # 结尾综合结论句
        assert "Jev 最终建议" in text


class TestCounterfactual:
    def test_reversible_buy_direction(self):
        explainer = _make_explainer()
        # 当前 sell；目标 buy：把 rsi 从 80 往下压到 <22 即可让 buy 占优
        cf = explainer.counterfactual(FEATURES, "buy")
        assert cf["feasible"] is True
        chg = cf["changes"][0]
        assert chg["feature"] == "rsi"
        assert chg["from"] == pytest.approx(80.0)
        assert chg["change"] < 0  # 减小 rsi
        assert chg["to"] < 80.0
        # 反事实后 buy 成为最高概率
        assert max(cf["final_probs"], key=cf["final_probs"].get) == "buy"

    def test_infeasible_when_decision_locked(self):
        # predict 恒定：sell 恒为 0.8，任何特征都不影响概率 → 无法反转
        def locked_predict(features):
            return {"buy": 0.1, "sell": 0.8, "hold": 0.1}

        explainer = JevExplainer(predict_callable=locked_predict)
        cf = explainer.counterfactual(FEATURES, "buy")
        assert cf["feasible"] is False
        assert "无法" in cf["reason"]

    def test_already_target_no_change(self):
        explainer = _make_explainer()
        cf = explainer.counterfactual(FEATURES, "sell")
        assert cf["feasible"] is True
        assert cf["already_target"] is True
        assert cf["changes"] == []


class TestShapApprox:
    def test_structure_and_availability_flag(self):
        explainer = _make_explainer()
        sv = explainer.shap_values(FEATURES)
        assert isinstance(sv["shap_available"], bool)
        names = {f["feature"] for f in FEATURES}
        assert set(sv["values"].keys()) == names
        for vals in sv["values"].values():
            assert set(vals.keys()) == {"buy", "sell", "hold"}


class TestDecisionPath:
    def test_first_input_last_decision(self):
        explainer = _make_explainer()
        dp = explainer.decision_path(FEATURES)
        assert dp["steps"][0]["type"] == "input"
        assert dp["steps"][0]["features"]  # 非空特征列表
        assert dp["steps"][-1]["type"] == "decision"
        assert dp["steps"][-1]["action"] == "sell"
        # 中间为特征贡献节点
        middle = [s for s in dp["steps"] if s["type"] == "feature_contribution"]
        assert len(middle) == len(FEATURES)
        assert dp["final_action"] == "sell"


# ---------------------------------------------------------------------------
# API 集成测试
# ---------------------------------------------------------------------------

def _build_client() -> TestClient:
    from _routes_explain import register_explain_routes

    app = FastAPI()

    def ok(data=None, message="success"):
        return {"code": 0, "message": message, "data": data}

    def err(code, message, http_status=400):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=http_status,
                            content={"code": code, "message": message, "data": None})

    register_explain_routes(app, ok, err)
    return TestClient(app)


FEATURE_BODY = [
    {"feature": "price", "value": 100.0},
    {"feature": "price_change_5d", "value": 0.02},
    {"feature": "ma5_ma20_ratio", "value": 1.10},
    {"feature": "volume_ratio", "value": 1.5},
    {"feature": "rsi", "value": 80.0},
    {"feature": "macd_signal", "value": -1.0},
    {"feature": "volatility_20d", "value": 0.20},
]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return _build_client()


class TestExplainAPI:
    def test_post_explain_then_get_history(self, client):
        r = client.post("/api/jev/explain", json={"features": FEATURE_BODY})
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        data = body["data"]
        assert data["decision_id"]
        assert data["final_action"] in ("buy", "sell", "hold")
        assert "explain_text" in data
        assert len(data["contributions"]) == len(FEATURE_BODY)

        # 历史可取回
        rid = data["decision_id"]
        r2 = client.get(f"/api/jev/explain/{rid}")
        assert r2.status_code == 200
        assert r2.json()["data"]["decision_id"] == rid

    def test_history_404(self, client):
        r = client.get("/api/jev/explain/doesnotexist999")
        assert r.status_code == 404

    def test_explain_requires_features_400(self, client):
        r = client.post("/api/jev/explain", json={"decision_id": "abc"})
        assert r.status_code == 400


class TestImportanceAPI:
    def test_single_features(self, client):
        r = client.post("/api/jev/feature_importance", json={"features": FEATURE_BODY})
        assert r.status_code == 200
        data = r.json()["data"]
        assert "overall" in data and len(data["overall"]) == len(FEATURE_BODY)
        # 降序
        imps = [row["importance"] for row in data["overall"]]
        assert imps == sorted(imps, reverse=True)

    def test_samples_global(self, client):
        r = client.post("/api/jev/feature_importance",
                       json={"samples": [FEATURE_BODY, FEATURE_BODY]})
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["n_samples"] == 2

    def test_missing_input_400(self, client):
        r = client.post("/api/jev/feature_importance", json={})
        assert r.status_code == 400


class TestCounterfactualAPI:
    def test_counterfactual_ok(self, client):
        r = client.post("/api/jev/counterfactual",
                        json={"features": FEATURE_BODY, "target_action": "buy"})
        assert r.status_code == 200
        data = r.json()["data"]
        assert "feasible" in data

    def test_bad_target_action_400(self, client):
        r = client.post("/api/jev/counterfactual",
                        json={"features": FEATURE_BODY, "target_action": "holdon"})
        assert r.status_code == 400
