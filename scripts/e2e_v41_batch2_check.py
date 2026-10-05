"""v4.1 第2批 3 项需求端到端（E2E）验证脚本。

对运行在 8766 的量化服务发起真实 HTTP 调用，覆盖：
  - REQ-P2-09 数据校验与对账：交叉验证 / 质量评分 / 异常列表 / 修复
  - REQ-P2-12 Jev 可解释性：单次解释 / 特征重要性 / 反事实 / 历史
  - REQ-P1-06 移动端/PWA：manifest / sw / icons / 页面移动友好性

用法：python3 scripts/e2e_v41_batch2_check.py [base_url]
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8766"

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: Any = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -> {detail}")


def call(method: str, path: str, **kw) -> requests.Response:
    return requests.request(method, BASE + path, timeout=60, **kw)


# 标准特征集（RSI 偏高场景）
FEATURES: List[Dict[str, Any]] = [
    {"feature": "price", "value": 100.0},
    {"feature": "price_change_5d", "value": 0.04},
    {"feature": "ma5_ma20_ratio", "value": 1.03},
    {"feature": "volume_ratio", "value": 1.6},
    {"feature": "rsi", "value": 72.0},
    {"feature": "macd_signal", "value": -1.0},
    {"feature": "volatility_20d", "value": 0.25},
]

# ===========================================================================
# 1. 数据校验与对账 — REQ-P2-09
# ===========================================================================
print("== REQ-P2-09 数据校验与对账 ==")

reconcile_body = {
    "symbols": ["600519.SH", "000001.SZ"],
    "sources": {
        # 偏差 0.4% < 0.5% → validated
        "600519.SH": {"quantdash": 100.0, "tencent": 100.4},
        # 偏差 1.0% > 0.5% → flagged
        "000001.SZ": {"quantdash": 100.0, "tencent": 101.0},
    },
}
r = call("POST", "/api/data/reconcile", json=reconcile_body)
b = r.json()
check("交叉验证 code=0", b.get("code") == 0, b)
per = (b.get("data") or {}).get("per_symbol", {})
pc1 = per.get("600519.SH", {}).get("price_check", {})
pc2 = per.get("000001.SZ", {}).get("price_check", {})
check("0.4% 偏差 validated", pc1.get("status") == "validated", pc1)
check("1.0% 偏差 flagged", pc2.get("status") == "flagged", pc2)

# 异常列表（应有价格偏差异常）
r = call("GET", "/api/data/anomalies")
b = r.json()
anoms = b.get("data") or []
check("异常列表 code=0 且含记录", b.get("code") == 0 and len(anoms) > 0, b)
open_anoms = [a for a in anoms if a.get("status") == "open"]
check("存在 open 异常", len(open_anoms) > 0, anoms)

# status 过滤
r = call("GET", "/api/data/anomalies", params={"status": "fixed"})
fixed_list = r.json().get("data") or []
check("status=fixed 过滤返回列表", isinstance(fixed_list, list), r.json())

# 质量评分
r = call("GET", "/api/data/quality")
b = r.json()
check("质量端点 code=0", b.get("code") == 0, b)
qd = b.get("data") or {}
check("含 scores 与 report", "scores" in qd and "report" in qd, qd)

# 修复端点（对一条 open 异常触发；无备用源时应返回非0说明而非500崩溃）
if open_anoms:
    aid = open_anoms[0].get("id")
    r = call("POST", f"/api/data/anomalies/{aid}/fix")
    b = r.json()
    check("修复端点响应连贯（fixed 或明确失败）",
          b.get("code") == 0 or (b.get("code") != 0 and b.get("message")), b)

# 不存在异常修复
r = call("POST", "/api/data/anomalies/NOPE/fix")
check("修复不存在异常返回非0", r.json().get("code") != 0, r.json())

# 空 symbols
r = call("POST", "/api/data/reconcile", json={"symbols": []})
check("空 symbols 被拒", r.json().get("code") != 0, r.json())

# ===========================================================================
# 2. Jev 可解释性 — REQ-P2-12
# ===========================================================================
print("== REQ-P2-12 Jev 可解释性 ==")

r = call("POST", "/api/jev/explain", json={"features": FEATURES})
b = r.json()
check("解释端点 code=0", b.get("code") == 0, b)
ed = b.get("data") or {}
check("含解释文本 explain_text", bool(ed.get("explain_text")), ed)
check("解释文本含真实数值", "72" in ed.get("explain_text", ""), ed.get("explain_text"))
check("含概率 base_probs", isinstance(ed.get("base_probs"), dict), ed)
check("含最终动作", bool(ed.get("final_action")), ed)
check("含特征贡献 contributions", ed.get("contributions") is not None, ed)
decision_id = ed.get("decision_id", "")
check("返回 decision_id", bool(decision_id), ed)

# 历史解释
r = call("GET", f"/api/jev/explain/{decision_id}")
check("历史解释可取 code=0", r.json().get("code") == 0, r.json())
r = call("GET", "/api/jev/explain/NOPE")
check("历史解释不存在返回非0", r.json().get("code") != 0, r.json())

# 特征重要性
r = call("POST", "/api/jev/feature_importance", json={"features": FEATURES})
b = r.json()
check("特征重要性 code=0", b.get("code") == 0, b)
imp = b.get("data")
ranking = imp.get("overall", []) if isinstance(imp, dict) else imp
check("重要性返回排序且非空", isinstance(ranking, list) and len(ranking) > 0
      and ranking[0].get("feature") == "macd_signal", b)

# 反事实（目标 buy）
r = call("POST", "/api/jev/counterfactual",
         json={"features": FEATURES, "target_action": "buy"})
b = r.json()
check("反事实端点 code=0", b.get("code") == 0, b)
cf = b.get("data") or {}
check("反事实含 feasible 标志", "feasible" in cf, cf)
if cf.get("feasible"):
    check("反事实含变更集", isinstance(cf.get("changes"), list) and len(cf["changes"]) > 0, cf)

# ===========================================================================
# 3. 移动端 / PWA — REQ-P1-06
# ===========================================================================
print("== REQ-P1-06 移动端 / PWA ==")

for p, ct in [
    ("manifest.json", "application/manifest+json"),
    ("sw.js", "application/javascript"),
    ("icons/icon-192.png", "image/png"),
    ("icons/icon-512.png", "image/png"),
]:
    r = call("GET", "/" + p)
    check(f"{p} 可访问", r.status_code == 200 and ct in r.headers.get("content-type", ""),
          r.status_code)

r = call("GET", "/manifest.json")
mani = r.json()
check("manifest 含 name/short_name/start_url/display",
      mani.get("name") and mani.get("short_name") and mani.get("start_url")
      and mani.get("display") == "standalone", mani)
check("manifest 含图标", len(mani.get("icons", [])) >= 2, mani)

# 页面含移动端关键标记
r = call("GET", "/")
html = r.text
check("页面含 viewport", "width=device-width" in html, "")
check("页面引用 manifest", 'rel="manifest"' in html, "")
check("页面含底部 Tab 栏", "mobile-tabbar" in html, "")
check("页面注册 Service Worker", "serviceWorker" in html, "")
check("页面含 5 个 Tab", all(t in html for t in ["行情", "交易", "回测", "通知", "我的"]), "")

# 路径穿越防护
r = call("GET", "/icons/..%2f..%2fserver.py")
check("icons 路径穿越被拦", r.status_code in (400, 404), r.status_code)

# ===========================================================================
# 汇总
# ===========================================================================
print(f"\n==== 第2批 E2E：{PASS} passed, {FAIL} failed ====")
sys.exit(1 if FAIL else 0)
