"""v4.1 第1批 4 项需求端到端（E2E）验证脚本。

对运行在 8766 的量化服务发起真实 HTTP 调用，覆盖：
  - REQ-P0-02 OMS：下单 / 查询 / 详情 / 撤单 / 废单 / 成交记录
  - REQ-P1-03 Brinson：提交归因 / 取结果 / 闭合验证
  - REQ-P1-09 因子暴露：暴露计算 / 超限告警 / 中性化建议 / 历史 / 告警列表
  - REQ-P1-10 算法交易：TWAP 创建 / 状态 / 控制 / 报告

用法：python3 scripts/e2e_v41_check.py [base_url]
"""
from __future__ import annotations

import sys
from typing import Any, Dict

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8766"

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: Any = "") -> None:
    """断言并打印结果。"""
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -> {detail}")


def call(method: str, path: str, **kw) -> requests.Response:
    return requests.request(method, BASE + path, timeout=30, **kw)


# ===========================================================================
# 1. OMS — REQ-P0-02
# ===========================================================================
print("== REQ-P0-02 OMS 订单管理 ==")

order_body = {
    "symbol": "600519.SH", "side": "buy", "order_type": "limit",
    "quantity": 100, "limit_price": 1500.0,
    "strategy_name": "e2e_test", "timeout_seconds": 300,
}
r = call("POST", "/api/oms/orders", json=order_body)
body = r.json()
check("下单返回 code=0", body.get("code") == 0, body)
oid1 = (body.get("data") or {}).get("order_id", "")
check("订单自动生成 order_id", bool(oid1), body)

r = call("GET", f"/api/oms/orders/{oid1}")
b = r.json()
check("订单详情 SUBMITTED", (b.get("data") or {}).get("status") == "SUBMITTED", b)

# 第二单（用于废单）
order_body2 = dict(order_body, symbol="000001.SZ", quantity=200)
r = call("POST", "/api/oms/orders", json=order_body2)
oid2 = (r.json().get("data") or {}).get("order_id", "")
check("第二单提交", bool(oid2))

# 列表 + 状态过滤
r = call("GET", "/api/oms/orders", params={"symbol": "600519.SH"})
b = r.json()
check("按 symbol 过滤订单", any(o.get("order_id") == oid1 for o in (b.get("data") or [])), b)

# 撤单
r = call("DELETE", f"/api/oms/orders/{oid1}", params={"reason": "e2e撤单"})
b = r.json()
check("撤单返回 code=0", b.get("code") == 0, b)
r = call("GET", f"/api/oms/orders/{oid1}")
check("撤单后状态 CANCELLED", (r.json().get("data") or {}).get("status") == "CANCELLED", r.json())

# 废单
r = call("POST", f"/api/oms/orders/{oid2}/reject", json={"reason": "e2e废单"})
check("废单返回 code=0", r.json().get("code") == 0, r.json())
r = call("GET", f"/api/oms/orders/{oid2}")
b = r.json()
check("废单后状态 REJECTED", (b.get("data") or {}).get("status") == "REJECTED", b)

# 成交记录（空列表结构）
r = call("GET", "/api/oms/fills")
b = r.json()
check("成交记录端点 code=0", b.get("code") == 0 and isinstance(b.get("data"), (list, dict)), b)

# 非法撤单（对已撤单再撤）
r = call("DELETE", f"/api/oms/orders/{oid1}")
check("重复撤单被拒(非0)", r.json().get("code") != 0, r.json())

# ===========================================================================
# 2. Brinson — REQ-P1-03
# ===========================================================================
print("== REQ-P1-03 Brinson 绩效归因 ==")

brinson_body: Dict[str, Any] = {
    "groups": [
        {"name": "科技", "w_p": 0.5, "w_b": 0.4, "r_p": 0.10, "r_b": 0.08},
        {"name": "消费", "w_p": 0.3, "w_b": 0.4, "r_p": 0.05, "r_b": 0.06},
        {"name": "金融", "w_p": 0.2, "w_b": 0.2, "r_p": 0.02, "r_b": 0.01},
    ],
    "model": "bhb",
}
r = call("POST", "/api/attribution/brinson", json=brinson_body)
b = r.json()
check("Brinson 提交 code=0", b.get("code") == 0, b)
task_id = (b.get("data") or {}).get("task_id", "")
check("返回 task_id", bool(task_id), b)

r = call("GET", f"/api/attribution/brinson/{task_id}")
b = r.json()
check("取归因结果 code=0", b.get("code") == 0, b)
data = b.get("data") or {}
resid = abs(data.get("closure_residual", 1))
check(f"闭合残差 <1e-6（实测 {resid:.2e}）", resid < 1e-6, data)
check("含瀑布图数据", bool(data.get("waterfall")), data)
# 显式提供分组时 sector_source="provided" 为正确行为；未提供分组时核心类
# 走 mock_simplified 路径（已在核心层单独验证，含 P0-04 切换提示）。
check("分组来源标注 provided", data.get("sector_source") == "provided", data)

r = call("GET", "/api/attribution/brinson/nonexistent-task")
check("不存在 task 返回非0", r.json().get("code") != 0, r.json())

# ===========================================================================
# 3. 因子风险暴露 — REQ-P1-09
# ===========================================================================
print("== REQ-P1-09 因子风险暴露 ==")

holdings = [
    {"symbol": "AAA", "weight": 0.5},
    {"symbol": "BBB", "weight": 0.3},
    {"symbol": "CCC", "weight": 0.2},
]
# 构造 f1 必超限（组合暴露 = 0.5*2 + 0.3*1 + 0.2*0 = 1.3 > 1.0）
factor_scores = {
    "AAA": {"f1": 2.0, "f2": 0.2},
    "BBB": {"f1": 1.0, "f2": -0.2},
    "CCC": {"f1": 0.0, "f2": 0.1},
}
r = call("POST", "/api/risk/factor_exposure",
         json={"holdings": holdings, "factor_scores": factor_scores})
b = r.json()
check("暴露计算 code=0", b.get("code") == 0, b)
d = b.get("data") or {}
exposures = d.get("exposures", {})
check("f1 组合暴露≈1.3", abs(exposures.get("f1", 0) - 1.3) < 1e-6, d)
breaches = d.get("breaches", [])
check("f1 超限被识别", any(x.get("factor") == "f1" for x in breaches), d)

r = call("POST", "/api/risk/factor_exposure/neutralize",
         json={"holdings": holdings, "factor_scores": factor_scores})
b = r.json()
check("中性化建议 code=0", b.get("code") == 0, b)
sugs = b.get("data")
sugs = sugs.get("suggestions", sugs) if isinstance(sugs, dict) else sugs
check("返回调仓建议", isinstance(sugs, list) and len(sugs) > 0, b)

r = call("GET", "/api/risk/factor_exposure/history")
check("历史暴露端点 code=0", r.json().get("code") == 0, r.json())
r = call("GET", "/api/risk/factor_exposure/alerts")
b = r.json()
check("告警列表端点 code=0", b.get("code") == 0, b)

# 权重不闭合
r = call("POST", "/api/risk/factor_exposure",
         json={"holdings": [{"symbol": "AAA", "weight": 0.5}], "factor_scores": factor_scores})
check("权重不闭合被拒(非0)", r.json().get("code") != 0, r.json())

# ===========================================================================
# 4. 算法交易 — REQ-P1-10
# ===========================================================================
print("== REQ-P1-10 算法交易 TWAP/VWAP ==")

twap_body = {
    "symbol": "600519.SH", "side": "buy", "total_quantity": 100,
    "num_slices": 4, "duration_minutes": 20, "auto_start": False,
}
r = call("POST", "/api/algo/twap", json=twap_body)
b = r.json()
check("TWAP 创建 code=0", b.get("code") == 0, b)
algo_id = (b.get("data") or {}).get("algo_order_id", "")
check("返回 algo_order_id", bool(algo_id), b)

r = call("GET", f"/api/algo/orders/{algo_id}")
b = r.json()
check("执行状态查询 code=0", b.get("code") == 0, b)
st = (b.get("data") or {}).get("status")
check("初始状态 PENDING", st == "PENDING", b)

r = call("GET", f"/api/algo/orders/{algo_id}/report")
check("执行报告端点 code=0", r.json().get("code") == 0, r.json())

# stop（PENDING 直接终止应可接受）
r = call("POST", f"/api/algo/orders/{algo_id}/stop")
check("stop 返回 code=0", r.json().get("code") == 0, r.json())
r = call("GET", f"/api/algo/orders/{algo_id}")
check("stop 后状态 STOPPED", (r.json().get("data") or {}).get("status") == "STOPPED", r.json())

# VWAP 创建
vwap_body = {
    "symbol": "600519.SH", "side": "buy", "total_quantity": 1000,
    "volume_profile": [
        {"time": "09:30-10:00", "period": 30, "volume": 3000},
        {"time": "10:00-10:30", "period": 30, "volume": 1500},
        {"time": "10:30-11:00", "period": 30, "volume": 1000},
        {"time": "13:00-13:30", "period": 30, "volume": 1200},
        {"time": "14:30-15:00", "period": 30, "volume": 3300},
    ],
    "participation_rate": 1.0, "auto_start": False,
}
r = call("POST", "/api/algo/vwap", json=vwap_body)
b = r.json()
check("VWAP 创建 code=0", b.get("code") == 0, b)
vid = (b.get("data") or {}).get("algo_order_id", "")
r = call("GET", f"/api/algo/orders/{vid}")
b = r.json()
slices = (b.get("data") or {}).get("slices", [])
total_planned = sum(float(s.get("planned_quantity", 0)) for s in slices)
check(f"VWAP 拆单 Σ=1000（实测 {total_planned:.2f}）", abs(total_planned - 1000) < 1e-6, b)
check("VWAP U型：首片>午间片", len(slices) >= 3 and
      slices[0]["planned_quantity"] > slices[2]["planned_quantity"], b)
call("POST", f"/api/algo/orders/{vid}/stop")

# 不存在 id
r = call("GET", "/api/algo/orders/NOPE")
check("不存在 algo 单返回非0", r.json().get("code") != 0, r.json())

# ===========================================================================
# 汇总
# ===========================================================================
print(f"\n==== E2E 结果：{PASS} passed, {FAIL} failed ====")
sys.exit(1 if FAIL else 0)
