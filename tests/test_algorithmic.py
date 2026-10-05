"""算法交易（TWAP/VWAP）模块测试。

覆盖：
1. TWAP 拆单：等分正确、Σ=total；尾单吸收余数（100/3 → 33/33/34）
2. VWAP 成交量分布拆单：U 型 profile，各片量=total×占比，Σ=total；
   参与率截断生效、缺口记录
3. 执行进度：手动 execute_next_slice + oms.process_fill 后
   filled_quantity/avg_fill_price/progress 正确，全部完成 → COMPLETED
4. pause/resume/stop 状态流转；stop 后活动子单被撤
5. 非法操作（COMPLETED 后 pause）抛异常
6. 偏差分析：已知基准价/实际成交价手算 slippage/shortfall；未成交机会成本
7. 修改参与率后未执行切片重算
8. 全部 API 端点 TestClient 集成（创建/查询/控制/报告/404/400 分支）
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web-dashboard"))

from trading.oms import (  # noqa: E402
    STATUS_CANCELLED,
    STATUS_FILLED,
    OrderManagementSystem,
)
from trading.algorithmic import (  # noqa: E402
    ALGO_COMPLETED,
    ALGO_PAUSED,
    ALGO_RUNNING,
    ALGO_STOPPED,
    SLICE_CANCELLED,
    SLICE_FILLED,
    TWAPExecutor,
    VWAPExecutor,
    AlgoOrderError,
)
from _routes_algo import register_algo_routes  # noqa: E402


# ---------------------------------------------------------------------------
# 公共夹具 / 工具
# ---------------------------------------------------------------------------

#: U 型历史分时成交量（30 分钟 bar）：开盘/收盘高，午间低
U_SHAPED_PROFILE = [
    {"time": "09:30", "period": 30, "volume": 1000.0},
    {"time": "10:00", "period": 30, "volume": 300.0},
    {"time": "10:30", "period": 30, "volume": 100.0},
    {"time": "11:00", "period": 30, "volume": 200.0},
    {"time": "13:00", "period": 30, "volume": 150.0},
    {"time": "14:00", "period": 30, "volume": 400.0},
    {"time": "14:30", "period": 30, "volume": 800.0},
]
#: 总成交量 = 2950
TOTAL_VOL = sum(b["volume"] for b in U_SHAPED_PROFILE)


@pytest.fixture
def oms(tmp_path) -> OrderManagementSystem:
    """基于临时 SQLite 的 OMS。"""
    db = tmp_path / "algo_test.db"
    return OrderManagementSystem(db_path=str(db))


def make_twap(
    oms, total=90.0, n=3, interval=60.0, benchmark=None, side="buy"
) -> TWAPExecutor:
    """构造一个已 start 的 TWAP 执行器。"""
    exe = TWAPExecutor(
        oms=oms,
        symbol="AAPL",
        side=side,
        total_quantity=total,
        num_slices=n,
        interval_seconds=interval,
        benchmark_prices=benchmark or [100.0] * n,
    )
    exe.start()
    return exe


# ---------------------------------------------------------------------------
# 1. TWAP 拆单
# ---------------------------------------------------------------------------


class TestTwapSlicing:
    def test_equal_slices_sum_to_total(self, oms):
        exe = TWAPExecutor(
            oms=oms, symbol="AAPL", side="buy", total_quantity=90.0,
            num_slices=3, interval_seconds=60.0,
        )
        qtys = [s.planned_quantity for s in exe.order.slices]
        assert qtys == [30.0, 30.0, 30.0]
        assert sum(qtys) == pytest.approx(90.0)

    def test_tail_absorbs_remainder(self, oms):
        """total=100, n=3 → 33/33/34。"""
        exe = TWAPExecutor(
            oms=oms, symbol="AAPL", side="buy", total_quantity=100.0,
            num_slices=3, interval_seconds=60.0,
        )
        qtys = [s.planned_quantity for s in exe.order.slices]
        assert qtys == [33.0, 33.0, 34.0]
        assert sum(qtys) == pytest.approx(100.0)

    def test_offsets_evenly_spaced(self, oms):
        exe = TWAPExecutor(
            oms=oms, symbol="AAPL", side="buy", total_quantity=90.0,
            num_slices=3, interval_seconds=60.0,
        )
        offsets = [s.offset_seconds for s in exe.order.slices]
        assert offsets == [0.0, 60.0, 120.0]

    def test_duration_derives_interval(self, oms):
        """duration_minutes=10, n=3 → interval = 600/2 = 300s。"""
        exe = TWAPExecutor(
            oms=oms, symbol="AAPL", side="buy", total_quantity=90.0,
            num_slices=3, duration_minutes=10.0,
        )
        assert exe.interval_seconds == pytest.approx(300.0)
        offsets = [s.offset_seconds for s in exe.order.slices]
        assert offsets == [0.0, 300.0, 600.0]

    def test_duration_and_interval_inconsistent_raises(self, oms):
        with pytest.raises(AlgoOrderError):
            TWAPExecutor(
                oms=oms, symbol="AAPL", side="buy", total_quantity=90.0,
                num_slices=3, duration_minutes=10.0, interval_seconds=100.0,
            )

    def test_requires_duration_or_interval(self, oms):
        with pytest.raises(AlgoOrderError):
            TWAPExecutor(
                oms=oms, symbol="AAPL", side="buy", total_quantity=90.0,
                num_slices=3,
            )


# ---------------------------------------------------------------------------
# 2. VWAP 拆单
# ---------------------------------------------------------------------------


class TestVwapSlicing:
    def test_u_shaped_weights_sum_to_total(self, oms):
        """无参与率截断时，各片 = total×占比，Σ=total。"""
        exe = VWAPExecutor(
            oms=oms, symbol="AAPL", side="buy", total_quantity=1000.0,
            volume_profile=U_SHAPED_PROFILE, participation_rate=1.0,
        )
        qtys = [s.planned_quantity for s in exe.order.slices]
        # 占比最高的开盘 bar 量最大，午间 bar 最小（U 型）
        assert qtys[0] == pytest.approx(1000.0 * 1000.0 / TOTAL_VOL)
        assert qtys[2] == pytest.approx(1000.0 * 100.0 / TOTAL_VOL)
        assert qtys[6] == pytest.approx(1000.0 * 800.0 / TOTAL_VOL)
        assert sum(qtys) == pytest.approx(1000.0)
        assert exe.order.leftover_quantity == pytest.approx(0.0)

    def test_participation_cap_truncates(self, oms):
        """低参与率下各片截断到 vol×rate，缺口记 leftover。"""
        exe = VWAPExecutor(
            oms=oms, symbol="AAPL", side="buy", total_quantity=1000.0,
            volume_profile=U_SHAPED_PROFILE, participation_rate=0.05,
        )
        qtys = [s.planned_quantity for s in exe.order.slices]
        # 全部被 cap 截断：planned_i = vol_i × 0.05
        expected = [b["volume"] * 0.05 for b in U_SHAPED_PROFILE]
        assert qtys == pytest.approx(expected)
        allocated = sum(expected)
        assert exe.order.leftover_quantity == pytest.approx(1000.0 - allocated)
        assert exe.order.leftover_quantity > 0

    def test_empty_profile_raises(self, oms):
        with pytest.raises(AlgoOrderError):
            VWAPExecutor(
                oms=oms, symbol="AAPL", side="buy", total_quantity=100.0,
                volume_profile=[],
            )


# ---------------------------------------------------------------------------
# 3. 执行进度聚合
# ---------------------------------------------------------------------------


class TestExecutionProgress:
    def test_filled_aggregation_and_completion(self, oms):
        exe = make_twap(oms, total=90.0, n=3, benchmark=[100.0, 101.0, 102.0])
        prices = [100.0, 101.0, 102.0]
        for i in range(3):
            child = exe.execute_next_slice()
            assert child is not None
            oms.process_fill(child.order_id, 30.0, prices[i])

        assert exe.order.filled_quantity == pytest.approx(90.0)
        # 加权均价 = (30*100+30*101+30*102)/90
        expected_avg = (30 * 100.0 + 30 * 101.0 + 30 * 102.0) / 90.0
        assert exe.order.avg_fill_price == pytest.approx(expected_avg)
        assert exe.order.progress_percent == pytest.approx(100.0)
        assert exe.order.remaining_quantity == pytest.approx(0.0)
        assert exe.order.status == ALGO_COMPLETED

    def test_partial_progress(self, oms):
        exe = make_twap(oms, total=90.0, n=3)
        c1 = exe.execute_next_slice()
        oms.process_fill(c1.order_id, 30.0, 100.0)
        assert exe.order.filled_quantity == pytest.approx(30.0)
        assert exe.order.progress_percent == pytest.approx(30.0 / 90.0 * 100)
        assert exe.order.remaining_quantity == pytest.approx(60.0)
        assert exe.order.status == ALGO_RUNNING  # 未全部完成

    def test_execute_next_slice_when_not_running_raises(self, oms):
        exe = make_twap(oms, total=90.0, n=3)
        exe.pause()
        with pytest.raises(AlgoOrderError):
            exe.execute_next_slice()


# ---------------------------------------------------------------------------
# 4. pause / resume / stop
# ---------------------------------------------------------------------------


class TestStateTransitions:
    def test_pause_resume(self, oms):
        exe = make_twap(oms, total=100.0, n=4)
        assert exe.order.status == ALGO_RUNNING
        exe.pause()
        assert exe.order.status == ALGO_PAUSED
        with pytest.raises(AlgoOrderError):
            exe.execute_next_slice()
        exe.resume()
        assert exe.order.status == ALGO_RUNNING
        assert exe.execute_next_slice() is not None

    def test_stop_cancels_active_child_and_voids_pending(self, oms):
        exe = make_twap(oms, total=100.0, n=4)  # 25/25/25/25
        s0 = exe.execute_next_slice()
        s1 = exe.execute_next_slice()
        s2 = exe.execute_next_slice()
        # s2 尚未成交，处于活动 SUBMITTED
        exe.stop()
        assert exe.order.status == ALGO_STOPPED
        # 已提交的活动子单被撤
        assert oms.get_order(s2.order_id).status == STATUS_CANCELLED
        assert oms.get_order(s0.order_id).status == STATUS_CANCELLED
        # 未提交的 S3 切片标记作废
        s3 = exe.order.slices[3]
        assert s3.status == SLICE_CANCELLED
        assert s3.child_order_id is None

    def test_stop_twice_raises(self, oms):
        exe = make_twap(oms, total=100.0, n=2)
        exe.stop()
        with pytest.raises(AlgoOrderError):
            exe.stop()


# ---------------------------------------------------------------------------
# 5. 非法操作
# ---------------------------------------------------------------------------


class TestIllegalOps:
    def test_pause_completed_raises(self, oms):
        exe = make_twap(oms, total=60.0, n=2)
        for _ in range(2):
            c = exe.execute_next_slice()
            oms.process_fill(c.order_id, 30.0, 100.0)
        assert exe.order.status == ALGO_COMPLETED
        with pytest.raises(AlgoOrderError):
            exe.pause()
        with pytest.raises(AlgoOrderError):
            exe.resume()
        with pytest.raises(AlgoOrderError):
            exe.stop()

    def test_resume_non_paused_raises(self, oms):
        exe = make_twap(oms, total=60.0, n=2)
        with pytest.raises(AlgoOrderError):
            exe.resume()


# ---------------------------------------------------------------------------
# 6. 执行偏差分析
# ---------------------------------------------------------------------------


class TestExecutionReport:
    def test_known_slippage_shortfall(self, oms):
        """benchmark=100，实际成交 101/103，手算短差。"""
        exe = make_twap(
            oms, total=100.0, n=2, benchmark=[100.0, 100.0]
        )
        c0 = exe.execute_next_slice()
        oms.process_fill(c0.order_id, 50.0, 101.0)
        c1 = exe.execute_next_slice()
        oms.process_fill(c1.order_id, 50.0, 103.0)

        report = exe.execution_quality_report()
        assert report["filled_qty"] == pytest.approx(100.0)
        assert report["avg_price"] == pytest.approx(102.0)
        assert report["benchmark_price"] == pytest.approx(100.0)
        # 每份额短差 = 102 - 100 = 2
        assert report["slippage_per_share"] == pytest.approx(2.0)
        # 实现短差 = (101-100)*50 + (103-100)*50 = 50 + 150 = 200
        assert report["realized_shortfall"] == pytest.approx(200.0)
        assert report["opportunity_cost"] == pytest.approx(0.0)
        assert report["completed"] is True
        assert report["stages"] == "final"

    def test_opportunity_cost_on_unfilled(self, oms):
        """部分成交后停止，未成交部分按基准价相对到达价的不利变动计机会成本。"""
        exe = make_twap(
            oms, total=100.0, n=2, benchmark=[100.0, 110.0]
        )
        c0 = exe.execute_next_slice()
        oms.process_fill(c0.order_id, 50.0, 100.0)   # bench 100
        c1 = exe.execute_next_slice()
        oms.process_fill(c1.order_id, 25.0, 110.0)   # bench 110，部分成交
        exe.stop()

        report = exe.execution_quality_report()
        # filled = 75, avg = (50*100+25*110)/75 = 103.333
        assert report["filled_qty"] == pytest.approx(75.0)
        assert report["avg_price"] == pytest.approx(103.3333, rel=1e-3)
        # benchmark_price 按成交量加权 = (50*100+25*110)/75 = 103.333
        assert report["benchmark_price"] == pytest.approx(103.3333, rel=1e-3)
        # realized shortfall = (100-100)*50 + (110-110)*25 = 0
        assert report["realized_shortfall"] == pytest.approx(0.0)
        # arrival=100, bench=103.333, remaining=25 → opp = 25*3.333 = 83.33
        assert report["opportunity_cost"] == pytest.approx(83.3333, rel=1e-3)
        assert report["remaining_qty"] == pytest.approx(25.0)
        assert report["stages"] == "interim"

    def test_sell_side_sign(self, oms):
        """sell 方向：实际卖价低于基准为跑输（短差为负）。"""
        exe = TWAPExecutor(
            oms=oms, symbol="AAPL", side="sell", total_quantity=100.0,
            num_slices=2, interval_seconds=60.0,
            benchmark_prices=[100.0, 100.0],
        )
        exe.start()
        c0 = exe.execute_next_slice()
        oms.process_fill(c0.order_id, 50.0, 99.0)   # 卖低了 1 元
        c1 = exe.execute_next_slice()
        oms.process_fill(c1.order_id, 50.0, 99.0)
        report = exe.execution_quality_report()
        # sell 跑输 = (99-100)*100 = -100
        assert report["realized_shortfall"] == pytest.approx(-100.0)


# ---------------------------------------------------------------------------
# 7. 运行中修改参与率
# ---------------------------------------------------------------------------


class TestRuntimeAdjust:
    def test_set_participation_rate_recalculates_pending(self, oms):
        exe = VWAPExecutor(
            oms=oms, symbol="AAPL", side="buy", total_quantity=1000.0,
            volume_profile=U_SHAPED_PROFILE, participation_rate=1.0,
        )
        exe.start()
        # 提交 S0（已提交，受保护）
        s0 = exe.order.slices[0]
        exe.execute_next_slice()
        s0_planned_before = s0.planned_quantity
        s1_before = exe.order.slices[1].planned_quantity

        # 调低参与率 → 未提交切片重算
        exe.set_participation_rate(0.05)
        # S0 已提交，不变
        assert exe.order.slices[0].planned_quantity == pytest.approx(s0_planned_before)
        # S1 未提交，被截断到 vol_1 × 0.05 = 300 × 0.05 = 15
        assert exe.order.slices[1].planned_quantity == pytest.approx(15.0)
        assert exe.order.slices[1].planned_quantity != pytest.approx(s1_before)
        # leftover 重算
        assert exe.order.leftover_quantity > 0

    def test_set_interval_recalculates_pending_offsets(self, oms):
        exe = make_twap(oms, total=90.0, n=3, interval=60.0)
        exe.execute_next_slice()  # S0 已提交
        exe.set_interval(120.0)
        # S1/S2 未提交，offset 重算
        assert exe.order.slices[1].offset_seconds == pytest.approx(120.0)
        assert exe.order.slices[2].offset_seconds == pytest.approx(240.0)


# ---------------------------------------------------------------------------
# 8. API 集成测试
# ---------------------------------------------------------------------------


def _ok(data=None, message="success"):
    """与 server.py 一致的成功封装。"""
    return {"code": 0, "message": message, "data": data}


def _err(code, message, http_status=400):
    """与 server.py 一致的错误封装。"""
    return JSONResponse(
        status_code=http_status,
        content={"code": code, "message": message, "data": None},
    )


@pytest.fixture
def api_env(tmp_path):
    """构造注入临时库 + 独立注册表的 TestClient。"""
    db = tmp_path / "api_algo.db"
    oms = OrderManagementSystem(db_path=str(db))
    registry: dict = {}
    app = FastAPI()
    register_algo_routes(app, _ok, _err, oms=oms, registry=registry)
    return TestClient(app), oms, registry


class TestApi:
    def test_create_twap(self, api_env):
        c, _, registry = api_env
        r = c.post("/api/algo/twap", json={
            "symbol": "AAPL", "side": "buy", "total_quantity": 100.0,
            "num_slices": 3, "interval_seconds": 60.0,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        data = body["data"]
        assert data["strategy"] == "TWAP"
        assert data["status"] == ALGO_RUNNING
        assert len(data["slices"]) == 3
        assert [s["planned_quantity"] for s in data["slices"]] == [33.0, 33.0, 34.0]
        assert data["algo_order_id"] in registry

    def test_create_vwap(self, api_env):
        c, _, _ = api_env
        r = c.post("/api/algo/vwap", json={
            "symbol": "AAPL", "side": "buy", "total_quantity": 1000.0,
            "volume_profile": U_SHAPED_PROFILE, "participation_rate": 1.0,
        })
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["strategy"] == "VWAP"
        assert len(data["slices"]) == 7
        assert sum(s["planned_quantity"] for s in data["slices"]) == pytest.approx(1000.0)

    def test_get_status(self, api_env):
        c, oms, registry = api_env
        r = c.post("/api/algo/twap", json={
            "symbol": "AAPL", "side": "buy", "total_quantity": 90.0,
            "num_slices": 3, "interval_seconds": 60.0,
        })
        algo_id = r.json()["data"]["algo_order_id"]
        exe = registry[algo_id]
        child = exe.execute_next_slice()
        oms.process_fill(child.order_id, 30.0, 100.0)

        r2 = c.get(f"/api/algo/orders/{algo_id}")
        assert r2.status_code == 200
        d = r2.json()["data"]
        assert d["filled_quantity"] == pytest.approx(30.0)
        assert d["progress_percent"] == pytest.approx(30.0 / 90.0 * 100)
        assert d["slices"][0]["status"] == SLICE_FILLED

    def test_get_not_found_404(self, api_env):
        c, _, _ = api_env
        r = c.get("/api/algo/orders/nope")
        assert r.status_code == 404

    def test_pause_resume_stop(self, api_env):
        c, _, _ = api_env
        r = c.post("/api/algo/twap", json={
            "symbol": "AAPL", "side": "buy", "total_quantity": 90.0,
            "num_slices": 3, "interval_seconds": 60.0,
        })
        algo_id = r.json()["data"]["algo_order_id"]

        rp = c.post(f"/api/algo/orders/{algo_id}/pause")
        assert rp.status_code == 200
        assert rp.json()["data"]["status"] == ALGO_PAUSED

        rr = c.post(f"/api/algo/orders/{algo_id}/resume")
        assert rr.status_code == 200
        assert rr.json()["data"]["status"] == ALGO_RUNNING

        rs = c.post(f"/api/algo/orders/{algo_id}/stop")
        assert rs.status_code == 200
        assert rs.json()["data"]["status"] == ALGO_STOPPED

    def test_pause_not_running_400(self, api_env):
        """auto_start=False → PENDING，pause 应返回 400。"""
        c, _, _ = api_env
        r = c.post("/api/algo/twap", json={
            "symbol": "AAPL", "side": "buy", "total_quantity": 90.0,
            "num_slices": 3, "interval_seconds": 60.0, "auto_start": False,
        })
        algo_id = r.json()["data"]["algo_order_id"]
        rp = c.post(f"/api/algo/orders/{algo_id}/pause")
        assert rp.status_code == 400
        assert rp.json()["code"] == 400

    def test_control_not_found_404(self, api_env):
        c, _, _ = api_env
        assert c.post("/api/algo/orders/nope/pause").status_code == 404
        assert c.post("/api/algo/orders/nope/stop").status_code == 404

    def test_report_interim(self, api_env):
        c, oms, registry = api_env
        r = c.post("/api/algo/twap", json={
            "symbol": "AAPL", "side": "buy", "total_quantity": 90.0,
            "num_slices": 3, "interval_seconds": 60.0,
        })
        algo_id = r.json()["data"]["algo_order_id"]
        exe = registry[algo_id]
        child = exe.execute_next_slice()
        oms.process_fill(child.order_id, 30.0, 100.0)

        rr = c.get(f"/api/algo/orders/{algo_id}/report")
        assert rr.status_code == 200
        rep = rr.json()["data"]
        assert rep["filled_qty"] == pytest.approx(30.0)
        assert rep["stages"] == "interim"
        assert rep["completed"] is False
        assert len(rep["slices"]) == 3

    def test_report_not_found_404(self, api_env):
        c, _, _ = api_env
        assert c.get("/api/algo/orders/nope/report").status_code == 404

    def test_invalid_twap_params_400(self, api_env):
        c, _, _ = api_env
        # duration 与 interval 不一致
        r = c.post("/api/algo/twap", json={
            "symbol": "AAPL", "side": "buy", "total_quantity": 90.0,
            "num_slices": 3, "duration_minutes": 10.0, "interval_seconds": 100.0,
        })
        assert r.status_code == 400
