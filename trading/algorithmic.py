"""算法交易（TWAP / VWAP）执行模块。

将一笔大额母单（AlgorithmOrder）按时间（TWAP）或成交量分布（VWAP）拆成若干
子单（SlicePlan），通过已开发完成的 :class:`~trading.oms.OrderManagementSystem`
(OMS) 逐片下发，并聚合成交、计算执行偏差（implementation shortfall）。

核心组件：

- :class:`SlicePlan`：单片拆单计划（数量 / 计划时间 / 子单 ID / 状态）。
- :class:`AlgorithmOrder`：母单数据结构（状态机 + 进度聚合）。
- :class:`BaseAlgoExecutor`：TWAP/VWAP 共用执行驱动（与 OMS 集成、状态流转、
  成交聚合、后台线程、执行质量报告）。
- :class:`TWAPExecutor`：按时间等分拆单（尾单吸收余数）。
- :class:`VWAPExecutor`：按历史分时成交量占比拆单，受参与率（participation rate）
  上限截断，缺口记为 leftover。

可测性约定：拆单计划 ``build_schedule()`` 为纯计算、不 sleep；执行推进由
``execute_next_slice()`` / ``tick(now=None)`` 手动驱动，后台 daemon 线程仅在
显式 ``start(background=True)`` 时启用，且可被 ``stop()`` 用 ``threading.Event``
干净终止。基准价格序列作为构造输入，不依赖外部行情。

典型用法::

    oms = OrderManagementSystem(db_path=":memory:")
    exe = TWAPExecutor(
        oms=oms, symbol="AAPL", side="buy", total_quantity=100.0,
        duration_minutes=10.0, num_slices=3,
        benchmark_prices=[150.0, 150.1, 150.2],
    )
    exe.start()
    while exe.execute_next_slice():
        pass  # 每片到点后由外部/行情侧 process_fill
    report = exe.execution_quality_report()
"""
from __future__ import annotations

import logging
import math
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from trading.oms import (
    STATUS_CANCELLED,
    STATUS_FILLED,
    STATUS_PARTIAL_FILLED,
    STATUS_SUBMITTED,
    Fill,
    InvalidOrderStateError,
    Order,
    OrderManagementSystem,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 母单状态机
# ---------------------------------------------------------------------------

#: 母单待启动
ALGO_PENDING = "PENDING"
#: 母单运行中
ALGO_RUNNING = "RUNNING"
#: 母单暂停
ALGO_PAUSED = "PAUSED"
#: 母单全部子单成交完成
ALGO_COMPLETED = "COMPLETED"
#: 母单被手动停止（未成交部分作废 / 撤单）
ALGO_STOPPED = "STOPPED"

#: 母单合法状态集合
ALGO_STATES = frozenset(
    {ALGO_PENDING, ALGO_RUNNING, ALGO_PAUSED, ALGO_COMPLETED, ALGO_STOPPED}
)

#: 子单（切片）状态
SLICE_PENDING = "PENDING"
SLICE_SUBMITTED = "SUBMITTED"
SLICE_PARTIAL = "PARTIAL_FILLED"
SLICE_FILLED = "FILLED"
SLICE_CANCELLED = "CANCELLED"

#: 数值比较容差
_EPS = 1e-6


class AlgoOrderError(Exception):
    """算法母单非法状态操作或参数错误时抛出。"""


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class SlicePlan:
    """单片子单执行计划。

    Attributes:
        slice_index: 切片序号（从 0 开始）。
        offset_seconds: 相对母单启动时刻的偏移秒数。
        planned_quantity: 本片计划下单数量。
        filled_quantity: 本片已成交数量（从 OMS 同步）。
        child_order_id: 对应 OMS 子单 ID，未提交为 None。
        status: 本片状态（PENDING/SUBMITTED/PARTIAL_FILLED/FILLED/CANCELLED）。
        scheduled_time: 计划下单绝对时间（启动后填充）。
        market_volume: 本片对应时段的市场成交量（VWAP 用，TWAP 为 0）。
        raw_quantity: 未参与率截断前的理想量（VWAP 用，用于 leftover 对账）。
    """

    slice_index: int
    offset_seconds: float
    planned_quantity: float
    filled_quantity: float = 0.0
    child_order_id: Optional[str] = None
    status: str = SLICE_PENDING
    scheduled_time: Optional[datetime] = None
    market_volume: float = 0.0
    raw_quantity: float = 0.0


@dataclass
class AlgorithmOrder:
    """算法母单（大额拆分执行的入口对象）。

    Attributes:
        algo_order_id: 母单唯一标识。
        symbol: 标的代码。
        side: 买卖方向（buy/sell）。
        strategy: 策略名（"TWAP"/"VWAP"）。
        total_quantity: 母单总数量。
        filled_quantity: 已成交数量（所有子单加总）。
        avg_fill_price: 已成交部分成交量加权平均价。
        status: 母单状态（PENDING/RUNNING/PAUSED/COMPLETED/STOPPED）。
        slices: 拆单计划列表。
        created_at: 创建时间。
        started_at: 启动时间。
        completed_at: 完成 / 停止时间。
        params: 执行参数（duration/interval/participation_rate 等）。
        leftover_quantity: 因参与率截断等原因未能排入计划的缺口量。
    """

    algo_order_id: str
    symbol: str
    side: str
    strategy: str
    total_quantity: float
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    status: str = ALGO_PENDING
    slices: List[SlicePlan] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    params: Dict[str, Any] = field(default_factory=dict)
    leftover_quantity: float = 0.0

    @property
    def progress_percent(self) -> float:
        """执行进度（已成交 / 总量 × 100）。"""
        if self.total_quantity <= 0:
            return 0.0
        return self.filled_quantity / self.total_quantity * 100.0

    @property
    def remaining_quantity(self) -> float:
        """剩余未成交数量。"""
        return max(0.0, self.total_quantity - self.filled_quantity)


# ---------------------------------------------------------------------------
# 执行器基类
# ---------------------------------------------------------------------------


class BaseAlgoExecutor(ABC):
    """TWAP / VWAP 共用的执行驱动基类。

    职责：与 OMS 集成下发子单、聚合成交、维护母单状态机、提供手动/自动两种
    推进方式、生成执行质量报告。

    Args:
        oms: 必传的订单管理系统实例。
        symbol: 标的代码。
        side: buy/sell。
        total_quantity: 母单总数量。
        strategy: "TWAP"/"VWAP"。
        algo_order_id: 母单 ID，缺省自动生成。
        order_type: 子单类型（market/limit），默认 market。
        limit_price: 限价单价格，market 单为 None。
        benchmark_prices: 每个切片对应的基准价（到达价 / 切片时刻市场价），
            用于偏差分析；长度应与切片数一致。缺省则报告中相关字段为 0。
        params: 透传到 :attr:`AlgorithmOrder.params` 的执行参数。

    Raises:
        AlgoOrderError: oms 为空、side 非法或数量非法。
    """

    def __init__(
        self,
        oms: OrderManagementSystem,
        symbol: str,
        side: str,
        total_quantity: float,
        strategy: str,
        algo_order_id: Optional[str] = None,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        benchmark_prices: Optional[List[float]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """初始化执行器并构建拆单计划。"""
        if oms is None:
            raise AlgoOrderError("oms 为必传参数")
        if side not in ("buy", "sell"):
            raise AlgoOrderError(f"非法方向: {side}（仅 buy/sell）")
        if total_quantity <= 0:
            raise AlgoOrderError(f"总数量必须为正: {total_quantity}")

        self.oms = oms
        self.order = AlgorithmOrder(
            algo_order_id=algo_order_id
            or f"ALGO{datetime.now().strftime('%Y%m%d%H%M%S')}{uuid.uuid4().hex[:6]}",
            symbol=symbol,
            side=side,
            strategy=strategy,
            total_quantity=float(total_quantity),
            params=dict(params or {}),
        )
        self.order.params["order_type"] = order_type
        self.order_type = order_type
        self.limit_price = limit_price
        self.benchmark_prices: List[float] = list(benchmark_prices or [])

        # 后台线程控制
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # 构建拆单计划（纯计算，不 sleep）
        self.build_schedule()

        # 注册成交回报回调，OMS 每笔成交后自动聚合
        self.oms.register_fill_callback(self._on_oms_fill)

    # ------------------------------------------------------------------
    # 拆单计划（子类实现）
    # ------------------------------------------------------------------

    @abstractmethod
    def build_schedule(self) -> List[SlicePlan]:
        """构建拆单计划，填充 :attr:`order.slices`。

        Returns:
            切片计划列表。
        """

    # ------------------------------------------------------------------
    # 状态流转
    # ------------------------------------------------------------------

    def start(self, background: bool = False, now: Optional[datetime] = None) -> None:
        """启动母单：PENDING → RUNNING。

        Args:
            background: 是否启用后台 daemon 线程按 interval 自动 ``tick``。
                测试环境默认 False，由 ``execute_next_slice()`` 手动推进。
            now: 注入启动时间（测试用）。

        Raises:
            AlgoOrderError: 当前状态不允许启动。
        """
        if self.order.status not in (ALGO_PENDING,):
            raise AlgoOrderError(
                f"仅 PENDING 母单可启动，当前={self.order.status}"
            )
        self.order.status = ALGO_RUNNING
        self.order.started_at = now or datetime.now()
        for s in self.order.slices:
            if s.scheduled_time is None and self.order.started_at is not None:
                s.scheduled_time = self.order.started_at + _seconds(s.offset_seconds)
        if background:
            self._start_background_loop()

    def pause(self) -> None:
        """暂停：RUNNING → PAUSED。

        Raises:
            AlgoOrderError: 非 RUNNING 状态不可暂停。
        """
        if self.order.status != ALGO_RUNNING:
            raise AlgoOrderError(
                f"仅 RUNNING 母单可暂停，当前={self.order.status}"
            )
        self.order.status = ALGO_PAUSED

    def resume(self) -> None:
        """恢复：PAUSED → RUNNING。

        Raises:
            AlgoOrderError: 非 PAUSED 状态不可恢复。
        """
        if self.order.status != ALGO_PAUSED:
            raise AlgoOrderError(
                f"仅 PAUSED 母单可恢复，当前={self.order.status}"
            )
        self.order.status = ALGO_RUNNING

    def stop(self, reason: str = "manual stop") -> None:
        """停止母单：→ STOPPED；未提交切片作废，活动子单调 OMS 撤单。

        Args:
            reason: 撤单原因。

        Raises:
            AlgoOrderError: 对 COMPLETED/STOPPED 母单再次 stop。
        """
        if self.order.status in (ALGO_COMPLETED, ALGO_STOPPED):
            raise AlgoOrderError(
                f"母单已终结（{self.order.status}），无法 stop"
            )
        self.order.status = ALGO_STOPPED
        self.order.completed_at = datetime.now()
        self._stop_event.set()

        for s in self.order.slices:
            if s.child_order_id is None:
                # 未提交切片直接作废
                s.status = SLICE_CANCELLED
                continue
            child = self.oms.get_order(s.child_order_id)
            if child is not None and child.status in (
                STATUS_SUBMITTED,
                STATUS_PARTIAL_FILLED,
            ):
                try:
                    self.oms.cancel_order(child.order_id, reason=reason)
                except InvalidOrderStateError:
                    logger.warning("撤单时子单状态已变化: %s", child.order_id)
        self._sync_fills()

    # ------------------------------------------------------------------
    # 子单下发与成交聚合
    # ------------------------------------------------------------------

    def _submit_slice(self, sl: SlicePlan) -> Order:
        """向 OMS 提交一个切片对应的子单。

        Args:
            sl: 待提交切片。

        Returns:
            OMS 中的子单对象。
        """
        child = Order(
            order_id=f"{self.order.algo_order_id}-S{sl.slice_index}",
            symbol=self.order.symbol,
            side=self.order.side,
            order_type=self.order_type,
            quantity=sl.planned_quantity,
            limit_price=self.limit_price,
            strategy_name=self.order.strategy,
        )
        self.oms.submit_order(child)
        sl.child_order_id = child.order_id
        sl.status = SLICE_SUBMITTED
        return child

    def execute_next_slice(self) -> Optional[Order]:
        """手动提交下一个尚未提交的切片。

        Returns:
            提交的子单；所有切片均已提交返回 None。

        Raises:
            AlgoOrderError: 母单非 RUNNING 状态。
        """
        if self.order.status != ALGO_RUNNING:
            raise AlgoOrderError(
                f"仅 RUNNING 母单可下发子单，当前={self.order.status}"
            )
        for sl in self.order.slices:
            if sl.status == SLICE_PENDING:
                child = self._submit_slice(sl)
                self._sync_fills()
                return child
        return None

    def tick(self, now: Optional[datetime] = None) -> List[Order]:
        """按时间推进：提交所有 offset 已到点但尚未提交的切片。

        Args:
            now: 注入当前时间（测试用），缺省取系统时间。

        Returns:
            本次 tick 新提交的子单列表。
        """
        if self.order.status != ALGO_RUNNING or self.order.started_at is None:
            return []
        now = now or datetime.now()
        elapsed = (now - self.order.started_at).total_seconds()
        submitted: List[Order] = []
        for sl in self.order.slices:
            if sl.status == SLICE_PENDING and sl.offset_seconds <= elapsed + _EPS:
                submitted.append(self._submit_slice(sl))
        if submitted:
            self._sync_fills()
        return submitted

    def update_from_oms(self) -> AlgorithmOrder:
        """主动从 OMS 轮询子单状态并聚合母单成交。

        Returns:
            更新后的 :attr:`order`。
        """
        self._sync_fills()
        return self.order

    def _on_oms_fill(self, fill: Fill) -> None:
        """OMS 成交回调：按子单 ID 反查切片并聚合。"""
        if not fill.order_id.startswith(f"{self.order.algo_order_id}-S"):
            return
        self._sync_fills()

    @staticmethod
    def _map_child_status(child_status: str) -> str:
        """将 OMS 订单状态映射为切片状态。"""
        return {
            STATUS_SUBMITTED: SLICE_SUBMITTED,
            STATUS_PARTIAL_FILLED: SLICE_PARTIAL,
            STATUS_FILLED: SLICE_FILLED,
            STATUS_CANCELLED: SLICE_CANCELLED,
        }.get(child_status, child_status)

    def _sync_fills(self) -> None:
        """从 OMS 拉取每个已提交子单的成交，聚合到切片与母单。"""
        filled_qty = 0.0
        cost = 0.0
        for sl in self.order.slices:
            if sl.child_order_id is None:
                continue
            child = self.oms.get_order(sl.child_order_id)
            if child is None:
                continue
            sl.filled_quantity = child.filled_quantity
            sl.status = self._map_child_status(child.status)
            filled_qty += child.filled_quantity
            cost += child.avg_fill_price * child.filled_quantity

        self.order.filled_quantity = filled_qty
        self.order.avg_fill_price = cost / filled_qty if filled_qty > _EPS else 0.0
        self._maybe_complete()

    def _maybe_complete(self) -> None:
        """全部子单已提交且全部成交 → COMPLETED。"""
        if self.order.status != ALGO_RUNNING:
            return
        if not self.order.slices:
            return
        all_submitted = all(s.child_order_id is not None for s in self.order.slices)
        all_filled = True
        for s in self.order.slices:
            child = self.oms.get_order(s.child_order_id) if s.child_order_id else None
            if child is None or child.status != STATUS_FILLED:
                all_filled = False
                break
        if all_submitted and all_filled:
            self.order.status = ALGO_COMPLETED
            self.order.completed_at = datetime.now()

    # ------------------------------------------------------------------
    # 后台线程
    # ------------------------------------------------------------------

    def _start_background_loop(self) -> None:
        """启动 daemon 线程按 interval 自动 tick（测试不依赖此路径）。"""
        interval = float(self.order.params.get("tick_interval_seconds", 1.0))

        def _loop() -> None:
            while not self._stop_event.wait(interval):
                try:
                    self.tick()
                except Exception:  # pragma: no cover
                    logger.exception("algo tick 失败")

        self._thread = threading.Thread(target=_loop, daemon=True, name=f"algo-{self.order.algo_order_id}")
        self._thread.start()

    # ------------------------------------------------------------------
    # 运行中调参
    # ------------------------------------------------------------------

    def set_interval(self, seconds: float) -> None:
        """修改切片间隔（仅重算未提交切片的 offset，TWAP 用）。

        Args:
            seconds: 新的间隔秒数。
        """
        self.order.params["interval_seconds"] = seconds
        for sl in self.order.slices:
            if sl.status == SLICE_PENDING:
                sl.offset_seconds = round(sl.slice_index * seconds, 6)

    def set_participation_rate(self, rate: float) -> None:
        """修改参与率（仅重算未提交切片的计划量，VWAP 用）。

        Args:
            rate: 新的参与率 (0, 1]。
        """
        raise NotImplementedError("请在 VWAPExecutor 中实现")

    # ------------------------------------------------------------------
    # 执行质量报告
    # ------------------------------------------------------------------

    def execution_quality_report(self) -> Dict[str, Any]:
        """生成执行偏差分析报告。

        基准价序列在构造时注入（arrival price / 各切片时刻市场价），
        不依赖外部行情。实现短差（implementation shortfall）=
        Σ(实际成交价 − 该切片基准价)×成交量（buy 为正表示跑输基准）。

        Returns:
            报告字典，含：
            - algo_order_id / symbol / side / strategy / status
            - filled_qty / avg_price / remaining_qty / progress_percent
            - benchmark_price（按成交量加权的基准均价）/ arrival_price
            - realized_shortfall（已成交部分短差，货币单位）
            - slippage_per_share（每份额短差）
            - opportunity_cost（未成交部分机会成本）
            - total_shortfall（realized + opportunity）
            - participation_realized（实际参与率）
            - duration_seconds / leftover_qty / completed
            - slices：逐片明细
        """
        self._sync_fills()
        o = self.order
        filled = o.filled_quantity
        total = o.total_quantity
        remaining = max(0.0, total - filled)

        arrival_price = self.benchmark_prices[0] if self.benchmark_prices else 0.0

        bench_cost = 0.0
        realized = 0.0
        slice_details: List[Dict[str, Any]] = []
        for i, sl in enumerate(o.slices):
            bench_i = (
                self.benchmark_prices[i] if i < len(self.benchmark_prices) else None
            )
            child = (
                self.oms.get_order(sl.child_order_id) if sl.child_order_id else None
            )
            s_avg = child.avg_fill_price if child else 0.0
            s_fill = sl.filled_quantity
            if bench_i is not None and s_fill > _EPS:
                bench_cost += bench_i * s_fill
                realized += (s_avg - bench_i) * s_fill
            slice_details.append(
                {
                    "slice_index": sl.slice_index,
                    "offset_seconds": sl.offset_seconds,
                    "planned_quantity": sl.planned_quantity,
                    "filled_quantity": s_fill,
                    "avg_price": s_avg,
                    "benchmark_price": bench_i,
                    "child_order_id": sl.child_order_id,
                    "status": sl.status,
                }
            )

        benchmark_price = bench_cost / filled if filled > _EPS else arrival_price

        # 严格按需求定义：shortfall = Σ(实际价 − 基准价)×成交量。
        # buy 为正表示跑输基准；sell 方向符号自然相反（卖低了为负）。
        realized_shortfall = realized
        slippage_per_share = (
            (o.avg_fill_price - benchmark_price) if filled > _EPS else 0.0
        )

        # 机会成本：未成交部分按基准价相对到达价的变动估算
        # （buy：价格上行错过买入；sell：价格下行错过卖出，符号随公式自然体现）
        if arrival_price and benchmark_price:
            opportunity_cost = remaining * (benchmark_price - arrival_price)
        else:
            opportunity_cost = 0.0

        # 实际参与率：优先用 VWAP 历史市场成交量做分母
        market_volume = sum(s.market_volume for s in o.slices)
        if market_volume > _EPS:
            participation_realized = filled / market_volume
        else:
            participation_realized = filled / total if total > _EPS else 0.0

        duration = None
        if o.started_at is not None:
            end = o.completed_at or datetime.now()
            duration = round((end - o.started_at).total_seconds(), 3)

        return {
            "algo_order_id": o.algo_order_id,
            "symbol": o.symbol,
            "side": o.side,
            "strategy": o.strategy,
            "status": o.status,
            "filled_qty": filled,
            "avg_price": o.avg_fill_price,
            "remaining_qty": remaining,
            "progress_percent": o.progress_percent,
            "arrival_price": arrival_price,
            "benchmark_price": benchmark_price,
            "slippage_per_share": slippage_per_share,
            "realized_shortfall": realized_shortfall,
            "opportunity_cost": opportunity_cost,
            "total_shortfall": realized_shortfall + opportunity_cost,
            "participation_realized": participation_realized,
            "duration_seconds": duration,
            "leftover_qty": o.leftover_quantity,
            "completed": o.status == ALGO_COMPLETED,
            "stages": "final" if o.status == ALGO_COMPLETED else "interim",
            "slices": slice_details,
        }

    def to_dict(self) -> Dict[str, Any]:
        """将母单序列化为可 JSON 化字典（路由层状态查询用）。"""
        data = asdict(self.order)
        for key in ("created_at", "started_at", "completed_at"):
            if isinstance(data.get(key), datetime):
                data[key] = data[key].isoformat()
        return data


def _seconds(value: float):
    """秒数转 timedelta 的小工具（延迟导入避免顶部循环依赖）。"""
    from datetime import timedelta

    return timedelta(seconds=value)


# ---------------------------------------------------------------------------
# TWAP 执行器
# ---------------------------------------------------------------------------


class TWAPExecutor(BaseAlgoExecutor):
    """时间加权平均价格（TWAP）执行器。

    将母单在 ``duration_minutes`` 内等分为 ``num_slices`` 个子单，按固定
    ``interval_seconds`` 逐片下发。数量不可整除时，前 N-1 片向下取整，
    最后一片吸收余数，保证 Σ planned = total。

    Args:
        oms: 必传 OMS 实例。
        symbol: 标的。
        side: buy/sell。
        total_quantity: 总数量。
        algo_order_id: 母单 ID，缺省自动生成。
        duration_minutes: 执行总时长（分钟），与 interval_seconds 二选一。
        num_slices: 切片数量。
        interval_seconds: 切片间隔秒数，与 duration_minutes 二选一。
        order_type: 子单类型，默认 market。
        limit_price: 限价。
        benchmark_prices: 各切片基准价序列。

    Raises:
        AlgoOrderError: 切片数非法、duration 与 interval 同时给出且不一致。
    """

    def __init__(
        self,
        oms: OrderManagementSystem,
        symbol: str,
        side: str,
        total_quantity: float,
        num_slices: int,
        algo_order_id: Optional[str] = None,
        duration_minutes: Optional[float] = None,
        interval_seconds: Optional[float] = None,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        benchmark_prices: Optional[List[float]] = None,
    ) -> None:
        """校验参数一致性并推导 interval / duration。"""
        if num_slices <= 0:
            raise AlgoOrderError(f"num_slices 必须为正整数: {num_slices}")

        # duration 与 interval 二选一推导，同时给出时校验一致性
        if duration_minutes is not None and interval_seconds is not None:
            given = duration_minutes * 60.0
            implied = interval_seconds * (num_slices - 1) if num_slices > 1 else 0.0
            if abs(given - implied) > 1e-6:
                raise AlgoOrderError(
                    f"duration_minutes({duration_minutes}min) 与 interval_seconds"
                    f"({interval_seconds}s) 推导的总时长不一致"
                )
        elif duration_minutes is not None:
            span = duration_minutes * 60.0
            interval_seconds = (
                span / (num_slices - 1) if num_slices > 1 else 0.0
            )
        elif interval_seconds is not None:
            duration_minutes = (
                interval_seconds * (num_slices - 1) / 60.0 if num_slices > 1 else 0.0
            )
        else:
            raise AlgoOrderError("必须提供 duration_minutes 或 interval_seconds 之一")

        self.num_slices = int(num_slices)
        self.duration_minutes = float(duration_minutes)
        self.interval_seconds = float(interval_seconds)

        super().__init__(
            oms=oms,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            strategy="TWAP",
            algo_order_id=algo_order_id,
            order_type=order_type,
            limit_price=limit_price,
            benchmark_prices=benchmark_prices,
            params={
                "duration_minutes": self.duration_minutes,
                "interval_seconds": self.interval_seconds,
                "num_slices": self.num_slices,
            },
        )

    def build_schedule(self) -> List[SlicePlan]:
        """按时间等分构建切片计划（尾单吸收余数）。

        Returns:
            切片列表，Σ planned_quantity == total_quantity。
        """
        n = self.num_slices
        base = math.floor(self.order.total_quantity / n)
        slices: List[SlicePlan] = []
        for i in range(n):
            if i < n - 1:
                qty = base
            else:
                # 最后一片吸收余数，保证 Σ = total
                qty = self.order.total_quantity - base * (n - 1)
            slices.append(
                SlicePlan(
                    slice_index=i,
                    offset_seconds=round(i * self.interval_seconds, 6),
                    planned_quantity=float(qty),
                    raw_quantity=float(qty),
                )
            )
        self.order.slices = slices
        return slices


# ---------------------------------------------------------------------------
# VWAP 执行器
# ---------------------------------------------------------------------------


class VWAPExecutor(BaseAlgoExecutor):
    """成交量加权平均价格（VWAP）执行器。

    依据历史分时成交量（``volume_profile``，30 分钟 bar）计算各时段占比，
    按占比分配母单量；再用参与率（participation_rate）上限截断：若某时段
    ``市场成交量 × participation_rate`` 小于理想计划量，则该片截断到上限，
    未排出的缺口累计到 ``leftover_quantity`` 并在报告中披露。

    Volume profile 数据来源说明：
        典型 ``volume_profile`` 由历史（如前 20 个交易日）相同时段的 30 分钟
        成交量聚合而成，呈 U 型分布——开盘 / 收盘时段成交活跃（占比高），
        午间成交清淡（占比低）。生产环境来自行情数据仓库；此处作为构造注入，
        测试使用合成的 U 型数据。

    Args:
        oms: 必传 OMS 实例。
        symbol: 标的。
        side: buy/sell。
        total_quantity: 总数量。
        volume_profile: 分时成交量列表，每项 ``{"time"/"period"/"volume"}``。
        participation_rate: 参与率上限，默认 0.10。
        algo_order_id: 母单 ID，缺省自动生成。
        order_type: 子单类型，默认 market。
        limit_price: 限价。
        benchmark_prices: 各切片基准价序列。

    Raises:
        AlgoOrderError: volume_profile 为空或总成交量为 0。
    """

    def __init__(
        self,
        oms: OrderManagementSystem,
        symbol: str,
        side: str,
        total_quantity: float,
        volume_profile: List[Dict[str, Any]],
        participation_rate: float = 0.10,
        algo_order_id: Optional[str] = None,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        benchmark_prices: Optional[List[float]] = None,
    ) -> None:
        """校验 volume_profile 并初始化。"""
        if not volume_profile:
            raise AlgoOrderError("volume_profile 不能为空")
        self.volume_profile = [dict(b) for b in volume_profile]
        self.participation_rate = float(participation_rate)
        super().__init__(
            oms=oms,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            strategy="VWAP",
            algo_order_id=algo_order_id,
            order_type=order_type,
            limit_price=limit_price,
            benchmark_prices=benchmark_prices,
            params={
                "participation_rate": self.participation_rate,
                "num_bars": len(self.volume_profile),
            },
        )

    def build_schedule(self) -> List[SlicePlan]:
        """按成交量占比 + 参与率截断构建切片计划。

        Returns:
            切片列表；Σ planned_quantity ≤ total_quantity，差额记 leftover。
        """
        bars = self.volume_profile
        total_vol = sum(float(b.get("volume", 0.0)) for b in bars)
        if total_vol <= _EPS:
            raise AlgoOrderError("volume_profile 总成交量为 0")

        # 每个 bar 的时长（分钟）；period 可能是时长数值，也可能是 "09:30-10:00"
        # 形式的时段标签（见方法 docstring），标签无法转 float 时回退 30 分钟。
        raw_period = bars[0].get("period_minutes", bars[0].get("period", 30))
        try:
            period_minutes = float(raw_period)
        except (TypeError, ValueError):
            period_minutes = 30.0
        period_seconds = period_minutes * 60.0

        slices: List[SlicePlan] = []
        allocated = 0.0
        for i, bar in enumerate(bars):
            vol = float(bar.get("volume", 0.0))
            weight = vol / total_vol
            raw = self.order.total_quantity * weight
            cap = vol * self.participation_rate
            planned = min(raw, cap)
            planned = max(0.0, planned)
            allocated += planned
            slices.append(
                SlicePlan(
                    slice_index=i,
                    offset_seconds=round(i * period_seconds, 6),
                    planned_quantity=planned,
                    raw_quantity=raw,
                    market_volume=vol,
                )
            )
        self.order.slices = slices
        # 参与率截断导致的缺口，显式记录
        self.order.leftover_quantity = max(
            0.0, self.order.total_quantity - allocated
        )
        return slices

    def set_participation_rate(self, rate: float) -> None:
        """修改参与率，仅重算未提交（PENDING）切片的计划量。

        已提交切片对应的子单已发出，不再变动；缺口（leftover）随之重算。

        Args:
            rate: 新的参与率 (0, 1]。

        Raises:
            AlgoOrderError: rate 不在 (0, 1]。
        """
        if not (0.0 < rate <= 1.0):
            raise AlgoOrderError(f"participation_rate 须在 (0,1]: {rate}")
        self.participation_rate = float(rate)
        self.order.params["participation_rate"] = self.participation_rate

        total_vol = sum(s.market_volume for s in self.order.slices)
        allocated = 0.0
        for s in self.order.slices:
            if s.status == SLICE_PENDING:
                raw = self.order.total_quantity * (
                    s.market_volume / total_vol if total_vol > _EPS else 0.0
                )
                cap = s.market_volume * self.participation_rate
                s.raw_quantity = raw
                s.planned_quantity = max(0.0, min(raw, cap))
            allocated += s.planned_quantity
        self.order.leftover_quantity = max(
            0.0, self.order.total_quantity - allocated
        )
