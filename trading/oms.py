"""订单管理系统（Order Management System, OMS）。

负责量化交易系统中订单的完整生命周期管理：

- 订单状态机：PENDING → SUBMITTED → (PARTIAL_FILLED) → FILLED / CANCELLED / REJECTED。
- 内存订单簿：活动订单 / 待提交订单 / 历史订单三类分桶，线程安全（``threading.RLock``）。
- 成交回报处理：累加成交量、按量加权重算成交均价，生成 :class:`Fill` 成交记录。
- 成交回调：支持注册多个 ``on_fill`` 回调，用于持仓 / 资金更新 / 通知。
- 持久化：内置 :class:`OmsRepository``，直接使用 SQLite（``data/quant_trading.db``），
  orders 表与 fills 表；传入 ``:memory:`` 或 None 时使用内存库（测试用）。
- 审计：所有下单 / 撤单 / 成交 / 废单均通过 ``security.audit`` 记录审计事件。

典型用法::

    oms = OrderManagementSystem(db_path="data/quant_trading.db")
    order = Order(order_id="o1", symbol="AAPL", side="buy",
                  order_type="limit", quantity=100, limit_price=150.0)
    oms.submit_order(order)
    oms.process_fill(order.order_id, fill_quantity=40, fill_price=150.5)
    oms.cancel_order(order.order_id, reason="risk")
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from security.audit import ActionType, get_audit_logger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 状态机定义
# ---------------------------------------------------------------------------

#: 合法订单状态
STATUS_PENDING = "PENDING"
STATUS_SUBMITTED = "SUBMITTED"
STATUS_PARTIAL_FILLED = "PARTIAL_FILLED"
STATUS_FILLED = "FILLED"
STATUS_CANCELLED = "CANCELLED"
STATUS_REJECTED = "REJECTED"

#: 终结态（不可再转换）
TERMINAL_STATES = frozenset({STATUS_FILLED, STATUS_CANCELLED, STATUS_REJECTED})

#: 合法状态转换映射：当前状态 -> 允许的下一状态集合
ORDER_TRANSITIONS: Dict[str, frozenset] = {
    STATUS_PENDING: frozenset({STATUS_SUBMITTED, STATUS_REJECTED}),
    STATUS_SUBMITTED: frozenset({
        STATUS_PARTIAL_FILLED, STATUS_FILLED, STATUS_CANCELLED, STATUS_REJECTED,
    }),
    STATUS_PARTIAL_FILLED: frozenset({
        STATUS_PARTIAL_FILLED, STATUS_FILLED, STATUS_CANCELLED,
    }),
    STATUS_FILLED: frozenset(),
    STATUS_CANCELLED: frozenset(),
    STATUS_REJECTED: frozenset(),
}

#: 合法买卖方向
VALID_SIDES = frozenset({"buy", "sell"})
#: 合法订单类型
VALID_ORDER_TYPES = frozenset({
    "market", "limit", "stop", "stop_limit", "trailing_stop",
})


class InvalidOrderStateError(Exception):
    """非法订单状态转换时抛出。

    例如对已终结（FILLED/CANCELLED/REJECTED）订单继续成交或撤单，
    或触发了 :data:`ORDER_TRANSITIONS` 中未定义的状态跳转。
    """


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Order:
    """单笔委托订单。

    Attributes:
        order_id: 订单唯一标识。
        symbol: 标的代码。
        side: 买卖方向，仅 ``buy`` / ``sell``。
        order_type: 订单类型，market/limit/stop/stop_limit/trailing_stop。
        quantity: 委托数量。
        filled_quantity: 已成交数量，默认 0。
        avg_fill_price: 已成交部分的加权平均成交价，默认 0。
        limit_price: 限价单价格，市价单为 None。
        stop_price: 止损/止触发价，无触发价为 None。
        status: 订单状态，默认 PENDING。
        created_at: 创建时间。
        submitted_at: 提交到交易所时间。
        filled_at: 全部成交时间。
        cancelled_at: 撤单时间。
        strategy_name: 来源策略名。
        account_id: 账户 ID。
        timeout_seconds: 活动订单超时秒数，超时自动撤单，默认 300。
        reject_reason: 废单原因。
    """

    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = STATUS_PENDING
    created_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    strategy_name: str = ""
    account_id: str = ""
    timeout_seconds: int = 300
    reject_reason: Optional[str] = None

    def __post_init__(self) -> None:
        """补默认创建时间。"""
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Fill:
    """一笔成交回报记录。

    Attributes:
        order_id: 所属订单 ID。
        fill_id: 成交唯一标识。
        symbol: 标的代码。
        side: 买卖方向。
        quantity: 本笔成交数量。
        price: 本笔成交价。
        timestamp: 成交时间。
    """

    order_id: str
    fill_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# 持久化
# ---------------------------------------------------------------------------


def _dt_to_iso(value: Optional[datetime]) -> Optional[str]:
    """datetime 转 ISO 字符串，None 透传。"""
    return value.isoformat() if value is not None else None


def _iso_to_dt(value: Optional[str]) -> Optional[datetime]:
    """ISO 字符串转 datetime，None/空串透传。"""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class OmsRepository:
    """OMS 持久化仓库（直接使用 sqlite3）。

    建表使用 ``CREATE TABLE IF NOT EXISTS``，连接开启 WAL 模式与
    ``row_factory=sqlite3.Row``。与 ``persistence.database.Database`` 相互独立，
    避免与其他并行开发冲突。

    Attributes:
        db_path: SQLite 路径；``:memory:`` 或 None 时使用内存库。
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """初始化仓库并建表。

        Args:
            db_path: SQLite 数据库文件路径；None / ``:memory:`` 使用内存库。
        """
        self.db_path: str = db_path if db_path else ":memory:"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            # 内存库不支持 WAL，忽略
            pass
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._create_tables()

    def _create_tables(self) -> None:
        """创建 orders / fills 表（若不存在）。"""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                filled_quantity REAL NOT NULL DEFAULT 0,
                avg_fill_price REAL NOT NULL DEFAULT 0,
                limit_price REAL,
                stop_price REAL,
                status TEXT NOT NULL,
                created_at TEXT,
                submitted_at TEXT,
                filled_at TEXT,
                cancelled_at TEXT,
                strategy_name TEXT DEFAULT '',
                account_id TEXT DEFAULT '',
                timeout_seconds INTEGER DEFAULT 300,
                reject_reason TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )

    # -- orders ----------------------------------------------------------

    def insert_order(self, order: Order) -> None:
        """插入一条订单记录（新订单）。

        Args:
            order: 待持久化的 :class:`Order`。
        """
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO orders (
                    order_id, symbol, side, order_type, quantity, filled_quantity,
                    avg_fill_price, limit_price, stop_price, status, created_at,
                    submitted_at, filled_at, cancelled_at, strategy_name,
                    account_id, timeout_seconds, reject_reason
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    order.order_id, order.symbol, order.side, order.order_type,
                    order.quantity, order.filled_quantity, order.avg_fill_price,
                    order.limit_price, order.stop_price, order.status,
                    _dt_to_iso(order.created_at), _dt_to_iso(order.submitted_at),
                    _dt_to_iso(order.filled_at), _dt_to_iso(order.cancelled_at),
                    order.strategy_name, order.account_id, order.timeout_seconds,
                    order.reject_reason,
                ),
            )

    def update_order(self, order: Order) -> None:
        """更新一条订单记录的全字段（与 insert 同构）。"""
        self.insert_order(order)

    def get_order(self, order_id: str) -> Optional[Order]:
        """按 order_id 查询订单，不存在返回 None。"""
        rows = self._conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchall()
        if not rows:
            return None
        return self._row_to_order(rows[0])

    def get_orders(
        self,
        status: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[Order]:
        """查询订单列表，可按状态 / 标的过滤。

        Args:
            status: 精确状态过滤，可选。
            symbol: 精确标的过滤，可选。

        Returns:
            匹配的 :class:`Order` 列表。
        """
        clauses: List[str] = []
        params: List[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM orders{where} ORDER BY created_at", params
        ).fetchall()
        return [self._row_to_order(r) for r in rows]

    @staticmethod
    def _row_to_order(row: sqlite3.Row) -> Order:
        """将数据库行重建为 :class:`Order`。"""
        return Order(
            order_id=row["order_id"],
            symbol=row["symbol"],
            side=row["side"],
            order_type=row["order_type"],
            quantity=row["quantity"],
            filled_quantity=row["filled_quantity"] or 0.0,
            avg_fill_price=row["avg_fill_price"] or 0.0,
            limit_price=row["limit_price"],
            stop_price=row["stop_price"],
            status=row["status"],
            created_at=_iso_to_dt(row["created_at"]),
            submitted_at=_iso_to_dt(row["submitted_at"]),
            filled_at=_iso_to_dt(row["filled_at"]),
            cancelled_at=_iso_to_dt(row["cancelled_at"]),
            strategy_name=row["strategy_name"] or "",
            account_id=row["account_id"] or "",
            timeout_seconds=row["timeout_seconds"] or 300,
            reject_reason=row["reject_reason"],
        )

    # -- fills -----------------------------------------------------------

    def insert_fill(self, fill: Fill) -> None:
        """插入一条成交记录。"""
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO fills (
                    fill_id, order_id, symbol, side, quantity, price, timestamp
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    fill.fill_id, fill.order_id, fill.symbol, fill.side,
                    fill.quantity, fill.price, _dt_to_iso(fill.timestamp),
                ),
            )

    def get_fills(self, order_id: Optional[str] = None) -> List[Fill]:
        """查询成交记录，可按订单过滤。"""
        if order_id:
            rows = self._conn.execute(
                "SELECT * FROM fills WHERE order_id = ? ORDER BY timestamp",
                (order_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM fills ORDER BY timestamp"
            ).fetchall()
        return [
            Fill(
                order_id=r["order_id"],
                fill_id=r["fill_id"],
                symbol=r["symbol"],
                side=r["side"],
                quantity=r["quantity"],
                price=r["price"],
                timestamp=_iso_to_dt(r["timestamp"]) or datetime.now(),
            )
            for r in rows
        ]

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()


# ---------------------------------------------------------------------------
# 订单管理系统
# ---------------------------------------------------------------------------


class OrderManagementSystem:
    """订单管理系统：完整订单生命周期管理。

    线程安全：所有对内存订单簿的读写均在 ``threading.RLock`` 保护下进行。

    Args:
        db_path: SQLite 路径；None / ``:memory:`` 使用内存库（测试用）。
        on_fill: 可选的首个成交回调，每笔成交后触发。
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        on_fill: Optional[Callable[[Fill], None]] = None,
    ) -> None:
        """初始化 OMS，加载历史订单与成交。"""
        self._repo = OmsRepository(db_path=db_path)
        self._lock = threading.RLock()
        self._callbacks: List[Callable[[Fill], None]] = []
        if on_fill is not None:
            self._callbacks.append(on_fill)

        # 订单簿分桶：master 索引 + 三类桶
        self._orders: Dict[str, Order] = {}
        self._pending: Dict[str, Order] = {}
        self._active: Dict[str, Order] = {}
        self._historical: Dict[str, Order] = {}
        self._fills: List[Fill] = []

        # 每订单成交序号（用于生成 fill_id）
        self._fill_seq: Dict[str, int] = {}

        self._load_from_db()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _load_from_db(self) -> None:
        """从持久层加载订单与成交到内存订单簿。"""
        for order in self._repo.get_orders():
            self._orders[order.order_id] = order
            self._rebucket(order)
        for fill in self._repo.get_fills():
            self._fills.append(fill)
            self._fill_seq[fill.order_id] = self._fill_seq.get(fill.order_id, 0) + 1

    def _rebucket(self, order: Order) -> None:
        """根据订单当前状态将其放入正确的订单簿分桶。"""
        oid = order.order_id
        self._pending.pop(oid, None)
        self._active.pop(oid, None)
        self._historical.pop(oid, None)
        if order.status == STATUS_PENDING:
            self._pending[oid] = order
        elif order.status in (STATUS_SUBMITTED, STATUS_PARTIAL_FILLED):
            self._active[oid] = order
        else:  # 终结态
            self._historical[oid] = order

    @staticmethod
    def _can_transition(current: str, target: str) -> bool:
        """判断从 current 到 target 的状态转换是否合法。"""
        return target in ORDER_TRANSITIONS.get(current, frozenset())

    def _transition(self, order: Order, target: str) -> None:
        """执行状态转换，非法转换抛 :class:`InvalidOrderStateError`。

        Args:
            order: 目标订单。
            target: 目标状态。

        Raises:
            InvalidOrderStateError: 状态转换不在 :data:`ORDER_TRANSITIONS` 中。
        """
        current = order.status
        if not self._can_transition(current, target):
            raise InvalidOrderStateError(
                f"非法状态转换: {current} -> {target} (order_id={order.order_id})"
            )
        order.status = target

    @staticmethod
    def _audit(
        action: str,
        order: Order,
        extra: Optional[Dict[str, Any]] = None,
        result: str = "success",
    ) -> None:
        """记录审计事件（失败不阻断主流程）。

        Args:
            action: 审计动作类型（ActionType 枚举或字符串）。
            order: 关联订单。
            extra: 额外参数字典。
            result: 结果描述。
        """
        params: Dict[str, Any] = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
        }
        if extra:
            params.update(extra)
        try:
            get_audit_logger().log(
                operator="oms",
                action_type=action,
                target=order.symbol,
                params=params,
                result=result,
            )
        except Exception:  # pragma: no cover - 审计失败不应阻断交易
            logger.exception("写入 OMS 审计日志失败: action=%s", action)

    def _trigger_fill(self, fill: Fill) -> None:
        """触发全部成交回调；单个回调异常不阻断主流程。"""
        for cb in list(self._callbacks):
            try:
                cb(fill)
            except Exception:
                logger.exception("on_fill 回调执行失败: order_id=%s", fill.order_id)

    # ------------------------------------------------------------------
    # 回调注册
    # ------------------------------------------------------------------

    def register_fill_callback(self, callback: Callable[[Fill], None]) -> None:
        """注册一个成交回报回调。

        Args:
            callback: 签名 ``callback(fill: Fill) -> None``。
        """
        with self._lock:
            self._callbacks.append(callback)

    # ------------------------------------------------------------------
    # 订单生命周期
    # ------------------------------------------------------------------

    @staticmethod
    def generate_order_id() -> str:
        """生成自动订单 ID（时间戳 + uuid 前 8 位）。"""
        return f"ORD{datetime.now().strftime('%Y%m%d%H%M%S%f')}{uuid.uuid4().hex[:8]}"

    def submit_order(self, order: Order) -> Order:
        """提交订单：PENDING → SUBMITTED。

        Args:
            order: 待提交订单（须为 PENDING）。

        Returns:
            提交后的订单（状态 SUBMITTED）。

        Raises:
            ValueError: 方向 / 类型非法。
            InvalidOrderStateError: 订单不在 PENDING 状态。
        """
        if order.side not in VALID_SIDES:
            raise ValueError(f"非法订单方向: {order.side}（仅 buy/sell）")
        if order.order_type not in VALID_ORDER_TYPES:
            raise ValueError(f"非法订单类型: {order.order_type}")

        with self._lock:
            if order.status != STATUS_PENDING:
                raise InvalidOrderStateError(
                    f"仅 PENDING 订单可提交，当前={order.status} "
                    f"(order_id={order.order_id})"
                )
            self._orders[order.order_id] = order
            self._rebucket(order)

            self._transition(order, STATUS_SUBMITTED)
            order.submitted_at = datetime.now()
            self._rebucket(order)

            self._repo.insert_order(order)
            self._audit(ActionType.ORDER_SUBMIT, order)
            return order

    def cancel_order(self, order_id: str, reason: str = "") -> Order:
        """撤销活动订单：SUBMITTED/PARTIAL_FILLED → CANCELLED。

        Args:
            order_id: 订单 ID。
            reason: 撤单原因。

        Returns:
            撤单后的订单。

        Raises:
            KeyError: 订单不存在。
            InvalidOrderStateError: 订单非活动状态，无法撤单。
        """
        with self._lock:
            order = self._must_get(order_id)
            if order.status not in (STATUS_SUBMITTED, STATUS_PARTIAL_FILLED):
                raise InvalidOrderStateError(
                    f"仅活动订单可撤单，当前={order.status} "
                    f"(order_id={order_id})"
                )
            self._transition(order, STATUS_CANCELLED)
            order.cancelled_at = datetime.now()
            self._rebucket(order)

            self._repo.update_order(order)
            self._audit(
                ActionType.ORDER_CANCEL, order, extra={"reason": reason}
            )
            return order

    def process_fill(
        self,
        order_id: str,
        fill_quantity: float,
        fill_price: float,
    ) -> Fill:
        """处理一笔成交回报。

        - 累加 filled_quantity；
        - 按量加权重算 avg_fill_price；
        - 未满量 → PARTIAL_FILLED；满量 → FILLED（记录 filled_at）；
        - 生成 :class:`Fill` 记录并触发成交回调；
        - 持久化 fill 与订单状态。

        Args:
            order_id: 订单 ID。
            fill_quantity: 本笔成交数量。
            fill_price: 本笔成交价。

        Returns:
            生成的 :class:`Fill` 记录。

        Raises:
            KeyError: 订单不存在。
            ValueError: 成交量超过委托数量。
            InvalidOrderStateError: 订单非活动状态，无法成交。
        """
        with self._lock:
            order = self._must_get(order_id)
            if order.status not in (STATUS_SUBMITTED, STATUS_PARTIAL_FILLED):
                raise InvalidOrderStateError(
                    f"仅活动订单可成交，当前={order.status} "
                    f"(order_id={order_id})"
                )
            new_filled = order.filled_quantity + fill_quantity
            if new_filled > order.quantity + 1e-9:
                raise ValueError(
                    f"成交数量({new_filled})超过委托数量({order.quantity}) "
                    f"(order_id={order_id})"
                )

            # 重算加权平均成交价
            old_qty = order.filled_quantity
            new_avg = (
                (order.avg_fill_price * old_qty + fill_price * fill_quantity)
                / new_filled
            ) if new_filled > 0 else 0.0
            order.filled_quantity = new_filled
            order.avg_fill_price = new_avg

            # 状态推进
            if abs(new_filled - order.quantity) <= 1e-9:
                self._transition(order, STATUS_FILLED)
                order.filled_at = datetime.now()
            else:
                self._transition(order, STATUS_PARTIAL_FILLED)
            self._rebucket(order)

            # 生成 Fill 记录
            self._fill_seq[order_id] = self._fill_seq.get(order_id, 0) + 1
            fill = Fill(
                order_id=order_id,
                fill_id=f"{order_id}-{self._fill_seq[order_id]}",
                symbol=order.symbol,
                side=order.side,
                quantity=fill_quantity,
                price=fill_price,
                timestamp=datetime.now(),
            )
            self._fills.append(fill)

            self._repo.insert_fill(fill)
            self._repo.update_order(order)
            self._audit(
                "ORDER_FILL",
                order,
                extra={
                    "fill_id": fill.fill_id,
                    "fill_quantity": fill_quantity,
                    "fill_price": fill_price,
                    "filled_quantity": order.filled_quantity,
                    "avg_fill_price": order.avg_fill_price,
                },
            )

        # 回调在锁外触发，避免回调内反向调用造成死锁
        self._trigger_fill(fill)
        return fill

    def reject_order(self, order_id: str, reason: str) -> Order:
        """废单：PENDING/SUBMITTED → REJECTED。

        Args:
            order_id: 订单 ID。
            reason: 废单原因（必填）。

        Returns:
            废单后的订单。

        Raises:
            KeyError: 订单不存在。
            InvalidOrderStateError: 订单状态不允许废单。
        """
        with self._lock:
            order = self._must_get(order_id)
            if not reason:
                raise ValueError("废单原因不能为空")
            self._transition(order, STATUS_REJECTED)
            order.reject_reason = reason
            self._rebucket(order)

            self._repo.update_order(order)
            self._audit(
                "ORDER_REJECT", order, extra={"reason": reason},
            )
            return order

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def _must_get(self, order_id: str) -> Order:
        """按 order_id 取订单，不存在抛 KeyError。"""
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"订单不存在: {order_id}")
        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """查询订单详情，不存在返回 None。"""
        with self._lock:
            return self._orders.get(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """查询活动订单（SUBMITTED/PARTIAL_FILLED），可按标的过滤。"""
        with self._lock:
            orders = list(self._active.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """查询待提交订单（PENDING），可按标的过滤。"""
        with self._lock:
            orders = list(self._pending.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_historical_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """查询历史（已终结）订单，可按标的过滤。"""
        with self._lock:
            orders = list(self._historical.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_fills(self, order_id: Optional[str] = None) -> List[Fill]:
        """查询成交记录，可按订单过滤。"""
        with self._lock:
            fills = list(self._fills)
        if order_id:
            fills = [f for f in fills if f.order_id == order_id]
        return fills

    def get_all_orders(self) -> List[Order]:
        """返回全部订单（含三类订单簿）。"""
        with self._lock:
            return list(self._orders.values())

    # ------------------------------------------------------------------
    # 超时撤单
    # ------------------------------------------------------------------

    def check_timeouts(self, now: Optional[datetime] = None) -> List[str]:
        """检查活动订单是否超时，超时自动撤单。

        Args:
            now: 注入当前时间（测试用），默认 ``datetime.now()``。

        Returns:
            被自动撤单的订单 ID 列表。
        """
        now = now or datetime.now()
        expired: List[str] = []
        # 先收集候选，避免在迭代字典时改动
        with self._lock:
            candidates = [
                o for o in self._active.values()
                if o.submitted_at is not None
                and (now - o.submitted_at).total_seconds() > o.timeout_seconds
            ]
        for order in candidates:
            try:
                self.cancel_order(order.order_id, reason="超时自动撤单")
                expired.append(order.order_id)
            except InvalidOrderStateError:
                logger.warning("超时撤单失败（状态已变化）: %s", order.order_id)
        return expired


# ---------------------------------------------------------------------------
# 序列化辅助（供路由层输出）
# ---------------------------------------------------------------------------


def order_to_dict(order: Order) -> Dict[str, Any]:
    """将 :class:`Order` 序列化为可 JSON 化的字典。

    Args:
        order: 订单对象。

    Returns:
        字段字典，datetime 转 ISO 字符串。
    """
    data = asdict(order)
    for key in (
        "created_at", "submitted_at", "filled_at", "cancelled_at",
    ):
        if isinstance(data.get(key), datetime):
            data[key] = data[key].isoformat()
    return data


def fill_to_dict(fill: Fill) -> Dict[str, Any]:
    """将 :class:`Fill` 序列化为可 JSON 化的字典。"""
    data = asdict(fill)
    if isinstance(data.get("timestamp"), datetime):
        data["timestamp"] = data["timestamp"].isoformat()
    return data
