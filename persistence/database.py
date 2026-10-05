"""SQLite 持久化数据库。

四张表：
  trades           — 每笔成交记录（买卖/价格/数量/手续费/盈亏）
  jev_decisions    — Jev 决策审计（输入特征/概率分布/最终决策/延迟）
  account_snapshots — 账户快照（总资产/现金/持仓市值/持仓JSON）
  daily_reports    — 每日绩效报告（日收益/交易次数/胜率）

设计原则：
  - 内置 sqlite3，零额外依赖
  - WAL 模式支持并发读写
  - JSON 字段用 TEXT 存储，读写时自动序列化/反序列化
  - 每次写入独立事务，崩溃不丢数据
  - 查询返回 dict 列表，直接可序列化为 JSON
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 默认数据库路径（项目根目录/data/quant_trading.db）
_DEFAULT_DB_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "quant_trading.db"
)


class Database:
    """SQLite 持久化管理器。

    线程安全：每个线程使用独立连接（thread-local），避免 sqlite3 跨线程问题。
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        """初始化数据库连接并建表。

        Args:
            db_path: SQLite 数据库文件路径，默认 data/quant_trading.db。
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()
        logger.info("数据库已初始化: %s", db_path)

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接（懒加载）。"""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """关闭当前线程的数据库连接。"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------
    # 建表
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """创建所有表（如果不存在）。"""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                name TEXT DEFAULT '',
                side TEXT NOT NULL,           -- buy / sell
                price REAL NOT NULL,
                fill_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                amount REAL NOT NULL,
                commission REAL NOT NULL DEFAULT 0,
                stamp_tax REAL NOT NULL DEFAULT 0,
                realized_pnl REAL NOT NULL DEFAULT 0,
                reason TEXT DEFAULT '',
                account_id TEXT DEFAULT 'default',
                strategy TEXT DEFAULT '',
                jev_confidence REAL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_account ON trades(account_id);

            CREATE TABLE IF NOT EXISTS jev_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy_signal TEXT DEFAULT '',
                strategy_confidence REAL DEFAULT 0,
                market_state_json TEXT DEFAULT '{}',
                probabilities_json TEXT DEFAULT '{}',
                final_action TEXT NOT NULL,
                final_confidence REAL NOT NULL,
                executed INTEGER NOT NULL DEFAULT 0,  -- 0/1
                reason TEXT DEFAULT '',
                mode TEXT DEFAULT 'mock',            -- real / mock
                latency_ms REAL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_jev_timestamp ON jev_decisions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_jev_symbol ON jev_decisions(symbol);

            CREATE TABLE IF NOT EXISTS account_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                account_id TEXT NOT NULL,
                total_asset REAL NOT NULL,
                cash REAL NOT NULL,
                position_value REAL NOT NULL,
                daily_pnl REAL DEFAULT 0,
                positions_json TEXT DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_snapshot_timestamp ON account_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_snapshot_account ON account_snapshots(account_id);

            CREATE TABLE IF NOT EXISTS daily_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                account_id TEXT NOT NULL,
                start_asset REAL NOT NULL,
                end_asset REAL NOT NULL,
                daily_return REAL NOT NULL,
                trades_count INTEGER NOT NULL DEFAULT 0,
                win_count INTEGER NOT NULL DEFAULT 0,
                win_rate REAL NOT NULL DEFAULT 0,
                total_pnl REAL NOT NULL DEFAULT 0,
                UNIQUE(date, account_id)
            );

            CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_reports(date);

            CREATE TABLE IF NOT EXISTS walkthroughs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                snapshots_json TEXT NOT NULL DEFAULT '[]',
                trades_json TEXT NOT NULL DEFAULT '[]',
                meta_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_walkthroughs_created ON walkthroughs(created_at);

            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                operator TEXT NOT NULL DEFAULT '',
                action_type TEXT NOT NULL,
                target TEXT DEFAULT '',
                params TEXT DEFAULT '{}',
                result TEXT DEFAULT '',
                ip TEXT DEFAULT '',
                request_id TEXT DEFAULT '',
                prev_hash TEXT DEFAULT '',
                hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action_type);
            CREATE INDEX IF NOT EXISTS idx_audit_operator ON audit_logs(operator);
        """)

        # 兼容已有库：为 daily_reports 追加扩展列（列已存在时静默跳过）。
        # 这些列由 DailyReportGenerator 写入，用于更细粒度的每日绩效统计。
        for alter_sql in (
            "ALTER TABLE daily_reports ADD COLUMN max_positions INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE daily_reports ADD COLUMN risk_events_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE daily_reports ADD COLUMN jev_decisions_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE daily_reports ADD COLUMN jev_executed_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE daily_reports ADD COLUMN extra_json TEXT NOT NULL DEFAULT '{}'",
        ):
            try:
                conn.execute(alter_sql)
            except sqlite3.OperationalError:
                # 列已存在（SQLite 对重复 ADD COLUMN 抛 OperationalError），忽略。
                pass
        conn.commit()

    # ------------------------------------------------------------------
    # 写入：交易记录
    # ------------------------------------------------------------------

    def insert_trade(
        self,
        timestamp: str,
        symbol: str,
        side: str,
        price: float,
        fill_price: float,
        quantity: int,
        amount: float,
        commission: float = 0.0,
        stamp_tax: float = 0.0,
        realized_pnl: float = 0.0,
        reason: str = "",
        account_id: str = "default",
        strategy: str = "",
        jev_confidence: float = 0.0,
        name: str = "",
    ) -> int:
        """插入一笔成交记录。

        Returns:
            新记录的行 ID。
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO trades
               (timestamp, symbol, name, side, price, fill_price, quantity,
                amount, commission, stamp_tax, realized_pnl, reason,
                account_id, strategy, jev_confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, symbol, name, side, price, fill_price, quantity,
             amount, commission, stamp_tax, realized_pnl, reason,
             account_id, strategy, jev_confidence),
        )
        conn.commit()
        return cursor.lastrowid

    def get_trades(
        self,
        limit: int = 50,
        symbol: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """查询最近的成交记录（倒序，最新在前）。

        Args:
            limit: 返回条数上限。
            symbol: 按标的过滤（可选）。
            account_id: 按账户过滤（可选）。

        Returns:
            成交记录字典列表。
        """
        conn = self._get_conn()
        query = "SELECT * FROM trades WHERE 1=1"
        params: list = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if account_id:
            query += " AND account_id = ?"
            params.append(account_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # 写入：Jev 决策
    # ------------------------------------------------------------------

    def insert_jev_decision(
        self,
        timestamp: str,
        symbol: str,
        strategy_signal: str,
        strategy_confidence: float,
        market_state: Dict[str, Any],
        probabilities: Dict[str, float],
        final_action: str,
        final_confidence: float,
        executed: bool,
        reason: str = "",
        mode: str = "mock",
        latency_ms: float = 0.0,
    ) -> int:
        """插入一条 Jev 决策审计记录。

        Returns:
            新记录的行 ID。
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO jev_decisions
               (timestamp, symbol, strategy_signal, strategy_confidence,
                market_state_json, probabilities_json, final_action,
                final_confidence, executed, reason, mode, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp, symbol, strategy_signal, strategy_confidence,
                json.dumps(market_state, ensure_ascii=False),
                json.dumps(probabilities, ensure_ascii=False),
                final_action, final_confidence,
                1 if executed else 0, reason, mode, latency_ms,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def get_jev_decisions(
        self, limit: int = 50, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """查询最近的 Jev 决策记录（倒序），自动反序列化 JSON 字段。"""
        conn = self._get_conn()
        query = "SELECT * FROM jev_decisions"
        params: list = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["market_state"] = json.loads(d.pop("market_state_json", "{}"))
            d["probabilities"] = json.loads(d.pop("probabilities_json", "{}"))
            d["executed"] = bool(d["executed"])
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # 写入：账户快照
    # ------------------------------------------------------------------

    def insert_account_snapshot(
        self,
        timestamp: str,
        account_id: str,
        total_asset: float,
        cash: float,
        position_value: float,
        daily_pnl: float = 0.0,
        positions: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """插入一条账户快照。

        Args:
            positions: 持仓列表，每个元素含 symbol/shares/avg_cost/market_value。

        Returns:
            新记录的行 ID。
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO account_snapshots
               (timestamp, account_id, total_asset, cash, position_value,
                daily_pnl, positions_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp, account_id, total_asset, cash, position_value,
                daily_pnl, json.dumps(positions or [], ensure_ascii=False),
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def get_account_snapshots(
        self, account_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """查询某账户的历史快照（倒序），自动反序列化 positions。"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM account_snapshots WHERE account_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (account_id, limit),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["positions"] = json.loads(d.pop("positions_json", "[]"))
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # 写入：每日报告
    # ------------------------------------------------------------------

    def upsert_daily_report(
        self,
        date: str,
        account_id: str,
        start_asset: float,
        end_asset: float,
        trades_count: int = 0,
        win_count: int = 0,
        total_pnl: float = 0.0,
        max_positions: int = 0,
        risk_events_count: int = 0,
        jev_decisions_count: int = 0,
        jev_executed_count: int = 0,
        extra_json: Optional[str] = None,
    ) -> None:
        """插入或更新每日报告（按 date+account_id 去重）。

        Args:
            date: 交易日期，格式 YYYY-MM-DD。
            account_id: 账户 ID。
            start_asset: 当日初始资产。
            end_asset: 当日期末资产。
            trades_count: 当日交易笔数。
            win_count: 当日盈利卖出笔数。
            total_pnl: 当日已实现盈亏合计。
            max_positions: 当日最大持仓标的数。
            risk_events_count: 当日风控事件数（止损/止盈/仓位限制）。
            jev_decisions_count: 当日 Jev 决策总数。
            jev_executed_count: 当日 Jev 决策执行数。
            extra_json: 额外扩展字段（JSON 字符串），默认 "{}"。
        """
        daily_return = (
            (end_asset - start_asset) / start_asset if start_asset > 0 else 0.0
        )
        win_rate = win_count / trades_count if trades_count > 0 else 0.0
        extra_json = extra_json if extra_json is not None else "{}"
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO daily_reports
               (date, account_id, start_asset, end_asset, daily_return,
                trades_count, win_count, win_rate, total_pnl,
                max_positions, risk_events_count,
                jev_decisions_count, jev_executed_count, extra_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(date, account_id) DO UPDATE SET
                 end_asset=excluded.end_asset,
                 daily_return=excluded.daily_return,
                 trades_count=excluded.trades_count,
                 win_count=excluded.win_count,
                 win_rate=excluded.win_rate,
                 total_pnl=excluded.total_pnl,
                 max_positions=excluded.max_positions,
                 risk_events_count=excluded.risk_events_count,
                 jev_decisions_count=excluded.jev_decisions_count,
                 jev_executed_count=excluded.jev_executed_count,
                 extra_json=excluded.extra_json""",
            (date, account_id, start_asset, end_asset, daily_return,
             trades_count, win_count, win_rate, total_pnl,
             max_positions, risk_events_count,
             jev_decisions_count, jev_executed_count, extra_json),
        )
        conn.commit()

    def get_daily_reports(
        self, account_id: Optional[str] = None, limit: int = 30
    ) -> List[Dict[str, Any]]:
        """查询每日报告（倒序，最新在前）。"""
        conn = self._get_conn()
        query = "SELECT * FROM daily_reports"
        params: list = []
        if account_id:
            query += " WHERE account_id = ?"
            params.append(account_id)
        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # 写入：回测走查（walkthrough）
    # ------------------------------------------------------------------

    def insert_walkthrough(
        self,
        id: str,
        created_at: str,
        symbol: str,
        strategy: str,
        start_date: str,
        end_date: str,
        snapshots: List[Dict[str, Any]],
        trades: List[Dict[str, Any]],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """插入一条回测走查记录（快照/交易/元信息以 JSON TEXT 存储）。

        Args:
            id: 走查唯一 ID（uuid4 短码）。
            created_at: 创建时间 ISO 字符串。
            symbol: 标的代码。
            strategy: 策略名称。
            start_date: 回测开始日期 YYYY-MM-DD。
            end_date: 回测结束日期 YYYY-MM-DD。
            snapshots: 逐日快照列表。
            trades: 成交明细列表。
            meta: 额外元信息（初始资金/快照数/交易数等）。
        """
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO walkthroughs
               (id, created_at, symbol, strategy, start_date, end_date,
                snapshots_json, trades_json, meta_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                id, created_at, symbol, strategy, start_date, end_date,
                json.dumps(snapshots, ensure_ascii=False),
                json.dumps(trades, ensure_ascii=False),
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )
        conn.commit()

    def get_walkthrough(self, walkthrough_id: str) -> Optional[Dict[str, Any]]:
        """按 ID 查询走查记录，自动反序列化 JSON 字段。

        Returns:
            走查字典（含 snapshots/trades/meta），不存在返回 None。
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM walkthroughs WHERE id = ?",
            (walkthrough_id,),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["snapshots"] = json.loads(d.pop("snapshots_json", "[]"))
        d["trades"] = json.loads(d.pop("trades_json", "[]"))
        d["meta"] = json.loads(d.pop("meta_json", "{}"))
        return d

    def list_walkthroughs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """列出最近的走查记录（倒序），不展开大体积 JSON 字段。"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, created_at, symbol, strategy, start_date, end_date, meta_json "
            "FROM walkthroughs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["meta"] = json.loads(d.pop("meta_json", "{}"))
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # 写入：安全审计日志（audit_logs，哈希链）
    # ------------------------------------------------------------------

    def insert_audit_log(
        self,
        timestamp: str,
        operator: str,
        action_type: str,
        target: str = "",
        params: str = "{}",
        result: str = "",
        ip: str = "",
        request_id: str = "",
        prev_hash: str = "",
        entry_hash: str = "",
    ) -> int:
        """插入一条安全审计记录。

        Args:
            timestamp: 记录时间 ISO 字符串。
            operator: 操作人名称。
            action_type: 操作类型（LOGIN / ORDER_SUBMIT / ...）。
            target: 操作目标。
            params: 参数字典的 JSON 字符串。
            result: 操作结果描述。
            ip: 来源 IP。
            request_id: 请求唯一标识。
            prev_hash: 前一条记录的 hash。
            entry_hash: 本条记录的 hash（SHA-256 链）。

        Returns:
            新记录的行 ID。
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO audit_logs
               (timestamp, operator, action_type, target, params,
                result, ip, request_id, prev_hash, hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, operator, action_type, target, params,
             result, ip, request_id, prev_hash, entry_hash),
        )
        conn.commit()
        return cursor.lastrowid

    def query_audit_logs(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        action_type: Optional[str] = None,
        operator: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """查询审计日志（默认按 id 正序返回，供完整性校验用）。

        Args:
            start: 起始时间过滤（>=），可选。
            end: 结束时间过滤（<=），可选。
            action_type: 按操作类型精确过滤，可选。
            operator: 按操作人精确过滤，可选。
            limit: 返回条数上限；None 表示不限制（校验链时取全量）。
            offset: 偏移量（分页用）。

        Returns:
            审计记录字典列表（按 id 正序）。
        """
        conn = self._get_conn()
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params: list = []
        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)
        if action_type:
            query += " AND action_type = ?"
            params.append(action_type)
        if operator:
            query += " AND operator = ?"
            params.append(operator)
        query += " ORDER BY id ASC"
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_last_audit_hash(self) -> str:
        """获取最后一条审计记录的 hash（用于哈希链续接）。

        Returns:
            末条记录的 hash；表为空时返回空串（由调用方回退到创世哈希）。
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT hash FROM audit_logs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["hash"] if row else ""

    def count_audit_logs(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        action_type: Optional[str] = None,
        operator: Optional[str] = None,
    ) -> int:
        """统计符合过滤条件的审计记录总数。"""
        conn = self._get_conn()
        query = "SELECT COUNT(*) as cnt FROM audit_logs WHERE 1=1"
        params: list = []
        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)
        if action_type:
            query += " AND action_type = ?"
            params.append(action_type)
        if operator:
            query += " AND operator = ?"
            params.append(operator)
        row = conn.execute(query, params).fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # 统计查询
    # ------------------------------------------------------------------

    def get_trade_stats(self, account_id: Optional[str] = None) -> Dict[str, Any]:
        """获取交易统计概览。

        Returns:
            含 total_trades / total_pnl / win_count / loss_count / win_rate。
        """
        conn = self._get_conn()
        query = (
            "SELECT COUNT(*) as total, "
            "COALESCE(SUM(realized_pnl), 0) as total_pnl, "
            "COALESCE(SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END), 0) as wins, "
            "COALESCE(SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END), 0) as losses "
            "FROM trades WHERE side = 'sell'"
        )
        params: list = []
        if account_id:
            query += " AND account_id = ?"
            params.append(account_id)
        row = conn.execute(query, params).fetchone()
        total = row["total"] if row else 0
        wins = row["wins"] if row else 0
        return {
            "total_trades": total,
            "total_pnl": round(row["total_pnl"], 2) if row else 0.0,
            "win_count": wins,
            "loss_count": row["losses"] if row else 0,
            "win_rate": round(wins / total, 4) if total > 0 else 0.0,
        }

    def get_db_size(self) -> int:
        """获取数据库文件大小（字节）。"""
        return Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0

    def count_table(self, table: str, where: str = "", params: tuple = ()) -> int:
        """统计表记录数。

        Args:
            table: 表名。
            where: 可选 WHERE 子句（不含 WHERE 关键字）。
            params: WHERE 参数。

        Returns:
            记录数。
        """
        conn = self._get_conn()
        query = f"SELECT COUNT(*) as cnt FROM {table}"
        if where:
            query += f" WHERE {where}"
        row = conn.execute(query, params).fetchone()
        return row["cnt"] if row else 0
