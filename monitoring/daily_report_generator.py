"""每日绩效报告自动生成器。

从 SQLite 聚合当日 trades / account_snapshots / jev_decisions 数据，
生成结构化的每日报告并写入 ``daily_reports`` 表，支持幂等重复调用、
收盘后自动生成（15:30）以及导出 CSV。

与 monitoring/monitor.py 的 ``DailyReporter`` 的区别：
  - DailyReporter 接收内存中的实时数据，输出文本文件；
  - DailyReportGenerator 从 SQLite 聚合历史数据，结构化落库并可导出。
"""
from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from persistence.database import Database

logger = logging.getLogger(__name__)

# 默认账户列表（与 config/config.yaml accounts 节保持一致）
DEFAULT_ACCOUNT_IDS: List[str] = ["acc_1", "acc_2", "acc_3"]

# 默认收盘后自动生成时间：15:30
DEFAULT_REPORT_HOUR = 15
DEFAULT_REPORT_MINUTE = 30

# CSV 导出表头
CSV_HEADER = [
    "日期", "账户ID", "初始资产", "期末资产", "日收益率",
    "交易笔数", "胜率", "最大持仓", "风控事件数",
    "Jev决策数", "Jev执行数",
]


def _load_default_initial_capital() -> Dict[str, float]:
    """从 config/config.yaml 读取各账户初始资金（失败时返回空 dict）。"""
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    try:
        import yaml  # 延迟导入，避免未安装时影响核心功能

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        result: Dict[str, float] = {}
        for acc in cfg.get("accounts", []) or []:
            acc_id = acc.get("account_id")
            capital = acc.get("initial_capital")
            if acc_id and capital:
                result[acc_id] = float(capital)
        return result
    except Exception as exc:  # noqa: BLE001 - 兜底，不阻断报告生成
        logger.warning("读取 config.yaml 初始资金失败: %s", exc)
        return {}


class DailyReportGenerator:
    """每日绩效报告生成器：聚合 SQLite 数据 -> upsert 到 daily_reports 表。"""

    def __init__(
        self,
        db: Database,
        alert_manager: Any = None,
        account_ids: Optional[List[str]] = None,
        account_initial_capital: Optional[Dict[str, float]] = None,
    ) -> None:
        """初始化。

        Args:
            db: Database 持久化实例。
            alert_manager: AlertManager 实例（可选，用于报告生成异常告警）。
            account_ids: 需要生成报告的账户 ID 列表，默认 acc_1/acc_2/acc_3。
            account_initial_capital: 账户初始资金映射，用于无任何快照时的兜底。
                为 None 时自动从 config/config.yaml 读取。
        """
        self.db = db
        self.alert_manager = alert_manager
        self.account_ids = list(account_ids) if account_ids else list(DEFAULT_ACCOUNT_IDS)
        self.account_initial_capital = (
            dict(account_initial_capital)
            if account_initial_capital is not None
            else _load_default_initial_capital()
        )

    # ------------------------------------------------------------------
    # 内部聚合查询
    # ------------------------------------------------------------------

    def _conn(self):
        return self.db._get_conn()

    def _first_last_snapshot(
        self, account_id: str, day_prefix: str
    ) -> tuple[Optional[dict], Optional[dict]]:
        """返回当日 (第一条快照, 最后一条快照)，无快照时为 (None, None)。"""
        conn = self._conn()
        first = conn.execute(
            "SELECT total_asset, positions_json FROM account_snapshots "
            "WHERE account_id = ? AND timestamp LIKE ? ORDER BY id ASC LIMIT 1",
            (account_id, f"{day_prefix}%"),
        ).fetchone()
        last = conn.execute(
            "SELECT total_asset, positions_json FROM account_snapshots "
            "WHERE account_id = ? AND timestamp LIKE ? ORDER BY id DESC LIMIT 1",
            (account_id, f"{day_prefix}%"),
        ).fetchone()
        return (dict(first) if first else None, dict(last) if last else None)

    def _prev_day_end_asset(self, account_id: str, day_prefix: str) -> Optional[float]:
        """返回 day_prefix 之前最近一条快照的 total_asset（前一交易日期末资产）。"""
        conn = self._conn()
        row = conn.execute(
            "SELECT total_asset FROM account_snapshots "
            "WHERE account_id = ? AND timestamp < ? ORDER BY id DESC LIMIT 1",
            (account_id, day_prefix),
        ).fetchone()
        return float(row["total_asset"]) if row else None

    def _day_trades(self, account_id: str, day_prefix: str) -> List[dict]:
        """返回当日该账户的全部成交记录。"""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM trades WHERE account_id = ? AND timestamp LIKE ?",
            (account_id, f"{day_prefix}%"),
        ).fetchall()
        return [dict(r) for r in rows]

    def _max_positions_count(self, account_id: str, day_prefix: str) -> int:
        """当日所有快照中 positions 数组的最大长度。"""
        conn = self._conn()
        rows = conn.execute(
            "SELECT positions_json FROM account_snapshots "
            "WHERE account_id = ? AND timestamp LIKE ?",
            (account_id, f"{day_prefix}%"),
        ).fetchall()
        max_len = 0
        for r in rows:
            try:
                positions = json.loads(r["positions_json"] or "[]")
            except (json.JSONDecodeError, TypeError):
                positions = []
            if isinstance(positions, list):
                max_len = max(max_len, len(positions))
        return max_len

    def _risk_events_count(self, account_id: str, day_prefix: str) -> int:
        """当日 reason 含 止损/止盈/仓位 的交易记录数。"""
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM trades "
            "WHERE account_id = ? AND timestamp LIKE ? "
            "AND (reason LIKE '%止损%' OR reason LIKE '%止盈%' OR reason LIKE '%仓位%')",
            (account_id, f"{day_prefix}%"),
        ).fetchone()
        return int(row["c"]) if row else 0

    def _jev_stats(self, day_prefix: str) -> tuple[int, int]:
        """当日 Jev 决策 (总数, 执行数)。jev_decisions 无 account_id，按日统计。"""
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(executed), 0) AS executed_cnt "
            "FROM jev_decisions WHERE timestamp LIKE ?",
            (f"{day_prefix}%",),
        ).fetchone()
        if not row:
            return 0, 0
        return int(row["total"]), int(row["executed_cnt"])

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def generate(self, account_id: str, date: str) -> Dict[str, Any]:
        """生成指定日期的每日绩效报告，写入 SQLite 并返回报告 dict。

        Args:
            account_id: 账户 ID，如 "acc_1"。
            date: 交易日期，格式 "YYYY-MM-DD"。

        Returns:
            完整报告 dict，含初始/期末资产、日收益率、交易统计、持仓、风控、Jev 等。
        """
        try:
            first, last = self._first_last_snapshot(account_id, date)

            # 初始资产：当日第一条快照；否则前一交易日期末；否则配置初始资金
            if first is not None:
                start_asset = float(first["total_asset"])
            else:
                prev = self._prev_day_end_asset(account_id, date)
                start_asset = (
                    prev
                    if prev is not None
                    else float(self.account_initial_capital.get(account_id, 0.0))
                )

            # 期末资产：当日最后一条快照；无快照则等于初始资产
            end_asset = float(last["total_asset"]) if last is not None else start_asset

            trades = self._day_trades(account_id, date)
            trades_count = len(trades)

            # 胜率：只统计 side='sell' 且 realized_pnl != 0 的交易
            sell_closed = [t for t in trades if t.get("side") == "sell" and t.get("realized_pnl", 0) != 0]
            win_count = sum(1 for t in sell_closed if t.get("realized_pnl", 0) > 0)

            total_pnl = round(sum(float(t.get("realized_pnl", 0) or 0) for t in trades), 2)
            max_positions = self._max_positions_count(account_id, date)
            risk_events_count = self._risk_events_count(account_id, date)
            jev_total, jev_executed = self._jev_stats(date)

            self.db.upsert_daily_report(
                date=date,
                account_id=account_id,
                start_asset=start_asset,
                end_asset=end_asset,
                trades_count=trades_count,
                win_count=win_count,
                total_pnl=total_pnl,
                max_positions=max_positions,
                risk_events_count=risk_events_count,
                jev_decisions_count=jev_total,
                jev_executed_count=jev_executed,
            )

            report = self.get_report(account_id, date)
            if report is None:  # 理论上不会发生，兜底
                report = {
                    "date": date,
                    "account_id": account_id,
                    "start_asset": start_asset,
                    "end_asset": end_asset,
                    "daily_return": 0.0,
                    "trades_count": trades_count,
                    "win_count": win_count,
                    "win_rate": 0.0,
                    "total_pnl": total_pnl,
                    "max_positions": max_positions,
                    "risk_events_count": risk_events_count,
                    "jev_decisions_count": jev_total,
                    "jev_executed_count": jev_executed,
                    "extra_json": "{}",
                }
            return report
        except Exception as exc:  # noqa: BLE001 - 异常上报告警，不抛出
            logger.exception("生成每日报告失败: account=%s date=%s", account_id, date)
            if self.alert_manager is not None:
                try:
                    self.alert_manager.send_alert(
                        f"每日报告生成失败 account={account_id} date={date}: {exc}"
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("告警发送失败")
            raise

    def auto_generate(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """收盘后自动生成今日报告（幂等）。

        逻辑：
          1. 若当前时间早于今日 15:30，直接返回空列表；
          2. 对所有账户，若今日报告尚未生成则调用 generate()；
          3. 已生成过的账户跳过（幂等）。

        Args:
            now: 注入当前时间（测试用），默认 datetime.now()。

        Returns:
            本次新生成的报告 dict 列表。
        """
        if now is None:
            now = datetime.now()

        today_str = now.strftime("%Y-%m-%d")
        cutoff = now.replace(
            hour=DEFAULT_REPORT_HOUR, minute=DEFAULT_REPORT_MINUTE,
            second=0, microsecond=0,
        )
        if now < cutoff:
            logger.debug("未到报告时间 %s，跳过自动生成", cutoff.time())
            return []

        generated: List[Dict[str, Any]] = []
        for account_id in self.account_ids:
            if self.get_report(account_id, today_str) is not None:
                logger.info("今日报告已存在，跳过: %s %s", account_id, today_str)
                continue
            report = self.generate(account_id, today_str)
            generated.append(report)
        return generated

    def get_report(self, account_id: str, date: str) -> Optional[Dict[str, Any]]:
        """查询指定日期的报告，不存在返回 None。"""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM daily_reports WHERE account_id = ? AND date = ?",
            (account_id, date),
        ).fetchone()
        return dict(row) if row else None

    def export_csv(
        self,
        account_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """导出指定账户的每日报告为 CSV 字符串。

        Args:
            account_id: 账户 ID。
            start_date: 起始日期（含），格式 YYYY-MM-DD，可选。
            end_date: 结束日期（含），格式 YYYY-MM-DD，可选。

        Returns:
            CSV 格式字符串（UTF-8），含表头与数据行，按日期升序排列。
        """
        rows = self.db.get_daily_reports(account_id=account_id, limit=100000)
        # get_daily_reports 为倒序，这里反转为日期升序
        rows = list(reversed(rows))

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(CSV_HEADER)

        for r in rows:
            d = r["date"]
            if start_date and d < start_date:
                continue
            if end_date and d > end_date:
                continue
            writer.writerow([
                r["date"],
                r["account_id"],
                f'{r["start_asset"]:.2f}',
                f'{r["end_asset"]:.2f}',
                f'{r["daily_return"]:.6f}',
                r["trades_count"],
                f'{r["win_rate"]:.6f}',
                r.get("max_positions", 0),
                r.get("risk_events_count", 0),
                r.get("jev_decisions_count", 0),
                r.get("jev_executed_count", 0),
            ])
        return buf.getvalue()
