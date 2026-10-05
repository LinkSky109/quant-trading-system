"""DailyReportGenerator 单元测试。

使用临时 SQLite 数据库，手动插入 trades / account_snapshots / jev_decisions，
验证报告聚合、upsert 幂等、自动生成时间门控、CSV 导出与旧表兼容。
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime

import pytest

from monitoring.daily_report_generator import DailyReportGenerator
from persistence.database import Database


@pytest.fixture
def db(tmp_path) -> Database:
    """每个测试独立的临时数据库。"""
    return Database(db_path=str(tmp_path / "test_report.db"))


@pytest.fixture
def gen(db) -> DailyReportGenerator:
    """只针对 acc_1 的生成器，避免依赖真实 config。"""
    return DailyReportGenerator(
        db=db,
        account_ids=["acc_1"],
        account_initial_capital={"acc_1": 1_000_000.0},
    )


def _seed_full_day(db: Database, date: str = "2024-06-03", account_id: str = "acc_1") -> None:
    """为某日插入一组完整的快照/交易/Jev 决策测试数据。"""
    # 快照：开盘 100w，持仓 1 个；收盘 102w，持仓 2 个
    db.insert_account_snapshot(
        timestamp=f"{date}T09:30:00", account_id=account_id,
        total_asset=1_000_000.0, cash=1_000_000.0, position_value=0.0,
        positions=[{"symbol": "600519.SH", "shares": 100}],
    )
    db.insert_account_snapshot(
        timestamp=f"{date}T15:00:00", account_id=account_id,
        total_asset=1_020_000.0, cash=500_000.0, position_value=520_000.0,
        positions=[
            {"symbol": "600519.SH", "shares": 100},
            {"symbol": "300750.SZ", "shares": 200},
        ],
    )
    # 交易：1 笔买入 + 3 笔卖出（2 盈 1 亏）
    db.insert_trade(f"{date}T10:00:00", "600519.SH", "buy", 100, 100, 100, 10_000,
                    realized_pnl=0.0, reason="买入成交", account_id=account_id)
    db.insert_trade(f"{date}T10:30:00", "600519.SH", "sell", 110, 110, 100, 11_000,
                    realized_pnl=500.0, reason="止盈卖出", account_id=account_id)
    db.insert_trade(f"{date}T11:00:00", "300750.SZ", "sell", 200, 200, 100, 20_000,
                    realized_pnl=-300.0, reason="止损卖出", account_id=account_id)
    db.insert_trade(f"{date}T13:00:00", "600036.SH", "sell", 50, 50, 100, 5_000,
                    realized_pnl=100.0, reason="策略卖出", account_id=account_id)
    # Jev 决策：3 条，2 条执行
    for i, executed in enumerate([True, True, False]):
        db.insert_jev_decision(
            timestamp=f"{date}T1{i}:00:00", symbol="600519.SH",
            strategy_signal="buy", strategy_confidence=0.7,
            market_state={}, probabilities={},
            final_action="buy" if executed else "hold",
            final_confidence=0.65, executed=executed,
        )


# ---------------------------------------------------------------------------
# 1. generate() 生成并写入 SQLite
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_generate_writes_report(self, gen, db):
        _seed_full_day(db, "2024-06-03")
        report = gen.generate("acc_1", "2024-06-03")

        assert report["date"] == "2024-06-03"
        assert report["account_id"] == "acc_1"
        # 落库
        rows = db.get_daily_reports(account_id="acc_1")
        assert len(rows) == 1

    def test_report_data_consistency(self, gen, db):
        _seed_full_day(db, "2024-06-03")
        r = gen.generate("acc_1", "2024-06-03")

        assert r["start_asset"] == pytest.approx(1_000_000.0)
        assert r["end_asset"] == pytest.approx(1_020_000.0)
        assert r["daily_return"] == pytest.approx(0.02)
        assert r["trades_count"] == 4
        assert r["win_count"] == 2          # 卖出且 pnl>0：500、100
        assert r["win_rate"] == pytest.approx(2 / 4)  # upsert 口径 = win_count/trades_count
        assert r["total_pnl"] == pytest.approx(300.0)
        assert r["max_positions"] == 2
        assert r["risk_events_count"] == 2  # 止盈 + 止损
        assert r["jev_decisions_count"] == 3
        assert r["jev_executed_count"] == 2

    def test_upsert_idempotent_no_duplicate(self, gen, db):
        _seed_full_day(db, "2024-06-03")
        gen.generate("acc_1", "2024-06-03")
        gen.generate("acc_1", "2024-06-03")  # 重复调用同一天
        rows = db.get_daily_reports(account_id="acc_1")
        assert len(rows) == 1

    def test_no_data_returns_zero_report(self, gen, db):
        # 当日无任何数据：用配置初始资金兜底，不应崩溃
        r = gen.generate("acc_1", "2024-01-01")
        assert r["start_asset"] == pytest.approx(1_000_000.0)
        assert r["end_asset"] == pytest.approx(1_000_000.0)
        assert r["daily_return"] == 0.0
        assert r["trades_count"] == 0
        assert r["win_count"] == 0
        assert r["total_pnl"] == 0.0
        assert r["max_positions"] == 0
        assert r["risk_events_count"] == 0
        assert r["jev_decisions_count"] == 0

    def test_start_asset_falls_back_to_prev_day_snapshot(self, gen, db):
        # 当日无快照，但前一日有快照 -> 初始资产取前一日期末
        db.insert_account_snapshot(
            "2024-06-02T15:00:00", "acc_1", 950_000.0, 950_000.0, 0.0, positions=[],
        )
        r = gen.generate("acc_1", "2024-06-03")
        assert r["start_asset"] == pytest.approx(950_000.0)
        assert r["end_asset"] == pytest.approx(950_000.0)  # 当日无快照

    def test_trades_filtered_by_account(self, gen, db):
        # 其他账户的交易不计入
        _seed_full_day(db, "2024-06-03", account_id="acc_1")
        db.insert_trade("2024-06-03T12:00:00", "X", "sell", 1, 1, 1, 1,
                        realized_pnl=999.0, reason="x", account_id="acc_2")
        r = gen.generate("acc_1", "2024-06-03")
        assert r["trades_count"] == 4
        assert r["total_pnl"] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# auto_generate()
# ---------------------------------------------------------------------------

class TestAutoGenerate:
    def test_after_1530_generates(self, gen, db):
        _seed_full_day(db, "2024-06-03")
        now = datetime(2024, 6, 3, 15, 45)
        generated = gen.auto_generate(now=now)
        # auto_generate 用的是"今天"，即 now 的日期
        assert len(generated) == 1
        assert generated[0]["date"] == "2024-06-03"
        assert generated[0]["account_id"] == "acc_1"

    def test_before_1530_returns_empty(self, gen, db):
        _seed_full_day(db, "2024-06-03")
        now = datetime(2024, 6, 3, 10, 0)
        assert gen.auto_generate(now=now) == []
        # 未生成 -> 表中无记录
        assert db.get_daily_reports(account_id="acc_1") == []

    def test_auto_generate_idempotent(self, gen, db):
        _seed_full_day(db, "2024-06-03")
        now = datetime(2024, 6, 3, 16, 0)
        first = gen.auto_generate(now=now)
        second = gen.auto_generate(now=now)
        assert len(first) == 1
        assert second == []  # 已生成，跳过
        rows = db.get_daily_reports(account_id="acc_1")
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# get_report / export_csv
# ---------------------------------------------------------------------------

class TestQueryAndExport:
    def test_get_report_exists_and_missing(self, gen, db):
        _seed_full_day(db, "2024-06-03")
        gen.generate("acc_1", "2024-06-03")

        existing = gen.get_report("acc_1", "2024-06-03")
        assert existing is not None
        assert existing["account_id"] == "acc_1"
        assert existing["max_positions"] == 2

        assert gen.get_report("acc_1", "2020-01-01") is None

    def test_export_csv_format(self, gen, db):
        _seed_full_day(db, "2024-06-03")
        gen.generate("acc_1", "2024-06-03")
        csv_text = gen.export_csv("acc_1")

        lines = csv_text.strip().splitlines()
        # 表头 + 1 行数据
        assert len(lines) == 2
        assert lines[0].startswith("日期,账户ID,初始资产,期末资产,日收益率,交易笔数,胜率")
        assert "2024-06-03" in lines[1]
        assert "acc_1" in lines[1]

    def test_export_csv_date_range_filter(self, gen, db):
        for d in ["2024-06-03", "2024-06-04", "2024-06-05"]:
            db.insert_account_snapshot(f"{d}T09:30:00", "acc_1",
                                      1_000_000.0, 1_000_000.0, 0.0, positions=[])
            gen.generate("acc_1", d)

        full = gen.export_csv("acc_1").strip().splitlines()
        assert len(full) == 4  # header + 3 rows

        partial = gen.export_csv("acc_1", start_date="2024-06-04",
                                 end_date="2024-06-04").strip().splitlines()
        assert len(partial) == 2  # header + 1 row
        assert "2024-06-04" in partial[1]
        assert "2024-06-03" not in partial[1]


# ---------------------------------------------------------------------------
# 旧表兼容：ALTER TABLE 不丢失已有数据
# ---------------------------------------------------------------------------

class TestLegacyCompat:
    def test_alter_table_preserves_legacy_rows(self, tmp_path):
        db_path = str(tmp_path / "legacy.db")

        # 1) 先用裸连接建"旧版" daily_reports 表（只有老字段），并写入一行旧数据
        raw = sqlite3.connect(db_path)
        raw.executescript("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
                price REAL NOT NULL, fill_price REAL NOT NULL, quantity INTEGER NOT NULL,
                amount REAL NOT NULL, commission REAL NOT NULL DEFAULT 0,
                stamp_tax REAL NOT NULL DEFAULT 0, realized_pnl REAL NOT NULL DEFAULT 0,
                reason TEXT DEFAULT '', account_id TEXT DEFAULT 'default',
                strategy TEXT DEFAULT '', jev_confidence REAL DEFAULT 0
            );
            CREATE TABLE jev_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, symbol TEXT NOT NULL, strategy_signal TEXT DEFAULT '',
                strategy_confidence REAL DEFAULT 0, market_state_json TEXT DEFAULT '{}',
                probabilities_json TEXT DEFAULT '{}', final_action TEXT NOT NULL,
                final_confidence REAL NOT NULL, executed INTEGER NOT NULL DEFAULT 0,
                reason TEXT DEFAULT '', mode TEXT DEFAULT 'mock', latency_ms REAL DEFAULT 0
            );
            CREATE TABLE account_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, account_id TEXT NOT NULL, total_asset REAL NOT NULL,
                cash REAL NOT NULL, position_value REAL NOT NULL, daily_pnl REAL DEFAULT 0,
                positions_json TEXT DEFAULT '[]'
            );
            CREATE TABLE daily_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL, account_id TEXT NOT NULL, start_asset REAL NOT NULL,
                end_asset REAL NOT NULL, daily_return REAL NOT NULL,
                trades_count INTEGER NOT NULL DEFAULT 0, win_count INTEGER NOT NULL DEFAULT 0,
                win_rate REAL NOT NULL DEFAULT 0, total_pnl REAL NOT NULL DEFAULT 0,
                UNIQUE(date, account_id)
            );
            INSERT INTO daily_reports
                (date, account_id, start_asset, end_asset, daily_return,
                 trades_count, win_count, win_rate, total_pnl)
                VALUES ('2024-01-01', 'acc_1', 1000000, 1010000, 0.01, 3, 2, 0.6667, 10000);
        """)
        raw.commit()
        raw.close()

        # 2) 用新版 Database 打开（触发 ALTER TABLE），旧数据必须保留
        db = Database(db_path=db_path)
        rows = db.get_daily_reports(account_id="acc_1")
        assert len(rows) == 1
        assert rows[0]["date"] == "2024-01-01"
        assert rows[0]["total_pnl"] == pytest.approx(10000)
        # 新列存在且有默认值
        assert rows[0]["max_positions"] == 0
        assert rows[0]["risk_events_count"] == 0
        assert rows[0]["jev_decisions_count"] == 0
        assert rows[0]["jev_executed_count"] == 0
        assert rows[0]["extra_json"] == "{}"

        # 3) 再次初始化（列已存在）不应报错 —— 幂等
        db2 = Database(db_path=db_path)
        assert db2.get_daily_reports(account_id="acc_1")[0]["date"] == "2024-01-01"
