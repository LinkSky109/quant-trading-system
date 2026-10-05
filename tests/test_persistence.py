"""持久化层（SQLite）单元测试。

覆盖: 建表 / 交易记录CRUD / Jev决策审计 / 账户快照 / 每日报告upsert / 统计查询
使用临时数据库文件，测试后自动清理。
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from persistence.database import Database


@pytest.fixture
def db(tmp_path) -> Database:
    """每个测试使用独立的临时数据库。"""
    db_path = str(tmp_path / "test_quant.db")
    return Database(db_path=db_path)


# ---------------------------------------------------------------------------
# 建表与初始化
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_db_file(self, db):
        assert os.path.exists(db.db_path)

    def test_tables_exist(self, db):
        conn = db._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        assert "trades" in table_names
        assert "jev_decisions" in table_names
        assert "account_snapshots" in table_names
        assert "daily_reports" in table_names

    def test_wal_mode(self, db):
        conn = db._get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()
        assert mode[0] == "wal"


# ---------------------------------------------------------------------------
# 交易记录
# ---------------------------------------------------------------------------

class TestTrades:
    def test_insert_and_query(self, db):
        trade_id = db.insert_trade(
            timestamp="2024-01-15T10:30:00",
            symbol="600519.SH",
            side="buy",
            price=1000.0,
            fill_price=1001.0,
            quantity=100,
            amount=100100.0,
            commission=25.03,
            stamp_tax=0.0,
            realized_pnl=0.0,
            reason="买入成交",
            account_id="balanced",
            strategy="ma_cross",
            jev_confidence=0.75,
            name="贵州茅台",
        )
        assert trade_id > 0

        trades = db.get_trades(limit=10)
        assert len(trades) == 1
        assert trades[0]["symbol"] == "600519.SH"
        assert trades[0]["side"] == "buy"
        assert trades[0]["quantity"] == 100
        assert trades[0]["name"] == "贵州茅台"

    def test_query_by_symbol(self, db):
        db.insert_trade("2024-01-01", "600519.SH", "buy", 100, 100, 10, 1000)
        db.insert_trade("2024-01-02", "300750.SZ", "buy", 200, 200, 10, 2000)
        trades = db.get_trades(symbol="600519.SH")
        assert len(trades) == 1
        assert trades[0]["symbol"] == "600519.SH"

    def test_query_by_account(self, db):
        db.insert_trade("2024-01-01", "A", "buy", 100, 100, 10, 1000, account_id="acc1")
        db.insert_trade("2024-01-02", "B", "buy", 200, 200, 10, 2000, account_id="acc2")
        trades = db.get_trades(account_id="acc1")
        assert len(trades) == 1

    def test_query_limit(self, db):
        for i in range(10):
            db.insert_trade(f"2024-01-{i+1:02d}", "A", "buy", 100, 100, 10, 1000)
        trades = db.get_trades(limit=3)
        assert len(trades) == 3

    def test_query_descending_order(self, db):
        db.insert_trade("2024-01-01", "A", "buy", 100, 100, 10, 1000)
        db.insert_trade("2024-01-02", "B", "buy", 200, 200, 10, 2000)
        trades = db.get_trades(limit=10)
        # 最新的在前（id大的在前）
        assert trades[0]["symbol"] == "B"

    def test_empty_trades(self, db):
        assert db.get_trades() == []


# ---------------------------------------------------------------------------
# Jev 决策
# ---------------------------------------------------------------------------

class TestJevDecisions:
    def test_insert_and_query(self, db):
        market_state = {"price": 100.0, "rsi": 55.0, "macd_signal": 1}
        probabilities = {"buy": 0.65, "sell": 0.15, "hold": 0.20}

        decision_id = db.insert_jev_decision(
            timestamp="2024-01-15T10:30:00",
            symbol="600519.SH",
            strategy_signal="buy",
            strategy_confidence=0.8,
            market_state=market_state,
            probabilities=probabilities,
            final_action="buy",
            final_confidence=0.65,
            executed=True,
            reason="通过Jev过滤",
            mode="real",
            latency_ms=78.5,
        )
        assert decision_id > 0

        decisions = db.get_jev_decisions(limit=10)
        assert len(decisions) == 1
        d = decisions[0]
        assert d["symbol"] == "600519.SH"
        assert d["final_action"] == "buy"
        assert d["executed"] is True
        assert d["mode"] == "real"
        assert d["latency_ms"] == 78.5
        # JSON 字段自动反序列化
        assert d["market_state"]["rsi"] == 55.0
        assert d["probabilities"]["buy"] == 0.65

    def test_executed_false_stored_as_0(self, db):
        db.insert_jev_decision(
            "2024-01-01", "A", "buy", 0.5, {}, {},
            "hold", 0.4, False, "置信度不足", "mock", 0,
        )
        decisions = db.get_jev_decisions()
        assert decisions[0]["executed"] is False

    def test_query_by_symbol(self, db):
        db.insert_jev_decision("2024-01-01", "600519.SH", "buy", 0.8, {}, {}, "buy", 0.7, True)
        db.insert_jev_decision("2024-01-02", "300750.SZ", "sell", 0.7, {}, {}, "sell", 0.6, True)
        decisions = db.get_jev_decisions(symbol="600519.SH")
        assert len(decisions) == 1


# ---------------------------------------------------------------------------
# 账户快照
# ---------------------------------------------------------------------------

class TestAccountSnapshots:
    def test_insert_and_query(self, db):
        positions = [
            {"symbol": "600519.SH", "shares": 100, "avg_cost": 1001.0, "market_value": 100500.0},
        ]
        snap_id = db.insert_account_snapshot(
            timestamp="2024-01-15T10:30:00",
            account_id="balanced",
            total_asset=1_100_500.0,
            cash=999_500.0,
            position_value=100_500.0,
            daily_pnl=500.0,
            positions=positions,
        )
        assert snap_id > 0

        snapshots = db.get_account_snapshots(account_id="balanced", limit=10)
        assert len(snapshots) == 1
        s = snapshots[0]
        assert s["total_asset"] == 1_100_500.0
        assert s["cash"] == 999_500.0
        assert s["daily_pnl"] == 500.0
        assert len(s["positions"]) == 1
        assert s["positions"][0]["symbol"] == "600519.SH"

    def test_empty_positions(self, db):
        db.insert_account_snapshot(
            "2024-01-01", "acc1", 1_000_000, 1_000_000, 0, 0, [],
        )
        snapshots = db.get_account_snapshots("acc1")
        assert snapshots[0]["positions"] == []

    def test_query_descending(self, db):
        db.insert_account_snapshot("2024-01-01", "acc1", 1_000_000, 1_000_000, 0)
        db.insert_account_snapshot("2024-01-02", "acc1", 1_010_000, 1_010_000, 0)
        snapshots = db.get_account_snapshots("acc1")
        # get_account_snapshots 返回倒序（最新在前）
        assert snapshots[0]["total_asset"] == 1_010_000.0


# ---------------------------------------------------------------------------
# 每日报告
# ---------------------------------------------------------------------------

class TestDailyReports:
    def test_insert(self, db):
        db.upsert_daily_report(
            date="2024-01-15",
            account_id="balanced",
            start_asset=1_000_000,
            end_asset=1_025_000,
            trades_count=5,
            win_count=3,
            total_pnl=25_000,
        )
        reports = db.get_daily_reports()
        assert len(reports) == 1
        assert reports[0]["daily_return"] == pytest.approx(0.025)
        assert reports[0]["win_rate"] == pytest.approx(0.6)

    def test_upsert_updates_existing(self, db):
        db.upsert_daily_report("2024-01-15", "acc1", 1_000_000, 1_010_000, 2, 1, 10_000)
        # 同一天同一账户再次写入，应更新而非新增
        db.upsert_daily_report("2024-01-15", "acc1", 1_000_000, 1_050_000, 5, 3, 50_000)
        reports = db.get_daily_reports(account_id="acc1")
        assert len(reports) == 1
        assert reports[0]["end_asset"] == 1_050_000.0
        assert reports[0]["trades_count"] == 5

    def test_different_accounts_separate(self, db):
        db.upsert_daily_report("2024-01-15", "acc1", 1_000_000, 1_010_000)
        db.upsert_daily_report("2024-01-15", "acc2", 500_000, 510_000)
        reports = db.get_daily_reports()
        assert len(reports) == 2

    def test_zero_trades_win_rate_zero(self, db):
        db.upsert_daily_report("2024-01-15", "acc1", 1_000_000, 1_000_000, 0, 0, 0)
        reports = db.get_daily_reports()
        assert reports[0]["win_rate"] == 0.0

    def test_zero_start_asset_no_division_error(self, db):
        db.upsert_daily_report("2024-01-15", "acc1", 0, 0, 0, 0, 0)
        reports = db.get_daily_reports()
        assert reports[0]["daily_return"] == 0.0


# ---------------------------------------------------------------------------
# 统计查询
# ---------------------------------------------------------------------------

class TestTradeStats:
    def test_stats_with_trades(self, db):
        # 3笔卖出：2赢1亏
        db.insert_trade("2024-01-01", "A", "sell", 100, 100, 10, 1000, realized_pnl=500)
        db.insert_trade("2024-01-02", "B", "sell", 200, 200, 10, 2000, realized_pnl=-200)
        db.insert_trade("2024-01-03", "C", "sell", 300, 300, 10, 3000, realized_pnl=800)
        # 买入不计入统计
        db.insert_trade("2024-01-04", "D", "buy", 100, 100, 10, 1000, realized_pnl=0)

        stats = db.get_trade_stats()
        assert stats["total_trades"] == 3
        assert stats["total_pnl"] == 1100.0
        assert stats["win_count"] == 2
        assert stats["loss_count"] == 1
        assert stats["win_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_stats_empty(self, db):
        stats = db.get_trade_stats()
        assert stats["total_trades"] == 0
        assert stats["total_pnl"] == 0.0
        assert stats["win_rate"] == 0.0

    def test_stats_by_account(self, db):
        db.insert_trade("2024-01-01", "A", "sell", 100, 100, 10, 1000,
                        realized_pnl=500, account_id="acc1")
        db.insert_trade("2024-01-02", "B", "sell", 200, 200, 10, 2000,
                        realized_pnl=-200, account_id="acc2")
        stats1 = db.get_trade_stats(account_id="acc1")
        assert stats1["total_trades"] == 1
        assert stats1["total_pnl"] == 500.0


# ---------------------------------------------------------------------------
# 数据库大小
# ---------------------------------------------------------------------------

class TestDbSize:
    def test_db_size_positive_after_write(self, db):
        db.insert_trade("2024-01-01", "A", "buy", 100, 100, 10, 1000)
        assert db.get_db_size() > 0

    def test_db_size_zero_for_nonexistent(self, tmp_path):
        db = Database(db_path=str(tmp_path / "nonexistent.db"))
        # 建表后应该有大小
        assert db.get_db_size() > 0
