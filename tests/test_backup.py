"""备份 / 恢复管理器单元测试。

覆盖: 创建备份 / 带标签备份 / 列表 / 完整性校验 / 恢复 / 恢复前兜底备份 /
过期清理 / 损坏文件校验。全部使用临时目录，不触碰真实数据库。
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from persistence.backup import BackupManager
from persistence.database import Database


@pytest.fixture
def env(tmp_path: Path):
    """构造隔离的临时数据库 + 备份目录，并写入一条基础交易。"""
    db_path = str(tmp_path / "test_quant.db")
    backup_dir = str(tmp_path / "backups")
    db = Database(db_path=db_path)
    db.insert_trade(
        timestamp="2024-01-02 09:30:00",
        symbol="600519.SH", side="buy",
        price=1700.0, fill_price=1700.0,
        quantity=100, amount=170000.0,
    )
    mgr = BackupManager(db_path=db_path, backup_dir=backup_dir, keep_days=30)
    return {"db": db, "mgr": mgr, "db_path": db_path, "backup_dir": backup_dir}


# ---------------------------------------------------------------------------
# 创建 / 列表
# ---------------------------------------------------------------------------

class TestCreate:
    def test_create_backup(self, env):
        path = env["mgr"].backup()
        assert os.path.exists(path)
        assert Path(path).stat().st_size > 0
        assert Path(path).name.startswith("backup_")
        assert Path(path).suffix == ".db"

    def test_backup_with_tag(self, env):
        path = env["mgr"].backup(tag="pre_upgrade")
        name = Path(path).name
        assert name.startswith("backup_")
        assert name.endswith("_pre_upgrade.db")

    def test_list_backups(self, env):
        env["mgr"].backup(tag="first")
        time.sleep(1.0)  # 保证时间戳不同
        env["mgr"].backup(tag="second")
        items = env["mgr"].list_backups()
        assert len(items) == 2
        # 倒序：最新在前
        assert items[0]["tag"] == "second"
        assert items[1]["tag"] == "first"
        for it in items:
            assert {"filename", "path", "size_bytes", "created_at", "tag"} <= set(it)
            assert it["size_bytes"] > 0


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

class TestVerify:
    def test_verify_backup(self, env):
        path = env["mgr"].backup()
        result = env["mgr"].verify(path)
        assert result["valid"] is True
        assert result["error"] == ""
        # 5 张核心表都在，trades 至少有 fixture 写入的 1 条
        assert set(result["tables"].keys()) == {
            "trades", "jev_decisions", "account_snapshots",
            "daily_reports", "walkthroughs",
        }
        assert result["tables"]["trades"] >= 1

    def test_verify_invalid_file(self, env, tmp_path: Path):
        bad = tmp_path / "corrupt.db"
        bad.write_bytes(b"this is not a sqlite database")
        result = env["mgr"].verify(str(bad))
        assert result["valid"] is False
        assert result["error"] != ""


# ---------------------------------------------------------------------------
# 恢复
# ---------------------------------------------------------------------------

class TestRestore:
    def test_restore(self, env):
        db, mgr = env["db"], env["mgr"]
        backup = mgr.backup()
        before = db.count_table("trades")

        # 修改数据：再插入 5 笔
        for i in range(5):
            db.insert_trade(
                timestamp=f"2024-01-03 09:3{i}:00",
                symbol="000001.SZ", side="buy",
                price=10.0, fill_price=10.0,
                quantity=100, amount=1000.0,
            )
        db.close()  # 回收连接，确保读到恢复后的最新内容
        assert Database(env["db_path"]).count_table("trades") == before + 5

        # 执行恢复
        ok = mgr.restore(backup)
        assert ok is True

        # 恢复后数据应回到备份时的状态
        db2 = Database(env["db_path"])
        after = db2.count_table("trades")
        assert after == before

    def test_restore_creates_pre_backup(self, env):
        db, mgr = env["db"], env["mgr"]
        backup = mgr.backup()
        db.insert_trade(
            timestamp="2024-01-04 10:00:00",
            symbol="600000.SH", side="sell",
            price=8.0, fill_price=8.0,
            quantity=50, amount=400.0, realized_pnl=50.0,
        )
        db.close()

        mgr.restore(backup)
        names = [it["filename"] for it in mgr.list_backups()]
        # 原始备份 + 恢复前兜底备份
        assert any(n.startswith("pre_restore_") for n in names)
        assert len(names) >= 2


# ---------------------------------------------------------------------------
# 清理
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_cleanup(self, env):
        mgr = env["mgr"]
        # 新备份（应保留）
        keep_path = mgr.backup(tag="fresh")
        # 旧备份（伪造 mtime 为 40 天前，应被清理）
        old_path = mgr.backup(tag="old")
        old_time = time.time() - 40 * 24 * 3600
        os.utime(old_path, (old_time, old_time))

        removed = mgr.cleanup(keep_days=30)
        assert removed >= 1
        assert not Path(old_path).exists()
        assert Path(keep_path).exists()


# ---------------------------------------------------------------------------
# 自动备份触发判断
# ---------------------------------------------------------------------------

class TestAutoBackup:
    def test_should_auto_backup(self, env):
        mgr = env["mgr"]
        # 早上 10 点，未到 15:30
        morning = datetime(2024, 6, 3, 10, 0, 0)
        assert mgr.should_auto_backup(now=morning, auto_time="15:30") is False

        # 下午 16 点，到点且当天未备份 -> True
        evening = datetime(2024, 6, 3, 16, 0, 0)
        assert mgr.should_auto_backup(now=evening, auto_time="15:30") is True

        # 当天已备份过（伪造一个与 injected 同日时间戳的备份文件）-> 幂等返回 False
        (Path(env["backup_dir"]) / "backup_20240603_160500_manual.db").touch()
        assert mgr.should_auto_backup(now=evening, auto_time="15:30") is False
