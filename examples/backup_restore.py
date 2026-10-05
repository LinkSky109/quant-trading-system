#!/usr/bin/env python3
"""数据库备份 / 恢复完整流程示例。

在临时目录中演示：
    创建备份 -> 列出备份 -> 完整性校验 -> 模拟修改数据 -> 从备份恢复 -> 清理旧备份

全程不触碰生产数据库（data/quant_trading.db）。

运行方式:
    cd quant_trading_system
    python3 examples/backup_restore.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from persistence.backup import BackupManager  # noqa: E402
from persistence.database import Database  # noqa: E402


def main() -> None:
    # 使用临时目录，演示完自动删除，绝不影响真实数据
    workdir = Path(tempfile.mkdtemp(prefix="backup_demo_"))
    db_path = workdir / "demo.db"
    backup_dir = workdir / "backups"
    print(f"[演示目录] {workdir}")

    # 1) 准备一张库并写入一条交易
    db = Database(db_path=str(db_path))
    db.insert_trade(
        timestamp="2024-01-02 09:30:00",
        symbol="600519.SH", side="buy",
        price=1700.0, fill_price=1700.0,
        quantity=100, amount=170000.0,
    )
    print(f"[1] 初始化演示库，trades 记录数 = {db.count_table('trades')}")

    mgr = BackupManager(db_path=str(db_path), backup_dir=str(backup_dir), keep_days=30)

    # 2) 创建备份
    backup_path = mgr.backup(tag="demo")
    print(f"[2] 已创建备份: {Path(backup_path).name}")

    # 3) 列出备份
    print("[3] 当前备份列表:")
    for item in mgr.list_backups():
        print(f"      - {item['filename']:40s} {item['size_bytes']:>8d} bytes  "
              f"@ {item['created_at']}  tag={item['tag']}")

    # 4) 校验备份完整性
    v = mgr.verify(backup_path)
    print(f"[4] 校验结果 valid={v['valid']} 各表记录数={v['tables']}  error={v['error'] or '无'}")

    # 5) 模拟修改数据（再写 3 笔）
    for i in range(3):
        db.insert_trade(
            timestamp=f"2024-01-03 09:3{i}:00",
            symbol="000001.SZ", side="buy",
            price=10.0, fill_price=10.0,
            quantity=100, amount=1000.0,
        )
    db.close()
    print(f"[5] 模拟修改后，trades 记录数 = {Database(str(db_path)).count_table('trades')}")

    # 6) 从备份恢复（会自动先把当前库兜底备份为 pre_restore_*.db）
    ok = mgr.restore(backup_path)
    restored_count = Database(str(db_path)).count_table("trades")
    print(f"[6] 恢复成功={ok}，恢复后 trades 记录数 = {restored_count}（应回到 1）")

    # 7) 清理：删除超过保留天数的旧备份
    removed = mgr.cleanup(keep_days=30)
    print(f"[7] 清理过期备份，删除 {removed} 个；剩余 {len(mgr.list_backups())} 个")

    # 收尾：删除临时目录
    shutil.rmtree(workdir, ignore_errors=True)
    print("[完成] 演示结束，临时目录已清理，未影响任何真实数据。")


if __name__ == "__main__":
    main()
