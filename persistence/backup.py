"""SQLite 数据库在线备份 / 恢复 / 清理。

基于 sqlite3 官方在线备份 API（``Connection.backup()``）实现，
不需要停服务、不需要锁库，可在交易引擎运行时执行：

  - :meth:`BackupManager.backup`        在线生成带时间戳的一致性快照
  - :meth:`BackupManager.restore`       恢复前自动备份当前库，再把备份内容写回线上库
  - :meth:`BackupManager.verify`        校验备份文件能否打开、5 张核心表是否齐全、记录数
  - :meth:`BackupManager.cleanup`       按保留天数清理过期备份
  - :meth:`BackupManager.should_auto_backup`  定时任务的幂等触发判断

备份文件命名：``backup_YYYYMMDD_HHMMSS[_tag].db``，默认存放 ``data/backups/``。
"""
from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from persistence.database import _DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

# 备份文件名中的时间戳 + 可选标签： backup_20240101_120000[_tag].db
_BACKUP_NAME_RE = re.compile(
    r"^(?P<prefix>[a-z_]+?)_(?P<date>\d{8})_(?P<time>\d{6})(?:_(?P<tag>.+))?\.db$"
)

# 核心业务表（用于 verify 完整性校验）
CORE_TABLES = (
    "trades",
    "jev_decisions",
    "account_snapshots",
    "daily_reports",
    "walkthroughs",
)


class BackupManager:
    """交易数据库在线备份与恢复管理器。

    线程安全：每次操作临时打开短连接，不复用长连接，避免与业务线程的
    thread-local 连接互相干扰。
    """

    def __init__(
        self,
        db_path: str = "",
        backup_dir: str = "",
        keep_days: int = 30,
    ) -> None:
        """初始化备份管理器。

        Args:
            db_path: 源 SQLite 数据库路径，为空时使用
                :mod:`persistence.database` 的默认路径（``data/quant_trading.db``）。
            backup_dir: 备份存放目录，为空时使用 ``data/backups/``。
            keep_days: 默认备份保留天数，供 cleanup / auto-backup 使用。
        """
        self.db_path = str(db_path) if db_path else _DEFAULT_DB_PATH
        if backup_dir:
            self.backup_dir = str(backup_dir)
        else:
            self.backup_dir = str(
                Path(self.db_path).resolve().parent / "backups"
            )
        self.keep_days = keep_days
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "BackupManager 初始化: db=%s, dir=%s, keep_days=%s",
            self.db_path, self.backup_dir, keep_days,
        )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_name(filename: str) -> Optional[Dict[str, Any]]:
        """从备份文件名解析前缀/标签/时间戳，无法识别返回 None。"""
        m = _BACKUP_NAME_RE.match(filename)
        if not m:
            return None
        try:
            created_at = datetime.strptime(
                m.group("date") + m.group("time"), "%Y%m%d%H%M%S"
            )
        except ValueError:
            return None
        tag = m.group("tag") or m.group("prefix")
        return {
            "prefix": m.group("prefix"),
            "tag": tag,
            "created_at": created_at,
        }

    def _connect_source(self) -> sqlite3.Connection:
        """打开源库只读连接（在线备份的源）。"""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=30.0)
        return conn

    # ------------------------------------------------------------------
    # 备份
    # ------------------------------------------------------------------

    def backup(self, tag: str = "") -> str:
        """创建带时间戳的一致性备份文件。

        使用 sqlite3 在线备份 API 逐页拷贝，源库可继续读写，无需停服。

        Args:
            tag: 可选标签（如 ``manual`` / ``pre_upgrade``），会附加到文件名。

        Returns:
            备份文件的绝对路径。
        """
        return self._write_backup(prefix="backup", tag=tag)

    def _write_backup(self, prefix: str, tag: str = "") -> str:
        """执行一次在线快照并落盘，返回文件路径。

        Args:
            prefix: 文件名前缀（``backup`` / ``pre_restore`` 等）。
            tag: 可选附加标签。
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_tag = re.sub(r"[^0-9A-Za-z_\-]", "", tag)
        name = f"{prefix}_{ts}" + (f"_{safe_tag}" if safe_tag else "") + ".db"
        dest_path = Path(self.backup_dir) / name

        src = self._connect_source()
        try:
            dst = sqlite3.connect(str(dest_path), timeout=30.0)
            try:
                # 在线备份：源(只读) -> 目标新文件
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()

        logger.info("数据库备份完成: %s", dest_path)
        return str(dest_path.resolve())

    # ------------------------------------------------------------------
    # 列表
    # ------------------------------------------------------------------

    def list_backups(self) -> List[Dict[str, Any]]:
        """列出所有备份文件，按创建时间倒序（最新在前）。

        Returns:
            每项含 ``filename`` / ``path`` / ``size_bytes`` / ``created_at``
            （ISO 字符串）/ ``tag``。
        """
        result: List[Dict[str, Any]] = []
        for p in Path(self.backup_dir).glob("*.db"):
            info = self._parse_name(p.name)
            if info is None:
                continue
            result.append({
                "filename": p.name,
                "path": str(p.resolve()),
                "size_bytes": p.stat().st_size,
                "created_at": info["created_at"].isoformat(),
                "tag": info["tag"],
            })
        result.sort(key=lambda x: x["created_at"], reverse=True)
        return result

    # ------------------------------------------------------------------
    # 恢复
    # ------------------------------------------------------------------

    def restore(self, backup_file: str) -> bool:
        """从备份文件恢复数据库。

        恢复流程：
          1. 先把当前线上库在线备份为 ``pre_restore_<时间戳>.db``（兜底，可回滚）；
          2. 通过 sqlite3 在线备份 API 把备份文件内容逐页写回线上库，
             覆盖现有数据，无需手动停服/删文件。

        Args:
            backup_file: 备份文件路径（可传相对 backup_dir 的文件名或绝对路径）。

        Returns:
            是否成功。
        """
        backup_path = Path(backup_file)
        if not backup_path.is_absolute():
            backup_path = Path(self.backup_dir) / backup_file
        backup_path = backup_path.resolve()
        if not backup_path.exists():
            logger.error("恢复失败，备份文件不存在: %s", backup_path)
            return False

        # 1) 恢复前兜底备份当前数据
        try:
            pre = self._write_backup(prefix="pre_restore")
            logger.info("恢复前已自动备份当前库: %s", pre)
        except Exception:  # noqa: BLE001 - 兜底备份失败不阻断恢复，但记录日志
            logger.exception("恢复前自动备份失败，继续执行恢复")

        # 2) 在线恢复：备份文件(源) -> 线上库(目标)
        # sqlite 语义: X.backup(Y) 把 X 的库拷贝到 Y。
        dst = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            src = sqlite3.connect(f"file:{backup_path}?mode=ro", uri=True, timeout=30.0)
            try:
                src.backup(dst)
            finally:
                src.close()
        finally:
            dst.close()

        logger.info("数据库已从备份恢复: %s", backup_path)
        return True

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def cleanup(self, keep_days: Optional[int] = None) -> int:
        """删除超过保留天数的旧备份。

        Args:
            keep_days: 保留天数，为 None 时使用构造函数传入的值。

        Returns:
            实际删除的文件数。
        """
        days = self.keep_days if keep_days is None else keep_days
        cutoff = datetime.now() - timedelta(days=days)
        removed = 0
        for p in Path(self.backup_dir).glob("*.db"):
            if self._parse_name(p.name) is None:
                continue
            created = datetime.fromtimestamp(p.stat().st_mtime)
            if created < cutoff:
                try:
                    p.unlink()
                    removed += 1
                    logger.info("清理过期备份: %s", p.name)
                except OSError:
                    logger.exception("清理备份失败: %s", p)
        if removed:
            logger.info("共清理 %d 个过期备份（保留 %d 天）", removed, days)
        return removed

    # ------------------------------------------------------------------
    # 完整性校验
    # ------------------------------------------------------------------

    def verify(self, backup_file: str) -> Dict[str, Any]:
        """校验备份文件完整性。

        检查项：
          - 文件能否以 sqlite3 正常打开；
          - 5 张核心表是否全部存在；
          - 每张表的记录数统计。

        Args:
            backup_file: 备份文件路径或文件名。

        Returns:
            ``{"valid": bool, "tables": {表名: 记录数}, "error": str}``。
        """
        path = Path(backup_file)
        if not path.is_absolute():
            path = Path(self.backup_dir) / backup_file
        result: Dict[str, Any] = {"valid": False, "tables": {}, "error": ""}
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)
        except sqlite3.Error as e:
            result["error"] = f"无法打开备份文件: {e}"
            return result

        try:
            existing = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            missing = [t for t in CORE_TABLES if t not in existing]
            if missing:
                result["error"] = f"缺少核心表: {', '.join(missing)}"
                return result
            for t in CORE_TABLES:
                cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                result["tables"][t] = cnt
            result["valid"] = True
            return result
        except sqlite3.Error as e:
            result["error"] = f"读取备份内容失败: {e}"
            return result
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 自动备份触发判断
    # ------------------------------------------------------------------

    def should_auto_backup(
        self,
        now: Optional[datetime] = None,
        auto_time: str = "15:30",
    ) -> bool:
        """判断当前是否应当执行自动备份（供定时任务幂等调用）。

        规则：
          - 当前时间早于今日 ``auto_time`` -> False（还没到点）；
          - 已到点，但今天已经存在过任意备份 -> False（避免一天重复备份）；
          - 已到点且今天尚未备份 -> True。

        Args:
            now: 注入的当前时间（测试用），为空取真实时间。
            auto_time: 每日自动备份时刻，格式 ``"HH:MM"``。

        Returns:
            是否应触发一次备份。
        """
        now = now or datetime.now()
        try:
            hour, minute = (int(x) for x in auto_time.split(":"))
        except ValueError:
            logger.warning("auto_backup_time 格式非法: %s，按 15:30 处理", auto_time)
            hour, minute = 15, 30

        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now < target:
            return False

        # 今天是否已经备份过
        today = now.date()
        for item in self.list_backups():
            created = datetime.fromisoformat(item["created_at"])
            if created.date() == today:
                return False
        return True
