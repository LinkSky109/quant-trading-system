"""监控与日志系统模块。

包含：
1. 统一日志配置
2. 交易记录器（每笔交易明细）
3. 每日绩效报告
4. 异常预警（连续亏损、回撤超标等）
"""
from __future__ import annotations

import json
import logging
import logging.handlers
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

def setup_logging(
    log_dir: str = "./logs",
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """配置全局日志系统。

    同时输出到控制台和文件（按大小轮转）。
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 清除已有 handler
    root_logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    # 文件
    fh = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "quant_system.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)


# ---------------------------------------------------------------------------
# 交易记录器
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """交易记录。"""
    date: str
    symbol: str
    action: str
    price: float
    quantity: int
    amount: float
    pnl: Optional[float]
    commission: float
    reason: str


class TradeLogger:
    """交易记录器，持久化到 JSONL。"""

    def __init__(self, log_path: str = "./logs/trades.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[TradeRecord] = []

    def record(self, trade: TradeRecord) -> None:
        self.records.append(trade)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(trade), ensure_ascii=False) + "\n")

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame([asdict(r) for r in self.records])

    def get_closed_trades(self) -> List[TradeRecord]:
        return [r for r in self.records if r.pnl is not None]


# ---------------------------------------------------------------------------
# 每日绩效报告
# ---------------------------------------------------------------------------

class DailyReporter:
    """每日绩效报告生成器。"""

    def __init__(self, report_dir: str = "./reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        date: pd.Timestamp,
        equity: float,
        daily_return: float,
        trades: List[Dict[str, Any]],
        positions: List[Dict[str, Any]],
        risk_summary: Dict[str, Any],
    ) -> str:
        """生成每日绩效报告文本。"""
        lines = [
            f"{'='*60}",
            f"每日绩效报告 - {date.strftime('%Y-%m-%d')}",
            f"{'='*60}",
            f"账户权益: {equity:,.2f}",
            f"当日收益率: {daily_return*100:.2f}%",
            f"当日交易笔数: {len(trades)}",
            f"当前持仓数: {len(positions)}",
            f"风控状态: {'暂停' if risk_summary.get('paused') else '正常'}",
            f"当前回撤: {risk_summary.get('current_drawdown', 0)*100:.2f}%",
            f"",
            f"--- 当日交易明细 ---",
        ]
        for t in trades:
            pnl_str = f", 盈亏: {t['pnl']:,.2f}" if t.get("pnl") is not None else ""
            lines.append(
                f"  {t['date']} {t['action'].upper()} {t['symbol']} "
                f"{t['quantity']}股 @ {t['price']:.2f}{pnl_str}"
            )

        report = "\n".join(lines)

        # 保存到文件
        report_file = self.report_dir / f"daily_{date.strftime('%Y%m%d')}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        return report


# ---------------------------------------------------------------------------
# 异常预警
# ---------------------------------------------------------------------------

class AlertMonitor:
    """异常预警监控器。"""

    def __init__(
        self,
        max_consecutive_losses: int = 3,
        drawdown_alert: float = 0.08,
        daily_loss_alert: float = 0.015,
    ):
        self.max_consecutive_losses = max_consecutive_losses
        self.drawdown_alert = drawdown_alert
        self.daily_loss_alert = daily_loss_alert
        self.alerts: List[str] = []

    def check(self, equity_series: pd.Series, trades: List[Dict[str, Any]]) -> List[str]:
        """检查异常条件，返回预警列表。"""
        new_alerts = []

        # 连续亏损
        closed = [t for t in trades if t.get("pnl") is not None]
        if len(closed) >= self.max_consecutive_losses:
            recent = closed[-self.max_consecutive_losses:]
            if all(t["pnl"] < 0 for t in recent):
                msg = f"连续 {self.max_consecutive_losses} 笔亏损，建议暂停交易检查策略"
                new_alerts.append(msg)
                logger.warning(msg)

        # 回撤预警
        if len(equity_series) > 1:
            peak = equity_series.cummax().iloc[-1]
            dd = (peak - equity_series.iloc[-1]) / peak
            if dd >= self.drawdown_alert:
                msg = f"回撤 {dd*100:.1f}% 接近暂停阈值，请注意风险"
                new_alerts.append(msg)
                logger.warning(msg)

        self.alerts.extend(new_alerts)
        return new_alerts
