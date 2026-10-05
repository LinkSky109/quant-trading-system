"""数据质量监控模块（P1-2）。

``DataQualityMonitor`` 在交易主循环/定时任务中被调用，对实时行情与日线数据
做基础质量校验，尽早发现数据通道异常、数据滞后、K线缺失、价格跳变与成交量异常。

设计原则:
- **绝不阻断**正常数据获取流程: 每一项检测都被独立 ``try/except`` 包裹，
  检测自身抛错只会记录日志，不会影响主交易链路。
- 告警通过传入的 ``AlertManager`` 发送（``AlertLevel.CRITICAL`` / ``WARNING``）；
  当 ``alert_manager`` 为 ``None`` 时，仅把异常写入内存历史，不崩溃。

检测项:
1. API 断连检测: 连续 N 次请求失败触发 CRITICAL（成功后重置计数）。
2. 数据滞后检测: 实时行情时间戳与当前时间差超过阈值触发 WARNING。
3. 缺失K线检测: 日线最后日期距最近交易日超过 2 个工作日触发 WARNING。
4. 价格异常跳变: 涨跌幅超过 warning 阈值标记为异常记录（不告警，可能真实涨跌停）；
   超过 critical 阈值触发 CRITICAL（疑似数据错误）。
5. 成交量异常: 成交量为 0 或为负触发 WARNING。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

try:  # AlertManager 是可选依赖，测试中可传 mock 或 None
    from .alert import AlertLevel
except ImportError:  # pragma: no cover - 防御性导入
    class AlertLevel:  # type: ignore
        CRITICAL = "CRITICAL"
        WARNING = "WARNING"
        INFO = "INFO"

logger = logging.getLogger(__name__)


# 告警类别常量
ALERT_API_DISCONNECT = "data_quality_api_disconnect"
ALERT_LAG = "data_quality_lag"
ALERT_MISSING_KLINES = "data_quality_missing_klines"
ALERT_PRICE_JUMP = "data_quality_price_jump"
ALERT_VOLUME_ANOMALY = "data_quality_volume_anomaly"


class DataQualityMonitor:
    """量化交易数据质量监控器。

    Args:
        alert_manager: 告警管理器实例，为 ``None`` 时仅记录异常历史不推送。
        config: 配置字典，可包含以下键:
            - ``lag_threshold_seconds``: 数据滞后阈值（秒），默认 300。
            - ``price_jump_warning``: 价格跳变异常标记阈值（比例），默认 0.10。
            - ``price_jump_critical``: 价格跳变严重阈值（比例），默认 0.20。
            - ``max_consecutive_failures``: 连续失败触发告警次数，默认 3。
            - ``max_history``: 异常历史保留条数，默认 200。
    """

    def __init__(self, alert_manager: Any = None, config: Optional[dict] = None):
        cfg = config or {}
        self.alert_manager = alert_manager

        self.lag_threshold_seconds: float = float(cfg.get("lag_threshold_seconds", 300))
        self.price_jump_warning: float = float(cfg.get("price_jump_warning", 0.10))
        self.price_jump_critical: float = float(cfg.get("price_jump_critical", 0.20))
        self.max_consecutive_failures: int = int(cfg.get("max_consecutive_failures", 3))
        # 兼容配置项 anomaly_history_limit
        self.max_history: int = int(
            cfg.get("max_history", cfg.get("anomaly_history_limit", 200))
        )

        # 每个数据源的连续失败计数
        self._consecutive_failures: Dict[str, int] = {}
        # 历史异常记录（最新在前的追加顺序，查询时倒序返回）
        self._anomaly_history: List[Dict[str, Any]] = []
        # 最近一次 check() 报告
        self._latest_report: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # API 断连检测
    # ------------------------------------------------------------------

    def record_fetch_result(
        self, source: str, success: bool, symbol: str = ""
    ) -> None:
        """记录一次数据获取结果，用于断连检测。

        成功时重置该数据源的连续失败计数；失败时累加，达到阈值立即触发
        CRITICAL 告警。计数在每次成功后重置。

        Args:
            source: 数据源名称，如 ``"quantdash"`` / ``"tencent"``。
            success: 本次获取是否成功。
            symbol: 关联标的（可选，仅用于告警文案）。
        """
        try:
            if success:
                self._consecutive_failures[source] = 0
                return

            count = self._consecutive_failures.get(source, 0) + 1
            self._consecutive_failures[source] = count

            if count >= self.max_consecutive_failures:
                self._send_alert(
                    AlertLevel.CRITICAL,
                    ALERT_API_DISCONNECT,
                    f"数据源 {source} 连续 {count} 次请求失败，疑似断连",
                    symbol=symbol or None,
                    current_value=float(count),
                    threshold=float(self.max_consecutive_failures),
                    extra={"source": source, "consecutive_failures": count},
                )
                self._record_anomaly(
                    type_=ALERT_API_DISCONNECT,
                    symbol=symbol or None,
                    message=f"数据源 {source} 连续 {count} 次请求失败",
                    value=float(count),
                    threshold=float(self.max_consecutive_failures),
                )
        except Exception as e:  # 绝不阻断主流程
            logger.warning("record_fetch_result 处理异常（已忽略）: %s", e)

    # ------------------------------------------------------------------
    # 主检测入口
    # ------------------------------------------------------------------

    def check(self, quotes: dict, klines: Optional[dict] = None) -> dict:
        """执行全部检测项，返回质量报告。

        所有检测均独立包裹 ``try/except``，任一检测出错只记录日志，不抛出。

        Args:
            quotes: 实时行情字典 ``{symbol: {price/last_price, prev_close,
                volume, timestamp, ...}}``。
            klines: 日线数据字典 ``{symbol: pd.DataFrame}``，索引为日期。

        Returns:
            质量报告 dict，结构见模块 docstring 中的返回示例。
        """
        checks: Dict[str, Dict[str, Any]] = {}
        anomalies: List[Dict[str, Any]] = []

        checks["api_connection"] = self._check_api_connection()
        checks["data_lag"] = self._check_data_lag(quotes, anomalies)
        checks["missing_klines"] = self._check_missing_klines(klines or {}, anomalies)
        checks["price_jump"] = self._check_price_jump(quotes or {}, anomalies)
        checks["volume_anomaly"] = self._check_volume(quotes or {}, anomalies)

        # 汇总总体状态
        statuses = [c["status"] for c in checks.values()]
        if "critical" in statuses:
            overall = "critical"
        elif "warning" in statuses:
            overall = "warning"
        else:
            overall = "healthy"

        report = {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
            "overall_status": overall,
            "checks": checks,
            "anomalies": anomalies,
        }
        # 持久化本次检测到的异常到历史（带时间戳），并裁剪
        for a in anomalies:
            self._anomaly_history.append({**a, "timestamp": time.time()})
        if len(self._anomaly_history) > self.max_history:
            self._anomaly_history = self._anomaly_history[-self.max_history:]

        self._latest_report = report
        return report

    # ------------------------------------------------------------------
    # 各检测项
    # ------------------------------------------------------------------

    def _check_api_connection(self) -> Dict[str, Any]:
        """API 断连检测（基于累计的连续失败计数）。"""
        try:
            details: Dict[str, Any] = {
                "consecutive_failures": dict(self._consecutive_failures),
                "max_consecutive_failures": self.max_consecutive_failures,
            }
            over = {
                src: cnt
                for src, cnt in self._consecutive_failures.items()
                if cnt >= self.max_consecutive_failures
            }
            if over:
                details["failed_sources"] = over
                return {"status": "critical", "details": details}
            if any(cnt > 0 for cnt in self._consecutive_failures.values()):
                details["note"] = "存在失败但未达连续阈值"
                return {"status": "warning", "details": details}
            return {"status": "ok", "details": details}
        except Exception as e:
            logger.warning("api_connection 检测异常（已忽略）: %s", e)
            return {"status": "ok", "details": {"error": str(e)}}

    def _check_data_lag(
        self, quotes: dict, anomalies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """数据滞后检测: 行情时间戳过旧触发 WARNING。"""
        try:
            now = time.time()
            ages: Dict[str, float] = {}
            stale: Dict[str, float] = {}
            for symbol, q in (quotes or {}).items():
                ts = self._parse_timestamp(q.get("timestamp")) if q else None
                if ts is None:
                    continue
                age = now - ts
                ages[symbol] = round(age, 1)
                if age > self.lag_threshold_seconds:
                    stale[symbol] = round(age, 1)

            details: Dict[str, Any] = {
                "lag_threshold_seconds": self.lag_threshold_seconds,
                "ages_seconds": ages,
            }
            if stale:
                details["stale_symbols"] = stale
                for symbol, age in stale.items():
                    self._send_alert(
                        AlertLevel.WARNING,
                        ALERT_LAG,
                        f"行情数据滞后: {symbol} 已滞后 {age:.0f}s "
                        f"(阈值 {self.lag_threshold_seconds:.0f}s)",
                        symbol=symbol,
                        current_value=age,
                        threshold=self.lag_threshold_seconds,
                    )
                    anomalies.append({
                        "type": ALERT_LAG,
                        "symbol": symbol,
                        "message": f"行情滞后 {age:.0f}s",
                        "value": age,
                        "threshold": self.lag_threshold_seconds,
                    })
                return {"status": "warning", "details": details}
            return {"status": "ok", "details": details}
        except Exception as e:
            logger.warning("data_lag 检测异常（已忽略）: %s", e)
            return {"status": "ok", "details": {"error": str(e)}}

    def _check_missing_klines(
        self, klines: dict, anomalies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """缺失K线检测: 日线最后日期距最近交易日超过 2 个工作日触发 WARNING。"""
        try:
            today = pd.Timestamp.now().normalize()
            # 最近若干个工作日（含今天），用于判断"最近交易日"
            recent_business_days = pd.bdate_range(end=today, periods=10)
            # 最近一个已结束的交易日（取 <= today 的最后一个工作日）
            last_bday = recent_business_days[-1]

            details: Dict[str, Any] = {
                "last_business_day": last_bday.strftime("%Y-%m-%d"),
                "gaps_business_days": {},
            }
            missing: Dict[str, Any] = {}

            for symbol, df in klines.items():
                if df is None or len(df) == 0:
                    continue
                try:
                    last_date = pd.Timestamp(df.index[-1]).normalize()
                except Exception:
                    continue

                # 最后日期到最近交易日之间的工作日数
                bdays = pd.bdate_range(start=last_date, end=last_bday)
                gap = len(bdays) - 1  # 0 表示当天
                details["gaps_business_days"][symbol] = int(gap)

                # 缺失超过 2 个工作日（即 last_date 落后最近交易日 >= 3 天的工作日跨度）
                if gap > 2:
                    missing[symbol] = {
                        "last_kline_date": last_date.strftime("%Y-%m-%d"),
                        "gap_business_days": int(gap),
                    }

            if missing:
                details["missing_symbols"] = missing
                for symbol, info in missing.items():
                    self._send_alert(
                        AlertLevel.WARNING,
                        ALERT_MISSING_KLINES,
                        f"日线数据缺失: {symbol} 最后K线日期 "
                        f"{info['last_kline_date']}，落后 {info['gap_business_days']} 个工作日",
                        symbol=symbol,
                        current_value=float(info["gap_business_days"]),
                        threshold=2.0,
                    )
                    anomalies.append({
                        "type": ALERT_MISSING_KLINES,
                        "symbol": symbol,
                        "message": (
                            f"日线最后日期 {info['last_kline_date']}，"
                            f"落后 {info['gap_business_days']} 个工作日"
                        ),
                        "value": float(info["gap_business_days"]),
                        "threshold": 2.0,
                    })
                return {"status": "warning", "details": details}
            return {"status": "ok", "details": details}
        except Exception as e:
            logger.warning("missing_klines 检测异常（已忽略）: %s", e)
            return {"status": "ok", "details": {"error": str(e)}}

    def _check_price_jump(
        self, quotes: dict, anomalies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """价格跳变检测: >critical 触发 CRITICAL；>warning 仅记录异常。"""
        try:
            changes: Dict[str, float] = {}
            critical: Dict[str, float] = {}
            flagged: Dict[str, float] = {}

            for symbol, q in (quotes or {}).items():
                if not q:
                    continue
                price = self._extract_price(q)
                prev_close = q.get("prev_close")
                if price is None or not prev_close:
                    continue
                try:
                    change = abs((float(price) - float(prev_close)) / float(prev_close))
                except (ZeroDivisionError, ValueError, TypeError):
                    continue
                changes[symbol] = round(change, 4)

                if change > self.price_jump_critical:
                    critical[symbol] = round(change, 4)
                elif change > self.price_jump_warning:
                    flagged[symbol] = round(change, 4)

            details: Dict[str, Any] = {
                "changes": changes,
                "price_jump_warning": self.price_jump_warning,
                "price_jump_critical": self.price_jump_critical,
            }

            # >warning 一律记录为异常（含 critical）
            for symbol, ch in {**flagged, **critical}.items():
                anomalies.append({
                    "type": ALERT_PRICE_JUMP,
                    "symbol": symbol,
                    "message": f"价格跳变 {ch*100:.2f}%",
                    "value": ch,
                    "threshold": self.price_jump_warning,
                    "critical": symbol in critical,
                })

            if critical:
                details["critical_symbols"] = critical
                for symbol, ch in critical.items():
                    self._send_alert(
                        AlertLevel.CRITICAL,
                        ALERT_PRICE_JUMP,
                        f"价格异常跳变(疑似数据错误): {symbol} 涨跌幅 {ch*100:.2f}% "
                        f"> 阈值 {self.price_jump_critical*100:.0f}%",
                        symbol=symbol,
                        current_value=ch,
                        threshold=self.price_jump_critical,
                    )
                return {"status": "critical", "details": details}

            if flagged:
                details["flagged_symbols"] = flagged
                # 仅标记异常，不发送告警（可能为真实涨跌停）
                return {"status": "warning", "details": details}

            return {"status": "ok", "details": details}
        except Exception as e:
            logger.warning("price_jump 检测异常（已忽略）: %s", e)
            return {"status": "ok", "details": {"error": str(e)}}

    def _check_volume(
        self, quotes: dict, anomalies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """成交量异常检测: volume <= 0 触发 WARNING。"""
        try:
            bad: Dict[str, Any] = {}
            for symbol, q in (quotes or {}).items():
                if not q or "volume" not in q:
                    continue
                try:
                    vol = float(q.get("volume"))
                except (TypeError, ValueError):
                    continue
                if vol <= 0:
                    bad[symbol] = vol

            details: Dict[str, Any] = {"bad_symbols": bad}
            if bad:
                for symbol, vol in bad.items():
                    reason = "成交量为0" if vol == 0 else f"成交量为负({vol})"
                    self._send_alert(
                        AlertLevel.WARNING,
                        ALERT_VOLUME_ANOMALY,
                        f"成交量异常: {symbol} {reason}",
                        symbol=symbol,
                        current_value=vol,
                        threshold=0.0,
                    )
                    anomalies.append({
                        "type": ALERT_VOLUME_ANOMALY,
                        "symbol": symbol,
                        "message": reason,
                        "value": vol,
                        "threshold": 0.0,
                    })
                return {"status": "warning", "details": details}
            return {"status": "ok", "details": details}
        except Exception as e:
            logger.warning("volume_anomaly 检测异常（已忽略）: %s", e)
            return {"status": "ok", "details": {"error": str(e)}}

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get_latest_report(self) -> Optional[dict]:
        """返回最近一次质量报告，未执行过 check() 时返回 None。"""
        return self._latest_report

    def get_anomaly_history(self, limit: int = 50) -> list:
        """返回历史异常记录（最新在前）。

        Args:
            limit: 返回条数上限。

        Returns:
            异常记录列表。
        """
        return list(reversed(self._anomaly_history[-limit:]))

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_price(q: dict) -> Optional[float]:
        """从行情字典中提取价格，兼容 ``price`` 与 ``last_price`` 字段。"""
        for key in ("price", "last_price", "close"):
            val = q.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _parse_timestamp(ts: Any) -> Optional[float]:
        """兼容解析 ISO 字符串 / epoch（秒或毫秒）/ pandas Timestamp。

        Returns:
            Unix 时间戳（秒，本地时区基准），无法解析时返回 None。
        """
        if ts is None:
            return None
        try:
            if isinstance(ts, (int, float)):
                v = float(ts)
                # 毫秒级 epoch
                if v > 1e12:
                    v = v / 1000.0
                return v
            if isinstance(ts, pd.Timestamp):
                return ts.timestamp()
            s = str(ts)
            # 兼容 "Z" 结尾
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                #  naive 时间按本地时区解释
                return dt.timestamp()
            return dt.timestamp()
        except Exception:
            return None

    def _send_alert(
        self,
        level: Any,
        category: str,
        message: str,
        symbol: Optional[str] = None,
        current_value: Optional[float] = None,
        threshold: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """通过 alert_manager 发送告警；manager 为 None 时静默跳过。"""
        if self.alert_manager is None:
            return
        try:
            self.alert_manager.alert(
                level,
                category,
                message,
                symbol=symbol,
                current_value=current_value,
                threshold=threshold,
                extra=extra,
            )
        except Exception as e:
            logger.warning("发送数据质量告警失败（已忽略）: %s", e)

    def _record_anomaly(
        self,
        type_: str,
        symbol: Optional[str],
        message: str,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """追加一条异常记录到历史，并按 max_history 裁剪。"""
        try:
            self._anomaly_history.append({
                "type": type_,
                "symbol": symbol,
                "message": message,
                "value": value,
                "threshold": threshold,
                "timestamp": time.time(),
            })
            if len(self._anomaly_history) > self.max_history:
                self._anomaly_history = self._anomaly_history[-self.max_history:]
        except Exception as e:  # pragma: no cover
            logger.warning("记录异常历史失败（已忽略）: %s", e)
