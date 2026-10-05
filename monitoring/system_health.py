"""系统健康监控模块。

聚合以下子系统的运行状态，统一计算 0-100 健康评分:

1. **服务状态**  — 量化服务(8766) 进程存活/uptime/内存/CPU，Jev 服务(8765) 连通性。
2. **API 性能**  — 通过 :class:`ApiMetricsCollector` 统计各端点请求量、平均/P95 延迟、错误率。
3. **Jev 服务**  — real / mock / disconnected 模式、模型加载状态、累计推理次数与平均延迟。
4. **数据库**    — SQLite 连接、各表记录数、库文件大小、WAL 模式。
5. **数据质量**  — 复用 :class:`~monitoring.data_quality.DataQualityMonitor` 的最近报告。
6. **交易引擎**  — RealtimeTrader 运行状态、当前持仓、今日交易数/盈亏、风控暂停状态。
7. **告警**      — :class:`~monitoring.alert.AlertManager` 各级别告警计数。

设计原则:
- **绝不阻断**: 每个 ``collect_*`` 方法独立 ``try/except``，单项失败只记录日志，
  不影响整体健康报告组装。
- **依赖可选**: 所有外部依赖（db / data_quality_monitor / alert_manager /
  realtime_trader）均可为 ``None``，对应子项标记为 ``"unknown"``。
- **psutil 可选**: 未安装时 memory/cpu 返回 ``None``，不抛异常。
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# psutil 是可选依赖：未安装时降级为 memory/cpu = None，不崩溃。
try:  # pragma: no cover - 取决于运行环境
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


# 健康评分权重（合计 100）
_W_SERVICE_QUANT = 15   # 量化服务正常
_W_SERVICE_JEV = 10     # Jev 服务正常
_W_API_ERROR = 10       # API 错误率
_W_API_LATENCY = 10     # API P95 延迟
_W_DB_CONN = 10         # 数据库连接
_W_DB_TABLES = 5        # 数据库表可查
_W_DATA_QUALITY = 15    # 数据质量
_W_TRADER_RUNNING = 8   # 交易引擎运行中
_W_TRADER_RISK = 7      # 风控正常
_W_ALERTS = 10          # 告警（每 CRITICAL 扣 5）


def _percentile(sorted_values: list, pct: float) -> Optional[float]:
    """计算已排序数组的百分位数（线性插值）。空数组返回 None。"""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    # 0.95 -> 95 分位：位置 = ceil(pct * n) - 1（经典 NIST 方法）
    import math

    idx = math.ceil(pct * len(sorted_values)) - 1
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])


class ApiMetricsCollector:
    """API 请求指标收集器（线程安全）。

    为每个端点保留最近 ``max_samples`` 条延迟样本（用 ``collections.deque``
    滚动覆盖），并原子计数错误数。统计计算时加锁快照，避免遍历期间被修改。

    Args:
        max_samples: 每个端点保留的最大延迟样本数，默认 1000。
    """

    def __init__(self, max_samples: int = 1000) -> None:
        self.max_samples = max_samples
        self._lock = threading.Lock()
        # endpoint -> {"latencies": deque[float], "errors": int, "total": int}
        self._data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"latencies": deque(maxlen=max_samples), "errors": 0, "total": 0}
        )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def record_request(
        self, endpoint: str, method: str, latency_ms: float, status_code: int
    ) -> None:
        """记录一次请求。

        Args:
            endpoint: 请求路径，如 ``"/api/health"``。
            method: HTTP 方法（仅用于日志/调试，统计按 endpoint 聚合）。
            latency_ms: 服务端耗时（毫秒）。
            status_code: HTTP 状态码，>=400 计为错误。
        """
        try:
            key = f"{method} {endpoint}"
            with self._lock:
                bucket = self._data[key]
                bucket["latencies"].append(float(latency_ms))
                bucket["total"] += 1
                if int(status_code) >= 400:
                    bucket["errors"] += 1
        except Exception as e:  # 指标收集绝不影响业务请求
            logger.warning("record_request 异常（已忽略）: %s", e)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """返回各端点统计。

        Returns:
            ``{endpoint: {count, avg_latency_ms, p95_latency_ms,
            error_count, error_rate}}``；无样本时各字段为 0 / None。
        """
        with self._lock:
            snapshot = {
                k: {"latencies": list(v["latencies"]),
                    "errors": v["errors"], "total": v["total"]}
                for k, v in self._data.items()
            }

        result: Dict[str, Dict[str, Any]] = {}
        for endpoint, info in snapshot.items():
            lats = sorted(info["latencies"])
            total = info["total"]
            errors = info["errors"]
            avg = sum(lats) / len(lats) if lats else None
            result[endpoint] = {
                "count": total,
                "avg_latency_ms": round(avg, 2) if avg is not None else None,
                "p95_latency_ms": (
                    round(_percentile(lats, 0.95), 2) if lats else None
                ),
                "error_count": errors,
                "error_rate": round(errors / total, 4) if total > 0 else 0.0,
            }
        return result

    def get_summary(self) -> Dict[str, Any]:
        """返回全局汇总统计。

        Returns:
            ``{total_requests, avg_latency_ms, p95_latency_ms, error_rate}``；
            无任何请求时 ``total_requests=0``，延迟字段为 None。
        """
        with self._lock:
            all_latencies = [
                lat for bucket in self._data.values() for lat in bucket["latencies"]
            ]
            total = sum(b["total"] for b in self._data.values())
            errors = sum(b["errors"] for b in self._data.values())

        lats = sorted(all_latencies)
        return {
            "total_requests": total,
            "avg_latency_ms": round(sum(lats) / len(lats), 2) if lats else None,
            "p95_latency_ms": round(_percentile(lats, 0.95), 2) if lats else None,
            "error_rate": round(errors / total, 4) if total > 0 else 0.0,
        }


class SystemHealthMonitor:
    """系统健康监控器。

    聚合服务状态、API 性能、Jev 服务、数据库、数据质量、交易引擎、告警，
    计算 0-100 健康评分。

    Args:
        db: 持久化数据库实例，为 ``None`` 时数据库项标记 unknown。
        data_quality_monitor: 数据质量监控器，为 ``None`` 时标记 unknown。
        alert_manager: 告警管理器，为 ``None`` 时返回空统计。
        realtime_trader: 实时交易引擎，为 ``None`` 时交易引擎项标记 unknown。
        api_metrics: 共享的 API 指标收集器；不传则内部自建一个。
        jev_base_url: Jev 服务地址，默认 ``http://localhost:8765``。
        quant_port: 量化服务端口，仅用于展示，默认 8766。
        start_time: 量化服务启动时间戳（秒），默认本监控器初始化时刻。
    """

    def __init__(
        self,
        db: Any = None,
        data_quality_monitor: Any = None,
        alert_manager: Any = None,
        realtime_trader: Any = None,
        api_metrics: Optional[ApiMetricsCollector] = None,
        jev_base_url: str = "http://localhost:8765",
        quant_port: int = 8766,
        start_time: Optional[float] = None,
    ) -> None:
        self.db = db
        self.data_quality_monitor = data_quality_monitor
        self.alert_manager = alert_manager
        self.realtime_trader = realtime_trader
        self.api_metrics = api_metrics or ApiMetricsCollector()
        self.jev_base_url = jev_base_url.rstrip("/")
        self.quant_port = quant_port
        self.start_time = start_time or time.time()

        self._latest_report: Optional[Dict[str, Any]] = None
        self._stop_flag = threading.Event()
        self._auto_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # 各子系统采集
    # ------------------------------------------------------------------

    def collect_service_status(self) -> Dict[str, Any]:
        """采集量化服务与 Jev 服务的进程/连通状态。"""
        report: Dict[str, Any] = {"quant": {}, "jev": {}}
        try:
            uptime = round(time.time() - self.start_time, 1)
            mem_mb: Optional[float] = None
            cpu_percent: Optional[float] = None
            if _HAS_PSUTIL:
                try:
                    proc = psutil.Process()  # type: ignore[union-attr]
                    mem_mb = round(proc.memory_info().rss / 1024 / 1024, 1)
                    cpu_percent = round(proc.cpu_percent(interval=0.05), 1)
                except Exception as e:  # pragma: no cover - 防御性
                    logger.warning("psutil 采集失败（已忽略）: %s", e)
            report["quant"] = {
                "status": "up",
                "port": self.quant_port,
                "uptime_seconds": uptime,
                "memory_mb": mem_mb,
                "cpu_percent": cpu_percent,
                "psutil_available": _HAS_PSUTIL,
            }
        except Exception as e:
            logger.warning("collect_service_status quant 失败: %s", e)
            report["quant"] = {"status": "unknown", "error": str(e)}

        # Jev 服务连通性
        try:
            data = self._http_get_jev_health()
            if data is None:
                report["jev"] = {
                    "status": "disconnected",
                    "port": self._port_of(self.jev_base_url),
                    "reachable": False,
                }
            else:
                model_loaded = bool(data.get("model_loaded"))
                report["jev"] = {
                    "status": "ok" if model_loaded else "loading",
                    "port": self._port_of(self.jev_base_url),
                    "reachable": True,
                    "model_loaded": model_loaded,
                    "model": data.get("model", "unknown"),
                }
        except Exception as e:
            logger.warning("collect_service_status jev 失败: %s", e)
            report["jev"] = {"status": "unknown", "error": str(e)}
        return report

    def collect_api_performance(self) -> Dict[str, Any]:
        """从 ApiMetricsCollector 获取各端点统计与全局汇总。"""
        try:
            return {
                "endpoints": self.api_metrics.get_stats(),
                "summary": self.api_metrics.get_summary(),
            }
        except Exception as e:
            logger.warning("collect_api_performance 失败: %s", e)
            return {"endpoints": {}, "summary": {}, "error": str(e)}

    def collect_jev_status(self) -> Dict[str, Any]:
        """采集 Jev 服务模式（real/mock/disconnected）与推理统计。"""
        report: Dict[str, Any] = {"mode": "unknown"}
        try:
            data = self._http_get_jev_health()
            if data is None:
                report["mode"] = "disconnected"
                report["reachable"] = False
            else:
                model_loaded = bool(data.get("model_loaded"))
                report["reachable"] = True
                report["model_loaded"] = model_loaded
                report["model"] = data.get("model", "unknown")
                report["mode"] = "real" if model_loaded else "mock"
        except Exception as e:
            logger.warning("collect_jev_status 健康探测失败: %s", e)
            report["mode"] = "unknown"
            report["error"] = str(e)

        # 推理次数 / 平均延迟 / 最近一次时间（从 jev_decisions 表）
        try:
            if self.db is not None:
                count = self.db.count_table("jev_decisions")
                report["inference_count"] = count
                conn = self.db._get_conn()
                row = conn.execute(
                    "SELECT AVG(latency_ms) AS avg_lat, MAX(timestamp) AS last_ts "
                    "FROM jev_decisions"
                ).fetchone()
                if row is not None:
                    avg_lat = row["avg_lat"] if hasattr(row, "keys") else row[0]
                    last_ts = row["last_ts"] if hasattr(row, "keys") else row[1]
                    report["avg_latency_ms"] = (
                        round(float(avg_lat), 2) if avg_lat is not None else None
                    )
                    report["last_decision_time"] = last_ts
            else:
                report["inference_count"] = None
        except Exception as e:
            logger.warning("collect_jev_status 推理统计失败: %s", e)
            report["inference_count"] = report.get("inference_count")
        return report

    def collect_database_status(self) -> Dict[str, Any]:
        """采集 SQLite 连接状态、表记录数、库大小与 WAL 模式。"""
        report: Dict[str, Any] = {"status": "unknown"}
        if self.db is None:
            return {"status": "unknown"}
        try:
            conn = self.db._get_conn()
            conn.execute("SELECT 1").fetchone()
            report["status"] = "connected"
        except Exception as e:
            logger.warning("collect_database_status 连接检查失败: %s", e)
            report["status"] = "disconnected"
            report["error"] = str(e)
            return report

        # 各表记录数（walkthroughs 等可选表不存在时标记 None）
        tables = ["trades", "jev_decisions", "account_snapshots",
                  "daily_reports", "walkthroughs"]
        counts: Dict[str, Optional[int]] = {}
        for t in tables:
            try:
                counts[t] = self.db.count_table(t)
            except Exception:
                counts[t] = None
        report["table_counts"] = counts

        # 库大小（MB）
        try:
            size_bytes = self.db.get_db_size()
            report["size_mb"] = round(size_bytes / 1024 / 1024, 2)
        except Exception as e:
            logger.warning("collect_database_status 大小查询失败: %s", e)
            report["size_mb"] = None

        # WAL 模式
        try:
            row = conn.execute("PRAGMA journal_mode").fetchone()
            report["journal_mode"] = row[0] if row is not None else None
        except Exception as e:
            logger.warning("collect_database_status WAL 查询失败: %s", e)
            report["journal_mode"] = None
        return report

    def collect_data_quality(self) -> Dict[str, Any]:
        """复用 DataQualityMonitor.get_latest_report()。"""
        if self.data_quality_monitor is None:
            return {"status": "unknown"}
        try:
            report = self.data_quality_monitor.get_latest_report()
            if report is None:
                return {"status": "unknown"}
            return report
        except Exception as e:
            logger.warning("collect_data_quality 失败: %s", e)
            return {"status": "unknown", "error": str(e)}

    def collect_trading_engine(self) -> Dict[str, Any]:
        """采集交易引擎运行状态、持仓、今日交易/盈亏与风控状态。"""
        trader = self.realtime_trader
        if trader is None:
            return {"status": "unknown"}
        report: Dict[str, Any] = {}
        try:
            running = bool(getattr(trader, "running", False))
            report["running"] = running
            status: Dict[str, Any] = {}
            if hasattr(trader, "get_status"):
                try:
                    status = trader.get_status() or {}
                except Exception:
                    status = {}
            positions = status.get("positions", []) or []
            report["position_count"] = len(positions) if isinstance(positions, list) else positions
            # 今日盈亏：优先 trader.today_pnl
            today_pnl = status.get("today_pnl")
            if today_pnl is None:
                today_pnl = getattr(trader, "today_pnl", None)
            report["today_pnl"] = today_pnl

            # 风控是否暂停
            risk_summary = status.get("risk_summary", {}) or {}
            paused = bool(
                risk_summary.get("paused")
                or risk_summary.get("risk_paused")
                or risk_summary.get("status") == "paused"
            )
            report["risk_paused"] = paused
            report["risk_summary"] = risk_summary
        except Exception as e:
            logger.warning("collect_trading_engine 状态采集失败: %s", e)
            report["status"] = "unknown"
            report["error"] = str(e)
            return report

        # 今日交易数（从 trades 表按本地日期过滤）
        try:
            if self.db is not None:
                report["today_trades"] = self.db.count_table(
                    "trades",
                    where="date(timestamp) = date('now', 'localtime')",
                )
            else:
                report["today_trades"] = None
        except Exception as e:
            logger.warning("collect_trading_engine 今日交易数失败: %s", e)
            report["today_trades"] = None
        return report

    def collect_alerts(self) -> Dict[str, Any]:
        """从 AlertManager.stats 获取各级别告警计数。"""
        if self.alert_manager is None:
            return {"status": "unknown", "stats": {}}
        try:
            return {"status": "ok", "stats": dict(self.alert_manager.stats)}
        except Exception as e:
            logger.warning("collect_alerts 失败: %s", e)
            return {"status": "unknown", "stats": {}, "error": str(e)}

    # ------------------------------------------------------------------
    # 健康评分
    # ------------------------------------------------------------------

    def calculate_health_score(self, report: Dict[str, Any]) -> int:
        """根据完整健康报告计算 0-100 健康评分。

        权重:
            - 服务状态 25（量化 15 + Jev 10）
            - API 性能 20（错误率 10 + P95 延迟 10）
            - 数据库 15（连接 10 + 表可查 5）
            - 数据质量 15
            - 交易引擎 15（运行 8 + 风控 7）
            - 告警 10（每 CRITICAL 扣 5）
        """
        score = 0.0

        # ---- 服务状态 25 ----
        svc = report.get("service_status", {}) or {}
        quant = svc.get("quant", {}) or {}
        if quant.get("status") == "up":
            score += _W_SERVICE_QUANT
        elif quant.get("status") == "unknown":
            score += _W_SERVICE_QUANT * 0.5  # 无法确认，给一半
        jev = svc.get("jev", {}) or {}
        if jev.get("status") == "ok":
            score += _W_SERVICE_JEV
        elif jev.get("status") == "unknown":
            score += _W_SERVICE_JEV * 0.5
        # disconnected / loading -> 0

        # ---- API 性能 20 ----
        api = report.get("api_performance", {}) or {}
        summary = api.get("summary", {}) or {}
        if summary.get("total_requests", 0) == 0:
            # 暂无流量：不扣分（无错误可观察）
            score += _W_API_ERROR + _W_API_LATENCY
        else:
            err_rate = float(summary.get("error_rate") or 0.0)
            # <1% 满分；>5% 零分；线性插值
            score += self._linear_score(
                err_rate, low=0.01, high=0.05,
                full=_W_API_ERROR, invert=True,
            )
            p95 = summary.get("p95_latency_ms")
            if p95 is not None:
                score += self._linear_score(
                    float(p95), low=500.0, high=2000.0,
                    full=_W_API_LATENCY, invert=True,
                )
            else:
                score += _W_API_LATENCY * 0.5

        # ---- 数据库 15 ----
        db_st = report.get("database", {}) or {}
        if db_st.get("status") == "connected":
            score += _W_DB_CONN
            counts = db_st.get("table_counts", {}) or {}
            queryable = sum(1 for v in counts.values() if v is not None)
            if queryable >= 4:
                score += _W_DB_TABLES
            elif queryable > 0:
                score += _W_DB_TABLES * (queryable / 5.0)
        elif db_st.get("status") == "unknown":
            score += (_W_DB_CONN + _W_DB_TABLES) * 0.5

        # ---- 数据质量 15 ----
        dq = report.get("data_quality", {}) or {}
        dq_status = dq.get("overall_status")
        if dq_status == "healthy" or (
            dq_status is None and dq.get("status") == "ok"
        ):
            score += _W_DATA_QUALITY
        elif dq_status == "warning":
            score += _W_DATA_QUALITY * 0.5
        elif dq_status == "unknown" or dq.get("status") == "unknown":
            score += _W_DATA_QUALITY * 0.5
        # critical -> 0

        # ---- 交易引擎 15 ----
        te = report.get("trading_engine", {}) or {}
        if te.get("status") == "unknown" and "running" not in te:
            score += (_W_TRADER_RUNNING + _W_TRADER_RISK) * 0.5
        else:
            if te.get("running"):
                score += _W_TRADER_RUNNING
            if not te.get("risk_paused"):
                score += _W_TRADER_RISK

        # ---- 告警 10 ----
        alerts = report.get("alerts", {}) or {}
        stats = alerts.get("stats", {}) or {}
        score += _W_ALERTS
        critical = int(stats.get("critical", 0) or 0)
        score -= 5 * critical
        score = max(score, 0.0)

        return int(round(score))

    @staticmethod
    def _linear_score(
        value: float, low: float, high: float, full: float, invert: bool = True
    ) -> float:
        """线性插值评分。

        invert=True: value <= low 得满分，value >= high 得 0 分。
        invert=False: value >= low 得满分，value <= high 得 0 分。
        """
        if invert:
            if value <= low:
                return full
            if value >= high:
                return 0.0
            return full * (high - value) / (high - low)
        else:
            if value >= low:
                return full
            if value <= high:
                return 0.0
            return full * (value - high) / (low - high)

    # ------------------------------------------------------------------
    # 汇总采集
    # ------------------------------------------------------------------

    def collect_all(self) -> Dict[str, Any]:
        """采集全部子系统报告并计算健康评分。

        每个子项独立 try/except，单项失败不影响整体。
        """
        report: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        }
        collectors = {
            "service_status": self.collect_service_status,
            "api_performance": self.collect_api_performance,
            "jev_status": self.collect_jev_status,
            "database": self.collect_database_status,
            "data_quality": self.collect_data_quality,
            "trading_engine": self.collect_trading_engine,
            "alerts": self.collect_alerts,
        }
        for name, fn in collectors.items():
            try:
                report[name] = fn()
            except Exception as e:  # 终极兜底，绝不崩溃
                logger.warning("collect_all: %s 采集失败: %s", name, e)
                report[name] = {"status": "unknown", "error": str(e)}

        score = self.calculate_health_score(report)
        report["health_score"] = score
        if score >= 80:
            level = "healthy"
        elif score >= 60:
            level = "warning"
        else:
            level = "critical"
        report["health_level"] = level
        self._latest_report = report
        return report

    # ------------------------------------------------------------------
    # 后台自动采集
    # ------------------------------------------------------------------

    def start_auto_collect(self, interval: int = 30) -> None:
        """启动后台线程，每 ``interval`` 秒自动采集一次。"""
        if self._auto_thread is not None and self._auto_thread.is_alive():
            return
        self._stop_flag = threading.Event()

        def _loop() -> None:
            while not self._stop_flag.is_set():
                try:
                    self.collect_all()
                except Exception:  # pragma: no cover
                    logger.exception("自动采集异常（已忽略）")
                # 用 Event.wait 以便 stop 时立即唤醒
                self._stop_flag.wait(timeout=interval)

        self._auto_thread = threading.Thread(
            target=_loop, daemon=True, name="system-health-auto-collect"
        )
        self._auto_thread.start()
        logger.info("系统健康自动采集已启动（间隔 %ss）", interval)

    def stop_auto_collect(self) -> None:
        """停止后台自动采集线程。"""
        self._stop_flag.set()
        if self._auto_thread is not None:
            self._auto_thread.join(timeout=2.0)
            self._auto_thread = None

    def get_latest_report(self) -> Optional[Dict[str, Any]]:
        """返回最近一次自动/手动采集的报告，从未采集时为 None。"""
        return self._latest_report

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _http_get_jev_health(self) -> Optional[Dict[str, Any]]:
        """GET Jev 服务 /api/health，失败返回 None（不抛异常）。"""
        try:
            import requests

            r = requests.get(f"{self.jev_base_url}/api/health", timeout=2.5)
            if r.status_code == 200:
                return r.json()
            return None
        except Exception:
            return None

    @staticmethod
    def _port_of(base_url: str) -> Optional[int]:
        """从 base_url 提取端口（无显式端口时返回 None）。"""
        try:
            parsed = urlparse(base_url)
            return parsed.port
        except Exception:
            return None
