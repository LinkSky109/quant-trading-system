"""数据校验与对账模块（REQ-P2-09）。

提供多数据源价格交叉验证、K线数据异常自动检测、数据血缘留痕、异常自动修复
与数据质量评分能力。设计目标：

- **价格交叉验证**：同一标的、同一时点在不同数据源（如 QuantDash / 腾讯）的报价偏差比对，
  超阈值即判为不一致并生成异常记录。
- **异常自动检测**：对 K线 ``DataFrame`` 做价格跳变、成交量异常、缺失K线、OHLC 逻辑错误
  四类检测。
- **数据血缘**：每条/每批数据记录来源、抓取时间与校验状态（validated/flagged/corrected）。
- **自动修复**：异常数据从备用源（或外部传入的备用数据）重新获取并补全，全程留痕；
  备用源也缺数据时保持 ``open``，不假装成功。
- **质量评分**：完整率 / 准确率 / 及时率三项加权（默认 0.4 / 0.4 / 0.2）输出 0-100 分，
  并可生成每日质量报告。

清洗类动作（去重、正价格修正等）委托 :mod:`data.data_cleaner`，本模块聚焦
「校验 / 对账 / 血缘 / 评分」。

典型用法：

    >>> from data.validation import DataReconciler
    >>> rc = DataReconciler()
    >>> rc.reconcile_prices("600519.SH", {"quantdash": 1680.0, "tencent": 1688.4})
    {"status": "flagged", "deviation": 0.005, ...}
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# 复用数据清洗能力（不重复实现）
try:  # pragma: no cover - 导入失败仅影响 live 分支，不影响纯检测
    from data.data_cleaner import clean_klines
except Exception:  # noqa: BLE001
    clean_klines = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class AnomalyRecord:
    """数据异常记录。

    Attributes:
        id: 异常唯一标识（自动生成）。
        symbol: 标的代码。
        type: 异常类型（price_jump / volume_surge / volume_dry /
            missing_kline / ohlc_logic / price_mismatch）。
        severity: 严重度（high / medium / low）。
        date: 异常对应的 K线日期（ISO 字符串；批量/对账类可为空串）。
        detail: 异常描述。
        status: 状态，默认 ``open``，修复后为 ``fixed``。
        detected_at: 检测时间（ISO 字符串）。
        fix_history: 修复动作留痕列表（每次修复一条 dict）。
    """

    id: str
    symbol: str
    type: str
    severity: str = "medium"
    date: str = ""
    detail: str = ""
    status: str = "open"
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    fix_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "type": self.type,
            "severity": self.severity,
            "date": self.date,
            "detail": self.detail,
            "status": self.status,
            "detected_at": self.detected_at,
            "fix_history": self.fix_history,
        }


@dataclass
class LineageRecord:
    """数据血缘记录。

    Attributes:
        symbol: 标的代码。
        source: 数据来源（quantdash / tencent / 备用源 等）。
        fetched_at: 抓取时间（ISO 字符串）。
        status: 血缘状态 ``validated`` / ``flagged`` / ``corrected``。
        detail: 备注。
    """

    symbol: str
    source: str
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "validated"
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "symbol": self.symbol,
            "source": self.source,
            "fetched_at": self.fetched_at,
            "status": self.status,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# 主对账器
# ---------------------------------------------------------------------------


class DataReconciler:
    """多数据源交叉验证 + 异常检测 + 血缘 + 评分。

    Args:
        fetcher: 数据获取对象（需具备 ``get_realtime_quote`` / ``get_klines`` 方法）；
            为 ``None`` 时 live 分支（自动取数 / 自动修复取备用源）不可用，但纯检测、
            手工传入数据的路径仍可正常工作。
        db_path: SQLite 库路径；为 ``None`` 时不落盘（仅内存态）。
        price_tolerance: 价格交叉验证容忍偏差率，默认 0.005（0.5%）。
        jump_threshold: 价格跳变阈值（单日涨跌幅绝对值），非 A股默认 0.20。
        a_jump_threshold: A股价格跳变阈值，默认 0.10（主板 ±10% 停板）。
        volume_ma_window: 成交量均线窗口，默认 20。
        volume_surge_mult: 放量倍数（volume > MA*mult），默认 5.0。
        volume_dry_ratio: 地量比例（volume < MA*ratio），默认 0.1（即 MA/10）。
        quality_weights: 质量评分权重 ``(完整率, 准确率, 及时率)``，默认 (0.4, 0.4, 0.2)。
    """

    #: 偏差率定义：``|p1-p2| / min(p1, p2)``（以较低价为基准，避免高价源压低偏差）。
    DEVIATION_FORMULA = "abs(p1-p2)/min(p1,p2)"

    def __init__(
        self,
        fetcher: Any = None,
        db_path: Optional[str] = None,
        price_tolerance: float = 0.005,
        jump_threshold: float = 0.20,
        a_jump_threshold: float = 0.10,
        volume_ma_window: int = 20,
        volume_surge_mult: float = 5.0,
        volume_dry_ratio: float = 0.1,
        quality_weights: tuple = (0.4, 0.4, 0.2),
    ) -> None:
        self.fetcher = fetcher
        self.db_path = db_path
        self.price_tolerance = float(price_tolerance)
        self.jump_threshold = float(jump_threshold)
        self.a_jump_threshold = float(a_jump_threshold)
        self.volume_ma_window = int(volume_ma_window)
        self.volume_surge_mult = float(volume_surge_mult)
        self.volume_dry_ratio = float(volume_dry_ratio)
        self.quality_weights = tuple(quality_weights)

        self._anomalies: Dict[str, AnomalyRecord] = {}
        self._lineage: List[LineageRecord] = []
        self._init_db()

    # ------------------------------------------------------------------
    # SQLite 持久化（本模块自管，不改动 persistence/database.py）
    # ------------------------------------------------------------------

    def _connect(self) -> Optional[sqlite3.Connection]:
        """建立 SQLite 连接（WAL、Row 工厂）；db_path 为 None 返回 None。"""
        if not self.db_path:
            return None
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:  # pragma: no cover - 某些环境不支持 WAL
            pass
        return conn

    def _init_db(self) -> None:
        """建表（IF NOT EXISTS）并从库中恢复已有异常/血缘到内存态。"""
        conn = self._connect()
        if conn is None:
            return
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_anomalies (
                    id TEXT PRIMARY KEY,
                    symbol TEXT, type TEXT, severity TEXT,
                    date TEXT, detail TEXT, status TEXT,
                    detected_at TEXT, fix_history TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, source TEXT, fetched_at TEXT,
                    status TEXT, detail TEXT
                )
                """
            )
            conn.commit()
            for row in conn.execute("SELECT * FROM data_anomalies"):
                self._anomalies[row["id"]] = AnomalyRecord(
                    id=row["id"], symbol=row["symbol"], type=row["type"],
                    severity=row["severity"], date=row["date"] or "",
                    detail=row["detail"] or "", status=row["status"],
                    detected_at=row["detected_at"],
                    fix_history=self._decode_fix_history(row["fix_history"]),
                )
            for row in conn.execute(
                "SELECT * FROM data_lineage ORDER BY id"
            ):
                self._lineage.append(
                    LineageRecord(
                        symbol=row["symbol"], source=row["source"],
                        fetched_at=row["fetched_at"], status=row["status"],
                        detail=row["detail"] or "",
                    )
                )
        finally:
            conn.close()

    @staticmethod
    def _decode_fix_history(raw: Any) -> List[Dict[str, Any]]:
        """安全解码 fix_history JSON 字段。"""
        import json

        if not raw:
            return []
        try:
            v = json.loads(raw)
            return v if isinstance(v, list) else []
        except (ValueError, TypeError):
            return []

    def _persist_anomaly(self, rec: AnomalyRecord) -> None:
        """把异常记录写回 SQLite（全字段覆盖）。"""
        import json

        conn = self._connect()
        if conn is None:
            return
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO data_anomalies
                  (id, symbol, type, severity, date, detail, status,
                   detected_at, fix_history)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    rec.id, rec.symbol, rec.type, rec.severity, rec.date,
                    rec.detail, rec.status, rec.detected_at,
                    json.dumps(rec.fix_history, ensure_ascii=False),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _persist_lineage(self, rec: LineageRecord) -> None:
        """追加一条血缘到 SQLite。"""
        conn = self._connect()
        if conn is None:
            return
        try:
            conn.execute(
                """
                INSERT INTO data_lineage
                  (symbol, source, fetched_at, status, detail)
                VALUES (?,?,?,?,?)
                """,
                (rec.symbol, rec.source, rec.fetched_at, rec.status, rec.detail),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 价格交叉验证
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_price(quote: Any) -> Optional[float]:
        """从行情快照中容错取价（兼容 price / last_price 字段）。"""
        if not isinstance(quote, dict):
            return None
        for key in ("price", "last_price"):
            val = quote.get(key)
            if val is not None:
                try:
                    p = float(val)
                    if p > 0:
                        return p
                except (TypeError, ValueError):
                    continue
        return None

    def reconcile_prices(
        self,
        symbol: str,
        sources: Optional[Dict[str, float]] = None,
        tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """多数据源价格交叉验证。

        Args:
            symbol: 标的代码。
            sources: ``{数据源名: 价格}``，如 ``{"quantdash": 1680.0, "tencent": 1688.4}``；
                不传时通过注入的 ``fetcher`` 分别取 QuantDash 默认源与腾讯备用源
                （此为防御性 live 分支；未注入 fetcher 时返回说明）。
            tolerance: 覆盖实例默认的偏差容忍率。

        Returns:
            结果字典，含 ``symbol / prices / deviation / tolerance / status /
            anomaly_id（若 flagged）``。status 为 ``validated`` 或 ``flagged``。
        """
        tol = self.price_tolerance if tolerance is None else float(tolerance)

        if sources is None:
            sources = self._fetch_cross_source_prices(symbol)
            if not sources:
                return {
                    "symbol": symbol, "prices": {}, "deviation": None,
                    "tolerance": tol, "status": "unavailable",
                    "message": "未提供 sources 且未注入可用 fetcher，无法交叉验证",
                }

        # 至少需要两个有效价格
        valid = {k: float(v) for k, v in sources.items() if v is not None and v > 0}
        if len(valid) < 2:
            return {
                "symbol": symbol, "prices": valid, "deviation": None,
                "tolerance": tol, "status": "insufficient",
                "message": "有效价格源不足 2 个",
            }

        # 两两比对，取最大偏差率
        names = list(valid.keys())
        max_dev = 0.0
        worst_pair = (names[0], names[1])
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = valid[names[i]], valid[names[j]]
                dev = abs(a - b) / min(a, b)
                if dev > max_dev:
                    max_dev = dev
                    worst_pair = (names[i], names[j])

        result: Dict[str, Any] = {
            "symbol": symbol,
            "prices": valid,
            "deviation": round(max_dev, 6),
            "tolerance": tol,
            "worst_pair": worst_pair,
            "formula": self.DEVIATION_FORMULA,
        }

        if max_dev > tol:
            rec = self._add_anomaly(
                symbol=symbol,
                atype="price_mismatch",
                severity="high",
                date="",
                detail=(
                    f"价格交叉验证偏差 {max_dev:.4f} > 容忍 {tol:.4f}"
                    f"（{worst_pair[0]}={valid[worst_pair[0]]} vs "
                    f"{worst_pair[1]}={valid[worst_pair[1]]}）"
                ),
            )
            result["status"] = "flagged"
            result["anomaly_id"] = rec.id
            self.record_lineage(symbol, source=worst_pair[0], status="flagged",
                                detail=rec.detail)
        else:
            result["status"] = "validated"
            self.record_lineage(symbol, source=",".join(names), status="validated",
                                detail=f"偏差 {max_dev:.4f} <= {tol:.4f}")
        return result

    def _fetch_cross_source_prices(self, symbol: str) -> Dict[str, float]:
        """通过注入的 fetcher 取 QuantDash 默认源 + 腾讯备用源实时价。

        防御性实现：任一来源失败/取到 None 都会被跳过；全部失败返回空 dict。
        """
        prices: Dict[str, float] = {}
        if self.fetcher is None:
            return prices
        # 默认源（QuantDash）
        try:
            quote = self.fetcher.get_realtime_quote(symbol)
            p = self._extract_price(quote)
            if p is not None:
                src = quote.get("source", "quantdash") if isinstance(quote, dict) else "quantdash"
                prices[src] = p
        except Exception as e:  # noqa: BLE001
            logger.warning("默认源取价失败(%s): %s", symbol, e)
        # 腾讯备用源
        try:
            from data.data_fetcher import _fetch_tencent_realtime

            tq = _fetch_tencent_realtime(symbol)
            p = self._extract_price(tq)
            if p is not None:
                prices["tencent"] = p
        except Exception as e:  # noqa: BLE001
            logger.warning("腾讯备用源取价失败(%s): %s", symbol, e)
        return prices

    # ------------------------------------------------------------------
    # 异常检测
    # ------------------------------------------------------------------

    def detect_anomalies(
        self,
        symbol: str,
        df: pd.DataFrame,
        jump_threshold: Optional[float] = None,
    ) -> List[AnomalyRecord]:
        """对 K线 DataFrame 做全量异常检测。

        检测四类异常：

        1. **价格跳变**：单日 ``|close.pct_change()|`` 超阈值（A股默认 0.10，其余 0.20，
           可被 ``jump_threshold`` 覆盖）。
        2. **成交量异常**：volume > 20日均量 × ``volume_surge_mult``（放量）或
           volume < 均量 × ``volume_dry_ratio``（地量）。
        3. **缺失K线**：以工作日（Mon-Fri）启发式交易日历比对日期索引，列出缺失日期。
           .. warning:: 真实交易日历（含节假日）尚未接入，按工作日启发式会在节假日误报。
        4. **OHLC 逻辑错误**：high<low、high<open/close、low>open/close、
           close/open/high/low ≤ 0。

        Args:
            symbol: 标的代码。
            df: K线 DataFrame，需含 open/high/low/close/volume 列，索引为日期。
            jump_threshold: 显式覆盖跳变阈值；缺省按市场自动选择。

        Returns:
            本次新检测到的异常记录列表（已写入内存与库）。
        """
        found: List[AnomalyRecord] = []
        if df is None or len(df) == 0:
            return found

        df = df.copy()
        required = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required):
            raise ValueError(f"K线缺少必需列，需要 {required}")

        threshold = jump_threshold
        if threshold is None:
            threshold = self._auto_jump_threshold(symbol)

        # 1. 价格跳变
        close = pd.to_numeric(df["close"], errors="coerce")
        pct = close.pct_change()
        for dt, val in pct.items():
            if pd.isna(val):
                continue
            if abs(val) > threshold:
                found.append(self._add_anomaly(
                    symbol=symbol, atype="price_jump", severity="high",
                    date=self._date_str(dt),
                    detail=f"单日涨跌幅 {val:.4f} 超阈值 {threshold:.4f}",
                ))

        # 2. 成交量异常（均线窗口不足时跳过）
        vol = pd.to_numeric(df["volume"], errors="coerce")
        ma = vol.rolling(window=self.volume_ma_window, min_periods=2).mean()
        for dt, v in vol.items():
            m = ma.loc[dt]
            if pd.isna(v) or pd.isna(m) or m <= 0:
                continue
            if v > m * self.volume_surge_mult:
                found.append(self._add_anomaly(
                    symbol=symbol, atype="volume_surge", severity="medium",
                    date=self._date_str(dt),
                    detail=f"成交量 {v:.0f} > {self.volume_ma_window}日均量 {m:.0f}"
                           f"×{self.volume_surge_mult}",
                ))
            elif v < m * self.volume_dry_ratio:
                found.append(self._add_anomaly(
                    symbol=symbol, atype="volume_dry", severity="low",
                    date=self._date_str(dt),
                    detail=f"成交量 {v:.0f} < 均量 {m:.0f}×{self.volume_dry_ratio}",
                ))

        # 3. 缺失K线（工作日启发式；节假日会误报）
        missing = self._find_missing_trading_days(df)
        for d in missing:
            found.append(self._add_anomaly(
                symbol=symbol, atype="missing_kline", severity="medium",
                date=self._date_str(d),
                detail="该工作日无K线（按 Mon-Fri 启发式，节假日可能误报）",
            ))

        # 4. OHLC 逻辑错误
        for dt, row in df.iterrows():
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            problems: List[str] = []
            for name, x in (("open", o), ("high", h), ("low", l), ("close", c)):
                try:
                    if float(x) <= 0:
                        problems.append(f"{name}={x} 非正")
                except (TypeError, ValueError):
                    problems.append(f"{name}={x} 非数值")
                    continue
            try:
                if float(h) < float(l):
                    problems.append(f"high({h}) < low({l})")
                if float(h) < max(float(o), float(c)):
                    problems.append(f"high({h}) 未覆盖 open({o})/close({c})")
                if float(l) > min(float(o), float(c)):
                    problems.append(f"low({l}) 未落在 open({o})/close({c}) 区间")
            except (TypeError, ValueError):
                pass
            if problems:
                found.append(self._add_anomaly(
                    symbol=symbol, atype="ohlc_logic", severity="high",
                    date=self._date_str(dt),
                    detail="; ".join(problems),
                ))

        if found:
            self.record_lineage(symbol, source="anomaly_scan",
                                status="flagged" if any(
                                    a.severity == "high" for a in found) else "validated",
                                detail=f"检测到 {len(found)} 条异常")
        return found

    def _auto_jump_threshold(self, symbol: str) -> float:
        """按市场自动选择跳变阈值：A股用 a_jump_threshold，其余用 jump_threshold。"""
        try:
            from data.data_fetcher import get_market

            if get_market(symbol) == "A股":
                return self.a_jump_threshold
        except Exception:  # noqa: BLE001
            pass
        return self.jump_threshold

    @staticmethod
    def _date_str(dt: Any) -> str:
        """把索引元素转为 YYYY-MM-DD 字符串。"""
        try:
            return pd.Timestamp(dt).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            return str(dt)

    def _find_missing_trading_days(self, df: pd.DataFrame) -> List[Any]:
        """以工作日（Mon-Fri）启发式比对日期索引，返回缺失的工作日列表。"""
        if len(df) == 0:
            return []
        start = pd.Timestamp(df.index.min())
        end = pd.Timestamp(df.index.max())
        full = pd.bdate_range(start=start.normalize(), end=end.normalize())
        have = {pd.Timestamp(d).normalize() for d in df.index}
        return [d for d in full if d not in have]

    def _add_anomaly(
        self, symbol: str, atype: str, severity: str, date: str, detail: str
    ) -> AnomalyRecord:
        """登记一条异常（内存 + 库），并返回该记录。"""
        rec = AnomalyRecord(
            id=f"AN{uuid.uuid4().hex[:10]}",
            symbol=symbol, type=atype, severity=severity,
            date=date, detail=detail,
        )
        self._anomalies[rec.id] = rec
        self._persist_anomaly(rec)
        return rec

    # ------------------------------------------------------------------
    # 数据血缘
    # ------------------------------------------------------------------

    def record_lineage(
        self,
        symbol: str,
        source: str,
        fetched_at: Optional[str] = None,
        status: str = "validated",
        detail: str = "",
    ) -> LineageRecord:
        """记录一条数据血缘。

        Args:
            symbol: 标的代码。
            source: 数据来源。
            fetched_at: 抓取时间；缺省取当前时间。
            status: validated / flagged / corrected。
            detail: 备注。

        Returns:
            已登记的 :class:`LineageRecord`。
        """
        rec = LineageRecord(
            symbol=symbol, source=source,
            fetched_at=fetched_at or datetime.now().isoformat(),
            status=status, detail=detail,
        )
        self._lineage.append(rec)
        self._persist_lineage(rec)
        return rec

    def get_lineage(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """查询血缘记录，可按标的过滤。"""
        recs = self._lineage
        if symbol is not None:
            recs = [r for r in recs if r.symbol == symbol]
        return [r.to_dict() for r in recs]

    # ------------------------------------------------------------------
    # 异常自动修复
    # ------------------------------------------------------------------

    def fix_anomaly(
        self,
        anomaly_id: str,
        backup_data: Any = None,
    ) -> Dict[str, Any]:
        """自动修复异常。

        修复策略（按优先级）：

        1. 使用显式传入的 ``backup_data``（备用数据 / 备用源报价）；
        2. 未传入且注入了 fetcher 时，从腾讯备用源重新取数补全。

        修复成功：异常 ``status="fixed"``，写入 ``fix_history``（前后值留痕），
        并补一条 ``corrected`` 血缘。备用源也缺数据时：异常保持 ``open``，
        返回说明，**不假装成功**。

        Args:
            anomaly_id: 异常 ID。
            backup_data: 备用数据（dict，可含 corrected_value / corrected_df）。

        Returns:
            结果字典 ``{id, status, message, before, after}``。

        Raises:
            KeyError: 异常不存在。
        """
        rec = self._anomalies.get(anomaly_id)
        if rec is None:
            raise KeyError(f"异常不存在: {anomaly_id}")

        if rec.status == "fixed":
            return {"id": rec.id, "status": "fixed",
                    "message": "该异常已修复", "fix_history": rec.fix_history}

        before = rec.detail
        after: Any = None

        # 1) 显式备用数据
        if isinstance(backup_data, dict):
            if "corrected_value" in backup_data or "corrected_df" in backup_data:
                after = backup_data.get("corrected_value") or "corrected_df"
            elif backup_data:
                after = backup_data
        # 2) 从备用源（腾讯）重新取价
        if after is None and self.fetcher is not None:
            try:
                from data.data_fetcher import _fetch_tencent_realtime

                tq = _fetch_tencent_realtime(rec.symbol)
                p = self._extract_price(tq)
                if p is not None:
                    after = {"tencent_price": p}
            except Exception as e:  # noqa: BLE001
                logger.warning("备用源修复取数失败(%s): %s", rec.symbol, e)

        if after is None:
            # 备用也缺：保持 open，如实返回
            return {
                "id": rec.id, "status": "open",
                "message": "备用数据/备用源均不可用，无法自动修复，保持 open",
                "before": before, "after": None,
            }

        rec.status = "fixed"
        rec.fix_history.append({
            "action": "auto_fix",
            "before": before,
            "after": after,
            "fixed_at": datetime.now().isoformat(),
        })
        self._persist_anomaly(rec)
        self.record_lineage(rec.symbol, source="backup_source",
                            status="corrected",
                            detail=f"修复异常 {rec.id}: {before} -> {after}")
        return {
            "id": rec.id, "status": "fixed",
            "message": "修复成功", "before": before, "after": after,
        }

    def get_anomalies(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """返回异常列表，可按 status 过滤。"""
        recs = list(self._anomalies.values())
        if status:
            recs = [r for r in recs if r.status == status]
        return [r.to_dict() for r in recs]

    # ------------------------------------------------------------------
    # 数据质量评分
    # ------------------------------------------------------------------

    def quality_score(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        *,
        completeness: Optional[float] = None,
        anomaly_count: Optional[int] = None,
        total_records: Optional[int] = None,
        timely_ratio: Optional[float] = None,
        weights: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """计算 0-100 数据质量评分（完整率 / 准确率 / 及时率加权）。

        子项定义：

        - **完整率 completeness**：非缺失K线 / 预期K线（工作日启发式）；可直接传入覆盖。
        - **准确率 accuracy**：``1 - 异常条数 / 总记录数``；``anomaly_count`` 缺省取当前
          该标的 open 异常数。
        - **及时率 timeliness**：简化定义，可由 ``timely_ratio`` 直接传入；缺省为 1.0
          （真实抓取时效统计待接入）。

        Args:
            symbol: 标的代码。
            df: 可选 K线，用于推导完整率。
            completeness: 直接给定完整率（0-1）。
            anomaly_count: 异常条数（缺省取该标的 open 异常数）。
            total_records: 总记录数（缺省取 df 行数或异常数下限）。
            timely_ratio: 及时率（0-1）。
            weights: 覆盖实例权重。

        Returns:
            ``{symbol, total, completeness, accuracy, timeliness, weights}``，
            total 为 0-100。
        """
        w = tuple(weights) if weights else self.quality_weights

        # 完整率
        if completeness is None:
            if df is not None and len(df) > 0:
                expected = len(pd.bdate_range(
                    start=pd.Timestamp(df.index.min()).normalize(),
                    end=pd.Timestamp(df.index.max()).normalize(),
                ))
                completeness = (len(df) / expected) if expected > 0 else 1.0
            else:
                completeness = 1.0
        completeness = max(0.0, min(1.0, float(completeness)))

        # 准确率
        if anomaly_count is None:
            anomaly_count = sum(
                1 for r in self._anomalies.values()
                if r.symbol == symbol and r.status == "open"
            )
        if total_records is None:
            total_records = len(df) if df is not None else max(int(anomaly_count), 1)
        total_records = max(int(total_records), 1)
        accuracy = max(0.0, 1.0 - float(anomaly_count) / total_records)

        # 及时率（简化）
        if timely_ratio is None:
            timely_ratio = 1.0
        timely_ratio = max(0.0, min(1.0, float(timely_ratio)))

        total = 100.0 * (w[0] * completeness + w[1] * accuracy + w[2] * timely_ratio)
        return {
            "symbol": symbol,
            "total": round(total, 2),
            "completeness": round(completeness, 4),
            "accuracy": round(accuracy, 4),
            "timeliness": round(timely_ratio, 4),
            "weights": w,
        }

    def daily_quality_report(
        self,
        symbols: List[str],
        scores: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """生成每日质量报告。

        Args:
            symbols: 标的列表。
            scores: 预先算好的 ``{symbol: quality_score 返回}``；缺省时对每个标的按
                当前 open 异常自动评分（无 df，完整率默认 1.0）。

        Returns:
            结构化报告 ``{date, symbols, per_symbol, anomaly_summary, fix_summary}``。
        """
        date = datetime.now().strftime("%Y-%m-%d")
        per_symbol: Dict[str, Any] = {}
        for sym in symbols:
            sc = scores.get(sym) if scores else self.quality_score(sym)
            per_symbol[sym] = sc

        open_anoms = [a for a in self._anomalies.values() if a.status == "open"]
        fixed_anoms = [a for a in self._anomalies.values() if a.status == "fixed"]
        return {
            "date": date,
            "symbols": list(symbols),
            "per_symbol": per_symbol,
            "anomaly_summary": {
                "total_open": len(open_anoms),
                "by_type": self._count_by_type(open_anoms),
            },
            "fix_summary": {
                "total_fixed": len(fixed_anoms),
                "fixed_ids": [a.id for a in fixed_anoms],
            },
        }

    @staticmethod
    def _count_by_type(recs: List[AnomalyRecord]) -> Dict[str, int]:
        """按异常类型计数。"""
        out: Dict[str, int] = {}
        for r in recs:
            out[r.type] = out.get(r.type, 0) + 1
        return out
