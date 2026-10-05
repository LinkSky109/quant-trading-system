"""因子风险暴露监控模块（REQ-P1-09）。

提供组合层面的因子暴露度量、中性化限额监控、调仓建议、暴露时序持久化、
因子收益贡献度分解以及因子间相关性矩阵计算。

核心概念
--------
- **单票因子 z-score**：对某只标的而言，某因子的标准化暴露度（已由
  :class:`factors.factor_engine.FactorEngine` 在其历史时序上做 z-score 标准化，
  理论均值≈0、标准差≈1）。
- **组合因子暴露**：组合在因子 f 上的暴露为持仓加权和

  .. math:: E_f = \\sum_i w_i \\cdot z_{i,f}

  其中缺失某标的 z-score 时按 0 参与计算并记录警告。
- **类别暴露**：把因子按 7 大类（价值/成长/质量/动量/波动率/技术/流动性，
  以 :class:`FactorEngine` 注册表 ``category`` 字段为准）归类，**类别暴露取类内
  各因子暴露的算术平均（带符号，未取绝对值）**；即 ``E_category = mean(E_f)``。
  未在引擎注册表中登记的因子归入 ``未分类``。
- **中性化限额**：单因子暴露绝对值超过 ``threshold``（默认 ±1.0σ）即视为超限。

典型用法::

    monitor = FactorExposureMonitor(db_path="data/quant_trading.db",
                                    alert_manager=alert_mgr)
    res = monitor.calculate_exposure(holdings, factor_scores)
    breaches = monitor.check_limits(res["exposures"])
    monitor.record_snapshot("2026-10-03", res["exposures"])
"""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

#: 权重闭合容差：持仓权重之和应≈1，偏差超过该值抛 ValueError。
WEIGHT_TOLERANCE = 1e-4

#: 未登记因子归入的类别名。
UNKNOWN_CATEGORY = "未分类"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Holding:
    """组合持仓单元。

    Attributes:
        symbol: 标的代码。
        weight: 权重（占组合净值比例，0~1）。
    """

    symbol: str
    weight: float


@dataclass
class ExposureResult:
    """组合因子暴露计算结果。

    Attributes:
        exposures: ``{factor_name: 组合暴露}``。
        category_exposures: ``{category: 类内因子暴露算术平均}``。
        factor_categories: ``{factor_name: category}`` 因子归类映射。
        missing_symbols: 完全没有提供 z-score 的标的列表（按 0 处理）。
        weight_sum: 权重之和。
    """

    exposures: Dict[str, float] = field(default_factory=dict)
    category_exposures: Dict[str, float] = field(default_factory=dict)
    factor_categories: Dict[str, str] = field(default_factory=dict)
    missing_symbols: List[str] = field(default_factory=list)
    weight_sum: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转成可 JSON 序列化的字典。"""
        return {
            "exposures": dict(self.exposures),
            "category_exposures": dict(self.category_exposures),
            "factor_categories": dict(self.factor_categories),
            "missing_symbols": list(self.missing_symbols),
            "weight_sum": self.weight_sum,
        }


@dataclass
class LimitBreach:
    """单因子超限记录。

    Attributes:
        factor: 因子名。
        exposure: 当前组合暴露（带符号）。
        threshold: 限额绝对值（|bound|）。
        exceeded_by: 超出限额的幅度（``|exposure| - threshold``）。
    """

    factor: str
    exposure: float
    threshold: float
    exceeded_by: float

    def to_dict(self) -> Dict[str, Any]:
        """转成字典。"""
        return {
            "factor": self.factor,
            "exposure": self.exposure,
            "threshold": self.threshold,
            "exceeded_by": self.exceeded_by,
        }


# ---------------------------------------------------------------------------
# 监控器
# ---------------------------------------------------------------------------


class FactorExposureMonitor:
    """组合因子暴露监控器。

    Args:
        db_path: SQLite 库文件路径，默认 ``data/quant_trading.db``。
            测试时可传入 ``tmp_path`` 下的临时库。
        alert_manager: 可选的告警管理器（:class:`monitoring.alert.AlertManager`）。
            为 None 时超限只记录返回值，不触发告警。
        thresholds: 按因子单独配置的限额 ``{factor: threshold}``。
        default_threshold: 未单独配置因子的默认限额（绝对值），默认 1.0σ。
    """

    def __init__(
        self,
        db_path: str = "data/quant_trading.db",
        alert_manager: Any = None,
        thresholds: Optional[Dict[str, float]] = None,
        default_threshold: float = 1.0,
    ) -> None:
        self.db_path = db_path
        self.alert_manager = alert_manager
        self.thresholds: Dict[str, float] = dict(thresholds or {})
        self.default_threshold = float(default_threshold)
        # 因子 -> 类别 映射（惰性从 FactorEngine 注册表加载，缓存）
        self._factor_category: Optional[Dict[str, str]] = None
        self._init_db()

    # ------------------------------------------------------------------ #
    # SQLite 持久化（直连，不改 persistence/database.py）
    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        """建立带 Row 工厂与 WAL 模式的连接。"""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """建表（幂等）。"""
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS factor_exposure_history (
                    date TEXT NOT NULL,
                    factor TEXT NOT NULL,
                    exposure REAL NOT NULL,
                    PRIMARY KEY (date, factor)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # 因子类别映射
    # ------------------------------------------------------------------ #
    def _load_factor_categories(self) -> Dict[str, str]:
        """从 FactorEngine 注册表加载 ``{factor_name: category}``。

        引擎不可用或因子未登记时返回空映射，调用方据此把因子归入
        :data:`UNKNOWN_CATEGORY`。结果会被缓存。
        """
        if self._factor_category is not None:
            return self._factor_category
        mapping: Dict[str, str] = {}
        try:
            from factors.factor_engine import FactorEngine

            for item in FactorEngine().get_factor_list():
                mapping[item["name"]] = item.get("category", UNKNOWN_CATEGORY)
        except Exception as exc:  # pragma: no cover - 引擎缺失时降级
            logger.warning("FactorEngine 不可用，因子类别退化为未分类: %s", exc)
        self._factor_category = mapping
        return mapping

    # ------------------------------------------------------------------ #
    # 组合暴露计算
    # ------------------------------------------------------------------ #
    def calculate_exposure(
        self,
        holdings: Sequence[Dict[str, Any]],
        factor_scores: Dict[str, Dict[str, float]],
    ) -> ExposureResult:
        """计算组合在各因子上的暴露。

        组合暴露 ``E_f = Σ_i w_i · z_{i,f}``。某标的完全缺失 z-score，或某因子在
        该标的上缺失时，该标的在该因子上按 0 参与计算；完全缺失 z-score 的标的
        会被记录到 ``missing_symbols`` 并输出 warning 日志。

        Args:
            holdings: 持仓列表 ``[{symbol, weight}]``，权重之和应≈1。
            factor_scores: 各标的因子 z-score ``{symbol: {factor_name: z}}``。

        Returns:
            :class:`ExposureResult`，含逐因子暴露、类别汇总与缺失标的。

        Raises:
            ValueError: 权重之和与 1 的偏差超过 :data:`WEIGHT_TOLERANCE`。
        """
        if not holdings:
            raise ValueError("持仓列表为空")

        weight_sum = float(sum(float(h["weight"]) for h in holdings))
        if abs(weight_sum - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(
                f"持仓权重之和={weight_sum:.6f} 偏离 1.0 超过容差 "
                f"{WEIGHT_TOLERANCE:g}（须先归一化或补全权重）"
            )

        # 汇总所有出现过的因子名
        all_factors: List[str] = []
        seen = set()
        for scores in factor_scores.values():
            for f in (scores or {}):
                if f not in seen:
                    seen.add(f)
                    all_factors.append(f)

        exposures: Dict[str, float] = {f: 0.0 for f in all_factors}
        missing_symbols: List[str] = []

        for h in holdings:
            symbol = h["symbol"]
            w = float(h["weight"])
            scores = factor_scores.get(symbol)
            if not scores:
                missing_symbols.append(symbol)
                logger.warning("标的 %s 缺少因子 z-score，按 0 暴露处理", symbol)
                continue
            for f in all_factors:
                z = scores.get(f)
                if z is None:
                    logger.warning("标的 %s 在因子 %s 上缺失 z-score，按 0 处理",
                                   symbol, f)
                    continue
                exposures[f] += w * float(z)

        # 类别汇总：类内因子暴露算术平均（带符号）
        cat_map = self._load_factor_categories()
        factor_categories = {
            f: cat_map.get(f, UNKNOWN_CATEGORY) for f in all_factors
        }
        cat_buckets: Dict[str, List[float]] = {}
        for f, e in exposures.items():
            cat = factor_categories[f]
            cat_buckets.setdefault(cat, []).append(e)
        category_exposures = {
            cat: float(sum(vals) / len(vals)) for cat, vals in cat_buckets.items()
        }

        return ExposureResult(
            exposures=exposures,
            category_exposures=category_exposures,
            factor_categories=factor_categories,
            missing_symbols=missing_symbols,
            weight_sum=weight_sum,
        )

    # ------------------------------------------------------------------ #
    # 中性化限额
    # ------------------------------------------------------------------ #
    def _threshold_for(self, factor: str) -> float:
        """取某因子的限额绝对值。"""
        return float(self.thresholds.get(factor, self.default_threshold))

    def check_limits(self, exposures: Dict[str, float]) -> List[LimitBreach]:
        """检查各因子暴露是否超限，并在挂载告警管理器时触发 WARNING 告警。

        Args:
            exposures: ``{factor: exposure}`` 组合暴露。

        Returns:
            超限因子列表（未超限的因子不出现在列表中）。若挂载了
            ``alert_manager``，每个超限因子都会调用一次
            ``alert(level=WARNING, category="factor_exposure_limit", ...)``。
        """
        breaches: List[LimitBreach] = []
        for factor, exp in exposures.items():
            threshold = self._threshold_for(factor)
            abs_exp = abs(float(exp))
            if abs_exp > threshold:
                exceeded_by = abs_exp - threshold
                breach = LimitBreach(
                    factor=factor,
                    exposure=float(exp),
                    threshold=threshold,
                    exceeded_by=float(exceeded_by),
                )
                breaches.append(breach)
                self._fire_alert(breach)
        return breaches

    def _fire_alert(self, breach: LimitBreach) -> None:
        """触发一条因子超限告警（无 alert_manager 时静默跳过）。"""
        if self.alert_manager is None:
            return
        try:
            from monitoring.alert import AlertLevel

            self.alert_manager.alert(
                level=AlertLevel.WARNING,
                category="factor_exposure_limit",
                message=(
                    f"因子 {breach.factor} 组合暴露 {breach.exposure:+.3f} "
                    f"超出限额 ±{breach.threshold:.2f}（超限 {breach.exceeded_by:.3f}）"
                ),
                current_value=breach.exposure,
                threshold=breach.threshold,
            )
        except Exception as exc:  # pragma: no cover - 告警失败不阻断主流程
            logger.warning("因子暴露告警发送失败（已忽略）: %s", exc)

    # ------------------------------------------------------------------ #
    # 中性化调仓建议
    # ------------------------------------------------------------------ #
    def neutralize(
        self,
        holdings: Sequence[Dict[str, Any]],
        factor_scores: Dict[str, Dict[str, float]],
        target_factor: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """针对超限因子给出可执行的调仓方向与近似权重调整量。

        对每个超限因子 f：
        - 若组合暴露为**正超限**（E_f > threshold）：建议**减持**在该因子上
          z-score 最高的标的、**增持** z-score 最低（最负）的标的，以压低暴露。
        - 若为**负超限**（E_f < -threshold）：方向相反——增持高 z 标的、
          减持低 z 标的。
        - 近似权重调整量 ``weight_delta`` 由对称调仓推得：设最高 z 为 z_hi、
          最低 z 为 z_lo，对称调整 δ 使 ``δ·(z_lo - z_hi) = -超限幅度``，
          故 ``δ = |exposure| - threshold / (z_hi - z_lo)``。该 δ 即各标的
          建议的近似权重调整幅度（正负由 action 决定方向）。

        Args:
            holdings: 持仓 ``[{symbol, weight}]``。
            factor_scores: ``{symbol: {factor: z}}``。
            target_factor: 只针对该因子给建议；为 None 时对所有超限因子给建议。

        Returns:
            调仓建议列表，每项含 ``symbol / action(increase|decrease) / reason /
            target_factor / current_zscore / weight_delta``。
        """
        result = self.calculate_exposure(holdings, factor_scores)
        exposures = result.exposures

        # 确定需要处理的超限因子集合
        if target_factor is not None:
            if target_factor not in exposures:
                return []
            candidates = [target_factor] if self._is_breach(
                target_factor, exposures[target_factor]) else []
        else:
            candidates = [
                f for f, e in exposures.items() if self._is_breach(f, e)
            ]

        suggestions: List[Dict[str, Any]] = []
        for factor in candidates:
            exp = float(exposures[factor])
            threshold = self._threshold_for(factor)
            over_mag = abs(exp) - threshold  # 需要拉回的幅度

            # 收集有该因子 z-score 的标的
            rows: List[Dict[str, float]] = []
            for h in holdings:
                sym = h["symbol"]
                z = (factor_scores.get(sym) or {}).get(factor)
                if z is None:
                    continue
                rows.append({"symbol": sym, "z": float(z)})
            if len(rows) < 2:
                continue

            rows.sort(key=lambda r: r["z"])
            low = rows[0]    # 最低 z
            high = rows[-1]  # 最高 z
            z_span = high["z"] - low["z"]
            if z_span == 0:
                continue
            delta = over_mag / z_span

            if exp > 0:
                # 正超限：减持高 z，增持低 z
                suggestions.append(self._build_suggestion(
                    high, "decrease", factor, exp, threshold, delta))
                suggestions.append(self._build_suggestion(
                    low, "increase", factor, exp, threshold, delta))
            else:
                # 负超限：增持高 z，减持低 z
                suggestions.append(self._build_suggestion(
                    high, "increase", factor, exp, threshold, delta))
                suggestions.append(self._build_suggestion(
                    low, "decrease", factor, exp, threshold, delta))
        return suggestions

    def _is_breach(self, factor: str, exposure: float) -> bool:
        """判断某因子当前是否超限。"""
        return abs(float(exposure)) > self._threshold_for(factor)

    @staticmethod
    def _build_suggestion(
        row: Dict[str, float],
        action: str,
        factor: str,
        exposure: float,
        threshold: float,
        weight_delta: float,
    ) -> Dict[str, Any]:
        """构造单条调仓建议字典。"""
        direction = "压低" if exposure > 0 else "抬高"
        return {
            "symbol": row["symbol"],
            "action": action,
            "target_factor": factor,
            "current_zscore": row["z"],
            "weight_delta": round(weight_delta, 6),
            "reason": (
                f"因子 {factor} 组合暴露 {exposure:+.3f} 超出 ±{threshold:.2f}，"
                f"需{direction}该因子暴露；当前标的 z={row['z']:+.3f}，"
                f"建议{('减持' if action == 'decrease' else '增持')}"
            ),
        }

    # ------------------------------------------------------------------ #
    # 暴露时序
    # ------------------------------------------------------------------ #
    def record_snapshot(
        self, date: str, exposure: Dict[str, float]
    ) -> None:
        """记录某日组合因子暴露快照（按 date+factor 主键 upsert）。

        Args:
            date: 交易日，建议 ``YYYY-MM-DD``；为空时取当前日期。
            exposure: ``{factor: exposure}``。
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        conn = self._connect()
        try:
            for factor, value in exposure.items():
                conn.execute(
                    """
                    INSERT INTO factor_exposure_history (date, factor, exposure)
                    VALUES (?, ?, ?)
                    ON CONFLICT(date, factor) DO UPDATE SET exposure=excluded.exposure
                    """,
                    (date, factor, float(value)),
                )
            conn.commit()
        finally:
            conn.close()

    def get_history(
        self,
        factor: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """查询暴露历史。

        Args:
            factor: 只查某因子；为 None 时查全部。
            start: 起始日期（含，闭区间）。
            end: 结束日期（含，闭区间）。

        Returns:
            ``[{date, factor, exposure}]``，按 date 升序。
        """
        sql = "SELECT date, factor, exposure FROM factor_exposure_history WHERE 1=1"
        params: List[Any] = []
        if factor:
            sql += " AND factor = ?"
            params.append(factor)
        if start:
            sql += " AND date >= ?"
            params.append(start)
        if end:
            sql += " AND date <= ?"
            params.append(end)
        sql += " ORDER BY date ASC"

        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [
                {"date": r["date"], "factor": r["factor"],
                 "exposure": float(r["exposure"])}
                for r in rows
            ]
        finally:
            conn.close()

    def time_series_chart_data(self, factor: str) -> List[Dict[str, Any]]:
        """为前端折线图组装单因子暴露时序数据。

        Args:
            factor: 因子名。

        Returns:
            ``[{date, exposure, threshold}]``，threshold 取该因子限额绝对值。
        """
        history = self.get_history(factor=factor)
        threshold = self._threshold_for(factor)
        return [
            {
                "date": row["date"],
                "exposure": row["exposure"],
                "threshold": threshold,
            }
            for row in history
        ]

    # ------------------------------------------------------------------ #
    # 因子收益贡献度
    # ------------------------------------------------------------------ #
    def factor_contributions(
        self,
        exposure: Dict[str, float],
        factor_returns: Dict[str, float],
    ) -> Dict[str, Any]:
        """计算各因子对组合收益的贡献度。

        单因子贡献 = 组合暴露 × 该因子当期收益率；合计为所有因子贡献之和。

        Args:
            exposure: ``{factor: 组合暴露}``。
            factor_returns: ``{factor: 因子收益率}``。

        Returns:
            ``{"contributions": {factor: 贡献}, "total": 合计贡献}``。
        """
        contributions: Dict[str, float] = {}
        total = 0.0
        for factor, exp in exposure.items():
            ret = factor_returns.get(factor, 0.0)
            contrib = float(exp) * float(ret)
            contributions[factor] = contrib
            total += contrib
        return {"contributions": contributions, "total": float(total)}

    # ------------------------------------------------------------------ #
    # 因子相关性
    # ------------------------------------------------------------------ #
    def factor_correlation(
        self, factor_scores: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """从多标的因子 z-score 矩阵计算因子间 Pearson 相关系数矩阵。

        Args:
            factor_scores: ``{symbol: {factor: z}}``。

        Returns:
            因子×因子 的 Pearson 相关系数 :class:`pandas.DataFrame`；
            无数据时返回空 DataFrame。
        """
        if not factor_scores:
            return pd.DataFrame()
        df = pd.DataFrame(factor_scores).T  # 行=标的，列=因子
        if df.shape[1] < 1:
            return pd.DataFrame()
        return df.corr(method="pearson")
