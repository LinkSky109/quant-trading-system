"""数据校验与对账 API 路由（REQ-P2-09 扩展模块）。

按 server.py 现有扩展路由模式挂载：

    try:
        from _routes_data import register_data_routes
        register_data_routes(app, manager, SYMBOL_SET, ok, err)
    except Exception as e:
        logger.warning("数据校验路由注册失败: %s", e)

端点：
    POST /api/data/reconcile                     —— 多源价格交叉验证 + 全量异常检测
    GET  /api/data/quality                       —— 数据质量评分与每日报告
    GET  /api/data/anomalies                     —— 异常列表（支持 status 过滤）
    POST /api/data/anomalies/{anomaly_id}/fix    —— 触发异常自动修复
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

from data.validation import DataReconciler

logger = logging.getLogger("realtime_server")

# 项目根目录与默认 SQLite 路径（锚定文件位置，不受启动工作目录影响）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB_PATH = str(_PROJECT_ROOT / "data" / "quant_trading.db")

# 模块级 DataReconciler 单例（懒加载）
_default_reconciler: Optional[DataReconciler] = None


def _get_default_reconciler() -> DataReconciler:
    """获取模块级 DataReconciler 单例（首次调用时初始化）。"""
    global _default_reconciler
    if _default_reconciler is None:
        try:
            from data.data_fetcher import DataFetcher

            fetcher = DataFetcher()
        except Exception:  # noqa: BLE001
            fetcher = None
        _default_reconciler = DataReconciler(fetcher=fetcher, db_path=_DEFAULT_DB_PATH)
    return _default_reconciler


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class ReconcileReq(BaseModel):
    """交叉验证请求体。

    Attributes:
        symbols: 待校验标的列表。
        sources: 可选，``{symbol: {源名: 价格}}``；不传时由 reconciler 取 live 数。
        tolerance: 可选，价格偏差容忍率（覆盖默认 0.005）。
    """

    symbols: List[str] = []
    sources: Optional[Dict[str, Dict[str, float]]] = None
    tolerance: Optional[float] = None


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_data_routes(
    app: FastAPI,
    manager: Any,
    symbol_set: set,
    ok: Any,
    err: Any,
    reconciler: Optional[DataReconciler] = None,
) -> None:
    """注册数据校验与对账相关路由。

    Args:
        app: FastAPI 实例。
        manager: 实时数据管理器（可传 None/mock，本模块主要走 reconciler）。
        symbol_set: 股票池标的集合。
        ok / err: server.py 的统一响应封装。
        reconciler: 可选注入的 :class:`DataReconciler`（测试隔离用）；缺省使用模块级单例。
    """
    rc: DataReconciler = reconciler if reconciler is not None else _get_default_reconciler()

    @app.post("/api/data/reconcile")
    async def reconcile(req: ReconcileReq):
        """对给定标的执行多源价格交叉验证 + 全量异常检测。

        请求体见 :class:`ReconcileReq`。
        """
        symbols = [s for s in (req.symbols or []) if s]
        if not symbols:
            return err(40001, "至少选择 1 只标的")

        per_symbol: Dict[str, Any] = {}
        for sym in symbols:
            entry: Dict[str, Any] = {}
            # 1) 价格交叉验证
            src_prices = None
            if req.sources and sym in req.sources:
                src_prices = req.sources[sym]
            try:
                entry["price_check"] = rc.reconcile_prices(
                    sym, sources=src_prices, tolerance=req.tolerance
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("价格交叉验证失败: %s", sym)
                entry["price_check"] = {"status": "error", "message": str(e)}

            # 2) K线异常检测（有 fetcher 时取 live K线；否则跳过并说明）
            try:
                df = None
                if rc.fetcher is not None:
                    df = rc.fetcher.get_klines(sym, count=120)
                if df is not None and not getattr(df, "empty", True):
                    anoms = rc.detect_anomalies(sym, df)
                    entry["anomalies"] = [a.to_dict() for a in anoms]
                else:
                    entry["anomalies"] = []
                    entry["anomaly_note"] = "未注入 fetcher，跳过 K线异常检测"
            except Exception as e:  # noqa: BLE001
                logger.exception("异常检测失败: %s", sym)
                entry["anomalies"] = []
                entry["anomaly_note"] = f"异常检测失败: {e}"

            per_symbol[sym] = entry

        return ok({"per_symbol": per_symbol})

    @app.get("/api/data/quality")
    async def quality(
        symbol: Optional[str] = Query(default=None),
        date: Optional[str] = Query(default=None),
    ):
        """返回数据质量评分与最新每日报告。

        Args:
            symbol: 可选，只看某标的；缺省时对当前异常涉及的全部标的评分。
            date: 可选，报告日期（仅回显，真实历史报告待接入）。
        """
        symbols: List[str]
        if symbol:
            symbols = [symbol]
        else:
            seen = {a["symbol"] for a in rc.get_anomalies()}
            symbols = sorted(seen) if seen else []
        scores = {s: rc.quality_score(s) for s in symbols}
        report = rc.daily_quality_report(symbols, scores=scores) if symbols else {
            "date": date, "per_symbol": {}, "anomaly_summary": {}, "fix_summary": {}
        }
        return ok({"symbols": symbols, "scores": scores, "report": report})

    @app.get("/api/data/anomalies")
    async def list_anomalies(
        status: Optional[str] = Query(default=None),
    ):
        """返回异常列表，支持按 status（open/fixed）过滤。"""
        return ok(rc.get_anomalies(status=status))

    @app.post("/api/data/anomalies/{anomaly_id}/fix")
    async def fix_anomaly(anomaly_id: str):
        """触发异常自动修复（备用源/备用数据补全）。"""
        try:
            result = rc.fix_anomaly(anomaly_id)
        except KeyError:
            return err(40401, f"异常不存在: {anomaly_id}", http_status=404)
        except Exception as e:  # noqa: BLE001
            logger.exception("异常修复失败: %s", anomaly_id)
            return err(50000, f"修复失败: {e}", http_status=500)

        if result.get("status") != "fixed":
            # 备用源也缺数据：返回 400 说明，不假装成功
            return err(40002, result.get("message", "无法修复"), http_status=400)
        return ok(result)
