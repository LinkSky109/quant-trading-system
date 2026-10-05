"""另类数据 API 路由（扩展模块 #26，REQ-P3-04）。

端点：
    GET  /api/alternative/sources   —— 查询可用另类数据源列表
    POST /api/alternative/preview   —— 预览指定数据源的 mock 原始数据
    POST /api/alternative/factors   —— 计算另类因子（shift(1) 防未来函数）
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from data.alternative import (
    ALTERNATIVE_FACTOR_NAMES,
    AlternativeFactorEngine,
    DEFAULT_SOURCES,
    get_source,
)

logger = logging.getLogger("realtime_server")


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class AlternativePreviewReq(BaseModel):
    """另类数据预览请求。"""

    source: str
    symbol: str = "600519.SH"
    days: int = 60


class AlternativeFactorsReq(BaseModel):
    """另类因子计算请求。"""

    symbol: str
    days: int = 250
    lag: int = 1


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_alternative_routes(
    app: FastAPI,
    manager: Any,
    symbol_set: List[str],
    ok: Any,
    err: Any,
) -> None:
    """注册另类数据路由到 FastAPI app。"""

    @app.get("/api/alternative/sources")
    async def alternative_sources():
        """查询可用另类数据源列表。"""
        try:
            sources = [src.meta() for src in DEFAULT_SOURCES.values()]
            return ok({
                "sources": sources,
                "factors": ALTERNATIVE_FACTOR_NAMES,
                "real_source_reserved": True,  # 真实数据源接口已预留
            })
        except Exception as e:
            logger.exception("查询另类数据源失败")
            return err(500, f"查询失败: {e}")

    @app.post("/api/alternative/preview")
    async def alternative_preview(req: AlternativePreviewReq):
        """预览指定数据源的 mock 原始数据。"""
        try:
            symbol = req.symbol.strip().upper()
            if not symbol:
                return err(400, "symbol 不能为空")
            if req.days <= 0 or req.days > 1500:
                return err(422, "days 须在 1~1500 之间")
            try:
                src = get_source(req.source.strip().lower())
            except KeyError as e:
                return err(422, str(e).strip("'"))

            df = src.fetch(symbol, days=req.days)
            records = []
            for idx, row in df.iterrows():
                records.append({
                    "date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                    **{k: float(v) for k, v in row.items()},
                })
            return ok({
                "source": src.name,
                "symbol": symbol,
                "count": len(records),
                "columns": list(df.columns),
                "data": records[:200],
            })
        except Exception as e:
            logger.exception("预览另类数据失败")
            return err(500, f"预览失败: {e}")

    @app.post("/api/alternative/factors")
    async def alternative_factors(req: AlternativeFactorsReq):
        """计算另类因子（默认 shift(1) 防未来函数）。"""
        try:
            symbol = req.symbol.strip().upper()
            if not symbol:
                return err(400, "symbol 不能为空")
            if req.days <= 0 or req.days > 1500:
                return err(422, "days 须在 1~1500 之间")
            if req.lag < 0:
                return err(422, "lag 须 >= 0")

            engine = AlternativeFactorEngine(lag=req.lag)
            factors = engine.calculate_all(symbol, days=req.days)
            records = []
            for idx, row in factors.iterrows():
                records.append({
                    "date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                    **{k: (None if v != v else float(v)) for k, v in row.items()},
                })
            return ok({
                "symbol": symbol,
                "lag": req.lag,
                "count": len(records),
                "factors": list(factors.columns),
                "data": records[:200],
            })
        except Exception as e:
            logger.exception("计算另类因子失败")
            return err(500, f"计算失败: {e}")
