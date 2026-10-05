"""高频因子 API 路由（扩展模块 #21）。

端点：
    POST /api/highfreq/calculate  —— 对单只标的计算全部高频因子
    GET  /api/highfreq/list       —— 返回支持的高频因子列表
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from factors.high_frequency import HighFrequencyFactorEngine

logger = logging.getLogger("realtime_server")


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class HighFreqCalculateReq(BaseModel):
    """高频因子计算请求。"""

    symbol: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_highfreq_routes(
    app: FastAPI,
    manager: Any,
    symbol_set: List[str],
    ok: Any,
    err: Any,
) -> None:
    """注册高频因子路由到 FastAPI app。"""

    @app.post("/api/highfreq/calculate")
    async def highfreq_calculate(req: HighFreqCalculateReq):
        """计算指定标的的高频因子。"""
        try:
            symbol = req.symbol.strip().upper()
            if not symbol:
                return err(400, "symbol 不能为空")

            # 尝试从 manager 获取 K 线数据
            try:
                df = manager.get_daily_klines(symbol)
                if df is None or df.empty:
                    return err(404, f"未找到 {symbol} 的日K线数据")
            except Exception:
                return err(404, f"未找到 {symbol} 的日K线数据")

            if req.start_date:
                df = df[df.index >= req.start_date]
            if req.end_date:
                df = df[df.index <= req.end_date]

            engine = HighFrequencyFactorEngine()
            result = engine.calculate_all(df)

            # 只返回高频因子列 + close 列
            factor_cols = [c for c in result.columns if c in engine.FACTOR_NAMES] + ["close"]
            payload = result[factor_cols].reset_index().rename(columns={"index": "date"})
            payload = payload.where(pd.notnull(payload), None)
            return ok({
                "symbol": symbol,
                "factors": payload.to_dict(orient="records"),
                "count": len(payload),
            })
        except Exception as e:
            logger.exception("高频因子计算失败")
            return err(500, f"计算失败: {e}")

    @app.get("/api/highfreq/list")
    async def highfreq_list():
        """返回支持的高频因子列表。"""
        engine = HighFrequencyFactorEngine()
        return ok({"factors": engine.get_factor_list()})
