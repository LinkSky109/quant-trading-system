"""技术指标计算 API 路由（扩展模块）。

本文件不直接被 server.py 运行，按 server.py 中现有扩展路由的注册模式挂载：

    # 在 server.py 的 _register_extension_routes() 中追加：
    try:
        from _routes_indicators import register_indicator_routes
        register_indicator_routes(app, manager, SYMBOL_SET, ok, err)
        logger.info("技术指标路由已注册: /api/indicators/*")
    except Exception as e:
        logger.warning("技术指标路由注册失败: %s", e)

端点：
    GET  /api/indicators/list     —— 可用指标列表（含分类/参数默认值/输出列）
    POST /api/indicators/calculate —— 计算指定标的的指定指标
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel

from indicators.technical import TechnicalIndicators

logger = logging.getLogger("realtime_server")

# 模块级门面实例（无状态，可复用）
_TI = TechnicalIndicators()


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class IndicatorCalculateReq(BaseModel):
    """计算指定指标的请求体。"""

    symbol: str = "600519.SH"
    indicator: str = "adx"
    # 指标参数覆盖，如 {"period": 20}；为空则使用默认参数
    params: Dict[str, Any] = {}
    # 可选日期过滤（YYYY-MM-DD），不传则使用全部K线
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# ---------------------------------------------------------------------------
# 序列化工具
# ---------------------------------------------------------------------------


def _series_to_list(s: pd.Series) -> List[Optional[float]]:
    """pd.Series -> JSON 友好列表（NaN -> None）。"""
    return [None if pd.isna(v) else float(v) for v in s.tolist()]


def _result_to_payload(result: Any) -> Dict[str, List[Optional[float]]]:
    """指标计算结果（Series 或 DataFrame）-> {列名: [值...]}。"""
    if isinstance(result, pd.DataFrame):
        return {col: _series_to_list(result[col]) for col in result.columns}
    name = result.name or "value"
    return {name: _series_to_list(result)}


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_indicator_routes(
    app: FastAPI,
    manager: Any,
    symbol_set: set,
    ok: Any,
    err: Any,
) -> None:
    """注册技术指标相关路由。

    Args:
        app: FastAPI 实例。
        manager: 实时数据管理器（manager.get(symbol).klines 获取K线）。
        symbol_set: 股票池标的集合（用于合法性校验）。
        ok: server.py 的成功响应封装函数。
        err: server.py 的错误响应封装函数。
    """
    from data.data_fetcher import normalize_symbol

    @app.get("/api/indicators/list")
    async def list_indicators():
        """返回可用技术指标列表（含分类、参数默认值、输出列说明）。"""
        items = _TI.list_indicators()
        return ok({"count": len(items), "indicators": items})

    @app.post("/api/indicators/calculate")
    async def calculate_indicator(req: IndicatorCalculateReq):
        """计算指定标的的指定技术指标。

        请求体:
            symbol: 标的代码（须在股票池内）。
            indicator: 指标名称，见 GET /api/indicators/list。
            params: 指标参数覆盖（可选）。
            start_date / end_date: 可选日期过滤。
        """
        sym = normalize_symbol(req.symbol)
        if sym not in symbol_set:
            return err(40001, f"标的不在股票池内: {req.symbol}")

        valid_names = {item["name"] for item in _TI.list_indicators()}
        if req.indicator not in valid_names:
            return err(
                40003,
                f"未知指标: {req.indicator}，可用: {sorted(valid_names)}",
            )

        try:
            sim = manager.get(sym)
            df = sim.klines.copy()
            if df is None or df.empty:
                return err(40004, f"标的 {sym} 暂无K线数据")

            if req.start_date:
                df = df.loc[df.index >= pd.Timestamp(req.start_date)]
            if req.end_date:
                df = df.loc[df.index <= pd.Timestamp(req.end_date)]
            if len(df) < 30:
                return err(40006, "所选数据区间有效K线不足(<30根)")

            result = _TI.calculate(df, req.indicator, **(req.params or {}))
            payload = _result_to_payload(result)

            return ok({
                "symbol": getattr(sim, "symbol", sym),
                "indicator": req.indicator,
                "params": req.params or {},
                "dates": [d.strftime("%Y-%m-%d") for d in df.index],
                "data": payload,
            })
        except KeyError as e:
            return err(40003, f"未知指标: {e}")
        except ValueError as e:
            return err(40005, f"指标参数错误: {e}")
        except Exception as e:  # 与 server.py 错误口径保持一致，不暴露堆栈
            logger.exception("指标计算失败 %s %s", sym, req.indicator)
            return err(50000, f"指标计算失败: {e}")
