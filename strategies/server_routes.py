"""事件驱动策略 API 路由（可插拔 snippet）。

本模块**不修改** ``web-dashboard/server.py``，而是暴露
:func:`register_event_routes`，在 server 启动后一行挂载：

.. code-block:: python

    from strategies.server_routes import register_event_routes
    register_event_routes(app)

提供三个接口：
    - ``POST /api/event/detect``  检测事件
    - ``POST /api/event/study``    事件研究（CAR 曲线 + t 检验）
    - ``POST /api/event/backtest`` 事件驱动策略回测
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel

from backtest.engine import BacktestEngine
from config import load_config
from data.data_fetcher import DataFetcher, normalize_symbol
from strategies.event_driven import (
    EXTERNAL_EVENT_TYPES,
    KLINE_EVENT_TYPES,
    EventDrivenStrategy,
    SUPPORTED_MODES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic 请求体模型（模块级，确保 FastAPI 正确识别为 body 参数）
# ---------------------------------------------------------------------------

class EventDetectReq(BaseModel):
    """事件检测请求。"""
    symbol: str
    event_types: Optional[List[str]] = None  # None=全部K线事件


class EventStudyReq(BaseModel):
    """事件研究请求。"""
    symbol: str
    event_type: str = "limit_up"
    window: Optional[int] = None


class EventBacktestReq(BaseModel):
    """事件驱动回测请求。"""
    symbol: str
    mode: str = "pead"
    hold_days: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    count: int = 500

#: 项目根目录（server_routes 位于 strategies/ 下，上溯两级）
_PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def _ok(data: Any = None, message: str = "success") -> Dict[str, Any]:
    """成功响应统一封装。"""
    return {"code": 0, "message": message, "data": data}


def _err(code: int, message: str, http_status: int = 400) -> Dict[str, Any]:
    """错误响应（与 server.py 格式一致）。"""
    return {"code": code, "message": message, "data": None,
            "_http_status": http_status}


def _load_cfg() -> Dict[str, Any]:
    """加载主配置。"""
    try:
        return load_config(str(_PROJECT_ROOT / "config" / "config.yaml"))
    except Exception:  # pragma: no cover - 配置加载失败时降级空字典
        logger.warning("配置加载失败，使用默认参数", exc_info=True)
        return {}


def _get_fetcher(cfg: Dict[str, Any]) -> DataFetcher:
    """构造 DataFetcher。"""
    data_cfg = cfg.get("data", {})
    return DataFetcher(
        api_key=data_cfg.get("api_key", ""),
        cache_dir=data_cfg.get("cache_dir", str(_PROJECT_ROOT / "cache")),
        cache_ttl_hours=float(data_cfg.get("cache_ttl_hours", 4)),
    )


def _fetch_klines(fetcher: DataFetcher, symbol: str,
                 count: int = 500) -> pd.DataFrame:
    """拉取日线数据。"""
    return fetcher.get_klines(symbol, period="1d", count=count,
                              adjust="qfq", use_cache=True)


def register_event_routes(app: Any) -> None:
    """在 FastAPI app 上注册事件驱动策略路由。

    Args:
        app: FastAPI 实例。
    """
    cfg = _load_cfg()
    es_cfg = cfg.get("event_strategy", {})
    fetcher = _get_fetcher(cfg)

    # ------------------------------------------------------------------
    # POST /api/event/detect
    # ------------------------------------------------------------------

    @app.post("/api/event/detect")
    async def api_event_detect(req: EventDetectReq) -> Dict[str, Any]:
        """检测标的上的市场事件。

        body: ``{symbol, event_types?: [...]}``
        """
        sym = normalize_symbol(req.symbol)
        try:
            df = _fetch_klines(fetcher, sym)
        except Exception as e:
            logger.exception("拉取K线失败: %s", sym)
            return _err(50001, f"拉取K线失败: {e}")

        strategy = EventDrivenStrategy({
            "limit_up_threshold": es_cfg.get("limit_up_threshold", 0.098),
            "volume_spike_multiplier": es_cfg.get("volume_spike_multiplier", 2.0),
            "price_gap_threshold": es_cfg.get("price_gap_threshold", 0.02),
        })
        events = strategy.detect_events(df, symbol=sym)
        external = strategy.load_external_events(sym)

        # 过滤事件类型
        wanted = set(req.event_types) if req.event_types else \
            set(KLINE_EVENT_TYPES) | set(EXTERNAL_EVENT_TYPES)
        filtered = [e for e in events if e["event_type"] in wanted]
        filtered += [e for e in external if e["event_type"] in wanted]
        filtered.sort(key=lambda e: e["date"])

        return _ok({
            "symbol": sym,
            "count": len(filtered),
            "events": [
                {
                    "date": pd.Timestamp(e["date"]).strftime("%Y-%m-%d"),
                    "event_type": e["event_type"],
                    "symbol": e["symbol"],
                    "metadata": e.get("metadata", {}),
                }
                for e in filtered
            ],
        })

    # ------------------------------------------------------------------
    # POST /api/event/study
    # ------------------------------------------------------------------

    @app.post("/api/event/study")
    async def api_event_study(req: EventStudyReq) -> Dict[str, Any]:
        """对指定事件类型做事件研究，返回 CAR 曲线与 t 检验。

        body: ``{symbol, event_type, window?}``
        """
        sym = normalize_symbol(req.symbol)
        window = int(req.window if req.window is not None
                     else es_cfg.get("event_study_window", 20))
        event_type = req.event_type

        try:
            df = _fetch_klines(fetcher, sym)
        except Exception as e:
            logger.exception("拉取K线失败: %s", sym)
            return _err(50001, f"拉取K线失败: {e}")

        strategy = EventDrivenStrategy({
            "limit_up_threshold": es_cfg.get("limit_up_threshold", 0.098),
            "volume_spike_multiplier": es_cfg.get("volume_spike_multiplier", 2.0),
            "price_gap_threshold": es_cfg.get("price_gap_threshold", 0.02),
        })
        all_events = strategy.detect_events(df, symbol=sym)
        events = [e for e in all_events if e["event_type"] == event_type]

        result = strategy.event_study(df, events, window=window)
        car_series = result["car_series"]
        return _ok({
            "symbol": sym,
            "event_type": event_type,
            "window": window,
            "event_count": result["event_count"],
            "t_statistic": result["t_statistic"],
            "p_value": result["p_value"],
            "car_curve": [
                {"offset": int(off),
                 "car": round(float(val), 6),
                 "std": round(float(result["car_std"].loc[off]), 6)}
                for off, val in car_series.items()
            ] if not car_series.empty else [],
        })

    # ------------------------------------------------------------------
    # POST /api/event/backtest
    # ------------------------------------------------------------------

    @app.post("/api/event/backtest")
    async def api_event_backtest(req: EventBacktestReq) -> Dict[str, Any]:
        """运行事件驱动策略回测。

        body: ``{symbol, mode, hold_days?, start_date?, end_date?, count?}``
        """
        sym = normalize_symbol(req.symbol)
        if req.mode not in SUPPORTED_MODES:
            return _err(40002,
                        f"未知模式: {req.mode}，可选: {list(SUPPORTED_MODES)}")

        hold_days = int(req.hold_days if req.hold_days is not None
                        else es_cfg.get("default_hold_days", 5))
        try:
            df = _fetch_klines(fetcher, sym, count=req.count)
        except Exception as e:
            logger.exception("拉取K线失败: %s", sym)
            return _err(50001, f"拉取K线失败: {e}")

        if req.start_date:
            df = df[df.index >= pd.Timestamp(req.start_date)]
        if req.end_date:
            df = df[df.index <= pd.Timestamp(req.end_date)]

        if len(df) < 30:
            return _err(40003, f"有效K线不足30条（当前 {len(df)} 条）")

        strategy = EventDrivenStrategy({
            "mode": req.mode,
            "hold_days": hold_days,
            "limit_up_threshold": es_cfg.get("limit_up_threshold", 0.098),
            "volume_spike_multiplier": es_cfg.get("volume_spike_multiplier", 2.0),
            "price_gap_threshold": es_cfg.get("price_gap_threshold", 0.02),
        })

        bt_cfg = cfg.get("backtest", {})
        engine = BacktestEngine(
            initial_capital=float(bt_cfg.get("initial_capital", 1_000_000.0)),
            commission_rate=float(bt_cfg.get("commission_rate", 0.00025)),
            stamp_tax_rate=float(bt_cfg.get("stamp_tax_rate", 0.0005)),
            slippage_rate=float(bt_cfg.get("slippage_rate", 0.001)),
            risk_free_rate=float(bt_cfg.get("risk_free_rate", 0.02)),
            trading_days=int(bt_cfg.get("trading_days_per_year", 252)),
        )
        result = engine.run(df, strategy, symbol=sym)

        return _ok({
            "symbol": sym,
            "mode": req.mode,
            "hold_days": hold_days,
            "metrics": result.metrics,
            "equity_curve": [
                [d.strftime("%Y-%m-%d"), round(float(v), 2)]
                for d, v in result.equity_curve.items()
            ],
            "trades": [
                {
                    "date": t.date.strftime("%Y-%m-%d"),
                    "action": t.action,
                    "price": round(float(t.price), 2),
                    "shares": int(t.shares),
                    "pnl": round(float(t.pnl), 2) if t.pnl is not None else None,
                }
                for t in result.trades
            ],
        })

    logger.info("事件驱动策略路由已注册: /api/event/{detect,study,backtest}")
