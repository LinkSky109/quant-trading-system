"""回测走查（walkthrough）路由集成代码片段。

本文件**不被 server.py 直接导入**，而是提供可直接粘贴到 ``web-dashboard/server.py``
的代码块。按注释分隔逐段复制即可。

假设 server.py 中已存在以下符号（见现有代码）：
  - app            : FastAPI 实例
  - ok() / err()   : 统一响应封装
  - manager        : MultiSymbolManager，manager.get(symbol).klines 为 K 线 DataFrame
  - _strategy_instance(name) : 策略工厂
  - STRATEGY_NAMES : 允许的策略名列表
  - pd             : pandas

粘贴顺序：
  1) 在文件顶部 import 区追加「== 1. IMPORTS ==」
  2) 在合适位置（如 BacktestReq 附近）追加「== 2. PYDANTIC 模型 ==」
  3) 在全局区追加「== 3. 走查缓存管理器 ==」
  4) 在路由区追加「== 4. 路由处理器 ==」
"""
from __future__ import annotations

from typing import Dict, Optional  # noqa: F401  (供全局注解使用)

# =============================================================================
# == 1. IMPORTS == （粘贴到 server.py 顶部 import 区）
# =============================================================================
# from typing import Any, Dict, List, Optional
# from fastapi import Query
# from backtest.walkthrough import BacktestWalkthrough


# =============================================================================
# == 2. PYDANTIC 模型 == （粘贴到其他 *Req 模型附近，如 BacktestReq 之后）
# =============================================================================
# class WalkthroughReq(BaseModel):
#     """运行一次回测走查的请求体。"""
#     symbol: str = "600519.SH"
#     strategy: str = "ma_cross"
#     start_date: str = "2024-01-02"
#     end_date: str = "2025-12-31"
#     initial_capital: float = 1_000_000.0
#     use_jev: bool = False


# =============================================================================
# == 3. 走查缓存管理器 == （粘贴为模块级全局对象）
# =============================================================================
# 简单内存缓存：{walkthrough_id: BacktestWalkthrough 实例}
_WALKTHROUGH_CACHE: Dict[str, BacktestWalkthrough] = {}
_WALKTHROUGH_MAX = 50  # 最多保留最近 50 次走查，避免内存膨胀
# 可选：如需持久化到 SQLite，传入数据库文件路径（None 表示仅内存）
_WALKTHROUGH_DB_PATH: Optional[str] = None


def _walkthrough_put(wt: BacktestWalkthrough) -> None:
    """登记一次走查结果（超出容量时淘汰最早的）。"""
    if wt.walkthrough_id is None:
        return
    _WALKTHROUGH_CACHE[wt.walkthrough_id] = wt
    # 简单 LRU：超出上限时删除最早插入的键
    while len(_WALKTHROUGH_CACHE) > _WALKTHROUGH_MAX:
        _WALKTHROUGH_CACHE.pop(next(iter(_WALKTHROUGH_CACHE)))


def _walkthrough_get(walkthrough_id: str) -> Optional[BacktestWalkthrough]:
    return _WALKTHROUGH_CACHE.get(walkthrough_id)


# =============================================================================
# == 4. 路由处理器 == （粘贴到其他 @app 路由附近）
# =============================================================================

# -----------------------------------------------------------------------------
# POST /api/walkthrough/run —— 运行走查回测，返回 walkthrough_id
# -----------------------------------------------------------------------------
# @app.post("/api/walkthrough/run")
# async def walkthrough_run(req: WalkthroughReq):
#     if req.strategy not in STRATEGY_NAMES:
#         return err(40002, f"未知策略: {req.strategy}，可选: {STRATEGY_NAMES}")
#
#     sim = manager.get(req.symbol)
#     if sim is None or sim.klines is None or sim.klines.empty:
#         return err(40404, f"无行情数据: {req.symbol}")
#
#     df = sim.klines.copy()
#     df = df.loc[(df.index >= pd.Timestamp(req.start_date)) &
#                 (df.index <= pd.Timestamp(req.end_date))]
#     if len(df) < 30:
#         return err(40010, "区间内K线不足30根，无法走查")
#
#     try:
#         wt = BacktestWalkthrough(
#             symbol=req.symbol,
#             strategy_name=req.strategy,
#             start_date=req.start_date,
#             end_date=req.end_date,
#             initial_capital=req.initial_capital,
#             use_jev=req.use_jev,
#             db_path=_WALKTHROUGH_DB_PATH,
#         )
#         walkthrough_id = wt.run(df)
#         _walkthrough_put(wt)
#     except Exception as e:
#         logger.exception("走查回测失败")
#         return err(50000, f"走查回测失败: {e}")
#
#     return ok({
#         "walkthrough_id": walkthrough_id,
#         "summary": wt.to_dict(),
#     })


# -----------------------------------------------------------------------------
# GET /api/walkthrough/{id}/day?date=YYYY-MM-DD
# -----------------------------------------------------------------------------
# @app.get("/api/walkthrough/{walkthrough_id}/day")
# async def walkthrough_day(
#     walkthrough_id: str,
#     date: str = Query(..., description="YYYY-MM-DD"),
# ):
#     wt = _walkthrough_get(walkthrough_id)
#     if wt is None:
#         return err(40404, f"走查不存在或已过期: {walkthrough_id}")
#     day = wt.get_day(date)
#     if day is None:
#         return err(40404, f"该日无快照: {date}")
#     return ok(day)


# -----------------------------------------------------------------------------
# GET /api/walkthrough/{id}/range?start=&end=
# -----------------------------------------------------------------------------
# @app.get("/api/walkthrough/{walkthrough_id}/range")
# async def walkthrough_range(
#     walkthrough_id: str,
#     start: str = Query(..., description="YYYY-MM-DD"),
#     end: str = Query(..., description="YYYY-MM-DD"),
# ):
#     wt = _walkthrough_get(walkthrough_id)
#     if wt is None:
#         return err(40404, f"走查不存在或已过期: {walkthrough_id}")
#     snapshots = wt.get_range(start, end)
#     return ok({
#         "start": start, "end": end,
#         "count": len(snapshots),
#         "snapshots": snapshots,
#     })


# -----------------------------------------------------------------------------
# GET /api/walkthrough/{id}/trades
# -----------------------------------------------------------------------------
# @app.get("/api/walkthrough/{walkthrough_id}/trades")
# async def walkthrough_trades(walkthrough_id: str):
#     wt = _walkthrough_get(walkthrough_id)
#     if wt is None:
#         return err(40404, f"走查不存在或已过期: {walkthrough_id}")
#     return ok({
#         "walkthrough_id": walkthrough_id,
#         "trades": wt.get_trades(),
#         "signal_vs_action": wt.get_signal_vs_action(),
#     })
