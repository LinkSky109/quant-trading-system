"""server.py 集成代码片段（System Health Monitor）。

本文件**不被 server.py 直接 import**，而是提供可手工粘贴到
``web-dashboard/server.py`` 的三段代码。按 A -> B -> C 的顺序放置：

    A. 请求统计中间件  —— 放在 ``app = FastAPI(...)`` 之后（约第 49 行后）。
    B. 健康监控实例    —— 放在 ``trader = RealtimeTrader(...)`` 初始化之后
                         （约第 818 行后）。
    C. 路由            —— 放在其它 ``@app.get(...)`` 路由附近
                         （如 ``/api/data_quality`` 路由旁）。

约定：server.py 中已存在的变量/函数直接复用，不要重复定义：
    - ``app``            : FastAPI 实例
    - ``ok(data, msg)``  : 统一响应封装（返回 {"code":0,"message":...,"data":...}）
    - ``db``             : persistence.database.Database 实例
    - ``alert_manager``  : monitoring.alert.AlertManager 实例
    - ``data_quality_monitor`` : monitoring.data_quality.DataQualityMonitor 实例
    - ``trader``         : realtime_trader.RealtimeTrader 实例
    - ``jev_client``     : JevRealClient 实例（取 base_url）
    - ``START_TIME``     : 服务启动时间戳（秒）
    - ``time``           : 已 import
"""

# ===========================================================================
# A. 请求统计中间件（粘贴到 app = FastAPI(...) 之后）
# ===========================================================================
#
# from monitoring.system_health import ApiMetricsCollector
#
# api_metrics = ApiMetricsCollector()
#
# @app.middleware("http")
# async def api_stats_middleware(request, call_next):
#     start = time.time()
#     response = await call_next(request)
#     latency = (time.time() - start) * 1000  # 毫秒
#     endpoint = request.url.path
#     # 只统计 /api/ 路径，排除静态文件、WebSocket、文档页
#     if endpoint.startswith("/api/"):
#         api_metrics.record_request(
#             endpoint, request.method, latency, response.status_code
#         )
#     return response


# ===========================================================================
# B. 全局 SystemHealthMonitor 实例（粘贴到 trader = RealtimeTrader(...) 之后）
# ===========================================================================
#
# from monitoring.system_health import SystemHealthMonitor
#
# health_monitor = SystemHealthMonitor(
#     db=db,
#     data_quality_monitor=data_quality_monitor,
#     alert_manager=alert_manager,
#     realtime_trader=trader,
#     api_metrics=api_metrics,          # 与上面中间件共享同一个收集器
#     jev_base_url=jev_client.base_url,  # http://localhost:8765
#     quant_port=8766,
#     start_time=START_TIME,
# )
# # 启动后台线程，每 30 秒自动采集一次健康报告
# health_monitor.start_auto_collect(interval=30)
#
# # 建议在 shutdown_event() 里补充停止：
# #   async def shutdown_event():
# #       health_monitor.stop_auto_collect()
# #       jev_client.shutdown()


# ===========================================================================
# C. 路由（粘贴到其它 @app.get 路由附近）
# ===========================================================================
#
# @app.get("/api/system/health")
# async def system_health(
#     refresh: bool = Query(False, description="是否强制实时采集（默认读缓存）"),
# ):
#     """全链路系统健康报告。
#
#     聚合服务进程、API 延迟、Jev 服务、数据库、数据质量、交易引擎、告警，
#     并给出 0-100 健康评分与 healthy/warning/critical 级别。
#     """
#     if refresh or health_monitor.get_latest_report() is None:
#         report = await asyncio.to_thread(health_monitor.collect_all)
#     else:
#         report = health_monitor.get_latest_report()
#     return ok(report)
#
#
# @app.get("/api/system/health/metrics")
# async def system_health_metrics():
#     """仅返回 API 性能指标（各端点延迟/错误率），便于前端做图表。"""
#     return ok({
#         "summary": api_metrics.get_summary(),
#         "endpoints": api_metrics.get_stats(),
#     })


# ===========================================================================
# 说明：健康报告字段结构
# ===========================================================================
# {
#   "timestamp": "2026-09-30T22:00:00+08:00",
#   "health_score": 92,            # 0-100
#   "health_level": "healthy",     # healthy / warning / critical (<60)
#   "service_status": {
#       "quant": {"status":"up", "port":8766, "uptime_seconds":123.4,
#                 "memory_mb":45.2, "cpu_percent":3.1, "psutil_available":true},
#       "jev":   {"status":"ok", "port":8765, "reachable":true,
#                 "model_loaded":true, "model":"laya_mlx"}
#   },
#   "api_performance": {
#       "summary": {"total_requests":120, "avg_latency_ms":12.3,
#                   "p95_latency_ms":45.6, "error_rate":0.0},
#       "endpoints": {"GET /api/health": {...}, ...}
#   },
#   "jev_status": {"mode":"real", "reachable":true, "model_loaded":true,
#                  "inference_count":42, "avg_latency_ms":18.5,
#                  "last_decision_time":"..."},
#   "database": {"status":"connected", "size_mb":3.2, "journal_mode":"wal",
#                "table_counts": {"trades":10, "jev_decisions":42,
#                                 "account_snapshots":100, "daily_reports":5}},
#   "data_quality": {"overall_status":"healthy", "checks": {...}, ...},
#   "trading_engine": {"running":true, "position_count":3,
#                      "today_trades":2, "today_pnl":123.45,
#                      "risk_paused":false},
#   "alerts": {"status":"ok", "stats": {"total":3, "critical":0,
#                                       "warning":1, "info":2}}
# }
