"""Jev 训练数据导出 —— server.py 集成代码片段。

本文件**不要直接运行**，其中的代码片段用于粘贴到 ``web-dashboard/server.py``。
它假设 server.py 已存在：
  - ``app``（FastAPI 实例）、``ok()`` / ``err()`` 统一响应封装
  - ``PROJECT_ROOT``（Path，项目根目录）
  - ``logger``

集成步骤：
  1. 在 server.py 顶部 import 区追加下方 ``# ---- 需要的 import ----``。
  2. 把 ``# ---- 路由开始 ----`` 到 ``# ---- 路由结束 ----`` 之间的路由函数
     粘贴到 server.py 中其它路由附近（@app.post / @app.get 装饰器会自动注册）。
  3. 重启服务即可访问。

端点：
  - POST /api/jev/export_training  触发导出，返回文件路径与统计
  - GET  /api/jev/training_stats   读取最近一次导出的统计
"""
from __future__ import annotations

# ===========================================================================
# ---- 需要的 import ----（追加到 server.py 顶部 import 区）
# ===========================================================================
# import json
# from pathlib import Path
# from pydantic import BaseModel
# from typing import Optional
#
# from jev.training_data import TrainingDataExporter


# ===========================================================================
# ---- 请求体模型 ----（可与其它 BaseModel 放一起）
# ===========================================================================
# class TrainingExportReq(BaseModel):
#     start_date: Optional[str] = None
#     end_date: Optional[str] = None
#     symbol: Optional[str] = None
#     strategy: Optional[str] = None
#     forward_days: int = 5
#     hold_threshold: float = 0.02


# ===========================================================================
# ---- 路由开始 ----
# ===========================================================================

# #: 训练数据导出目录（相对项目根目录）
# TRAINING_DATA_DIR = PROJECT_ROOT / "output" / "training_data"
#
#
# @app.post("/api/jev/export_training")
# async def export_jev_training(req: TrainingExportReq):
#     """导出带标签的 Jev 训练数据（JSONL + CSV，按时间切分 train/val/test）。
#
#     返回导出文件路径与统计信息。K线在线拉取失败的标的会被静默跳过。
#     """
#     try:
#         exporter = TrainingDataExporter(
#             hold_threshold=req.hold_threshold,
#             forward_days=req.forward_days,
#         )
#         result = exporter.export_all(
#             output_dir=str(TRAINING_DATA_DIR),
#             start_date=req.start_date,
#             end_date=req.end_date,
#             symbol=req.symbol,
#             strategy=req.strategy,
#         )
#         # 文件路径转成可下载的 URL 路径（前端再拼接主机地址）
#         download_urls = {
#             k: f"/training_data/{Path(v).name}"
#             for k, v in result["file_paths"].items()
#         }
#         return ok({
#             "sample_count": result["sample_count"],
#             "stats": result["stats"],
#             "files": result["file_paths"],
#             "download_urls": download_urls,
#         })
#     except Exception as e:
#         logger.exception("导出 Jev 训练数据失败")
#         return err(50010, f"导出失败: {e}", http_status=500)
#
#
# @app.get("/api/jev/training_stats")
# async def get_jev_training_stats():
#     """返回最近一次训练数据导出的统计信息。"""
#     stats_file = TRAINING_DATA_DIR / "last_export_stats.json"
#     if not stats_file.exists():
#         return ok({"available": False, "message": "尚未导出训练数据"})
#     try:
#         data = json.loads(stats_file.read_text(encoding="utf-8"))
#         return ok({"available": True, **data})
#     except Exception as e:
#         logger.exception("读取训练数据统计失败")
#         return err(50011, f"读取统计失败: {e}", http_status=500)


# ===========================================================================
# ---- 路由结束 ----
# ===========================================================================
#
# 可选：若希望通过 HTTP 下载导出的文件，再加一个静态文件路由：
#
# @app.get("/training_data/{name}")
# async def download_training_file(name: str):
#     safe_name = Path(name).name  # 防目录穿越
#     file_path = TRAINING_DATA_DIR / safe_name
#     if not file_path.exists():
#         return err(40404, "文件不存在", http_status=404)
#     return FileResponse(
#         str(file_path), filename=safe_name,
#         media_type="application/octet-stream",
#     )
