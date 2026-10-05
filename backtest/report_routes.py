"""回测 HTML 报告生成与下载路由。

本模块提供三个端点的注册工厂：

- ``POST /api/backtest/report`` —— 运行回测并生成专业 HTML 报告，
  返回报告文件名与下载 URL。
- ``GET  /api/reports/list`` —— 列出 ``output/reports/`` 下已生成的报告。
- ``GET  /api/reports/{report_id}/download`` —— 下载指定报告 HTML 文件。

设计要点：
- **不修改** ``web-dashboard/server.py``。由 server.py 在启动时调用
  :func:`register_report_routes` 注入依赖（app、manager、SYMBOL_SET 等）。
- 核心列表 / 路径解析逻辑抽成纯函数（:func:`list_reports` /
  :func:`resolve_report_path`），便于在无 FastAPI 环境下单测。
- 报告统一保存到 ``output/reports/<symbol>_<strategy>_<ts>.html``。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# 默认报告输出目录（项目根 / output / reports）
_DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "output" / "reports"

# 报告文件后缀
_REPORT_SUFFIX = ".html"


# ---------------------------------------------------------------------------
# 依赖容器
# ---------------------------------------------------------------------------
@dataclass
class ReportRouteDeps:
    """路由运行所需的外部依赖（由 server.py 注入）。

    Attributes:
        project_root: 项目根目录。
        reports_dir: 报告输出目录（默认为 ``<project_root>/output/reports``）。
        symbol_set: 允许交易的标的集合（如 ``{"600519.SH", ...}``）。
        strategy_names: 允许的策略名列表。
        normalize_symbol: 标的规范化函数。
        run_backtest: 运行回测并返回 ``BacktestResult`` 的可调用对象，
            签名 ``(symbol, strategy, start_date, end_date, use_jev) -> BacktestResult``。
        ok: 统一成功响应封装 ``(data, message) -> JSONResponse/dict``。
        err: 统一错误响应封装 ``(code, message, http_status) -> JSONResponse``。
    """

    project_root: Path
    reports_dir: Path = field(default_factory=lambda: _DEFAULT_REPORTS_DIR)
    symbol_set: set = field(default_factory=set)
    strategy_names: list = field(default_factory=list)
    normalize_symbol: Optional[Callable[[str], str]] = None
    run_backtest: Optional[Callable[..., Any]] = None
    ok: Optional[Callable[..., Any]] = None
    err: Optional[Callable[..., Any]] = None


# ---------------------------------------------------------------------------
# 请求体模型（延迟导入 pydantic，避免无 FastAPI 环境时崩溃）
# ---------------------------------------------------------------------------
def _build_report_req_model():
    """延迟构建 Pydantic 模型，便于在无 FastAPI 环境下导入本模块。"""
    try:
        from pydantic import BaseModel  # noqa: WPS433
    except Exception:  # pragma: no cover - 无 pydantic 时退化为普通 dict
        return dict

    class ReportReq(BaseModel):
        """生成回测报告请求体。"""

        symbol: str = "600519.SH"
        strategy: str = "ma_cross"
        start_date: str = "2024-01-02"
        end_date: str = "2025-12-31"
        use_jev: bool = False

    return ReportReq


ReportReq = _build_report_req_model()


# ---------------------------------------------------------------------------
# 纯函数：列表 / 路径解析（便于单测）
# ---------------------------------------------------------------------------
def list_reports(reports_dir: Path) -> List[Dict[str, Any]]:
    """列出 ``reports_dir`` 下所有 ``*.html`` 报告。

    Args:
        reports_dir: 报告目录。

    Returns:
        报告元信息列表，按修改时间倒序：
        ``[{"report_id": 文件名(不含扩展名), "filename": 文件名,
            "size_bytes": 字节数, "modified_at": ISO 时间}]``。
        目录不存在时返回空列表。
    """
    reports_dir = Path(reports_dir)
    if not reports_dir.exists() or not reports_dir.is_dir():
        return []
    items: List[Dict[str, Any]] = []
    for p in sorted(reports_dir.glob(f"*{_REPORT_SUFFIX}")):
        try:
            stat = p.stat()
            items.append({
                "report_id": p.stem,
                "filename": p.name,
                "size_bytes": int(stat.st_size),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            })
        except OSError:  # pragma: no cover - 文件被并发删除
            continue
    # 按修改时间倒序
    items.sort(key=lambda x: x["modified_at"], reverse=True)
    return items


def resolve_report_path(reports_dir: Path, report_id: str) -> Optional[Path]:
    """根据 report_id 解析报告文件的绝对路径。

    防止路径穿越：report_id 仅允许文件名 stem，拒绝含 ``/`` 或 ``..`` 的输入。

    Args:
        reports_dir: 报告目录。
        report_id: 报告 ID（即文件名不含 ``.html`` 的 stem）。

    Returns:
        存在且位于 ``reports_dir`` 内的 Path；否则返回 None。
    """
    if not report_id:
        return None
    # 拒绝路径分隔符与 ..
    if "/" in report_id or "\\" in report_id or ".." in report_id:
        return None
    reports_dir = Path(reports_dir).resolve()
    candidate = (reports_dir / f"{report_id}{_REPORT_SUFFIX}").resolve()
    # 确保仍在 reports_dir 内
    try:
        candidate.relative_to(reports_dir)
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    return None


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------
def register_report_routes(app: Any, deps: ReportRouteDeps) -> None:
    """在 FastAPI ``app`` 上注册报告相关路由。

    Args:
        app: FastAPI 实例。
        deps: 路由依赖容器。
    """
    # 延迟导入，避免 server.py 未启动时强依赖 fastapi
    from fastapi.responses import FileResponse  # noqa: WPS433

    reports_dir = Path(deps.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    def _normalize(symbol: str) -> str:
        if deps.normalize_symbol is not None:
            return deps.normalize_symbol(symbol)
        return symbol

    @app.post("/api/backtest/report")
    async def generate_backtest_report(req: "ReportReq"):  # type: ignore[valid-type]
        """运行回测并生成专业 HTML 报告。

        Body:
            symbol / strategy / start_date / end_date / use_jev。

        Returns:
            ``{"report_id": ..., "filename": ..., "download_url": ..., "size_bytes": ...}``
        """
        # 统一成 dict（pydantic 模型或降级后的 dict 都能处理）
        body = req if isinstance(req, dict) else req.model_dump()

        symbol = _normalize(str(body.get("symbol", "")))
        strategy = str(body.get("strategy", ""))
        start_date = str(body.get("start_date", ""))
        end_date = str(body.get("end_date", ""))
        use_jev = bool(body.get("use_jev", False))

        if deps.symbol_set and symbol not in deps.symbol_set:
            return deps.err(40001, f"标的不在股票池内: {symbol}")
        if deps.strategy_names and strategy not in deps.strategy_names:
            return deps.err(40002, f"未知策略: {strategy}，可选: {deps.strategy_names}")
        if deps.run_backtest is None:
            return deps.err(50000, "回测执行器未注入")

        try:
            result = deps.run_backtest(symbol, strategy, start_date, end_date, use_jev)
        except Exception as e:  # pragma: no cover - 回测异常透传
            logger.exception("回测报告生成失败")
            return deps.err(50001, f"回测失败: {e}", http_status=500)

        # 延迟导入 ReportGenerator（避免循环依赖）
        from backtest.report_generator import ReportGenerator  # noqa: WPS433

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{strategy}_{ts}{_REPORT_SUFFIX}"
        out_path = reports_dir / filename

        title = f"回测报告 - {symbol} - {strategy}"
        params = {
            "标的": symbol,
            "策略": strategy,
            "开始日期": start_date,
            "结束日期": end_date,
            "Jev信号过滤": "是" if use_jev else "否",
        }
        try:
            html_path = ReportGenerator().generate_html_report(
                result, str(out_path), title=title, params=params,
            )
        except Exception as e:  # pragma: no cover
            logger.exception("HTML 报告渲染失败")
            return deps.err(50002, f"报告渲染失败: {e}", http_status=500)

        size = Path(html_path).stat().st_size
        report_id = Path(html_path).stem
        logger.info("回测报告已导出: %s", html_path)
        return deps.ok({
            "report_id": report_id,
            "filename": Path(html_path).name,
            "download_url": f"/api/reports/{report_id}/download",
            "size_bytes": int(size),
        })

    @app.get("/api/reports/list")
    async def list_generated_reports():
        """列出 ``output/reports/`` 下已生成的报告。"""
        return deps.ok({"reports": list_reports(reports_dir)})

    @app.get("/api/reports/{report_id}/download")
    async def download_report(report_id: str):
        """下载指定报告 HTML 文件。

        Args:
            report_id: 报告 ID（文件名 stem，不含 .html）。
        """
        path = resolve_report_path(reports_dir, report_id)
        if path is None:
            return deps.err(40404, f"报告不存在: {report_id}", http_status=404)
        return FileResponse(
            path,
            media_type="text/html; charset=utf-8",
            filename=path.name,
        )
