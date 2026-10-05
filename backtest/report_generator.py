"""回测报告 HTML 导出模块。

将 :class:`backtest.engine.BacktestResult` 渲染为自包含的深色主题 HTML 报告：
- 封面（策略名称 / 标的 / 回测区间 / 初始资金 / 生成时间）
- 绩效指标卡片网格（核心指标 + 索提诺 / 卡玛比率）
- 净值曲线图（ECharts 折线，策略净值 vs 基准净值）
- 回撤图（ECharts 面积图，回撤百分比）
- 月度收益热力图（ECharts heatmap，年×月收益矩阵）
- 持仓分析（持仓时长分桶柱状图 + 最大盈亏 / 平均盈亏统计）
- 风险指标（VaR(95%) / CVaR(95%) / 最大连续亏损天数）
- 信号分析（reason 分布 + 各信号胜率）
- 交易明细表
- 参数说明区

报告完全自包含：CSS 内联在 ``<style>`` 中，ECharts 通过 CDN 引入，
图表数据通过内联 ``<script>`` 注入为 JS 变量，生成的单个 ``.html`` 文件
可直接在浏览器打开。同时提供 ``@media print`` 打印样式优化分页。
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult, Trade

logger = logging.getLogger(__name__)

# ECharts CDN（与看板保持一致）
ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"

# 深色主题配色
COLOR_BG = "#0f0f1a"
COLOR_CARD = "#16213e"
COLOR_CARD_BORDER = "#1f2b4d"
COLOR_TEXT = "#e0e0e0"
COLOR_TEXT_MUTED = "#8a93b2"
COLOR_ACCENT = "#00d4ff"   # 策略净值线
COLOR_BENCHMARK = "#9aa5c0"  # 基准净值线
COLOR_DRAWDOWN = "#ff6b6b"  # 回撤面积
COLOR_PROFIT = "#2ecc71"
COLOR_LOSS = "#ff6b6b"

# 需要以百分比形式展示的指标 key
_PERCENT_METRICS = {"累计收益率", "年化收益率", "最大回撤", "胜率"}
# 需要以金额形式展示的指标 key
_MONEY_METRICS = {"总盈利", "总亏损", "最大单笔盈利", "最大单笔亏损", "平均盈利", "平均亏损"}
# 整数展示的指标 key
_INT_METRICS = {"交易次数"}
# 比率类指标（保留 3 位小数）
_RATIO_METRICS = {"夏普比率", "索提诺比率", "卡玛比率"}

# 封面优先展示的参数 key（按顺序）
_COVER_KEYS = [
    "标的", "策略", "开始日期", "结束日期",
    "回测区间", "初始资金", "Jev信号过滤",
]


class ReportGenerator:
    """将回测结果渲染为自包含 HTML 报告。"""

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def generate_html_report(
        self,
        backtest_result: BacktestResult,
        output_path: str,
        title: str = "回测报告",
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """生成自包含 HTML 报告并写入 ``output_path``。

        Args:
            backtest_result: 回测引擎产出的结果容器。
            output_path: 输出 HTML 文件路径（父目录不存在会自动创建）。
            title: 报告标题。
            params: 附加参数（回测参数 / 策略参数），展示在参数说明区与封面。

        Returns:
            生成的 HTML 文件绝对路径字符串。
        """
        params = params or {}
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 准备图表数据
        dates, strategy_vals, benchmark_vals = self._build_curve_data(backtest_result)
        _, drawdown_vals = self._build_drawdown_data(backtest_result)

        # ---- 新增分析数据 ----
        daily_returns = backtest_result.daily_returns
        monthly_pivot = self._build_monthly_returns(daily_returns)
        var_95, cvar_95 = self._calc_var_cvar(daily_returns, confidence=0.95)
        max_consec_losses = self._calc_max_consecutive_losses(daily_returns)
        pos_analysis = self._analyze_positions(backtest_result.trades)
        sig_analysis = self._analyze_signals(backtest_result.trades)

        # 在 metrics 副本上补充索提诺 / 卡玛比率（不修改原 metrics）
        enriched_metrics: Dict[str, float] = dict(backtest_result.metrics or {})
        enriched_metrics["索提诺比率"] = self._calc_sortino_ratio(daily_returns)
        enriched_metrics["卡玛比率"] = self._calc_calmar_ratio(backtest_result.equity_curve)

        # 渲染各区块
        cover_html = self._render_cover(title, generated_at, params)
        cards_html = self._render_metric_cards(enriched_metrics)
        monthly_html = self._render_monthly_section(monthly_pivot)
        position_html = self._render_position_section(pos_analysis)
        risk_html = self._render_risk_section(var_95, cvar_95, max_consec_losses)
        signal_html = self._render_signal_section(sig_analysis)
        trades_html = self._render_trades_table(backtest_result.trades)
        params_html = self._render_params(title, generated_at, params)

        # 注入 JS 数据
        chart_payload: Dict[str, Any] = {
            "dates": dates,
            "strategy": strategy_vals,
            "benchmark": benchmark_vals,
            "drawdown": drawdown_vals,
            # 月度热力图
            "monthly_years": [int(y) for y in monthly_pivot.index],
            "monthly_months": [f"{int(m)}月" for m in monthly_pivot.columns],
            "monthly_data": self._monthly_pivot_to_payload(monthly_pivot),
            "monthly_vmin": self._monthly_vmin(monthly_pivot),
            "monthly_vmax": self._monthly_vmax(monthly_pivot),
            # 持仓分桶
            "holding_bucket_labels": list(pos_analysis["holding_buckets"].keys()),
            "holding_bucket_values": list(pos_analysis["holding_buckets"].values()),
        }

        html = self._render_template(
            title=title,
            generated_at=generated_at,
            cover_html=cover_html,
            cards_html=cards_html,
            monthly_html=monthly_html,
            position_html=position_html,
            risk_html=risk_html,
            signal_html=signal_html,
            trades_html=trades_html,
            params_html=params_html,
            chart_payload=chart_payload,
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)

        abs_path = str(out.resolve())
        logger.info("回测报告已生成: %s", abs_path)
        return abs_path

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------
    @staticmethod
    def _series_to_pairs(series: pd.Series) -> tuple[List[str], List[Optional[float]]]:
        """将 pd.Series（index 为日期）转为 (日期字符串列表, 数值列表)。"""
        if series is None or len(series) == 0:
            return [], []
        dates: List[str] = []
        values: List[Optional[float]] = []
        for idx, val in series.items():
            try:
                dates.append(pd.Timestamp(idx).strftime("%Y-%m-%d"))
            except Exception:  # pragma: no cover - 防御性
                dates.append(str(idx))
            v = float(val)
            values.append(None if math.isnan(v) else round(v, 4))
        return dates, values

    def _build_curve_data(
        self, result: BacktestResult
    ) -> tuple[List[str], List[Optional[float]], List[Optional[float]]]:
        """构建净值曲线数据：以策略净值的日期轴为准对齐基准。"""
        eq_dates, eq_vals = self._series_to_pairs(result.equity_curve)
        # 基准按日期对齐到策略轴（缺失填 None）
        bench_map: Dict[str, float] = {}
        if result.benchmark_curve is not None and len(result.benchmark_curve) > 0:
            for idx, val in result.benchmark_curve.items():
                bench_map[pd.Timestamp(idx).strftime("%Y-%m-%d")] = round(float(val), 4)
        bench_vals: List[Optional[float]] = [bench_map.get(d) for d in eq_dates]
        return eq_dates, eq_vals, bench_vals

    @staticmethod
    def _build_drawdown_data(
        result: BacktestResult,
    ) -> tuple[List[str], List[float]]:
        """根据净值曲线计算回撤序列（百分比，<=0）。"""
        eq = result.equity_curve
        if eq is None or len(eq) == 0:
            return [], []
        try:
            peak = eq.cummax()
            drawdown = (eq - peak) / peak * 100.0
            dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in eq.index]
            vals = [round(float(v), 2) for v in drawdown.values]
            return dates, vals
        except Exception:  # pragma: no cover - 防御性
            return [], []

    # ------------------------------------------------------------------
    # 新增：月度收益 / 风险 / 持仓 / 信号分析
    # ------------------------------------------------------------------
    @staticmethod
    def _build_monthly_returns(daily_returns: pd.Series) -> pd.DataFrame:
        """从日收益率序列构建 年×月 收益率透视表。

        Args:
            daily_returns: 日收益率 pd.Series（index 为日期）。

        Returns:
            DataFrame，index=年份(int)，columns=月份(1..12)，值为该月复利收益率。
            无数据时返回空 DataFrame。
        """
        if daily_returns is None or len(daily_returns) == 0:
            return pd.DataFrame()
        s = daily_returns.dropna()
        if len(s) == 0:
            return pd.DataFrame()
        df = pd.DataFrame({"ret": s.values}, index=pd.DatetimeIndex(s.index))
        df["year"] = df.index.year
        df["month"] = df.index.month
        monthly = df.groupby(["year", "month"])["ret"].apply(
            lambda x: float((1.0 + x).prod() - 1.0)
        )
        pivot = monthly.unstack("month")
        # 补齐 1..12 月列，便于热力图对齐
        for m in range(1, 13):
            if m not in pivot.columns:
                pivot[m] = np.nan
        pivot = pivot[sorted(pivot.columns)]
        pivot.index.name = "year"
        return pivot

    @staticmethod
    def _calc_var_cvar(
        daily_returns: pd.Series, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """历史模拟法 VaR 与 CVaR。

        Args:
            daily_returns: 日收益率序列。
            confidence: 置信水平，默认 0.95（即 95% VaR）。

        Returns:
            (VaR, CVaR) 均为小数（负数表示损失）。无数据时返回 (0.0, 0.0)。
        """
        if daily_returns is None or len(daily_returns) == 0:
            return 0.0, 0.0
        rets = daily_returns.dropna().values
        if len(rets) == 0:
            return 0.0, 0.0
        alpha = (1.0 - confidence) * 100.0  # 例如 5.0
        var = float(np.percentile(rets, alpha))
        tail = rets[rets <= var]
        cvar = float(tail.mean()) if len(tail) > 0 else var
        return var, cvar

    @staticmethod
    def _calc_max_consecutive_losses(daily_returns: pd.Series) -> int:
        """计算最大连续亏损天数（日收益率 < 0 的最长连续段）。

        Args:
            daily_returns: 日收益率序列。

        Returns:
            最大连续亏损交易日数。无数据返回 0。
        """
        if daily_returns is None or len(daily_returns) == 0:
            return 0
        rets = daily_returns.dropna().values
        max_streak = 0
        cur = 0
        for v in rets:
            if v < 0:
                cur += 1
                if cur > max_streak:
                    max_streak = cur
            else:
                cur = 0
        return int(max_streak)

    @staticmethod
    def _analyze_positions(trades: List[Trade]) -> Dict[str, Any]:
        """从交易记录统计持仓分析指标。

        仅统计平仓交易（``pnl is not None``）。

        Args:
            trades: 交易记录列表。

        Returns:
            字典包含：
            - closed_count: 平仓交易笔数
            - holding_buckets: 持仓时长分桶计数
              （1-3天 / 4-7天 / 8-14天 / 15-30天 / 30+天）
            - max_profit / max_loss: 最大单笔盈利 / 亏损
            - avg_profit / avg_loss: 平均盈利 / 平均亏损
            - avg_holding_days: 平均持仓天数
        """
        buckets: Dict[str, int] = {
            "1-3天": 0, "4-7天": 0, "8-14天": 0, "15-30天": 0, "30+天": 0,
        }
        pnls: List[float] = []
        holding_days_list: List[float] = []
        for t in trades:
            if t.pnl is None:
                continue
            pnls.append(float(t.pnl))
            if t.entry_date is not None:
                try:
                    days = (pd.Timestamp(t.date) - pd.Timestamp(t.entry_date)).days
                    holding_days_list.append(float(days))
                    if days <= 3:
                        buckets["1-3天"] += 1
                    elif days <= 7:
                        buckets["4-7天"] += 1
                    elif days <= 14:
                        buckets["8-14天"] += 1
                    elif days <= 30:
                        buckets["15-30天"] += 1
                    else:
                        buckets["30+天"] += 1
                except Exception:  # pragma: no cover - 防御性
                    pass
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        return {
            "closed_count": len(pnls),
            "holding_buckets": buckets,
            "max_profit": float(max(pnls)) if pnls else 0.0,
            "max_loss": float(min(pnls)) if pnls else 0.0,
            "avg_profit": float(np.mean(wins)) if wins else 0.0,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "avg_holding_days": float(np.mean(holding_days_list)) if holding_days_list else 0.0,
        }

    @staticmethod
    def _analyze_signals(trades: List[Trade]) -> Dict[str, Any]:
        """从交易 reason 字段统计信号分布与各信号胜率。

        Args:
            trades: 交易记录列表。

        Returns:
            字典包含：
            - distribution: {信号reason: 出现次数}
            - winrate: {信号reason: 胜率(0..1)}，仅含平仓交易
        """
        dist: Dict[str, int] = {}
        win_by_signal: Dict[str, List[bool]] = {}
        for t in trades:
            reason = (t.reason or "").strip() or "未标注"
            dist[reason] = dist.get(reason, 0) + 1
            if t.pnl is not None:
                win_by_signal.setdefault(reason, []).append(bool(t.pnl > 0))
        signal_winrate: Dict[str, float] = {}
        for sig, results in win_by_signal.items():
            if results:
                signal_winrate[sig] = float(sum(results) / len(results))
        return {"distribution": dist, "winrate": signal_winrate}

    @staticmethod
    def _calc_sortino_ratio(
        daily_returns: pd.Series,
        risk_free_rate: float = 0.02,
        trading_days: int = 252,
    ) -> float:
        """索提诺比率（Sortino Ratio）。

        下行偏差仅统计低于无风险利率的收益段：
        ``sqrt(252) * mean(r - rf/252) / sqrt(mean(min(0, r-rf/252)^2))``

        Args:
            daily_returns: 日收益率序列。
            risk_free_rate: 年化无风险利率。
            trading_days: 年交易日数。

        Returns:
            Sortino 比率；样本不足或下行偏差为 0 时返回 0.0。
        """
        if daily_returns is None or len(daily_returns) < 2:
            return 0.0
        rets = daily_returns.dropna()
        if len(rets) < 2:
            return 0.0
        mar = risk_free_rate / trading_days
        excess = rets - mar
        downside = excess[excess < 0]
        if len(downside) == 0:
            return 0.0
        downside_dev = float(np.sqrt((downside ** 2).mean()))
        if downside_dev == 0:
            return 0.0
        return float(np.sqrt(trading_days) * excess.mean() / downside_dev)

    @staticmethod
    def _calc_calmar_ratio(
        equity: pd.Series, trading_days: int = 252
    ) -> float:
        """卡玛比率（Calmar Ratio）= 年化收益率 / |最大回撤|。

        Args:
            equity: 净值曲线（index 为日期）。
            trading_days: 年交易日数。

        Returns:
            Calmar 比率；最大回撤为 0 时返回 0.0。
        """
        if equity is None or len(equity) < 2:
            return 0.0
        eq = equity.dropna()
        if len(eq) < 2:
            return 0.0
        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        if total_return <= -1.0:
            return -1.0
        n = len(eq)
        ann_return = float((1.0 + total_return) ** (trading_days / n) - 1.0)
        peak = eq.cummax()
        mdd = float(((eq - peak) / peak).min())
        if mdd == 0:
            return 0.0
        return float(ann_return / abs(mdd))

    # ------------------------------------------------------------------
    # 月度透视表 -> ECharts payload 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _monthly_pivot_to_payload(pivot: pd.DataFrame) -> List[List[Optional[float]]]:
        """将年×月透视表转为 ECharts heatmap 的 [xIndex, yIndex, value] 列表。"""
        if pivot is None or pivot.empty:
            return []
        out: List[List[Optional[float]]] = []
        for yi in range(len(pivot.index)):
            for xi in range(len(pivot.columns)):
                v = pivot.iloc[yi, xi]
                if pd.isna(v):
                    out.append([xi, yi, None])
                else:
                    out.append([xi, yi, round(float(v), 4)])
        return out

    @staticmethod
    def _monthly_vmin(pivot: pd.DataFrame) -> float:
        """热力图 visualMap 下界（取最小月收益，向下取整到 1%）。"""
        if pivot is None or pivot.empty:
            return -0.1
        vals = pivot.values.flatten()
        vals = vals[~pd.isna(vals)]
        if len(vals) == 0:
            return -0.1
        return float(np.floor(np.min(vals) * 100) / 100.0)

    @staticmethod
    def _monthly_vmax(pivot: pd.DataFrame) -> float:
        """热力图 visualMap 上界（取最大月收益，向上取整到 1%）。"""
        if pivot is None or pivot.empty:
            return 0.1
        vals = pivot.values.flatten()
        vals = vals[~pd.isna(vals)]
        if len(vals) == 0:
            return 0.1
        return float(np.ceil(np.max(vals) * 100) / 100.0)

    # ------------------------------------------------------------------
    # 区块渲染
    # ------------------------------------------------------------------
    @staticmethod
    def _format_metric(key: str, value: Any) -> str:
        """根据指标 key 格式化展示文本。"""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return str(value)
        if key in _PERCENT_METRICS:
            return f"{v * 100:.2f}%"
        if key in _MONEY_METRICS:
            return f"¥{v:,.2f}"
        if key in _INT_METRICS:
            return f"{int(round(v))}"
        if key in _RATIO_METRICS:
            return f"{v:.3f}"
        if key == "盈亏比":
            return f"{v:.2f}"
        return f"{v:.4f}"

    def _render_metric_cards(self, metrics: Dict[str, float]) -> str:
        """渲染绩效指标卡片网格 HTML。"""
        # 固定展示顺序，取核心指标
        order = [
            "累计收益率", "年化收益率", "最大回撤", "夏普比率",
            "索提诺比率", "卡玛比率",
            "胜率", "盈亏比", "交易次数", "总盈利", "总亏损",
        ]
        keys = [k for k in order if k in metrics]
        # 兼容：若 metrics 有额外 key 则追加
        keys += [k for k in metrics.keys() if k not in keys]

        cards: List[str] = []
        for k in keys:
            display = self._format_metric(k, metrics.get(k, 0.0))
            # 涨跌着色
            color = COLOR_TEXT
            try:
                fv = float(metrics.get(k, 0.0))
                if k in {"累计收益率", "年化收益率", "总盈利", "索提诺比率",
                         "卡玛比率", "夏普比率", "平均盈利"}:
                    color = COLOR_PROFIT if fv > 0 else (COLOR_LOSS if fv < 0 else COLOR_TEXT)
                elif k in {"最大回撤", "总亏损", "最大单笔亏损", "平均亏损"}:
                    color = COLOR_LOSS if fv < 0 else COLOR_TEXT
            except (TypeError, ValueError):
                pass
            cards.append(
                f'<div class="card">'
                f'<div class="card-label">{self._esc(k)}</div>'
                f'<div class="card-value" style="color:{color}">{self._esc(display)}</div>'
                f"</div>"
            )
        return '<div class="cards-grid">' + "".join(cards) + "</div>"

    def _render_cover(
        self, title: str, generated_at: str, params: Dict[str, Any]
    ) -> str:
        """渲染封面区：标题 + 关键信息网格（标的/策略/区间/资金/生成时间）。"""
        items: List[Tuple[str, str]] = []
        for k in _COVER_KEYS:
            if k in params:
                items.append((k, str(params[k])))
        items.append(("生成时间", generated_at))
        grid = "".join(
            f'<div class="cover-item"><div class="cover-key">{self._esc(k)}</div>'
            f'<div class="cover-val">{self._esc(v)}</div></div>'
            for k, v in items
        )
        return (
            f'<header class="header">'
            f'<h1>{self._esc(title)}</h1>'
            f'<div class="cover-grid">{grid}</div>'
            f"</header>"
        )

    def _render_monthly_section(self, pivot: pd.DataFrame) -> str:
        """渲染月度收益热力图区块（容器 div，图表由 JS 初始化）。"""
        if pivot is None or pivot.empty:
            return '<p class="empty-hint">无足够日收益数据构建月度热力图。</p>'
        return '<div id="monthly-heatmap-chart" class="chart"></div>'

    def _render_position_section(self, pos: Dict[str, Any]) -> str:
        """渲染持仓分析区块：统计卡片 + 持仓时长分桶柱状图容器。"""
        # 统计小卡片
        mini_metrics = {
            "平仓笔数": float(pos["closed_count"]),
            "平均持仓天数": pos["avg_holding_days"],
            "最大单笔盈利": pos["max_profit"],
            "最大单笔亏损": pos["max_loss"],
            "平均盈利": pos["avg_profit"],
            "平均亏损": pos["avg_loss"],
        }
        # 平仓笔数按整数展示
        cards_html = self._render_metric_cards(mini_metrics)
        bucket_chart = '<div id="holding-bucket-chart" class="chart"></div>'
        return cards_html + bucket_chart

    def _render_risk_section(
        self, var_95: float, cvar_95: float, max_consec_losses: int
    ) -> str:
        """渲染风险指标区块：VaR / CVaR / 最大连续亏损天数卡片。"""
        # 用小数展示，加 % 格式化
        var_pct = f"{var_95 * 100:.2f}%"
        cvar_pct = f"{cvar_95 * 100:.2f}%"
        cards = [
            ("VaR(95%)", var_pct, COLOR_LOSS if var_95 < 0 else COLOR_TEXT),
            ("CVaR(95%)", cvar_pct, COLOR_LOSS if cvar_95 < 0 else COLOR_TEXT),
            ("最大连续亏损天数", f"{max_consec_losses} 天", COLOR_TEXT),
        ]
        cards_html = "".join(
            f'<div class="card"><div class="card-label">{self._esc(label)}</div>'
            f'<div class="card-value" style="color:{color}">{self._esc(val)}</div></div>'
            for label, val, color in cards
        )
        return (
            '<div class="cards-grid">' + cards_html + "</div>"
            '<p class="empty-hint">VaR/CVaR 采用历史模拟法，基于日收益率分布计算；'
            '负值表示在 95% 置信度下的预期损失幅度。</p>'
        )

    def _render_signal_section(self, sig: Dict[str, Any]) -> str:
        """渲染信号分析区块：信号分布 + 胜率表格。"""
        dist: Dict[str, int] = sig.get("distribution", {})
        winrate: Dict[str, float] = sig.get("winrate", {})
        if not dist:
            return '<p class="empty-hint">无信号记录（trades.reason 为空）。</p>'
        headers = ["信号类型", "出现次数", "胜率"]
        head = "".join(f"<th>{h}</th>" for h in headers)
        # 按出现次数降序
        rows: List[str] = []
        for reason, count in sorted(dist.items(), key=lambda x: -x[1]):
            wr = winrate.get(reason)
            wr_text = "—" if wr is None else f"{wr * 100:.2f}%"
            wr_color = COLOR_TEXT
            if wr is not None:
                wr_color = COLOR_PROFIT if wr >= 0.5 else COLOR_LOSS
            rows.append(
                "<tr>"
                f"<td>{self._esc(reason)}</td>"
                f"<td>{count}</td>"
                f'<td style="color:{wr_color}">{wr_text}</td>'
                "</tr>"
            )
        body = "".join(rows)
        return (
            '<div class="table-wrap"><table class="trades-table signal-table">'
            f"<thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody>"
            "</table></div>"
        )

    def _render_trades_table(self, trades: List[Trade]) -> str:
        """渲染交易明细 HTML 表格。"""
        headers = ["日期", "标的", "方向", "价格", "数量", "金额", "手续费", "盈亏", "原因"]
        head = "".join(f"<th>{h}</th>" for h in headers)

        if not trades:
            body = '<tr><td colspan="9" class="empty">本回测无交易记录</td></tr>'
        else:
            rows: List[str] = []
            for t in trades:
                pnl = t.pnl
                pnl_text = "—" if pnl is None else f"{pnl:,.2f}"
                pnl_color = COLOR_TEXT
                if pnl is not None:
                    pnl_color = COLOR_PROFIT if pnl > 0 else (COLOR_LOSS if pnl < 0 else COLOR_TEXT)
                try:
                    date_text = pd.Timestamp(t.date).strftime("%Y-%m-%d")
                except Exception:
                    date_text = str(t.date)
                direction = "买入" if str(t.action).lower() == "buy" else "卖出"
                rows.append(
                    "<tr>"
                    f"<td>{self._esc(date_text)}</td>"
                    f"<td>{self._esc(str(t.symbol))}</td>"
                    f'<td class="dir-{self._esc(str(t.action).lower())}">{self._esc(direction)}</td>'
                    f"<td>{float(t.price):.2f}</td>"
                    f"<td>{int(t.shares)}</td>"
                    f"<td>{float(t.amount):,.2f}</td>"
                    f"<td>{float(t.commission):,.2f}</td>"
                    f'<td style="color:{pnl_color}">{pnl_text}</td>'
                    f"<td>{self._esc(str(t.reason or ''))}</td>"
                    "</tr>"
                )
            body = "".join(rows)

        return (
            '<div class="table-wrap"><table class="trades-table">'
            f"<thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody>"
            "</table></div>"
        )

    def _render_params(
        self, title: str, generated_at: str, params: Dict[str, Any]
    ) -> str:
        """渲染参数说明区。"""
        items: List[str] = []
        items.append(("<报告标题>", title))
        items.append(("<生成时间>", generated_at))
        for k, v in params.items():
            items.append((str(k), str(v)))
        rows = "".join(
            f'<tr><td class="param-key">{self._esc(k)}</td>'
            f'<td class="param-val">{self._esc(v)}</td></tr>'
            for k, v in items
        )
        return f'<table class="params-table"><tbody>{rows}</tbody></table>'

    # ------------------------------------------------------------------
    # 模板组装
    # ------------------------------------------------------------------
    def _render_template(
        self,
        title: str,
        generated_at: str,
        cover_html: str,
        cards_html: str,
        monthly_html: str,
        position_html: str,
        risk_html: str,
        signal_html: str,
        trades_html: str,
        params_html: str,
        chart_payload: Dict[str, Any],
    ) -> str:
        """组装完整 HTML 文档。"""
        data_js = json.dumps(chart_payload, ensure_ascii=False)
        css = self._build_css()
        js = self._build_js()

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{self._esc(title)}</title>
<style>{css}</style>
<script src="{ECHARTS_CDN}"></script>
</head>
<body>
<div class="container">
  {cover_html}

  <section class="block">
    <h2>绩效指标</h2>
    {cards_html}
  </section>

  <section class="block">
    <h2>净值曲线</h2>
    <div id="equity-chart" class="chart"></div>
  </section>

  <section class="block">
    <h2>回撤曲线</h2>
    <div id="drawdown-chart" class="chart"></div>
  </section>

  <section class="block">
    <h2>月度收益热力图</h2>
    {monthly_html}
  </section>

  <section class="block">
    <h2>持仓分析</h2>
    {position_html}
  </section>

  <section class="block">
    <h2>风险指标</h2>
    {risk_html}
  </section>

  <section class="block">
    <h2>信号分析</h2>
    {signal_html}
  </section>

  <section class="block">
    <h2>交易明细</h2>
    {trades_html}
  </section>

  <section class="block">
    <h2>参数说明</h2>
    {params_html}
  </section>

  <footer class="footer">
    本报告由量化交易系统自动生成，回测结果不代表未来表现。
  </footer>
</div>

<script id="report-data" type="application/json">{data_js}</script>
<script>{js}</script>
</body>
</html>
"""

    @staticmethod
    def _build_css() -> str:
        """内联 CSS（深色主题 + 打印样式）。"""
        return f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: {COLOR_BG};
  color: {COLOR_TEXT};
  font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
  line-height: 1.6;
  padding: 24px;
}}
.container {{ max-width: 1200px; margin: 0 auto; }}
.header {{ margin-bottom: 8px; }}
.header h1 {{ font-size: 28px; font-weight: 600; }}
.header .meta {{ color: {COLOR_TEXT_MUTED}; margin-top: 6px; font-size: 14px; }}
.cover-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px; margin-top: 16px;
}}
.cover-item {{
  background: {COLOR_CARD}; border: 1px solid {COLOR_CARD_BORDER};
  border-radius: 8px; padding: 10px 14px;
}}
.cover-key {{ color: {COLOR_TEXT_MUTED}; font-size: 12px; }}
.cover-val {{ color: {COLOR_TEXT}; font-size: 15px; font-weight: 500; margin-top: 2px; }}
.block {{ margin-top: 28px; }}
.block h2 {{
  font-size: 18px; font-weight: 600; margin-bottom: 14px;
  padding-left: 10px; border-left: 3px solid {COLOR_ACCENT};
}}
.cards-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 14px;
}}
.card {{
  background: {COLOR_CARD}; border: 1px solid {COLOR_CARD_BORDER};
  border-radius: 10px; padding: 16px;
}}
.card-label {{ color: {COLOR_TEXT_MUTED}; font-size: 13px; }}
.card-value {{ font-size: 22px; font-weight: 600; margin-top: 6px; }}
.chart {{
  background: {COLOR_CARD}; border: 1px solid {COLOR_CARD_BORDER};
  border-radius: 10px; width: 100%; height: 380px;
}}
.table-wrap {{
  background: {COLOR_CARD}; border: 1px solid {COLOR_CARD_BORDER};
  border-radius: 10px; overflow: auto; max-height: 480px;
}}
.trades-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.trades-table th, .trades-table td {{
  padding: 9px 12px; text-align: right; white-space: nowrap;
  border-bottom: 1px solid {COLOR_CARD_BORDER};
}}
.trades-table th {{
  position: sticky; top: 0; background: {COLOR_CARD};
  color: {COLOR_TEXT_MUTED}; font-weight: 600;
}}
.trades-table td:first-child, .trades-table th:first-child {{ text-align: left; }}
.trades-table td:nth-child(2), .trades-table th:nth-child(2) {{ text-align: left; }}
.trades-table .empty {{ text-align: center; color: {COLOR_TEXT_MUTED}; padding: 24px; }}
.dir-buy {{ color: {COLOR_PROFIT}; }}
.dir-sell {{ color: {COLOR_ACCENT}; }}
.signal-table td:nth-child(1) {{ text-align: left; }}
.empty-hint {{ color: {COLOR_TEXT_MUTED}; font-size: 13px; padding: 12px 4px; }}
.params-table {{
  background: {COLOR_CARD}; border: 1px solid {COLOR_CARD_BORDER};
  border-radius: 10px; width: 100%; border-collapse: collapse; font-size: 14px;
}}
.params-table td {{ padding: 9px 14px; border-bottom: 1px solid {COLOR_CARD_BORDER}; }}
.param-key {{ color: {COLOR_TEXT_MUTED}; width: 220px; }}
.footer {{
  margin-top: 36px; color: {COLOR_TEXT_MUTED}; font-size: 12px;
  text-align: center;
}}

/* 打印样式：隐藏交互元素、优化分页 */
@media print {{
  body {{ background: #fff; color: #000; padding: 0; }}
  .chart {{ height: 280px !important; page-break-inside: avoid; }}
  .table-wrap {{ max-height: none !important; overflow: visible !important; }}
  .block {{ page-break-inside: avoid; break-inside: avoid; }}
  .card, .cover-item {{ box-shadow: none; border: 1px solid #ccc; }}
  .echarts-tooltip, .echarts-datazoom, .echarts-control {{ display: none !important; }}
  .header h1 {{ color: #000; }}
}}
"""

    @staticmethod
    def _build_js() -> str:
        """ECharts 初始化 JS（从内联 JSON 读取数据）。"""
        return f"""
(function () {{
  var el = document.getElementById('report-data');
  var data = JSON.parse(el.text || el.textContent);
  var axisStyle = {{ axisLine: {{ lineStyle: {{ color: '#3a4466' }} }},
                     axisLabel: {{ color: '#8a93b2' }},
                     splitLine: {{ lineStyle: {{ color: '#1f2b4d' }} }} }};
  var tooltip = {{ trigger: 'axis',
                  backgroundColor: '#16213e', borderColor: '#1f2b4d',
                  textStyle: {{ color: '#e0e0e0' }} }};
  var charts = [];

  // 净值曲线
  var eqEl = document.getElementById('equity-chart');
  if (eqEl) {{
    var eqChart = echarts.init(eqEl, 'dark');
    eqChart.setOption({{
      backgroundColor: 'transparent',
      tooltip: tooltip,
      legend: {{ data: ['策略净值', '基准净值'], textStyle: {{ color: '#e0e0e0' }} }},
      grid: {{ left: 60, right: 30, top: 40, bottom: 40 }},
      xAxis: Object.assign({{ type: 'category', data: data.dates }}, axisStyle),
      yAxis: Object.assign({{ type: 'value', scale: true }}, axisStyle),
      series: [
        {{ name: '策略净值', type: 'line', data: data.strategy, smooth: true,
           showSymbol: false, lineStyle: {{ color: '{COLOR_ACCENT}', width: 2 }},
           itemStyle: {{ color: '{COLOR_ACCENT}' }} }},
        {{ name: '基准净值', type: 'line', data: data.benchmark, smooth: true,
           showSymbol: false, lineStyle: {{ color: '{COLOR_BENCHMARK}', width: 1.5, type: 'dashed' }},
           itemStyle: {{ color: '{COLOR_BENCHMARK}' }} }}
      ]
    }});
    charts.push(eqChart);
  }}

  // 回撤面积图
  var ddEl = document.getElementById('drawdown-chart');
  if (ddEl) {{
    var ddChart = echarts.init(ddEl, 'dark');
    ddChart.setOption({{
      backgroundColor: 'transparent',
      tooltip: Object.assign({{ valueFormatter: function (v) {{ return v == null ? '-' : v + '%'; }} }}, tooltip),
      grid: {{ left: 60, right: 30, top: 30, bottom: 40 }},
      xAxis: Object.assign({{ type: 'category', data: data.dates }}, axisStyle),
      yAxis: Object.assign({{ type: 'value', axisLabel: {{ color: '#8a93b2', formatter: '{{value}}%' }} }}, axisStyle),
      series: [
        {{ name: '回撤', type: 'line', data: data.drawdown, smooth: true,
           showSymbol: false, areaStyle: {{ color: '{COLOR_DRAWDOWN}', opacity: 0.35 }},
           lineStyle: {{ color: '{COLOR_DRAWDOWN}', width: 1.5 }},
           itemStyle: {{ color: '{COLOR_DRAWDOWN}' }} }}
      ]
    }});
    charts.push(ddChart);
  }}

  // 月度收益热力图
  var hmEl = document.getElementById('monthly-heatmap-chart');
  if (hmEl && data.monthly_data && data.monthly_data.length > 0) {{
    var hmChart = echarts.init(hmEl, 'dark');
    hmChart.setOption({{
      backgroundColor: 'transparent',
      tooltip: {{
        backgroundColor: '#16213e', borderColor: '#1f2b4d',
        textStyle: {{ color: '#e0e0e0' }},
        formatter: function (p) {{
          var v = p.value[2];
          var y = data.monthly_years[p.value[1]];
          var m = data.monthly_months[p.value[0]];
          return y + '年 ' + m + '<br/>收益: ' + (v == null ? '-' : (v * 100).toFixed(2) + '%');
        }}
      }},
      grid: {{ left: 60, right: 30, top: 20, bottom: 70 }},
      xAxis: Object.assign({{ type: 'category', data: data.monthly_months, splitArea: {{ show: true }} }}, axisStyle),
      yAxis: Object.assign({{ type: 'category', data: data.monthly_years.map(String), splitArea: {{ show: true }} }}, axisStyle),
      visualMap: {{
        min: data.monthly_vmin, max: data.monthly_vmax, calculable: true,
        orient: 'horizontal', left: 'center', bottom: 10,
        inRange: {{ color: ['#ff6b6b', '#16213e', '#2ecc71'] }},
        textStyle: {{ color: '#8a93b2' }}
      }},
      series: [{{
        name: '月度收益', type: 'heatmap', data: data.monthly_data,
        label: {{ show: true, color: '#e0e0e0',
                 formatter: function (p) {{
                   var v = p.value[2];
                   return v == null ? '' : (v * 100).toFixed(1);
                 }} }},
        emphasis: {{ itemStyle: {{ shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' }} }}
      }}]
    }});
    charts.push(hmChart);
  }}

  // 持仓时长分桶柱状图
  var bkEl = document.getElementById('holding-bucket-chart');
  if (bkEl && data.holding_bucket_labels && data.holding_bucket_labels.length > 0) {{
    var bkChart = echarts.init(bkEl, 'dark');
    bkChart.setOption({{
      backgroundColor: 'transparent',
      tooltip: tooltip,
      grid: {{ left: 60, right: 30, top: 30, bottom: 40 }},
      xAxis: Object.assign({{ type: 'category', data: data.holding_bucket_labels }}, axisStyle),
      yAxis: Object.assign({{ type: 'value', minInterval: 1 }}, axisStyle),
      series: [{{
        name: '持仓笔数', type: 'bar', data: data.holding_bucket_values,
        itemStyle: {{ color: '{COLOR_ACCENT}' }},
        barMaxWidth: 60
      }}]
    }});
    charts.push(bkChart);
  }}

  window.addEventListener('resize', function () {{
    charts.forEach(function (c) {{ c.resize(); }});
  }});
}})();
"""

    @staticmethod
    def _esc(text: Any) -> str:
        """HTML 转义，防止参数/原因文本破坏页面。"""
        s = str(text)
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
