"""回测报告 HTML 生成器单元测试。

覆盖:
- 报告文件生成且非空
- HTML 包含关键区块（净值曲线/绩效指标/交易明细/echarts CDN）
- 空交易列表不崩溃
- 指标卡片值正确出现在 HTML 中
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestResult, Trade
from backtest.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_equity(n: int = 120, start: float = 1_000_000.0) -> pd.Series:
    """构造一条带波动的净值曲线。"""
    dates = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.RandomState(42)
    drift = np.linspace(0, 0.15, n)
    noise = rng.randn(n) * 0.01
    values = start * np.exp(drift + noise)
    return pd.Series(values, index=dates, name="equity")


@pytest.fixture
def sample_trades() -> list[Trade]:
    """构造示例交易记录。"""
    dates = pd.bdate_range("2024-01-02", periods=120)
    return [
        Trade(date=dates[10], symbol="600519.SH", action="buy",
              price=1700.0, shares=100, amount=170000.0,
              commission=42.5, stamp_tax=0.0, slippage_cost=170.0,
              pnl=None, reason="金叉买入"),
        Trade(date=dates[60], symbol="600519.SH", action="sell",
              price=1780.0, shares=100, amount=178000.0,
              commission=44.5, stamp_tax=89.0, slippage_cost=178.0,
              pnl=5200.0, reason="死叉卖出"),
    ]


@pytest.fixture
def sample_metrics() -> dict[str, float]:
    """构造示例绩效指标。"""
    return {
        "累计收益率": 0.1234,
        "年化收益率": 0.15,
        "最大回撤": -0.05,
        "夏普比率": 1.234,
        "胜率": 0.6,
        "盈亏比": 1.5,
        "交易次数": 1,
        "总盈利": 5200.0,
        "总亏损": 0.0,
    }


@pytest.fixture
def sample_result(sample_trades, sample_metrics) -> BacktestResult:
    """构造示例回测结果。"""
    equity = _make_equity()
    benchmark = equity * 0.95  # 基准略低于策略
    return BacktestResult(
        equity_curve=equity,
        benchmark_curve=benchmark,
        trades=sample_trades,
        metrics=sample_metrics,
        metrics_df=pd.DataFrame(),
        daily_returns=equity.pct_change().dropna(),
    )


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestReportGeneration:
    def test_generate_html_report_creates_file(self, sample_result, tmp_path: Path):
        """验证文件生成且非空。"""
        out = tmp_path / "report.html"
        path = ReportGenerator().generate_html_report(
            sample_result, str(out), title="测试报告",
        )
        assert Path(path).exists()
        assert Path(path).is_absolute()
        assert Path(path).stat().st_size > 1000  # 非空且有实质内容

    def test_html_contains_key_sections(self, sample_result, tmp_path: Path):
        """验证 HTML 包含关键区块与 echarts。"""
        out = tmp_path / "report.html"
        path = ReportGenerator().generate_html_report(
            sample_result, str(out), title="测试报告",
        )
        html = Path(path).read_text(encoding="utf-8")
        for keyword in ["净值曲线", "绩效指标", "交易明细", "echarts"]:
            assert keyword in html, f"HTML 缺少关键字: {keyword}"

    def test_report_with_empty_trades(self, sample_result, tmp_path: Path):
        """空交易列表不应崩溃。"""
        sample_result.trades = []
        out = tmp_path / "empty_trades.html"
        path = ReportGenerator().generate_html_report(
            sample_result, str(out), title="空交易报告",
        )
        html = Path(path).read_text(encoding="utf-8")
        assert Path(path).exists()
        # 仍应包含关键区块
        assert "净值曲线" in html
        assert "交易明细" in html

    def test_metrics_card_values(self, sample_result, tmp_path: Path):
        """验证指标值出现在 HTML 中。"""
        out = tmp_path / "metrics.html"
        path = ReportGenerator().generate_html_report(
            sample_result, str(out), title="指标报告",
        )
        html = Path(path).read_text(encoding="utf-8")
        # 累计收益率 0.1234 -> 12.34%
        assert "12.34%" in html
        # 夏普比率 1.234 -> 1.234
        assert "1.234" in html
        # 胜率 0.6 -> 60.00%
        assert "60.00%" in html
        # 总盈利 -> 带千分位金额
        assert "5,200.00" in html

    def test_params_section_rendered(self, sample_result, tmp_path: Path):
        """参数说明区应展示传入参数。"""
        out = tmp_path / "params.html"
        path = ReportGenerator().generate_html_report(
            sample_result, str(out),
            title="参数报告",
            params={"标的": "600519.SH", "策略": "ma_cross"},
        )
        html = Path(path).read_text(encoding="utf-8")
        assert "参数说明" in html
        assert "600519.SH" in html
        assert "ma_cross" in html

    def test_returns_absolute_path(self, sample_result, tmp_path: Path):
        """返回值应为可直接打开的绝对路径。"""
        out = tmp_path / "subdir" / "report.html"
        path = ReportGenerator().generate_html_report(sample_result, str(out))
        assert Path(path).is_absolute()
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# 新增分析方法单元测试
# ---------------------------------------------------------------------------
class TestNewAnalysisMethods:
    """覆盖月度收益 / VaR / 连续亏损 / 持仓 / 信号 / Sortino / Calmar。"""

    def test_monthly_returns_matrix_shape(self):
        """年月透视表应正确聚合：2024-01 与 2024-03 的月收益可复算。"""
        rng = np.random.RandomState(0)
        dates = pd.bdate_range("2024-01-02", periods=60)  # 跨 1-3 月
        rets = pd.Series(rng.randn(60) * 0.01 + 0.001, index=dates)
        pivot = ReportGenerator._build_monthly_returns(rets)
        # index 应含 2024，columns 应含 1..12
        assert 2024 in pivot.index
        for m in range(1, 13):
            assert m in pivot.columns
        # 手动复算 2024-01 的月收益
        jan = rets.loc[:"2024-01-31"]
        expected = float((1 + jan).prod() - 1)
        got = float(pivot.loc[2024, 1])
        assert abs(got - expected) < 1e-10

    def test_monthly_returns_empty(self):
        """空日收益序列应返回空 DataFrame，不崩溃。"""
        out = ReportGenerator._build_monthly_returns(pd.Series(dtype=float))
        assert isinstance(out, pd.DataFrame)
        assert out.empty

    def test_var_cvar_known_distribution(self):
        """VaR/CVaR：与 np.percentile 对齐，且 CVaR <= VaR。"""
        # 构造一个有明确尾部的分布
        rng = np.random.RandomState(7)
        rets = pd.Series(rng.normal(0.001, 0.02, 500))
        var, cvar = ReportGenerator._calc_var_cvar(rets, confidence=0.95)
        expected_var = float(np.percentile(rets.values, 5))
        assert abs(var - expected_var) < 1e-10
        # CVaR 是尾部均值，应 <= VaR（更负）
        assert cvar <= var + 1e-12

    def test_var_cvar_empty_returns_zero(self):
        """空序列 VaR/CVaR 返回 (0.0, 0.0)。"""
        var, cvar = ReportGenerator._calc_var_cvar(pd.Series(dtype=float))
        assert var == 0.0 and cvar == 0.0

    def test_max_consecutive_losses(self):
        """最大连续亏损天数：设计序列 [+,-,-,-,0,+,-] 应为 3。"""
        rets = pd.Series([0.01, -0.01, -0.02, -0.01, 0.0, 0.01, -0.005])
        assert ReportGenerator._calc_max_consecutive_losses(rets) == 3

    def test_max_consecutive_losses_all_positive(self):
        """全正收益序列应返回 0。"""
        rets = pd.Series([0.01, 0.02, 0.03])
        assert ReportGenerator._calc_max_consecutive_losses(rets) == 0

    def test_analyze_positions_fields(self):
        """持仓分析应返回要求字段，且分桶计数正确。"""
        dates = pd.bdate_range("2024-01-02", periods=120)
        trades = [
            # 平仓 1：持仓 5 天（落入 4-7 天桶）
            Trade(date=dates[10], symbol="X", action="buy", price=10.0,
                  shares=100, amount=1000.0, commission=1.0, stamp_tax=0.0,
                  slippage_cost=0.0, pnl=None, reason="金叉买入",
                  entry_date=dates[5]),
            Trade(date=dates[10], symbol="X", action="sell", price=11.0,
                  shares=100, amount=1100.0, commission=1.0, stamp_tax=0.5,
                  slippage_cost=0.0, pnl=98.5, reason="止盈",
                  entry_date=dates[5]),
            # 平仓 2：持仓 20 天（落入 15-30 天桶），亏损
            Trade(date=dates[40], symbol="Y", action="buy", price=20.0,
                  shares=100, amount=2000.0, commission=2.0, stamp_tax=0.0,
                  slippage_cost=0.0, pnl=None, reason="突破买入",
                  entry_date=dates[20]),
            Trade(date=dates[40], symbol="Y", action="sell", price=19.0,
                  shares=100, amount=1900.0, commission=1.9, stamp_tax=0.95,
                  slippage_cost=0.0, pnl=-104.85, reason="止损",
                  entry_date=dates[20]),
        ]
        out = ReportGenerator._analyze_positions(trades)
        assert out["closed_count"] == 2
        assert out["holding_buckets"]["4-7天"] == 1
        assert out["holding_buckets"]["15-30天"] == 1
        assert out["max_profit"] == pytest.approx(98.5)
        assert out["max_loss"] == pytest.approx(-104.85)
        assert out["avg_profit"] == pytest.approx(98.5)
        assert out["avg_loss"] == pytest.approx(-104.85)

    def test_analyze_signals_distribution(self):
        """信号分析应按 reason 统计次数与胜率。"""
        trades = [
            Trade(date=pd.Timestamp("2024-01-02"), symbol="X", action="buy",
                  price=10.0, shares=100, amount=1000.0, commission=1.0,
                  stamp_tax=0.0, slippage_cost=0.0, pnl=None, reason="金叉买入"),
            Trade(date=pd.Timestamp("2024-01-03"), symbol="X", action="sell",
                  price=11.0, shares=100, amount=1100.0, commission=1.0,
                  stamp_tax=0.5, slippage_cost=0.0, pnl=98.5, reason="金叉买入"),
            Trade(date=pd.Timestamp("2024-01-04"), symbol="Y", action="sell",
                  price=20.0, shares=100, amount=2000.0, commission=2.0,
                  stamp_tax=1.0, slippage_cost=0.0, pnl=-50.0, reason="止损"),
        ]
        out = ReportGenerator._analyze_signals(trades)
        assert out["distribution"]["金叉买入"] == 2
        assert out["distribution"]["止损"] == 1
        assert out["winrate"]["金叉买入"] == 1.0  # 1/1 平仓盈利
        assert out["winrate"]["止损"] == 0.0

    def test_sortino_ratio_positive(self):
        """Sortino 比率：正漂移 + 少量下行应 > 0。"""
        rng = np.random.RandomState(1)
        # 多数正收益（均值 +0.005），少量小负收益（均值 -0.005），整体均值为正
        rets = pd.Series(np.concatenate([
            rng.normal(0.005, 0.005, 80),
            rng.normal(-0.005, 0.003, 20),
        ]))
        s = ReportGenerator._calc_sortino_ratio(rets)
        assert isinstance(s, float)
        assert s > 0

    def test_sortino_ratio_empty(self):
        """空序列 Sortino 返回 0。"""
        assert ReportGenerator._calc_sortino_ratio(pd.Series(dtype=float)) == 0.0

    def test_calmar_ratio(self):
        """Calmar 比率：有回撤的上涨净值应 > 0；空净值返回 0。"""
        dates = pd.bdate_range("2024-01-02", periods=252)
        # 先涨后小幅回撤再继续涨，保证存在非零最大回撤
        base = np.linspace(1_000_000, 1_080_000, 252)
        # 在中段插入一个 -3% 的回撤
        base[120:150] = base[120:150] * 0.97
        eq = pd.Series(base, index=dates)
        c = ReportGenerator._calc_calmar_ratio(eq)
        assert isinstance(c, float)
        assert c > 0
        # 空序列
        assert ReportGenerator._calc_calmar_ratio(pd.Series(dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# 新增 HTML 章节 / 打印样式测试
# ---------------------------------------------------------------------------
class TestNewHtmlSections:
    def test_html_contains_new_section_keywords(self, sample_result, tmp_path: Path):
        """HTML 应包含月度收益 / 持仓分析 / 风险指标 / VaR 等新增章节。"""
        out = tmp_path / "enhanced.html"
        path = ReportGenerator().generate_html_report(
            sample_result, str(out), title="增强报告",
        )
        html = Path(path).read_text(encoding="utf-8")
        for kw in ["月度收益热力图", "持仓分析", "风险指标", "VaR",
                   "CVaR", "信号分析", "索提诺比率", "卡玛比率"]:
            assert kw in html, f"HTML 缺少新增章节关键词: {kw}"

    def test_html_has_print_media_query(self, sample_result, tmp_path: Path):
        """CSS 应包含 @media print 打印样式。"""
        out = tmp_path / "print.html"
        path = ReportGenerator().generate_html_report(
            sample_result, str(out), title="打印测试",
        )
        html = Path(path).read_text(encoding="utf-8")
        assert "@media print" in html

    def test_html_cover_shows_params(self, sample_result, tmp_path: Path):
        """封面区应展示标的/策略/初始资金等参数。"""
        out = tmp_path / "cover.html"
        path = ReportGenerator().generate_html_report(
            sample_result, str(out), title="封面测试",
            params={"标的": "600519.SH", "策略": "ma_cross", "初始资金": "1,000,000"},
        )
        html = Path(path).read_text(encoding="utf-8")
        assert "cover-grid" in html
        assert "600519.SH" in html
        assert "1,000,000" in html

    def test_empty_data_boundary(self, tmp_path: Path):
        """空 BacktestResult（无交易、无净值）不应崩溃。"""
        empty_result = BacktestResult(
            equity_curve=pd.Series(dtype=float),
            benchmark_curve=pd.Series(dtype=float),
            trades=[],
            metrics={},
            metrics_df=pd.DataFrame(),
            daily_returns=pd.Series(dtype=float),
        )
        out = tmp_path / "empty.html"
        path = ReportGenerator().generate_html_report(
            empty_result, str(out), title="空数据报告",
        )
        html = Path(path).read_text(encoding="utf-8")
        assert Path(path).exists()
        assert "净值曲线" in html
        assert "风险指标" in html  # 仍渲染风险指标区（VaR=0）


# ---------------------------------------------------------------------------
# 报告路由纯函数测试
# ---------------------------------------------------------------------------
class TestReportRouteHelpers:
    def test_list_reports_empty_dir(self, tmp_path: Path):
        """不存在的目录应返回空列表。"""
        from backtest.report_routes import list_reports
        assert list_reports(tmp_path / "nope") == []

    def test_list_reports_returns_meta(self, tmp_path: Path):
        """list_reports 应返回文件名 / 大小 / 修改时间。"""
        from backtest.report_routes import list_reports
        (tmp_path / "aaa_report.html").write_text("<html>x</html>", encoding="utf-8")
        (tmp_path / "bbb_report.html").write_text("<html>yy</html>", encoding="utf-8")
        items = list_reports(tmp_path)
        assert len(items) == 2
        ids = {it["report_id"] for it in items}
        assert ids == {"aaa_report", "bbb_report"}
        for it in items:
            assert "size_bytes" in it and it["size_bytes"] > 0
            assert "modified_at" in it and "T" in it["modified_at"]

    def test_resolve_report_path_normal(self, tmp_path: Path):
        """正常 report_id 应解析到文件。"""
        from backtest.report_routes import resolve_report_path
        f = tmp_path / "my_report.html"
        f.write_text("<html/>", encoding="utf-8")
        p = resolve_report_path(tmp_path, "my_report")
        assert p is not None and p.name == "my_report.html"

    def test_resolve_report_path_path_traversal_blocked(self, tmp_path: Path):
        """路径穿越（../）应被拒绝。"""
        from backtest.report_routes import resolve_report_path
        assert resolve_report_path(tmp_path, "../etc") is None
        assert resolve_report_path(tmp_path, "foo/bar") is None
        assert resolve_report_path(tmp_path, "") is None

