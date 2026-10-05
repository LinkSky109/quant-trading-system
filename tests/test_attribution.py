"""绩效归因分析单元测试。

覆盖: 完整归因报告 / 月度收益分解 / 盈亏贡献校验 / 卡玛比率 / 索提诺比率
      无交易边界 / 单标的持仓归因 / 组合标的归因 / 信号质量评估
全部使用 mock 数据，不依赖网络。
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from analysis.attribution import PerformanceAttribution
from backtest.engine import BacktestResult, Trade


# ---------------------------------------------------------------------------
# 构造工具
# ---------------------------------------------------------------------------

def _make_trade(
    date: pd.Timestamp,
    symbol: str,
    action: str,
    price: float = 10.0,
    shares: int = 100,
    pnl: float | None = None,
) -> Trade:
    """构造一笔 mock 成交。"""
    amount = price * shares
    return Trade(
        date=date,
        symbol=symbol,
        action=action,
        price=price,
        shares=shares,
        amount=amount,
        commission=max(amount * 0.00025, 5.0),
        stamp_tax=amount * 0.0005 if action == "sell" else 0.0,
        slippage_cost=0.0,
        pnl=pnl,
        reason="test",
    )


def _make_equity(values: List[float], start: str = "2024-01-02") -> pd.Series:
    """按交易日顺序构造净值曲线（每个值一个交易日）。"""
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=idx, name="equity")


def _mixed_trades() -> List[Trade]:
    """混合盈亏的平仓记录。"""
    d = pd.Timestamp("2024-02-01")
    return [
        _make_trade(d, "AAA", "buy", pnl=None),
        _make_trade(d + pd.Timedelta(days=1), "AAA", "sell", pnl=500.0),
        _make_trade(d + pd.Timedelta(days=2), "BBB", "buy", pnl=None),
        _make_trade(d + pd.Timedelta(days=3), "BBB", "sell", pnl=-200.0),
        _make_trade(d + pd.Timedelta(days=4), "CCC", "sell", pnl=300.0),
        _make_trade(d + pd.Timedelta(days=5), "CCC", "sell", pnl=-100.0),
    ]


# ---------------------------------------------------------------------------
# 1. 完整归因报告
# ---------------------------------------------------------------------------

class TestFullReport:
    def test_all_dimensions_present(self):
        attr = PerformanceAttribution(risk_free_rate=0.02, trading_days=252)
        equity = _make_equity([1.0, 1.05, 1.02, 1.10, 1.08, 1.15])
        report = attr.analyze(equity_curve=equity, trades=_mixed_trades())

        assert set(report.keys()) == {
            "summary", "trade_attribution", "time_attribution",
            "holding_attribution", "strategy_attribution", "risk_adjusted",
        }
        assert set(report["summary"].keys()) == {
            "total_return", "annualized_return", "max_drawdown", "total_trades",
        }
        assert report["summary"]["total_trades"] == 6
        # 交易归因字段齐全
        ta = report["trade_attribution"]
        for k in ("profit_contribution", "loss_contribution", "profit_count",
                  "loss_count", "max_single_profit", "max_single_loss",
                  "avg_profit", "avg_loss", "total_realized_pnl"):
            assert k in ta

    def test_trade_attribution_values(self):
        attr = PerformanceAttribution()
        report = attr.analyze(
            equity_curve=_make_equity([1.0, 1.0]), trades=_mixed_trades()
        )
        ta = report["trade_attribution"]
        # 盈利: 500 + 300 = 800；亏损: -200 + -100 = -300
        assert ta["profit_contribution"] == pytest.approx(800.0)
        assert ta["loss_contribution"] == pytest.approx(-300.0)
        assert ta["profit_count"] == 2
        assert ta["loss_count"] == 2
        assert ta["max_single_profit"] == pytest.approx(500.0)
        assert ta["max_single_loss"] == pytest.approx(-200.0)
        assert ta["avg_profit"] == pytest.approx(400.0)
        assert ta["avg_loss"] == pytest.approx(-150.0)

    def test_profit_plus_loss_equals_total(self):
        """盈利贡献 + 亏损贡献 ≈ 总已实现盈亏。"""
        attr = PerformanceAttribution()
        report = attr.analyze(
            equity_curve=_make_equity([1.0, 1.0]), trades=_mixed_trades()
        )
        ta = report["trade_attribution"]
        assert ta["profit_contribution"] + ta["loss_contribution"] == pytest.approx(
            ta["total_realized_pnl"], abs=1e-6
        )

    def test_strategy_attribution_empty_without_data(self):
        attr = PerformanceAttribution()
        report = attr.analyze(
            equity_curve=_make_equity([1.0, 1.0]), trades=_mixed_trades()
        )
        sa = report["strategy_attribution"]
        assert sa["buy_signal_quality"]["sample_count"] == 0
        assert sa["sell_signal_quality"]["sample_count"] == 0


# ---------------------------------------------------------------------------
# 2. 月度收益分解
# ---------------------------------------------------------------------------

class TestTimeAttribution:
    def test_monthly_returns_decomposed_correctly(self):
        """构造跨三个月的已知净值，验证月度收益分段正确。"""
        # 2024-01: 1.00 -> 1.10  (月内涨 10%)
        # 2024-02: 1.10 -> 1.21  (月内再涨 10%)
        # 2024-03: 1.21 -> 1.089 (跌 10%)
        idx = pd.DatetimeIndex([
            "2024-01-02", "2024-01-31",
            "2024-02-29",
            "2024-03-29",
        ])
        equity = pd.Series([1.00, 1.10, 1.21, 1.089], index=idx)

        attr = PerformanceAttribution()
        report = attr.analyze(equity_curve=equity, trades=[])
        ma = report["time_attribution"]["monthly_returns"]

        months = [m["month"] for m in ma]
        assert months == ["2024-01", "2024-02", "2024-03"]
        # 首月: 1.10/1.00 - 1 = 0.10
        assert ma[0]["return"] == pytest.approx(0.10, abs=1e-6)
        # 次月: 1.21/1.10 - 1 = 0.10
        assert ma[1]["return"] == pytest.approx(0.10, abs=1e-6)
        # 三月: 1.089/1.21 - 1 = -0.10
        assert ma[2]["return"] == pytest.approx(-0.10, abs=1e-6)

    def test_best_and_worst_month(self):
        idx = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-29"])
        equity = pd.Series([1.00, 1.20, 0.90], index=idx)
        attr = PerformanceAttribution()
        report = attr.analyze(equity_curve=equity, trades=[])
        ta = report["time_attribution"]
        assert ta["best_month"]["month"] == "2024-02"
        assert ta["best_month"]["return"] == pytest.approx(0.20)
        assert ta["worst_month"]["month"] == "2024-03"
        assert ta["worst_month"]["return"] == pytest.approx(-0.25)

    def test_weekly_returns_format(self):
        equity = _make_equity([1.0, 1.02, 1.05, 1.03, 1.06], start="2024-01-01")
        attr = PerformanceAttribution()
        report = attr.analyze(equity_curve=equity, trades=[])
        wr = report["time_attribution"]["weekly_returns"]
        assert len(wr) > 0
        assert "week" in wr[0] and "return" in wr[0]
        assert wr[0]["week"].startswith("2024-W")

    def test_empty_equity_time_attribution(self):
        attr = PerformanceAttribution()
        report = attr.analyze(equity_curve=pd.Series(dtype=float), trades=[])
        ta = report["time_attribution"]
        assert ta["monthly_returns"] == []
        assert ta["best_month"]["month"] is None


# ---------------------------------------------------------------------------
# 4. 风险调整指标
# ---------------------------------------------------------------------------

class TestRiskAdjusted:
    def test_calmar_ratio(self):
        """构造已知回撤的净值：[1.0, 0.8, 1.1] -> max_dd = -0.2。"""
        equity = _make_equity([1.0, 0.8, 1.1])
        attr = PerformanceAttribution(risk_free_rate=0.02, trading_days=252)
        report = attr.analyze(equity_curve=equity, trades=[])
        ra = report["risk_adjusted"]

        # 最大回撤 = (0.8 - 1.0)/1.0 = -0.2
        assert report["summary"]["max_drawdown"] == pytest.approx(-0.2, abs=1e-6)
        # 年化 = (1.1/1.0)^(252/3) - 1
        annualized = (1.1) ** (252 / 3) - 1
        assert ra["calmar_ratio"] == pytest.approx(annualized / 0.2, rel=1e-6)

    def test_calmar_none_when_no_drawdown(self):
        """单调上涨 -> 回撤为 0 -> Calmar 返回 None。"""
        equity = _make_equity([1.0, 1.1, 1.2])
        attr = PerformanceAttribution()
        report = attr.analyze(equity_curve=equity, trades=[])
        assert report["risk_adjusted"]["calmar_ratio"] is None
        assert report["risk_adjusted"]["return_drawdown_ratio"] is None

    def test_sortino_ratio(self):
        """构造已知下行波动的日收益序列。

        日收益序列: +1%, -1%, +1%, -1%（pct_change 去首日 NaN 后）。
        min(x, 0) = [0, -0.01, 0, -0.01]，下行标准差按全部样本数取均值后年化。
        """
        # 由日收益反推净值
        daily = np.array([0.01, -0.01, 0.01, -0.01])
        values = np.concatenate([[1.0], 1.0 * np.cumprod(1 + daily)])
        equity = pd.Series(values, index=pd.bdate_range("2024-01-02", periods=len(values)))

        rf = 0.02
        attr = PerformanceAttribution(risk_free_rate=rf, trading_days=252)
        report = attr.analyze(equity_curve=equity, trades=[])

        downside = np.array([0.0, -0.01, 0.0, -0.01])
        expected_downside_dev = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(252))
        expected_sortino = (
            report["summary"]["annualized_return"] - rf
        ) / expected_downside_dev
        assert report["risk_adjusted"]["sortino_ratio"] == pytest.approx(
            expected_sortino, rel=1e-6
        )

    def test_return_drawdown_ratio(self):
        # [1.0, 0.8, 1.1] -> total_return = 0.1, |dd| = 0.2
        equity = _make_equity([1.0, 0.8, 1.1])
        attr = PerformanceAttribution()
        report = attr.analyze(equity_curve=equity, trades=[])
        assert report["risk_adjusted"]["return_drawdown_ratio"] == pytest.approx(0.1 / 0.2)


# ---------------------------------------------------------------------------
# 6. 无交易 / 空数据边界
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_trades_no_crash(self):
        attr = PerformanceAttribution()
        report = attr.analyze(equity_curve=_make_equity([1.0, 1.01, 1.02]), trades=[])
        ta = report["trade_attribution"]
        assert ta["profit_count"] == 0
        assert ta["loss_count"] == 0
        assert ta["total_realized_pnl"] == 0.0
        assert ta["avg_profit"] == 0.0

    def test_empty_equity_no_crash(self):
        attr = PerformanceAttribution()
        report = attr.analyze(equity_curve=pd.Series(dtype=float), trades=[])
        assert report["summary"]["total_return"] == 0.0
        assert report["risk_adjusted"]["sortino_ratio"] is None

    def test_single_value_equity_no_crash(self):
        attr = PerformanceAttribution()
        report = attr.analyze(equity_curve=pd.Series([1.0]), trades=[])
        assert report["summary"]["total_return"] == 0.0


# ---------------------------------------------------------------------------
# 7/8. 持仓归因
# ---------------------------------------------------------------------------

def _fake_result(symbol: str, pnls: List[float]) -> BacktestResult:
    """构造仅含 trades 的伪 BacktestResult。"""
    trades = [
        _make_trade(pd.Timestamp("2024-01-02"), symbol, "sell", pnl=p)
        for p in pnls
    ]
    equity = pd.Series([1.0, 1.0], index=pd.bdate_range("2024-01-02", periods=2))
    return BacktestResult(
        equity_curve=equity,
        benchmark_curve=equity,
        trades=trades,
        metrics={},
        metrics_df=pd.DataFrame(),
        daily_returns=equity.pct_change().fillna(0.0),
    )


class TestHoldingAttribution:
    def test_single_symbol_holding(self):
        """单标的回测: symbol_results 仅一条。"""
        results = {"AAA": _fake_result("AAA", [500.0, -100.0])}
        attr = PerformanceAttribution()
        report = attr.analyze(
            equity_curve=pd.Series([1.0, 1.05], index=pd.bdate_range("2024-01-02", periods=2)),
            trades=results["AAA"].trades,
            symbol_results=results,
        )
        ha = report["holding_attribution"]
        assert len(ha["symbol_contributions"]) == 1
        item = ha["symbol_contributions"][0]
        assert item["symbol"] == "AAA"
        assert item["contribution"] == pytest.approx(400.0)
        assert item["pct"] == pytest.approx(1.0)  # 唯一标的 -> 100%
        assert ha["top_contributor"]["symbol"] == "AAA"
        assert ha["top_drag"]["symbol"] == "AAA"

    def test_portfolio_symbol_contributions(self):
        """组合回测: 各标的贡献正确汇总，最大贡献/拖累识别正确。"""
        results = {
            "AAA": _fake_result("AAA", [800.0, 200.0]),       # +1000
            "BBB": _fake_result("BBB", [-300.0, -100.0]),     # -400
            "CCC": _fake_result("CCC", [500.0]),              # +500
        }
        all_trades = []
        for r in results.values():
            all_trades.extend(r.trades)

        attr = PerformanceAttribution()
        report = attr.analyze(
            equity_curve=pd.Series([1.0, 1.1], index=pd.bdate_range("2024-01-02", periods=2)),
            trades=all_trades,
            symbol_results=results,
        )
        ha = report["holding_attribution"]
        contribs = {c["symbol"]: c["contribution"] for c in ha["symbol_contributions"]}
        assert contribs["AAA"] == pytest.approx(1000.0)
        assert contribs["BBB"] == pytest.approx(-400.0)
        assert contribs["CCC"] == pytest.approx(500.0)

        # 净值之和 = 1100，AAA 贡献占比 = 1000/1100
        aaa_item = next(c for c in ha["symbol_contributions"] if c["symbol"] == "AAA")
        assert aaa_item["pct"] == pytest.approx(1000.0 / 1100.0)

        assert ha["top_contributor"]["symbol"] == "AAA"
        assert ha["top_contributor"]["contribution"] == pytest.approx(1000.0)
        assert ha["top_drag"]["symbol"] == "BBB"
        assert ha["top_drag"]["contribution"] == pytest.approx(-400.0)

    def test_no_symbol_results_returns_empty(self):
        attr = PerformanceAttribution()
        report = attr.analyze(equity_curve=_make_equity([1.0, 1.0]), trades=[])
        ha = report["holding_attribution"]
        assert ha["symbol_contributions"] == []
        assert ha["top_contributor"]["symbol"] is None


# ---------------------------------------------------------------------------
# 信号质量（策略归因）
# ---------------------------------------------------------------------------

class TestSignalQuality:
    def test_buy_and_sell_signal_quality(self):
        """构造单调上涨行情，验证买入后 N 日收益为正、卖出后仍上涨。"""
        # 10 个交易日，价格从 10 涨到 19（每日 +1）
        idx = pd.bdate_range("2024-01-02", periods=10)
        close = pd.Series(np.arange(10, 20, dtype=float), index=idx)
        df = pd.DataFrame({"close": close, "open": close})

        trades = [
            _make_trade(idx[0], "AAA", "buy", price=10.0),
            _make_trade(idx[1], "AAA", "sell", price=11.0, pnl=100.0),
        ]
        attr = PerformanceAttribution()
        report = attr.analyze(
            equity_curve=pd.Series([1.0, 1.01], index=idx[:2]),
            trades=trades,
            lookforward_days=3,
            data={"AAA": df},
        )
        sa = report["strategy_attribution"]
        # 买入(idx0, close=10) -> idx3 close=13 -> 收益 = 30%
        assert sa["buy_signal_quality"]["sample_count"] == 1
        assert sa["buy_signal_quality"]["avg_return_after_n_days"] == pytest.approx(0.30)
        # 卖出(idx1, close=11) -> idx4 close=14 -> 收益 = 27.3%（卖早了）
        assert sa["sell_signal_quality"]["sample_count"] == 1
        assert sa["sell_signal_quality"]["avg_return_after_n_days"] == pytest.approx(
            14.0 / 11.0 - 1.0
        )

    def test_lookforward_out_of_range_skipped(self):
        """信号后不足 N 日的样本被跳过。"""
        idx = pd.bdate_range("2024-01-02", periods=5)
        close = pd.Series(np.arange(10, 15, dtype=float), index=idx)
        df = pd.DataFrame({"close": close, "open": close})
        trades = [_make_trade(idx[-1], "AAA", "buy", price=14.0)]  # 最后一天买入
        attr = PerformanceAttribution()
        report = attr.analyze(
            equity_curve=pd.Series([1.0] * 5, index=idx),
            trades=trades,
            lookforward_days=5,
            data={"AAA": df},
        )
        assert report["strategy_attribution"]["buy_signal_quality"]["sample_count"] == 0
