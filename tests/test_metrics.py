"""回测绩效指标单元测试。

覆盖: 累计收益/年化/最大回撤/夏普/胜率/盈亏比/总盈利/总亏损/全量指标
边界: 空数据/单条/全赢/全亏/零波动/负资产
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    calc_all_metrics,
    calc_annualized_return,
    calc_cumulative_return,
    calc_max_drawdown,
    calc_profit_loss_ratio,
    calc_sharpe_ratio,
    calc_total_loss,
    calc_total_profit,
    calc_win_rate,
    metrics_to_dataframe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def equity_up() -> pd.Series:
    """稳定上涨的净值曲线。"""
    return pd.Series([1.0, 1.02, 1.05, 1.08, 1.10], name="equity")


@pytest.fixture
def equity_down() -> pd.Series:
    """持续下跌的净值曲线。"""
    return pd.Series([1.0, 0.98, 0.95, 0.92, 0.90], name="equity")


@pytest.fixture
def equity_volatile() -> pd.Series:
    """有涨有跌、含回撤的净值曲线。"""
    return pd.Series([1.0, 1.10, 1.05, 1.15, 1.08, 1.20], name="equity")


@pytest.fixture
def mixed_trades() -> list[dict]:
    """混合盈亏交易记录。"""
    return [
        {"pnl": 500.0, "action": "sell"},
        {"pnl": -200.0, "action": "sell"},
        {"pnl": 800.0, "action": "sell"},
        {"pnl": -100.0, "action": "sell"},
        {"pnl": None, "action": "buy"},  # 未平仓
    ]


# ---------------------------------------------------------------------------
# calc_cumulative_return
# ---------------------------------------------------------------------------

class TestCumulativeReturn:
    def test_up_trend(self, equity_up):
        result = calc_cumulative_return(equity_up)
        assert result == pytest.approx(0.10, abs=1e-6)

    def test_down_trend(self, equity_down):
        result = calc_cumulative_return(equity_down)
        assert result == pytest.approx(-0.10, abs=1e-6)

    def test_empty_returns_zero(self):
        assert calc_cumulative_return(pd.Series([], dtype=float)) == 0.0

    def test_single_value_returns_zero(self):
        assert calc_cumulative_return(pd.Series([1.0])) == 0.0

    def test_flat_returns_zero(self):
        assert calc_cumulative_return(pd.Series([1.0, 1.0, 1.0])) == 0.0


# ---------------------------------------------------------------------------
# calc_annualized_return
# ---------------------------------------------------------------------------

class TestAnnualizedReturn:
    def test_up_trend_252_days(self, equity_up):
        result = calc_annualized_return(equity_up, trading_days=252)
        # 5天涨10%，年化 = 1.1^(252/5) - 1
        expected = 1.1 ** (252 / 5) - 1
        assert result == pytest.approx(expected, rel=1e-4)

    def test_empty_returns_zero(self):
        assert calc_annualized_return(pd.Series([], dtype=float)) == 0.0

    def test_single_value_returns_zero(self):
        assert calc_annualized_return(pd.Series([1.0])) == 0.0

    def test_negative_total_returns_minus_one(self):
        # total_return <= 0 时返回 -1.0
        equity = pd.Series([1.0, 0.0])
        assert calc_annualized_return(equity) == -1.0


# ---------------------------------------------------------------------------
# calc_max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_volatile_has_drawdown(self, equity_volatile):
        result = calc_max_drawdown(equity_volatile)
        # 峰值1.10(第2天)，之后最低1.05(第3天)，回撤=(1.05-1.10)/1.10=-0.04545
        # 峰值1.15(第4天)，之后最低1.08(第5天)，回撤=(1.08-1.15)/1.15=-0.06087
        assert result == pytest.approx(-0.06087, abs=1e-4)

    def test_up_trend_no_drawdown(self, equity_up):
        assert calc_max_drawdown(equity_up) == 0.0

    def test_empty_returns_zero(self):
        assert calc_max_drawdown(pd.Series([], dtype=float)) == 0.0

    def test_single_value_returns_zero(self):
        assert calc_max_drawdown(pd.Series([1.0])) == 0.0

    def test_drawdown_is_negative_or_zero(self):
        equity = pd.Series([1.0, 0.8, 0.9])
        assert calc_max_drawdown(equity) <= 0.0


# ---------------------------------------------------------------------------
# calc_sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_volatile_has_sharpe(self, equity_volatile):
        result = calc_sharpe_ratio(equity_volatile, risk_free_rate=0.02, trading_days=252)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_empty_returns_zero(self):
        assert calc_sharpe_ratio(pd.Series([], dtype=float)) == 0.0

    def test_less_than_three_returns_zero(self):
        assert calc_sharpe_ratio(pd.Series([1.0, 1.01])) == 0.0

    def test_zero_volatility_returns_zero(self):
        # 完全平滑的曲线，std=0
        equity = pd.Series([1.0, 1.01, 1.02, 1.03])
        # pct_change = [0.01, 0.0099, 0.0098] 不是零，需要构造真正零波动
        equity = pd.Series([1.0, 1.0, 1.0, 1.0])
        assert calc_sharpe_ratio(equity) == 0.0


# ---------------------------------------------------------------------------
# calc_win_rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_mixed_trades(self, mixed_trades):
        # 2赢2亏，胜率=0.5
        assert calc_win_rate(mixed_trades) == pytest.approx(0.5)

    def test_all_wins(self):
        trades = [{"pnl": 100.0}, {"pnl": 200.0}]
        assert calc_win_rate(trades) == 1.0

    def test_all_losses(self):
        trades = [{"pnl": -100.0}, {"pnl": -200.0}]
        assert calc_win_rate(trades) == 0.0

    def test_empty_returns_zero(self):
        assert calc_win_rate([]) == 0.0

    def test_open_trades_excluded(self):
        trades = [{"pnl": None}, {"pnl": None}]
        assert calc_win_rate(trades) == 0.0

    def test_zero_pnl_not_counted_as_win(self):
        # pnl=0 不算赢
        trades = [{"pnl": 0.0}, {"pnl": 100.0}]
        assert calc_win_rate(trades) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# calc_profit_loss_ratio
# ---------------------------------------------------------------------------

class TestProfitLossRatio:
    def test_mixed_trades(self, mixed_trades):
        # 平均盈利=(500+800)/2=650, 平均亏损=|-150|=150, 比率=650/150=4.333
        assert calc_profit_loss_ratio(mixed_trades) == pytest.approx(4.333, abs=1e-3)

    def test_no_wins_returns_zero(self):
        trades = [{"pnl": -100.0}, {"pnl": -200.0}]
        assert calc_profit_loss_ratio(trades) == 0.0

    def test_no_losses_returns_zero(self):
        trades = [{"pnl": 100.0}, {"pnl": 200.0}]
        assert calc_profit_loss_ratio(trades) == 0.0

    def test_empty_returns_zero(self):
        assert calc_profit_loss_ratio([]) == 0.0


# ---------------------------------------------------------------------------
# calc_total_profit / calc_total_loss
# ---------------------------------------------------------------------------

class TestTotalProfitLoss:
    def test_total_profit_mixed(self, mixed_trades):
        assert calc_total_profit(mixed_trades) == pytest.approx(1300.0)

    def test_total_loss_mixed(self, mixed_trades):
        assert calc_total_loss(mixed_trades) == pytest.approx(300.0)

    def test_total_profit_all_losses(self):
        trades = [{"pnl": -100.0}, {"pnl": -200.0}]
        assert calc_total_profit(trades) == 0.0

    def test_total_loss_all_wins(self):
        trades = [{"pnl": 100.0}, {"pnl": 200.0}]
        assert calc_total_loss(trades) == 0.0

    def test_empty_returns_zero(self):
        assert calc_total_profit([]) == 0.0
        assert calc_total_loss([]) == 0.0

    def test_open_trades_excluded(self):
        trades = [{"pnl": None}, {"pnl": 100.0}]
        assert calc_total_profit(trades) == 100.0
        assert calc_total_loss(trades) == 0.0


# ---------------------------------------------------------------------------
# calc_all_metrics
# ---------------------------------------------------------------------------

class TestAllMetrics:
    def test_returns_all_eleven_keys(self, equity_volatile, mixed_trades):
        result = calc_all_metrics(equity_volatile, mixed_trades)
        expected_keys = {
            "累计收益率", "年化收益率", "最大回撤", "夏普比率",
            "胜率", "盈亏比", "交易次数", "总盈利", "总亏损",
            "订单成交率", "平均持仓时间",
        }
        assert set(result.keys()) == expected_keys

    def test_trade_count_excludes_open(self, mixed_trades):
        result = calc_all_metrics(pd.Series([1.0, 1.1, 1.2]), mixed_trades)
        assert result["交易次数"] == 4  # 2赢2亏，排除1个未平仓

    def test_empty_trades(self, equity_up):
        result = calc_all_metrics(equity_up, [])
        assert result["交易次数"] == 0
        assert result["胜率"] == 0.0
        assert result["总盈利"] == 0.0
        assert result["总亏损"] == 0.0

    def test_values_are_floats(self, equity_volatile, mixed_trades):
        result = calc_all_metrics(equity_volatile, mixed_trades)
        for v in result.values():
            assert isinstance(v, (int, float, np.floating))


# ---------------------------------------------------------------------------
# metrics_to_dataframe
# ---------------------------------------------------------------------------

class TestMetricsToDataframe:
    def test_returns_dataframe(self, equity_volatile, mixed_trades):
        metrics = calc_all_metrics(equity_volatile, mixed_trades)
        df = metrics_to_dataframe(metrics)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 11
        assert list(df.columns) == ["指标", "数值", "原始值"]

    def test_percentage_formatting(self):
        metrics = {"累计收益率": 0.10}
        df = metrics_to_dataframe(metrics)
        assert df.iloc[0]["数值"] == "10.00%"

    def test_trade_count_formatting(self):
        metrics = {"交易次数": 7}
        df = metrics_to_dataframe(metrics)
        assert df.iloc[0]["数值"] == "7"

    def test_profit_formatting(self):
        metrics = {"总盈利": 12345.67}
        df = metrics_to_dataframe(metrics)
        assert "12,345.67" in df.iloc[0]["数值"]
