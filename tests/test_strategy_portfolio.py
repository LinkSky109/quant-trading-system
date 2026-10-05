"""多策略组合引擎单元测试。

运行:
    cd quant_trading_system
    python -m pytest tests/test_strategy_portfolio.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

from backtest.engine import BacktestEngine
from data.data_fetcher import _generate_mock_klines
from strategies.ma_cross import MACrossStrategy
from strategies.strategy_portfolio import (
    StrategyPortfolio,
    StrategyPortfolioResult,
)

# 组合级绩效必须包含的核心指标
CORE_METRICS = [
    "累计收益率", "年化收益率", "最大回撤",
    "夏普比率", "胜率", "盈亏比",
]

# mock 基准价：高价蓝筹等权拆分资金后单笔 20% 仓位不足 1 手，
# 用亲民价格确保策略信号能真实成交（与 examples/run_portfolio_backtest.py 一致）
MOCK_BASE_PRICE = 25.0


# ---------------------------------------------------------------------------
# 测试数据工具
# ---------------------------------------------------------------------------

def make_data(count: int = 250, start_date: str = "2024-01-02") -> pd.DataFrame:
    """生成单标的 mock 行情数据（确定性，不依赖网络）。"""
    return _generate_mock_klines(
        "600519.SH", period="1d", count=count,
        start_date=start_date, base_price=MOCK_BASE_PRICE,
    )


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestStrategyPortfolio:
    """多策略组合引擎测试。"""

    def test_three_strategies_equal_weight_runs(self):
        """1. 3 策略等权组合回测 -> 返回组合净值 + 绩效 + 单策略明细。"""
        data = make_data()
        sp = StrategyPortfolio(
            ["ma_cross", "bollinger", "momentum_breakout"],
            initial_capital=1_000_000.0,
        )
        result = sp.run(data, symbol="600519.SH")

        assert isinstance(result, StrategyPortfolioResult)
        # 组合净值
        assert len(result.portfolio_equity_curve) > 0
        # 组合绩效包含核心指标
        for key in CORE_METRICS:
            assert key in result.portfolio_metrics, f"缺少核心指标: {key}"
        # 三个策略明细都在
        assert set(result.strategy_results.keys()) == {
            "ma_cross", "bollinger", "momentum_breakout"
        }
        # 每个策略都有独立 metrics
        for name, res in result.strategy_results.items():
            assert "累计收益率" in res.metrics

    def test_equal_weight_capital_allocation(self):
        """2. 等权时每个策略分到的初始资金 = 总资金 / 3。"""
        data = make_data()
        total = 1_000_000.0
        sp = StrategyPortfolio(
            ["ma_cross", "bollinger", "momentum_breakout"],
            initial_capital=total,
        )
        result = sp.run(data, symbol="600519.SH")

        # 权重 = 1/3
        expected_w = pytest.approx(1.0 / 3.0, abs=1e-9)
        for w in result.strategy_weights.values():
            assert w == expected_w
        assert sum(result.strategy_weights.values()) == pytest.approx(1.0, abs=1e-9)

        # 每个子账户首日净值 = 分配到的资金 = total/3
        for res in result.strategy_results.values():
            assert float(res.equity_curve.iloc[0]) == pytest.approx(total / 3.0, rel=1e-6)

        # 组合首日净值 = 总资金
        assert float(result.portfolio_equity_curve.iloc[0]) == pytest.approx(total, rel=1e-6)

    def test_custom_weights_take_effect(self):
        """3. 自定义权重生效（自动归一化）。"""
        data = make_data()
        sp = StrategyPortfolio(
            ["ma_cross", "bollinger"],
            weights=[1.0, 3.0],
            initial_capital=1_000_000.0,
        )
        result = sp.run(data, symbol="600519.SH")

        w = result.strategy_weights
        assert w["ma_cross"] == pytest.approx(0.25, abs=1e-9)
        assert w["bollinger"] == pytest.approx(0.75, abs=1e-9)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

        # 子账户资金与权重成正比
        eq_ma = result.strategy_results["ma_cross"].equity_curve.iloc[0]
        eq_bo = result.strategy_results["bollinger"].equity_curve.iloc[0]
        assert float(eq_bo) / float(eq_ma) == pytest.approx(3.0, abs=1e-6)

    def test_conflict_log_recorded(self):
        """4. 信号冲突有记录，且记录结构正确、确实存在买/卖分歧。"""
        data = make_data()
        sp = StrategyPortfolio(
            ["ma_cross", "bollinger", "momentum_breakout"],
            initial_capital=1_000_000.0,
        )
        result = sp.run(data, symbol="600519.SH")

        # mock 数据确定性，3 策略应产生若干分歧日
        assert len(result.conflict_log) >= 1, "预期至少出现一个信号分歧日"

        for entry in result.conflict_log:
            # 必填字段
            assert "date" in entry
            assert "directions" in entry
            assert "net_vote" in entry
            assert "decision" in entry
            # directions 覆盖全部策略
            assert set(entry["directions"].keys()) == {
                "ma_cross", "bollinger", "momentum_breakout"
            }
            # 确实同时存在买(+1)与卖(-1)
            dirs = list(entry["directions"].values())
            assert 1 in dirs and -1 in dirs
            # decision 合法
            assert entry["decision"] in ("buy", "sell", "hold")
            # net_vote 与 decision 自洽
            if entry["net_vote"] > 0.1:
                assert entry["decision"] == "buy"
            elif entry["net_vote"] < -0.1:
                assert entry["decision"] == "sell"
            else:
                assert entry["decision"] == "hold"

    def test_single_strategy_equals_direct_backtest(self):
        """5. 单策略组合的结果 == 直接用 BacktestEngine 回测。"""
        data = make_data()
        total = 800_000.0

        # 组合：只放一个策略，权重自动 = 1.0
        sp = StrategyPortfolio(["ma_cross"], initial_capital=total)
        port_result = sp.run(data, symbol="600519.SH")

        # 直接回测（相同资金）
        standalone = BacktestEngine(initial_capital=total).run(
            data, MACrossStrategy(), symbol="600519.SH"
        )

        # 权重 = 1.0
        assert port_result.strategy_weights["ma_cross"] == pytest.approx(1.0)
        # 组合净值 == 单策略净值
        pd.testing.assert_series_equal(
            port_result.portfolio_equity_curve,
            standalone.equity_curve,
            check_names=False,
        )
        # 组合级 metrics 与单策略一致
        assert port_result.portfolio_metrics["累计收益率"] == pytest.approx(
            standalone.metrics["累计收益率"], rel=1e-9
        )

    def test_contribution_analysis(self):
        """6. 贡献度分析：pnl/return/contribution_pct 齐全且占比和≈1。"""
        data = make_data()
        sp = StrategyPortfolio(
            ["ma_cross", "bollinger"], initial_capital=1_000_000.0
        )
        result = sp.run(data, symbol="600519.SH")

        assert set(result.contribution.keys()) == {"ma_cross", "bollinger"}
        total_pnl = 0.0
        for name, info in result.contribution.items():
            assert {"pnl", "return_pct", "contribution_pct"} <= set(info.keys())
            total_pnl += info["pnl"]

        # 组合总盈亏 = 各策略 pnl 之和
        # 注意：contribution 中的 pnl 已 round 到 2 位小数，因此用 abs 容差
        port_pnl = (result.portfolio_equity_curve.iloc[-1]
                    - result.portfolio_equity_curve.iloc[0])
        assert total_pnl == pytest.approx(float(port_pnl), abs=0.02)

        # contribution_pct 之和 ≈ 1（组合整体盈利或亏损时）
        if abs(port_pnl) > 1e-6:
            contrib_sum = sum(i["contribution_pct"] for i in result.contribution.values())
            assert contrib_sum == pytest.approx(1.0, abs=1e-6)

    def test_empty_data_no_crash(self):
        """7. 空数据输入不崩溃。"""
        sp = StrategyPortfolio(["ma_cross", "bollinger"])
        result = sp.run(pd.DataFrame(), symbol="600519.SH")
        assert len(result.portfolio_equity_curve) == 0
        assert result.strategy_results == {}
        assert result.conflict_log == []

    def test_empty_strategies_raises(self):
        """8. 空策略列表应抛出 ValueError。"""
        with pytest.raises(ValueError):
            StrategyPortfolio([])

    def test_weights_length_mismatch_raises(self):
        """9. 权重长度与策略数不一致应抛出 ValueError。"""
        with pytest.raises(ValueError):
            StrategyPortfolio(["ma_cross", "bollinger"], weights=[0.5])

    def test_unknown_strategy_raises(self):
        """10. 未知策略名在 run 时抛出 ValueError。"""
        data = make_data()
        sp = StrategyPortfolio(["not_a_strategy"])
        with pytest.raises(ValueError):
            sp.run(data, symbol="600519.SH")
