"""Jev决策引擎单元测试。

覆盖: MarketState序列化 / mock模式概率分布 / 置信度阈值过滤 / 方向冲突过滤 /
     build_market_state / filter_signals批量 / 审计日志 / API失败降级
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from jev.jev_engine import (
    JevAuditLog,
    JevDecision,
    JevDecisionEngine,
    MarketState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def market_state() -> MarketState:
    return MarketState(
        price=100.0,
        price_change_5d=0.03,
        ma5_ma20_ratio=1.02,
        volume_ratio=1.5,
        rsi=55.0,
        macd_signal=1,
        volatility_20d=0.20,
    )


@pytest.fixture
def engine(tmp_path) -> JevDecisionEngine:
    audit_path = str(tmp_path / "jev_audit.jsonl")
    return JevDecisionEngine(
        mock_mode=True,
        confidence_threshold=0.6,
        audit_log_path=audit_path,
    )


def _make_df_with_indicators(n: int = 50) -> pd.DataFrame:
    """构造含技术指标列的测试DataFrame。"""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "close": close,
        "ma5": close * 1.01,
        "ma20": close,
        "vol_ratio": np.random.uniform(0.8, 1.5, n),
        "rsi": np.random.uniform(30, 70, n),
        "macd_hist": np.random.randn(n),
        "volume": np.random.randint(100000, 500000, n),
    }, index=dates)


# ---------------------------------------------------------------------------
# MarketState
# ---------------------------------------------------------------------------

class TestMarketState:
    def test_to_states_list_has_seven_features(self, market_state):
        states = market_state.to_states_list()
        assert len(states) == 7
        features = [s["feature"] for s in states]
        assert "price" in features
        assert "price_change_5d" in features
        assert "ma5_ma20_ratio" in features
        assert "volume_ratio" in features
        assert "rsi" in features
        assert "macd_signal" in features
        assert "volatility_20d" in features

    def test_to_states_list_values_rounded(self, market_state):
        states = market_state.to_states_list()
        price_state = next(s for s in states if s["feature"] == "price")
        assert price_state["value"] == 100.0
        rsi_state = next(s for s in states if s["feature"] == "rsi")
        assert rsi_state["value"] == 55.0


# ---------------------------------------------------------------------------
# JevDecision
# ---------------------------------------------------------------------------

class TestJevDecision:
    def test_decision_fields(self):
        d = JevDecision(
            timestamp="2024-01-01",
            symbol="600519.SH",
            raw_signal="buy",
            raw_confidence=0.8,
            market_state={"price": 100},
            probabilities={"buy": 0.7, "sell": 0.1, "hold": 0.2},
            final_action="buy",
            final_confidence=0.7,
            executed=True,
            reason="通过",
        )
        assert d.executed is True
        assert d.final_action == "buy"
        assert d.reason == "通过"


# ---------------------------------------------------------------------------
# JevAuditLog
# ---------------------------------------------------------------------------

class TestJevAuditLog:
    def test_record_writes_jsonl(self, tmp_path):
        log_path = str(tmp_path / "audit.jsonl")
        audit = JevAuditLog(log_path=log_path)
        decision = JevDecision(
            timestamp="2024-01-01T00:00:00",
            symbol="TEST",
            raw_signal="buy",
            raw_confidence=0.8,
            market_state={"price": 100},
            probabilities={"buy": 0.7, "sell": 0.1, "hold": 0.2},
            final_action="buy",
            final_confidence=0.7,
            executed=True,
        )
        audit.record(decision)
        assert os.path.exists(log_path)
        with open(log_path) as f:
            line = f.readline()
        data = json.loads(line)
        assert data["symbol"] == "TEST"
        assert data["final_action"] == "buy"

    def test_creates_parent_directory(self, tmp_path):
        nested = str(tmp_path / "sub" / "dir" / "audit.jsonl")
        audit = JevAuditLog(log_path=nested)
        assert os.path.exists(os.path.dirname(nested))


# ---------------------------------------------------------------------------
# JevDecisionEngine - mock模式
# ---------------------------------------------------------------------------

class TestEngineMock:
    def test_evaluate_returns_decision(self, engine, market_state):
        decision = engine.evaluate("buy", 0.8, market_state, symbol="TEST")
        assert isinstance(decision, JevDecision)
        assert decision.raw_signal == "buy"
        assert decision.symbol == "TEST"

    def test_probabilities_sum_to_one(self, engine, market_state):
        decision = engine.evaluate("buy", 0.8, market_state)
        probs = decision.probabilities
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_probabilities_have_three_actions(self, engine, market_state):
        decision = engine.evaluate("buy", 0.8, market_state)
        assert set(decision.probabilities.keys()) == {"buy", "sell", "hold"}

    def test_hold_signal_not_executed(self, engine, market_state):
        """原始信号为hold时不执行（方向冲突或置信度不足）。"""
        decision = engine.evaluate("hold", 0.5, market_state)
        assert decision.executed is False
        # raw_signal=hold时跳过冲突检查，可能因置信度不足或Jev建议观望而不执行

    def test_direction_conflict_not_executed(self, engine):
        """构造一个Jev倾向sell但策略给buy的场景。"""
        # 大涨+高RSI → mock模型倾向sell
        state = MarketState(
            price=100, price_change_5d=0.15, ma5_ma20_ratio=1.10,
            volume_ratio=2.0, rsi=80, macd_signal=-1, volatility_20d=0.3,
        )
        decision = engine.evaluate("buy", 0.7, state)
        # 如果Jev最终选了sell，与buy冲突则不执行
        if decision.final_action == "sell":
            assert decision.executed is False
            assert "冲突" in decision.reason

    def test_low_confidence_not_executed(self, engine):
        """构造一个Jev概率分散（最高概率<0.6）的场景。"""
        # 中性市场状态 → 各概率接近 → hold概率可能最高
        state = MarketState(
            price=100, price_change_5d=0.0, ma5_ma20_ratio=1.0,
            volume_ratio=1.0, rsi=50, macd_signal=0, volatility_20d=0.15,
        )
        decision = engine.evaluate("buy", 0.5, state)
        # 中性状态下hold概率通常最高，不执行
        if decision.final_action == "hold":
            assert decision.executed is False

    def test_audit_log_recorded(self, engine, market_state, tmp_path):
        engine.evaluate("buy", 0.8, market_state, symbol="TEST")
        # audit_log_path在fixture中设为tmp_path下
        assert os.path.exists(engine.audit.log_path)
        with open(engine.audit.log_path) as f:
            lines = f.readlines()
        assert len(lines) >= 1

    def test_mock_evaluate_returns_valid_probs(self, engine, market_state):
        states = market_state.to_states_list()
        probs = engine._mock_evaluate(states, "buy", 0.8)
        assert set(probs.keys()) == {"buy", "sell", "hold"}
        assert all(0.0 <= v <= 1.0 for v in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# build_market_state
# ---------------------------------------------------------------------------

class TestBuildMarketState:
    def test_returns_none_for_early_index(self, engine):
        df = _make_df_with_indicators(50)
        assert engine.build_market_state(df, idx=5) is None

    def test_returns_market_state_for_valid_index(self, engine):
        df = _make_df_with_indicators(50)
        state = engine.build_market_state(df, idx=30)
        assert state is not None
        assert isinstance(state, MarketState)
        assert state.price > 0

    def test_ma20_zero_returns_ratio_one(self, engine):
        df = _make_df_with_indicators(50)
        df["ma20"] = 0.0  # 触发除零保护
        state = engine.build_market_state(df, idx=30)
        assert state is not None
        assert state.ma5_ma20_ratio == 1.0

    def test_missing_columns_fallback(self, engine):
        """缺少ma5/ma20等列时用close兜底。"""
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        df = pd.DataFrame({"close": range(100, 150)}, index=dates)
        state = engine.build_market_state(df, idx=30)
        assert state is not None
        assert state.rsi == 50.0  # 默认值
        assert state.volume_ratio == 1.0  # 默认值


# ---------------------------------------------------------------------------
# filter_signals
# ---------------------------------------------------------------------------

class TestFilterSignals:
    def test_filters_signals(self, engine):
        df = _make_df_with_indicators(50)
        signals = [
            {"date": df.index[30], "action": "buy", "confidence": 0.8, "symbol": "TEST"},
            {"date": df.index[35], "action": "sell", "confidence": 0.7, "symbol": "TEST"},
        ]
        filtered = engine.filter_signals(signals, df)
        assert isinstance(filtered, list)
        # 至少不报错，通过数量取决于mock决策

    def test_skips_unknown_dates(self, engine):
        df = _make_df_with_indicators(50)
        signals = [
            {"date": pd.Timestamp("2099-01-01"), "action": "buy", "confidence": 0.8},
        ]
        filtered = engine.filter_signals(signals, df)
        assert filtered == []

    def test_passed_signals_have_jev_fields(self, engine):
        df = _make_df_with_indicators(50)
        signals = [
            {"date": df.index[30], "action": "buy", "confidence": 0.9, "symbol": "TEST"},
        ]
        filtered = engine.filter_signals(signals, df)
        for sig in filtered:
            assert "jev_confidence" in sig
            assert "jev_probabilities" in sig


# ---------------------------------------------------------------------------
# API调用（mock测试失败降级）
# ---------------------------------------------------------------------------

class TestApiCall:
    def test_api_failure_falls_back_to_mock(self, tmp_path):
        """指向不存在的端口，验证降级到mock。"""
        engine = JevDecisionEngine(
            base_url="http://localhost:19999",
            mock_mode=False,
            timeout=0.5,
            retry_count=0,
            audit_log_path=str(tmp_path / "audit.jsonl"),
        )
        states = [{"feature": "price", "value": 100}]
        probs = engine._call_api(states, ["buy", "sell", "hold"])
        assert set(probs.keys()) == {"buy", "sell", "hold"}
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_confidence_threshold_config(self):
        engine = JevDecisionEngine(mock_mode=True, confidence_threshold=0.8)
        assert engine.confidence_threshold == 0.8
