"""机器学习进阶策略测试（XGBoost / LightGBM / Stacking）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_mock_klines(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """构造带趋势的模拟 K 线数据。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    ret = rng.normal(0.0005, 0.02, size=n)
    close = 100.0 * np.cumprod(1 + ret)
    open_ = close * (1 + rng.normal(0, 0.005, size=n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.01, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.01, size=n))
    volume = rng.integers(500_000, 2_000_000, size=n).astype(float)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)


# ---------------------------------------------------------------------------
# AdvancedMLPredictor
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    pytest.importorskip("xgboost", reason="xgboost 未安装") is None,
    reason="xgboost 未安装",
)
def test_advanced_predictor_xgboost_train():
    """XGBoost predictor 应能训练并预测。"""
    from ml.predictor import MLPredictor
    from strategies.ml_advanced import AdvancedMLPredictor

    df = _make_mock_klines(200)
    features = MLPredictor.build_features(df)
    labels = MLPredictor.build_labels(df, forward_days=5)

    combined = features.copy()
    combined["label"] = labels
    combined = combined.dropna()
    X = combined[features.columns]
    y = combined["label"]

    predictor = AdvancedMLPredictor(model_type="xgboost")
    metrics = predictor.train(X, y)

    assert metrics["model_type"] == "xgboost"
    assert metrics["train_samples"] > 50
    assert "train_accuracy" in metrics

    probs = predictor.predict_proba(X.iloc[:10])
    assert len(probs) == 10
    assert all(0 <= p <= 1 for p in probs)


@pytest.mark.skipif(
    pytest.importorskip("lightgbm", reason="lightgbm 未安装") is None,
    reason="lightgbm 未安装",
)
def test_advanced_predictor_lightgbm_train():
    """LightGBM predictor 应能训练并预测。"""
    from ml.predictor import MLPredictor
    from strategies.ml_advanced import AdvancedMLPredictor

    df = _make_mock_klines(200)
    features = MLPredictor.build_features(df)
    labels = MLPredictor.build_labels(df, forward_days=5)

    combined = features.copy()
    combined["label"] = labels
    combined = combined.dropna()
    X = combined[features.columns]
    y = combined["label"]

    predictor = AdvancedMLPredictor(model_type="lightgbm")
    metrics = predictor.train(X, y)

    assert metrics["model_type"] == "lightgbm"
    assert metrics["train_samples"] > 50

    probs = predictor.predict_proba(X.iloc[:10])
    assert len(probs) == 10


# ---------------------------------------------------------------------------
# StackingEnsemblePredictor
# ---------------------------------------------------------------------------

def test_stacking_ensemble_train():
    """Stacking 集成应能训练并预测。"""
    from ml.predictor import MLPredictor
    from strategies.ml_advanced import StackingEnsemblePredictor

    df = _make_mock_klines(200)
    features = MLPredictor.build_features(df)
    labels = MLPredictor.build_labels(df, forward_days=5)

    combined = features.copy()
    combined["label"] = labels
    combined = combined.dropna()
    X = combined[features.columns]
    y = combined["label"]

    ensemble = StackingEnsemblePredictor(forward_days=5)
    metrics = ensemble.train(X, y)

    assert metrics["ensemble_type"] == "stacking"
    assert "random_forest" in metrics["base_models"]
    assert "xgboost" in metrics["base_models"]
    assert "lightgbm" in metrics["base_models"]
    assert metrics["train_samples"] > 50

    probs = ensemble.predict_proba(X.iloc[:10])
    assert len(probs) == 10
    assert all(0 <= p <= 1 for p in probs)


# ---------------------------------------------------------------------------
# MLAdvancedStrategy
# ---------------------------------------------------------------------------

def test_ml_advanced_strategy_untrained():
    """未训练时应返回空信号列表。"""
    from strategies.ml_advanced import MLAdvancedStrategy

    df = _make_mock_klines(100)
    strategy = MLAdvancedStrategy(params={"model_type": "xgboost"})
    signals = strategy.generate_signals(df, symbol="TEST")
    assert isinstance(signals, list)


@pytest.mark.skipif(
    pytest.importorskip("xgboost", reason="xgboost 未安装") is None,
    reason="xgboost 未安装",
)
def test_ml_advanced_strategy_train_and_signal():
    """训练后应生成合法信号。"""
    from strategies.ml_advanced import MLAdvancedStrategy

    df = _make_mock_klines(200)
    strategy = MLAdvancedStrategy(params={"model_type": "xgboost"})
    strategy.train_model(df)

    signals = strategy.generate_signals(df, symbol="TEST")
    assert isinstance(signals, list)
    if len(signals) > 0:
        s = signals[0]
        assert s.strategy == "ml_advanced"
        assert s.action in ("buy", "sell")
        assert 0.0 <= s.confidence <= 1.0


def test_ml_advanced_stacking_untrained():
    """Stacking 未训练时应返回空信号。"""
    from strategies.ml_advanced import MLAdvancedStrategy

    df = _make_mock_klines(100)
    strategy = MLAdvancedStrategy(params={"use_stacking": True})
    signals = strategy.generate_signals(df, symbol="TEST")
    assert isinstance(signals, list)
