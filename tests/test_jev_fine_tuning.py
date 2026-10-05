"""Jev 微调 Pipeline 测试。"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from jev.fine_tuning import FineTuningConfig, JevFineTuner


@pytest.fixture
def sample_training_df():
    """构造模拟训练数据。"""
    n = 200
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "price": rng.normal(100, 10, n),
        "price_change_5d": rng.normal(0, 0.05, n),
        "ma5_ma20_ratio": rng.normal(1.0, 0.1, n),
        "volume_ratio": rng.normal(1.0, 0.5, n),
        "rsi": rng.uniform(0, 100, n),
        "macd_signal": rng.choice([-1, 0, 1], n),
        "volatility_20d": rng.uniform(0.01, 0.05, n),
        "label": rng.choice(["buy", "sell", "hold"], n),
    })


@pytest.fixture
def sample_jsonl_path(sample_training_df):
    """将训练数据写成临时 JSONL。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for _, row in sample_training_df.iterrows():
            features = [
                {"feature": col, "value": float(row[col])}
                for col in sample_training_df.columns if col != "label"
            ]
            rec = {"features": features, "label": row["label"]}
            f.write(json.dumps(rec) + "\n")
        path = f.name
    yield path
    os.unlink(path)


def test_config_validation():
    """不支持的模型类型应抛 ValueError。"""
    with pytest.raises(ValueError):
        FineTuningConfig(model_type="unsupported")


def test_load_jsonl(sample_jsonl_path):
    """JSONL 加载应正确还原 DataFrame。"""
    df = JevFineTuner.load_jsonl(sample_jsonl_path)
    assert len(df) == 200
    assert "label" in df.columns
    assert "price" in df.columns


def test_train_gradient_boosting(sample_training_df):
    """GradientBoosting 训练应成功并返回指标。"""
    config = FineTuningConfig(model_type="gradient_boosting", test_size=0.2)
    tuner = JevFineTuner(config)
    result = tuner.train(sample_training_df)

    assert result.model_type == "gradient_boosting"
    assert 0 <= result.accuracy <= 1.0
    assert result.train_samples > 0
    assert result.val_samples > 0
    assert result.model_path != ""
    assert len(result.feature_importance) > 0


@pytest.mark.skipif(
    pytest.importorskip("lightgbm", reason="lightgbm 未安装") is None,
    reason="lightgbm 未安装",
)
def test_train_lightgbm(sample_training_df):
    """LightGBM 训练应成功。"""
    config = FineTuningConfig(model_type="lightgbm", test_size=0.2)
    tuner = JevFineTuner(config)
    result = tuner.train(sample_training_df)

    assert result.model_type == "lightgbm"
    assert 0 <= result.accuracy <= 1.0
    assert result.model_path != ""


@pytest.mark.skipif(
    pytest.importorskip("xgboost", reason="xgboost 未安装") is None,
    reason="xgboost 未安装",
)
def test_train_xgboost(sample_training_df):
    """XGBoost 训练应成功。"""
    config = FineTuningConfig(model_type="xgboost", test_size=0.2)
    tuner = JevFineTuner(config)
    result = tuner.train(sample_training_df)

    assert result.model_type == "xgboost"
    assert 0 <= result.accuracy <= 1.0
    assert result.model_path != ""


def test_predict_after_train(sample_training_df):
    """训练后应能推理单条样本。"""
    config = FineTuningConfig(model_type="gradient_boosting", test_size=0.2)
    tuner = JevFineTuner(config)
    tuner.train(sample_training_df)

    features = {
        "price": 100.0,
        "price_change_5d": 0.02,
        "ma5_ma20_ratio": 1.05,
        "volume_ratio": 1.2,
        "rsi": 55.0,
        "macd_signal": 1,
        "volatility_20d": 0.02,
    }
    label, probs = tuner.predict(features)
    assert label in ("0", "1", "2")
    assert sum(probs.values()) > 0.99


def test_save_and_load(sample_training_df):
    """模型应能正确保存和加载。"""
    config = FineTuningConfig(model_type="gradient_boosting", test_size=0.2)
    tuner = JevFineTuner(config)
    result = tuner.train(sample_training_df)

    loaded = JevFineTuner.load(result.model_path)
    assert loaded.model is not None
    assert loaded.feature_names == tuner.feature_names


def test_result_to_dict(sample_training_df):
    """FineTuningResult 应能序列化为 dict。"""
    config = FineTuningConfig(model_type="gradient_boosting", test_size=0.2)
    tuner = JevFineTuner(config)
    result = tuner.train(sample_training_df)
    d = result.to_dict()
    assert "accuracy" in d
    assert "feature_importance" in d
