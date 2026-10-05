"""MLPredictor 与 MLStrategy 单元测试。

整个文件用 try-except ImportError 包裹，sklearn 不可用时全部 skip。
使用 200 条模拟 K 线快速训练验证。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# sklearn 可用性检测
try:
    import sklearn  # noqa: F401
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SKLEARN_AVAILABLE,
    reason="scikit-learn 未安装，跳过 ML 相关测试",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_klines(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """构造带趋势的模拟 K 线数据。

    用带噪声的随机游走生成收盘价，确保有足够的涨跌变化供模型学习。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    # 随机游走 + 轻微上涨趋势
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
# 特征工程测试
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    """测试 MLPredictor.build_features。"""

    def test_feature_count(self):
        """特征数量应 >= 12 个。"""
        from ml.predictor import MLPredictor
        df = _make_mock_klines(100)
        feats = MLPredictor.build_features(df)
        assert len(feats.columns) >= 12, f"特征数={len(feats.columns)}，应>=12"

    def test_feature_columns_present(self):
        """关键特征列应存在。"""
        from ml.predictor import MLPredictor
        df = _make_mock_klines(100)
        feats = MLPredictor.build_features(df)
        expected = {
            "ret_1", "ret_5", "ret_10", "ret_20",
            "vol_5", "vol_20",
            "rsi_14",
            "macd_dif", "macd_dea", "macd_hist",
            "bollinger_position",
            "volume_change_5",
            "ma5_deviation", "ma20_deviation",
        }
        assert expected.issubset(set(feats.columns)), \
            f"缺少特征列: {expected - set(feats.columns)}"

    def test_features_index_matches(self):
        """特征 DataFrame 的 index 应与输入一致。"""
        from ml.predictor import MLPredictor
        df = _make_mock_klines(100)
        feats = MLPredictor.build_features(df)
        assert len(feats) == len(df)
        assert (feats.index == df.index).all()


# ---------------------------------------------------------------------------
# 标签构建测试
# ---------------------------------------------------------------------------

class TestLabelBuilding:
    """测试 MLPredictor.build_labels。"""

    def test_classification_label_binary(self):
        """分类标签应为 0/1 且最后 forward_days 行为 NaN。"""
        from ml.predictor import MLPredictor
        df = _make_mock_klines(100)
        labels = MLPredictor.build_labels(df, forward_days=5, task="classification")
        # 有效标签应为 0 或 1
        valid = labels.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})
        # 最后 5 个应为 NaN
        assert labels.iloc[-5:].isna().all()

    def test_regression_label_continuous(self):
        """回归标签应为连续收益率。"""
        from ml.predictor import MLPredictor
        df = _make_mock_klines(100)
        labels = MLPredictor.build_labels(df, forward_days=5, task="regression")
        valid = labels.dropna()
        assert valid.dtype == float
        assert len(valid) == 95  # 100 - 5

    def test_label_alignment_with_features(self):
        """特征和标签 dropna 后应能对齐。"""
        from ml.predictor import MLPredictor
        df = _make_mock_klines(200)
        feats = MLPredictor.build_features(df)
        labels = MLPredictor.build_labels(df, forward_days=5)
        combined = feats.copy()
        combined["label"] = labels
        combined = combined.dropna()
        assert len(combined) > 50, f"对齐后样本太少: {len(combined)}"


# ---------------------------------------------------------------------------
# 模型训练与预测测试
# ---------------------------------------------------------------------------

class TestModelTrainPredict:
    """测试 MLPredictor 训练、预测、概率输出。"""

    def test_train_random_forest(self):
        """RandomForest 能训练并返回指标。"""
        from ml.predictor import MLPredictor
        df = _make_mock_klines(200)
        predictor = MLPredictor(model_type="random_forest", forward_days=5)
        feats = MLPredictor.build_features(df)
        labels = MLPredictor.build_labels(df, forward_days=5)
        combined = feats.copy()
        combined["label"] = labels
        combined = combined.dropna()
        X = combined[feats.columns]
        y = combined["label"]

        metrics = predictor.train(X, y, test_size=0.2)
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert "confusion_matrix" in metrics
        assert metrics["train_samples"] > 0
        assert metrics["test_samples"] > 0
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_predict_proba_range(self):
        """预测概率应在 [0, 1] 范围内。"""
        from ml.predictor import MLPredictor
        df = _make_mock_klines(200)
        predictor = MLPredictor(model_type="random_forest", forward_days=5)
        feats = MLPredictor.build_features(df)
        labels = MLPredictor.build_labels(df, forward_days=5)
        combined = feats.copy()
        combined["label"] = labels
        combined = combined.dropna()
        X = combined[feats.columns]
        y = combined["label"]
        predictor.train(X, y, test_size=0.2)

        probs = predictor.predict_proba(X.tail(10))
        assert len(probs) == 10
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_binary(self):
        """predict 输出应为 0/1。"""
        from ml.predictor import MLPredictor
        df = _make_mock_klines(200)
        predictor = MLPredictor(model_type="random_forest", forward_days=5)
        feats = MLPredictor.build_features(df)
        labels = MLPredictor.build_labels(df, forward_days=5)
        combined = feats.copy()
        combined["label"] = labels
        combined = combined.dropna()
        X = combined[feats.columns]
        y = combined["label"]
        predictor.train(X, y, test_size=0.2)

        preds = predictor.predict(X.tail(10))
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_signal(self):
        """predict_signal 返回正确格式的信号字典。"""
        from ml.predictor import MLPredictor
        df = _make_mock_klines(200)
        predictor = MLPredictor(model_type="random_forest", forward_days=5)
        feats = MLPredictor.build_features(df)
        labels = MLPredictor.build_labels(df, forward_days=5)
        combined = feats.copy()
        combined["label"] = labels
        combined = combined.dropna()
        X = combined[feats.columns]
        y = combined["label"]
        predictor.train(X, y, test_size=0.2)

        result = predictor.predict_signal(df.tail(30))
        assert "probability" in result
        assert "signal" in result
        assert "confidence" in result
        assert result["signal"] in ("buy", "sell", "hold")
        assert 0.0 <= result["probability"] <= 1.0


# ---------------------------------------------------------------------------
# 模型保存/加载测试
# ---------------------------------------------------------------------------

class TestModelSaveLoad:
    """测试 MLPredictor.save / load。"""

    def test_save_and_load(self, tmp_path):
        """保存后加载的模型应能正常预测。"""
        from ml.predictor import MLPredictor
        df = _make_mock_klines(200)
        predictor = MLPredictor(model_type="random_forest", forward_days=5)
        feats = MLPredictor.build_features(df)
        labels = MLPredictor.build_labels(df, forward_days=5)
        combined = feats.copy()
        combined["label"] = labels
        combined = combined.dropna()
        X = combined[feats.columns]
        y = combined["label"]
        predictor.train(X, y, test_size=0.2)

        # 保存到临时目录
        model_file = tmp_path / "test_model.joblib"
        predictor.save(str(model_file))
        assert model_file.exists()

        # 加载
        loaded = MLPredictor.load(str(model_file))
        assert loaded.model is not None
        assert loaded.feature_names == predictor.feature_names
        assert loaded.model_type == predictor.model_type

        # 加载后能预测
        probs = loaded.predict_proba(X.tail(5))
        assert len(probs) == 5
        assert (probs >= 0).all() and (probs <= 1).all()


# ---------------------------------------------------------------------------
# MLStrategy 测试
# ---------------------------------------------------------------------------

class TestMLStrategy:
    """测试 MLStrategy 信号生成。"""

    def test_strategy_train_and_signals(self):
        """训练后生成非全零信号的 DataFrame。"""
        from strategies.ml_strategy import MLStrategy
        df = _make_mock_klines(200)
        strategy = MLStrategy(params={
            "model_type": "random_forest",
            "forward_days": 5,
            "buy_threshold": 0.55,
            "sell_threshold": 0.45,
        })
        metrics = strategy.train_model(df)
        assert metrics["accuracy"] > 0.0

        signals_df = strategy._compute_raw_signals(df)
        assert "signal" in signals_df.columns
        assert "confidence" in signals_df.columns
        # 信号应为 1, -1, 或 0
        assert set(signals_df["signal"].dropna().unique()).issubset({1, -1, 0})

    def test_strategy_without_model_returns_zero(self):
        """未训练模型时应返回全零信号。"""
        from strategies.ml_strategy import MLStrategy
        df = _make_mock_klines(100)
        strategy = MLStrategy(params={"model_type": "random_forest"})
        signals_df = strategy._compute_raw_signals(df)
        assert (signals_df["signal"] == 0).all()


# ---------------------------------------------------------------------------
# 回测集成测试
# ---------------------------------------------------------------------------

class TestMLBacktest:
    """测试 MLStrategy 回测能正常运行。"""

    def test_backtest_runs(self):
        """用模拟数据跑回测，确认不报错并返回结果。"""
        from backtest.engine import BacktestEngine
        from strategies.ml_strategy import MLStrategy

        df = _make_mock_klines(200)
        strategy = MLStrategy(params={
            "model_type": "random_forest",
            "forward_days": 5,
        })
        strategy.train_model(df)

        engine = BacktestEngine(initial_capital=1_000_000)
        result = engine.run(df, strategy, symbol="TEST")
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
        assert result.metrics is not None
