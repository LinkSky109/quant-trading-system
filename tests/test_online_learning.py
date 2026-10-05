"""Jev 在线学习模块测试（REQ-P3-05）。

覆盖：
  - 滑动窗口缓冲（容量、淡出、指数衰减权重）
  - 概念漂移检测（基线、连续越界、再训练信号、基线重置）
  - 在线线性模型（增量训练降 loss、维度校验）
  - OnlineLearningPipeline（增量更新统计、漂移触发、acknowledge_retrain）
  - JevDecisionEngine.update_model() 集成
"""
from __future__ import annotations

import numpy as np
import pytest

from jev.online_learning import (
    ConceptDriftDetector,
    OnlineLearningPipeline,
    OnlineLinearModel,
    SlidingWindowBuffer,
    attach_online_learning,
    update_model,
)


# ---------------------------------------------------------------------------
# SlidingWindowBuffer
# ---------------------------------------------------------------------------


class TestSlidingWindowBuffer:
    def test_push_and_arrays(self):
        buf = SlidingWindowBuffer(max_size=5)
        buf.push_batch(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, -1.0]))
        X, y = buf.as_arrays()
        assert X.shape == (2, 2)
        assert np.allclose(y, [1.0, -1.0])

    def test_capacity_fifo(self):
        buf = SlidingWindowBuffer(max_size=3)
        buf.push_batch(np.arange(6, dtype=float).reshape(3, 2), np.zeros(3))
        buf.push_batch(np.array([[100.0, 200.0]]), np.array([9.0]))
        X, y = buf.as_arrays()
        assert len(buf) == 3
        assert np.allclose(X[:, 0], [2.0, 4.0, 100.0])  # 旧样本被挤出
        assert y[-1] == 9.0

    def test_decay_weights_recency(self):
        buf = SlidingWindowBuffer(max_size=4)
        buf.push_batch(np.zeros((4, 1)), np.zeros(4))
        w = buf.decay_weights(half_life=2)
        assert w[-1] > w[0]  # 最新样本权重最大
        # half_life=2：相邻样本权重比 = 2^(1/2)
        assert np.isclose(w[-1] / w[-2], 2 ** 0.5)
        assert np.isclose(w.sum(), w.sum())  # 无 NaN

    def test_decay_half_life_none_zero(self):
        buf = SlidingWindowBuffer(max_size=3)
        buf.push_batch(np.zeros((3, 1)), np.zeros(3))
        w = buf.decay_weights(half_life=10**9)
        assert np.allclose(w, w[0])  # 超长半衰期近似均匀权重


# ---------------------------------------------------------------------------
# ConceptDriftDetector
# ---------------------------------------------------------------------------


class TestConceptDriftDetector:
    def test_no_drift_below_baseline(self):
        d = ConceptDriftDetector(baseline_loss=1.0, threshold=0.2, patience=3)
        for _ in range(10):
            st = d.update(0.5)
        assert st["drift_detected"] is False
        assert st["consecutive_breaches"] == 0

    def test_drift_after_patience(self):
        d = ConceptDriftDetector(baseline_loss=1.0, threshold=0.2, window=20, patience=3)
        st = None
        for _ in range(5):
            st = d.update(2.0)
        assert st["drift_detected"] is True
        assert st["retrain_recommended"] is True

    def test_breach_reset_by_good_loss(self):
        d = ConceptDriftDetector(baseline_loss=1.0, threshold=0.2, patience=3)
        d.update(2.0)
        d.update(2.0)
        # 滚动均值含历史：连续多个好样本后均值回落到阈值内，计数清零
        st = d.update(0.5)
        st = d.update(0.5)
        st = d.update(0.5)
        assert st["consecutive_breaches"] == 0
        assert st["breach"] is False

    def test_invalid_baseline(self):
        with pytest.raises(ValueError):
            ConceptDriftDetector(baseline_loss=0.0)

    def test_reset_baseline(self):
        d = ConceptDriftDetector(baseline_loss=1.0, patience=3)
        d.update(5.0)
        d.update(5.0)
        d.reset_baseline(5.0)
        st = d.update(5.0)
        assert st["baseline_loss"] == 5.0
        assert st["drift_detected"] is False


# ---------------------------------------------------------------------------
# OnlineLinearModel：增量训练
# ---------------------------------------------------------------------------


class TestOnlineLinearModel:
    def test_loss_decreases_incrementally(self):
        rng = np.random.RandomState(7)
        X = rng.normal(0, 1, size=(400, 4))
        w_true = np.array([1.5, -2.0, 0.5, 0.0])
        y = X @ w_true + 0.05 * rng.normal(0, 1, 400)
        model = OnlineLinearModel(n_features=4, lr=0.05)
        first = model.partial_fit(X[:20], y[:20])
        for i in range(20, 400, 20):
            last = model.partial_fit(X[i:i + 20], y[i:i + 20])
        assert last < first * 0.5  # 增量训练显著降低 loss
        pred = model.predict(X[:10])
        assert np.corrcoef(pred, y[:10])[0, 1] > 0.9

    def test_dim_mismatch(self):
        model = OnlineLinearModel(n_features=3)
        with pytest.raises(ValueError):
            model.partial_fit(np.zeros((2, 4)), np.zeros(2))

    def test_no_full_retrain_flag(self):
        """验证是增量接口：partial_fit 后 n_updates 递增，模型参数连续演化。"""
        model = OnlineLinearModel(n_features=2, lr=0.01)
        w_before = model.w.copy()
        model.partial_fit(np.array([[1.0, 1.0]]), np.array([1.0]))
        assert model.n_updates == 1
        assert not np.allclose(model.w, w_before)
        model.partial_fit(np.array([[1.0, 1.0]]), np.array([1.0]))
        assert model.n_updates == 2  # 增量而非全量重训


# ---------------------------------------------------------------------------
# OnlineLearningPipeline
# ---------------------------------------------------------------------------


class TestOnlineLearningPipeline:
    def _batch(self, n, scale=1.0, offset=0.0, seed=0):
        rng = np.random.RandomState(seed)
        X = rng.normal(0, 1, size=(n, 3)) * scale + offset
        y = X @ np.array([1.0, -1.0, 0.5]) + 0.01 * rng.normal(0, 1, n)
        return X, y

    def test_update_returns_monitoring_fields(self):
        pipe = OnlineLearningPipeline(n_features=3, window_size=100)
        X, y = self._batch(30, seed=1)
        s = pipe.update(X, y)
        for key in ("batch_loss", "pre_window_loss", "post_window_loss",
                    "loss_delta", "window_samples", "model_updates", "drift"):
            assert key in s
        assert s["window_samples"] == 30

    def test_sliding_window_fade_out(self):
        pipe = OnlineLearningPipeline(n_features=3, window_size=50)
        X, y = self._batch(80, seed=2)
        pipe.update(X, y)
        assert len(pipe.buffer) == 50  # 旧数据淡出，窗口封顶

    def test_baseline_and_drift_trigger(self):
        pipe = OnlineLearningPipeline(n_features=3)
        X, y = self._batch(20, seed=3)
        pipe.update(X, y)  # 建立基线
        assert pipe.drift_detector is not None
        # 注入持续恶化的 loss 触发漂移
        for _ in range(5):
            Xb, yb = self._batch(10, seed=99)
            Xb = Xb * 50  # 让 loss 大幅超过基线
            pipe.update(Xb, yb)
        st = pipe.monitor_history[-1]["drift"]
        assert st["drift_detected"] is True

    def test_acknowledge_retrain(self):
        pipe = OnlineLearningPipeline(n_features=3)
        X, y = self._batch(20, seed=4)
        pipe.update(X, y)
        pipe.acknowledge_retrain(0.5)
        assert pipe.drift_detector.baseline_loss == 0.5
        assert pipe.drift_detector._consecutive_breaches == 0

    def test_get_stats_and_accuracy_proxy(self):
        pipe = OnlineLearningPipeline(n_features=3)
        X, y = self._batch(40, seed=5)
        pipe.update(X, y)
        stats = pipe.get_stats()
        assert stats["n_updates"] == 1
        assert len(stats["loss_history"]) == 1
        acc = pipe.accuracy_proxy()
        assert acc is not None and 0.0 <= acc <= 1.0

    def test_empty_update_raises(self):
        pipe = OnlineLearningPipeline(n_features=3)
        with pytest.raises(ValueError):
            pipe.update(np.zeros((0, 3)), np.zeros(0))

    def test_dim_mismatch_raises(self):
        pipe = OnlineLearningPipeline(n_features=3)
        with pytest.raises(ValueError):
            pipe.update(np.zeros((5, 7)), np.zeros(5))


# ---------------------------------------------------------------------------
# JevDecisionEngine 集成
# ---------------------------------------------------------------------------


class TestJevEngineIntegration:
    def _records(self, n, seed=0):
        rng = np.random.RandomState(seed)
        return [
            {
                "features": {
                    "momentum": float(rng.normal()),
                    "volatility": abs(float(rng.normal())),
                    "volume_ratio": float(rng.normal()),
                },
                "realized_return": float(rng.normal(0, 0.01)),
            }
            for _ in range(n)
        ]

    def test_update_model_attaches_and_updates(self):
        from jev.jev_engine import JevDecisionEngine

        engine = JevDecisionEngine()
        assert engine.get_online_learning_stats() is None  # 未挂载
        s = engine.update_model(self._records(30, seed=1))
        assert "batch_loss" in s
        assert s["window_samples"] == 30
        stats = engine.get_online_learning_stats()
        assert stats["n_updates"] == 1

    def test_update_model_incremental_second_call(self):
        from jev.jev_engine import JevDecisionEngine

        engine = JevDecisionEngine()
        engine.update_model(self._records(20, seed=2))
        s2 = engine.update_model(self._records(20, seed=3))
        assert s2["model_updates"] > 0
        stats = engine.get_online_learning_stats()
        assert stats["n_updates"] == 2

    def test_update_model_explicit_feature_keys(self):
        from jev.jev_engine import JevDecisionEngine

        engine = JevDecisionEngine()
        s = engine.update_model(
            self._records(15, seed=4), feature_keys=["momentum", "volatility"]
        )
        assert s["feature_keys"] == ["momentum", "volatility"]

    def test_update_model_empty_raises(self):
        from jev.jev_engine import JevDecisionEngine

        engine = JevDecisionEngine()
        with pytest.raises(ValueError):
            engine.update_model([])

    def test_attach_online_learning_idempotent(self):
        from jev.jev_engine import JevDecisionEngine

        engine = JevDecisionEngine()
        p1 = attach_online_learning(engine, n_features=3)
        p2 = attach_online_learning(engine, n_features=3)
        assert p1 is p2
