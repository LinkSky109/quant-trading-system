"""Jev 在线学习模块（REQ-P3-05）。

组成：
  - ``SlidingWindowBuffer``：滑动窗口样本缓冲，旧样本按指数衰减淡出
  - ``ConceptDriftDetector``：概念漂移检测（滚动误差相对基线的偏移监控）
  - ``OnlineLinearModel``：可增量更新的线性模型（SGD 逐批 partial_fit，
    非全量重训）
  - ``OnlineLearningPipeline``：串起 增量更新 -> 性能监控 -> 漂移检测 的
    在线学习 pipeline，并暴露 ``update_model`` 接口供 ``JevDecisionEngine``
    集成调用。

所有计算基于 numpy，测试使用 mock 数据，不依赖外部 API。
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 滑动窗口缓冲
# --------------------------------------------------------------------------- #


class SlidingWindowBuffer:
    """滑动窗口样本缓冲（最近 N 条参与监控/评估，旧样本自然淡出）。"""

    def __init__(self, max_size: int = 500) -> None:
        if max_size <= 0:
            raise ValueError("max_size 必须 > 0")
        self.max_size = max_size
        self._X: Deque[np.ndarray] = deque(maxlen=max_size)
        self._y: Deque[float] = deque(maxlen=max_size)

    def push(self, x: np.ndarray, y: float) -> None:
        """写入一条样本。"""
        self._X.append(np.asarray(x, dtype=float))
        self._y.append(float(y))

    def push_batch(self, X: np.ndarray, y: np.ndarray) -> None:
        """批量写入。"""
        for xi, yi in zip(np.asarray(X, dtype=float), np.asarray(y, dtype=float)):
            self.push(xi, yi)

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """导出窗口内样本。"""
        if not self._X:
            return np.zeros((0, 0)), np.zeros(0)
        return np.array(self._X), np.array(self._y)

    def decay_weights(self, half_life: int = 100) -> np.ndarray:
        """旧样本淡出权重：按样本序龄指数衰减（half_life 为半衰期）。

        最新样本权重 1.0，越旧权重越低；仅用于监控指标加权。
        """
        n = len(self._y)
        if n == 0:
            return np.zeros(0)
        # deque  oldest-first：最新样本权重 1.0，越旧权重越低
        ages = np.arange(n, dtype=float)[::-1]
        return np.power(0.5, ages / float(half_life))

    def __len__(self) -> int:
        return len(self._y)


# --------------------------------------------------------------------------- #
# 概念漂移检测
# --------------------------------------------------------------------------- #


class ConceptDriftDetector:
    """概念漂移检测器（滚动误差相对基线的偏移监控）。

    规则：
      - 维护最近 ``window`` 条 loss 的滚动均值；
      - 滚动均值连续 ``patience`` 次超过 ``baseline_loss * (1 + threshold)``
        判定概念漂移，触发再训练信号。
    """

    def __init__(
        self,
        baseline_loss: float,
        threshold: float = 0.2,
        window: int = 20,
        patience: int = 3,
    ) -> None:
        if baseline_loss <= 0:
            raise ValueError("baseline_loss 必须 > 0")
        self.baseline_loss = baseline_loss
        self.threshold = threshold
        self.window = window
        self.patience = patience
        self._losses: Deque[float] = deque(maxlen=window)
        self._consecutive_breaches = 0

    def update(self, loss: float) -> Dict[str, Any]:
        """写入一次评估 loss，返回检测状态。"""
        self._losses.append(float(loss))
        rolling = float(np.mean(self._losses)) if self._losses else float(loss)
        breach = rolling > self.baseline_loss * (1.0 + self.threshold)
        self._consecutive_breaches = self._consecutive_breaches + 1 if breach else 0
        drift = self._consecutive_breaches >= self.patience
        return {
            "rolling_loss": rolling,
            "baseline_loss": self.baseline_loss,
            "breach": breach,
            "consecutive_breaches": self._consecutive_breaches,
            "drift_detected": drift,
            "retrain_recommended": drift,
        }

    def reset_baseline(self, baseline_loss: float) -> None:
        """再训练完成后重置基线并清空计数。"""
        if baseline_loss <= 0:
            raise ValueError("baseline_loss 必须 > 0")
        self.baseline_loss = baseline_loss
        self._losses.clear()
        self._consecutive_breaches = 0


# --------------------------------------------------------------------------- #
# 可增量更新的线性模型
# --------------------------------------------------------------------------- #


class OnlineLinearModel:
    """SGD 增量更新的线性回归模型（增量训练，非全量重训）。

    每次 ``partial_fit`` 只用新到的批次做一步（或若干步）梯度下降，
    不保留历史数据、不做全量重拟合。
    """

    def __init__(self, n_features: int, lr: float = 0.01, l2: float = 1e-4) -> None:
        if n_features <= 0:
            raise ValueError("n_features 必须 > 0")
        self.n_features = n_features
        self.lr = lr
        self.l2 = l2
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.n_updates = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return X @ self.w + self.b

    def partial_fit(self, X: np.ndarray, y: np.ndarray, steps: int = 1) -> float:
        """用新批次数据做 steps 步 SGD，返回该批次的 MSE loss。

        若传入特征维度与模型不一致，抛 ValueError（调用方应做特征对齐）。
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"特征维度不匹配: 期望 {self.n_features}, 实际 {X.shape[1]}"
            )
        if len(X) != len(y):
            raise ValueError("X 与 y 数量不一致")
        loss = 0.0
        for _ in range(max(1, steps)):
            pred = self.predict(X)
            err = pred - y
            loss = float(np.mean(err ** 2))
            grad_w = X.T @ err / max(len(y), 1) + self.l2 * self.w
            grad_b = float(np.mean(err))
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            self.n_updates += 1
        return loss


# --------------------------------------------------------------------------- #
# 在线学习 pipeline
# --------------------------------------------------------------------------- #


class OnlineLearningPipeline:
    """Jev 在线学习 pipeline：增量更新 + 性能监控 + 概念漂移检测。

    用法::

        pipe = OnlineLearningPipeline(n_features=8)
        stats = pipe.update(X_new, y_new)     # 新数据到来时增量更新
        stats["drift"]["drift_detected"]      # 是否需要再训练
    """

    def __init__(
        self,
        n_features: int,
        lr: float = 0.01,
        window_size: int = 500,
        drift_threshold: float = 0.2,
        drift_patience: int = 3,
        half_life: int = 100,
    ) -> None:
        self.model = OnlineLinearModel(n_features, lr=lr)
        self.buffer = SlidingWindowBuffer(window_size)
        self.monitor_history: List[Dict[str, Any]] = []
        self._baseline_loss: Optional[float] = None
        self.drift_detector: Optional[ConceptDriftDetector] = None
        self.half_life = half_life
        self.created_at = time.time()

    # ---------------- 内部 ----------------
    def _window_loss(self) -> Optional[float]:
        """窗口内加权 MSE（旧样本淡出权重）。"""
        X, y = self.buffer.as_arrays()
        if len(y) < 5:
            return None
        pred = self.model.predict(X)
        w = self.buffer.decay_weights(self.half_life)
        w = w / w.sum()
        return float(np.sum(w * (pred - y) ** 2))

    # ---------------- 对外 ----------------
    def update(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """新数据到来：增量训练（仅本批）+ 性能监控 + 漂移检测。

        Returns:
            {"batch_loss", "window_loss", "model_updates", "drift", ...}
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if len(X) == 0:
            raise ValueError("更新样本为空")
        if X.shape[1] != self.model.n_features:
            raise ValueError(
                f"特征维度不匹配: 期望 {self.model.n_features}, 实际 {X.shape[1]}"
            )

        pre_loss = self._window_loss()
        batch_loss = self.model.partial_fit(X, y)
        self.buffer.push_batch(X, y)
        post_loss = self._window_loss()

        # 基线：首次更新时用本批 loss 作为漂移检测基线
        if self._baseline_loss is None and batch_loss > 0:
            self._baseline_loss = batch_loss
        if self._baseline_loss is not None:
            if self.drift_detector is None:
                self.drift_detector = ConceptDriftDetector(
                    self._baseline_loss,
                    threshold=self.drift_threshold_default(),
                    patience=self.drift_patience_default(),
                )
            eval_loss = post_loss if post_loss is not None else batch_loss
            drift_status = self.drift_detector.update(eval_loss)
        else:
            drift_status = {"drift_detected": False, "rolling_loss": batch_loss}

        stats = {
            "timestamp": time.time(),
            "batch_size": len(y),
            "batch_loss": batch_loss,
            "pre_window_loss": pre_loss,
            "post_window_loss": post_loss,
            "loss_delta": (post_loss - pre_loss)
            if (pre_loss is not None and post_loss is not None) else None,
            "model_updates": self.model.n_updates,
            "window_samples": len(self.buffer),
            "drift": drift_status,
        }
        self.monitor_history.append(stats)
        return stats

    # 供子类/配置覆盖的默认漂移参数
    def drift_threshold_default(self) -> float:
        return 0.2

    def drift_patience_default(self) -> int:
        return 3

    def acknowledge_retrain(self, new_baseline_loss: float) -> None:
        """再训练完成后调用：重置漂移基线。"""
        if self.drift_detector is None:
            self.drift_detector = ConceptDriftDetector(
                new_baseline_loss,
                threshold=self.drift_threshold_default(),
                patience=self.drift_patience_default(),
            )
        else:
            self.drift_detector.reset_baseline(new_baseline_loss)

    def get_stats(self) -> Dict[str, Any]:
        """性能监控摘要：每次更新的 loss / 准确率代理变化。"""
        losses = [s["batch_loss"] for s in self.monitor_history]
        return {
            "n_updates": len(self.monitor_history),
            "model_updates": self.model.n_updates,
            "window_samples": len(self.buffer),
            "first_batch_loss": losses[0] if losses else None,
            "latest_batch_loss": losses[-1] if losses else None,
            "mean_batch_loss": float(np.mean(losses)) if losses else None,
            "loss_history": losses,
            "baseline_loss": self._baseline_loss,
            "drift_detected": (
                self.drift_detector._consecutive_breaches >= self.drift_detector.patience
                if self.drift_detector else False
            ),
        }

    def accuracy_proxy(self) -> Optional[float]:
        """准确率代理：窗口内预测方向与真实方向的一致率（y 视作收益符号）。"""
        X, y = self.buffer.as_arrays()
        if len(y) < 5:
            return None
        pred = self.model.predict(X)
        return float(np.mean(np.sign(pred) == np.sign(y)))


# --------------------------------------------------------------------------- #
# JevDecisionEngine 集成接口（update_model）
# --------------------------------------------------------------------------- #

# 存放在引擎实例上的 pipeline 属性名
_PIPELINE_ATTR = "_online_pipeline"


def attach_online_learning(
    engine: Any,
    n_features: int = 8,
    **pipeline_kwargs: Any,
) -> OnlineLearningPipeline:
    """给 ``JevDecisionEngine``（或任意对象）挂载在线学习 pipeline。"""
    if not hasattr(engine, _PIPELINE_ATTR):
        setattr(engine, _PIPELINE_ATTR, OnlineLearningPipeline(n_features, **pipeline_kwargs))
    return getattr(engine, _PIPELINE_ATTR)


def update_model(
    engine: Any,
    feedback_records: List[Dict[str, Any]],
    feature_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """``JevDecisionEngine.update_model`` 的实现体。

    Args:
        engine: JevDecisionEngine 实例。
        feedback_records: 决策反馈记录列表，每条形如
            ``{"features": {...}, "realized_return": 0.012}``；
            label 默认取 realized_return 的符号对应的数值收益。
        feature_keys: 特征提取顺序；None 则按第一条记录的 key 排序。

    Returns:
        在线学习更新统计（见 ``OnlineLearningPipeline.update``）。
    """
    if not feedback_records:
        raise ValueError("feedback_records 为空，无需更新")
    feats = [r["features"] for r in feedback_records]
    if feature_keys is None:
        feature_keys = sorted(feats[0].keys())
    X = np.array([[float(f[k]) for k in feature_keys] for f in feats])
    y = np.array([float(r.get("realized_return", 0.0)) for r in feedback_records])

    pipe = attach_online_learning(engine, n_features=X.shape[1])
    stats = pipe.update(X, y)
    stats["feature_keys"] = feature_keys
    return stats
