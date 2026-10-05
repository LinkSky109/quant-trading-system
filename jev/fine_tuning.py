"""Jev 模型微调 Pipeline。

从 TrainingDataExporter 产出的训练数据出发，训练本地 surrogate 模型，
用于在 Jev 服务不可用时提供降级决策，或作为 Jev 的本地缓存层。

支持的模型：LightGBM、XGBoost、sklearn GradientBoosting。
所有 boosting 库的 import 均延迟到函数内部，确保本模块在无库环境下也能正常 import。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from jev.training_data import FEATURE_NAMES

logger = logging.getLogger(__name__)

#: 模型持久化目录
FINE_TUNED_DIR = Path(__file__).resolve().parent.parent / "models" / "fine_tuned"
FINE_TUNED_DIR.mkdir(parents=True, exist_ok=True)

#: 支持的微调模型类型
_SUPPORTED_MODELS = ["lightgbm", "xgboost", "gradient_boosting"]


@dataclass
class FineTuningConfig:
    """微调配置。"""
    model_type: str = "lightgbm"
    test_size: float = 0.2
    random_state: int = 42
    hyperparameter_search: bool = False
    n_trials: int = 20
    model_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_type not in _SUPPORTED_MODELS:
            raise ValueError(
                f"不支持的模型类型: {self.model_type}, "
                f"可选: {_SUPPORTED_MODELS}"
            )


@dataclass
class FineTuningResult:
    """微调结果。"""
    model_type: str
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    best_params: Dict[str, Any]
    feature_importance: Dict[str, float]
    model_path: str
    train_samples: int
    val_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "best_params": self.best_params,
            "feature_importance": self.feature_importance,
            "model_path": self.model_path,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
        }


def _try_import_lightgbm():
    try:
        import lightgbm as lgb  # noqa: F401
        return lgb
    except ImportError as e:
        raise ImportError("lightgbm 未安装，请运行: pip install lightgbm") from e


def _try_import_xgboost():
    try:
        import xgboost as xgb  # noqa: F401
        return xgb
    except ImportError as e:
        raise ImportError("xgboost 未安装，请运行: pip install xgboost") from e


def _try_import_sklearn_metrics():
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        return accuracy_score, f1_score, precision_score, recall_score
    except ImportError as e:
        raise ImportError("scikit-learn 未安装") from e


class JevFineTuner:
    """Jev 模型微调器。

    从 JSONL/CSV 训练数据加载样本，训练 surrogate 分类器，
    输出可持久化的模型和评估指标。
    """

    def __init__(self, config: Optional[FineTuningConfig] = None) -> None:
        self.config = config or FineTuningConfig()
        self.model: Any = None
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    @staticmethod
    def load_jsonl(path: str) -> pd.DataFrame:
        """从 JSONL 加载训练数据。"""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return JevFineTuner._records_to_df(records)

    @staticmethod
    def load_csv(path: str) -> pd.DataFrame:
        """从 CSV 加载训练数据。"""
        return pd.read_csv(path)

    @staticmethod
    def _records_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
        """将 JSONL 记录列表转为 DataFrame，展开 features。"""
        rows = []
        for rec in records:
            row: Dict[str, Any] = {}
            # 扁平化 features
            features = rec.get("features", {})
            if isinstance(features, list):
                # [{feature, value}, ...] 格式
                for fv in features:
                    row[fv["feature"]] = fv["value"]
            elif isinstance(features, dict):
                row.update(features)
            else:
                raise ValueError(f"不支持的 features 格式: {type(features)}")
            row["label"] = rec.get("label", rec.get("optimal_action", "hold"))
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> FineTuningResult:
        """训练 surrogate 模型。

        Args:
            df: 含特征列和 label 列的 DataFrame。
                label 为 buy/sell/hold 字符串，自动编码为 0/1/2。

        Returns:
            FineTuningResult 含评估指标和模型路径。
        """
        X, y, label_map = self._prepare_data(df)
        self.feature_names = list(X.columns)

        train_size = int(len(X) * (1 - self.config.test_size))
        X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]

        logger.info(
            "开始微调 Jev surrogate 模型: %s, 训练样本=%d, 验证样本=%d",
            self.config.model_type, len(X_train), len(X_val),
        )

        self.model = self._fit_model(X_train, y_train)

        # 评估
        accuracy_score, f1_score, precision_score, recall_score = _try_import_sklearn_metrics()
        y_pred_raw = self.model.predict(X_val)
        # lgb.train 返回的 Booster 在 multiclass 下输出概率矩阵，需取 argmax
        if y_pred_raw.ndim > 1:
            y_pred = np.argmax(y_pred_raw, axis=1)
        else:
            y_pred = y_pred_raw

        result = FineTuningResult(
            model_type=self.config.model_type,
            accuracy=float(accuracy_score(y_val, y_pred)),
            f1_score=float(f1_score(y_val, y_pred, average="weighted", zero_division=0)),
            precision=float(precision_score(y_val, y_pred, average="weighted", zero_division=0)),
            recall=float(recall_score(y_val, y_pred, average="weighted", zero_division=0)),
            best_params=self._get_model_params(),
            feature_importance=self._feature_importance(),
            model_path="",
            train_samples=len(X_train),
            val_samples=len(X_val),
        )

        # 持久化
        model_path = self._save_model(label_map)
        result.model_path = model_path
        logger.info("微调完成，模型已保存至: %s", model_path)
        return result

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
        """准备 X, y 和标签映射。"""
        # 自动识别特征列（排除 label 和 metadata）
        exclude = {"label", "optimal_action", "symbol", "date", "timestamp"}
        feature_cols = [c for c in df.columns if c not in exclude]
        if not feature_cols:
            feature_cols = FEATURE_NAMES

        X = df[feature_cols].copy()
        # 处理 NaN
        X = X.fillna(0.0)

        label_col = "label" if "label" in df.columns else "optimal_action"
        raw_labels = df[label_col].astype(str)
        unique_labels = sorted(raw_labels.unique())
        label_map = {name: i for i, name in enumerate(unique_labels)}
        y = raw_labels.map(label_map)

        return X, y, label_map

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """根据配置训练具体模型。"""
        model_type = self.config.model_type
        params = dict(self.config.model_params)

        if model_type == "lightgbm":
            lgb = _try_import_lightgbm()
            default_params = {
                "objective": "multiclass",
                "num_class": len(y.unique()),
                "metric": "multi_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "random_state": self.config.random_state,
            }
            default_params.update(params)
            train_data = lgb.Dataset(X, label=y)
            model = lgb.train(
                default_params,
                train_data,
                num_boost_round=100,
            )
            return model

        if model_type == "xgboost":
            xgb = _try_import_xgboost()
            default_params = {
                "objective": "multi:softprob",
                "num_class": len(y.unique()),
                "eval_metric": "mlogloss",
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.config.random_state,
            }
            default_params.update(params)
            model = xgb.XGBClassifier(**default_params)
            model.fit(X, y)
            return model

        if model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=self.config.random_state,
                **params,
            )
            model.fit(X, y)
            return model

        raise ValueError(f"不支持的模型类型: {model_type}")

    def _get_model_params(self) -> Dict[str, Any]:
        """提取模型参数。"""
        if self.model is None:
            return {}
        if hasattr(self.model, "get_params"):
            return self.model.get_params()
        return {}

    def _feature_importance(self) -> Dict[str, float]:
        """提取特征重要性。"""
        if self.model is None or not self.feature_names:
            return {}
        importances = None
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "feature_importance"):
            # lightgbm.Booster
            importances = self.model.feature_importance(importance_type="gain")
        if importances is not None:
            total = float(np.sum(importances))
            if total > 0:
                return {
                    name: float(imp) / total
                    for name, imp in zip(self.feature_names, importances)
                }
        return {}

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _save_model(self, label_map: Dict[str, int]) -> str:
        """保存模型和元数据。"""
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_dir = FINE_TUNED_DIR / f"{self.config.model_type}_{ts}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型
        model_path = model_dir / "model.pkl"
        import joblib
        joblib.dump(self.model, model_path)

        # 保存元数据
        meta = {
            "model_type": self.config.model_type,
            "feature_names": self.feature_names,
            "label_map": label_map,
            "config": {
                "model_type": self.config.model_type,
                "test_size": self.config.test_size,
                "random_state": self.config.random_state,
            },
        }
        meta_path = model_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return str(model_dir)

    @classmethod
    def load(cls, model_dir: str) -> "JevFineTuner":
        """加载已保存的微调器。"""
        model_dir = Path(model_dir)
        meta_path = model_dir / "meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        import joblib
        model = joblib.load(model_dir / "model.pkl")

        config = FineTuningConfig(**meta["config"])
        tuner = cls(config)
        tuner.model = model
        tuner.feature_names = meta["feature_names"]
        return tuner

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def predict(self, features: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """单条推理。

        Args:
            features: 特征名 -> 值的字典。

        Returns:
            (预测标签, 各类别概率字典)
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 train()")

        X = pd.DataFrame([{name: features.get(name, 0.0) for name in self.feature_names}])

        if self.config.model_type == "lightgbm":
            probs = self.model.predict(X)
            if probs.ndim == 2:
                probs = probs[0]
        elif hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
        else:
            label = int(self.model.predict(X)[0])
            return str(label), {str(label): 1.0}

        # 加载 label_map 反查
        model_dir = FINE_TUNED_DIR
        # 简化：假设 caller 知道 label 含义
        prob_dict = {str(i): float(p) for i, p in enumerate(probs)}
        best_label = str(int(np.argmax(probs)))
        return best_label, prob_dict
