"""机器学习进阶策略（XGBoost / LightGBM 集成）。

在 MLPredictor 基础上扩展 XGBoost 和 LightGBM 模型支持，
并提供 Stacking 集成方式综合多个模型的预测结果。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.predictor import MLPredictor
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

#: 延迟导入的模型注册表
_ADVANCED_MODEL_REGISTRY: Dict[str, str] = {
    "xgboost": "XGBClassifier",
    "lightgbm": "LGBMClassifier",
}


def _try_import_xgboost():
    try:
        import xgboost as xgb  # noqa: F401
        return xgb
    except ImportError as e:
        raise ImportError("xgboost 未安装，请运行: pip install xgboost") from e


def _try_import_lightgbm():
    try:
        import lightgbm as lgb  # noqa: F401
        return lgb
    except ImportError as e:
        raise ImportError("lightgbm 未安装，请运行: pip install lightgbm") from e


class AdvancedMLPredictor(MLPredictor):
    """进阶 ML 预测器，支持 XGBoost / LightGBM。

    继承 MLPredictor 的全部特征工程和标签构建能力，
    仅替换模型训练/预测部分。
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        model_params: Optional[Dict[str, Any]] = None,
        forward_days: int = 5,
        task: str = "classification",
    ):
        super().__init__(
            model_type=model_type,
            model_params=model_params,
            forward_days=forward_days,
            task=task,
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """训练进阶模型。"""
        self.feature_names = list(X.columns)
        self.train_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        model_type = self.model_type.lower()
        if model_type == "xgboost":
            xgb = _try_import_xgboost()
            params = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "use_label_encoder": False,
                "eval_metric": "logloss",
            }
            params.update(self.model_params)
            self.model = xgb.XGBClassifier(**params)
        elif model_type == "lightgbm":
            lgb = _try_import_lightgbm()
            params = {
                "n_estimators": 100,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbosity": -1,
            }
            params.update(self.model_params)
            self.model = lgb.LGBMClassifier(**params)
        else:
            # 回退到父类支持的模型
            return super().train(X, y)

        self.model.fit(X, y)

        from sklearn.metrics import accuracy_score
        train_pred = self.model.predict(X)
        acc = accuracy_score(y, train_pred)

        return {
            "model_type": self.model_type,
            "train_samples": len(X),
            "feature_count": len(X.columns),
            "train_accuracy": acc,
        }


class StackingEnsemblePredictor:
    """Stacking 集成预测器。

    训练多个基模型（RF + XGB + LGBM），用逻辑回归作为元模型进行 stacking。
    """

    def __init__(
        self,
        forward_days: int = 5,
        meta_model_type: str = "logistic_regression",
    ):
        self.forward_days = forward_days
        self.meta_model_type = meta_model_type
        self.base_predictors: List[AdvancedMLPredictor] = []
        self.meta_model: Any = None
        self.feature_names: List[str] = []
        self.train_date: Optional[str] = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """训练 stacking 集成。"""
        from sklearn.model_selection import KFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier

        self.feature_names = list(X.columns)
        self.train_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        # 基模型配置
        base_configs = [
            ("random_forest", {"n_estimators": 50, "max_depth": 5, "random_state": 42}),
            ("xgboost", {"n_estimators": 50, "max_depth": 4, "random_state": 42}),
            ("lightgbm", {"n_estimators": 50, "num_leaves": 20, "random_state": 42}),
        ]

        # 用 KFold 生成元特征（防止过拟合）
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X), len(base_configs)))

        for idx, (model_type, params) in enumerate(base_configs):
            predictor = AdvancedMLPredictor(model_type=model_type, model_params=params)
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                predictor.train(X_train, y_train)
                probs = predictor.predict_proba(X_val)
                meta_features[val_idx, idx] = probs

            # 最终在全量数据上重训练
            predictor.train(X, y)
            self.base_predictors.append(predictor)

        # 训练元模型
        self.meta_model = LogisticRegression(max_iter=1000, random_state=42)
        self.meta_model.fit(meta_features, y)

        from sklearn.metrics import accuracy_score
        meta_pred = self.meta_model.predict(meta_features)
        acc = accuracy_score(y, meta_pred)

        return {
            "ensemble_type": "stacking",
            "base_models": [c[0] for c in base_configs],
            "meta_model": self.meta_model_type,
            "train_samples": len(X),
            "train_accuracy": acc,
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测上涨概率。"""
        if self.meta_model is None:
            raise RuntimeError("模型未训练")

        meta_features = np.zeros((len(X), len(self.base_predictors)))
        for idx, predictor in enumerate(self.base_predictors):
            meta_features[:, idx] = predictor.predict_proba(X)

        probs = self.meta_model.predict_proba(meta_features)
        return probs[:, 1]  # 上涨概率


class MLAdvancedStrategy(BaseStrategy):
    """机器学习进阶策略（XGBoost / LightGBM / Stacking）。

    与 MLStrategy 接口一致，但底层使用 AdvancedMLPredictor 或 StackingEnsemblePredictor。
    """

    name = "ml_advanced"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.model_type: str = str(self.params.get("model_type", "xgboost"))
        self.model_params: Dict[str, Any] = dict(self.params.get("model_params", {}))
        self.forward_days: int = int(self.params.get("forward_days", 5))
        self.buy_threshold: float = float(self.params.get("buy_threshold", 0.6))
        self.sell_threshold: float = float(self.params.get("sell_threshold", 0.4))
        self.use_stacking: bool = bool(self.params.get("use_stacking", False))
        self.model_path: Optional[str] = self.params.get("model_path")

        if self.use_stacking:
            self.predictor = StackingEnsemblePredictor(
                forward_days=self.forward_days,
            )
        else:
            self.predictor = AdvancedMLPredictor(
                model_type=self.model_type,
                model_params=self.model_params,
                forward_days=self.forward_days,
            )

        if self.model_path:
            self.load_model(self.model_path)

    def train_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """训练模型。"""
        features = MLPredictor.build_features(df)
        labels = MLPredictor.build_labels(df, forward_days=self.forward_days)

        combined = features.copy()
        combined["label"] = labels
        combined = combined.dropna()

        X = combined[features.columns]
        y = combined["label"]

        logger.info(
            "开始训练 %s 模型: 样本数=%d, 特征数=%d",
            self.model_type, len(X), len(features.columns),
        )
        return self.predictor.train(X, y)

    def load_model(self, model_path: str) -> None:
        """加载已训练模型。"""
        import joblib
        self.predictor = joblib.load(model_path)
        logger.info("MLAdvancedStrategy 已加载模型: %s", model_path)

    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成信号。"""
        out = df.copy()
        out["signal"] = 0
        out["confidence"] = 0.0

        if self.use_stacking:
            predictor = self.predictor
            if not isinstance(predictor, StackingEnsemblePredictor):
                predictor = StackingEnsemblePredictor()
                predictor.meta_model = getattr(self.predictor, "meta_model", None)
                predictor.base_predictors = getattr(self.predictor, "base_predictors", [])
        else:
            predictor = self.predictor
            if not isinstance(predictor, AdvancedMLPredictor):
                predictor = AdvancedMLPredictor()
                predictor.model = getattr(self.predictor, "model", None)
                predictor.feature_names = getattr(self.predictor, "feature_names", [])

        meta_model = getattr(predictor, "meta_model", None)
        base_model = getattr(predictor, "model", None)
        if meta_model is None and base_model is None:
            logger.warning("ML 进阶模型未训练，输出全零信号")
            return out

        # 如果是 Stacking 且 meta_model 为 None，也跳过
        if self.use_stacking and getattr(predictor, "meta_model", None) is None:
            logger.warning("Stacking 模型未训练，输出全零信号")
            return out

        features = MLPredictor.build_features(df)
        valid = features.dropna()
        if valid.empty:
            return out

        X = valid[predictor.feature_names] if predictor.feature_names else valid
        probs = predictor.predict_proba(X)

        prob_series = pd.Series(probs, index=valid.index)
        out["probability"] = prob_series.reindex(out.index)

        buy_mask = out["probability"] > self.buy_threshold
        sell_mask = out["probability"] < self.sell_threshold
        out.loc[buy_mask, "signal"] = 1
        out.loc[sell_mask, "signal"] = -1

        out["confidence"] = (out["probability"] - 0.5).abs() * 2
        out["confidence"] = out["confidence"].fillna(0.0)
        out.loc[out["signal"] == 0, "confidence"] = 0.0

        return out
