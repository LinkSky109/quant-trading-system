"""机器学习价格方向预测器。

封装特征工程、标签构建、模型训练、概率预测与模型持久化。
所有 sklearn 的 import 均延迟到函数内部，确保本模块在无 sklearn 环境下也能正常 import。
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.indicators import bollinger_bands, macd, rsi as rsi_indicator

logger = logging.getLogger(__name__)

#: models/ 目录路径（项目根目录下）
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

#: 支持的模型类型映射（字符串 -> sklearn 类名）
_MODEL_REGISTRY: Dict[str, str] = {
    "random_forest": "RandomForestClassifier",
    "gradient_boosting": "GradientBoostingClassifier",
    "logistic_regression": "LogisticRegression",
    "svm": "SVC",
}


def _try_import_sklearn():
    """尝试导入 sklearn 相关模块，失败抛出 ImportError。

    Raises:
        ImportError: 当 scikit-learn 未安装时。
    """
    try:
        from sklearn.ensemble import (  # noqa: F401
            GradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        from sklearn.svm import SVC  # noqa: F401
        from sklearn.metrics import (  # noqa: F401
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError as e:
        raise ImportError(
            "scikit-learn 未安装，请运行: pip install scikit-learn"
        ) from e


def _get_model_class(model_type: str):
    """根据模型类型字符串获取 sklearn 模型类。

    Args:
        model_type: 模型类型（random_forest / gradient_boosting /
            logistic_regression / svm）。

    Returns:
        sklearn 分类器类。

    Raises:
        ValueError: 不支持的模型类型。
        ImportError: scikit-learn 未安装。
    """
    _try_import_sklearn()
    class_name = _MODEL_REGISTRY.get(model_type)
    if class_name is None:
        raise ValueError(
            f"不支持的模型类型: {model_type}，"
            f"可选: {list(_MODEL_REGISTRY.keys())}"
        )
    import sklearn.ensemble as sk_ensemble
    import sklearn.linear_model as sk_linear
    import sklearn.svm as sk_svm

    mapping = {
        "RandomForestClassifier": sk_ensemble.RandomForestClassifier,
        "GradientBoostingClassifier": sk_ensemble.GradientBoostingClassifier,
        "LogisticRegression": sk_linear.LogisticRegression,
        "SVC": sk_svm.SVC,
    }
    return mapping[class_name]


class MLPredictor:
    """机器学习价格方向预测器。

    从 K 线 DataFrame 构建技术特征，训练分类模型预测未来 N 日涨跌方向，
    并输出上涨概率用于信号生成。

    Attributes:
        model_type: 模型类型字符串。
        model_params: 模型超参数字典。
        forward_days: 标签前瞻天数。
        task: 任务类型（classification / regression）。
        model: 已训练的 sklearn 模型实例（训练后可用）。
        feature_names: 特征列名列表。
        train_date: 模型训练日期字符串。
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        model_params: Optional[Dict[str, Any]] = None,
        forward_days: int = 5,
        task: str = "classification",
    ):
        """初始化预测器。

        Args:
            model_type: 模型类型，默认 random_forest。
            model_params: 模型超参数，None 时使用默认值。
            forward_days: 标签前瞻天数。
            task: classification 或 regression。
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.forward_days = forward_days
        self.task = task
        self.model: Any = None
        self.feature_names: List[str] = []
        self.train_date: Optional[str] = None

    # ------------------------------------------------------------------
    # 特征工程
    # ------------------------------------------------------------------

    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        """从 K 线 DataFrame 构建特征矩阵。

        特征列表（共约 15 个）:
            - ret_1, ret_5, ret_10, ret_20: N 日收益率
            - vol_5, vol_20: N 日收益率标准差（波动率）
            - rsi_14: 14 日 RSI
            - macd_dif, macd_dea, macd_hist: MACD 三要素
            - bollinger_position: 布林带位置 (close-lower)/(upper-lower)
            - volume_change_5: 成交量变化率
            - ma5_deviation, ma20_deviation: 均线偏离度

        Args:
            df: 含 open/high/low/close/volume 列的 K 线 DataFrame，index 为日期。

        Returns:
            含特征列的 DataFrame，index 与输入一致。前若干行因 rolling 窗口为 NaN。
        """
        out = pd.DataFrame(index=df.index)
        close = df["close"]
        volume = df["volume"]

        # --- 收益率 ---
        out["ret_1"] = close.pct_change(1)
        out["ret_5"] = close.pct_change(5)
        out["ret_10"] = close.pct_change(10)
        out["ret_20"] = close.pct_change(20)

        # --- 波动率 ---
        daily_ret = close.pct_change()
        out["vol_5"] = daily_ret.rolling(window=5, min_periods=5).std()
        out["vol_20"] = daily_ret.rolling(window=20, min_periods=20).std()

        # --- RSI ---
        out["rsi_14"] = rsi_indicator(close, 14)

        # --- MACD ---
        dif, dea, hist = macd(close)
        out["macd_dif"] = dif
        out["macd_dea"] = dea
        out["macd_hist"] = hist

        # --- 布林带位置 ---
        _, bb_upper, bb_lower = bollinger_bands(close, period=20, num_std=2.0)
        bb_width = (bb_upper - bb_lower).replace(0, np.nan)
        out["bollinger_position"] = (close - bb_lower) / bb_width

        # --- 成交量变化率 ---
        vol_ma5 = volume.rolling(window=5, min_periods=5).mean()
        out["volume_change_5"] = volume / vol_ma5.replace(0, np.nan) - 1.0

        # --- 均线偏离度 ---
        ma5 = close.rolling(window=5, min_periods=5).mean()
        ma20 = close.rolling(window=20, min_periods=20).mean()
        out["ma5_deviation"] = close / ma5.replace(0, np.nan) - 1.0
        out["ma20_deviation"] = close / ma20.replace(0, np.nan) - 1.0

        return out

    # ------------------------------------------------------------------
    # 标签构建
    # ------------------------------------------------------------------

    @staticmethod
    def build_labels(
        df: pd.DataFrame,
        forward_days: int = 5,
        task: str = "classification",
    ) -> pd.Series:
        """构建标签。

        Args:
            df: 含 close 列的 K 线 DataFrame。
            forward_days: 前瞻天数。
            task: classification 返回上涨=1/下跌=0；regression 返回未来收益率。

        Returns:
            标签 Series，index 与输入一致。最后 forward_days 个值为 NaN（未来不可知）。
        """
        close = df["close"]
        # 未来 N 日收益率 = close[t+N] / close[t] - 1
        future_ret = close.shift(-forward_days) / close - 1.0

        if task == "classification":
            label = (future_ret > 0).astype(float)
            # 最后 forward_days 行未来数据不可得，置 NaN
            label[future_ret.isna()] = np.nan
            return label
        else:
            return future_ret

    # ------------------------------------------------------------------
    # 模型训练
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """训练模型，按时间顺序划分训练/测试集（避免数据泄露）。

        Args:
            X: 特征矩阵（已 dropna）。
            y: 标签（与 X 对齐，已 dropna）。
            test_size: 测试集比例（从尾部截取）。

        Returns:
            性能指标字典: accuracy, auc, confusion_matrix, precision, recall, f1,
            train_samples, test_samples。

        Raises:
            ImportError: scikit-learn 未安装。
            ValueError: 数据量不足。
        """
        _try_import_sklearn()
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        if len(X) < 30:
            raise ValueError(f"训练数据不足（仅 {len(X)} 条），至少需要 30 条")

        # 按时间顺序划分：前 (1-test_size) 训练，后 test_size 测试
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        if len(X_train) < 10 or len(X_test) < 5:
            raise ValueError(
                f"划分后样本不足: train={len(X_train)}, test={len(X_test)}"
            )

        self.feature_names = list(X.columns)

        # 默认超参数
        default_params: Dict[str, Any] = {"random_state": 42}
        if self.model_type == "random_forest":
            default_params.update({"n_estimators": 100, "max_depth": 10})
        default_params.update(self.model_params)

        model_cls = _get_model_class(self.model_type)
        self.model = model_cls(**default_params)

        # SVM 概率输出需要 probability=True
        if self.model_type == "svm":
            self.model.set_params(probability=True)

        self.model.fit(X_train, y_train)
        self.train_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 评估
        y_pred = self.model.predict(X_test)
        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        }

        # AUC 需要预测概率
        try:
            y_proba = self.model.predict_proba(X_test)
            # 取正类（label=1）概率
            pos_idx = list(self.model.classes_).index(1) if 1 in self.model.classes_ else 1
            metrics["auc"] = float(roc_auc_score(y_test, y_proba[:, pos_idx]))
        except (ValueError, IndexError):
            metrics["auc"] = 0.5

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        metrics["confusion_matrix"] = cm.tolist()

        logger.info(
            "ML模型训练完成: %s, accuracy=%.4f, auc=%.4f, train=%d, test=%d",
            self.model_type, metrics["accuracy"], metrics["auc"],
            len(X_train), len(X_test),
        )
        return metrics

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """输出上涨概率（0-1）。

        Args:
            X: 特征矩阵，列名需与训练时一致。

        Returns:
            上涨概率数组，shape=(n_samples,)。

        Raises:
            RuntimeError: 模型尚未训练。
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 train() 或 load()")
        proba = self.model.predict_proba(X)
        # 取正类（1=上涨）概率
        pos_idx = list(self.model.classes_).index(1) if 1 in self.model.classes_ else 1
        return proba[:, pos_idx]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """输出分类预测（0/1）。

        Args:
            X: 特征矩阵。

        Returns:
            预测标签数组（0=下跌, 1=上涨）。
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 train() 或 load()")
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # 预测信号
    # ------------------------------------------------------------------

    def predict_signal(
        self,
        df: pd.DataFrame,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.4,
    ) -> Dict[str, Any]:
        """基于最新 K 线预测信号。

        Args:
            df: 最近 K 线 DataFrame（至少需要 25 行以计算特征）。
            buy_threshold: 买入概率阈值。
            sell_threshold: 卖出概率阈值。

        Returns:
            {probability, signal: buy/sell/hold, confidence}
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练")

        features = self.build_features(df)
        # 取最后一行有效特征
        latest = features.dropna()
        if latest.empty:
            return {"probability": 0.5, "signal": "hold", "confidence": 0.5}

        X = latest.iloc[[-1]][self.feature_names]
        prob = float(self.predict_proba(X)[0])

        if prob > buy_threshold:
            signal = "buy"
        elif prob < sell_threshold:
            signal = "sell"
        else:
            signal = "hold"

        return {
            "probability": round(prob, 4),
            "signal": signal,
            "confidence": round(abs(prob - 0.5) * 2, 4),
        }

    # ------------------------------------------------------------------
    # 模型保存 / 加载
    # ------------------------------------------------------------------

    def save(self, model_path: str) -> None:
        """保存模型到文件。

        Args:
            model_path: 模型文件路径（.joblib 或 .pkl）。
        """
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "forward_days": self.forward_days,
            "task": self.task,
            "train_date": self.train_date,
        }
        path = Path(model_path)
        if not path.is_absolute():
            path = MODELS_DIR / path

        try:
            import joblib
            joblib.dump(payload, str(path))
        except ImportError:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(payload, f)

        logger.info("模型已保存到: %s", path)

    @classmethod
    def load(cls, model_path: str) -> "MLPredictor":
        """从文件加载模型。

        Args:
            model_path: 模型文件路径。

        Returns:
            加载后的 MLPredictor 实例。
        """
        path = Path(model_path)
        if not path.is_absolute():
            path = MODELS_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")

        try:
            import joblib
            payload = joblib.load(str(path))
        except ImportError:
            import pickle
            with open(path, "rb") as f:
                payload = pickle.load(f)

        obj = cls(
            model_type=payload.get("model_type", "random_forest"),
            forward_days=payload.get("forward_days", 5),
            task=payload.get("task", "classification"),
        )
        obj.model = payload["model"]
        obj.feature_names = payload.get("feature_names", [])
        obj.train_date = payload.get("train_date")
        logger.info("模型已加载: %s (训练于 %s)", path, obj.train_date)
        return obj
