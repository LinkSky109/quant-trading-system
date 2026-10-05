"""机器学习价格方向预测策略。

基于 MLPredictor 的预测概率生成交易信号：
- 上涨概率 > buy_threshold → 买入信号 (signal=1)
- 上涨概率 < sell_threshold → 卖出信号 (signal=-1)
- 其余 → 持有 (signal=0)

严格避免未来函数：每个时间点 t 的预测仅使用 t 日及之前的数据。
模型在回测开始前训练一次（使用全量或训练窗口数据），随后逐日推理。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ml.predictor import MLPredictor
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """机器学习方向预测策略。

    Attributes:
        name: 策略名称 "ml_direction"。
        predictor: MLPredictor 实例（训练或加载后可用）。
        buy_threshold: 买入概率阈值。
        sell_threshold: 卖出概率阈值。
        retrain_days: 重训练间隔天数（0 表示不自动重训练）。
    """

    name = "ml_direction"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """初始化 ML 策略。

        Args:
            params: 策略参数字典，支持:
                - model_type: 模型类型（默认 random_forest）
                - model_params: 模型超参数字典
                - forward_days: 前瞻标签天数（默认 5）
                - buy_threshold: 买入阈值（默认 0.6）
                - sell_threshold: 卖出阈值（默认 0.4）
                - retrain_days: 重训练间隔（默认 0，不自动重训练）
                - model_path: 可选，已训练模型路径
        """
        super().__init__(params)
        self.model_type: str = str(self.params.get("model_type", "random_forest"))
        self.model_params: Dict[str, Any] = dict(self.params.get("model_params", {}))
        self.forward_days: int = int(self.params.get("forward_days", 5))
        self.buy_threshold: float = float(self.params.get("buy_threshold", 0.6))
        self.sell_threshold: float = float(self.params.get("sell_threshold", 0.4))
        self.retrain_days: int = int(self.params.get("retrain_days", 0))
        self.model_path: Optional[str] = self.params.get("model_path")

        self.predictor = MLPredictor(
            model_type=self.model_type,
            model_params=self.model_params,
            forward_days=self.forward_days,
        )

        # 若指定了模型路径，自动加载
        if self.model_path:
            self.load_model(self.model_path)

    # ------------------------------------------------------------------
    # 模型训练 / 加载
    # ------------------------------------------------------------------

    def train_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """用 K 线数据训练模型。

        Args:
            df: 含 open/high/low/close/volume 的完整 K 线 DataFrame。

        Returns:
            训练性能指标字典。
        """
        features = MLPredictor.build_features(df)
        labels = MLPredictor.build_labels(df, forward_days=self.forward_days)

        # 对齐：drop NaN（前 20 行 rolling 预热 + 后 forward_days 行标签不可知）
        combined = features.copy()
        combined["label"] = labels
        combined = combined.dropna()

        X = combined[features.columns]
        y = combined["label"]

        logger.info(
            "开始训练 ML 模型: %s, 样本数=%d, 特征数=%d",
            self.model_type, len(X), len(features.columns),
        )
        return self.predictor.train(X, y)

    def load_model(self, model_path: str) -> None:
        """加载已训练模型。

        Args:
            model_path: 模型文件路径。
        """
        self.predictor = MLPredictor.load(model_path)
        # 同步阈值参数（保留策略实例上的阈值设置）
        logger.info("MLStrategy 已加载模型: %s", model_path)

    # ------------------------------------------------------------------
    # 信号计算（基类要求实现）
    # ------------------------------------------------------------------

    def _compute_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """对每个交易日用已训练模型预测未来方向，生成 signal 和 confidence。

        严格避免未来函数：
        - 特征全部基于 rolling/pct_change，天然只使用当日及之前数据
        - 基类 generate_signals 还会额外 shift(1) 延迟一天执行

        Args:
            df: 行情 DataFrame（回测引擎已添加技术指标列，此处不依赖）。

        Returns:
            含 signal(1/-1/0) 和 confidence 列的 DataFrame。
        """
        out = df.copy()
        out["signal"] = 0
        out["confidence"] = 0.0

        if self.predictor.model is None:
            logger.warning("ML 模型未训练，输出全零信号")
            return out

        # 构建全量特征
        features = MLPredictor.build_features(df)

        # 逐行预测（向量化：一次 predict_proba 全部有效行）
        valid = features.dropna()
        if valid.empty:
            return out

        X = valid[self.predictor.feature_names]
        probs = self.predictor.predict_proba(X)

        # 填充回 out
        prob_series = pd.Series(probs, index=valid.index)
        out["probability"] = prob_series.reindex(out.index)

        # 信号生成
        buy_mask = out["probability"] > self.buy_threshold
        sell_mask = out["probability"] < self.sell_threshold
        out.loc[buy_mask, "signal"] = 1
        out.loc[sell_mask, "signal"] = -1

        # 置信度：概率偏离 0.5 的程度，映射到 [0, 1]
        out["confidence"] = (out["probability"] - 0.5).abs() * 2
        out["confidence"] = out["confidence"].fillna(0.0)
        # 无信号日置信度归零
        out.loc[out["signal"] == 0, "confidence"] = 0.0

        return out
