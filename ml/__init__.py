"""机器学习价格方向预测模块。

提供 MLPredictor 封装特征工程、标签构建、模型训练/预测/保存加载，
以及 MLStrategy 策略子类。

注意: scikit-learn 为可选依赖，未安装时本模块可正常 import，
但训练/预测相关功能会在调用时抛出 ImportError。
"""
from __future__ import annotations

__version__ = "0.1.0"
