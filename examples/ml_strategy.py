#!/usr/bin/env python3
"""机器学习价格方向预测策略示例。

演示完整流程:
    1. 构造模拟 K 线数据
    2. 创建 MLPredictor 并训练模型
    3. 打印性能指标
    4. 预测最新信号
    5. 保存 / 加载模型
    6. 用 MLStrategy 跑回测

运行方式:
    cd quant_trading_system
    python examples/ml_strategy.py

注意: 需要安装 scikit-learn: pip install scikit-learn
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def make_mock_klines(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """构造带趋势的模拟 K 线数据。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
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


def main() -> None:
    print("=" * 60)
    print("机器学习价格方向预测策略示例")
    print("=" * 60)

    # 检查 sklearn
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("\n[错误] scikit-learn 未安装，请运行: pip install scikit-learn")
        sys.exit(1)

    from ml.predictor import MLPredictor
    from strategies.ml_strategy import MLStrategy
    from backtest.engine import BacktestEngine

    # 1. 构造模拟数据
    print("\n[1] 构造模拟 K 线数据 (500 条)...")
    df = make_mock_klines(500)
    print(f"    数据范围: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"    最新收盘价: {df['close'].iloc[-1]:.2f}")

    # 2. 特征工程
    print("\n[2] 构建特征工程...")
    features = MLPredictor.build_features(df)
    labels = MLPredictor.build_labels(df, forward_days=5)
    print(f"    特征数: {len(features.columns)}")
    print(f"    特征列: {list(features.columns)}")

    # 3. 训练模型
    print("\n[3] 训练 RandomForest 模型...")
    combined = features.copy()
    combined["label"] = labels
    combined = combined.dropna()
    X = combined[features.columns]
    y = combined["label"]

    predictor = MLPredictor(model_type="random_forest", forward_days=5)
    metrics = predictor.train(X, y, test_size=0.2)

    print(f"    训练样本数: {metrics['train_samples']}")
    print(f"    测试样本数: {metrics['test_samples']}")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    AUC:       {metrics['auc']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1:        {metrics['f1']:.4f}")
    print(f"    混淆矩阵:  {metrics['confusion_matrix']}")

    # 4. 预测最新信号
    print("\n[4] 预测最新交易日信号...")
    signal = predictor.predict_signal(df.tail(30))
    print(f"    上涨概率: {signal['probability']:.4f}")
    print(f"    信号:     {signal['signal']}")
    print(f"    置信度:   {signal['confidence']:.4f}")

    # 5. 保存 / 加载模型
    print("\n[5] 保存模型到 models/ 目录...")
    model_path = "demo_ml_model.joblib"
    predictor.save(model_path)
    print(f"    已保存: {model_path}")

    print("\n[6] 从文件加载模型...")
    loaded = MLPredictor.load(model_path)
    signal2 = loaded.predict_signal(df.tail(30))
    print(f"    加载后预测概率: {signal2['probability']:.4f}")
    print(f"    加载后信号:     {signal2['signal']}")

    # 6. MLStrategy 回测
    print("\n[7] 使用 MLStrategy 跑回测...")
    strategy = MLStrategy(params={
        "model_type": "random_forest",
        "forward_days": 5,
        "buy_threshold": 0.6,
        "sell_threshold": 0.4,
    })
    strategy.train_model(df)

    engine = BacktestEngine(initial_capital=1_000_000)
    result = engine.run(df, strategy, symbol="DEMO")

    print(f"    交易次数:   {result.metrics.get('交易次数', 0)}")
    print(f"    累计收益率: {result.metrics.get('累计收益率', 0)*100:.2f}%")
    print(f"    最大回撤:   {result.metrics.get('最大回撤', 0)*100:.2f}%")
    print(f"    夏普比率:   {result.metrics.get('夏普比率', 0):.4f}")

    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
