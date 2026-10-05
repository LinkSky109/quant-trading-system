"""机器学习策略 API 路由（可插拔 snippet）。

本模块**不修改** ``web-dashboard/server.py``，而是暴露
:func:`register_ml_routes`，在 server 启动后一行挂载：

.. code-block:: python

    from web_dashboard.ml_routes import register_ml_routes
    register_ml_routes(app, manager, SYMBOL_SET, ok, err)

提供接口:
    - ``POST /api/ml/train``      训练模型
    - ``POST /api/ml/predict``    预测指定标的上涨概率
    - ``POST /api/ml/backtest``   ML 策略回测
    - ``GET  /api/ml/models``     已训练模型列表
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

#: 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = _PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Pydantic 请求体模型
# ---------------------------------------------------------------------------

class MLTrainReq(BaseModel):
    """训练模型请求体。"""
    symbol: str = Field(..., description="标的代码")
    start_date: Optional[str] = Field(default=None, description="起始日期 YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="结束日期 YYYY-MM-DD")
    model_type: str = Field(default="random_forest", description="模型类型")
    forward_days: int = Field(default=5, ge=1, le=60, description="前瞻天数")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="测试集比例")


class MLPredictReq(BaseModel):
    """预测请求体。"""
    symbol: str = Field(..., description="标的代码")
    model_path: Optional[str] = Field(default=None, description="模型文件路径")


class MLBacktestReq(BaseModel):
    """ML 策略回测请求体。"""
    symbol: str = Field(..., description="标的代码")
    start_date: str = Field(..., description="回测起始日期")
    end_date: str = Field(..., description="回测结束日期")
    model_type: str = Field(default="random_forest", description="模型类型")
    forward_days: int = Field(default=5, ge=1, le=60, description="前瞻天数")
    buy_threshold: float = Field(default=0.6, ge=0.5, le=0.95, description="买入阈值")
    sell_threshold: float = Field(default=0.4, ge=0.05, le=0.5, description="卖出阈值")
    initial_capital: float = Field(default=1_000_000.0, description="初始资金")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _check_sklearn() -> Optional[Dict[str, Any]]:
    """检查 sklearn 是否可用，不可用时返回错误响应字典。"""
    try:
        import sklearn  # noqa: F401
        return None
    except ImportError:
        return {
            "code": 50002,
            "message": "scikit-learn未安装，请 pip install scikit-learn",
            "data": None,
            "_http_status": 503,
        }


def _get_klines(manager: Any, symbol: str) -> Optional[pd.DataFrame]:
    """从 manager 获取标的 K 线。"""
    try:
        sim = manager.get(symbol)
    except Exception:
        return None
    if sim is None or getattr(sim, "klines", None) is None:
        return None
    df = sim.klines
    return df if isinstance(df, pd.DataFrame) and not df.empty else None


def _list_models() -> list:
    """扫描 models/ 目录，返回已训练模型列表。"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for ext in ("*.joblib", "*.pkl"):
        for p in sorted(MODELS_DIR.glob(ext)):
            stat = p.stat()
            result.append({
                "name": p.name,
                "path": str(p),
                "created_at": pd.Timestamp(stat.st_ctime, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                "size_kb": round(stat.st_size / 1024, 1),
            })
    return result


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------

def register_ml_routes(
    app: Any,
    manager: Any,
    symbol_set: Optional[Set[str]] = None,
    ok: Any = None,
    err: Any = None,
) -> None:
    """在 FastAPI app 上注册 ML 策略路由。

    Args:
        app: FastAPI 实例。
        manager: MultiSymbolManager 实例，用于获取 K 线数据。
        symbol_set: 允许交易的标的集合（可选）。
        ok: 成功响应封装函数。
        err: 错误响应封装函数。
    """
    # 兼容：若未传入 ok/err，使用默认封装
    if ok is None:
        def ok(data: Any = None, message: str = "success") -> Dict[str, Any]:
            return {"code": 0, "message": message, "data": data}
    if err is None:
        def err(code: int, message: str, http_status: int = 400) -> Dict[str, Any]:
            return {"code": code, "message": message, "data": None,
                    "_http_status": http_status}

    # ------------------------------------------------------------------
    # POST /api/ml/train
    # ------------------------------------------------------------------

    @app.post("/api/ml/train")
    async def api_ml_train(req: MLTrainReq) -> Dict[str, Any]:
        """训练 ML 模型。

        body: ``{symbol, start_date?, end_date?, model_type?, forward_days?, test_size?}``
        """
        sk_err = _check_sklearn()
        if sk_err is not None:
            return sk_err

        df = _get_klines(manager, req.symbol)
        if df is None or df.empty:
            return err(50001, f"无法获取标的 {req.symbol} 的 K 线数据")

        if req.start_date:
            df = df[df.index >= pd.Timestamp(req.start_date)]
        if req.end_date:
            df = df[df.index <= pd.Timestamp(req.end_date)]

        if len(df) < 60:
            return err(40003, f"有效K线不足60条（当前 {len(df)} 条），无法训练")

        from ml.predictor import MLPredictor

        try:
            predictor = MLPredictor(
                model_type=req.model_type,
                forward_days=req.forward_days,
            )
            features = MLPredictor.build_features(df)
            labels = MLPredictor.build_labels(df, forward_days=req.forward_days)

            combined = features.copy()
            combined["label"] = labels
            combined = combined.dropna()
            X = combined[features.columns]
            y = combined["label"]

            metrics = predictor.train(X, y, test_size=req.test_size)

            # 保存模型
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_filename = (
                f"ml_{req.symbol.replace('.', '_')}_{req.model_type}_"
                f"fd{req.forward_days}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            )
            model_path = MODELS_DIR / model_filename
            predictor.save(str(model_path))

            return ok({
                "accuracy": metrics["accuracy"],
                "auc": metrics["auc"],
                "confusion_matrix": metrics["confusion_matrix"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "model_type": req.model_type,
                "feature_count": len(predictor.feature_names),
                "train_samples": metrics["train_samples"],
                "test_samples": metrics["test_samples"],
                "model_path": str(model_path),
            })
        except ImportError as e:
            return err(50002, str(e), http_status=503)
        except Exception as e:
            logger.exception("ML 模型训练失败")
            return err(50003, f"训练失败: {e}")

    # ------------------------------------------------------------------
    # POST /api/ml/predict
    # ------------------------------------------------------------------

    @app.post("/api/ml/predict")
    async def api_ml_predict(req: MLPredictReq) -> Dict[str, Any]:
        """预测指定标的最新上涨概率。

        body: ``{symbol, model_path?}``
        """
        sk_err = _check_sklearn()
        if sk_err is not None:
            return sk_err

        df = _get_klines(manager, req.symbol)
        if df is None or df.empty:
            return err(50001, f"无法获取标的 {req.symbol} 的 K 线数据")

        from ml.predictor import MLPredictor

        try:
            if req.model_path:
                predictor = MLPredictor.load(req.model_path)
            else:
                # 自动找最新模型
                models = _list_models()
                if not models:
                    return err(40004, "未找到已训练模型，请先调用 /api/ml/train")
                predictor = MLPredictor.load(models[-1]["path"])

            result = predictor.predict_signal(df)
            return ok({
                "symbol": req.symbol,
                "probability": result["probability"],
                "signal": result["signal"],
                "confidence": result["confidence"],
                "as_of_date": pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d"),
            })
        except ImportError as e:
            return err(50002, str(e), http_status=503)
        except Exception as e:
            logger.exception("ML 预测失败")
            return err(50005, f"预测失败: {e}")

    # ------------------------------------------------------------------
    # POST /api/ml/backtest
    # ------------------------------------------------------------------

    @app.post("/api/ml/backtest")
    async def api_ml_backtest(req: MLBacktestReq) -> Dict[str, Any]:
        """运行 ML 策略回测。

        body: ``{symbol, start_date, end_date, model_type?, forward_days?,
                 buy_threshold?, sell_threshold?, initial_capital?}``
        """
        sk_err = _check_sklearn()
        if sk_err is not None:
            return sk_err

        df = _get_klines(manager, req.symbol)
        if df is None or df.empty:
            return err(50001, f"无法获取标的 {req.symbol} 的 K 线数据")

        # 筛选回测区间
        mask = (df.index >= pd.Timestamp(req.start_date)) & \
               (df.index <= pd.Timestamp(req.end_date))
        bt_df = df[mask].copy()
        if len(bt_df) < 30:
            return err(40003, f"回测区间有效K线不足30条（当前 {len(bt_df)} 条）")

        from backtest.engine import BacktestEngine
        from strategies.ml_strategy import MLStrategy

        try:
            # 先用全量数据训练模型（含回测区间数据，简化处理）
            strategy = MLStrategy(params={
                "model_type": req.model_type,
                "forward_days": req.forward_days,
                "buy_threshold": req.buy_threshold,
                "sell_threshold": req.sell_threshold,
            })
            train_metrics = strategy.train_model(df)

            engine = BacktestEngine(initial_capital=req.initial_capital)
            result = engine.run(bt_df, strategy, symbol=req.symbol)

            return ok({
                "symbol": req.symbol,
                "metrics": result.metrics,
                "train_metrics": train_metrics,
                "equity_curve": [
                    [d.strftime("%Y-%m-%d"), round(float(v), 2)]
                    for d, v in result.equity_curve.items()
                ],
                "benchmark_curve": [
                    [d.strftime("%Y-%m-%d"), round(float(v), 2)]
                    for d, v in result.benchmark_curve.items()
                ],
                "trades": [
                    {
                        "date": t.date.strftime("%Y-%m-%d"),
                        "action": t.action,
                        "price": round(float(t.price), 2),
                        "shares": int(t.shares),
                        "pnl": round(float(t.pnl), 2) if t.pnl is not None else None,
                    }
                    for t in result.trades
                ],
            })
        except ImportError as e:
            return err(50002, str(e), http_status=503)
        except Exception as e:
            logger.exception("ML 策略回测失败")
            return err(50006, f"回测失败: {e}")

    # ------------------------------------------------------------------
    # GET /api/ml/models
    # ------------------------------------------------------------------

    @app.get("/api/ml/models")
    async def api_ml_models() -> Dict[str, Any]:
        """列出已训练模型。"""
        models = _list_models()
        return ok({"models": models, "count": len(models)})

    logger.info("ML 策略路由已注册: /api/ml/{train,predict,backtest,models}")
