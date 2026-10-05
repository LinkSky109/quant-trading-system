"""TrainingDataExporter 单元测试。

全部使用临时 SQLite + 构造的已知走势 K线，不依赖网络和真实数据库：
  - load_decisions 正确读取并反序列化 JSON 字段
  - compute_labels 标签判定（上涨→buy / 下跌→sell / 横盘→hold）
  - export_jsonl 每行合法 JSON 且字段完整
  - export_csv 含展开的特征列与标签列
  - split_dataset 按时间划分，train/val/test 无重叠且比例正确
  - compute_stats 统计正确
  - export_all 端到端生成所有文件

运行:
    cd 项目根目录 && python -m pytest tests/test_training_data.py -v
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from jev.training_data import FEATURE_NAMES, TrainingDataExporter
from persistence.database import Database


# ---------------------------------------------------------------------------
# Fixtures / 构造工具
# ---------------------------------------------------------------------------

def _make_klines(closes) -> pd.DataFrame:
    """构造 index 为工作日、含 close 列的 K线 DataFrame。"""
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"close": list(closes)}, index=idx)


# 30 个交易日的确定性走势
_UP = _make_klines([100.0 * (1 + 0.01 * i) for i in range(30)])     # 稳定上涨 +1%/日
_DOWN = _make_klines([100.0 * (1 - 0.01 * i) for i in range(30)])   # 稳定下跌 -1%/日
_FLAT = _make_klines([100.0 for _ in range(30)])                    # 横盘


def _market_state() -> dict:
    """构造含全部 7 个特征的扁平 market_state（与落库结构一致）。"""
    return {
        "price": 100.0,
        "price_change_5d": 0.02,
        "ma5_ma20_ratio": 1.01,
        "volume_ratio": 1.2,
        "rsi": 55.0,
        "macd_signal": 1,
        "volatility_20d": 0.15,
    }


@pytest.fixture
def env(tmp_path):
    """临时数据库 + 三个标的的 mock K线。返回 (db_path, price_data)。"""
    db_path = str(tmp_path / "train_test.db")
    db = Database(db_path=db_path)
    price_data = {"UP.SH": _UP, "DOWN.SZ": _DOWN, "FLAT.SH": _FLAT}
    return db_path, price_data


def _date(df: pd.DataFrame, pos: int) -> str:
    """取 df 第 pos 个交易日的日期字符串（与落库 timestamp 同格式）。"""
    return df.index[pos].strftime("%Y-%m-%d")


def _record(db_path, timestamp, symbol, final_action, executed=True,
            strategy_signal="ma_cross", market_state=None):
    """插入一条 mock jev_decision（带完整 market_state）。"""
    db = Database(db_path=db_path)
    return db.insert_jev_decision(
        timestamp=timestamp,
        symbol=symbol,
        strategy_signal=strategy_signal,
        strategy_confidence=0.6,
        market_state=market_state if market_state is not None else _market_state(),
        probabilities={"buy": 0.5, "sell": 0.3, "hold": 0.2},
        final_action=final_action,
        final_confidence=0.7,
        executed=executed,
        reason="test",
        mode="mock",
    )


# ---------------------------------------------------------------------------
# load_decisions
# ---------------------------------------------------------------------------

def test_load_decisions_deserializes(env):
    db_path, price_data = env
    _record(db_path, _date(_UP, 5), "UP.SH", "buy", strategy_signal="ma_cross")
    _record(db_path, _date(_DOWN, 5), "DOWN.SZ", "sell", strategy_signal="boll")

    exporter = TrainingDataExporter(db_path=db_path, price_data=price_data)
    decisions = exporter.load_decisions()

    assert len(decisions) == 2
    # JSON 字段已反序列化
    d0 = decisions[0]
    assert isinstance(d0["market_state"], dict)
    assert d0["market_state"]["price"] == 100.0
    assert isinstance(d0["probabilities"], dict)
    assert d0["probabilities"]["buy"] == 0.5
    assert d0["executed"] is True


def test_load_decisions_filters(env):
    db_path, price_data = env
    _record(db_path, _date(_UP, 5), "UP.SH", "buy", strategy_signal="ma_cross")
    _record(db_path, _date(_DOWN, 5), "DOWN.SZ", "sell", strategy_signal="boll")

    exporter = TrainingDataExporter(db_path=db_path, price_data=price_data)
    only_up = exporter.load_decisions(symbol="UP.SH")
    assert len(only_up) == 1
    assert only_up[0]["symbol"] == "UP.SH"

    only_ma = exporter.load_decisions(strategy="ma_cross")
    assert len(only_ma) == 1
    assert only_ma[0]["strategy_signal"] == "ma_cross"


# ---------------------------------------------------------------------------
# compute_labels 标签判定
# ---------------------------------------------------------------------------

def test_label_up_trend_is_buy(env):
    db_path, price_data = env
    # 决策日 idx=5，forward_days=5 → idx=10，UP 走势 +约5% > 2% → buy 最优
    _record(db_path, _date(_UP, 5), "UP.SH", "buy", executed=True)

    exporter = TrainingDataExporter(db_path=db_path, price_data=price_data,
                                    forward_days=5)
    samples = exporter.compute_labels(exporter.load_decisions())

    assert len(samples) == 1
    s = samples[0]
    assert s["label"] == "buy"
    assert s["future_return"] > 0.02
    # 上涨窗口内未跌破 -3% 止损
    assert s["stop_loss_triggered"] is False
    # features 已转为 states 数组
    assert isinstance(s["features"], list)
    feat_map = {f["feature"]: f["value"] for f in s["features"]}
    assert feat_map["price"] == 100.0
    assert "rsi" in feat_map


def test_label_down_trend_is_sell(env):
    db_path, price_data = env
    # 决策日 idx=5 → idx=10，DOWN 走势 -约5% < -2% → sell 最优
    _record(db_path, _date(_DOWN, 5), "DOWN.SZ", "sell", executed=True)

    exporter = TrainingDataExporter(db_path=db_path, price_data=price_data,
                                    forward_days=5)
    samples = exporter.compute_labels(exporter.load_decisions())

    assert len(samples) == 1
    s = samples[0]
    assert s["label"] == "sell"
    assert s["future_return"] < -0.02
    # 下跌窗口内跌幅远超 3% → 触发止损
    assert s["stop_loss_triggered"] is True


def test_label_flat_is_hold(env):
    db_path, price_data = env
    # 横盘 |收益|≈0 <= 2% → hold 最优
    _record(db_path, _date(_FLAT, 5), "FLAT.SH", "hold", executed=False)

    exporter = TrainingDataExporter(db_path=db_path, price_data=price_data,
                                    forward_days=5)
    samples = exporter.compute_labels(exporter.load_decisions())

    assert len(samples) == 1
    s = samples[0]
    assert s["label"] == "hold"
    assert abs(s["future_return"]) <= 0.02
    # 原始动作保留
    assert s["original_action"] == "hold"
    assert s["original_executed"] is False


def test_insufficient_future_data_skipped(env):
    db_path, price_data = env
    # 决策日 idx=28，forward_days=5 → 需要 idx=33，超出 30 根 → 丢弃
    _record(db_path, _date(_UP, 28), "UP.SH", "buy")
    # 正常可标注
    _record(db_path, _date(_UP, 5), "UP.SH", "buy")

    exporter = TrainingDataExporter(db_path=db_path, price_data=price_data,
                                    forward_days=5)
    samples = exporter.compute_labels(exporter.load_decisions())
    assert len(samples) == 1


def test_missing_symbol_klines_no_crash(env, monkeypatch):
    db_path, price_data = env
    from data.data_fetcher import DataFetcher

    def _boom(self, *args, **kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr(DataFetcher, "get_klines", _boom)
    _record(db_path, "2024-01-08", "NOPE.SH", "buy")

    exporter = TrainingDataExporter(db_path=db_path, price_data=price_data,
                                    forward_days=5)
    # 在线拉取失败应静默跳过，不抛异常
    samples = exporter.compute_labels(exporter.load_decisions())
    assert samples == []


# ---------------------------------------------------------------------------
# export_jsonl / export_csv
# ---------------------------------------------------------------------------

def _one_sample(env) -> list:
    db_path, price_data = env
    _record(db_path, _date(_UP, 5), "UP.SH", "buy", executed=True)
    exporter = TrainingDataExporter(db_path=db_path, price_data=price_data,
                                    forward_days=5)
    return exporter.compute_labels(exporter.load_decisions())


def test_export_jsonl(env, tmp_path):
    samples = _one_sample(env)
    exporter = TrainingDataExporter(price_data={})
    out = str(tmp_path / "out.jsonl")
    path = exporter.export_jsonl(samples, out)

    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])  # 每行合法 JSON
    for key in ("id", "timestamp", "symbol", "strategy_signal", "features",
                "label", "future_return", "max_drawdown",
                "stop_loss_triggered", "original_action", "original_executed"):
        assert key in row


def test_export_csv(env, tmp_path):
    samples = _one_sample(env)
    exporter = TrainingDataExporter(price_data={})
    out = str(tmp_path / "out.csv")
    path = exporter.export_csv(samples, out)

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames

    assert len(rows) == 1
    # 7 个特征列都存在
    for feat in FEATURE_NAMES:
        assert feat in cols
    # 标签与辅助列存在
    for col in ("label", "future_return", "max_drawdown",
                "stop_loss_triggered", "original_action"):
        assert col in cols
    # 特征值已展开
    assert float(rows[0]["price"]) == 100.0
    assert rows[0]["label"] == "buy"


# ---------------------------------------------------------------------------
# split_dataset（按时间，无泄露）
# ---------------------------------------------------------------------------

def test_split_dataset_by_time(env):
    db_path, price_data = env
    # 在 UP 上插入 10 条时间递增的决策（pos 5..14，均有足够未来数据）
    for pos in range(5, 15):
        _record(db_path, _date(_UP, pos), "UP.SH", "buy")

    exporter = TrainingDataExporter(db_path=db_path, price_data=price_data,
                                    forward_days=5)
    samples = exporter.compute_labels(exporter.load_decisions())
    assert len(samples) == 10

    splits = exporter.split_dataset(samples, train_ratio=0.7,
                                   val_ratio=0.2, test_ratio=0.1)
    assert len(splits["train"]) == 7
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 1

    # 无重叠：id 集合互不相交
    train_ids = {s["id"] for s in splits["train"]}
    val_ids = {s["id"] for s in splits["val"]}
    test_ids = {s["id"] for s in splits["test"]}
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)

    # 按时间单调：train 所有时间 <= val 所有时间 <= test 所有时间
    def _ts(ss):
        return [pd.Timestamp(s["timestamp"]) for s in ss]
    assert max(_ts(splits["train"])) <= min(_ts(splits["val"]))
    assert max(_ts(splits["val"])) <= min(_ts(splits["test"]))


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------

def test_compute_stats(env):
    db_path, price_data = env
    # 1 buy 最优(UP) + 1 sell 最优(DOWN) + 1 hold(FLAT)
    _record(db_path, _date(_UP, 5), "UP.SH", "buy", executed=True)
    _record(db_path, _date(_DOWN, 5), "DOWN.SZ", "sell", executed=True)
    _record(db_path, _date(_FLAT, 5), "FLAT.SH", "hold", executed=False)

    exporter = TrainingDataExporter(db_path=db_path, price_data=price_data,
                                    forward_days=5)
    samples = exporter.compute_labels(exporter.load_decisions())
    stats = exporter.compute_stats(samples)

    assert stats["total"] == 3
    assert stats["label_distribution"]["buy"]["count"] == 1
    assert stats["label_distribution"]["sell"]["count"] == 1
    assert stats["label_distribution"]["hold"]["count"] == 1
    # 三个样本原始动作都等于最优标签 → accuracy = 1.0
    assert stats["accuracy"] == 1.0
    assert stats["by_symbol"] == {"UP.SH": 1, "DOWN.SZ": 1, "FLAT.SH": 1}


def test_compute_stats_empty():
    exporter = TrainingDataExporter(price_data={})
    stats = exporter.compute_stats([])
    assert stats["total"] == 0
    assert stats["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# export_all 端到端
# ---------------------------------------------------------------------------

def test_export_all_end_to_end(env, tmp_path):
    db_path, price_data = env
    _record(db_path, _date(_UP, 5), "UP.SH", "buy", executed=True)
    _record(db_path, _date(_DOWN, 5), "DOWN.SZ", "sell", executed=True)
    _record(db_path, _date(_FLAT, 5), "FLAT.SH", "hold", executed=False)

    exporter = TrainingDataExporter(db_path=db_path, price_data=price_data,
                                    forward_days=5)
    out_dir = str(tmp_path / "training_data")
    result = exporter.export_all(output_dir=out_dir)

    assert result["sample_count"] == 3
    paths = result["file_paths"]
    # train/val/test 的 jsonl + csv 都生成了
    for part in ("train", "val", "test"):
        assert Path(paths[f"{part}_jsonl"]).exists()
        assert Path(paths[f"{part}_csv"]).exists()
    assert Path(paths["stats_json"]).exists()

    # 统计信息齐全
    assert result["stats"]["total"] == 3
    assert "label_distribution" in result["stats"]
