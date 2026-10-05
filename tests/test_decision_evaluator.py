"""DecisionEvaluator 单元测试。

用临时 SQLite + 构造的已知走势 K线验证评估逻辑，不依赖真实数据库或网络：
  - 买入后上涨→正确买入；买入后下跌→错误买入
  - 卖出后下跌→正确卖出；卖出后上涨→错误卖出
  - hold 后小幅波动→正确观望；hold 后大幅波动→错误观望
  - 置信度分桶统计
  - 空数据 / 后续数据不足时不崩溃

运行:
    cd 项目根目录 && python -m pytest tests/test_decision_evaluator.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

from jev.decision_evaluator import DecisionEvaluator
from persistence.database import Database


# ---------------------------------------------------------------------------
# Fixtures / 构造工具
# ---------------------------------------------------------------------------

def _make_klines(closes) -> pd.DataFrame:
    """构造 index 为工作日、含 close 列的 K线 DataFrame。"""
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"close": list(closes)}, index=idx)


# 30 个交易日的确定性走势
_UP = _make_klines([100.0 * (1 + 0.01 * i) for i in range(30)])      # 稳定上涨
_DOWN = _make_klines([100.0 * (1 - 0.01 * i) for i in range(30)])    # 稳定下跌
_FLAT = _make_klines([100.0 for _ in range(30)])                      # 横盘


@pytest.fixture
def env(tmp_path):
    """临时数据库 + 三个标的的 mock K线。返回 (db, price_data)。"""
    db_path = str(tmp_path / "eval_test.db")
    db = Database(db_path=db_path)
    price_data = {"UP.SH": _UP, "DOWN.SZ": _DOWN, "FLAT.SH": _FLAT}
    return db, price_data


def _date(df: pd.DataFrame, pos: int) -> str:
    """取 df 第 pos 个交易日的日期字符串。"""
    return df.index[pos].strftime("%Y-%m-%d")


def _record(db, timestamp, symbol, action, confidence, executed=True,
            strategy_signal="ma_cross") -> None:
    db.insert_jev_decision(
        timestamp=timestamp,
        symbol=symbol,
        strategy_signal=strategy_signal,
        strategy_confidence=0.6,
        market_state={},
        probabilities={action: confidence},
        final_action=action,
        final_confidence=confidence,
        executed=executed,
        reason="test",
        mode="mock",
    )


# ---------------------------------------------------------------------------
# 买入 / 卖出 / hold 判定
# ---------------------------------------------------------------------------

def test_buy_after_up_is_correct(env):
    db, price_data = env
    # 决策日 idx=5，days=5 → 观察 idx=10，UP 走势上涨
    _record(db, _date(_UP, 5), "UP.SH", "buy", 0.7, executed=True)

    ev = DecisionEvaluator(db_path=db.db_path, price_data=price_data)
    report = ev.evaluate(days=5)

    assert report["summary"]["evaluated_decisions"] == 1
    assert report["summary"]["correct_buy"] == 1
    assert report["summary"]["wrong_buy"] == 0
    assert report["summary"]["overall_accuracy"] == 1.0


def test_buy_after_down_is_wrong(env):
    db, price_data = env
    _record(db, _date(_DOWN, 5), "DOWN.SZ", "buy", 0.7, executed=True)

    ev = DecisionEvaluator(db_path=db.db_path, price_data=price_data)
    report = ev.evaluate(days=5)

    assert report["summary"]["wrong_buy"] == 1
    assert report["summary"]["correct_buy"] == 0
    assert report["summary"]["overall_accuracy"] == 0.0


def test_sell_after_down_is_correct(env):
    db, price_data = env
    # 卖出后下跌 → 正确卖出（避开下跌）
    _record(db, _date(_DOWN, 5), "DOWN.SZ", "sell", 0.7, executed=True)

    ev = DecisionEvaluator(db_path=db.db_path, price_data=price_data)
    report = ev.evaluate(days=5)

    assert report["summary"]["correct_sell"] == 1
    assert report["summary"]["wrong_sell"] == 0


def test_sell_after_up_is_wrong(env):
    db, price_data = env
    # 卖出后上涨 → 错误卖出（卖飞）
    _record(db, _date(_UP, 5), "UP.SH", "sell", 0.7, executed=True)

    ev = DecisionEvaluator(db_path=db.db_path, price_data=price_data)
    report = ev.evaluate(days=5)

    assert report["summary"]["wrong_sell"] == 1
    assert report["summary"]["correct_sell"] == 0


def test_hold_small_move_is_correct(env):
    db, price_data = env
    # 横盘 |收益|≈0 < 2% → 正确观望
    _record(db, _date(_FLAT, 5), "FLAT.SH", "hold", 0.5, executed=False)

    ev = DecisionEvaluator(db_path=db.db_path, price_data=price_data)
    report = ev.evaluate(days=5)

    assert report["summary"]["correct_hold"] == 1
    assert report["summary"]["wrong_hold"] == 0


def test_hold_big_move_is_wrong(env):
    db, price_data = env
    # 横盘标的上 hold，但用 UP 走势（5 日 +约5% ≥ 2%）→ 错误观望（错过行情）
    _record(db, _date(_UP, 5), "UP.SH", "hold", 0.5, executed=False)

    ev = DecisionEvaluator(db_path=db.db_path, price_data=price_data)
    report = ev.evaluate(days=5)

    assert report["summary"]["wrong_hold"] == 1
    assert report["summary"]["correct_hold"] == 0


# ---------------------------------------------------------------------------
# 置信度分桶
# ---------------------------------------------------------------------------

def test_confidence_buckets(env):
    db, price_data = env
    # 高置信(0.9)正确买入；中低置信(0.5)错误买入
    _record(db, _date(_UP, 5), "UP.SH", "buy", 0.9, executed=True)
    _record(db, _date(_DOWN, 5), "DOWN.SZ", "buy", 0.5, executed=True)

    ev = DecisionEvaluator(db_path=db.db_path, price_data=price_data)
    report = ev.evaluate(days=5)

    hi = report["confidence_buckets"]["0.8-1.0"]
    mid = report["confidence_buckets"]["0.4-0.6"]
    assert hi["count"] == 1 and hi["correct"] == 1 and hi["accuracy"] == 1.0
    assert mid["count"] == 1 and mid["correct"] == 0 and mid["accuracy"] == 0.0
    # 其余桶为空
    assert report["confidence_buckets"]["0-0.4"]["count"] == 0


def test_confidence_bucket_boundary():
    """边界：0.4 归入 0.4-0.6，0.6 归入 0.6-0.8。"""
    assert DecisionEvaluator._conf_bucket(0.4) == "0.4-0.6"
    assert DecisionEvaluator._conf_bucket(0.6) == "0.6-0.8"
    assert DecisionEvaluator._conf_bucket(0.399) == "0-0.4"
    assert DecisionEvaluator._conf_bucket(0.8) == "0.8-1.0"
    assert DecisionEvaluator._conf_bucket(1.0) == "0.8-1.0"


# ---------------------------------------------------------------------------
# 按策略统计
# ---------------------------------------------------------------------------

def test_by_strategy_executed_accuracy(env):
    db, price_data = env
    # ma_cross: 1 笔执行且正确，1 笔被否决
    _record(db, _date(_UP, 5), "UP.SH", "buy", 0.8, executed=True,
            strategy_signal="ma_cross")
    _record(db, _date(_DOWN, 5), "DOWN.SZ", "buy", 0.8, executed=False,
            strategy_signal="ma_cross")

    ev = DecisionEvaluator(db_path=db.db_path, price_data=price_data)
    report = ev.evaluate(days=5)

    s = report["by_strategy"]["ma_cross"]
    assert s["total"] == 2
    assert s["executed"] == 1
    assert s["vetoed"] == 1
    assert s["executed_accuracy"] == 1.0  # 唯一被执行的那笔是正确买入


# ---------------------------------------------------------------------------
# 空数据 / 数据不足
# ---------------------------------------------------------------------------

def test_empty_db_no_crash(tmp_path):
    db_path = str(tmp_path / "empty.db")
    db = Database(db_path=db_path)  # noqa: F841
    ev = DecisionEvaluator(db_path=db_path, price_data={})
    report = ev.evaluate(days=5)

    assert report["summary"]["total_decisions"] == 0
    assert report["summary"]["evaluated_decisions"] == 0
    assert report["recent_decisions"] == []
    assert "0.8-1.0" in report["confidence_buckets"]


def test_insufficient_future_data_skipped(env):
    db, price_data = env
    # 决策日 idx=28，days=5 → 需要 idx=33，超出 30 根 K线 → 不评估
    _record(db, _date(_UP, 28), "UP.SH", "buy", 0.9, executed=True)
    # 正常可评估的一笔
    _record(db, _date(_UP, 5), "UP.SH", "buy", 0.9, executed=True)

    ev = DecisionEvaluator(db_path=db.db_path, price_data=price_data)
    report = ev.evaluate(days=5)

    assert report["summary"]["total_decisions"] == 2
    assert report["summary"]["evaluated_decisions"] == 1
    assert len(report["recent_decisions"]) == 1


def test_missing_symbol_klines_no_crash(env, monkeypatch):
    db, price_data = env
    # 该标的不在 price_data 中；强制在线拉取失败（模拟无网络）→ 应跳过而不是抛异常
    from data.data_fetcher import DataFetcher

    def _boom(self, *args, **kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr(DataFetcher, "get_klines", _boom)

    _record(db, "2024-01-08", "NOPE.SH", "buy", 0.9, executed=True)

    ev = DecisionEvaluator(db_path=db.db_path, price_data=price_data)
    report = ev.evaluate(days=5)

    assert report["summary"]["total_decisions"] == 1
    assert report["summary"]["evaluated_decisions"] == 0


def test_recent_decisions_shape(env):
    db, price_data = env
    _record(db, _date(_UP, 5), "UP.SH", "buy", 0.9, executed=True)

    ev = DecisionEvaluator(db_path=db.db_path, price_data=price_data)
    report = ev.evaluate(days=5)

    row = report["recent_decisions"][0]
    for key in ("timestamp", "symbol", "action", "confidence",
                "future_return", "correct", "reason"):
        assert key in row
    assert row["action"] == "buy"
    assert row["correct"] is True
