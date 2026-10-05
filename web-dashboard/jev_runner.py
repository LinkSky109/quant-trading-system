#!/usr/bin/env python3
"""Jev runner 子进程：stdin/stdout JSON 行协议，懒加载 laya_mlx。

由 server.py 通过 subprocess 启动（解释器为 Jev venv 内的 python）。
协议：
  - 启动后先输出一行就绪消息：{"ok":true,"ready":true}
    若 laya_mlx 不可用，则输出 {"ok":false,"error":"laya_unavailable:..."} 后退出。
  - 从 stdin 读一行 JSON：{"action":"predict","state":{...},"questions":{...}}
  - 调用 laya.load("aac6fef/laya-mlx")（模块级缓存）+ agent.predict(state, questions)
  - 向 stdout 写一行：
      成功 {"ok":true,"probabilities":{"buy":x,"sell":x,"hold":x},"latency_ms":123.4}
      失败 {"ok":false,"error":"...","latency_ms":123.4}
  - 收到 {"action":"quit"} 退出。
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict

# 模块级缓存：laya agent 只加载一次
_AGENT: Any = None

# questions 模板（decision 选择题）
QUESTIONS_TEMPLATE: Dict[str, Any] = {
    "decision": {
        "type": "choice",
        "instructions": "基于当前市场状态，应该买入、卖出还是持有？",
        "criteria": {
            "buy": "多头趋势明确，技术指标支撑上涨，建议买入",
            "sell": "空头趋势明确，技术指标支撑下跌，建议卖出",
            "hold": "趋势不明或震荡，建议观望等待",
        },
    }
}


def _load_agent():
    """懒加载 laya_mlx agent，首次调用时加载。"""
    global _AGENT
    if _AGENT is not None:
        return _AGENT
    import laya_mlx as laya  # type: ignore

    _AGENT = laya.load("aac6fef/laya-mlx")
    return _AGENT


def _normalize_probabilities(result: Any) -> Dict[str, float]:
    """将 laya predict 返回结果归一化为 buy/sell/hold 概率分布。

    兼容多种返回形态：
      - dict 含 "probabilities" 子字典
      - dict 含 "decision" 字符串选择 → 该选项=1.0
      - dict 含 "decision" 子字典（choice 或 probabilities）
      - 直接返回字符串
    """
    probs = {"buy": 0.0, "sell": 0.0, "hold": 0.0}

    if isinstance(result, dict):
        # 顶层 probabilities
        if isinstance(result.get("probabilities"), dict):
            for k in probs:
                v = result["probabilities"].get(k, 0.0)
                try:
                    probs[k] = float(v)
                except (TypeError, ValueError):
                    probs[k] = 0.0
        else:
            decision = result.get("decision")
            if isinstance(decision, str) and decision in probs:
                probs[decision] = 1.0
            elif isinstance(decision, dict):
                if isinstance(decision.get("probabilities"), dict):
                    for k in probs:
                        v = decision["probabilities"].get(k, 0.0)
                        try:
                            probs[k] = float(v)
                        except (TypeError, ValueError):
                            probs[k] = 0.0
                elif decision.get("choice") in probs:
                    probs[decision["choice"]] = 1.0
    elif isinstance(result, str) and result in probs:
        probs[result] = 1.0

    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    else:
        probs = {"buy": 0.0, "sell": 0.0, "hold": 1.0}
    return probs


def _write(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    # 启动即尝试加载 laya（>60s 超时由 server.py 侧 readline 超时兜底）
    try:
        _load_agent()
    except Exception as e:  # noqa: BLE001
        _write({"ok": False, "error": f"laya_unavailable: {e}"})
        return 0

    _write({"ok": True, "ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        action = req.get("action")
        if action == "quit":
            break
        if action != "predict":
            continue

        state = req.get("state", {}) or {}
        questions = req.get("questions") or QUESTIONS_TEMPLATE

        t0 = time.time()
        try:
            agent = _load_agent()
            result = agent.predict(state, questions)
            latency_ms = (time.time() - t0) * 1000.0
            probs = _normalize_probabilities(result)
            _write({
                "ok": True,
                "probabilities": {k: round(v, 4) for k, v in probs.items()},
                "latency_ms": round(latency_ms, 1),
            })
        except Exception as e:  # noqa: BLE001
            latency_ms = (time.time() - t0) * 1000.0
            _write({"ok": False, "error": str(e), "latency_ms": round(latency_ms, 1)})

    return 0


if __name__ == "__main__":
    sys.exit(main())
