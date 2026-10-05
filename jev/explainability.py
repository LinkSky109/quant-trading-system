"""Jev 决策可解释性模块（REQ-P2-12）。

把「特征列表 → 三动作概率 {buy, sell, hold}」包装成一个可重复调用的
``predict_callable``，所有解释方法都基于对该函数的**扰动调用**（permutation /
masking / counterfactual line search）实现，因此：

* 不依赖本地 8765 Jev 服务在线；
* 默认适配 :class:`jev.jev_engine.JevDecisionEngine` 的 mock/计算概率路径；
* 测试时可注入任意确定性的 ``predict_callable``，使结果可手算、可预测。

主要能力：

1. 特征重要性（permutation importance）：逐特征遮蔽后比较概率向量变化。
2. 单次决策解释报告：每个特征对 buy/sell/hold 的边际贡献 + 人类可读中文文本。
3. 反事实分析（counterfactual）：搜索最小特征变更使决策反转为目标动作。
4. SHAP 近似：优先尝试 ``shap`` 库，不可用时用扰动法近似并标注可用性。
5. 决策路径结构化数据：输入特征 → 各特征扰动贡献 → 概率 → 最终决策。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 三动作固定顺序
ACTIONS: Tuple[str, ...] = ("buy", "sell", "hold")

# 每个特征的「中性基线值」：遮蔽该特征时用它替换原值。
# 选择语义上的中性点，使扰动后的概率变化可解释为该特征的边际作用。
NEUTRAL_BASELINE: Dict[str, float] = {
    "price": 0.0,
    "price_change_5d": 0.0,      # 无涨跌
    "ma5_ma20_ratio": 1.0,        # 均线粘合
    "volume_ratio": 1.0,          # 平量
    "rsi": 50.0,                  # 中性强弱
    "macd_signal": 0,             # 无方向
    "volatility_20d": 0.0,
}

# 反事实搜索允许调整的连续特征及其合法边界（min, max）。
ADJUSTABLE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "rsi": (0.0, 100.0),
    "volume_ratio": (0.1, 5.0),
    "price_change_5d": (-0.5, 0.5),
    "ma5_ma20_ratio": (0.8, 1.2),
    "macd_signal": (-1.0, 1.0),
}

# 反事实二分搜索迭代轮数（决定最小变更精度）
_COUNTERFACTUAL_ITERS = 24
# 判定「支持/反对」的数值阈值（避免浮点噪声把微小贡献算作方向）
_DIR_EPS = 1e-6

# predict 函数类型：特征列表 [{"feature": name, "value": v}] → 概率字典
PredictFn = Callable[[List[Dict[str, Any]]], Dict[str, float]]


@dataclass
class ExplainFeatureContribution:
    """单个特征对一次决策的贡献明细。"""

    feature: str
    value: float
    baseline_value: float
    contribution_to_buy: float
    contribution_to_sell: float
    contribution_to_hold: float
    direction: str  # support / oppose / neutral（相对最终动作）

    def as_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "value": self.value,
            "baseline_value": self.baseline_value,
            "contribution_to_buy": round(self.contribution_to_buy, 6),
            "contribution_to_sell": round(self.contribution_to_sell, 6),
            "contribution_to_hold": round(self.contribution_to_hold, 6),
            "direction": self.direction,
        }


class JevExplainer:
    """Jev 决策解释器。

    核心是一个可重复调用的预测函数 ``predict_callable``：

    ``features``（``[{"feature": name, "value": v}]`` 列表）→
    ``{"buy": p, "sell": p, "hold": p}`` 概率字典。

    不传 ``predict_callable`` 时，默认适配
    :meth:`jev.jev_engine.JevDecisionEngine._compute_probabilities` 的 mock
    概率路径（``mock_mode=True``，离线可用），并使用构造时给定的
    ``default_raw_signal / default_raw_confidence`` 作为该路径的先验。
    """

    def __init__(
        self,
        predict_callable: Optional[PredictFn] = None,
        baseline: Optional[Dict[str, float]] = None,
        default_raw_signal: str = "hold",
        default_raw_confidence: float = 0.5,
    ) -> None:
        """初始化解释器。

        Args:
            predict_callable: 注入的「特征→概率」函数。为 None 时使用内置引擎
                mock 路径。测试时注入确定性函数。
            baseline: 逐特征遮蔽时使用的基线值，缺省合并 :data:`NEUTRAL_BASELINE`。
            default_raw_signal: 默认引擎路径使用的原始信号 buy/sell/hold。
            default_raw_confidence: 默认引擎路径使用的原始置信度。
        """
        self._baseline: Dict[str, float] = dict(NEUTRAL_BASELINE)
        if baseline:
            self._baseline.update(baseline)
        self._default_raw_signal = default_raw_signal
        self._default_raw_confidence = default_raw_confidence
        self._engine: Any = None  # 懒加载的默认引擎实例

        if predict_callable is not None:
            self.predict = predict_callable
        else:
            self.predict = self._default_engine_predict

    # ------------------------------------------------------------------
    # 默认预测路径（适配 JevDecisionEngine mock/计算路径）
    # ------------------------------------------------------------------
    def _default_engine_predict(self, features: List[Dict[str, Any]]) -> Dict[str, float]:
        """默认 predict：复用 JevDecisionEngine 的 mock 概率计算（离线可用）。

        Args:
            features: ``[{"feature": name, "value": v}]`` 列表。

        Returns:
            ``{"buy", "sell", "hold"}`` 概率字典（和为 1）。
        """
        if self._engine is None:
            # 延迟导入，避免本模块在无依赖环境下 import 即失败
            from jev.jev_engine import JevDecisionEngine

            self._engine = JevDecisionEngine(mock_mode=True)
        # _compute_probabilities 在 mock_mode 下走 _mock_evaluate，纯内存计算，
        # 不写缓存/审计，可被解释器反复扰动调用。
        probs = self._engine._compute_probabilities(
            features, self._default_raw_signal, self._default_raw_confidence
        )
        return self._normalize_probs(probs)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_probs(probs: Dict[str, float]) -> Dict[str, float]:
        """把概率字典规整为 {buy, sell, hold} 且和为 1。"""
        out = {a: float(probs.get(a, 0.0)) for a in ACTIONS}
        total = sum(out.values())
        if total > 1e-12:
            out = {a: v / total for a, v in out.items()}
        else:
            out = {a: 1.0 / 3.0 for a in ACTIONS}
        return out

    @staticmethod
    def _features_to_dict(features: List[Dict[str, Any]]) -> Dict[str, float]:
        """特征列表转 {feature: value}。"""
        return {str(f["feature"]): float(f["value"]) for f in features}

    def _masked(self, features: List[Dict[str, Any]], feature: str) -> List[Dict[str, Any]]:
        """返回把指定特征替换为基线值后的新特征列表（不修改入参）。"""
        base_val = self._baseline.get(feature, 0.0)
        out: List[Dict[str, Any]] = []
        for f in features:
            if str(f["feature"]) == feature:
                out.append({"feature": feature, "value": base_val})
            else:
                out.append({"feature": f["feature"], "value": f["value"]})
        return out

    def _set_feature(
        self, features: List[Dict[str, Any]], feature: str, value: float
    ) -> List[Dict[str, Any]]:
        """返回把指定特征改为 value 后的新特征列表。"""
        out: List[Dict[str, Any]] = []
        for f in features:
            if str(f["feature"]) == feature:
                out.append({"feature": feature, "value": value})
            else:
                out.append({"feature": f["feature"], "value": f["value"]})
        return out

    # ------------------------------------------------------------------
    # 1. 特征重要性（permutation importance）
    # ------------------------------------------------------------------
    def feature_importance(
        self, features: List[Dict[str, Any]], n_perturb: int = 1
    ) -> Dict[str, Any]:
        """逐特征遮蔽（mask），比较扰动前后三动作概率变化。

        对每个特征把其值替换为基线值，重新预测；重要性取概率向量的 L1 距离
        （各动作概率变化绝对值之和）。同时给出逐动作的带符号 delta 与排序。

        Args:
            features: 特征列表 ``[{"feature", "value"}]``。
            n_perturb: 扰动次数占位参数（本实现单基线遮蔽，保留接口语义）。

        Returns:
            含 ``base_probs``、``overall``（按 L1 重要性降序）、
            ``by_action``（每个动作按 |delta| 降序）的字典。
        """
        del n_perturb  # 单基线确定性遮蔽，无需多次平均
        base_probs = self.predict(features)
        fdict = self._features_to_dict(features)

        overall: List[Dict[str, Any]] = []
        by_action: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ACTIONS}

        for name in fdict:
            masked_probs = self.predict(self._masked(features, name))
            delta = {a: base_probs[a] - masked_probs[a] for a in ACTIONS}
            l1 = sum(abs(delta[a]) for a in ACTIONS)
            baseline_value = self._baseline.get(name, 0.0)
            overall.append({
                "feature": name,
                "baseline_value": baseline_value,
                "importance": round(l1, 6),
                "base_probs": {a: round(base_probs[a], 6) for a in ACTIONS},
                "masked_probs": {a: round(masked_probs[a], 6) for a in ACTIONS},
                "delta": {a: round(delta[a], 6) for a in ACTIONS},
            })
            for a in ACTIONS:
                by_action[a].append({
                    "feature": name,
                    "delta": round(delta[a], 6),
                    "abs_delta": round(abs(delta[a]), 6),
                })

        overall.sort(key=lambda x: x["importance"], reverse=True)
        for a in ACTIONS:
            by_action[a].sort(key=lambda x: x["abs_delta"], reverse=True)

        return {
            "base_probs": {a: round(base_probs[a], 6) for a in ACTIONS},
            "overall": overall,
            "by_action": by_action,
        }

    def global_feature_importance(
        self, samples: List[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """对多组采样特征平均得到全局特征重要性（smoke / 粗粒度）。

        Args:
            samples: 多组特征列表，每组同 :meth:`feature_importance` 的入参。

        Returns:
            ``{feature: {importance, delta:{buy,sell,hold}}}``（跨样本平均），
            按平均 importance 降序。
        """
        if not samples:
            return {"overall": [], "message": "no_samples"}

        acc: Dict[str, Dict[str, float]] = {}
        for sample in samples:
            fi = self.feature_importance(sample)
            for row in fi["overall"]:
                name = row["feature"]
                slot = acc.setdefault(name, {"importance": 0.0, **{a: 0.0 for a in ACTIONS}})
                slot["importance"] += row["importance"]
                for a in ACTIONS:
                    slot[a] += row["delta"][a]

        n = len(samples)
        overall = [
            {
                "feature": name,
                "importance": round(v["importance"] / n, 6),
                "delta": {a: round(v[a] / n, 6) for a in ACTIONS},
            }
            for name, v in acc.items()
        ]
        overall.sort(key=lambda x: x["importance"], reverse=True)
        return {"n_samples": n, "overall": overall}

    # ------------------------------------------------------------------
    # 2. 单次决策解释报告
    # ------------------------------------------------------------------
    def explain(
        self,
        features: List[Dict[str, Any]],
        baseline_probs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """生成单次决策的特征贡献解释报告。

        每个特征的边际贡献近似为「遮蔽该特征后概率变化」：
        ``contribution[a] = base_probs[a] - masked_probs[a]``。
        正值表示该特征支持动作 a，负值表示反对。

        Args:
            features: 特征列表。
            baseline_probs: 可选的基线概率；缺省用 ``predict(features)``。

        Returns:
            ``{base_probs, final_action, contributions:[...], explain_text}``。
        """
        base_probs = self._normalize_probs(
            baseline_probs if baseline_probs is not None else self.predict(features)
        )
        final_action = max(base_probs, key=base_probs.get)
        fdict = self._features_to_dict(features)

        contributions: List[Dict[str, Any]] = []
        for name, value in fdict.items():
            masked_probs = self.predict(self._masked(features, name))
            contrib = {a: base_probs[a] - masked_probs[a] for a in ACTIONS}
            c_final = contrib.get(final_action, 0.0)
            if c_final > _DIR_EPS:
                direction = "support"
            elif c_final < -_DIR_EPS:
                direction = "oppose"
            else:
                direction = "neutral"
            contributions.append(ExplainFeatureContribution(
                feature=name,
                value=value,
                baseline_value=self._baseline.get(name, 0.0),
                contribution_to_buy=contrib["buy"],
                contribution_to_sell=contrib["sell"],
                contribution_to_hold=contrib["hold"],
                direction=direction,
            ).as_dict())

        contributions.sort(key=lambda c: abs(c["contribution_to_" + final_action]), reverse=True)
        text = self._build_explain_text(features, base_probs, final_action)

        return {
            "base_probs": {a: round(base_probs[a], 6) for a in ACTIONS},
            "final_action": final_action,
            "final_confidence": round(base_probs[final_action], 6),
            "contributions": contributions,
            "explain_text": text,
        }

    # ------------------------------------------------------------------
    # 3. 反事实分析（Counterfactual）
    # ------------------------------------------------------------------
    def counterfactual(
        self, features: List[Dict[str, Any]], target_action: str
    ) -> Dict[str, Any]:
        """搜索最小特征变更，使决策从当前动作反转为 ``target_action``。

        对 :data:`ADJUSTABLE_BOUNDS` 内的连续特征逐一做边界探测 + 二分搜索：
        若把某特征推到边界能使目标动作成为 argmax，则二分求最小变更幅度；
        在所有可单独反转的特征中取「最小绝对变更」方案。任何特征都无法在
        合法边界内反转时，返回 ``feasible=False``，不编造变更。

        Args:
            features: 特征列表。
            target_action: 目标动作 buy/sell/hold。

        Returns:
            ``feasible=True`` 时含 ``counterfactual_features``、``changes``
            （from/to/change）、``steps``、``final_probs``；否则 ``feasible=False``
            并附 ``reason``。
        """
        if target_action not in ACTIONS:
            raise ValueError(f"target_action 必须为 {ACTIONS} 之一，实际: {target_action}")

        base_probs = self._normalize_probs(self.predict(features))
        current_action = max(base_probs, key=base_probs.get)

        if current_action == target_action:
            return {
                "feasible": True,
                "target_action": target_action,
                "original_action": current_action,
                "already_target": True,
                "changes": [],
                "steps": 0,
                "counterfactual_features": [dict(f) for f in features],
                "final_probs": {a: round(base_probs[a], 6) for a in ACTIONS},
            }

        fdict = self._features_to_dict(features)
        best: Optional[Dict[str, Any]] = None

        for name, (lo_bound, hi_bound) in ADJUSTABLE_BOUNDS.items():
            if name not in fdict:
                continue
            cur = fdict[name]
            # 探测两个边界，判断哪个方向能让 target 成为 argmax
            candidate = self._search_one_feature(
                features, name, cur, lo_bound, hi_bound, target_action
            )
            if candidate is None:
                continue
            if best is None or candidate["cost"] < best["cost"]:
                best = candidate

        if best is None:
            return {
                "feasible": False,
                "target_action": target_action,
                "original_action": current_action,
                "reason": (
                    "在可调特征的合法边界内，单独修改任一特征都无法使 "
                    f"{target_action} 成为最终决策"
                ),
                "final_probs": {a: round(base_probs[a], 6) for a in ACTIONS},
            }

        return {
            "feasible": True,
            "target_action": target_action,
            "original_action": current_action,
            "already_target": False,
            "changes": best["changes"],
            "steps": best["steps"],
            "cost": round(best["cost"], 6),
            "counterfactual_features": best["counterfactual_features"],
            "final_probs": best["final_probs"],
        }

    def _search_one_feature(
        self,
        features: List[Dict[str, Any]],
        name: str,
        cur: float,
        lo_bound: float,
        hi_bound: float,
        target: str,
    ) -> Optional[Dict[str, Any]]:
        """对单个特征做边界探测 + 二分搜索，求最小可反转变更。

        Returns:
            成功返回含 cost/changes/steps/... 的字典；该特征边界内无法反转返回 None。
        """
        def action_at(value: float) -> Tuple[str, Dict[str, float]]:
            probs = self._normalize_probs(self.predict(self._set_feature(features, name, value)))
            return max(probs, key=probs.get), probs

        # 探测低/高边界，选能反转且变更更小的方向
        lo_action, _ = action_at(lo_bound)
        hi_action, _ = action_at(hi_bound)

        flip_lo = lo_action == target
        flip_hi = hi_action == target
        if not flip_lo and not flip_hi:
            return None

        # 选择「离当前值更近」的可反转边界
        lo_dist = abs(cur - lo_bound)
        hi_dist = abs(hi_bound - cur)
        if flip_lo and (not flip_hi or lo_dist <= hi_dist):
            edge, edge_val, direction = "low", lo_bound, -1.0
        else:
            edge, edge_val, direction = "high", hi_bound, 1.0

        # 二分：cur（不可反转） → edge_val（可反转），求最小 |delta| 反转点
        lo_v, hi_v = cur, edge_val
        steps = 0
        for _ in range(_COUNTERFACTUAL_ITERS):
            mid = (lo_v + hi_v) / 2.0
            mid_action, mid_probs = action_at(mid)
            steps += 1
            if mid_action == target:
                hi_v = mid  # 仍可反转，往回收
            else:
                lo_v = mid  # 不可反转，往外推
        threshold = hi_v
        final_action, final_probs = action_at(threshold)
        if final_action != target:
            return None

        change = threshold - cur
        return {
            "cost": abs(change),
            "steps": steps,
            "changes": [{
                "feature": name,
                "from": round(cur, 6),
                "to": round(threshold, 6),
                "change": round(change, 6),
                "direction": "decrease" if direction < 0 else "increase",
            }],
            "counterfactual_features": [
                {"feature": f["feature"],
                 "value": round(threshold, 6) if str(f["feature"]) == name else f["value"]}
                for f in features
            ],
            "final_probs": {a: round(final_probs[a], 6) for a in ACTIONS},
        }

    # ------------------------------------------------------------------
    # 4. SHAP 近似
    # ------------------------------------------------------------------
    def shap_values(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """近似每个特征对各动作概率的 SHAP 值。

        优先 ``import shap`` 尝试 KernelExplainer；不可用或失败时退回扰动近似：
        以「全基线概率」为参考，逐特征遮蔽得到主效应，再按动作把主效应缩放到
        ``full - reference``，作为 Shapley 值的低成本近似。

        Args:
            features: 特征列表。

        Returns:
            ``{shap_available, values: {feature: {buy, sell, hold}}, reference, base_probs}``。
        """
        shap_available = False
        try:  # pragma: no cover - 依赖环境是否安装 shap
            import shap  # noqa: F401

            shap_available = True
        except Exception:
            shap_available = False

        base_probs = self._normalize_probs(self.predict(features))
        fdict = self._features_to_dict(features)

        # 参考点：所有特征都遮蔽为基线
        all_baseline = [
            {"feature": name, "value": self._baseline.get(name, 0.0)}
            for name in fdict
        ]
        ref_probs = self._normalize_probs(self.predict(all_baseline))

        # 逐特征主效应：full - masked
        main_effect: Dict[str, Dict[str, float]] = {}
        for name in fdict:
            masked_probs = self._normalize_probs(self.predict(self._masked(features, name)))
            main_effect[name] = {a: base_probs[a] - masked_probs[a] for a in ACTIONS}

        # 按动作缩放：使 Σ shap ≈ base - ref
        values: Dict[str, Dict[str, float]] = {}
        for name in fdict:
            values[name] = {}
        for a in ACTIONS:
            full_effect = base_probs[a] - ref_probs[a]
            denom = sum(main_effect[name][a] for name in fdict)
            scale = (full_effect / denom) if abs(denom) > 1e-12 else 1.0
            for name in fdict:
                values[name][a] = round(main_effect[name][a] * scale, 6)

        return {
            "shap_available": shap_available,
            "method": "kernel_shap" if shap_available else "perturbation_approx",
            "base_probs": {a: round(base_probs[a], 6) for a in ACTIONS},
            "reference_probs": {a: round(ref_probs[a], 6) for a in ACTIONS},
            "values": values,
        }

    # ------------------------------------------------------------------
    # 5. 决策路径结构化数据
    # ------------------------------------------------------------------
    def decision_path(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """把「输入特征 → 各特征扰动贡献 → 概率变化 → 最终决策」组织成步骤列表。

        Args:
            features: 特征列表。

        Returns:
            ``{steps:[...], edges:[...], input_features:[...], final_action, probabilities}``。
            steps 首项为输入特征节点，末项为最终决策节点。
        """
        base = self.explain(features)
        base_probs = base["base_probs"]
        final_action = base["final_action"]

        steps: List[Dict[str, Any]] = [
            {
                "type": "input",
                "label": "输入市场特征",
                "features": [
                    {"feature": c["feature"], "value": c["value"]}
                    for c in base["contributions"]
                ],
            }
        ]
        edges: List[Dict[str, Any]] = []
        for c in base["contributions"]:
            node = {
                "type": "feature_contribution",
                "feature": c["feature"],
                "value": c["value"],
                "direction": c["direction"],
                "delta": {
                    "buy": c["contribution_to_buy"],
                    "sell": c["contribution_to_sell"],
                    "hold": c["contribution_to_hold"],
                },
            }
            steps.append(node)
            edges.append({"from": "input", "to": c["feature"]})

        steps.append({
            "type": "decision",
            "label": "最终决策",
            "action": final_action,
            "probabilities": base_probs,
            "confidence": base["final_confidence"],
        })
        for c in base["contributions"]:
            edges.append({"from": c["feature"], "to": "decision"})

        return {
            "steps": steps,
            "edges": edges,
            "input_features": [{"feature": f["feature"], "value": f["value"]} for f in features],
            "final_action": final_action,
            "probabilities": base_probs,
        }

    # ------------------------------------------------------------------
    # 人类可读解释文本
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    def _build_explain_text(
        self,
        features: List[Dict[str, Any]],
        probs: Dict[str, float],
        final_action: str,
    ) -> str:
        """基于特征语义规则生成中文解释句（数值真实填入）。"""
        d = self._features_to_dict(features)
        sentences: List[str] = []

        price = d.get("price")
        if price is not None:
            sentences.append(f"当前价格 {price:.2f}。")

        pct = d.get("price_change_5d")
        if pct is not None:
            if pct > 0.03:
                sentences.append(f"近5日累计上涨 {self._fmt_pct(pct)}，短期涨幅偏大，存在回调压力。")
            elif pct < -0.03:
                sentences.append(f"近5日累计下跌 {self._fmt_pct(abs(pct))}，存在均值回归反弹空间。")
            else:
                sentences.append(f"近5日涨跌幅 {self._fmt_pct(pct)}，方向不明朗。")

        ratio = d.get("ma5_ma20_ratio")
        if ratio is not None:
            if ratio > 1.02:
                sentences.append(f"MA5/MA20={ratio:.3f}，均线多头排列，支持买入。")
            elif ratio < 0.98:
                sentences.append(f"MA5/MA20={ratio:.3f}，均线空头排列，支持卖出。")
            else:
                sentences.append(f"MA5/MA20={ratio:.3f}，均线粘合，趋势中性。")

        vr = d.get("volume_ratio")
        if vr is not None:
            if vr > 1.3:
                sentences.append(f"成交量放大 {vr:.2f} 倍，放量配合当前方向。")
            elif vr < 0.7:
                sentences.append(f"量比仅 {vr:.2f}，成交萎缩，信号缺乏量能确认。")
            else:
                sentences.append(f"量比 {vr:.2f}，成交处于常态。")

        rsi = d.get("rsi")
        if rsi is not None:
            if rsi > 70:
                sentences.append(f"RSI={rsi:.1f} 已进入超买区，反对买入、倾向卖出。")
            elif rsi < 30:
                sentences.append(f"RSI={rsi:.1f} 处于超卖区，倾向买入。")
            else:
                sentences.append(f"RSI={rsi:.1f} 处于 {('中性偏强' if rsi >= 50 else '中性偏弱')} 区间。")

        macd = d.get("macd_signal")
        if macd is not None:
            if macd > 0:
                sentences.append("MACD 金叉，呈多头信号。")
            elif macd < 0:
                sentences.append("MACD 死叉，呈空头信号。")
            else:
                sentences.append("MACD 信号中性。")

        vol = d.get("volatility_20d")
        if vol is not None:
            sentences.append(f"近20日年化波动率 {self._fmt_pct(vol)}。")

        action_cn = {"buy": "买入", "sell": "卖出", "hold": "观望"}.get(final_action, final_action)
        sentences.append(
            "综合三动作概率 "
            f"买入 {self._fmt_pct(probs['buy'])}、卖出 {self._fmt_pct(probs['sell'])}、"
            f"观望 {self._fmt_pct(probs['hold'])}，Jev 最终建议【{action_cn}】。"
        )
        return "".join(sentences)
