"""灰度发布与A/B测试框架。

支持:
1. 策略版本管理 (参数+代码hash)
2. 灰度运行: 小资金/小比例测试新策略
3. A/B测试: 新旧策略并行, 统计显著性检验
4. 自动回滚: 新策略表现差于阈值自动切回
5. 实验报告生成
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyVersion:
    """策略版本信息。"""
    name: str
    params: Dict[str, Any]
    code_hash: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def compute_hash(self) -> str:
        """计算版本hash。"""
        content = json.dumps({"name": self.name, "params": self.params}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ExperimentConfig:
    """实验配置。"""
    experiment_id: str
    control_version: StrategyVersion
    treatment_version: StrategyVersion
    traffic_split: float = 0.5  # 流量分配给新策略的比例
    capital_split: float = 0.2  # 资金分配给新策略的比例
    min_samples: int = 30       # 最小样本量
    rollback_threshold: float = -0.05  # 回撤超此阈值自动回滚
    significance_level: float = 0.05
    duration_days: int = 30


@dataclass
class ExperimentResult:
    """实验结果。"""
    experiment_id: str
    control_return: float
    treatment_return: float
    control_sharpe: float
    treatment_sharpe: float
    control_drawdown: float
    treatment_drawdown: float
    control_trades: int
    treatment_trades: int
    p_value: float
    is_significant: bool
    recommendation: str  # "promote" | "rollback" | "continue"
    sample_size: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ABTestingFramework:
    """A/B测试框架。"""

    def __init__(self):
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: Dict[str, List[ExperimentResult]] = {}
        self._active_experiments: Dict[str, bool] = {}

    def register_experiment(self, config: ExperimentConfig) -> str:
        """注册新实验。"""
        self.experiments[config.experiment_id] = config
        self.results[config.experiment_id] = []
        self._active_experiments[config.experiment_id] = True
        logger.info("实验已注册: %s (treatment=%s, split=%.0f%%)",
                    config.experiment_id, config.treatment_version.name,
                    config.traffic_split * 100)
        return config.experiment_id

    def stop_experiment(self, experiment_id: str) -> bool:
        """停止实验。"""
        if experiment_id in self._active_experiments:
            self._active_experiments[experiment_id] = False
            logger.info("实验已停止: %s", experiment_id)
            return True
        return False

    def is_active(self, experiment_id: str) -> bool:
        return self._active_experiments.get(experiment_id, False)

    def route(self, experiment_id: str, symbol: str) -> str:
        """根据实验配置决定路由到哪个版本。

        Returns:
            "control" 或 "treatment"
        """
        config = self.experiments.get(experiment_id)
        if not config or not self.is_active(experiment_id):
            return "control"

        # 基于symbol做确定性路由, 保证同一标的始终分配到同一版本
        hash_val = int(hashlib.md5(f"{experiment_id}:{symbol}".encode()).hexdigest(), 16)
        if (hash_val % 1000) / 1000.0 < config.traffic_split:
            return "treatment"
        return "control"

    def evaluate(self, experiment_id: str,
                 control_returns: List[float],
                 treatment_returns: List[float]) -> ExperimentResult:
        """评估实验结果并给出建议。"""
        from scipy import stats

        config = self.experiments.get(experiment_id)
        if not config:
            raise ValueError(f"实验不存在: {experiment_id}")

        n_control = len(control_returns)
        n_treatment = len(treatment_returns)

        if n_control < 2 or n_treatment < 2:
            return ExperimentResult(
                experiment_id=experiment_id,
                control_return=0.0, treatment_return=0.0,
                control_sharpe=0.0, treatment_sharpe=0.0,
                control_drawdown=0.0, treatment_drawdown=0.0,
                control_trades=n_control, treatment_trades=n_treatment,
                p_value=1.0, is_significant=False,
                recommendation="continue", sample_size=n_control + n_treatment,
            )

        c_ret = float(np.mean(control_returns))
        t_ret = float(np.mean(treatment_returns))
        c_sharpe = float(np.mean(control_returns) / (np.std(control_returns) + 1e-9) * np.sqrt(252))
        t_sharpe = float(np.mean(treatment_returns) / (np.std(treatment_returns) + 1e-9) * np.sqrt(252))
        c_dd = float(self._max_drawdown(control_returns))
        t_dd = float(self._max_drawdown(treatment_returns))

        # t-test
        _, p_value = stats.ttest_ind(treatment_returns, control_returns, equal_var=False)
        p_value = float(p_value) if not np.isnan(p_value) else 1.0

        is_sig = p_value < config.significance_level and n_treatment >= config.min_samples

        # 决策逻辑
        if t_dd < config.rollback_threshold:
            recommendation = "rollback"
        elif is_sig and t_ret > c_ret and t_sharpe > c_sharpe:
            recommendation = "promote"
        else:
            recommendation = "continue"

        result = ExperimentResult(
            experiment_id=experiment_id,
            control_return=c_ret,
            treatment_return=t_ret,
            control_sharpe=c_sharpe,
            treatment_sharpe=t_sharpe,
            control_drawdown=c_dd,
            treatment_drawdown=t_dd,
            control_trades=n_control,
            treatment_trades=n_treatment,
            p_value=p_value,
            is_significant=is_sig,
            recommendation=recommendation,
            sample_size=n_control + n_treatment,
        )
        self.results[experiment_id].append(result)
        return result

    @staticmethod
    def _max_drawdown(returns: List[float]) -> float:
        """计算最大回撤。"""
        if not returns:
            return 0.0
        cum = np.cumsum(returns)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        return float(dd.min())

    def get_report(self, experiment_id: str) -> Dict[str, Any]:
        """生成实验报告。"""
        config = self.experiments.get(experiment_id)
        results = self.results.get(experiment_id, [])
        if not config:
            return {"error": "实验不存在"}

        latest = results[-1] if results else None
        return {
            "experiment_id": experiment_id,
            "active": self.is_active(experiment_id),
            "control_version": asdict(config.control_version),
            "treatment_version": asdict(config.treatment_version),
            "traffic_split": config.traffic_split,
            "capital_split": config.capital_split,
            "evaluation_count": len(results),
            "latest_result": latest.to_dict() if latest else None,
            "all_results": [r.to_dict() for r in results],
        }

    def list_experiments(self) -> List[Dict[str, Any]]:
        """列出所有实验。"""
        return [
            {
                "experiment_id": eid,
                "active": self.is_active(eid),
                "control": e.control_version.name,
                "treatment": e.treatment_version.name,
            }
            for eid, e in self.experiments.items()
        ]


# 全局实例
_default_ab_framework: Optional[ABTestingFramework] = None


def get_ab_framework() -> ABTestingFramework:
    global _default_ab_framework
    if _default_ab_framework is None:
        _default_ab_framework = ABTestingFramework()
    return _default_ab_framework
