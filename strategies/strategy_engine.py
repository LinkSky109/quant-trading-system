"""多策略并行引擎。

同时运行多个策略，汇总信号，支持信号加权融合。
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from strategies.base_strategy import BaseStrategy, Signal
from strategies.bollinger import BollingerStrategy
from strategies.ma_cross import MACrossStrategy
from strategies.momentum_breakout import MomentumBreakoutStrategy

logger = logging.getLogger(__name__)


class StrategyEngine:
    """多策略并行引擎。"""

    def __init__(self, strategies: List[BaseStrategy] | None = None):
        self.strategies: List[BaseStrategy] = strategies or []

    @classmethod
    def from_config(cls, strategy_config: Dict) -> "StrategyEngine":
        """从配置字典构建策略引擎。"""
        strategy_map = {
            "ma_cross": MACrossStrategy,
            "bollinger": BollingerStrategy,
            "momentum_breakout": MomentumBreakoutStrategy,
        }
        instances = []
        for name, cfg in strategy_config.items():
            if not cfg.get("enabled", True):
                continue
            cls_type = strategy_map.get(name)
            if cls_type is None:
                logger.warning("未知策略: %s", name)
                continue
            params = {k: v for k, v in cfg.items() if k != "enabled"}
            instances.append(cls_type(params))
            logger.info("已加载策略: %s", name)
        return cls(instances)

    def add_strategy(self, strategy: BaseStrategy) -> None:
        self.strategies.append(strategy)

    def run_all(self, df: pd.DataFrame, symbol: str = "") -> List[Signal]:
        """运行所有策略，返回合并后的信号列表（按日期排序）。"""
        all_signals: List[Signal] = []
        for strat in self.strategies:
            try:
                sigs = strat.generate_signals(df, symbol)
                all_signals.extend(sigs)
                logger.debug("策略 %s 产生 %d 条信号", strat.name, len(sigs))
            except Exception as e:
                logger.error("策略 %s 运行失败: %s", strat.name, e)
        all_signals.sort(key=lambda s: s.date)
        return all_signals

    def get_combined_signals(
        self, df: pd.DataFrame, symbol: str = ""
    ) -> pd.DataFrame:
        """获取多策略融合后的信号 DataFrame。

        融合规则：同一日期多策略同向则置信度取平均加权，
        方向冲突则取置信度高的一方。
        """
        combined = pd.DataFrame(index=df.index)
        combined["signal"] = 0
        combined["confidence"] = 0.0
        combined["strategy_count"] = 0

        for strat in self.strategies:
            sig_df = strat.get_signal_dataframe(df, symbol)
            for idx in sig_df.index:
                s = sig_df.loc[idx, "signal"]
                c = sig_df.loc[idx, "confidence"]
                if pd.isna(s) or s == 0:
                    continue
                cur_sig = combined.loc[idx, "signal"]
                cur_conf = combined.loc[idx, "confidence"]
                cur_cnt = combined.loc[idx, "strategy_count"]

                if cur_sig == 0:
                    combined.loc[idx, "signal"] = s
                    combined.loc[idx, "confidence"] = c
                elif cur_sig == s:
                    # 同向：加权平均
                    combined.loc[idx, "confidence"] = (
                        cur_conf * cur_cnt + c
                    ) / (cur_cnt + 1)
                else:
                    # 冲突：取置信度高的
                    if c > cur_conf:
                        combined.loc[idx, "signal"] = s
                        combined.loc[idx, "confidence"] = c
                combined.loc[idx, "strategy_count"] = cur_cnt + 1

        return combined
