"""回测走查器（Backtest Walkthrough）。

包装 :class:`BacktestEngine`，在回测过程中逐日记录完整快照，
用于复盘"策略为什么买/卖"：

- 每日信号（symbol / signal / confidence）
- Jev 决策（概率分布 / 最终动作 / 是否执行 / 拦截原因）
- 风控拦截与止损/止盈
- 实际成交（成交价 / 股数）
- 日末持仓 / 现金 / 总权益 / 当日盈亏 / 累计盈亏

走查结果同时驻留内存缓存与 SQLite（``walkthroughs`` 表），
支持按日 / 区间 / 成交 / 信号-操作对比等查询。

设计原则：
- 走查器本身不依赖网络；行情数据由 ``run(data)`` 传入。
- 测试可用构造的 mock DataFrame 离线运行。
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.engine import BacktestEngine
from jev.jev_engine import JevDecisionEngine

logger = logging.getLogger(__name__)

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class BacktestWalkthrough:
    """回测走查器：包装 BacktestEngine，逐日记录快照并提供查询接口。

    Attributes:
        symbol: 标的代码。
        strategy_name: 策略名称。
        start_date / end_date: 回测区间（YYYY-MM-DD）。
        initial_capital: 初始资金。
        use_jev: 是否启用 Jev 信号过滤。
    """

    # 类级内存缓存：{walkthrough_id: 结果字典}
    _cache: Dict[str, Dict[str, Any]] = {}

    # 策略工厂映射（与 web-dashboard/server.py 的 _strategy_instance 保持一致）
    _STRATEGY_CLASS_MAP: Dict[str, type] = {}

    def __init__(
        self,
        symbol: str,
        strategy_name: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000.0,
        use_jev: bool = False,
        db_path: Optional[str] = None,
    ):
        """初始化走查器。

        Args:
            symbol: 标的代码。
            strategy_name: 策略名称（ma_cross / bollinger / momentum_breakout
                / rsi / macd / grid_trading）。
            start_date: 回测开始日期 YYYY-MM-DD。
            end_date: 回测结束日期 YYYY-MM-DD。
            initial_capital: 初始资金。
            use_jev: 是否启用 Jev 信号过滤（mock 模式）。
            db_path: SQLite 路径；传入则持久化走查结果，None 仅存内存。
        """
        self.symbol = symbol
        self.strategy_name = strategy_name
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.use_jev = use_jev

        self.walkthrough_id: Optional[str] = None
        self.snapshots: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []

        # 可选 SQLite 持久化
        self._db = None
        if db_path:
            from persistence.database import Database
            self._db = Database(db_path)

    # ------------------------------------------------------------------
    # 策略工厂
    # ------------------------------------------------------------------

    @classmethod
    def _build_strategy(cls, strategy_name: str):
        """根据名称实例化策略（参考 server.py 的 _strategy_instance）。

        优先从 config.yaml 的 strategies.<name>.params 读取参数；
        无配置时使用策略默认参数（与 server.py 行为一致）。
        """
        if not cls._STRATEGY_CLASS_MAP:
            from strategies.ma_cross import MACrossStrategy
            from strategies.bollinger import BollingerStrategy
            from strategies.momentum_breakout import MomentumBreakoutStrategy
            from strategies.rsi import RSIStrategy
            from strategies.macd import MACDStrategy
            from strategies.grid_trading import GridTradingStrategy
            cls._STRATEGY_CLASS_MAP.update({
                "ma_cross": MACrossStrategy,
                "bollinger": BollingerStrategy,
                "momentum_breakout": MomentumBreakoutStrategy,
                "rsi": RSIStrategy,
                "macd": MACDStrategy,
                "grid_trading": GridTradingStrategy,
            })

        strategy_cls = cls._STRATEGY_CLASS_MAP.get(strategy_name)
        if strategy_cls is None:
            raise ValueError(
                f"未知策略: {strategy_name}，可选: {list(cls._STRATEGY_CLASS_MAP)}"
            )

        # 尝试从 config.yaml 读取策略参数（容错：读不到则用默认）
        params: Optional[Dict[str, Any]] = None
        try:
            import yaml
            cfg_path = _PROJECT_ROOT / "config" / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                strat_cfg = (cfg.get("strategies", {}) or {}).get(strategy_name, {}) or {}
                if isinstance(strat_cfg, dict) and strat_cfg.get("params"):
                    params = dict(strat_cfg["params"])
        except Exception as e:  # pragma: no cover - 配置读取失败不阻断回测
            logger.debug("读取策略参数失败，使用默认: %s", e)

        return strategy_cls(params) if params else strategy_cls()

    # ------------------------------------------------------------------
    # 运行
    # ------------------------------------------------------------------

    def run(self, data: pd.DataFrame) -> str:
        """执行带走查的回测。

        Args:
            data: 单标的 K 线 DataFrame（index 为交易日，含 open/high/low/close/volume）。

        Returns:
            walkthrough_id（uuid4 短码）。
        """
        strategy = self._build_strategy(self.strategy_name)
        jev_engine = JevDecisionEngine(mock_mode=True) if self.use_jev else None

        # 走查快照列表由引擎逐日填充
        self.snapshots = []
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            jev_engine=jev_engine,
            walkthrough_snapshots=self.snapshots,
        )
        result = engine.run(data, strategy, symbol=self.symbol)

        # 序列化成交明细
        self.trades = [
            {
                "date": t.date.strftime("%Y-%m-%d"),
                "symbol": t.symbol,
                "action": t.action,
                "price": round(float(t.price), 4),
                "shares": int(t.shares),
                "amount": round(float(t.amount), 2),
                "commission": round(float(t.commission), 2),
                "stamp_tax": round(float(t.stamp_tax), 2),
                "pnl": round(float(t.pnl), 2) if t.pnl is not None else None,
                "reason": t.reason,
            }
            for t in result.trades
        ]

        self.walkthrough_id = uuid.uuid4().hex[:12]
        created_at = datetime.now().isoformat()

        meta = {
            "symbol": self.symbol,
            "strategy": self.strategy_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "use_jev": self.use_jev,
            "snapshots_count": len(self.snapshots),
            "trades_count": len(self.trades),
            "metrics": result.metrics,
        }

        # 写内存缓存
        self.__class__._cache[self.walkthrough_id] = {
            "id": self.walkthrough_id,
            "created_at": created_at,
            "symbol": self.symbol,
            "strategy": self.strategy_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "snapshots": self.snapshots,
            "trades": self.trades,
            "meta": meta,
        }

        # 写 SQLite（可选）
        if self._db is not None:
            self._db.insert_walkthrough(
                id=self.walkthrough_id,
                created_at=created_at,
                symbol=self.symbol,
                strategy=self.strategy_name,
                start_date=self.start_date,
                end_date=self.end_date,
                snapshots=self.snapshots,
                trades=self.trades,
                meta=meta,
            )

        logger.info(
            "走查完成: id=%s 快照=%d 成交=%d",
            self.walkthrough_id, len(self.snapshots), len(self.trades),
        )
        return self.walkthrough_id

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get_day(self, date: str) -> Optional[Dict[str, Any]]:
        """返回某日完整快照。

        Args:
            date: YYYY-MM-DD。

        Returns:
            该日快照字典，不存在返回 None。
        """
        for snap in self.snapshots:
            if snap["date"] == date:
                return snap
        return None

    def get_range(self, start: str, end: str) -> List[Dict[str, Any]]:
        """返回 [start, end] 闭区间内的所有日快照（含两端）。"""
        return [
            snap for snap in self.snapshots
            if start <= snap["date"] <= end
        ]

    def get_trades(self) -> List[Dict[str, Any]]:
        """返回所有成交日的详细交易信息。"""
        return list(self.trades)

    def get_signal_vs_action(self) -> List[Dict[str, Any]]:
        """信号 vs 实际操作对比，识别被 Jev 过滤 / 风控拦截的信号。

        Returns:
            每日每条信号一行：是否执行、被谁拦截、拦截原因、成交价/股数。
        """
        rows: List[Dict[str, Any]] = []
        for snap in self.snapshots:
            for sig in snap.get("signals", []):
                rows.append({
                    "date": snap["date"],
                    "symbol": sig.get("symbol"),
                    "signal": sig.get("signal"),
                    "confidence": sig.get("confidence"),
                    "executed": sig.get("executed", False),
                    "jev_filtered": sig.get("jev_filtered", False),
                    "risk_blocked": sig.get("risk_blocked", False),
                    "jev_final_action":
                        (sig.get("jev_decision") or {}).get("final_action"),
                    "jev_reason":
                        (sig.get("jev_decision") or {}).get("reason"),
                    "block_reason": sig.get("reason", ""),
                    "fill_price": sig.get("fill_price"),
                    "shares": sig.get("shares"),
                })
        return rows

    def to_dict(self) -> Dict[str, Any]:
        """完整走查结果（元信息 + 快照数 + 交易数）。"""
        return {
            "id": self.walkthrough_id,
            "symbol": self.symbol,
            "strategy": self.strategy_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "use_jev": self.use_jev,
            "snapshots_count": len(self.snapshots),
            "trades_count": len(self.trades),
        }

    # ------------------------------------------------------------------
    # 类级缓存查询
    # ------------------------------------------------------------------

    @classmethod
    def from_cache(cls, walkthrough_id: str) -> Optional[Dict[str, Any]]:
        """从内存缓存取回完整走查结果。"""
        return cls._cache.get(walkthrough_id)

    @classmethod
    def list_cache(cls) -> List[str]:
        """列出所有内存缓存中的 walkthrough_id。"""
        return list(cls._cache.keys())
