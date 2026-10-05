"""另类数据模块（REQ-P3-04）。

数据源类型：
  - 卫星图像衍生指标（停车场饱和度、夜间灯光）
  - 社交媒体情绪（提及量、情绪得分）
  - 供应链数据（交付履约率、库存压力）
  - ESG 评分（总分 + E/S/G 分项）

设计：
  - 所有数据源为确定性 mock 生成器（同参数同序列，回测可复现），并预留
    真实数据源接入接口（API key + 数据格式转换器）。
  - ``AlternativeFactorEngine`` 把另类数据转化为可交易因子，默认 shift(1)
    防未来函数（t 日因子值来自 t-1 日及以前的另类数据）。
  - 与 ``FactorEngine`` 集成：另类因子注册到因子注册表（category=另类）。
"""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #


def _seed(symbol: str, source: str) -> int:
    """确定性随机种子（symbol + 数据源）。"""
    key = f"{source}:{symbol}".upper()
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 10_000


# --------------------------------------------------------------------------- #
# 数据源接口（预留真实接入）
# --------------------------------------------------------------------------- #


class AlternativeDataSource(ABC):
    """另类数据源接口。

    所有数据源统一输出：DataFrame(index=日期, columns=各指标列)。
    """

    name: str = "base"
    description: str = ""

    @abstractmethod
    def fetch(self, symbol: str, days: int = 250) -> pd.DataFrame:
        """获取另类数据时间序列。"""

    def meta(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description}


class RealAlternativeDataSource(AlternativeDataSource):
    """真实另类数据源占位实现。

    预留接入点：设置 api_key 与 base_url 后，由数据格式转换器
    （``convert``，子类/后续实现）把第三方响应统一为本模块 DataFrame 格式。
    当前版本固定抛 NotImplementedError，所有调用方应降级到 mock。
    """

    def __init__(self, api_key: str = "", base_url: str = "", source_name: str = "real"):
        self.api_key = api_key
        self.base_url = base_url
        self.name = source_name

    def fetch(self, symbol: str, days: int = 250) -> pd.DataFrame:
        raise NotImplementedError(
            "真实另类数据源尚未接入；请使用对应 Mock 生成器"
        )

    @staticmethod
    def convert(payload: Any) -> pd.DataFrame:
        """数据格式转换器占位：第三方响应 -> 统一 DataFrame。"""
        raise NotImplementedError("数据格式转换器待真实数据源确定后实现")


# --------------------------------------------------------------------------- #
# Mock 生成器（确定性 + 可配置参数）
# --------------------------------------------------------------------------- #


@dataclass
class SatelliteDataGenerator(AlternativeDataSource):
    """卫星图像衍生指标 mock（停车场饱和度 / 夜间灯光指数）。"""

    name: str = "satellite"
    description: str = "卫星图像衍生：停车场饱和度、夜间灯光指数"
    base_saturation: float = 0.65        # 停车场饱和度基准
    saturation_vol: float = 0.03         # 日波动
    light_trend: float = 0.0005          # 夜间灯光日漂移

    def fetch(self, symbol: str, days: int = 250) -> pd.DataFrame:
        rng = np.random.RandomState(_seed(symbol, self.name))
        sat = np.clip(
            self.base_saturation + rng.normal(0, self.saturation_vol, size=days).cumsum() * 0.1
            + rng.normal(0, self.saturation_vol, size=days),
            0.0, 1.0,
        )
        light = 1.0 + self.light_trend * np.arange(days) + rng.normal(
            0, 0.01, size=days
        )
        # 归一化到 0~1（min-max，保持趋势形状）
        lmin, lmax = float(light.min()), float(light.max())
        light = (light - lmin) / (lmax - lmin) if lmax > lmin else np.full(days, 0.5)
        light = np.clip(light, 0.0, 1.0)
        idx = pd.date_range("2026-01-01", periods=days, freq="D")
        return pd.DataFrame(
            {"parking_lot_saturation": np.round(sat, 4),
             "night_light_index": np.round(light, 4)},
            index=idx,
        )


@dataclass
class SocialSentimentGenerator(AlternativeDataSource):
    """社交媒体情绪 mock（提及量 / 情绪得分）。"""

    name: str = "social"
    description: str = "社交媒体：提及量、情绪得分（-1~1）"
    base_sentiment: float = 0.05
    sentiment_mean_reversion: float = 0.9   # 情绪均值回复系数
    base_mentions: float = 1000.0

    def fetch(self, symbol: str, days: int = 250) -> pd.DataFrame:
        rng = np.random.RandomState(_seed(symbol, self.name))
        shocks = rng.normal(0, 0.15, size=days)
        sentiment = np.zeros(days)
        s = self.base_sentiment
        for i in range(days):
            s = self.sentiment_mean_reversion * s + (1 - self.sentiment_mean_reversion) * \
                self.base_sentiment + shocks[i] * 0.2
            sentiment[i] = np.clip(s, -1.0, 1.0)
        mentions = self.base_mentions * np.exp(rng.normal(0, 0.25, size=days))
        idx = pd.date_range("2026-01-01", periods=days, freq="D")
        return pd.DataFrame(
            {"sentiment_raw": np.round(sentiment, 4),
             "mentions_volume": np.round(mentions, 1)},
            index=idx,
        )


@dataclass
class SupplyChainGenerator(AlternativeDataSource):
    """供应链数据 mock（交付履约率 / 库存压力）。"""

    name: str = "supply_chain"
    description: str = "供应链：交付履约率、库存压力指数"
    base_delivery_rate: float = 0.92
    delivery_vol: float = 0.02
    base_inventory_pressure: float = 0.5

    def fetch(self, symbol: str, days: int = 250) -> pd.DataFrame:
        rng = np.random.RandomState(_seed(symbol, self.name))
        delivery = np.clip(
            self.base_delivery_rate + rng.normal(0, self.delivery_vol, size=days),
            0.0, 1.0,
        )
        pressure = np.clip(
            self.base_inventory_pressure + rng.normal(0, 0.08, size=days).cumsum() * 0.05,
            0.0, 1.0,
        )
        idx = pd.date_range("2026-01-01", periods=days, freq="D")
        return pd.DataFrame(
            {"supplier_delivery_rate": np.round(delivery, 4),
             "inventory_pressure": np.round(pressure, 4)},
            index=idx,
        )


@dataclass
class ESGScoreGenerator(AlternativeDataSource):
    """ESG 评分 mock（总分 0~100 + E/S/G 分项）。"""

    name: str = "esg"
    description: str = "ESG 评分：总分及环境/社会/治理分项（0~100）"
    base_score: float = 65.0
    score_vol: float = 0.6

    def fetch(self, symbol: str, days: int = 250) -> pd.DataFrame:
        rng = np.random.RandomState(_seed(symbol, self.name))
        base = self.base_score + rng.normal(0, 5.0)  # 每只标的基础分不同
        drift = rng.normal(0, self.score_vol, size=days).cumsum() * 0.1
        e = np.clip(base + drift + rng.normal(0, 4.0, size=days), 0.0, 100.0)
        s = np.clip(base + drift + rng.normal(0, 4.0, size=days), 0.0, 100.0)
        g = np.clip(base + drift + rng.normal(0, 4.0, size=days), 0.0, 100.0)
        total = np.clip((e + s + g) / 3.0, 0.0, 100.0)  # 总分 = 分项均值
        idx = pd.date_range("2026-01-01", periods=days, freq="D")
        return pd.DataFrame(
            {"esg_score": np.round(total, 2),
             "esg_e": np.round(e, 2), "esg_s": np.round(s, 2), "esg_g": np.round(g, 2)},
            index=idx,
        )


# --------------------------------------------------------------------------- #
# 聚合器
# --------------------------------------------------------------------------- #

DEFAULT_SOURCES: Dict[str, AlternativeDataSource] = {
    "satellite": SatelliteDataGenerator(),
    "social": SocialSentimentGenerator(),
    "supply_chain": SupplyChainGenerator(),
    "esg": ESGScoreGenerator(),
}


def get_source(name: str) -> AlternativeDataSource:
    """按名称取数据源；未知名称抛 KeyError。"""
    if name not in DEFAULT_SOURCES:
        raise KeyError(f"未知另类数据源: {name}，可选: {list(DEFAULT_SOURCES)}")
    return DEFAULT_SOURCES[name]


def fetch_all(symbol: str, days: int = 250) -> pd.DataFrame:
    """聚合全部另类数据源为一张表（外连接合并，按日期排序）。"""
    frames = [src.fetch(symbol, days) for src in DEFAULT_SOURCES.values()]
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="outer")
    return merged.sort_index()


# --------------------------------------------------------------------------- #
# 另类因子计算（shift(1) 防未来函数）
# --------------------------------------------------------------------------- #

ALTERNATIVE_FACTOR_NAMES: List[str] = [
    "sentiment_score",        # 情绪得分（标准化）
    "sentiment_momentum",     # 情绪动量（环比变化）
    "supply_chain_health",    # 供应链健康指数
    "esg_score",              # ESG 总分（0~1 归一）
    "esg_momentum",           # ESG 改善幅度（5 日变化）
    "satellite_activity",     # 卫星活跃度（停车场饱和度 z-score）
    "alternative_composite",  # 另类综合因子（等权 z-score 合成）
]


class AlternativeFactorEngine:
    """把另类数据转化为可交易因子。

    默认 ``lag=1``：t 日因子值由 t-1 日及以前的另类数据计算（shift(1)
    防未来函数）。设 ``lag=0`` 可查看当日对齐的原始因子。
    """

    def __init__(self, lag: int = 1) -> None:
        if lag < 0:
            raise ValueError("lag 必须 >= 0")
        self.lag = lag

    # ---------------- 原始因子（当日对齐） ----------------
    @staticmethod
    def compute_raw(altd: pd.DataFrame) -> pd.DataFrame:
        """由另类数据表计算原始因子列（不 shift）。

        Args:
            altd: ``fetch_all`` 输出的另类数据表。
        """
        out = pd.DataFrame(index=altd.index)

        # 情绪：raw 已在 [-1,1]，再做 20 日 z-score 标准化
        raw_sent = altd.get("sentiment_raw")
        if raw_sent is not None:
            mean = raw_sent.rolling(20, min_periods=5).mean()
            std = raw_sent.rolling(20, min_periods=5).std().replace(0, np.nan)
            out["sentiment_score"] = (raw_sent - mean) / std
            out["sentiment_momentum"] = raw_sent.diff(5)

        # 供应链健康：履约率 - 库存压力（两者都在 [0,1]）
        delivery = altd.get("supplier_delivery_rate")
        pressure = altd.get("inventory_pressure")
        if delivery is not None and pressure is not None:
            out["supply_chain_health"] = delivery - pressure

        # ESG：归一到 [0,1]
        esg = altd.get("esg_score")
        if esg is not None:
            out["esg_score"] = esg / 100.0
            out["esg_momentum"] = esg.diff(5) / 100.0

        # 卫星活跃度：停车场饱和度 60 日 z-score
        sat = altd.get("parking_lot_saturation")
        if sat is not None:
            mean = sat.rolling(60, min_periods=10).mean()
            std = sat.rolling(60, min_periods=10).std().replace(0, np.nan)
            out["satellite_activity"] = (sat - mean) / std

        # 综合因子：各分项 z-score 等权平均
        z_parts = []
        for col in ("sentiment_score", "supply_chain_health", "esg_score",
                    "satellite_activity"):
            if col in out.columns:
                s = out[col]
                m = s.rolling(60, min_periods=10).mean()
                sd = s.rolling(60, min_periods=10).std().replace(0, np.nan)
                z_parts.append((s - m) / sd)
        if z_parts:
            out["alternative_composite"] = pd.concat(z_parts, axis=1).mean(axis=1)

        return out

    # ---------------- 对外入口（默认 shift(1)） ----------------
    def calculate_all(self, symbol: str, days: int = 250) -> pd.DataFrame:
        """计算另类因子表（index=日期, columns=因子列，已按 lag 平移）。"""
        altd = fetch_all(symbol, days)
        raw = self.compute_raw(altd)
        if self.lag > 0:
            raw = raw.shift(self.lag)
        return raw

    def attach_to_klines(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """把另类因子按日期合并到 K 线 DataFrame（缺失日期为 NaN）。"""
        factors = self.calculate_all(symbol, days=max(len(df), 60))
        out = df.copy()
        for col in factors.columns:
            out[col] = factors[col].reindex(out.index)
        return out

    @staticmethod
    def factor_meta() -> List[Dict[str, Any]]:
        """因子元数据（供 FactorEngine 注册表使用）。"""
        return [
            {"name": "sentiment_score", "category": "另类",
             "description": "社交媒体情绪 z-score（20日，shift(1)）", "direction": +1},
            {"name": "sentiment_momentum", "category": "另类",
             "description": "情绪 5 日变化（shift(1)）", "direction": +1},
            {"name": "supply_chain_health", "category": "另类",
             "description": "供应链健康指数 = 履约率 - 库存压力（shift(1)）",
             "direction": +1},
            {"name": "esg_score", "category": "另类",
             "description": "ESG 总分归一 0~1（shift(1)）", "direction": +1},
            {"name": "esg_momentum", "category": "另类",
             "description": "ESG 5 日改善幅度（shift(1)）", "direction": +1},
            {"name": "satellite_activity", "category": "另类",
             "description": "停车场饱和度 z-score（60日，shift(1)）", "direction": +1},
            {"name": "alternative_composite", "category": "另类",
             "description": "另类综合因子（分项 z-score 等权，shift(1)）",
             "direction": +1},
        ]
