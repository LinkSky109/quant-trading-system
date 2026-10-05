"""财务数据接口模块。

提供 A股/美股/港股的财务数据获取接口。
当前为Mock实现 + 真实数据源接口预留，支持PE/PB/PS/ROE等核心指标。

当 REQ-P0-04 的真实数据源就绪后，切换 provider 即可。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from data.data_fetcher import normalize_symbol

logger = logging.getLogger(__name__)

# 缓存目录
_CACHE_DIR = Path(__file__).parent / "fundamental_cache"
_CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class FundamentalData:
    """单只标的的财务数据快照。"""

    symbol: str
    date: str

    # 估值指标
    pe_ttm: Optional[float] = None
    pb: Optional[float] = None
    ps_ttm: Optional[float] = None
    dividend_yield: Optional[float] = None
    ev_ebitda: Optional[float] = None

    # 盈利能力
    roe: Optional[float] = None
    roa: Optional[float] = None
    gross_margin: Optional[float] = None
    net_margin: Optional[float] = None

    # 成长能力
    revenue_growth_yoy: Optional[float] = None
    profit_growth_yoy: Optional[float] = None

    # 偿债能力
    debt_to_asset: Optional[float] = None
    current_ratio: Optional[float] = None

    # 运营效率
    inventory_turnover: Optional[float] = None
    receivable_turnover: Optional[float] = None

    # 数据来源标记
    source: str = "mock"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FundamentalProvider:
    """财务数据提供者基类。"""

    def fetch(self, symbol: str) -> FundamentalData:
        raise NotImplementedError

    def fetch_batch(self, symbols: List[str]) -> Dict[str, FundamentalData]:
        return {s: self.fetch(s) for s in symbols}


class MockFundamentalProvider(FundamentalProvider):
    """Mock财务数据提供者。

    生成基于标的代码哈希的确定性Mock数据，保证同一标的返回一致值。
    """

    # 行业基准映射 (简化)
    SECTOR_PROFILES = {
        "SH": {"pe_avg": 15.0, "pb_avg": 1.5, "roe_avg": 0.12, "margin_avg": 0.08},
        "SZ": {"pe_avg": 25.0, "pb_avg": 2.5, "roe_avg": 0.10, "margin_avg": 0.06},
        "HK": {"pe_avg": 10.0, "pb_avg": 1.0, "roe_avg": 0.08, "margin_avg": 0.15},
        "US": {"pe_avg": 30.0, "pb_avg": 5.0, "roe_avg": 0.18, "margin_avg": 0.20},
    }

    def _hash_seed(self, symbol: str) -> int:
        return int(hashlib.md5(symbol.encode()).hexdigest(), 16)

    def _mock_value(self, seed: int, base: float, std: float, min_val: float, max_val: float) -> float:
        import hashlib
        import random
        rng = random.Random(seed)
        val = rng.gauss(base, std)
        return round(max(min_val, min(max_val, val)), 4)

    def fetch(self, symbol: str) -> FundamentalData:
        import hashlib
        import random
        norm = normalize_symbol(symbol)
        suffix = norm.split(".")[-1] if "." in norm else "SH"
        seed = self._hash_seed(norm)
        rng = random.Random(seed)

        profile = self.SECTOR_PROFILES.get(suffix, self.SECTOR_PROFILES["SH"])

        date_str = pd.Timestamp.now().strftime("%Y-%m-%d")

        return FundamentalData(
            symbol=norm,
            date=date_str,
            pe_ttm=self._mock_value(seed, profile["pe_avg"], profile["pe_avg"] * 0.3, 3.0, 100.0),
            pb=self._mock_value(seed + 1, profile["pb_avg"], profile["pb_avg"] * 0.3, 0.5, 15.0),
            ps_ttm=self._mock_value(seed + 2, profile["pe_avg"] * 0.5, 2.0, 0.3, 30.0),
            dividend_yield=self._mock_value(seed + 3, 0.02, 0.01, 0.0, 0.15),
            ev_ebitda=self._mock_value(seed + 4, profile["pe_avg"] * 0.8, 5.0, 1.0, 50.0),
            roe=self._mock_value(seed + 5, profile["roe_avg"], 0.05, -0.2, 0.5),
            roa=self._mock_value(seed + 6, profile["roe_avg"] * 0.5, 0.03, -0.1, 0.3),
            gross_margin=self._mock_value(seed + 7, profile["margin_avg"], 0.05, 0.0, 0.9),
            net_margin=self._mock_value(seed + 8, profile["margin_avg"] * 0.6, 0.04, -0.3, 0.7),
            revenue_growth_yoy=self._mock_value(seed + 9, 0.15, 0.20, -0.5, 2.0),
            profit_growth_yoy=self._mock_value(seed + 10, 0.12, 0.25, -1.0, 3.0),
            debt_to_asset=self._mock_value(seed + 11, 0.45, 0.15, 0.05, 0.95),
            current_ratio=self._mock_value(seed + 12, 1.5, 0.5, 0.3, 5.0),
            source="mock",
        )


class CachedFundamentalProvider(FundamentalProvider):
    """带本地缓存的财务数据提供者。"""

    def __init__(self, provider: FundamentalProvider, cache_ttl_hours: int = 24):
        self.provider = provider
        self.cache_ttl = cache_ttl_hours * 3600

    def _cache_path(self, symbol: str) -> Path:
        return _CACHE_DIR / f"{symbol.replace('.', '_')}.json"

    def _is_cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < self.cache_ttl

    def fetch(self, symbol: str) -> FundamentalData:
        cache_path = self._cache_path(symbol)
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return FundamentalData(**data)
            except Exception:
                pass

        result = self.provider.fetch(symbol)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("缓存写入失败 %s: %s", symbol, e)
        return result

    def fetch_batch(self, symbols: List[str]) -> Dict[str, FundamentalData]:
        return {s: self.fetch(s) for s in symbols}


class FundamentalDataService:
    """财务数据服务入口。"""

    def __init__(self, provider: Optional[FundamentalProvider] = None):
        self.provider = provider or CachedFundamentalProvider(MockFundamentalProvider())

    def get(self, symbol: str) -> FundamentalData:
        return self.provider.fetch(symbol)

    def get_batch(self, symbols: List[str]) -> Dict[str, FundamentalData]:
        return self.provider.fetch_batch(symbols)

    def to_dataframe(self, symbols: List[str]) -> pd.DataFrame:
        data = self.get_batch(symbols)
        rows = [d.to_dict() for d in data.values()]
        return pd.DataFrame(rows)


# 全局默认实例
_default_service: Optional[FundamentalDataService] = None


def get_fundamental_service() -> FundamentalDataService:
    global _default_service
    if _default_service is None:
        _default_service = FundamentalDataService()
    return _default_service
