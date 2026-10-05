"""数据获取模块：基于 QuantDash SDK，支持 A股/美股/港股 K线、实时行情、五档盘口。

包含本地磁盘缓存机制，避免重复请求。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from trading.market_config import get_currency as _market_get_currency

logger = logging.getLogger(__name__)

# 尝试导入 QuantDash SDK，未安装时降级为 mock
try:
    from quantdash import QuantDash  # type: ignore

    _QUANTDASH_AVAILABLE = True
except ImportError:
    _QUANTDASH_AVAILABLE = False
    logger.warning("QuantDash SDK 未安装，数据获取将使用 mock 模式。")


# ---------------------------------------------------------------------------
# 标的代码格式统一
# ---------------------------------------------------------------------------

def normalize_symbol(symbol: str) -> str:
    """将标的代码统一为 交易所后缀 格式。

    支持输入:
        - 600519 / 600519.SH / sh600519 -> 600519.SH
        - 000001 / 000001.SZ / sz000001 -> 000001.SZ
        - AAPL / AAPL.US -> AAPL.US
        - 00700 / 00700.HK -> 00700.HK

    Args:
        symbol: 原始标的代码。

    Returns:
        标准化后的代码。
    """
    s = symbol.strip().upper()

    # 已带后缀
    if "." in s:
        return s

    # 带交易所前缀如 sh600519
    if s.startswith(("SH", "SZ", "HK")) and len(s) > 2 and s[2:].isdigit():
        prefix = s[:2]
        code = s[2:]
        return f"{code}.{prefix}"

    # 纯数字判断市场
    if s.isdigit():
        if len(s) == 6:
            if s.startswith(("6", "9")):
                return f"{s}.SH"
            elif s.startswith(("0", "3")):
                return f"{s}.SZ"
            elif s.startswith(("4", "8")):
                return f"{s}.BJ"
        elif len(s) == 5:
            return f"{s}.HK"

    # 字母代码默认美股
    if s.isalpha():
        return f"{s}.US"

    return s


def get_market(symbol: str) -> str:
    """根据标准化代码判断市场。"""
    norm = normalize_symbol(symbol)
    suffix = norm.split(".")[-1]
    mapping = {"SH": "A股", "SZ": "A股", "BJ": "A股", "US": "美股", "HK": "港股"}
    return mapping.get(suffix, "未知")


# 腾讯财经接口的交易所前缀映射
_TENCENT_PREFIX = {"SH": "sh", "SZ": "sz", "BJ": "bj", "US": "us", "HK": "hk"}


def to_tencent_format(symbol: str) -> str:
    """将标准化代码转为腾讯财经接口格式。

    示例:
        - ``600519.SH`` -> ``sh600519``
        - ``000001.SZ`` -> ``sz000001``
        - ``AAPL.US``   -> ``usAAPL``
        - ``00700.HK``  -> ``hk00700``（港股代码自动补零至 5 位）

    Args:
        symbol: 标的代码（任意写法，自动标准化）。

    Returns:
        腾讯财经接口使用的代码字符串。
    """
    norm = normalize_symbol(symbol)
    code, _, suffix = norm.partition(".")
    prefix = _TENCENT_PREFIX.get(suffix, "sh")
    if suffix == "HK":
        code = code.zfill(5)
    return f"{prefix}{code}"


def get_currency(symbol: str) -> str:
    """返回标的计价货币代码。

    Returns:
        ``"CNY"`` / ``"USD"`` / ``"HKD"``。
    """
    return _market_get_currency(symbol)


def _default_mock_base_price(symbol: str) -> float:
    """根据市场为 mock K线选择合理的基准价。

    A股默认 1680.0（保持历史行为），美股 ~180，港股 ~380。
    """
    norm = normalize_symbol(symbol)
    suffix = norm.split(".")[-1]
    if suffix == "US":
        return 180.0
    if suffix == "HK":
        return 380.0
    return 1680.0


def _fetch_tencent_realtime(symbol: str) -> Optional[Dict[str, Any]]:
    """通过腾讯财经 HTTP 接口获取实时行情快照（美股/港股降级通道）。

    接口: ``https://qt.gtimg.cn/q=usAAPL`` / ``q=hk00700``。
    返回内容为 ``~`` 分隔的字符串，本函数做容错解析。

    Returns:
        标准化行情字典；网络失败或解析失败返回 None。
    """
    tc = to_tencent_format(symbol)
    url = f"https://qt.gtimg.cn/q={tc}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("gbk", errors="ignore")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        logger.warning("腾讯行情请求失败(%s): %s", tc, e)
        return None

    # 形如: v_usAAPL="1~Apple~AAPL~180.0~179.0~...";
    try:
        payload = raw.split("=", 1)[1].strip().strip(";").strip('"')
        f = payload.split("~")
        if len(f) < 6:
            return None
        last_price = float(f[3]) if f[3] else 0.0
        prev_close = float(f[4]) if f[4] else 0.0
        open_ = float(f[5]) if f[5] else 0.0
        # high/low/volume 字段位置在美股/港股间略有差异，做容错取值
        high = float(f[33]) if len(f) > 33 and f[33] else max(open_, last_price)
        low = float(f[34]) if len(f) > 34 and f[34] else min(open_, last_price)
        volume = int(float(f[6])) if len(f) > 6 and f[6] else 0
        norm = normalize_symbol(symbol)
        return {
            "symbol": norm,
            "name": f[1],
            "last_price": last_price,
            "open": open_,
            "high": high,
            "low": low,
            "prev_close": prev_close,
            "volume": volume,
            "amount": float(last_price) * volume,
            "currency": get_currency(norm),
            "source": "tencent",
            "timestamp": pd.Timestamp.now().isoformat(),
        }
    except (ValueError, IndexError) as e:
        logger.warning("腾讯行情解析失败(%s): %s", tc, e)
        return None


# ---------------------------------------------------------------------------
# 缓存工具
# ---------------------------------------------------------------------------

class DataCache:
    """基于本地文件的简易数据缓存。"""

    def __init__(self, cache_dir: str = "./cache", ttl_hours: float = 4.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600

    def _key(self, **kwargs: Any) -> str:
        raw = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, **kwargs: Any) -> Optional[pd.DataFrame]:
        """读取缓存，过期或不存在返回 None。"""
        key = self._key(**kwargs)
        path = self.cache_dir / f"{key}.parquet"
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.ttl_seconds:
            return None
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning("缓存读取失败: %s", e)
            return None

    def set(self, df: pd.DataFrame, **kwargs: Any) -> None:
        """写入缓存。"""
        key = self._key(**kwargs)
        path = self.cache_dir / f"{key}.parquet"
        try:
            df.to_parquet(path)
        except Exception as e:
            logger.warning("缓存写入失败: %s", e)


# ---------------------------------------------------------------------------
# Mock 数据生成（无 QuantDash 时使用）
# ---------------------------------------------------------------------------

def _generate_mock_klines(
    symbol: str,
    period: str = "1d",
    count: int = 300,
    start_date: str = "2024-01-02",
    base_price: float = 1680.0,
) -> pd.DataFrame:
    """生成模拟 K线数据，用于无 API Key 时的回测演示。

    使用几何布朗运动 + 趋势项，价格行为接近真实 A 股蓝筹。
    """
    import numpy as np

    np.random.seed(hash(normalize_symbol(symbol)) % (2**31))

    dates = pd.bdate_range(start=start_date, periods=count)
    # 年化波动率 25%，年化漂移 5%
    daily_vol = 0.25 / np.sqrt(252)
    daily_drift = 0.05 / 252

    returns = np.random.normal(loc=daily_drift, scale=daily_vol, size=count)
    # 叠加一个缓慢趋势
    trend = np.linspace(0, 0.08, count)
    returns = returns + np.diff(trend, prepend=trend[0])

    close = base_price * np.cumprod(1 + returns)
    open_ = close * (1 + np.random.normal(0, 0.008, count))
    # 确保 high >= max(open, close), low <= min(open, close)
    upper = np.maximum(open_, close)
    lower = np.minimum(open_, close)
    high = upper * (1 + np.abs(np.random.normal(0, 0.012, count)))
    low = lower * (1 - np.abs(np.random.normal(0, 0.012, count)))
    volume = np.random.lognormal(mean=15, sigma=0.5, size=count).astype(int)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": close.round(2),
            "volume": volume,
            "amount": (close * volume).round(2),
        }
    )
    df.set_index("date", inplace=True)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# 主数据获取类
# ---------------------------------------------------------------------------

class DataFetcher:
    """统一数据获取接口，封装 QuantDash SDK 与缓存。"""

    def __init__(
        self,
        api_key: str = "",
        cache_dir: str = "./cache",
        cache_ttl_hours: float = 4.0,
        use_mock: bool = False,
    ):
        self.api_key = api_key
        self.cache = DataCache(cache_dir=cache_dir, ttl_hours=cache_ttl_hours)
        self.use_mock = use_mock or (not _QUANTDASH_AVAILABLE) or (not api_key)

        if self.use_mock:
            logger.info("DataFetcher 运行在 mock 模式。")
            self._client = None
        else:
            self._client = QuantDash(api_key=api_key)
            logger.info("DataFetcher 已连接 QuantDash。")

    # -- K线 ---------------------------------------------------------------

    def get_klines(
        self,
        symbol: str,
        period: str = "1d",
        count: int = 300,
        adjust: str = "qfq",
        start_date: str | None = None,
        end_date: str | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """获取 K线数据。

        Args:
            symbol: 标的代码（自动标准化）。
            period: 周期，1d / 1m / 5m / 15m / 30m / 60m。
            count: K线数量。
            adjust: 复权方式 qfq(前复权) / hfq(后复权) / none(不复权)。
            start_date: 起始日期 YYYY-MM-DD。
            end_date: 结束日期 YYYY-MM-DD。
            use_cache: 是否使用本地缓存。

        Returns:
            包含 open/high/low/close/volume 的 DataFrame，索引为日期。
        """
        norm = normalize_symbol(symbol)
        cache_kwargs = dict(
            symbol=norm, period=period, count=count, adjust=adjust,
            start_date=start_date, end_date=end_date,
        )

        if use_cache:
            cached = self.cache.get(**cache_kwargs)
            if cached is not None and not cached.empty:
                logger.debug("缓存命中: %s %s", norm, period)
                return cached

        if self.use_mock:
            df = _generate_mock_klines(
                norm, period=period, count=count,
                start_date=start_date or "2024-01-02",
                base_price=_default_mock_base_price(norm),
            )
        else:
            # QuantDash adjust 参数映射: qfq→forward, hfq→backward
            qd_adjust = {"qfq": "forward", "hfq": "backward", "none": "none"}.get(adjust, "forward")
            df = self._client.klines.get(
                norm, period=period, count=count,
                adjust=qd_adjust, to_dataframe=True,
            )
            df = self._standardize_columns(df)

        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        if use_cache and not df.empty:
            self.cache.set(df, **cache_kwargs)

        return df

    # -- 实时行情 -----------------------------------------------------------

    def get_realtime_quote(self, symbol: str) -> Dict[str, Any]:
        """获取实时行情快照。

        Returns:
            包含 last_price, open, high, low, prev_close, volume, amount 等。
        """
        norm = normalize_symbol(symbol)
        if self.use_mock or self._client is None:
            return self._quote_from_recent_kline(norm)
        # QuantDash 实时行情接口（quotes.get），无权限时依次降级：
        # 美股/港股 -> 腾讯财经 HTTP 接口 -> 最近K线
        try:
            raw = self._client.quotes.get(norm)
            if raw is None:
                raise ValueError("empty response")
            return raw
        except Exception as e:
            logger.warning("实时行情获取失败(%s): %s，尝试腾讯财经降级", norm, e)

        # QuantDash 不可用：美股/港股走腾讯财经 HTTP 接口
        if normalize_symbol(norm).split(".")[-1] in ("US", "HK"):
            tq = _fetch_tencent_realtime(norm)
            if tq is not None:
                return tq

        # 最终降级：最近K线
        logger.warning("降级为最近K线: %s", norm)
        return self._quote_from_recent_kline(norm)

    def _quote_from_recent_kline(self, norm: str) -> Dict[str, Any]:
        """用最近 2 根日K线合成一个实时行情快照。"""
        df = self.get_klines(norm, count=2, use_cache=False)
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        return {
            "symbol": norm,
            "last_price": float(last["close"]),
            "open": float(last["open"]),
            "high": float(last["high"]),
            "low": float(last["low"]),
            "prev_close": float(prev["close"]),
            "volume": int(last["volume"]),
            "amount": float(last.get("amount", 0)),
            "currency": get_currency(norm),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    # -- 五档盘口 -----------------------------------------------------------

    def get_order_book(self, symbol: str) -> Dict[str, Any]:
        """获取五档盘口。

        Returns:
            包含 bids(买五档) 和 asks(卖五档)，每档为 [price, volume]。
        """
        norm = normalize_symbol(symbol)
        if self.use_mock or self._client is None:
            quote = self.get_realtime_quote(norm)
            price = quote["last_price"]
            bids = [[round(price * (1 - 0.001 * i), 2), 1000 * (i + 1)] for i in range(1, 6)]
            asks = [[round(price * (1 + 0.001 * i), 2), 1000 * (i + 1)] for i in range(1, 6)]
            return {"symbol": norm, "bids": bids, "asks": asks}
        # QuantDash 五档盘口接口（depth.get），无权限时降级为模拟盘口
        try:
            raw = self._client.depth.get(norm)
            if raw is None:
                raise ValueError("empty response")
            return raw
        except Exception as e:
            logger.warning("五档盘口获取失败(%s)，降级为模拟盘口", e)
            quote = self.get_realtime_quote(norm)
            price = quote["last_price"]
            bids = [[round(price * (1 - 0.001 * i), 2), 1000 * (i + 1)] for i in range(1, 6)]
            asks = [[round(price * (1 + 0.001 * i), 2), 1000 * (i + 1)] for i in range(1, 6)]
            return {"symbol": norm, "bids": bids, "asks": asks}

    # -- 批量 ---------------------------------------------------------------

    def get_klines_batch(
        self,
        symbols: List[str],
        period: str = "1d",
        count: int = 300,
        adjust: str = "qfq",
    ) -> Dict[str, pd.DataFrame]:
        """批量获取多标的 K线。"""
        result = {}
        for sym in symbols:
            try:
                result[normalize_symbol(sym)] = self.get_klines(
                    sym, period=period, count=count, adjust=adjust
                )
            except Exception as e:
                logger.error("获取 %s K线失败: %s", sym, e)
        return result

    # -- 内部工具 -----------------------------------------------------------

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """将 QuantDash 返回的列名统一为小写标准列。"""
        # 优先用 trade_date 作为日期列，其次 timestamp，再次 time/datetime
        date_col = None
        for col in ["trade_date", "timestamp", "trade_time", "time", "datetime", "Date"]:
            if col in df.columns:
                date_col = col
                break

        col_map = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume", "Amount": "amount",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if date_col:
            if df[date_col].dtype.kind in "if":  # int/float = Unix 毫秒时间戳
                df["date"] = pd.to_datetime(df[date_col], unit="ms")
            else:
                df["date"] = pd.to_datetime(df[date_col])
            df.set_index("date", inplace=True)
            if date_col != "date":
                df = df.drop(columns=[date_col], errors="ignore")

        df.index.name = "date"
        # 只保留标准列
        keep = [c for c in ["open", "high", "low", "close", "volume", "amount"] if c in df.columns]
        return df[keep]
