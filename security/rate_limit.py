"""API 限流模块（令牌桶算法）。

按「限流维度（API Token 优先，未认证按 IP）× 分级」维护独立的令牌桶，
对高频/高成本接口做配额限制，防止单一方刷爆服务。

特性：
  - 令牌桶：桶容量 = 每分钟配额，按 ``capacity / 60`` 每秒匀速补充；桶满不再
    堆积令牌。允许短时突发，但长期不超过每分钟配额。
  - 分级配额：``read``(默认查询) / ``trade``(交易开关) / ``admin``(管理操作) /
    ``backtest``(回测/优化等重计算) 四档独立配额。
  - 限流维度：认证用户优先用 token 名称作为 key；未认证请求用来源 IP。
  - 白名单：``127.0.0.1`` / ``localhost`` / ``::1`` 等本机地址不限流。
  - LRU 清理：超过闲置 TTL（默认 1 小时）未访问的桶自动删除，防止内存泄漏。
  - 线程安全：``threading.Lock`` 保护桶字典与每个桶的令牌计数。

用法::

    limiter = RateLimiter(config_dict)
    allowed, retry_after, info = limiter.check_rate_limit(key="alice", tier="read")
    if not allowed:
        return 429 + Retry-After
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# 默认分级配额（次/分钟）
DEFAULT_TIERS: Dict[str, float] = {
    "read": 100.0,
    "trade": 20.0,
    "admin": 10.0,
    "backtest": 5.0,
}

# 默认白名单（本机回环地址不限流）
DEFAULT_WHITELIST = ("127.0.0.1", "localhost", "::1")

# 桶闲置超过该秒数后清理（1 小时）
_BUCKET_IDLE_TTL = 3600.0

# 主动清理的最小时间间隔（秒），避免每个请求都全量扫描桶字典
_CLEANUP_INTERVAL = 60.0


class TokenBucket:
    """令牌桶：容量即每分钟配额，按固定速率匀速补充。

    Attributes:
        capacity: 桶容量（= 每分钟允许的请求数）。
        tokens: 当前剩余令牌数。
        refill_rate: 每秒补充的令牌数（= capacity / 60）。
        last_refill: 上次补充令牌的时间戳（epoch 秒）。
        last_accessed: 上次被访问（消费）的时间戳，用于 LRU 清理。
    """

    __slots__ = ("capacity", "tokens", "refill_rate", "last_refill", "last_accessed")

    def __init__(self, capacity: float, now: Optional[float] = None) -> None:
        """初始化一个满桶。

        Args:
            capacity: 桶容量（每分钟配额）。
            now: 当前时间戳（注入便于测试）；默认 ``time.time()``。
        """
        self.capacity = float(capacity)
        self.tokens = float(capacity)  # 初始满桶，允许一次突发
        self.refill_rate = self.capacity / 60.0  # 每秒补充速率
        self.last_refill = float(now if now is not None else time.time())
        self.last_accessed = self.last_refill

    def _refill(self, now: float) -> None:
        """按经过时间补充令牌，桶满即止。"""
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(
                self.capacity, self.tokens + elapsed * self.refill_rate
            )
            self.last_refill = now

    def consume(
        self, tokens: float = 1.0, now: Optional[float] = None
    ) -> Tuple[bool, float]:
        """尝试消费 ``tokens`` 个令牌。

        Args:
            tokens: 要消费的令牌数（默认 1 次请求）。
            now: 当前时间戳（注入便于测试）；默认 ``time.time()``。

        Returns:
            ``(是否允许, 需等待的秒数)``。允许时第二项为 0；超限时第二项为
            攒够所需令牌预计要等待的秒数（即 Retry-After）。
        """
        now = float(now if now is not None else time.time())
        self._refill(now)
        self.last_accessed = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, 0.0

        # 令牌不足：计算补够 deficit 个令牌需要的秒数
        deficit = tokens - self.tokens
        retry_after = deficit / self.refill_rate if self.refill_rate > 0 else 0.0
        return False, retry_after


class RateLimiter:
    """按「维度 × 分级」维护令牌桶的限流器。

    构造参数为 ``config.yaml`` 中 ``rate_limit`` 节的字典::

        rate_limit:
          enabled: true
          tiers: {read: 100, trade: 20, admin: 10, backtest: 5}
          whitelist: ["127.0.0.1", "localhost", "::1"]
          default_tier: "read"
    """

    def __init__(self, config: Optional[Dict[str, Any]]) -> None:
        """初始化限流器。

        Args:
            config: ``rate_limit`` 节配置字典；为 None 时按 disabled + 默认配额处理。
        """
        cfg: Dict[str, Any] = dict(config or {})
        self._enabled: bool = bool(cfg.get("enabled", False))

        # 分级配额：以默认值为底，覆盖配置值
        tiers: Dict[str, float] = dict(DEFAULT_TIERS)
        raw_tiers = cfg.get("tiers") or {}
        if isinstance(raw_tiers, dict):
            for name, value in raw_tiers.items():
                try:
                    tiers[str(name)] = float(value)
                except (TypeError, ValueError):
                    logger.warning("rate_limit.tiers 中 %r=%r 无法解析为数字，已忽略",
                                   name, value)
        self._tiers: Dict[str, float] = tiers

        # 白名单（去重）
        raw_whitelist = cfg.get("whitelist") or list(DEFAULT_WHITELIST)
        self._whitelist = {str(x) for x in raw_whitelist}

        self._default_tier: str = str(cfg.get("default_tier", "read"))

        # 桶字典：key = f"{dimension}|{tier}" -> TokenBucket
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._last_cleanup: float = time.time()

        if self._enabled:
            logger.info(
                "API 限流已启用: tiers=%s, whitelist=%s",
                {k: int(v) for k, v in self._tiers.items()},
                sorted(self._whitelist),
            )
        else:
            logger.info("API 限流未启用（rate_limit.enabled=false）")

    # ------------------------------------------------------------------
    # 基础属性
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """返回是否启用限流。"""
        return self._enabled

    def is_whitelisted(self, ip: str) -> bool:
        """判断来源 IP 是否在白名单内（白名单不限流）。"""
        return bool(ip) and ip in self._whitelist

    # ------------------------------------------------------------------
    # 核心：检查并消费配额
    # ------------------------------------------------------------------

    @staticmethod
    def _bucket_key(dimension: str, tier: str) -> str:
        """组合维度与分级为桶字典键。"""
        return f"{dimension}|{tier}"

    def check_rate_limit(
        self,
        key: str,
        tier: str,
        now: Optional[float] = None,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """检查某个维度在指定分级下是否还能再消费 1 个请求。

        Args:
            key: 限流维度标识（token 名称 或 客户端 IP）。
            tier: 分级名（read / trade / admin / backtest）。
            now: 当前时间戳（注入便于测试）；默认 ``time.time()``。

        Returns:
            ``(是否允许, retry_after_秒, 状态信息字典)``。状态信息包含
            ``tier`` / ``limit`` / ``remaining`` / ``retry_after``。
            当限流未启用时直接返回 ``(True, 0.0, {...})``，不创建桶。
        """
        now = float(now if now is not None else time.time())

        if not self._enabled:
            return True, 0.0, {
                "tier": tier,
                "limit": int(self._tiers.get(tier, self._tiers.get(self._default_tier, 0))),
                "remaining": None,
                "retry_after": 0.0,
            }

        capacity = self._tiers.get(
            tier, self._tiers.get(self._default_tier, DEFAULT_TIERS["read"])
        )
        bucket_key = self._bucket_key(key, tier)

        with self._lock:
            self._maybe_cleanup(now)
            bucket = self._buckets.get(bucket_key)
            if bucket is None:
                bucket = TokenBucket(capacity=capacity, now=now)
                self._buckets[bucket_key] = bucket

            allowed, retry_after = bucket.consume(tokens=1.0, now=now)
            status: Dict[str, Any] = {
                "tier": tier,
                "limit": int(capacity),
                "remaining": int(max(0.0, bucket.tokens)),
                "retry_after": round(retry_after, 2),
            }

        return allowed, retry_after, status

    # ------------------------------------------------------------------
    # LRU 清理
    # ------------------------------------------------------------------

    def _maybe_cleanup(self, now: float) -> None:
        """距离上次清理不足间隔则跳过；否则触发一次过期桶清理。

        调用方须已持有 ``self._lock``。
        """
        if now - self._last_cleanup < _CLEANUP_INTERVAL:
            return
        self._last_cleanup = now
        self._cleanup_expired(now)

    def cleanup_expired(self, now: Optional[float] = None) -> int:
        """主动清理超过闲置 TTL 未访问的桶（线程安全）。

        Args:
            now: 当前时间戳（注入便于测试）；默认 ``time.time()``。

        Returns:
            清理掉的桶数量。
        """
        now = float(now if now is not None else time.time())
        with self._lock:
            return self._cleanup_expired(now)

    def _cleanup_expired(self, now: float) -> int:
        """实际清理逻辑（调用方须已持有 ``self._lock``）。"""
        expired = [
            key for key, b in self._buckets.items()
            if now - b.last_accessed > _BUCKET_IDLE_TTL
        ]
        for key in expired:
            del self._buckets[key]
        if expired:
            logger.info("限流 LRU 清理: 移除 %d 个闲置桶", len(expired))
        return len(expired)

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """返回限流器运行状态（线程安全快照）。

        Returns:
            ``{enabled, tiers, whitelist, default_tier, current_limits, bucket_count}``。
            ``current_limits`` 按分级聚合，列出当前每个维度的剩余配额。
        """
        with self._lock:
            current_limits: Dict[str, Any] = {}
            for bucket_key, b in self._buckets.items():
                # bucket_key = "dimension|tier"，维度名中理论上不会含 "|"
                dimension, _, tier = bucket_key.rpartition("|")
                tier_dict = current_limits.setdefault(tier, {})
                tier_dict[dimension] = {
                    "limit": int(b.capacity),
                    "remaining": int(max(0.0, b.tokens)),
                }
            return {
                "enabled": self._enabled,
                "tiers": {k: int(v) for k, v in self._tiers.items()},
                "whitelist": sorted(self._whitelist),
                "default_tier": self._default_tier,
                "current_limits": current_limits,
                "bucket_count": len(self._buckets),
            }
