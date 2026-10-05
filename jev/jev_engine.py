"""本地 Jev 决策模型集成模块。

通过 HTTP API 调用本地部署的 Jev 服务（默认 http://localhost:8765），
对策略原始信号进行二次过滤和决策增强。

Jev 返回各候选动作的概率分布，结合置信度阈值决定是否执行交易。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """当前市场状态特征，作为 Jev 决策的上下文。"""
    price: float
    price_change_5d: float
    ma5_ma20_ratio: float
    volume_ratio: float
    rsi: float
    macd_signal: int  # 1=金叉/多头, -1=死叉/空头, 0=中性
    volatility_20d: float

    def to_states_list(self) -> List[Dict[str, Any]]:
        """转为 Jev API 要求的 states 格式。"""
        return [
            {"feature": "price", "value": round(self.price, 4)},
            {"feature": "price_change_5d", "value": round(self.price_change_5d, 4)},
            {"feature": "ma5_ma20_ratio", "value": round(self.ma5_ma20_ratio, 4)},
            {"feature": "volume_ratio", "value": round(self.volume_ratio, 4)},
            {"feature": "rsi", "value": round(self.rsi, 2)},
            {"feature": "macd_signal", "value": self.macd_signal},
            {"feature": "volatility_20d", "value": round(self.volatility_20d, 4)},
        ]


@dataclass
class JevDecision:
    """Jev 决策结果。"""
    timestamp: str
    symbol: str
    raw_signal: str  # 策略原始信号 buy/sell/hold
    raw_confidence: float
    market_state: Dict[str, Any]
    probabilities: Dict[str, float]  # {"buy": 0.x, "sell": 0.x, "hold": 0.x}
    final_action: str  # 最终决策
    final_confidence: float
    executed: bool  # 是否通过阈值执行
    reason: str = ""


@dataclass
class JevAuditLog:
    """Jev 决策审计日志，持久化到 JSONL。"""
    log_path: str = "./logs/jev_audit.jsonl"

    def __post_init__(self) -> None:
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

    def record(self, decision: JevDecision) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")


class JevDecisionEngine:
    """Jev 决策引擎。

    接收策略信号，构造市场状态，调用本地 Jev 服务获取概率分布，
    应用置信度阈值后输出最终决策。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        endpoint: str = "/api/evaluate",
        timeout: float = 5.0,
        confidence_threshold: float = 0.6,
        retry_count: int = 2,
        mock_mode: bool = True,
        audit_log_path: str = "./logs/jev_audit.jsonl",
        max_cache_size: int = 100,
        max_concurrency: int = 3,
        latency_window: int = 1000,
    ):
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.timeout = timeout
        self.confidence_threshold = confidence_threshold
        self.retry_count = retry_count
        self.mock_mode = mock_mode
        self.audit = JevAuditLog(log_path=audit_log_path)
        self._session = requests.Session()

        # ---- 性能优化：KV 缓存 ----
        self._max_cache_size = max_cache_size
        self._cache: "OrderedDict[Tuple[Any, ...], JevDecision]" = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        # ---- 性能优化：延迟统计（环形缓冲） ----
        self._latencies: deque[float] = deque(maxlen=latency_window)
        self._total_requests = 0
        self._perf_start = time.perf_counter()

        # ---- 性能优化：异步队列 + 并发控制 ----
        self._max_concurrency = max_concurrency
        self._queue: Optional[asyncio.Queue] = None
        self._workers: List[asyncio.Task] = []
        self._worker_tasks: List[asyncio.Task] = []
        self._running = False
        self._semaphore: Optional[asyncio.Semaphore] = None
        # 用于观测并发峰值（测试/统计）
        self._active_inferences = 0
        self._peak_inferences = 0

        if mock_mode:
            logger.info("JevDecisionEngine 运行在 mock 模式（使用内置启发式模型）。")
        else:
            logger.info("JevDecisionEngine 已连接 %s", self.base_url)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def evaluate(
        self,
        raw_signal: str,
        raw_confidence: float,
        market_state: MarketState,
        symbol: str = "",
    ) -> JevDecision:
        """对单条策略信号进行 Jev 决策评估。

        集成 KV 缓存与延迟统计：相同输入第二次命中缓存，不再重复推理。

        Args:
            raw_signal: 策略原始动作 buy/sell/hold。
            raw_confidence: 策略原始置信度。
            market_state: 当前市场状态。
            symbol: 标的代码。

        Returns:
            JevDecision 决策结果。
        """
        t0 = time.perf_counter()

        # 1) KV 缓存命中直接返回
        cached = self.get_cached({
            "symbol": symbol,
            "signal": raw_signal,
            "confidence": raw_confidence,
            "market_state": market_state,
        })
        if cached is not None:
            self._record_latency(t0)
            return cached

        # 2) 缓存未命中，执行真实推理
        states = market_state.to_states_list()
        probabilities = self._compute_probabilities(states, raw_signal, raw_confidence)

        decision = self._build_decision(
            probabilities, raw_signal, raw_confidence, market_state, symbol,
        )

        # 3) 写缓存 + 审计
        self.set_cached({
            "symbol": symbol,
            "signal": raw_signal,
            "confidence": raw_confidence,
            "market_state": market_state,
        }, decision)
        self.audit.record(decision)
        self._record_latency(t0)

        logger.debug(
            "Jev决策: %s | 原始=%s(%.2f) → 最终=%s(%.2f) | 执行=%s | %s",
            symbol, raw_signal, raw_confidence,
            decision.final_action, decision.final_confidence,
            decision.executed, decision.reason,
        )
        return decision

    # ------------------------------------------------------------------
    # 性能优化：批量推理
    # ------------------------------------------------------------------

    def evaluate_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量决策推理。

        每个请求字典需包含:
            - symbol: str 标的代码
            - signal: str 原始动作 buy/sell/hold
            - confidence: float 原始置信度
            - market_state: MarketState 或 dict（MarketState 字段）

        在非 mock 模式下，若服务端支持 /api/evaluate_batch 则一次性批量调用，
        否则串行单条调用并合并结果。mock 模式下逐条走内置启发式。

        Args:
            requests: 决策请求列表。

        Returns:
            与输入等长的决策结果字典列表（顺序一致）。
        """
        t0 = time.perf_counter()
        results: List[Optional[JevDecision]] = [None] * len(requests)

        # 先尝试缓存命中
        cache_miss_idx: List[int] = []
        cache_miss_req: List[Dict[str, Any]] = []
        for i, req in enumerate(requests):
            cached = self.get_cached(req)
            if cached is not None:
                results[i] = cached
            else:
                cache_miss_idx.append(i)
                cache_miss_req.append(req)

        if cache_miss_req:
            if self.mock_mode:
                # mock 模式：逐条推理（纯内存计算，无 HTTP 开销）
                for i, req in zip(cache_miss_idx, cache_miss_req):
                    signal, confidence, mstate, symbol = self._normalize_request(req)
                    states = mstate.to_states_list()
                    probs = self._compute_probabilities(states, signal, confidence)
                    decision = self._build_decision(probs, signal, confidence, mstate, symbol)
                    self.set_cached(req, decision)
                    self.audit.record(decision)
                    results[i] = decision
            else:
                # 真实模式：优先调用服务端批量接口，失败则串行降级
                batch_results = self._call_api_batch(cache_miss_req)
                for i, req, decision in zip(cache_miss_idx, cache_miss_req, batch_results):
                    self.set_cached(req, decision)
                    self.audit.record(decision)
                    results[i] = decision

        self._record_latency(t0)
        logger.info(
            "Jev 批量推理: %d 条（缓存命中 %d，实算 %d）",
            len(requests), len(requests) - len(cache_miss_req), len(cache_miss_req),
        )
        return [asdict(d) for d in results]  # type: ignore[list-item]

    # ------------------------------------------------------------------
    # 性能优化：异步请求队列 + worker
    # ------------------------------------------------------------------

    def start_workers(self, n: int = 2) -> None:
        """启动 N 个 worker 协程，从队列消费请求并批量处理。

        必须在运行中的事件循环内调用。重复调用安全（已有 worker 则忽略）。
        """
        if self._running:
            return
        loop = asyncio.get_event_loop()
        self._queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._running = True
        self._worker_tasks = [
            loop.create_task(self._worker_loop(wid))
            for wid in range(n)
        ]
        logger.info("Jev 异步 worker 已启动: %d 个, 并发上限=%d", n, self._max_concurrency)

    async def stop_workers(self) -> None:
        """优雅停止所有 worker 协程，等待队列消费完毕。"""
        if not self._running:
            return
        self._running = False
        # 放入哨兵唤醒阻塞的 worker
        assert self._queue is not None
        for _ in self._worker_tasks:
            await self._queue.put(None)  # type: ignore[arg-type]
        # 等待 worker 结束
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        logger.info("Jev 异步 worker 已停止")

    async def evaluate_async(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """异步决策推理。

        请求进入内部队列，由 worker 协程批量消费，受信号量控制最大并发。

        Args:
            request: 决策请求 dict（同 evaluate_batch 单条格式）。

        Returns:
            决策结果字典。
        """
        if not self._running:
            self.start_workers()
        assert self._queue is not None and self._semaphore is not None

        async with self._semaphore:
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            await self._queue.put((request, fut))
            return await fut

    async def _worker_loop(self, worker_id: int) -> None:
        """worker 主循环：从队列取请求，凑批后统一处理。"""
        assert self._queue is not None
        while self._running:
            item = await self._queue.get()
            if item is None:  # 哨兵
                break
            first_req, first_fut = item
            batch: List[Tuple[Dict[str, Any], asyncio.Future]] = [(first_req, first_fut)]

            # 短窗口内尽量多收几条，凑成微批
            deadline = time.monotonic() + 0.005
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                try:
                    nxt = await asyncio.wait_for(self._queue.get(), timeout=max(remaining, 0.0005))
                except asyncio.TimeoutError:
                    break
                if nxt is None:
                    # 停止哨兵：放回给下一个 worker
                    await self._queue.put(None)  # type: ignore[arg-type]
                    break
                batch.append(nxt)

            # 逐条处理（mock 为纯内存计算；真实推理用线程池避免阻塞事件循环）
            for req, fut in batch:
                if fut.done():
                    continue
                try:
                    result = await asyncio.to_thread(self._process_one_sync, req)
                    if not fut.done():
                        fut.set_result(result)
                except Exception as e:  # noqa: BLE001
                    if not fut.done():
                        fut.set_exception(e)

    def _process_one_sync(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """同步处理单条异步请求（在 worker 线程池中执行）。"""
        self._active_inferences += 1
        self._peak_inferences = max(self._peak_inferences, self._active_inferences)
        try:
            # 复用 evaluate（内部带缓存与延迟统计）
            signal, confidence, mstate, symbol = self._normalize_request(req)
            decision = self.evaluate(signal, confidence, mstate, symbol=symbol)
            return asdict(decision)
        finally:
            self._active_inferences -= 1

    # ------------------------------------------------------------------
    # 性能优化：KV 缓存
    # ------------------------------------------------------------------

    def _cache_key(self, request: Dict[str, Any]) -> Tuple[Any, ...]:
        """根据请求内容生成缓存 key。"""
        signal, confidence, mstate, symbol = self._normalize_request(request)
        states_tuple = tuple(
            sorted((s["feature"], s["value"]) for s in mstate.to_states_list())
        )
        return (symbol, signal, round(float(confidence), 6), states_tuple)

    def get_cached(self, request: Dict[str, Any]) -> Optional[JevDecision]:
        """查询 KV 缓存，命中返回缓存的 JevDecision，否则 None。"""
        key = self._cache_key(request)
        if key in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._cache_misses += 1
        return None

    def set_cached(self, request: Dict[str, Any], result: JevDecision) -> None:
        """写入 KV 缓存（LRU，超出 maxsize 淘汰最久未使用项）。"""
        key = self._cache_key(request)
        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_cache_size:
            self._cache.popitem(last=False)

    # ------------------------------------------------------------------
    # 性能优化：延迟统计
    # ------------------------------------------------------------------

    def _record_latency(self, t0: float) -> None:
        """记录一次推理延迟（毫秒）。"""
        self._latencies.append((time.perf_counter() - t0) * 1000.0)
        self._total_requests += 1

    def get_performance_stats(self) -> Dict[str, Any]:
        """返回推理性能统计。

        Returns:
            {p50, p95, p99, total_requests, cache_hits, cache_misses,
             cache_hit_rate, throughput}（延迟单位 ms，throughput 单位 req/s）。
        """
        if self._latencies:
            arr = np.asarray(self._latencies, dtype=float)
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
        else:
            p50 = p95 = p99 = 0.0

        elapsed = time.perf_counter() - self._perf_start
        throughput = self._total_requests / elapsed if elapsed > 1e-6 else 0.0
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total) if total > 0 else 0.0

        return {
            "p50": round(p50, 3),
            "p95": round(p95, 3),
            "p99": round(p99, 3),
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": round(hit_rate, 4),
            "throughput": round(throughput, 4),
            "peak_concurrency": self._peak_inferences,
            "cache_size": len(self._cache),
            "cache_maxsize": self._max_cache_size,
        }

    # ------------------------------------------------------------------
    # 在线学习（REQ-P3-05）
    # ------------------------------------------------------------------
    def update_model(
        self,
        feedback_records: List[Dict[str, Any]],
        feature_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """在线增量更新模型（委托 jev.online_learning 实现）。

        首次调用自动挂载 :class:`~jev.online_learning.OnlineLearningPipeline`；
        后续调用走增量训练（SGD partial_fit，非全量重训），并自动做
        概念漂移检测与滑动窗口衰减。

        Args:
            feedback_records: 决策反馈列表，每条形如
                ``{"features": {...}, "realized_return": 0.012}``。
            feature_keys: 特征顺序；None 则按首条记录的 key 排序。

        Returns:
            本次更新的统计（batch_loss / window / drift 等）。
        """
        from jev.online_learning import update_model as _update_model

        return _update_model(self, feedback_records, feature_keys)

    def get_online_learning_stats(self) -> Optional[Dict[str, Any]]:
        """返回在线学习管线统计；尚未挂载时返回 None。"""
        pipeline = getattr(self, "_online_pipeline", None)
        if pipeline is None:
            return None
        stats = pipeline.get_stats()
        stats["accuracy_proxy"] = pipeline.accuracy_proxy()
        return stats

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _normalize_request(self, request: Dict[str, Any]) -> Tuple[str, float, MarketState, str]:
        """将批量/异步请求 dict 规范化为 evaluate() 所需参数。"""
        symbol = request.get("symbol", "")
        signal = request.get("signal", "hold")
        confidence = float(request.get("confidence", 0.5))
        ms = request.get("market_state")
        if isinstance(ms, MarketState):
            mstate = ms
        elif isinstance(ms, dict):
            # 仅取 MarketState 支持的字段，忽略多余 key
            fields = {f: ms[f] for f in MarketState.__dataclass_fields__ if f in ms}
            mstate = MarketState(**fields)
        else:
            raise ValueError(
                "request['market_state'] 必须是 MarketState 或 dict，"
                f"实际为 {type(ms).__name__}"
            )
        return signal, confidence, mstate, symbol

    def _compute_probabilities(
        self,
        states: List[Dict[str, Any]],
        raw_signal: str,
        raw_confidence: float,
    ) -> Dict[str, float]:
        """根据模式计算概率分布（mock 或 HTTP）。"""
        if self.mock_mode:
            return self._mock_evaluate(states, raw_signal, raw_confidence)
        return self._call_api(states, ["buy", "sell", "hold"])

    def _build_decision(
        self,
        probabilities: Dict[str, float],
        raw_signal: str,
        raw_confidence: float,
        market_state: MarketState,
        symbol: str,
    ) -> JevDecision:
        """根据概率分布构建 JevDecision（含阈值判定）。"""
        final_action = max(probabilities, key=probabilities.get)
        final_confidence = probabilities[final_action]

        if final_action == "hold":
            reason = "Jev 建议观望"
            executed = False
        elif final_action != raw_signal and raw_signal != "hold":
            reason = f"Jev 方向({final_action})与策略方向({raw_signal})冲突"
            executed = False
        elif final_confidence < self.confidence_threshold:
            reason = f"置信度 {final_confidence:.3f} 低于阈值 {self.confidence_threshold}"
            executed = False
        else:
            executed = True
            reason = "通过 Jev 过滤，执行交易"

        return JevDecision(
            timestamp=pd.Timestamp.now().isoformat(),
            symbol=symbol,
            raw_signal=raw_signal,
            raw_confidence=round(raw_confidence, 4),
            market_state=asdict(market_state),
            probabilities={k: round(v, 4) for k, v in probabilities.items()},
            final_action=final_action,
            final_confidence=round(final_confidence, 4),
            executed=executed,
            reason=reason,
        )

    def _call_api_batch(
        self, requests: List[Dict[str, Any]]
    ) -> List[JevDecision]:
        """非 mock 模式下的批量推理：优先服务端 /api/evaluate_batch，失败串行降级。"""
        # 尝试服务端批量接口
        try:
            url = f"{self.base_url}/api/evaluate_batch"
            batch_payload = []
            for req in requests:
                signal, confidence, mstate, symbol = self._normalize_request(req)
                batch_payload.append({
                    "states": mstate.to_states_list(),
                    "candidate_actions": ["buy", "sell", "hold"],
                })
            resp = self._session.post(
                url, json={"batch": batch_payload},
                timeout=self.timeout * max(len(requests), 1),
            )
            resp.raise_for_status()
            data = resp.json()
            server_results = data.get("results", [])
            decisions: List[JevDecision] = []
            for req, sres in zip(requests, server_results):
                signal, confidence, mstate, symbol = self._normalize_request(req)
                probs = sres.get("probabilities", {})
                total = sum(probs.values())
                if total > 0:
                    probs = {k: v / total for k, v in probs.items()}
                probs = {a: float(probs.get(a, 0.0)) for a in ["buy", "sell", "hold"]}
                decisions.append(
                    self._build_decision(probs, signal, confidence, mstate, symbol)
                )
            return decisions
        except Exception as e:
            logger.warning("服务端批量推理失败，降级为串行单条: %s", e)

        # 串行降级
        decisions = []
        for req in requests:
            signal, confidence, mstate, symbol = self._normalize_request(req)
            states = mstate.to_states_list()
            probs = self._compute_probabilities(states, signal, confidence)
            decisions.append(
                self._build_decision(probs, signal, confidence, mstate, symbol)
            )
        return decisions

    def build_market_state(self, df: pd.DataFrame, idx: int) -> Optional[MarketState]:
        """从行情 DataFrame 的指定位置构造 MarketState。

        Args:
            df: 包含技术指标的行情数据（需已调用 add_indicators）。
            idx: 行索引位置。

        Returns:
            MarketState 或 None（数据不足时）。
        """
        if idx < 20:
            return None
        row = df.iloc[idx]
        prev_row = df.iloc[max(0, idx - 5)]

        try:
            price = float(row["close"])
            price_change_5d = float(row["close"] / prev_row["close"] - 1)
            ma5 = float(row.get("ma5", row["close"]))
            ma20 = float(row.get("ma20", row["close"]))
            ma5_ma20_ratio = ma5 / ma20 if ma20 != 0 else 1.0
            volume_ratio = float(row.get("vol_ratio", 1.0))
            rsi_val = float(row.get("rsi", 50.0))
            macd_hist = float(row.get("macd_hist", 0.0))
            macd_signal = 1 if macd_hist > 0 else (-1 if macd_hist < 0 else 0)
            # 20日波动率
            returns = df["close"].pct_change().iloc[max(0, idx - 20):idx + 1]
            volatility_20d = float(returns.std() * np.sqrt(252))
        except (KeyError, ValueError, ZeroDivisionError):
            return None

        return MarketState(
            price=price,
            price_change_5d=price_change_5d,
            ma5_ma20_ratio=ma5_ma20_ratio,
            volume_ratio=volume_ratio,
            rsi=rsi_val,
            macd_signal=macd_signal,
            volatility_20d=volatility_20d,
        )

    def filter_signals(
        self,
        signals: List[Dict[str, Any]],
        df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """批量过滤信号：对每条信号调用 Jev，返回通过过滤的信号。

        Args:
            signals: 信号字典列表，需含 date/action/confidence 字段。
            df: 行情数据（含技术指标）。

        Returns:
            通过 Jev 过滤的信号列表。
        """
        filtered = []
        date_to_idx = {d: i for i, d in enumerate(df.index)}

        for sig in signals:
            sig_date = sig["date"]
            if sig_date not in date_to_idx:
                continue
            idx = date_to_idx[sig_date]
            state = self.build_market_state(df, idx)
            if state is None:
                continue

            decision = self.evaluate(
                raw_signal=sig["action"],
                raw_confidence=sig["confidence"],
                market_state=state,
                symbol=sig.get("symbol", ""),
            )
            if decision.executed:
                sig_copy = dict(sig)
                sig_copy["jev_confidence"] = decision.final_confidence
                sig_copy["jev_probabilities"] = decision.probabilities
                filtered.append(sig_copy)

        logger.info("Jev 过滤: %d 条信号 → %d 条通过", len(signals), len(filtered))
        return filtered

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _call_api(
        self, states: List[Dict[str, Any]], candidate_actions: List[str]
    ) -> Dict[str, float]:
        """调用本地 Jev HTTP API。"""
        url = f"{self.base_url}{self.endpoint}"
        payload = {
            "states": states,
            "candidate_actions": candidate_actions,
        }

        for attempt in range(self.retry_count + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                # 期望返回 {"probabilities": {"buy": 0.x, "sell": 0.x, "hold": 0.x}}
                probs = data.get("probabilities", data)
                # 归一化
                total = sum(probs.values())
                if total > 0:
                    probs = {k: v / total for k, v in probs.items()}
                return {a: float(probs.get(a, 0.0)) for a in candidate_actions}
            except Exception as e:
                logger.warning("Jev API 调用失败(第%d次): %s", attempt + 1, e)
                if attempt < self.retry_count:
                    time.sleep(0.5 * (attempt + 1))
                else:
                    logger.error("Jev API 不可用，降级为 mock 模式")
                    return self._mock_evaluate(states, "hold", 0.5)
        return {"buy": 0.0, "sell": 0.0, "hold": 1.0}

    def _mock_evaluate(
        self,
        states: List[Dict[str, Any]],
        raw_signal: str,
        raw_confidence: float,
    ) -> Dict[str, float]:
        """内置启发式 mock 模型，模拟 Jev 概率输出。

        基于市场状态特征计算各动作的倾向分数，再 softmax 归一化。
        这使得"有 Jev 过滤"的回测具有实际的信号筛选效果。
        """
        state_dict = {s["feature"]: s["value"] for s in states}

        # 各特征打分
        score_buy = 0.0
        score_sell = 0.0

        # 5日涨跌幅：下跌过多倾向买入（均值回归），上涨过多倾向卖出
        pct_5d = state_dict.get("price_change_5d", 0.0)
        score_buy += -pct_5d * 3.0   # 跌得多 → 买
        score_sell += pct_5d * 3.0   # 涨得多 → 卖

        # 均线比率：ma5 > ma20 多头 → 买
        ma_ratio = state_dict.get("ma5_ma20_ratio", 1.0)
        score_buy += (ma_ratio - 1.0) * 10.0
        score_sell += (1.0 - ma_ratio) * 10.0

        # RSI：超卖(<30)买，超买(>70)卖
        rsi_val = state_dict.get("rsi", 50.0)
        if rsi_val < 30:
            score_buy += (30 - rsi_val) * 0.1
        elif rsi_val > 70:
            score_sell += (rsi_val - 70) * 0.1

        # MACD 信号
        macd_sig = state_dict.get("macd_signal", 0)
        score_buy += macd_sig * 0.8
        score_sell += -macd_sig * 0.8

        # 量比：放量增强当前方向
        vol_ratio = state_dict.get("volume_ratio", 1.0)
        if raw_signal == "buy":
            score_buy += vol_ratio * 0.3
        elif raw_signal == "sell":
            score_sell += vol_ratio * 0.3

        # 原始策略信号作为先验
        if raw_signal == "buy":
            score_buy += raw_confidence * 1.5
        elif raw_signal == "sell":
            score_sell += raw_confidence * 1.5

        # hold 基准分（始终有一定概率观望）
        score_hold = 0.5

        # softmax
        scores = np.array([score_buy, score_sell, score_hold])
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()

        return {
            "buy": float(probs[0]),
            "sell": float(probs[1]),
            "hold": float(probs[2]),
        }
