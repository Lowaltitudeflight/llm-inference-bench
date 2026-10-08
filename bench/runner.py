"""并发压测执行器。

按并发梯度逐档加压，每档位并发发起请求并收集计时结果。
每路连接使用**独立**的客户端实例，避免共享连接池引入的排队
干扰，保证测到的是服务端能力而非客户端瓶颈。
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from config.settings import BenchConfig, EngineTarget, MonitorConfig
from engines.client import UnifiedLLMClient
from metrics.stats import RequestResult, StageMetrics, aggregate
from monitor.gpu import GpuMonitor

logger = logging.getLogger(__name__)


class LoadRunner:
    """并发负载执行器。

    Args:
        target: 引擎接入配置。
        config: 压测参数。
        monitor_config: GPU 监控参数；传入且启用时会为整个扫描过程
            采样显存曲线，并在返回结果里附上峰值显存。
    """

    def __init__(
        self,
        target: EngineTarget,
        config: BenchConfig,
        monitor_config: Optional[MonitorConfig] = None,
    ):
        self.target = target
        self.config = config
        self.monitor_config = monitor_config
        self.monitor: Optional[GpuMonitor] = None
        # 每档位显存峰值，供报告按并发对齐显存曲线
        self.stage_vram_peak: Dict[int, Optional[float]] = {}

    # ------------------------------------------------------------------
    # 单档位执行
    # ------------------------------------------------------------------
    def run_stage(self, concurrency: int) -> StageMetrics:
        """跑完一个并发档位，返回聚合指标。

        执行流程：
        1. 预热（结果丢弃，避免首请求的模型预热成本污染数据）
        2. 并发发起请求，墙壁时钟从**首个请求发出前**开始计时
        3. 收集全部结果并聚合

        Args:
            concurrency: 该档位并发数。
        """
        total = self.config.requests_per_level or concurrency
        # 保证每个并发通道至少分到一个请求
        total = max(total, concurrency)

        logger.info(
            "档位 concurrency=%d，请求总数=%d，预热=%d",
            concurrency,
            total,
            self.config.warmup_requests,
        )

        self._warmup(concurrency)

        results: List[RequestResult] = []
        lock = threading.Lock()

        def worker() -> RequestResult:
            # 每路请求自建客户端：避免连接池复用导致的排队
            client = UnifiedLLMClient(self.target, timeout=self.config.timeout)
            try:
                return client.complete(
                    prompt=self.config.prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    stream=self.config.stream,
                )
            finally:
                client.close()

        wall_start = time.perf_counter()
        monitor = self.monitor
        # 监控若可用，档位内额外高频采一次样，保证曲线覆盖加压时段
        monitor_before = len(monitor.samples) if monitor and monitor.available else 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(worker) for _ in range(total)]
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as exc:
                    res = RequestResult(
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                with lock:
                    results.append(res)
        wall_time = time.perf_counter() - wall_start

        # 记录该档位的显存增量峰值（相对加压前），反映并发带来的额外占用
        if monitor and monitor.available:
            self.stage_vram_peak[concurrency] = self._stage_vram_peak(monitor_before)

        stage = aggregate(
            results,
            concurrency=concurrency,
            wall_time=wall_time,
            max_failure_rate=self.config.max_failure_rate,
        )

        if stage.unstable:
            logger.warning(
                "档位 concurrency=%d 不稳定：失败率 %.1f%%",
                concurrency,
                stage.failure_rate * 100,
            )
        return stage

    def _stage_vram_peak(self, since_index: int) -> Optional[float]:
        """取加压期间所有卡的显存总占用峰值（MiB）。

        Args:
            since_index: 只统计该下标之后的样本（即本档位加压期间）。
        """
        if not self.monitor:
            return None
        batch = self.monitor.samples[since_index:]
        if not batch:
            return None
        # 按时间点聚合所有卡的显存之和，再取最大
        by_t: Dict[float, float] = {}
        for s in batch:
            if s.memory_used_mib is None:
                continue
            by_t[s.t] = by_t.get(s.t, 0.0) + s.memory_used_mib
        return max(by_t.values()) if by_t else None

    def _warmup(self, concurrency: int) -> None:
        """预热。失败不致命，仅记录日志。

        第一波请求通常包含模型首次前向、CUDA 内核编译、显存分配等
        一次性成本，若不预热，第一个档位的延迟会显著偏高。
        """
        if self.config.warmup_requests <= 0:
            return

        def one() -> None:
            client = UnifiedLLMClient(self.target, timeout=self.config.timeout)
            try:
                client.complete(
                    prompt=self.config.prompt,
                    max_tokens=min(self.config.max_tokens, 32),
                    temperature=self.config.temperature,
                    stream=self.config.stream,
                )
            except Exception as exc:
                logger.debug("预热请求失败（忽略）: %s", exc)
            finally:
                client.close()

        with ThreadPoolExecutor(max_workers=min(concurrency, self.config.warmup_requests)) as pool:
            list(pool.map(lambda _: one(), range(self.config.warmup_requests)))

    # ------------------------------------------------------------------
    # 全梯度扫描
    # ------------------------------------------------------------------
    def run_sweep(
        self,
        levels: Optional[List[int]] = None,
        on_stage: Optional[Callable[[StageMetrics], None]] = None,
    ) -> List[StageMetrics]:
        """按并发梯度依次执行。

        若配置了 GPU 监控，会在整个扫描期间持续采样显存，
        扫描结束后可通过 :meth:`vram_curve` 取到曲线。

        Args:
            levels: 并发档位列表；默认取 config 中的配置。
            on_stage: 每档完成后的回调，便于实时上报进度。

        Returns:
            按并发升序的档位指标列表。
        """
        levels = sorted(levels or self.config.concurrency_levels)
        stages: List[StageMetrics] = []

        if self.monitor_config is not None:
            self.monitor = GpuMonitor(
                interval=self.monitor_config.interval,
                enabled=self.monitor_config.enabled,
            )
            self.monitor.start()

        try:
            for level in levels:
                stage = self.run_stage(level)
                stages.append(stage)
                if on_stage:
                    on_stage(stage)
        finally:
            if self.monitor is not None:
                self.monitor.stop()

        return stages

    def vram_curve(self) -> Dict[str, Any]:
        """取本次扫描的显存曲线；未启用监控时返回 available=False。

        除 :meth:`GpuMonitor.curve` 的内容外，额外附加 ``stage_peak_mib``
        ——以并发数为键的档位显存峰值，供报告对齐「并发 - 显存」表。
        """
        if self.monitor is None:
            return {"available": False, "gpus": [], "message": "未启用 GPU 监控"}
        curve = self.monitor.curve()
        curve["stage_peak_mib"] = {
            str(k): v for k, v in sorted(self.stage_vram_peak.items())
        }
        return curve

    def health(self) -> Dict[str, Any]:
        """探测目标引擎可达性。"""
        client = UnifiedLLMClient(self.target, timeout=min(self.config.timeout, 10.0))
        try:
            return client.health()
        finally:
            client.close()
