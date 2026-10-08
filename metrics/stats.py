"""压测指标计算。

大模型推理服务的性能评估有四个核心指标，本模块负责从单请求
的原始计时数据中算出它们：

- **TTFT**（Time To First Token，首包延迟）：从发出请求到收到第一个
  token 的时间。直接决定用户的「感知响应速度」。
- **TPOT**（Time Per Output Token，单 token 生成延迟）：生成阶段
  平均每个 token 的耗时。反映解码速度。
- **吞吐**（tokens/s）：单位时间内服务产出的 token 总数。反映服务
  整体处理能力，是容量规划的核心。
- **尾延迟**（P99 / P95）：高分位延迟。平均值会掩盖长尾问题，
  线上体验往往由 P99 决定。

注意：TTFT 与 TPOT 只有在**流式**响应下才能准确测量。非流式请求
拿不到首 token 的时刻，只能得到端到端总延迟。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 失败样本最多保留条数，见 aggregate() 中的说明
MAX_ERROR_SAMPLES = 5


@dataclass
class RequestResult:
    """单次请求的原始观测结果。

    时间单位统一为**秒**，token 数为整数。
    """

    # 是否成功完成
    success: bool

    # 端到端总耗时（发出请求到收到完整响应）
    total_latency: Optional[float] = None

    # 首 token 延迟；非流式或失败时为 None
    ttft: Optional[float] = None

    # 生成阶段耗时（= total_latency - ttft）
    generation_time: Optional[float] = None

    # 输出 token 数
    output_tokens: int = 0

    # 输入 token 数
    input_tokens: int = 0

    # 失败时的错误信息
    error: Optional[str] = None

    # 单 token 生成延迟
    @property
    def tpot(self) -> Optional[float]:
        """Time Per Output Token。

        生成阶段耗时除以「生成阶段产出的 token 数」。
        首 token 的耗时已计入 TTFT，因此分母要减去它，
        否则会把预填充阶段的时间摊到解码阶段，低估 TPOT。
        """
        if self.generation_time is None or self.output_tokens <= 1:
            return None
        return self.generation_time / (self.output_tokens - 1)

    # 该请求的输出速率
    @property
    def throughput(self) -> Optional[float]:
        """单请求的输出 token 速率（tokens/s）。"""
        if not self.total_latency or self.total_latency <= 0 or self.output_tokens <= 0:
            return None
        return self.output_tokens / self.total_latency


def percentile(values: List[float], p: float) -> Optional[float]:
    """线性插值分位数。

    ``p`` 取 0~100。空列表返回 None。

    使用线性插值（等价 numpy 默认的 'linear' 方法），
    避免小样本下取到极端值导致分位数失真。
    """
    if not values:
        return None
    if not 0 <= p <= 100:
        raise ValueError(f"分位数 p 必须在 0~100 之间，收到 {p}")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    # 线性插值位置
    rank = (p / 100) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]

    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _safe_mean(values: List[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def _safe_stdev(values: List[float]) -> Optional[float]:
    return statistics.stdev(values) if len(values) > 1 else None


@dataclass
class StageMetrics:
    """一个并发档位的聚合指标。"""

    concurrency: int = 0
    total_requests: int = 0
    success_requests: int = 0
    failed_requests: int = 0

    # 端到端延迟
    latency_mean: Optional[float] = None
    latency_p50: Optional[float] = None
    latency_p95: Optional[float] = None
    latency_p99: Optional[float] = None
    latency_min: Optional[float] = None
    latency_max: Optional[float] = None
    latency_stdev: Optional[float] = None

    # 首 token 延迟
    ttft_mean: Optional[float] = None
    ttft_p50: Optional[float] = None
    ttft_p95: Optional[float] = None
    ttft_p99: Optional[float] = None

    # 单 token 生成延迟
    tpot_mean: Optional[float] = None
    tpot_p50: Optional[float] = None
    tpot_p95: Optional[float] = None
    tpot_p99: Optional[float] = None

    # 吞吐
    output_tokens_total: int = 0
    input_tokens_total: int = 0
    # 服务整体输出吞吐（tokens/s），基于墙壁时钟计算
    throughput_output: Optional[float] = None
    throughput_total: Optional[float] = None
    # 每路请求的平均输出速率
    per_request_throughput_mean: Optional[float] = None

    # 墙壁时钟：从首个请求发出到最后一个请求完成
    wall_time: Optional[float] = None
    # 请求速率
    request_rate: Optional[float] = None

    # 失败率
    failure_rate: float = 0.0
    # 该档位是否被视为不稳定
    unstable: bool = False
    errors: List[str] = field(default_factory=list)
    # 失败原因归类计数
    error_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典（用于落盘与上报）。"""
        return {
            "concurrency": self.concurrency,
            "total_requests": self.total_requests,
            "success_requests": self.success_requests,
            "failed_requests": self.failed_requests,
            "failure_rate": self.failure_rate,
            "unstable": self.unstable,
            "latency": {
                "mean": self.latency_mean,
                "p50": self.latency_p50,
                "p95": self.latency_p95,
                "p99": self.latency_p99,
                "min": self.latency_min,
                "max": self.latency_max,
                "stdev": self.latency_stdev,
            },
            "ttft": {
                "mean": self.ttft_mean,
                "p50": self.ttft_p50,
                "p95": self.ttft_p95,
                "p99": self.ttft_p99,
            },
            "tpot": {
                "mean": self.tpot_mean,
                "p50": self.tpot_p50,
                "p95": self.tpot_p95,
                "p99": self.tpot_p99,
            },
            "throughput": {
                "output_tokens_total": self.output_tokens_total,
                "input_tokens_total": self.input_tokens_total,
                "output_tokens_per_second": self.throughput_output,
                "total_tokens_per_second": self.throughput_total,
                "per_request_output_tps_mean": self.per_request_throughput_mean,
            },
            "timing": {
                "wall_time": self.wall_time,
                "request_rate": self.request_rate,
            },
            "errors": {
                "count": self.failed_requests,
                "breakdown": self.error_breakdown,
                "samples": self.errors[:MAX_ERROR_SAMPLES],
            },
        }


def _classify_error(error: Optional[str]) -> str:
    """把错误信息归到便于统计的类别。"""
    if not error:
        return "unknown"
    text = error.lower()
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "connection" in text or "refused" in text or "unreachable" in text:
        return "connection"
    if "memory" in text or "oom" in text:
        return "oom"
    if "rate" in text and "limit" in text:
        return "rate_limit"
    if "500" in text or "internal" in text:
        return "server_error"
    if "400" in text or "invalid" in text:
        return "bad_request"
    return "other"


def aggregate(
    results: List[RequestResult],
    concurrency: int,
    wall_time: Optional[float] = None,
    max_failure_rate: float = 0.2,
) -> StageMetrics:
    """把一组请求结果聚合为档位指标。

    参数
    ----
    results:
        该并发档位下全部请求的观测结果（含失败请求）。
    concurrency:
        该档位的并发数。
    wall_time:
        墙壁时钟总耗时。**建议传入实测值**；未传入时退化为
        「所有成功请求延迟之和 / 并发数」的估算。
        实测值更准确，因为它包含了调度与排队开销。
    max_failure_rate:
        失败率超过该阈值时把档位标记为 unstable。压测中若服务
        大面积失败，其延迟数据已不可信，报告里应显式提示。
    """
    metrics = StageMetrics(concurrency=concurrency, total_requests=len(results))

    if not results:
        return metrics

    ok = [r for r in results if r.success]
    metrics.success_requests = len(ok)
    metrics.failed_requests = len(results) - len(ok)
    metrics.failure_rate = round(metrics.failed_requests / len(results), 4)
    metrics.unstable = metrics.failure_rate > max_failure_rate

    for r in results:
        if not r.success:
            # 只留前 5 条样本：一次全失败的档位可能产生上千条错误，
            # 全存进结果文件会让 JSON 膨胀几十倍，而排查看前几条就够。
            if len(metrics.errors) < MAX_ERROR_SAMPLES:
                metrics.errors.append(r.error or "unknown error")
            key = _classify_error(r.error)
            metrics.error_breakdown[key] = metrics.error_breakdown.get(key, 0) + 1

    if not ok:
        return metrics

    # ---- 端到端延迟 ----
    latencies = [r.total_latency for r in ok if r.total_latency is not None]
    if latencies:
        metrics.latency_mean = _safe_mean(latencies)
        metrics.latency_p50 = percentile(latencies, 50)
        metrics.latency_p95 = percentile(latencies, 95)
        metrics.latency_p99 = percentile(latencies, 99)
        metrics.latency_min = min(latencies)
        metrics.latency_max = max(latencies)
        metrics.latency_stdev = _safe_stdev(latencies)

    # ---- 首 token 延迟 ----
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    if ttfts:
        metrics.ttft_mean = _safe_mean(ttfts)
        metrics.ttft_p50 = percentile(ttfts, 50)
        metrics.ttft_p95 = percentile(ttfts, 95)
        metrics.ttft_p99 = percentile(ttfts, 99)

    # ---- 单 token 生成延迟 ----
    tpots = [r.tpot for r in ok if r.tpot is not None]
    if tpots:
        metrics.tpot_mean = _safe_mean(tpots)
        metrics.tpot_p50 = percentile(tpots, 50)
        metrics.tpot_p95 = percentile(tpots, 95)
        metrics.tpot_p99 = percentile(tpots, 99)

    # ---- 吞吐 ----
    metrics.output_tokens_total = sum(r.output_tokens for r in ok)
    metrics.input_tokens_total = sum(r.input_tokens for r in ok)

    effective_wall = wall_time
    if effective_wall is None:
        # 退化估算：总延迟 / 并发。仅在拿不到实测墙壁时钟时使用。
        effective_wall = sum(latencies) / concurrency if latencies else None

    if effective_wall and effective_wall > 0:
        metrics.wall_time = effective_wall
        metrics.throughput_output = metrics.output_tokens_total / effective_wall
        metrics.throughput_total = (
            metrics.output_tokens_total + metrics.input_tokens_total
        ) / effective_wall
        metrics.request_rate = metrics.success_requests / effective_wall

    per_req = [r.throughput for r in ok if r.throughput is not None]
    metrics.per_request_throughput_mean = _safe_mean(per_req)

    return metrics


def summarize_run(stages: List[StageMetrics]) -> Dict[str, Any]:
    """跨档位汇总，用于报告结论段。"""
    valid = [s for s in stages if s.success_requests > 0]
    if not valid:
        return {
            "best_throughput": None,
            "best_throughput_concurrency": None,
            "lowest_ttft": None,
            "lowest_ttft_concurrency": None,
            "max_concurrency_tested": max((s.concurrency for s in stages), default=None),
            "unstable_stages": [s.concurrency for s in stages if s.unstable],
        }

    with_tp = [s for s in valid if s.throughput_output is not None]
    best_tp = max(with_tp, key=lambda s: s.throughput_output) if with_tp else None

    with_ttft = [s for s in valid if s.ttft_mean is not None]
    best_ttft = min(with_ttft, key=lambda s: s.ttft_mean) if with_ttft else None

    return {
        "best_throughput": best_tp.throughput_output if best_tp else None,
        "best_throughput_concurrency": best_tp.concurrency if best_tp else None,
        "lowest_ttft": best_ttft.ttft_mean if best_ttft else None,
        "lowest_ttft_concurrency": best_ttft.concurrency if best_ttft else None,
        "max_concurrency_tested": max(s.concurrency for s in stages),
        "unstable_stages": [s.concurrency for s in stages if s.unstable],
    }
