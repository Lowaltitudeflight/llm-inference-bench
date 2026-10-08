"""Prometheus 指标注册与暴露。

把压测过程中的关键指标暴露成 Prometheus 格式，让 Grafana 能画实时面板。
指标设计围绕「一个 Grafana 面板就能看出当前压测在干什么」：

======================  ===================================
指标名                   含义
======================  ===================================
``llmbench_requests_total``          请求计数（按引擎/结果）
``llmbench_request_latency_seconds`` 端到端延迟直方图
``llmbench_ttft_seconds``            首包延迟直方图
``llmbench_tpot_seconds``            单 token 延迟直方图
``llmbench_throughput_tokens_per_second`` 各档位吞吐（Gauge）
``llmbench_concurrency``             当前并发档位
``llmbench_gpu_memory_used_mib``     GPU 显存占用
``llmbench_gpu_utilization_percent`` GPU 利用率
``llmbench_stage_failure_rate``      档位失败率
``llmbench_run_info``                当前运行的元信息（Info）
======================  ===================================

直方图分桶按大模型推理的真实量级设置：TTFT 从 10ms 到 30s，
TPOT 从 5ms 到 500ms。默认分桶会把这些全部塞进一个桶，看不出分布。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info

from metrics.stats import RequestResult, StageMetrics

logger = logging.getLogger(__name__)

# 延迟分桶：覆盖 10ms ~ 60s
LATENCY_BUCKETS = (
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0
)

# TTFT 分桶：大模型首包通常 50ms ~ 5s
TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.0)

# TPOT 分桶：单 token 通常 5ms ~ 500ms
TPOT_BUCKETS = (0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28)


class BenchMetrics:
    """压测指标集合。

    每个实例自带独立 Registry，便于测试时互不干扰；
    ``collect()`` 可拿到供 :class:`prometheus_client` 直接暴露的输出。
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        self.requests_total = Counter(
            "llmbench_requests_total",
            "压测请求总数",
            ["engine", "model", "quantization", "status"],
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "llmbench_request_latency_seconds",
            "端到端请求延迟",
            ["engine", "model", "quantization"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.ttft = Histogram(
            "llmbench_ttft_seconds",
            "首 token 延迟（TTFT）",
            ["engine", "model", "quantization"],
            buckets=TTFT_BUCKETS,
            registry=self.registry,
        )
        self.tpot = Histogram(
            "llmbench_tpot_seconds",
            "单 token 生成延迟（TPOT）",
            ["engine", "model", "quantization"],
            buckets=TPOT_BUCKETS,
            registry=self.registry,
        )
        self.throughput = Gauge(
            "llmbench_throughput_tokens_per_second",
            "档位输出吞吐",
            ["engine", "model", "quantization", "concurrency"],
            registry=self.registry,
        )
        self.concurrency = Gauge(
            "llmbench_concurrency",
            "当前并发档位",
            ["engine", "model", "quantization"],
            registry=self.registry,
        )
        self.stage_failure_rate = Gauge(
            "llmbench_stage_failure_rate",
            "档位失败率（0~1）",
            ["engine", "model", "quantization", "concurrency"],
            registry=self.registry,
        )
        self.ttft_p99 = Gauge(
            "llmbench_ttft_p99_seconds",
            "档位 TTFT P99",
            ["engine", "model", "quantization", "concurrency"],
            registry=self.registry,
        )
        self.latency_p99 = Gauge(
            "llmbench_latency_p99_seconds",
            "档位端到端 P99",
            ["engine", "model", "quantization", "concurrency"],
            registry=self.registry,
        )
        self.gpu_memory_used = Gauge(
            "llmbench_gpu_memory_used_mib",
            "GPU 显存已用",
            ["gpu", "name"],
            registry=self.registry,
        )
        self.gpu_memory_total = Gauge(
            "llmbench_gpu_memory_total_mib",
            "GPU 显存总量",
            ["gpu", "name"],
            registry=self.registry,
        )
        self.gpu_utilization = Gauge(
            "llmbench_gpu_utilization_percent",
            "GPU 利用率",
            ["gpu", "name"],
            registry=self.registry,
        )
        self.gpu_power = Gauge(
            "llmbench_gpu_power_watts",
            "GPU 功耗",
            ["gpu", "name"],
            registry=self.registry,
        )
        self.run_info = Info(
            "llmbench_run",
            "当前压测运行的元信息",
            registry=self.registry,
        )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def observe_request(
        self, result: RequestResult, engine: str, model: str, quantization: str
    ) -> None:
        """记录单次请求。"""
        status = "success" if result.success else "failure"
        labels = (engine, model, quantization)
        self.requests_total.labels(*labels, status).inc()

        if not result.success:
            return
        if result.total_latency is not None:
            self.request_latency.labels(*labels).observe(result.total_latency)
        if result.ttft is not None:
            self.ttft.labels(*labels).observe(result.ttft)
        if result.tpot is not None:
            self.tpot.labels(*labels).observe(result.tpot)

    def observe_stage(
        self, stage: StageMetrics, engine: str, model: str, quantization: str
    ) -> None:
        """记录一个并发档位的聚合结果。"""
        conc = str(stage.concurrency)
        labels = (engine, model, quantization)
        self.concurrency.labels(*labels).set(stage.concurrency)

        if stage.throughput_output is not None:
            self.throughput.labels(*labels, conc).set(stage.throughput_output)
        self.stage_failure_rate.labels(*labels, conc).set(stage.failure_rate)
        if stage.ttft_p99 is not None:
            self.ttft_p99.labels(*labels, conc).set(stage.ttft_p99)
        if stage.latency_p99 is not None:
            self.latency_p99.labels(*labels, conc).set(stage.latency_p99)

    def observe_gpu(self, samples: List[Any]) -> None:
        """记录一批 GPU 采样。

        Args:
            samples: :class:`monitor.gpu.GpuSample` 列表。
        """
        for s in samples:
            labels = (str(s.index), s.name or "unknown")
            if s.memory_used_mib is not None:
                self.gpu_memory_used.labels(*labels).set(s.memory_used_mib)
            if s.memory_total_mib is not None:
                self.gpu_memory_total.labels(*labels).set(s.memory_total_mib)
            if s.util_percent is not None:
                self.gpu_utilization.labels(*labels).set(s.util_percent)
            if s.power_watts is not None:
                self.gpu_power.labels(*labels).set(s.power_watts)

    def set_run_info(self, **fields: Any) -> None:
        """写入运行元信息（字符串化后上报）。

        必须**一次性**传入全部字段：``Info.info()`` 每次调用会整体替换
        标签集合，如果按字段逐个调用，最后只会剩下最后一个字段。
        """
        self.run_info.clear()
        if not fields:
            return
        self.run_info.info(
            {key: "" if value is None else str(value) for key, value in fields.items()}
        )

    def reset(self) -> None:
        """清空所有指标（新一轮压测前调用）。"""
        new = BenchMetrics(self.registry)
        self.__dict__.update(new.__dict__)


# 全局单例：API 服务直接 import 使用
METRICS = BenchMetrics()


def render_latest() -> bytes:
    """渲染 Prometheus 文本格式（供 /metrics 端点使用）。"""
    from prometheus_client import generate_latest

    return generate_latest(METRICS.registry)


def snapshot() -> Dict[str, Any]:
    """把关键指标汇总成 JSON，供 API /status 使用。

    只取「看板要看的那几个数」，不做全量导出——全量交给 /metrics。
    """
    out: Dict[str, Any] = {"counters": {}, "gauges": {}}
    for metric in METRICS.registry.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                out["counters"][sample.name] = out["counters"].get(sample.name, 0) + sample.value
            elif sample.name.endswith(("_gauge", "_seconds", "_mib", "_percent", "_watts", "_rate", "_per_second")):
                key = sample.name
                if sample.labels:
                    key = f"{sample.name}{{{','.join(f'{k}={v}' for k, v in sorted(sample.labels.items()))}}}"
                out["gauges"][key] = sample.value
    return out


# 供 prometheus_client 直接挂载的 ASGI app
def make_metrics_app():
    """返回可直接 mount 的 Prometheus ASGI 应用。"""
    from prometheus_client import make_asgi_app

    return make_asgi_app(registry=METRICS.registry)
