"""压测执行器测试（用假引擎，不依赖真实推理服务）。

通过 ``httpx.MockTransport`` 拦截 HTTP 请求，模拟一个 OpenAI 兼容
的流式推理服务。这样才能在没有 GPU、没有 vLLM 的机器上验证：

- 并发档位扫描是否按梯度执行
- TTFT / TPOT 是否从流式分片里被正确测量
- 预热请求是否真的不计入统计
- 失败请求是否被正确归类而不是让整个压测崩掉
"""

from __future__ import annotations

import json
import time
from typing import Callable, List

import httpx
import pytest

from bench.runner import LoadRunner
from config.settings import BenchConfig, EngineTarget, MonitorConfig
from engines.client import UnifiedLLMClient


# ----------------------------------------------------------------------
# 假引擎
# ----------------------------------------------------------------------
def _sse_chunk(content: str) -> bytes:
    return f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n".encode()


def _sse_usage(prompt_tokens: int, completion_tokens: int) -> bytes:
    payload = {
        "choices": [],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def make_stream_transport(
    tokens: int = 10,
    ttft_seconds: float = 0.05,
    per_token_seconds: float = 0.01,
    fail_first: int = 0,
    fail_status: int = 500,
    counter: dict | None = None,
) -> httpx.MockTransport:
    """构造一个模拟流式推理服务的 transport。

    Args:
        tokens: 每次请求输出的 token 数。
        ttft_seconds: 首 token 之前的延迟。
        per_token_seconds: 相邻 token 的间隔。
        fail_first: 前 N 次请求返回错误（用于测失败路径）。
        fail_status: 失败时返回的 HTTP 状态码。
        counter: 外部传入的计数器。``run_stage`` 会为每个并发通道建一个
            客户端（现在只在档位开始时建一次），「前 N 次失败」的计数
            仍需跨 transport 实例共享。
    """
    state = counter if counter is not None else {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1

        if state["calls"] <= fail_first:
            return httpx.Response(
                fail_status, json={"error": {"message": "boom"}}
            )

        def stream():
            time.sleep(ttft_seconds)
            for i in range(tokens):
                if i > 0:
                    time.sleep(per_token_seconds)
                yield _sse_chunk(f"tok{i} ")
            yield _sse_usage(prompt_tokens=20, completion_tokens=tokens)
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, content=stream(), headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


def make_client(
    transport: httpx.MockTransport,
    model: str = "test-model",
    timeout: float = 30.0,
) -> UnifiedLLMClient:
    """构造使用假 transport 的客户端。"""
    target = EngineTarget(name="mock", base_url="http://mock.local", model=model)
    client = UnifiedLLMClient(target, timeout=timeout)
    # 换掉内部 httpx client，让请求走 MockTransport
    client._client = client._client.__class__(
        base_url="http://mock.local/v1",
        api_key="EMPTY",
        timeout=timeout,
        max_retries=0,
        http_client=httpx.Client(transport=transport, base_url="http://mock.local/v1"),
    )
    return client


# ----------------------------------------------------------------------
# 客户端计时
# ----------------------------------------------------------------------
class TestClientTiming:
    def test_stream_measures_ttft_and_tpot(self):
        """TTFT 必须明显小于总延迟，且 TPOT 与 token 间隔同量级。

        注意：这里不能断言 TTFT ≈ 设定的 sleep 时间。httpx.MockTransport
        用生成器模拟流式时会做内部缓冲，首片到达时刻会晚于生成器里的
        sleep。所以改为验证**结构性质**：TTFT 是总延迟的一部分、
        且 TPOT 与设定的 per_token_seconds 同数量级。
        精确的计时验证在真实服务上做。
        """
        client = make_client(
            make_stream_transport(tokens=10, ttft_seconds=0.05, per_token_seconds=0.01)
        )
        result = client.complete("hi", max_tokens=10, stream=True)

        assert result.success is True
        assert result.output_tokens == 10
        assert result.ttft is not None
        assert result.generation_time is not None
        # TTFT 是总延迟的一部分（含首 token 前的等待）
        assert 0 < result.ttft < result.total_latency
        # 生成阶段至少覆盖后 9 个 token
        assert result.generation_time > 0
        # TTFT + 生成时间 应约等于总延迟
        assert result.ttft + result.generation_time == pytest.approx(
            result.total_latency, abs=0.02
        )
        # TPOT 与 token 间隔同量级（MockTransport 有缓冲，放宽到 10 倍）
        assert 0 < result.tpot < 0.01 * 10

    def test_stream_uses_server_usage(self):
        """服务端回了 usage 时，token 数应采信 usage 而不是分片计数。"""
        def handler(request: httpx.Request) -> httpx.Response:
            def stream():
                yield _sse_chunk("a")
                # usage 报 99，但实际只有 1 个内容分片
                yield _sse_usage(prompt_tokens=7, completion_tokens=99)
                yield b"data: [DONE]\n\n"

            return httpx.Response(200, content=stream())

        client = make_client(httpx.MockTransport(handler))
        result = client.complete("hi", stream=True)
        assert result.output_tokens == 99
        assert result.input_tokens == 7

    def test_non_streaming_has_no_ttft(self):
        """非流式只有端到端延迟，TTFT / TPOT 必须为 None。"""
        def handler(request: httpx.Request) -> httpx.Response:
            payload = {
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }
            return httpx.Response(200, json=payload)

        client = make_client(httpx.MockTransport(handler))
        result = client.complete("hi", stream=False)

        assert result.success is True
        assert result.ttft is None
        assert result.tpot is None
        assert result.output_tokens == 3
        assert result.total_latency > 0

    def test_http_error_is_captured_not_raised(self):
        """5xx 不能抛出去，要变成 success=False 的结果。"""
        transport = make_stream_transport(fail_first=1, fail_status=500)
        client = make_client(transport)
        result = client.complete("hi", stream=True)

        assert result.success is False
        assert result.error
        assert "500" in result.error or "server" in result.error.lower()

    def test_bad_request_is_captured(self):
        """400 也要被捕获（例如模型名写错）。"""
        transport = make_stream_transport(fail_first=1, fail_status=400)
        result = make_client(transport).complete("hi", stream=True)
        assert result.success is False
        assert "400" in result.error or "invalid" in result.error.lower()

    def test_health_ok(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"id": "test-model"}, {"id": "other"}]})

        target = EngineTarget(name="mock", base_url="http://mock.local", model="test-model")
        client = UnifiedLLMClient(target)
        # health() 用独立的 httpx.Client，这里直接测解析逻辑
        assert client._model_present(["test-model", "other"]) is True

    def test_model_present_prefix_matching(self):
        """模型名带后缀（-AWQ）时也要能匹配上。"""
        target = EngineTarget(
            name="mock", base_url="http://mock.local", model="Qwen2.5-72B-Instruct"
        )
        client = UnifiedLLMClient(target)
        assert client._model_present(["Qwen2.5-72B-Instruct-AWQ"]) is True
        assert client._model_present(["other-model"]) is False
        assert client._model_present([]) is False

    def test_base_url_trailing_slash_normalized(self):
        """base_url 结尾带斜杠不能拼出 //v1。"""
        target = EngineTarget(name="mock", base_url="http://mock.local/", model="m")
        assert target.normalized_base_url() == "http://mock.local"


# ----------------------------------------------------------------------
# 压测执行器
# ----------------------------------------------------------------------
class TestLoadRunner:
    def _bench(self, **kwargs) -> BenchConfig:
        defaults = dict(
            concurrency_levels=[1, 2, 4],
            warmup_requests=0,
            max_tokens=10,
            timeout=30.0,
            stream=True,
            prompt="hi",
            max_failure_rate=0.2,
        )
        defaults.update(kwargs)
        return BenchConfig(**defaults)

    def _runner(
        self, transport, bench=None, monitor=None, fail_first: int = 0, tokens: int = 10
    ) -> LoadRunner:
        """构造一个把内部客户端替换为假 transport 的 LoadRunner。

        关键点：``run_stage`` 为每个请求新建客户端，所以不能复用同一个
        transport 实例（否则失败计数会被重置）。这里用一个**共享计数器**
        生成全新 transport，让「前 N 次请求失败」在并发下也成立。
        """
        target = EngineTarget(name="mock", base_url="http://mock.local", model="test-model")
        runner = LoadRunner(target, bench or self._bench(), monitor_config=monitor)

        import bench.runner as runner_mod

        original = runner_mod.UnifiedLLMClient
        shared = {"calls": 0}

        def patched(t, timeout=120.0):
            c = original(t, timeout=timeout)
            tr = (
                make_stream_transport(
                    tokens=tokens, fail_first=fail_first, counter=shared
                )
                if fail_first
                else transport
            )
            c._client = c._client.__class__(
                base_url=t.normalized_base_url() + "/v1",
                api_key="EMPTY",
                timeout=timeout,
                max_retries=0,
                http_client=httpx.Client(
                    transport=tr, base_url=t.normalized_base_url() + "/v1"
                ),
            )
            return c

        runner_mod.UnifiedLLMClient = patched
        runner._restore = lambda: setattr(runner_mod, "UnifiedLLMClient", original)
        return runner

    def test_single_stage_produces_metrics(self):
        runner = self._runner(make_stream_transport(tokens=10))
        try:
            stage = runner.run_stage(2)
        finally:
            runner._restore()

        assert stage.concurrency == 2
        # 默认样本量是 max(16, 2*concurrency)，不再是「等于并发数」
        assert stage.total_requests == 16
        assert stage.success_requests == 16
        assert stage.failure_rate == 0.0
        assert stage.ttft_mean is not None
        assert stage.tpot_mean is not None
        assert stage.throughput_output is not None
        assert stage.wall_time is not None

    def test_requests_scale_with_concurrency(self):
        """requests_per_level 未设时，请求数取 max(16, 2*并发)。

        旧行为是「等于并发数」，导致并发 1 的档位只有 1 个样本，
        P99 退化成最大值却仍被当作尾延迟结论。
        """
        runner = self._runner(make_stream_transport(tokens=5))
        try:
            stage = runner.run_stage(4)
        finally:
            runner._restore()
        assert stage.total_requests == 16

    def test_default_sample_size_scales_past_the_floor(self):
        runner = self._runner(make_stream_transport(tokens=3))
        try:
            stage = runner.run_stage(16)
        finally:
            runner._restore()
        assert stage.total_requests == 32

    def test_requests_per_level_override(self):
        runner = self._runner(
            make_stream_transport(tokens=5), bench=self._bench(requests_per_level=8)
        )
        try:
            stage = runner.run_stage(2)
        finally:
            runner._restore()
        assert stage.total_requests == 8

    def test_requests_per_level_below_concurrency_is_clamped(self):
        """请求数不能少于并发数，否则有空闲通道，测不出真实并发。"""
        runner = self._runner(
            make_stream_transport(tokens=5), bench=self._bench(requests_per_level=1)
        )
        try:
            stage = runner.run_stage(8)
        finally:
            runner._restore()
        assert stage.total_requests == 8

    def test_sweep_runs_all_levels_in_order(self):
        runner = self._runner(make_stream_transport(tokens=5))
        seen: List[int] = []
        try:
            stages = runner.run_sweep(levels=[4, 1, 2], on_stage=lambda s: seen.append(s.concurrency))
        finally:
            runner._restore()

        assert [s.concurrency for s in stages] == [1, 2, 4]
        assert seen == [1, 2, 4]

    def test_warmup_requests_excluded_from_stats(self):
        """预热请求不能计入 total_requests。"""
        runner = self._runner(
            make_stream_transport(tokens=5), bench=self._bench(warmup_requests=3)
        )
        try:
            stage = runner.run_stage(2)
        finally:
            runner._restore()
        assert stage.total_requests == 16  # 预热 3 次被丢弃，16 个有效请求

    def test_failures_recorded_and_classified(self):
        """部分请求失败时，失败要进统计且档位可被标记不稳定。

        ``warmup_requests=0`` 且 ``requests_per_level=2`` 时两个请求都是
        有效请求，前 2 次失败会让该档位 100% 失败并被标记不稳定。
        """
        runner = self._runner(
            None,
            bench=self._bench(
                warmup_requests=0, max_failure_rate=0.2, requests_per_level=2
            ),
            fail_first=2,
            tokens=5,
        )
        try:
            stage = runner.run_stage(2)
        finally:
            runner._restore()

        assert stage.total_requests == 2
        assert stage.failed_requests == 2
        assert stage.success_requests == 0
        assert stage.failure_rate == 1.0
        assert stage.unstable is True
        assert stage.errors
        assert stage.error_breakdown

    def test_partial_failure_rate(self):
        """4 个请求里 2 个失败 => 50% 失败率。"""
        runner = self._runner(
            None,
            bench=self._bench(warmup_requests=0, requests_per_level=4),
            fail_first=2,
            tokens=5,
        )
        try:
            stage = runner.run_stage(4)
        finally:
            runner._restore()

        assert stage.total_requests == 4
        assert stage.failed_requests == 2
        assert stage.failure_rate == pytest.approx(0.5)
        assert stage.success_requests == 2
        # 成功请求的延迟仍被统计
        assert stage.latency_mean is not None

    def test_connection_error_is_graceful(self):
        """目标不可达时不能崩，要产出全失败的档位指标。"""
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        runner = self._runner(httpx.MockTransport(handler))
        try:
            stage = runner.run_stage(2)
        finally:
            runner._restore()

        assert stage.total_requests == 16
        assert stage.success_requests == 0
        assert stage.failure_rate == 1.0
        assert stage.unstable is True
        assert stage.latency_mean is None
        assert "connection" in stage.error_breakdown

    def test_timeout_is_graceful(self):
        """超时要被归类为 timeout 而不是崩溃。"""
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        runner = self._runner(httpx.MockTransport(handler))
        try:
            stage = runner.run_stage(1)
        finally:
            runner._restore()

        assert stage.success_requests == 0
        assert "timeout" in stage.error_breakdown

    def test_vram_none_without_monitor(self):
        """没配监控时，显存曲线返回 unavailable。"""
        runner = self._runner(make_stream_transport(tokens=5))
        try:
            runner.run_sweep(levels=[1])
        finally:
            runner._restore()
        curve = runner.vram_curve()
        assert curve["available"] is False

    def test_vram_curve_with_monitor_config(self):
        """配了监控但环境无 GPU 时，曲线应优雅降级而不是报错。"""
        runner = self._runner(
            make_stream_transport(tokens=5),
            bench=self._bench(),
            monitor=MonitorConfig(interval=0.05, enabled=True),
        )
        try:
            runner.run_sweep(levels=[1])
        finally:
            runner._restore()

        curve = runner.vram_curve()
        assert "available" in curve
        assert "gpus" in curve
        # 无 GPU 环境必然不可用；有 GPU 时才有 series
        if not curve["available"]:
            assert curve["gpus"] == []


# ----------------------------------------------------------------------
# 端到端：压测 -> 落盘 -> 报告
# ----------------------------------------------------------------------
class TestEndToEndSanity:
    def test_stats_pipeline_consistent(self):
        """验证「墙钟吞吐 <= 单请求吞吐之和」这一物理约束。

        墙钟吞吐 = 总 token / 墙钟时间，单请求吞吐 = token / 该请求延迟。
        并发时墙钟时间通常大于单请求延迟，所以前者更小——若反过来，
        说明吞吐计算用了错误的时钟。
        """
        from metrics.stats import RequestResult, aggregate

        results = [
            RequestResult(
                success=True,
                total_latency=2.0,
                ttft=0.2,
                generation_time=1.8,
                output_tokens=100,
                input_tokens=20,
            )
            for _ in range(4)
        ]
        m = aggregate(results, concurrency=4, wall_time=2.2)

        per_request_total = sum(r.throughput for r in results)
        assert per_request_total == pytest.approx(200.0)  # 4 * 50
        assert m.throughput_output == pytest.approx(400 / 2.2)
        assert m.throughput_output < per_request_total




class TestClientLifecycle:
    """客户端构造必须发生在计时窗口之外。"""

    def test_client_construction_is_not_counted_in_wall_time(self):
        import bench.runner as runner_mod
        import time as _time

        original = runner_mod.UnifiedLLMClient
        transport = make_stream_transport(
            tokens=2, ttft_seconds=0.0, per_token_seconds=0.0
        )
        target = EngineTarget(name="vllm", base_url="http://mock:8000", model="mock")

        def patched(t, timeout=120.0):
            _time.sleep(2.0)  # 模拟建连接 / SSL context 的开销
            client = original(t, timeout=timeout)
            client._client = client._client.__class__(
                base_url=t.api_base(),
                api_key="EMPTY",
                timeout=timeout,
                max_retries=0,
                http_client=httpx.Client(transport=transport, base_url=t.api_base()),
            )
            return client

        bench = BenchConfig(
            concurrency_levels=[2],
            requests_per_level=8,
            warmup_requests=0,
            stream=True,
            prompt="hi",
        )
        runner_mod.UnifiedLLMClient = patched
        try:
            runner = LoadRunner(target, bench)
            stage = runner.run_stage(2)
        finally:
            runner_mod.UnifiedLLMClient = original

        assert stage.total_requests == 8
        # 两个客户端各睡 2 秒；若构造进入计时窗口，wall_time 必然 > 4 秒
        assert stage.wall_time < 3.0

    def test_empty_stream_counts_as_failure(self):
        """200 但没有 content token 的流不能算成功。"""
        nl = chr(10)
        payload = "".join(
            [
                'data: {"choices":[{"delta":{},"finish_reason":null}]}' + nl + nl,
                'data: {"choices":[],"usage":{"completion_tokens":0,"prompt_tokens":5}}'
                + nl
                + nl,
                "data: [DONE]" + nl + nl,
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=payload
            )

        target = EngineTarget(name="vllm", base_url="http://mock:8000", model="mock")
        client = UnifiedLLMClient(target)
        client._client = client._client.__class__(
            base_url=target.api_base(),
            api_key="EMPTY",
            timeout=5,
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(handler), base_url=target.api_base()
            ),
        )
        try:
            res = client.complete(
                prompt="hi", max_tokens=8, temperature=0.0, stream=True
            )
        finally:
            client.close()

        assert res.success is False
        assert "empty stream" in (res.error or "")
