"""指标计算测试。

重点验证三件事：
1. **TPOT 的分母**是 ``output_tokens - 1`` 而不是 ``output_tokens``
   ——首 token 的耗时已计入 TTFT，重复计入会低估 TPOT。
2. **分位数用线性插值**，小样本下不能取到极端值。
3. **失败率超阈值时档位被标记 unstable**，避免拿不可信数据下结论。
"""

from __future__ import annotations

import math

import pytest

from metrics.stats import (
    RequestResult,
    StageMetrics,
    aggregate,
    percentile,
    summarize_run,
)


# ----------------------------------------------------------------------
# RequestResult
# ----------------------------------------------------------------------
class TestRequestResult:
    def test_tpot_excludes_first_token(self):
        """TPOT = 生成阶段耗时 / (输出 token - 1)。

        生成 5 个 token、生成阶段耗时 0.4s（已扣除 TTFT），
        则 TPOT = 0.4 / 4 = 0.1s，而不是 0.4 / 5 = 0.08s。
        """
        r = RequestResult(
            success=True,
            total_latency=0.5,
            ttft=0.1,
            generation_time=0.4,
            output_tokens=5,
        )
        assert r.tpot == pytest.approx(0.1)

    def test_tpot_none_when_single_token(self):
        """只生成 1 个 token 时无法计算 TPOT（分母为 0）。"""
        r = RequestResult(
            success=True, total_latency=0.2, ttft=0.2, generation_time=0.0, output_tokens=1
        )
        assert r.tpot is None

    def test_tpot_none_without_generation_time(self):
        """非流式请求没有 generation_time，TPOT 应为 None。"""
        r = RequestResult(success=True, total_latency=1.0, output_tokens=50)
        assert r.tpot is None

    def test_tpot_none_on_zero_tokens(self):
        r = RequestResult(
            success=True, total_latency=0.5, ttft=0.1, generation_time=0.4, output_tokens=0
        )
        assert r.tpot is None

    def test_throughput(self):
        """吞吐 = 输出 token / 端到端延迟。"""
        r = RequestResult(success=True, total_latency=2.0, output_tokens=100)
        assert r.throughput == pytest.approx(50.0)

    def test_throughput_none_on_zero_latency(self):
        r = RequestResult(success=True, total_latency=0.0, output_tokens=10)
        assert r.throughput is None

    def test_throughput_none_on_zero_tokens(self):
        r = RequestResult(success=True, total_latency=1.0, output_tokens=0)
        assert r.throughput is None

    def test_failed_request_has_no_derived_metrics(self):
        r = RequestResult(success=False, error="timeout")
        assert r.tpot is None
        assert r.throughput is None


# ----------------------------------------------------------------------
# percentile
# ----------------------------------------------------------------------
class TestPercentile:
    def test_empty_returns_none(self):
        assert percentile([], 50) is None

    def test_single_value(self):
        assert percentile([3.0], 99) == 3.0

    def test_median_odd_count(self):
        """奇数个元素的 P50 就是中间那个。"""
        assert percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_p50_linear_interpolation(self):
        """偶数个元素：P50 落在 2 和 3 之间，取中点 2.5。"""
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)

    def test_p0_is_min(self):
        assert percentile([5.0, 1.0, 9.0], 0) == 1.0

    def test_p100_is_max(self):
        assert percentile([5.0, 1.0, 9.0], 100) == 9.0

    def test_p25_linear(self):
        # 4 个元素，P25 位置 = 0.25*3 = 0.75 -> 1 + 0.75*(2-1) = 1.75
        assert percentile([1.0, 2.0, 3.0, 4.0], 25) == pytest.approx(1.75)

    def test_p99_small_sample_not_extreme(self):
        """小样本下 P99 不应是最大值本身——线性插值给出 98。"""
        values = [float(i) for i in range(1, 101)]
        p99 = percentile(values, 99)
        assert p99 == pytest.approx(99.01, abs=0.05)
        assert p99 < 100.0

    def test_unordered_input(self):
        assert percentile([9.0, 1.0, 5.0], 50) == 5.0

    def test_duplicates(self):
        assert percentile([2.0, 2.0, 2.0, 2.0], 75) == 2.0

    @pytest.mark.parametrize("bad", [-1, 101, 1000])
    def test_invalid_p_raises(self, bad):
        with pytest.raises(ValueError):
            percentile([1.0, 2.0], bad)


# ----------------------------------------------------------------------
# aggregate
# ----------------------------------------------------------------------
def _ok(latency: float, ttft: float = 0.1, output_tokens: int = 11) -> RequestResult:
    """构造一个成功的流式请求结果。

    默认生成 10 个解码 token（output_tokens=11，首 token 占 1 个），
    generation_time = latency - ttft，TPOT = generation_time / 10。
    """
    return RequestResult(
        success=True,
        total_latency=latency,
        ttft=ttft,
        generation_time=latency - ttft,
        output_tokens=output_tokens,
        input_tokens=20,
    )


class TestAggregate:
    def test_empty_results(self):
        m = aggregate([], concurrency=4, wall_time=1.0)
        assert m.total_requests == 0
        assert m.success_requests == 0
        assert m.concurrency == 4

    def test_all_success(self):
        results = [_ok(1.0), _ok(2.0), _ok(3.0), _ok(4.0)]
        m = aggregate(results, concurrency=4, wall_time=2.0)

        assert m.total_requests == 4
        assert m.success_requests == 4
        assert m.failed_requests == 0
        assert m.failure_rate == 0.0
        assert m.unstable is False
        assert m.latency_mean == pytest.approx(2.5)
        assert m.latency_min == 1.0
        assert m.latency_max == 4.0

    def test_throughput_uses_wall_time(self):
        """吞吐必须用墙壁时钟，而不是「延迟之和 / 并发」的估算。"""
        results = [_ok(1.0) for _ in range(8)]  # 每个产出 11 token
        m = aggregate(results, concurrency=8, wall_time=2.0)
        assert m.output_tokens_total == 88
        assert m.throughput_output == pytest.approx(44.0)  # 88 / 2.0
        assert m.wall_time == 2.0

    def test_throughput_falls_back_when_wall_time_missing(self):
        """没给 wall_time 时退化为 总延迟/并发 的估算，但不该崩。"""
        results = [_ok(2.0) for _ in range(4)]
        m = aggregate(results, concurrency=4, wall_time=None)
        assert m.wall_time == pytest.approx(2.0)  # 4*2/4
        assert m.throughput_output == pytest.approx(44 / 2.0)

    def test_request_rate(self):
        results = [_ok(1.0) for _ in range(10)]
        m = aggregate(results, concurrency=10, wall_time=2.0)
        assert m.request_rate == pytest.approx(5.0)

    def test_tpot_aggregated(self):
        """TTFT=0.1，延迟 1.1 => generation=1.0，输出 11 token => TPOT=0.1。"""
        results = [_ok(1.1, ttft=0.1, output_tokens=11) for _ in range(4)]
        m = aggregate(results, concurrency=4, wall_time=1.1)
        assert m.tpot_mean == pytest.approx(0.1)

    def test_failures_counted_and_classified(self):
        results = [
            _ok(1.0),
            RequestResult(success=False, error="Request timed out after 120s"),
            RequestResult(success=False, error="Connection refused"),
            RequestResult(success=False, error="CUDA out of memory"),
        ]
        m = aggregate(results, concurrency=4, wall_time=2.0)

        assert m.success_requests == 1
        assert m.failed_requests == 3
        assert m.failure_rate == pytest.approx(0.75)
        assert m.error_breakdown == {"timeout": 1, "connection": 1, "oom": 1}

    def test_unstable_when_failure_rate_exceeds_threshold(self):
        """4 个请求里 1 个失败 => 25% > 20% 阈值 => 标记不稳定。"""
        results = [_ok(1.0)] * 3 + [RequestResult(success=False, error="boom")]
        m = aggregate(results, concurrency=4, wall_time=1.0, max_failure_rate=0.2)
        assert m.failure_rate == pytest.approx(0.25)
        assert m.unstable is True

    def test_stable_at_exactly_threshold(self):
        """失败率 == 阈值时不标记（用 >，不是 >=）。"""
        results = [_ok(1.0)] * 4 + [RequestResult(success=False, error="boom")]
        m = aggregate(results, concurrency=5, wall_time=1.0, max_failure_rate=0.2)
        assert m.failure_rate == pytest.approx(0.2)
        assert m.unstable is False

    def test_all_failed_still_returns_metrics(self):
        """全部失败时不能抛异常，且延迟字段保持 None。"""
        results = [RequestResult(success=False, error="Connection refused")] * 3
        m = aggregate(results, concurrency=3, wall_time=1.0)
        assert m.success_requests == 0
        assert m.failure_rate == 1.0
        assert m.unstable is True
        assert m.latency_mean is None
        assert m.throughput_output is None
        assert m.ttft_mean is None

    def test_error_samples_capped(self):
        """错误样本只留前 5 条，避免结果文件膨胀。"""
        results = [RequestResult(success=False, error=f"err {i}") for i in range(20)]
        m = aggregate(results, concurrency=20, wall_time=1.0)
        assert len(m.errors) == 5
        assert m.error_breakdown["other"] == 20

    def test_latency_ignores_failed_requests(self):
        """失败请求的耗时不应污染延迟统计。"""
        results = [
            _ok(1.0),
            _ok(3.0),
            RequestResult(success=False, total_latency=999.0, error="timeout"),
        ]
        m = aggregate(results, concurrency=3, wall_time=2.0)
        assert m.latency_mean == pytest.approx(2.0)
        assert m.latency_max == 3.0

    def test_non_streaming_has_no_ttft(self):
        """非流式请求 TTFT 为 None，聚合时该组字段应保持 None。"""
        results = [
            RequestResult(success=True, total_latency=1.0, output_tokens=10, input_tokens=5),
            RequestResult(success=True, total_latency=2.0, output_tokens=10, input_tokens=5),
        ]
        m = aggregate(results, concurrency=2, wall_time=2.0)
        assert m.ttft_mean is None
        assert m.tpot_mean is None
        assert m.latency_mean == pytest.approx(1.5)

    def test_per_request_throughput(self):
        results = [_ok(2.0, ttft=0.1, output_tokens=101)] * 2  # 101/2 = 50.5
        m = aggregate(results, concurrency=2, wall_time=2.0)
        assert m.per_request_throughput_mean == pytest.approx(50.5)

    def test_to_dict_shape(self):
        """to_dict 的嵌套结构是落盘契约，必须稳定。"""
        m = aggregate([_ok(1.0)], concurrency=1, wall_time=1.0)
        d = m.to_dict()
        assert set(d) >= {
            "concurrency",
            "total_requests",
            "success_requests",
            "failed_requests",
            "failure_rate",
            "unstable",
            "latency",
            "ttft",
            "tpot",
            "throughput",
            "timing",
            "errors",
        }
        assert set(d["latency"]) == {"mean", "p50", "p95", "p99", "min", "max", "stdev"}
        assert "output_tokens_per_second" in d["throughput"]
        assert "wall_time" in d["timing"]

    def test_to_dict_roundtrip_through_quant(self):
        """to_dict 的产物要能被监控模块还原（跨模块契约）。"""
        from monitor.quant import _stage_from_dict

        original = aggregate([_ok(1.5), _ok(2.5)], concurrency=2, wall_time=2.0)
        revived = _stage_from_dict(original.to_dict())

        assert revived.concurrency == original.concurrency
        assert revived.success_requests == original.success_requests
        assert revived.latency_mean == pytest.approx(original.latency_mean)
        assert revived.latency_p99 == pytest.approx(original.latency_p99)
        assert revived.throughput_output == pytest.approx(original.throughput_output)
        assert revived.ttft_mean == pytest.approx(original.ttft_mean)
        assert revived.tpot_mean == pytest.approx(original.tpot_mean)


# ----------------------------------------------------------------------
# summarize_run
# ----------------------------------------------------------------------
class TestSummarizeRun:
    def _stage(self, concurrency: int, tp: float, ttft: float, unstable: bool = False):
        s = StageMetrics(
            concurrency=concurrency,
            total_requests=concurrency,
            success_requests=concurrency,
            throughput_output=tp,
            ttft_mean=ttft,
        )
        s.unstable = unstable
        return s

    def test_empty(self):
        result = summarize_run([])
        assert result["best_throughput"] is None
        assert result["unstable_stages"] == []

    def test_picks_max_throughput(self):
        stages = [
            self._stage(1, 40.0, 0.1),
            self._stage(8, 300.0, 0.3),
            self._stage(32, 900.0, 0.9),
        ]
        r = summarize_run(stages)
        assert r["best_throughput"] == 900.0
        assert r["best_throughput_concurrency"] == 32

    def test_picks_min_ttft(self):
        """最优 TTFT 通常出现在最低并发——取最小值而非峰值吞吐那一档。"""
        stages = [
            self._stage(1, 40.0, 0.08),
            self._stage(8, 300.0, 0.3),
            self._stage(32, 900.0, 0.9),
        ]
        r = summarize_run(stages)
        assert r["lowest_ttft"] == pytest.approx(0.08)
        assert r["lowest_ttft_concurrency"] == 1

    def test_collects_unstable_stages(self):
        stages = [
            self._stage(1, 40.0, 0.1),
            self._stage(32, 900.0, 0.9, unstable=True),
        ]
        r = summarize_run(stages)
        assert r["unstable_stages"] == [32]
        assert r["max_concurrency_tested"] == 32

    def test_ignores_stages_without_success(self):
        """全失败的档位不参与最优值评选。"""
        bad = StageMetrics(concurrency=64, total_requests=64, success_requests=0)
        good = self._stage(8, 100.0, 0.2)
        r = summarize_run([good, bad])
        assert r["best_throughput_concurrency"] == 8

    def test_handles_missing_throughput(self):
        s = StageMetrics(concurrency=1, total_requests=1, success_requests=1, ttft_mean=0.5)
        r = summarize_run([s])
        assert r["best_throughput"] is None
        assert r["lowest_ttft"] == pytest.approx(0.5)
