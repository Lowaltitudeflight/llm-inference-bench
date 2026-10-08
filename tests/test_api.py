"""API 服务与 Prometheus 指标测试。

服务层测试关注契约：路由存在、请求体校验、任务状态机、
以及**错误输入要有明确的 4xx 而不是 500**。

Prometheus 测试关注指标名称与标签——Grafana 面板里的 PromQL
直接依赖它们，改名就会让面板空白。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.server import app
from metrics import prometheus as prom
from metrics.stats import RequestResult, StageMetrics


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def metrics():
    """每个测试用独立 Registry，避免相互污染。"""
    return prom.BenchMetrics()


# ----------------------------------------------------------------------
# 基础路由
# ----------------------------------------------------------------------
class TestBasicRoutes:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["service"] == "llm-inference-bench"
        assert "/bench" in " ".join(body["endpoints"])

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_engines_lists_config(self, client):
        r = client.get("/engines")
        assert r.status_code == 200
        body = r.json()
        names = {e["name"] for e in body["engines"]}
        assert {"vllm", "ollama"} <= names
        vllm = next(e for e in body["engines"] if e["name"] == "vllm")
        assert vllm["base_url"].startswith("http")
        assert vllm["model"]
        # 默认压测参数也要暴露，前端表单才能预填
        assert "concurrency_levels" in body["default_bench"]
        assert "max_tokens" in body["default_bench"]

    def test_openapi_docs_available(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = r.json()["paths"]
        assert "/bench" in paths
        assert "/metrics" in paths


class TestEngineCheck:
    def test_unknown_engine_returns_400(self, client):
        """未知引擎名要说清可选值，而不是 500。"""
        r = client.post("/engines/check", json={"engines": ["no-such-engine"]})
        assert r.status_code == 400
        assert "未知引擎" in r.json()["detail"]

    def test_check_unreachable_engine_does_not_500(self, client):
        """目标不可达时也要返回 200 + 错误详情，而不是崩。"""
        r = client.post("/engines/check", json={"engines": ["vllm"]})
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 1
        assert results[0]["engine"] == "vllm"
        assert "ok" in results[0]


# ----------------------------------------------------------------------
# 任务管理
# ----------------------------------------------------------------------
class TestJobRoutes:
    def test_unknown_job_404(self, client):
        assert client.get("/jobs/does-not-exist").status_code == 404
        assert client.get("/jobs/does-not-exist/result").status_code == 404

    def test_jobs_list_shape(self, client):
        r = client.get("/jobs")
        assert r.status_code == 200
        body = r.json()
        assert "count" in body
        assert isinstance(body["jobs"], list)

    def test_cancel_unknown_job_404(self, client):
        assert client.delete("/jobs/nope").status_code == 404

    def test_bench_rejects_unknown_engine(self, client):
        r = client.post("/bench", json={"engines": ["bogus"]})
        assert r.status_code == 400

    def test_bench_validation_rejects_bad_max_tokens(self, client):
        """max_tokens 超范围要 422（pydantic 校验），而不是进了压测才炸。"""
        r = client.post("/bench", json={"engines": ["vllm"], "max_tokens": 999999})
        assert r.status_code == 422

    def test_bench_validation_rejects_negative_warmup(self, client):
        r = client.post("/bench", json={"engines": ["vllm"], "warmup_requests": -1})
        assert r.status_code == 422

    def test_bench_accepts_minimal_body(self, client, monkeypatch):
        """只给 engines 也要能提交（其他字段用默认值）。

        这里把后台执行替换成空操作，避免真的发起压测。
        """
        import api.server as srv

        monkeypatch.setattr(srv, "_run_job", lambda job_id, req: None)
        r = client.post("/bench", json={"engines": ["vllm"]})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "pending"
        assert body["job_id"]
        assert body["run_id"].startswith("run-")
        assert body["poll"] == f"/jobs/{body['job_id']}"

    def test_bench_request_defaults_applied(self, client, monkeypatch):
        """请求体里的可选字段应覆盖全局配置。"""
        import api.server as srv
        from config.settings import BENCH

        monkeypatch.setattr(srv, "_run_job", lambda job_id, req: None)
        original = list(BENCH.concurrency_levels)

        r = client.post(
            "/bench",
            json={"engines": ["vllm"], "concurrency_levels": [1, 3, 5], "max_tokens": 64},
        )
        assert r.status_code == 200
        assert BENCH.concurrency_levels == [1, 3, 5]
        assert BENCH.max_tokens == 64

        BENCH.concurrency_levels = original  # 复原，避免影响其他测试


class TestReportsRoutes:
    def test_reports_list(self, client):
        r = client.get("/reports")
        assert r.status_code == 200
        assert "reports" in r.json()

    def test_missing_report_404(self, client):
        assert client.get("/reports/no-run/nope.html").status_code == 404

    def test_rejects_unsupported_extension(self, client):
        """只允许读已知的报告格式，避免路径穿越读任意文件。"""
        assert client.get("/reports/x/y.txt").status_code == 404
        assert client.get("/reports/x/y.exe").status_code == 404

    def test_reads_existing_report(self, client, tmp_path, monkeypatch):
        from config.settings import BENCH

        run_dir = tmp_path / "run-x"
        run_dir.mkdir()
        (run_dir / "report.html").write_text("<h1>hello</h1>", encoding="utf-8")

        monkeypatch.setattr(BENCH, "output_dir", tmp_path)
        r = client.get("/reports/run-x/report.html")
        assert r.status_code == 200
        assert "hello" in r.text


class TestCompareRoute:
    def test_missing_run_dir_404(self, client):
        r = client.post("/compare", json={"run_id": "nope"})
        assert r.status_code == 404

    def test_needs_at_least_two_results(self, client, tmp_path, monkeypatch):
        from config.settings import BENCH

        run_dir = tmp_path / "run-y"
        run_dir.mkdir()
        (run_dir / "only.json").write_text("{}", encoding="utf-8")

        monkeypatch.setattr(BENCH, "output_dir", tmp_path)
        r = client.post("/compare", json={"run_id": "run-y"})
        assert r.status_code == 400
        assert "至少需要 2 份" in r.json()["detail"]


# ----------------------------------------------------------------------
# Prometheus 指标
# ----------------------------------------------------------------------
class TestPrometheusMetrics:
    def test_metrics_endpoint(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]

    def test_metrics_endpoint_has_metric_names(self, client):
        text = client.get("/metrics").text
        for name in (
            "llmbench_requests_total",
            "llmbench_request_latency_seconds",
            "llmbench_ttft_seconds",
            "llmbench_tpot_seconds",
            "llmbench_throughput_tokens_per_second",
            "llmbench_concurrency",
            "llmbench_stage_failure_rate",
        ):
            assert name in text, f"缺少指标 {name}"

    def test_snapshot_endpoint(self, client):
        r = client.get("/metrics/snapshot")
        assert r.status_code == 200
        assert "counters" in r.json()
        assert "gauges" in r.json()

    def test_histogram_uses_sensible_buckets(self, metrics):
        """默认分桶会把大模型延迟全塞进一个桶，必须自定义分桶。

        TTFT 的量级是 50ms~5s，用默认的 (0.005,0.01,0.025,0.05,0.075,0.1,
        0.25,0.5,0.75,1,2.5,5,7.5,10,+Inf) 会在 0.1~0.25 处出现大跳变，
        分位数算不准。
        """
        metrics.ttft.labels("vllm", "m", "fp16").observe(0.35)
        output = _render(metrics)
        # 0.35s 应落在 le=0.4 这个自定义桶里（默认分桶没有 0.4）
        assert 'le="0.4"' in output
        assert 'le="0.8"' in output
        assert 'le="0.075"' not in output  # 默认桶不该出现

    def test_observe_request_counts_success(self, metrics):
        r = RequestResult(success=True, total_latency=1.0, ttft=0.2,
                          generation_time=0.8, output_tokens=21)
        metrics.observe_request(r, "vllm", "m", "fp16")
        output = _render(metrics)
        assert 'status="success"} 1.0' in output
        # 成功请求要产生延迟观测
        assert 'llmbench_request_latency_seconds_count{engine="vllm",model="m",quantization="fp16"} 1.0' in output
        assert 'llmbench_ttft_seconds_count{engine="vllm",model="m",quantization="fp16"} 1.0' in output

    def test_observe_request_counts_failure(self, metrics):
        r = RequestResult(success=False, error="timeout")
        metrics.observe_request(r, "vllm", "m", "fp16")
        output = _render(metrics)
        assert 'status="failure"} 1.0' in output
        # 失败请求不产生延迟观测。这些 series 根本不该被创建
        # （而不是写一个假的 0），否则延迟分位数会被 0 拉低。
        assert 'llmbench_request_latency_seconds_count{engine="vllm"' not in output
        assert 'llmbench_ttft_seconds_count{engine="vllm"' not in output

    def test_observe_stage_sets_gauges(self, metrics):
        stage = StageMetrics(
            concurrency=8,
            total_requests=8,
            success_requests=8,
            throughput_output=456.7,
            ttft_p99=0.9,
            latency_p99=12.3,
            failure_rate=0.0,
        )
        metrics.observe_stage(stage, "vllm", "m", "fp16")
        output = _render(metrics)
        assert 'llmbench_throughput_tokens_per_second{concurrency="8",engine="vllm",model="m",quantization="fp16"} 456.7' in output
        assert 'llmbench_concurrency{engine="vllm",model="m",quantization="fp16"} 8.0' in output

    def test_observe_stage_skips_none_values(self, metrics):
        """档位没有 TTFT 时（非流式）不该写入 0，否则面板会显示假数据。"""
        stage = StageMetrics(concurrency=1, total_requests=1, success_requests=1)
        metrics.observe_stage(stage, "vllm", "m", "fp16")
        output = _render(metrics)
        assert "llmbench_ttft_p99_seconds" not in output.split("# HELP")[-1] or \
            'llmbench_ttft_p99_seconds{' not in output

    def test_observe_gpu(self, metrics):
        from monitor.gpu import GpuSample

        samples = [
            GpuSample(index=0, name="RTX PRO 5000", memory_total_mib=49152,
                      memory_used_mib=34000, util_percent=95.0, power_watts=280.0),
            GpuSample(index=1, name="RTX PRO 5000", memory_total_mib=49152,
                      memory_used_mib=32000, util_percent=93.0, power_watts=275.0),
        ]
        metrics.observe_gpu(samples)
        output = _render(metrics)
        assert 'llmbench_gpu_memory_used_mib{gpu="0",name="RTX PRO 5000"} 34000.0' in output
        assert 'llmbench_gpu_memory_total_mib{gpu="1",name="RTX PRO 5000"} 49152.0' in output
        assert 'llmbench_gpu_utilization_percent{gpu="0",name="RTX PRO 5000"} 95.0' in output
        assert 'llmbench_gpu_power_watts{gpu="0",name="RTX PRO 5000"} 280.0' in output

    def test_observe_gpu_tolerates_missing_fields(self, metrics):
        """部分字段为 None（如容器里读不到功耗）时不能崩。"""
        from monitor.gpu import GpuSample

        metrics.observe_gpu([GpuSample(index=0, name="X", memory_used_mib=100)])
        output = _render(metrics)
        assert 'llmbench_gpu_memory_used_mib{gpu="0",name="X"} 100.0' in output
        assert "llmbench_gpu_power_watts{" not in output

    def test_set_run_info(self, metrics):
        metrics.set_run_info(run_id="run-1", engine="vllm", status="running")
        output = _render(metrics)
        assert "llmbench_run_info" in output
        assert 'run_id="run-1"' in output
        assert 'status="running"' in output
        # 所有字段必须同时存在：Info.info() 每次调用替换整个标签集，
        # 逐字段调用会只剩最后一个字段
        assert 'engine="vllm"' in output

    def test_set_run_info_clears_previous(self, metrics):
        """重复设置元信息不能累积旧标签，否则面板上会有幽灵数据。"""
        metrics.set_run_info(run_id="run-1", status="running")
        metrics.set_run_info(run_id="run-2", status="completed")
        output = _render(metrics)
        assert 'run_id="run-2"' in output
        assert 'run_id="run-1"' not in output
        assert 'status="running"' not in output

    def test_multiple_engines_separate_series(self, metrics):
        """不同引擎的指标必须按标签分离，否则对比无意义。"""
        r = RequestResult(success=True, total_latency=1.0, ttft=0.1,
                          generation_time=0.9, output_tokens=11)
        metrics.observe_request(r, "vllm", "big", "fp16")
        metrics.observe_request(r, "ollama", "small", "q4_K_M")
        output = _render(metrics)
        assert 'engine="vllm",model="big",quantization="fp16"' in output
        assert 'engine="ollama",model="small",quantization="q4_K_M"' in output

    def test_render_latest_uses_global_metrics(self):
        """全局单例的 render_latest 也要能正常工作。"""
        assert b"llmbench" in prom.render_latest()

    def test_snapshot_shape(self):
        snap = prom.snapshot()
        assert isinstance(snap["counters"], dict)
        assert isinstance(snap["gauges"], dict)


def _render(metrics: prom.BenchMetrics) -> str:
    from prometheus_client import generate_latest

    return generate_latest(metrics.registry).decode("utf-8")
