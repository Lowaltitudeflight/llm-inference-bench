"""结果持久化测试。

落盘的 JSON / CSV 是**跨进程契约**：压测进程写、报告进程读。
结构一旦变化，历史的压测结果就读不出来了。所以这里对字段名
和嵌套结构做严格断言。
"""

from __future__ import annotations

import csv
import json

import pytest

from config.settings import BenchConfig, EngineTarget
from metrics.stats import RequestResult, StageMetrics, aggregate, summarize_run
from metrics.store import (
    CSV_COLUMNS,
    build_result,
    environment_info,
    list_results,
    load_latest,
    load_result,
    new_run_id,
    save_result,
    stage_rows,
)


def _stages(count: int = 2) -> list[StageMetrics]:
    stages = []
    for lvl in (1, 4, 16)[:count]:
        results = [
            RequestResult(
                success=True,
                total_latency=1.0 + i * 0.1,
                ttft=0.2,
                generation_time=0.8 + i * 0.1,
                output_tokens=21,
                input_tokens=20,
            )
            for i in range(lvl)
        ]
        stages.append(aggregate(results, concurrency=lvl, wall_time=1.5))
    return stages


def _target() -> EngineTarget:
    return EngineTarget(
        name="vllm",
        base_url="http://localhost:8000",
        model="Qwen2.5-72B-Instruct",
        quantization="fp16",
        note="8x RTX PRO 5000",
    )


def _bench() -> BenchConfig:
    return BenchConfig(concurrency_levels=[1, 4, 16], max_tokens=256, prompt="test prompt")


class TestRunId:
    def test_format(self):
        rid = new_run_id()
        assert rid.startswith("run-")
        # run-YYYYMMDD-HHMMSS
        assert len(rid) == len("run-20260518-143002")
        assert rid.count("-") == 2

    def test_unique_across_prefixes(self):
        assert new_run_id("a") != new_run_id("b")


class TestEnvironmentInfo:
    def test_has_required_keys(self):
        env = environment_info()
        for key in ("platform", "python", "machine", "hostname"):
            assert key in env
            assert isinstance(env[key], str)

    def test_processor_may_be_empty_but_not_missing(self):
        """部分 Linux 上 processor 返回空串，归一成 unknown。"""
        env = environment_info()
        assert env["processor"]  # 不能是空字符串


class TestBuildResult:
    def test_full_shape(self):
        result = build_result(
            target=_target(),
            bench=_bench(),
            stages=_stages(),
            vram_curve={"available": False, "gpus": []},
            health={"ok": True, "model_loaded": True},
            run_id="run-test",
            duration=12.34,
        )
        assert result["run_id"] == "run-test"
        assert result["engine"] == "vllm"
        assert result["model"] == "Qwen2.5-72B-Instruct"
        assert result["quantization"] == "fp16"
        assert result["base_url"] == "http://localhost:8000"
        assert result["duration_seconds"] == pytest.approx(12.34)
        assert len(result["stages"]) == 2

    def test_config_snapshot(self):
        """配置要一起存下来，否则历史结果无法复现。"""
        result = build_result(_target(), _bench(), _stages(), run_id="x")
        cfg = result["config"]
        assert cfg["concurrency_levels"] == [1, 4, 16]
        assert cfg["max_tokens"] == 256
        assert cfg["prompt"] == "test prompt"
        assert "stream" in cfg
        assert "warmup_requests" in cfg

    def test_summary_included(self):
        result = build_result(_target(), _bench(), _stages(), run_id="x")
        assert "best_throughput" in result["summary"]
        assert "unstable_stages" in result["summary"]

    def test_vram_peak_extracted(self):
        result = build_result(
            _target(),
            _bench(),
            _stages(),
            vram_curve={"available": True, "gpus": [], "peak_memory_used_mib": 68300},
            run_id="x",
        )
        assert result["vram_peak_mib"] == 68300

    def test_vram_peak_none_without_curve(self):
        result = build_result(_target(), _bench(), _stages(), run_id="x")
        assert result["vram_peak_mib"] is None

    def test_generated_at_is_isoformat(self):
        result = build_result(_target(), _bench(), _stages(), run_id="x")
        assert "T" in result["generated_at"]

    def test_roundtrip_through_stage_from_dict(self):
        """stages 落盘后必须能无损还原成 StageMetrics。"""
        from monitor.quant import _stage_from_dict

        original = _stages(count=3)
        result = build_result(_target(), _bench(), original, run_id="x")
        restored = [_stage_from_dict(s) for s in result["stages"]]

        assert len(restored) == len(original)
        for o, r in zip(original, restored):
            assert r.concurrency == o.concurrency
            assert r.success_requests == o.success_requests
            assert r.failure_rate == pytest.approx(o.failure_rate)
            assert r.latency_mean == pytest.approx(o.latency_mean)
            assert r.latency_p95 == pytest.approx(o.latency_p95)
            assert r.latency_p99 == pytest.approx(o.latency_p99)
            assert r.ttft_mean == pytest.approx(o.ttft_mean)
            assert r.tpot_mean == pytest.approx(o.tpot_mean)
            assert r.throughput_output == pytest.approx(o.throughput_output)
            assert r.wall_time == pytest.approx(o.wall_time)


class TestSaveResult:
    def test_writes_json_and_csv(self, tmp_path):
        result = build_result(_target(), _bench(), _stages(), run_id="run-1")
        paths = save_result(result, tmp_path, "run-1")

        assert paths["json"].exists()
        assert paths["csv"].exists()
        assert paths["json"].parent.name == "run-1"
        # 文件名包含引擎、模型与量化等级三段，避免多配置压测互相覆盖
        assert paths["json"].name == "vllm__Qwen2.5-72B-Instruct__fp16.json"

    def test_json_is_valid(self, tmp_path):
        result = build_result(_target(), _bench(), _stages(), run_id="run-1")
        paths = save_result(result, tmp_path, "run-1")

        loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert loaded["engine"] == "vllm"
        assert loaded["run_id"] == "run-1"

    def test_json_is_not_ascii_escaped(self, tmp_path):
        """中文必须原样写入，不能变成 \\uXXXX——否则人看不了。"""
        bench = BenchConfig(prompt="请用三句话说明什么是张量并行。", concurrency_levels=[1])
        result = build_result(_target(), bench, _stages(count=1), run_id="run-1")
        paths = save_result(result, tmp_path, "run-1")
        raw = paths["json"].read_text(encoding="utf-8")
        assert "请用三句话说明什么是张量并行。" in raw
        assert "\\u8bf7" not in raw

    def test_csv_headers_match_contract(self, tmp_path):
        result = build_result(_target(), _bench(), _stages(), run_id="run-1")
        paths = save_result(result, tmp_path, "run-1")

        with paths["csv"].open(encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == CSV_COLUMNS
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["concurrency"] == "1"
        assert float(rows[0]["latency_mean"]) > 0

    def test_csv_has_bom_for_excel(self, tmp_path):
        """带 BOM 才能让中文 Excel 正确识别 UTF-8。"""
        result = build_result(_target(), _bench(), _stages(), run_id="run-1")
        paths = save_result(result, tmp_path, "run-1")
        assert paths["csv"].read_bytes().startswith(b"\xef\xbb\xbf")

    def test_csv_escapes_commas_in_prompt(self, tmp_path):
        """配置里有逗号时 CSV 也不能串列。"""
        bench = BenchConfig(prompt="a,b,c", concurrency_levels=[1])
        result = build_result(_target(), bench, _stages(count=1), run_id="run-1")
        paths = save_result(result, tmp_path, "run-1")
        with paths["csv"].open(encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["concurrency"] == "1"

    def test_overwrite_same_run(self, tmp_path):
        """同 run_id 重跑应覆盖而不是追加。"""
        r1 = build_result(_target(), _bench(), _stages(count=1), run_id="run-1")
        save_result(r1, tmp_path, "run-1")
        r2 = build_result(_target(), _bench(), _stages(count=2), run_id="run-1")
        paths = save_result(r2, tmp_path, "run-1")

        loaded = load_result(paths["json"])
        assert len(loaded["stages"]) == 2

    def test_result_with_no_stages(self, tmp_path):
        """全失败/空跑也要能落盘。"""
        result = build_result(_target(), _bench(), [], run_id="run-1")
        paths = save_result(result, tmp_path, "run-1")
        assert paths["json"].exists()
        assert paths["csv"].exists()

    def test_creates_nested_dirs(self, tmp_path):
        result = build_result(_target(), _bench(), _stages(), run_id="run-1")
        deep = tmp_path / "a" / "b"
        paths = save_result(result, deep, "run-1")
        assert paths["json"].exists()


class TestLoadAndList:
    def test_list_results_empty(self, tmp_path):
        assert list_results(tmp_path) == []
        assert list_results(tmp_path / "nope") == []

    def test_list_results_sorted_newest_first(self, tmp_path):
        """结果按修改时间倒序，最新一次在最前。

        直接用 os.utime 设定 mtime，避免为排序测试引入真实 sleep
        （那会让整个测试套件慢几十秒，而排序逻辑与真实时间无关）。
        """
        import os

        for i, rid in enumerate(("run-1", "run-2", "run-3")):
            r = build_result(_target(), _bench(), _stages(count=1), run_id=rid)
            paths = save_result(r, tmp_path, rid)
            # mtime 递增：run-3 最新
            stamp = 1_700_000_000 + i * 10
            os.utime(paths["json"], (stamp, stamp))

        files = list_results(tmp_path)
        assert len(files) == 3
        assert files[0].parent.name == "run-3"
        assert files[-1].parent.name == "run-1"

    def test_load_latest(self, tmp_path):
        import os

        for i, rid in enumerate(("run-1", "run-2")):
            r = build_result(_target(), _bench(), _stages(count=1), run_id=rid)
            paths = save_result(r, tmp_path, rid)
            stamp = 1_700_000_000 + i * 10
            os.utime(paths["json"], (stamp, stamp))

        latest = load_latest(tmp_path)
        assert latest["run_id"] == "run-2"

    def test_load_latest_empty(self, tmp_path):
        assert load_latest(tmp_path) is None

    def test_load_result_roundtrip(self, tmp_path):
        result = build_result(_target(), _bench(), _stages(), run_id="run-1")
        paths = save_result(result, tmp_path, "run-1")
        loaded = load_result(paths["json"])
        assert loaded == result


class TestStageRows:
    def test_flattens_stages(self):
        result = build_result(_target(), _bench(), _stages(), run_id="run-1")
        rows = stage_rows(result)
        assert len(rows) == 2
        assert set(rows[0]) == set(CSV_COLUMNS)

    def test_values_match(self):
        result = build_result(_target(), _bench(), _stages(), run_id="run-1")
        rows = stage_rows(result)
        assert rows[0]["concurrency"] == 1
        assert rows[0]["success_requests"] == 1
        assert rows[1]["concurrency"] == 4
        assert rows[1]["success_requests"] == 4

    def test_empty_result(self):
        result = build_result(_target(), _bench(), [], run_id="run-1")
        assert stage_rows(result) == []


class TestSummarizeIntegration:
    def test_summary_from_saved_result(self, tmp_path):
        """验证 summarize_run 的结论能从落盘结果里重建。"""
        stages = _stages(count=3)
        result = build_result(_target(), _bench(), stages, run_id="run-1")
        paths = save_result(result, tmp_path, "run-1")
        loaded = load_result(paths["json"])

        from monitor.quant import _stage_from_dict

        restored = [_stage_from_dict(s) for s in loaded["stages"]]
        assert summarize_run(restored) == loaded["summary"]
