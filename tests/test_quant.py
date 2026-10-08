"""量化横向对比测试。

重点验证：
1. **基准选择优先级**（显式 is_baseline > fp16/bf16 > 第一组）
2. **相对倍数**计算正确，且分母为 0/None 时返回 None 而不是崩
3. **只对比共同并发档位**，避免拿 32 并发比 1 并发
4. 结论文案能从数据里正确提炼
"""

from __future__ import annotations

import pytest

from metrics.stats import StageMetrics
from monitor.quant import (
    QuantGroup,
    _build_conclusion,
    _pct_delta,
    _ratio,
    _stage_from_dict,
    compare,
    from_result_dicts,
    pick_baseline,
)


# ----------------------------------------------------------------------
# 构造辅助
# ----------------------------------------------------------------------
def _stage(concurrency: int, tp: float, ttft: float = 0.2, p99: float = 1.0) -> StageMetrics:
    """构造一个档位指标。tp 为输出吞吐 tokens/s，ttft 为秒。"""
    s = StageMetrics(
        concurrency=concurrency,
        total_requests=concurrency,
        success_requests=concurrency,
        throughput_output=tp,
        ttft_mean=ttft,
        latency_p99=p99,
        tpot_mean=0.02,
    )
    return s


def _group(
    label: str,
    quant: str,
    base_tp: float,
    vram: float | None = None,
    levels=(1, 2, 4, 8, 16, 32),
    ttft: float = 0.2,
    engine: str = "vllm",
    model: str = "Qwen2.5-72B-Instruct",
    is_baseline: bool | None = None,
) -> QuantGroup:
    """构造一个量化组。

    ``base_tp`` 是并发 1 的吞吐，随并发按 0.72 次幂放大，
    模拟真实的「次线性扩展」。
    """
    stages = [
        _stage(lvl, base_tp * (lvl**0.72), ttft * (1 + lvl * 0.06)) for lvl in levels
    ]
    extra = {}
    if is_baseline is not None:
        extra["is_baseline"] = is_baseline
    return QuantGroup(
        label=label,
        engine=engine,
        model=model,
        quantization=quant,
        stages=stages,
        vram_peak_mib=vram,
        extra=extra,
    )


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
class TestRatio:
    def test_normal(self):
        assert _ratio(150.0, 100.0) == pytest.approx(1.5)

    def test_less_than_one(self):
        assert _ratio(25.0, 100.0) == pytest.approx(0.25)

    def test_none_numerator(self):
        assert _ratio(None, 100.0) is None

    def test_none_denominator(self):
        assert _ratio(100.0, None) is None

    def test_zero_denominator(self):
        """分母为 0 时返回 None，不能抛 ZeroDivisionError。"""
        assert _ratio(100.0, 0.0) is None


class TestPctDelta:
    def test_increase(self):
        assert _pct_delta(110.0, 100.0) == pytest.approx(10.0)

    def test_decrease_is_negative(self):
        assert _pct_delta(80.0, 100.0) == pytest.approx(-20.0)

    def test_none_inputs(self):
        assert _pct_delta(None, 100.0) is None
        assert _pct_delta(100.0, None) is None

    def test_zero_base(self):
        assert _pct_delta(100.0, 0.0) is None


# ----------------------------------------------------------------------
# 基准选择
# ----------------------------------------------------------------------
class TestPickBaseline:
    def test_explicit_flag_wins(self):
        """显式 is_baseline=True 优先于量化等级推测。"""
        g1 = _group("A", "awq", 100.0, is_baseline=True)
        g2 = _group("B", "fp16", 80.0)
        assert pick_baseline([g1, g2]) is g1

    def test_fp16_preferred_over_int8(self):
        """没有显式标记时，fp16 优先。"""
        g1 = _group("A", "int8", 100.0)
        g2 = _group("B", "fp16", 80.0)
        assert pick_baseline([g1, g2]) is g2

    def test_bf16_also_counts(self):
        g1 = _group("A", "awq", 100.0)
        g2 = _group("B", "bf16", 80.0)
        assert pick_baseline([g1, g2]) is g2

    def test_falls_back_to_first(self):
        """全是量化配置时退化用第一组，并给出警告。"""
        g1 = _group("A", "awq", 100.0)
        g2 = _group("B", "gptq", 80.0)
        assert pick_baseline([g1, g2]) is g1

    def test_empty_list(self):
        assert pick_baseline([]) is None

    def test_case_insensitive(self):
        g1 = _group("A", "AWQ", 100.0)
        g2 = _group("B", "FP16", 80.0)
        assert pick_baseline([g1, g2]) is g2


# ----------------------------------------------------------------------
# QuantGroup 自身
# ----------------------------------------------------------------------
class TestQuantGroup:
    def test_best_throughput(self):
        g = _group("A", "fp16", 100.0)
        assert g.best_throughput() == pytest.approx(100.0 * 32**0.72)

    def test_best_throughput_concurrency(self):
        g = _group("A", "fp16", 100.0)
        assert g.best_throughput_concurrency() == 32

    def test_best_throughput_on_empty(self):
        g = QuantGroup(label="x", engine="vllm", model="m", quantization="fp16")
        assert g.best_throughput() is None
        assert g.best_throughput_concurrency() is None

    def test_lowest_ttft(self):
        """最低 TTFT 出现在并发 1 那一档（ttft 随并发线性上升）。"""
        g = _group("A", "fp16", 100.0, ttft=0.2)
        # _group 里 ttft *= (1 + lvl*0.06)，并发 1 => 0.2 * 1.06
        assert g.lowest_ttft() == pytest.approx(0.2 * 1.06)

    def test_mean_tpot(self):
        g = _group("A", "fp16", 100.0)
        assert g.mean_tpot() == pytest.approx(0.02)

    def test_peak_stable_concurrency(self):
        g = QuantGroup(
            label="A",
            engine="vllm",
            model="m",
            quantization="fp16",
            stages=[_stage(1, 10.0), _stage(8, 80.0), _stage(32, 100.0)],
        )
        # 把并发 32 标成不稳定
        g.stages[2].failure_rate = 0.3
        assert g.peak_concurrency_stable(max_failure_rate=0.05) == 8

    def test_peak_stable_concurrency_all_failed(self):
        g = QuantGroup(
            label="A",
            engine="vllm",
            model="m",
            quantization="fp16",
            stages=[_stage(1, 10.0)],
        )
        g.stages[0].success_requests = 0
        assert g.peak_concurrency_stable() is None

    def test_stage_map(self):
        g = _group("A", "fp16", 100.0, levels=(1, 2, 4))
        assert sorted(g.stage_map()) == [1, 2, 4]


# ----------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------
class TestCompare:
    def test_empty_groups(self):
        result = compare([])
        assert result["baseline"] is None
        assert result["groups"] == []
        assert "没有可对比" in result["conclusion"]

    def test_single_group(self):
        result = compare([_group("A", "fp16", 100.0, vram=68000)])
        assert result["baseline"]["quantization"] == "fp16"
        assert len(result["groups"]) == 1
        row = result["groups"][0]
        assert row["is_baseline"] is True
        assert row["throughput_vs_baseline"] == pytest.approx(1.0)
        assert row["vram_ratio_vs_baseline"] == pytest.approx(1.0)

    def test_awq_beats_fp16(self):
        """AWQ 吞吐更高、显存更低，应该两项都占优。"""
        fp16 = _group("FP16", "fp16", 90.0, vram=68000)
        awq = _group("AWQ", "awq", 138.0, vram=42500)
        result = compare([fp16, awq])

        assert result["baseline"]["label"] == "FP16"
        rows = {r["label"]: r for r in result["groups"]}
        assert rows["AWQ"]["throughput_vs_baseline"] > 1.4
        assert rows["AWQ"]["vram_ratio_vs_baseline"] < 0.7
        assert rows["FP16"]["is_baseline"] is True

    def test_throughput_per_gib(self):
        """每 GiB 显存吞吐：量化收益的核心指标。"""
        fp16 = _group("FP16", "fp16", 100.0, vram=1024.0)
        awq = _group("AWQ", "awq", 100.0, vram=512.0)
        result = compare([fp16, awq])
        rows = {r["label"]: r for r in result["groups"]}

        # fp16: 100/1 GiB = 100；awq: 100/0.5 GiB = 200
        assert rows["FP16"]["throughput_per_gib"] == pytest.approx(
            rows["FP16"]["best_throughput_tps"] / 1.0
        )
        assert rows["AWQ"]["throughput_per_gib"] == pytest.approx(
            rows["AWQ"]["best_throughput_tps"] / 0.5
        )
        assert rows["AWQ"]["throughput_per_gib"] > rows["FP16"]["throughput_per_gib"]

    def test_missing_baseline_vram_gives_none_ratio(self):
        """基准组缺显存数据时，相对基准的显存比应为 None。

        注意 ``throughput_per_gib`` 只用**本组**的显存，基准缺数据不影响它。
        """
        fp16 = _group("FP16", "fp16", 100.0, vram=None)
        awq = _group("AWQ", "awq", 120.0, vram=40000)
        result = compare([fp16, awq])
        rows = {r["label"]: r for r in result["groups"]}
        assert rows["AWQ"]["vram_ratio_vs_baseline"] is None
        assert rows["FP16"]["vram_ratio_vs_baseline"] is None
        assert rows["AWQ"]["throughput_vs_baseline"] is not None
        # 本组有显存数据，所以每 GiB 吞吐仍可算
        assert rows["AWQ"]["throughput_per_gib"] == pytest.approx(
            rows["AWQ"]["best_throughput_tps"] / (40000 / 1024)
        )

    def test_missing_own_vram_gives_none_per_gib(self):
        """本组自己缺显存数据时，每 GiB 吞吐才是 None。"""
        fp16 = _group("FP16", "fp16", 100.0, vram=68000)
        awq = _group("AWQ", "awq", 120.0, vram=None)
        result = compare([fp16, awq])
        rows = {r["label"]: r for r in result["groups"]}
        assert rows["AWQ"]["throughput_per_gib"] is None
        assert rows["AWQ"]["vram_ratio_vs_baseline"] is None

    def test_common_concurrency_only(self):
        """不同组档位不一致时，只对比共同档位。"""
        a = _group("A", "fp16", 100.0, levels=(1, 2, 4, 8))
        b = _group("B", "awq", 120.0, levels=(2, 4, 16))
        result = compare([a, b])
        assert result["common_concurrency"] == [2, 4]
        assert "16" not in result["by_concurrency"]
        assert "1" not in result["by_concurrency"]

    def test_by_concurrency_entries(self):
        a = _group("A", "fp16", 100.0, levels=(1, 4))
        b = _group("B", "awq", 150.0, levels=(1, 4))
        result = compare([a, b])
        entry = result["by_concurrency"]["4"]
        assert len(entry) == 2
        assert {e["label"] for e in entry} == {"A", "B"}
        assert all(e["throughput_output"] is not None for e in entry)

    def test_no_common_concurrency(self):
        a = _group("A", "fp16", 100.0, levels=(1, 2))
        b = _group("B", "awq", 120.0, levels=(32, 64))
        result = compare([a, b])
        assert result["common_concurrency"] == []
        assert result["by_concurrency"] == {}

    def test_three_way_comparison(self):
        groups = [
            _group("FP16", "fp16", 90.0, vram=68000),
            _group("INT8", "int8", 105.0, vram=36000),
            _group("AWQ", "awq", 138.0, vram=42500),
        ]
        result = compare(groups)
        assert len(result["groups"]) == 3
        assert result["baseline"]["quantization"] == "fp16"
        rows = {r["label"]: r for r in result["groups"]}
        # AWQ 吞吐最高
        assert rows["AWQ"]["throughput_vs_baseline"] > rows["INT8"]["throughput_vs_baseline"]

    def test_baseline_ordering_independent(self):
        """基准在列表末尾时也要能正确识别。"""
        awq = _group("AWQ", "awq", 138.0, vram=42500)
        fp16 = _group("FP16", "fp16", 90.0, vram=68000)
        result = compare([awq, fp16])
        assert result["baseline"]["label"] == "FP16"
        rows = {r["label"]: r for r in result["groups"]}
        assert rows["FP16"]["is_baseline"] is True
        assert rows["AWQ"]["is_baseline"] is False

    def test_row_fields_complete(self):
        result = compare([_group("A", "fp16", 100.0, vram=1000)])
        row = result["groups"][0]
        for field in (
            "label",
            "engine",
            "model",
            "quantization",
            "is_baseline",
            "best_throughput_tps",
            "best_throughput_concurrency",
            "lowest_ttft_s",
            "mean_tpot_s",
            "vram_peak_mib",
            "peak_stable_concurrency",
            "throughput_vs_baseline",
            "ttft_delta_pct_vs_baseline",
            "tpot_delta_pct_vs_baseline",
            "vram_ratio_vs_baseline",
            "throughput_per_gib",
            "stage_count",
        ):
            assert field in row, f"缺少字段 {field}"

    def test_baseline_without_throughput_does_not_crash(self):
        """基准组没跑出吞吐时，其他组不该崩，倍数应为 None。"""
        base = QuantGroup(
            label="FP16",
            engine="vllm",
            model="m",
            quantization="fp16",
            stages=[StageMetrics(concurrency=1, success_requests=0)],
            vram_peak_mib=68000,
        )
        other = _group("AWQ", "awq", 138.0, vram=42500)
        result = compare([base, other])
        rows = {r["label"]: r for r in result["groups"]}
        assert rows["AWQ"]["throughput_vs_baseline"] is None

    def test_groups_without_stages(self):
        a = QuantGroup(label="A", engine="vllm", model="m", quantization="fp16")
        b = QuantGroup(label="B", engine="vllm", model="m", quantization="awq")
        result = compare([a, b])
        assert len(result["groups"]) == 2
        assert result["common_concurrency"] == []


# ----------------------------------------------------------------------
# 结论生成
# ----------------------------------------------------------------------
class TestConclusion:
    def test_quantization_gain_reported(self):
        rows = [
            {
                "label": "FP16",
                "quantization": "fp16",
                "is_baseline": True,
                "throughput_vs_baseline": 1.0,
                "vram_ratio_vs_baseline": 1.0,
                "ttft_delta_pct_vs_baseline": 0.0,
            },
            {
                "label": "AWQ",
                "quantization": "awq",
                "is_baseline": False,
                "throughput_vs_baseline": 1.5,
                "vram_ratio_vs_baseline": 0.6,
                "ttft_delta_pct_vs_baseline": -5.0,
            },
        ]
        text = _build_conclusion(rows)
        assert "AWQ" in text
        assert "50.0%" in text
        assert "40.0%" in text

    def test_no_gain_reported_honestly(self):
        """量化没带来提升时要如实说，不能编。"""
        rows = [
            {"label": "FP16", "quantization": "fp16", "is_baseline": True,
             "throughput_vs_baseline": 1.0, "vram_ratio_vs_baseline": 1.0,
             "ttft_delta_pct_vs_baseline": 0.0},
            {"label": "INT8", "quantization": "int8", "is_baseline": False,
             "throughput_vs_baseline": 0.98, "vram_ratio_vs_baseline": 0.55,
             "ttft_delta_pct_vs_baseline": 1.0},
        ]
        text = _build_conclusion(rows)
        assert "未带来吞吐提升" in text

    def test_ttft_regression_flagged(self):
        rows = [
            {"label": "FP16", "quantization": "fp16", "is_baseline": True,
             "throughput_vs_baseline": 1.0, "vram_ratio_vs_baseline": 1.0,
             "ttft_delta_pct_vs_baseline": 0.0},
            {"label": "GPTQ", "quantization": "gptq", "is_baseline": False,
             "throughput_vs_baseline": 1.3, "vram_ratio_vs_baseline": 0.4,
             "ttft_delta_pct_vs_baseline": 35.0},
        ]
        text = _build_conclusion(rows)
        assert "首包延迟较基准恶化" in text

    def test_only_baseline(self):
        rows = [
            {"label": "FP16", "quantization": "fp16", "is_baseline": True,
             "throughput_vs_baseline": 1.0, "vram_ratio_vs_baseline": 1.0,
             "ttft_delta_pct_vs_baseline": 0.0},
        ]
        assert "仅有基准组" in _build_conclusion(rows)

    def test_empty_rows(self):
        assert "没有可对比" in _build_conclusion([])


# ----------------------------------------------------------------------
# 反序列化
# ----------------------------------------------------------------------
class TestFromResultDicts:
    def test_rebuild_groups(self):
        items = [
            {
                "label": "Qwen2.5-72B @ FP16",
                "engine": "vllm",
                "model": "Qwen2.5-72B-Instruct",
                "quantization": "fp16",
                "vram_peak_mib": 68300,
                "stages": [
                    {
                        "concurrency": 1,
                        "total_requests": 1,
                        "success_requests": 1,
                        "failed_requests": 0,
                        "failure_rate": 0.0,
                        "unstable": False,
                        "latency": {"mean": 1.0, "p50": 1.0, "p95": 1.0, "p99": 1.0,
                                    "min": 1.0, "max": 1.0, "stdev": None},
                        "ttft": {"mean": 0.2, "p50": 0.2, "p95": 0.2, "p99": 0.2},
                        "tpot": {"mean": 0.02, "p50": 0.02, "p95": 0.02, "p99": 0.02},
                        "throughput": {"output_tokens_total": 256, "input_tokens_total": 32,
                                       "output_tokens_per_second": 100.0,
                                       "total_tokens_per_second": 110.0,
                                       "per_request_output_tps_mean": 50.0},
                        "timing": {"wall_time": 2.56, "request_rate": 0.4},
                        "errors": {"count": 0, "breakdown": {}, "samples": []},
                    }
                ],
            }
        ]
        groups = from_result_dicts(items)
        assert len(groups) == 1
        g = groups[0]
        assert g.label == "Qwen2.5-72B @ FP16"
        assert g.vram_peak_mib == 68300
        assert g.best_throughput() == pytest.approx(100.0)
        assert g.best_throughput_concurrency() == 1
        assert g.stages[0].ttft_mean == pytest.approx(0.2)

    def test_defaults_when_fields_missing(self):
        groups = from_result_dicts([{}])
        assert len(groups) == 1
        assert groups[0].engine == "unknown"
        assert groups[0].stages == []

    def test_stage_from_dict_tolerates_partial(self):
        """字段缺失时补默认值，不能 KeyError。"""
        s = _stage_from_dict({"concurrency": 8})
        assert s.concurrency == 8
        assert s.latency_mean is None
        assert s.throughput_output is None
        assert s.failure_rate == 0.0

    def test_roundtrip_via_compare(self):
        """完整链路：结果字典 -> 量化组 -> 对比矩阵。"""
        fp16 = _group("FP16", "fp16", 90.0, vram=68000)
        awq = _group("AWQ", "awq", 138.0, vram=42500)
        items = [
            {
                "label": g.label,
                "engine": g.engine,
                "model": g.model,
                "quantization": g.quantization,
                "vram_peak_mib": g.vram_peak_mib,
                "stages": [s.to_dict() for s in g.stages],
            }
            for g in (fp16, awq)
        ]
        result = compare(from_result_dicts(items))
        assert result["baseline"]["quantization"] == "fp16"
        rows = {r["label"]: r for r in result["groups"]}
        assert rows["AWQ"]["throughput_vs_baseline"] > 1.0
