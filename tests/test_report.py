"""报告生成测试。

报告是给人看的最终产物，所以测试关注：
1. **结构完整**（七节都在，标题不丢）
2. **数据正确**（报告里的数字与源数据一致）
3. **缺失数据不崩**（无显存、无吞吐、无量化对比时要有占位说明）
4. **HTML 离线自包含**（不引 CDN，图表是内联 SVG）

第 4 点尤其重要：交付给客户的 HTML 一旦依赖 CDN，
在客户内网环境下就是白屏。
"""

from __future__ import annotations

import json
import re

import pytest

from config.settings import BenchConfig, EngineTarget
from metrics.stats import RequestResult, aggregate, summarize_run
from monitor.quant import QuantGroup, compare
from report import generate_reports
from report.fmt import (
    adaptive_seconds,
    mib,
    ms,
    num,
    pct,
    ratio,
    seconds,
    signed_pct,
    tps,
)
from report.html import build_html
from report.markdown import build_markdown
from report.svgchart import bar_chart, multi_line_chart


# ----------------------------------------------------------------------
# 测试数据
# ----------------------------------------------------------------------
def _make_stages(base_tp: float, ttft: float, levels=(1, 2, 4, 8, 16, 32)):
    stages = []
    for lvl in levels:
        tp = base_tp * (lvl**0.72)
        otok = 256
        gen = otok / (tp / lvl)
        latency = ttft * (1 + lvl * 0.06) + gen
        results = [
            RequestResult(
                success=True,
                total_latency=latency,
                ttft=ttft * (1 + lvl * 0.06),
                generation_time=gen,
                output_tokens=otok,
                input_tokens=32,
            )
            for _ in range(lvl)
        ]
        stages.append(aggregate(results, concurrency=lvl, wall_time=gen))
    return stages


def _result(
    engine: str,
    model: str,
    quant: str,
    base_tp: float,
    ttft: float,
    vram: float | None = None,
    with_vram_curve: bool = False,
) -> dict:
    stages = _make_stages(base_tp, ttft)
    curve: dict = {"available": False, "gpus": []}
    if with_vram_curve:
        curve = {
            "available": True,
            "interval": 0.5,
            "gpu_count": 2,
            "peak_memory_used_mib": 66000.0,
            "total_memory_mib": 98304.0,
            "peak_memory_used_percent": 67.1,
            "stage_peak_mib": {"1": 32000.0, "4": 48000.0, "16": 66000.0},
            "message": "检测到 2 张 GPU",
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA RTX PRO 5000",
                    "memory_total_mib": 49152.0,
                    "sample_count": 120,
                    "peak_memory_used_mib": 34000.0,
                    "peak_memory_used_percent": 69.2,
                    "mean_util_percent": 78.5,
                    "max_util_percent": 99.0,
                    "series": [
                        {"t": 0.0, "memory_used_mib": 20000.0, "memory_used_percent": 40.7,
                         "util_percent": 0.0, "power_watts": 80.0, "temperature_c": 45.0},
                        {"t": 1.0, "memory_used_mib": 34000.0, "memory_used_percent": 69.2,
                         "util_percent": 95.0, "power_watts": 280.0, "temperature_c": 72.0},
                    ],
                },
                {
                    "index": 1,
                    "name": "NVIDIA RTX PRO 5000",
                    "memory_total_mib": 49152.0,
                    "sample_count": 120,
                    "peak_memory_used_mib": 32000.0,
                    "peak_memory_used_percent": 65.1,
                    "mean_util_percent": 76.0,
                    "max_util_percent": 98.0,
                    "series": [
                        {"t": 0.0, "memory_used_mib": 19000.0, "memory_used_percent": 38.7,
                         "util_percent": 0.0, "power_watts": 78.0, "temperature_c": 44.0},
                        {"t": 1.0, "memory_used_mib": 32000.0, "memory_used_percent": 65.1,
                         "util_percent": 93.0, "power_watts": 275.0, "temperature_c": 70.0},
                    ],
                },
            ],
        }
    return {
        "run_id": "run-test",
        "engine": engine,
        "model": model,
        "base_url": "http://localhost:8000",
        "quantization": quant,
        "note": "",
        "generated_at": "2026-05-18T14:30:02+08:00",
        "duration_seconds": 120.0,
        "config": {
            "concurrency_levels": [1, 2, 4, 8, 16, 32],
            "max_tokens": 256,
            "warmup_requests": 2,
            "temperature": 0.0,
            "stream": True,
            "timeout": 120.0,
            "prompt": "请用三句话说明什么是张量并行（Tensor Parallelism）。",
        },
        "environment": {
            "platform": "Linux-6.5.0",
            "python": "3.11.7",
            "processor": "x86_64",
            "machine": "x86_64",
            "hostname": "gpu-node-01",
        },
        "health": {"ok": True, "model_loaded": True, "models": [model]},
        "summary": summarize_run(stages),
        "stages": [s.to_dict() for s in stages],
        "vram_curve": curve,
        "vram_peak_mib": vram,
    }


@pytest.fixture
def two_results():
    return [
        _result("vllm", "Qwen2.5-72B-Instruct", "fp16", 90.0, 0.42, vram=68000.0,
                with_vram_curve=True),
        _result("ollama", "qwen2.5:7b", "q4_K_M", 210.0, 0.11, vram=6200.0),
    ]


@pytest.fixture
def quant_groups():
    return [
        QuantGroup(
            label="Qwen2.5-72B @ FP16", engine="vllm", model="Qwen2.5-72B-Instruct",
            quantization="fp16", stages=_make_stages(90.0, 0.42), vram_peak_mib=68000.0,
        ),
        QuantGroup(
            label="Qwen2.5-72B @ AWQ", engine="vllm", model="Qwen2.5-72B-Instruct-AWQ",
            quantization="awq", stages=_make_stages(138.0, 0.38), vram_peak_mib=42500.0,
        ),
    ]


# ----------------------------------------------------------------------
# 格式化
# ----------------------------------------------------------------------
class TestFormatting:
    @pytest.mark.parametrize(
        "value,expected",
        [(0.0, "0.0000 s"), (1.5, "1.5000 s"), (None, "N/A")],
    )
    def test_seconds(self, value, expected):
        assert seconds(value) == expected

    def test_ms_converts_from_seconds(self):
        assert ms(0.42) == "420.00 ms"
        assert ms(None) == "N/A"

    @pytest.mark.parametrize(
        "value,needle",
        [
            (0.0, "0 s"),
            (2.5, "2.500 s"),
            (0.42, "420.000 ms"),
            (0.00042, "420.000 µs"),
            (0.00000042, "420.000 ns"),
        ],
    )
    def test_adaptive_units(self, value, needle):
        """自适应单位：耗时跨 4 个数量级，固定单位会显示成 0.0000。"""
        assert adaptive_seconds(value) == needle

    def test_adaptive_none(self):
        assert adaptive_seconds(None) == "N/A"

    def test_pct_from_ratio(self):
        assert pct(0.0523) == "5.2%"
        assert pct(None) == "N/A"

    def test_tps(self):
        assert tps(1234.56) == "1234.6 tokens/s"

    def test_mib_switches_to_gib(self):
        assert mib(512.0) == "512.0 MiB"
        assert mib(2048.0) == "2.0 GiB"

    def test_ratio(self):
        assert ratio(1.53) == "1.53×"

    def test_signed_pct(self):
        assert signed_pct(12.34) == "+12.3%"
        assert signed_pct(-5.0) == "-5.0%"

    def test_num_digits(self):
        assert num(1.23456, 2) == "1.23"
        assert num(None) == "N/A"

    def test_num_handles_non_numeric(self):
        """传入字符串不能崩——报告里可能有已经格式化好的值。"""
        assert num("abc") == "N/A"


# ----------------------------------------------------------------------
# SVG 图表
# ----------------------------------------------------------------------
class TestSvgCharts:
    def test_multi_line_produces_valid_svg(self):
        series = [
            {"name": "A", "color": "#2563eb", "points": [(1, 10.0), (2, 18.0), (4, 30.0)]},
            {"name": "B", "color": "#ea580c", "points": [(1, 20.0), (2, 36.0), (4, 60.0)]},
        ]
        svg = multi_line_chart(series, x_label="并发", y_label="tokens/s", title="吞吐")
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert "并发" in svg
        assert "tokens/s" in svg
        # 每条线一个 polyline
        assert svg.count("<polyline") == 2
        # 每个点一个 circle
        assert svg.count("<circle") == 6

    def test_empty_series_shows_placeholder(self):
        svg = multi_line_chart([])
        assert "<svg" in svg
        assert "暂无数据" in svg

    def test_points_with_none_are_skipped(self):
        series = [{"name": "A", "color": "#000", "points": [(1, 10.0), (2, None), (4, 30.0)]}]
        svg = multi_line_chart(series)
        assert svg.count("<circle") == 2

    def test_single_point_no_polyline(self):
        series = [{"name": "A", "color": "#000", "points": [(1, 10.0)]}]
        svg = multi_line_chart(series)
        assert "<circle" in svg
        assert "<polyline" not in svg

    def test_log_axis_for_wide_range(self):
        """并发 1..64 跨度大时用对数轴，否则小并发会挤在左侧。"""
        series = [
            {"name": "A", "color": "#000", "points": [(1, 10.0), (8, 50.0), (64, 200.0)]}
        ]
        svg = multi_line_chart(series)
        # 对数轴下，1 在中点左侧但不会紧贴左边界，8 应落在中间附近
        xs = [float(m) for m in re.findall(r'<circle cx="([\d.]+)"', svg)]
        assert len(xs) == 3
        # 线性轴时 8 会在约 11% 处；对数轴时应在 50% 左右
        plot_left, plot_w = 78, 860 - 78 - 34
        rel = (xs[1] - plot_left) / plot_w
        assert 0.4 < rel < 0.6, f"8 的相对位置 {rel:.2f}，不像对数轴"

    def test_bar_chart(self):
        items = [
            {"label": "FP16", "value": 100.0, "color": "#2563eb"},
            {"label": "AWQ", "value": 150.0, "color": "#059669"},
        ]
        svg = bar_chart(items, y_label="tokens/s", title="对比")
        assert svg.startswith("<svg")
        assert svg.count("<rect") >= 3  # 背景 + 2 柱
        assert "150" in svg or "150.0" in svg

    def test_bar_chart_empty(self):
        assert "暂无数据" in bar_chart([])

    def test_bar_chart_skips_none_values(self):
        items = [{"label": "A", "value": 10.0}, {"label": "B", "value": None}]
        svg = bar_chart(items)
        assert "B" in svg  # 标签仍在
        assert svg.count('rx="4"') == 1  # 只有一个柱

    def test_multiline_labels_in_bar(self):
        """柱状图标签含换行时要渲染成多行 text。"""
        items = [{"label": "line1\nline2", "value": 10.0}]
        svg = bar_chart(items)
        assert "line1" in svg
        assert "line2" in svg

    def test_html_escaping_in_labels(self):
        """标签里的尖括号必须被转义，否则会破坏 HTML。"""
        series = [{"name": "<script>", "color": "#000", "points": [(1, 1.0)]}]
        svg = multi_line_chart(series)
        assert "<script>" not in svg
        assert "&lt;script&gt;" in svg


# ----------------------------------------------------------------------
# Markdown 报告
# ----------------------------------------------------------------------
class TestMarkdownReport:
    def test_all_sections_present(self, two_results):
        md = build_markdown(two_results)
        for heading in (
            "# 大模型推理性能压测报告",
            "## 1. 概览",
            "## 2. 环境与配置",
            "## 3. 吞吐 - 并发曲线",
            "## 4. 延迟指标",
            "## 5. 显存占用曲线",
            "## 6. 量化等级横向对比",
            "## 7. 结论与建议",
        ):
            assert heading in md, f"缺少章节：{heading}"

    def test_engines_listed(self, two_results):
        md = build_markdown(two_results)
        assert "vllm" in md
        assert "ollama" in md
        assert "Qwen2.5-72B-Instruct" in md
        assert "qwen2.5:7b" in md

    def test_environment_recorded(self, two_results):
        md = build_markdown(two_results)
        assert "Linux-6.5.0" in md
        assert "3.11.7" in md
        assert "gpu-node-01" in md

    def test_prompt_recorded_for_reproducibility(self, two_results):
        md = build_markdown(two_results)
        assert "请用三句话说明什么是张量并行" in md

    def test_no_none_leaks(self, two_results):
        """报告里不能出现裸的 None——那是没处理的空值。"""
        md = build_markdown(two_results)
        # 允许出现在代码块里的 Python None，但表格单元格里不该有
        table_lines = [line for line in md.splitlines() if line.startswith("|")]
        for line in table_lines:
            assert "| None |" not in line, f"表格里有裸 None: {line}"

    def test_conclusion_mentions_best_engine(self, two_results):
        md = build_markdown(two_results)
        # ollama 吞吐更高（210 基准 > 90）
        assert "吞吐最优" in md
        assert "ollama" in md

    def test_quant_section_with_matrix(self, two_results, quant_groups):
        md = build_markdown(two_results, compare(quant_groups))
        assert "对比基准" in md
        assert "AWQ" in md
        assert "每 GiB 吞吐" in md

    def test_quant_section_without_matrix(self, two_results):
        """没给量化对比时要有占位说明，而不是空着。"""
        md = build_markdown(two_results)
        assert "未提供多量化配置" in md
        assert "compare()" in md

    def test_vram_placeholder_when_unavailable(self):
        """无显存数据时必须给出原因说明，而不是留空。"""
        results = [_result("vllm", "m", "fp16", 90.0, 0.42)]  # 不带 vram_curve
        md = build_markdown(results)
        assert "未采集到 GPU 显存数据" in md
        assert "nvidia-smi" in md

    def test_vram_table_when_available(self):
        results = [_result("vllm", "m", "fp16", 90.0, 0.42, vram=68000.0,
                           with_vram_curve=True)]
        md = build_markdown(results)
        assert "RTX PRO 5000" in md
        assert "峰值占比" in md
        assert "显存随并发变化" in md
        # 有数据时不该出现"未采集到"的占位文案
        assert "未采集到 GPU 显存数据" not in md

    def test_empty_results_does_not_crash(self):
        md = build_markdown([])
        assert "# 大模型推理性能压测报告" in md

    def test_single_result(self):
        r = _result("vllm", "m", "fp16", 90.0, 0.42)
        md = build_markdown([r])
        assert "vllm" in md
        assert "## 7. 结论与建议" in md

    def test_reproduce_commands_present(self, two_results):
        md = build_markdown(two_results)
        assert "python scripts/run_bench.py" in md

    def test_custom_title(self, two_results):
        md = build_markdown(two_results, title="自定义标题")
        assert "# 自定义标题" in md


# ----------------------------------------------------------------------
# HTML 报告
# ----------------------------------------------------------------------
class TestHtmlReport:
    def test_valid_document(self, two_results):
        html = build_html(two_results)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert '<meta charset="utf-8">' in html

    def test_no_external_resources(self, two_results):
        """交付的 HTML 必须能离线打开——不能有任何外链。"""
        html = build_html(two_results)
        assert "src=" not in html or "data:" in html
        assert "href=\"http" not in html
        assert "cdn." not in html
        assert "<script" not in html

    def test_has_inline_svg_charts(self, two_results):
        html = build_html(two_results)
        assert html.count("<svg") >= 3  # 吞吐 / TTFT / P99
        assert "</svg>" in html

    def test_has_all_sections(self, two_results):
        html = build_html(two_results)
        for heading in (
            "1. 概览", "2. 环境与配置", "3. 吞吐 - 并发曲线",
            "4. 延迟指标", "5. 显存占用曲线", "6. 量化等级横向对比",
            "7. 结论与建议",
        ):
            assert heading in html

    def test_style_inlined(self, two_results):
        html = build_html(two_results)
        assert "<style>" in html
        assert "--accent" in html

    def test_chinese_lang_attribute(self, two_results):
        html = build_html(two_results)
        assert 'lang="zh-CN"' in html

    def test_escapes_user_content(self):
        """模型名里带尖括号时不能破坏 HTML 结构。"""
        r = _result("vllm<script>alert(1)</script>", "m", "fp16", 90.0, 0.42)
        html = build_html([r])
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_quant_bar_chart_with_matrix(self, two_results, quant_groups):
        html = build_html(two_results, compare(quant_groups))
        assert "各量化配置峰值吞吐" in html

    def test_vram_chart_when_available(self):
        results = [_result("vllm", "m", "fp16", 90.0, 0.42, vram=68000.0,
                           with_vram_curve=True)]
        html = build_html(results)
        assert "显存占用曲线" in html
        assert "RTX PRO 5000" in html

    def test_empty_results_does_not_crash(self):
        html = build_html([])
        assert "<!DOCTYPE html>" in html

    def test_badge_classes_used(self, two_results):
        html = build_html(two_results)
        assert "badge" in html
        assert "health" not in html.lower() or "健康" in html


# ----------------------------------------------------------------------
# 统一入口
# ----------------------------------------------------------------------
class TestGenerateReports:
    def test_writes_both_formats(self, tmp_path, two_results):
        paths = generate_reports(two_results, tmp_path)
        assert paths["markdown"].exists()
        assert paths["html"].exists()
        assert paths["markdown"].suffix == ".md"
        assert paths["html"].suffix == ".html"

    def test_computes_matrix_from_groups(self, tmp_path, two_results, quant_groups):
        paths = generate_reports(two_results, tmp_path, quant_groups=quant_groups)
        matrix = paths["matrix"]
        assert len(matrix["groups"]) == 2
        assert matrix["baseline"]["quantization"] == "fp16"

    def test_accepts_precomputed_matrix(self, tmp_path, two_results, quant_groups):
        matrix = compare(quant_groups)
        paths = generate_reports(two_results, tmp_path, quant_matrix=matrix)
        assert paths["matrix"] is matrix

    def test_matrix_takes_precedence(self, tmp_path, two_results, quant_groups):
        """同时给 groups 和 matrix 时，以 matrix 为准，不重复计算。"""
        matrix = compare(quant_groups)
        matrix["custom_marker"] = True
        paths = generate_reports(
            two_results, tmp_path, quant_groups=quant_groups, quant_matrix=matrix
        )
        assert paths["matrix"].get("custom_marker") is True

    def test_creates_output_dir(self, tmp_path, two_results):
        target = tmp_path / "deep" / "nested"
        paths = generate_reports(two_results, target)
        assert paths["html"].exists()

    def test_custom_basename(self, tmp_path, two_results):
        paths = generate_reports(two_results, tmp_path, basename="bench_report")
        assert paths["markdown"].name == "bench_report.md"
        assert paths["html"].name == "bench_report.html"

    def test_custom_title(self, tmp_path, two_results):
        paths = generate_reports(two_results, tmp_path, title="专项压测")
        assert "专项压测" in paths["markdown"].read_text(encoding="utf-8")

    def test_empty_matrix_when_no_groups(self, tmp_path, two_results):
        paths = generate_reports(two_results, tmp_path)
        assert paths["matrix"] == {}

    def test_html_is_utf8_readable(self, tmp_path, two_results):
        paths = generate_reports(two_results, tmp_path)
        content = paths["html"].read_text(encoding="utf-8")
        assert "大模型推理性能压测报告" in content
