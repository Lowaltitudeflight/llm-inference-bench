"""量化等级横向评测。

同一个模型用不同量化精度部署，代价与收益完全不同：

===========  ==========  ==============  ================
量化         显存占用     推理速度        精度损失
===========  ==========  ==============  ================
FP16/BF16    基准 100%   基准 100%       无（基准）
INT8         约 50%      约 100~130%     极小
AWQ (W4A16)  约 25~30%   约 120~180%     小幅
GPTQ (W4A16) 约 25~30%   约 120~180%     小幅
===========  ==========  ==============  ================

本模块不负责「跑压测」——那是 :mod:`bench.runner` 的职责；
它负责把多组「引擎 + 量化等级」的压测结果**对齐成可比较的矩阵**，
并算出相对 FP16 基准的收益倍数。

设计要点
--------
1. **基准可选**：默认以第一个标记为 ``is_baseline`` 的组为基准，
   找不到则取量化等级为 fp16/bf16 的组，再找不到就取第一组。
2. **指标归一**：不同组的 max_tokens / 并发档位可能不同，只对
   **共同存在的并发档位**做对比，避免拿 32 并发去比 1 并发。
3. **吞吐与显存是主要看板**：量化最直接的收益就是「同样显存能塞下
   更多并发」与「同样并发吞吐更高」。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from metrics.stats import StageMetrics, percentile

logger = logging.getLogger(__name__)

# 视为「无量化基准」的等级名
BASELINE_QUANTS = {"fp16", "bf16", "float16", "bfloat16", "none", "unknown"}


@dataclass
class QuantGroup:
    """一组「同一模型 + 同一量化等级」的压测结果。

    Attributes:
        label: 展示名，如 ``Qwen2.5-72B @ AWQ``。
        engine: 引擎名（vllm / ollama）。
        model: 模型名。
        quantization: 量化等级字符串。
        stages: 该组的各并发档位指标。
        vram_peak_mib: 压测期间显存峰值（MiB），可选。
        extra: 其他元信息（卡数、机器规格等）。
    """

    label: str
    engine: str
    model: str
    quantization: str
    stages: List[StageMetrics] = field(default_factory=list)
    vram_peak_mib: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_baseline_candidate(self) -> bool:
        """是否可作为未量化基准。"""
        return self.quantization.strip().lower() in BASELINE_QUANTS

    def stage_map(self) -> Dict[int, StageMetrics]:
        """并发数 -> 档位指标。"""
        return {s.concurrency: s for s in self.stages}

    def best_throughput(self) -> Optional[float]:
        """该组达到的最高输出吞吐（tokens/s）。"""
        values = [s.throughput_output for s in self.stages if s.throughput_output is not None]
        return max(values) if values else None

    def best_throughput_concurrency(self) -> Optional[int]:
        candidates = [s for s in self.stages if s.throughput_output is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.throughput_output).concurrency

    def lowest_ttft(self) -> Optional[float]:
        values = [s.ttft_mean for s in self.stages if s.ttft_mean is not None]
        return min(values) if values else None

    def mean_tpot(self) -> Optional[float]:
        """所有成功请求的 TPOT 均值（按档位平均再平均，近似即可）。"""
        values = [s.tpot_mean for s in self.stages if s.tpot_mean is not None]
        return sum(values) / len(values) if values else None

    def peak_concurrency_stable(self, max_failure_rate: float = 0.05) -> Optional[int]:
        """仍保持稳定的最高并发档位。用于判断「能扛到多少并发」。"""
        stable = [
            s.concurrency
            for s in self.stages
            if s.success_requests > 0 and s.failure_rate <= max_failure_rate
        ]
        return max(stable) if stable else None


def _ratio(value: Optional[float], base: Optional[float]) -> Optional[float]:
    """相对倍数，分母为 0 或 None 时返回 None。"""
    if value is None or base is None or base == 0:
        return None
    return value / base


def _pct_delta(value: Optional[float], base: Optional[float]) -> Optional[float]:
    """相对变化百分比（正数表示变差/变大）。"""
    if value is None or base is None or base == 0:
        return None
    return (value - base) / base * 100.0


def pick_baseline(groups: List[QuantGroup]) -> Optional[QuantGroup]:
    """选出基准组。

    优先级：
    1. ``extra['is_baseline'] is True`` 的组；
    2. 量化等级属于 :data:`BASELINE_QUANTS` 的组；
    3. 第一组。
    """
    if not groups:
        return None
    for g in groups:
        if g.extra.get("is_baseline") is True:
            return g
    for g in groups:
        if g.is_baseline_candidate:
            return g
    logger.warning("未找到 FP16/BF16 基准组，退化使用第一组 %s 作为对比基准", groups[0].label)
    return groups[0]


def compare(
    groups: List[QuantGroup],
    max_failure_rate: float = 0.05,
) -> Dict[str, Any]:
    """生成量化横向对比矩阵。

    Args:
        groups: 待对比的量化组，至少 1 组。
        max_failure_rate: 判定档位「稳定」的失败率阈值。

    Returns:
        {
          "baseline": {label, quantization},
          "groups": [每组一行，含绝对指标与相对基准的倍数],
          "common_concurrency": [共同并发档位],
          "by_concurrency": {并发: [每组在该并发的吞吐/延迟]},
          "conclusion": 文字结论,
        }
    """
    if not groups:
        return {
            "baseline": None,
            "groups": [],
            "common_concurrency": [],
            "by_concurrency": {},
            "conclusion": "没有可对比的量化组。",
        }

    baseline = pick_baseline(groups)
    base_tp = baseline.best_throughput() if baseline else None
    base_ttft = baseline.lowest_ttft() if baseline else None
    base_tpot = baseline.mean_tpot() if baseline else None
    base_vram = baseline.vram_peak_mib if baseline else None

    rows: List[Dict[str, Any]] = []
    for g in groups:
        best_tp = g.best_throughput()
        ttft = g.lowest_ttft()
        tpot = g.mean_tpot()
        rows.append(
            {
                "label": g.label,
                "engine": g.engine,
                "model": g.model,
                "quantization": g.quantization,
                "is_baseline": g is baseline,
                "best_throughput_tps": best_tp,
                "best_throughput_concurrency": g.best_throughput_concurrency(),
                "lowest_ttft_s": ttft,
                "mean_tpot_s": tpot,
                "vram_peak_mib": g.vram_peak_mib,
                "peak_stable_concurrency": g.peak_concurrency_stable(max_failure_rate),
                # 相对基准
                "throughput_vs_baseline": _ratio(best_tp, base_tp),
                "ttft_delta_pct_vs_baseline": _pct_delta(ttft, base_ttft),
                "tpot_delta_pct_vs_baseline": _pct_delta(tpot, base_tpot),
                "vram_ratio_vs_baseline": _ratio(g.vram_peak_mib, base_vram),
                # 显存效率：单位显存产出的吞吐，量化收益的核心指标
                "throughput_per_gib": (
                    best_tp / (g.vram_peak_mib / 1024.0)
                    if best_tp and g.vram_peak_mib
                    else None
                ),
                "stage_count": len(g.stages),
            }
        )

    # ---- 共同并发档位上的逐点对比 ----
    maps = [g.stage_map() for g in groups]
    common = sorted(set.intersection(*[set(m.keys()) for m in maps])) if maps else []
    by_concurrency: Dict[str, List[Dict[str, Any]]] = {}
    for level in common:
        by_concurrency[str(level)] = [
            {
                "label": g.label,
                "quantization": g.quantization,
                "throughput_output": (m[level].throughput_output if m.get(level) else None),
                "ttft_mean": (m[level].ttft_mean if m.get(level) else None),
                "tpot_mean": (m[level].tpot_mean if m.get(level) else None),
                "latency_p99": (m[level].latency_p99 if m.get(level) else None),
                "failure_rate": (m[level].failure_rate if m.get(level) else None),
            }
            for g, m in zip(groups, maps)
        ]

    return {
        "baseline": (
            {"label": baseline.label, "quantization": baseline.quantization}
            if baseline
            else None
        ),
        "groups": rows,
        "common_concurrency": common,
        "by_concurrency": by_concurrency,
        "conclusion": _build_conclusion(rows),
    }


def _build_conclusion(rows: List[Dict[str, Any]]) -> str:
    """从对比矩阵生成一句话结论。"""
    if not rows:
        return "没有可对比的量化组。"

    non_base = [r for r in rows if not r["is_baseline"] and r["throughput_vs_baseline"]]
    if not non_base:
        return "仅有基准组，缺少可对比的量化配置。"

    # 吞吐提升最多的一组
    best_tp = max(non_base, key=lambda r: r["throughput_vs_baseline"])
    parts: List[str] = []

    gain = (best_tp["throughput_vs_baseline"] - 1) * 100
    if gain > 1:
        parts.append(
            f"吞吐最优为 {best_tp['label']}（{best_tp['quantization']}），"
            f"相比基准提升 {gain:.1f}%"
        )
    else:
        parts.append(
            f"量化未带来吞吐提升，最佳组 {best_tp['label']} 仅与基准持平"
            f"（{gain:+.1f}%）"
        )

    # 显存压力最小的一组
    vram_rows = [r for r in non_base if r["vram_ratio_vs_baseline"] is not None]
    if vram_rows:
        least = min(vram_rows, key=lambda r: r["vram_ratio_vs_baseline"])
        saving = (1 - least["vram_ratio_vs_baseline"]) * 100
        if saving > 1:
            parts.append(
                f"显存占用最低为 {least['label']}，较基准降低 {saving:.1f}%"
            )

    # 延迟是否恶化
    ttft_rows = [r for r in non_base if r["ttft_delta_pct_vs_baseline"] is not None]
    if ttft_rows:
        worst = max(ttft_rows, key=lambda r: r["ttft_delta_pct_vs_baseline"])
        if worst["ttft_delta_pct_vs_baseline"] > 10:
            parts.append(
                f"注意 {worst['label']} 的首包延迟较基准恶化 "
                f"{worst['ttft_delta_pct_vs_baseline']:.1f}%"
            )

    return "；".join(parts) + "。"


def summary_verdict(matrix: Dict[str, Any]) -> List[str]:
    """把对比矩阵转成几条可直接贴进报告的要点。"""
    rows = matrix.get("groups") or []
    if not rows:
        return ["无有效数据。"]

    lines: List[str] = []
    base = matrix.get("baseline")
    if base:
        lines.append(f"对比基准：{base['label']}（{base['quantization']}）")

    for r in rows:
        if r["is_baseline"]:
            continue
        tp = r["throughput_vs_baseline"]
        vram = r["vram_ratio_vs_baseline"]
        seg = f"- **{r['label']}**: "
        seg += f"吞吐 {_fmt(r['best_throughput_tps'], 'tokens/s')}"
        if tp is not None:
            seg += f"（基准 {tp:.2f}×）"
        if vram is not None:
            seg += f"，显存 {vram:.2f}× 基准"
        if r["throughput_per_gib"]:
            seg += f"，每 GiB 显存吞吐 {r['throughput_per_gib']:.1f} tokens/s"
        lines.append(seg)

    lines.append(matrix.get("conclusion", ""))
    return lines


def _fmt(value: Optional[float], unit: str, digits: int = 2) -> str:
    return f"{value:.{digits}f} {unit}" if value is not None else "N/A"


def from_result_dicts(items: List[Dict[str, Any]]) -> List[QuantGroup]:
    """从落盘的 JSON 结果重建量化组。

    便于「先分头压测、后统一对比」的工作流：每次压测把结果存成
    JSON，最后统一喂给 :func:`compare`。

    Args:
        items: 形如
            [{"label","engine","model","quantization",
              "vram_peak_mib","stages":[StageMetrics.to_dict()...,]}]
    """
    groups: List[QuantGroup] = []
    for item in items:
        stages: List[StageMetrics] = []
        for raw in item.get("stages", []):
            stages.append(_stage_from_dict(raw))
        groups.append(
            QuantGroup(
                label=item.get("label") or item.get("model", "unknown"),
                engine=item.get("engine", "unknown"),
                model=item.get("model", "unknown"),
                quantization=item.get("quantization", "unknown"),
                stages=stages,
                vram_peak_mib=item.get("vram_peak_mib"),
                extra=item.get("extra", {}),
            )
        )
    return groups


def _stage_from_dict(raw: Dict[str, Any]) -> StageMetrics:
    """把 :meth:`StageMetrics.to_dict` 的嵌套结构还原成对象。"""
    latency = raw.get("latency", {})
    ttft = raw.get("ttft", {})
    tpot = raw.get("tpot", {})
    throughput = raw.get("throughput", {})
    timing = raw.get("timing", {})
    errors = raw.get("errors", {})

    return StageMetrics(
        concurrency=raw.get("concurrency", 0),
        total_requests=raw.get("total_requests", 0),
        success_requests=raw.get("success_requests", 0),
        failed_requests=raw.get("failed_requests", 0),
        failure_rate=raw.get("failure_rate", 0.0),
        unstable=raw.get("unstable", False),
        latency_mean=latency.get("mean"),
        latency_p50=latency.get("p50"),
        latency_p95=latency.get("p95"),
        latency_p99=latency.get("p99"),
        latency_min=latency.get("min"),
        latency_max=latency.get("max"),
        latency_stdev=latency.get("stdev"),
        ttft_mean=ttft.get("mean"),
        ttft_p50=ttft.get("p50"),
        ttft_p95=ttft.get("p95"),
        ttft_p99=ttft.get("p99"),
        tpot_mean=tpot.get("mean"),
        tpot_p50=tpot.get("p50"),
        tpot_p95=tpot.get("p95"),
        tpot_p99=tpot.get("p99"),
        output_tokens_total=throughput.get("output_tokens_total", 0),
        input_tokens_total=throughput.get("input_tokens_total", 0),
        throughput_output=throughput.get("output_tokens_per_second"),
        throughput_total=throughput.get("total_tokens_per_second"),
        per_request_throughput_mean=throughput.get("per_request_output_tps_mean"),
        wall_time=timing.get("wall_time"),
        request_rate=timing.get("request_rate"),
        errors=errors.get("samples", []),
        error_breakdown=errors.get("breakdown", {}),
    )
