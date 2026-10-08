"""Markdown 对比报告生成。

报告结构固定为七节，对应一个「能拿给客户/面试官看」的压测结论：

1. 概览（结论先行）
2. 环境与配置（可复现性）
3. 吞吐 - 并发曲线（容量规划的核心）
4. 延迟指标（TTFT / TPOT / P99，体验的核心）
5. 显存占用曲线（成本的核心）
6. 量化横向对比（性价比）
7. 结论与建议

设计上坚持「先结论、后数据」。压测报告最常见的失败是给出一堆
表格让人自己算，正确的做法是每节开头直接给判断。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from metrics.stats import StageMetrics
from monitor.quant import _stage_from_dict
from report.fmt import adaptive_seconds, mib, ms, num, pct, ratio, signed_pct, tps

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 表格构造
# ----------------------------------------------------------------------
def _table(headers: List[str], rows: List[List[str]]) -> str:
    """构造 Markdown 表格。"""
    if not rows:
        return "_无数据_\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def _stages_of(result: Dict[str, Any]) -> List[StageMetrics]:
    return [_stage_from_dict(s) for s in result.get("stages", [])]


# ----------------------------------------------------------------------
# 各小节
# ----------------------------------------------------------------------
def _section_overview(results: List[Dict[str, Any]]) -> str:
    """结论先行：把最关键的数字提到最前面。"""
    lines = ["## 1. 概览\n"]
    rows = []
    for r in results:
        summary = r.get("summary") or {}
        stages = _stages_of(r)
        peak = summary.get("best_throughput")
        peak_c = summary.get("best_throughput_concurrency")
        ttft = summary.get("lowest_ttft")
        vram = r.get("vram_peak_mib")

        # P99 取达到峰值吞吐那一档的 P99，代表「满载体验」
        p99 = None
        tpot = None
        if peak_c is not None:
            for s in stages:
                if s.concurrency == peak_c:
                    p99 = s.latency_p99
                    tpot = s.tpot_mean
                    break

        rows.append(
            [
                f"**{r.get('engine', '?')}** / `{r.get('model', '?')}`",
                r.get("quantization", "unknown"),
                tps(peak),
                str(peak_c) if peak_c is not None else "N/A",
                ms(ttft),
                ms(tpot),
                ms(p99),
                mib(vram),
            ]
        )

    lines.append(
        _table(
            [
                "引擎 / 模型",
                "量化",
                "峰值吞吐",
                "对应并发",
                "最优 TTFT",
                "TPOT",
                "P99 延迟",
                "显存峰值",
            ],
            rows,
        )
    )

    # 一段自动生成的结论
    lines.append("**结论**\n")
    for line in _verdict_lines(results):
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def _verdict_lines(results: List[Dict[str, Any]]) -> List[str]:
    """跨引擎自动生成结论要点。"""
    lines: List[str] = []
    scored = [
        (r, (r.get("summary") or {}).get("best_throughput"))
        for r in results
    ]
    scored = [(r, tp) for r, tp in scored if tp]
    if not scored:
        return ["- 本次压测未取得有效吞吐数据，请检查引擎健康状态与网络连通性。"]

    winner, best_tp = max(scored, key=lambda x: x[1])
    lines.append(
        f"- 吞吐最优：**{winner.get('engine')}**（`{winner.get('model')}`，"
        f"{winner.get('quantization')}），峰值 **{tps(best_tp)}**，"
        f"出现在并发 {winner.get('summary', {}).get('best_throughput_concurrency')}。"
    )

    if len(scored) > 1:
        for r, tp in scored:
            if r is winner:
                continue
            gap = (best_tp - tp) / tp * 100 if tp else None
            lines.append(
                f"- {r.get('engine')}（{r.get('quantization')}）峰值 "
                f"{tps(tp)}，比最优低 {gap:.1f}%。"
                if gap is not None
                else f"- {r.get('engine')} 未取得吞吐数据。"
            )

    # 稳定性提示
    unstable_all: List[str] = []
    for r in results:
        unstable = (r.get("summary") or {}).get("unstable_stages") or []
        if unstable:
            unstable_all.append(f"{r.get('engine')}@{unstable}")
    if unstable_all:
        lines.append(
            f"- ⚠️ 以下档位失败率超阈值，数据仅供参考：{', '.join(unstable_all)}。"
        )
    return lines


def _section_environment(results: List[Dict[str, Any]]) -> str:
    """环境与配置，保证结果可复现。"""
    lines = ["## 2. 环境与配置\n"]
    first = results[0] if results else {}
    cfg = first.get("config", {})
    env = first.get("environment", {})

    lines.append("### 2.1 运行环境\n")
    lines.append(
        _table(
            ["项", "值"],
            [
                ["操作系统", env.get("platform", "N/A")],
                ["Python", env.get("python", "N/A")],
                ["CPU", env.get("processor", "N/A")],
                ["主机名", env.get("hostname", "N/A")],
                ["生成时间", first.get("generated_at", "N/A")],
            ],
        )
    )

    lines.append("\n### 2.2 压测参数\n")
    lines.append(
        _table(
            ["项", "值"],
            [
                ["并发梯度", ", ".join(str(c) for c in cfg.get("concurrency_levels", []))],
                ["每请求最大生成", f"{cfg.get('max_tokens', 'N/A')} tokens"],
                ["预热请求数", str(cfg.get("warmup_requests", "N/A"))],
                ["温度", str(cfg.get("temperature", "N/A"))],
                ["流式", "是" if cfg.get("stream") else "否"],
                ["单请求超时", f"{cfg.get('timeout', 'N/A')} s"],
            ],
        )
    )

    lines.append("\n### 2.3 被测端点\n")
    lines.append(
        _table(
            ["引擎", "模型", "量化", "接入地址", "健康检查"],
            [
                [
                    r.get("engine", "?"),
                    f"`{r.get('model', '?')}`",
                    r.get("quantization", "unknown"),
                    r.get("base_url", "N/A"),
                    _health_text(r.get("health", {})),
                ]
                for r in results
            ],
        )
    )
    lines.append(
        f"\n> 压测提示词（固定）：`{cfg.get('prompt', 'N/A')}`\n"
    )
    return "\n".join(lines)


def _health_text(health: Dict[str, Any]) -> str:
    if not health:
        return "未检测"
    if health.get("ok"):
        loaded = health.get("model_loaded")
        return "✅ 正常" + ("" if loaded else "（⚠️ 目标模型未在列表中）")
    return f"❌ {health.get('error', '不可达')}"


def _section_throughput(results: List[Dict[str, Any]]) -> str:
    """吞吐 - 并发曲线。这是容量规划最直接的一张表。"""
    lines = ["## 3. 吞吐 - 并发曲线\n"]
    lines.append(
        "输出吞吐（tokens/s）随并发上升而增长，直到触及算力或显存瓶颈后饱和。"
        "**饱和点即服务的实际容量上限**，超过它继续加压只会推高延迟。\n"
    )

    headers = ["并发"] + [f"{r.get('engine')} ({r.get('quantization')})" for r in results]
    all_levels: List[int] = []
    for r in results:
        all_levels.extend(s.concurrency for s in _stages_of(r))
    levels = sorted(set(all_levels))

    rows = []
    for level in levels:
        row = [str(level)]
        for r in results:
            stage = next((s for s in _stages_of(r) if s.concurrency == level), None)
            row.append(tps(stage.throughput_output) if stage else "—")
        rows.append(row)
    lines.append(_table(headers, rows))

    lines.append("\n### 3.1 相对加速比\n")
    lines.append("以各引擎自身并发 1 的吞吐为 1.0 基准，观察扩展效率。\n")
    rows2 = []
    for level in levels:
        row = [str(level)]
        for r in results:
            stages = _stages_of(r)
            base = next((s for s in stages if s.concurrency == 1), None)
            cur = next((s for s in stages if s.concurrency == level), None)
            if base and cur and base.throughput_output and cur.throughput_output:
                row.append(f"{cur.throughput_output / base.throughput_output:.2f}×")
            else:
                row.append("—")
        rows2.append(row)
    lines.append(_table(headers, rows2))
    return "\n".join(lines)


def _section_latency(results: List[Dict[str, Any]]) -> str:
    """延迟指标：TTFT / TPOT / P99。"""
    lines = ["## 4. 延迟指标\n"]
    lines.append(
        "**TTFT**（首包延迟）决定用户感知的响应速度；**TPOT**（单 token 延迟）"
        "决定出字的流畅度；**P99** 反映长尾，线上体验往往由它决定。\n"
    )

    for r in results:
        head = f"### 4.{results.index(r) + 1} {r.get('engine')} / `{r.get('model')}`（{r.get('quantization')}）\n"
        lines.append(head)
        rows = []
        for s in _stages_of(r):
            rows.append(
                [
                    str(s.concurrency),
                    pct(s.failure_rate),
                    ms(s.ttft_mean),
                    ms(s.ttft_p95),
                    ms(s.ttft_p99),
                    ms(s.tpot_mean),
                    ms(s.latency_mean),
                    ms(s.latency_p95),
                    ms(s.latency_p99),
                    ms(s.latency_max),
                ]
            )
        lines.append(
            _table(
                [
                    "并发",
                    "失败率",
                    "TTFT 均值",
                    "TTFT P95",
                    "TTFT P99",
                    "TPOT 均值",
                    "端到端均值",
                    "P95",
                    "P99",
                    "最大值",
                ],
                rows,
            )
        )
        lines.append("")
    return "\n".join(lines)


def _section_vram(results: List[Dict[str, Any]]) -> str:
    """显存占用曲线。"""
    lines = ["## 5. 显存占用曲线\n"]
    available = [r for r in results if (r.get("vram_curve") or {}).get("available")]
    if not available:
        lines.append(
            "本次压测未采集到 GPU 显存数据（可能原因：非 NVIDIA 环境、"
            "容器内无 `nvidia-smi`、或监控被关闭）。\n"
        )
        lines.append(
            "采集方式：压测全程由后台线程调用 "
            "`nvidia-smi --query-gpu=...` 采样，间隔见配置。\n"
        )
        return "\n".join(lines)

    lines.append(
        "显存占用决定「同样一台机器能同时服务多少请求」，是私有化交付里"
        "最硬的成本约束。以下为压测全程采样得到的峰值。\n"
    )

    rows = []
    for r in available:
        curve = r["vram_curve"]
        for g in curve.get("gpus", []):
            rows.append(
                [
                    r.get("engine", "?"),
                    str(g.get("index")),
                    g.get("name", "N/A"),
                    mib(g.get("memory_total_mib")),
                    mib(g.get("peak_memory_used_mib")),
                    pct(
                        (g.get("peak_memory_used_percent") or 0) / 100.0
                        if g.get("peak_memory_used_percent") is not None
                        else None
                    ),
                    pct(
                        (g.get("mean_util_percent") or 0) / 100.0
                        if g.get("mean_util_percent") is not None
                        else None
                    ),
                    str(g.get("sample_count", 0)),
                ]
            )
    lines.append(
        _table(
            ["引擎", "卡号", "型号", "总显存", "峰值占用", "峰值占比", "平均利用率", "采样点"],
            rows,
        )
    )

    lines.append("\n### 5.1 显存随并发变化\n")
    lines.append(
        "下表为各并发档位加压期间的全卡显存占用之和。若曲线随并发明显上升，"
        "说明 KV Cache 是显存增长的主因，进而决定了最大并发。\n"
    )
    header = ["并发"] + [f"{r.get('engine')}" for r in available]
    rows2 = []
    all_levels: List[int] = []
    for r in results:
        all_levels.extend(s.concurrency for s in _stages_of(r))
    for level in sorted(set(all_levels)):
        row = [str(level)]
        for r in available:
            row.append(_level_mem(r, level))
        rows2.append(row)
    lines.append(_table(header, rows2))
    return "\n".join(lines)


def _level_mem(result: Dict[str, Any], level: int) -> str:
    """取某并发档位的显存峰值。优先用 curve 里的分段信息。"""
    peaks = (result.get("vram_curve") or {}).get("stage_peak_mib") or {}
    value = peaks.get(str(level))
    return mib(value) if value is not None else "—"


def _section_quant(matrix: Dict[str, Any]) -> str:
    """量化横向对比。"""
    lines = ["## 6. 量化等级横向对比\n"]
    rows = matrix.get("groups") or []
    if not rows:
        lines.append("未提供多量化配置，跳过本节。\n")
        lines.append(
            "> 使用方法：分别用 FP16 / INT8 / AWQ 部署同一模型，各跑一次压测，"
            "把结果 JSON 一起喂给 `compare()` 即可生成本节。\n"
        )
        return "\n".join(lines)

    base = matrix.get("baseline") or {}
    lines.append(f"对比基准：**{base.get('label', 'N/A')}**（{base.get('quantization', '?')}）\n")
    lines.append(
        _table(
            [
                "配置",
                "量化",
                "峰值吞吐",
                "吞吐 vs 基准",
                "最优 TTFT",
                "TTFT 变化",
                "TPOT 变化",
                "显存峰值",
                "显存 vs 基准",
                "每 GiB 吞吐",
            ],
            [
                [
                    r["label"],
                    r["quantization"],
                    tps(r["best_throughput_tps"]),
                    ratio(r["throughput_vs_baseline"]),
                    ms(r["lowest_ttft_s"]),
                    signed_pct(r["ttft_delta_pct_vs_baseline"]),
                    signed_pct(r["tpot_delta_pct_vs_baseline"]),
                    mib(r["vram_peak_mib"]),
                    ratio(r["vram_ratio_vs_baseline"]),
                    f"{r['throughput_per_gib']:.1f}" if r["throughput_per_gib"] else "N/A",
                ]
                for r in rows
            ],
        )
    )

    common = matrix.get("common_concurrency") or []
    if common:
        lines.append("\n### 6.1 共同并发档位逐点对比\n")
        for level in common:
            entries = (matrix.get("by_concurrency") or {}).get(str(level)) or []
            lines.append(f"**并发 {level}**\n")
            lines.append(
                _table(
                    ["配置", "吞吐", "TTFT", "TPOT", "P99", "失败率"],
                    [
                        [
                            e["label"],
                            tps(e["throughput_output"]),
                            ms(e["ttft_mean"]),
                            ms(e["tpot_mean"]),
                            ms(e["latency_p99"]),
                            pct(e["failure_rate"]),
                        ]
                        for e in entries
                    ],
                )
            )
            lines.append("")

    lines.append(f"\n**量化结论**：{matrix.get('conclusion', '')}\n")
    return "\n".join(lines)


def _section_conclusion(results: List[Dict[str, Any]], matrix: Dict[str, Any]) -> str:
    """结论与选型建议。"""
    lines = ["## 7. 结论与建议\n"]
    lines.append("### 7.1 关键结论\n")
    for line in _verdict_lines(results):
        lines.append(line)

    lines.append("\n### 7.2 选型建议\n")
    lines.append(_advice(results, matrix))
    lines.append(
        "\n### 7.3 复现方式\n"
        "```bash\n"
        "# 单引擎压测\n"
        "python scripts/run_bench.py --engine vllm\n\n"
        "# 多引擎对比 + 出报告\n"
        "python scripts/run_bench.py --engine vllm,ollama --report\n"
        "```\n"
    )
    return "\n".join(lines)


def _advice(results: List[Dict[str, Any]], matrix: Dict[str, Any]) -> str:
    """根据数据给出场景化建议。"""
    tips: List[str] = []
    scored = [
        (r, (r.get("summary") or {}).get("best_throughput")) for r in results
    ]
    scored = [(r, tp) for r, tp in scored if tp]
    if not scored:
        return "- 数据不足，建议先确认引擎可达后重跑。"

    winner, _ = max(scored, key=lambda x: x[1])
    winner_c = (winner.get("summary") or {}).get("best_throughput_concurrency")

    # 延迟敏感场景
    ttfts = [
        (r, (r.get("summary") or {}).get("lowest_ttft")) for r in results
    ]
    ttfts = [(r, t) for r, t in ttfts if t]
    if ttfts:
        fastest, t = min(ttfts, key=lambda x: x[1])
        tips.append(
            f"- **交互式场景（对首字速度敏感）**：优先 {fastest.get('engine')}"
            f"（{fastest.get('quantization')}），最低 TTFT {ms(t)}。"
            f"问答、代码补全类应用建议把并发控制在 "
            f"{fastest.get('summary', {}).get('lowest_ttft_concurrency', 1)} 附近，"
            "超过后 TTFT 会随排队显著上升。"
        )

    # 吞吐优先场景
    tips.append(
        f"- **离线批处理场景（吞吐优先）**：把并发开到 "
        f"**{winner_c}**，此时 {winner.get('engine')} 吞吐达到峰值。"
        "再往上加压不会提升吞吐，只会拉长单请求时间。"
    )

    # 量化建议
    rows = matrix.get("groups") or []
    best_eff = [r for r in rows if r.get("throughput_per_gib")]
    if best_eff:
        eff = max(best_eff, key=lambda r: r["throughput_per_gib"])
        tips.append(
            f"- **显存受限场景**：{eff['label']} 的「每 GiB 显存吞吐」最高"
            f"（{eff['throughput_per_gib']:.1f} tokens/s/GiB），"
            "在单卡装不下 FP16 模型时优先考虑该量化方案。"
        )
    else:
        tips.append(
            "- **显存受限场景**：建议补测 AWQ/INT8 配置以获得「每 GiB 显存吞吐」"
            "对比，量化通常能把显存压到 FP16 的 25%~50%。"
        )

    # 稳定性
    unstable = [
        f"{r.get('engine')}@{c}"
        for r in results
        for c in ((r.get("summary") or {}).get("unstable_stages") or [])
    ]
    if unstable:
        tips.append(
            f"- **稳定性风险**：{', '.join(unstable)} 档位失败率已超阈值，"
            "上线前需排查是否显存不足或请求队列溢出。"
        )

    return "\n".join(tips)


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def build_markdown(
    results: List[Dict[str, Any]],
    quant_matrix: Optional[Dict[str, Any]] = None,
    title: str = "大模型推理性能压测报告",
) -> str:
    """生成完整 Markdown 报告。

    Args:
        results: 各引擎的压测结果字典（``metrics.store.build_result`` 产物）。
        quant_matrix: ``monitor.quant.compare`` 的产物；None 时第 6 节显示占位说明。
        title: 报告标题。

    Returns:
        Markdown 文本。
    """
    quant_matrix = quant_matrix or {"groups": [], "conclusion": ""}
    parts = [
        f"# {title}\n",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　|　"
        f"被测引擎：{', '.join(r.get('engine', '?') for r in results)}\n",
        "---\n",
        _section_overview(results),
        "---\n",
        _section_environment(results),
        "---\n",
        _section_throughput(results),
        "---\n",
        _section_latency(results),
        "---\n",
        _section_vram(results),
        "---\n",
        _section_quant(quant_matrix),
        "---\n",
        _section_conclusion(results, quant_matrix),
    ]
    return "\n".join(parts)


def save_markdown(content: str, path: Path) -> Path:
    """写出 Markdown 报告。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Markdown 报告已保存：%s", path)
    return path
