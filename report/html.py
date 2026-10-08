"""HTML 报告生成（自包含，无外链依赖）。

为什么要额外出一份 HTML：
- 面试/客户演示时 Markdown 表格不好看，图表更直观；
- 交付物要能**离线打开**，所以不引用任何 CDN，图表用内联 SVG 手写。

图表类型：
- 吞吐 - 并发折线图
- TTFT / P99 延迟 - 并发折线图
- 显存占用曲线（时间轴多卡）
- 量化对比柱状图

SVG 全部由 :mod:`report.svgchart` 生成，本模块只负责拼装。
"""

from __future__ import annotations

import html
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from metrics.stats import StageMetrics
from monitor.quant import _stage_from_dict
from report.fmt import mib, ms, num, pct, ratio, signed_pct, tps
from report.svgchart import bar_chart, multi_line_chart

logger = logging.getLogger(__name__)

# 配色：蓝 / 橙 / 绿 / 紫 / 红，区分度足够且色盲友好
PALETTE = ["#2563eb", "#ea580c", "#059669", "#7c3aed", "#dc2626"]


def _stages_of(result: Dict[str, Any]) -> List[StageMetrics]:
    return [_stage_from_dict(s) for s in result.get("stages", [])]


def _esc(text: Any) -> str:
    return html.escape(str(text))


# ----------------------------------------------------------------------
# 图表数据准备
# ----------------------------------------------------------------------
def _throughput_series(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """吞吐 - 并发折线图数据。"""
    series = []
    for i, r in enumerate(results):
        stages = _stages_of(r)
        points = [
            (s.concurrency, s.throughput_output)
            for s in stages
            if s.throughput_output is not None
        ]
        if points:
            series.append(
                {
                    "name": f"{r.get('engine')} ({r.get('quantization')})",
                    "color": PALETTE[i % len(PALETTE)],
                    "points": points,
                }
            )
    return {
        "title": "吞吐 vs 并发",
        "x_label": "并发数",
        "y_label": "输出吞吐 (tokens/s)",
    } | {"series": series}


def _latency_series(results: List[Dict[str, Any]], metric: str, label: str, title: str) -> Dict[str, Any]:
    """延迟类折线图：TTFT 均值 / 端到端 P99。"""
    series = []
    for i, r in enumerate(results):
        stages = _stages_of(r)
        points = []
        for s in stages:
            value = getattr(s, metric, None)
            if value is not None:
                points.append((s.concurrency, value * 1000.0))  # 转 ms
        if points:
            series.append(
                {
                    "name": f"{r.get('engine')} ({r.get('quantization')})",
                    "color": PALETTE[i % len(PALETTE)],
                    "points": points,
                }
            )
    return {
        "title": title,
        "x_label": "并发数",
        "y_label": label,
    } | {"series": series}


def _vram_series(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """显存占用曲线：时间轴，每张卡一条线。"""
    series = []
    color_i = 0
    for r in results:
        curve = r.get("vram_curve") or {}
        if not curve.get("available"):
            continue
        for g in curve.get("gpus", []):
            pts = [
                (row["t"], row["memory_used_mib"])
                for row in g.get("series", [])
                if row.get("memory_used_mib") is not None
            ]
            if not pts:
                continue
            series.append(
                {
                    "name": f"{r.get('engine')} · GPU{g.get('index')}",
                    "color": PALETTE[color_i % len(PALETTE)],
                    "points": pts,
                }
            )
            color_i += 1
    if not series:
        return None
    return {
        "title": "显存占用曲线",
        "x_label": "压测时间 (s)",
        "y_label": "显存占用 (MiB)",
        "series": series,
    }


def _quant_bar(matrix: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """量化对比柱状图。"""
    rows = [r for r in (matrix.get("groups") or []) if r.get("best_throughput_tps")]
    if len(rows) < 2:
        return None
    return {
        "title": "各量化配置峰值吞吐",
        "y_label": "tokens/s",
        "items": [
            {
                "label": f"{r['label']}\n({r['quantization']})",
                "value": r["best_throughput_tps"],
                "color": "#059669" if r.get("is_baseline") else PALETTE[i % len(PALETTE)],
            }
            for i, r in enumerate(rows)
        ],
    }


# ----------------------------------------------------------------------
# HTML 拼装
# ----------------------------------------------------------------------
CSS = """
:root {
  --bg: #ffffff;
  --fg: #1f2430;
  --muted: #6b7280;
  --line: #e5e7eb;
  --card: #f9fafb;
  --accent: #2563eb;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 0 64px;
  background: var(--bg); color: var(--fg);
  font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 15px; line-height: 1.7;
}
header {
  padding: 40px 48px 28px;
  border-bottom: 1px solid var(--line);
  background: linear-gradient(180deg,#f8fafc 0%,#ffffff 100%);
}
header h1 { margin: 0 0 8px; font-size: 27px; letter-spacing: -.3px; }
header .meta { color: var(--muted); font-size: 13px; }
main { max-width: 1180px; margin: 0 auto; padding: 0 48px; }
h2 {
  margin: 44px 0 14px; font-size: 20px;
  padding-bottom: 8px; border-bottom: 2px solid var(--line);
}
h3 { margin: 26px 0 10px; font-size: 16px; color: #374151; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 20px; font-size: 13.5px; }
th, td { border: 1px solid var(--line); padding: 8px 11px; text-align: right; }
th { background: var(--card); font-weight: 600; text-align: right; }
th:first-child, td:first-child { text-align: left; }
tr:nth-child(even) td { background: #fcfcfd; }
code {
  background: #f3f4f6; padding: 1px 5px; border-radius: 4px;
  font-family: "SFMono-Regular", Consolas, monospace; font-size: .9em;
}
.card {
  background: var(--card); border: 1px solid var(--line);
  border-radius: 10px; padding: 18px 22px; margin: 14px 0;
}
.chart { margin: 18px 0 26px; overflow-x: auto; }
.conclusion { border-left: 4px solid var(--accent); padding-left: 16px; }
.conclusion ul { margin: 8px 0; padding-left: 20px; }
.badge {
  display: inline-block; padding: 2px 9px; border-radius: 999px;
  font-size: 12px; font-weight: 600;
}
.badge.ok { background: #dcfce7; color: #166534; }
.badge.bad { background: #fee2e2; color: #991b1b; }
.badge.warn { background: #fef3c7; color: #92400e; }
.muted { color: var(--muted); }
footer {
  max-width: 1180px; margin: 48px auto 0; padding: 20px 48px 0;
  border-top: 1px solid var(--line); color: var(--muted); font-size: 12.5px;
}
"""


def _table_html(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return '<p class="muted">暂无数据</p>'
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _badge(text: str, kind: str) -> str:
    return f'<span class="badge {kind}">{_esc(text)}</span>'


def _health_badge(health: Dict[str, Any]) -> str:
    if not health:
        return _badge("未检测", "warn")
    if health.get("ok"):
        if health.get("model_loaded"):
            return _badge("正常", "ok")
        return _badge("模型未加载", "warn")
    return _badge("不可达", "bad")


def _overview_html(results: List[Dict[str, Any]]) -> str:
    rows = []
    for r in results:
        s = r.get("summary") or {}
        stages = _stages_of(r)
        peak_c = s.get("best_throughput_concurrency")
        p99 = tpot = None
        for st in stages:
            if st.concurrency == peak_c:
                p99, tpot = st.latency_p99, st.tpot_mean
                break
        unstable = s.get("unstable_stages") or []
        status = (
            _badge(f"{len(unstable)} 档不稳定", "warn")
            if unstable
            else _badge("全部稳定", "ok")
        )
        rows.append(
            [
                f"<b>{_esc(r.get('engine'))}</b><br><code>{_esc(r.get('model'))}</code>",
                _esc(r.get("quantization", "unknown")),
                f"<b>{tps(s.get('best_throughput'))}</b>",
                str(peak_c) if peak_c is not None else "—",
                ms(s.get("lowest_ttft")),
                ms(tpot),
                ms(p99),
                mib(r.get("vram_peak_mib")),
                status,
                _health_badge(r.get("health", {})),
            ]
        )
    return _table_html(
        [
            "引擎 / 模型",
            "量化",
            "峰值吞吐",
            "对应并发",
            "最优 TTFT",
            "TPOT",
            "P99",
            "显存峰值",
            "稳定性",
            "健康",
        ],
        rows,
    )


def _verdict_html(results: List[Dict[str, Any]]) -> str:
    scored = [(r, (r.get("summary") or {}).get("best_throughput")) for r in results]
    scored = [(r, tp) for r, tp in scored if tp]
    if not scored:
        return '<p class="muted">本次压测未取得有效吞吐数据。</p>'

    items: List[str] = []
    winner, best = max(scored, key=lambda x: x[1])
    items.append(
        f"吞吐最优：<b>{_esc(winner.get('engine'))}</b>"
        f"（{_esc(winner.get('quantization'))}）峰值 <b>{tps(best)}</b>，"
        f"出现在并发 {winner.get('summary', {}).get('best_throughput_concurrency')}。"
    )
    for r, tp in scored:
        if r is winner:
            continue
        gap = (best - tp) / tp * 100 if tp else None
        items.append(
            f"{_esc(r.get('engine'))} 峰值 {tps(tp)}"
            + (f"，比最优低 {gap:.1f}%。" if gap is not None else "。")
        )
    for r in results:
        unstable = (r.get("summary") or {}).get("unstable_stages") or []
        if unstable:
            items.append(
                f"⚠️ {_esc(r.get('engine'))} 在并发 {unstable} 档位失败率超阈值，数据仅供参考。"
            )
    return '<div class="conclusion"><ul>' + "".join(f"<li>{i}</li>" for i in items) + "</ul></div>"


def _env_html(results: List[Dict[str, Any]]) -> str:
    first = results[0] if results else {}
    env = first.get("environment", {})
    cfg = first.get("config", {})
    env_rows = [
        ["操作系统", _esc(env.get("platform", "—"))],
        ["Python", _esc(env.get("python", "—"))],
        ["CPU", _esc(env.get("processor", "—"))],
        ["主机名", _esc(env.get("hostname", "—"))],
        ["生成时间", _esc(first.get("generated_at", "—"))],
    ]
    cfg_rows = [
        ["并发梯度", _esc(", ".join(str(c) for c in cfg.get("concurrency_levels", [])))],
        ["每请求最大生成", f"{cfg.get('max_tokens', '—')} tokens"],
        ["预热请求数", str(cfg.get("warmup_requests", "—"))],
        ["温度", str(cfg.get("temperature", "—"))],
        ["流式", "是" if cfg.get("stream") else "否"],
        ["单请求超时", f"{cfg.get('timeout', '—')} s"],
        ["提示词", f"<code>{_esc(cfg.get('prompt', '—'))}</code>"],
    ]
    return (
        "<div class='card'><h3>运行环境</h3>"
        + _table_html(["项", "值"], env_rows)
        + "</div><div class='card'><h3>压测参数</h3>"
        + _table_html(["项", "值"], cfg_rows)
        + "</div>"
    )


def _latency_tables(results: List[Dict[str, Any]]) -> str:
    out = []
    for r in results:
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
        out.append(
            f"<h3>{_esc(r.get('engine'))} / <code>{_esc(r.get('model'))}</code>"
            f"（{_esc(r.get('quantization'))}）</h3>"
            + _table_html(
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
                    "最大",
                ],
                rows,
            )
        )
    return "".join(out)


def _vram_table(results: List[Dict[str, Any]]) -> str:
    rows = []
    for r in results:
        curve = r.get("vram_curve") or {}
        if not curve.get("available"):
            continue
        for g in curve.get("gpus", []):
            peak_pct = g.get("peak_memory_used_percent")
            rows.append(
                [
                    _esc(r.get("engine")),
                    str(g.get("index")),
                    _esc(g.get("name", "—")),
                    mib(g.get("memory_total_mib")),
                    mib(g.get("peak_memory_used_mib")),
                    f"{peak_pct:.1f}%" if peak_pct is not None else "—",
                    f"{g.get('mean_util_percent'):.1f}%"
                    if g.get("mean_util_percent") is not None
                    else "—",
                    str(g.get("sample_count", 0)),
                ]
            )
    if not rows:
        return (
            '<p class="muted">本次压测未采集到 GPU 显存数据'
            "（非 NVIDIA 环境 / 容器内无 nvidia-smi / 监控关闭）。</p>"
        )
    return _table_html(
        ["引擎", "卡号", "型号", "总显存", "峰值占用", "峰值占比", "平均利用率", "采样点"],
        rows,
    )


def _stage_vram_table(results: List[Dict[str, Any]]) -> str:
    levels: List[int] = []
    for r in results:
        levels.extend(s.concurrency for s in _stages_of(r))
    levels = sorted(set(levels))
    if not levels:
        return ""
    rows = []
    for level in levels:
        row = [str(level)]
        for r in results:
            peaks = (r.get("vram_curve") or {}).get("stage_peak_mib") or {}
            row.append(mib(peaks.get(str(level))))
        rows.append(row)
    return _table_html(["并发"] + [f"{_esc(r.get('engine'))}" for r in results], rows)


def _quant_html(matrix: Dict[str, Any]) -> str:
    rows_data = matrix.get("groups") or []
    if not rows_data:
        return (
            '<p class="muted">未提供多量化配置，本节跳过。</p>'
            "<p class='muted'>用法：分别用 FP16 / INT8 / AWQ 部署同一模型，"
            "各跑一次压测，把结果一起喂给 <code>compare()</code>。</p>"
        )
    base = matrix.get("baseline") or {}
    rows = []
    for r in rows_data:
        tag = " <b>(基准)</b>" if r.get("is_baseline") else ""
        rows.append(
            [
                f"{_esc(r['label'])}{tag}",
                _esc(r["quantization"]),
                tps(r["best_throughput_tps"]),
                ratio(r["throughput_vs_baseline"]),
                ms(r["lowest_ttft_s"]),
                signed_pct(r["ttft_delta_pct_vs_baseline"]),
                mib(r["vram_peak_mib"]),
                ratio(r["vram_ratio_vs_baseline"]),
                f"{r['throughput_per_gib']:.1f}" if r["throughput_per_gib"] else "—",
            ]
        )
    out = [
        f'<p>对比基准：<b>{_esc(base.get("label", "—"))}</b>'
        f'（{_esc(base.get("quantization", "?"))}）</p>',
        _table_html(
            [
                "配置",
                "量化",
                "峰值吞吐",
                "吞吐 vs 基准",
                "最优 TTFT",
                "TTFT 变化",
                "显存峰值",
                "显存 vs 基准",
                "每 GiB 吞吐",
            ],
            rows,
        ),
    ]

    common = matrix.get("common_concurrency") or []
    for level in common:
        entries = (matrix.get("by_concurrency") or {}).get(str(level)) or []
        out.append(f"<h3>并发 {level}</h3>")
        out.append(
            _table_html(
                ["配置", "吞吐", "TTFT", "TPOT", "P99", "失败率"],
                [
                    [
                        _esc(e["label"]),
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

    out.append(f'<div class="conclusion"><p>{_esc(matrix.get("conclusion", ""))}</p></div>')
    return "".join(out)


def _advice_html(results: List[Dict[str, Any]], matrix: Dict[str, Any]) -> str:
    """场景化选型建议。"""
    scored = [(r, (r.get("summary") or {}).get("best_throughput")) for r in results]
    scored = [(r, tp) for r, tp in scored if tp]
    if not scored:
        return '<p class="muted">数据不足，建议确认引擎可达后重跑。</p>'

    items: List[str] = []
    winner, _ = max(scored, key=lambda x: x[1])
    winner_c = (winner.get("summary") or {}).get("best_throughput_concurrency")

    ttfts = [(r, (r.get("summary") or {}).get("lowest_ttft")) for r in results]
    ttfts = [(r, t) for r, t in ttfts if t]
    if ttfts:
        fastest, ttft = min(ttfts, key=lambda x: x[1])
        items.append(
            f"<b>交互式场景（首字优先）</b>：选 <b>{_esc(fastest.get('engine'))}</b>"
            f"（{_esc(fastest.get('quantization'))}），最低 TTFT {ms(ttft)}。"
            "问答/补全类应用建议并发不超过该档位附近，再往上 TTFT 会因排队快速上升。"
        )
    items.append(
        f"<b>离线批处理（吞吐优先）</b>：并发开到 <b>{winner_c}</b> 时"
        f"{_esc(winner.get('engine'))} 吞吐见顶；继续加压只增延迟不增吞吐。"
    )

    eff = [r for r in (matrix.get("groups") or []) if r.get("throughput_per_gib")]
    if eff:
        best_eff = max(eff, key=lambda r: r["throughput_per_gib"])
        items.append(
            f"<b>显存受限场景</b>：{_esc(best_eff['label'])} 的每 GiB 显存吞吐最高"
            f"（{best_eff['throughput_per_gib']:.1f} tokens/s/GiB），优先考虑。"
        )
    else:
        items.append(
            "<b>显存受限场景</b>：建议补测 AWQ/INT8，量化通常可把显存压到 FP16 的 25%~50%。"
        )

    unstable = [
        f"{r.get('engine')}@{c}"
        for r in results
        for c in ((r.get("summary") or {}).get("unstable_stages") or [])
    ]
    if unstable:
        items.append(
            f"<b>稳定性风险</b>：{_esc(', '.join(unstable))} 档位失败率超阈值，"
            "上线前需排查显存不足或队列溢出。"
        )

    return '<div class="conclusion"><ul>' + "".join(f"<li>{i}</li>" for i in items) + "</ul></div>"


def _chart_html(spec: Optional[Dict[str, Any]]) -> str:
    if spec is None:
        return ""
    series = spec.get("series")
    if series is not None:
        svg = multi_line_chart(
            series,
            x_label=spec.get("x_label", ""),
            y_label=spec.get("y_label", ""),
            title=spec.get("title", ""),
        )
    else:
        svg = bar_chart(spec.get("items", []), y_label=spec.get("y_label", ""), title=spec.get("title", ""))
    return f'<div class="chart">{svg}</div>'


def build_html(
    results: List[Dict[str, Any]],
    quant_matrix: Optional[Dict[str, Any]] = None,
    title: str = "大模型推理性能压测报告",
) -> str:
    """生成完整自包含 HTML 报告。"""
    quant_matrix = quant_matrix or {"groups": [], "conclusion": ""}
    engines = ", ".join(r.get("engine", "?") for r in results)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{_esc(title)}</title>",
        f"<style>{CSS}</style>",
        "</head><body>",
        "<header>",
        f"<h1>{_esc(title)}</h1>",
        f'<div class="meta">被测引擎：{_esc(engines)}　·　生成时间：{generated}</div>',
        "</header>",
        "<main>",
        # 1 概览
        "<h2>1. 概览</h2>",
        _overview_html(results),
        _verdict_html(results),
        # 2 环境
        "<h2>2. 环境与配置</h2>",
        _env_html(results),
        # 3 吞吐
        "<h2>3. 吞吐 - 并发曲线</h2>",
        "<p>吞吐随并发上升直至饱和，<b>饱和点即容量上限</b>，超过后继续加压只推高延迟。</p>",
        _chart_html(_throughput_series(results)),
        # 4 延迟
        "<h2>4. 延迟指标</h2>",
        "<p><b>TTFT</b> 决定感知响应速度，<b>TPOT</b> 决定出字流畅度，<b>P99</b> 决定长尾体验。</p>",
        _chart_html(_latency_series(results, "ttft_mean", "TTFT (ms)", "首包延迟 TTFT vs 并发")),
        _chart_html(_latency_series(results, "latency_p99", "端到端 P99 (ms)", "端到端 P99 延迟 vs 并发")),
        _latency_tables(results),
        # 5 显存
        "<h2>5. 显存占用曲线</h2>",
        "<p>显存决定「一台机器能同时服务多少请求」，是私有化交付最硬的成本约束。</p>",
        _chart_html(_vram_series(results)),
        _vram_table(results),
        "<h3>5.1 显存随并发变化</h3>",
        _stage_vram_table(results),
        # 6 量化
        "<h2>6. 量化等级横向对比</h2>",
        _chart_html(_quant_bar(quant_matrix)),
        _quant_html(quant_matrix),
        # 7 结论
        "<h2>7. 结论与建议</h2>",
        _advice_html(results, quant_matrix),
        "</main>",
        "<footer>由 llm-inference-bench 自动生成 · 自包含页面，可离线打开 · "
        "所有图表为内联 SVG，无外部依赖</footer>",
        "</body></html>",
    ]
    return "\n".join(parts)


def save_html(content: str, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("HTML 报告已保存：%s", path)
    return path
