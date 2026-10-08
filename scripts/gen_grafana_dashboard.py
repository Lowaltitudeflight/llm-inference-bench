#!/usr/bin/env python
"""生成 Grafana 面板 JSON。

单独成脚本而不是手写 JSON：面板有 13 个、每个几十行，手改容易漏逗号。
需要调整面板时改这里的 Python，重新执行即可。

    python scripts/gen_grafana_dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DS = {"type": "prometheus", "uid": "prometheus"}
OUT = Path(__file__).resolve().parent.parent / "deploy" / "grafana" / "dashboards" / "llm-inference-bench.json"


def ts(
    title: str,
    expr: str,
    unit: str = "short",
    legend: str = "{{engine}} ({{quantization}})",
    x: int = 0,
    y: int = 0,
    w: int = 12,
    h: int = 8,
    desc: str = "",
) -> dict:
    """时间序列面板。"""
    return {
        "type": "timeseries",
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [
            {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": "A", "range": True}
        ],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "custom": {
                    "drawStyle": "line",
                    "lineWidth": 2,
                    "fillOpacity": 10,
                    "showPoints": "auto",
                    "spanNulls": True,
                    "lineInterpolation": "smooth",
                },
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 0.05},
                        {"color": "red", "value": 0.2},
                    ],
                },
            },
            "overrides": [],
        },
        "options": {
            "legend": {
                "displayMode": "table",
                "placement": "right",
                "calcs": ["lastNotNull", "max", "mean"],
            },
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def stat(
    title: str,
    expr: str,
    unit: str = "short",
    x: int = 0,
    y: int = 0,
    w: int = 6,
    h: int = 4,
    color: str = "blue",
    decimals: int = 1,
) -> dict:
    """单值数字面板。"""
    return {
        "type": "stat",
        "title": title,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [
            {
                "datasource": DS,
                "expr": expr,
                "refId": "A",
                "instant": True,
                "range": False,
                "legendFormat": "__auto",
            }
        ],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "decimals": decimals,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": color, "value": None}],
                },
            },
            "overrides": [],
        },
        "options": {
            "graphMode": "area",
            "textMode": "auto",
            "colorMode": "value",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
        },
    }


def build() -> dict:
    sel = 'engine=~"$engine"'

    panels = [
        # ---------- 第一行：核心数字 ----------
        stat("当前并发", f"max(llmbench_concurrency{{{sel}}})", x=0, y=0, w=4, h=4, decimals=0),
        stat(
            "峰值吞吐",
            f"max(llmbench_throughput_tokens_per_second{{{sel}}})",
            x=4, y=0, w=5, h=4, color="green",
        ),
        stat(
            "失败率",
            f"max(llmbench_stage_failure_rate{{{sel}}})",
            unit="percentunit", x=9, y=0, w=5, h=4, color="red",
        ),
        stat(
            "GPU 显存已用",
            "max(llmbench_gpu_memory_used_mib)",
            unit="decmbytes", x=14, y=0, w=5, h=4, color="orange", decimals=0,
        ),
        stat(
            "GPU 利用率",
            "max(llmbench_gpu_utilization_percent)",
            unit="percent", x=19, y=0, w=5, h=4, color="purple",
        ),
        # ---------- 第二行：吞吐与首包 ----------
        ts(
            "输出吞吐 vs 时间",
            f"llmbench_throughput_tokens_per_second{{{sel}}}",
            x=0, y=4, w=12, h=9,
            legend="{{engine}} · 并发{{concurrency}}",
            desc="每个档位完成后打点。台阶跳变对应并发梯度上调。",
        ),
        ts(
            "首包延迟 TTFT P99",
            f"llmbench_ttft_p99_seconds{{{sel}}}",
            unit="s", x=12, y=4, w=12, h=9,
            legend="{{engine}} · 并发{{concurrency}}",
            desc="TTFT 是交互式体验的决定性指标；随并发上升的斜率反映排队程度。",
        ),
        # ---------- 第三行：延迟分位 ----------
        ts(
            "端到端延迟 P50",
            f"histogram_quantile(0.50, sum by (le, engine) "
            f"(rate(llmbench_request_latency_seconds_bucket{{{sel}}}[1m])))",
            unit="s", x=0, y=13, w=8, h=9,
            legend="{{engine}} P50",
            desc="来自直方图统计，反映实际请求分布的中位数。",
        ),
        ts(
            "端到端延迟 P99",
            f"histogram_quantile(0.99, sum by (le, engine) "
            f"(rate(llmbench_request_latency_seconds_bucket{{{sel}}}[1m])))",
            unit="s", x=8, y=13, w=8, h=9,
            legend="{{engine}} P99",
            desc="长尾延迟，线上体验往往由它决定。",
        ),
        ts(
            "TPOT 单 token 延迟 P50",
            f"histogram_quantile(0.50, sum by (le, engine) "
            f"(rate(llmbench_tpot_seconds_bucket{{{sel}}}[1m])))",
            unit="s", x=16, y=13, w=8, h=9,
            legend="{{engine}} TPOT P50",
            desc="解码速度，决定出字是否流畅。",
        ),
        # ---------- 第四行：速率与显存 ----------
        ts(
            "请求速率",
            f"sum by (engine, status) (rate(llmbench_requests_total{{{sel}}}[1m]))",
            unit="reqps", x=0, y=22, w=8, h=9,
            legend="{{engine}} · {{status}}",
            desc="成功/失败请求速率。失败突然抬头要立刻排查。",
        ),
        ts(
            "GPU 显存占用曲线",
            "llmbench_gpu_memory_used_mib",
            unit="decmbytes", x=8, y=22, w=8, h=9,
            legend="GPU{{gpu}} {{name}}",
            desc="显存决定一台机器能同时服务多少请求，是私有化交付最硬的约束。",
        ),
        ts(
            "GPU 利用率",
            "llmbench_gpu_utilization_percent",
            unit="percent", x=16, y=22, w=8, h=9,
            legend="GPU{{gpu}} {{name}}",
        ),
        # ---------- 第五行 ----------
        ts(
            "GPU 功耗 (W)",
            "llmbench_gpu_power_watts",
            unit="watt", x=0, y=31, w=12, h=8,
            legend="GPU{{gpu}} {{name}}",
            desc="功耗曲线能看出是否真的把卡跑满了。",
        ),
        ts(
            "档位失败率",
            f"llmbench_stage_failure_rate{{{sel}}}",
            unit="percentunit", x=12, y=31, w=12, h=8,
            legend="{{engine}} · 并发{{concurrency}}",
            desc="失败率超 20% 的档位数据已不可信，报告中会被标记为不稳定。",
        ),
    ]

    return {
        "uid": "llmbench-main",
        "title": "LLM 推理压测面板",
        "description": "llm-inference-bench 自动生成 · 吞吐 / TTFT / TPOT / 尾延迟 / 显存",
        "tags": ["llm", "benchmark", "vllm", "ollama", "gpu"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "5s",
        "editable": True,
        "time": {"from": "now-30m", "to": "now"},
        "templating": {
            "list": [
                {
                    "name": "engine",
                    "label": "引擎",
                    "type": "query",
                    "datasource": DS,
                    "query": {
                        "query": "label_values(llmbench_requests_total, engine)",
                        "refId": "A",
                    },
                    "includeAll": True,
                    "multi": True,
                    "current": {"selected": True, "text": ["All"], "value": ["$__all"]},
                    "refresh": 1,
                }
            ]
        },
        "panels": panels,
    }


def main() -> None:
    dashboard = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成 {OUT}（{len(dashboard['panels'])} 个面板，{OUT.stat().st_size} 字节）")


if __name__ == "__main__":
    main()
