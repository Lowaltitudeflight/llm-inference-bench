"""压测结果的持久化。

每次压测产出两类文件，放在 ``data/results/<run_id>/`` 下：

- ``<engine>.json`` — 结构化结果（元信息 + 各档位指标 + 显存曲线）
- ``<engine>.csv``  — 档位指标的扁平表，便于丢进 Excel / pandas

用 ``run_id`` 目录隔离不同批次，避免多次压测互相覆盖。
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import BenchConfig, EngineTarget
from metrics.stats import StageMetrics, summarize_run

logger = logging.getLogger(__name__)

# CSV 扁平行，顺序即列顺序
CSV_COLUMNS = [
    "concurrency",
    "total_requests",
    "success_requests",
    "failed_requests",
    "failure_rate",
    "unstable",
    "latency_mean",
    "latency_p50",
    "latency_p95",
    "latency_p99",
    "latency_min",
    "latency_max",
    "latency_stdev",
    "ttft_mean",
    "ttft_p50",
    "ttft_p95",
    "ttft_p99",
    "tpot_mean",
    "tpot_p50",
    "tpot_p95",
    "tpot_p99",
    "throughput_output",
    "throughput_total",
    "per_request_throughput_mean",
    "output_tokens_total",
    "input_tokens_total",
    "wall_time",
    "request_rate",
]


def new_run_id(prefix: str = "run") -> str:
    """生成按时间排序的 run id，如 ``run-20260518-143002``。"""
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def environment_info() -> Dict[str, Any]:
    """记录环境信息。报告里要能说明「这个数是哪台机器跑的」。"""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "hostname": platform.node(),
    }


def _stage_row(stage: StageMetrics) -> Dict[str, Any]:
    """把档位指标拍平成一行。"""
    return {
        "concurrency": stage.concurrency,
        "total_requests": stage.total_requests,
        "success_requests": stage.success_requests,
        "failed_requests": stage.failed_requests,
        "failure_rate": stage.failure_rate,
        "unstable": stage.unstable,
        "latency_mean": stage.latency_mean,
        "latency_p50": stage.latency_p50,
        "latency_p95": stage.latency_p95,
        "latency_p99": stage.latency_p99,
        "latency_min": stage.latency_min,
        "latency_max": stage.latency_max,
        "latency_stdev": stage.latency_stdev,
        "ttft_mean": stage.ttft_mean,
        "ttft_p50": stage.ttft_p50,
        "ttft_p95": stage.ttft_p95,
        "ttft_p99": stage.ttft_p99,
        "tpot_mean": stage.tpot_mean,
        "tpot_p50": stage.tpot_p50,
        "tpot_p95": stage.tpot_p95,
        "tpot_p99": stage.tpot_p99,
        "throughput_output": stage.throughput_output,
        "throughput_total": stage.throughput_total,
        "per_request_throughput_mean": stage.per_request_throughput_mean,
        "output_tokens_total": stage.output_tokens_total,
        "input_tokens_total": stage.input_tokens_total,
        "wall_time": stage.wall_time,
        "request_rate": stage.request_rate,
    }


def build_result(
    target: EngineTarget,
    bench: BenchConfig,
    stages: List[StageMetrics],
    vram_curve: Optional[Dict[str, Any]] = None,
    health: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    duration: Optional[float] = None,
) -> Dict[str, Any]:
    """组装一次完整压测的结果字典（也是落盘 JSON 的结构）。"""
    return {
        "run_id": run_id or new_run_id(),
        "engine": target.name,
        "model": target.model,
        "base_url": target.normalized_base_url(),
        # 落盘文件名（引擎__模型__量化），多配置对比时避免互相覆盖
        "result_stem": result_stem(target),
        "quantization": target.quantization,
        "note": target.note,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "duration_seconds": round(duration, 2) if duration is not None else None,
        "config": {
            "concurrency_levels": bench.concurrency_levels,
            "max_tokens": bench.max_tokens,
            "warmup_requests": bench.warmup_requests,
            "temperature": bench.temperature,
            "stream": bench.stream,
            "timeout": bench.timeout,
            "prompt": bench.prompt,
        },
        "environment": environment_info(),
        "health": health or {},
        "summary": summarize_run(stages),
        "stages": [s.to_dict() for s in stages],
        "vram_curve": vram_curve or {"available": False, "gpus": []},
        "vram_peak_mib": (
            (vram_curve or {}).get("peak_memory_used_mib") if vram_curve else None
        ),
    }


def _safe_slug(text: str) -> str:
    """把任意文本变成安全的文件名片段。

    多配置对比时同一引擎会跑多次（不同量化/模型），若文件名只用
    引擎名，后一次会覆盖前一次的结果。因此文件名要带上量化与模型。
    """
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in "-_.@:":
            keep.append(ch)
        else:
            keep.append("_")
    slug = "".join(keep).strip("_")
    return slug or "unknown"


def result_stem(target: EngineTarget) -> str:
    """结果文件名（不含扩展名）。

    形如 ``vllm__Qwen2.5-72B-Instruct__fp16``；量化等级为 unknown
    时省略该段，保持「单配置压测」的文件名简洁。
    """
    parts = [target.name]
    model = (target.model or "").strip()
    if model:
        parts.append(_safe_slug(model))
    quant = (target.quantization or "").strip().lower()
    if quant and quant != "unknown":
        parts.append(_safe_slug(quant))
    return "__".join(parts)


def save_result(result: Dict[str, Any], output_dir: Path, run_id: str) -> Dict[str, Path]:
    """把结果写到 ``output_dir/run_id/``，返回产出文件路径。

    文件名由引擎 + 模型 + 量化组成，避免同批次多配置互相覆盖。

    Returns:
        {"json": Path, "csv": Path}
    """
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 优先用结果里已有的 stem（run_one 会显式写入），
    # 否则从字段拼一个出来，保证旧格式的结果也能落盘
    stem = result.get("result_stem") or _slug_from_result(result)

    json_path = run_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = run_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for stage in result.get("stages", []):
            # stages 是嵌套结构，先还原成 StageMetrics 再拍平
            writer.writerow(_stage_row(_revive_stage(stage)))

    logger.info("结果已保存：%s", run_dir)
    return {"json": json_path, "csv": csv_path}


def _slug_from_result(result: Dict[str, Any]) -> str:
    """从结果字典拼出文件名 stem。"""
    parts = [str(result.get("engine", "engine"))]
    model = str(result.get("model") or "").strip()
    if model:
        parts.append(_safe_slug(model))
    quant = str(result.get("quantization") or "").strip().lower()
    if quant and quant != "unknown":
        parts.append(_safe_slug(quant))
    return "__".join(parts)


def _revive_stage(raw: Dict[str, Any]) -> StageMetrics:
    """把嵌套 dict 还原成 StageMetrics，供拍平与报告共用。"""
    from monitor.quant import _stage_from_dict

    return _stage_from_dict(raw)


def load_result(path: Path) -> Dict[str, Any]:
    """读取一个结果 JSON。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_results(output_dir: Path) -> List[Path]:
    """列出结果目录下所有 run 的 JSON，按修改时间倒序。"""
    root = Path(output_dir)
    if not root.exists():
        return []
    files = sorted(root.glob("*/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def load_latest(output_dir: Path) -> Optional[Dict[str, Any]]:
    """读取最近一次压测结果。"""
    files = list_results(output_dir)
    return load_result(files[0]) if files else None


def stage_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从结果字典取扁平行，供 CSV 之外的报表复用。"""
    return [_stage_row(_revive_stage(s)) for s in result.get("stages", [])]
