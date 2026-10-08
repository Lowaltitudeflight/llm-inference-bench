"""FastAPI 服务。

把压测能力包成 HTTP 接口，让压测可以远程触发、异步执行、进度可查：

========================  ==========================================
方法 路径                  说明
========================  ==========================================
GET  ``/``                服务信息与可用端点
GET  ``/health``          服务自身健康
GET  ``/engines``         已配置的推理引擎列表
POST ``/engines/check``   探活指定引擎
POST ``/bench``           提交一次压测任务（异步，立即返回 job_id）
GET  ``/jobs``            列出全部任务
GET  ``/jobs/{job_id}``   查询任务状态与实时进度
GET  ``/jobs/{job_id}/result`` 取任务结果（含各档位指标）
GET  ``/reports``         列出已生成的报告
POST ``/compare``         对已有结果做量化横向对比
GET  ``/metrics``         Prometheus 指标
========================  ==========================================

为什么任务要异步：一次完整压测（6 档并发 × 数十秒）动辄几分钟，
同步 HTTP 请求必然超时。这里用后台线程执行，前端轮询 ``/jobs/{id}``。
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from bench.runner import LoadRunner
from config.settings import (
    API_HOST,
    API_PORT,
    BENCH,
    LOG_LEVEL,
    MONITOR,
    EngineTarget,
    default_engines,
)
from engines.client import UnifiedLLMClient
from metrics.store import build_result, new_run_id, save_result
from monitor.quant import QuantGroup, compare, from_result_dicts
from report import generate_reports
from metrics import prometheus as prom

logger = logging.getLogger(__name__)

app = FastAPI(
    title="llm-inference-bench API",
    description="大模型推理部署与压测评测平台",
    version="0.1.0",
)

# ----------------------------------------------------------------------
# 任务注册表（内存态）
# ----------------------------------------------------------------------
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


class BenchRequest(BaseModel):
    """压测请求体。所有字段可选，缺省用全局配置。"""

    engines: List[str] = Field(
        default=["vllm"], description="待测引擎名列表，取值见 GET /engines"
    )
    concurrency_levels: Optional[List[int]] = Field(
        default=None, description="并发梯度，如 [1,2,4,8,16,32]"
    )
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    warmup_requests: Optional[int] = Field(default=None, ge=0, le=32)
    timeout: Optional[float] = Field(default=None, gt=0, le=3600)
    stream: Optional[bool] = None
    prompt: Optional[str] = None
    requests_per_level: Optional[int] = Field(
        default=None, ge=1, le=10000, description="每档位请求总数，默认 = 并发数"
    )
    enable_monitor: bool = Field(default=True, description="是否采集 GPU 显存")
    generate_report: bool = Field(default=True, description="压测后是否生成报告")
    # 覆盖端点参数（单引擎时生效）
    base_url: Optional[str] = None
    model: Optional[str] = None
    quantization: Optional[str] = None


class CheckRequest(BaseModel):
    engines: List[str] = Field(default=["vllm"])


class CompareRequest(BaseModel):
    """基于已有结果做量化对比。"""

    run_id: str = Field(description="结果所在的 run 目录名")
    output_dir: Optional[str] = None


# ----------------------------------------------------------------------
# 基础端点
# ----------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "llm-inference-bench",
        "version": app.version,
        "endpoints": [
            "GET  /health",
            "GET  /engines",
            "POST /engines/check",
            "POST /bench",
            "GET  /jobs",
            "GET  /jobs/{job_id}",
            "GET  /jobs/{job_id}/result",
            "GET  /jobs/{job_id}/report",
            "GET  /reports",
            "POST /compare",
            "GET  /metrics",
            "GET  /docs",
        ],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "time": datetime.now().isoformat(timespec="seconds"),
        "jobs_total": len(_JOBS),
    }


@app.get("/engines")
def list_engines() -> Dict[str, Any]:
    """列出已配置引擎。"""
    engines = default_engines()
    return {
        "engines": [
            {
                "name": t.name,
                "base_url": t.normalized_base_url(),
                "model": t.model,
                "quantization": t.quantization,
                "note": t.note,
            }
            for t in engines.values()
        ],
        "default_bench": {
            "concurrency_levels": BENCH.concurrency_levels,
            "max_tokens": BENCH.max_tokens,
            "warmup_requests": BENCH.warmup_requests,
            "stream": BENCH.stream,
            "timeout": BENCH.timeout,
            "prompt": BENCH.prompt,
        },
    }


@app.post("/engines/check")
def check_engines(req: CheckRequest) -> Dict[str, Any]:
    """探活。用于压测前确认目标服务可达、模型已加载。"""
    available = default_engines()
    unknown = [n for n in req.engines if n not in available]
    if unknown:
        raise HTTPException(400, f"未知引擎：{unknown}，可选 {list(available)}")

    results = []
    for name in req.engines:
        target = available[name]
        client = UnifiedLLMClient(target, timeout=min(BENCH.timeout, 10.0))
        try:
            h = client.health()
        finally:
            client.close()
        results.append({"engine": name, "base_url": target.normalized_base_url(), **h})
    return {"results": results}


# ----------------------------------------------------------------------
# 压测任务
# ----------------------------------------------------------------------
def _apply_request_config(req: BenchRequest) -> None:
    """把请求参数写入全局 BENCH。"""
    if req.concurrency_levels:
        BENCH.concurrency_levels = sorted(req.concurrency_levels)
    if req.max_tokens is not None:
        BENCH.max_tokens = req.max_tokens
    if req.warmup_requests is not None:
        BENCH.warmup_requests = req.warmup_requests
    if req.timeout is not None:
        BENCH.timeout = req.timeout
    if req.stream is not None:
        BENCH.stream = req.stream
    if req.prompt:
        BENCH.prompt = req.prompt
    if req.requests_per_level is not None:
        BENCH.requests_per_level = req.requests_per_level


def _resolve_targets(req: BenchRequest) -> List[EngineTarget]:
    available = default_engines()
    unknown = [n for n in req.engines if n not in available]
    if unknown:
        raise HTTPException(400, f"未知引擎：{unknown}")
    targets = [available[n] for n in req.engines]
    if len(targets) == 1:
        t = targets[0]
        if req.base_url:
            t.base_url = req.base_url
        if req.model:
            t.model = req.model
        if req.quantization:
            t.quantization = req.quantization
    return targets


def _run_job(job_id: str, req: BenchRequest) -> None:
    """后台执行的压测任务主体。

    全程更新 ``_JOBS[job_id]``，前端靠轮询拿进度。
    """
    job = _JOBS[job_id]
    run_id = job["run_id"]
    output_dir = Path(job["output_dir"])
    targets = job["targets"]

    try:
        job["status"] = "running"
        job["started_at"] = datetime.now().isoformat(timespec="seconds")
        started = time.perf_counter()
        results: List[Dict[str, Any]] = []

        for target in targets:
            if job.get("cancel"):
                job["status"] = "cancelled"
                return

            job["current_engine"] = target.name
            job["stage_index"] = 0
            job["stage_total"] = len(BENCH.concurrency_levels)

            prom.METRICS.set_run_info(
                run_id=run_id, engine=target.name, model=target.model, status="running"
            )

            runner = LoadRunner(
                target,
                BENCH,
                monitor_config=MONITOR if req.enable_monitor and MONITOR.enabled else None,
            )
            health = runner.health()

            def on_stage(stage, _t=target) -> None:
                job["stage_index"] += 1
                job["last_stage"] = stage.to_dict()
                job["progress"] = round(
                    job["stage_index"] / max(job["stage_total"], 1), 4
                )
                # 把档位结果推给 Prometheus
                prom.METRICS.observe_stage(
                    stage, _t.name, _t.model, _t.quantization
                )

            stages = runner.run_sweep(on_stage=on_stage)
            duration = time.perf_counter() - started
            vram_curve = runner.vram_curve()

            result = build_result(
                target=target,
                bench=BENCH,
                stages=stages,
                vram_curve=vram_curve,
                health=health,
                run_id=run_id,
                duration=duration,
            )
            save_result(result, output_dir, run_id)
            results.append(result)

        job["results"] = results
        job["summary"] = [
            {
                "engine": r["engine"],
                "model": r["model"],
                "quantization": r["quantization"],
                "best_throughput": (r.get("summary") or {}).get("best_throughput"),
                "best_throughput_concurrency": (
                    r.get("summary") or {}
                ).get("best_throughput_concurrency"),
                "unstable_stages": (r.get("summary") or {}).get("unstable_stages") or [],
            }
            for r in results
        ]

        # 报告
        if req.generate_report and results:
            groups = _build_groups(results)
            matrix = compare(groups) if len(groups) > 1 else None
            paths = generate_reports(
                results, output_dir / run_id, quant_matrix=matrix
            )
            job["report"] = {
                "markdown": str(paths["markdown"]),
                "html": str(paths["html"]),
                "quant_matrix": matrix,
            }

        job["status"] = "completed"
        job["progress"] = 1.0
        job["duration_seconds"] = round(time.perf_counter() - started, 2)
        prom.METRICS.set_run_info(run_id=run_id, status="completed")

    except Exception as exc:
        logger.exception("任务 %s 执行失败", job_id)
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
        prom.METRICS.set_run_info(run_id=run_id, status="failed")
    finally:
        job["finished_at"] = datetime.now().isoformat(timespec="seconds")


def _build_groups(results: List[Dict[str, Any]]) -> List[QuantGroup]:
    """从结果构造量化组。"""
    from monitor.quant import _stage_from_dict

    return [
        QuantGroup(
            label=f"{r.get('model')} @ {r.get('quantization')}",
            engine=r.get("engine", "?"),
            model=r.get("model", "?"),
            quantization=r.get("quantization", "unknown"),
            stages=[_stage_from_dict(s) for s in r.get("stages", [])],
            vram_peak_mib=r.get("vram_peak_mib"),
            extra={
                "is_baseline": r.get("quantization", "").lower() in {"fp16", "bf16"}
            },
        )
        for r in results
    ]


@app.post("/bench")
def create_bench(req: BenchRequest, background: BackgroundTasks) -> Dict[str, Any]:
    """提交压测任务，立即返回 job_id。"""
    _apply_request_config(req)
    targets = _resolve_targets(req)

    job_id = uuid.uuid4().hex[:12]
    run_id = new_run_id()

    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "run_id": run_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "engines": req.engines,
            "targets": targets,
            "output_dir": BENCH.output_dir,
            "progress": 0.0,
            "stage_index": 0,
            "stage_total": len(BENCH.concurrency_levels),
            "request": req.model_dump(),
        }

    background.add_task(_run_job, job_id, req)
    return {
        "job_id": job_id,
        "run_id": run_id,
        "status": "pending",
        "poll": f"/jobs/{job_id}",
    }


@app.get("/jobs")
def list_jobs() -> Dict[str, Any]:
    with _JOBS_LOCK:
        items = [
            {k: v for k, v in j.items() if k not in {"targets", "results"}}
            for j in _JOBS.values()
        ]
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"count": len(items), "jobs": items}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"任务不存在：{job_id}")
    return {k: v for k, v in job.items() if k not in {"targets", "results"}}


@app.get("/jobs/{job_id}/result")
def get_job_result(job_id: str) -> Dict[str, Any]:
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"任务不存在：{job_id}")
    if job["status"] not in {"completed", "failed", "cancelled"}:
        raise HTTPException(409, f"任务尚未结束，当前状态：{job['status']}")
    return {
        "job_id": job_id,
        "status": job["status"],
        "error": job.get("error"),
        "summary": job.get("summary"),
        "results": job.get("results") or [],
        "report": job.get("report"),
    }


@app.get("/jobs/{job_id}/report", response_class=HTMLResponse)
def get_job_report(job_id: str, fmt: str = "html") -> Response:
    """直接返回报告内容（默认 HTML）。"""
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"任务不存在：{job_id}")
    report = job.get("report")
    if not report:
        raise HTTPException(404, "该任务没有生成报告")

    path = Path(report["html"] if fmt == "html" else report["markdown"])
    if not path.exists():
        raise HTTPException(404, f"报告文件不存在：{path}")
    media = "text/html; charset=utf-8" if fmt == "html" else "text/markdown; charset=utf-8"
    return Response(path.read_text(encoding="utf-8"), media_type=media)


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str) -> Dict[str, Any]:
    """请求取消任务（仅在档位之间生效）。"""
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"任务不存在：{job_id}")
    if job["status"] in {"completed", "failed"}:
        raise HTTPException(409, f"任务已结束：{job['status']}")
    job["cancel"] = True
    return {"job_id": job_id, "status": "cancelling"}


# ----------------------------------------------------------------------
# 报告与对比
# ----------------------------------------------------------------------
@app.get("/reports")
def list_reports(limit: int = 20) -> Dict[str, Any]:
    """列出已生成的报告文件。"""
    root = Path(BENCH.output_dir)
    if not root.exists():
        return {"count": 0, "reports": []}
    files = sorted(
        list(root.glob("*/*.html")) + list(root.glob("*/*.md")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    return {
        "count": len(files),
        "reports": [
            {
                "path": str(p),
                "run_id": p.parent.name,
                "name": p.name,
                "format": p.suffix.lstrip("."),
                "size_bytes": p.stat().st_size,
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(
                    timespec="seconds"
                ),
            }
            for p in files
        ],
    }


@app.get("/reports/{run_id}/{filename}")
def get_report(run_id: str, filename: str) -> Response:
    """按 run_id + 文件名读取报告。"""
    path = Path(BENCH.output_dir) / run_id / filename
    if not path.exists() or path.suffix not in {".html", ".md", ".json", ".csv"}:
        raise HTTPException(404, "报告不存在")
    media = {
        ".html": "text/html; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
        ".json": "application/json",
        ".csv": "text/csv; charset=utf-8",
    }[path.suffix]
    return Response(path.read_text(encoding="utf-8"), media_type=media)


@app.post("/compare")
def post_compare(req: CompareRequest) -> Dict[str, Any]:
    """对某次 run 的多份结果做量化横向对比。"""
    root = Path(req.output_dir) if req.output_dir else Path(BENCH.output_dir)
    run_dir = root / req.run_id
    if not run_dir.exists():
        raise HTTPException(404, f"结果目录不存在：{run_dir}")

    import json

    items = []
    for p in sorted(run_dir.glob("*.json")):
        if p.name in {"report.json", "compare.json"}:
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        items.append(data)

    if len(items) < 2:
        raise HTTPException(400, f"至少需要 2 份结果才能对比，当前 {len(items)} 份")

    groups = from_result_dicts(items)
    matrix = compare(groups)
    return {"run_id": req.run_id, "matrix": matrix}


# ----------------------------------------------------------------------
# Prometheus
# ----------------------------------------------------------------------
@app.get("/metrics")
def metrics_endpoint() -> Response:
    """Prometheus 抓取端点。"""
    return Response(prom.render_latest(), media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/metrics/snapshot")
def metrics_snapshot() -> JSONResponse:
    """指标的 JSON 快照，便于前端/CI 直接消费。"""
    return JSONResponse(prom.snapshot())


def serve() -> None:
    """以脚本方式启动服务。"""
    import uvicorn

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    print(f"\n  llm-inference-bench API  →  http://{API_HOST}:{API_PORT}")
    print(f"  交互式文档               →  http://{API_HOST}:{API_PORT}/docs")
    print(f"  Prometheus 指标          →  http://{API_HOST}:{API_PORT}/metrics\n")
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level=LOG_LEVEL.lower())


if __name__ == "__main__":
    serve()
