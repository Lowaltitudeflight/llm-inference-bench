"""GPU 显存与利用率采样。

通过 ``nvidia-smi`` 的 CSV 查询模式采集每张卡的显存占用、利用率、
功耗与温度。设计目标是**任何环境下都不抛异常**：

- 没有 NVIDIA 驱动 → 返回 available=False，监控自动降级为「不监控」；
- 采样失败 → 记录错误但压测继续，不能因为监控挂掉而中断压测。

采样在独立线程中按固定间隔执行，与压测主流程解耦。
"""

from __future__ import annotations

import csv
import io
import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 单次 nvidia-smi 调用的超时（秒）。驱动卡死时不能拖住整个压测。
QUERY_TIMEOUT = 5.0

# 查询字段：索引 / 卡名 / 显存总量 / 显存已用 / 利用率 / 功耗 / 温度
QUERY_FIELDS = (
    "index,name,memory.total,memory.used,utilization.gpu,"
    "power.draw,temperature.gpu"
)


@dataclass
class GpuSample:
    """单张卡的一次瞬时采样。

    显存单位统一为 MiB，功率为 W，温度为摄氏度。
    """

    index: int
    name: str = ""
    memory_total_mib: Optional[float] = None
    memory_used_mib: Optional[float] = None
    util_percent: Optional[float] = None
    power_watts: Optional[float] = None
    temperature_c: Optional[float] = None
    # 采样时刻（相对压测开始的秒数），由 Recorder 填充
    t: float = 0.0

    @property
    def memory_used_percent(self) -> Optional[float]:
        """显存占用率（%）。总量为 0 或缺失时返回 None。"""
        if not self.memory_total_mib or self.memory_used_mib is None:
            return None
        return self.memory_used_mib / self.memory_total_mib * 100.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["memory_used_percent"] = self.memory_used_percent
        return data


def _to_float(text: str) -> Optional[float]:
    """把 nvidia-smi 的字段转成 float，兼容 'N/A' 与 '[N/A]'。"""
    cleaned = (text or "").strip().strip("[]")
    if not cleaned or cleaned.upper() in {"N/A", "NA", "UNKNOWN"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def nvidia_smi_available() -> bool:
    """检测 nvidia-smi 是否可用（命令存在且能成功执行）。"""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={QUERY_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=QUERY_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if proc.returncode != 0:
        return False
    # 有输出且首行能被逗号切开，才认为真的可用
    first = (proc.stdout or "").strip().splitlines()
    return bool(first) and len(first[0].split(",")) >= 3


def query_gpus() -> Dict[str, Any]:
    """查询一次全部 GPU 状态。

    Returns:
        {"available": bool, "gpus": [GpuSample, ...], "message": str}
        任何异常都转成 available=False，绝不向上抛。
    """
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "gpus": [], "message": "nvidia-smi 未安装"}

    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={QUERY_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=QUERY_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"available": False, "gpus": [], "message": "nvidia-smi 调用超时"}
    except OSError as exc:
        return {"available": False, "gpus": [], "message": f"nvidia-smi 调用失败: {exc}"}

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()[:200]
        return {
            "available": False,
            "gpus": [],
            "message": f"nvidia-smi 退出码 {proc.returncode}: {err}",
        }

    gpus: List[GpuSample] = []
    reader = csv.reader(io.StringIO(proc.stdout or ""))
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            continue
        # 字段缺失时用 None 补齐，保持容错
        row = [cell.strip() for cell in row] + [""] * len(QUERY_FIELDS.split(","))
        index = _to_float(row[0])
        if index is None:
            continue
        gpus.append(
            GpuSample(
                index=int(index),
                name=row[1],
                memory_total_mib=_to_float(row[2]),
                memory_used_mib=_to_float(row[3]),
                util_percent=_to_float(row[4]),
                power_watts=_to_float(row[5]),
                temperature_c=_to_float(row[6]),
            )
        )

    if not gpus:
        return {"available": False, "gpus": [], "message": "未解析到任何 GPU"}
    return {"available": True, "gpus": gpus, "message": f"检测到 {len(gpus)} 张 GPU"}


class GpuMonitor:
    """后台显存采样器。

    用法::

        with GpuMonitor(interval=0.5) as mon:
            ...  # 跑压测
        curve = mon.curve()   # 拿到显存曲线

    以线程实现，不阻塞主流程；``stop()`` 后可按时间戳对齐压测各阶段。
    """

    def __init__(self, interval: float = 0.5, enabled: bool = True):
        self.interval = max(0.05, interval)
        self.enabled = enabled
        self.available = False
        self.message = ""
        self._samples: List[GpuSample] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> "GpuMonitor":
        """开始采样。无 GPU 时静默降级，不启动线程。"""
        if not self.enabled:
            self.message = "监控已关闭"
            return self

        probe = query_gpus()
        self.available = probe["available"]
        self.message = probe["message"]
        if not self.available:
            logger.warning("GPU 监控不可用，已降级：%s", self.message)
            return self

        self._start_time = time.perf_counter()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="gpu-monitor", daemon=True)
        self._thread.start()
        logger.info("GPU 监控已启动，采样间隔 %.2fs，%s", self.interval, self.message)
        return self

    def stop(self) -> None:
        """停止采样并等待线程退出。"""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self.interval * 4 + 1.0)
        self._thread = None

    def __enter__(self) -> "GpuMonitor":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self.sample_once()
            self._stop_event.wait(self.interval)

    def sample_once(self) -> List[GpuSample]:
        """采一次样并记录，返回本次样本。异常一律吞掉只记日志。"""
        try:
            probe = query_gpus()
            if not probe["available"]:
                return []
            now = time.perf_counter()
            elapsed = now - self._start_time if self._start_time else 0.0
            batch: List[GpuSample] = []
            for sample in probe["gpus"]:
                sample.t = round(elapsed, 4)
                batch.append(sample)
            with self._lock:
                self._samples.extend(batch)
            return batch
        except Exception as exc:  # 监控绝不能影响压测
            logger.debug("GPU 采样失败（忽略）: %s", exc)
            return []

    # ------------------------------------------------------------------
    # 结果
    # ------------------------------------------------------------------
    @property
    def samples(self) -> List[GpuSample]:
        with self._lock:
            return list(self._samples)

    def curve(self) -> Dict[str, Any]:
        """按卡聚合出显存/利用率曲线，供报告绘图使用。

        Returns:
            {
              "available": bool,
              "interval": float,
              "gpus": [
                 {"index", "name", "memory_total_mib",
                  "series": [{"t","memory_used_mib","memory_used_percent",
                              "util_percent","power_watts","temperature_c"}],
                  "peak_memory_used_mib", "peak_memory_used_percent",
                  "mean_util_percent"}
              ],
              "peak_memory_used_mib": 全局峰值,
              "message": str
            }
        """
        samples = self.samples
        grouped: Dict[int, List[GpuSample]] = {}
        for s in samples:
            grouped.setdefault(s.index, []).append(s)

        # 全局峰值必须是「同一时刻各卡之和」的最大值。
        # 若直接取各卡峰值的最大者，多卡场景下会严重低估——
        # 8 卡各用 60GiB 时真实占用 480GiB，而按单卡峰值只报 60GiB。
        by_t: Dict[float, float] = {}
        for s in samples:
            if s.memory_used_mib is None:
                continue
            by_t[s.t] = by_t.get(s.t, 0.0) + s.memory_used_mib

        gpu_series: List[Dict[str, Any]] = []
        for index in sorted(grouped):
            rows = sorted(grouped[index], key=lambda x: x.t)
            memory_used = [r.memory_used_mib for r in rows if r.memory_used_mib is not None]
            util = [r.util_percent for r in rows if r.util_percent is not None]
            peak_used = max(memory_used) if memory_used else None
            total = rows[0].memory_total_mib if rows else None
            gpu_series.append(
                {
                    "index": index,
                    "name": rows[0].name if rows else "",
                    "memory_total_mib": total,
                    "sample_count": len(rows),
                    "peak_memory_used_mib": peak_used,
                    "peak_memory_used_percent": (
                        peak_used / total * 100.0 if peak_used is not None and total else None
                    ),
                    "mean_util_percent": sum(util) / len(util) if util else None,
                    "max_util_percent": max(util) if util else None,
                    "series": [
                        {
                            "t": r.t,
                            "memory_used_mib": r.memory_used_mib,
                            "memory_used_percent": r.memory_used_percent,
                            "util_percent": r.util_percent,
                            "power_watts": r.power_watts,
                            "temperature_c": r.temperature_c,
                        }
                        for r in rows
                    ],
                }
            )

        total_peak = max(by_t.values()) if by_t else None
        total_capacity = sum(
            g["memory_total_mib"] for g in gpu_series if g["memory_total_mib"]
        ) or None

        return {
            "available": self.available and bool(gpu_series),
            "interval": self.interval,
            "gpu_count": len(gpu_series),
            # 全卡合计峰值（同一时刻求和），以及它占总显存的比例
            "peak_memory_used_mib": round(total_peak, 1) if total_peak is not None else None,
            "total_memory_mib": total_capacity,
            "peak_memory_used_percent": (
                total_peak / total_capacity * 100.0
                if total_peak is not None and total_capacity
                else None
            ),
            "gpus": gpu_series,
            "message": self.message,
        }

    def summary(self) -> Dict[str, Any]:
        """轻量摘要，用于阶段指标里附带一行显存信息。"""
        curve = self.curve()
        if not curve["available"]:
            return {"available": False}
        return {
            "available": True,
            "gpu_count": curve["gpu_count"],
            "peak_memory_used_mib": curve["peak_memory_used_mib"],
            "per_gpu_peak_mib": {
                g["index"]: g["peak_memory_used_mib"] for g in curve["gpus"]
            },
        }
