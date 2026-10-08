"""监控与横向评测模块。

- :mod:`monitor.gpu`   — GPU 显存 / 利用率采样（显存占用曲线）
- :mod:`monitor.quant` — 量化等级（FP16/INT8/AWQ）横向对比矩阵
"""

from monitor.gpu import GpuMonitor, GpuSample, nvidia_smi_available, query_gpus
from monitor.quant import (
    QuantGroup,
    compare,
    from_result_dicts,
    pick_baseline,
    summary_verdict,
)

__all__ = [
    "GpuMonitor",
    "GpuSample",
    "nvidia_smi_available",
    "query_gpus",
    "QuantGroup",
    "compare",
    "from_result_dicts",
    "pick_baseline",
    "summary_verdict",
]
