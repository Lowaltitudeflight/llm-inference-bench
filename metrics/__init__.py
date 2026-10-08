"""指标计算与结果存储。

- :mod:`metrics.stats` — TTFT / TPOT / 吞吐 / 分位数聚合
- :mod:`metrics.store` — 结果落盘（JSON / CSV）与环境信息采集
"""

from metrics.stats import (
    RequestResult,
    StageMetrics,
    aggregate,
    percentile,
    summarize_run,
)
from metrics.store import (
    build_result,
    environment_info,
    list_results,
    load_latest,
    load_result,
    new_run_id,
    save_result,
    stage_rows,
)

__all__ = [
    "RequestResult",
    "StageMetrics",
    "aggregate",
    "percentile",
    "summarize_run",
    "build_result",
    "environment_info",
    "list_results",
    "load_latest",
    "load_result",
    "new_run_id",
    "save_result",
    "stage_rows",
]
