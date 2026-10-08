"""格式化辅助函数。

报告的每个数字都要带单位、有合理精度、缺失时显示 N/A 而不是
"None"。集中在这里，避免各模板各写一套。

同时提供:**自适应单位**的耗时格式化。大模型推理的耗时跨了 4 个
数量级——TTFT 可能是 80ms，TPOT 可能是 12ms，单层算子可能是 400ns。
固定用秒会显示成一串 0.0000004，固定用毫秒会把 80s 显示成 80000ms。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

NA = "N/A"


def num(value: Optional[float], digits: int = 3) -> str:
    """数字格式化，None -> N/A。"""
    if value is None:
        return NA
    try:
        return f"{value:.{digits}f}"
    except (TypeError, ValueError):
        return NA


def seconds(value: Optional[float], digits: int = 4) -> str:
    """秒格式化，带单位。"""
    if value is None:
        return NA
    return f"{value:.{digits}f} s"


def ms(value: Optional[float], digits: int = 2) -> str:
    """毫秒格式化。输入为秒。

    大模型推理的延迟基本都在毫秒量级，报告里统一用 ms 更直观。
    """
    if value is None:
        return NA
    return f"{value * 1000:.{digits}f} ms"


def adaptive_seconds(value: Optional[float], digits: int = 3) -> str:
    """按数量级自动选择 s / ms / µs / ns。输入为秒。"""
    if value is None:
        return NA
    if value == 0:
        return "0 s"
    abs_v = abs(value)
    if abs_v >= 1:
        return f"{value:.{digits}f} s"
    if abs_v >= 1e-3:
        return f"{value * 1e3:.{digits}f} ms"
    if abs_v >= 1e-6:
        return f"{value * 1e6:.{digits}f} µs"
    return f"{value * 1e9:.{digits}f} ns"


def pct(value: Optional[float], digits: int = 1) -> str:
    """百分比格式化。输入为 0~1 的比例。"""
    if value is None:
        return NA
    return f"{value * 100:.{digits}f}%"


def tps(value: Optional[float], digits: int = 1) -> str:
    """吞吐格式化（tokens/s）。"""
    if value is None:
        return NA
    return f"{value:.{digits}f} tokens/s"


def mib(value: Optional[float], digits: int = 1) -> str:
    """显存格式化，自动 MiB -> GiB。"""
    if value is None:
        return NA
    if abs(value) >= 1024:
        return f"{value / 1024:.{digits}f} GiB"
    return f"{value:.{digits}f} MiB"


def ratio(value: Optional[float], digits: int = 2) -> str:
    """倍数格式化。"""
    if value is None:
        return NA
    return f"{value:.{digits}f}×"


def signed_pct(value: Optional[float], digits: int = 1) -> str:
    """带符号百分比（输入已是百分数）。"""
    if value is None:
        return NA
    return f"{value:+.{digits}f}%"


def render_number(value: Any, spec: str = "") -> str:
    """模板过滤器：``{{ v|n }}`` / ``{{ v|n(2) }}``。

    Jinja2 模板里对 None 做格式化会报错，统一走这里。
    """
    if value is None:
        return NA
    if isinstance(value, float):
        digits = int(spec) if spec else 3
        return f"{value:.{digits}f}"
    return str(value)


FILTERS: Dict[str, Any] = {
    "n": render_number,
    "seconds": lambda v, d=4: seconds(v, d),
    "ms": lambda v, d=2: ms(v, d),
    "adaptive": lambda v, d=3: adaptive_seconds(v, d),
    "pct": lambda v, d=1: pct(v, d),
    "tps": lambda v, d=1: tps(v, d),
    "mib": lambda v, d=1: mib(v, d),
    "ratio": lambda v, d=2: ratio(v, d),
    "signed_pct": lambda v, d=1: signed_pct(v, d),
}
