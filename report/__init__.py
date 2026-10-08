"""报告生成模块。

- :mod:`report.fmt`       — 数字/单位格式化
- :mod:`report.svgchart`  — 手写 SVG 图表（离线可用）
- :mod:`report.markdown`  — Markdown 报告
- :mod:`report.html`      — 自包含 HTML 报告

统一入口 :func:`report.generator.generate_reports`，一次产出两种格式。
"""

from report.generator import generate_reports
from report.markdown import build_markdown, save_markdown
from report.html import build_html, save_html

__all__ = [
    "generate_reports",
    "build_markdown",
    "save_markdown",
    "build_html",
    "save_html",
]
