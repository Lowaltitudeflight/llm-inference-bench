"""报告生成的统一入口。

把「压测结果 → 报告文件」这一步收口到一个函数，调用方不需要关心
Markdown / HTML 各自的细节，也不需要自己拼量化对比矩阵。

典型用法::

    from report import generate_reports
    from monitor.quant import QuantGroup

    paths = generate_reports(
        results=[vllm_result, ollama_result],
        quant_groups=[fp16_group, awq_group],   # 可选
        output_dir=Path("data/results/run-xxx"),
    )
    print(paths["html"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from monitor.quant import QuantGroup, compare
from report.html import build_html, save_html
from report.markdown import build_markdown, save_markdown

logger = logging.getLogger(__name__)


def generate_reports(
    results: List[Dict[str, Any]],
    output_dir: Path,
    quant_groups: Optional[List[QuantGroup]] = None,
    quant_matrix: Optional[Dict[str, Any]] = None,
    title: str = "大模型推理性能压测报告",
    basename: str = "report",
) -> Dict[str, Path]:
    """生成 Markdown + HTML 两份报告。

    Args:
        results: 各引擎压测结果字典列表。
        output_dir: 输出目录（通常是某次 run 的目录）。
        quant_groups: 量化组列表；给了就自动算对比矩阵。
        quant_matrix: 已算好的对比矩阵；与 ``quant_groups`` 二选一，
            两个都给时以 ``quant_matrix`` 为准。
        title: 报告标题。
        basename: 输出文件名前缀。

    Returns:
        ``{"markdown": Path, "html": Path, "matrix": ...}``
        —— ``matrix`` 是本次用到的量化矩阵（可能为空）。
    """
    if quant_matrix is None and quant_groups:
        quant_matrix = compare(quant_groups)
        logger.info("已生成量化对比矩阵：%d 组", len(quant_groups))

    md_text = build_markdown(results, quant_matrix, title=title)
    html_text = build_html(results, quant_matrix, title=title)

    output_dir = Path(output_dir)
    md_path = save_markdown(md_text, output_dir / f"{basename}.md")
    html_path = save_html(html_text, output_dir / f"{basename}.html")

    return {
        "markdown": md_path,
        "html": html_path,
        "matrix": quant_matrix or {},
    }
