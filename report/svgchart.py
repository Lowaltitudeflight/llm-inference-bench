"""手写 SVG 图表。

为什么不引 ECharts / Chart.js：
- HTML 报告必须能**离线打开**，引 CDN 一断网就白屏；
- 压测报告的图表类型很固定（折线 + 柱状），手写 SVG 只有几百行，
  比引一个 1MB 的库更干净，也更好嵌进模板。

生成的都是纯静态 SVG 字符串，无 JS 交互。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 画布尺寸
WIDTH = 860
HEIGHT = 340
PAD_LEFT = 78
PAD_RIGHT = 150  # 右侧留给图例
PAD_TOP = 48
PAD_BOTTOM = 58

PLOT_W = WIDTH - PAD_LEFT - PAD_RIGHT
PLOT_H = HEIGHT - PAD_TOP - PAD_BOTTOM

GRID_COLOR = "#e5e7eb"
AXIS_COLOR = "#9ca3af"
TEXT_COLOR = "#4b5563"
TITLE_COLOR = "#1f2430"
BG_COLOR = "#ffffff"


def _esc(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _nice_ticks(vmin: float, vmax: float, count: int = 5) -> List[float]:
    """生成「好看」的刻度值（1/2/5 × 10^n）。"""
    if vmax == vmin:
        vmax = vmin + 1.0
    span = vmax - vmin
    raw_step = span / max(1, count)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for mult in (1, 2, 2.5, 5, 10):
        step = magnitude * mult
        if step >= raw_step:
            break
    start = math.floor(vmin / step) * step
    ticks = []
    value = start
    while value <= vmax + step * 0.5:
        if value >= vmin - step * 0.5:
            ticks.append(round(value, 10))
        value += step
    return ticks


def _fmt_tick(value: float) -> str:
    """刻度标签：大数用 k / M 缩写，避免挤在一起。"""
    abs_v = abs(value)
    if abs_v >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_v >= 10_000:
        return f"{value / 1000:.0f}k"
    if abs_v >= 1000:
        return f"{value / 1000:.1f}k"
    if abs_v == 0:
        return "0"
    if abs_v < 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if abs_v < 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.0f}"


def _log_axis(values: Sequence[float]) -> bool:
    """是否该用对数 X 轴：并发梯度通常 1,2,4,8,16,32，线性轴会把
    小并发全挤在左侧。跨度 >= 8 倍时切对数轴。"""
    positive = [v for v in values if v > 0]
    if len(positive) < 3:
        return False
    return max(positive) / min(positive) >= 8


def _legend(names_colors: List[Tuple[str, str]], x: float, y: float) -> str:
    """右上/右侧图例。"""
    out = []
    cy = y
    for name, color in names_colors:
        out.append(
            f'<rect x="{x:.1f}" y="{cy - 5:.1f}" width="11" height="11" '
            f'rx="2" fill="{color}"/>'
        )
        out.append(
            f'<text x="{x + 17:.1f}" y="{cy + 4:.1f}" font-size="11" '
            f'fill="{TEXT_COLOR}" font-family="sans-serif">{_esc(name)}</text>'
        )
        cy += 20
    return "".join(out)


def multi_line_chart(
    series: List[Dict[str, Any]],
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    width: int = WIDTH,
    height: int = HEIGHT,
) -> str:
    """多系列折线图。

    Args:
        series: ``[{"name": str, "color": str, "points": [(x, y), ...]}]``
        x_label / y_label / title: 轴标签与标题。
    """
    if not series:
        return _empty_svg("暂无数据", width, height)

    pad_r = PAD_RIGHT if len(series) > 1 else 34
    plot_w = width - PAD_LEFT - pad_r

    xs = [p[0] for s in series for p in s["points"]]
    ys = [p[1] for s in series for p in s["points"] if p[1] is not None]
    if not xs or not ys:
        return _empty_svg("暂无数据", width, height)

    use_log = _log_axis(xs)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_max == y_min:
        y_max = y_min + 1.0
    # Y 轴从 0 起更直观（但显存曲线除外时也接受）
    y_lo = 0.0 if y_min >= 0 else y_min
    y_hi = y_max * 1.06

    def px(x: float) -> float:
        if use_log:
            lx = math.log10(max(x, 1e-9))
            lo, hi = math.log10(max(x_min, 1e-9)), math.log10(x_max)
            if hi == lo:
                return PAD_LEFT + plot_w / 2
            return PAD_LEFT + (lx - lo) / (hi - lo) * plot_w
        if x_max == x_min:
            return PAD_LEFT + plot_w / 2
        return PAD_LEFT + (x - x_min) / (x_max - x_min) * plot_w

    def py(y: float) -> float:
        if y_hi == y_lo:
            return PAD_TOP + PLOT_H / 2
        return PAD_TOP + PLOT_H - (y - y_lo) / (y_hi - y_lo) * PLOT_H

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" style="max-width:{width}px" role="img">',
        f'<rect width="{width}" height="{height}" fill="{BG_COLOR}" rx="10"/>',
    ]
    if title:
        out.append(
            f'<text x="{PAD_LEFT}" y="26" font-size="14.5" font-weight="600" '
            f'fill="{TITLE_COLOR}" font-family="sans-serif">{_esc(title)}</text>'
        )

    # Y 轴网格与刻度
    for tick in _nice_ticks(y_lo, y_hi, 5):
        if tick < y_lo - 1e-9 or tick > y_hi + 1e-9:
            continue
        y = py(tick)
        out.append(
            f'<line x1="{PAD_LEFT}" y1="{y:.1f}" x2="{PAD_LEFT + plot_w}" '
            f'y2="{y:.1f}" stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{PAD_LEFT - 10}" y="{y + 4:.1f}" font-size="11" '
            f'text-anchor="end" fill="{TEXT_COLOR}" font-family="sans-serif">'
            f"{_fmt_tick(tick)}</text>"
        )

    # X 轴刻度
    x_ticks = sorted(set(xs))
    if len(x_ticks) > 12:
        step = max(1, len(x_ticks) // 10)
        x_ticks = x_ticks[::step]
    for xt in x_ticks:
        x = px(xt)
        out.append(
            f'<line x1="{x:.1f}" y1="{PAD_TOP + PLOT_H}" x2="{x:.1f}" '
            f'y2="{PAD_TOP + PLOT_H + 5}" stroke="{AXIS_COLOR}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{PAD_TOP + PLOT_H + 20}" font-size="11" '
            f'text-anchor="middle" fill="{TEXT_COLOR}" font-family="sans-serif">'
            f"{_fmt_tick(xt)}</text>"
        )

    # 坐标轴
    out.append(
        f'<line x1="{PAD_LEFT}" y1="{PAD_TOP + PLOT_H}" x2="{PAD_LEFT + plot_w}" '
        f'y2="{PAD_TOP + PLOT_H}" stroke="{AXIS_COLOR}" stroke-width="1.4"/>'
    )
    out.append(
        f'<line x1="{PAD_LEFT}" y1="{PAD_TOP}" x2="{PAD_LEFT}" '
        f'y2="{PAD_TOP + PLOT_H}" stroke="{AXIS_COLOR}" stroke-width="1.4"/>'
    )

    # 折线与数据点
    for s in series:
        pts = [(px(x), py(y)) for x, y in s["points"] if y is not None]
        if not pts:
            continue
        color = s.get("color", "#2563eb")
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        if len(pts) > 1:
            out.append(
                f'<polyline points="{poly}" fill="none" stroke="{color}" '
                f'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        for x, y in pts:
            out.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="#ffffff" '
                f'stroke="{color}" stroke-width="2.2"/>'
            )

    # 轴标签
    if x_label:
        out.append(
            f'<text x="{PAD_LEFT + plot_w / 2:.1f}" y="{height - 8}" font-size="11.5" '
            f'text-anchor="middle" fill="{TEXT_COLOR}" font-family="sans-serif">'
            f"{_esc(x_label)}</text>"
        )
    if y_label:
        cy = PAD_TOP + PLOT_H / 2
        out.append(
            f'<text x="16" y="{cy:.1f}" font-size="11.5" text-anchor="middle" '
            f'fill="{TEXT_COLOR}" font-family="sans-serif" '
            f'transform="rotate(-90 16 {cy:.1f})">{_esc(y_label)}</text>'
        )

    # 图例（右侧竖排）
    if len(series) > 1:
        out.append(_legend([(s["name"], s.get("color", "#2563eb")) for s in series], PAD_LEFT + plot_w + 18, PAD_TOP + 12))
    elif series:
        out.append(
            f'<text x="{PAD_LEFT + plot_w / 2:.1f}" y="{PAD_TOP - 10}" font-size="11.5" '
            f'text-anchor="middle" fill="{TEXT_COLOR}" font-family="sans-serif">'
            f"{_esc(series[0]['name'])}</text>"
        )

    out.append("</svg>")
    return "".join(out)


def bar_chart(
    items: List[Dict[str, Any]],
    y_label: str = "",
    title: str = "",
    width: int = WIDTH,
    height: int = HEIGHT,
) -> str:
    """柱状图。

    Args:
        items: ``[{"label": str, "value": float, "color": str}]``
    """
    if not items:
        return _empty_svg("暂无数据", width, height)

    plot_w = width - PAD_LEFT - 40
    values = [i["value"] for i in items if i.get("value") is not None]
    if not values:
        return _empty_svg("暂无数据", width, height)
    y_hi = max(values) * 1.15 or 1.0

    n = len(items)
    slot = plot_w / n
    bar_w = min(slot * 0.56, 86)

    def py(y: float) -> float:
        return PAD_TOP + PLOT_H - (y / y_hi) * PLOT_H

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" style="max-width:{width}px" role="img">',
        f'<rect width="{width}" height="{height}" fill="{BG_COLOR}" rx="10"/>',
    ]
    if title:
        out.append(
            f'<text x="{PAD_LEFT}" y="26" font-size="14.5" font-weight="600" '
            f'fill="{TITLE_COLOR}" font-family="sans-serif">{_esc(title)}</text>'
        )

    for tick in _nice_ticks(0, y_hi, 5):
        if tick > y_hi + 1e-9:
            continue
        y = py(tick)
        out.append(
            f'<line x1="{PAD_LEFT}" y1="{y:.1f}" x2="{PAD_LEFT + plot_w}" '
            f'y2="{y:.1f}" stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{PAD_LEFT - 10}" y="{y + 4:.1f}" font-size="11" '
            f'text-anchor="end" fill="{TEXT_COLOR}" font-family="sans-serif">'
            f"{_fmt_tick(tick)}</text>"
        )

    for idx, item in enumerate(items):
        cx = PAD_LEFT + slot * idx + slot / 2
        value = item.get("value")
        if value is None:
            continue
        y = py(value)
        bar_h = PAD_TOP + PLOT_H - y
        color = item.get("color", "#2563eb")
        out.append(
            f'<rect x="{cx - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{max(bar_h, 1):.1f}" rx="4" fill="{color}" opacity="0.88"/>'
        )
        out.append(
            f'<text x="{cx:.1f}" y="{y - 7:.1f}" font-size="11.5" text-anchor="middle" '
            f'fill="{TITLE_COLOR}" font-weight="600" font-family="sans-serif">'
            f"{_fmt_tick(value)}</text>"
        )
        # 标签可能含换行符，逐行渲染
        for li, line in enumerate(str(item.get("label", "")).split("\n")):
            out.append(
                f'<text x="{cx:.1f}" y="{PAD_TOP + PLOT_H + 19 + li * 14:.1f}" '
                f'font-size="11" text-anchor="middle" fill="{TEXT_COLOR}" '
                f'font-family="sans-serif">{_esc(line)}</text>'
            )

    out.append(
        f'<line x1="{PAD_LEFT}" y1="{PAD_TOP + PLOT_H}" x2="{PAD_LEFT + plot_w}" '
        f'y2="{PAD_TOP + PLOT_H}" stroke="{AXIS_COLOR}" stroke-width="1.4"/>'
    )
    out.append(
        f'<line x1="{PAD_LEFT}" y1="{PAD_TOP}" x2="{PAD_LEFT}" '
        f'y2="{PAD_TOP + PLOT_H}" stroke="{AXIS_COLOR}" stroke-width="1.4"/>'
    )
    if y_label:
        cy = PAD_TOP + PLOT_H / 2
        out.append(
            f'<text x="16" y="{cy:.1f}" font-size="11.5" text-anchor="middle" '
            f'fill="{TEXT_COLOR}" font-family="sans-serif" '
            f'transform="rotate(-90 16 {cy:.1f})">{_esc(y_label)}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def _empty_svg(message: str, width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" style="max-width:{width}px">'
        f'<rect width="{width}" height="{height}" fill="{BG_COLOR}" rx="10"/>'
        f'<text x="{width / 2}" y="{height / 2}" font-size="13" text-anchor="middle" '
        f'fill="{AXIS_COLOR}" font-family="sans-serif">{_esc(message)}</text></svg>'
    )
