#!/usr/bin/env python3
"""Generate signal-flow diagrams and sonograms for the reverb primer."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageChops, ImageDraw, ImageFont
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "reverb_primer"
AUDIO = ROOT / "examples" / "audio"

WIDTH = 1600
HEIGHT = 940
FDN_OUTPUT_PROJECTION_BOX = (1620, 10, 1940, 160)
FDN_GAIN_BOX = (1660, 190, 1900, 720)
FDN_MATRIX_BOX = (2050, 190, 2290, 720)
WHITE = "#ffffff"
INK = "#123431"
MUTED = "#526762"
GRID = "#d9dfdc"
GOLD = "#d5a84b"
TEAL = "#2b8c7f"
BLUE = "#2f6680"
RUST = "#ae5f3e"
CREAM = "#f8f4ea"
PALE_GREEN = "#e8f0ed"


def font(
    size: int,
    bold: bool = False,
    italic: bool = False,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    georgia_style = {
        (False, False): "Georgia.ttf",
        (True, False): "Georgia Bold.ttf",
        (False, True): "Georgia Italic.ttf",
        (True, True): "Georgia Bold Italic.ttf",
    }[(bold, italic)]
    arial_style = {
        (False, False): "Arial.ttf",
        (True, False): "Arial Bold.ttf",
        (False, True): "Arial Italic.ttf",
        (True, True): "Arial Bold Italic.ttf",
    }[(bold, italic)]
    candidates = [
        f"/System/Library/Fonts/Supplemental/{georgia_style}",
        f"/System/Library/Fonts/Supplemental/{arial_style}",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font(46, True)
F_SUBTITLE = font(22)
F_NODE = font(23, True)
F_SMALL = font(18)
F_TINY = font(15)
F_FLOW = font(34)
F_FLOW_BOLD = font(34, True)
F_FLOW_SMALL = font(25)


def _scaled_font(selected_font: ImageFont.ImageFont, scale: float) -> ImageFont.ImageFont:
    if isinstance(selected_font, ImageFont.FreeTypeFont):
        return selected_font.font_variant(size=max(8, round(selected_font.size * scale)))
    return selected_font


def _italic_font(selected_font: ImageFont.ImageFont) -> ImageFont.ImageFont:
    if not isinstance(selected_font, ImageFont.FreeTypeFont):
        return selected_font
    _, style = selected_font.getname()
    return font(selected_font.size, bold="bold" in style.lower(), italic=True)


def _script_group(value: str, start: int) -> tuple[str, int]:
    if start >= len(value):
        return "", start
    if value[start] != "{":
        return value[start], start + 1

    depth = 1
    index = start + 1
    while index < len(value) and depth:
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
        index += 1
    return value[start + 1 : index - 1], index


def _math_runs(
    draw: ImageDraw.ImageDraw,
    value: str,
    selected_font: ImageFont.ImageFont,
) -> tuple[list[tuple[str, ImageFont.ImageFont, float, float]], float]:
    """Lay out a compact TeX-like expression with real scripts."""

    runs: list[tuple[str, ImageFont.ImageFont, float, float]] = []
    x = 0.0
    index = 0
    while index < len(value):
        start = index
        while index < len(value) and value[index] not in "^_{}":
            index += 1
        if index > start:
            text = value[start:index].replace("-", "\N{EN DASH}")
            run_start = 0
            while run_start < len(text):
                alphabetic = text[run_start].isalpha()
                run_end = run_start + 1
                while run_end < len(text) and text[run_end].isalpha() == alphabetic:
                    run_end += 1
                run_text = text[run_start:run_end]
                run_font = _italic_font(selected_font) if alphabetic else selected_font
                runs.append((run_text, run_font, x, 0.0))
                x += draw.textlength(run_text, font=run_font)
                run_start = run_end
        if index >= len(value):
            break
        if value[index] in "{}":
            index += 1
            continue

        script_x = x
        script_width = 0.0
        while index < len(value) and value[index] in "^_":
            operator = value[index]
            content, index = _script_group(value, index + 1)
            script_font = _scaled_font(selected_font, 0.64)
            child_runs, child_width = _math_runs(draw, content, script_font)
            baseline_offset = (
                -0.52 * getattr(selected_font, "size", 20)
                if operator == "^"
                else 0.32 * getattr(selected_font, "size", 20)
            )
            runs.extend(
                (text, run_font, script_x + run_x, run_y + baseline_offset)
                for text, run_font, run_x, run_y in child_runs
            )
            script_width = max(script_width, child_width)
        x = script_x + script_width
    return runs, x


def _math_layout(
    draw: ImageDraw.ImageDraw,
    value: str,
    selected_font: ImageFont.ImageFont,
) -> tuple[list[tuple[str, ImageFont.ImageFont, float, float]], float, float, float]:
    runs, width = _math_runs(draw, value, selected_font)
    bounds = [
        draw.textbbox((run_x, run_y), text, font=run_font, anchor="ls")
        for text, run_font, run_x, run_y in runs
        if text
    ]
    if not bounds:
        return runs, width, 0.0, 0.0
    return runs, width, min(item[1] for item in bounds), max(item[3] for item in bounds)


def _math_size(
    draw: ImageDraw.ImageDraw,
    value: str,
    selected_font: ImageFont.ImageFont,
) -> tuple[float, float]:
    _, width, top, bottom = _math_layout(draw, value, selected_font)
    return width, bottom - top


def _draw_math(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    value: str,
    *,
    fill: str,
    selected_font: ImageFont.ImageFont,
) -> None:
    runs, _, top, _ = _math_layout(draw, value, selected_font)
    baseline = position[1] - top
    for text, run_font, run_x, run_y in runs:
        draw.text(
            (position[0] + run_x, baseline + run_y),
            text,
            fill=fill,
            font=run_font,
            anchor="ls",
        )


def _line_size(
    draw: ImageDraw.ImageDraw,
    value: str,
    selected_font: ImageFont.ImageFont,
) -> tuple[float, float]:
    if value.startswith("$") and value.endswith("$") and value.count("$") == 2:
        return _math_size(draw, value[1:-1], selected_font)
    if "$" in value:
        segments = value.split("$")
        sizes = [
            _math_size(draw, segment, selected_font)
            if index % 2
            else _plain_text_size(draw, segment, selected_font)
            for index, segment in enumerate(segments)
        ]
        return sum(width for width, _ in sizes), max(height for _, height in sizes)
    return _plain_text_size(draw, value, selected_font)


def _plain_text_size(
    draw: ImageDraw.ImageDraw,
    value: str,
    selected_font: ImageFont.ImageFont,
) -> tuple[float, float]:
    bounds = draw.textbbox((0, 0), value, font=selected_font)
    return bounds[2] - bounds[0], bounds[3] - bounds[1]


def _draw_line(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    value: str,
    *,
    fill: str,
    selected_font: ImageFont.ImageFont,
) -> None:
    if value.startswith("$") and value.endswith("$") and value.count("$") == 2:
        _draw_math(
            draw,
            position,
            value[1:-1],
            fill=fill,
            selected_font=selected_font,
        )
    elif "$" in value:
        segments = value.split("$")
        sizes = [
            _math_size(draw, segment, selected_font)
            if index % 2
            else _plain_text_size(draw, segment, selected_font)
            for index, segment in enumerate(segments)
        ]
        line_height = max(height for _, height in sizes)
        x = position[0]
        for index, (segment, (segment_width, segment_height)) in enumerate(
            zip(segments, sizes, strict=True)
        ):
            y = position[1] + (line_height - segment_height) / 2
            if index % 2:
                _draw_math(
                    draw,
                    (x, y),
                    segment,
                    fill=fill,
                    selected_font=selected_font,
                )
            else:
                bounds = draw.textbbox((0, 0), segment, font=selected_font)
                draw.text(
                    (x, y - bounds[1]),
                    segment,
                    fill=fill,
                    font=selected_font,
                )
            x += segment_width
    else:
        draw.text(position, value, fill=fill, font=selected_font)


def _draw_fraction(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    prefix: str,
    numerator: str,
    denominator: str,
    *,
    selected_font: ImageFont.ImageFont,
) -> None:
    prefix_width, prefix_height = _math_size(draw, prefix, selected_font)
    numerator_width, numerator_height = _math_size(draw, numerator, selected_font)
    denominator_width, denominator_height = _math_size(draw, denominator, selected_font)
    fraction_width = max(numerator_width, denominator_width) + 20
    fraction_height = numerator_height + denominator_height + 18
    x, y = position
    _draw_math(
        draw,
        (x, y + (fraction_height - prefix_height) / 2),
        prefix,
        fill="black",
        selected_font=selected_font,
    )
    fraction_x = x + prefix_width + 12
    _draw_math(
        draw,
        (fraction_x + (fraction_width - numerator_width) / 2, y),
        numerator,
        fill="black",
        selected_font=selected_font,
    )
    rule_y = y + numerator_height + 7
    draw.line((fraction_x, rule_y, fraction_x + fraction_width, rule_y), fill="black", width=2)
    _draw_math(
        draw,
        (fraction_x + (fraction_width - denominator_width) / 2, rule_y + 7),
        denominator,
        fill="black",
        selected_font=selected_font,
    )


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    draw.line((70, 116, WIDTH - 70, 116), fill=GOLD, width=5)
    draw.text((70, 34), title, fill=INK, font=F_TITLE)
    draw.text((72, 90), subtitle, fill=MUTED, font=F_SUBTITLE)
    return image, draw


def save(image: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    background = Image.new(image.mode, image.size, WHITE)
    content_bounds = ImageChops.difference(image, background).getbbox()
    if content_bounds is not None:
        bottom = min(image.height, content_bounds[3] + 24)
        image = image.crop((0, 0, image.width, bottom))
    image.save(OUT / name, optimize=True)


def _centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    fill: str = INK,
    selected_font: ImageFont.ImageFont = F_NODE,
) -> None:
    lines = text.split("\n")
    sizes = [_line_size(draw, line, selected_font) for line in lines]
    heights = [item[1] for item in sizes]
    total_height = sum(heights) + 8 * (len(lines) - 1)
    y = (box[1] + box[3] - total_height) / 2
    for line, (line_width, _), line_height in zip(lines, sizes, heights, strict=True):
        x = (box[0] + box[2] - line_width) / 2
        _draw_line(draw, (x, y), line, fill=fill, selected_font=selected_font)
        y += line_height + 8


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    width: int = 4,
) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 15
    left = (
        end[0] - size * math.cos(angle - 0.5),
        end[1] - size * math.sin(angle - 0.5),
    )
    right = (
        end[0] - size * math.cos(angle + 0.5),
        end[1] - size * math.sin(angle + 0.5),
    )
    draw.polygon((end, left, right), fill=color)


def _node(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    accent: str,
) -> None:
    draw.rounded_rectangle(box, radius=22, fill=CREAM, outline=accent, width=5)
    _centered_text(draw, box, label)


def _edge_points(
    source: tuple[int, int, int, int], target: tuple[int, int, int, int]
) -> tuple[tuple[float, float], tuple[float, float]]:
    source_center = ((source[0] + source[2]) / 2, (source[1] + source[3]) / 2)
    target_center = ((target[0] + target[2]) / 2, (target[1] + target[3]) / 2)
    if abs(target_center[0] - source_center[0]) >= abs(target_center[1] - source_center[1]):
        if target_center[0] >= source_center[0]:
            return (source[2], source_center[1]), (target[0], target_center[1])
        return (source[0], source_center[1]), (target[2], target_center[1])
    if target_center[1] >= source_center[1]:
        return (source_center[0], source[3]), (target_center[0], target[1])
    return (source_center[0], source[1]), (target_center[0], target[3])


def diagram(
    name: str,
    title: str,
    subtitle: str,
    nodes: dict[str, tuple[str, tuple[int, int, int, int], str]],
    edges: list[tuple[str, str, str]],
    feedback_edges: list[tuple[str, str, str]] | None = None,
) -> None:
    image, draw = canvas(title, subtitle)
    for source_name, target_name, label in edges:
        start, end = _edge_points(nodes[source_name][1], nodes[target_name][1])
        _arrow(draw, start, end, color=MUTED)
        if label:
            x = (start[0] + end[0]) / 2
            y = (start[1] + end[1]) / 2 - 25
            label_width, _ = _line_size(draw, label, F_SMALL)
            _draw_line(
                draw,
                (x - label_width / 2, y),
                label,
                fill=MUTED,
                selected_font=F_SMALL,
            )
    for source_name, target_name, label in feedback_edges or []:
        source = nodes[source_name][1]
        target = nodes[target_name][1]
        start = ((source[0] + source[2]) / 2, source[3])
        end = ((target[0] + target[2]) / 2, target[3])
        loop_y = max(source[3], target[3]) + 110
        points = [start, (start[0], loop_y), (end[0], loop_y), end]
        draw.line(points, fill=RUST, width=4, joint="curve")
        _arrow(draw, points[-2], points[-1], color=RUST)
        label_width, _ = _line_size(draw, label, F_SMALL)
        _draw_line(
            draw,
            ((start[0] + end[0] - label_width) / 2, loop_y + 10),
            label,
            fill=RUST,
            selected_font=F_SMALL,
        )
    for label, box, accent in nodes.values():
        _node(draw, box, label, accent)
    save(image, name)


def _generate_acoustic_diagram() -> None:
    diagram(
        "01_acoustic_event_anatomy.png",
        "Anatomy of a Reverberant Event",
        "Direct sound establishes location; reflections establish enclosure and scale.",
        {
            "source": ("Source", (80, 365, 300, 495), BLUE),
            "direct": ("Direct\npath", (430, 190, 670, 320), TEAL),
            "early": ("Early\nreflections", (430, 365, 670, 495), GOLD),
            "late": ("Late diffuse\nfield", (430, 540, 670, 670), RUST),
            "listener": ("Listener /\nmicrophone", (890, 365, 1160, 495), INK),
            "meaning": ("Distance\nsize\nmaterial", (1270, 365, 1510, 495), TEAL),
        },
        [
            ("source", "direct", "milliseconds"),
            ("source", "early", "10–80 ms"),
            ("source", "late", "dense tail"),
            ("direct", "listener", ""),
            ("early", "listener", ""),
            ("late", "listener", ""),
            ("listener", "meaning", "perception"),
        ],
    )


def _flow_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (2600, 900), WHITE)
    return image, ImageDraw.Draw(image)


def _flow_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    lines: tuple[str, ...],
) -> None:
    draw.rectangle(box, fill=WHITE, outline="black", width=6)
    fonts = (F_FLOW_BOLD, *(F_FLOW_SMALL for _ in lines[1:]))
    sizes = [_line_size(draw, line, selected) for line, selected in zip(lines, fonts, strict=True)]
    heights = [item[1] for item in sizes]
    total_height = sum(heights) + 11 * (len(lines) - 1)
    y = (box[1] + box[3] - total_height) / 2
    for line, selected, (line_width, _), line_height in zip(
        lines, fonts, sizes, heights, strict=True
    ):
        _draw_line(
            draw,
            ((box[0] + box[2] - line_width) / 2, y),
            line,
            fill="black",
            selected_font=selected,
        )
        y += line_height + 11


def _flow_sum(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int = 42) -> None:
    x, y = center
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius), fill=WHITE, outline="black", width=5
    )
    text_box = draw.textbbox((0, 0), "+", font=F_FLOW_BOLD)
    draw.text(
        (x - (text_box[2] - text_box[0]) / 2, y - (text_box[3] - text_box[1]) / 2 - 4),
        "+",
        fill="black",
        font=F_FLOW_BOLD,
    )


def _flow_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    label: str = "",
    *,
    label_offset: tuple[int, int] = (0, -42),
) -> None:
    _arrow(draw, start, end, color="black", width=5)
    if label:
        label_width, _ = _line_size(draw, label, F_FLOW_SMALL)
        x = (start[0] + end[0] - label_width) / 2 + label_offset[0]
        y = (start[1] + end[1]) / 2 + label_offset[1]
        _draw_line(draw, (x, y), label, fill="black", selected_font=F_FLOW_SMALL)


def _flow_note(draw: ImageDraw.ImageDraw, value: str) -> None:
    _draw_line(draw, (70, 840), value, fill=MUTED, selected_font=F_FLOW_SMALL)


def flowgraph_feedback_comb() -> None:
    image, draw = _flow_canvas()
    _flow_sum(draw, (550, 380))
    _flow_box(
        draw,
        (900, 280, 1330, 480),
        ("$z^{-M}$", "$M$\N{EN DASH}sample delay"),
    )
    _flow_box(draw, (1160, 640, 1450, 780), ("$g$", "loop gain"))
    _flow_arrow(draw, (80, 380), (505, 380), "$x[n]$")
    _flow_arrow(draw, (595, 380), (900, 380), "$w[n]$")
    _flow_arrow(draw, (1330, 380), (1980, 380), "$y[n]$")
    draw.line((1540, 380, 1540, 710, 1450, 710), fill="black", width=5)
    _flow_arrow(draw, (1160, 710), (550, 425), "$gy[n]$", label_offset=(0, 28))
    _draw_fraction(draw, (2010, 305), "H(z)=", "1", "1-gz^{-M}", selected_font=F_FLOW)
    _flow_note(
        draw,
        "Feedback comb: each loop traversal adds $M$ samples and multiplies amplitude by $g$.",
    )
    save(image, "23_feedback_comb_flowgraph.png")


def flowgraph_schroeder_allpass() -> None:
    image, draw = _flow_canvas()
    _flow_sum(draw, (470, 330))
    _flow_box(
        draw,
        (760, 230, 1180, 430),
        ("$z^{-M}$", "$M$\N{EN DASH}sample delay"),
    )
    _flow_box(draw, (770, 600, 1040, 740), ("$-g$", "feedforward"))
    _flow_box(draw, (1260, 600, 1530, 740), ("$g$", "feedback"))
    _flow_sum(draw, (1770, 330))
    _flow_arrow(draw, (70, 330), (425, 330), "$x[n]$")
    _flow_arrow(draw, (515, 330), (760, 330), "$w[n]$")
    _flow_arrow(draw, (1180, 330), (1725, 330), "$w[n-M]$")
    _flow_arrow(draw, (1815, 330), (2200, 330), "$y[n]$")
    draw.line((650, 330, 650, 670, 770, 670), fill="black", width=5)
    _flow_arrow(draw, (1040, 670), (1770, 375), "$-gw[n]$", label_offset=(0, 28))
    draw.line((1380, 330, 1380, 600), fill="black", width=5)
    _flow_arrow(draw, (1260, 670), (470, 375), "$gw[n-M]$", label_offset=(0, 28))
    _draw_fraction(
        draw,
        (2200, 270),
        "H(z)=",
        "-g+z^{-M}",
        "1-gz^{-M}",
        selected_font=F_FLOW_SMALL,
    )
    _flow_note(
        draw, "Schroeder allpass: matched feedforward and feedback paths preserve ideal magnitude."
    )
    save(image, "24_schroeder_allpass_flowgraph.png")


def flowgraph_parameterized_schroeder() -> None:
    image, draw = _flow_canvas()
    allpasses = [
        ((150, 320, 470, 500), ("AP", "$N$=337, $g$=0.70")),
        ((540, 320, 860, 500), ("AP", "$N$=113, $g$=0.70")),
        ((930, 320, 1250, 500), ("AP", "$N$=41, $g$=0.70")),
    ]
    for box, lines in allpasses:
        _flow_box(draw, box, lines)
    _flow_arrow(draw, (20, 410), (150, 410), "RevIn")
    _flow_arrow(draw, (470, 410), (540, 410))
    _flow_arrow(draw, (860, 410), (930, 410))
    bus_x = 1310
    draw.line((1250, 410, bus_x, 410), fill="black", width=5)
    draw.line((bus_x, 115, bus_x, 745), fill="black", width=5)

    delays = (1499, 1601, 1877, 2137)
    gains = tuple(10 ** (-3 * delay / (48000 * 2.4)) for delay in delays)
    row_centers = (145, 330, 515, 700)
    for index, (delay, gain, y) in enumerate(zip(delays, gains, row_centers, strict=True), 1):
        _flow_arrow(draw, (bus_x, y), (1420, y))
        _flow_box(
            draw,
            (1420, y - 65, 1850, y + 65),
            ("FBCF", f"$N$={delay}, $g$={gain:.3f}"),
        )
        _flow_arrow(draw, (1850, y), (2010, y), f"$x_{index}$", label_offset=(0, -38))

    _flow_box(draw, (2010, 75, 2250, 770), ("$H_{4/2}$", "output matrix"))
    for index, y in enumerate(row_centers):
        _flow_arrow(draw, (2250, y), (2560, y), f"Out{chr(65 + index)}")
    _flow_note(
        draw,
        "Illustrative 48 kHz, $T_{60}$=2.4 s design; values demonstrate notation "
        "and are not verbx defaults.",
    )
    save(image, "25_parameterized_schroeder_flowgraph.png")


def flowgraph_expanded_fdn() -> None:
    image, draw = _flow_canvas()
    _flow_box(draw, (80, 340, 290, 500), ("$B$", "input projection"))
    _flow_sum(draw, (440, 420))
    _flow_arrow(draw, (10, 420), (80, 420), "$x[n]$")
    _flow_arrow(draw, (290, 420), (395, 420), "$u[n]$")
    delay_rows = (150, 330, 510, 690)
    draw.line((485, 420, 610, 420), fill="black", width=5)
    draw.line((610, 150, 610, 690), fill="black", width=5)
    for index, y in enumerate(delay_rows, 1):
        _flow_arrow(draw, (610, y), (720, y))
        _flow_box(
            draw,
            (720, y - 58, 1010, y + 58),
            (f"$z^{{-m_{index}}}$", f"delay $m_{index}$"),
        )
        _flow_arrow(draw, (1010, y), (1110, y), f"$s_{index}$", label_offset=(0, -36))
        _flow_box(draw, (1110, y - 58, 1410, y + 58), (f"$H_{index}(z)$", "loop damping"))
        _flow_arrow(draw, (1410, y), (1530, y))
    draw.line((1530, 150, 1530, 690), fill="black", width=5)
    _flow_box(draw, FDN_GAIN_BOX, ("$G$", "$T_{60}$ gains"))
    _flow_arrow(draw, (1530, 420), (1660, 420), "$N$-vector")
    _flow_box(draw, FDN_MATRIX_BOX, ("$M$", "unitary matrix"))
    _flow_arrow(draw, (1900, 420), (2050, 420), "$G s[n]$")
    draw.line((2290, 420, 2400, 420, 2400, 805, 440, 805, 440, 465), fill="black", width=5)
    _flow_arrow(draw, (1530, 270), (1660, 270))
    _flow_box(draw, FDN_OUTPUT_PROJECTION_BOX, ("$C^{T}$", "output projection"))
    draw.line((1530, 150, 1580, 150, 1580, 85, 1620, 85), fill="black", width=5)
    _flow_arrow(draw, (1940, 85), (2440, 85), "$y[n]$")
    _flow_note(
        draw,
        "Expanded FDN: unequal delays, per-line damping, $T_{60}$ gains, unitary "
        "feedback, and output projection.",
    )
    save(image, "26_expanded_fdn_flowgraph.png")


def flowgraph_multiband_loop_filter() -> None:
    image, draw = _flow_canvas()
    draw.line((250, 420, 500, 420), fill="black", width=5)
    draw.line((500, 150, 500, 690), fill="black", width=5)
    _flow_arrow(draw, (40, 420), (250, 420), "$s_i[n]$")
    bands = [
        (150, "$L_i(z)$", "$g_{low}=10^{-3d_i/T_{60,low}}$"),
        (420, "$M_i(z)$", "$g_{mid}=10^{-3d_i/T_{60,mid}}$"),
        (690, "$H_i(z)$", "$g_{high}=10^{-3d_i/T_{60,high}}$"),
    ]
    for y, filter_name, gain_name in bands:
        _flow_arrow(draw, (500, y), (700, y))
        _flow_box(draw, (700, y - 65, 1050, y + 65), (filter_name, "band filter"))
        _flow_arrow(draw, (1050, y), (1200, y))
        _flow_box(draw, (1200, y - 65, 1800, y + 65), (gain_name,))
        _flow_arrow(draw, (1800, y), (2110, 420))
    _flow_sum(draw, (2155, 420))
    _flow_arrow(draw, (2200, 420), (2550, 420), "conditioned line $i$")
    _flow_note(
        draw,
        "Frequency-dependent loop loss: each crossover band receives its own "
        "calibrated decay gain.",
    )
    save(image, "27_multiband_loop_filter_flowgraph.png")


def flowgraph_stereo_projection() -> None:
    image, draw = _flow_canvas()
    state_rows = tuple(135 + index * 85 for index in range(8))
    draw.line((260, state_rows[0], 260, state_rows[-1]), fill="black", width=5)
    for index, y in enumerate(state_rows, 1):
        _flow_arrow(draw, (40, y), (260, y), f"$s_{index}[n]$")
    _flow_box(
        draw, (560, 120, 1120, 405), ("$C_L^T$", "+ - + - - + - +", "normalized left projection")
    )
    _flow_box(
        draw, (560, 500, 1120, 785), ("$C_R^T$", "+ + - - + - - +", "normalized right projection")
    )
    draw.line((260, 300, 430, 300, 430, 260, 560, 260), fill="black", width=5)
    draw.line((260, 620, 430, 620, 430, 640, 560, 640), fill="black", width=5)
    _flow_arrow(draw, (1120, 260), (1500, 260), "wet L")
    _flow_arrow(draw, (1120, 640), (1500, 640), "wet R")
    _flow_box(draw, (1640, 145, 2050, 375), ("Mix", "dry L + wet L"))
    _flow_box(draw, (1640, 525, 2050, 755), ("Mix", "dry R + wet R"))
    _flow_arrow(draw, (1500, 260), (1640, 260))
    _flow_arrow(draw, (1500, 640), (1640, 640))
    _flow_arrow(draw, (2050, 260), (2500, 260), "Out L")
    _flow_arrow(draw, (2050, 640), (2500, 640), "Out R")
    _flow_note(
        draw,
        "Stereo projection: both channels hear one shared FDN state through "
        "different signed weight vectors.",
    )
    save(image, "28_stereo_projection_flowgraph.png")


def _roots_on_radius(count: int, radius: float, phase: float = 0.0) -> list[complex]:
    return [
        radius * np.exp(1j * (phase + 2.0 * math.pi * index / count))
        for index in range(count)
    ]


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    value: str,
    selected_font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in value.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=selected_font) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def pole_zero_plot(
    name: str,
    title: str,
    subtitle: str,
    poles: list[complex],
    zeros: list[complex],
    notes: tuple[str, ...],
    *,
    zero_at_origin_count: int = 0,
) -> None:
    """Draw a true-aspect unit-circle pole-zero diagram."""

    image, draw = canvas(title, subtitle)
    left, top, size = 95, 175, 660
    right, bottom = left + size, top + size
    center_x = left + size / 2
    center_y = top + size / 2
    scale = size / 2.4

    draw.rectangle((left, top, right, bottom), fill="#fbfaf6", outline=GRID, width=3)
    for value in (-1.0, -0.5, 0.5, 1.0):
        x = center_x + value * scale
        y = center_y - value * scale
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        draw.line((left, y, right, y), fill=GRID, width=2)
    draw.line((left, center_y, right, center_y), fill=MUTED, width=3)
    draw.line((center_x, top, center_x, bottom), fill=MUTED, width=3)
    radius = scale
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        outline=BLUE,
        width=5,
    )

    def map_point(value: complex) -> tuple[float, float]:
        return center_x + value.real * scale, center_y - value.imag * scale

    for value in zeros:
        x, y = map_point(value)
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=WHITE, outline=TEAL, width=4)
    for value in poles:
        x, y = map_point(value)
        draw.line((x - 10, y - 10, x + 10, y + 10), fill=RUST, width=5)
        draw.line((x - 10, y + 10, x + 10, y - 10), fill=RUST, width=5)

    if zero_at_origin_count:
        x, y = map_point(0j)
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=WHITE, outline=TEAL, width=4)
        draw.text((x + 18, y - 18), f"x{zero_at_origin_count}", fill=TEAL, font=F_SMALL)

    draw.text((center_x, bottom + 22), "Real part (unitless)", fill=MUTED, font=F_SMALL, anchor="ma")
    vertical_text = "Imaginary part (unitless)"
    vertical_box = draw.textbbox((0, 0), vertical_text, font=F_SMALL)
    vertical_image = Image.new(
        "RGBA",
        (vertical_box[2] - vertical_box[0] + 12, vertical_box[3] - vertical_box[1] + 12),
        (255, 255, 255, 0),
    )
    vertical_draw = ImageDraw.Draw(vertical_image)
    vertical_draw.text((6, 6), vertical_text, fill=MUTED, font=F_SMALL)
    vertical_image = vertical_image.rotate(90, expand=True)
    image.paste(
        vertical_image,
        (24, round(center_y - vertical_image.height / 2)),
        vertical_image,
    )
    draw.text((center_x + radius - 8, center_y + 12), "unit circle", fill=BLUE, font=F_TINY, anchor="ra")

    text_x = 850
    draw.ellipse((text_x, 224, text_x + 20, 244), fill=WHITE, outline=TEAL, width=4)
    draw.text((text_x + 36, 218), "zero", fill=INK, font=F_SMALL)
    draw.line((text_x, 280, text_x + 20, 300), fill=RUST, width=5)
    draw.line((text_x, 300, text_x + 20, 280), fill=RUST, width=5)
    draw.text((text_x + 36, 274), "pole", fill=INK, font=F_SMALL)
    y = 360
    for note in notes:
        draw.ellipse((text_x, y + 8, text_x + 9, y + 17), fill=GOLD)
        lines = _wrap_text(draw, note, F_SMALL, 620)
        for line in lines:
            draw.text((text_x + 25, y), line, fill=INK, font=F_SMALL)
            y += 27
        y += 22
    save(image, name)


def generate_pole_zero_plots() -> None:
    comb_poles = _roots_on_radius(12, 0.86, math.pi / 12)
    pole_zero_plot(
        "38_feedback_comb_pz.png",
        "Feedback Comb Pole-Zero Map",
        "The delay order sets angular density; loop gain sets modal radius.",
        comb_poles,
        [],
        (
            "Representative order M = 12 and positive loop gain g.",
            "All poles share radius |g|^(1/M); moving them outward lengthens decay.",
            "Polynomial form contributes M coincident zeros at the origin.",
        ),
        zero_at_origin_count=12,
    )

    allpass_poles = _roots_on_radius(8, 0.82, math.pi / 8)
    allpass_zeros = _roots_on_radius(8, 1.0 / 0.82, math.pi / 8)
    pole_zero_plot(
        "39_schroeder_allpass_pz.png",
        "Schroeder Allpass Pole-Zero Map",
        "Reciprocal pole-zero geometry preserves magnitude while rotating phase.",
        allpass_poles,
        allpass_zeros,
        (
            "Representative delay M = 8; each zero is reciprocal to a pole.",
            "Stable poles remain inside the unit circle while zeros may lie outside it.",
            "The phase rotation disperses attacks without imposing an ideal spectral tilt.",
        ),
    )

    schroeder_poles: list[complex] = []
    for count, radius, phase in ((7, 0.79, 0.0), (9, 0.84, 0.10), (11, 0.88, 0.04), (13, 0.91, 0.08)):
        schroeder_poles.extend(_roots_on_radius(count, radius, phase))
    schroeder_zeros = _roots_on_radius(6, 1.08, math.pi / 6)
    pole_zero_plot(
        "40_parameterized_schroeder_pz.png",
        "Parameterized Schroeder Modal Map",
        "Several incommensurate comb rings interleave into a denser modal field.",
        schroeder_poles,
        schroeder_zeros,
        (
            "Reduced-order illustration of four comb sections and serial allpasses.",
            "Unequal delay orders avoid exact modal coincidence and obvious periodicity.",
            "Allpass zeros shown outside the circle are paired with stable interior poles.",
        ),
    )

    fdn_poles: list[complex] = []
    for count, radius, phase in ((8, 0.72, 0.02), (9, 0.80, 0.11), (10, 0.87, 0.06), (11, 0.93, 0.13)):
        fdn_poles.extend(_roots_on_radius(count, radius, phase))
    fdn_zeros = [0.36 * np.exp(1j * angle) for angle in (0.3, 1.2, 2.1, 3.4, 4.8, 5.5)]
    pole_zero_plot(
        "41_expanded_fdn_pz.png",
        "Expanded FDN Modal Projection",
        "Feedback eigenmodes determine poles; input and output projections determine zeros.",
        fdn_poles,
        fdn_zeros,
        (
            "Illustrative reduced-order modal projection, not one fixed verbx preset.",
            "Unitary mixing redistributes energy while per-line damping controls pole radii.",
            "Transmission zeros change when the B or C projection vector changes.",
        ),
    )

    multiband_poles = (
        _roots_on_radius(9, 0.95, 0.0)
        + _roots_on_radius(10, 0.86, 0.08)
        + _roots_on_radius(11, 0.72, 0.04)
    )
    multiband_zeros = _roots_on_radius(8, 0.48, math.pi / 8)
    pole_zero_plot(
        "42_multiband_loop_filter_pz.png",
        "Multiband Decay Pole-Zero Map",
        "Low-, mid-, and high-band losses create distinct modal-radius families.",
        multiband_poles,
        multiband_zeros,
        (
            "Outer poles represent a longer low-band decay; inner poles decay faster.",
            "Crossover and damping filters contribute zeros as well as poles.",
            "A stable design keeps every feedback eigenmode strictly inside the unit circle.",
        ),
    )

    stereo_poles = _roots_on_radius(18, 0.89, 0.04)
    stereo_zeros = [0.62 * np.exp(1j * angle) for angle in (0.2, 0.9, 1.8, 2.8, 3.7, 4.5, 5.6)]
    pole_zero_plot(
        "43_stereo_projection_pz.png",
        "Stereo Projection Pole-Zero Map",
        "Left and right outputs share poles but acquire different transmission zeros.",
        stereo_poles,
        stereo_zeros,
        (
            "The shared FDN state gives both channels the same internal modal poles.",
            "Signed output vectors C_L and C_R select different modal combinations.",
            "Decorrelated zeros change channel color without creating two unrelated rooms.",
        ),
    )


def _generate_detailed_schroeder_diagram() -> None:
    image, draw = canvas(
        "Classic Schroeder Reverberator",
        "Four decay-calibrated comb loops feed three serial diffusion stages.",
    )
    input_box = (35, 400, 165, 520)
    split_box = (210, 400, 350, 520)
    sum_box = (875, 400, 975, 520)
    allpass_boxes = (
        (1015, 385, 1155, 535),
        (1190, 385, 1330, 535),
        (1365, 385, 1505, 535),
    )
    comb_specs = (
        ("FBCF 1\n$M$=1499 samples\n$g$=0.914", 245),
        ("FBCF 2\n$M$=1601 samples\n$g$=0.908", 385),
        ("FBCF 3\n$M$=1877 samples\n$g$=0.894", 535),
        ("FBCF 4\n$M$=2137 samples\n$g$=0.880", 675),
    )
    fan_out_x = 410
    sum_bus_x = 835

    draw.text((465, 157), "PARALLEL DECAY BANK", fill=GOLD, font=F_SMALL)
    draw.text((1090, 318), "SERIAL DIFFUSION", fill=TEAL, font=F_SMALL)
    _arrow(draw, (input_box[2], 460), (split_box[0], 460), color=MUTED)
    _arrow(draw, (split_box[2], 460), (fan_out_x, 460), color=MUTED)
    draw.line((fan_out_x, 245, fan_out_x, 675), fill=MUTED, width=4)
    draw.line((sum_bus_x, 245, sum_bus_x, 675), fill=MUTED, width=4)

    for label, y in comb_specs:
        comb_box = (465, y - 56, 780, y + 56)
        _arrow(draw, (fan_out_x, y), (comb_box[0], y), color=MUTED)
        _arrow(draw, (comb_box[2], y), (sum_bus_x, y), color=MUTED)
        _node(draw, comb_box, label, GOLD)

    _arrow(draw, (sum_bus_x, 460), (sum_box[0], 460), color=MUTED)
    _node(draw, input_box, "Input", BLUE)
    _node(draw, split_box, "Fan-out", TEAL)
    _node(draw, sum_box, "Sum", RUST)

    allpass_specs = (
        ("AP 1\n$M$=337\n$g$=0.70", allpass_boxes[0]),
        ("AP 2\n$M$=113\n$g$=0.70", allpass_boxes[1]),
        ("AP 3\n$M$=41\n$g$=0.70", allpass_boxes[2]),
    )
    _arrow(draw, (sum_box[2], 460), (allpass_boxes[0][0], 460), color=MUTED)
    for index, (label, box) in enumerate(allpass_specs):
        _node(draw, box, label, TEAL)
        if index + 1 < len(allpass_specs):
            _arrow(draw, (box[2], 460), (allpass_boxes[index + 1][0], 460), color=MUTED)
    _arrow(draw, (allpass_boxes[-1][2], 460), (1570, 460), color=MUTED)
    draw.text((1508, 420), "Wet out", fill=MUTED, font=F_TINY)
    _draw_line(
        draw,
        (70, 780),
        "Illustrative 48 kHz, $T_{60}$=2.4 s values; $M$ is delay length and $g$ is loop "
        "gain, not a verbx preset.",
        fill=MUTED,
        selected_font=F_SMALL,
    )
    save(image, "02_schroeder_reverberator.png")


def _generate_partitioned_convolution_diagram() -> None:
    diagram(
        "08_partitioned_convolution.png",
        "Partitioned Convolution Reverb",
        "Short front partitions reduce latency while long FFT partitions carry the tail.",
        {
            "input": ("Input\nblocks", (60, 350, 260, 490), BLUE),
            "fft": ("FFT", (340, 350, 520, 490), TEAL),
            "parts": (
                "IR partitions\n$H_0$  $H_1$  ...  $H_K$",
                (610, 300, 900, 540),
                GOLD,
            ),
            "mac": ("Complex multiply\n+ overlap-save", (990, 315, 1260, 525), RUST),
            "ifft": ("IFFT", (1350, 350, 1535, 490), INK),
        },
        [
            ("input", "fft", "$P$ samples"),
            ("fft", "parts", "$X[k]$"),
            ("parts", "mac", "sum"),
            ("mac", "ifft", "wet"),
        ],
    )


def _generate_remaining_diagrams() -> None:
    _generate_detailed_schroeder_diagram()
    diagram(
        "03_feedback_comb_filter.png",
        "Feedback Comb Filter",
        "A delayed, attenuated copy returns to the input and creates a modal series.",
        {
            "input": ("$x[n]$", (100, 350, 280, 470), BLUE),
            "sum": ("+", (390, 350, 550, 470), TEAL),
            "delay": ("Delay\n$z^{-M}$", (700, 350, 940, 470), GOLD),
            "output": ("$y[n]$", (1110, 350, 1290, 470), INK),
            "gain": ("Feedback gain\n$g$", (700, 640, 940, 760), RUST),
        },
        [("input", "sum", ""), ("sum", "delay", ""), ("delay", "output", "")],
        [("delay", "gain", "loop"), ("gain", "sum", "$g y[n-M]$")],
    )
    diagram(
        "04_schroeder_allpass.png",
        "Schroeder Allpass Diffuser",
        "Feedforward and feedback paths preserve magnitude while rotating phase.",
        {
            "input": ("$x[n]$", (80, 350, 260, 470), BLUE),
            "split": ("Split", (350, 350, 530, 470), TEAL),
            "delay": ("Delay\n$z^{-M}$", (690, 250, 930, 370), GOLD),
            "direct": ("Direct gain\n$-g$", (690, 520, 930, 640), RUST),
            "sum": ("+", (1090, 350, 1250, 470), TEAL),
            "output": ("$y[n]$", (1360, 350, 1530, 470), INK),
        },
        [
            ("input", "split", ""),
            ("split", "delay", "delayed"),
            ("split", "direct", "direct"),
            ("delay", "sum", "+1"),
            ("direct", "sum", "$-g$"),
            ("sum", "output", ""),
        ],
        [("delay", "split", "$g$ feedback")],
    )
    diagram(
        "05_allpass_diffusion_network.png",
        "Serial Allpass Diffusion Network",
        "Incommensurate delays progressively turn discrete echoes into a smooth onset.",
        {
            "input": ("Input\nimpulse", (60, 350, 245, 480), BLUE),
            "a1": ("Allpass 1\n5.1 ms", (320, 350, 555, 480), TEAL),
            "a2": ("Allpass 2\n7.7 ms", (630, 350, 865, 480), GOLD),
            "a3": ("Allpass 3\n12.3 ms", (940, 350, 1175, 480), RUST),
            "output": ("Dense\nexcitation", (1280, 350, 1515, 480), INK),
        },
        [
            ("input", "a1", ""),
            ("a1", "a2", "scatter"),
            ("a2", "a3", "scatter"),
            ("a3", "output", ""),
        ],
    )
    diagram(
        "06_fdn_signal_flow.png",
        "Feedback Delay Network (FDN)",
        "A vector feedback loop separates delay distribution, decay, and mixing topology.",
        {
            "input": ("Input\nprojection $B$", (45, 350, 275, 490), BLUE),
            "sum": ("Vector sum", (350, 350, 570, 490), TEAL),
            "delays": ("Delay bank $D(z)$\n$N$ unequal lines", (650, 315, 930, 525), GOLD),
            "filters": ("Damping +\n$T_{60}$ gains $G$", (1010, 315, 1260, 525), RUST),
            "output": ("Output\nprojection $C$", (1340, 350, 1560, 490), INK),
            "matrix": ("Orthonormal feedback matrix $M$", (720, 665, 1190, 790), TEAL),
        },
        [
            ("input", "sum", "$u[n]$"),
            ("sum", "delays", "state"),
            ("delays", "filters", "$N$ lines"),
            ("filters", "output", "wet"),
        ],
        [("filters", "matrix", "mix"), ("matrix", "sum", "feedback")],
    )
    diagram(
        "07_multiband_fdn.png",
        "Frequency-Dependent FDN Decay",
        "Loop filters assign different loss rates to low, middle, and high bands.",
        {
            "state": ("Delay-line\nstate", (70, 350, 290, 490), BLUE),
            "split": ("Crossover\nfilter bank", (380, 350, 620, 490), TEAL),
            "low": ("Low band\n$T_{60}$ = 5.0 s", (730, 180, 990, 300), GOLD),
            "mid": ("Mid band\n$T_{60}$ = 3.0 s", (730, 350, 990, 470), TEAL),
            "high": ("High band\n$T_{60}$ = 1.4 s", (730, 520, 990, 640), RUST),
            "sum": ("Band sum", (1110, 350, 1325, 490), BLUE),
            "matrix": ("Feedback\nmatrix", (1405, 350, 1560, 490), INK),
        },
        [
            ("state", "split", ""),
            ("split", "low", "< 250 Hz"),
            ("split", "mid", "250–4k"),
            ("split", "high", "> 4 kHz"),
            ("low", "sum", ""),
            ("mid", "sum", ""),
            ("high", "sum", ""),
            ("sum", "matrix", ""),
        ],
    )
    _generate_partitioned_convolution_diagram()
    diagram(
        "09_verbx_hybrid_path.png",
        "verbx Algorithmic Signal Path",
        "The engine separates onset, diffusion, late field, spectral design, and safety.",
        {
            "input": ("Input", (35, 350, 190, 480), BLUE),
            "predelay": ("Pre-delay", (245, 350, 440, 480), TEAL),
            "diffuse": ("Allpass\ndiffusion", (495, 330, 720, 500), GOLD),
            "fdn": ("FDN late\nfield", (775, 330, 1000, 500), RUST),
            "creative": ("Shimmer / bloom\nduck / freeze", (1055, 315, 1320, 515), TEAL),
            "output": ("Mix + limiter\nOutput", (1375, 340, 1565, 490), INK),
        },
        [
            ("input", "predelay", ""),
            ("predelay", "diffuse", "onset"),
            ("diffuse", "fdn", "density"),
            ("fdn", "creative", "tail"),
            ("creative", "output", "safe"),
        ],
    )
    diagram(
        "10_musical_send_return.png",
        "Musical Send/Return Reverb",
        "The dry path carries articulation while the return carries spatial counterpoint.",
        {
            "source": ("Voice /\ninstrument", (60, 350, 290, 490), BLUE),
            "dry": ("Dry fader", (430, 190, 660, 320), TEAL),
            "send": ("Send level", (430, 520, 660, 650), GOLD),
            "reverb": ("verbx\n100% wet", (800, 520, 1070, 650), RUST),
            "return": ("Return fader\n+ automation", (1190, 520, 1480, 650), TEAL),
            "mix": ("Stereo /\nimmersive mix", (1190, 190, 1480, 320), INK),
        },
        [
            ("source", "dry", "direct"),
            ("source", "send", "aux"),
            ("send", "reverb", "wet input"),
            ("reverb", "return", "tail"),
            ("dry", "mix", "articulation"),
            ("return", "mix", "space"),
        ],
    )


def _generate_dsp_extension_diagrams() -> None:
    diagram(
        "29_modulated_delay_control.png",
        "Modulated Delay: Audio and Control Paths",
        "Parameter smoothing and fractional interpolation keep moving delays continuous.",
        {
            "input": ("Audio\ninput", (55, 480, 245, 610), BLUE),
            "write": ("Delay-buffer\nwrite", (360, 480, 620, 610), TEAL),
            "read": ("Fractional\nread", (790, 480, 1050, 610), GOLD),
            "output": ("Audio\noutput", (1300, 480, 1510, 610), INK),
            "control": ("Automation /\nLFO target", (360, 190, 620, 320), RUST),
            "smooth": ("Control-rate\nsmoother", (790, 190, 1050, 320), TEAL),
        },
        [
            ("input", "write", "samples"),
            ("write", "read", "circular memory"),
            ("read", "output", "interpolated"),
            ("control", "smooth", "target delay"),
            ("smooth", "read", "$d[n]$"),
        ],
    )
    diagram(
        "30_fdn_design_coordinates.png",
        "Five Design Coordinates of an FDN",
        "Timing, coupling, loss, excitation, and observation jointly determine the tail.",
        {
            "state": ("Recursive\nFDN state", (650, 390, 950, 555), RUST),
            "input": ("Input\nprojection $B$", (75, 405, 345, 540), BLUE),
            "delays": ("Unequal delays\n$m_1 … m_N$", (650, 160, 950, 295), GOLD),
            "matrix": ("Feedback\nmatrix $M$", (1190, 175, 1490, 310), TEAL),
            "loss": ("Loop filters\nand gains $G$", (1190, 655, 1490, 790), RUST),
            "output": ("Output\nprojection $C^T$", (650, 655, 950, 790), INK),
        },
        [
            ("input", "state", "excitation"),
            ("delays", "state", "modal timing"),
            ("matrix", "state", "coupling"),
            ("loss", "state", "decay"),
            ("state", "output", "observation"),
        ],
    )
    diagram(
        "31_energy_decay_measurement.png",
        "From Impulse Response to Decay Metrics",
        "Backward energy integration turns a measured response into fitted decay slopes.",
        {
            "ir": ("Impulse\nresponse $h[n]$", (75, 220, 320, 355), BLUE),
            "square": ("Energy\n$h^2[n]$", (505, 220, 750, 355), TEAL),
            "integrate": ("Backward\nintegration", (930, 220, 1190, 355), GOLD),
            "db": ("Normalize and\nconvert to dB", (1260, 540, 1530, 675), RUST),
            "fit": ("Fit EDT, $T_{20}$,\nand $T_{30}$ slopes", (760, 540, 1040, 675), TEAL),
            "report": ("$T_{60}$, clarity,\nand confidence", (210, 540, 510, 675), INK),
        },
        [
            ("ir", "square", "sample energy"),
            ("square", "integrate", "future energy"),
            ("integrate", "db", "decay curve"),
            ("db", "fit", "valid windows"),
            ("fit", "report", "estimates"),
        ],
    )
    diagram(
        "32_hybrid_early_late_reverb.png",
        "Hybrid Early and Late Reverberation",
        "A measured onset and an algorithmic tail can share one spatial output stage.",
        {
            "source": ("Source", (55, 405, 250, 535), BLUE),
            "split": ("Band-limited\nenergy split", (365, 390, 635, 550), TEAL),
            "early": ("Measured or\nray-traced early IR", (785, 195, 1085, 350), GOLD),
            "late": ("Algorithmic FDN\nlate field", (785, 600, 1085, 755), RUST),
            "join": ("Time and level\ntransition", (1220, 390, 1495, 550), TEAL),
            "decode": ("Stereo / immersive\nprojection", (1220, 680, 1495, 820), INK),
        },
        [
            ("source", "split", "direct"),
            ("split", "early", "geometry"),
            ("split", "late", "diffuse energy"),
            ("early", "join", "early field"),
            ("late", "join", "late field"),
            ("join", "decode", "coherent state"),
        ],
    )
    diagram(
        "33_dsp_validation_loop.png",
        "The Reverb DSP Validation Loop",
        "A trustworthy design closes the loop between targets, measurements, and listening.",
        {
            "design": ("Topology and\nparameter target", (140, 190, 450, 340), BLUE),
            "probe": ("Impulse, burst,\nand music probes", (1120, 190, 1430, 340), GOLD),
            "measure": ("Decay, spectrum,\nlevel, and latency", (1120, 610, 1430, 760), RUST),
            "listen": ("Critical listening\nand failure notes", (140, 610, 450, 760), TEAL),
            "decision": ("Accept, revise,\nor bound", (645, 400, 955, 550), INK),
        },
        [
            ("design", "probe", "render deterministically"),
            ("probe", "measure", "analyze"),
            ("measure", "decision", "compare targets"),
            ("decision", "listen", "audition"),
            ("listen", "design", "revise"),
        ],
    )


def _generate_dereverberation_diagrams() -> None:
    diagram(
        "34_dereverb_inverse_problem.png",
        "Dereverberation as a Regularized Inverse Problem",
        "The observation is known; the dry source, room response, and noise are latent.",
        {
            "source": ("Dry source\n$x[n]$", (90, 175, 330, 315), BLUE),
            "room": ("Room response\n$h[n]$", (665, 175, 925, 315), GOLD),
            "noise": ("Additive noise\n$v[n]$", (1195, 175, 1460, 315), RUST),
            "observation": (
                "Microphone signal\n$y[n] = h[n] * x[n] + v[n]$",
                (615, 430, 995, 585),
                INK,
            ),
            "assumptions": ("Acoustic and source\npriors", (90, 675, 350, 815), TEAL),
            "estimate": ("Regularized estimator\n$x_{est}[n]$", (610, 665, 980, 815), TEAL),
            "residual": ("Residual late field\n$r[n]$", (1195, 675, 1460, 815), RUST),
        },
        [
            ("source", "observation", "convolution"),
            ("room", "observation", "direct + early + late"),
            ("noise", "observation", "addition"),
            ("observation", "estimate", "ill-posed evidence"),
            ("assumptions", "estimate", "regularization"),
            ("estimate", "residual", "unexplained energy"),
        ],
    )
    diagram(
        "35_statistical_dereverb_estimator.png",
        "Statistical Late-Reverberation Suppression",
        "A causal estimator converts delayed spectral evidence into a bounded gain mask.",
        {
            "stft": ("STFT frame\n$Y_{k,l}$", (80, 375, 310, 515), BLUE),
            "delay": ("Delayed history\n$Y_{k,l-D}$", (420, 145, 700, 285), GOLD),
            "late": ("Late-field PSD\n$λ_{r,k,l}$", (825, 145, 1110, 285), RUST),
            "speech": ("Desired PSD\n$λ_{x,k,l}$", (420, 645, 700, 785), TEAL),
            "gain": ("Bounded gain\n$G_{k,l}$", (825, 410, 1110, 550), INK),
            "istft": ("iSTFT + OLA\n$x_{est}[n]$", (1290, 410, 1525, 550), BLUE),
        },
        [
            ("stft", "delay", "past frames"),
            ("delay", "late", "decay model"),
            ("stft", "speech", "current evidence"),
            ("late", "gain", "interference"),
            ("speech", "gain", "desired signal"),
            ("stft", "gain", "$Y_{k,l}$"),
            ("gain", "istft", "$X_{est,k,l}$"),
        ],
    )
    diagram(
        "36_wpe_prediction_loop.png",
        "Weighted Prediction Error Dereverberation",
        "Delayed multichannel history predicts late reverberation without canceling the onset.",
        {
            "stack": ("Microphone STFT\n$y_{t,f}$", (55, 380, 315, 535), BLUE),
            "history": ("Delay $Δ$ and stack\n$y_{t-Δ,f}$", (420, 380, 710, 535), GOLD),
            "predictor": ("Weighted predictor\n$g_f^H$", (825, 380, 1110, 535), RUST),
            "subtract": ("Prediction-error\nresidual $X_{est,t,f}$", (1280, 380, 1570, 535), INK),
            "reference": ("Undelayed reference\n$Y_{t,f}$", (825, 165, 1110, 305), TEAL),
            "variance": ("Source variance\n$λ_{t,f}$", (825, 665, 1110, 805), TEAL),
        },
        [
            ("stack", "history", ""),
            ("history", "predictor", "past frames"),
            ("predictor", "subtract", "late estimate"),
            ("stack", "reference", "current frame"),
            ("reference", "subtract", "reference"),
            ("subtract", "variance", "update power"),
            ("variance", "predictor", "weights"),
        ],
    )
    diagram(
        "37_multichannel_dereverb_stack.png",
        "Multichannel Dereverberation and Spatial Filtering",
        "Temporal prediction and spatial covariance estimation solve complementary problems.",
        {
            "array": ("Microphone array\n$y_1 … y_M$", (80, 385, 320, 535), BLUE),
            "sync": ("Clock, gain, and\ngeometry validation", (430, 150, 730, 300), GOLD),
            "wpe": ("Multichannel WPE\nlate-tail prediction", (430, 620, 730, 770), RUST),
            "cov": ("Spatial covariance\n$R_{xx}$ and $R_{vv}$", (850, 150, 1140, 300), TEAL),
            "beam": ("MVDR / MWF / WPD\nspatial filter", (850, 620, 1140, 770), INK),
            "output": ("Dereverberated\nspatial output", (1300, 385, 1540, 535), BLUE),
        },
        [
            ("array", "sync", "calibrate"),
            ("array", "wpe", "delayed frames"),
            ("sync", "cov", "array model"),
            ("wpe", "beam", "shortened"),
            ("cov", "beam", "spatial statistics"),
            ("beam", "output", "distortion constraint"),
        ],
    )


def generate_high_level_diagrams() -> None:
    _generate_acoustic_diagram()
    _generate_remaining_diagrams()
    _generate_dsp_extension_diagrams()
    _generate_dereverberation_diagrams()


def generate_technical_flowgraphs() -> None:
    flowgraph_feedback_comb()
    flowgraph_schroeder_allpass()
    flowgraph_parameterized_schroeder()
    flowgraph_expanded_fdn()
    flowgraph_multiband_loop_filter()
    flowgraph_stereo_projection()


def _mono_resampled(path: Path, target_rate: int = 16000) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float64")
    mono = np.mean(audio, axis=1)
    if sample_rate != target_rate:
        divisor = math.gcd(sample_rate, target_rate)
        mono = signal.resample_poly(mono, target_rate // divisor, sample_rate // divisor)
        sample_rate = target_rate
    peak = float(np.max(np.abs(mono)))
    if peak > 0:
        mono = mono / peak
    return mono, sample_rate


def _sonogram(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies, times, spectrum = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=512,
        noverlap=448,
        boundary=None,
        padded=False,
    )
    magnitude = np.maximum(np.abs(spectrum), 1e-10)
    decibels = 20.0 * np.log10(magnitude)
    decibels -= float(np.max(decibels))
    return frequencies, times, np.clip(decibels, -80.0, 0.0)


def _colorize(values: np.ndarray) -> np.ndarray:
    stops = np.array(
        [
            [9, 25, 29],
            [18, 52, 49],
            [31, 102, 112],
            [43, 140, 127],
            [213, 168, 75],
            [255, 248, 220],
        ],
        dtype=np.float64,
    )
    normalized = np.clip((values + 80.0) / 80.0, 0.0, 1.0)
    positions = normalized * (len(stops) - 1)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (positions - lower)[..., None]
    return ((1.0 - fraction) * stops[lower] + fraction * stops[upper]).astype(np.uint8)


def _shared_time_limit(paths: list[Path]) -> float:
    """Return one full-duration time extent for every panel in a figure."""

    durations = []
    for path in paths:
        info = sf.info(path)
        durations.append(float(info.frames / info.samplerate))
    return max(durations, default=0.0)


def _sonogram_pixels(
    frequencies: np.ndarray,
    times: np.ndarray,
    decibels: np.ndarray,
    *,
    width: int,
    height: int,
    time_start_s: float,
    time_limit_s: float,
) -> np.ndarray:
    """Sample one sonogram without stretching it beyond its recorded duration."""

    target_times = np.linspace(time_start_s, time_limit_s, width)
    target_frequencies = np.geomspace(50.0, min(8000.0, frequencies[-1]), height)[::-1]
    frequency_indices = np.clip(
        np.searchsorted(frequencies, target_frequencies), 0, len(frequencies) - 1
    )
    pixels = np.full((height, width), -80.0, dtype=np.float64)
    valid = (target_times >= times[0]) & (target_times <= times[-1])
    time_indices = np.clip(
        np.searchsorted(times, target_times[valid]), 0, len(times) - 1
    )
    pixels[:, valid] = decibels[np.ix_(frequency_indices, time_indices)]
    return pixels


def _render_sonogram_panel(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    path: Path,
    label: str,
    box: tuple[int, int, int, int],
    shared_time_limit_s: float | None,
) -> None:
    audio, sample_rate = _mono_resampled(path)
    frequencies, times, decibels = _sonogram(audio, sample_rate)
    x0, y0, x1, y1 = box
    plot_width = x1 - x0
    plot_height = y1 - y0
    time_start_s = 0.0 if shared_time_limit_s is not None else float(times[0])
    time_limit_s = (
        float(shared_time_limit_s) if shared_time_limit_s is not None else float(times[-1])
    )
    pixels = _sonogram_pixels(
        frequencies,
        times,
        decibels,
        width=plot_width,
        height=plot_height,
        time_start_s=time_start_s,
        time_limit_s=time_limit_s,
    )
    rendered = Image.fromarray(_colorize(pixels), mode="RGB")
    image.paste(rendered, (x0, y0))
    draw.rectangle(box, outline=INK, width=3)
    draw.text((x0, y0 - 31), label, fill=INK, font=F_NODE)

    for index in range(5):
        x = x0 + index * plot_width / 4
        draw.line((x, y1, x, y1 + 8), fill=INK, width=2)
        value = time_limit_s * index / 4
        draw.text((x - 18, y1 + 10), f"{value:.1f}", fill=MUTED, font=F_TINY)
    for frequency in (100, 500, 2000, 8000):
        if frequency > frequencies[-1]:
            continue
        ratio = math.log(frequency / 50.0) / math.log(min(8000.0, frequencies[-1]) / 50.0)
        y = y1 - ratio * plot_height
        draw.line((x0 - 8, y, x0, y), fill=INK, width=2)
        draw.text((x0 - 64, y - 10), f"{frequency:g}", fill=MUTED, font=F_TINY)
    x_label = "Time (s)"
    x_label_box = draw.textbbox((0, 0), x_label, font=F_SMALL)
    draw.text(
        ((x0 + x1 - (x_label_box[2] - x_label_box[0])) / 2, y1 + 34),
        x_label,
        fill=MUTED,
        font=F_SMALL,
    )
    draw.text((x0 - 108, (y0 + y1) / 2 - 10), "Hz", fill=MUTED, font=F_SMALL)


def sonogram_figure(
    name: str,
    title: str,
    subtitle: str,
    panels: list[tuple[str, str]],
) -> None:
    image, draw = canvas(title, subtitle)
    panel_count = len(panels)
    if panel_count == 1:
        boxes = [(170, 205, 1440, 790)]
    else:
        boxes = [(170, 195, 1440, 470), (170, 590, 1440, 865)]
    panel_paths = [AUDIO / filename for filename, _label in panels]
    shared_time_limit_s = _shared_time_limit(panel_paths) if panel_count > 1 else None
    for (filename, label), box in zip(panels, boxes, strict=False):
        _render_sonogram_panel(
            draw,
            image,
            AUDIO / filename,
            label,
            box,
            shared_time_limit_s,
        )

    bar_x0, bar_y0, bar_x1, bar_y1 = 1485, boxes[0][1], 1515, boxes[-1][3]
    values = np.linspace(0.0, -80.0, bar_y1 - bar_y0)[:, None]
    colorbar = Image.fromarray(_colorize(values).repeat(bar_x1 - bar_x0, axis=1), mode="RGB")
    image.paste(colorbar, (bar_x0, bar_y0))
    draw.rectangle((bar_x0, bar_y0, bar_x1, bar_y1), outline=INK, width=2)
    draw.text((1452, bar_y0 - 27), "0 dB", fill=MUTED, font=F_TINY)
    draw.text((1440, bar_y1 + 4), "–80 dB", fill=MUTED, font=F_TINY)
    save(image, name)


def generate_sonograms() -> None:
    specifications = [
        (
            "11_click_room_sonogram.png",
            "Impulse to Enclosure",
            "A click exposes onset spacing and the growth of the late field.",
            [("dry_click.wav", "Dry click"), ("dry_click_reverbed.wav", "Reverberated click")],
        ),
        (
            "12_music_hall_sonogram.png",
            "Music in a Hall",
            "A dry harmonic phrase acquires connected spectral tails and phrase overlap.",
            [
                ("realistic_music_dry.wav", "Dry musical phrase"),
                ("realistic_music_hall.wav", "Hall render"),
            ],
        ),
        (
            "13_drums_room_sonogram.png",
            "Drums in a Room",
            "Early reflections add body while the direct attacks remain visible.",
            [("realistic_drums_dry.wav", "Dry drums"), ("realistic_drums_room.wav", "Room render")],
        ),
        (
            "14_cathedral_drums_sonogram.png",
            "Cathedral-Scale Percussion",
            "Long broadband tails turn isolated strikes into overlapping harmonic clouds.",
            [("extreme_cathedral_drums.wav", "8-second Hadamard FDN cathedral")],
        ),
        (
            "15_shimmer_sonogram.png",
            "Octave Shimmer",
            "Pitch-shifted feedback lifts persistent energy into upper spectral bands.",
            [("extreme_shimmer_music.wav", "6-second shimmer study")],
        ),
        (
            "16_freeze_sonogram.png",
            "Frozen Harmonic Field",
            "A captured musical interval becomes a sustained, slowly changing spectrum.",
            [("extreme_frozen_music.wav", "30-second near-infinite tail")],
        ),
        (
            "17_sparse_and_phase_sonograms.png",
            "Sparse Space and Phase Rhythm",
            "Reverb can preserve silence or bind a dense rhythmic lattice.",
            [
                ("feldman_sparse_room.wav", "Sparse-note room study"),
                ("reich_phase_drums.wav", "Phase-drum study"),
            ],
        ),
        (
            "18_drone_time_sonograms.png",
            "Slow Harmonic Time",
            "Long tails reveal whether a drone evolves by register, density, or spectral drift.",
            [
                ("eno_discreet_music.wav", "Layered ambient study"),
                ("radigue_drone.wav", "Long-form drone study"),
            ],
        ),
        (
            "19_recirculation_sonograms.png",
            "Recirculation as Form",
            "Repeated returns progressively replace source articulation with room resonance.",
            [
                ("fripp_frippertronics.wav", "Tape-loop-style study"),
                ("lucier_sitting_room.wav", "Room-recirculation study"),
            ],
        ),
        (
            "20_dense_field_sonograms.png",
            "Dense Spatial Fields",
            "Broadband production and deep-listening textures occupy space in different ways.",
            [
                ("mbv_shoegaze.wav", "Dense guitar-field study"),
                ("oliveros_deep_listening.wav", "Deep-listening spatial study"),
            ],
        ),
        (
            "21_speech_room_sonogram.png",
            "Speech Intelligibility and Room Tail",
            "Consonants are short broadband events; vowels form sustained harmonic ridges.",
            [
                ("realistic_speech_dry.wav", "Dry speech"),
                ("realistic_speech_room.wav", "Speech in room"),
            ],
        ),
        (
            "22_plate_speech_sonogram.png",
            "Plate-Like Speech Reverb",
            "A bright, dense tail extends sibilance differently from vowel energy.",
            [("extreme_plate_speech.wav", "Circulant-FDN plate study")],
        ),
    ]
    for specification in specifications:
        sonogram_figure(*specification)


def main() -> int:
    generate_high_level_diagrams()
    generate_technical_flowgraphs()
    generate_pole_zero_plots()
    generate_sonograms()
    print(
        "Wrote 19 diagrams, 6 technical flowgraphs, 6 pole-zero plots, "
        f"and 12 sonograms to {OUT.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
