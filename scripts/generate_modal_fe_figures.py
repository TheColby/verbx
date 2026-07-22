#!/usr/bin/env python3
"""Render the README's finite-element reverb diagrams as PDF-safe PNG assets."""

from __future__ import annotations

from math import exp
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_ITALIC = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"
FONT_BOLD_ITALIC = "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf"


def _font(
    size: int, bold: bool = False, italic: bool = False
) -> ImageFont.FreeTypeFont:
    if bold and italic:
        path = FONT_BOLD_ITALIC
    elif bold:
        path = FONT_BOLD
    elif italic:
        path = FONT_ITALIC
    else:
        path = FONT
    return ImageFont.truetype(path, size)


def _text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    size: int,
    *,
    bold: bool = False,
    italic: bool = False,
    fill: str = "#152338",
) -> float:
    font = _font(size, bold, italic)
    draw.text(xy, text, font=font, fill=fill)
    return draw.textlength(text, font=font)


def _composed_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    runs: tuple[tuple[str, str, bool], ...],
    size: int,
    *,
    fill: str = "#152338",
) -> None:
    """Draw upright prose and italic math with real super/subscript placement."""

    cursor_x, base_y = xy
    for text, position, italic in runs:
        run_size = size if position == "baseline" else round(size * 0.68)
        if position == "superscript":
            run_y = base_y - round(size * 0.28)
        elif position == "subscript":
            run_y = base_y + round(size * 0.42)
        else:
            run_y = base_y
        cursor_x += _text(
            draw,
            (round(cursor_x), run_y),
            text,
            run_size,
            italic=italic,
            fill=fill,
        )


def spring_tank() -> None:
    image = Image.new("RGB", (1920, 600), "#f8fafc")
    draw = ImageDraw.Draw(image)
    _text(draw, (90, 56), "Coupled mass-spring-damper tank", 50, bold=True)
    _text(draw, (90, 112), "Lumped mass nodes and spring elements form a bounded finite-element tank.", 29, fill="#43536a")
    for row, y in enumerate((250, 420), 1):
        _text(draw, (95, y - 18), f"spring {row}", 26)
        start = 270
        for section in range(3):
            x = start + section * 380
            points = [(x, y), (x + 38, y), (x + 55, y - 25), (x + 85, y + 25), (x + 115, y - 25), (x + 145, y + 25), (x + 175, y - 25), (x + 205, y + 25), (x + 222, y), (x + 260, y)]
            draw.line(points, fill="#506a8c", width=6, joint="curve")
            draw.ellipse((x + 245, y - 20, x + 285, y + 20), fill="#34a0a4", outline="#1a5f70", width=4)
            draw.line((x + 265, y + 22, x + 265, y + 80), fill="#e76f51", width=5)
        if row == 1:
            _composed_text(
                draw,
                (270, y - 70),
                (("drive ", "baseline", False), ("e", "baseline", True)),
                25,
            )
            _composed_text(
                draw,
                (1395, y - 70),
                (("pickup ", "baseline", False), ("p", "baseline", True)),
                25,
            )
    draw.line((1500, 330, 1730, 330), fill="#152338", width=6)
    draw.polygon(((1730, 330), (1698, 312), (1698, 348)), fill="#152338")
    _text(draw, (1495, 285), "wet output", 24)
    _composed_text(
        draw,
        (745, 305),
        (
            ("coupling ", "baseline", False),
            ("k", "baseline", True),
            ("c", "subscript", True),
        ),
        25,
        fill="#c44f35",
    )
    image.save(ASSETS / "modal_fe_spring_tank.png", optimize=True)


def plate_grid() -> None:
    image = Image.new("RGB", (1920, 880), "#f8fafc")
    draw = ImageDraw.Draw(image)
    _text(draw, (90, 56), "Structured mass-lumped plate grid", 50, bold=True)
    _text(draw, (90, 112), "Fixed-edge displacement, discrete bending stiffness, optional tension, and interpolated pickup.", 29, fill="#43536a")
    box = (220, 210, 1440, 710)
    draw.rectangle(box, fill="#e6eef5", outline="#152338", width=14)
    for x in range(342, 1440, 122):
        draw.line((x, 210, x, 710), fill="#6d8aa8", width=4)
    for y in range(280, 710, 72):
        draw.line((220, y, 1440, y), fill="#6d8aa8", width=4)
    drive, pickup = (585, 565), (1195, 350)
    labels = (
        (
            drive,
            (0, 90),
            (
                ("drive ", "baseline", False),
                ("e", "baseline", True),
                ("(", "baseline", False),
                ("x", "baseline", True),
                ("d", "subscript", True),
                (", ", "baseline", False),
                ("y", "baseline", True),
                ("d", "subscript", True),
                (")", "baseline", False),
            ),
        ),
        (
            pickup,
            (-80, -90),
            (
                ("pickup ", "baseline", False),
                ("p", "baseline", True),
                ("(", "baseline", False),
                ("x", "baseline", True),
                ("p", "subscript", True),
                (", ", "baseline", False),
                ("y", "baseline", True),
                ("p", "subscript", True),
                (")", "baseline", False),
            ),
        ),
    )
    for (x, y), offset, runs in labels:
        draw.ellipse((x - 23, y - 23, x + 23, y + 23), fill="#34a0a4", outline="#1a5f70", width=4)
        draw.line((x, y, x, y + offset[1]), fill="#e76f51", width=5)
        _composed_text(
            draw,
            (x + offset[0], y + offset[1]),
            runs,
            25,
            fill="#c44f35",
        )
    _composed_text(
        draw,
        (1510, 365),
        (
            ("D L", "baseline", True),
            ("T", "superscript", True),
            (" L", "baseline", True),
        ),
        29,
    )
    _text(draw, (1510, 402), "bending", 26)
    _text(draw, (1510, 515), "T L", 29, italic=True)
    _text(draw, (1510, 552), "tension", 26)
    _composed_text(
        draw,
        (220, 785),
        (
            ("Each interior node has mass ", "baseline", False),
            ("ρ h Δx Δy", "baseline", True),
            ("; the heavy frame is the fixed boundary.", "baseline", False),
        ),
        27,
        fill="#43536a",
    )
    image.save(ASSETS / "modal_fe_plate_grid.png", optimize=True)


def convolution_process() -> None:
    """Show discrete convolution as shift, multiply, and sum."""

    image = Image.new("RGB", (1920, 1180), "#f8fafc")
    draw = ImageDraw.Draw(image)
    _text(draw, (90, 48), "Convolution: shift, multiply, and sum", 50, bold=True)
    _text(
        draw,
        (90, 106),
        "An impulse response weights delayed input samples; each output sample is one accumulated overlap.",
        28,
        fill="#43536a",
    )

    input_signal = [0.0, 0.25, 0.78, 1.0, 0.62, 0.20, 0.0]
    impulse_response = [exp(-0.42 * index) for index in range(7)]
    output = [
        sum(
            input_signal[k] * impulse_response[n - k]
            for k in range(len(input_signal))
            if 0 <= n - k < len(impulse_response)
        )
        for n in range(len(input_signal) + len(impulse_response) - 1)
    ]
    output_peak = max(output)
    output = [value / output_peak for value in output]

    left, right = 180, 1760
    panel_tops = (205, 455, 705, 955)

    def sample_x(index: int, count: int) -> float:
        return left + index * (right - left) / (count - 1)

    def axes(top: int, title_runs: tuple[tuple[str, str, bool], ...]) -> int:
        baseline = top + 132
        draw.line((left, baseline, right, baseline), fill="#506078", width=3)
        draw.line((left, top + 8, left, baseline + 12), fill="#506078", width=3)
        draw.polygon(
            ((right, baseline), (right - 18, baseline - 9), (right - 18, baseline + 9)),
            fill="#506078",
        )
        _composed_text(draw, (left, top - 4), title_runs, 29)
        _text(
            draw,
            (right - 195, baseline + 15),
            "sample index",
            20,
            fill="#43536a",
        )
        _text(draw, (38, top + 52), "normalized amplitude", 19, fill="#43536a")
        return baseline

    def stems(values: list[float], top: int, color: str) -> None:
        baseline = top + 132
        for index, value in enumerate(values):
            x = round(sample_x(index, len(values)))
            y = round(baseline - value * 104)
            draw.line((x, baseline, x, y), fill=color, width=5)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)

    axes(
        panel_tops[0],
        (
            ("Input ", "baseline", False),
            ("x", "baseline", True),
            ("[", "baseline", False),
            ("k", "baseline", True),
            ("]", "baseline", False),
        ),
    )
    stems(input_signal, panel_tops[0], "#167d9a")

    axes(
        panel_tops[1],
        (
            ("Impulse response ", "baseline", False),
            ("h", "baseline", True),
            ("[", "baseline", False),
            ("k", "baseline", True),
            ("]", "baseline", False),
        ),
    )
    stems(impulse_response, panel_tops[1], "#d89b21")

    overlap_top = panel_tops[2]
    overlap_baseline = axes(
        overlap_top,
        (
            ("At output sample ", "baseline", False),
            ("n", "baseline", True),
            (" = 6: products ", "baseline", False),
            ("x", "baseline", True),
            ("[", "baseline", False),
            ("k", "baseline", True),
            ("] ", "baseline", False),
            ("h", "baseline", True),
            ("[", "baseline", False),
            ("n", "baseline", True),
            ("−", "baseline", False),
            ("k", "baseline", True),
            ("]", "baseline", False),
        ),
    )
    n = 6
    products = [input_signal[k] * impulse_response[n - k] for k in range(7)]
    product_peak = max(products)
    for index, value in enumerate(products):
        x = round(sample_x(index, len(products)))
        y = round(overlap_baseline - (value / product_peak) * 104)
        draw.rectangle((x - 26, y, x + 26, overlap_baseline), fill="#c45532")
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#7d2e1d")
    _text(
        draw,
        (1270, overlap_top + 40),
        "sum this overlap",
        22,
        bold=True,
        fill="#c45532",
    )
    draw.line(
        (1480, overlap_top + 62, 1640, overlap_top + 106),
        fill="#c45532",
        width=4,
    )

    axes(
        panel_tops[3],
        (
            ("Output ", "baseline", False),
            ("y", "baseline", True),
            ("[", "baseline", False),
            ("n", "baseline", True),
            ("]", "baseline", False),
        ),
    )
    stems(output, panel_tops[3], "#178d78")
    selected_x = round(sample_x(n, len(output)))
    selected_y = round(panel_tops[3] + 132 - output[n] * 104)
    draw.ellipse(
        (selected_x - 14, selected_y - 14, selected_x + 14, selected_y + 14),
        outline="#c45532",
        width=5,
    )
    _composed_text(
        draw,
        (1180, 1100),
        (
            ("y", "baseline", True),
            ("[", "baseline", False),
            ("n", "baseline", True),
            ("] = Σ", "baseline", False),
            ("k", "subscript", True),
            (" x", "baseline", True),
            ("[", "baseline", False),
            ("k", "baseline", True),
            ("] ", "baseline", False),
            ("h", "baseline", True),
            ("[", "baseline", False),
            ("n", "baseline", True),
            ("−", "baseline", False),
            ("k", "baseline", True),
            ("]", "baseline", False),
        ),
        31,
    )
    image.save(ASSETS / "modal_fe_convolution_process.png", optimize=True)


if __name__ == "__main__":
    convolution_process()
    spring_tank()
    plate_grid()
