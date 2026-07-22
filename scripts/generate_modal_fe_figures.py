#!/usr/bin/env python3
"""Render the README's finite-element reverb diagrams as PDF-safe PNG assets."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int, *, bold: bool = False, fill: str = "#152338") -> None:
    draw.text(xy, text, font=_font(size, bold), fill=fill)


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
            _text(draw, (270, y - 70), "drive e", 25)
            _text(draw, (1395, y - 70), "pickup p", 25)
    draw.line((1500, 330, 1730, 330), fill="#152338", width=6)
    draw.polygon(((1730, 330), (1698, 312), (1698, 348)), fill="#152338")
    _text(draw, (1495, 285), "wet output", 24)
    _text(draw, (745, 305), "coupling kc", 25, fill="#c44f35")
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
    for x, y, label, offset in ((drive[0], drive[1], "drive e(xd, yd)", (0, 90)), (pickup[0], pickup[1], "pickup p(xp, yp)", (-80, -90))):
        draw.ellipse((x - 23, y - 23, x + 23, y + 23), fill="#34a0a4", outline="#1a5f70", width=4)
        draw.line((x, y, x, y + offset[1]), fill="#e76f51", width=5)
        _text(draw, (x + offset[0], y + offset[1]), label, 25, fill="#c44f35")
    _text(draw, (1510, 365), "D L^T L", 29)
    _text(draw, (1510, 402), "bending", 26)
    _text(draw, (1510, 515), "T L", 29)
    _text(draw, (1510, 552), "tension", 26)
    _text(draw, (220, 785), "Each interior node has mass rho h dx dy; the heavy frame is the fixed boundary.", 27, fill="#43536a")
    image.save(ASSETS / "modal_fe_plate_grid.png", optimize=True)


if __name__ == "__main__":
    spring_tank()
    plate_grid()
