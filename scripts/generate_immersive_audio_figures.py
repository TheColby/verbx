#!/usr/bin/env python3
"""Generate the diagrams for the immersive-audio chapter."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "immersive_audio"
WIDTH = 1800
HEIGHT = 1040

INK = "#073b38"
MUTED = "#526b66"
PAPER = "#fbf8ef"
PANEL = "#f4ecd9"
GOLD = "#dda62b"
TEAL = "#008f83"
BLUE = "#087b9b"
RUST = "#bf5b35"
VIOLET = "#7151a5"
LINE = "#48645e"


def font(size: int, *, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    suffix = (
        "BoldItalic" if bold and italic else "Bold" if bold else "Italic" if italic else "Regular"
    )
    candidates = [
        f"/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/tex-gyre/texgyreschola-{suffix.lower()}.otf",
        f"/Library/Fonts/texgyreschola-{suffix.lower()}.otf",
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Georgia.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font(58, bold=True)
F_SUBTITLE = font(25)
F_HEAD = font(30, bold=True)
F_BODY = font(23)
F_SMALL = font(19)
F_TINY = font(16)
F_LABEL = font(21, bold=True)


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 145), fill="#efe0c4")
    draw.text((74, 37), title, font=F_TITLE, fill=INK)
    draw.text((78, 106), subtitle, font=F_SUBTITLE, fill=MUTED)
    draw.line((74, 144, WIDTH - 74, 144), fill=GOLD, width=6)
    return image, draw


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    *,
    outline: str = TEAL,
    fill: str = PANEL,
    detail: str | None = None,
) -> None:
    draw.rounded_rectangle(box, radius=24, fill=fill, outline=outline, width=5)
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    bbox = draw.multiline_textbbox((0, 0), label, font=F_LABEL, spacing=5, align="center")
    y = cy - (bbox[3] - bbox[1]) // 2 - (16 if detail else 0)
    draw.multiline_text(
        (cx, y), label, font=F_LABEL, fill=INK, anchor="mm", align="center", spacing=5
    )
    if detail:
        draw.text((cx, box[3] - 28), detail, font=F_TINY, fill=MUTED, anchor="ms")


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = LINE,
    width: int = 5,
    label: str | None = None,
) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 17
    spread = 0.6
    points = [
        end,
        (
            end[0] - length * math.cos(angle - spread),
            end[1] - length * math.sin(angle - spread),
        ),
        (
            end[0] - length * math.cos(angle + spread),
            end[1] - length * math.sin(angle + spread),
        ),
    ]
    draw.polygon(points, fill=color)
    if label:
        draw.text(
            ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2 - 13),
            label,
            font=F_SMALL,
            fill=MUTED,
            anchor="ms",
        )


def footer(draw: ImageDraw.ImageDraw, note: str) -> None:
    draw.line((75, HEIGHT - 76, WIDTH - 75, HEIGHT - 76), fill="#d6c8ad", width=2)
    draw.text((WIDTH // 2, HEIGHT - 42), note, font=F_TINY, fill=MUTED, anchor="mm")


def save(image: Image.Image, filename: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT / filename, optimize=True)


def speaker(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, color: str) -> None:
    draw.ellipse((x - 24, y - 24, x + 24, y + 24), fill=color, outline=INK, width=2)
    draw.text((x, y + 35), label, font=F_TINY, fill=INK, anchor="ma")


def draw_layout(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    *,
    four_top: bool,
    heading: str,
) -> None:
    cx, cy = center
    draw.rounded_rectangle(
        (cx - 360, cy - 310, cx + 360, cy + 310), radius=22, outline="#cbbd9f", width=3
    )
    draw.text((cx, cy - 275), heading, font=F_HEAD, fill=INK, anchor="mm")
    draw.text((cx, cy - 235), "screen / front", font=F_SMALL, fill=MUTED, anchor="mm")
    draw.arc((cx - 160, cy - 150, cx + 160, cy + 170), 190, 350, fill="#d3c5aa", width=2)
    draw.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), fill=INK)
    draw.text((cx, cy + 28), "listener", font=F_TINY, fill=MUTED, anchor="ma")
    positions = {
        "L": (-180, -170),
        "C": (0, -195),
        "R": (180, -170),
        "Ls": (-260, 10),
        "Rs": (260, 10),
        "Lrs": (-185, 185),
        "Rrs": (185, 185),
        "LFE": (-305, -190),
    }
    for label, (dx, dy) in positions.items():
        speaker(draw, cx + dx, cy + dy, label, BLUE if label != "LFE" else RUST)
    tops = [("Ltm", -105, 5), ("Rtm", 105, 5)]
    if four_top:
        tops = [("Ltf", -120, -105), ("Rtf", 120, -105), ("Ltr", -120, 115), ("Rtr", 120, 115)]
    for label, dx, dy in tops:
        speaker(draw, cx + dx, cy + dy, label, GOLD)
    draw.text((cx, cy + 255), "ear-level", font=F_TINY, fill=BLUE, anchor="mm")
    draw.text((cx + 110, cy + 255), "height", font=F_TINY, fill=GOLD, anchor="mm")


def figure_layouts() -> None:
    image, draw = canvas(
        "Bed Channels Are Not the Monitor Layout",
        "A 7.1.2 Atmos bed and a 7.1.4 loudspeaker render describe different layers of the system.",
    )
    draw_layout(draw, (465, 540), four_top=False, heading="7.1.2 bed channels")
    draw_layout(draw, (1335, 540), four_top=True, heading="7.1.4 monitor layout")
    arrow(draw, (835, 540), (965, 540), color=VIOLET, label="Renderer maps metadata")
    footer(
        draw,
        "Blue: ear-level channels. Gold: height channels. "
        "Positions are schematic, not installation angles.",
    )
    save(image, "01_bed_vs_monitor_layout.png")


def figure_renderer() -> None:
    image, draw = canvas(
        "Dolby Atmos: Bed, Objects, and Renderer",
        "The renderer combines fixed-channel audio and position-aware object audio "
        "for each listening endpoint.",
    )
    rounded_box(
        draw, (85, 240, 390, 410), "7.1.2 bed", outline=BLUE, detail="fixed channel assignments"
    )
    rounded_box(
        draw,
        (85, 505, 390, 675),
        "Audio objects",
        outline=RUST,
        detail="audio + time-varying metadata",
    )
    rounded_box(
        draw, (85, 770, 390, 910), "Session metadata", outline=GOLD, detail="trim, binaural, timing"
    )
    rounded_box(
        draw,
        (620, 345, 1055, 790),
        "Dolby Atmos\nRenderer",
        outline=VIOLET,
        detail="endpoint-aware spatial rendering",
    )
    for y in (325, 590, 840):
        arrow(draw, (390, y), (620, 520 + (y - 590) // 5), label="input")
    outputs = [
        ("7.1.4 speakers", 1265, 235, BLUE),
        ("5.1 / 7.1", 1425, 405, TEAL),
        ("Stereo", 1265, 610, GOLD),
        ("Binaural", 1425, 790, RUST),
    ]
    for label, x, y, color in outputs:
        rounded_box(draw, (x, y, x + 280, y + 130), label, outline=color)
        arrow(draw, (1055, 570), (x, y + 65), color=color)
    draw.text((837, 300), "up to 128 input paths", font=F_SMALL, fill=MUTED, anchor="mm")
    footer(
        draw,
        "A channel-based WAV alone contains no object trajectories or "
        "endpoint-specific rendering instructions.",
    )
    save(image, "02_atmos_renderer_architecture.png")


def figure_hybrid_reverb() -> None:
    image, draw = canvas(
        "Hybrid Immersive-Reverb Topology",
        "Separate early definition, diffuse envelopment, and authored object effects "
        "instead of treating every return alike.",
    )
    rounded_box(draw, (60, 430, 300, 590), "Dry source", outline=BLUE)
    rounded_box(
        draw,
        (430, 220, 770, 390),
        "Early reflections",
        outline=GOLD,
        detail="localization and room size",
    )
    rounded_box(
        draw,
        (430, 455, 770, 625),
        "Diffuse late field",
        outline=TEAL,
        detail="decorrelated bed return",
    )
    rounded_box(
        draw,
        (430, 690, 770, 860),
        "Designed spot tail",
        outline=RUST,
        detail="freeze, reverse, moving wash",
    )
    rounded_box(draw, (920, 220, 1260, 390), "Front / side bed", outline=GOLD)
    rounded_box(draw, (920, 455, 1260, 625), "7.1.2 bed return", outline=TEAL)
    rounded_box(draw, (920, 690, 1260, 860), "Mono/stereo object", outline=RUST)
    rounded_box(
        draw,
        (1430, 425, 1740, 615),
        "Atmos Renderer",
        outline=VIOLET,
        detail="speaker and binaural renders",
    )
    for y in (305, 540, 775):
        arrow(draw, (300, 510), (430, y), label="send")
        arrow(draw, (770, y), (920, y), label="route")
        arrow(draw, (1260, y), (1430, 520), color=VIOLET)
    draw.text((1088, 416), "preserve anchors", font=F_TINY, fill=MUTED, anchor="mm")
    draw.text((1088, 651), "build envelopment", font=F_TINY, fill=MUTED, anchor="mm")
    draw.text((1088, 886), "reserve motion for intent", font=F_TINY, fill=MUTED, anchor="mm")
    footer(
        draw,
        "Keep the perceptual role of each return explicit; motion is metadata "
        "authoring, not merely multichannel panning.",
    )
    save(image, "03_hybrid_reverb_topology.png")


def figure_monitoring() -> None:
    image, draw = canvas(
        "Immersive Translation and Quality-Control Loop",
        "The master is not proven until spatial intent survives every required "
        "renderer and listening endpoint.",
    )
    rounded_box(
        draw, (70, 390, 370, 610), "Source session", outline=INK, detail="beds, objects, automation"
    )
    rounded_box(
        draw,
        (560, 390, 900, 610),
        "Reference renderer",
        outline=VIOLET,
        detail="same master, multiple re-renders",
    )
    endpoints = [
        ("7.1.4 room", 1110, 205, BLUE),
        ("5.1 room", 1430, 205, TEAL),
        ("Stereo speakers", 1110, 525, GOLD),
        ("Binaural phones", 1430, 525, RUST),
    ]
    arrow(draw, (370, 500), (560, 500), label="master")
    for label, x, y, color in endpoints:
        rounded_box(draw, (x, y, x + 275, y + 165), label, outline=color)
        arrow(draw, (900, 500), (x, y + 82), color=color)
    draw.rounded_rectangle((1080, 780, 1735, 930), radius=20, fill="#fffaf0", outline=INK, width=4)
    draw.text((1408, 815), "Compare", font=F_HEAD, fill=INK, anchor="mm")
    draw.text(
        (1408, 862),
        "localization  |  height  |  envelopment  |  timbre",
        font=F_BODY,
        fill=MUTED,
        anchor="mm",
    )
    draw.text(
        (1408, 900),
        "dialogue/music anchors  |  loudness  |  true peak",
        font=F_BODY,
        fill=MUTED,
        anchor="mm",
    )
    arrow(draw, (1408, 780), (335, 610), color=INK, label="revise mix or metadata")
    footer(
        draw,
        "Binaural and stereo are renderer outputs, not substitutes for checking "
        "the authored immersive master.",
    )
    save(image, "04_translation_qc_loop.png")


def figure_delivery() -> None:
    image, draw = canvas(
        "Where verbx Ends and Atmos Mastering Begins",
        "Audio preparation and spatial-master authoring are connected stages, "
        "but they are not interchangeable file operations.",
    )
    draw.rounded_rectangle((55, 205, 790, 900), radius=28, fill="#edf6f2", outline=TEAL, width=5)
    draw.text((422, 255), "verbx domain", font=F_HEAD, fill=INK, anchor="mm")
    left = [
        ("Discrete WAV stems", "mono, stereo, 5.1, 7.1.2, 7.1.4"),
        ("Ambisonic scenes", "ACN with SN3D/N3D; FuMa for FOA"),
        ("Matrix convolution", "explicit input-to-output IR routing"),
        ("Reports and manifests", "analysis, labels, routing, QC evidence"),
    ]
    for index, (label, detail) in enumerate(left):
        y = 315 + index * 135
        rounded_box(draw, (115, y, 730, y + 100), label, outline=TEAL, detail=detail)
    draw.rounded_rectangle((1010, 205, 1745, 900), radius=28, fill="#f7eff0", outline=RUST, width=5)
    draw.text((1377, 255), "Atmos authoring domain", font=F_HEAD, fill=INK, anchor="mm")
    right = [
        ("Bed and object assignment", "including mono/stereo object paths"),
        ("Position metadata", "automation, size, binaural render mode"),
        ("Dolby Renderer", "speaker, stereo, and binaural re-renders"),
        ("Master deliverable", "ADM BWF, DAMF, or IMF IAB as required"),
    ]
    for index, (label, detail) in enumerate(right):
        y = 315 + index * 135
        rounded_box(draw, (1070, y, 1685, y + 100), label, outline=RUST, detail=detail)
    arrow(
        draw,
        (790, 550),
        (1010, 550),
        color=VIOLET,
        width=7,
        label="DAW import and explicit mapping",
    )
    footer(
        draw,
        "A verbx sidecar can document a handoff, but it is not a Dolby ADM BWF "
        "metadata chunk or a DAMF master.",
    )
    save(image, "05_delivery_boundary.png")


def main() -> None:
    figure_layouts()
    figure_renderer()
    figure_hybrid_reverb()
    figure_monitoring()
    figure_delivery()
    print(f"Generated 5 immersive-audio figures in {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
