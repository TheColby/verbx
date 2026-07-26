#!/usr/bin/env python3
"""Generate figures for the Audio AI and data-augmentation chapter."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "ai_augmentation"
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
GREEN = "#3d8b43"
LINE = "#48645e"
GRID = "#d8ccb5"


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


F_TITLE = font(54, bold=True)
F_SUBTITLE = font(24)
F_HEAD = font(29, bold=True)
F_BODY = font(22)
F_SMALL = font(18)
F_TINY = font(15)
F_LABEL = font(21, bold=True)
F_BIG = font(38, bold=True)


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 145), fill="#efe0c4")
    draw.text((74, 38), title, font=F_TITLE, fill=INK)
    draw.text((78, 108), subtitle, font=F_SUBTITLE, fill=MUTED)
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
    draw.rounded_rectangle(box, radius=22, fill=fill, outline=outline, width=5)
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    bbox = draw.multiline_textbbox((0, 0), label, font=F_LABEL, spacing=5, align="center")
    label_height = bbox[3] - bbox[1]
    draw.multiline_text(
        (cx, cy - (14 if detail else 0)),
        label,
        font=F_LABEL,
        fill=INK,
        anchor="mm",
        align="center",
        spacing=5,
    )
    if detail:
        detail_y = min(box[3] - 26, cy + label_height // 2 + 22)
        draw.text((cx, detail_y), detail, font=F_TINY, fill=MUTED, anchor="mm")


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
    spread = 0.58
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
            ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2 - 14),
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


def figure_evidence_chain() -> None:
    image, draw = canvas(
        "From Source Audio to Auditable Model Evidence",
        "Generation, lineage, training, and evaluation stay connected by machine-readable records.",
    )
    boxes = [
        ((55, 355, 295, 555), "Dry source", BLUE, "licensed + split assigned"),
        ((355, 355, 620, 555), "Manifest", GOLD, "profile + seed + labels"),
        ((680, 355, 970, 555), "verbx render plan", TEAL, "deterministic variants"),
        ((1030, 250, 1315, 430), "Audio corpus", RUST, "dry/wet pairs"),
        ((1030, 520, 1315, 700), "Evidence bundle", VIOLET, "JSONL + QA + hashes"),
        ((1410, 355, 1745, 555), "Model experiment", GREEN, "metrics + failure slices"),
    ]
    for box, label, color, detail in boxes:
        rounded_box(draw, box, label, outline=color, detail=detail)
    arrow(draw, (295, 455), (355, 455))
    arrow(draw, (620, 455), (680, 455))
    arrow(draw, (970, 430), (1030, 340), color=RUST)
    arrow(draw, (970, 480), (1030, 610), color=VIOLET)
    arrow(draw, (1315, 340), (1410, 430), color=RUST)
    arrow(draw, (1315, 610), (1410, 495), color=VIOLET)
    arrow(draw, (1580, 555), (1580, 790), color=GREEN, label="evaluate")
    rounded_box(
        draw,
        (1280, 790, 1745, 925),
        "Held-out rooms and sources",
        outline=BLUE,
        detail="report by severity and domain",
    )
    footer(
        draw,
        "The audio file is not the dataset by itself; the dataset is audio plus identity, split, parameters, and evidence.",
    )
    save(image, "01_dataset_evidence_chain.png")


def figure_split_isolation() -> None:
    image, draw = canvas(
        "Split Before Augmentation",
        "All descendants of one identity remain together; variants never decide their own split.",
    )
    split_specs = [
        ("TRAIN", 90, BLUE, ["speaker 01", "song 04", "room A"]),
        ("VALIDATION", 650, GOLD, ["speaker 07", "song 11", "room C"]),
        ("TEST", 1210, RUST, ["speaker 12", "song 18", "room F"]),
    ]
    for heading, x, color, identities in split_specs:
        draw.rounded_rectangle((x, 215, x + 500, 865), radius=26, outline=color, width=5)
        draw.text((x + 250, 255), heading, font=F_HEAD, fill=color, anchor="mm")
        for row, identity in enumerate(identities):
            y = 335 + row * 165
            rounded_box(
                draw,
                (x + 55, y, x + 255, y + 105),
                identity,
                outline=color,
                detail="source group",
            )
            for variant in range(3):
                vy = y + 5 + variant * 37
                draw.ellipse((x + 330, vy, x + 356, vy + 26), fill=color)
                draw.text(
                    (x + 370, vy + 13),
                    f"variant {variant + 1}",
                    font=F_TINY,
                    fill=INK,
                    anchor="lm",
                )
                arrow(draw, (x + 255, y + 52), (x + 330, vy + 13), color=color, width=3)
    draw.line((610, 205, 610, 885), fill=INK, width=2)
    draw.line((1170, 205, 1170, 885), fill=INK, width=2)
    footer(
        draw,
        "Group by performer, utterance family, song, session, room, or impulse response before rendering descendants.",
    )
    save(image, "02_split_isolation_map.png")


def figure_profile_envelopes() -> None:
    image, draw = canvas(
        "Built-in Profile Envelopes",
        "The three profiles occupy different RT60 and pre-delay regions; ranges are sampling bounds, not measurements.",
    )
    left, top, right, bottom = 190, 230, 1650, 825
    draw.rectangle((left, top, right, bottom), outline=GRID, width=3)
    for seconds in range(0, 7):
        x = left + (right - left) * seconds / 6
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        draw.text((x, bottom + 20), str(seconds), font=F_SMALL, fill=MUTED, anchor="ma")
    for delay in range(0, 81, 10):
        y = bottom - (bottom - top) * delay / 80
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 20, y), str(delay), font=F_SMALL, fill=MUTED, anchor="rm")
    draw.text(((left + right) // 2, 885), "RT60 range (seconds)", font=F_LABEL, fill=INK, anchor="mm")
    draw.text(
        (left, top - 16),
        "Pre-delay range (milliseconds)",
        font=F_LABEL,
        fill=INK,
        anchor="ls",
    )
    profiles = [
        ("ASR", BLUE, (0.18, 3.40, 0.0, 52.0)),
        ("Drums", RUST, (0.22, 3.20, 0.0, 34.0)),
        ("Music", TEAL, (0.70, 6.00, 6.0, 80.0)),
    ]
    for index, (label, color, (x0, x1, y0, y1)) in enumerate(profiles):
        px0 = left + (right - left) * x0 / 6
        px1 = left + (right - left) * x1 / 6
        py0 = bottom - (bottom - top) * y0 / 80
        py1 = bottom - (bottom - top) * y1 / 80
        draw.rounded_rectangle((px0, py1, px1, py0), radius=18, outline=color, width=6)
        draw.rectangle((1280, 180 + index * 45, 1310, 210 + index * 45), fill=color)
        draw.text((1325, 195 + index * 45), label, font=F_SMALL, fill=INK, anchor="lm")
    footer(
        draw,
        "ASR emphasizes shorter rooms; music reaches six seconds; drums preserve a stronger direct component and shorter pre-delay.",
    )
    save(image, "03_profile_envelopes.png")


def figure_supervision_geometry() -> None:
    image, draw = canvas(
        "Supervision Geometry for Audio AI",
        "One rendered signal can support enhancement, estimation, classification, contrastive learning, and control prediction.",
    )
    rounded_box(draw, (70, 385, 310, 555), "Dry audio", outline=BLUE, detail="target or positive view")
    rounded_box(draw, (425, 385, 720, 555), "Reverb renderer", outline=TEAL, detail="known parameter vector")
    rounded_box(draw, (835, 385, 1090, 555), "Wet audio", outline=RUST, detail="model input or view")
    arrow(draw, (310, 470), (425, 470))
    arrow(draw, (720, 470), (835, 470))
    targets = [
        ("Dry waveform", 1235, 205, BLUE),
        ("RT60 / early-late", 1425, 335, GOLD),
        ("Room archetype", 1235, 500, VIOLET),
        ("Matched embedding", 1425, 665, GREEN),
        ("Render controls", 1235, 795, TEAL),
    ]
    for label, x, y, color in targets:
        rounded_box(draw, (x, y, x + 300, y + 115), label, outline=color)
        arrow(draw, (1090, 470), (x, y + 58), color=color)
    draw.text((575, 330), "metadata: seed, profile, parameters", font=F_SMALL, fill=MUTED, anchor="mm")
    footer(
        draw,
        "Choose the target before generating the corpus; a convenient field is not automatically a scientifically valid label.",
    )
    save(image, "04_supervision_geometry.png")


def figure_task_matrix() -> None:
    image, draw = canvas(
        "Task-to-Evidence Matrix",
        "Different Audio AI tasks require different pair structures, labels, and held-out dimensions.",
    )
    rows = ["ASR robustness", "Dereverberation", "Room estimation", "Music tagging", "Neural reverb", "Retrieval"]
    cols = ["Dry pair", "Wet audio", "Render labels", "Analysis metrics", "Held-out room", "Listening test"]
    values = [
        [2, 2, 1, 1, 2, 0],
        [2, 2, 2, 2, 2, 1],
        [0, 2, 2, 2, 2, 1],
        [1, 2, 1, 1, 2, 1],
        [2, 2, 2, 2, 2, 2],
        [1, 2, 1, 1, 2, 2],
    ]
    left, top = 430, 260
    cell_w, cell_h = 205, 95
    for col, label in enumerate(cols):
        x = left + col * cell_w + cell_w // 2
        draw.multiline_text((x, top - 30), label.replace(" ", "\n"), font=F_SMALL, fill=INK, anchor="ms", align="center")
    colors = {0: "#e8e1d2", 1: "#9acbc3", 2: TEAL}
    for row, label in enumerate(rows):
        y = top + row * cell_h
        draw.text((left - 25, y + cell_h // 2), label, font=F_LABEL, fill=INK, anchor="rm")
        for col, value in enumerate(values[row]):
            x = left + col * cell_w
            draw.rectangle((x + 5, y + 5, x + cell_w - 5, y + cell_h - 5), fill=colors[value], outline=PAPER, width=3)
            marker = "required" if value == 2 else "useful" if value == 1 else "optional"
            draw.text((x + cell_w // 2, y + cell_h // 2), marker, font=F_TINY, fill=INK, anchor="mm")
    footer(
        draw,
        "Dark cells are usually required, pale teal cells are useful, and cream cells are task-dependent.",
    )
    save(image, "05_task_evidence_matrix.png")


def figure_corpus_growth() -> None:
    image, draw = canvas(
        "Corpus Growth Is Multiplicative",
        "Rendered-file count grows with source count and variants per source before dry copies or analysis sidecars are added.",
    )
    left, top, right, bottom = 180, 220, 1640, 830
    draw.rectangle((left, top, right, bottom), outline=GRID, width=3)
    source_ticks = [0, 250, 500, 750, 1000]
    for source_count in source_ticks:
        x = left + (right - left) * source_count / 1000
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        draw.text((x, bottom + 20), str(source_count), font=F_SMALL, fill=MUTED, anchor="ma")
    for files in range(0, 8001, 1000):
        y = bottom - (bottom - top) * files / 8000
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 22, y), f"{files:,}", font=F_SMALL, fill=MUTED, anchor="rm")
    draw.text(((left + right) // 2, 900), "Clean sources (files)", font=F_LABEL, fill=INK, anchor="mm")
    draw.text(
        (left, top - 16),
        "Rendered wet outputs (files)",
        font=F_LABEL,
        fill=INK,
        anchor="ls",
    )
    for variants, color in [(2, BLUE), (4, TEAL), (8, RUST)]:
        points = []
        for source_count in range(0, 1001, 50):
            files = source_count * variants
            x = left + (right - left) * source_count / 1000
            y = bottom - (bottom - top) * files / 8000
            points.append((x, y))
        draw.line(points, fill=color, width=6)
        y_end = bottom - (bottom - top) * variants * 1000 / 8000
        draw.text((right - 12, y_end - 12), f"{variants} variants/source", font=F_SMALL, fill=color, anchor="rs")
    footer(
        draw,
        "Storage planning must also include copied dry audio, per-output analysis JSON, JSONL metadata, and checkpoints.",
    )
    save(image, "06_corpus_growth.png")


def figure_quality_funnel() -> None:
    image, draw = canvas(
        "Quality Gates Before Training",
        "A corpus should shrink when evidence is weak; silent acceptance converts render defects into label noise.",
    )
    stages = [
        ("Planned", 100, 1700, BLUE, "manifest expands deterministically"),
        ("Rendered", 200, 1600, TEAL, "no failed jobs or missing files"),
        ("Audio valid", 330, 1470, GOLD, "duration, channels, finite samples"),
        ("Acoustically plausible", 470, 1330, VIOLET, "metrics within declared envelope"),
        ("Split clean", 620, 1180, RUST, "no shared source identity"),
        ("Training ready", 670, 1130, GREEN, "card, hashes, labels, audit sample"),
    ]
    for index, (label, x0, x1, color, detail) in enumerate(stages):
        y0 = 205 + index * 120
        polygon = [(x0, y0), (x1, y0), (x1 - 55, y0 + 88), (x0 + 55, y0 + 88)]
        draw.polygon(polygon, fill=color)
        draw.text(((x0 + x1) // 2, y0 + 34), label, font=F_HEAD, fill=PAPER, anchor="mm")
        draw.text(((x0 + x1) // 2, y0 + 66), detail, font=F_TINY, fill=PAPER, anchor="mm")
    footer(
        draw,
        "Fail-fast generation and explicit exclusions make the final training set smaller, clearer, and easier to reproduce.",
    )
    save(image, "07_quality_gate_funnel.png")


def figure_regeneration_drift() -> None:
    image, draw = canvas(
        "Regeneration Drift Review",
        "Conceptual class-ratio deltas show how a new corpus can move even when every render succeeds.",
    )
    left, top, right, bottom = 210, 240, 1630, 820
    zero_y = (top + bottom) // 2
    draw.rectangle((left, top, right, bottom), outline=GRID, width=3)
    for delta in range(-10, 11, 5):
        y = zero_y - delta * 22
        draw.line((left, y, right, y), fill=INK if delta == 0 else GRID, width=3 if delta == 0 else 2)
        draw.text((left - 22, y), f"{delta:+d}", font=F_SMALL, fill=MUTED, anchor="rm")
    draw.text(
        (left, top - 16),
        "Class-ratio delta (percentage points)",
        font=F_LABEL,
        fill=INK,
        anchor="ls",
    )
    groups = [
        ("Train", [-2, 4, -2]),
        ("Validation", [1, -5, 4]),
        ("Test", [0, 2, -2]),
    ]
    colors = [BLUE, TEAL, RUST]
    labels = ["speech", "music", "effects"]
    for group_index, (group, values) in enumerate(groups):
        center = left + 280 + group_index * 430
        for item_index, value in enumerate(values):
            x0 = center - 105 + item_index * 80
            y = zero_y - value * 22
            draw.rectangle((x0, min(y, zero_y), x0 + 55, max(y, zero_y)), fill=colors[item_index])
            draw.text((x0 + 27, bottom + 18), labels[item_index], font=F_TINY, fill=MUTED, anchor="ma")
        draw.text((center, bottom + 68), group, font=F_LABEL, fill=INK, anchor="ma")
    footer(
        draw,
        "Illustrative values only. verbx reports exact count and ratio deltas from the supplied baseline summary or QA bundle.",
    )
    save(image, "08_regeneration_drift.png")


def figure_evaluation_grid() -> None:
    image, draw = canvas(
        "Evaluation Grid: Domain by Reverberation Severity",
        "One average score can hide the exact room and severity region where a model fails.",
    )
    rows = ["Seen source / seen room", "Unseen source / seen room", "Seen source / unseen room", "Unseen source / unseen room"]
    cols = ["Dry", "Short", "Medium", "Long", "Extreme"]
    left, top = 510, 250
    cell_w, cell_h = 220, 135
    colors = ["#dcebdc", "#cae5dd", "#eadcae", "#e7b98f", "#d98f75"]
    for col, label in enumerate(cols):
        draw.text((left + col * cell_w + cell_w // 2, top - 30), label, font=F_LABEL, fill=INK, anchor="ms")
    for row, label in enumerate(rows):
        y = top + row * cell_h
        draw.text((left - 25, y + cell_h // 2), label, font=F_LABEL, fill=INK, anchor="rm")
        for col in range(len(cols)):
            severity = min(4, col + (1 if row >= 2 else 0))
            x = left + col * cell_w
            draw.rectangle((x + 5, y + 5, x + cell_w - 5, y + cell_h - 5), fill=colors[severity], outline=PAPER, width=3)
            draw.text((x + cell_w // 2, y + cell_h // 2), "report", font=F_SMALL, fill=INK, anchor="mm")
    footer(
        draw,
        "Report each cell separately, then aggregate only after the held-out room and source penalties are visible.",
    )
    save(image, "09_evaluation_grid.png")


def figure_lineage_graph() -> None:
    image, draw = canvas(
        "Reproducibility Lineage",
        "A result should be traceable from model metric back to audio output, render plan, manifest, and source signature.",
    )
    rounded_box(draw, (75, 230, 365, 390), "Source signatures", outline=BLUE, detail="SHA-256 + audio metadata")
    rounded_box(draw, (75, 600, 365, 760), "Manifest payload", outline=GOLD, detail="canonical JSON + seed")
    rounded_box(draw, (535, 410, 845, 590), "Provenance hash", outline=VIOLET, detail="dataset identity")
    rounded_box(draw, (1010, 230, 1305, 390), "Rendered audio", outline=RUST, detail="one file per variant")
    rounded_box(draw, (1010, 600, 1305, 760), "JSONL records", outline=TEAL, detail="labels + resolved config")
    rounded_box(draw, (1450, 410, 1735, 590), "Experiment report", outline=GREEN, detail="code + model + metrics")
    arrow(draw, (365, 310), (535, 470), color=BLUE)
    arrow(draw, (365, 680), (535, 530), color=GOLD)
    arrow(draw, (845, 470), (1010, 310), color=RUST)
    arrow(draw, (845, 530), (1010, 680), color=TEAL)
    arrow(draw, (1305, 310), (1450, 470), color=RUST)
    arrow(draw, (1305, 680), (1450, 530), color=TEAL)
    draw.text((690, 700), "same inputs + same seed", font=F_SMALL, fill=MUTED, anchor="mm")
    draw.text((690, 740), "=> same augmentation plan", font=F_SMALL, fill=MUTED, anchor="mm")
    footer(
        draw,
        "The provenance hash identifies generation inputs; model reproducibility still requires code, environment, and training-state records.",
    )
    save(image, "10_reproducibility_lineage.png")


def figure_ablation_grid() -> None:
    image, draw = canvas(
        "Ablation Design for Reverb Augmentation",
        "Change one evidence-bearing factor at a time and keep the source split, training budget, and evaluation grid fixed.",
    )
    rows = ["No augmentation", "RT60 only", "RT60 + pre-delay", "Full profile", "Full + real IR holdout"]
    cols = ["Decay", "Distance", "Color", "Topology", "Measured RIR"]
    left, top = 520, 235
    cell_w, cell_h = 220, 125
    active = [
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ]
    for col, label in enumerate(cols):
        draw.text((left + col * cell_w + cell_w // 2, top - 30), label, font=F_LABEL, fill=INK, anchor="ms")
    for row, label in enumerate(rows):
        y = top + row * cell_h
        draw.text((left - 25, y + cell_h // 2), label, font=F_LABEL, fill=INK, anchor="rm")
        for col, enabled in enumerate(active[row]):
            x = left + col * cell_w
            color = TEAL if enabled else "#e6dfd0"
            draw.rounded_rectangle((x + 16, y + 16, x + cell_w - 16, y + cell_h - 16), radius=18, fill=color)
            draw.text((x + cell_w // 2, y + cell_h // 2), "included" if enabled else "held out", font=F_SMALL, fill=INK, anchor="mm")
    footer(
        draw,
        "This ladder separates the value of duration, distance cues, coloration, algorithmic diversity, and measured-room transfer.",
    )
    save(image, "11_ablation_design.png")


def figure_application_map() -> None:
    image, draw = canvas(
        "Applications to Audio AI and Machine Learning",
        "The same deterministic acoustic generator supports perception, restoration, understanding, retrieval, and creative control.",
    )
    center = (900, 535)
    nodes = [
        ("ASR and speaker\nrobustness", 235, 240, BLUE, (535, 330)),
        ("Dereverberation\nand enhancement", 700, 205, TEAL, (850, 345)),
        ("Room and scene\nestimation", 1260, 240, VIOLET, (1260, 330)),
        ("Music tagging and\nsource separation", 1370, 650, RUST, (1370, 720)),
        ("Neural reverb and\nparameter control", 750, 785, GREEN, (900, 785)),
        ("Acoustic retrieval\nand forensics", 185, 650, GOLD, (485, 720)),
    ]
    for _label, _x, _y, color, endpoint in nodes:
        arrow(draw, center, endpoint, color=color, width=4)
    draw.ellipse((720, 355, 1080, 715), fill="#efe0c4", outline=GOLD, width=6)
    draw.multiline_text(
        center,
        "verbx\nacoustic variants",
        font=F_BIG,
        fill=INK,
        anchor="mm",
        align="center",
        spacing=8,
    )
    for label, x, y, color, _endpoint in nodes:
        rounded_box(draw, (x, y, x + 300, y + 140), label, outline=color)
    footer(
        draw,
        "The renderer supplies controlled variation; task validity still depends on lawful sources, defensible labels, and held-out evaluation.",
    )
    save(image, "12_audio_ai_application_map.png")


def main() -> None:
    figure_evidence_chain()
    figure_split_isolation()
    figure_profile_envelopes()
    figure_supervision_geometry()
    figure_task_matrix()
    figure_corpus_growth()
    figure_quality_funnel()
    figure_regeneration_drift()
    figure_evaluation_grid()
    figure_lineage_graph()
    figure_ablation_grid()
    figure_application_map()
    print(f"Wrote 12 Audio AI figures to {OUTPUT}")


if __name__ == "__main__":
    main()
