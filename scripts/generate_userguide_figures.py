#!/usr/bin/env python3
"""Generate PNG figures used by the verbx user guide."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "userguide_figures"

W, H = 1600, 920
BG = "#f7f2e8"
INK = "#17211f"
MUTED = "#66736f"
GRID = "#d7cdbc"
BLUE = "#1f6f8b"
TEAL = "#249a88"
GOLD = "#c78f2b"
RUST = "#b75135"
PLUM = "#6a4c93"
GREEN = "#4f8a42"
PAPER = "#fffaf0"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/New Century Schoolbook.ttc",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


F_TITLE = font(52, True)
F_SUB = font(28)
F_BODY = font(24)
F_SMALL = font(20)
F_MONO = font(22)
F_PANEL = font(28, True)
F_CHANNEL = font(17, True)
F_TINY = font(15)


def canvas(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 118), fill="#efe2cb")
    d.text((64, 34), title, fill=INK, font=F_TITLE)
    if subtitle:
        d.text((66, 92), subtitle, fill=MUTED, font=F_SMALL)
    return img, d


def save(img: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / name, optimize=True)


def text_center(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill: str = INK, fnt=F_BODY) -> None:
    lines = text.split("\n")
    heights = [d.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    total = sum(heights) + (len(lines) - 1) * 8
    y = (box[1] + box[3] - total) / 2
    for line, height in zip(lines, heights):
        bbox = d.textbbox((0, 0), line, font=fnt)
        x = (box[0] + box[2] - (bbox[2] - bbox[0])) / 2
        d.text((x, y), line, fill=fill, font=fnt)
        y += height + 8


def arrow(d: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color: str = INK, width: int = 5) -> None:
    d.line((a, b), fill=color, width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    size = 18
    p1 = (b[0] - size * math.cos(ang - 0.45), b[1] - size * math.sin(ang - 0.45))
    p2 = (b[0] - size * math.cos(ang + 0.45), b[1] - size * math.sin(ang + 0.45))
    d.polygon((b, p1, p2), fill=color)


def node(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, color: str) -> None:
    d.rounded_rectangle(box, radius=24, fill=PAPER, outline=color, width=5)
    text_center(d, box, label, fill=INK)


def plot_axes(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], xlab: str, ylab: str) -> None:
    x0, y0, x1, y1 = box
    for i in range(6):
        x = x0 + i * (x1 - x0) / 5
        d.line((x, y0, x, y1), fill=GRID, width=1)
    for i in range(5):
        y = y0 + i * (y1 - y0) / 4
        d.line((x0, y, x1, y), fill=GRID, width=1)
    d.line((x0, y1, x1, y1), fill=INK, width=3)
    d.line((x0, y0, x0, y1), fill=INK, width=3)
    axis_labels(d, box, xlab, ylab)


def axis_labels(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], xlab: str, ylab: str) -> None:
    """Draw centered axis labels, rotating the ordinate to preserve plot width."""
    x0, y0, x1, y1 = box
    x_bbox = d.textbbox((0, 0), xlab, font=F_SMALL)
    d.text(((x0 + x1 - (x_bbox[2] - x_bbox[0])) / 2, y1 + 24), xlab, fill=MUTED, font=F_SMALL)

    y_bbox = F_SMALL.getbbox(ylab)
    label = Image.new("L", (y_bbox[2] - y_bbox[0] + 12, y_bbox[3] - y_bbox[1] + 12), 0)
    ImageDraw.Draw(label).text((6 - y_bbox[0], 6 - y_bbox[1]), ylab, fill=255, font=F_SMALL)
    label = label.rotate(90, expand=True)
    d.bitmap((x0 - 72, (y0 + y1 - label.height) / 2), label, fill=MUTED)


def chart_labels(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], xlab: str, ylab: str) -> None:
    """Label non-line charts whose marks already provide their own baseline."""
    x0, y0, x1, y1 = box
    axis_labels(d, (x0, y0, x1, y1 + 42), xlab, ylab)


def line_plot(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], xs, ys, color: str, width: int = 5) -> None:
    x0, y0, x1, y1 = box
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    px = x0 + (xs - xs.min()) / (xs.max() - xs.min()) * (x1 - x0)
    py = y1 - (ys - ys.min()) / (ys.max() - ys.min()) * (y1 - y0)
    d.line(list(zip(px, py)), fill=color, width=width, joint="curve")


def bars(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], vals, labels, colors) -> None:
    x0, y0, x1, y1 = box
    maxv = max(vals)
    n = len(vals)
    gap = 18
    bw = ((x1 - x0) - gap * (n - 1)) / n
    for i, (v, label, color) in enumerate(zip(vals, labels, colors)):
        bx0 = x0 + i * (bw + gap)
        bx1 = bx0 + bw
        by0 = y1 - v / maxv * (y1 - y0)
        d.rounded_rectangle((bx0, by0, bx1, y1), radius=10, fill=color)
        text_center(d, (int(bx0), y1 + 12, int(bx1), y1 + 58), label, fill=INK, fnt=F_SMALL)
        text_center(d, (int(bx0), int(by0) - 42, int(bx1), int(by0) - 4), str(v), fill=INK, fnt=F_SMALL)


def fig_signal_flow() -> None:
    img, d = canvas("End-to-End Render Signal Flow", "The high-level path from input audio to analyzed output.")
    boxes = [(70, 280, 270, 400), (355, 280, 555, 400), (640, 280, 840, 400), (925, 280, 1125, 400), (1210, 280, 1410, 400)]
    labels = ["Input\nWAV/FLAC", "Preflight\n+ config", "Engine\nconv/algo", "Post FX\nlimit/loud", "Output\n+ JSON"]
    colors = [BLUE, TEAL, GOLD, RUST, PLUM]
    for b, label, c in zip(boxes, labels, colors):
        node(d, b, label, c)
    for a, b in zip(boxes[:-1], boxes[1:]):
        arrow(d, (a[2] + 12, (a[1] + a[3]) // 2), (b[0] - 14, (b[1] + b[3]) // 2), MUTED)
    d.rounded_rectangle((180, 560, 1360, 720), radius=28, outline=GRID, width=4, fill="#f3ead9")
    text_center(d, (180, 560, 1360, 720), "Progress bars, deterministic seeds, analysis sidecars, and reproducible CLI options\nwrap the signal path so renders can be audited later.", fill=INK)
    save(img, "01_signal_flow.png")


def fig_realtime_latency() -> None:
    img, d = canvas("Realtime Latency Budget", "Total latency is block + device buffering + algorithm lookahead.")
    box = (180, 230, 1420, 650)
    vals = [2.7, 5.3, 10.7, 21.3, 42.7]
    labels = ["128", "256", "512", "1024", "2048"]
    bars(d, box, vals, labels, [TEAL, BLUE, GOLD, RUST, PLUM])
    chart_labels(d, box, "Audio block size (frames at 48 kHz)", "One-block duration (ms)")
    d.text((180, 790), "Use the smallest stable block size your audio driver can sustain; dereverb lookahead adds on top.", fill=INK, font=F_BODY)
    save(img, "02_realtime_latency.png")


def fig_rt60_curves() -> None:
    img, d = canvas("RT60 Decay Families", "A 60 dB drop defines the nominal reverberation time.")
    box = (190, 190, 1420, 710)
    plot_axes(d, box, "Time after excitation (s)", "Relative decay level (dB)")
    t = np.linspace(0, 8, 300)
    for rt, color in [(1.2, BLUE), (2.5, TEAL), (5.0, GOLD), (8.0, RUST)]:
        y = -60 * t / rt
        y = np.maximum(y, -72)
        px = box[0] + t / 8 * (box[2] - box[0])
        py = box[3] - (y + 72) / 72 * (box[3] - box[1])
        d.line(list(zip(px, py)), fill=color, width=5)
        d.text((box[2] - 170, int(py[min(len(py)-1, int(rt / 8 * 299))]) - 16), f"{rt:g}s", fill=color, font=F_SMALL)
    save(img, "03_rt60_decay_families.png")


def fig_edc_windows() -> None:
    img, d = canvas("Energy Decay Curve Windows", "EDT, T20, and T30 fit different portions of the same EDC.")
    box = (190, 190, 1420, 700)
    plot_axes(d, box, "Normalized decay time (0-1)", "Energy decay level (dB)")
    t = np.linspace(0, 1, 280)
    y = -65 * (t ** 0.92) + 2 * np.sin(18 * t)
    xpix = box[0] + t * (box[2] - box[0])
    ypix = box[3] - (y + 70) / 72 * (box[3] - box[1])
    d.line(list(zip(xpix, ypix)), fill=BLUE, width=6)
    for start, end, label, color in [(0, 10, "EDT", TEAL), (5, 25, "T20", GOLD), (5, 35, "T30", RUST)]:
        yy = box[3] - ((-end) + 70) / 72 * (box[3] - box[1])
        d.line((box[0], yy, box[2], yy), fill=color, width=2)
        d.text((box[2] - 95, yy - 24), label, fill=color, font=F_BODY)
    save(img, "04_edc_fit_windows.png")


def fig_fdn_matrix() -> None:
    img, d = canvas("Feedback Matrix Texture", "Dense orthogonal feedback spreads energy across delay lines.")
    n = 16
    rng = np.random.default_rng(7)
    mat = rng.normal(size=(n, n))
    q, _ = np.linalg.qr(mat)
    x0, y0, cell = 420, 170, 38
    for i in range(n):
        for j in range(n):
            v = abs(q[i, j])
            shade = int(245 - v * 220)
            color = (shade, int(shade * 0.9 + 20), int(255 - v * 110))
            d.rectangle((x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell - 2, y0 + (i + 1) * cell - 2), fill=color)
    d.text((150, 270), "delay lines", fill=MUTED, font=F_BODY)
    d.text((1000, 280), "lighter cells = weak coupling\nblue cells = stronger exchange", fill=INK, font=F_BODY)
    axis_labels(d, (420, 170, 1028, 778), "Destination delay line (index)", "Source delay line (index)")
    d.text((1030, 390), "Color scale: absolute coupling coefficient (0-1)", fill=MUTED, font=F_SMALL)
    save(img, "05_fdn_matrix_heatmap.png")


def fig_window_functions() -> None:
    img, d = canvas("Analysis Window Options", "Different windows trade leakage for transient sharpness.")
    box = (180, 190, 1420, 700)
    plot_axes(d, box, "Normalized sample position (0-1)", "Window amplitude (linear, 0-1)")
    x = np.linspace(0, 1, 512)
    curves = [
        ("hann", 0.5 - 0.5 * np.cos(2 * np.pi * x), BLUE),
        ("blackman", 0.42 - 0.5 * np.cos(2 * np.pi * x) + 0.08 * np.cos(4 * np.pi * x), TEAL),
        ("kaiser", np.kaiser(512, 8), GOLD),
        ("tukey", np.where(x < .2, .5 * (1 + np.cos(np.pi * (x / .2 - 1))), np.where(x > .8, .5 * (1 + np.cos(np.pi * ((x - .8) / .2))), 1)), RUST),
    ]
    for name, y, color in curves:
        px = box[0] + x * (box[2] - box[0])
        py = box[3] - y * (box[3] - box[1])
        d.line(list(zip(px, py)), fill=color, width=4)
    for i, (name, _, color) in enumerate(curves):
        d.text((1030, 220 + i * 44), name, fill=color, font=F_BODY)
    save(img, "06_window_functions.png")


def fig_limiter_curve() -> None:
    img, d = canvas("Limiter Transfer Curves", "Soft knees preserve shape; hard ceilings prioritize safety.")
    box = (210, 190, 1350, 700)
    plot_axes(d, box, "Input level (dBFS)", "Output level (dBFS)")
    x = np.linspace(-36, 6, 400)
    curves = [
        ("hard", np.minimum(x, -1), RUST),
        ("soft", -1 - 10 * np.log10(1 + 10 ** ((-1 - x) / 10)), GOLD),
        ("transparent", np.where(x < -6, x, -6 + (x + 6) * 0.45), TEAL),
    ]
    for name, y, color in curves:
        px = box[0] + (x - x.min()) / (x.max() - x.min()) * (box[2] - box[0])
        py = box[3] - (y - (-36)) / (6 - (-36)) * (box[3] - box[1])
        d.line(list(zip(px, py)), fill=color, width=5)
    d.text((980, 235), "hard ceiling\nsoft knee\ntransparent", fill=INK, font=F_BODY)
    save(img, "07_limiter_transfer.png")


def fig_ducking() -> None:
    img, d = canvas("Reverb Ducking Envelope", "Wet level backs away while the dry source is active.")
    box = (170, 190, 1420, 700)
    plot_axes(d, box, "Time (s)", "Relative signal level (linear, 0-1)")
    t = np.linspace(0, 8, 500)
    dry = 0.15 + 0.85 * ((np.sin(2 * np.pi * t * 0.55) > 0.2).astype(float))
    wet = 1 - 0.55 * dry
    for y, color, label, yoff in [(dry, BLUE, "dry source", 0), (wet, RUST, "ducked wet", 46)]:
        px = box[0] + t / t.max() * (box[2] - box[0])
        py = box[3] - y * (box[3] - box[1])
        d.line(list(zip(px, py)), fill=color, width=5)
        d.text((1100, 230 + yoff), label, fill=color, font=F_BODY)
    save(img, "08_ducking_envelope.png")


def fig_multiband_decay() -> None:
    img, d = canvas("Frequency-Dependent Decay", "Low, mid, and high bands can carry different RT60 targets.")
    box = (170, 190, 1420, 700)
    plot_axes(d, box, "Time after excitation (s)", "Relative band level (linear, 0-1)")
    t = np.linspace(0, 6, 300)
    bands = [("low", 5.5, BLUE), ("mid", 3.3, TEAL), ("high", 1.7, GOLD)]
    for name, rt, color in bands:
        y = np.exp(-t / rt)
        px = box[0] + t / 6 * (box[2] - box[0])
        py = box[3] - y * (box[3] - box[1])
        d.line(list(zip(px, py)), fill=color, width=5)
        d.text((1120, int(py[120]) - 18), f"{name}: {rt}s", fill=color, font=F_BODY)
    save(img, "09_multiband_decay.png")


def fig_dereverb_strength() -> None:
    img, d = canvas("Dereverb Strength vs Artifacts", "More reduction eventually trades clarity for processing damage.")
    box = (180, 190, 1420, 700)
    plot_axes(d, box, "Dereverb amount (%)", "Perceptual score (normalized, 0-1)")
    x = np.linspace(0, 1, 300)
    clarity = 1 - np.exp(-4 * x)
    natural = 1 - x ** 2.2
    useful = clarity * natural
    for y, color, label, yoff in [(clarity, TEAL, "clarity", 0), (natural, GOLD, "naturalness", 46), (useful, RUST, "sweet spot", 92)]:
        px = box[0] + x * (box[2] - box[0])
        py = box[3] - y * (box[3] - box[1])
        d.line(list(zip(px, py)), fill=color, width=5)
        d.text((1070, 240 + yoff), label, fill=color, font=F_BODY)
    save(img, "10_dereverb_tradeoff.png")


def fig_partitioned_convolution() -> None:
    img, d = canvas("Partitioned Convolution Layout", "Long IRs are split so early sound stays responsive.")
    x0, y0 = 120, 250
    widths = [120, 160, 220, 280, 360, 440]
    colors = [BLUE, TEAL, GOLD, RUST, PLUM, GREEN]
    for i, (w, c) in enumerate(zip(widths, colors)):
        y = y0 + i * 72
        d.rounded_rectangle((x0, y, x0 + w, y + 42), radius=10, fill=c)
        d.text((x0 + w + 24, y + 8), f"partition {i}: {2 ** (i + 8)} samples", fill=INK, font=F_BODY)
    arrow(d, (880, 335), (1210, 335), MUTED)
    node(d, (1220, 270, 1480, 410), "FFT blocks\n+ overlap-add", PLUM)
    d.text((950, 540), "Smaller first partitions reduce perceived latency; larger late partitions reduce CPU.", fill=INK, font=F_BODY)
    save(img, "11_partitioned_convolution.png")


def fig_ir_morph() -> None:
    img, d = canvas("IR Morphing Blend Space", "A dry room can become a plate, cathedral, or texture through controlled interpolation.")
    points = [(300, 650, "Room", BLUE), (850, 230, "Plate", GOLD), (1280, 640, "Cathedral", RUST)]
    d.polygon([(p[0], p[1]) for p in points], outline=MUTED, fill="#efe8d7")
    for x, y, label, color in points:
        d.ellipse((x - 38, y - 38, x + 38, y + 38), fill=color)
        d.text((x - 46, y + 54), label, fill=INK, font=F_BODY)
    d.ellipse((790 - 24, 505 - 24, 790 + 24, 505 + 24), fill=PLUM)
    d.text((830, 488), "morphed IR", fill=PLUM, font=F_BODY)
    d.text((1000, 780), "Coordinates: normalized IR blend weights (0-1)", fill=MUTED, font=F_SMALL)
    save(img, "12_ir_morph_space.png")


def fig_spatial_layouts() -> None:
    img, d = canvas("Spatial Layout Families", "Channel layout changes how reverberant energy wraps the listener.")
    centers = [(330, 460, "stereo"), (800, 460, "5.1"), (1270, 460, "7.1.4")]
    for cx, cy, label in centers:
        d.ellipse((cx - 58, cy - 58, cx + 58, cy + 58), fill=PAPER, outline=INK, width=4)
        d.text((cx - 38, cy - 14), "you", fill=INK, font=F_BODY)
        n = {"stereo": 2, "5.1": 6, "7.1.4": 12}[label]
        for i in range(n):
            ang = 2 * math.pi * i / n - math.pi / 2
            r = 150 if i < 8 else 95
            x, y = cx + r * math.cos(ang), cy + r * math.sin(ang)
            d.rounded_rectangle((x - 24, y - 24, x + 24, y + 24), radius=8, fill=BLUE if i < 8 else GOLD)
        d.text((cx - 42, 690), label, fill=INK, font=F_BODY)
    save(img, "13_spatial_layouts.png")


def fig_ambisonics_order() -> None:
    img, d = canvas("Ambisonics Order Growth", "Channel count rises quadratically with order: (N + 1)².")
    orders = list(range(0, 8))
    vals = [(n + 1) ** 2 for n in orders]
    labels = [str(n) for n in orders]
    bars(d, (170, 210, 1420, 700), vals, labels, [BLUE, TEAL, GOLD, RUST, PLUM, GREEN, "#8f6f3f", "#3f7f8f"])
    chart_labels(d, (170, 210, 1420, 700), "Ambisonics order N (integer)", "Channel count (channels)")
    save(img, "14_ambisonics_order.png")


def fig_shimmer_path() -> None:
    img, d = canvas("Shimmer Feedback Path", "Pitch-shifted feedback turns late energy into harmonic bloom.")
    boxes = [(120, 300, 330, 420), (430, 300, 640, 420), (740, 300, 950, 420), (1050, 300, 1260, 420)]
    labels = ["wet tail", "pitch\nshift", "diffuse\nagain", "feedback\nmix"]
    colors = [BLUE, GOLD, TEAL, RUST]
    for b, l, c in zip(boxes, labels, colors):
        node(d, b, l, c)
    for a, b in zip(boxes[:-1], boxes[1:]):
        arrow(d, (a[2] + 10, 360), (b[0] - 12, 360), MUTED)
    arrow(d, (1155, 430), (220, 520), PLUM)
    arrow(d, (220, 520), (220, 430), PLUM)
    d.text((510, 585), "feedback controls sustain; semitones control harmonic color", fill=INK, font=F_BODY)
    save(img, "15_shimmer_feedback.png")


def fig_room_inference() -> None:
    img, d = canvas("Room Size Inference", "RT60 plus absorption estimate maps to plausible room volume.")
    box = (210, 190, 1350, 700)
    plot_axes(d, box, "Measured RT60 (s)", "Estimated room volume (m³)")
    rt = np.linspace(0.2, 4.5, 200)
    for alpha, color in [(0.15, BLUE), (0.3, TEAL), (0.55, GOLD)]:
        vol = rt * (200 * alpha) / 0.161
        px = box[0] + (rt - rt.min()) / (rt.max() - rt.min()) * (box[2] - box[0])
        py = box[3] - (vol - vol.min()) / (2800 - vol.min()) * (box[3] - box[1])
        d.line(list(zip(px, py)), fill=color, width=5)
        d.text((1080, int(py[130]) - 20), f"alpha {alpha}", fill=color, font=F_BODY)
    save(img, "16_room_size_inference.png")


def fig_analysis_dashboard() -> None:
    img, d = canvas("Analysis Metrics Dashboard", "The JSON sidecar converts audio into comparable acoustic metrics.")
    metrics = [("RT60", "2.84 s", BLUE), ("DRR", "–6.1 dB", TEAL), ("C80", "–3.8 dB", GOLD), ("Peak", "–1.0 dB", RUST), ("LUFS", "–16.4", PLUM), ("EDT", "2.12 s", GREEN)]
    for i, (name, value, color) in enumerate(metrics):
        x = 120 + (i % 3) * 480
        y = 210 + (i // 3) * 250
        d.rounded_rectangle((x, y, x + 380, y + 180), radius=24, fill=PAPER, outline=color, width=5)
        d.text((x + 28, y + 28), name, fill=MUTED, font=F_BODY)
        d.text((x + 28, y + 82), value, fill=color, font=F_TITLE)
    save(img, "17_analysis_dashboard.png")


def fig_cli_map() -> None:
    img, d = canvas("CLI Command Map", "verbx is organized around render, realtime, analyze, IR, and utility workflows.")
    center = (800, 455)
    d.ellipse((center[0] - 80, center[1] - 80, center[0] + 80, center[1] + 80), fill=INK)
    text_center(d, (720, 390, 880, 520), "verbx", fill=PAPER, fnt=F_BODY)
    items = [("render", BLUE, -90), ("realtime", TEAL, -30), ("analyze", GOLD, 30), ("ir", RUST, 90), ("batch", PLUM, 150), ("presets", GREEN, 210)]
    for label, color, deg in items:
        ang = math.radians(deg)
        x, y = center[0] + 430 * math.cos(ang), center[1] + 300 * math.sin(ang)
        node(d, (int(x - 115), int(y - 54), int(x + 115), int(y + 54)), label, color)
        arrow(d, center, (int(x - 120 * math.cos(ang)), int(y - 60 * math.sin(ang))), color, 4)
    save(img, "18_cli_command_map.png")


def fig_references_map() -> None:
    img, d = canvas("Reference Corpus Shape", "Curated citations stay separate from the larger Crossref discovery index.")
    vals = [100, 900]
    labels = ["curated\nannotated", "extended\nCrossref"]
    bars(d, (300, 220, 1300, 700), vals, labels, [TEAL, PLUM])
    chart_labels(d, (300, 220, 1300, 700), "Reference collection (category)", "Bibliography entries (count)")
    d.text((260, 830), "The guide has 1,000 references total, but only the curated set is treated as implementation authority.", fill=INK, font=F_BODY)
    save(img, "19_reference_corpus.png")


def fig_ir_grid() -> None:
    img, d = canvas("IR Library Coverage Grid", "Duration bands crossed with synthesis families.")
    rows = ["tiny", "short", "medium", "long"]
    cols = ["fdn", "modal", "stochastic", "hybrid"]
    x0, y0, cell = 420, 210, 150
    for i, row in enumerate(rows):
        d.text((250, y0 + i * cell + 52), row, fill=INK, font=F_BODY)
    for j, col in enumerate(cols):
        d.text((x0 + j * cell + 28, 160), col, fill=INK, font=F_BODY)
    for i in range(4):
        for j in range(4):
            color = [BLUE, TEAL, GOLD, RUST][(i + j) % 4]
            d.rounded_rectangle((x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell - 16, y0 + (i + 1) * cell - 16), radius=16, fill=color)
            text_center(d, (x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell - 16, y0 + (i + 1) * cell - 16), "4 IRs", fill=PAPER)
    axis_labels(d, (420, 210, 1020, 810), "Synthesis family (category)", "Duration family (category)")
    d.text((1070, 700), "Cell value: impulse responses (count)", fill=MUTED, font=F_SMALL)
    save(img, "20_ir_library_grid.png")


def fig_cpu_block() -> None:
    img, d = canvas("Block Size CPU Tradeoff", "Small blocks feel fast; large blocks reduce scheduling pressure.")
    box = (170, 190, 1420, 700)
    plot_axes(d, box, "Audio block size (frames)", "Normalized cost or latency (0-1)")
    x = np.array([128, 256, 512, 1024, 2048])
    cpu = np.array([1.0, 0.62, 0.38, 0.25, 0.18])
    latency = np.array([0.08, 0.16, 0.32, 0.64, 1.0])
    for y, color, label, yoff in [(cpu, TEAL, "CPU pressure", 0), (latency, RUST, "latency", 46)]:
        px = box[0] + (np.log2(x) - 7) / 4 * (box[2] - box[0])
        py = box[3] - y * (box[3] - box[1])
        d.line(list(zip(px, py)), fill=color, width=6)
        d.text((1030, 250 + yoff), label, fill=color, font=F_BODY)
    save(img, "21_cpu_block_tradeoff.png")


def fig_json_tree() -> None:
    img, d = canvas("Analysis JSON Structure", "Sidecar output is designed for automation and regression checks.")
    root = (160, 210, 420, 310)
    node(d, root, "analysis.json", BLUE)
    branches = [("input", 590, 170, TEAL), ("metrics", 590, 310, GOLD), ("render", 590, 450, RUST), ("warnings", 590, 590, PLUM)]
    for label, x, y, color in branches:
        node(d, (x, y, x + 260, y + 90), label, color)
        arrow(d, (root[2], 260), (x - 12, y + 45), color, 4)
        for k in range(3):
            d.rounded_rectangle((x + 390, y + k * 32, x + 680, y + 24 + k * 32), radius=8, fill="#efe4cd", outline=GRID)
    save(img, "22_json_tree.png")


def fig_preset_space() -> None:
    img, d = canvas("Preset Design Space", "Presets balance time, tone, width, modulation, and safety.")
    cx, cy, r = 800, 470, 260
    axes = [("time", -90), ("tone", -18), ("width", 54), ("motion", 126), ("safety", 198)]
    values = [0.86, 0.62, 0.78, 0.48, 0.72]
    pts = []
    for (label, deg), val in zip(axes, values):
        ang = math.radians(deg)
        end = (cx + r * math.cos(ang), cy + r * math.sin(ang))
        d.line((cx, cy, end[0], end[1]), fill=GRID, width=3)
        d.text((end[0] - 35, end[1] - 16), label, fill=INK, font=F_BODY)
        pts.append((cx + r * val * math.cos(ang), cy + r * val * math.sin(ang)))
    d.polygon(pts, fill="#d8a24d", outline=RUST)
    d.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=INK)
    d.text((1030, 760), "Radial scale: normalized parameter amount (0-1)", fill=MUTED, font=F_SMALL)
    save(img, "23_preset_radar.png")


def fig_infinite_reverb() -> None:
    img, d = canvas("Infinite-Style Reverb Modes", "Very high RT60 values behave more like sustain instruments than rooms.")
    box = (170, 190, 1420, 700)
    plot_axes(d, box, "Normalized elapsed time (0-1)", "Relative tail energy (linear, 0-1)")
    t = np.linspace(0, 1, 400)
    normal = np.exp(-5 * t)
    long = np.exp(-1.3 * t)
    freeze = 0.68 + 0.05 * np.sin(20 * t)
    for y, color, label, yoff in [(normal, BLUE, "room", 0), (long, GOLD, "extreme", 46), (freeze, PLUM, "infinite/freeze", 92)]:
        px = box[0] + t * (box[2] - box[0])
        py = box[3] - y * (box[3] - box[1])
        d.line(list(zip(px, py)), fill=color, width=5)
        d.text((1060, 230 + yoff), label, fill=color, font=F_BODY)
    save(img, "24_infinite_reverb.png")


EXTRA_FIGURES: tuple[tuple[str, str, str, str, int], ...] = (
    ("Early Reflection Timing", "Tap spacing sketches perceived room geometry before the late field blooms.", "25_early_reflections.png", "timeline", 25),
    ("Pre-Delay Perception", "A few milliseconds can separate source presence from room size.", "26_predelay_perception.png", "curve", 26),
    ("Diffusion Build-Up", "Dense tails emerge as echo density crosses the fusion threshold.", "27_diffusion_buildup.png", "curve", 27),
    ("Damping EQ Targets", "Low, mid, and high shelves shape the perceived material of a room.", "28_damping_eq_targets.png", "multi", 28),
    ("Modulation Depth Safety", "Depth and rate interact: motion is useful until pitch smear takes over.", "29_modulation_depth_safety.png", "heat", 29),
    ("Stereo Width Correlation", "Width controls should preserve mono safety while expanding ambience.", "30_stereo_width_correlation.png", "curve", 30),
    ("Haas Zone", "Tiny left/right delays can widen image before becoming discrete echoes.", "31_haas_zone.png", "bands", 31),
    ("Gate Tail Shapes", "Classic gated reverb depends on hold, release, and threshold timing.", "32_gate_tail_shapes.png", "timeline", 32),
    ("Reverse Reverb Envelope", "Reverse tails rise into the transient instead of decaying away from it.", "33_reverse_reverb_envelope.png", "curve", 33),
    ("Spectral Tilt Analyzer", "Broadband tilt gives a fast visual clue for dark versus bright renders.", "34_spectral_tilt_analyzer.png", "multi", 34),
    ("Loudness Normalization Path", "Peak, RMS, and LUFS views catch different gain staging problems.", "35_loudness_normalization_path.png", "bars", 35),
    ("Sample Rate Cost", "Higher sample rates improve bandwidth at a predictable CPU cost.", "36_sample_rate_cost.png", "bars", 36),
    ("Oversampling Alias Guard", "Limiter oversampling moves foldback artifacts away from the audible band.", "37_oversampling_alias_guard.png", "multi", 37),
    ("Lookahead Limiter Timing", "Lookahead catches peaks before gain reduction becomes audible pumping.", "38_lookahead_limiter_timing.png", "timeline", 38),
    ("Dry/Wet Crossfade Laws", "Linear, equal-power, and DJ-style blends feel different near the center.", "39_dry_wet_crossfade_laws.png", "multi", 39),
    ("IR Trim Finder", "Trim logic balances silence removal against preserving natural onset.", "40_ir_trim_finder.png", "timeline", 40),
    ("Silence Detector Thresholds", "Noise floors and thresholds decide where batch processing can skip work.", "41_silence_detector_thresholds.png", "curve", 41),
    ("Batch Render Throughput", "Parallel workers help until IO and memory pressure dominate.", "42_batch_render_throughput.png", "bars", 42),
    ("Cache Hit Savings", "Reusable analysis and IR material turn repeated renders into fast paths.", "43_cache_hit_savings.png", "bars", 43),
    ("Native Parity Slice", "A narrow deterministic slice keeps Python and native engines aligned.", "44_native_parity_slice.png", "stack", 44),
    ("Test Matrix Coverage", "Golden audio, CLI smoke, realtime, and docs checks cover different risks.", "45_test_matrix_coverage.png", "heat", 45),
    ("Preset Morph Trajectory", "Morph paths should move smoothly through perceptual control space.", "46_preset_morph_trajectory.png", "space", 46),
    ("Realtime Dropout Risk", "CPU load, block size, and driver buffers define the danger zone.", "47_realtime_dropout_risk.png", "heat", 47),
    ("Release Readiness Dashboard", "The release gate is healthiest when docs, tests, render, and realtime agree.", "48_release_readiness_dashboard.png", "radar", 48),
)


MORE_FIGURES: tuple[tuple[str, str, str, str, int], ...] = (
    ("Comb Filter Notches", "Short delays carve predictable notches that can make tails metallic.", "49_comb_filter_notches.png", "multi", 49),
    ("Allpass Diffuser Response", "Allpass stages preserve energy while scrambling phase and timing.", "50_allpass_diffuser_response.png", "curve", 50),
    ("FDN Delay Distribution", "Prime-ish delay spacing avoids obvious repeating echo patterns.", "51_fdn_delay_distribution.png", "bars", 51),
    ("Modal Density Growth", "Large rooms pack more resonances into each octave.", "52_modal_density_growth.png", "curve", 52),
    ("Schroeder Frequency Estimate", "Below the transition band, individual modes matter more.", "53_schroeder_frequency_estimate.png", "bands", 53),
    ("Air Absorption Roll-Off", "Long bright tails need damping to avoid synthetic glare.", "54_air_absorption_rolloff.png", "multi", 54),
    ("Material Absorption Map", "Wall, carpet, curtain, and glass assumptions shape decay by band.", "55_material_absorption_map.png", "heat", 55),
    ("Early Late Balance", "Presence comes from early energy; envelopment comes from late energy.", "56_early_late_balance.png", "bars", 56),
    ("Source Distance Cue", "Distance changes direct-to-reverberant balance before it changes tone.", "57_source_distance_cue.png", "curve", 57),
    ("Mic Pattern Pickup", "Cardioid, omni, and figure-eight captures feed the room differently.", "58_mic_pattern_pickup.png", "multi", 58),
    ("Sidechain Detector Modes", "Peak and RMS detectors react on different musical timescales.", "59_sidechain_detector_modes.png", "multi", 59),
    ("Ducking Release Families", "Release curvature decides whether ambience breathes or pumps.", "60_ducking_release_families.png", "multi", 60),
    ("Limiter Knee Families", "Knee width trades transparent onset against strict peak containment.", "61_limiter_knee_families.png", "multi", 61),
    ("True Peak Margin", "Inter-sample peaks make safety margin useful even when samples look safe.", "62_true_peak_margin.png", "timeline", 62),
    ("LUFS Integration Windows", "Momentary, short-term, and integrated loudness answer different questions.", "63_lufs_integration_windows.png", "timeline", 63),
    ("Crest Factor Map", "Transient-heavy inputs need different limiter behavior from pads.", "64_crest_factor_map.png", "heat", 64),
    ("Transient Preservation", "Dereverb should reduce tail energy without flattening attack detail.", "65_transient_preservation.png", "timeline", 65),
    ("Dereverb Mask Strength", "Mask aggressiveness governs the speech-cleanup versus artifact tradeoff.", "66_dereverb_mask_strength.png", "curve", 66),
    ("Spectral Gate Residuals", "Residual maps reveal where dereverb is leaving flutter or musical noise.", "67_spectral_gate_residuals.png", "heat", 67),
    ("Noise Floor Tracking", "Slow floor estimates avoid chasing quiet reverb as if it were noise.", "68_noise_floor_tracking.png", "curve", 68),
    ("Multichannel Routing Matrix", "Channel maps keep immersive and stereo renders auditable.", "69_multichannel_routing_matrix.png", "heat", 69),
    ("Ambisonic Decode Spread", "Decode spread converts abstract soundfield order into speaker energy.", "70_ambisonic_decode_spread.png", "space", 70),
    ("Binaural HRTF Blend", "HRTF blending needs smooth interpolation across azimuth and elevation.", "71_binaural_hrtf_blend.png", "space", 71),
    (
        "Loudspeaker Layouts: Plan and Elevation",
        "Nominal channel bearings for stereo, 5.1, and 7.1.4, with the immersive "
        "height layer shown separately.",
        "72_speaker_layout_coverage.png",
        "layout",
        72,
    ),
    ("IR Capture Checklist", "Capture quality depends on sweep level, silence, trim, and calibration.", "73_ir_capture_checklist.png", "stack", 73),
    ("Sweep Deconvolution Path", "Measured IRs move through sweep, inverse filter, trim, and normalize steps.", "74_sweep_deconvolution_path.png", "stack", 74),
    ("IR Tail Trim Decision", "Trim should stop after useful decay, not after the first low-energy valley.", "75_ir_tail_trim_decision.png", "timeline", 75),
    ("IR Normalization Modes", "Peak, energy, and loudness normalization each preserve a different invariant.", "76_ir_normalization_modes.png", "bars", 76),
    ("Convolution Partition Plan", "Small early partitions and larger late partitions balance CPU and latency.", "77_convolution_partition_plan.png", "timeline", 77),
    ("FFT Size Efficiency", "FFT cost rises in steps, so partition sizes should sit on friendly powers.", "78_fft_size_efficiency.png", "bars", 78),
    ("SIMD Batch Shape", "Native kernels are fastest when channel and frame batches align cleanly.", "79_simd_batch_shape.png", "heat", 79),
    ("Memory Bandwidth Pressure", "Long tails can become bandwidth-bound before they become math-bound.", "80_memory_bandwidth_pressure.png", "curve", 80),
    ("Thread Pool Scaling", "More workers help until contention and IO erase the gain.", "81_thread_pool_scaling.png", "bars", 81),
    ("Realtime Callback Budget", "Audio callbacks need headroom below the hard deadline.", "82_realtime_callback_budget.png", "timeline", 82),
    ("XRuns By Block Size", "Dropout risk falls with buffer size but latency rises in exchange.", "83_xruns_by_block_size.png", "curve", 83),
    ("Device Buffer Stack", "Round-trip latency is the sum of API, driver, hardware, and DSP buffers.", "84_device_buffer_stack.png", "stack", 84),
    ("CLI Option Families", "Render, analysis, IR, limiter, realtime, and batch flags cluster by job.", "85_cli_option_families.png", "radar", 85),
    ("Preset Taxonomy", "Presets should be searchable by space, tone, motion, and safety intent.", "86_preset_taxonomy.png", "heat", 86),
    ("JSON Schema Coverage", "Schema fields make render results machine-checkable across releases.", "87_json_schema_coverage.png", "heat", 87),
    ("Analysis Regression Bands", "Metric tolerances should be tight where outputs are deterministic.", "88_analysis_regression_bands.png", "bands", 88),
    ("Golden Audio Drift", "Golden fixtures reveal unexpected changes in gain, tail, or spectrum.", "89_golden_audio_drift.png", "multi", 89),
    ("Documentation Build Pipeline", "Markdown, figures, Pandoc, LaTeX, and PDF checks form one reproducible chain.", "90_documentation_build_pipeline.png", "stack", 90),
    ("Table Wrap Stress Test", "Long CLI options and URLs need wrapping before they hit the page edge.", "91_table_wrap_stress_test.png", "heat", 91),
    ("Reference Annotation Flow", "Curated notes, extended entries, and cross-links serve different readers.", "92_reference_annotation_flow.png", "stack", 92),
    ("Citation Corpus Growth", "Reference expansion should grow breadth without diluting implementation guidance.", "93_citation_corpus_growth.png", "bars", 93),
    ("Release Branch Flow", "Feature, docs, CI, tag, and package steps should stay visible.", "94_release_branch_flow.png", "stack", 94),
    ("Homebrew Formula Refresh", "Version, checksum, bottle, and audit checks keep installs boring.", "95_homebrew_formula_refresh.png", "stack", 95),
    ("Platform Support Grid", "macOS, Linux, and Windows support differs across audio backends.", "96_platform_support_grid.png", "heat", 96),
    ("Error Message Quality", "Good CLI errors name the bad input, likely cause, and next command.", "97_error_message_quality.png", "radar", 97),
    ("User Workflow Map", "Explore, render, analyze, compare, and automate form the core loop.", "98_user_workflow_map.png", "stack", 98),
    ("Feature Maturity Radar", "Stable, beta, experimental, and roadmap features need distinct labeling.", "99_feature_maturity_radar.png", "radar", 99),
    ("End-to-End Confidence Map", "The full system is healthiest when audio, docs, tests, and packaging align.", "100_end_to_end_confidence_map.png", "heat", 100),
)


# Semantic labels for every atlas chart. The third value labels heatmap color or
# radar radius; schematics intentionally omit Cartesian axes.
ATLAS_AXES: dict[int, tuple[str, str, str]] = {
    25: ("Time after direct sound (ms)", "Reflection amplitude (dBFS)", ""),
    26: ("Pre-delay (ms)", "Perceived source-room separation (normalized, 0-1)", ""),
    27: ("Time after excitation (ms)", "Echo density (reflections/s)", ""),
    28: ("Frequency (Hz)", "Relative damping gain (dB)", ""),
    29: ("Modulation rate (Hz)", "Modulation depth (ms)", "Artifact risk (normalized, 0-1)"),
    30: ("Stereo width (%)", "Inter-channel correlation (unitless, –1 to +1)", ""),
    31: ("Inter-channel delay (ms)", "Perceptual region (category)", ""),
    32: ("Time after transient (ms)", "Wet level (dBFS)", ""),
    33: ("Time before transient (ms)", "Wet envelope level (linear, 0-1)", ""),
    34: ("Frequency (Hz)", "Spectral magnitude (dBFS)", ""),
    35: ("Normalization method (category)", "Target or measured level (dB)", ""),
    36: ("Sample rate (kHz)", "Relative CPU cost (%)", ""),
    37: ("Frequency (kHz)", "Alias energy (dBFS)", ""),
    38: ("Time relative to peak (ms)", "Gain reduction (dB)", ""),
    39: ("Dry/wet control (%)", "Channel gain (linear, 0-1)", ""),
    40: ("Time in impulse response (ms)", "IR amplitude (dBFS)", ""),
    41: ("Detector threshold (dBFS)", "Detected activity (%)", ""),
    42: ("Parallel workers (count)", "Render throughput (files/min)", ""),
    43: ("Cache condition (category)", "Elapsed processing time (%)", ""),
    45: ("Test family (category)", "Platform or engine (category)", "Coverage (%)"),
    46: ("Timbral brightness (normalized, 0-1)", "Spatial width (normalized, 0-1)", ""),
    47: ("Audio block size (frames)", "Callback CPU load (%)", "Dropout risk (normalized, 0-1)"),
    48: ("Readiness dimension (category)", "", "Completion score (normalized, 0-1)"),
    49: ("Frequency (Hz)", "Magnitude response (dB)", ""),
    50: ("Frequency (Hz)", "Group delay (ms)", ""),
    51: ("FDN delay line (index)", "Delay length (samples)", ""),
    52: ("Frequency (Hz)", "Modes per octave (count)", ""),
    53: ("Frequency (Hz)", "Acoustic behavior (category)", ""),
    54: ("Frequency (Hz)", "Air attenuation (dB/m)", ""),
    55: ("Frequency band (Hz)", "Surface material (category)", "Absorption coefficient (0-1)"),
    56: ("Energy component (category)", "Relative energy (%)", ""),
    57: ("Source distance (m)", "Direct-to-reverberant ratio (dB)", ""),
    58: ("Arrival angle (degrees)", "Relative microphone sensitivity (dB)", ""),
    59: ("Time after onset (ms)", "Detector envelope (linear, 0-1)", ""),
    60: ("Time after sidechain release (ms)", "Wet gain (dB)", ""),
    61: ("Input level relative to threshold (dB)", "Output level (dBFS)", ""),
    62: ("Time around sample peak (µs)", "Signal level (dBFS)", ""),
    63: ("Program time (s)", "Integrated loudness (LUFS)", ""),
    64: ("RMS level (dBFS)", "Peak level (dBFS)", "Crest factor (dB)"),
    65: ("Time around transient (ms)", "Signal amplitude (linear, –1 to +1)", ""),
    66: ("Dereverb mask strength (%)", "Artifact or suppression score (normalized, 0-1)", ""),
    67: ("Time (s)", "Frequency (Hz)", "Residual magnitude (dBFS)"),
    68: ("Program time (s)", "Estimated noise floor (dBFS)", ""),
    69: ("Input channel (index)", "Output channel (index)", "Routing gain (dB)"),
    70: ("Speaker azimuth (degrees)", "Speaker elevation (degrees)", "Relative decode energy (0-1)"),
    71: ("Source azimuth (degrees)", "Source elevation (degrees)", "HRTF blend weight (0-1)"),
    72: ("", "", ""),
    75: ("Time in impulse response (s)", "IR decay level (dBFS)", ""),
    76: ("Normalization mode (category)", "Resulting reference level (dB)", ""),
    77: ("IR time offset (samples)", "Partition size (samples)", ""),
    78: ("FFT size (samples)", "Relative processing cost (%)", ""),
    79: ("Frame batch size (frames)", "Channel batch size (channels)", "SIMD utilization (%)"),
    80: ("Impulse-response duration (s)", "Memory bandwidth (GB/s)", ""),
    81: ("Worker threads (count)", "Speedup (× realtime)", ""),
    82: ("Callback time (ms)", "Budget use (%)", ""),
    83: ("Audio block size (frames)", "XRuns (count/hour)", ""),
    85: ("CLI family (category)", "", "Option coverage (normalized, 0-1)"),
    86: ("Preset attribute (category)", "Preset family (category)", "Library density (presets/cell)"),
    87: ("Schema field group (category)", "Command (category)", "Fields implemented (%)"),
    88: ("Regression metric (category)", "Allowed deviation (%)", ""),
    89: ("Frequency (Hz)", "Golden-output drift (dB)", ""),
    91: ("Table column width (characters)", "Content length (characters)", "Overflow risk (normalized, 0-1)"),
    93: ("Documentation release (version index)", "References (count)", ""),
    96: ("Feature or backend (category)", "Operating system (category)", "Support state (category)"),
    97: ("Error-message quality (category)", "", "Quality score (normalized, 0-1)"),
    99: ("Maturity dimension (category)", "", "Maturity score (normalized, 0-1)"),
    100: ("Subsystem (category)", "Verification layer (category)", "Confidence score (normalized, 0-1)"),
}


def _polar_position(
    center: tuple[int, int], radius: float, azimuth_degrees: float
) -> tuple[float, float]:
    angle = math.radians(azimuth_degrees)
    return center[0] + radius * math.sin(angle), center[1] - radius * math.cos(angle)


def _speaker_mark(
    d: ImageDraw.ImageDraw,
    center: tuple[float, float],
    label: str,
    color: str,
) -> None:
    x, y = center
    radius = 20 if len(label) <= 2 else 23
    d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    text_center(
        d,
        (int(x - radius), int(y - radius), int(x + radius), int(y + radius)),
        label,
        fill=PAPER,
        fnt=F_CHANNEL,
    )


def _listener_mark(d: ImageDraw.ImageDraw, center: tuple[int, int]) -> None:
    x, y = center
    d.ellipse((x - 25, y - 25, x + 25, y + 25), fill=PAPER, outline=INK, width=4)
    d.arc((x - 35, y - 16, x - 20, y + 16), 90, 270, fill=INK, width=3)
    d.arc((x + 20, y - 16, x + 35, y + 16), -90, 90, fill=INK, width=3)
    arrow(d, (x, y - 30), (x, y - 58), INK, 3)


def _speaker_plan(
    d: ImageDraw.ImageDraw,
    *,
    center: tuple[int, int],
    title: str,
    speakers: tuple[tuple[str, float, str], ...],
    lfe: bool,
) -> None:
    cx, cy = center
    radius = 150
    frame = (cx - 225, 172, cx + 225, 650)
    d.rounded_rectangle(frame, radius=24, fill=PAPER, outline=GRID, width=3)
    title_box = d.textbbox((0, 0), title, font=F_PANEL)
    d.text((cx - (title_box[2] - title_box[0]) / 2, 192), title, fill=INK, font=F_PANEL)
    d.text((cx - 52, 228), "FRONT / 0°", fill=MUTED, font=F_TINY)
    d.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        outline=GRID,
        width=3,
    )
    d.line((cx, cy - radius, cx, cy + radius), fill=GRID, width=2)
    d.line((cx - radius, cy, cx + radius, cy), fill=GRID, width=2)
    d.text((cx - 48, cy + radius - 25), "REAR / 180°", fill=MUTED, font=F_TINY)
    _listener_mark(d, center)

    for label, azimuth, layer in speakers:
        speaker_radius = 92 if layer == "height" else radius
        position = _polar_position(center, speaker_radius, azimuth)
        d.line((cx, cy, position[0], position[1]), fill="#dbe2de", width=2)
        color = GOLD if layer == "height" else TEAL if "s" in label.lower() else BLUE
        _speaker_mark(d, position, label, color)

    if lfe:
        d.rounded_rectangle(
            (frame[0] + 18, frame[3] - 62, frame[0] + 112, frame[3] - 18),
            radius=12,
            fill=RUST,
        )
        text_center(
            d,
            (frame[0] + 18, frame[3] - 62, frame[0] + 112, frame[3] - 18),
            "LFE",
            fill=PAPER,
            fnt=F_CHANNEL,
        )
        d.text(
            (frame[0] + 122, frame[3] - 50),
            "non-positional",
            fill=MUTED,
            font=F_TINY,
        )


def fig_speaker_layout_coverage(filename: str = "72_speaker_layout_coverage.png") -> None:
    img, d = canvas(
        "Loudspeaker Layouts: Plan and Elevation",
        "Channel labels show nominal listener-relative bearings; connecting lines are "
        "bearing guides, not signal flow.",
    )
    _speaker_plan(
        d,
        center=(275, 420),
        title="Stereo / 2.0",
        speakers=(("L", -30, "bed"), ("R", 30, "bed")),
        lfe=False,
    )
    _speaker_plan(
        d,
        center=(800, 420),
        title="5.1 Bed",
        speakers=(
            ("L", -30, "bed"),
            ("C", 0, "bed"),
            ("R", 30, "bed"),
            ("Ls", -110, "bed"),
            ("Rs", 110, "bed"),
        ),
        lfe=True,
    )
    _speaker_plan(
        d,
        center=(1325, 420),
        title="7.1.4 Immersive",
        speakers=(
            ("L", -30, "bed"),
            ("C", 0, "bed"),
            ("R", 30, "bed"),
            ("Lss", -90, "bed"),
            ("Rss", 90, "bed"),
            ("Lrs", -135, "bed"),
            ("Rrs", 135, "bed"),
            ("Ltf", -45, "height"),
            ("Rtf", 45, "height"),
            ("Ltr", -135, "height"),
            ("Rtr", 135, "height"),
        ),
        lfe=True,
    )

    d.rounded_rectangle(
        (65, 682, 1535, 860),
        radius=22,
        fill="#efe4cd",
        outline=GRID,
        width=3,
    )
    d.text((95, 706), "KEY", fill=INK, font=F_PANEL)
    _speaker_mark(d, (125, 770), "L", BLUE)
    d.text((158, 758), "front / center bed", fill=INK, font=F_SMALL)
    _speaker_mark(d, (125, 820), "Ls", TEAL)
    d.text((158, 808), "side / rear bed", fill=INK, font=F_SMALL)
    _speaker_mark(d, (425, 770), "Ltf", GOLD)
    d.text((460, 758), "height channel", fill=INK, font=F_SMALL)
    d.rounded_rectangle((402, 802, 448, 838), radius=9, fill=RUST)
    d.text((460, 808), "LFE has no bearing", fill=INK, font=F_SMALL)

    elevation_title = "7.1.4 SIDE ELEVATION"
    elevation_box = d.textbbox((0, 0), elevation_title, font=F_PANEL)
    d.text(
        (1110 - (elevation_box[2] - elevation_box[0]) / 2, 695),
        elevation_title,
        fill=INK,
        font=F_PANEL,
    )
    listener = (1110, 815)
    d.line((755, 815, 1480, 815), fill=MUTED, width=3)
    _listener_mark(d, listener)
    for x in (835, 1385):
        d.line((listener[0], listener[1], x, 815), fill="#dbe2de", width=2)
        _speaker_mark(d, (x, 815), "bed", TEAL)
    for x in (1045, 1175):
        d.line((listener[0], listener[1], x, 750), fill="#dbe2de", width=2)
        _speaker_mark(d, (x, 750), "top", GOLD)
    d.text((1210, 741), "height: +45°", fill=MUTED, font=F_TINY)
    save(img, filename)


def fig_extra(title: str, subtitle: str, filename: str, kind: str, seed: int) -> None:
    if kind == "layout":
        fig_speaker_layout_coverage(filename)
        return
    img, d = canvas(title, subtitle)
    rng = np.random.default_rng(seed)
    box = (170, 190, 1420, 700)
    palette = [BLUE, TEAL, GOLD, RUST, PLUM, GREEN]

    xlab, ylab, scale_lab = ATLAS_AXES.get(seed, ("Horizontal dimension (normalized, 0-1)", "Vertical dimension (normalized, 0-1)", ""))

    if kind in {"curve", "multi"}:
        plot_axes(d, box, xlab, ylab)
        x = np.linspace(0, 1, 360)
        count = 1 if kind == "curve" else 4
        for i in range(count):
            phase = rng.uniform(0, 1)
            bend = 0.65 + i * 0.28
            y = np.clip((x ** bend) * (0.82 - i * 0.08) + 0.08 * np.sin((i + 1.5) * np.pi * x + phase), 0, 1)
            if seed in {33, 39} and i % 2 == 0:
                y = 1 - y
            px = box[0] + x * (box[2] - box[0])
            py = box[3] - y * (box[3] - box[1])
            d.line(list(zip(px, py)), fill=palette[i], width=5)
        for i, color in enumerate(palette[:count]):
            d.text((1040, 225 + i * 42), f"trace {i + 1}", fill=color, font=F_BODY)

    elif kind == "bars":
        labels = ["A", "B", "C", "D", "E"]
        vals = list((rng.uniform(0.28, 1.0, size=5) * 100).astype(int))
        bars(d, box, vals, labels, palette[:5])
        chart_labels(d, box, xlab, ylab)

    elif kind == "timeline":
        plot_axes(d, box, xlab, ylab)
        x = np.linspace(0, 1, 500)
        base = np.exp(-3.5 * x)
        px = box[0] + x * (box[2] - box[0])
        py = box[3] - base * (box[3] - box[1])
        d.line(list(zip(px, py)), fill=BLUE, width=5)
        for i, tap in enumerate(np.sort(rng.uniform(0.05, 0.9, size=10))):
            height = rng.uniform(0.18, 0.86)
            tx = box[0] + tap * (box[2] - box[0])
            ty = box[3] - height * (box[3] - box[1])
            d.line((tx, box[3], tx, ty), fill=palette[i % len(palette)], width=4)
            d.ellipse((tx - 7, ty - 7, tx + 7, ty + 7), fill=palette[i % len(palette)])

    elif kind == "heat":
        rows, cols = 8, 14
        cell_w, cell_h = 76, 48
        x0, y0 = 280, 210
        for i in range(rows):
            for j in range(cols):
                v = (math.sin(i * 0.7 + seed) + math.cos(j * 0.45 + seed / 3) + 2) / 4
                v = 0.65 * v + 0.35 * rng.uniform()
                color = (int(235 - 110 * v), int(222 - 70 * v), int(190 + 55 * v))
                d.rounded_rectangle((x0 + j * cell_w, y0 + i * cell_h, x0 + (j + 1) * cell_w - 8, y0 + (i + 1) * cell_h - 8), radius=8, fill=color)
        axis_labels(d, (280, 210, 1344, 594), xlab, ylab)
        d.text((280, 660), f"Color scale: {scale_lab}", fill=INK, font=F_BODY)

    elif kind == "bands":
        x0, y0, x1, y1 = box
        bands_data = [(0.0, 0.24, TEAL, "fusion"), (0.24, 0.52, GOLD, "wide"), (0.52, 1.0, RUST, "echo")]
        for start, end, color, label in bands_data:
            bx0 = x0 + start * (x1 - x0)
            bx1 = x0 + end * (x1 - x0)
            d.rounded_rectangle((bx0, y0, bx1, y1), radius=24, fill=color)
            text_center(d, (int(bx0), y0, int(bx1), y1), label, fill=PAPER, fnt=F_BODY)
        axis_labels(d, box, xlab, ylab)

    elif kind == "stack":
        stages = ["Python", "CLI", "DSP", "analysis", "native"]
        x = 210
        for i, stage in enumerate(stages):
            node(d, (x, 300, x + 210, 430), stage, palette[i])
            if i:
                arrow(d, (x - 54, 365), (x - 10, 365), MUTED, 4)
            x += 250
        d.rounded_rectangle((250, 570, 1350, 690), radius=26, fill="#efe4cd", outline=GRID, width=3)
        text_center(d, (250, 570, 1350, 690), "Parity tests lock deterministic inputs, seeds, windows, and expected metrics.", fill=INK)

    elif kind == "space":
        x0, y0, x1, y1 = box
        d.rectangle(box, outline=GRID, width=3, fill="#fbf4e5")
        pts = rng.uniform(size=(9, 2))
        pts[:, 0] = x0 + pts[:, 0] * (x1 - x0)
        pts[:, 1] = y0 + pts[:, 1] * (y1 - y0)
        d.line([tuple(p) for p in pts], fill=PLUM, width=5)
        for i, p in enumerate(pts):
            d.ellipse((p[0] - 13, p[1] - 13, p[0] + 13, p[1] + 13), fill=palette[i % len(palette)])
        axis_labels(d, box, xlab, ylab)

    elif kind == "radar":
        cx, cy, r = 800, 460, 250
        axes = ["docs", "tests", "render", "rt", "native", "release"]
        vals = rng.uniform(0.52, 0.95, size=len(axes))
        pts = []
        for i, (label, val) in enumerate(zip(axes, vals)):
            ang = -math.pi / 2 + 2 * math.pi * i / len(axes)
            end = (cx + r * math.cos(ang), cy + r * math.sin(ang))
            d.line((cx, cy, end[0], end[1]), fill=GRID, width=3)
            d.text((end[0] - 32, end[1] - 18), label, fill=INK, font=F_BODY)
            pts.append((cx + r * val * math.cos(ang), cy + r * val * math.sin(ang)))
        d.polygon(pts, fill="#8ab6a8", outline=BLUE)
        d.text((1030, 750), f"Radial scale: {scale_lab}", fill=MUTED, font=F_SMALL)

    save(img, filename)


def fig_extra_gallery() -> None:
    for spec in (*EXTRA_FIGURES, *MORE_FIGURES):
        fig_extra(*spec)


def main() -> int:
    for fn in [
        fig_signal_flow,
        fig_realtime_latency,
        fig_rt60_curves,
        fig_edc_windows,
        fig_fdn_matrix,
        fig_window_functions,
        fig_limiter_curve,
        fig_ducking,
        fig_multiband_decay,
        fig_dereverb_strength,
        fig_partitioned_convolution,
        fig_ir_morph,
        fig_spatial_layouts,
        fig_ambisonics_order,
        fig_shimmer_path,
        fig_room_inference,
        fig_analysis_dashboard,
        fig_cli_map,
        fig_references_map,
        fig_ir_grid,
        fig_cpu_block,
        fig_json_tree,
        fig_preset_space,
        fig_infinite_reverb,
    ]:
        fn()
    fig_extra_gallery()
    print(f"Wrote figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
