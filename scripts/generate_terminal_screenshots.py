#!/usr/bin/env python3
"""Generate reproducible terminal transcript figures from verified verbx runs."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "terminal"
PROMPT = "cleider@Colby-Leiders-M3-Pro-MacBook-Pro-2024 verbx % "


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ("Menlo Bold.ttf", "Menlo.ttc") if bold else ("Menlo.ttc", "Menlo.ttf")
    for name in names:
        for base in ("/System/Library/Fonts", "/System/Library/Fonts/Supplemental"):
            try:
                return ImageFont.truetype(f"{base}/{name}", size)
            except OSError:
                pass
    return ImageFont.load_default()


def terminal_figure(name: str, command: str, transcript: tuple[str, ...]) -> None:
    font = _font(25)
    bold = _font(25, True)
    line_height = 39
    width = 1800
    height = 118 + line_height * (len(transcript) + 3)
    image = Image.new("RGB", (width, height), "#111816")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((1, 1, width - 2, height - 2), radius=20, outline="#43524d", width=3)
    draw.rectangle((1, 1, width - 2, 72), fill="#26312e")
    for x, color in ((35, "#ff605c"), (75, "#ffbd44"), (115, "#00ca4e")):
        draw.ellipse((x - 10, 36 - 10, x + 10, 36 + 10), fill=color)
    draw.text((width / 2, 36), "verbx terminal", fill="#d8e3de", font=bold, anchor="mm")

    y = 96
    draw.text((28, y), PROMPT, fill="#69c8b9", font=bold)
    prompt_width = draw.textlength(PROMPT, font=bold)
    draw.text((28 + prompt_width, y), command, fill="#f4f0e6", font=font)
    y += line_height * 1.5
    for line in transcript:
        color = "#8fd5a6" if "PASS" in line else "#d8e3de"
        draw.text((28, y), line, fill=color, font=font)
        y += line_height

    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / name, optimize=True)


def main() -> int:
    terminal_figure(
        "01_doctor.png",
        "uv run verbx doctor --strict",
        (
            "Platform: macOS-26.4.1-arm64",
            "CPU count: 12",
            "Apple Silicon: True",
            "MPS policy: auto",
            "PASS  Python runtime",
            "PASS  Audio dependencies",
            "PASS  CLI entry point",
            "PASS  Accelerated backend discovery",
            "PASS  Output directory access",
        ),
    )
    terminal_figure(
        "02_render_dry_run.png",
        "uv run verbx render \\",
        (
            "  examples/audio/realistic_music_dry.wav tmp/terminal_demo.wav \\",
            "  --preset warm_small_hall --dry-run \\",
            "  --json-out tmp/terminal_demo.json",
            "Resolved render plan",
            "Preset                 warm_small_hall",
            "RT60                   4.800 s",
            "Wet / dry              0.600 / 0.520",
            "FDN matrix / lines      householder / 12",
            "Estimated output        10.338 s",
            "Estimated file size     3.786 MB",
            "Dry run complete: audio write skipped",
        ),
    )
    print(f"Wrote 2 terminal transcript figures to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
