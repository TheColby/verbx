#!/usr/bin/env python3
"""Generate a large curated IR library grouped by length bucket and mode."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import soundfile as sf

from verbx.ir.generator import IRGenConfig, generate_ir

OutFormat = Literal["flac", "wav", "aiff", "aif", "ogg", "caf"]


BUCKET_LENGTHS: dict[str, tuple[float, ...]] = {
    "tiny": (0.5, 1.0, 2.0, 4.0),
    "short": (6.0, 8.0, 12.0, 16.0),
    "medium": (24.0, 32.0, 45.0, 60.0),
    "long": (75.0, 90.0, 120.0, 180.0),
}
MODES: tuple[str, ...] = ("fdn", "stochastic", "modal", "hybrid")
MATRICES: tuple[str, ...] = ("hadamard", "householder", "random_orthogonal", "circulant")


def _build_config(
    *,
    mode: str,
    length: float,
    seed: int,
    sr: int,
    channels: int,
    variant: int,
) -> IRGenConfig:
    damping = min(0.88, 0.22 + (0.08 * (variant % 7)))
    diffusion = min(1.0, 0.30 + (0.12 * (variant % 6)))
    density = 0.75 + (0.18 * (variant % 6))
    rt60 = max(0.3, min(length * 0.85, max(8.0, length * 0.65)))
    matrix = MATRICES[variant % len(MATRICES)]

    return IRGenConfig(
        mode=mode,
        length=length,
        sr=sr,
        channels=channels,
        seed=seed,
        rt60=rt60,
        damping=damping,
        lowcut=40.0 + (20.0 * (variant % 5)),
        highcut=6000.0 + (1500.0 * (variant % 6)),
        tilt=-2.0 + (0.55 * (variant % 9)),
        normalize="peak",
        peak_dbfs=-1.0,
        target_lufs=None,
        true_peak=True,
        er_count=18 + (variant % 20),
        er_max_delay_ms=40.0 + (10.0 * (variant % 9)),
        er_decay_shape="exp",
        er_stereo_width=0.7 + (0.15 * (variant % 7)),
        er_room=0.8 + (0.25 * (variant % 6)),
        diffusion=diffusion,
        mod_depth_ms=0.5 + (0.55 * (variant % 8)),
        mod_rate_hz=0.03 + (0.025 * (variant % 9)),
        density=density,
        tuning="A4=440" if (variant % 3) else "A4=432",
        modal_count=24 + (8 * (variant % 10)),
        modal_q_min=4.0 + (variant % 5),
        modal_q_max=45.0 + (7.5 * (variant % 7)),
        modal_spread_cents=1.5 + (1.2 * (variant % 7)),
        modal_low_hz=55.0 + (12.0 * (variant % 6)),
        modal_high_hz=6500.0 + (1200.0 * (variant % 7)),
        fdn_lines=8 + ((variant % 4) * 4),
        fdn_matrix=matrix,
        fdn_stereo_inject=0.55 + (0.08 * (variant % 6)),
    )


def _format_length_tag(length_seconds: float) -> str:
    if abs(length_seconds - round(length_seconds)) < 1e-9:
        return f"{round(length_seconds)}s"
    return f"{str(length_seconds).replace('.', 'p')}s"


def generate_library(
    *,
    out_root: Path,
    sr: int,
    channels: int,
    out_format: OutFormat,
    seeds_per_shape: int,
) -> dict[str, object]:
    out_root.mkdir(parents=True, exist_ok=True)
    ext = "aiff" if out_format == "aiff" else out_format
    manifest_entries: list[dict[str, object]] = []
    counter = 0

    for bucket_name, lengths in BUCKET_LENGTHS.items():
        for mode in MODES:
            mode_dir = out_root / bucket_name / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            for length in lengths:
                for seed_idx in range(max(1, seeds_per_shape)):
                    counter += 1
                    seed = 10_000 + (counter * 17) + seed_idx
                    cfg = _build_config(
                        mode=mode,
                        length=length,
                        seed=seed,
                        sr=sr,
                        channels=channels,
                        variant=counter + seed_idx,
                    )
                    audio, out_sr, meta = generate_ir(cfg)
                    length_tag = _format_length_tag(length)
                    stem = f"ir_{mode}_{length_tag}_v{seed_idx + 1:02d}_s{seed}"
                    audio_path = mode_dir / f"{stem}.{ext}"
                    meta_path = mode_dir / f"{stem}.ir.meta.json"
                    sf.write(str(audio_path), audio, out_sr)
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                    manifest_entries.append(
                        {
                            "bucket": bucket_name,
                            "mode": mode,
                            "length_seconds": length,
                            "seed": seed,
                            "audio_file": str(audio_path.relative_to(out_root)),
                            "meta_file": str(meta_path.relative_to(out_root)),
                            "config": asdict(cfg),
                        }
                    )

    manifest = {
        "schema": "verbx-ir-library-v1",
        "root": str(out_root),
        "sample_rate": sr,
        "channels": channels,
        "format": out_format,
        "total_files": len(manifest_entries),
        "buckets": {name: list(lengths) for name, lengths in BUCKET_LENGTHS.items()},
        "modes": list(MODES),
        "entries": manifest_entries,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a curated folder-sorted IR library.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("IRs/library"),
        help="Output root directory for generated IR library.",
    )
    parser.add_argument("--sr", type=int, default=12_000, help="Output sample rate.")
    parser.add_argument("--channels", type=int, default=2, help="Output channel count.")
    parser.add_argument(
        "--format",
        choices=["flac", "wav", "aiff", "aif", "ogg", "caf"],
        default="flac",
        help="Audio output format/container.",
    )
    parser.add_argument(
        "--seeds-per-shape",
        type=int,
        default=1,
        help="How many seed variants to generate for each mode/length shape.",
    )
    args = parser.parse_args()

    manifest = generate_library(
        out_root=args.out,
        sr=max(8_000, int(args.sr)),
        channels=max(1, int(args.channels)),
        out_format=args.format,
        seeds_per_shape=max(1, int(args.seeds_per_shape)),
    )
    print(
        "Generated IR library:",
        f"{manifest['total_files']} files at {manifest['sample_rate']} Hz,",
        f"{manifest['channels']} ch, format={manifest['format']}",
    )


if __name__ == "__main__":
    main()
