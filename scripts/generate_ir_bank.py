#!/usr/bin/env python3
"""Generate a bank of synthetic IRs with varied parameters."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import soundfile as sf

from verbx.ir.generator import IRGenConfig, generate_ir

OutFormat = Literal["wav", "flac", "aiff", "aif", "ogg", "caf"]


def _build_config(index: int, length: float, sr: int, channels: int) -> IRGenConfig:
    mode_cycle = ("hybrid", "fdn", "stochastic", "modal")
    mode = mode_cycle[index % len(mode_cycle)]

    rt60 = max(8.0, min(140.0, length * (0.45 + (index % 7) * 0.08)))
    damping = min(0.85, 0.2 + ((index % 6) * 0.1))
    tilt = -2.0 + ((index % 9) * 0.5)
    diffusion = min(1.0, 0.25 + ((index % 5) * 0.15))
    density = 0.7 + ((index % 6) * 0.18)

    return IRGenConfig(
        mode=mode,
        length=length,
        sr=sr,
        channels=channels,
        seed=1000 + index,
        rt60=rt60,
        damping=damping,
        lowcut=40.0 + (index % 5) * 20.0,
        highcut=8000.0 + (index % 4) * 2000.0,
        tilt=tilt,
        normalize="peak",
        peak_dbfs=-1.0,
        target_lufs=None,
        true_peak=True,
        er_count=20 + (index % 12),
        er_max_delay_ms=70.0 + (index % 8) * 10.0,
        er_decay_shape="exp",
        er_stereo_width=0.8 + (index % 5) * 0.2,
        er_room=0.8 + (index % 6) * 0.2,
        diffusion=diffusion,
        mod_depth_ms=0.7 + (index % 5) * 0.6,
        mod_rate_hz=0.04 + (index % 6) * 0.03,
        density=density,
        tuning="A4=440" if index % 3 else "A4=432",
        modal_count=32 + (index % 7) * 8,
        modal_q_min=4.0 + (index % 4),
        modal_q_max=50.0 + (index % 6) * 10.0,
        modal_spread_cents=2.0 + (index % 6) * 1.5,
        modal_low_hz=55.0 + (index % 5) * 15.0,
        modal_high_hz=7000.0 + (index % 6) * 1200.0,
        fdn_lines=8,
        fdn_matrix=("hadamard", "householder", "random_orthogonal")[index % 3],
        fdn_stereo_inject=0.55 + (index % 5) * 0.1,
    )


def generate_bank(
    out_dir: Path,
    count: int,
    sr: int,
    channels: int,
    out_format: OutFormat,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    durations = [60.0, 75.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 300.0, 360.0]
    manifest: list[dict[str, object]] = []

    file_ext = ".aiff" if out_format == "aiff" else f".{out_format}"

    for idx in range(count):
        length = durations[idx % len(durations)]
        cfg = _build_config(idx, length=length, sr=sr, channels=channels)
        audio, out_sr, meta = generate_ir(cfg)

        stem = f"ir_{idx + 1:02d}_{cfg.mode}_{int(length)}s"
        path = out_dir / f"{stem}{file_ext}"
        sf.write(str(path), audio, out_sr)

        meta_path = out_dir / f"{stem}.ir.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        manifest.append(
            {
                "file": str(path.name),
                "meta": str(meta_path.name),
                "config": asdict(cfg),
            }
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 25-IR (or custom) synthetic bank.")
    parser.add_argument("--out", type=Path, default=Path("IRs/generated_25"), help="Output folder")
    parser.add_argument("--count", type=int, default=25, help="Number of IRs to generate")
    parser.add_argument("--sr", type=int, default=12000, help="Sample rate")
    parser.add_argument("--channels", type=int, default=2, help="Channel count")
    parser.add_argument(
        "--format",
        choices=["wav", "flac", "aiff", "aif", "ogg", "caf"],
        default="flac",
        help="Output container/codec extension",
    )
    args = parser.parse_args()

    generate_bank(
        out_dir=args.out,
        count=max(1, args.count),
        sr=max(8000, args.sr),
        channels=max(1, args.channels),
        out_format=args.format,
    )


if __name__ == "__main__":
    main()
