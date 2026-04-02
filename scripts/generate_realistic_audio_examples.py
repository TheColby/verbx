#!/usr/bin/env python3
"""Generate the full verbx audio example pack.

This script creates deterministic dry sources plus the rendered example assets used
throughout the README and docs. The delivery spec is intentionally fixed at a
higher-quality format than the original launch pack:

- 48 kHz sample rate
- 24-bit PCM WAV

The original script only covered the "realistic" subset. It now owns the whole
example pack so the repo has one canonical regeneration path instead of a pile
of hand-tweaked artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from verbx.api import generate_ir
from verbx.config import RenderConfig
from verbx.core.pipeline import run_render_pipeline
from verbx.ir.generator import IRGenConfig

OUTPUT_SUBTYPES = {
    "pcm16": "PCM_16",
    "pcm24": "PCM_24",
    "pcm32": "PCM_32",
    "float32": "FLOAT",
    "float64": "DOUBLE",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("examples/audio"),
        help="Output directory for dry/wet example assets.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48_000,
        help="Output sample rate for the example pack.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=5.5,
        help="Duration in seconds for each synthesized dry example.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260314,
        help="Deterministic RNG seed.",
    )
    parser.add_argument(
        "--output-subtype",
        choices=sorted(OUTPUT_SUBTYPES.keys()),
        default="pcm24",
        help="Output subtype for the written examples.",
    )
    parser.add_argument(
        "--peak-target-dbfs",
        type=float,
        default=-2.0,
        help="Target sample-peak normalization applied to rendered wet examples.",
    )
    parser.add_argument(
        "--skip-renders",
        action="store_true",
        help="Only write dry source examples, utility files, IR, and metadata scaffolding.",
    )
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    sr = int(max(8_000, args.sample_rate))
    duration_s = float(max(1.0, args.duration_s))
    seed = int(args.seed)
    output_subtype = str(args.output_subtype)
    peak_target_dbfs = float(args.peak_target_dbfs)
    rng = np.random.default_rng(seed)

    dry_sources: dict[str, np.ndarray] = {
        "dry_click.wav": _synth_dry_click(sr=sr, seconds=2.0),
        "realistic_speech_dry.wav": _synth_speech_like(sr=sr, seconds=duration_s, rng=rng),
        "realistic_music_dry.wav": _synth_music_like(sr=sr, seconds=duration_s, rng=rng),
        "realistic_drums_dry.wav": _synth_drums_like(sr=sr, seconds=duration_s, rng=rng),
    }
    for filename, audio in dry_sources.items():
        _write_audio_file(out_dir / filename, audio, sr, output_subtype)

    ir_audio, ir_sr, ir_meta = generate_ir(
        IRGenConfig(
            mode="hybrid",
            length=1.5,
            sr=sr,
            channels=1,
            seed=seed + 17,
            rt60=1.1,
            damping=0.5,
            diffusion=0.6,
            normalize="peak",
            peak_dbfs=peak_target_dbfs,
        )
    )
    _write_audio_file(out_dir / "hybrid_ir_short.wav", ir_audio, ir_sr, output_subtype)

    render_events: list[dict[str, Any]] = []
    if not bool(args.skip_renders):
        for outfile_name, infile_name, config in _render_plan(
            repo_root=repo_root,
            sr=sr,
            output_subtype=output_subtype,
            peak_target_dbfs=peak_target_dbfs,
        ):
            outfile = out_dir / outfile_name
            infile = out_dir / infile_name
            run_render_pipeline(infile=infile, outfile=outfile, config=config)
            render_events.append(
                {
                    "file": outfile_name,
                    "source": infile_name,
                    "config": _render_config_snapshot(config),
                }
            )

    full_manifest = {
        "generated_by": "scripts/generate_realistic_audio_examples.py",
        "seed": seed,
        "sample_rate": sr,
        "output_subtype": output_subtype,
        "peak_target_dbfs": peak_target_dbfs,
        "duration_seconds": duration_s,
        "dry_assets": sorted(dry_sources.keys()),
        "utility_assets": ["hybrid_ir_short.wav"],
        "rendered_assets": render_events,
        "files": _collect_audio_metadata(out_dir),
        "hybrid_ir": {
            "file": "hybrid_ir_short.wav",
            "meta": ir_meta,
        },
    }
    (out_dir / "example_pack.meta.json").write_text(
        json.dumps(full_manifest, indent=2),
        encoding="utf-8",
    )

    realistic_manifest = {
        "generated_by": "scripts/generate_realistic_audio_examples.py",
        "seed": seed,
        "sample_rate": sr,
        "output_subtype": output_subtype,
        "peak_target_dbfs": peak_target_dbfs,
        "duration_seconds": duration_s,
        "dry_assets": [
            "realistic_drums_dry.wav",
            "realistic_music_dry.wav",
            "realistic_speech_dry.wav",
        ],
        "wet_assets": [
            event
            for event in render_events
            if event["file"]
            in {
                "realistic_speech_room.wav",
                "realistic_music_hall.wav",
                "realistic_drums_room.wav",
            }
        ],
    }
    (out_dir / "realistic_examples.meta.json").write_text(
        json.dumps(realistic_manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote audio example pack to: {out_dir}")
    return 0


def _render_plan(
    *,
    repo_root: Path,
    sr: int,
    output_subtype: str,
    peak_target_dbfs: float,
) -> list[tuple[str, str, RenderConfig]]:
    base: dict[str, Any] = {
        "target_sr": sr,
        "output_subtype": output_subtype,
        "normalize_stage": "none",
        "output_peak_norm": "target",
        "output_peak_target_dbfs": peak_target_dbfs,
        "silent": True,
        "progress": False,
    }
    return [
        (
            "dry_click_reverbed.wav",
            "dry_click.wav",
            RenderConfig(
                engine="algo",
                rt60=1.35,
                wet=1.0,
                dry=0.0,
                pre_delay_ms=0.0,
                damping=0.58,
                width=0.0,
                fdn_lines=8,
                fdn_matrix="hadamard",
                **base,
            ),
        ),
        (
            "realistic_speech_room.wav",
            "realistic_speech_dry.wav",
            RenderConfig(
                engine="algo",
                rt60=2.1,
                wet=0.46,
                dry=0.84,
                pre_delay_ms=22.0,
                damping=0.52,
                width=1.15,
                fdn_matrix="hadamard",
                **base,
            ),
        ),
        (
            "realistic_music_hall.wav",
            "realistic_music_dry.wav",
            RenderConfig(
                engine="conv",
                ir=str((repo_root / "IRs/ir_hybrid_60s.flac").resolve()),
                wet=0.58,
                dry=0.74,
                tail_limit=3.5,
                partition_size=8192,
                **base,
            ),
        ),
        (
            "realistic_drums_room.wav",
            "realistic_drums_dry.wav",
            RenderConfig(
                engine="algo",
                rt60=1.05,
                wet=0.34,
                dry=0.92,
                pre_delay_ms=9.0,
                damping=0.63,
                width=1.02,
                fdn_matrix="householder",
                **base,
            ),
        ),
        (
            "extreme_cathedral_drums.wav",
            "realistic_drums_dry.wav",
            RenderConfig(
                engine="algo",
                rt60=8.0,
                wet=0.85,
                dry=0.25,
                pre_delay_ms=45.0,
                fdn_lines=16,
                fdn_matrix="hadamard",
                lowcut=80.0,
                highcut=12_000.0,
                **base,
            ),
        ),
        (
            "extreme_shimmer_music.wav",
            "realistic_music_dry.wav",
            RenderConfig(
                engine="algo",
                rt60=6.0,
                wet=0.8,
                dry=0.3,
                shimmer=True,
                shimmer_semitones=12.0,
                shimmer_mix=0.35,
                shimmer_feedback=0.65,
                pre_delay_ms=30.0,
                fdn_lines=16,
                **base,
            ),
        ),
        (
            "extreme_plate_speech.wav",
            "realistic_speech_dry.wav",
            RenderConfig(
                engine="algo",
                rt60=1.8,
                wet=0.7,
                dry=0.4,
                fdn_matrix="circulant",
                lowcut=200.0,
                highcut=6_000.0,
                pre_delay_ms=12.0,
                **base,
            ),
        ),
        (
            "extreme_frozen_music.wav",
            "realistic_music_dry.wav",
            RenderConfig(
                engine="algo",
                rt60=30.0,
                wet=0.95,
                dry=0.1,
                fdn_lines=32,
                pre_delay_ms=60.0,
                fdn_matrix="hadamard",
                **base,
            ),
        ),
        (
            "lucier_sitting_room.wav",
            "realistic_speech_dry.wav",
            RenderConfig(
                engine="algo",
                rt60=4.5,
                wet=1.0,
                dry=0.0,
                fdn_lines=16,
                fdn_matrix="hadamard",
                repeat=7,
                lowcut=60.0,
                **base,
            ),
        ),
        (
            "eno_discreet_music.wav",
            "realistic_music_dry.wav",
            RenderConfig(
                engine="algo",
                rt60=12.0,
                wet=0.92,
                dry=0.08,
                fdn_lines=16,
                fdn_matrix="hadamard",
                pre_delay_ms=35.0,
                damping=0.25,
                lowcut=50.0,
                **base,
            ),
        ),
        (
            "oliveros_deep_listening.wav",
            "realistic_music_dry.wav",
            RenderConfig(
                engine="algo",
                rt60=18.0,
                wet=0.95,
                dry=0.10,
                fdn_lines=32,
                fdn_matrix="hadamard",
                pre_delay_ms=55.0,
                damping=0.15,
                lowcut=30.0,
                **base,
            ),
        ),
        (
            "fripp_frippertronics.wav",
            "realistic_music_dry.wav",
            RenderConfig(
                engine="algo",
                rt60=8.0,
                wet=0.82,
                dry=0.28,
                fdn_lines=16,
                fdn_matrix="hadamard",
                shimmer=True,
                shimmer_semitones=12.0,
                shimmer_mix=0.45,
                shimmer_feedback=0.78,
                pre_delay_ms=25.0,
                **base,
            ),
        ),
        (
            "mbv_shoegaze.wav",
            "realistic_music_dry.wav",
            RenderConfig(
                engine="algo",
                rt60=5.0,
                wet=0.88,
                dry=0.22,
                fdn_lines=16,
                fdn_matrix="circulant",
                shimmer=True,
                shimmer_semitones=12.0,
                shimmer_mix=0.55,
                shimmer_feedback=0.72,
                pre_delay_ms=8.0,
                lowcut=80.0,
                **base,
            ),
        ),
        (
            "reich_phase_drums.wav",
            "realistic_drums_dry.wav",
            RenderConfig(
                engine="algo",
                rt60=0.7,
                wet=0.55,
                dry=0.50,
                fdn_lines=8,
                fdn_matrix="circulant",
                pre_delay_ms=18.0,
                damping=0.6,
                lowcut=60.0,
                **base,
            ),
        ),
        (
            "radigue_drone.wav",
            "realistic_music_dry.wav",
            RenderConfig(
                engine="algo",
                rt60=45.0,
                wet=0.97,
                dry=0.05,
                fdn_lines=32,
                fdn_matrix="hadamard",
                damping=0.10,
                lowcut=20.0,
                **base,
            ),
        ),
        (
            "feldman_sparse_room.wav",
            "realistic_music_dry.wav",
            RenderConfig(
                engine="algo",
                rt60=3.8,
                wet=0.52,
                dry=0.52,
                fdn_lines=8,
                fdn_matrix="circulant",
                pre_delay_ms=30.0,
                damping=0.50,
                allpass_stages=4,
                **base,
            ),
        ),
    ]


def _write_audio_file(path: Path, audio: np.ndarray, sr: int, subtype_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subtype = _resolve_sf_subtype(subtype_mode)
    sf.write(
        str(path),
        np.asarray(audio, dtype=np.float64),
        int(sr),
        subtype=subtype,
    )


def _resolve_sf_subtype(mode: str) -> str:
    try:
        return OUTPUT_SUBTYPES[mode]
    except KeyError as exc:  # pragma: no cover - argparse keeps this honest.
        raise ValueError(f"unsupported output subtype mode: {mode}") from exc


def _render_config_snapshot(config: RenderConfig) -> dict[str, Any]:
    payload = asdict(config)
    keys = (
        "engine",
        "rt60",
        "wet",
        "dry",
        "pre_delay_ms",
        "damping",
        "width",
        "fdn_lines",
        "fdn_matrix",
        "shimmer",
        "shimmer_mix",
        "shimmer_feedback",
        "lowcut",
        "highcut",
        "ir",
        "tail_limit",
        "partition_size",
        "target_sr",
        "output_subtype",
        "output_peak_norm",
        "output_peak_target_dbfs",
    )
    return {key: payload.get(key) for key in keys}


def _collect_audio_metadata(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*.wav")):
        info = sf.info(str(path))
        rows.append(
            {
                "file": path.name,
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "subtype": str(info.subtype),
                "seconds": float(info.duration),
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_normalize(audio: np.ndarray, target_peak: float = 0.90) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size > 0 else 0.0
    if peak <= 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return np.asarray((target_peak / peak) * x, dtype=np.float64)


def _synth_dry_click(*, sr: int, seconds: float) -> np.ndarray:
    n = max(8, int(sr * seconds))
    click = np.zeros(n, dtype=np.float64)
    click[0] = 0.80
    return click


def _synth_speech_like(
    *,
    sr: int,
    seconds: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float64) / float(sr)
    f0 = 115.0 + (18.0 * np.sin(2.0 * np.pi * 0.42 * t)) + (7.0 * np.sin(2.0 * np.pi * 0.18 * t))
    f0 = np.clip(f0, 80.0, 180.0)

    vowels = np.asarray(
        [
            [730.0, 1090.0, 2440.0],
            [530.0, 1840.0, 2480.0],
            [300.0, 2200.0, 3000.0],
            [570.0, 840.0, 2410.0],
            [440.0, 1020.0, 2240.0],
        ],
        dtype=np.float64,
    )
    seg_phase = t / 0.42
    seg_idx = np.floor(seg_phase).astype(np.int64) % vowels.shape[0]
    seg_next = (seg_idx + 1) % vowels.shape[0]
    seg_alpha = seg_phase - np.floor(seg_phase)
    formants = ((1.0 - seg_alpha)[:, np.newaxis] * vowels[seg_idx]) + (
        seg_alpha[:, np.newaxis] * vowels[seg_next]
    )
    f1 = formants[:, 0]
    f2 = formants[:, 1]
    f3 = formants[:, 2]

    voice = np.zeros(n, dtype=np.float64)
    for harmonic in range(1, 22):
        inst_freq = harmonic * f0
        phase = 2.0 * np.pi * np.cumsum(inst_freq / float(sr), dtype=np.float64)
        g1 = np.exp(-0.5 * np.square((inst_freq - f1) / 120.0))
        g2 = np.exp(-0.5 * np.square((inst_freq - f2) / 180.0))
        g3 = np.exp(-0.5 * np.square((inst_freq - f3) / 230.0))
        gain = ((1.8 * g1) + (1.1 * g2) + (0.85 * g3)) / float(harmonic)
        voice += gain * np.sin(phase)

    syllable_env = np.power(np.clip(np.sin(2.0 * np.pi * 2.55 * t), 0.0, 1.0), 1.7)
    phrase_env = np.power(np.hanning(max(8, n)), 0.35)
    breath = np.diff(
        np.concatenate(([0.0], rng.standard_normal(n, dtype=np.float64))),
        n=1,
    )
    mono = (voice * (0.32 + (0.68 * syllable_env)) * phrase_env) + (
        0.012 * breath * syllable_env
    )
    mono = _safe_normalize(mono, target_peak=0.86)
    width = 0.035 * np.sin(2.0 * np.pi * 0.24 * t)
    left = mono * (1.0 + width)
    right = mono * (1.0 - width)
    return np.column_stack((left, right))


def _synth_music_like(
    *,
    sr: int,
    seconds: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(sr * seconds)
    out = np.zeros((n, 2), dtype=np.float64)

    def midi_to_hz(midi: float) -> float:
        return float(440.0 * np.power(2.0, (midi - 69.0) / 12.0))

    progression = [
        (0.00, [57, 60, 64, 67]),
        (1.35, [55, 59, 62, 67]),
        (2.70, [53, 57, 60, 65]),
        (4.05, [52, 55, 59, 64]),
    ]
    note_len = 1.8
    for start_s, chord in progression:
        start = int(start_s * sr)
        length = int(min(n - start, note_len * sr))
        if length <= 0:
            continue
        tt = np.arange(length, dtype=np.float64) / float(sr)
        env = (1.0 - np.exp(-tt * 55.0)) * np.exp(-tt * 2.1)
        env += 0.10 * np.exp(-tt * 8.0)
        for note_idx, midi in enumerate(chord):
            freq = midi_to_hz(float(midi))
            detune_cents = float(rng.uniform(-4.0, 4.0))
            detune = np.power(2.0, detune_cents / 1200.0)
            f_base = freq * detune
            phase_a = 2.0 * np.pi * (f_base * tt)
            phase_b = 2.0 * np.pi * ((f_base * 1.0028) * tt)
            tone = (
                0.72 * np.sin(phase_a)
                + 0.42 * np.sin(2.0 * phase_a + 0.4)
                + 0.21 * np.sin(3.0 * phase_a + 1.1)
                + 0.22 * np.sin(phase_b + 0.7)
            )
            attack_noise = (
                0.016 * rng.standard_normal(length, dtype=np.float64) * np.exp(-tt * 90.0)
            )
            voice = (tone + attack_noise) * env / float(1.0 + (0.22 * note_idx))
            pan = np.clip(-0.65 + (1.30 * note_idx / max(1, len(chord) - 1)), -0.95, 0.95)
            pan_l = np.sqrt(0.5 * (1.0 - pan))
            pan_r = np.sqrt(0.5 * (1.0 + pan))
            out[start : start + length, 0] += pan_l * voice
            out[start : start + length, 1] += pan_r * voice

    trem = 1.0 + (0.07 * np.sin(2.0 * np.pi * 4.2 * np.arange(n, dtype=np.float64) / float(sr)))
    out[:, 0] *= trem
    out[:, 1] *= trem[::-1]
    return _safe_normalize(out, target_peak=0.90)


def _synth_drums_like(
    *,
    sr: int,
    seconds: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(sr * seconds)
    out = np.zeros((n, 2), dtype=np.float64)
    bpm = 102.0
    beat_s = 60.0 / bpm

    def add_event(start_s: float, signal: np.ndarray, pan: float = 0.0) -> None:
        start = int(start_s * sr)
        if start >= n:
            return
        length = min(signal.shape[0], n - start)
        if length <= 0:
            return
        pan_l = np.sqrt(0.5 * (1.0 - pan))
        pan_r = np.sqrt(0.5 * (1.0 + pan))
        out[start : start + length, 0] += pan_l * signal[:length]
        out[start : start + length, 1] += pan_r * signal[:length]

    def kick() -> np.ndarray:
        length = int(0.26 * sr)
        tt = np.arange(length, dtype=np.float64) / float(sr)
        sweep = 42.0 + (120.0 * np.exp(-tt * 20.0))
        phase = 2.0 * np.pi * np.cumsum(sweep / float(sr), dtype=np.float64)
        env = np.exp(-tt * 10.0)
        click = 0.10 * np.exp(-tt * 110.0) * np.sin(2.0 * np.pi * 1600.0 * tt)
        return (0.95 * np.sin(phase) * env) + click

    def snare() -> np.ndarray:
        length = int(0.23 * sr)
        tt = np.arange(length, dtype=np.float64) / float(sr)
        noise = rng.standard_normal(length, dtype=np.float64)
        noise = np.diff(np.concatenate(([0.0], noise)))
        env = np.exp(-tt * 16.0)
        tone = 0.30 * np.sin(2.0 * np.pi * 192.0 * tt) * np.exp(-tt * 18.0)
        body = 0.18 * np.sin(2.0 * np.pi * 330.0 * tt) * np.exp(-tt * 26.0)
        return (0.68 * noise * env) + tone + body

    def hat(open_hat: bool) -> np.ndarray:
        length = int((0.30 if open_hat else 0.09) * sr)
        tt = np.arange(length, dtype=np.float64) / float(sr)
        noise = rng.standard_normal(length, dtype=np.float64)
        bright = np.diff(np.concatenate(([0.0], noise)))
        bright = np.diff(np.concatenate(([0.0], bright)))
        env = np.exp(-tt * (12.0 if open_hat else 45.0))
        metallic = (
            0.12 * np.sin(2.0 * np.pi * 6300.0 * tt)
            + 0.09 * np.sin(2.0 * np.pi * 8100.0 * tt + 0.8)
        )
        return (0.42 * bright * env) + (metallic * env)

    kicks = [0.0, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0]
    snares = [1.0, 3.0]
    hats = [0.0 + (0.5 * i) for i in range(int(np.ceil(seconds / (beat_s * 0.5))) + 2)]
    sec_per_beat = beat_s
    for beat in kicks:
        add_event(beat * sec_per_beat, kick(), pan=0.0)
    for beat in snares:
        add_event(beat * sec_per_beat, snare(), pan=0.05)
    for idx, beat in enumerate(hats):
        open_hat = (idx % 8) == 7
        pan = -0.22 if (idx % 2 == 0) else 0.22
        add_event(beat * sec_per_beat, hat(open_hat=open_hat), pan=pan)

    return _safe_normalize(out, target_peak=0.91)


if __name__ == "__main__":
    raise SystemExit(_main())
