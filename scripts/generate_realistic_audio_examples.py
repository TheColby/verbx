#!/usr/bin/env python3
"""Generate realistic-ish demo program material and reverb examples.

This script creates three dry sources:
- speech-like
- harmonic music-like
- drum-loop-like

It then renders reverb versions using verbx so docs can link to concrete,
audible examples without requiring users to source input content first.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from verbx.config import RenderConfig
from verbx.core.pipeline import run_render_pipeline


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
        default=24_000,
        help="Sample rate for synthesized dry material.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=5.5,
        help="Duration in seconds for each dry example.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260314,
        help="Deterministic RNG seed.",
    )
    parser.add_argument(
        "--skip-renders",
        action="store_true",
        help="Only write dry source examples and metadata.",
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
    rng = np.random.default_rng(seed)

    dry_sources: dict[str, np.ndarray] = {
        "realistic_speech_dry.wav": _synth_speech_like(sr=sr, seconds=duration_s, rng=rng),
        "realistic_music_dry.wav": _synth_music_like(sr=sr, seconds=duration_s, rng=rng),
        "realistic_drums_dry.wav": _synth_drums_like(sr=sr, seconds=duration_s, rng=rng),
    }

    for filename, audio in dry_sources.items():
        _write_wav(out_dir / filename, audio, sr)

    wet_assets: list[dict[str, Any]] = []
    if not bool(args.skip_renders):
        render_plan = [
            (
                out_dir / "realistic_speech_room.wav",
                out_dir / "realistic_speech_dry.wav",
                RenderConfig(
                    engine="algo",
                    rt60=2.1,
                    wet=0.46,
                    dry=0.84,
                    pre_delay_ms=22.0,
                    damping=0.52,
                    width=1.15,
                    fdn_matrix="hadamard",
                    output_subtype="pcm16",
                    normalize_stage="none",
                    output_peak_norm="input",
                    silent=True,
                    progress=False,
                ),
            ),
            (
                out_dir / "realistic_music_hall.wav",
                out_dir / "realistic_music_dry.wav",
                RenderConfig(
                    engine="conv",
                    ir=str((repo_root / "IRs/ir_hybrid_60s.flac").resolve()),
                    wet=0.58,
                    dry=0.74,
                    tail_limit=3.5,
                    partition_size=8192,
                    output_subtype="pcm16",
                    normalize_stage="none",
                    output_peak_norm="input",
                    silent=True,
                    progress=False,
                ),
            ),
            (
                out_dir / "realistic_drums_room.wav",
                out_dir / "realistic_drums_dry.wav",
                RenderConfig(
                    engine="algo",
                    rt60=1.05,
                    wet=0.34,
                    dry=0.92,
                    pre_delay_ms=9.0,
                    damping=0.63,
                    width=1.02,
                    fdn_matrix="householder",
                    output_subtype="pcm16",
                    normalize_stage="none",
                    output_peak_norm="input",
                    silent=True,
                    progress=False,
                ),
            ),
        ]
        for outfile, infile, config in render_plan:
            run_render_pipeline(infile=infile, outfile=outfile, config=config)
            wet_assets.append(
                {
                    "file": outfile.name,
                    "source": infile.name,
                    "engine": config.engine,
                    "config": _render_config_snapshot(config),
                }
            )

    metadata = {
        "generated_by": "scripts/generate_realistic_audio_examples.py",
        "seed": seed,
        "sample_rate": sr,
        "duration_seconds": duration_s,
        "dry_assets": sorted(dry_sources.keys()),
        "wet_assets": wet_assets,
    }
    (out_dir / "realistic_examples.meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote realistic audio examples to: {out_dir}")
    return 0


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.asarray(audio, dtype=np.float64),
        int(sr),
        subtype="PCM_16",
    )


def _render_config_snapshot(config: RenderConfig) -> dict[str, Any]:
    payload = asdict(config)
    # Keep metadata compact and useful; no one needs 100 fields for demo provenance.
    keys = (
        "engine",
        "rt60",
        "wet",
        "dry",
        "pre_delay_ms",
        "damping",
        "width",
        "fdn_matrix",
        "ir",
        "tail_limit",
        "partition_size",
    )
    return {key: payload.get(key) for key in keys}


def _safe_normalize(audio: np.ndarray, target_peak: float = 0.90) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size > 0 else 0.0
    if peak <= 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return np.asarray((target_peak / peak) * x, dtype=np.float64)


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
            [730.0, 1090.0, 2440.0],  # /a/
            [530.0, 1840.0, 2480.0],  # /e/
            [300.0, 2200.0, 3000.0],  # /i/
            [570.0, 840.0, 2410.0],  # /o/
            [440.0, 1020.0, 2240.0],  # /u/
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

    # Four-chord loop with gentle inversions for a realistic pad/pluck hybrid vibe.
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
