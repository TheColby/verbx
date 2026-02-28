"""IR synthesis entrypoints and cache helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import soundfile as sf

from verbx.ir.early_reflections import generate_early_reflections
from verbx.ir.metrics import analyze_ir
from verbx.ir.modes_fdn import generate_fdn_ir
from verbx.ir.modes_modal import generate_modal_ir
from verbx.ir.modes_stochastic import generate_stochastic_ir
from verbx.ir.shaping import apply_ir_shaping
from verbx.ir.tuning import apply_harmonic_alignment

AudioArray = npt.NDArray[np.float32]
IRMode = Literal["fdn", "stochastic", "modal", "hybrid"]
IRNormalize = Literal["none", "peak", "rms"]


@dataclass(slots=True)
class IRGenConfig:
    """Configuration for IR generation modes and shaping."""

    mode: IRMode = "hybrid"
    length: float = 60.0
    sr: int = 48_000
    channels: int = 2
    seed: int = 0

    rt60: float | None = 12.0
    rt60_low: float | None = None
    rt60_high: float | None = None
    damping: float = 0.4
    lowcut: float | None = None
    highcut: float | None = None
    tilt: float = 0.0
    normalize: IRNormalize = "peak"
    peak_dbfs: float = -1.0
    target_lufs: float | None = None
    true_peak: bool = True

    er_count: int = 24
    er_max_delay_ms: float = 90.0
    er_decay_shape: str = "exp"
    er_stereo_width: float = 1.0
    er_room: float = 1.0

    diffusion: float = 0.5
    mod_depth_ms: float = 1.5
    mod_rate_hz: float = 0.12
    density: float = 1.0

    tuning: str = "A4=440"
    modal_count: int = 48
    modal_q_min: float = 5.0
    modal_q_max: float = 60.0
    modal_spread_cents: float = 5.0
    modal_low_hz: float = 80.0
    modal_high_hz: float = 12_000.0

    fdn_lines: int = 8
    fdn_matrix: str = "hadamard"
    fdn_stereo_inject: float = 1.0

    f0_hz: float | None = None
    harmonic_targets_hz: tuple[float, ...] = ()
    harmonic_align_strength: float = 0.75


def generate_ir(config: IRGenConfig) -> tuple[AudioArray, int, dict[str, Any]]:
    """Generate IR and return audio, sample rate, and metadata."""
    length_samples = max(1, int(config.length * config.sr))
    rt60_low, rt60_high = _resolve_rt60_band(config)

    if config.mode == "fdn":
        ir = generate_fdn_ir(
            length_samples=length_samples,
            sr=config.sr,
            channels=config.channels,
            rt60=config.rt60 if config.rt60 is not None else rt60_high,
            damping=config.damping,
            mod_depth_ms=config.mod_depth_ms,
            mod_rate_hz=config.mod_rate_hz,
            fdn_lines=config.fdn_lines,
            fdn_matrix=config.fdn_matrix,
            fdn_stereo_inject=config.fdn_stereo_inject,
            seed=config.seed,
        )
    elif config.mode == "stochastic":
        ir = generate_stochastic_ir(
            length_samples=length_samples,
            sr=config.sr,
            channels=config.channels,
            rt60_low=rt60_low,
            rt60_high=rt60_high,
            damping=config.damping,
            diffusion=config.diffusion,
            density=config.density,
            seed=config.seed,
        )
    elif config.mode == "modal":
        ir = generate_modal_ir(
            length_samples=length_samples,
            sr=config.sr,
            channels=config.channels,
            rt60=config.rt60 if config.rt60 is not None else rt60_high,
            seed=config.seed,
            tuning=config.tuning,
            modal_count=config.modal_count,
            modal_q_min=config.modal_q_min,
            modal_q_max=config.modal_q_max,
            modal_spread_cents=config.modal_spread_cents,
            modal_low_hz=config.modal_low_hz,
            modal_high_hz=config.modal_high_hz,
            f0_hz=config.f0_hz,
            harmonic_targets_hz=config.harmonic_targets_hz,
            align_strength=config.harmonic_align_strength,
        )
    else:
        # Hybrid: early reflections + blended stochastic/modal/fdn tails.
        rng = np.random.default_rng(config.seed)
        early = generate_early_reflections(
            sr=config.sr,
            channels=config.channels,
            er_count=config.er_count,
            er_max_delay_ms=config.er_max_delay_ms,
            er_decay_shape=config.er_decay_shape,
            er_stereo_width=config.er_stereo_width,
            er_room=config.er_room,
            rng=rng,
        )

        stoch = generate_stochastic_ir(
            length_samples=length_samples,
            sr=config.sr,
            channels=config.channels,
            rt60_low=rt60_low,
            rt60_high=rt60_high,
            damping=config.damping,
            diffusion=config.diffusion,
            density=config.density,
            seed=config.seed + 11,
        )
        modal = generate_modal_ir(
            length_samples=length_samples,
            sr=config.sr,
            channels=config.channels,
            rt60=config.rt60 if config.rt60 is not None else rt60_high,
            seed=config.seed + 17,
            tuning=config.tuning,
            modal_count=max(8, config.modal_count // 2),
            modal_q_min=config.modal_q_min,
            modal_q_max=config.modal_q_max,
            modal_spread_cents=config.modal_spread_cents,
            modal_low_hz=config.modal_low_hz,
            modal_high_hz=config.modal_high_hz,
            f0_hz=config.f0_hz,
            harmonic_targets_hz=config.harmonic_targets_hz,
            align_strength=config.harmonic_align_strength,
        )
        fdn = generate_fdn_ir(
            length_samples=length_samples,
            sr=config.sr,
            channels=config.channels,
            rt60=config.rt60 if config.rt60 is not None else rt60_high,
            damping=config.damping,
            mod_depth_ms=config.mod_depth_ms,
            mod_rate_hz=config.mod_rate_hz,
            fdn_lines=config.fdn_lines,
            fdn_matrix=config.fdn_matrix,
            fdn_stereo_inject=config.fdn_stereo_inject,
            seed=config.seed + 23,
        )

        ir = (0.55 * stoch) + (0.25 * modal) + (0.20 * fdn)
        early_len = min(early.shape[0], ir.shape[0])
        ir[:early_len, :] += early[:early_len, :]
        ir = np.asarray(ir, dtype=np.float32)

    ir = apply_harmonic_alignment(
        ir=ir,
        sr=config.sr,
        f0_hz=config.f0_hz,
        harmonic_targets_hz=config.harmonic_targets_hz,
        strength=config.harmonic_align_strength,
    )

    shaped = apply_ir_shaping(
        ir,
        sr=config.sr,
        damping=config.damping,
        lowcut=config.lowcut,
        highcut=config.highcut,
        tilt=config.tilt,
        normalize=config.normalize,
        peak_dbfs=config.peak_dbfs,
        target_lufs=config.target_lufs,
        use_true_peak=config.true_peak,
    )

    meta: dict[str, Any] = {
        "version": "0.3.0",
        "mode": config.mode,
        "seed": config.seed,
        "params": asdict(config),
        "metrics": analyze_ir(shaped, config.sr),
    }

    return shaped, config.sr, meta


def generate_or_load_cached_ir(
    config: IRGenConfig,
    cache_dir: Path,
) -> tuple[AudioArray, int, dict[str, Any], Path, bool]:
    """Generate or load cached IR and return audio/sr/meta/path/cache_hit."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(config)
    wav_path = cache_dir / f"{key}.wav"
    meta_path = cache_dir / f"{key}.meta.json"

    if wav_path.exists() and meta_path.exists():
        audio, sr = sf.read(str(wav_path), always_2d=True, dtype="float32")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return np.asarray(audio, dtype=np.float32), int(sr), meta, wav_path, True

    audio, sr, meta = generate_ir(config)
    sf.write(str(wav_path), np.asarray(audio, dtype=np.float32), sr)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return audio, sr, meta, wav_path, False


def write_ir_artifacts(
    out_path: Path,
    audio: AudioArray,
    sr: int,
    meta: dict[str, Any],
    silent: bool,
) -> None:
    """Write IR wav and optional metadata sidecar."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), np.asarray(audio, dtype=np.float32), sr)

    if not silent:
        meta_path = out_path.with_suffix(f"{out_path.suffix}.ir.meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _resolve_rt60_band(config: IRGenConfig) -> tuple[float, float]:
    if config.rt60 is not None:
        value = max(0.1, float(config.rt60))
        return value, value

    low = 6.0 if config.rt60_low is None else max(0.1, float(config.rt60_low))
    high = 12.0 if config.rt60_high is None else max(low, float(config.rt60_high))
    return low, high


def _cache_key(config: IRGenConfig) -> str:
    payload = asdict(config)
    payload["_schema"] = "verbx-ir-v0.3"
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
