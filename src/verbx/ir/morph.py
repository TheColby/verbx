"""IR morphing and blend-cache helpers for Track D workflows."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy.signal import resample_poly

from verbx.ir.metrics import analyze_ir

AudioArray = npt.NDArray[np.float64]
IRMorphMode = Literal["linear", "equal-power", "spectral", "envelope-aware"]
IRMorphMismatchPolicy = Literal["coerce", "strict"]

_MORPH_MODE_CHOICES = {
    "linear",
    "equal-power",
    "spectral",
    "envelope-aware",
}
_MORPH_MISMATCH_POLICY_CHOICES = {
    "coerce",
    "strict",
}
_CACHE_KEY_SCHEMA = "ir-morph-v3"


@dataclass(slots=True)
class IRMorphConfig:
    """Configuration for pairwise IR morphing."""

    mode: IRMorphMode = "equal-power"
    alpha: float = 0.5
    early_ms: float = 80.0
    early_alpha: float | None = None
    late_alpha: float | None = None
    align_decay: bool = True
    phase_coherence: float = 0.75
    spectral_smooth_bins: int = 3
    mismatch_policy: IRMorphMismatchPolicy = "coerce"


def normalize_ir_morph_mode_name(value: str) -> str:
    """Normalize IR morph mode names to canonical CLI/API identifiers."""
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in {"equalpower", "eqp"}:
        return "equal-power"
    if normalized in {"envelope", "envelopeaware", "env-aware"}:
        return "envelope-aware"
    return normalized


def validate_ir_morph_mode_name(value: str) -> str:
    """Validate and return normalized IR morph mode name."""
    normalized = normalize_ir_morph_mode_name(value)
    if normalized not in _MORPH_MODE_CHOICES:
        options = ", ".join(sorted(_MORPH_MODE_CHOICES))
        msg = f"IR morph mode must be one of: {options}."
        raise ValueError(msg)
    return normalized


def normalize_ir_morph_mismatch_policy_name(value: str) -> str:
    """Normalize mismatch-policy names to canonical CLI/API identifiers."""
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in {"align", "auto", "coercive"}:
        return "coerce"
    return normalized


def validate_ir_morph_mismatch_policy_name(value: str) -> str:
    """Validate and return normalized mismatch-policy name."""
    normalized = normalize_ir_morph_mismatch_policy_name(value)
    if normalized not in _MORPH_MISMATCH_POLICY_CHOICES:
        options = ", ".join(sorted(_MORPH_MISMATCH_POLICY_CHOICES))
        msg = f"IR morph mismatch policy must be one of: {options}."
        raise ValueError(msg)
    return normalized


def resolve_blend_mix_values(raw_mix: Sequence[float], n_blends: int) -> tuple[float, ...]:
    """Resolve repeatable blend weights for ``n_blends`` additional IR paths."""
    if n_blends < 0:
        raise ValueError("n_blends must be >= 0.")
    if n_blends == 0:
        return ()
    if len(raw_mix) == 0:
        return tuple(1.0 for _ in range(n_blends))
    if len(raw_mix) == 1 and n_blends > 1:
        value = float(np.clip(float(raw_mix[0]), 0.0, 1.0))
        return tuple(value for _ in range(n_blends))
    if len(raw_mix) != n_blends:
        msg = f"Expected either 1 or {n_blends} blend-weight values, got {len(raw_mix)}."
        raise ValueError(msg)
    return tuple(float(np.clip(float(v), 0.0, 1.0)) for v in raw_mix)


def morph_ir_arrays(
    ir_a: AudioArray,
    ir_b: AudioArray,
    *,
    sr: int,
    config: IRMorphConfig,
) -> tuple[AudioArray, dict[str, Any]]:
    """Morph two in-memory IR arrays and return output + quality summary."""
    policy = validate_ir_morph_mismatch_policy_name(config.mismatch_policy)
    a, b = _align_ir_pair(
        np.asarray(ir_a, dtype=np.float64),
        np.asarray(ir_b, dtype=np.float64),
        mismatch_policy=cast(IRMorphMismatchPolicy, policy),
    )
    mode = validate_ir_morph_mode_name(config.mode)
    alpha = float(np.clip(config.alpha, 0.0, 1.0))
    early_alpha = float(
        np.clip(config.early_alpha if config.early_alpha is not None else alpha, 0.0, 1.0)
    )
    late_alpha = float(
        np.clip(config.late_alpha if config.late_alpha is not None else alpha, 0.0, 1.0)
    )
    early_samples = int(max(0.0, float(config.early_ms)) * float(sr) / 1000.0)
    early_samples = min(early_samples, int(a.shape[0]))

    if bool(config.align_decay):
        target_rt60 = _resolve_target_rt60(a, b, sr=sr, alpha=alpha)
        if target_rt60 is not None:
            a = _align_decay_shape(a, sr=sr, target_rt60=target_rt60, early_samples=early_samples)
            b = _align_decay_shape(b, sr=sr, target_rt60=target_rt60, early_samples=early_samples)

    if mode == "linear":
        out = _blend_early_late(
            a, b, early_samples=early_samples, early_alpha=early_alpha, late_alpha=late_alpha
        )
    elif mode == "equal-power":
        out = _blend_early_late(
            a,
            b,
            early_samples=early_samples,
            early_alpha=early_alpha,
            late_alpha=late_alpha,
            equal_power=True,
        )
    elif mode == "spectral":
        if abs(early_alpha - late_alpha) <= 1e-9:
            out = _blend_spectral(
                a,
                b,
                alpha=alpha,
                phase_coherence=float(np.clip(config.phase_coherence, 0.0, 1.0)),
                smooth_bins=max(0, int(config.spectral_smooth_bins)),
            )
        else:
            out = _blend_early_late(
                a,
                b,
                early_samples=early_samples,
                early_alpha=early_alpha,
                late_alpha=late_alpha,
            )
            if int(out.shape[0]) > early_samples:
                tail = _blend_spectral(
                    a[early_samples:, :],
                    b[early_samples:, :],
                    alpha=late_alpha,
                    phase_coherence=float(np.clip(config.phase_coherence, 0.0, 1.0)),
                    smooth_bins=max(0, int(config.spectral_smooth_bins)),
                )
                out[early_samples:, :] = tail
    else:
        out = _blend_early_late(
            a, b, early_samples=early_samples, early_alpha=early_alpha, late_alpha=late_alpha
        )
        if int(out.shape[0]) > early_samples:
            tail = _blend_spectral(
                a[early_samples:, :],
                b[early_samples:, :],
                alpha=late_alpha,
                phase_coherence=float(np.clip(config.phase_coherence, 0.0, 1.0)),
                smooth_bins=max(0, int(config.spectral_smooth_bins)),
            )
            target_env = _blend_envelope(
                np.abs(a[early_samples:, :]),
                np.abs(b[early_samples:, :]),
                alpha=late_alpha,
            )
            tail = _impose_envelope(tail, target_env)
            out[early_samples:, :] = tail

    out = _normalize_channel_energy(a, b, out, alpha=alpha)
    out = np.asarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
    quality = compute_morph_quality_metrics(a, b, out, sr=sr, alpha=alpha)
    quality["mode"] = mode
    quality["alpha"] = alpha
    quality["early_alpha"] = early_alpha
    quality["late_alpha"] = late_alpha
    quality["aligned_decay"] = bool(config.align_decay)
    quality["mismatch_policy"] = policy
    return out, quality


def generate_or_load_cached_morphed_ir(
    *,
    ir_a_path: Path,
    ir_b_path: Path,
    config: IRMorphConfig,
    cache_dir: Path,
    target_sr: int | None = None,
) -> tuple[AudioArray, int, dict[str, Any], Path, bool]:
    """Morph two IR files with deterministic cache lookup."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig_a = _source_signature(ir_a_path)
    sig_b = _source_signature(ir_b_path)
    mismatch_policy = validate_ir_morph_mismatch_policy_name(config.mismatch_policy)
    _validate_source_mismatch_policy(
        reference=sig_a,
        candidate=sig_b,
        candidate_label="IR_B",
        mismatch_policy=cast(IRMorphMismatchPolicy, mismatch_policy),
    )
    payload = {
        "cache_schema": _CACHE_KEY_SCHEMA,
        "op": "pair-morph",
        "a": sig_a,
        "b": sig_b,
        "config": asdict(config),
        "target_sr": None if target_sr is None else int(target_sr),
    }
    key = _payload_hash(payload)
    wav_path = cache_dir / f"{key}.wav"
    meta_path = cache_dir / f"{key}.meta.json"

    if wav_path.exists() and meta_path.exists():
        audio, sr = sf.read(str(wav_path), always_2d=True, dtype="float64")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return np.asarray(audio, dtype=np.float64), int(sr), meta, wav_path, True

    chosen_sr = _resolve_target_sample_rate((ir_a_path, ir_b_path), target_sr=target_sr)
    a = _load_ir_resampled(ir_a_path, target_sr=chosen_sr)
    b = _load_ir_resampled(ir_b_path, target_sr=chosen_sr)
    morphed, quality = morph_ir_arrays(a, b, sr=chosen_sr, config=config)

    meta = {
        "mode": "ir-morph",
        "cache_schema": _CACHE_KEY_SCHEMA,
        "cache_key": key,
        "sample_rate": chosen_sr,
        "params": asdict(config),
        "sources": [str(ir_a_path), str(ir_b_path)],
        "source_signatures": {
            "a": sig_a,
            "b": sig_b,
        },
        "quality": quality,
    }
    sf.write(str(wav_path), morphed, chosen_sr)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return morphed, chosen_sr, meta, wav_path, False


def generate_or_load_cached_blended_ir(
    *,
    base_ir_path: Path,
    blend_ir_paths: Sequence[Path],
    blend_mix: Sequence[float],
    config: IRMorphConfig,
    cache_dir: Path,
    target_sr: int | None = None,
) -> tuple[AudioArray, int, dict[str, Any], Path, bool]:
    """Blend one base IR plus N additional IRs with deterministic cache lookup."""
    if len(blend_ir_paths) == 0:
        msg = "blend_ir_paths must contain at least one IR path."
        raise ValueError(msg)

    mix_values = resolve_blend_mix_values(blend_mix, len(blend_ir_paths))
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_sig = _source_signature(base_ir_path)
    blend_sigs = [_source_signature(path) for path in blend_ir_paths]
    mismatch_policy = validate_ir_morph_mismatch_policy_name(config.mismatch_policy)
    for idx, blend_sig in enumerate(blend_sigs, start=1):
        _validate_source_mismatch_policy(
            reference=base_sig,
            candidate=blend_sig,
            candidate_label=f"IR_BLEND[{idx}]",
            mismatch_policy=cast(IRMorphMismatchPolicy, mismatch_policy),
        )
    payload = {
        "cache_schema": _CACHE_KEY_SCHEMA,
        "op": "multi-blend",
        "base": base_sig,
        "blend": blend_sigs,
        "mix": list(mix_values),
        "config": asdict(config),
        "target_sr": None if target_sr is None else int(target_sr),
    }
    key = _payload_hash(payload)
    wav_path = cache_dir / f"{key}.wav"
    meta_path = cache_dir / f"{key}.meta.json"

    if wav_path.exists() and meta_path.exists():
        audio, sr = sf.read(str(wav_path), always_2d=True, dtype="float64")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return np.asarray(audio, dtype=np.float64), int(sr), meta, wav_path, True

    sources: list[Path] = [base_ir_path, *blend_ir_paths]
    chosen_sr = _resolve_target_sample_rate(sources, target_sr=target_sr)
    arrays = [_load_ir_resampled(path, target_sr=chosen_sr) for path in sources]

    weights = np.asarray([1.0, *mix_values], dtype=np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        weights[:] = 1.0
        weight_sum = float(np.sum(weights))
    weights /= weight_sum

    current = np.asarray(arrays[0], dtype=np.float64)
    cumulative = float(weights[0])
    last_quality: dict[str, Any] | None = None
    for idx, nxt in enumerate(arrays[1:], start=1):
        denom = cumulative + float(weights[idx])
        step_alpha = 0.0 if denom <= 1e-12 else float(weights[idx] / denom)
        step_cfg = IRMorphConfig(
            mode=config.mode,
            alpha=step_alpha,
            early_ms=config.early_ms,
            early_alpha=config.early_alpha,
            late_alpha=config.late_alpha,
            align_decay=config.align_decay,
            phase_coherence=config.phase_coherence,
            spectral_smooth_bins=config.spectral_smooth_bins,
        )
        current, last_quality = morph_ir_arrays(current, nxt, sr=chosen_sr, config=step_cfg)
        cumulative = denom

    if last_quality is None:
        last_quality = {}
    meta = {
        "mode": "ir-blend",
        "cache_schema": _CACHE_KEY_SCHEMA,
        "cache_key": key,
        "sample_rate": chosen_sr,
        "params": asdict(config),
        "sources": [str(path) for path in sources],
        "source_signatures": {
            "base": base_sig,
            "blend": blend_sigs,
        },
        "weights": [float(w) for w in weights],
        "blend_mix_input": [float(v) for v in mix_values],
        "quality_last_step": last_quality,
    }
    sf.write(str(wav_path), current, chosen_sr)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return current, chosen_sr, meta, wav_path, False


def compute_morph_quality_metrics(
    ir_a: AudioArray,
    ir_b: AudioArray,
    out: AudioArray,
    *,
    sr: int,
    alpha: float,
) -> dict[str, Any]:
    """Compute compact objective quality metrics for one morph result."""
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    a_metrics = analyze_ir(np.asarray(ir_a, dtype=np.float64), sr)
    b_metrics = analyze_ir(np.asarray(ir_b, dtype=np.float64), sr)
    out_metrics = analyze_ir(np.asarray(out, dtype=np.float64), sr)

    rt_a = _metric_float(a_metrics, "rt60_estimate_seconds")
    rt_b = _metric_float(b_metrics, "rt60_estimate_seconds")
    rt_out = _metric_float(out_metrics, "rt60_estimate_seconds")
    target_rt = _blend_scalar(rt_a, rt_b, alpha_clamped)

    elr_a = _metric_float(a_metrics, "early_late_ratio_db")
    elr_b = _metric_float(b_metrics, "early_late_ratio_db")
    elr_out = _metric_float(out_metrics, "early_late_ratio_db")
    target_elr = _blend_scalar(elr_a, elr_b, alpha_clamped)

    coh_a = _interchannel_coherence_score(ir_a)
    coh_b = _interchannel_coherence_score(ir_b)
    coh_out = _interchannel_coherence_score(out)
    target_coh = _blend_scalar(coh_a, coh_b, alpha_clamped)

    spectral_distance_db = _spectral_distance_vs_target(ir_a, ir_b, out, alpha=alpha_clamped)
    return {
        "rt60_target_s": target_rt,
        "rt60_out_s": rt_out,
        "rt60_drift_s": None
        if target_rt is None or rt_out is None
        else float(abs(rt_out - target_rt)),
        "early_late_target_db": target_elr,
        "early_late_out_db": elr_out,
        "early_late_drift_db": (
            None if target_elr is None or elr_out is None else float(abs(elr_out - target_elr))
        ),
        "spectral_distance_db": float(spectral_distance_db),
        "interchannel_coherence_target": target_coh,
        "interchannel_coherence_out": coh_out,
        "interchannel_coherence_delta": (
            None if target_coh is None or coh_out is None else float(abs(coh_out - target_coh))
        ),
    }


def _resolve_target_sample_rate(paths: Sequence[Path], *, target_sr: int | None) -> int:
    if target_sr is not None and int(target_sr) > 0:
        return int(target_sr)
    return max(int(sf.info(str(path)).samplerate) for path in paths)


def _load_ir_resampled(path: Path, *, target_sr: int) -> AudioArray:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float64")
    x = np.asarray(audio, dtype=np.float64)
    src_sr = int(sr)
    if src_sr == int(target_sr):
        return x
    gcd = math.gcd(src_sr, int(target_sr))
    up = int(target_sr // gcd)
    down = int(src_sr // gcd)
    return np.asarray(resample_poly(x, up=up, down=down, axis=0), dtype=np.float64)


def _align_ir_pair(
    ir_a: AudioArray,
    ir_b: AudioArray,
    *,
    mismatch_policy: IRMorphMismatchPolicy = "coerce",
) -> tuple[AudioArray, AudioArray]:
    a = np.asarray(ir_a, dtype=np.float64)
    b = np.asarray(ir_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("IR arrays must be 2D with shape [samples, channels].")
    policy = validate_ir_morph_mismatch_policy_name(mismatch_policy)
    if policy == "strict":
        if int(a.shape[1]) != int(b.shape[1]):
            msg = (
                "IR channel-layout mismatch under strict policy: "
                f"{int(a.shape[1])} vs {int(b.shape[1])}."
            )
            raise ValueError(msg)
        if int(a.shape[0]) != int(b.shape[0]):
            msg = (
                "IR duration mismatch under strict policy: "
                f"{int(a.shape[0])} vs {int(b.shape[0])} samples."
            )
            raise ValueError(msg)
        return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)

    target_channels = max(int(a.shape[1]), int(b.shape[1]))
    a = _match_channels(a, target_channels)
    b = _match_channels(b, target_channels)
    target_len = max(int(a.shape[0]), int(b.shape[0]))
    a = _pad_to_len(a, target_len)
    b = _pad_to_len(b, target_len)
    return a, b


def _match_channels(x: AudioArray, channels: int) -> AudioArray:
    current = int(x.shape[1])
    if current == channels:
        return x
    if current == 1 and channels > 1:
        return np.repeat(x, channels, axis=1).astype(np.float64)
    if channels == 1:
        return np.asarray(np.mean(x, axis=1, keepdims=True, dtype=np.float64), dtype=np.float64)
    idx = np.linspace(0, max(0, current - 1), channels, dtype=np.float64)
    mapped = np.rint(idx).astype(np.int32)
    mapped = np.clip(mapped, 0, max(0, current - 1))
    return np.asarray(x[:, mapped], dtype=np.float64)


def _pad_to_len(x: AudioArray, length: int) -> AudioArray:
    n = int(x.shape[0])
    if n >= length:
        return np.asarray(x[:length, :], dtype=np.float64)
    out = np.zeros((length, int(x.shape[1])), dtype=np.float64)
    out[:n, :] = x
    return out


def _resolve_target_rt60(
    ir_a: AudioArray, ir_b: AudioArray, *, sr: int, alpha: float
) -> float | None:
    metrics_a = analyze_ir(ir_a, sr)
    metrics_b = analyze_ir(ir_b, sr)
    rt_a = _metric_float(metrics_a, "rt60_estimate_seconds")
    rt_b = _metric_float(metrics_b, "rt60_estimate_seconds")
    target = _blend_scalar(rt_a, rt_b, alpha)
    if target is None or target <= 0.0:
        return None
    return float(np.clip(target, 0.05, 300.0))


def _align_decay_shape(
    ir: AudioArray,
    *,
    sr: int,
    target_rt60: float,
    early_samples: int,
) -> AudioArray:
    metrics = analyze_ir(ir, sr)
    current_rt60 = _metric_float(metrics, "rt60_estimate_seconds")
    if current_rt60 is None or current_rt60 <= 0.0:
        return ir

    k_current = math.log(1000.0) / float(max(1e-6, current_rt60))
    k_target = math.log(1000.0) / float(max(1e-6, target_rt60))
    n = int(ir.shape[0])
    t = np.arange(n, dtype=np.float64) / float(max(1, sr))
    t0 = float(max(0, min(n - 1, early_samples))) / float(max(1, sr))
    late_t = np.maximum(0.0, t - t0)
    gain = np.exp((k_current - k_target) * late_t)
    gain = np.clip(gain, 0.05, 20.0).astype(np.float64)
    gain[: min(n, max(0, early_samples))] = np.float64(1.0)
    return np.asarray(ir * gain[:, np.newaxis], dtype=np.float64)


def _blend_early_late(
    a: AudioArray,
    b: AudioArray,
    *,
    early_samples: int,
    early_alpha: float,
    late_alpha: float,
    equal_power: bool = False,
) -> AudioArray:
    out = np.zeros_like(a, dtype=np.float64)
    split = int(np.clip(early_samples, 0, int(a.shape[0])))
    if split > 0:
        out[:split, :] = _blend_weighted(
            a[:split, :], b[:split, :], alpha=early_alpha, equal_power=equal_power
        )
    if split < int(a.shape[0]):
        out[split:, :] = _blend_weighted(
            a[split:, :], b[split:, :], alpha=late_alpha, equal_power=equal_power
        )
    return out


def _blend_weighted(a: AudioArray, b: AudioArray, *, alpha: float, equal_power: bool) -> AudioArray:
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    if equal_power:
        g_a = float(np.cos(0.5 * np.pi * alpha_clamped))
        g_b = float(np.sin(0.5 * np.pi * alpha_clamped))
    else:
        g_a = 1.0 - alpha_clamped
        g_b = alpha_clamped
    return np.asarray((g_a * a) + (g_b * b), dtype=np.float64)


def _blend_spectral(
    a: AudioArray,
    b: AudioArray,
    *,
    alpha: float,
    phase_coherence: float,
    smooth_bins: int,
) -> AudioArray:
    n = int(max(a.shape[0], b.shape[0]))
    if n <= 0:
        return np.zeros((0, max(int(a.shape[1]), int(b.shape[1]))), dtype=np.float64)
    a2, b2 = _align_ir_pair(a, b)
    out = np.zeros_like(a2, dtype=np.float64)
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    coherence = float(np.clip(phase_coherence, 0.0, 1.0))
    eps = 1e-9

    for ch in range(int(a2.shape[1])):
        xa = np.asarray(a2[:, ch], dtype=np.float64)
        xb = np.asarray(b2[:, ch], dtype=np.float64)
        fa = np.fft.rfft(xa)
        fb = np.fft.rfft(xb)

        mag_a = np.abs(fa)
        mag_b = np.abs(fb)
        mag = ((1.0 - alpha_clamped) * mag_a) + (alpha_clamped * mag_b)
        if smooth_bins > 0:
            mag = _smooth_vector(mag, bins=smooth_bins)

        unit_a = fa / (mag_a + eps)
        unit_b = fb / (mag_b + eps)
        unit_mix = ((1.0 - alpha_clamped) * unit_a) + (alpha_clamped * unit_b)
        unit_mix /= np.abs(unit_mix) + eps
        phase_safe = ((1.0 - coherence) * unit_a) + (coherence * unit_mix)
        phase_safe /= np.abs(phase_safe) + eps
        merged = mag * phase_safe
        out[:, ch] = np.asarray(np.fft.irfft(merged, n=n), dtype=np.float64)

    return out


def _smooth_vector(x: npt.NDArray[np.float64], *, bins: int) -> npt.NDArray[np.float64]:
    width = max(1, int(bins))
    kernel_len = (2 * width) + 1
    window = np.hanning(kernel_len).astype(np.float64)
    if np.sum(window) <= 1e-12:
        return x
    kernel = window / np.sum(window)
    padded = np.pad(x, (width, width), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float64)


def _blend_envelope(a_env: AudioArray, b_env: AudioArray, *, alpha: float) -> AudioArray:
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    target = ((1.0 - alpha_clamped) * a_env.astype(np.float64)) + (
        alpha_clamped * b_env.astype(np.float64)
    )
    # Light temporal smoothing to avoid zipper-like gain modulation.
    win = max(3, int(min(513, max(3, (target.shape[0] // 200) * 2 + 1))))
    kernel = np.ones((win,), dtype=np.float64) / float(win)
    out = np.zeros_like(target, dtype=np.float64)
    for ch in range(int(target.shape[1])):
        padded = np.pad(target[:, ch], (win // 2, win // 2), mode="edge")
        out[:, ch] = np.convolve(padded, kernel, mode="valid")
    return np.asarray(out, dtype=np.float64)


def _impose_envelope(x: AudioArray, target_env: AudioArray) -> AudioArray:
    current = np.abs(np.asarray(x, dtype=np.float64))
    scale = target_env / (current + 1e-6)
    scale = np.clip(scale, 0.2, 5.0).astype(np.float64)
    return np.asarray(x * scale, dtype=np.float64)


def _normalize_channel_energy(
    a: AudioArray,
    b: AudioArray,
    out: AudioArray,
    *,
    alpha: float,
) -> AudioArray:
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    y = np.asarray(out, dtype=np.float64).copy()
    for ch in range(int(y.shape[1])):
        rms_a = float(np.sqrt(np.mean(np.square(a[:, ch], dtype=np.float64)) + 1e-12))
        rms_b = float(np.sqrt(np.mean(np.square(b[:, ch], dtype=np.float64)) + 1e-12))
        rms_t = ((1.0 - alpha_clamped) * rms_a) + (alpha_clamped * rms_b)
        rms_y = float(np.sqrt(np.mean(np.square(y[:, ch], dtype=np.float64)) + 1e-12))
        gain = float(np.clip(rms_t / max(1e-9, rms_y), 0.25, 4.0))
        y[:, ch] = np.asarray(y[:, ch] * gain, dtype=np.float64)
    return y


def _metric_float(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _blend_scalar(a: float | None, b: float | None, alpha: float) -> float | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    return float(((1.0 - alpha_clamped) * a) + (alpha_clamped * b))


def _interchannel_coherence_score(ir: AudioArray) -> float | None:
    if int(ir.shape[1]) < 2:
        return 1.0
    ref = np.asarray(ir[:, 0], dtype=np.float64)
    ref_std = float(np.std(ref))
    if ref_std <= 1e-12:
        return None
    values: list[float] = []
    for ch in range(1, int(ir.shape[1])):
        cur = np.asarray(ir[:, ch], dtype=np.float64)
        cur_std = float(np.std(cur))
        if cur_std <= 1e-12:
            continue
        corr = float(np.corrcoef(ref, cur)[0, 1])
        if np.isfinite(corr):
            values.append(abs(corr))
    if len(values) == 0:
        return None
    return float(np.mean(values))


def _spectral_distance_vs_target(
    ir_a: AudioArray,
    ir_b: AudioArray,
    out: AudioArray,
    *,
    alpha: float,
) -> float:
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    mono_a = np.mean(np.asarray(ir_a, dtype=np.float64), axis=1)
    mono_b = np.mean(np.asarray(ir_b, dtype=np.float64), axis=1)
    mono_o = np.mean(np.asarray(out, dtype=np.float64), axis=1)
    n = int(max(mono_a.shape[0], mono_b.shape[0], mono_o.shape[0]))
    if n <= 1:
        return 0.0
    mono_a = _pad_vec(mono_a, n)
    mono_b = _pad_vec(mono_b, n)
    mono_o = _pad_vec(mono_o, n)
    mag_a = np.abs(np.fft.rfft(mono_a))
    mag_b = np.abs(np.fft.rfft(mono_b))
    mag_t = ((1.0 - alpha_clamped) * mag_a) + (alpha_clamped * mag_b)
    mag_o = np.abs(np.fft.rfft(mono_o))
    eps = 1e-9
    db_t = 20.0 * np.log10(mag_t + eps)
    db_o = 20.0 * np.log10(mag_o + eps)
    return float(np.sqrt(np.mean(np.square(db_o - db_t), dtype=np.float64)))


def _pad_vec(x: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.float64]:
    if int(x.shape[0]) >= n:
        return np.asarray(x[:n], dtype=np.float64)
    out = np.zeros((n,), dtype=np.float64)
    out[: int(x.shape[0])] = x
    return out


def _source_signature(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    return {
        "sha256": _file_sha256(path),
        "frames": int(info.frames),
        "channels": int(info.channels),
        "sample_rate": int(info.samplerate),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


def _validate_source_mismatch_policy(
    *,
    reference: dict[str, Any],
    candidate: dict[str, Any],
    candidate_label: str,
    mismatch_policy: IRMorphMismatchPolicy,
) -> None:
    policy = validate_ir_morph_mismatch_policy_name(mismatch_policy)
    if policy != "strict":
        return
    ref_sr = int(reference.get("sample_rate", 0))
    cand_sr = int(candidate.get("sample_rate", 0))
    if ref_sr != cand_sr:
        msg = (
            f"{candidate_label} sample-rate mismatch under strict policy: "
            f"{cand_sr} vs {ref_sr}."
        )
        raise ValueError(msg)
    ref_channels = int(reference.get("channels", 0))
    cand_channels = int(candidate.get("channels", 0))
    if ref_channels != cand_channels:
        msg = (
            f"{candidate_label} channel-layout mismatch under strict policy: "
            f"{cand_channels} vs {ref_channels}."
        )
        raise ValueError(msg)
    ref_frames = int(reference.get("frames", 0))
    cand_frames = int(candidate.get("frames", 0))
    if ref_frames != cand_frames:
        msg = (
            f"{candidate_label} duration mismatch under strict policy: "
            f"{cand_frames} vs {ref_frames} frames."
        )
        raise ValueError(msg)


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
