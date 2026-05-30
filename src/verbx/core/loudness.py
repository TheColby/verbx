"""Loudness and peak targeting utilities.

The API is intentionally pragmatic for offline rendering:

- LUFS normalization via ``pyloudnorm`` when available,
- deterministic fallbacks when that dependency is missing,
- optional true-peak approximation by oversampling.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly

from verbx.io.audio import LimiterDetect, LimiterMode, ensure_mono_or_stereo, soft_limiter

try:
    import pyloudnorm as pyln  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    pyln = None

AudioArray = npt.NDArray[np.float64]


def sample_peak_dbfs(audio: AudioArray) -> float:
    """Return sample peak in dBFS."""
    peak = float(np.max(np.abs(audio)))
    peak = max(peak, 1e-12)
    return float(20.0 * np.log10(peak))


def integrated_lufs(audio: AudioArray, sr: int) -> float:
    """Compute EBU R128 integrated loudness (LUFS)."""
    x = ensure_mono_or_stereo(audio)
    if x.shape[0] == 0:
        return -float("inf")

    if pyln is not None:
        meter = pyln.Meter(sr)
        try:
            return float(meter.integrated_loudness(x.astype(np.float64)))
        except ValueError:
            pass

    # Fallback approximation when pyloudnorm is unavailable. This is not a full
    # EBU implementation, but it keeps CLI behavior predictable in minimal envs.
    rms = float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))
    return float(20.0 * np.log10(max(rms, 1e-12)))


def loudness_range_lu(audio: AudioArray, sr: int) -> float:
    """Compute loudness range (LRA) in LU."""
    x = ensure_mono_or_stereo(audio)
    if x.shape[0] == 0:
        return 0.0

    if pyln is not None:
        meter = pyln.Meter(sr)
        try:
            return float(meter.loudness_range(x.astype(np.float64)))
        except ValueError:
            return 0.0

    # Fallback proxy using short-window RMS spread.
    window = max(256, sr // 5)
    mono = np.mean(x, axis=1)
    frames: list[float] = []
    for start in range(0, max(1, mono.shape[0] - window + 1), max(1, window // 2)):
        frame = mono[start : start + window]
        if frame.size == 0:
            continue
        val = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        frames.append(20.0 * math.log10(max(val, 1e-12)))

    if len(frames) < 2:
        return 0.0
    lo = float(np.percentile(frames, 10.0))
    hi = float(np.percentile(frames, 95.0))
    return max(0.0, hi - lo)


def loudness_normalize(audio: AudioArray, sr: int, target_lufs: float) -> AudioArray:
    """Normalize audio toward target integrated LUFS."""
    x = ensure_mono_or_stereo(audio)
    if x.shape[0] == 0:
        return x.copy()

    if pyln is not None:
        meter = pyln.Meter(sr)
        try:
            loudness = float(meter.integrated_loudness(x.astype(np.float64)))
            normalized = pyln.normalize.loudness(x.astype(np.float64), loudness, target_lufs)
            return np.asarray(normalized, dtype=np.float64)
        except ValueError:
            pass

    current = integrated_lufs(x, sr)
    gain_db = target_lufs - current
    gain = float(10.0 ** (gain_db / 20.0))
    return np.asarray(x * gain, dtype=np.float64)


def true_peak_dbfs(audio: AudioArray, sr: int, oversample: int = 4) -> float:
    """Approximate true-peak dBFS using oversampling."""
    x = ensure_mono_or_stereo(audio)
    if x.shape[0] == 0:
        return -float("inf")

    over = max(1, int(oversample))
    if over == 1:
        return sample_peak_dbfs(x)

    upsampled = resample_poly(x.astype(np.float64), up=over, down=1, axis=0)
    peak = float(np.max(np.abs(upsampled)))
    peak = max(peak, 1e-12)
    return float(20.0 * np.log10(peak))


def peak_limit(audio: AudioArray, target_peak_dbfs: float) -> AudioArray:
    """Apply a hard sample-peak ceiling.

    This is a final safety clamp and not a transparent mastering limiter.
    """
    x = ensure_mono_or_stereo(audio)
    ceiling = float(10.0 ** (target_peak_dbfs / 20.0))
    limited = np.clip(x, -ceiling, ceiling)
    return np.asarray(limited, dtype=np.float64)


def apply_output_targets(
    audio: AudioArray,
    sr: int,
    target_lufs: float | None,
    target_peak_dbfs: float | None,
    limiter: bool = True,
    use_true_peak: bool = True,
    oversample: int = 4,
    limiter_mode: LimiterMode = "tanh",
    limiter_detect: LimiterDetect = "peak",
    limiter_threshold_dbfs: float | None = None,
    limiter_ceiling_dbfs: float | None = None,
    limiter_knee_db: float = 6.0,
    limiter_drive: float = 1.0,
    limiter_mix: float = 1.0,
    limiter_attack_ms: float = 0.5,
    limiter_release_ms: float = 80.0,
    limiter_lookahead_ms: float = 1.5,
    limiter_stereo_link: bool = True,
    limiter_oversample: int = 2,
    limiter_pre_gain_db: float = 0.0,
    limiter_post_gain_db: float = 0.0,
    limiter_dc_block: bool = False,
) -> AudioArray:
    """Apply loudness and peak targets to output audio.

    Processing order:
    1. loudness normalization (optional),
    2. peak check and gain trim,
    3. soft limiter (optional),
    4. hard sample peak ceiling,
    5. optional true-peak correction pass.
    """
    x = ensure_mono_or_stereo(audio)
    if x.shape[0] == 0:
        return x.copy()

    out = x.copy()
    if target_lufs is not None and np.isfinite(target_lufs):
        out = loudness_normalize(out, sr, target_lufs)

    if target_peak_dbfs is not None and np.isfinite(target_peak_dbfs):
        resolved_limiter_threshold_dbfs = (
            float(target_peak_dbfs)
            if limiter_threshold_dbfs is None
            else float(limiter_threshold_dbfs)
        )
        resolved_limiter_ceiling_dbfs = (
            float(target_peak_dbfs)
            if limiter_ceiling_dbfs is None
            else float(limiter_ceiling_dbfs)
        )
        measured = (
            true_peak_dbfs(out, sr, oversample=oversample)
            if use_true_peak
            else sample_peak_dbfs(out)
        )
        if measured > target_peak_dbfs:
            gain_db = target_peak_dbfs - measured
            gain = float(10.0 ** (gain_db / 20.0))
            out = np.asarray(out * gain, dtype=np.float64)

        if limiter:
            out = soft_limiter(
                out,
                sr=sr,
                threshold_dbfs=resolved_limiter_threshold_dbfs,
                ceiling_dbfs=resolved_limiter_ceiling_dbfs,
                knee_db=limiter_knee_db,
                mode=cast(LimiterMode, str(limiter_mode).strip().lower()),
                detect=cast(LimiterDetect, str(limiter_detect).strip().lower()),
                drive=limiter_drive,
                mix=limiter_mix,
                attack_ms=limiter_attack_ms,
                release_ms=limiter_release_ms,
                lookahead_ms=limiter_lookahead_ms,
                stereo_link=bool(limiter_stereo_link),
                oversample=limiter_oversample,
                pre_gain_db=limiter_pre_gain_db,
                post_gain_db=limiter_post_gain_db,
                dc_block=bool(limiter_dc_block),
            )

        out = peak_limit(out, target_peak_dbfs)

        # True-peak correction pass after limiting.
        if use_true_peak:
            post_tp = true_peak_dbfs(out, sr, oversample=oversample)
            if post_tp > target_peak_dbfs:
                gain_db = target_peak_dbfs - post_tp
                gain = float(10.0 ** (gain_db / 20.0))
                out = np.asarray(out * gain, dtype=np.float64)
                out = peak_limit(out, target_peak_dbfs)

    return np.asarray(out, dtype=np.float64)
