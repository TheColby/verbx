"""IR post-shaping: filters, EQ, normalization, and loudness targets.

This module encapsulates "finishing" stages used after raw IR synthesis.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from verbx.core.ambient import apply_tilt_eq
from verbx.core.loudness import apply_output_targets
from verbx.io.audio import ensure_mono_or_stereo

AudioArray = npt.NDArray[np.float32]


def normalize_ir(audio: AudioArray, mode: str, peak_dbfs: float) -> AudioArray:
    """Normalize IR according to mode.

    ``mode`` may be ``none``, ``peak``, or ``rms``.
    """
    x = ensure_mono_or_stereo(audio)
    norm_mode = mode.strip().lower()

    if norm_mode == "none":
        return x

    if norm_mode == "rms":
        rms = float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))
        if rms <= 1e-12:
            return x
        target = float(10.0 ** (peak_dbfs / 20.0))
        return np.asarray(x * (target / rms), dtype=np.float32)

    peak = float(np.max(np.abs(x)))
    if peak <= 1e-12:
        return x
    target = float(10.0 ** (peak_dbfs / 20.0))
    return np.asarray(x * (target / peak), dtype=np.float32)


def apply_ir_shaping(
    audio: AudioArray,
    sr: int,
    damping: float,
    lowcut: float | None,
    highcut: float | None,
    tilt: float,
    normalize: str,
    peak_dbfs: float,
    target_lufs: float | None,
    use_true_peak: bool,
) -> AudioArray:
    """Apply deterministic shaping chain to generated IR.

    Processing order:
    1) tilt/filters, 2) damping, 3) normalization, 4) optional loudness target.
    """
    x = ensure_mono_or_stereo(audio)

    out = apply_tilt_eq(x, sr=sr, tilt_db=tilt, lowcut=lowcut, highcut=highcut)

    damp = float(np.clip(damping, 0.0, 1.0))
    if damp > 0.0:
        # One-pole smoothing darkens high-frequency tail content.
        alpha = np.float32(0.1 + (0.85 * damp))
        state = np.zeros(out.shape[1], dtype=np.float32)
        for i in range(out.shape[0]):
            state = ((1.0 - alpha) * out[i, :]) + (alpha * state)
            out[i, :] = state

    out = normalize_ir(out, mode=normalize, peak_dbfs=peak_dbfs)

    if target_lufs is not None:
        out = apply_output_targets(
            out,
            sr,
            target_lufs=target_lufs,
            target_peak_dbfs=peak_dbfs,
            limiter=True,
            use_true_peak=use_true_peak,
        )

    return np.asarray(out, dtype=np.float32)
