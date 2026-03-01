"""IR analysis metrics and decay estimation.

These routines provide compact room-response diagnostics for generated or
captured IRs, including Schroeder decay and simple RT60 estimation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from verbx.analysis.features_time import peak_dbfs, rms_dbfs

AudioArray = npt.NDArray[np.float32]


def schroeder_decay_db(ir: AudioArray) -> npt.NDArray[np.float64]:
    """Compute Schroeder integrated decay curve in dB.

    The decay is energy-integrated from tail to start and normalized to 0 dB at
    the first sample.
    """
    mono = np.mean(ir, axis=1).astype(np.float64)
    energy = np.square(mono) + 1e-18
    integ = np.cumsum(energy[::-1])[::-1]
    integ = np.maximum(integ, 1e-24)
    norm = integ / integ[0]
    return 10.0 * np.log10(norm)


def estimate_rt60(ir: AudioArray, sr: int) -> float:
    """Estimate RT60 via linear regression on decay curve.

    The default window is ``-5`` to ``-35`` dB, with a shallower fallback for
    short/noisy IRs.
    """
    decay = schroeder_decay_db(ir)
    t = np.arange(decay.shape[0], dtype=np.float64) / float(sr)

    mask = (decay <= -5.0) & (decay >= -35.0)
    if np.count_nonzero(mask) < 8:
        mask = (decay <= -5.0) & (decay >= -20.0)
        if np.count_nonzero(mask) < 8:
            return 0.0

    x = t[mask]
    y = decay[mask]
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = float(np.sum((x - x_mean) * (y - y_mean)))
    den = float(np.sum((x - x_mean) ** 2))
    if den <= 1e-18:
        return 0.0

    slope = num / den
    if slope >= 0.0:
        return 0.0

    rt60 = -60.0 / slope
    if np.max(y) > -20.0:
        # Heuristic compensation for short-range regression windows.
        rt60 *= 3.0
    return float(max(0.0, rt60))


def early_late_ratio_db(ir: AudioArray, sr: int, split_ms: float = 80.0) -> float:
    """Compute early/late energy ratio in dB at the requested split time."""
    mono = np.mean(ir, axis=1).astype(np.float64)
    split = max(1, int((split_ms / 1000.0) * sr))
    split = min(split, mono.shape[0] - 1)

    early = np.sum(np.square(mono[:split]))
    late = np.sum(np.square(mono[split:]))
    return float(10.0 * np.log10((early + 1e-18) / (late + 1e-18)))


def stereo_coherence(ir: AudioArray) -> float:
    """Compute stereo coherence (L/R normalized dot product)."""
    if ir.shape[1] < 2:
        return 1.0
    left = ir[:, 0].astype(np.float64)
    right = ir[:, 1].astype(np.float64)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def analyze_ir(ir: AudioArray, sr: int) -> dict[str, float | list[float]]:
    """Return IR analysis metrics and a compact decay curve.

    The full decay curve is downsampled to at most 256 points for readable JSON.
    """
    decay = schroeder_decay_db(ir)
    if decay.shape[0] > 256:
        idx = np.linspace(0, decay.shape[0] - 1, 256, dtype=np.int32)
        compact = decay[idx]
    else:
        compact = decay

    return {
        "duration_seconds": float(ir.shape[0] / sr),
        "peak_dbfs": peak_dbfs(ir),
        "rms_dbfs": rms_dbfs(ir),
        "rt60_estimate_seconds": estimate_rt60(ir, sr),
        "early_late_ratio_db": early_late_ratio_db(ir, sr),
        "stereo_coherence": stereo_coherence(ir),
        "decay_curve_db": compact.astype(np.float64).tolist(),
    }
