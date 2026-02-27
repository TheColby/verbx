"""Time-domain feature extraction."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

AudioArray = npt.NDArray[np.float32]


def duration_seconds(audio: AudioArray, sr: int) -> float:
    """Compute duration in seconds."""
    return float(audio.shape[0] / sr)


def rms(audio: AudioArray) -> float:
    """Compute global RMS across all channels."""
    squared = np.square(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(squared, dtype=np.float64)))


def peak(audio: AudioArray) -> float:
    """Compute absolute sample peak."""
    return float(np.max(np.abs(audio)))


def peak_dbfs(audio: AudioArray) -> float:
    """Compute sample peak in dBFS."""
    p = max(peak(audio), 1e-12)
    return float(20.0 * np.log10(p))


def rms_dbfs(audio: AudioArray) -> float:
    """Compute RMS in dBFS."""
    value = max(rms(audio), 1e-12)
    return float(20.0 * np.log10(value))


def zero_crossing_rate(audio: AudioArray) -> float:
    """Compute average zero-crossing rate."""
    mono = np.mean(audio, axis=1)
    signs = np.signbit(mono)
    crossings = np.count_nonzero(signs[1:] != signs[:-1])
    denom = max(mono.shape[0] - 1, 1)
    return float(crossings / denom)


def crest_factor(audio: AudioArray) -> float:
    """Compute crest factor (peak/rms)."""
    r = max(rms(audio), 1e-12)
    return float(peak(audio) / r)


def dc_offset(audio: AudioArray) -> float:
    """Compute mean offset across all channels."""
    return float(np.mean(audio, dtype=np.float64))


def dynamic_range(audio: AudioArray) -> float:
    """Approximate dynamic range in dB using 95th/5th percentile envelope."""
    envelope = np.abs(np.mean(audio, axis=1))
    hi = float(np.percentile(envelope, 95.0))
    lo = float(np.percentile(envelope, 5.0))
    hi = max(hi, 1e-12)
    lo = max(lo, 1e-12)
    return float(20.0 * np.log10(hi / lo))


def energy(audio: AudioArray) -> float:
    """Compute total signal energy."""
    return float(np.sum(np.square(audio), dtype=np.float64))


def l1_energy(audio: AudioArray) -> float:
    """Compute total L1 magnitude."""
    return float(np.sum(np.abs(audio), dtype=np.float64))


def silence_ratio(audio: AudioArray, threshold_dbfs: float = -50.0) -> float:
    """Estimate proportion of samples below threshold."""
    threshold = float(10.0 ** (threshold_dbfs / 20.0))
    envelope = np.abs(np.mean(audio, axis=1))
    silent = np.count_nonzero(envelope < threshold)
    return float(silent / max(1, envelope.shape[0]))


def transient_density(audio: AudioArray) -> float:
    """Compute transient density via first-order envelope differencing."""
    env = np.abs(np.mean(audio, axis=1))
    diff = np.diff(env, prepend=env[:1])
    threshold = float(np.mean(diff) + (2.0 * np.std(diff)))
    hits = np.count_nonzero(diff > threshold)
    return float(hits / max(1, env.shape[0]))


def stereo_correlation(audio: AudioArray) -> float:
    """Compute stereo L/R correlation when available."""
    if audio.shape[1] < 2:
        return 1.0
    left = audio[:, 0]
    right = audio[:, 1]
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def stereo_width(audio: AudioArray) -> float:
    """Compute mid/side RMS ratio as width proxy."""
    if audio.shape[1] < 2:
        return 0.0
    mid = 0.5 * (audio[:, 0] + audio[:, 1])
    side = 0.5 * (audio[:, 0] - audio[:, 1])
    mid_rms = float(np.sqrt(np.mean(np.square(mid), dtype=np.float64)))
    side_rms = float(np.sqrt(np.mean(np.square(side), dtype=np.float64)))
    return float(side_rms / max(mid_rms, 1e-12))
