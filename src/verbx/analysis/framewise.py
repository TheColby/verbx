"""Framewise analysis export helpers.

The framewise path is aimed at workflow diagnostics (CSV plotting, modulation
inspection) and complements the single-shot metrics in :mod:`analyzer`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from verbx.analysis.features_spectral import spectral_centroid
from verbx.analysis.features_time import peak_dbfs, rms_dbfs, zero_crossing_rate

AudioArray = npt.NDArray[np.float64]


@dataclass(slots=True)
class _FrameFeatures:
    """Internal typed container for per-frame feature snapshots."""

    start_s: float
    end_s: float
    rms_dbfs: float
    peak_dbfs: float
    spectral_centroid: float
    zero_crossing_rate: float
    rms_linear: float
    channel_coherence: float


def framewise_metrics(
    audio: AudioArray,
    sr: int,
    frame_size: int = 2048,
    hop_size: int = 1024,
) -> list[dict[str, float]]:
    """Compute framewise metrics with simple modulation descriptors.

    ``amp_mod_*`` values are computed on short local windows over frame RMS.
    ``centroid_mod_*`` mirrors the same procedure for spectral centroid.
    """
    features: list[_FrameFeatures] = []
    n = audio.shape[0]
    if n == 0:
        return []

    for start in range(0, max(1, n - frame_size + 1), hop_size):
        end = min(n, start + frame_size)
        frame = audio[start:end, :]
        if frame.shape[0] < 8:
            continue

        rms_linear = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        features.append(
            _FrameFeatures(
                start_s=float(start / sr),
                end_s=float(end / sr),
                rms_dbfs=rms_dbfs(frame),
                peak_dbfs=peak_dbfs(frame),
                spectral_centroid=spectral_centroid(frame, sr),
                zero_crossing_rate=zero_crossing_rate(frame),
                rms_linear=rms_linear,
                channel_coherence=_frame_channel_coherence(frame),
            )
        )

    if not features:
        return []

    frame_rate_hz = float(sr) / float(hop_size)
    # Two-second local context is a practical compromise for modulation cues.
    window_frames = max(8, round(2.0 * frame_rate_hz))
    half_window = max(1, window_frames // 2)
    rms_series = np.asarray([item.rms_linear for item in features], dtype=np.float64)
    centroid_series = np.asarray([item.spectral_centroid for item in features], dtype=np.float64)
    coherence_series = np.asarray([item.channel_coherence for item in features], dtype=np.float64)

    rows: list[dict[str, float]] = []
    for idx, item in enumerate(features):
        lo = max(0, idx - half_window)
        hi = min(len(features), idx + half_window + 1)
        rms_window = rms_series[lo:hi]
        centroid_window = centroid_series[lo:hi]

        rows.append(
            {
                "start_s": item.start_s,
                "end_s": item.end_s,
                "rms_dbfs": item.rms_dbfs,
                "peak_dbfs": item.peak_dbfs,
                "spectral_centroid": item.spectral_centroid,
                "zero_crossing_rate": item.zero_crossing_rate,
                "amp_mod_depth": _relative_mod_depth(rms_window),
                "amp_mod_rate_hz": _dominant_mod_rate_hz(rms_window, frame_rate_hz),
                "amp_mod_confidence": _modulation_confidence(rms_window, frame_rate_hz),
                "centroid_mod_depth": _relative_mod_depth(centroid_window),
                "centroid_mod_rate_hz": _dominant_mod_rate_hz(centroid_window, frame_rate_hz),
                "centroid_mod_confidence": _modulation_confidence(centroid_window, frame_rate_hz),
                "channel_coherence": item.channel_coherence,
                "coherence_drift": _drift_rate_hz(coherence_series[lo:hi], frame_rate_hz),
            }
        )

    return rows


def write_framewise_csv(
    path: Path,
    audio: AudioArray,
    sr: int,
    frame_size: int = 2048,
    hop_size: int = 1024,
) -> None:
    """Write framewise analysis CSV to disk."""
    rows = framewise_metrics(audio=audio, sr=sr, frame_size=frame_size, hop_size=hop_size)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "start_s",
                "end_s",
                "rms_dbfs",
                "peak_dbfs",
                "spectral_centroid",
                "zero_crossing_rate",
                "amp_mod_depth",
                "amp_mod_rate_hz",
                "amp_mod_confidence",
                "centroid_mod_depth",
                "centroid_mod_rate_hz",
                "centroid_mod_confidence",
                "channel_coherence",
                "coherence_drift",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _relative_mod_depth(values: npt.NDArray[np.float64]) -> float:
    """Return normalized modulation depth (std/mean)."""
    if values.size < 2:
        return 0.0
    mean = float(np.mean(np.abs(values)))
    if mean <= 1e-12:
        return 0.0
    return float(np.std(values) / mean)


def _dominant_mod_rate_hz(
    values: npt.NDArray[np.float64],
    frame_rate_hz: float,
    min_hz: float = 0.05,
    max_hz: float = 20.0,
) -> float:
    """Return dominant modulation frequency from a short scalar series."""
    if values.size < 8:
        return 0.0

    centered = np.asarray(values - np.mean(values), dtype=np.float64)
    if np.max(np.abs(centered)) <= 1e-12:
        return 0.0

    window = np.hanning(centered.shape[0])
    spectrum = np.abs(np.fft.rfft(centered * window))
    freqs = np.fft.rfftfreq(centered.shape[0], d=1.0 / frame_rate_hz)
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    if np.count_nonzero(mask) == 0:
        return 0.0

    masked = spectrum[mask]
    if masked.size == 0:
        return 0.0
    peak_idx = int(np.argmax(masked))
    bins = np.flatnonzero(mask)
    return float(freqs[bins[peak_idx]])


def _modulation_confidence(
    values: npt.NDArray[np.float64],
    frame_rate_hz: float,
    min_hz: float = 0.05,
    max_hz: float = 20.0,
) -> float:
    """Return confidence for modulation-rate estimate using spectral concentration."""
    if values.size < 8:
        return 0.0
    centered = np.asarray(values - np.mean(values), dtype=np.float64)
    if np.max(np.abs(centered)) <= 1e-12:
        return 0.0

    window = np.hanning(centered.shape[0])
    spectrum = np.abs(np.fft.rfft(centered * window))
    freqs = np.fft.rfftfreq(centered.shape[0], d=1.0 / frame_rate_hz)
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    if np.count_nonzero(mask) == 0:
        return 0.0

    masked = spectrum[mask] ** 2.0
    total = float(np.sum(masked))
    if total <= 1e-12:
        return 0.0
    return float(np.clip(np.max(masked) / total, 0.0, 1.0))


def _frame_channel_coherence(frame: AudioArray) -> float:
    """Compute average adjacent-channel coherence for one frame."""
    channels = int(frame.shape[1])
    if channels <= 1:
        return 1.0

    coherences: list[float] = []
    for idx in range(channels - 1):
        a = np.asarray(frame[:, idx], dtype=np.float64)
        b = np.asarray(frame[:, idx + 1], dtype=np.float64)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            coherences.append(0.0)
            continue
        coherences.append(float(np.clip(np.dot(a, b) / denom, -1.0, 1.0)))
    return float(np.mean(np.abs(np.asarray(coherences, dtype=np.float64))))


def _drift_rate_hz(values: npt.NDArray[np.float64], frame_rate_hz: float) -> float:
    """Return mean absolute drift rate per second for a scalar framewise series."""
    if values.size < 2:
        return 0.0
    diff = np.abs(np.diff(values))
    return float(np.mean(diff) * frame_rate_hz)
