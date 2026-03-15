"""FDN QA metrics for decay-density and ringing regression checks.

These helpers are designed for automated regression tests that compare
algorithmic FDN topology changes over time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

AudioArray = npt.NDArray[np.float64]


@dataclass(slots=True)
class FDNQAMetrics:
    """Compact metrics used in FDN regression harnesses."""

    echo_density_start: float
    echo_density_end: float
    echo_density_growth: float
    ringing_index: float


def _as_mono(signal: AudioArray) -> AudioArray:
    """Convert input signal to mono vector for QA metric estimation."""
    if signal.ndim == 1:
        return np.asarray(signal, dtype=np.float64)
    if signal.ndim == 2 and signal.shape[1] > 0:
        return np.asarray(np.mean(signal, axis=1), dtype=np.float64)
    return np.zeros((0,), dtype=np.float64)


def echo_density_curve(
    signal: AudioArray,
    sr: int,
    *,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    threshold_db: float = -45.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Estimate normalized echo density over time from an impulse response."""
    x = np.abs(_as_mono(signal))
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    frame = max(1, int((float(frame_ms) / 1000.0) * sr))
    hop = max(1, int((float(hop_ms) / 1000.0) * sr))
    reference = float(np.max(x)) + 1e-12
    threshold = reference * (10.0 ** (float(threshold_db) / 20.0))

    time_values: list[float] = []
    density_values: list[float] = []
    for start in range(0, max(1, x.shape[0] - frame + 1), hop):
        end = min(x.shape[0], start + frame)
        if end <= start:
            continue
        frame_slice = x[start:end]
        density = float(np.mean(frame_slice >= threshold))
        center_time = (start + ((end - start) * 0.5)) / float(sr)
        time_values.append(center_time)
        density_values.append(density)

    return (
        np.asarray(time_values, dtype=np.float64),
        np.asarray(density_values, dtype=np.float64),
    )


def ringing_index(
    signal: AudioArray,
    sr: int,
    *,
    tail_start_s: float = 0.15,
    analysis_window_s: float = 1.2,
    max_samples: int = 8192,
) -> float:
    """Estimate ringing risk from normalized late-tail autocorrelation peak."""
    x = _as_mono(signal)
    if x.shape[0] == 0:
        return 0.0

    start = int(max(0.0, tail_start_s) * sr)
    if start >= x.shape[0]:
        return 0.0

    end = min(x.shape[0], start + int(max(0.1, analysis_window_s) * sr))
    seg = np.asarray(x[start:end], dtype=np.float64)
    if seg.shape[0] > max_samples:
        seg = seg[:max_samples]
    if seg.shape[0] < 64:
        return 0.0

    seg = np.asarray(seg - np.mean(seg), dtype=np.float64)
    norm = float(np.linalg.norm(seg))
    if norm <= 1e-10:
        return 0.0

    n = int(seg.shape[0])
    n_fft = 1 << int(np.ceil(np.log2((2 * n) - 1)))
    spectrum = np.fft.rfft(seg.astype(np.float64), n=n_fft)
    power = spectrum * np.conjugate(spectrum)
    corr = np.fft.irfft(power, n=n_fft)[:n].astype(np.float64)
    corr = corr / (corr[0] + 1e-12)

    min_lag = max(1, int(0.002 * sr))
    if min_lag >= corr.shape[0]:
        return 0.0

    return float(np.clip(np.max(np.abs(corr[min_lag:])), 0.0, 1.0))


def compute_fdn_qa_metrics(signal: AudioArray, sr: int) -> FDNQAMetrics:
    """Compute compact FDN QA regression metrics."""
    _, density = echo_density_curve(signal, sr)
    if density.shape[0] == 0:
        start_density = 0.0
        end_density = 0.0
    else:
        segment = max(1, density.shape[0] // 5)
        start_density = float(np.mean(density[:segment]))
        end_density = float(np.mean(density[-segment:]))

    return FDNQAMetrics(
        echo_density_start=start_density,
        echo_density_end=end_density,
        echo_density_growth=end_density - start_density,
        ringing_index=ringing_index(signal, sr),
    )
