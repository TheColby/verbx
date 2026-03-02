"""Energy Decay Relief (EDR) estimation helpers.

This module provides a practical EDR-style summary for arbitrary program audio.
It is not a full room-acoustics laboratory implementation, but it gives useful
frequency-dependent decay estimates from a single file for creative workflows.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

AudioArray = npt.NDArray[np.float32]


def edr_summary(
    audio: AudioArray,
    sr: int,
    n_fft: int = 2048,
    hop_size: int | None = None,
) -> dict[str, float]:
    """Estimate compact EDR metrics for one audio file.

    The algorithm computes a bandwise reverse cumulative STFT-energy decay and
    performs slope-based RT60 fits per frequency bin, then aggregates by band.
    """
    mono = np.mean(audio, axis=1).astype(np.float32)
    if mono.size < 32:
        return _empty_summary()

    fft_size = int(min(max(256, n_fft), 8192))
    if mono.size < fft_size:
        fft_size = int(2 ** np.floor(np.log2(max(32, mono.size))))
        fft_size = max(32, fft_size)
    hop = int(fft_size // 4 if hop_size is None else max(1, hop_size))

    power = _stft_power(mono, fft_size, hop)
    if power.shape[1] < 8:
        return _empty_summary()

    decay_db = _band_decay_db(power)
    frame_times = (np.arange(decay_db.shape[1], dtype=np.float64) * hop) / float(sr)
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / float(sr))

    t60_bins = np.asarray(
        [_fit_rt60_from_decay(decay_db[idx, :], frame_times) for idx in range(decay_db.shape[0])],
        dtype=np.float64,
    )

    valid = t60_bins > 0.0
    low = (freqs >= 20.0) & (freqs < 250.0)
    mid = (freqs >= 250.0) & (freqs < 2_000.0)
    high = freqs >= 2_000.0

    return {
        "edr_rt60_median_s": _median_safe(t60_bins[valid]),
        "edr_rt60_low_s": _median_safe(t60_bins[valid & low]),
        "edr_rt60_mid_s": _median_safe(t60_bins[valid & mid]),
        "edr_rt60_high_s": _median_safe(t60_bins[valid & high]),
        "edr_valid_bins": float(np.count_nonzero(valid)),
    }


def _stft_power(
    mono: npt.NDArray[np.float32],
    n_fft: int,
    hop: int,
) -> npt.NDArray[np.float64]:
    """Return STFT power matrix with shape ``(freq_bins, frames)``."""
    if mono.size < n_fft:
        padded = np.zeros(n_fft, dtype=np.float32)
        padded[: mono.size] = mono
        mono = padded

    starts = range(0, max(1, mono.size - n_fft + 1), hop)
    window = np.hanning(n_fft).astype(np.float32)
    frames: list[npt.NDArray[np.float64]] = []
    for start in starts:
        frame = mono[start : start + n_fft]
        if frame.size < n_fft:
            padded = np.zeros(n_fft, dtype=np.float32)
            padded[: frame.size] = frame
            frame = padded
        spec = np.fft.rfft(frame * window)
        power = np.square(np.abs(spec), dtype=np.float64)
        frames.append(power)

    if not frames:
        return np.zeros((1, 1), dtype=np.float64)
    return np.stack(frames, axis=1).astype(np.float64)


def _band_decay_db(power: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute reverse-integrated decay in dB for each frequency bin."""
    cumulative = np.cumsum(power[:, ::-1], axis=1)[:, ::-1]
    cumulative = np.maximum(cumulative, 1e-24)
    norm = cumulative / np.maximum(cumulative[:, :1], 1e-24)
    return 10.0 * np.log10(norm)


def _fit_rt60_from_decay(
    decay_db: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
) -> float:
    """Estimate RT60 from one decay curve using a slope fit window."""
    mask = (decay_db <= -5.0) & (decay_db >= -35.0)
    if np.count_nonzero(mask) < 8:
        mask = (decay_db <= -5.0) & (decay_db >= -20.0)
        if np.count_nonzero(mask) < 8:
            return 0.0

    x = times[mask]
    y = decay_db[mask]
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    num = float(np.sum((x - x_mean) * (y - y_mean)))
    den = float(np.sum((x - x_mean) ** 2))
    if den <= 1e-18:
        return 0.0

    slope = num / den
    if slope >= -1e-9:
        return 0.0
    return float(max(0.0, -60.0 / slope))


def _median_safe(values: npt.NDArray[np.float64]) -> float:
    """Return median value with empty-array fallback."""
    if values.size == 0:
        return 0.0
    return float(np.median(values))


def _empty_summary() -> dict[str, float]:
    """Return zeroed EDR summary for edge-case inputs."""
    return {
        "edr_rt60_median_s": 0.0,
        "edr_rt60_low_s": 0.0,
        "edr_rt60_mid_s": 0.0,
        "edr_rt60_high_s": 0.0,
        "edr_valid_bins": 0.0,
    }

