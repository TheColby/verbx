"""Helpers for IR harmonic tuning and source-audio pitch analysis.

These utilities let synthetic IR modes align to a fixed fundamental (``f0``)
or to harmonics estimated from user-provided source audio.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

try:
    import librosa  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    librosa = None

AudioArray = npt.NDArray[np.float32]

_FREQ_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(?:hz)?\s*$", re.IGNORECASE)


def parse_frequency_hz(value: str) -> float:
    """Parse frequency expressions like `64`, `64Hz`, or `64 Hz`."""
    match = _FREQ_RE.match(value)
    if not match:
        msg = f"Invalid frequency value: {value!r}"
        raise ValueError(msg)
    hz = float(match.group(1))
    if hz <= 0.0:
        msg = "Frequency must be > 0 Hz"
        raise ValueError(msg)
    return hz


def analyze_audio_for_tuning(
    path: Path,
    max_harmonics: int = 12,
    min_hz: float = 20.0,
    max_hz: float = 4_000.0,
) -> tuple[float, list[float]]:
    """Estimate fundamental and harmonic targets from an audio file.

    Analysis is intentionally bounded to a short prefix for deterministic
    runtime on long files.
    """
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    x = np.asarray(audio, dtype=np.float32)
    mono = np.asarray(np.mean(x, axis=1), dtype=np.float32)

    # Keep analysis bounded and deterministic.
    max_samples = min(mono.shape[0], int(sr * 12))
    mono = mono[:max_samples]
    if mono.shape[0] < 128:
        return 220.0, _harmonic_series(220.0, max_harmonics, max_hz=max_hz)

    f0 = _estimate_f0(mono, int(sr), min_hz=min_hz, max_hz=max_hz)
    harmonics = _find_harmonics_from_spectrum(
        mono,
        int(sr),
        f0,
        max_harmonics=max_harmonics,
        max_hz=max_hz,
    )
    return f0, harmonics


def tune_frequency_to_targets(
    freq_hz: float,
    f0_hz: float | None,
    harmonic_targets_hz: tuple[float, ...],
    align_strength: float,
    max_hz: float,
) -> float:
    """Move a frequency toward nearest harmonic target."""
    targets = list(harmonic_targets_hz)
    if not targets and f0_hz is not None:
        targets = _harmonic_series(f0_hz, 16, max_hz=max_hz)
    if not targets:
        return freq_hz

    nearest = min(targets, key=lambda value: abs(value - freq_hz))
    amount = float(np.clip(align_strength, 0.0, 1.0))
    tuned = ((1.0 - amount) * freq_hz) + (amount * nearest)
    return float(np.clip(tuned, 1.0, max_hz))


def apply_harmonic_alignment(
    ir: AudioArray,
    sr: int,
    f0_hz: float | None,
    harmonic_targets_hz: tuple[float, ...],
    strength: float,
) -> AudioArray:
    """Inject a harmonic decay bed so non-modal modes remain musically aligned.

    This is a gentle additive bed, not a hard pitch-quantization stage.
    """
    amount = float(np.clip(strength, 0.0, 1.0))
    if amount <= 0.0:
        return np.asarray(ir, dtype=np.float32)

    targets = list(harmonic_targets_hz)
    if not targets and f0_hz is not None:
        targets = _harmonic_series(f0_hz, 10, max_hz=sr * 0.45)
    if not targets:
        return np.asarray(ir, dtype=np.float32)

    n, channels = ir.shape
    t = np.arange(n, dtype=np.float64) / float(sr)
    bed = np.zeros((n, channels), dtype=np.float64)
    decay = np.exp(-t / max(0.05, n / sr / 5.0))

    for idx, freq in enumerate(targets[:12]):
        phase = 0.37 * idx
        amp = 1.0 / np.sqrt(idx + 1.0)
        harmonic = amp * np.sin((2.0 * np.pi * freq * t) + phase) * decay
        for ch in range(channels):
            spread_phase = phase + (0.17 * ch)
            bed[:, ch] += harmonic * np.cos(spread_phase)

    bed = bed.astype(np.float32)
    blend = 0.22 * amount
    out = np.asarray(ir + (blend * bed), dtype=np.float32)
    return out


def _estimate_f0(mono: npt.NDArray[np.float32], sr: int, min_hz: float, max_hz: float) -> float:
    """Estimate f0 via librosa YIN when available, else autocorrelation."""
    if librosa is not None:
        try:
            f0_track = librosa.yin(mono, fmin=min_hz, fmax=max_hz, sr=sr)
            valid = f0_track[np.isfinite(f0_track)]
            if valid.size > 0:
                return float(np.median(valid))
        except Exception:
            pass

    # Autocorrelation fallback.
    x = mono - np.mean(mono, dtype=np.float32)
    corr = np.correlate(x, x, mode="full")[x.shape[0] - 1 :]
    corr[0] = 0.0

    min_lag = max(1, int(sr / max_hz))
    max_lag = max(min_lag + 1, int(sr / min_hz))
    max_lag = min(max_lag, corr.shape[0] - 1)
    if max_lag <= min_lag:
        return 220.0

    lag = int(np.argmax(corr[min_lag:max_lag]) + min_lag)
    f0 = float(sr / max(lag, 1))
    return float(np.clip(f0, min_hz, max_hz))


def _find_harmonics_from_spectrum(
    mono: npt.NDArray[np.float32],
    sr: int,
    f0_hz: float,
    max_harmonics: int,
    max_hz: float,
) -> list[float]:
    """Find harmonic peak locations around integer multiples of ``f0_hz``."""
    if mono.shape[0] < 256:
        return _harmonic_series(f0_hz, max_harmonics, max_hz=max_hz)

    win = np.hanning(mono.shape[0]).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(mono * win)).astype(np.float64)
    freqs = np.fft.rfftfreq(mono.shape[0], d=1.0 / sr)

    harmonics: list[float] = []
    for k in range(1, max_harmonics + 1):
        target = f0_hz * k
        if target > max_hz:
            break
        lo = target * 0.94
        hi = target * 1.06
        mask = (freqs >= lo) & (freqs <= hi)
        if np.count_nonzero(mask) == 0:
            harmonics.append(float(target))
            continue
        local_idx = int(np.argmax(spectrum[mask]))
        bins = np.flatnonzero(mask)
        peak_bin = bins[local_idx]
        harmonics.append(float(freqs[peak_bin]))
    return harmonics


def _harmonic_series(f0_hz: float, count: int, max_hz: float) -> list[float]:
    """Generate ideal harmonic series up to ``count`` or ``max_hz``."""
    out: list[float] = []
    for k in range(1, count + 1):
        freq = f0_hz * k
        if freq > max_hz:
            break
        out.append(float(freq))
    return out
