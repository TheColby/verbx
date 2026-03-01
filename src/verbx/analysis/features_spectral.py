"""Spectral feature extraction utilities.

When available, librosa implementations are used for feature parity with common
audio-analysis tooling. NumPy fallbacks keep the CLI functional when librosa is
not installed or optional dependencies are unavailable.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

AudioArray = npt.NDArray[np.float32]

try:
    import librosa  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - import fallback branch
    librosa = None


def _mono(audio: AudioArray) -> npt.NDArray[np.float32]:
    """Return mono fold-down used by scalar spectral features."""
    return np.mean(audio, axis=1).astype(np.float32)


def _magnitude_spectrum(
    audio: AudioArray, sr: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return full-file RFFT magnitude and frequency bins."""
    mono = _mono(audio)
    if mono.shape[0] == 0:
        return np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(mono)).astype(np.float64)
    freqs = np.fft.rfftfreq(mono.shape[0], d=1.0 / sr).astype(np.float64)
    return spectrum, freqs


def spectral_centroid(audio: AudioArray, sr: int) -> float:
    """Compute spectral centroid."""
    mono = _mono(audio)
    if librosa is not None and mono.shape[0] > 8:
        n_fft = max(8, min(2048, mono.shape[0]))
        hop_length = max(1, n_fft // 4)
        centroid = librosa.feature.spectral_centroid(
            y=mono, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        return float(np.mean(centroid))

    spectrum, freqs = _magnitude_spectrum(audio, sr)
    denom = float(np.sum(spectrum))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(freqs * spectrum) / denom)


def spectral_bandwidth(audio: AudioArray, sr: int) -> float:
    """Compute spectral bandwidth (2nd central moment in Hz)."""
    mono = _mono(audio)
    if librosa is not None and mono.shape[0] > 8:
        n_fft = max(8, min(2048, mono.shape[0]))
        hop_length = max(1, n_fft // 4)
        bw = librosa.feature.spectral_bandwidth(y=mono, sr=sr, n_fft=n_fft, hop_length=hop_length)
        return float(np.mean(bw))

    spectrum, freqs = _magnitude_spectrum(audio, sr)
    denom = float(np.sum(spectrum))
    if denom <= 0.0:
        return 0.0
    centroid = np.sum(freqs * spectrum) / denom
    var = np.sum(((freqs - centroid) ** 2) * spectrum) / denom
    return float(np.sqrt(max(var, 0.0)))


def spectral_rolloff(audio: AudioArray, sr: int, roll_percent: float = 0.85) -> float:
    """Compute rolloff frequency containing ``roll_percent`` spectral energy."""
    mono = _mono(audio)
    if librosa is not None and mono.shape[0] > 8:
        n_fft = max(8, min(2048, mono.shape[0]))
        hop_length = max(1, n_fft // 4)
        rolloff = librosa.feature.spectral_rolloff(
            y=mono,
            sr=sr,
            roll_percent=roll_percent,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        return float(np.mean(rolloff))

    spectrum, freqs = _magnitude_spectrum(audio, sr)
    cumulative = np.cumsum(spectrum)
    if cumulative[-1] <= 0.0:
        return 0.0
    cutoff = roll_percent * cumulative[-1]
    idx = int(np.searchsorted(cumulative, cutoff))
    idx = min(idx, freqs.shape[0] - 1)
    return float(freqs[idx])


def spectral_flatness(audio: AudioArray, sr: int) -> float:
    """Compute spectral flatness."""
    mono = _mono(audio)
    if librosa is not None and mono.shape[0] > 8:
        n_fft = max(8, min(2048, mono.shape[0]))
        hop_length = max(1, n_fft // 4)
        flatness = librosa.feature.spectral_flatness(y=mono, n_fft=n_fft, hop_length=hop_length)
        return float(np.mean(flatness))

    spectrum, _ = _magnitude_spectrum(audio, sr)
    spectrum = np.maximum(spectrum, 1e-12)
    geometric = float(np.exp(np.mean(np.log(spectrum))))
    arithmetic = float(np.mean(spectrum))
    return float(geometric / max(arithmetic, 1e-12))


def spectral_flux(audio: AudioArray, sr: int) -> float:
    """Compute normalized spectral flux over short windows."""
    mono = _mono(audio)
    if mono.shape[0] < 32:
        return 0.0

    # Choose a stable power-of-two FFT size bounded for CLI speed.
    n_fft = min(1024, max(32, 2 ** int(np.floor(np.log2(mono.shape[0])) - 1)))
    hop = max(16, n_fft // 4)
    window = np.hanning(n_fft).astype(np.float32)

    frames: list[npt.NDArray[np.float32]] = []
    for start in range(0, mono.shape[0] - n_fft + 1, hop):
        frame = mono[start : start + n_fft] * window
        frames.append(np.abs(np.fft.rfft(frame)).astype(np.float32))

    if len(frames) < 2:
        return 0.0

    flux = 0.0
    for i in range(1, len(frames)):
        diff = frames[i] - frames[i - 1]
        flux += float(np.sqrt(np.sum(np.square(diff), dtype=np.float64)))

    return float(flux / (len(frames) - 1))


def spectral_slope(audio: AudioArray, sr: int) -> float:
    """Estimate spectral slope via linear regression in log-frequency domain."""
    spectrum, freqs = _magnitude_spectrum(audio, sr)
    valid = freqs > 0.0
    if np.count_nonzero(valid) < 2:
        return 0.0

    x = np.log10(freqs[valid])
    y = 20.0 * np.log10(np.maximum(spectrum[valid], 1e-12))
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))

    num = float(np.sum((x - x_mean) * (y - y_mean)))
    den = float(np.sum((x - x_mean) ** 2))
    if den <= 1e-12:
        return 0.0
    return float(num / den)
