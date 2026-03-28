"""Deterministic spectral dereverberation for offline CLI processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import istft, stft

from verbx.io.audio import ensure_mono_or_stereo

AudioArray = npt.NDArray[np.float64]
DereverbMode = Literal["wiener", "spectral_sub"]


@dataclass(slots=True)
class DereverbConfig:
    """Configuration for deterministic spectral dereverberation."""

    mode: DereverbMode = "wiener"
    strength: float = 0.65
    floor: float = 0.08
    window_ms: float = 42.67
    hop_ms: float = 10.67
    tail_ms: float = 220.0
    pre_emphasis: float = 0.0
    mix: float = 1.0


def apply_dereverb(audio: AudioArray, sr: int, config: DereverbConfig) -> AudioArray:
    """Apply single-file offline dereverberation using spectral suppression.

    This is intentionally deterministic and does not require model weights.
    """
    x = ensure_mono_or_stereo(audio)
    if x.shape[0] == 0:
        return x.copy()

    n_fft, hop = _resolve_stft_window_hop(
        sr=sr,
        window_ms=float(config.window_ms),
        hop_ms=float(config.hop_ms),
    )
    strength = float(np.clip(config.strength, 0.0, 2.0))
    floor = float(np.clip(config.floor, 1e-6, 1.0))
    tail_s = max(0.001, float(config.tail_ms) / 1000.0)
    mix = float(np.clip(config.mix, 0.0, 1.0))
    pre_emphasis = float(np.clip(config.pre_emphasis, 0.0, 0.98))
    mode = str(config.mode).strip().lower()
    if mode not in {"wiener", "spectral_sub"}:
        msg = f"Unsupported dereverb mode: {config.mode}"
        raise ValueError(msg)

    out = np.zeros_like(x, dtype=np.float64)
    for ch in range(x.shape[1]):
        chan = np.asarray(x[:, ch], dtype=np.float64)
        if pre_emphasis > 0.0:
            chan = _apply_pre_emphasis(chan, pre_emphasis)

        _, _, z = stft(
            chan,
            fs=float(sr),
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            nfft=n_fft,
            boundary="zeros",
            padded=True,
        )
        mag = np.abs(z)
        phase = np.angle(z)
        late_mag = _estimate_late_tail(mag, hop_seconds=float(hop) / float(sr), tail_seconds=tail_s)
        if mode == "spectral_sub":
            mag_hat = _apply_spectral_subtraction(mag, late_mag, strength=strength, floor=floor)
        else:
            mag_hat = _apply_wiener_gain(mag, late_mag, strength=strength, floor=floor)
        z_hat = mag_hat * np.exp(1j * phase)
        _, restored = istft(
            z_hat,
            fs=float(sr),
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            nfft=n_fft,
            input_onesided=True,
            boundary=True,
        )
        restored = np.asarray(restored[: x.shape[0]], dtype=np.float64)
        if restored.shape[0] < x.shape[0]:
            restored = np.pad(restored, (0, x.shape[0] - restored.shape[0]), mode="constant")
        if pre_emphasis > 0.0:
            restored = _apply_de_emphasis(restored, pre_emphasis)
        mixed = (mix * restored) + ((1.0 - mix) * x[:, ch])
        out[:, ch] = np.asarray(mixed, dtype=np.float64)

    return np.asarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)


def _resolve_stft_window_hop(*, sr: int, window_ms: float, hop_ms: float) -> tuple[int, int]:
    """Map millisecond window/hop settings to stable FFT-friendly samples."""
    win = max(32, round(max(2.0, window_ms) * float(sr) / 1000.0))
    hop = max(8, round(max(1.0, hop_ms) * float(sr) / 1000.0))
    if hop >= win:
        hop = max(8, win // 4)
    n_fft = 1 << max(5, int(np.ceil(np.log2(max(32, win)))))
    if n_fft < win:
        n_fft <<= 1
    return int(n_fft), int(hop)


def _estimate_late_tail(
    magnitude: AudioArray,
    *,
    hop_seconds: float,
    tail_seconds: float,
) -> AudioArray:
    """Estimate late reverberant magnitude with exponential smoothing."""
    if magnitude.ndim != 2 or magnitude.shape[1] == 0:
        return np.zeros_like(magnitude, dtype=np.float64)
    alpha = float(np.exp(-hop_seconds / max(hop_seconds, tail_seconds)))
    late = np.zeros_like(magnitude, dtype=np.float64)
    late[:, 0] = np.asarray(magnitude[:, 0], dtype=np.float64)
    blend = 1.0 - alpha
    for t in range(1, magnitude.shape[1]):
        late[:, t] = (alpha * late[:, t - 1]) + (blend * magnitude[:, t])
    return late


def _apply_spectral_subtraction(
    magnitude: AudioArray,
    late_magnitude: AudioArray,
    *,
    strength: float,
    floor: float,
) -> AudioArray:
    """Classical spectral subtraction with proportional floor safeguard."""
    residual = magnitude - (strength * late_magnitude)
    floor_mag = floor * magnitude
    return np.asarray(np.maximum(residual, floor_mag), dtype=np.float64)


def _apply_wiener_gain(
    magnitude: AudioArray,
    late_magnitude: AudioArray,
    *,
    strength: float,
    floor: float,
) -> AudioArray:
    """Apply Wiener-like gain from early-vs-late spectral energy estimate."""
    mag2 = np.square(magnitude)
    late2 = np.square(late_magnitude)
    early2 = np.maximum(mag2 - (strength * late2), 0.0)
    gain = early2 / (early2 + late2 + 1e-12)
    gain = np.clip(gain, floor, 1.0)
    return np.asarray(magnitude * gain, dtype=np.float64)


def _apply_pre_emphasis(signal: npt.NDArray[np.float64], coef: float) -> npt.NDArray[np.float64]:
    """Apply one-tap pre-emphasis before spectral processing."""
    if signal.shape[0] <= 1:
        return signal.copy()
    out = np.empty_like(signal, dtype=np.float64)
    out[0] = signal[0]
    out[1:] = signal[1:] - (coef * signal[:-1])
    return out


def _apply_de_emphasis(signal: npt.NDArray[np.float64], coef: float) -> npt.NDArray[np.float64]:
    """Invert pre-emphasis using stable one-pole recursion."""
    if signal.shape[0] <= 1:
        return signal.copy()
    out = np.empty_like(signal, dtype=np.float64)
    out[0] = signal[0]
    for idx in range(1, signal.shape[0]):
        out[idx] = signal[idx] + (coef * out[idx - 1])
    return out
