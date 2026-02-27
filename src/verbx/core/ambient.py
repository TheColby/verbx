"""Ambient enhancement modules: ducking, bloom, and tilt EQ."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, fftconvolve, sosfilt

from verbx.io.audio import ensure_mono_or_stereo

AudioArray = npt.NDArray[np.float32]


def apply_ducking(
    wet: AudioArray,
    sidechain: AudioArray,
    sr: int,
    attack_ms: float,
    release_ms: float,
    strength: float = 0.75,
) -> AudioArray:
    """Apply envelope-based ducking to wet signal."""
    x = ensure_mono_or_stereo(wet)
    sc = ensure_mono_or_stereo(sidechain)
    if sc.shape[0] < x.shape[0]:
        pad = np.zeros((x.shape[0] - sc.shape[0], sc.shape[1]), dtype=np.float32)
        sc = np.vstack((sc, pad))
    elif sc.shape[0] > x.shape[0]:
        sc = sc[: x.shape[0], :]

    env_in = np.abs(np.mean(sc, axis=1)).astype(np.float32)
    env = np.zeros_like(env_in)

    attack = max(attack_ms, 0.1) / 1000.0
    release = max(release_ms, 0.1) / 1000.0
    attack_alpha = np.exp(-1.0 / (attack * sr))
    release_alpha = np.exp(-1.0 / (release * sr))

    last = np.float32(0.0)
    for i, sample in enumerate(env_in):
        if sample > last:
            last = (attack_alpha * last) + ((1.0 - attack_alpha) * sample)
        else:
            last = (release_alpha * last) + ((1.0 - release_alpha) * sample)
        env[i] = last

    normalized = env / (np.percentile(env, 95.0) + 1e-9)
    reduction = np.clip(normalized, 0.0, 1.0) * float(np.clip(strength, 0.0, 1.0))
    gain = 1.0 - reduction

    return np.asarray(x * gain[:, np.newaxis], dtype=np.float32)


def apply_bloom(audio: AudioArray, sr: int, bloom_seconds: float) -> AudioArray:
    """Add soft trailing bloom via exponential convolution tail."""
    x = ensure_mono_or_stereo(audio)
    if bloom_seconds <= 0.0:
        return x

    tau = max(0.05, bloom_seconds)
    kernel_len = min(int(sr * tau * 3.0), sr * 3)
    kernel_len = max(16, kernel_len)

    t = np.arange(kernel_len, dtype=np.float64) / float(sr)
    kernel = np.exp(-t / tau)
    kernel = kernel / max(np.sum(kernel), 1e-12)

    mix = float(np.clip(bloom_seconds / 8.0, 0.0, 0.65))
    out = x.copy()
    for ch in range(x.shape[1]):
        tail = fftconvolve(x[:, ch], kernel, mode="full")[: x.shape[0]]
        out[:, ch] = ((1.0 - mix) * x[:, ch]) + (mix * tail.astype(np.float32))

    return np.asarray(out, dtype=np.float32)


def apply_tilt_eq(
    audio: AudioArray,
    sr: int,
    tilt_db: float,
    lowcut: float | None,
    highcut: float | None,
) -> AudioArray:
    """Apply tilt EQ around 1kHz plus optional low/high cuts."""
    x = ensure_mono_or_stereo(audio)
    out = x.copy()

    if lowcut is not None and 10.0 < lowcut < (sr * 0.49):
        sos = butter(2, lowcut / (0.5 * sr), btype="highpass", output="sos")
        for ch in range(out.shape[1]):
            filtered = sosfilt(sos, out[:, ch])
            if isinstance(filtered, tuple):
                filtered = filtered[0]
            out[:, ch] = np.asarray(filtered, dtype=np.float32)

    if highcut is not None and 10.0 < highcut < (sr * 0.49):
        sos = butter(2, highcut / (0.5 * sr), btype="lowpass", output="sos")
        for ch in range(out.shape[1]):
            filtered = sosfilt(sos, out[:, ch])
            if isinstance(filtered, tuple):
                filtered = filtered[0]
            out[:, ch] = np.asarray(filtered, dtype=np.float32)

    if abs(tilt_db) < 1e-4:
        return np.asarray(out, dtype=np.float32)

    n = out.shape[0]
    if n < 4:
        return np.asarray(out, dtype=np.float32)

    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    safe_freqs = np.maximum(freqs, 20.0)
    tilt_curve_db = tilt_db * np.log2(safe_freqs / 1000.0)
    tilt_curve_db = np.clip(tilt_curve_db, -18.0, 18.0)
    gain = np.power(10.0, tilt_curve_db / 20.0).astype(np.float32)

    for ch in range(out.shape[1]):
        spectrum = np.fft.rfft(out[:, ch]).astype(np.complex64)
        shaped = spectrum * gain
        out[:, ch] = np.fft.irfft(shaped, n=n).astype(np.float32)

    return np.asarray(out, dtype=np.float32)


def apply_ambient_processing(
    wet: AudioArray,
    dry_reference: AudioArray,
    sr: int,
    duck: bool,
    duck_attack: float,
    duck_release: float,
    bloom: float,
    lowcut: float | None,
    highcut: float | None,
    tilt: float,
) -> AudioArray:
    """Apply ambient enhancement chain to wet signal."""
    out = ensure_mono_or_stereo(wet)
    if duck:
        out = apply_ducking(out, dry_reference, sr, attack_ms=duck_attack, release_ms=duck_release)
    if bloom > 0.0:
        out = apply_bloom(out, sr, bloom_seconds=bloom)
    if lowcut is not None or highcut is not None or abs(tilt) > 1e-4:
        out = apply_tilt_eq(out, sr, tilt_db=tilt, lowcut=lowcut, highcut=highcut)
    return np.asarray(out, dtype=np.float32)
