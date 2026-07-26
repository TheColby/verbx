"""Ambient enhancement modules: ducking, bloom, and tilt EQ.

These are post-render color controls intended for musical shaping, not
physically exact room simulation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, fftconvolve, sosfilt

from verbx.io.audio import ensure_mono_or_stereo

AudioArray = npt.NDArray[np.float64]


def apply_ducking(
    wet: AudioArray,
    sidechain: AudioArray,
    sr: int,
    attack_ms: float,
    release_ms: float,
    strength: float = 0.75,
    floor: float = 0.0,
) -> AudioArray:
    """Apply envelope-based ducking to wet signal.

    The dry/reference signal drives the sidechain envelope follower. Attack and
    release constants are set in milliseconds and converted to one-pole
    coefficients.
    """
    x = ensure_mono_or_stereo(wet)
    sc = ensure_mono_or_stereo(sidechain)
    if sc.shape[0] < x.shape[0]:
        pad = np.zeros((x.shape[0] - sc.shape[0], sc.shape[1]), dtype=np.float64)
        sc = np.vstack((sc, pad))
    elif sc.shape[0] > x.shape[0]:
        sc = sc[: x.shape[0], :]

    env_in = np.abs(np.mean(sc, axis=1)).astype(np.float64)
    env = np.zeros_like(env_in)

    attack = max(attack_ms, 0.1) / 1000.0
    release = max(release_ms, 0.1) / 1000.0
    attack_alpha = np.exp(-1.0 / (attack * sr))
    release_alpha = np.exp(-1.0 / (release * sr))

    last = np.float64(0.0)
    for i, sample in enumerate(env_in):
        if sample > last:
            last = (attack_alpha * last) + ((1.0 - attack_alpha) * sample)
        else:
            last = (release_alpha * last) + ((1.0 - release_alpha) * sample)
        env[i] = last

    normalized = env / (np.percentile(env, 95.0) + 1e-9)
    reduction = np.clip(normalized, 0.0, 1.0) * float(np.clip(strength, 0.0, 1.0))
    gain = np.maximum(1.0 - reduction, float(np.clip(floor, 0.0, 1.0)))

    return np.asarray(x * gain[:, np.newaxis], dtype=np.float64)


def apply_bloom(
    audio: AudioArray,
    sr: int,
    bloom_seconds: float,
    bloom_mix: float | None = None,
) -> AudioArray:
    """Add soft trailing bloom via exponential convolution tail.

    This stage intentionally behaves like a gentle diffuse smear layer.
    """
    x = ensure_mono_or_stereo(audio)
    if bloom_seconds <= 0.0:
        return x

    tau = max(0.05, bloom_seconds)
    kernel_len = min(int(sr * tau * 3.0), sr * 3)
    kernel_len = max(16, kernel_len)

    t = np.arange(kernel_len, dtype=np.float64) / float(sr)
    kernel = np.exp(-t / tau)
    kernel = kernel / max(np.sum(kernel), 1e-12)

    auto_mix = float(np.clip(bloom_seconds / 8.0, 0.0, 0.65))
    mix = auto_mix if bloom_mix is None else float(np.clip(bloom_mix, 0.0, 1.0))
    out = x.copy()
    for ch in range(x.shape[1]):
        tail = fftconvolve(x[:, ch], kernel, mode="full")[: x.shape[0]]
        out[:, ch] = ((1.0 - mix) * x[:, ch]) + (mix * tail.astype(np.float64))

    return np.asarray(out, dtype=np.float64)


def apply_tilt_eq(
    audio: AudioArray,
    sr: int,
    tilt_db: float,
    lowcut: float | None,
    highcut: float | None,
    *,
    lowcut_order: int = 2,
    highcut_order: int = 2,
    pivot_hz: float = 1_000.0,
) -> AudioArray:
    """Apply tilt EQ around 1kHz plus optional low/high cuts.

    Positive tilt boosts highs and attenuates lows; negative tilt does the
    inverse. A frequency-domain shelf approximation is used for clarity and
    deterministic behavior across channels.
    """
    x = ensure_mono_or_stereo(audio)
    out = x.copy()

    if lowcut is not None and 10.0 < lowcut < (sr * 0.49):
        order = max(1, int(lowcut_order))
        sos = butter(order, lowcut / (0.5 * sr), btype="highpass", output="sos")
        for ch in range(out.shape[1]):
            filtered = sosfilt(sos, out[:, ch])
            if isinstance(filtered, tuple):
                filtered = filtered[0]
            out[:, ch] = np.asarray(filtered, dtype=np.float64)

    if highcut is not None and 10.0 < highcut < (sr * 0.49):
        order = max(1, int(highcut_order))
        sos = butter(order, highcut / (0.5 * sr), btype="lowpass", output="sos")
        for ch in range(out.shape[1]):
            filtered = sosfilt(sos, out[:, ch])
            if isinstance(filtered, tuple):
                filtered = filtered[0]
            out[:, ch] = np.asarray(filtered, dtype=np.float64)

    if abs(tilt_db) < 1e-4:
        return np.asarray(out, dtype=np.float64)

    n = out.shape[0]
    if n < 4:
        return np.asarray(out, dtype=np.float64)

    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    safe_freqs = np.maximum(freqs, 20.0)
    pivot = float(np.clip(pivot_hz, 20.0, sr * 0.45))
    tilt_curve_db = tilt_db * np.log2(safe_freqs / pivot)
    tilt_curve_db = np.clip(tilt_curve_db, -18.0, 18.0)
    gain = np.power(10.0, tilt_curve_db / 20.0).astype(np.float64)

    for ch in range(out.shape[1]):
        spectrum = np.fft.rfft(out[:, ch]).astype(np.complex128)
        shaped = spectrum * gain
        out[:, ch] = np.fft.irfft(shaped, n=n).astype(np.float64)

    return np.asarray(out, dtype=np.float64)


def apply_ambient_processing(
    wet: AudioArray,
    dry_reference: AudioArray,
    sr: int,
    duck: bool,
    duck_attack: float,
    duck_release: float,
    duck_strength: float,
    duck_floor: float,
    bloom: float,
    bloom_mix: float | None,
    lowcut: float | None,
    lowcut_order: int,
    highcut: float | None,
    highcut_order: int,
    tilt: float,
    tilt_pivot_hz: float,
) -> AudioArray:
    """Apply ambient enhancement chain to wet signal.

    Order: ducking -> bloom -> EQ.
    """
    out = ensure_mono_or_stereo(wet)
    if duck:
        out = apply_ducking(
            out,
            dry_reference,
            sr,
            attack_ms=duck_attack,
            release_ms=duck_release,
            strength=duck_strength,
            floor=duck_floor,
        )
    if bloom > 0.0:
        out = apply_bloom(out, sr, bloom_seconds=bloom, bloom_mix=bloom_mix)
    if lowcut is not None or highcut is not None or abs(tilt) > 1e-4:
        out = apply_tilt_eq(
            out,
            sr,
            tilt_db=tilt,
            lowcut=lowcut,
            highcut=highcut,
            lowcut_order=lowcut_order,
            highcut_order=highcut_order,
            pivot_hz=tilt_pivot_hz,
        )
    return np.asarray(out, dtype=np.float64)
