"""Modalys-inspired resonator layer for IR late-tail coloration.

This optional stage adds a physically-inspired resonant bed to the late tail,
useful for metallic, string-like, or plate-like IR coloration.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter

from verbx.io.audio import ensure_mono_or_stereo
from verbx.ir.tuning import tune_frequency_to_targets

AudioArray = npt.NDArray[np.float64]


def apply_modalys_resonator_layer(
    ir: AudioArray,
    sr: int,
    *,
    enabled: bool,
    mix: float,
    modes: int,
    q_min: float,
    q_max: float,
    low_hz: float,
    high_hz: float,
    late_start_ms: float,
    seed: int,
    f0_hz: float | None,
    harmonic_targets_hz: tuple[float, ...],
    align_strength: float,
) -> AudioArray:
    """Apply a deterministic modal resonator bank to the late IR tail.

    Resonators are excited by a mono fold-down of the IR and mixed back into
    all channels with deterministic random gains.
    """
    x = ensure_mono_or_stereo(ir).astype(np.float64, copy=True)
    if not enabled:
        return np.asarray(x, dtype=np.float64)

    n, channels = x.shape
    if n < 8 or sr <= 0:
        return np.asarray(x, dtype=np.float64)

    blend = float(np.clip(mix, 0.0, 1.0))
    mode_count = max(0, int(modes))
    if blend <= 0.0 or mode_count == 0:
        return np.asarray(x, dtype=np.float64)

    q_lo = max(0.5, float(q_min))
    q_hi = max(q_lo, float(q_max))
    f_lo = max(20.0, float(low_hz))
    f_hi = max(f_lo + 1.0, min(float(high_hz), 0.49 * float(sr)))
    start = int(max(0.0, late_start_ms) * float(sr) / 1000.0)
    start = min(max(0, start), n - 1)

    excitation = np.asarray(np.mean(x, axis=1), dtype=np.float64)
    excitation -= float(np.mean(excitation))
    excitation = _dc_block(excitation)
    if start > 0:
        excitation[:start] = 0.0

    if float(np.max(np.abs(excitation))) <= 1e-12:
        return np.asarray(x, dtype=np.float64)

    rng = np.random.default_rng(seed + 9_973)
    reson = np.zeros((n, channels), dtype=np.float64)

    for idx in range(mode_count):
        freq = _sample_mode_frequency(
            rng=rng,
            idx=idx,
            low_hz=f_lo,
            high_hz=f_hi,
            f0_hz=f0_hz,
            harmonic_targets_hz=harmonic_targets_hz,
            align_strength=align_strength,
        )
        q = float(rng.uniform(q_lo, q_hi))
        mode = _resonator_mode(excitation, sr, freq_hz=freq, q=q)
        mode *= 1.0 / np.sqrt(float(mode_count))
        gains = _sample_channel_gains(rng, channels)
        reson += mode[:, np.newaxis] * gains[np.newaxis, :]

    tail_rms = _rms(x[start:, :]) if start < n else _rms(x)
    res_rms = _rms(reson[start:, :]) if start < n else _rms(reson)
    if res_rms > 1e-12 and tail_rms > 0.0:
        # Match late-tail energy so resonator mix behaves predictably.
        reson *= 0.85 * (tail_rms / res_rms)

    envelope = _late_tail_envelope(n=n, sr=sr, start=start)
    out = x + (blend * reson * envelope[:, np.newaxis])
    out = np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(out)))
    if peak > 10.0:
        out *= 10.0 / peak

    return np.asarray(out, dtype=np.float64)


def _sample_mode_frequency(
    *,
    rng: np.random.Generator,
    idx: int,
    low_hz: float,
    high_hz: float,
    f0_hz: float | None,
    harmonic_targets_hz: tuple[float, ...],
    align_strength: float,
) -> float:
    """Sample a mode frequency and optionally align toward harmonic targets."""
    base = float(np.exp(rng.uniform(np.log(low_hz), np.log(high_hz))))

    if len(harmonic_targets_hz) > 0 and rng.random() < 0.8:
        target = harmonic_targets_hz[idx % len(harmonic_targets_hz)]
        detune_cents = float(rng.uniform(-14.0, 14.0))
        base = float(target * (2.0 ** (detune_cents / 1200.0)))

    tuned = tune_frequency_to_targets(
        freq_hz=base,
        f0_hz=f0_hz,
        harmonic_targets_hz=harmonic_targets_hz,
        align_strength=align_strength,
        max_hz=high_hz,
    )
    return float(np.clip(tuned, low_hz, high_hz))


def _resonator_mode(
    excitation: npt.NDArray[np.float64],
    sr: int,
    *,
    freq_hz: float,
    q: float,
) -> npt.NDArray[np.float64]:
    """Render one second-order resonator response from a shared excitation."""
    # Pole radius from equivalent resonator bandwidth.
    bandwidth = max(0.2, float(freq_hz) / max(0.5, float(q)))
    radius = float(np.exp((-np.pi * bandwidth) / float(sr)))
    radius = float(np.clip(radius, 0.0, 0.99995))
    w0 = (2.0 * np.pi * float(freq_hz)) / float(sr)

    b = np.array([1.0 - radius], dtype=np.float64)
    a = np.array([1.0, -2.0 * radius * np.cos(w0), radius * radius], dtype=np.float64)
    return np.asarray(lfilter(b, a, excitation), dtype=np.float64)


def _sample_channel_gains(rng: np.random.Generator, channels: int) -> npt.NDArray[np.float64]:
    """Sample normalized per-channel gains for one resonator mode."""
    if channels == 1:
        return np.ones(1, dtype=np.float64)

    gains = rng.uniform(0.2, 1.0, size=channels).astype(np.float64)
    norm = float(np.linalg.norm(gains))
    if norm <= 1e-12:
        return np.ones(channels, dtype=np.float64) / np.sqrt(float(channels))
    return gains / norm


def _late_tail_envelope(n: int, sr: int, start: int) -> npt.NDArray[np.float64]:
    """Create a smooth late-tail gate starting near ``start`` sample index."""
    env = np.ones(n, dtype=np.float64)
    if start <= 0:
        return env

    env[:start] = 0.0
    fade = min(start, max(8, int(0.03 * float(sr))))
    begin = max(0, start - fade)
    span = start - begin
    if span > 0:
        ramp = np.linspace(0.0, np.pi / 2.0, span, endpoint=False, dtype=np.float64)
        env[begin:start] = np.sin(ramp) ** 2
    return env


def _dc_block(signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """One-pole DC blocker for stable resonator excitation."""
    out = np.empty_like(signal)
    xm1 = 0.0
    ym1 = 0.0
    coeff = 0.995
    for i in range(signal.shape[0]):
        x = float(signal[i])
        y = x - xm1 + (coeff * ym1)
        out[i] = y
        xm1 = x
        ym1 = y
    return out


def _rms(x: npt.ArrayLike) -> float:
    """Return RMS with empty-array safety handling."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr), dtype=np.float64)))
