"""Shimmer processing utilities.

The shimmer stage is treated as a wet-path enhancer:

- optional band-limited input,
- pitch-shifted layer (librosa if available),
- controllable feedback memory,
- safety limiting before returning to the render chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

from verbx.io.audio import ensure_mono_or_stereo, soft_limiter

try:
    import librosa  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    librosa = None

AudioArray = npt.NDArray[np.float64]


@dataclass(slots=True)
class ShimmerConfig:
    """Configuration for shimmer processing."""

    enabled: bool = False
    semitones: float = 12.0
    mix: float = 0.25
    feedback: float = 0.35
    highcut: float | None = 10_000.0
    lowcut: float | None = 300.0
    unsafe_self_oscillate: bool = False
    spatial: bool = False
    spread_cents: float = 8.0
    decorrelation_ms: float = 1.5


class ShimmerProcessor:
    """Block-compatible shimmer processor with feedback memory."""

    __slots__ = ("_cfg", "_feedback_state")

    def __init__(self, cfg: ShimmerConfig) -> None:
        self._cfg = cfg
        self._feedback_state: AudioArray | None = None

    def process(self, wet: AudioArray, sr: int) -> AudioArray:
        """Apply shimmer enhancement to wet signal."""
        x = ensure_mono_or_stereo(wet)
        if not self._cfg.enabled:
            return x

        mix = float(np.clip(self._cfg.mix, 0.0, 1.0))
        feedback_cap = 1.25 if self._cfg.unsafe_self_oscillate else 0.98
        feedback = float(np.clip(self._cfg.feedback, 0.0, feedback_cap))
        if mix <= 0.0:
            return x

        bandlimited = _bandlimit(x, sr, lowcut=self._cfg.lowcut, highcut=self._cfg.highcut)
        shifted = _pitch_shift_audio(bandlimited, sr, self._cfg.semitones)
        shifted = _apply_spatial_decorrelation(
            shifted,
            sr=sr,
            enabled=bool(self._cfg.spatial),
            spread_cents=float(self._cfg.spread_cents),
            decorrelation_ms=float(self._cfg.decorrelation_ms),
        )

        if self._feedback_state is None or self._feedback_state.shape != shifted.shape:
            self._feedback_state = np.zeros_like(shifted)

        shimmer_wet = shifted + (feedback * self._feedback_state)
        self._feedback_state = shimmer_wet.astype(np.float64)

        out = ((1.0 - mix) * x) + (mix * shimmer_wet)
        out = soft_limiter(np.asarray(out, dtype=np.float64), threshold_dbfs=-1.0, knee_db=5.0)
        return np.asarray(out, dtype=np.float64)


def _pitch_shift_audio(audio: AudioArray, sr: int, semitones: float) -> AudioArray:
    """Pitch-shift each channel by semitone offset.

    ``librosa`` is used when available for better quality. A deterministic
    interpolation fallback is kept for minimal dependency environments.
    """
    x = ensure_mono_or_stereo(audio)
    if abs(semitones) < 1e-6:
        return x.copy()

    out = np.zeros_like(x, dtype=np.float64)
    for ch in range(x.shape[1]):
        signal = x[:, ch].astype(np.float64)
        if librosa is not None:
            shifted = librosa.effects.pitch_shift(signal, sr=sr, n_steps=semitones)
            if shifted.shape[0] != signal.shape[0]:
                shifted = librosa.util.fix_length(shifted, size=signal.shape[0])
            out[:, ch] = shifted.astype(np.float64)
            continue

        # librosa isn't here, so we do the janky version.
        # It works, but sounds rougher on fast material.
        ratio = float(2.0 ** (semitones / 12.0))
        # Fraction gives us a rational approximation of the pitch ratio so the
        # scale correction below doesn't have to deal with irrational floats.
        frac = Fraction(ratio).limit_denominator(1000)
        # Resample up then back down to approximate a simple pitch shift.
        tmp = np.asarray(
            np.interp(
                np.linspace(0.0, signal.shape[0] - 1.0, max(1, int(signal.shape[0] / ratio))),
                np.arange(signal.shape[0]),
                signal,
            ),
            dtype=np.float64,
        )
        restored = np.asarray(
            np.interp(
                np.linspace(0.0, tmp.shape[0] - 1.0, signal.shape[0]),
                np.arange(tmp.shape[0]),
                tmp,
            ),
            dtype=np.float64,
        )
        scale = np.sqrt(max(frac.numerator, 1) / max(frac.denominator, 1))
        out[:, ch] = restored * np.float64(scale)

    return out


def _bandlimit(
    audio: AudioArray, sr: int, lowcut: float | None, highcut: float | None
) -> AudioArray:
    """Apply optional high/low cut filtering around shimmer pitch stage."""
    x = ensure_mono_or_stereo(audio)
    if x.shape[0] == 0:
        return x.copy()
    out = x.copy()

    if lowcut is not None and lowcut > 1.0 and lowcut < (0.5 * sr):
        sos = butter(2, lowcut / (0.5 * sr), btype="highpass", output="sos")
        zi = sosfilt_zi(sos)
        for ch in range(out.shape[1]):
            y, _ = sosfilt(sos, out[:, ch], zi=zi * out[0, ch])
            out[:, ch] = y.astype(np.float64)

    if highcut is not None and highcut > 10.0 and highcut < (0.5 * sr):
        sos = butter(2, highcut / (0.5 * sr), btype="lowpass", output="sos")
        for ch in range(out.shape[1]):
            try:
                filtered = sosfiltfilt(sos, out[:, ch]).astype(np.float64)
            except ValueError:
                # sosfiltfilt needs the signal to be longer than padlen (3*filter_order).
                # Very short shimmer blocks can hit this, so fall back to causal filter.
                fallback = sosfilt(sos, out[:, ch])
                if isinstance(fallback, tuple):
                    fallback = fallback[0]
                filtered = np.asarray(fallback, dtype=np.float64)
            out[:, ch] = filtered

    return np.asarray(out, dtype=np.float64)


def _apply_spatial_decorrelation(
    audio: AudioArray,
    *,
    sr: int,
    enabled: bool,
    spread_cents: float,
    decorrelation_ms: float,
) -> AudioArray:
    """Apply lightweight multichannel shimmer decorrelation."""
    x = ensure_mono_or_stereo(audio)
    if not enabled or x.shape[1] <= 1:
        return x

    y = np.asarray(x, dtype=np.float64).copy()
    spread = float(max(0.0, spread_cents))
    max_delay = int(max(0, round(float(max(0.0, decorrelation_ms)) * float(sr) / 1000.0)))
    if max_delay <= 0 and spread <= 0.0:
        return y

    # Deterministic per-channel offsets to avoid mono collapse in large beds.
    channel_positions = np.linspace(-1.0, 1.0, y.shape[1], dtype=np.float64)
    for ch in range(y.shape[1]):
        cents = spread * float(channel_positions[ch])
        gain = float(2.0 ** (cents / 1200.0))
        delayed = y[:, ch]
        if max_delay > 0:
            delay = round(abs(float(channel_positions[ch])) * max_delay)
            if delay > 0:
                delayed = np.pad(delayed[:-delay], (delay, 0), mode="constant")
        # Tiny deterministic variation; keep energy bounded.
        y[:, ch] = np.asarray(0.92 * delayed * gain, dtype=np.float64)
    return np.asarray(y, dtype=np.float64)
