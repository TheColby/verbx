"""Shimmer processing utilities."""

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

AudioArray = npt.NDArray[np.float32]


@dataclass(slots=True)
class ShimmerConfig:
    """Configuration for shimmer processing."""

    enabled: bool = False
    semitones: float = 12.0
    mix: float = 0.25
    feedback: float = 0.35
    highcut: float | None = 10_000.0
    lowcut: float | None = 300.0


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
        feedback = float(np.clip(self._cfg.feedback, 0.0, 0.98))
        if mix <= 0.0:
            return x

        bandlimited = _bandlimit(x, sr, lowcut=self._cfg.lowcut, highcut=self._cfg.highcut)
        shifted = _pitch_shift_audio(bandlimited, sr, self._cfg.semitones)

        if self._feedback_state is None or self._feedback_state.shape != shifted.shape:
            self._feedback_state = np.zeros_like(shifted)

        shimmer_wet = shifted + (feedback * self._feedback_state)
        self._feedback_state = shimmer_wet.astype(np.float32)

        out = ((1.0 - mix) * x) + (mix * shimmer_wet)
        out = soft_limiter(np.asarray(out, dtype=np.float32), threshold_dbfs=-1.0, knee_db=5.0)
        return np.asarray(out, dtype=np.float32)


def _pitch_shift_audio(audio: AudioArray, sr: int, semitones: float) -> AudioArray:
    x = ensure_mono_or_stereo(audio)
    if abs(semitones) < 1e-6:
        return x.copy()

    out = np.zeros_like(x, dtype=np.float32)
    for ch in range(x.shape[1]):
        signal = x[:, ch].astype(np.float32)
        if librosa is not None:
            shifted = librosa.effects.pitch_shift(signal, sr=sr, n_steps=semitones)
            if shifted.shape[0] != signal.shape[0]:
                shifted = librosa.util.fix_length(shifted, size=signal.shape[0])
            out[:, ch] = shifted.astype(np.float32)
            continue

        ratio = float(2.0 ** (semitones / 12.0))
        frac = Fraction(ratio).limit_denominator(1000)
        # Resample up then back down to approximate a simple pitch shift.
        tmp = np.asarray(
            np.interp(
                np.linspace(0.0, signal.shape[0] - 1.0, max(1, int(signal.shape[0] / ratio))),
                np.arange(signal.shape[0]),
                signal,
            ),
            dtype=np.float32,
        )
        restored = np.asarray(
            np.interp(
                np.linspace(0.0, tmp.shape[0] - 1.0, signal.shape[0]),
                np.arange(tmp.shape[0]),
                tmp,
            ),
            dtype=np.float32,
        )
        scale = np.sqrt(max(frac.numerator, 1) / max(frac.denominator, 1))
        out[:, ch] = restored * np.float32(scale)

    return out


def _bandlimit(
    audio: AudioArray, sr: int, lowcut: float | None, highcut: float | None
) -> AudioArray:
    x = ensure_mono_or_stereo(audio)
    out = x.copy()

    if lowcut is not None and lowcut > 1.0 and lowcut < (0.5 * sr):
        sos = butter(2, lowcut / (0.5 * sr), btype="highpass", output="sos")
        zi = sosfilt_zi(sos)
        for ch in range(out.shape[1]):
            y, _ = sosfilt(sos, out[:, ch], zi=zi * out[0, ch])
            out[:, ch] = y.astype(np.float32)

    if highcut is not None and highcut > 10.0 and highcut < (0.5 * sr):
        sos = butter(2, highcut / (0.5 * sr), btype="lowpass", output="sos")
        for ch in range(out.shape[1]):
            try:
                filtered = sosfiltfilt(sos, out[:, ch]).astype(np.float32)
            except ValueError:
                fallback = sosfilt(sos, out[:, ch])
                if isinstance(fallback, tuple):
                    fallback = fallback[0]
                filtered = np.asarray(fallback, dtype=np.float32)
            out[:, ch] = filtered

    return np.asarray(out, dtype=np.float32)
