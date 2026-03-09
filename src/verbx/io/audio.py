"""Audio I/O and gain-utility helpers.

All public read/write utilities normalize to a predictable in-memory layout:
``float64`` arrays with shape ``(samples, channels)``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

AudioArray = npt.NDArray[np.float64]


def read_audio(path: str) -> tuple[AudioArray, int]:
    """Read audio as float64 with shape ``(samples, channels)``.

    ``soundfile`` handles WAV/FLAC/AIFF and other libsndfile-backed formats.
    """
    audio, sr = sf.read(path, always_2d=True, dtype="float64")
    array = np.asarray(audio, dtype=np.float64)
    return array, int(sr)


def write_audio(path: str, audio: AudioArray, sr: int, subtype: str | None = None) -> None:
    """Write audio with optional subtype override.

    The signal is always converted to float64 before writing to keep engine
    behavior deterministic across containers.
    """
    output = ensure_mono_or_stereo(audio).astype(np.float64, copy=False)
    sf.write(file=path, data=output, samplerate=sr, subtype=subtype)


def validate_audio_path(path: str) -> None:
    """Raise ``FileNotFoundError`` when input path does not exist."""
    if not Path(path).exists():
        msg = f"Input audio file not found: {path}"
        raise FileNotFoundError(msg)


def iter_audio_blocks(path: str, block_size: int) -> Iterator[AudioArray]:
    """Stream file blocks as float64 ``(samples, channels)`` arrays."""
    with sf.SoundFile(path, mode="r") as snd:
        for block in snd.blocks(blocksize=block_size, dtype="float64", always_2d=True):
            yield np.asarray(block, dtype=np.float64)


def ensure_mono_or_stereo(audio: npt.ArrayLike) -> AudioArray:
    """Ensure audio has shape (samples, channels) and float64 dtype.

    The function keeps arbitrary channel counts and does not force mono/stereo
    downmixing despite the historical name.
    """
    array = np.asarray(audio, dtype=np.float64)
    if array.ndim == 1:
        return array[:, np.newaxis]
    if array.ndim != 2:
        msg = f"Audio must be 1D or 2D, received shape {array.shape!r}"
        raise ValueError(msg)
    return array


def peak_normalize(audio: AudioArray, target_dbfs: float = -1.0) -> AudioArray:
    """Scale signal so absolute sample peak reaches ``target_dbfs``."""
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return audio.copy()
    target = float(10.0 ** (target_dbfs / 20.0))
    gain = target / peak
    return np.asarray(audio * gain, dtype=np.float64)


def soft_limiter(
    audio: AudioArray, threshold_dbfs: float = -1.0, knee_db: float = 6.0
) -> AudioArray:
    """Apply a soft-knee limiter using a tanh saturation stage.

    This is a light safety limiter, not a full mastering limiter.
    """
    threshold = float(10.0 ** (threshold_dbfs / 20.0))
    threshold = max(threshold, 1e-6)
    knee = max(knee_db, 0.1)
    drive = 1.0 + (knee / 6.0)

    x = np.asarray(audio, dtype=np.float64)
    abs_x = np.abs(x)
    out = x.copy()

    mask = abs_x > threshold
    if np.any(mask):
        sign = np.sign(x[mask])
        scaled = (abs_x[mask] - threshold) / threshold
        # Soft-knee mapping keeps slope continuous around threshold.
        shaped = threshold + threshold * np.tanh(scaled * drive) / np.tanh(drive)
        out[mask] = sign * shaped

    return np.asarray(out, dtype=np.float64)
