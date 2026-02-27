"""Audio I/O helpers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

AudioArray = npt.NDArray[np.float32]


def read_audio(path: str) -> tuple[AudioArray, int]:
    """Read an audio file as float32 with shape (samples, channels)."""
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    array = np.asarray(audio, dtype=np.float32)
    return array, int(sr)


def write_audio(path: str, audio: AudioArray, sr: int) -> None:
    """Write audio as float32."""
    output = ensure_mono_or_stereo(audio).astype(np.float32, copy=False)
    sf.write(file=path, data=output, samplerate=sr)


def validate_audio_path(path: str) -> None:
    """Raise an error if input path does not exist."""
    if not Path(path).exists():
        msg = f"Input audio file not found: {path}"
        raise FileNotFoundError(msg)


def iter_audio_blocks(path: str, block_size: int) -> Iterator[AudioArray]:
    """Yield float32 blocks with shape (samples, channels)."""
    with sf.SoundFile(path, mode="r") as snd:
        for block in snd.blocks(blocksize=block_size, dtype="float32", always_2d=True):
            yield np.asarray(block, dtype=np.float32)


def ensure_mono_or_stereo(audio: npt.ArrayLike) -> AudioArray:
    """Ensure audio has shape (samples, channels) and float32 dtype.

    The function keeps arbitrary channel counts and does not force mono/stereo downmixing.
    """
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array[:, np.newaxis]
    if array.ndim != 2:
        msg = f"Audio must be 1D or 2D, received shape {array.shape!r}"
        raise ValueError(msg)
    return array


def peak_normalize(audio: AudioArray, target_dbfs: float = -1.0) -> AudioArray:
    """Scale audio so absolute peak reaches target dBFS."""
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return audio.copy()
    target = float(10.0 ** (target_dbfs / 20.0))
    gain = target / peak
    return np.asarray(audio * gain, dtype=np.float32)


def soft_limiter(
    audio: AudioArray, threshold_dbfs: float = -1.0, knee_db: float = 6.0
) -> AudioArray:
    """Apply a soft-knee limiter using a tanh saturation stage."""
    threshold = float(10.0 ** (threshold_dbfs / 20.0))
    threshold = max(threshold, 1e-6)
    knee = max(knee_db, 0.1)
    drive = 1.0 + (knee / 6.0)

    x = np.asarray(audio, dtype=np.float32)
    abs_x = np.abs(x)
    out = x.copy()

    mask = abs_x > threshold
    if np.any(mask):
        sign = np.sign(x[mask])
        scaled = (abs_x[mask] - threshold) / threshold
        shaped = threshold + threshold * np.tanh(scaled * drive) / np.tanh(drive)
        out[mask] = sign * shaped

    return np.asarray(out, dtype=np.float32)
