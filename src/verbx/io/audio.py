from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf


def read_audio(filepath: str | Path) -> tuple[np.ndarray, int]:
    """
    Load an audio file into a numpy array (float32).

    Args:
        filepath: Path to the audio file.

    Returns:
        Tuple of (audio array, sample rate).
        Audio is always 2D: (n_samples, n_channels).
    """
    data, sr = sf.read(str(filepath), dtype="float32")
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return data, sr


def write_audio(filepath: str | Path, audio: np.ndarray, sr: int):
    """
    Save a numpy array to an audio file (float32).

    Args:
        filepath: Path to save the audio file.
        audio: Audio array to save.
        sr: Sample rate.
    """
    # Ensure audio is float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    sf.write(str(filepath), audio, sr)


def iter_audio_blocks(
    filepath: str | Path, block_size: int = 4096
) -> Iterator[np.ndarray]:
    """
    Iterate over audio blocks from a file.

    Args:
        filepath: Path to the audio file.
        block_size: Block size in samples.

    Yields:
        Audio block as numpy array (n_samples, n_channels).
    """
    with sf.SoundFile(str(filepath)) as f:
        while f.tell() < f.frames:
            data = f.read(block_size, dtype="float32")
            if data.ndim == 1:
                data = data[:, np.newaxis]
            yield data


def ensure_mono_or_stereo(audio: np.ndarray) -> np.ndarray:
    """
    Ensure audio is mono or stereo. If > 2 channels, downmix to stereo.

    Args:
        audio: Input audio array (n_samples, n_channels).

    Returns:
        Audio array with 1 or 2 channels.
    """
    if audio.ndim == 1:
        return audio[:, np.newaxis]

    channels = audio.shape[1]
    if channels <= 2:
        return audio

    # Downmix to stereo
    left = np.mean(audio[:, ::2], axis=1)
    right = np.mean(audio[:, 1::2], axis=1)
    return np.stack([left, right], axis=1)


def peak_normalize(audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """
    Peak normalize audio to target dBFS.

    Args:
        audio: Input audio.
        target_dbfs: Target peak level in dBFS.

    Returns:
        Normalized audio.
    """
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio

    target_amp = 10 ** (target_dbfs / 20)
    gain = target_amp / peak
    return audio * gain


def soft_limiter(
    audio: np.ndarray, threshold_dbfs: float = -1.0, knee_db: float = 6.0
) -> np.ndarray:
    """
    Soft limiter using tanh.

    Args:
        audio: Input audio.
        threshold_dbfs: Threshold in dBFS.
        knee_db: Knee width in dB (unused in simple tanh implementation, keeping for API).

    Returns:
        Limited audio.
    """
    threshold = 10 ** (threshold_dbfs / 20)

    # Simple tanh limiting: audio / threshold -> tanh -> * threshold
    # But this limits to threshold.
    # To act as a limiter that *mostly* preserves dynamics below threshold:
    # We want f(x) = x for small x, and f(x) -> threshold for large x.
    # tanh(x) ~ x for small x.
    # So threshold * tanh(x / threshold) works.

    return threshold * np.tanh(audio / threshold)
