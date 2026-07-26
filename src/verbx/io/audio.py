"""Audio I/O and gain-utility helpers.

All public read/write utilities normalize to a predictable in-memory layout:
``float64`` arrays with shape ``(samples, channels)``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy.signal import resample_poly

AudioArray = npt.NDArray[np.float64]
LimiterMode = Literal["tanh", "arctan", "softsign", "hard"]
LimiterDetect = Literal["peak", "rms"]


def read_audio(path: str) -> tuple[AudioArray, int]:
    """Read audio as float64 with shape ``(samples, channels)``.

    ``soundfile`` handles WAV/FLAC/AIFF and other libsndfile-backed formats.
    """
    audio, sr = sf.read(path, always_2d=True, dtype="float64")
    array = np.asarray(audio, dtype=np.float64)
    return array, int(sr)


def write_audio(
    path: str,
    audio: AudioArray,
    sr: int,
    subtype: str | None = None,
    format: str | None = None,
) -> None:
    """Write audio with optional subtype override.

    The signal is always converted to float64 before writing to keep engine
    behavior deterministic across containers.
    """
    output = ensure_mono_or_stereo(audio).astype(np.float64, copy=False)
    sf.write(file=path, data=output, samplerate=sr, subtype=subtype, format=format)


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
    audio: AudioArray,
    threshold_dbfs: float = -1.0,
    knee_db: float = 6.0,
    *,
    sr: int = 48_000,
    ceiling_dbfs: float | None = None,
    mode: LimiterMode = "tanh",
    detect: LimiterDetect = "peak",
    drive: float = 1.0,
    mix: float = 1.0,
    attack_ms: float = 0.5,
    release_ms: float = 80.0,
    lookahead_ms: float = 1.5,
    stereo_link: bool = True,
    oversample: int = 1,
    pre_gain_db: float = 0.0,
    post_gain_db: float = 0.0,
    dc_block: bool = False,
) -> AudioArray:
    """Apply a configurable safety limiter to offline audio."""
    x = ensure_mono_or_stereo(audio)
    if x.shape[0] == 0:
        return x.copy()

    resolved_ceiling_dbfs = threshold_dbfs if ceiling_dbfs is None else float(ceiling_dbfs)
    threshold = _dbfs_to_linear(min(float(threshold_dbfs), resolved_ceiling_dbfs))
    ceiling = _dbfs_to_linear(resolved_ceiling_dbfs)
    threshold = min(threshold, ceiling)
    mix_amount = float(np.clip(mix, 0.0, 1.0))
    if mix_amount <= 0.0:
        return np.asarray(x, dtype=np.float64)

    processed = np.asarray(x, dtype=np.float64)
    if dc_block:
        processed = _apply_dc_block(processed)
    processed *= float(10.0 ** (float(pre_gain_db) / 20.0))

    over = max(1, int(oversample))
    oversampled = (
        resample_poly(processed, up=over, down=1, axis=0)
        if over > 1
        else np.asarray(processed, dtype=np.float64)
    )
    os_sr = max(1, int(sr)) * over
    delayed = _delay_audio(oversampled, os_sr=os_sr, lookahead_ms=lookahead_ms)
    envelope = _compute_limiter_envelope(
        oversampled,
        detect=detect,
        stereo_link=bool(stereo_link),
        attack_ms=attack_ms,
        release_ms=release_ms,
        sr=os_sr,
    )
    gain = _compute_limiter_gain(
        envelope,
        threshold=threshold,
        ceiling=ceiling,
        knee_db=knee_db,
        mode=mode,
        drive=drive,
    )
    limited = delayed * gain
    limited = (mix_amount * limited) + ((1.0 - mix_amount) * delayed)
    limited *= float(10.0 ** (float(post_gain_db) / 20.0))
    limited = np.clip(limited, -ceiling, ceiling)

    if over > 1:
        limited = resample_poly(limited, up=1, down=over, axis=0)
        limited = np.asarray(limited[: x.shape[0], :], dtype=np.float64)

    return np.asarray(np.clip(limited, -ceiling, ceiling), dtype=np.float64)


def _dbfs_to_linear(dbfs: float) -> float:
    """Convert dBFS into a linear amplitude scalar."""
    return float(max(10.0 ** (float(dbfs) / 20.0), 1e-9))


def _apply_dc_block(audio: AudioArray) -> AudioArray:
    """Apply a gentle first-order DC blocker."""
    x = np.asarray(audio, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    pole = np.float64(0.995)
    for ch in range(x.shape[1]):
        last_x = np.float64(0.0)
        last_y = np.float64(0.0)
        for idx in range(x.shape[0]):
            sample = np.float64(x[idx, ch])
            y = sample - last_x + (pole * last_y)
            out[idx, ch] = y
            last_x = sample
            last_y = y
    return np.asarray(out, dtype=np.float64)


def _delay_audio(audio: AudioArray, *, os_sr: int, lookahead_ms: float) -> AudioArray:
    """Delay the signal so the detector can look ahead."""
    x = np.asarray(audio, dtype=np.float64)
    lookahead_samples = max(0, round(float(os_sr) * (max(0.0, float(lookahead_ms)) / 1000.0)))
    if lookahead_samples <= 0:
        return x.copy()
    padded = np.pad(x, ((lookahead_samples, 0), (0, 0)), mode="constant")
    return np.asarray(padded[: x.shape[0], :], dtype=np.float64)


def _compute_limiter_envelope(
    audio: AudioArray,
    *,
    detect: LimiterDetect,
    stereo_link: bool,
    attack_ms: float,
    release_ms: float,
    sr: int,
) -> AudioArray:
    """Return a smoothed amplitude envelope for the limiter detector."""
    x = np.asarray(audio, dtype=np.float64)
    if str(detect).strip().lower() == "rms":
        squared = np.square(x)
        if stereo_link:
            linked_power = np.mean(squared, axis=1, keepdims=True)
            smoothed = _smooth_detector(
                linked_power,
                attack_ms=attack_ms,
                release_ms=release_ms,
                sr=sr,
            )
            return np.asarray(np.sqrt(np.maximum(smoothed, 1e-18)), dtype=np.float64)
        smoothed = _smooth_detector(
            squared,
            attack_ms=attack_ms,
            release_ms=release_ms,
            sr=sr,
        )
        return np.asarray(np.sqrt(np.maximum(smoothed, 1e-18)), dtype=np.float64)
    else:
        detector = np.abs(x)
    if stereo_link:
        linked = np.max(detector, axis=1, keepdims=True)
        return _smooth_detector(linked, attack_ms=attack_ms, release_ms=release_ms, sr=sr)
    return _smooth_detector(detector, attack_ms=attack_ms, release_ms=release_ms, sr=sr)


def _smooth_detector(
    detector: AudioArray,
    *,
    attack_ms: float,
    release_ms: float,
    sr: int,
) -> AudioArray:
    """Apply one-pole attack/release smoothing to a detector signal."""
    x = np.asarray(detector, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    attack_seconds = max(1e-5, float(attack_ms) / 1000.0)
    release_seconds = max(1e-5, float(release_ms) / 1000.0)
    attack_alpha = float(np.exp(-1.0 / (attack_seconds * float(max(1, sr)))))
    release_alpha = float(np.exp(-1.0 / (release_seconds * float(max(1, sr)))))
    state = np.zeros((x.shape[1],), dtype=np.float64)
    for idx in range(x.shape[0]):
        sample = x[idx, :]
        rising = sample > state
        state = np.where(
            rising,
            (attack_alpha * state) + ((1.0 - attack_alpha) * sample),
            (release_alpha * state) + ((1.0 - release_alpha) * sample),
        )
        out[idx, :] = state
    return np.asarray(out, dtype=np.float64)


def _compute_limiter_gain(
    envelope: AudioArray,
    *,
    threshold: float,
    ceiling: float,
    knee_db: float,
    mode: LimiterMode,
    drive: float,
) -> AudioArray:
    """Map detector envelope into a gain curve."""
    env = np.asarray(envelope, dtype=np.float64)
    knee_start = threshold / float(10.0 ** (max(0.0, float(knee_db)) / 40.0))
    knee_start = min(knee_start, threshold)
    over = np.maximum(env - knee_start, 0.0)
    norm = over / max(threshold - knee_start, 1e-9)
    curve = _limiter_transfer(norm, mode=mode, drive=drive)
    shaped = np.where(
        env <= knee_start,
        env,
        knee_start + ((ceiling - knee_start) * curve),
    )
    gain = np.minimum(1.0, shaped / np.maximum(env, 1e-9))
    return np.asarray(np.clip(gain, 0.0, 1.0), dtype=np.float64)


def _limiter_transfer(norm: AudioArray, *, mode: LimiterMode, drive: float) -> AudioArray:
    """Return a normalized 0..1 shaping curve for limiter overflow."""
    x = np.asarray(np.maximum(norm, 0.0) * max(1e-6, float(drive)), dtype=np.float64)
    mode_name = str(mode).strip().lower()
    if mode_name == "hard":
        shaped = np.where(x > 0.0, 1.0, 0.0)
    elif mode_name == "arctan":
        shaped = np.arctan(x) / (np.pi / 2.0)
    elif mode_name == "softsign":
        shaped = x / (1.0 + x)
    else:
        shaped = np.tanh(x)
    return np.asarray(np.clip(shaped, 0.0, 1.0), dtype=np.float64)
