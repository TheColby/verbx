"""Freeze-segment preparation for reverb input streams.

The v0.4 freeze path is intentionally simple: it extracts a user-selected region
and tiles it to the original program length so downstream engines can run with
the same framing assumptions as normal rendering.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

from verbx.io.audio import ensure_mono_or_stereo

AudioArray = npt.NDArray[np.float32]


def freeze_segment(
    audio: AudioArray,
    sr: int,
    start: float | None,
    end: float | None,
    mode: str = "loop",
    xfade_ms: float = 100.0,
) -> AudioArray:
    """Freeze a selected segment and loop it across the render input.

    The segment is made loop-safe with an equal-power boundary blend.

    Parameters
    ----------
    audio:
        Input program material as ``(samples, channels)`` or mono vector.
    sr:
        Sample rate in Hz.
    start, end:
        Freeze region in seconds. Invalid or missing bounds fall back to
        pass-through behavior.
    mode:
        Currently only ``"loop"`` is implemented. Other values preserve input.
    xfade_ms:
        Boundary crossfade duration used to suppress discontinuity clicks.
    """
    x = ensure_mono_or_stereo(audio)
    if mode != "loop":
        return x.copy()

    if start is None or end is None or end <= start:
        return x.copy()

    start_idx = max(0, int(start * sr))
    end_idx = min(x.shape[0], int(end * sr))
    if end_idx - start_idx < 2:
        return x.copy()

    segment = x[start_idx:end_idx, :].copy()
    loop_len = segment.shape[0]

    xfade = int(max(0.0, xfade_ms) * sr / 1000.0)
    xfade = min(xfade, max(1, loop_len // 4))

    if xfade > 0:
        # Equal-power crossfade keeps perceived level steady at loop boundary.
        theta = np.linspace(0.0, np.pi / 2.0, xfade, dtype=np.float32)
        fade_out = np.cos(theta)
        fade_in = np.sin(theta)
        blended = (segment[-xfade:, :] * fade_out[:, np.newaxis]) + (
            segment[:xfade, :] * fade_in[:, np.newaxis]
        )
        segment[:xfade, :] = blended

    # Tile the frozen buffer to preserve input render duration for v0.4.
    repeats = int(np.ceil(x.shape[0] / loop_len))
    frozen = np.tile(segment, (repeats, 1))[: x.shape[0], :]
    return np.asarray(frozen, dtype=np.float32)


def freeze_generator(
    loop_buffer: np.ndarray, block_size: int = 4096
) -> Iterator[np.ndarray]:
    """
    Yield infinite blocks from the loop buffer.
    """
    idx = 0
    loop_len = len(loop_buffer)

    while True:
        if loop_len == 0:
            shape = (block_size, *loop_buffer.shape[1:])
            yield np.zeros(shape, dtype=loop_buffer.dtype)
            continue

        if idx + block_size <= loop_len:
            yield loop_buffer[idx : idx + block_size]
            idx = (idx + block_size) % loop_len
        else:
            parts = [loop_buffer[idx:]]
            rem = block_size - len(parts[0])
            while rem > 0:
                chunk = min(rem, loop_len)
                parts.append(loop_buffer[:chunk])
                rem -= chunk
            yield np.concatenate(parts, axis=0)
            idx = (idx + block_size) % loop_len
