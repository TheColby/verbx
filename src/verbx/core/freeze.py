"""Freeze-segment preparation for reverb input streams.

The v0.4 freeze path is intentionally simple: it extracts a user-selected region
and tiles it to the original program length so downstream engines can run with
the same framing assumptions as normal rendering.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from verbx.io.audio import ensure_mono_or_stereo

AudioArray = npt.NDArray[np.float32]


def create_crossfaded_loop(
    audio: np.ndarray, sr: int, start: float, end: float, xfade_ms: float = 100.0
) -> np.ndarray:
    """
    Create a seamless loop from an audio segment using crossfading.

    Args:
        audio: Input audio array.
        sr: Sample rate.
        start: Start time (s).
        end: End time (s).
        xfade_ms: Crossfade duration (ms).

    Returns:
        The loop buffer (numpy array).
    """
    x = ensure_mono_or_stereo(audio)
    start_idx = max(0, int(start * sr))
    end_idx = min(x.shape[0], int(end * sr))

    if end_idx <= start_idx:
        return np.array([], dtype=np.float32)

    segment = x[start_idx:end_idx, :].copy()
    loop_len = segment.shape[0]

    xfade = int(max(0.0, xfade_ms) * sr / 1000.0)
    xfade = min(xfade, max(1, loop_len // 4))

    if xfade > 0 and loop_len >= xfade * 2:
        # Equal-power crossfade keeps perceived level steady at loop boundary.
        theta = np.linspace(0.0, np.pi / 2.0, xfade, dtype=np.float32)
        fade_out = np.cos(theta)
        fade_in = np.sin(theta)

        # Mix the end of the segment into the start of the segment
        blended = (segment[-xfade:, :] * fade_out[:, np.newaxis]) + (
            segment[:xfade, :] * fade_in[:, np.newaxis]
        )
        segment[:xfade, :] = blended

        # Truncate the segment to remove the tail that was crossfaded
        # since it's already mixed into the front.
        segment = segment[:-xfade, :]

    return segment


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
