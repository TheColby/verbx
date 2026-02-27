"""Freeze-related processing."""

from __future__ import annotations

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
        theta = np.linspace(0.0, np.pi / 2.0, xfade, dtype=np.float32)
        fade_out = np.cos(theta)
        fade_in = np.sin(theta)
        blended = (segment[-xfade:, :] * fade_out[:, np.newaxis]) + (
            segment[:xfade, :] * fade_in[:, np.newaxis]
        )
        segment[:xfade, :] = blended

    repeats = int(np.ceil(x.shape[0] / loop_len))
    frozen = np.tile(segment, (repeats, 1))[: x.shape[0], :]
    return np.asarray(frozen, dtype=np.float32)
