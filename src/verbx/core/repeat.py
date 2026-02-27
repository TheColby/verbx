"""Repeat-processing with safety normalization."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from verbx.core.engine_base import ReverbEngine

AudioArray = npt.NDArray[np.float32]
ProgressCallback = Callable[[int, int], None]
PassPostProcessor = Callable[[AudioArray, int, int], AudioArray]


def repeat_process(
    engine: ReverbEngine,
    audio: AudioArray,
    sr: int,
    n: int,
    post_pass_processor: PassPostProcessor | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AudioArray:
    """Apply engine processing repeatedly with optional per-pass post processing."""
    passes = max(1, int(n))
    current = np.asarray(audio, dtype=np.float32)

    for idx in range(passes):
        current = engine.process(current, sr)
        if post_pass_processor is not None:
            current = post_pass_processor(current, idx + 1, passes)

        if progress_callback is not None:
            progress_callback(idx + 1, passes)

    return np.asarray(current, dtype=np.float32)
