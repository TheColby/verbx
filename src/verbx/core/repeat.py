"""Repeat-processing with safety normalization."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from verbx.core.engine_base import ReverbEngine
from verbx.io.audio import peak_normalize, soft_limiter

AudioArray = npt.NDArray[np.float32]
ProgressCallback = Callable[[int, int], None]


def repeat_process(
    engine: ReverbEngine,
    audio: AudioArray,
    sr: int,
    n: int,
    target_dbfs: float = -1.0,
    limiter: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> AudioArray:
    """Apply engine processing repeatedly with safety conditioning."""
    passes = max(1, int(n))
    current = np.asarray(audio, dtype=np.float32)

    for idx in range(passes):
        current = engine.process(current, sr)
        if limiter:
            current = soft_limiter(current, threshold_dbfs=target_dbfs, knee_db=6.0)
        current = peak_normalize(current, target_dbfs=target_dbfs)

        if progress_callback is not None:
            progress_callback(idx + 1, passes)

    return np.asarray(current, dtype=np.float32)
