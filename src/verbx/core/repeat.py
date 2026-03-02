"""Repeat-pass orchestration for recursive reverb workflows.

This module keeps repeat chaining policy separate from engine internals:
engines render one pass, while :func:`repeat_process` handles pass counting,
optional per-pass normalization/targeting, and progress callbacks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

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
    """Apply engine processing repeatedly with optional per-pass post processing.

    Parameters
    ----------
    engine:
        Reverb engine used for each pass.
    audio:
        Input audio array. Converted to float32 internally.
    sr:
        Sample rate in Hz.
    n:
        Requested number of passes. Values below 1 are clamped to 1.
    post_pass_processor:
        Optional callback executed after each pass. Typical use:
        normalization/limiting per pass.
    progress_callback:
        Optional callback receiving ``(current_pass, total_passes)``.
    """
    passes = max(1, int(n))
    current = np.asarray(audio, dtype=np.float32)

    for idx in range(passes):
        current = engine.process(current, sr)
        if post_pass_processor is not None:
            # Allow pipeline-level control (LUFS, peak ceilings, etc.).
            current = post_pass_processor(current, idx + 1, passes)

        if progress_callback is not None:
            progress_callback(idx + 1, passes)

    return np.asarray(current, dtype=np.float32)


def process_pass_stream(
    engine: ReverbEngine,
    input_stream: Iterator[np.ndarray],
    sr: int,
    wet_decay: float = 1.0,
    normalize: bool = True,
) -> Iterator[np.ndarray]:
    """Process a stream of audio blocks through a reverb engine.

    Parameters
    ----------
    engine:
        The reverb engine to use.
    input_stream:
        An iterator yielding audio blocks as numpy arrays.
    sr:
        Sample rate in Hz.
    wet_decay:
        A gain multiplier applied to the processed block.
    normalize:
        If True, normalizes the processed block to its peak amplitude before yielding.
        If the block is completely silent, normalization is skipped.

    Yields
    ------
    numpy.ndarray
        The processed audio block.
    """
    for block in input_stream:
        block = np.asarray(block, dtype=np.float32)
        processed = engine.process(block, sr)
        processed = processed * wet_decay

        if normalize:
            peak = np.max(np.abs(processed))
            if peak > 0.0:
                processed = processed / peak

        yield processed
