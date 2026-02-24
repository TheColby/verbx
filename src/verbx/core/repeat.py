from typing import Iterator

import numpy as np

from verbx.core.engine_base import ReverbEngine


def process_pass_stream(
    engine: ReverbEngine,
    input_stream: Iterator[np.ndarray],
    sr: int,
    wet_decay: float = 1.0,
    normalize: bool = True,
) -> Iterator[np.ndarray]:
    """
    Process one pass of repeat chaining.

    Args:
        engine: The reverb engine.
        input_stream: Iterator of audio blocks.
        sr: Sample rate.
        wet_decay: Gain factor applied to this pass.
        normalize: Whether to normalize output (block-wise? No, global requires 2 passes).

    Yields:
        Processed audio blocks.
    """
    # Note: Normalization usually requires global knowledge (2 passes).
    # For streaming, we can use a limiter or rolling normalization.
    # Or strict normalization requires writing to temp file first.
    # Here we apply wet_decay.
    # Normalization will be handled by the pipeline/outer loop which uses temp files.

    for block in input_stream:
        processed = engine.process(block, sr)

        if wet_decay != 1.0:
            processed *= wet_decay

        yield processed
