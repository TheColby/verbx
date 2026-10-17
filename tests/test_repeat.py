from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from verbx.core.engine_base import ReverbEngine
from verbx.core.repeat import process_pass_stream


class MockEngine(ReverbEngine):
    """A mock engine that simply multiplies input by a constant for tracking."""

    def __init__(self, multiplier: float = 2.0):
        self.multiplier = multiplier

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return audio * self.multiplier


def test_process_pass_stream_basic() -> None:
    engine = MockEngine(multiplier=1.0)
    sr = 48_000

    # Create a stream of two blocks
    blocks = [
        np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32),
        np.array([[0.1, 0.2], [-0.1, -0.2]], dtype=np.float32),
    ]

    def input_stream() -> Iterator[np.ndarray]:
        yield from blocks

    # No normalization, just wet_decay=0.5
    result_stream = process_pass_stream(
        engine, input_stream(), sr, wet_decay=0.5, normalize=False
    )
    result_blocks = list(result_stream)

    assert len(result_blocks) == 2
    np.testing.assert_allclose(result_blocks[0], blocks[0] * 0.5)
    np.testing.assert_allclose(result_blocks[1], blocks[1] * 0.5)


def test_process_pass_stream_normalize() -> None:
    engine = MockEngine(multiplier=2.0)
    sr = 44_100

    blocks = [
        np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32), # peak after engine is 0.8
    ]

    def input_stream() -> Iterator[np.ndarray]:
        yield from blocks

    # Engine multiplies by 2.0 -> peaks at 0.8
    # Normalize is True, so it should scale back by /0.8, effectively resulting
    # in the same relative shape but peak 1.0
    result_stream = process_pass_stream(
        engine, input_stream(), sr, wet_decay=1.0, normalize=True
    )
    result_blocks = list(result_stream)

    assert len(result_blocks) == 1
    output = result_blocks[0]
    assert np.isclose(np.max(np.abs(output)), 1.0)

    expected = (blocks[0] * 2.0) / 0.8
    np.testing.assert_allclose(output, expected)


def test_process_pass_stream_silent_normalize() -> None:
    engine = MockEngine(multiplier=1.0)
    sr = 44_100

    blocks = [
        np.zeros((100, 2), dtype=np.float32), # silent block
    ]

    def input_stream() -> Iterator[np.ndarray]:
        yield from blocks

    result_stream = process_pass_stream(
        engine, input_stream(), sr, wet_decay=1.0, normalize=True
    )
    result_blocks = list(result_stream)

    assert len(result_blocks) == 1
    output = result_blocks[0]
    assert np.max(np.abs(output)) == 0.0


def test_process_pass_stream_empty() -> None:
    engine = MockEngine(multiplier=1.0)
    sr = 48_000

    def empty_stream() -> Iterator[np.ndarray]:
        yield from []

    result_stream = process_pass_stream(
        engine, empty_stream(), sr, wet_decay=1.0, normalize=True
    )
    result_blocks = list(result_stream)

    assert len(result_blocks) == 0
