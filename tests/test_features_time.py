from __future__ import annotations

import numpy as np

from verbx.analysis.features_time import zero_crossing_rate


def test_zero_crossing_rate_happy_path() -> None:
    # 4 samples, signs: +, -, -, + -> 2 crossings
    # mono length = 4. crossings = 2. denom = max(3, 1) = 3. zcr = 2/3
    audio = np.array([[1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
    assert np.isclose(zero_crossing_rate(audio), 2.0 / 3.0)


def test_zero_crossing_rate_empty_array() -> None:
    audio = np.empty((0, 2), dtype=np.float32)
    assert zero_crossing_rate(audio) == 0.0


def test_zero_crossing_rate_single_element() -> None:
    audio = np.array([[1.0, 1.0]], dtype=np.float32)
    assert zero_crossing_rate(audio) == 0.0


def test_zero_crossing_rate_no_crossings() -> None:
    audio = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    assert zero_crossing_rate(audio) == 0.0


def test_zero_crossing_rate_all_crossings() -> None:
    audio = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
    assert np.isclose(zero_crossing_rate(audio), 3.0 / 3.0)
