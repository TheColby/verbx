from __future__ import annotations

import numpy as np

from verbx.analysis.analyzer import AudioAnalyzer

EXPECTED_KEYS = {
    "duration",
    "rms",
    "peak",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "stereo_width",
    "dynamic_range",
    "crest_factor",
}


def test_analyzer_returns_expected_keys() -> None:
    analyzer = AudioAnalyzer()
    audio = np.zeros((1024, 2), dtype=np.float32)

    metrics = analyzer.analyze(audio, sr=48_000)

    assert EXPECTED_KEYS.issubset(metrics.keys())
    assert metrics["duration"] > 0.0
