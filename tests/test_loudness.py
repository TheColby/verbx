from __future__ import annotations

import numpy as np

from verbx.core.loudness import (
    apply_output_targets,
    integrated_lufs,
    sample_peak_dbfs,
    true_peak_dbfs,
)


def test_lufs_normalization_moves_toward_target() -> None:
    rng = np.random.default_rng(42)
    audio = (0.03 * rng.standard_normal((48_000 * 2, 2))).astype(np.float32)
    target_lufs = -20.0

    before = integrated_lufs(audio, sr=48_000)
    normalized = apply_output_targets(
        audio,
        sr=48_000,
        target_lufs=target_lufs,
        target_peak_dbfs=None,
        limiter=False,
        use_true_peak=True,
    )
    after = integrated_lufs(normalized, sr=48_000)

    assert abs(after - target_lufs) < abs(before - target_lufs)


def test_peak_ceiling_enforced() -> None:
    t = np.linspace(0.0, 1.0, 48_000, endpoint=False, dtype=np.float32)
    audio = np.stack(
        [1.8 * np.sin(2.0 * np.pi * 440.0 * t), 1.8 * np.sin(2.0 * np.pi * 880.0 * t)], axis=1
    ).astype(np.float32)

    target_peak = -3.0
    out = apply_output_targets(
        audio,
        sr=48_000,
        target_lufs=None,
        target_peak_dbfs=target_peak,
        limiter=True,
        use_true_peak=True,
    )

    assert sample_peak_dbfs(out) <= target_peak + 0.25
    assert true_peak_dbfs(out, sr=48_000, oversample=4) <= target_peak + 0.35
