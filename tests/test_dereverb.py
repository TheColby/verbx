from __future__ import annotations

import numpy as np

from verbx.core.dereverb import DereverbConfig, apply_dereverb


def _synthetic_reverberant_impulse(*, sr: int, length_s: float) -> np.ndarray:
    n = int(sr * length_s)
    dry = np.zeros((n,), dtype=np.float64)
    dry[32] = 1.0
    ir = np.exp(-np.arange(int(0.6 * sr), dtype=np.float64) / (0.18 * sr))
    wet = np.convolve(dry, ir, mode="full")[:n]
    return wet[:, np.newaxis]


def test_apply_dereverb_reduces_late_tail_energy() -> None:
    sr = 16_000
    wet = _synthetic_reverberant_impulse(sr=sr, length_s=1.0)
    out = apply_dereverb(
        wet,
        sr,
        DereverbConfig(
            mode="wiener",
            strength=1.0,
            floor=0.05,
            window_ms=32.0,
            hop_ms=8.0,
            tail_ms=200.0,
            mix=1.0,
        ),
    )
    assert out.shape == wet.shape
    assert np.isfinite(out).all()
    late_start = int(0.25 * sr)
    in_late = float(np.mean(np.square(wet[late_start:, 0])))
    out_late = float(np.mean(np.square(out[late_start:, 0])))
    assert out_late < in_late


def test_apply_dereverb_spectral_sub_mode_is_stable() -> None:
    sr = 48_000
    x = np.zeros((2048, 2), dtype=np.float64)
    x[32:128, 0] = 0.5
    x[64:192, 1] = 0.3
    out = apply_dereverb(
        x,
        sr,
        DereverbConfig(
            mode="spectral_sub",
            strength=0.7,
            floor=0.1,
            window_ms=20.0,
            hop_ms=5.0,
            tail_ms=120.0,
            pre_emphasis=0.25,
            mix=0.9,
        ),
    )
    assert out.shape == x.shape
    assert np.isfinite(out).all()
