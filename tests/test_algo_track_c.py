from __future__ import annotations

import numpy as np

from verbx.core.algo_reverb import AlgoReverbConfig, AlgoReverbEngine


def test_tonal_correction_scales_default_to_unity_when_disabled() -> None:
    low = np.asarray([0.92, 0.91, 0.90], dtype=np.float64)
    mid = np.asarray([0.84, 0.83, 0.82], dtype=np.float64)
    high = np.asarray([0.71, 0.70, 0.69], dtype=np.float64)
    scales = AlgoReverbEngine.resolve_tonal_correction_scales(
        feedback_gain_low=low,
        feedback_gain_mid=mid,
        feedback_gain_high=high,
        strength=0.0,
    )
    assert scales == (np.float64(1.0), np.float64(1.0), np.float64(1.0))


def test_tonal_correction_scales_rebalance_low_and_high_decay_color() -> None:
    low = np.asarray([0.95, 0.94, 0.93], dtype=np.float64)
    mid = np.asarray([0.85, 0.84, 0.83], dtype=np.float64)
    high = np.asarray([0.66, 0.65, 0.64], dtype=np.float64)
    low_scale, mid_scale, high_scale = AlgoReverbEngine.resolve_tonal_correction_scales(
        feedback_gain_low=low,
        feedback_gain_mid=mid,
        feedback_gain_high=high,
        strength=1.0,
    )
    assert float(low_scale) < 1.0
    assert float(high_scale) > 1.0
    assert 0.5 <= float(low_scale) <= 2.0
    assert 0.5 <= float(mid_scale) <= 2.0
    assert 0.5 <= float(high_scale) <= 2.0
    rms = float(
        np.sqrt(
            (
                (low_scale * low_scale)
                + (mid_scale * mid_scale)
                + (high_scale * high_scale)
            )
            / 3.0
        )
    )
    assert abs(rms - 1.0) < 1e-6


def test_multiband_rt60_resolver_accepts_runtime_track_c_controls() -> None:
    cfg = AlgoReverbConfig(
        rt60=3.0,
        fdn_rt60_low=3.0,
        fdn_rt60_mid=3.0,
        fdn_rt60_high=3.0,
        fdn_rt60_tilt=0.0,
        clarity_macro=0.0,
        warmth_macro=0.0,
    )
    low, mid, high = AlgoReverbEngine.resolve_multiband_rt60(
        cfg,
        fdn_rt60_tilt=0.45,
        clarity_macro=-0.2,
        warmth_macro=0.6,
    )
    assert float(low) > float(mid)
    assert float(mid) > float(high)
