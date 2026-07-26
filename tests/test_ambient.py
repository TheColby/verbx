from __future__ import annotations

import numpy as np

from verbx.core.ambient import apply_bloom, apply_ducking, apply_tilt_eq


def test_apply_ducking_respects_floor() -> None:
    wet = np.ones((1024, 1), dtype=np.float64)
    sidechain = np.ones((1024, 1), dtype=np.float64)

    out = apply_ducking(
        wet,
        sidechain,
        sr=48_000,
        attack_ms=0.1,
        release_ms=10.0,
        strength=1.0,
        floor=0.35,
    )

    assert float(np.min(out)) >= 0.349
    assert float(out[-1, 0]) <= 0.351


def test_apply_bloom_mix_override_changes_output_shape() -> None:
    audio = np.zeros((4096, 1), dtype=np.float64)
    audio[0, 0] = 1.0

    auto = apply_bloom(audio, sr=48_000, bloom_seconds=2.0)
    forced = apply_bloom(audio, sr=48_000, bloom_seconds=2.0, bloom_mix=1.0)

    assert not np.allclose(auto, forced)
    assert float(np.max(np.abs(forced))) < float(np.max(np.abs(auto)))


def test_apply_tilt_eq_pivot_changes_spectral_balance() -> None:
    sr = 48_000
    t = np.arange(sr, dtype=np.float64) / float(sr)
    audio = np.column_stack(
        [
            0.7 * np.sin(2.0 * np.pi * 120.0 * t) + 0.3 * np.sin(2.0 * np.pi * 6_000.0 * t)
        ]
    )

    low_pivot = apply_tilt_eq(
        audio,
        sr=sr,
        tilt_db=6.0,
        lowcut=None,
        highcut=None,
        pivot_hz=250.0,
    )
    high_pivot = apply_tilt_eq(
        audio,
        sr=sr,
        tilt_db=6.0,
        lowcut=None,
        highcut=None,
        pivot_hz=4_000.0,
    )

    assert not np.allclose(low_pivot, high_pivot)
    assert float(np.sqrt(np.mean(np.square(low_pivot - high_pivot)))) > 0.01


def test_apply_tilt_eq_filter_orders_change_response() -> None:
    audio = np.zeros((2048, 1), dtype=np.float64)
    audio[32, 0] = 1.0

    gentle = apply_tilt_eq(
        audio,
        sr=48_000,
        tilt_db=0.0,
        lowcut=120.0,
        highcut=8_000.0,
        lowcut_order=1,
        highcut_order=1,
    )
    steep = apply_tilt_eq(
        audio,
        sr=48_000,
        tilt_db=0.0,
        lowcut=120.0,
        highcut=8_000.0,
        lowcut_order=6,
        highcut_order=6,
    )

    assert np.isfinite(gentle).all()
    assert np.isfinite(steep).all()
    assert not np.allclose(gentle, steep)
