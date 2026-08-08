from __future__ import annotations

import numpy as np

from verbx.core.algo_reverb import AlgoReverbConfig, AlgoReverbEngine


def test_spring_and_plate_models_render_distinct_finite_tails() -> None:
    sr = 16_000
    impulse = np.zeros((sr // 4, 1), dtype=np.float64)
    impulse[0, 0] = 1.0
    renders: dict[str, np.ndarray] = {}

    for model in ("spring", "plate"):
        renders[model] = AlgoReverbEngine(
            AlgoReverbConfig(
                algo_model=model,
                rt60=0.35,
                wet=1.0,
                dry=0.0,
                block_size=512,
            )
        ).process(impulse, sr)
        assert renders[model].shape == impulse.shape
        assert np.all(np.isfinite(renders[model]))
        assert float(np.max(np.abs(renders[model][64:, 0]))) > 1e-6

    assert not np.allclose(renders["spring"], renders["plate"])


def test_spring_specs_and_plate_material_controls_change_model_response() -> None:
    sr = 16_000
    impulse = np.zeros((2048, 1), dtype=np.float64)
    impulse[0, 0] = 1.0
    base_spring = AlgoReverbEngine(
        AlgoReverbConfig(algo_model="spring", rt60=0.3, wet=1.0, dry=0.0)
    ).process(impulse, sr)
    dual_spring = AlgoReverbEngine(
        AlgoReverbConfig(
            algo_model="spring",
            spring_count=2,
            spring_specs=(
                "length_m=0.35,mass_g=22,diameter_mm=0.9,compliance_mm_n=0.5,tension_n=7,damping=0.62",
                "length_m=0.62,mass_g=48,diameter_mm=1.6,compliance_mm_n=1.1,tension_n=3,damping=0.78",
            ),
            rt60=0.3,
            wet=1.0,
            dry=0.0,
        )
    ).process(impulse, sr)
    default_plate = AlgoReverbEngine(
        AlgoReverbConfig(algo_model="plate", rt60=0.3, wet=1.0, dry=0.0)
    ).process(impulse, sr)
    large_plate = AlgoReverbEngine(
        AlgoReverbConfig(
            algo_model="plate",
            plate_width_m=3.0,
            plate_height_m=2.0,
            plate_thickness_mm=0.9,
            plate_tension_n=800.0,
            plate_pickup_x=0.15,
            plate_pickup_y=0.82,
            rt60=0.3,
            wet=1.0,
            dry=0.0,
        )
    ).process(impulse, sr)

    for rendered in (base_spring, dual_spring, default_plate, large_plate):
        assert np.all(np.isfinite(rendered))
        assert float(np.max(np.abs(rendered))) > 1e-6
    assert not np.allclose(base_spring, dual_spring)
    assert not np.allclose(default_plate, large_plate)


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


def test_multiband_rt60_resolver_supports_long_decay_envelopes() -> None:
    cfg = AlgoReverbConfig(
        rt60=1_200.0,
        fdn_rt60_low=1_200.0,
        fdn_rt60_mid=1_200.0,
        fdn_rt60_high=1_200.0,
        fdn_rt60_tilt=0.0,
        clarity_macro=0.0,
        warmth_macro=0.0,
    )
    low, mid, high = AlgoReverbEngine.resolve_multiband_rt60(
        cfg,
        fdn_rt60_tilt=0.6,
        clarity_macro=0.0,
        warmth_macro=0.0,
    )
    assert float(low) > 300.0
    assert float(high) > 300.0
    assert float(mid) > 300.0
