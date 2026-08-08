from __future__ import annotations

import numpy as np

from verbx.core.early_reflections import apply_image_source_early_reflections
from verbx.core.room_geometry import RoomGeometry, infer_room_geometry_from_rt60


def test_room_geometry_summary_contains_physical_fields() -> None:
    geometry = RoomGeometry(
        room_dims_m=(6.0, 8.0, 3.0),
        source_pos_m=(1.0, 2.0, 1.5),
        listener_pos_m=(4.0, 5.0, 1.5),
    )
    summary = geometry.summary()
    assert summary["volume_m3"] == 144.0
    assert summary["surface_area_m2"] == 180.0
    assert summary["direct_distance_m"] > 0.0
    assert summary["direct_path_pre_delay_ms"] > 0.0
    assert 0.0 <= summary["bolt_score"] <= 1.0


def test_infer_room_geometry_from_rt60_builds_nonzero_room() -> None:
    geometry = infer_room_geometry_from_rt60(
        rt60_s=1.4,
        mean_absorption=0.3,
        wall_material="hall",
    )
    assert geometry.volume_m3 > 0.0
    assert geometry.width_m > 0.0
    assert geometry.depth_m > 0.0
    assert geometry.height_m > 0.0
    assert geometry.mean_absorption == 0.3


def test_geometry_derives_ordered_fdn_delays_and_band_rt60() -> None:
    geometry = RoomGeometry(
        room_dims_m=(6.0, 8.0, 3.0),
        wall_materials={
            "left": "stone",
            "right": "stone",
            "front": "studio",
            "rear": "studio",
            "floor": "carpet",
            "ceiling": "ceiling-tile",
        },
    )
    derived = geometry.derive_fdn_parameters(line_count=12)
    assert len(derived.delay_ms) == 12
    assert all(left < right for left, right in zip(derived.delay_ms, derived.delay_ms[1:], strict=False))
    assert derived.pre_delay_ms == geometry.direct_path_pre_delay_ms
    assert derived.mean_free_path_m == geometry.mean_free_path_m
    assert derived.rt60_low_s > derived.rt60_high_s
    assert geometry.sabine_rt60_by_band()[1000] == derived.rt60_mid_s


def test_geometry_fdn_derivation_preserves_custom_material_compatibility() -> None:
    geometry = RoomGeometry(room_dims_m=(6.0, 8.0, 3.0), wall_materials={"left": "custom"})
    assert geometry.derive_fdn_parameters().rt60_mid_s > 0.0


def test_image_source_order_adds_deterministic_material_aware_reflections() -> None:
    sr = 16_000
    impulse = np.zeros((4096, 1), dtype=np.float64)
    impulse[0, 0] = 1.0
    common = {
        "sr": sr,
        "room_dims_m": (6.0, 8.0, 3.0),
        "source_pos_m": (1.0, 2.0, 1.5),
        "listener_pos_m": (4.0, 5.0, 1.5),
        "absorption": 0.3,
    }
    first_order = apply_image_source_early_reflections(
        impulse,
        reflection_order=1,
        wall_materials={"left": "stone"},
        **common,
    )
    third_order = apply_image_source_early_reflections(
        impulse,
        reflection_order=3,
        wall_materials={"left": "stone"},
        **common,
    )
    assert first_order.shape == impulse.shape
    assert np.all(np.isfinite(third_order))
    assert not np.allclose(first_order, third_order)
    assert float(np.sum(np.abs(third_order))) > float(np.sum(np.abs(first_order)))
