from __future__ import annotations

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
