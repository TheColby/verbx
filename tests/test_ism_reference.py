from __future__ import annotations

import json
from pathlib import Path

import pytest

from verbx.core.early_reflections import SPEED_OF_SOUND_M_S, enumerate_image_source_paths


def test_rectangular_ism_matches_analytic_reference_corpus() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "ism_rectangular_reference.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    sr = int(fixture["sample_rate"])
    paths = enumerate_image_source_paths(
        sr=sr,
        room_dims_m=tuple(fixture["room_dims_m"]),
        source_pos_m=tuple(fixture["source_pos_m"]),
        listener_pos_m=tuple(fixture["listener_pos_m"]),
        absorption=float(fixture["absorption"]),
        reflection_order=int(fixture["reflection_order"]),
    )

    assert fixture["speed_of_sound_m_s"] == SPEED_OF_SOUND_M_S
    assert len(paths) == len(fixture["paths"])
    actual = {path.walls: path for path in paths}
    for expected in fixture["paths"]:
        walls = tuple(expected["walls"])
        path = actual[walls]
        expected_distance = float(expected["distance_m"])
        assert path.distance_m == pytest.approx(expected_distance, abs=1e-9)
        assert path.delay_samples == round(expected_distance / SPEED_OF_SOUND_M_S * sr)


def test_ism_reference_paths_retain_material_gain_before_tap_normalization() -> None:
    common = {
        "sr": 48_000,
        "room_dims_m": (6.0, 8.0, 3.0),
        "source_pos_m": (1.0, 2.0, 1.5),
        "listener_pos_m": (4.0, 5.0, 1.5),
        "absorption": 0.3,
        "reflection_order": 1,
    }
    stone = enumerate_image_source_paths(
        wall_materials={"left": "stone"},
        **common,
    )
    dead = enumerate_image_source_paths(
        wall_materials={"left": "dead"},
        **common,
    )
    stone_left = next(path for path in stone if path.walls == ("left",))
    dead_left = next(path for path in dead if path.walls == ("left",))
    assert stone_left.distance_m == dead_left.distance_m
    assert stone_left.gain > dead_left.gain
