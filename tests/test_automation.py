from __future__ import annotations

import numpy as np
import pytest

from verbx.core.automation import (
    collect_automation_targets,
    load_automation_bundle,
    parse_automation_clamp_overrides,
    parse_automation_point_specs,
)


def test_parse_automation_points_groups_sorts_and_normalizes_targets() -> None:
    lanes = parse_automation_point_specs(
        [
            "rt60-high:1.0:3:smooth",
            "fdn_rt60_high:0.0:1",
            "wet:0.5:0.75:hold",
        ]
    )

    assert lanes == [
        {
            "target": "fdn-rt60-high",
            "type": "breakpoints",
            "interp": "smooth",
            "points": [{"time": 0.0, "value": 1.0}, {"time": 1.0, "value": 3.0}],
        },
        {
            "target": "wet",
            "type": "breakpoints",
            "interp": "hold",
            "points": [{"time": 0.5, "value": 0.75}],
        },
    ]


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("wet:0:1:unknown", "Unsupported automation interpolation"),
        ("wet:-1:0.5", "time must be finite and >= 0"),
        ("not-a-target:0:1", "Unsupported automation point target"),
    ],
)
def test_parse_automation_points_rejects_invalid_contracts(spec: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_automation_point_specs([spec])


def test_parse_automation_clamps_normalizes_and_validates_bounds() -> None:
    assert parse_automation_clamp_overrides(["rt60-high:0.1:120", "wet:0:1"]) == {
        "fdn-rt60-high": (0.1, 120.0),
        "wet": (0.0, 1.0),
    }
    with pytest.raises(ValueError, match="min < max"):
        parse_automation_clamp_overrides(["wet:1:0"])


def test_inline_automation_bundle_is_deterministic_and_sample_aligned() -> None:
    point_specs = ["wet:0:0", "wet:1:1"]
    first = load_automation_bundle(
        path=None,
        point_specs=point_specs,
        sr=100,
        num_samples=101,
        mode="sample",
        smoothing_ms=0.0,
    )
    second = load_automation_bundle(
        path=None,
        point_specs=point_specs,
        sr=100,
        num_samples=101,
        mode="sample",
        smoothing_ms=0.0,
    )

    assert first.signature == second.signature
    assert first.mode == "sample"
    assert first.control_step == 1
    assert first.lanes_per_target == {"wet": 1}
    np.testing.assert_allclose(first.curves["wet"], np.linspace(0.0, 1.0, 101))
    assert collect_automation_targets(path=None, point_specs=point_specs) == {"wet"}
