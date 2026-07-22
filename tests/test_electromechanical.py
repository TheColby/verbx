"""Regression coverage for the bounded electro-mechanical modal solvers."""

from __future__ import annotations

import numpy as np

from verbx.core.algo_reverb import AlgoReverbConfig, AlgoReverbEngine


def test_modal_fe_spring_tank_is_finite_and_reports_modes() -> None:
    impulse = np.zeros((4096, 1), dtype=np.float64)
    impulse[0, 0] = 1.0
    engine = AlgoReverbEngine(
        AlgoReverbConfig(
            algo_model="spring",
            electromechanical_solver="modal-fe",
            spring_count=2,
            spring_specs=(
                "length_m=0.34,mass_g=21,compliance_mm_n=0.45,tension_n=7,damping=0.5",
                "length_m=0.62,mass_g=47,compliance_mm_n=0.90,tension_n=3,damping=0.7",
            ),
            spring_fe_nodes=12,
            spring_fe_modes=16,
            wet=1.0,
            dry=0.0,
        )
    )
    rendered = engine.process(impulse, 48_000)
    report = engine.electromechanical_report()

    assert np.all(np.isfinite(rendered))
    assert float(np.max(np.abs(rendered))) > 1e-5
    assert report is not None
    assert report["solver"] == "modal-fe"
    assert int(report["active_modes"]) > 0


def test_modal_fe_plate_pickup_changes_response() -> None:
    impulse = np.zeros((4096, 1), dtype=np.float64)
    impulse[0, 0] = 1.0
    left = AlgoReverbEngine(
        AlgoReverbConfig(
            algo_model="plate",
            electromechanical_solver="modal-fe",
            plate_fe_nx=8,
            plate_fe_ny=6,
            plate_fe_modes=20,
            plate_pickup_x=0.15,
            plate_pickup_y=0.20,
            wet=1.0,
            dry=0.0,
        )
    ).process(impulse, 48_000)
    right = AlgoReverbEngine(
        AlgoReverbConfig(
            algo_model="plate",
            electromechanical_solver="modal-fe",
            plate_fe_nx=8,
            plate_fe_ny=6,
            plate_fe_modes=20,
            plate_pickup_x=0.82,
            plate_pickup_y=0.72,
            wet=1.0,
            dry=0.0,
        )
    ).process(impulse, 48_000)

    assert np.all(np.isfinite(left)) and np.all(np.isfinite(right))
    assert not np.allclose(left, right)
