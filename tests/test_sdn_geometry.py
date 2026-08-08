from __future__ import annotations

import numpy as np

from verbx.core.room_geometry import RoomGeometry
from verbx.core.sdn_geometry import FACE_ORDER, derive_sdn_geometry


def test_sdn_geometry_has_six_face_nodes_and_analytic_delays() -> None:
    room = RoomGeometry(
        room_dims_m=(6.0, 8.0, 3.0),
        source_pos_m=(1.0, 2.0, 1.5),
        listener_pos_m=(4.0, 5.0, 1.5),
    )
    topology = derive_sdn_geometry(room)

    assert topology.face_order == FACE_ORDER
    assert topology.node_positions_m[0] == (0.0, 4.0, 1.5)
    assert topology.node_positions_m[1] == (6.0, 4.0, 1.5)
    assert topology.source_to_node_delay_ms[0] == np.hypot(1.0, 2.0) / 343.0 * 1000.0
    assert topology.node_to_listener_delay_ms[3] == np.hypot(1.0, 3.0) / 343.0 * 1000.0
    delays = np.asarray(topology.inter_node_delay_ms)
    assert np.allclose(delays, delays.T)
    assert np.allclose(np.diag(delays), 0.0)
    assert delays[0, 1] == 6.0 / 343.0 * 1000.0


def test_sdn_geometry_maps_face_materials_and_normalizes_coupling() -> None:
    topology = derive_sdn_geometry(
        RoomGeometry(
            room_dims_m=(6.0, 8.0, 3.0),
            wall_materials={"left": "stone", "right": "diffuser", "front": "custom"},
        )
    )
    absorption = np.asarray(topology.absorption_by_face)
    coupling = np.asarray(topology.inter_node_coupling)

    assert absorption.shape == (6, 6)
    assert np.allclose(absorption[2], 0.35)  # legacy unknown-material fallback
    assert topology.scattering_by_face[1] > topology.scattering_by_face[0]
    assert np.allclose(np.diag(coupling), 0.0)
    assert np.allclose(np.sum(coupling, axis=1), 1.0)
    assert coupling[0, 1] > coupling[0, 0]


def test_sdn_geometry_report_is_json_safe_and_keeps_contract_shape() -> None:
    report = derive_sdn_geometry(RoomGeometry(room_dims_m=(6.0, 8.0, 3.0))).to_report()
    assert report["face_order"] == list(FACE_ORDER)
    assert len(report["source_to_node_delay_ms"]) == 6
    assert len(report["inter_node_delay_ms"]) == 6
    assert all(len(row) == 6 for row in report["inter_node_coupling"])
