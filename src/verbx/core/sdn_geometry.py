"""Geometry-derived building blocks for an eventual Scattering Delay Network.

This module deliberately stops before DSP: it specifies the rectangular-room
topology and its physically meaningful propagation/absorption parameters so a
future realtime SDN can use one stable, testable contract.  It is *not* an
audio engine and does not make ``--engine sdn`` available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from verbx.core.room_geometry import RoomGeometry
from verbx.ir.materials import OCTAVE_BANDS_HZ, get_material_profile, material_absorption

SPEED_OF_SOUND_M_S = 343.0
FACE_ORDER: tuple[str, ...] = ("left", "right", "front", "rear", "floor", "ceiling")


@dataclass(frozen=True, slots=True)
class SDNGeometry:
    """The fixed six-node rectangular SDN geometry contract.

    ``inter_node_delay_ms`` and ``inter_node_coupling`` use ``FACE_ORDER`` on
    both axes.  Diagonal delay/coupling entries are zero: a node never sends a
    propagation path to itself.  Off-diagonal coupling rows are normalized, so
    an implementation may apply its own loss and scattering filters without
    inheriting accidental gain from this topology helper.
    """

    face_order: tuple[str, ...]
    node_positions_m: tuple[tuple[float, float, float], ...]
    source_to_node_delay_ms: tuple[float, ...]
    node_to_listener_delay_ms: tuple[float, ...]
    inter_node_delay_ms: tuple[tuple[float, ...], ...]
    absorption_by_face: tuple[tuple[float, ...], ...]
    scattering_by_face: tuple[float, ...]
    inter_node_coupling: tuple[tuple[float, ...], ...]

    def to_report(self) -> dict[str, Any]:
        """Return a JSON-safe representation for future render reports."""
        return {
            "face_order": list(self.face_order),
            "node_positions_m": [list(position) for position in self.node_positions_m],
            "source_to_node_delay_ms": list(self.source_to_node_delay_ms),
            "node_to_listener_delay_ms": list(self.node_to_listener_delay_ms),
            "inter_node_delay_ms": [list(row) for row in self.inter_node_delay_ms],
            "octave_bands_hz": list(OCTAVE_BANDS_HZ),
            "absorption_by_face": [list(row) for row in self.absorption_by_face],
            "scattering_by_face": list(self.scattering_by_face),
            "inter_node_coupling": [list(row) for row in self.inter_node_coupling],
        }


def derive_sdn_geometry(room: RoomGeometry) -> SDNGeometry:
    """Derive a bounded six-wall SDN topology from a rectangular room.

    Wall nodes are face centroids.  Each directed connection's delay is the
    Euclidean path between centroids; source/listener legs are calculated the
    same way.  Material profiles provide octave-band absorption and a scalar
    scattering weight.  Unknown legacy material labels retain RoomGeometry's
    existing broadband fallback and use zero scattering.
    """
    positions = _face_centres(room)
    source = np.asarray(room.source_pos_m, dtype=np.float64)
    listener = np.asarray(room.listener_pos_m, dtype=np.float64)
    nodes = np.asarray(positions, dtype=np.float64)
    source_delays = _distance_to_delay_ms(nodes - source)
    listener_delays = _distance_to_delay_ms(nodes - listener)
    pairwise = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]
    inter_node_delays = _distance_to_delay_ms(pairwise)
    np.fill_diagonal(inter_node_delays, 0.0)

    absorption, scattering = _material_parameters(room)
    # Each sender distributes to the other five receivers. The receiver's
    # scattering enters the weight, providing geometry/material-derived
    # coupling while row normalization guarantees no topology gain.
    receiver_weight = np.clip(np.asarray(scattering, dtype=np.float64), 0.0, 1.0)
    receiver_weight += 0.05  # retain a connected graph for perfectly smooth walls
    coupling = np.broadcast_to(receiver_weight, (len(FACE_ORDER), len(FACE_ORDER))).copy()
    np.fill_diagonal(coupling, 0.0)
    coupling /= np.sum(coupling, axis=1, keepdims=True)

    return SDNGeometry(
        face_order=FACE_ORDER,
        node_positions_m=positions,
        source_to_node_delay_ms=tuple(float(value) for value in source_delays),
        node_to_listener_delay_ms=tuple(float(value) for value in listener_delays),
        inter_node_delay_ms=tuple(tuple(float(value) for value in row) for row in inter_node_delays),
        absorption_by_face=absorption,
        scattering_by_face=scattering,
        inter_node_coupling=tuple(tuple(float(value) for value in row) for row in coupling),
    )


def _face_centres(room: RoomGeometry) -> tuple[tuple[float, float, float], ...]:
    width, depth, height = room.room_dims_m
    return (
        (0.0, depth / 2.0, height / 2.0),
        (width, depth / 2.0, height / 2.0),
        (width / 2.0, 0.0, height / 2.0),
        (width / 2.0, depth, height / 2.0),
        (width / 2.0, depth / 2.0, 0.0),
        (width / 2.0, depth / 2.0, height),
    )


def _distance_to_delay_ms(vectors: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return (np.linalg.norm(vectors, axis=-1) / SPEED_OF_SOUND_M_S) * 1000.0


def _material_parameters(
    room: RoomGeometry,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    absorption: list[tuple[float, ...]] = []
    scattering: list[float] = []
    for face in FACE_ORDER:
        material = room.wall_materials[face]
        try:
            profile = get_material_profile(material)
        except ValueError:
            fallback = material_absorption(material, 0.35)
            absorption.append(tuple(fallback for _ in OCTAVE_BANDS_HZ))
            scattering.append(0.0)
        else:
            absorption.append(tuple(float(value) for value in profile.absorption))
            scattering.append(float(profile.scattering))
    return tuple(absorption), tuple(scattering)
