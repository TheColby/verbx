"""Room geometry helpers for physically grounded acoustic workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from verbx.analysis.room_size import estimate_volume, project_dimensions
from verbx.core.early_reflections import material_absorption
from verbx.ir.materials import OCTAVE_BANDS_HZ, get_material_profile

_SPEED_OF_SOUND_M_S = 343.0
_BOLT_RATIO_BOUNDS = {
    "depth_over_height": (1.1, 3.2),
    "width_over_height": (1.0, 2.5),
}
_DEFAULT_WALL_MATERIALS = {
    "left": "studio",
    "right": "studio",
    "front": "studio",
    "rear": "studio",
    "ceiling": "studio",
    "floor": "studio",
}


@dataclass(frozen=True, slots=True)
class FDNRoomParameters:
    """Physically grounded starting controls for the algorithmic FDN.

    These values are deliberately *suggestions*, rather than a second room
    simulator. Delay lengths use rectangular-room propagation paths with a
    small deterministic prime-ratio perturbation to avoid coincident FDN
    modes; the three RT60 values are Sabine estimates at 125, 1 kHz, and
    4 kHz.
    """

    delay_ms: tuple[float, ...]
    rt60_low_s: float
    rt60_mid_s: float
    rt60_high_s: float
    pre_delay_ms: float
    mean_free_path_m: float

    def to_report(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "delay_ms": list(self.delay_ms),
            "rt60_low_s": self.rt60_low_s,
            "rt60_mid_s": self.rt60_mid_s,
            "rt60_high_s": self.rt60_high_s,
            "pre_delay_ms": self.pre_delay_ms,
            "mean_free_path_m": self.mean_free_path_m,
        }


@dataclass(slots=True)
class RoomGeometry:
    """Reusable rectangular room geometry model.

    The current implementation keeps geometry intentionally simple: rectangular
    rooms with explicit source/listener positions and one material tag per
    surface. That is enough to support image-source, SDN, and
    geometry-to-parameter work later without forcing those engines to invent
    their own slightly-different room structs.
    """

    room_dims_m: tuple[float, float, float]
    source_pos_m: tuple[float, float, float] = (2.0, 2.0, 1.5)
    listener_pos_m: tuple[float, float, float] = (5.0, 3.5, 1.5)
    wall_materials: dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_WALL_MATERIALS)
    )
    mean_absorption: float | None = None

    def __post_init__(self) -> None:
        dims = tuple(float(value) for value in self.room_dims_m)
        src = tuple(float(value) for value in self.source_pos_m)
        lst = tuple(float(value) for value in self.listener_pos_m)
        if len(dims) != 3 or any(value <= 0.0 for value in dims):
            raise ValueError("room_dims_m must contain three strictly positive values.")
        if len(src) != 3 or len(lst) != 3:
            raise ValueError("source_pos_m and listener_pos_m must contain exactly three values.")
        for label, pos in (("source_pos_m", src), ("listener_pos_m", lst)):
            for axis, (value, limit) in enumerate(zip(pos, dims, strict=True), start=1):
                if value < 0.0 or value > limit:
                    raise ValueError(
                        f"{label} axis {axis} must fall inside the room bounds "
                        f"0..{limit:.3f} m."
                    )
        self.room_dims_m = dims
        self.source_pos_m = src
        self.listener_pos_m = lst
        merged = dict(_DEFAULT_WALL_MATERIALS)
        merged.update(
            {str(key): str(value) for key, value in dict(self.wall_materials).items()}
        )
        self.wall_materials = merged
        if self.mean_absorption is None:
            material_values = [
                material_absorption(material, 0.35)
                for material in self.wall_materials.values()
            ]
            self.mean_absorption = float(np.mean(material_values, dtype=np.float64))
        self.mean_absorption = float(np.clip(float(self.mean_absorption), 0.01, 0.99))

    @property
    def width_m(self) -> float:
        return float(self.room_dims_m[0])

    @property
    def depth_m(self) -> float:
        return float(self.room_dims_m[1])

    @property
    def height_m(self) -> float:
        return float(self.room_dims_m[2])

    @property
    def volume_m3(self) -> float:
        width, depth, height = self.room_dims_m
        return float(width * depth * height)

    @property
    def surface_area_m2(self) -> float:
        width, depth, height = self.room_dims_m
        return float((2.0 * width * depth) + (2.0 * width * height) + (2.0 * depth * height))

    @property
    def direct_distance_m(self) -> float:
        src = np.asarray(self.source_pos_m, dtype=np.float64)
        lst = np.asarray(self.listener_pos_m, dtype=np.float64)
        return float(np.linalg.norm(lst - src))

    @property
    def direct_path_pre_delay_ms(self) -> float:
        return float((self.direct_distance_m / _SPEED_OF_SOUND_M_S) * 1000.0)

    @property
    def wall_areas_m2(self) -> dict[str, float]:
        """Return rectangular surface areas keyed by the material face names."""
        return {
            "left": self.depth_m * self.height_m,
            "right": self.depth_m * self.height_m,
            "front": self.width_m * self.height_m,
            "rear": self.width_m * self.height_m,
            "floor": self.width_m * self.depth_m,
            "ceiling": self.width_m * self.depth_m,
        }

    @property
    def mean_free_path_m(self) -> float:
        """Diffuse-field mean free path, ``4V/S``, for the room."""
        return float((4.0 * self.volume_m3) / max(self.surface_area_m2, 1e-9))

    def sabine_rt60_by_band(self) -> dict[int, float]:
        """Estimate octave-band RT60 using per-face material absorption.

        Sabine remains a stable parameter derivation rule here, not a claim
        that a small, highly absorptive room is perfectly diffuse. A
        conservative bound keeps derived controls safe for the real-time FDN.
        """
        areas = self.wall_areas_m2
        estimates: dict[int, float] = {}
        for band_index, band_hz in enumerate(OCTAVE_BANDS_HZ):
            absorption_area = 0.0
            for wall, area in areas.items():
                material = self.wall_materials[wall]
                try:
                    coefficient = get_material_profile(material).absorption[band_index]
                except ValueError:
                    # RoomGeometry accepts legacy/custom material labels. Keep
                    # them usable by treating the existing broadband fallback
                    # as a flat absorption profile.
                    coefficient = material_absorption(material, 0.35)
                absorption_area += area * coefficient
            estimates[int(band_hz)] = float(
                np.clip((0.161 * self.volume_m3) / max(absorption_area, 1e-6), 0.05, 60.0)
            )
        return estimates

    def derive_fdn_parameters(self, *, line_count: int = 8) -> FDNRoomParameters:
        """Derive decorrelated FDN delays and band RT60 targets from geometry.

        The path pool covers axial, planar-diagonal, and body-diagonal room
        dimensions. Repeating it for larger FDNs with prime-ratio scaling
        gives deterministic, non-coincident delays without depending on a
        sample rate or rounding sample counts prematurely.
        """
        requested = max(1, int(line_count))
        width, depth, height = self.room_dims_m
        path_lengths = np.asarray(
            (
                height,
                self.mean_free_path_m,
                width,
                depth,
                float(np.hypot(width, height)),
                float(np.hypot(depth, height)),
                float(np.hypot(width, depth)),
                float(np.linalg.norm((width, depth, height))),
            ),
            dtype=np.float64,
        )
        primes = np.asarray((2, 3, 5, 7, 11, 13, 17, 19), dtype=np.float64)
        values: list[float] = []
        for index in range(requested):
            cycle, slot = divmod(index, len(path_lengths))
            scale = (1.0 + (0.007 * primes[slot])) * (1.0 + (0.11 * cycle))
            values.append(float((path_lengths[slot] * scale / _SPEED_OF_SOUND_M_S) * 1000.0))
        delays = np.sort(np.clip(np.asarray(values, dtype=np.float64), 3.0, 120.0))
        for index in range(1, len(delays)):
            delays[index] = max(delays[index], delays[index - 1] + 0.11)

        rt60 = self.sabine_rt60_by_band()
        return FDNRoomParameters(
            delay_ms=tuple(float(value) for value in delays),
            rt60_low_s=rt60[125],
            rt60_mid_s=rt60[1000],
            rt60_high_s=rt60[4000],
            pre_delay_ms=self.direct_path_pre_delay_ms,
            mean_free_path_m=self.mean_free_path_m,
        )

    @property
    def aspect_ratios(self) -> dict[str, float]:
        return {
            "depth_over_width": float(self.depth_m / self.width_m),
            "height_over_width": float(self.height_m / self.width_m),
            "depth_over_height": float(self.depth_m / self.height_m),
            "width_over_height": float(self.width_m / self.height_m),
        }

    def bolt_score(self) -> float:
        """Return a simple 0..1 heuristic for small-room proportion sanity."""
        ratios = self.aspect_ratios
        score = 0.0
        depth_low, depth_high = _BOLT_RATIO_BOUNDS["depth_over_height"]
        width_low, width_high = _BOLT_RATIO_BOUNDS["width_over_height"]
        if depth_low <= ratios["depth_over_height"] <= depth_high:
            score += 0.5
        if width_low <= ratios["width_over_height"] <= width_high:
            score += 0.5
        return float(np.clip(score, 0.0, 1.0))

    def warnings(self) -> list[str]:
        """Return heuristic warnings about pathological rectangular proportions."""
        warnings: list[str] = []
        ratios = self.aspect_ratios
        if self.height_m < 2.1:
            warnings.append(
                "Low ceiling height may exaggerate floor/ceiling flutter and "
                "modal crowding."
            )
        if ratios["depth_over_height"] < _BOLT_RATIO_BOUNDS["depth_over_height"][0]:
            warnings.append(
                "Depth/height ratio is unusually small for a balanced "
                "rectangular room."
            )
        if ratios["depth_over_height"] > _BOLT_RATIO_BOUNDS["depth_over_height"][1]:
            warnings.append(
                "Depth/height ratio is unusually large; expect stretched axial "
                "spacing."
            )
        if ratios["width_over_height"] < _BOLT_RATIO_BOUNDS["width_over_height"][0]:
            warnings.append(
                "Width/height ratio is unusually small; the room may feel "
                "tunnel-like."
            )
        if ratios["width_over_height"] > _BOLT_RATIO_BOUNDS["width_over_height"][1]:
            warnings.append(
                "Width/height ratio is unusually large; lateral mode spacing "
                "may bunch up."
            )
        if self.direct_distance_m > max(self.room_dims_m):
            warnings.append(
                "Source-listener spacing is longer than the largest room "
                "dimension."
            )
        return warnings

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary payload."""
        ratios = self.aspect_ratios
        mean_absorption = self.mean_absorption
        assert mean_absorption is not None
        return {
            "room_dims_m": list(self.room_dims_m),
            "source_pos_m": list(self.source_pos_m),
            "listener_pos_m": list(self.listener_pos_m),
            "wall_materials": dict(self.wall_materials),
            "mean_absorption": float(mean_absorption),
            "volume_m3": self.volume_m3,
            "surface_area_m2": self.surface_area_m2,
            "direct_distance_m": self.direct_distance_m,
            "direct_path_pre_delay_ms": self.direct_path_pre_delay_ms,
            "mean_free_path_m": self.mean_free_path_m,
            "aspect_ratios": ratios,
            "bolt_score": self.bolt_score(),
            "warnings": self.warnings(),
        }


def infer_room_geometry_from_rt60(
    *,
    rt60_s: float,
    mean_absorption: float,
    source_pos_m: tuple[float, float, float] = (2.0, 2.0, 1.5),
    listener_pos_m: tuple[float, float, float] = (5.0, 3.5, 1.5),
    wall_material: str = "studio",
) -> RoomGeometry:
    """Infer a rectangular room geometry from RT60 plus absorption."""
    volume = estimate_volume(float(rt60_s), float(mean_absorption))
    dims = project_dimensions(float(volume["primary_m3"]))
    geometry = RoomGeometry(
        room_dims_m=(
            float(dims["width_m"]),
            float(dims["depth_m"]),
            float(dims["height_m"]),
        ),
        source_pos_m=source_pos_m,
        listener_pos_m=listener_pos_m,
        wall_materials={key: wall_material for key in _DEFAULT_WALL_MATERIALS},
        mean_absorption=float(mean_absorption),
    )
    return geometry
