"""Room geometry helpers for physically grounded acoustic workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from verbx.analysis.room_size import estimate_volume, project_dimensions
from verbx.core.early_reflections import material_absorption

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
