"""Experimental geometry-to-IR tracing helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from verbx.core.room_geometry import RoomGeometry
from verbx.ir.materials import get_material_profile
from verbx.ir.metrics import analyze_ir

AudioArray = npt.NDArray[np.float64]
_SPEED_OF_SOUND_M_S = 343.0
_SURFACE_NAMES = ("floor", "ceiling", "left", "right", "front", "rear")


@dataclass(slots=True)
class DXFTraceGeometry:
    """Constrained room-like geometry extracted from a simple DXF file."""

    vertices_xy: tuple[tuple[float, float], ...]
    height_m: float
    units: str
    warnings: tuple[str, ...] = ()
    surface_layers: dict[str, tuple[str, ...]] | None = None

    @property
    def width_m(self) -> float:
        xs = [point[0] for point in self.vertices_xy]
        return float(max(xs) - min(xs))

    @property
    def depth_m(self) -> float:
        ys = [point[1] for point in self.vertices_xy]
        return float(max(ys) - min(ys))

    @property
    def room_dims_m(self) -> tuple[float, float, float]:
        return (self.width_m, self.depth_m, float(self.height_m))

    def normalized_vertices_xy(self) -> list[list[float]]:
        min_x = min(point[0] for point in self.vertices_xy)
        min_y = min(point[1] for point in self.vertices_xy)
        return [[float(x - min_x), float(y - min_y)] for x, y in self.vertices_xy]

    def known_layers(self) -> tuple[str, ...]:
        """Return the deterministic set of CAD layers present in the outline."""
        layers = self.surface_layers or {}
        return tuple(sorted({layer for values in layers.values() for layer in values}))


def parse_trace_vector(value: str, *, label: str) -> tuple[float, float, float]:
    """Parse a CLI vector in x,y,z meters."""
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{label} must use x,y,z meters, for example 2,3,1.5.")
    try:
        parsed = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{label} must contain numeric x,y,z values.") from exc
    if any(not np.isfinite(value) for value in parsed):
        raise ValueError(f"{label} must contain finite x,y,z values.")
    return parsed  # type: ignore[return-value]


def parse_dxf_room_outline(path: Path, *, height_m: float = 3.0) -> DXFTraceGeometry:
    """Parse a constrained ASCII DXF room outline.

    Supported MVP entities:
    - ``LINE`` in the ENTITIES section
    - ``LWPOLYLINE`` vertex code pairs

    The first implementation intentionally derives an axis-aligned room bounding
    box from clean room-like geometry. Unsupported CAD cleanup remains deferred.
    """
    if float(height_m) <= 0.0:
        raise ValueError("--height must be > 0.")
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines()]
    pairs = list(zip(lines[0::2], lines[1::2], strict=False))
    units = _resolve_units(pairs)
    scale = _unit_scale_to_meters(units)
    entities: list[tuple[list[tuple[float, float]], str, bool]] = []
    warnings: list[str] = []

    index = 0
    while index < len(pairs):
        code, value = pairs[index]
        entity = value.upper()
        if code == "0" and entity == "LINE":
            parsed, layer, index = _parse_line_entity(pairs, index + 1, scale)
            entities.append((parsed, layer, False))
            continue
        if code == "0" and entity == "LWPOLYLINE":
            parsed, layer, is_closed, index = _parse_lwpolyline_entity(
                pairs, index + 1, scale
            )
            entities.append((parsed, layer, is_closed))
            continue
        index += 1

    points = [point for entity_points, _, _ in entities for point in entity_points]

    if len(points) < 2:
        raise ValueError(
            "DXF trace MVP requires LINE or LWPOLYLINE room-boundary entities."
        )

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    depth = max_y - min_y
    if width <= 0.0 or depth <= 0.0:
        raise ValueError("DXF trace geometry must span positive width and depth.")
    if len({(round(x, 6), round(y, 6)) for x, y in points}) < 4:
        warnings.append("Geometry has fewer than four unique XY points; using bounding box.")

    vertices = ((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y))
    return DXFTraceGeometry(
        vertices_xy=vertices,
        height_m=float(height_m),
        units=units,
        warnings=tuple(warnings),
        surface_layers=_surface_layers_from_entities(
            entities, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y
        ),
    )


def generate_trace_ir(
    *,
    geometry: DXFTraceGeometry,
    source_pos_m: tuple[float, float, float],
    listener_pos_m: tuple[float, float, float],
    material: str,
    layer_materials: dict[str, str] | None = None,
    rays: int,
    length_s: float,
    sr: int,
    seed: int = 0,
) -> tuple[AudioArray, dict[str, Any]]:
    """Generate a deterministic experimental trace IR and report payload."""
    if int(sr) < 8_000:
        raise ValueError("--target-sr must be >= 8000.")
    if float(length_s) <= 0.01:
        raise ValueError("--length must be > 0.01 seconds.")
    if int(rays) < 1:
        raise ValueError("--rays must be >= 1.")

    material_profile = get_material_profile(material)
    overrides = _validate_layer_materials(geometry, layer_materials or {})
    surface_materials = _resolve_surface_materials(
        geometry, default_material=material_profile.name, layer_materials=overrides
    )
    surface_profiles = {
        surface: get_material_profile(name) for surface, name in surface_materials.items()
    }
    room = RoomGeometry(
        room_dims_m=geometry.room_dims_m,
        source_pos_m=source_pos_m,
        listener_pos_m=listener_pos_m,
        wall_materials=surface_materials,
    )
    absorption = float(room.mean_absorption if room.mean_absorption is not None else 0.35)
    frames = max(1, round(float(length_s) * int(sr)))
    audio = np.zeros((frames, 2), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    reflection_rows: list[dict[str, Any]] = []

    direct_delay = room.direct_distance_m / _SPEED_OF_SOUND_M_S
    direct_amp = 1.0 / max(1.0, room.direct_distance_m)
    _add_stereo_impulse(audio, sr, direct_delay, direct_amp, pan=0.0)
    reflection_rows.append(
        {
            "kind": "direct",
            "surface": "direct",
            "time_s": direct_delay,
            "distance_m": room.direct_distance_m,
            "amplitude": direct_amp,
        }
    )

    for surface, image in _first_order_images(room).items():
        distance = float(np.linalg.norm(np.asarray(image) - np.asarray(listener_pos_m)))
        delay = distance / _SPEED_OF_SOUND_M_S
        surface_absorption = surface_profiles[surface].broadband_absorption()
        amp = (1.0 - surface_absorption) / max(1.0, distance)
        pan = _surface_pan(surface)
        _add_stereo_impulse(audio, sr, delay, amp, pan=pan)
        reflection_rows.append(
            {
                "kind": "first_order",
                "surface": surface,
                "time_s": delay,
                "distance_m": distance,
                "amplitude": amp,
            }
        )

    _add_late_tail(
        audio=audio,
        sr=sr,
        start_s=min(float(length_s) * 0.8, 0.08),
        rt60_s=_estimate_rt60(room),
        absorption=absorption,
        rays=int(rays),
        rng=rng,
    )
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0:
        audio *= float(10.0 ** (-1.0 / 20.0)) / peak

    report: dict[str, Any] = {
        "schema": "trace-report-v1",
        "experimental": True,
        "mode": "dxf-bounding-box-first-order-stochastic-tail",
        "geometry": {
            "source": "dxf",
            "units": geometry.units,
            "vertices_xy_m": geometry.normalized_vertices_xy(),
            "height_m": geometry.height_m,
            "room": room.summary(),
            "warnings": list(geometry.warnings) + room.warnings(),
        },
        "material": {
            "default": material_profile.name,
            "mean_absorption": absorption,
            "profile": material_profile.to_report(),
            "layer_material_overrides": dict(sorted(overrides.items())),
            "surface_layers": {
                surface: list((geometry.surface_layers or {}).get(surface, ()))
                for surface in _SURFACE_NAMES
            },
            "surface_profiles": {
                surface: surface_profiles[surface].to_report() for surface in _SURFACE_NAMES
            },
        },
        "trace": {
            "rays": int(rays),
            "seed": int(seed),
            "source_pos_m": list(source_pos_m),
            "listener_pos_m": list(listener_pos_m),
            "length_s": float(length_s),
            "target_sr": int(sr),
            "estimated_rt60_s": _estimate_rt60(room),
            "reflection_count": len(reflection_rows),
            "reflections": reflection_rows,
        },
        "metrics": analyze_ir(audio, int(sr)),
    }
    return audio, report


def write_trace_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _resolve_units(pairs: list[tuple[str, str]]) -> str:
    for index, (code, value) in enumerate(pairs):
        if code == "9" and value.upper() == "$INSUNITS" and index + 1 < len(pairs):
            raw = pairs[index + 1][1]
            return {
                "0": "unitless",
                "1": "inches",
                "2": "feet",
                "4": "millimeters",
                "5": "centimeters",
                "6": "meters",
            }.get(raw, f"acad-unit-{raw}")
    return "meters"


def _unit_scale_to_meters(units: str) -> float:
    return {
        "unitless": 1.0,
        "inches": 0.0254,
        "feet": 0.3048,
        "millimeters": 0.001,
        "centimeters": 0.01,
        "meters": 1.0,
    }.get(units, 1.0)


def _parse_line_entity(
    pairs: list[tuple[str, str]], index: int, scale: float
) -> tuple[list[tuple[float, float]], str, int]:
    values: dict[str, float] = {}
    layer = "0"
    while index < len(pairs):
        code, value = pairs[index]
        if code == "0":
            break
        if code in {"10", "20", "11", "21"}:
            values[code] = float(value) * scale
        elif code == "8":
            layer = value or "0"
        index += 1
    points: list[tuple[float, float]] = []
    if {"10", "20", "11", "21"}.issubset(values):
        points.append((values["10"], values["20"]))
        points.append((values["11"], values["21"]))
    return points, layer, index


def _parse_lwpolyline_entity(
    pairs: list[tuple[str, str]], index: int, scale: float
) -> tuple[list[tuple[float, float]], str, bool, int]:
    points: list[tuple[float, float]] = []
    pending_x: float | None = None
    layer = "0"
    is_closed = False
    while index < len(pairs):
        code, value = pairs[index]
        if code == "0":
            break
        if code == "10":
            pending_x = float(value) * scale
        elif code == "20" and pending_x is not None:
            points.append((pending_x, float(value) * scale))
            pending_x = None
        elif code == "8":
            layer = value or "0"
        elif code == "70":
            is_closed = bool(int(value) & 1)
        index += 1
    return points, layer, is_closed, index


def parse_layer_material_overrides(values: list[str] | None) -> dict[str, str]:
    """Parse repeatable ``LAYER=MATERIAL`` CLI options deterministically."""
    overrides: dict[str, str] = {}
    for raw_value in values or []:
        layer, separator, material = str(raw_value).partition("=")
        layer = layer.strip()
        material = material.strip()
        if separator == "" or layer == "" or material == "":
            raise ValueError(
                "--layer-material must use LAYER=MATERIAL, for example walls=concrete."
            )
        profile = get_material_profile(material)
        if layer in overrides:
            raise ValueError(f"--layer-material assigns layer '{layer}' more than once.")
        overrides[layer] = profile.name
    return dict(sorted(overrides.items()))


def _surface_layers_from_entities(
    entities: list[tuple[list[tuple[float, float]], str, bool]],
    *,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> dict[str, tuple[str, ...]]:
    """Map axis-aligned boundary *segments* to rectangular room faces.

    A polyline may turn at a room corner.  Its layer belongs only to the faces
    traversed by its individual segments, rather than to every bounding-box
    face incident to one of its vertices.
    """
    result: dict[str, set[str]] = {surface: set() for surface in _SURFACE_NAMES}
    tolerance = 1e-6
    for points, layer, is_closed in entities:
        if len(points) < 2:
            continue
        segments = list(pairwise(points))
        if is_closed:
            segments.append((points[-1], points[0]))
        for (x1, y1), (x2, y2) in segments:
            if abs(x1 - x2) <= tolerance:
                if abs(x1 - min_x) <= tolerance:
                    result["left"].add(layer)
                if abs(x1 - max_x) <= tolerance:
                    result["right"].add(layer)
            if abs(y1 - y2) <= tolerance:
                if abs(y1 - min_y) <= tolerance:
                    result["front"].add(layer)
                if abs(y1 - max_y) <= tolerance:
                    result["rear"].add(layer)
    return {surface: tuple(sorted(layers)) for surface, layers in result.items()}


def _validate_layer_materials(
    geometry: DXFTraceGeometry, layer_materials: dict[str, str]
) -> dict[str, str]:
    known_layers = set(geometry.known_layers())
    unknown = sorted(set(layer_materials) - known_layers)
    if unknown:
        available = ", ".join(sorted(known_layers)) or "(none)"
        raise ValueError(
            "--layer-material references unknown DXF layer(s): "
            f"{', '.join(unknown)}. Available layers: {available}."
        )
    return dict(sorted(layer_materials.items()))


def _resolve_surface_materials(
    geometry: DXFTraceGeometry, *, default_material: str, layer_materials: dict[str, str]
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for surface in _SURFACE_NAMES:
        layers = (geometry.surface_layers or {}).get(surface, ())
        mapped = [layer_materials[layer] for layer in layers if layer in layer_materials]
        if len(mapped) > 1 and len(set(mapped)) > 1:
            raise ValueError(
                f"DXF surface '{surface}' has conflicting --layer-material assignments: "
                f"{', '.join(sorted(set(mapped)))}."
            )
        resolved[surface] = mapped[0] if mapped else default_material
    return resolved


def _first_order_images(room: RoomGeometry) -> dict[str, tuple[float, float, float]]:
    x, y, z = room.source_pos_m
    width, depth, height = room.room_dims_m
    return {
        "left": (-x, y, z),
        "right": ((2.0 * width) - x, y, z),
        "front": (x, -y, z),
        "rear": (x, (2.0 * depth) - y, z),
        "floor": (x, y, -z),
        "ceiling": (x, y, (2.0 * height) - z),
    }


def _surface_pan(surface: str) -> float:
    return {"left": -0.75, "right": 0.75, "front": 0.0, "rear": 0.0}.get(surface, 0.0)


def _add_stereo_impulse(
    audio: AudioArray, sr: int, delay_s: float, amplitude: float, *, pan: float
) -> None:
    index = round(float(delay_s) * int(sr))
    if index < 0 or index >= audio.shape[0]:
        return
    left = float(amplitude) * float(np.sqrt(np.clip((1.0 - pan) * 0.5, 0.0, 1.0)))
    right = float(amplitude) * float(np.sqrt(np.clip((1.0 + pan) * 0.5, 0.0, 1.0)))
    audio[index, 0] += left
    audio[index, 1] += right


def _estimate_rt60(room: RoomGeometry) -> float:
    absorption = float(room.mean_absorption if room.mean_absorption is not None else 0.35)
    return float(0.161 * room.volume_m3 / max(1e-6, absorption * room.surface_area_m2))


def _add_late_tail(
    *,
    audio: AudioArray,
    sr: int,
    start_s: float,
    rt60_s: float,
    absorption: float,
    rays: int,
    rng: np.random.Generator,
) -> None:
    start = max(0, round(float(start_s) * int(sr)))
    if start >= audio.shape[0]:
        return
    n = audio.shape[0] - start
    t = np.arange(n, dtype=np.float64) / float(sr)
    decay = np.power(10.0, (-3.0 * t) / max(0.05, float(rt60_s)))
    density_scale = np.clip(np.log10(max(10, int(rays))) / 6.0, 0.15, 1.0)
    noise = rng.normal(0.0, 1.0, size=(n, 2))
    tail_gain = 0.035 * (1.0 - float(absorption)) * density_scale
    audio[start:, :] += noise * decay[:, np.newaxis] * tail_gain
