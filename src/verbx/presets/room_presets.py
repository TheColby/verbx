"""Dynamic room-derived render presets."""

from __future__ import annotations

from typing import Any

import numpy as np

from verbx.core.control_targets import RT60_MAX_SECONDS, RT60_MIN_SECONDS
from verbx.core.early_reflections import material_absorption
from verbx.core.room_geometry import RoomGeometry

_ROOM_PRESET_PREFIX = "room:"
_DEFAULT_ROOM_MATERIAL = "studio"


def is_room_preset_name(value: str) -> bool:
    """Return True when a preset token uses the dynamic room shorthand."""
    return str(value).strip().lower().startswith(_ROOM_PRESET_PREFIX)


def resolve_room_preset(value: str) -> tuple[str, dict[str, Any]]:
    """Resolve `room:WxDxH/material` into a RenderConfig payload."""
    raw = str(value).strip()
    if not is_room_preset_name(raw):
        raise ValueError(
            "Room preset must use the form room:<width>x<depth>x<height>/<material>."
        )

    body = raw.split(":", 1)[1].strip()
    dims_token, separator, material_token = body.partition("/")
    material = material_token.strip().lower() if separator else _DEFAULT_ROOM_MATERIAL
    if material == "":
        material = _DEFAULT_ROOM_MATERIAL

    dims = _parse_room_dims(dims_token)
    source_pos, listener_pos = _default_positions_for_room(dims)
    absorption = float(material_absorption(material, 0.35))
    geometry = RoomGeometry(
        room_dims_m=dims,
        source_pos_m=source_pos,
        listener_pos_m=listener_pos,
        wall_materials={
            "left": material,
            "right": material,
            "front": material,
            "rear": material,
            "ceiling": material,
            "floor": material,
        },
        mean_absorption=absorption,
    )

    equivalent_absorption_area = max(geometry.surface_area_m2 * absorption, 1e-6)
    rt60 = float(
        np.clip(
            (0.161 * geometry.volume_m3) / equivalent_absorption_area,
            RT60_MIN_SECONDS,
            RT60_MAX_SECONDS,
        )
    )
    room_size_macro = float(
        np.clip((np.log10(max(geometry.volume_m3, 1.0)) - np.log10(120.0)) / 1.1, -1.0, 1.0)
    )
    damping = float(np.clip(0.18 + (0.82 * absorption), 0.15, 0.92))
    clarity_macro = float(np.clip((absorption - 0.30) * 1.5, -0.35, 0.35))
    width = float(
        np.clip(
            0.92 + (0.14 * (geometry.width_m / max(geometry.height_m, 1e-6) - 1.0)),
            0.9,
            1.35,
        )
    )

    if geometry.volume_m3 < 120.0:
        fdn_lines = 8
    elif geometry.volume_m3 < 500.0:
        fdn_lines = 12
    elif geometry.volume_m3 < 1_500.0:
        fdn_lines = 16
    elif geometry.volume_m3 < 4_000.0:
        fdn_lines = 24
    else:
        fdn_lines = 32

    canonical_name = (
        f"room:{dims[0]:g}x{dims[1]:g}x{dims[2]:g}/{material}"
    )
    payload: dict[str, Any] = {
        "engine": "algo",
        "rt60": rt60,
        "wet": 0.34,
        "dry": 0.88,
        "pre_delay_ms": geometry.direct_path_pre_delay_ms,
        "damping": damping,
        "width": width,
        "fdn_lines": fdn_lines,
        "room_size_macro": room_size_macro,
        "clarity_macro": clarity_macro,
        "er_geometry": True,
        "er_room_dims_m": dims,
        "er_source_pos_m": source_pos,
        "er_listener_pos_m": listener_pos,
        "er_absorption": absorption,
        "er_material": material,
    }
    return canonical_name, payload


def _parse_room_dims(raw: str) -> tuple[float, float, float]:
    tokens = [token.strip() for token in raw.lower().replace(",", "x").split("x") if token.strip()]
    if len(tokens) != 3:
        raise ValueError(
            "Room preset dimensions must use widthxdepthxheight, for example room:6x8x3/hall."
        )
    try:
        values = tuple(float(token) for token in tokens)
    except ValueError as exc:
        raise ValueError(
            "Room preset dimensions must be numeric, for example room:6x8x3/hall."
        ) from exc
    if any(value <= 0.0 for value in values):
        raise ValueError("Room preset dimensions must be strictly positive.")
    return (values[0], values[1], values[2])


def _default_positions_for_room(
    dims: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    width, depth, height = dims
    ear_height = float(np.clip(min(1.5, 0.55 * height), 0.9, max(0.9, height - 0.25)))
    source = (
        _inside_margin(width, 0.22 * width),
        _inside_margin(depth, 0.24 * depth),
        _inside_margin(height, ear_height),
    )
    listener = (
        _inside_margin(width, 0.62 * width),
        _inside_margin(depth, 0.56 * depth),
        _inside_margin(height, ear_height),
    )
    return source, listener


def _inside_margin(limit: float, value: float) -> float:
    margin = min(0.35, max(0.08 * limit, 0.12))
    return float(np.clip(value, margin, max(margin, limit - margin)))
