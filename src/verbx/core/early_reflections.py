"""Simple image-source early-reflection synthesis for pre-reverb staging."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import fftconvolve

from verbx.io.audio import ensure_mono_or_stereo
from verbx.ir.materials import material_absorption as profile_material_absorption

AudioArray = npt.NDArray[np.float64]

SPEED_OF_SOUND_M_S = 343.0
MATERIAL_ABSORPTION: dict[str, float] = {
    "anechoic": 0.95,
    "dead": 0.75,
    "studio": 0.45,
    "hall": 0.30,
    "stone": 0.15,
}


def material_absorption(material: str, default: float) -> float:
    key = str(material).strip().lower()
    if key in MATERIAL_ABSORPTION:
        return float(MATERIAL_ABSORPTION[key])
    return profile_material_absorption(key, default)


def apply_image_source_early_reflections(
    audio: AudioArray,
    *,
    sr: int,
    room_dims_m: tuple[float, float, float],
    source_pos_m: tuple[float, float, float],
    listener_pos_m: tuple[float, float, float],
    absorption: float,
    reflection_order: int = 1,
    wall_materials: dict[str, str] | None = None,
) -> AudioArray:
    """Apply direct plus bounded image-source reflections in a rectangular room."""
    x = ensure_mono_or_stereo(audio)
    if x.shape[0] == 0:
        return x.copy()

    room = np.asarray(room_dims_m, dtype=np.float64)
    src = np.asarray(source_pos_m, dtype=np.float64)
    lst = np.asarray(listener_pos_m, dtype=np.float64)
    if room.shape != (3,) or src.shape != (3,) or lst.shape != (3,):
        return x.copy()
    if np.any(room <= 0.0):
        return x.copy()

    src = np.clip(src, [0.0, 0.0, 0.0], room)
    lst = np.clip(lst, [0.0, 0.0, 0.0], room)
    max_order = int(np.clip(reflection_order, 0, 6))
    default_reflectivity = float(1.0 - np.clip(absorption, 0.0, 0.99))
    material_map = {str(key): str(value) for key, value in (wall_materials or {}).items()}
    wall_reflectivity = {
        "left": 1.0 - material_absorption(material_map.get("left", ""), absorption),
        "right": 1.0 - material_absorption(material_map.get("right", ""), absorption),
        "front": 1.0 - material_absorption(material_map.get("front", ""), absorption),
        "rear": 1.0 - material_absorption(material_map.get("rear", ""), absorption),
        "floor": 1.0 - material_absorption(material_map.get("floor", ""), absorption),
        "ceiling": 1.0 - material_absorption(material_map.get("ceiling", ""), absorption),
    }
    wall_reflectivity = {
        name: float(np.clip(value if material_map.get(name) else default_reflectivity, 0.0, 0.99))
        for name, value in wall_reflectivity.items()
    }

    taps: dict[int, float] = {}

    def _add_path(distance_m: float, gain_scale: float) -> None:
        if not np.isfinite(distance_m) or distance_m <= 1e-6:
            return
        delay = round(float(distance_m / SPEED_OF_SOUND_M_S) * float(sr))
        if delay < 0:
            return
        gain = float(gain_scale / max(distance_m, 0.25))
        taps[delay] = taps.get(delay, 0.0) + gain

    # direct path
    _add_path(float(np.linalg.norm(src - lst)), 1.0)

    reflections = (
        ("left", 0, 0.0),
        ("right", 0, 2.0 * room[0]),
        ("front", 1, 0.0),
        ("rear", 1, 2.0 * room[1]),
        ("floor", 2, 0.0),
        ("ceiling", 2, 2.0 * room[2]),
    )
    frontier: list[tuple[np.ndarray, float, str | None]] = [(src, 1.0, None)]
    for _ in range(max_order):
        next_frontier: list[tuple[np.ndarray, float, str | None]] = []
        for image, gain, previous_wall in frontier:
            for wall, axis, boundary_twice in reflections:
                # Immediate bounce reversal produces duplicate image paths.
                if wall == previous_wall:
                    continue
                reflected = np.asarray(image.copy(), dtype=np.float64)
                reflected[axis] = boundary_twice - reflected[axis]
                reflected_gain = gain * wall_reflectivity[wall]
                _add_path(float(np.linalg.norm(reflected - lst)), reflected_gain)
                next_frontier.append((reflected, reflected_gain, wall))
        frontier = next_frontier

    if len(taps) == 0:
        return x.copy()

    max_delay = int(max(taps.keys()))
    kernel = np.zeros((max_delay + 1,), dtype=np.float64)
    for delay, gain in taps.items():
        kernel[int(delay)] += float(gain)
    # Normalize early reflection kernel to a sensible range.
    peak = float(np.max(np.abs(kernel)))
    if peak > 1e-12:
        kernel *= np.float64(min(1.0, 0.9 / peak))

    out = np.zeros_like(x, dtype=np.float64)
    for ch in range(x.shape[1]):
        wet = fftconvolve(x[:, ch], kernel, mode="full")[: x.shape[0]]
        out[:, ch] = np.asarray(wet, dtype=np.float64)

    # Mix direct signal to preserve transient identity.
    mixed = (0.65 * x) + (0.35 * out)
    return np.asarray(np.nan_to_num(mixed, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
