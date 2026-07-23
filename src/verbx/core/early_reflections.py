"""Simple image-source early-reflection synthesis for pre-reverb staging."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class ImageSourcePath:
    """One analytically derived path in a rectangular image-source model."""

    walls: tuple[str, ...]
    distance_m: float
    delay_samples: int
    gain: float


def material_absorption(material: str, default: float) -> float:
    key = str(material).strip().lower()
    if key in MATERIAL_ABSORPTION:
        return float(MATERIAL_ABSORPTION[key])
    return profile_material_absorption(key, default)


def enumerate_image_source_paths(
    *,
    sr: int,
    room_dims_m: tuple[float, float, float],
    source_pos_m: tuple[float, float, float],
    listener_pos_m: tuple[float, float, float],
    absorption: float,
    reflection_order: int = 1,
    wall_materials: dict[str, str] | None = None,
) -> tuple[ImageSourcePath, ...]:
    """Return deterministic direct and reflected paths before tap aggregation.

    Exposing the geometric path set separately from convolution makes the
    physical model testable against analytic or measured timing references.
    """
    if sr <= 0:
        raise ValueError("sr must be strictly positive.")
    room = np.asarray(room_dims_m, dtype=np.float64)
    src = np.asarray(source_pos_m, dtype=np.float64)
    lst = np.asarray(listener_pos_m, dtype=np.float64)
    if room.shape != (3,) or src.shape != (3,) or lst.shape != (3,):
        raise ValueError("room, source, and listener coordinates must contain three values.")
    if np.any(room <= 0.0):
        raise ValueError("room dimensions must be strictly positive.")

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

    paths: list[ImageSourcePath] = []

    def _add_path(distance_m: float, gain_scale: float, walls: tuple[str, ...]) -> None:
        if not np.isfinite(distance_m) or distance_m <= 1e-6:
            return
        paths.append(
            ImageSourcePath(
                walls=walls,
                distance_m=float(distance_m),
                delay_samples=round(float(distance_m / SPEED_OF_SOUND_M_S) * float(sr)),
                gain=float(gain_scale / max(distance_m, 0.25)),
            )
        )

    _add_path(float(np.linalg.norm(src - lst)), 1.0, ())
    reflections = (
        ("left", 0, 0.0),
        ("right", 0, 2.0 * room[0]),
        ("front", 1, 0.0),
        ("rear", 1, 2.0 * room[1]),
        ("floor", 2, 0.0),
        ("ceiling", 2, 2.0 * room[2]),
    )
    frontier: list[tuple[np.ndarray, float, str | None, tuple[str, ...]]] = [
        (src, 1.0, None, ())
    ]
    for _ in range(max_order):
        next_frontier: list[tuple[np.ndarray, float, str | None, tuple[str, ...]]] = []
        for image, gain, previous_wall, path_walls in frontier:
            for wall, axis, boundary_twice in reflections:
                if wall == previous_wall:
                    continue
                reflected = np.asarray(image.copy(), dtype=np.float64)
                reflected[axis] = boundary_twice - reflected[axis]
                reflected_gain = gain * wall_reflectivity[wall]
                reflected_walls = (*path_walls, wall)
                _add_path(
                    float(np.linalg.norm(reflected - lst)),
                    reflected_gain,
                    reflected_walls,
                )
                next_frontier.append((reflected, reflected_gain, wall, reflected_walls))
        frontier = next_frontier
    return tuple(paths)


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

    taps: dict[int, float] = {}
    paths = enumerate_image_source_paths(
        sr=sr,
        room_dims_m=room_dims_m,
        source_pos_m=source_pos_m,
        listener_pos_m=listener_pos_m,
        absorption=absorption,
        reflection_order=reflection_order,
        wall_materials=wall_materials,
    )
    for path in paths:
        taps[path.delay_samples] = taps.get(path.delay_samples, 0.0) + path.gain

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
