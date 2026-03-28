"""Simple image-source early-reflection synthesis for pre-reverb staging."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import fftconvolve

from verbx.io.audio import ensure_mono_or_stereo

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
    return float(np.clip(default, 0.0, 0.99))


def apply_image_source_early_reflections(
    audio: AudioArray,
    *,
    sr: int,
    room_dims_m: tuple[float, float, float],
    source_pos_m: tuple[float, float, float],
    listener_pos_m: tuple[float, float, float],
    absorption: float,
) -> AudioArray:
    """Apply direct + first-order image-source early reflections."""
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
    absorption_value = float(np.clip(absorption, 0.0, 0.99))
    reflectivity = float(1.0 - absorption_value)

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

    # first-order reflections across each wall.
    images = (
        np.array([-src[0], src[1], src[2]], dtype=np.float64),
        np.array([2.0 * room[0] - src[0], src[1], src[2]], dtype=np.float64),
        np.array([src[0], -src[1], src[2]], dtype=np.float64),
        np.array([src[0], 2.0 * room[1] - src[1], src[2]], dtype=np.float64),
        np.array([src[0], src[1], -src[2]], dtype=np.float64),
        np.array([src[0], src[1], 2.0 * room[2] - src[2]], dtype=np.float64),
    )
    for image in images:
        _add_path(float(np.linalg.norm(image - lst)), reflectivity)

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
