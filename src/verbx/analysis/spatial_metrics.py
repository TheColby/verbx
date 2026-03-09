"""Spatial/Ambisonics analysis helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from verbx.core.spatial import (
    convert_ambisonic_convention,
)

AudioArray = npt.NDArray[np.float64]


def compute_ambisonic_metrics(
    audio: AudioArray,
    *,
    order: int,
    normalization: str,
    channel_order: str,
) -> dict[str, float]:
    """Return compact spherical-energy and directionality stability metrics."""
    if int(order) < 0:
        raise ValueError("order must be >= 0")
    if int(audio.shape[0]) == 0:
        return {
            "ambi_order": float(order),
            "ambi_energy_total": 0.0,
            "ambi_energy_omni_ratio": 0.0,
            "ambi_energy_directional_ratio": 0.0,
            "ambi_directionality_mean": 0.0,
            "ambi_directionality_std": 0.0,
            "ambi_directionality_stability": 0.0,
            "ambi_front_ratio": 0.0,
            "ambi_back_ratio": 0.0,
            "ambi_left_ratio": 0.0,
            "ambi_right_ratio": 0.0,
            "ambi_up_ratio": 0.0,
            "ambi_down_ratio": 0.0,
        }

    canonical = convert_ambisonic_convention(
        np.asarray(audio, dtype=np.float64),
        order=order,
        source_normalization=normalization,
        source_channel_order=channel_order,
        target_normalization="sn3d",
        target_channel_order="acn",
    )

    total_energy = float(np.mean(np.sum(np.square(canonical), axis=1, dtype=np.float64)))
    omni_energy = float(np.mean(np.square(canonical[:, 0]), dtype=np.float64))
    directional_energy = max(0.0, total_energy - omni_energy)
    safe_total = max(1e-12, total_energy)

    result: dict[str, float] = {
        "ambi_order": float(order),
        "ambi_energy_total": total_energy,
        "ambi_energy_omni_ratio": float(np.clip(omni_energy / safe_total, 0.0, 1.0)),
        "ambi_energy_directional_ratio": float(np.clip(directional_energy / safe_total, 0.0, 1.0)),
    }

    if order < 1 or int(canonical.shape[1]) < 4:
        result.update(
            {
                "ambi_directionality_mean": 0.0,
                "ambi_directionality_std": 0.0,
                "ambi_directionality_stability": 0.0,
                "ambi_front_ratio": 0.0,
                "ambi_back_ratio": 0.0,
                "ambi_left_ratio": 0.0,
                "ambi_right_ratio": 0.0,
                "ambi_up_ratio": 0.0,
                "ambi_down_ratio": 0.0,
            }
        )
        return result

    # ACN FOA channels: W, Y, Z, X.
    w = np.asarray(canonical[:, 0], dtype=np.float64)
    y = np.asarray(canonical[:, 1], dtype=np.float64)
    z = np.asarray(canonical[:, 2], dtype=np.float64)
    x = np.asarray(canonical[:, 3], dtype=np.float64)
    vec_mag = np.sqrt((x * x) + (y * y) + (z * z))
    dir_strength = vec_mag / np.maximum(np.abs(w), 1e-9)
    dir_mean = float(np.mean(dir_strength))
    dir_std = float(np.std(dir_strength))
    stability = float(np.clip(1.0 - (dir_std / max(1e-9, dir_mean)), 0.0, 1.0))

    front = float(np.mean(np.square(np.maximum(y, 0.0)), dtype=np.float64))
    back = float(np.mean(np.square(np.maximum(-y, 0.0)), dtype=np.float64))
    left = float(np.mean(np.square(np.maximum(-x, 0.0)), dtype=np.float64))
    right = float(np.mean(np.square(np.maximum(x, 0.0)), dtype=np.float64))
    up = float(np.mean(np.square(np.maximum(z, 0.0)), dtype=np.float64))
    down = float(np.mean(np.square(np.maximum(-z, 0.0)), dtype=np.float64))
    axis_total = max(1e-12, front + back + left + right + up + down)

    result.update(
        {
            "ambi_directionality_mean": dir_mean,
            "ambi_directionality_std": dir_std,
            "ambi_directionality_stability": stability,
            "ambi_front_ratio": float(front / axis_total),
            "ambi_back_ratio": float(back / axis_total),
            "ambi_left_ratio": float(left / axis_total),
            "ambi_right_ratio": float(right / axis_total),
            "ambi_up_ratio": float(up / axis_total),
            "ambi_down_ratio": float(down / axis_total),
        }
    )
    return result

