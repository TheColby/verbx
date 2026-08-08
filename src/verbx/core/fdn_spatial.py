"""Spatial-coupling helpers for multichannel FDN wet buses."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

AudioArray = npt.NDArray[np.float64]
SPATIAL_COUPLING_MODES = frozenset(
    {"none", "adjacent", "front_rear", "bed_top", "all_to_all"}
)


def normalize_spatial_coupling_mode(mode: str) -> str:
    """Normalize and validate a spatial-coupling identifier."""
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in SPATIAL_COUPLING_MODES:
        return normalized
    raise ValueError(f"Unsupported FDN spatial coupling mode: {mode}")


def infer_layout(layout: str, channels: int) -> str:
    """Resolve ``auto`` to the deterministic layout used by the FDN engine."""
    normalized = str(layout).strip().lower()
    if normalized != "auto":
        return normalized
    return {
        3: "lcr",
        6: "5.1",
        8: "7.1",
        10: "7.1.2",
        12: "7.1.4",
        13: "7.2.4",
        16: "16.0",
        68: "64.4",
    }.get(channels, "auto")


def layout_channel_groups(
    layout: str,
    channels: int,
) -> tuple[list[int], list[int], list[int]]:
    """Return front, rear, and top channel groups for common bus layouts."""
    groups = {
        "lcr": ([0, 1, 2], [], []),
        "5.1": ([0, 1, 2, 3], [4, 5], []),
        "7.1": ([0, 1, 2, 3], [4, 5, 6, 7], []),
        "7.1.2": ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9]),
        "7.1.4": ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]),
        "7.2.4": ([0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]),
        "8.0": ([0, 1, 2], [3, 4, 5, 6, 7], []),
    }
    if layout in groups:
        return groups[layout]
    if layout == "16.0":
        return list(range(0, min(8, channels))), list(range(8, channels)), []
    if layout == "64.4" and channels >= 68:
        return list(range(0, 32)), list(range(32, 64)), list(range(64, 68))
    return list(range(channels)), [], []


def apply_spatial_coupling(
    wet: AudioArray,
    *,
    layout: str,
    mode: str,
    strength: float,
) -> AudioArray:
    """Apply deterministic directional coupling to a multichannel wet bus."""
    out = np.asarray(wet, dtype=np.float64)
    channels = int(out.shape[1])
    resolved_mode = normalize_spatial_coupling_mode(mode)
    resolved_strength = float(np.clip(strength, 0.0, 1.0))
    if channels <= 2 or resolved_strength <= 0.0 or resolved_mode == "none":
        return out

    coupled = np.asarray(out.copy(), dtype=np.float64)
    front, rear, top = layout_channel_groups(infer_layout(layout, channels), channels)
    if resolved_mode == "adjacent":
        coupled[:, :] = 0.5 * (np.roll(out, 1, axis=1) + np.roll(out, -1, axis=1))
    elif resolved_mode == "all_to_all":
        coupled[:, :] = (np.sum(out, axis=1, keepdims=True) - out) / float(channels - 1)
    elif resolved_mode == "front_rear":
        if not front or not rear:
            return out
        front_mean = np.mean(out[:, front], axis=1)
        rear_mean = np.mean(out[:, rear], axis=1)
        coupled[:, front] = rear_mean[:, np.newaxis]
        coupled[:, rear] = front_mean[:, np.newaxis]
    elif resolved_mode == "bed_top":
        bed = sorted({*front, *rear})
        if not bed or not top:
            return out
        bed_mean = np.mean(out[:, bed], axis=1)
        top_mean = np.mean(out[:, top], axis=1)
        coupled[:, bed] = top_mean[:, np.newaxis]
        coupled[:, top] = bed_mean[:, np.newaxis]
    mixed = ((1.0 - resolved_strength) * out) + (resolved_strength * coupled)
    return np.asarray(np.nan_to_num(mixed, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
