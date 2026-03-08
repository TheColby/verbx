"""Ambisonics helpers for convention handling, transforms, and validation.

This module intentionally focuses on practical workflow coverage:

- channel-count/order/normalization validation,
- ACN/SN3D internal canonicalization,
- FOA encode/decode helpers for bus<->Ambisonics transforms,
- listener yaw rotation (full for FOA, FOA-subset for HOA),
- lightweight layout inference for analysis/CLI guards.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt

AudioArray = npt.NDArray[np.float32]
AmbiNorm = Literal["auto", "sn3d", "n3d", "fuma"]
AmbiOrderConvention = Literal["auto", "acn", "fuma"]

_AMBI_NORM_CHOICES = {"auto", "sn3d", "n3d", "fuma"}
_AMBI_ORDER_CHOICES = {"auto", "acn", "fuma"}


def ambisonic_channel_count(order: int) -> int:
    """Return channel count for Ambisonics order ``N``."""
    if order < 0:
        msg = "Ambisonics order must be >= 0."
        raise ValueError(msg)
    return int((order + 1) ** 2)


def infer_ambisonic_order(channels: int, max_order: int = 7) -> int | None:
    """Infer Ambisonics order from channel count when exact-square mapping exists."""
    if channels < 1:
        return None
    for order in range(max(0, max_order) + 1):
        if ambisonic_channel_count(order) == int(channels):
            return order
    return None


def normalize_ambisonic_metadata(
    *,
    order: int,
    normalization: str,
    channel_order: str,
) -> tuple[int, str, str]:
    """Validate and normalize Ambisonics metadata choices."""
    normalized_order = int(order)
    if normalized_order < 0:
        msg = "--ambi-order must be >= 0."
        raise ValueError(msg)

    norm = str(normalization).strip().lower()
    if norm not in _AMBI_NORM_CHOICES:
        choices = ", ".join(sorted(_AMBI_NORM_CHOICES))
        msg = f"--ambi-normalization must be one of: {choices}."
        raise ValueError(msg)

    ch_order = str(channel_order).strip().lower().replace("-", "_")
    if ch_order not in _AMBI_ORDER_CHOICES:
        choices = ", ".join(sorted(_AMBI_ORDER_CHOICES))
        msg = f"--channel-order must be one of: {choices}."
        raise ValueError(msg)

    if norm == "auto":
        norm = "fuma" if ch_order == "fuma" else "sn3d"
    if ch_order == "auto":
        ch_order = "fuma" if norm == "fuma" else "acn"

    if normalized_order != 1 and norm == "fuma":
        msg = "FUMA normalization is only valid for first-order Ambisonics (order 1)."
        raise ValueError(msg)
    if normalized_order != 1 and ch_order == "fuma":
        msg = "FUMA channel order is only valid for first-order Ambisonics (order 1)."
        raise ValueError(msg)

    return normalized_order, norm, ch_order


def validate_ambisonic_channels(channels: int, order: int, *, context: str) -> None:
    """Fail fast when a signal channel count mismatches expected Ambisonics order."""
    expected = ambisonic_channel_count(order)
    if int(channels) != expected:
        msg = (
            f"{context} channels must match --ambi-order {order} "
            f"({expected} channels expected, got {int(channels)})."
        )
        raise ValueError(msg)


def convert_ambisonic_convention(
    audio: AudioArray,
    *,
    order: int,
    source_normalization: str,
    source_channel_order: str,
    target_normalization: str = "sn3d",
    target_channel_order: str = "acn",
) -> AudioArray:
    """Convert Ambisonics signal between channel-order/normalization conventions."""
    if int(order) < 0:
        msg = "order must be >= 0."
        raise ValueError(msg)
    if int(order) == 0:
        return np.asarray(audio, dtype=np.float32)

    _, src_norm, src_order = normalize_ambisonic_metadata(
        order=order,
        normalization=source_normalization,
        channel_order=source_channel_order,
    )
    _, dst_norm, dst_order = normalize_ambisonic_metadata(
        order=order,
        normalization=target_normalization,
        channel_order=target_channel_order,
    )

    expected = ambisonic_channel_count(order)
    if int(audio.shape[1]) != expected:
        msg = (
            f"Ambisonics signal has {int(audio.shape[1])} channels but "
            f"order {order} expects {expected}."
        )
        raise ValueError(msg)

    canonical = _to_acn_sn3d(
        np.asarray(audio, dtype=np.float32),
        order=order,
        source_normalization=src_norm,
        source_channel_order=src_order,
    )
    return _from_acn_sn3d(
        canonical,
        order=order,
        target_normalization=dst_norm,
        target_channel_order=dst_order,
    )


def rotate_ambisonic_yaw(
    audio: AudioArray,
    *,
    order: int,
    yaw_degrees: float,
    normalization: str,
    channel_order: str,
) -> AudioArray:
    """Rotate Ambisonics signal around vertical axis (listener yaw).

    For order > 1, FOA components are rotated and higher-order channels are
    preserved as-is. This keeps processing stable while allowing practical
    orientation-aware behavior with existing content.
    """
    if abs(float(yaw_degrees)) <= 1e-12:
        return np.asarray(audio, dtype=np.float32)
    if int(order) < 1:
        return np.asarray(audio, dtype=np.float32)

    canonical = convert_ambisonic_convention(
        audio,
        order=order,
        source_normalization=normalization,
        source_channel_order=channel_order,
        target_normalization="sn3d",
        target_channel_order="acn",
    )

    theta = float(np.deg2rad(float(yaw_degrees)))
    c = float(math.cos(theta))
    s = float(math.sin(theta))

    # ACN/SN3D FOA channel order: W, Y, Z, X.
    y = np.asarray(canonical[:, 1], dtype=np.float32)
    z = np.asarray(canonical[:, 2], dtype=np.float32)
    x = np.asarray(canonical[:, 3], dtype=np.float32)
    x_rot = (c * x) - (s * y)
    y_rot = (s * x) + (c * y)

    rotated = np.asarray(canonical.copy(), dtype=np.float32)
    rotated[:, 1] = np.asarray(y_rot, dtype=np.float32)
    rotated[:, 2] = z
    rotated[:, 3] = np.asarray(x_rot, dtype=np.float32)

    return convert_ambisonic_convention(
        rotated,
        order=order,
        source_normalization="sn3d",
        source_channel_order="acn",
        target_normalization=normalization,
        target_channel_order=channel_order,
    )


def encode_bus_to_foa(audio: AudioArray, *, source: str) -> AudioArray:
    """Encode mono/stereo bus into ACN/SN3D FOA channels.

    ``source`` accepts ``mono`` or ``stereo``.
    """
    mode = str(source).strip().lower()
    if mode not in {"mono", "stereo"}:
        msg = "source must be mono or stereo."
        raise ValueError(msg)

    x = np.asarray(audio, dtype=np.float32)
    if mode == "mono":
        if int(x.shape[1]) != 1:
            msg = "FOA mono encode expects 1 input channel."
            raise ValueError(msg)
        m = np.asarray(x[:, 0], dtype=np.float32)
        foa = np.zeros((x.shape[0], 4), dtype=np.float32)
        foa[:, 0] = m * np.float32(0.70710678)
        foa[:, 1] = m * np.float32(0.5)
        foa[:, 2] = np.float32(0.0)
        foa[:, 3] = np.float32(0.0)
        return foa

    if int(x.shape[1]) != 2:
        msg = "FOA stereo encode expects 2 input channels."
        raise ValueError(msg)
    left = np.asarray(x[:, 0], dtype=np.float32)
    right = np.asarray(x[:, 1], dtype=np.float32)
    mid = np.asarray(0.5 * (left + right), dtype=np.float32)
    side = np.asarray(0.5 * (left - right), dtype=np.float32)

    foa = np.zeros((x.shape[0], 4), dtype=np.float32)
    foa[:, 0] = mid * np.float32(0.70710678)
    foa[:, 1] = mid * np.float32(0.5)
    foa[:, 2] = np.float32(0.0)
    foa[:, 3] = side
    return foa


def decode_foa_to_stereo(
    audio: AudioArray,
    *,
    order: int,
    normalization: str,
    channel_order: str,
) -> AudioArray:
    """Decode FOA/HOA signal to stereo using FOA components."""
    if int(order) < 1:
        msg = "FOA decode requires --ambi-order >= 1."
        raise ValueError(msg)
    if int(audio.shape[1]) < 4:
        msg = "FOA decode expects at least 4 Ambisonic channels."
        raise ValueError(msg)

    canonical = convert_ambisonic_convention(
        np.asarray(audio, dtype=np.float32),
        order=order,
        source_normalization=normalization,
        source_channel_order=channel_order,
        target_normalization="sn3d",
        target_channel_order="acn",
    )
    w = np.asarray(canonical[:, 0], dtype=np.float32)
    y = np.asarray(canonical[:, 1], dtype=np.float32)
    x = np.asarray(canonical[:, 3], dtype=np.float32)

    left = (np.float32(0.70710678) * w) + (np.float32(0.5) * y) + (np.float32(0.5) * x)
    right = (np.float32(0.70710678) * w) + (np.float32(0.5) * y) - (np.float32(0.5) * x)
    return np.asarray(np.column_stack((left, right)), dtype=np.float32)


def _to_acn_sn3d(
    audio: AudioArray,
    *,
    order: int,
    source_normalization: str,
    source_channel_order: str,
) -> AudioArray:
    """Convert signal to ACN/SN3D internal representation."""
    converted = np.asarray(audio.copy(), dtype=np.float32)

    if source_channel_order == "fuma":
        # FUMA FOA: [W, X, Y, Z] -> ACN FOA: [W, Y, Z, X]
        w = np.asarray(converted[:, 0], dtype=np.float32)
        x = np.asarray(converted[:, 1], dtype=np.float32)
        y = np.asarray(converted[:, 2], dtype=np.float32)
        z = np.asarray(converted[:, 3], dtype=np.float32)
        converted[:, 0] = w
        converted[:, 1] = y
        converted[:, 2] = z
        converted[:, 3] = x

    if source_normalization == "fuma":
        converted[:, 0] *= np.float32(math.sqrt(2.0))
        return converted

    if source_normalization == "n3d":
        for n in range(order + 1):
            lo = n * n
            hi = lo + (2 * n) + 1
            factor = np.float32(1.0 / math.sqrt((2 * n) + 1.0))
            converted[:, lo:hi] *= factor

    return converted


def _from_acn_sn3d(
    audio: AudioArray,
    *,
    order: int,
    target_normalization: str,
    target_channel_order: str,
) -> AudioArray:
    """Convert ACN/SN3D internal representation to requested convention."""
    converted = np.asarray(audio.copy(), dtype=np.float32)

    if target_normalization == "n3d":
        for n in range(order + 1):
            lo = n * n
            hi = lo + (2 * n) + 1
            factor = np.float32(math.sqrt((2 * n) + 1.0))
            converted[:, lo:hi] *= factor
    elif target_normalization == "fuma":
        converted[:, 0] *= np.float32(1.0 / math.sqrt(2.0))

    if target_channel_order == "fuma":
        # ACN FOA: [W, Y, Z, X] -> FUMA FOA: [W, X, Y, Z]
        w = np.asarray(converted[:, 0], dtype=np.float32)
        y = np.asarray(converted[:, 1], dtype=np.float32)
        z = np.asarray(converted[:, 2], dtype=np.float32)
        x = np.asarray(converted[:, 3], dtype=np.float32)
        converted[:, 0] = w
        converted[:, 1] = x
        converted[:, 2] = y
        converted[:, 3] = z

    return converted

