"""Feature-vector-driven control primitives for deterministic render automation.

This module provides:
- frame-aligned feature extraction from audio,
- inline lane parsing for CLI-specified feature mappings,
- lane normalization/validation for automation-file feature lanes,
- curved and hysteretic mapping from features to target control lanes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from verbx.core.control_targets import normalize_control_target_name

AudioArray = npt.NDArray[np.float32]
FloatArray = npt.NDArray[np.float64]

FEATURE_CURVE_CHOICES = {
    "linear",
    "smooth",
    "smoothstep",
    "exp",
    "log",
    "tanh",
    "power",
}
FEATURE_LANE_COMBINE_CHOICES = {"replace", "add", "multiply"}

_FEATURE_SOURCE_ALIASES: dict[str, str] = {
    "loudness_db": "loudness_db",
    "rms_db": "loudness_db",
    "rms_dbfs": "loudness_db",
    "short_lufs_db": "loudness_db",
    "loudness_norm": "loudness_norm",
    "loudness": "loudness_norm",
    "rms_norm": "loudness_norm",
    "transient_strength": "transient_strength",
    "transientness": "transient_strength",
    "onset_strength": "transient_strength",
    "onset_norm": "transient_strength",
    "spectral_flux": "spectral_flux",
    "flux": "spectral_flux",
    "flux_norm": "spectral_flux",
    "spectral_centroid_hz": "spectral_centroid_hz",
    "spectral_centroid": "spectral_centroid_hz",
    "centroid_hz": "spectral_centroid_hz",
    "brightness_hz": "spectral_centroid_hz",
    "spectral_centroid_norm": "spectral_centroid_norm",
    "centroid_norm": "spectral_centroid_norm",
    "brightness_norm": "spectral_centroid_norm",
    "spectral_flatness": "spectral_flatness",
    "flatness": "spectral_flatness",
    "harmonic_ratio": "harmonic_ratio",
    "harm_ratio": "harmonic_ratio",
}
SUPPORTED_FEATURE_SOURCES = frozenset(_FEATURE_SOURCE_ALIASES.values())

_FEATURE_LANE_TYPE_CHOICES = {
    "feature",
    "feature-map",
    "feature_map",
    "feature-vector",
    "feature_vector",
}


@dataclass(slots=True)
class FeatureVectorBus:
    """Frame-aligned feature vectors resampled onto automation control time."""

    control_features: dict[str, FloatArray]
    sample_rate: int
    frame_size: int
    hop_size: int
    signature: str


def normalize_feature_source_name(value: str) -> str:
    """Normalize a feature source token and resolve aliases."""
    key = str(value).strip().lower().replace("-", "_")
    return _FEATURE_SOURCE_ALIASES.get(key, key)


def parse_feature_vector_lane_specs(specs: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
    """Parse inline ``--feature-vector-lane`` specs into normalized lane dicts.

    Spec format (comma-separated key/value pairs):

    ``target=wet,source=loudness_norm,weight=1.0,bias=0.0,curve=smoothstep,hysteresis_up=0.02,hysteresis_down=0.01,combine=replace,smoothing_ms=20``
    """
    lanes: list[dict[str, Any]] = []
    for lane_idx, raw in enumerate(specs, start=1):
        token = str(raw).strip()
        if token == "":
            continue
        fields = _parse_lane_kv_spec(token, lane_context=f"feature lane #{lane_idx}")
        lane_obj = normalize_feature_vector_lane(
            fields,
            lane_context=f"feature lane #{lane_idx}",
        )
        lanes.append(lane_obj)
    return lanes


def load_feature_vector_lane_file(path: Path) -> list[dict[str, Any]]:
    """Load feature-vector lanes from ``.json`` or ``.csv``."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            lanes_raw = payload.get("lanes", [])
        elif isinstance(payload, list):
            lanes_raw = payload
        else:
            raise ValueError("Feature-vector JSON must be an object with 'lanes' or a list.")
        if not isinstance(lanes_raw, list):
            raise ValueError("Feature-vector JSON requires lane list.")
        lanes: list[dict[str, Any]] = []
        for lane_idx, lane_raw in enumerate(lanes_raw, start=1):
            if not isinstance(lane_raw, dict):
                raise ValueError(f"Feature-vector lane #{lane_idx} must be an object.")
            lanes.append(
                normalize_feature_vector_lane(
                    lane_raw,
                    lane_context=f"feature lane #{lane_idx}",
                )
            )
        return lanes

    if suffix == ".csv":
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    {
                        str(key): str(value)
                        for key, value in row.items()
                        if key is not None and value is not None
                    }
                )
        lanes = []
        for lane_idx, row in enumerate(rows, start=1):
            lanes.append(
                normalize_feature_vector_lane(
                    row,
                    lane_context=f"feature lane #{lane_idx}",
                )
            )
        return lanes

    raise ValueError("Feature-vector file must use .json or .csv extension.")


def normalize_feature_vector_lane(
    lane: dict[str, Any],
    *,
    lane_context: str,
) -> dict[str, Any]:
    """Normalize one feature-vector lane from JSON/CSV/CLI mapping payload."""
    lane_copy = {
        str(key).strip().lower().replace("-", "_"): value
        for key, value in dict(lane).items()
    }
    lane_type = str(lane_copy.get("type", "feature-vector")).strip().lower()
    if lane_type not in _FEATURE_LANE_TYPE_CHOICES:
        choices = ", ".join(sorted(_FEATURE_LANE_TYPE_CHOICES))
        raise ValueError(f"{lane_context}: type must be one of: {choices}.")

    target = normalize_control_target_name(str(lane_copy.get("target", "")).strip())
    if target == "":
        raise ValueError(f"{lane_context}: missing required target.")

    source_raw = (
        lane_copy.get("source")
        if lane_copy.get("source") is not None
        else lane_copy.get("feature", lane_copy.get("input", ""))
    )
    source = normalize_feature_source_name(str(source_raw).strip())
    if source not in SUPPORTED_FEATURE_SOURCES:
        choices = ", ".join(sorted(SUPPORTED_FEATURE_SOURCES))
        raise ValueError(
            f"{lane_context}: unsupported source '{source_raw}'. Supported: {choices}."
        )

    curve = str(lane_copy.get("curve", "linear")).strip().lower()
    if curve == "":
        curve = "linear"
    if curve not in FEATURE_CURVE_CHOICES:
        choices = ", ".join(sorted(FEATURE_CURVE_CHOICES))
        raise ValueError(f"{lane_context}: unsupported curve '{curve}'. Supported: {choices}.")

    combine = str(lane_copy.get("combine", "replace")).strip().lower()
    if combine == "":
        combine = "replace"
    if combine not in FEATURE_LANE_COMBINE_CHOICES:
        choices = ", ".join(sorted(FEATURE_LANE_COMBINE_CHOICES))
        raise ValueError(
            f"{lane_context}: unsupported combine mode '{combine}'. Supported: {choices}."
        )

    weight = _parse_finite_float(lane_copy.get("weight", 1.0), context=f"{lane_context} weight")
    bias = _parse_finite_float(lane_copy.get("bias", 0.0), context=f"{lane_context} bias")
    curve_amount = _parse_finite_float(
        lane_copy.get("curve_amount", lane_copy.get("amount", 1.0)),
        context=f"{lane_context} curve_amount",
    )
    if curve_amount <= 0.0:
        raise ValueError(f"{lane_context}: curve_amount must be > 0.")

    hysteresis_up = _parse_finite_float(
        lane_copy.get("hysteresis_up", lane_copy.get("hyst_up", 0.0)),
        context=f"{lane_context} hysteresis_up",
    )
    hysteresis_down = _parse_finite_float(
        lane_copy.get("hysteresis_down", lane_copy.get("hyst_down", 0.0)),
        context=f"{lane_context} hysteresis_down",
    )
    if hysteresis_up < 0.0 or hysteresis_down < 0.0:
        raise ValueError(f"{lane_context}: hysteresis_up/down must be >= 0.")

    smoothing_raw = lane_copy.get("smoothing_ms")
    smoothing_ms: float | None
    if smoothing_raw is None or str(smoothing_raw).strip() == "":
        smoothing_ms = None
    else:
        smoothing_ms = _parse_finite_float(
            smoothing_raw,
            context=f"{lane_context} smoothing_ms",
        )
        if smoothing_ms < 0.0:
            raise ValueError(f"{lane_context}: smoothing_ms must be >= 0.")

    return {
        "type": "feature-vector",
        "target": target,
        "source": source,
        "weight": float(weight),
        "bias": float(bias),
        "curve": curve,
        "curve_amount": float(curve_amount),
        "hysteresis_up": float(hysteresis_up),
        "hysteresis_down": float(hysteresis_down),
        "combine": combine,
        "smoothing_ms": smoothing_ms,
    }


def is_feature_vector_lane(lane: dict[str, Any]) -> bool:
    """Return whether lane payload describes a feature-vector lane."""
    lane_type = str(lane.get("type", "")).strip().lower()
    return lane_type in _FEATURE_LANE_TYPE_CHOICES


def build_feature_vector_bus(
    *,
    audio: AudioArray,
    sr: int,
    ctrl_times: FloatArray,
    frame_ms: float,
    hop_ms: float,
    requested_sources: set[str],
) -> FeatureVectorBus:
    """Extract and resample requested feature vectors to control-rate times."""
    if frame_ms <= 0.0:
        raise ValueError("feature frame_ms must be > 0.")
    if hop_ms <= 0.0:
        raise ValueError("feature hop_ms must be > 0.")

    resolved_sources = sorted(
        normalize_feature_source_name(source) for source in requested_sources
    )
    unsupported = sorted(source for source in resolved_sources if source not in SUPPORTED_FEATURE_SOURCES)
    if len(unsupported) > 0:
        choices = ", ".join(sorted(SUPPORTED_FEATURE_SOURCES))
        raise ValueError(
            "Unsupported feature sources: " + ", ".join(unsupported) + f". Supported: {choices}."
        )

    frame_size = max(64, int(round(float(sr) * float(frame_ms) / 1000.0)))
    hop_size = max(1, int(round(float(sr) * float(hop_ms) / 1000.0)))

    frame_times, frame_features = _extract_frame_features(
        audio=audio,
        sr=int(sr),
        frame_size=frame_size,
        hop_size=hop_size,
    )

    control_features: dict[str, FloatArray] = {}
    for source in resolved_sources:
        values = frame_features[source]
        if values.shape[0] == 0:
            control_features[source] = np.zeros(ctrl_times.shape[0], dtype=np.float64)
            continue
        control_features[source] = np.interp(
            ctrl_times,
            frame_times,
            values,
            left=float(values[0]),
            right=float(values[-1]),
        ).astype(np.float64)

    signature = _feature_bus_signature(
        control_features=control_features,
        sample_rate=int(sr),
        frame_size=frame_size,
        hop_size=hop_size,
    )
    return FeatureVectorBus(
        control_features=control_features,
        sample_rate=int(sr),
        frame_size=frame_size,
        hop_size=hop_size,
        signature=signature,
    )


def render_feature_vector_lane(
    lane: dict[str, Any],
    *,
    feature_bus: FeatureVectorBus,
) -> FloatArray:
    """Render one feature-vector lane to control-rate values."""
    normalized = normalize_feature_vector_lane(
        lane,
        lane_context="feature-vector lane",
    )
    source = str(normalized["source"])
    source_values = feature_bus.control_features.get(source)
    if source_values is None:
        raise ValueError(f"feature-vector lane source '{source}' not present in feature bus.")

    mapped = _feature_to_unit_interval(
        source_values=np.asarray(source_values, dtype=np.float64),
        source=source,
        sample_rate=int(feature_bus.sample_rate),
    )
    curved = _apply_feature_curve(
        mapped,
        curve=str(normalized["curve"]),
        amount=float(normalized["curve_amount"]),
    )
    hysteretic = _apply_hysteresis(
        curved,
        up=float(normalized["hysteresis_up"]),
        down=float(normalized["hysteresis_down"]),
    )
    weight = float(normalized["weight"])
    bias = float(normalized["bias"])
    return np.asarray((weight * hysteretic) + bias, dtype=np.float64)


def _extract_frame_features(
    *,
    audio: AudioArray,
    sr: int,
    frame_size: int,
    hop_size: int,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    mono = np.asarray(np.mean(audio, axis=1), dtype=np.float64)
    if mono.shape[0] == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return empty, {
            "loudness_db": empty,
            "loudness_norm": empty,
            "transient_strength": empty,
            "spectral_flux": empty,
            "spectral_centroid_hz": empty,
            "spectral_centroid_norm": empty,
            "spectral_flatness": empty,
            "harmonic_ratio": empty,
        }

    starts = np.arange(0, mono.shape[0], hop_size, dtype=np.int64)
    frame_count = int(starts.shape[0])
    frame_times = np.zeros(frame_count, dtype=np.float64)

    loudness_db = np.zeros(frame_count, dtype=np.float64)
    transient_raw = np.zeros(frame_count, dtype=np.float64)
    spectral_flux_raw = np.zeros(frame_count, dtype=np.float64)
    centroid_hz = np.zeros(frame_count, dtype=np.float64)
    flatness = np.zeros(frame_count, dtype=np.float64)

    window = np.hanning(frame_size).astype(np.float64)
    prev_mag: FloatArray | None = None
    prev_rms = 0.0

    for idx, start in enumerate(starts):
        stop = min(int(start + frame_size), int(mono.shape[0]))
        frame = np.zeros((frame_size,), dtype=np.float64)
        frame[: stop - start] = mono[start:stop]

        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        loudness_db[idx] = float(20.0 * np.log10(max(rms, 1e-12)))
        transient_raw[idx] = max(0.0, rms - prev_rms)
        prev_rms = rms

        spectrum = np.abs(np.fft.rfft(frame * window)).astype(np.float64)
        if prev_mag is not None and prev_mag.shape[0] == spectrum.shape[0]:
            diff = np.maximum(spectrum - prev_mag, 0.0)
            spectral_flux_raw[idx] = float(
                np.sqrt(np.sum(np.square(diff), dtype=np.float64)) / max(1, diff.shape[0])
            )
        prev_mag = spectrum

        spec_sum = float(np.sum(spectrum))
        if spec_sum > 1e-12:
            freqs = np.fft.rfftfreq(frame_size, d=1.0 / float(sr)).astype(np.float64)
            centroid_hz[idx] = float(np.sum(freqs * spectrum) / spec_sum)
            spectrum_eps = np.maximum(spectrum, 1e-12)
            geometric = float(np.exp(np.mean(np.log(spectrum_eps))))
            arithmetic = float(np.mean(spectrum_eps))
            flatness[idx] = float(np.clip(geometric / max(arithmetic, 1e-12), 0.0, 1.0))
        else:
            centroid_hz[idx] = 0.0
            flatness[idx] = 0.0

        center_sample = float(start + (0.5 * frame_size))
        frame_times[idx] = center_sample / float(sr)

    transient = _robust_unit_scale(transient_raw)
    spectral_flux = _robust_unit_scale(spectral_flux_raw)
    loudness_norm = _robust_unit_scale(loudness_db)
    nyquist = max(1e-9, 0.5 * float(sr))
    centroid_norm = np.clip(centroid_hz / nyquist, 0.0, 1.0).astype(np.float64)
    harmonic_ratio = np.clip(1.0 - flatness, 0.0, 1.0).astype(np.float64)

    return frame_times, {
        "loudness_db": loudness_db.astype(np.float64),
        "loudness_norm": loudness_norm.astype(np.float64),
        "transient_strength": transient.astype(np.float64),
        "spectral_flux": spectral_flux.astype(np.float64),
        "spectral_centroid_hz": centroid_hz.astype(np.float64),
        "spectral_centroid_norm": centroid_norm.astype(np.float64),
        "spectral_flatness": flatness.astype(np.float64),
        "harmonic_ratio": harmonic_ratio.astype(np.float64),
    }


def _feature_to_unit_interval(
    *,
    source_values: FloatArray,
    source: str,
    sample_rate: int,
) -> FloatArray:
    if source == "loudness_db":
        return np.clip((source_values + 80.0) / 80.0, 0.0, 1.0).astype(np.float64)
    if source == "spectral_centroid_hz":
        nyquist = max(1e-9, 0.5 * float(sample_rate))
        return np.clip(source_values / nyquist, 0.0, 1.0).astype(np.float64)
    return np.clip(source_values, 0.0, 1.0).astype(np.float64)


def _apply_feature_curve(values: FloatArray, *, curve: str, amount: float) -> FloatArray:
    x = np.clip(values, 0.0, 1.0).astype(np.float64)
    mode = curve.strip().lower()
    shaped: FloatArray
    if mode in {"linear"}:
        shaped = x
    elif mode in {"smooth", "smoothstep"}:
        shaped = (x * x) * (3.0 - (2.0 * x))
    elif mode == "exp":
        k = float(max(1e-6, amount))
        denom = float(np.expm1(k))
        if abs(denom) <= 1e-12:
            shaped = x
        else:
            shaped = np.expm1(k * x) / denom
    elif mode == "log":
        k = float(max(1e-6, amount))
        denom = float(np.log1p(k))
        if abs(denom) <= 1e-12:
            shaped = x
        else:
            shaped = np.log1p(k * x) / denom
    elif mode == "tanh":
        k = float(max(1e-6, amount))
        edge = float(np.tanh(k))
        if abs(edge) <= 1e-12:
            shaped = x
        else:
            centered = (2.0 * x) - 1.0
            shaped = 0.5 * ((np.tanh(k * centered) / edge) + 1.0)
    else:
        power = float(max(1e-6, amount))
        shaped = np.power(x, power)
    return np.clip(shaped, 0.0, 1.0).astype(np.float64)


def _apply_hysteresis(values: FloatArray, *, up: float, down: float) -> FloatArray:
    if values.shape[0] <= 1 or (up <= 0.0 and down <= 0.0):
        return values.astype(np.float64)

    out = np.asarray(values, dtype=np.float64).copy()
    state = float(out[0])
    for idx in range(1, out.shape[0]):
        value = float(out[idx])
        if value >= state + up:
            state = value
        elif value <= state - down:
            state = value
        out[idx] = state
    return out


def _feature_bus_signature(
    *,
    control_features: dict[str, FloatArray],
    sample_rate: int,
    frame_size: int,
    hop_size: int,
) -> str:
    h = hashlib.sha256()
    h.update(str(int(sample_rate)).encode("utf-8"))
    h.update(b"|")
    h.update(str(int(frame_size)).encode("utf-8"))
    h.update(b"|")
    h.update(str(int(hop_size)).encode("utf-8"))
    for name in sorted(control_features.keys()):
        h.update(b"|")
        h.update(name.encode("utf-8"))
        h.update(np.asarray(control_features[name], dtype=np.float32).tobytes(order="C"))
    return h.hexdigest()[:16]


def _parse_lane_kv_spec(spec: str, *, lane_context: str) -> dict[str, str]:
    out: dict[str, str] = {}
    parts = [chunk.strip() for chunk in str(spec).split(",") if chunk.strip() != ""]
    if len(parts) == 0:
        raise ValueError(f"{lane_context}: empty feature lane spec.")
    for part in parts:
        if "=" not in part:
            raise ValueError(
                f"{lane_context}: expected key=value token, got '{part}'."
            )
        key, value = part.split("=", 1)
        norm_key = key.strip().lower().replace("-", "_")
        if norm_key == "":
            raise ValueError(f"{lane_context}: empty key in token '{part}'.")
        out[norm_key] = value.strip()
    return out


def _parse_finite_float(raw: Any, *, context: str) -> float:
    text = str(raw).strip()
    if text == "":
        raise ValueError(f"{context} must not be empty.")
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{context} must be numeric: {raw}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite.")
    return float(value)


def _robust_unit_scale(values: FloatArray) -> FloatArray:
    if values.size == 0:
        return values.astype(np.float64)
    lo = float(np.percentile(values, 5.0))
    hi = float(np.percentile(values, 95.0))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.zeros_like(values, dtype=np.float64)
    if hi <= lo + 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float64)
