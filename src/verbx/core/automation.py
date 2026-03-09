"""Render automation timeline parsing and application."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from verbx.core.control_targets import (
    CONV_CONTROL_TARGETS,
    CONTROL_TARGET_LIMITS,
    ENGINE_CONTROL_TARGETS,
    POST_RENDER_CONTROL_TARGETS,
    normalize_control_target_name,
)
from verbx.core.feature_vector import (
    FEATURE_LANE_COMBINE_CHOICES,
    build_feature_vector_bus,
    is_feature_vector_lane,
    normalize_feature_source_name,
    normalize_feature_vector_lane,
    parse_feature_vector_lane_specs,
    render_feature_vector_lane,
)

AudioArray = npt.NDArray[np.float64]

POST_RENDER_AUTOMATION_TARGETS = set(POST_RENDER_CONTROL_TARGETS)
ENGINE_AUTOMATION_TARGETS = set(ENGINE_CONTROL_TARGETS)
CONV_AUTOMATION_TARGETS = set(CONV_CONTROL_TARGETS)
SUPPORTED_AUTOMATION_TARGETS = (
    POST_RENDER_AUTOMATION_TARGETS | ENGINE_AUTOMATION_TARGETS | CONV_AUTOMATION_TARGETS
)
TARGET_LIMITS: dict[str, tuple[float, float]] = dict(CONTROL_TARGET_LIMITS)
BREAKPOINT_INTERP_CHOICES = {
    "linear",
    "hold",
    "step",
    "smooth",
    "smoothstep",
    "exp",
    "exponential",
}
LANE_COMBINE_CHOICES = {"replace", "add", "multiply"}


@dataclass(slots=True)
class AutomationBundle:
    """Resolved automation curves at sample rate."""

    source_path: str
    mode: str
    control_step: int
    sample_rate: int
    num_samples: int
    curves: dict[str, AudioArray]
    lanes_per_target: dict[str, int]
    signature: str
    feature_curves: dict[str, AudioArray] = field(default_factory=dict)
    feature_sources: tuple[str, ...] = ()
    feature_signature: str | None = None


def parse_automation_clamp_overrides(
    specs: tuple[str, ...] | list[str],
) -> dict[str, tuple[float, float]]:
    """Parse ``--automation-clamp target:min:max`` style overrides."""
    overrides: dict[str, tuple[float, float]] = {}
    for raw in specs:
        token = str(raw).strip()
        parts = token.split(":")
        if len(parts) != 3:
            msg = (
                "Automation clamp override must use target:min:max format. "
                f"Invalid value: {raw}"
            )
            raise ValueError(msg)
        target = _normalize_target_name(parts[0])
        if target not in SUPPORTED_AUTOMATION_TARGETS:
            options = ", ".join(sorted(SUPPORTED_AUTOMATION_TARGETS))
            msg = f"Unsupported automation clamp target '{target}'. Supported: {options}"
            raise ValueError(msg)
        try:
            minimum = float(parts[1])
            maximum = float(parts[2])
        except ValueError as exc:
            msg = f"Automation clamp values must be numeric: {raw}"
            raise ValueError(msg) from exc
        if minimum >= maximum:
            msg = f"Automation clamp requires min < max: {raw}"
            raise ValueError(msg)
        overrides[target] = (minimum, maximum)
    return overrides


def parse_automation_point_specs(specs: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
    """Parse repeatable ``--automation-point`` specs into breakpoint lanes.

    Point spec format:
    ``target:time_s:value[:interp]``
    """
    grouped: dict[str, list[tuple[float, float]]] = {}
    interp_per_target: dict[str, str] = {}
    for raw in specs:
        token = str(raw).strip()
        if token == "":
            continue
        parts = [chunk.strip() for chunk in token.split(":")]
        if len(parts) not in {3, 4}:
            raise ValueError(
                "Automation point must use target:time_s:value[:interp] format. "
                f"Invalid value: {raw}"
            )
        target = _normalize_target_name(parts[0])
        if target not in SUPPORTED_AUTOMATION_TARGETS:
            options = ", ".join(sorted(SUPPORTED_AUTOMATION_TARGETS))
            raise ValueError(
                f"Unsupported automation point target '{target}'. Supported: {options}"
            )
        try:
            time_s = float(parts[1])
            value = float(parts[2])
        except ValueError as exc:
            raise ValueError(
                f"Automation point time/value must be numeric: {raw}"
            ) from exc
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError(f"Automation point time must be finite and >= 0: {raw}")
        if not math.isfinite(value):
            raise ValueError(f"Automation point value must be finite: {raw}")
        grouped.setdefault(target, []).append((time_s, value))
        if len(parts) == 4 and parts[3] != "":
            interp = parts[3].strip().lower()
            if interp not in BREAKPOINT_INTERP_CHOICES:
                choices = ", ".join(sorted(BREAKPOINT_INTERP_CHOICES))
                raise ValueError(
                    f"Unsupported automation interpolation '{interp}'. "
                    f"Supported: {choices}"
                )
            interp_per_target[target] = interp

    lanes: list[dict[str, Any]] = []
    for target, points in grouped.items():
        sorted_points = sorted(points, key=lambda item: item[0])
        lanes.append(
            {
                "target": target,
                "type": "breakpoints",
                "interp": interp_per_target.get(target, "linear"),
                "points": [{"time": t, "value": v} for t, v in sorted_points],
            }
        )
    return lanes


def collect_automation_targets(
    *,
    path: Path | None,
    point_specs: tuple[str, ...] | list[str] = (),
    feature_lane_specs: tuple[str, ...] | list[str] = (),
) -> set[str]:
    """Collect normalized automation targets from file and inline point specs."""
    targets: set[str] = set()
    if path is not None:
        spec = _read_automation_spec(path)
        for lane in spec.get("lanes", []):
            if not isinstance(lane, dict):
                continue
            target = _normalize_target_name(lane.get("target", ""))
            if target == "":
                continue
            if target not in SUPPORTED_AUTOMATION_TARGETS:
                options = ", ".join(sorted(SUPPORTED_AUTOMATION_TARGETS))
                raise ValueError(
                    f"Unsupported automation target '{target}'. Supported: {options}."
                )
            targets.add(target)
    for lane in parse_automation_point_specs(point_specs):
        target = _normalize_target_name(lane.get("target", ""))
        if target != "":
            targets.add(target)
    for lane in parse_feature_vector_lane_specs(feature_lane_specs):
        target = _normalize_target_name(str(lane.get("target", "")))
        if target != "":
            targets.add(target)
    return targets


def load_automation_bundle(
    *,
    path: Path | None,
    point_specs: tuple[str, ...] | list[str] = (),
    feature_lane_specs: tuple[str, ...] | list[str] = (),
    feature_audio: AudioArray | None = None,
    sr: int,
    num_samples: int,
    mode: str = "auto",
    block_ms: float = 20.0,
    smoothing_ms: float = 20.0,
    feature_frame_ms: float = 40.0,
    feature_hop_ms: float = 20.0,
    clamp_overrides: dict[str, tuple[float, float]] | None = None,
) -> AutomationBundle:
    """Load automation lanes from JSON/CSV and evaluate to sample-rate curves."""
    spec: dict[str, Any] = {"mode": "block", "block_ms": float(block_ms), "lanes": []}
    source_tokens: list[str] = []
    if path is not None:
        if not path.exists():
            msg = f"Automation file not found: {path}"
            raise ValueError(msg)
        spec = _read_automation_spec(path)
        source_tokens.append(str(path))

    lanes = list(spec.get("lanes", []))
    inline_lanes = parse_automation_point_specs(point_specs)
    if len(inline_lanes) > 0:
        lanes.extend(inline_lanes)
        source_tokens.append("inline-cli")
    inline_feature_lanes = parse_feature_vector_lane_specs(feature_lane_specs)
    if len(inline_feature_lanes) > 0:
        lanes.extend(inline_feature_lanes)
        source_tokens.append("inline-feature")

    file_mode = str(spec.get("mode", "block")).strip().lower()
    file_block_ms = float(spec.get("block_ms", block_ms))
    resolved_mode = str(mode).strip().lower()
    if resolved_mode == "auto":
        resolved_mode = file_mode
    if resolved_mode not in {"sample", "block"}:
        msg = "--automation-mode must be one of: auto, sample, block."
        raise ValueError(msg)

    resolved_block_ms = float(block_ms) if mode != "auto" else file_block_ms
    if resolved_block_ms <= 0.0:
        msg = "--automation-block-ms must be > 0."
        raise ValueError(msg)
    control_step = (
        1
        if resolved_mode == "sample"
        else max(1, round(sr * resolved_block_ms / 1000.0))
    )
    ctrl_count = math.ceil(max(1, num_samples) / float(control_step))
    ctrl_times = np.arange(ctrl_count, dtype=np.float64) * (float(control_step) / float(sr))

    feature_lane_sources = _collect_feature_lane_sources(lanes)
    feature_bus = None
    if len(feature_lane_sources) > 0:
        if feature_audio is None:
            raise ValueError(
                "Feature-vector lanes require feature source audio. "
                "Provide --feature-vector-lane or feature-vector automation only in render context."
            )
        feature_bus = build_feature_vector_bus(
            audio=np.asarray(feature_audio, dtype=np.float64),
            sr=int(sr),
            ctrl_times=ctrl_times,
            frame_ms=float(feature_frame_ms),
            hop_ms=float(feature_hop_ms),
            requested_sources=feature_lane_sources,
        )

    limit_map = dict(TARGET_LIMITS)
    if clamp_overrides is not None:
        limit_map.update(clamp_overrides)

    controls: dict[str, npt.NDArray[np.float64]] = {}
    lanes_per_target: dict[str, int] = {}
    for lane_idx, lane_obj in enumerate(lanes, start=1):
        if not isinstance(lane_obj, dict):
            raise ValueError(f"Automation lane #{lane_idx} must be an object.")
        lane = dict(lane_obj)
        target = _normalize_target_name(lane.get("target", ""))
        if target not in SUPPORTED_AUTOMATION_TARGETS:
            options = ", ".join(sorted(SUPPORTED_AUTOMATION_TARGETS))
            msg = (
                f"Unsupported automation target '{target}' in lane #{lane_idx}. "
                f"Supported: {options}."
            )
            raise ValueError(msg)

        lane_context = f"lane #{lane_idx} (target '{target}')"
        combine = _normalize_lane_combine(lane.get("combine", "replace"), lane_context=lane_context)
        lane_smoothing_raw = lane.get("smoothing_ms")
        if lane_smoothing_raw is None:
            lane_smoothing_raw = smoothing_ms
        lane_smoothing_ms = _parse_lane_smoothing_ms(
            lane_smoothing_raw,
            lane_context=lane_context,
        )
        try:
            lane_values = _render_lane_values(
                lane=lane,
                ctrl_times=ctrl_times,
                feature_bus=feature_bus,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid automation {lane_context}: {exc}") from exc

        lane_values = _apply_smoothing(lane_values, lane_smoothing_ms, sr=sr, step=control_step)
        min_v, max_v = limit_map[target]
        lane_values = np.clip(lane_values, min_v, max_v)

        existing = controls.get(target, np.full(ctrl_count, np.nan, dtype=np.float64))
        if combine == "add":
            existing = np.nan_to_num(existing, nan=0.0) + lane_values
        elif combine == "multiply":
            existing = np.nan_to_num(existing, nan=1.0) * lane_values
        else:
            mask = ~np.isnan(lane_values)
            existing[mask] = lane_values[mask]
        controls[target] = existing
        lanes_per_target[target] = int(lanes_per_target.get(target, 0) + 1)

    curves: dict[str, AudioArray] = {}
    for target in sorted(controls.keys()):
        control = controls[target]
        expanded = _expand_control_to_samples(control, step=control_step, num_samples=num_samples)
        curves[target] = np.asarray(expanded, dtype=np.float64)
    feature_curves: dict[str, AudioArray] = {}
    feature_signature: str | None = None
    feature_sources: tuple[str, ...] = ()
    if feature_bus is not None:
        feature_signature = str(feature_bus.signature)
        feature_sources = tuple(sorted(feature_bus.control_features.keys()))
        for source in feature_sources:
            control_values = feature_bus.control_features[source]
            expanded_feature = _expand_control_to_samples(
                control_values,
                step=control_step,
                num_samples=num_samples,
            )
            feature_curves[source] = np.asarray(expanded_feature, dtype=np.float64)

    source_path = "+".join(source_tokens) if len(source_tokens) > 0 else "inline-empty"
    signature = _bundle_signature(
        curves=curves,
        mode=resolved_mode,
        control_step=control_step,
        sample_rate=int(sr),
        num_samples=int(num_samples),
    )

    return AutomationBundle(
        source_path=source_path,
        mode=resolved_mode,
        control_step=control_step,
        sample_rate=int(sr),
        num_samples=int(num_samples),
        curves=curves,
        lanes_per_target=lanes_per_target,
        signature=signature,
        feature_curves=feature_curves,
        feature_sources=feature_sources,
        feature_signature=feature_signature,
    )


def apply_render_automation(
    *,
    rendered: AudioArray,
    dry_reference: AudioArray,
    base_wet: float,
    base_dry: float,
    bundle: AutomationBundle,
) -> tuple[AudioArray, dict[str, Any]]:
    """Apply wet/dry/gain automation to rendered output."""
    n = int(rendered.shape[0])
    if n == 0:
        summary = {
            "source": bundle.source_path,
            "mode": bundle.mode,
            "control_step": int(bundle.control_step),
            "targets": [],
            "lanes_per_target": bundle.lanes_per_target,
            "signature": bundle.signature,
        }
        if len(bundle.feature_curves) > 0:
            summary["feature_vector"] = {
                "sources": list(bundle.feature_sources),
                "source_stats": _target_stats(bundle.feature_curves),
                "signature": bundle.feature_signature,
            }
        return np.asarray(rendered, dtype=np.float64), summary

    out = np.asarray(rendered, dtype=np.float64)
    active_targets = sorted(bundle.curves.keys())

    wet_curve = bundle.curves.get("wet")
    dry_curve = bundle.curves.get("dry")
    if wet_curve is not None or dry_curve is not None:
        wet_base = float(base_wet)
        dry_base = float(base_dry)
        if abs(wet_base) <= 1e-9:
            wet_estimate = np.zeros_like(out, dtype=np.float64)
        else:
            wet_estimate = np.asarray(
                (out - (dry_base * dry_reference)) / wet_base,
                dtype=np.float64,
            )

        wet_gain = wet_curve if wet_curve is not None else np.full(n, wet_base, dtype=np.float64)
        dry_gain = dry_curve if dry_curve is not None else np.full(n, dry_base, dtype=np.float64)
        out = np.asarray(
            (dry_gain[:, np.newaxis] * dry_reference) + (wet_gain[:, np.newaxis] * wet_estimate),
            dtype=np.float64,
        )

    gain_curve = bundle.curves.get("gain-db")
    if gain_curve is not None:
        linear = np.asarray(np.power(10.0, gain_curve / 20.0), dtype=np.float64)
        out = np.asarray(out * linear[:, np.newaxis], dtype=np.float64)

    summary = {
        "source": bundle.source_path,
        "mode": bundle.mode,
        "control_step": int(bundle.control_step),
        "targets": active_targets,
        "lanes_per_target": bundle.lanes_per_target,
        "target_stats": _target_stats(bundle.curves),
        "signature": bundle.signature,
    }
    if len(bundle.feature_curves) > 0:
        summary["feature_vector"] = {
            "sources": list(bundle.feature_sources),
            "source_stats": _target_stats(bundle.feature_curves),
            "signature": bundle.feature_signature,
        }
    sanitized = np.asarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
    return sanitized, summary


def write_automation_trace(path: Path, bundle: AutomationBundle) -> None:
    """Write resolved automation curves as CSV trace for QA/debugging."""
    if len(bundle.curves) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    targets = sorted(bundle.curves.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "time_s", *targets])
        for idx in range(bundle.num_samples):
            time_s = float(idx / float(bundle.sample_rate))
            row: list[float | int] = [idx, time_s]
            for target in targets:
                values = bundle.curves[target]
                row.append(float(values[idx]))
            writer.writerow(row)


def write_feature_vector_trace(path: Path, bundle: AutomationBundle) -> None:
    """Write feature + parameter trace export for Track B explainability."""
    if len(bundle.feature_curves) == 0 and len(bundle.curves) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    feature_names = sorted(bundle.feature_curves.keys())
    target_names = sorted(bundle.curves.keys())

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_index",
                "time_s",
                *[f"feature_{name}" for name in feature_names],
                *[f"target_{name}" for name in target_names],
            ]
        )
        for idx in range(bundle.num_samples):
            row: list[float | int] = [idx, float(idx / float(bundle.sample_rate))]
            for name in feature_names:
                row.append(float(bundle.feature_curves[name][idx]))
            for name in target_names:
                row.append(float(bundle.curves[name][idx]))
            writer.writerow(row)


def _read_automation_spec(path: Path) -> dict[str, Any]:
    """Read automation spec from JSON or CSV."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Automation JSON must be an object.")
        lanes = payload.get("lanes")
        if not isinstance(lanes, list):
            raise ValueError("Automation JSON requires a top-level 'lanes' list.")
        return payload

    if suffix == ".csv":
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    {
                        str(k): str(v)
                        for k, v in row.items()
                        if k is not None and v is not None
                    }
                )
        lanes = _lanes_from_csv_rows(rows)
        return {"mode": "block", "block_ms": 20.0, "lanes": lanes}

    raise ValueError("Automation file must be .json or .csv.")


def _lanes_from_csv_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build breakpoint lanes grouped by target from CSV rows."""
    grouped: dict[str, list[tuple[float, float]]] = {}
    interp_map: dict[str, str] = {}
    for row in rows:
        target = _normalize_target_name(row.get("target", ""))
        if target == "":
            continue
        if target not in SUPPORTED_AUTOMATION_TARGETS:
            options = ", ".join(sorted(SUPPORTED_AUTOMATION_TARGETS))
            msg = f"Unsupported automation target '{target}'. Supported: {options}."
            raise ValueError(msg)
        try:
            time_s = float(row.get("time_s", "0"))
            value = float(row.get("value", "0"))
        except ValueError as exc:
            raise ValueError("Automation CSV rows require numeric time_s and value.") from exc
        grouped.setdefault(target, []).append((time_s, value))
        interp = str(row.get("interp", "")).strip().lower()
        if interp != "":
            interp_map[target] = interp

    lanes: list[dict[str, Any]] = []
    for target, points in grouped.items():
        sorted_points = sorted(points, key=lambda item: item[0])
        lanes.append(
            {
                "target": target,
                "type": "breakpoints",
                "interp": interp_map.get(target, "linear"),
                "points": [{"time": t, "value": v} for t, v in sorted_points],
            }
        )
    return lanes


def _render_lane_values(
    *,
    lane: dict[str, Any],
    ctrl_times: npt.NDArray[np.float64],
    feature_bus: Any | None = None,
) -> npt.NDArray[np.float64]:
    """Render one automation lane at control-rate time samples."""
    lane_type = str(lane.get("type", "breakpoints")).strip().lower().replace("_", "-")
    if lane_type in {"breakpoints", "ramp", "curve"}:
        return _render_breakpoints(lane, ctrl_times)
    if lane_type == "lfo":
        return _render_lfo(lane, ctrl_times)
    if lane_type in {"segment", "segments"}:
        return _render_segment(lane, ctrl_times)
    if lane_type in {"feature", "feature-map", "feature-vector"}:
        if feature_bus is None:
            raise ValueError("Feature-vector lanes require resolved feature bus context.")
        return render_feature_vector_lane(lane, feature_bus=feature_bus)
    raise ValueError(f"Unsupported automation lane type: {lane_type}")


def _render_breakpoints(
    lane: dict[str, Any],
    ctrl_times: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Render breakpoint/ramp lane with interpolation mode."""
    points = lane.get("points")
    parsed = _parse_points(points)
    if len(parsed) == 0:
        raise ValueError("Breakpoint lane requires non-empty 'points'.")
    parsed.sort(key=lambda item: item[0])
    times = np.asarray([item[0] for item in parsed], dtype=np.float64)
    values = np.asarray([item[1] for item in parsed], dtype=np.float64)

    interp = str(lane.get("interp", lane.get("curve", "linear"))).strip().lower()
    if interp == "":
        interp = "linear"
    if interp in {"hold", "step"}:
        idx = np.searchsorted(times, ctrl_times, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return np.asarray(values[idx], dtype=np.float64)

    if interp in {"smooth", "smoothstep"}:
        return _smoothstep_interpolate(ctrl_times, times, values)

    if interp in {"exp", "exponential"}:
        return _exp_interpolate(ctrl_times, times, values)

    if interp == "linear":
        return np.interp(
            ctrl_times,
            times,
            values,
            left=values[0],
            right=values[-1],
        ).astype(np.float64)

    choices = ", ".join(sorted(BREAKPOINT_INTERP_CHOICES))
    raise ValueError(
        f"Unsupported automation interpolation '{interp}'. Supported: {choices}"
    )


def _render_lfo(
    lane: dict[str, Any],
    ctrl_times: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Render LFO lane."""
    shape = str(lane.get("shape", "sine")).strip().lower()
    allowed_shapes = {"sine", "triangle", "tri", "square", "pulse", "saw", "sawtooth"}
    if shape not in allowed_shapes:
        choices = ", ".join(sorted(allowed_shapes))
        raise ValueError(f"LFO lane shape must be one of: {choices}.")

    rate_hz = float(lane.get("rate_hz", 0.0))
    depth = float(lane.get("depth", 0.0))
    center = float(lane.get("center", 0.0))
    phase_deg = float(lane.get("phase_deg", 0.0))
    start_s = float(lane.get("start_s", 0.0))
    end_raw = lane.get("end_s")
    end_s = float(end_raw) if end_raw is not None else float(ctrl_times[-1]) + 1e9
    if not np.isfinite(rate_hz):
        raise ValueError("LFO lane rate_hz must be finite.")
    if rate_hz < 0.0:
        raise ValueError("LFO lane rate_hz must be >= 0.")
    if not np.isfinite(depth):
        raise ValueError("LFO lane depth must be finite.")
    if depth < 0.0:
        raise ValueError("LFO lane depth must be >= 0.")
    if not np.isfinite(center):
        raise ValueError("LFO lane center must be finite.")
    if not np.isfinite(phase_deg):
        raise ValueError("LFO lane phase_deg must be finite.")
    if not np.isfinite(start_s):
        raise ValueError("LFO lane start_s must be finite.")
    if not np.isfinite(end_s):
        raise ValueError("LFO lane end_s must be finite.")
    if end_s < start_s:
        raise ValueError("LFO lane end_s must be >= start_s.")

    out = np.full(ctrl_times.shape[0], np.nan, dtype=np.float64)
    mask = (ctrl_times >= start_s) & (ctrl_times <= end_s)
    if np.count_nonzero(mask) == 0:
        return out

    phase = (2.0 * np.pi * rate_hz * (ctrl_times[mask] - start_s)) + np.deg2rad(phase_deg)
    if shape in {"triangle", "tri"}:
        wave = (2.0 / np.pi) * np.arcsin(np.sin(phase))
    elif shape in {"square", "pulse"}:
        wave = np.sign(np.sin(phase))
    elif shape in {"saw", "sawtooth"}:
        wave = 2.0 * ((phase / (2.0 * np.pi)) - np.floor(0.5 + (phase / (2.0 * np.pi))))
    else:
        wave = np.sin(phase)
    out[mask] = center + (depth * wave)
    return out


def _render_segment(
    lane: dict[str, Any],
    ctrl_times: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Render constant segment lane with optional ramp in/out."""
    start_s = float(lane.get("start_s", 0.0))
    end_s = float(lane.get("end_s", start_s))
    value = float(lane.get("value", 0.0))
    ramp_ms = float(lane.get("ramp_ms", 0.0))
    if not np.isfinite(start_s):
        raise ValueError("Segment lane start_s must be finite.")
    if not np.isfinite(end_s):
        raise ValueError("Segment lane end_s must be finite.")
    if not np.isfinite(value):
        raise ValueError("Segment lane value must be finite.")
    if not np.isfinite(ramp_ms):
        raise ValueError("Segment lane ramp_ms must be finite.")
    if ramp_ms < 0.0:
        raise ValueError("Segment lane ramp_ms must be >= 0.")
    if end_s < start_s:
        raise ValueError("Segment lane end_s must be >= start_s.")

    out = np.full(ctrl_times.shape[0], np.nan, dtype=np.float64)
    mask = (ctrl_times >= start_s) & (ctrl_times <= end_s)
    out[mask] = value
    if ramp_ms <= 0.0:
        return out

    ramp_s = float(ramp_ms / 1000.0)
    in_mask = (ctrl_times >= start_s) & (ctrl_times < (start_s + ramp_s))
    if np.count_nonzero(in_mask) > 0:
        alpha = np.clip((ctrl_times[in_mask] - start_s) / max(1e-9, ramp_s), 0.0, 1.0)
        out[in_mask] = value * alpha
    out_mask = (ctrl_times > (end_s - ramp_s)) & (ctrl_times <= end_s)
    if np.count_nonzero(out_mask) > 0:
        alpha = np.clip((end_s - ctrl_times[out_mask]) / max(1e-9, ramp_s), 0.0, 1.0)
        out[out_mask] = value * alpha
    return out


def _parse_points(raw: Any) -> list[tuple[float, float]]:
    """Parse breakpoint points from JSON lane payload."""
    if not isinstance(raw, list):
        return []
    points: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, dict):
            try:
                t = float(item["time"])
                v = float(item["value"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Automation breakpoint dict requires numeric time and value."
                ) from exc
            if not np.isfinite(t) or t < 0.0:
                raise ValueError("Automation breakpoint time must be finite and >= 0.")
            if not np.isfinite(v):
                raise ValueError("Automation breakpoint value must be finite.")
            points.append((t, v))
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                t = float(item[0])
                v = float(item[1])
            except (TypeError, ValueError) as exc:
                raise ValueError("Automation breakpoint list items must be [time, value].") from exc
            if not np.isfinite(t) or t < 0.0:
                raise ValueError("Automation breakpoint time must be finite and >= 0.")
            if not np.isfinite(v):
                raise ValueError("Automation breakpoint value must be finite.")
            points.append((t, v))
            continue
        raise ValueError("Unsupported breakpoint point format.")
    return points


def _smoothstep_interpolate(
    x: npt.NDArray[np.float64],
    xp: npt.NDArray[np.float64],
    fp: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Piecewise smoothstep interpolation across keyframes."""
    out = np.empty_like(x, dtype=np.float64)
    out.fill(np.nan)
    if x.size == 0:
        return out

    lo_mask = x <= xp[0]
    hi_mask = x >= xp[-1]
    mid_mask = ~(lo_mask | hi_mask)
    out[lo_mask] = fp[0]
    out[hi_mask] = fp[-1]
    if not np.any(mid_mask):
        return out

    mid_x = x[mid_mask]
    hi_idx = np.searchsorted(xp, mid_x, side="right")
    hi_idx = np.clip(hi_idx, 1, xp.shape[0] - 1)
    lo_idx = hi_idx - 1
    x0 = xp[lo_idx]
    x1 = xp[hi_idx]
    y0 = fp[lo_idx]
    y1 = fp[hi_idx]
    t = np.clip((mid_x - x0) / np.maximum(1e-9, x1 - x0), 0.0, 1.0)
    s = (t * t) * (3.0 - (2.0 * t))
    out[mid_mask] = y0 + ((y1 - y0) * s)
    return out


def _exp_interpolate(
    x: npt.NDArray[np.float64],
    xp: npt.NDArray[np.float64],
    fp: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Piecewise exponential-ish interpolation for smoother ramps."""
    out = np.empty_like(x, dtype=np.float64)
    out.fill(np.nan)
    if x.size == 0:
        return out

    lo_mask = x <= xp[0]
    hi_mask = x >= xp[-1]
    mid_mask = ~(lo_mask | hi_mask)
    out[lo_mask] = fp[0]
    out[hi_mask] = fp[-1]
    if not np.any(mid_mask):
        return out

    mid_x = x[mid_mask]
    hi_idx = np.searchsorted(xp, mid_x, side="right")
    hi_idx = np.clip(hi_idx, 1, xp.shape[0] - 1)
    lo_idx = hi_idx - 1
    x0 = xp[lo_idx]
    x1 = xp[hi_idx]
    y0 = fp[lo_idx]
    y1 = fp[hi_idx]
    t = np.clip((mid_x - x0) / np.maximum(1e-9, x1 - x0), 0.0, 1.0)
    s_up = t * t
    s_down = 1.0 - ((1.0 - t) * (1.0 - t))
    s = np.where(y1 >= y0, s_up, s_down)
    out[mid_mask] = y0 + ((y1 - y0) * s)
    return out


def _apply_smoothing(
    values: npt.NDArray[np.float64],
    smoothing_ms: float,
    *,
    sr: int,
    step: int,
) -> npt.NDArray[np.float64]:
    """Apply one-pole smoothing only where lane values are defined."""
    if float(smoothing_ms) <= 0.0:
        return values
    out = values.copy()
    dt = float(step) / float(sr)
    tau = float(smoothing_ms / 1000.0)
    alpha = float(np.clip(np.exp(-dt / max(1e-9, tau)), 0.0, 1.0))

    valid = ~np.isnan(values)
    if np.count_nonzero(valid) <= 1:
        return out

    first = int(np.flatnonzero(valid)[0])
    state = float(values[first])
    out[first] = state
    for idx in range(first + 1, out.shape[0]):
        if np.isnan(values[idx]):
            continue
        state = (alpha * state) + ((1.0 - alpha) * float(values[idx]))
        out[idx] = state
    return out


def _expand_control_to_samples(
    control: npt.NDArray[np.float64],
    *,
    step: int,
    num_samples: int,
) -> npt.NDArray[np.float64]:
    """Expand control-rate sequence to sample-rate sequence."""
    if int(num_samples) <= 0:
        return np.zeros((0,), dtype=np.float64)
    expanded = np.repeat(control, max(1, int(step)))
    if expanded.shape[0] < int(num_samples):
        pad = np.full(int(num_samples) - expanded.shape[0], float(control[-1]), dtype=np.float64)
        expanded = np.concatenate((expanded, pad), axis=0)
    return np.asarray(expanded[: int(num_samples)], dtype=np.float64)


def _target_stats(curves: dict[str, AudioArray]) -> dict[str, dict[str, float]]:
    """Return compact per-target stats for render reports."""
    stats: dict[str, dict[str, float]] = {}
    for target, values in curves.items():
        vec = np.asarray(values, dtype=np.float64)
        if vec.size == 0:
            stats[target] = {"min": 0.0, "max": 0.0, "mean": 0.0}
            continue
        stats[target] = {
            "min": float(np.min(vec)),
            "max": float(np.max(vec)),
            "mean": float(np.mean(vec)),
        }
    return stats


def _collect_feature_lane_sources(lanes: list[Any]) -> set[str]:
    """Collect normalized feature sources referenced by feature-vector lanes."""
    sources: set[str] = set()
    for lane_idx, lane_obj in enumerate(lanes, start=1):
        if not isinstance(lane_obj, dict):
            continue
        if not is_feature_vector_lane(lane_obj):
            continue
        normalized_lane = normalize_feature_vector_lane(
            lane_obj,
            lane_context=f"lane #{lane_idx}",
        )
        sources.add(normalize_feature_source_name(str(normalized_lane["source"])))
    return sources


def _normalize_lane_combine(raw: Any, *, lane_context: str) -> str:
    combine = str(raw).strip().lower()
    if combine == "":
        return "replace"
    if combine in FEATURE_LANE_COMBINE_CHOICES:
        return combine
    if combine not in LANE_COMBINE_CHOICES:
        choices = ", ".join(sorted(LANE_COMBINE_CHOICES))
        raise ValueError(
            f"Invalid combine mode '{combine}' for {lane_context}. Supported: {choices}."
        )
    return combine


def _parse_lane_smoothing_ms(raw: Any, *, lane_context: str) -> float:
    try:
        smoothing_ms = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid smoothing_ms for {lane_context}; expected numeric value."
        ) from exc
    if not np.isfinite(smoothing_ms):
        raise ValueError(f"smoothing_ms for {lane_context} must be finite.")
    if smoothing_ms < 0.0:
        raise ValueError(f"smoothing_ms for {lane_context} must be >= 0.")
    return float(smoothing_ms)


def _bundle_signature(
    *,
    curves: dict[str, AudioArray],
    mode: str,
    control_step: int,
    sample_rate: int,
    num_samples: int,
) -> str:
    """Build stable digest for deterministic automation replay QA."""
    h = hashlib.sha256()
    h.update(str(mode).encode("utf-8"))
    h.update(b"|")
    h.update(str(int(control_step)).encode("utf-8"))
    h.update(b"|")
    h.update(str(int(sample_rate)).encode("utf-8"))
    h.update(b"|")
    h.update(str(int(num_samples)).encode("utf-8"))
    for target in sorted(curves.keys()):
        h.update(b"|")
        h.update(target.encode("utf-8"))
        vec = np.asarray(curves[target], dtype=np.float64)
        h.update(vec.tobytes(order="C"))
    return h.hexdigest()[:16]


def _normalize_target_name(value: str) -> str:
    """Normalize automation target name and aliases."""
    return normalize_control_target_name(value)
