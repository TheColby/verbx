"""Immersive interoperability helpers for v0.7 production workflows.

This module provides:

- object/bed policy checks used for Atmos-adjacent handoff preparation,
- ADM/BWF sidecar + deliverable manifest generation,
- reusable immersive QC gates (loudness, true-peak, fold-down, occupancy),
- file-backed distributed queue worker primitives with heartbeats and retries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import numpy.typing as npt
import soundfile as sf

from verbx.core.loudness import integrated_lufs, true_peak_dbfs
from verbx.io.audio import read_audio

AudioArray = npt.NDArray[np.float64]

LAYOUT_CHANNELS: dict[str, int] = {
    "mono": 1,
    "stereo": 2,
    "lcr": 3,
    "5.1": 6,
    "7.1": 8,
    "7.1.2": 10,
    "7.1.4": 12,
    "7.2.4": 13,
    "8.0": 8,
    "16.0": 16,
    "64.4": 68,
}

LAYOUT_CHANNEL_LABELS: dict[str, tuple[str, ...]] = {
    "mono": ("M",),
    "stereo": ("L", "R"),
    "lcr": ("L", "R", "C"),
    "5.1": ("L", "R", "C", "LFE", "Ls", "Rs"),
    "7.1": ("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs"),
    "7.1.2": ("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs", "Ltf", "Rtf"),
    "7.1.4": ("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs", "Ltf", "Rtf", "Ltr", "Rtr"),
    "7.2.4": ("L", "R", "C", "LFE1", "LFE2", "Ls", "Rs", "Lrs", "Rrs", "Ltf", "Rtf", "Ltr", "Rtr"),
    "8.0": ("L", "R", "C", "Ls", "Rs", "Lrs", "Rrs", "Cs"),
}

POLICY_MODES = {"bed-safe", "object-safe", "balanced"}
LAYOUT_HINT_CHOICES = frozenset({"auto", *LAYOUT_CHANNELS.keys()})


@dataclass(slots=True)
class ImmersiveQCGates:
    """Thresholds used by immersive QC checks."""

    target_lufs: float = -18.0
    lufs_tolerance: float = 3.0
    max_true_peak_dbfs: float = -1.0
    max_fold_down_delta_db: float = 4.0
    min_channel_occupancy: float = 0.34
    occupancy_threshold_dbfs: float = -45.0


@dataclass(slots=True)
class QueueWorkerConfig:
    """Runtime options for file-backed distributed queue workers."""

    worker_id: str
    heartbeat_dir: Path
    poll_ms: int = 800
    max_jobs: int = 0
    stale_claim_seconds: float = 120.0
    continue_on_error: bool = True


class QueueJobClaim(TypedDict):
    """Normalized claimed queue job."""

    id: str
    key: str
    infile: str
    outfile: str
    options: dict[str, Any]
    max_retries: int
    claim_path: str


QueueRunner = Callable[[QueueJobClaim], None]


def normalize_layout_name(layout: str) -> str:
    """Return canonical lowercase layout identifier."""
    value = str(layout).strip().lower().replace("_", ".")
    if value in {"l.c.r", "l-c-r"}:
        return "lcr"
    return value


def validate_layout_hint(layout: str) -> str:
    """Validate user-facing layout hint and return canonical identifier."""
    normalized = normalize_layout_name(layout)
    if normalized in LAYOUT_HINT_CHOICES:
        return normalized
    options = ", ".join(sorted(LAYOUT_HINT_CHOICES))
    raise ValueError(f"layout must be one of: {options}")


def infer_layout_from_channels(channels: int) -> str:
    """Infer immersive layout name from channel count when possible."""
    for name, count in LAYOUT_CHANNELS.items():
        if int(channels) == int(count):
            return name
    return f"{int(channels)}ch"


def channel_labels_for_layout(layout: str, channels: int) -> list[str]:
    """Return channel labels for a layout/channel-count combination."""
    normalized = normalize_layout_name(layout)
    labels = LAYOUT_CHANNEL_LABELS.get(normalized)
    if labels is not None and len(labels) == int(channels):
        return list(labels)
    return [f"ch{idx + 1}" for idx in range(int(channels))]


def fold_down_to_stereo(audio: AudioArray, layout: str = "auto") -> AudioArray:
    """Fold a multichannel signal to stereo using practical bed mappings."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 2:
        msg = f"audio must be 2D (samples, channels), got shape {x.shape!r}"
        raise ValueError(msg)
    channels = int(x.shape[1])
    if channels <= 0:
        msg = "audio must contain at least one channel."
        raise ValueError(msg)
    if channels == 1:
        return np.repeat(x, 2, axis=1)
    if channels == 2:
        return np.asarray(x.copy(), dtype=np.float64)

    inferred = infer_layout_from_channels(channels)
    normalized_layout = normalize_layout_name(layout)
    if normalized_layout in {"", "auto"}:
        normalized_layout = inferred

    if (
        normalized_layout in {"lcr", "5.1", "7.1", "7.1.2", "7.1.4", "7.2.4", "8.0"}
        and channels >= 3
    ):
        if normalized_layout == "7.2.4" and channels >= 13:
            # 7.2.4 assumption:
            # L(0), R(1), C(2), LFE1(3), LFE2(4), Ls(5), Rs(6), Lrs(7), Rrs(8),
            # Ltf(9), Rtf(10), Ltr(11), Rtr(12)
            left = np.asarray(x[:, 0], dtype=np.float64).copy()
            right = np.asarray(x[:, 1], dtype=np.float64).copy()
            center = np.asarray(x[:, 2], dtype=np.float64) * np.float64(0.7071)
            left += center
            right += center
            left += np.asarray(x[:, 5], dtype=np.float64) * np.float64(0.7071)
            right += np.asarray(x[:, 6], dtype=np.float64) * np.float64(0.7071)
            left += np.asarray(x[:, 7], dtype=np.float64) * np.float64(0.5)
            right += np.asarray(x[:, 8], dtype=np.float64) * np.float64(0.5)
            left += np.asarray(x[:, 9], dtype=np.float64) * np.float64(0.5)
            right += np.asarray(x[:, 10], dtype=np.float64) * np.float64(0.5)
            left += np.asarray(x[:, 11], dtype=np.float64) * np.float64(0.5)
            right += np.asarray(x[:, 12], dtype=np.float64) * np.float64(0.5)
            return np.asarray(np.column_stack((left, right)), dtype=np.float64)
        if normalized_layout == "8.0" and channels >= 8:
            # 8.0 assumption:
            # L(0), R(1), C(2), Ls(3), Rs(4), Lrs(5), Rrs(6), Cs(7)
            left = np.asarray(x[:, 0], dtype=np.float64).copy()
            right = np.asarray(x[:, 1], dtype=np.float64).copy()
            center = np.asarray(x[:, 2], dtype=np.float64) * np.float64(0.7071)
            left += center
            right += center
            left += np.asarray(x[:, 3], dtype=np.float64) * np.float64(0.7071)
            right += np.asarray(x[:, 4], dtype=np.float64) * np.float64(0.7071)
            left += np.asarray(x[:, 5], dtype=np.float64) * np.float64(0.5)
            right += np.asarray(x[:, 6], dtype=np.float64) * np.float64(0.5)
            rear_center = np.asarray(x[:, 7], dtype=np.float64) * np.float64(0.5)
            left += rear_center
            right += rear_center
            return np.asarray(np.column_stack((left, right)), dtype=np.float64)
        left = np.asarray(x[:, 0], dtype=np.float64).copy()
        right = np.asarray(x[:, 1], dtype=np.float64).copy()
        center = np.asarray(x[:, 2], dtype=np.float64) * np.float64(0.7071)
        left += center
        right += center
        if channels >= 6:
            left += np.asarray(x[:, 4], dtype=np.float64) * np.float64(0.7071)
            right += np.asarray(x[:, 5], dtype=np.float64) * np.float64(0.7071)
        if channels >= 8:
            left += np.asarray(x[:, 6], dtype=np.float64) * np.float64(0.5)
            right += np.asarray(x[:, 7], dtype=np.float64) * np.float64(0.5)
        if channels >= 10:
            left += np.asarray(x[:, 8], dtype=np.float64) * np.float64(0.5)
            right += np.asarray(x[:, 9], dtype=np.float64) * np.float64(0.5)
        if channels >= 12:
            left += np.asarray(x[:, 10], dtype=np.float64) * np.float64(0.5)
            right += np.asarray(x[:, 11], dtype=np.float64) * np.float64(0.5)
        return np.asarray(np.column_stack((left, right)), dtype=np.float64)

    left = np.mean(x[:, 0::2], axis=1, dtype=np.float64)
    if channels >= 2:
        right = np.mean(x[:, 1::2], axis=1, dtype=np.float64)
    else:
        right = left.copy()
    return np.asarray(np.column_stack((left, right)), dtype=np.float64)


def build_qc_gates(payload: dict[str, Any] | None) -> ImmersiveQCGates:
    """Build QC gate thresholds from optional scene payload."""
    if payload is None:
        return ImmersiveQCGates()
    gates = ImmersiveQCGates()
    for field in (
        "target_lufs",
        "lufs_tolerance",
        "max_true_peak_dbfs",
        "max_fold_down_delta_db",
        "min_channel_occupancy",
        "occupancy_threshold_dbfs",
    ):
        value = payload.get(field)
        if isinstance(value, (float, int)):
            setattr(gates, field, float(value))
    gates.lufs_tolerance = max(0.0, float(gates.lufs_tolerance))
    gates.min_channel_occupancy = float(np.clip(gates.min_channel_occupancy, 0.0, 1.0))
    return gates


def evaluate_immersive_qc(
    *,
    audio: AudioArray,
    sr: int,
    label: str,
    layout: str = "auto",
    gates: ImmersiveQCGates | None = None,
) -> dict[str, Any]:
    """Evaluate immersive production QC gates and return structured metrics."""
    qc_gates = ImmersiveQCGates() if gates is None else gates
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 2:
        msg = f"audio must be 2D (samples, channels), got {x.shape!r}"
        raise ValueError(msg)
    if int(sr) <= 0:
        msg = "sample rate must be > 0"
        raise ValueError(msg)

    inferred_layout = infer_layout_from_channels(int(x.shape[1]))
    resolved_layout = normalize_layout_name(layout)
    if resolved_layout in {"", "auto"}:
        resolved_layout = inferred_layout

    channel_peaks = np.max(np.abs(x), axis=0)
    channel_peaks_dbfs = [float(20.0 * np.log10(max(1e-12, float(p)))) for p in channel_peaks]
    channel_rms = np.sqrt(np.mean(np.square(x), axis=0, dtype=np.float64))
    channel_rms_dbfs = [float(20.0 * np.log10(max(1e-12, float(v)))) for v in channel_rms]

    max_channel_peak_dbfs = max(channel_peaks_dbfs)
    stereo_fold = fold_down_to_stereo(x, layout=resolved_layout)
    fold_peak = float(np.max(np.abs(stereo_fold)))
    fold_peak_dbfs = float(20.0 * np.log10(max(1e-12, fold_peak)))
    fold_down_delta_db = float(fold_peak_dbfs - max_channel_peak_dbfs)

    active_channels = sum(
        1 for value in channel_rms_dbfs if value >= float(qc_gates.occupancy_threshold_dbfs)
    )
    channel_occupancy = float(active_channels / max(1, int(x.shape[1])))

    measured_lufs = float(integrated_lufs(x, int(sr)))
    measured_true_peak = float(true_peak_dbfs(x, int(sr), oversample=4))

    loudness_error = float(abs(measured_lufs - float(qc_gates.target_lufs)))
    passes = {
        "loudness": loudness_error <= float(qc_gates.lufs_tolerance),
        "true_peak": measured_true_peak <= float(qc_gates.max_true_peak_dbfs),
        "fold_down_delta": abs(fold_down_delta_db) <= float(qc_gates.max_fold_down_delta_db),
        "channel_occupancy": channel_occupancy >= float(qc_gates.min_channel_occupancy),
    }
    failed = [name for name, state in passes.items() if not state]

    metrics = {
        "integrated_lufs": measured_lufs,
        "target_lufs": float(qc_gates.target_lufs),
        "lufs_error": loudness_error,
        "true_peak_dbfs": measured_true_peak,
        "max_true_peak_dbfs": float(qc_gates.max_true_peak_dbfs),
        "fold_down_delta_db": fold_down_delta_db,
        "max_fold_down_delta_db": float(qc_gates.max_fold_down_delta_db),
        "channel_occupancy": channel_occupancy,
        "min_channel_occupancy": float(qc_gates.min_channel_occupancy),
        "occupancy_threshold_dbfs": float(qc_gates.occupancy_threshold_dbfs),
        "max_channel_peak_dbfs": max_channel_peak_dbfs,
        "fold_down_peak_dbfs": fold_peak_dbfs,
        "channel_peaks_dbfs": channel_peaks_dbfs,
        "channel_rms_dbfs": channel_rms_dbfs,
        "channel_labels": channel_labels_for_layout(resolved_layout, int(x.shape[1])),
    }
    return {
        "label": label,
        "layout": resolved_layout,
        "inferred_layout": inferred_layout,
        "sample_rate": int(sr),
        "channels": int(x.shape[1]),
        "passes": passes,
        "failed_gates": failed,
        "pass": len(failed) == 0,
        "metrics": metrics,
    }


def evaluate_object_bed_policy(
    *,
    scene: dict[str, Any],
    bed_qc: dict[str, Any],
    object_qc: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate object/bed policy constraints and downmix-risk checks."""
    policy_raw = scene.get("policy")
    policy = policy_raw if isinstance(policy_raw, dict) else {}
    mode = str(policy.get("mode", "balanced")).strip().lower()
    if mode not in POLICY_MODES:
        mode = "balanced"

    defaults = {
        "bed-safe": {"max_bed_wet": 0.85, "max_object_wet": 0.60, "max_object_rt60": 12.0},
        "object-safe": {"max_bed_wet": 0.75, "max_object_wet": 0.50, "max_object_rt60": 8.0},
        "balanced": {"max_bed_wet": 0.80, "max_object_wet": 0.65, "max_object_rt60": 10.0},
    }[mode]
    max_bed_wet = float(policy.get("max_bed_wet", defaults["max_bed_wet"]))
    max_object_wet = float(policy.get("max_object_wet", defaults["max_object_wet"]))
    max_object_rt60 = float(policy.get("max_object_rt60", defaults["max_object_rt60"]))
    max_downmix_delta = float(policy.get("downmix_max_delta_db", 4.0))

    errors: list[str] = []
    warnings: list[str] = []

    bed_cfg_raw = scene.get("bed")
    bed_cfg = bed_cfg_raw if isinstance(bed_cfg_raw, dict) else {}
    bed_opts_raw = bed_cfg.get("render_options")
    bed_opts = bed_opts_raw if isinstance(bed_opts_raw, dict) else {}
    if "wet" in bed_opts and isinstance(bed_opts["wet"], (float, int)):
        bed_wet = float(bed_opts["wet"])
        if bed_wet > max_bed_wet:
            errors.append(
                f"bed wet={bed_wet:.3f} exceeds policy max_bed_wet={max_bed_wet:.3f} ({mode})."
            )
    else:
        warnings.append("bed.render_options.wet not provided; policy cannot verify bed wet cap.")

    bed_metrics_raw = bed_qc.get("metrics", {})
    bed_metrics = bed_metrics_raw if isinstance(bed_metrics_raw, dict) else {}
    bed_delta = float(bed_metrics.get("fold_down_delta_db", 0.0))
    if abs(bed_delta) > max_downmix_delta:
        errors.append(
            "bed fold-down delta "
            f"{bed_delta:.2f} dB exceeds policy downmix_max_delta_db={max_downmix_delta:.2f}."
        )

    objects_raw = scene.get("objects")
    objects = objects_raw if isinstance(objects_raw, list) else []
    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            warnings.append(f"objects[{idx}] is not an object; skipping policy checks.")
            continue
        obj_name = str(obj.get("name", f"object_{idx + 1}"))
        opts_raw = obj.get("render_options")
        opts = opts_raw if isinstance(opts_raw, dict) else {}
        wet_value = opts.get("wet")
        rt60_value = opts.get("rt60")
        if isinstance(wet_value, (float, int)):
            wet = float(wet_value)
            if wet > max_object_wet:
                errors.append(
                    f"{obj_name}: wet={wet:.3f} exceeds policy max_object_wet={max_object_wet:.3f}."
                )
        else:
            warnings.append(f"{obj_name}: render_options.wet not provided.")
        if isinstance(rt60_value, (float, int)):
            rt60 = float(rt60_value)
            if rt60 > max_object_rt60:
                warnings.append(
                    f"{obj_name}: rt60={rt60:.2f}s exceeds "
                    f"policy max_object_rt60={max_object_rt60:.2f}s."
                )

    for qc in object_qc:
        metrics_raw = qc.get("metrics")
        metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
        name = str(qc.get("label", "object"))
        delta = float(metrics.get("fold_down_delta_db", 0.0))
        if abs(delta) > max_downmix_delta:
            errors.append(
                f"{name}: fold-down delta {delta:.2f} dB exceeds "
                f"downmix_max_delta_db={max_downmix_delta:.2f}."
            )

    return {
        "mode": mode,
        "thresholds": {
            "max_bed_wet": max_bed_wet,
            "max_object_wet": max_object_wet,
            "max_object_rt60": max_object_rt60,
            "downmix_max_delta_db": max_downmix_delta,
        },
        "errors": errors,
        "warnings": warnings,
    }


def generate_immersive_handoff_package(
    *,
    scene: dict[str, Any],
    out_dir: Path,
    strict: bool = True,
) -> dict[str, Any]:
    """Generate ADM-sidecar + deliverable manifests + QC bundle."""
    scene_name = _slug(str(scene.get("scene_name", "immersive_scene")))
    bed_raw = scene.get("bed")
    if not isinstance(bed_raw, dict):
        raise ValueError("scene.bed must be an object with path/layout metadata.")
    bed_path = Path(str(bed_raw.get("path", "")))
    if str(bed_path).strip() == "":
        raise ValueError("scene.bed.path is required.")
    if not bed_path.exists():
        raise ValueError(f"bed file not found: {bed_path}")

    bed_layout = validate_layout_hint(str(bed_raw.get("layout", "auto")))
    bed_audio, bed_sr = read_audio(str(bed_path))
    if bed_layout not in {"auto", ""} and bed_layout in LAYOUT_CHANNELS:
        expected = int(LAYOUT_CHANNELS[bed_layout])
        if int(bed_audio.shape[1]) != expected:
            raise ValueError(
                f"bed layout '{bed_layout}' expects {expected} channels, "
                f"got {int(bed_audio.shape[1])}."
            )
    declared_sample_rate_raw = scene.get("sample_rate")
    if declared_sample_rate_raw is None:
        declared_sample_rate = int(bed_sr)
    elif isinstance(declared_sample_rate_raw, (int, float)):
        declared_sample_rate = int(declared_sample_rate_raw)
        if declared_sample_rate <= 0:
            raise ValueError("scene.sample_rate must be > 0 when provided.")
    else:
        raise ValueError("scene.sample_rate must be numeric when provided.")

    validation_errors: list[str] = []
    validation_warnings: list[str] = []
    if int(bed_sr) != int(declared_sample_rate):
        validation_errors.append(
            "bed sample rate does not match declared scene.sample_rate "
            f"({bed_sr} != {declared_sample_rate})."
        )

    gates_raw = scene.get("qc_gates")
    gates = build_qc_gates(gates_raw if isinstance(gates_raw, dict) else None)
    bed_qc = evaluate_immersive_qc(
        audio=bed_audio,
        sr=bed_sr,
        label=str(bed_raw.get("name", "bed")),
        layout=bed_layout,
        gates=gates,
    )

    objects_raw = scene.get("objects")
    objects = objects_raw if isinstance(objects_raw, list) else []
    object_entries: list[dict[str, Any]] = []
    object_qc: list[dict[str, Any]] = []
    ids_seen: set[str] = set()
    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError(f"objects[{idx}] must be an object.")
        obj_id = str(obj.get("id", f"obj_{idx + 1:03d}")).strip()
        if obj_id == "":
            obj_id = f"obj_{idx + 1:03d}"
        if obj_id in ids_seen:
            raise ValueError(f"Duplicate object id: {obj_id}")
        ids_seen.add(obj_id)

        obj_path = Path(str(obj.get("path", "")))
        if str(obj_path).strip() == "":
            raise ValueError(f"objects[{idx}].path is required.")
        if not obj_path.exists():
            raise ValueError(f"object file not found: {obj_path}")
        obj_audio, obj_sr = read_audio(str(obj_path))
        obj_label = str(obj.get("name", obj_id))
        qc = evaluate_immersive_qc(
            audio=obj_audio,
            sr=obj_sr,
            label=obj_label,
            layout=validate_layout_hint(str(obj.get("layout", "auto"))),
            gates=gates,
        )
        object_qc.append(qc)
        if int(obj_sr) != int(declared_sample_rate):
            validation_errors.append(
                f"{obj_label}: sample rate {obj_sr} does not match declared scene.sample_rate "
                f"{declared_sample_rate}."
            )

        info = sf.info(str(obj_path))
        object_entries.append(
            {
                "id": obj_id,
                "name": obj_label,
                "path": str(obj_path),
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "duration_seconds": (
                    float(info.frames) / float(info.samplerate) if info.samplerate > 0 else 0.0
                ),
                "start_s": float(obj.get("start_s", 0.0)),
                "gain_db": float(obj.get("gain_db", 0.0)),
                "position": {
                    "x": float(obj.get("x", 0.0)),
                    "y": float(obj.get("y", 0.0)),
                    "z": float(obj.get("z", 0.0)),
                },
                "render_options": obj.get("render_options", {}),
            }
        )

    policy_summary = evaluate_object_bed_policy(scene=scene, bed_qc=bed_qc, object_qc=object_qc)
    qa_all_pass = bool(bed_qc["pass"]) and all(bool(item["pass"]) for item in object_qc)
    if strict:
        strict_errors: list[str] = []
        strict_errors.extend(str(item) for item in policy_summary["errors"])
        strict_errors.extend(validation_errors)
        if not qa_all_pass:
            strict_errors.append("one or more QC gates failed.")
        if len(strict_errors) > 0:
            joined = "; ".join(strict_errors)
            raise ValueError(f"immersive strict checks failed: {joined}")

    out_dir.mkdir(parents=True, exist_ok=True)
    object_manifest = {
        "version": "0.7",
        "scene_name": str(scene.get("scene_name", scene_name)),
        "objects": object_entries,
    }
    qa_bundle = {
        "version": "0.7",
        "scene_name": str(scene.get("scene_name", scene_name)),
        "gates": {
            "target_lufs": gates.target_lufs,
            "lufs_tolerance": gates.lufs_tolerance,
            "max_true_peak_dbfs": gates.max_true_peak_dbfs,
            "max_fold_down_delta_db": gates.max_fold_down_delta_db,
            "min_channel_occupancy": gates.min_channel_occupancy,
            "occupancy_threshold_dbfs": gates.occupancy_threshold_dbfs,
        },
        "bed": bed_qc,
        "objects": object_qc,
        "policy": policy_summary,
        "validation": {
            "errors": validation_errors,
            "warnings": validation_warnings,
        },
        "summary": {
            "all_pass": qa_all_pass,
            "failed_tracks": [
                str(item.get("label", "unknown"))
                for item in [bed_qc, *object_qc]
                if not item["pass"]
            ],
        },
    }
    adm_sidecar = {
        "schema": "verbx.adm-bwf.sidecar.v0.7",
        "generated_utc": _iso_utc_now(),
        "scene_name": str(scene.get("scene_name", scene_name)),
        "sample_rate": int(declared_sample_rate),
        "bed": {
            "name": str(bed_raw.get("name", "bed")),
            "path": str(bed_path),
            "layout": bed_qc["layout"],
            "channels": int(bed_audio.shape[1]),
            "channel_labels": bed_qc["metrics"]["channel_labels"],
        },
        "objects": [
            {
                "id": entry["id"],
                "name": entry["name"],
                "path": entry["path"],
                "start_s": entry["start_s"],
                "gain_db": entry["gain_db"],
                "position": entry["position"],
            }
            for entry in object_entries
        ],
        "policy": policy_summary,
        "validation": qa_bundle["validation"],
        "qa_summary": qa_bundle["summary"],
    }

    deliverables_raw = scene.get("deliverables")
    deliverables_cfg = deliverables_raw if isinstance(deliverables_raw, dict) else {}
    write_adm = bool(deliverables_cfg.get("adm_sidecar", True))
    write_objects = bool(deliverables_cfg.get("object_stem_manifest", True))
    write_qa = bool(deliverables_cfg.get("qa_bundle", True))

    outputs: dict[str, str] = {}
    if write_adm:
        adm_path = out_dir / f"{scene_name}.adm-bwf.sidecar.json"
        _write_json(adm_path, adm_sidecar)
        outputs["adm_sidecar"] = str(adm_path)
    if write_objects:
        objects_path = out_dir / f"{scene_name}.object-stems.manifest.json"
        _write_json(objects_path, object_manifest)
        outputs["object_stem_manifest"] = str(objects_path)
    if write_qa:
        qa_path = out_dir / f"{scene_name}.qa.bundle.json"
        _write_json(qa_path, qa_bundle)
        outputs["qa_bundle"] = str(qa_path)

    deliverable_manifest = {
        "version": "0.7",
        "scene_name": str(scene.get("scene_name", scene_name)),
        "generated_utc": _iso_utc_now(),
        "scope_boundary": (
            "verbx emits prep metadata and QC sidecars; Dolby encoding/rendering remains external."
        ),
        "beds": [{"name": str(bed_raw.get("name", "bed")), "path": str(bed_path)}],
        "object_stems": [{"id": entry["id"], "path": entry["path"]} for entry in object_entries],
        "outputs": outputs,
        "policy": policy_summary,
        "validation": qa_bundle["validation"],
    }
    manifest_path = out_dir / f"{scene_name}.deliverables.manifest.json"
    _write_json(manifest_path, deliverable_manifest)
    outputs["deliverables_manifest"] = str(manifest_path)

    return {
        "scene_name": str(scene.get("scene_name", scene_name)),
        "strict": bool(strict),
        "outputs": outputs,
        "policy": policy_summary,
        "validation": qa_bundle["validation"],
        "qa_summary": qa_bundle["summary"],
    }


def run_file_queue_worker(
    *,
    queue_path: Path,
    runner: QueueRunner,
    config: QueueWorkerConfig,
) -> dict[str, Any]:
    """Run one file-backed distributed worker with heartbeat + retry semantics."""
    if not queue_path.exists():
        raise FileNotFoundError(f"queue file not found: {queue_path}")
    state_root = queue_path.parent / f".{queue_path.stem}.queue_state"
    claims_dir = state_root / "claims"
    results_dir = state_root / "results"
    attempts_dir = state_root / "attempts"
    for path in (state_root, claims_dir, results_dir, attempts_dir):
        path.mkdir(parents=True, exist_ok=True)
    config.heartbeat_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "worker_id": config.worker_id,
        "queue_path": str(queue_path),
        "state_root": str(state_root),
        "processed": 0,
        "success": 0,
        "failed": 0,
        "retried": 0,
        "skipped_done": 0,
    }
    idle_rounds = 0
    last_error: str | None = None

    while True:
        _write_heartbeat(config=config, state="polling", summary=summary)
        payload = _load_queue_payload(queue_path)
        jobs = _normalize_queue_jobs(payload.get("jobs"))
        claim = _claim_next_job(
            jobs=jobs,
            claims_dir=claims_dir,
            results_dir=results_dir,
            attempts_dir=attempts_dir,
            stale_claim_seconds=float(config.stale_claim_seconds),
            worker_id=config.worker_id,
        )

        if claim is None:
            idle_rounds += 1
            if idle_rounds >= 2:
                break
            time.sleep(max(0.05, float(config.poll_ms) / 1000.0))
            continue

        idle_rounds = 0
        summary["processed"] = int(summary["processed"]) + 1
        claim_path = Path(claim["claim_path"])
        attempt = _increment_attempt(attempts_dir=attempts_dir, job_id=claim["id"])
        _write_heartbeat(
            config=config,
            state="running",
            summary=summary,
            current_job_id=claim["id"],
            current_attempt=attempt,
        )
        started = time.perf_counter()

        try:
            runner(claim)
            duration = float(time.perf_counter() - started)
            _write_json(
                results_dir / f"{claim['key']}.success.json",
                {
                    "job_id": claim["id"],
                    "key": claim["key"],
                    "attempt": attempt,
                    "duration_seconds": duration,
                    "worker_id": config.worker_id,
                    "finished_utc": _iso_utc_now(),
                },
            )
            summary["success"] = int(summary["success"]) + 1
        except Exception as exc:
            duration = float(time.perf_counter() - started)
            last_error = str(exc)
            retriable = attempt <= int(claim["max_retries"])
            if retriable:
                summary["retried"] = int(summary["retried"]) + 1
                _write_json(
                    results_dir / f"{claim['key']}.retry.json",
                    {
                        "job_id": claim["id"],
                        "key": claim["key"],
                        "attempt": attempt,
                        "error": last_error,
                        "duration_seconds": duration,
                        "worker_id": config.worker_id,
                        "finished_utc": _iso_utc_now(),
                    },
                )
            else:
                summary["failed"] = int(summary["failed"]) + 1
                _write_json(
                    results_dir / f"{claim['key']}.failed.json",
                    {
                        "job_id": claim["id"],
                        "key": claim["key"],
                        "attempt": attempt,
                        "error": last_error,
                        "duration_seconds": duration,
                        "worker_id": config.worker_id,
                        "finished_utc": _iso_utc_now(),
                    },
                )
                if not config.continue_on_error:
                    _safe_unlink(claim_path)
                    raise RuntimeError(
                        f"Queue job {claim['id']} failed (attempt={attempt}): {last_error}"
                    ) from exc
        finally:
            _safe_unlink(claim_path)

        if int(config.max_jobs) > 0 and int(summary["processed"]) >= int(config.max_jobs):
            break

    summary["last_error"] = last_error
    _write_heartbeat(config=config, state="idle", summary=summary)
    return summary


def summarize_file_queue(queue_path: Path) -> dict[str, Any]:
    """Return queue status summary for automation/monitoring use."""
    payload = _load_queue_payload(queue_path)
    jobs = _normalize_queue_jobs(payload.get("jobs"))
    state_root = queue_path.parent / f".{queue_path.stem}.queue_state"
    claims_dir = state_root / "claims"
    results_dir = state_root / "results"
    attempts_dir = state_root / "attempts"
    claims_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir.mkdir(parents=True, exist_ok=True)

    total = len(jobs)
    success = 0
    failed = 0
    claimed = 0
    pending = 0
    for job in jobs:
        key = _queue_job_key(job)
        if (results_dir / f"{key}.success.json").exists():
            success += 1
            continue
        if (results_dir / f"{key}.failed.json").exists():
            failed += 1
            continue
        if (claims_dir / f"{job['id']}.claim.json").exists():
            claimed += 1
            continue
        pending += 1

    return {
        "queue_path": str(queue_path),
        "state_root": str(state_root),
        "total_jobs": total,
        "success_jobs": success,
        "failed_jobs": failed,
        "claimed_jobs": claimed,
        "pending_jobs": pending,
    }


def _normalize_queue_jobs(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    jobs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        infile = str(item.get("infile", "")).strip()
        outfile = str(item.get("outfile", "")).strip()
        if infile == "" or outfile == "":
            continue
        options_raw = item.get("options")
        options = options_raw if isinstance(options_raw, dict) else {}
        job_id = str(item.get("id", f"job_{idx + 1:04d}")).strip()
        if job_id == "":
            job_id = f"job_{idx + 1:04d}"
        if job_id in seen_ids:
            raise ValueError(f"Duplicate queue job id: {job_id}")
        seen_ids.add(job_id)
        max_retries_raw = item.get("max_retries", item.get("retries", 0))
        max_retries = int(max_retries_raw) if isinstance(max_retries_raw, (int, float)) else 0
        jobs.append(
            {
                "id": job_id,
                "infile": infile,
                "outfile": outfile,
                "options": options,
                "max_retries": max(0, max_retries),
            }
        )
    return jobs


def _claim_next_job(
    *,
    jobs: list[dict[str, Any]],
    claims_dir: Path,
    results_dir: Path,
    attempts_dir: Path,
    stale_claim_seconds: float,
    worker_id: str,
) -> QueueJobClaim | None:
    for job in jobs:
        key = _queue_job_key(job)
        job_id = str(job["id"])
        max_retries = int(job.get("max_retries", 0))

        if (results_dir / f"{key}.success.json").exists():
            continue
        attempts = _read_attempts(attempts_dir=attempts_dir, job_id=job_id)
        if attempts > max_retries and (results_dir / f"{key}.failed.json").exists():
            continue

        claim_path = claims_dir / f"{job_id}.claim.json"
        if claim_path.exists():
            age = time.time() - float(claim_path.stat().st_mtime)
            if age > stale_claim_seconds:
                _safe_unlink(claim_path)
            else:
                continue

        payload = {
            "job_id": job_id,
            "key": key,
            "worker_id": worker_id,
            "claimed_utc": _iso_utc_now(),
        }
        if not _atomic_create_json(claim_path, payload):
            continue

        return {
            "id": job_id,
            "key": key,
            "infile": str(job["infile"]),
            "outfile": str(job["outfile"]),
            "options": cast(dict[str, Any], job.get("options", {})),
            "max_retries": max_retries,
            "claim_path": str(claim_path),
        }
    return None


def _queue_job_key(job: dict[str, Any]) -> str:
    payload = {
        "infile": str(job.get("infile", "")),
        "outfile": str(job.get("outfile", "")),
        "options": cast(dict[str, Any], job.get("options", {})),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _read_attempts(*, attempts_dir: Path, job_id: str) -> int:
    path = attempts_dir / f"{job_id}.json"
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, dict):
        return 0
    count = payload.get("attempts", 0)
    return int(count) if isinstance(count, (int, float)) else 0


def _increment_attempt(*, attempts_dir: Path, job_id: str) -> int:
    count = _read_attempts(attempts_dir=attempts_dir, job_id=job_id) + 1
    _write_json(
        attempts_dir / f"{job_id}.json",
        {"job_id": job_id, "attempts": count, "updated_utc": _iso_utc_now()},
    )
    return count


def _load_queue_payload(queue_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(queue_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid queue JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Queue file must be a JSON object.")
    return payload


def _write_heartbeat(
    *,
    config: QueueWorkerConfig,
    state: str,
    summary: dict[str, Any],
    current_job_id: str | None = None,
    current_attempt: int | None = None,
) -> None:
    heartbeat = {
        "worker_id": config.worker_id,
        "state": state,
        "updated_utc": _iso_utc_now(),
        "summary": summary,
        "current_job_id": current_job_id,
        "current_attempt": current_attempt,
    }
    path = config.heartbeat_dir / f"{config.worker_id}.heartbeat.json"
    _write_json(path, heartbeat)


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()  # noqa: UP017


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned if cleaned != "" else "scene"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _atomic_create_json(path: Path, payload: dict[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(path), flags)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return True


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
