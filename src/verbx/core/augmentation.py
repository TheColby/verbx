"""Dataset-style augmentation planning for AI research workflows.

This module focuses on deterministic, metadata-rich render planning so users can
generate augmentation corpora with reproducible parameter sampling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verbx.config import RenderConfig
from verbx.core.schema_versions import AUGMENT_MANIFEST_VERSION
from verbx.core.tempo import parse_pre_delay_ms

_TOKEN_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_AUDIO_EXTENSIONS = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".caf"}

_AUGMENTATION_PROFILES: dict[str, dict[str, Any]] = {
    "asr-reverb-v1": {
        "description": "Speech robustness augmentation with mostly short/medium rooms.",
        "archetypes": [
            {
                "name": "booth",
                "weight": 0.34,
                "params": {
                    "rt60": (0.18, 0.65),
                    "pre_delay_ms": (0.0, 12.0),
                    "wet": (0.10, 0.28),
                    "dry": (0.82, 1.00),
                    "damping": (0.58, 0.85),
                    "width": (0.78, 1.05),
                    "fdn_matrix": ("hadamard", "householder"),
                    "fdn_lines": (6, 10),
                },
            },
            {
                "name": "room",
                "weight": 0.43,
                "params": {
                    "rt60": (0.45, 1.65),
                    "pre_delay_ms": (4.0, 24.0),
                    "wet": (0.18, 0.42),
                    "dry": (0.72, 1.00),
                    "damping": (0.45, 0.76),
                    "width": (0.90, 1.20),
                    "fdn_matrix": ("hadamard", "householder", "circulant"),
                    "fdn_lines": (8, 14),
                },
            },
            {
                "name": "hall",
                "weight": 0.23,
                "params": {
                    "rt60": (1.40, 3.40),
                    "pre_delay_ms": (18.0, 52.0),
                    "wet": (0.28, 0.58),
                    "dry": (0.62, 0.92),
                    "damping": (0.32, 0.68),
                    "width": (1.00, 1.38),
                    "fdn_matrix": ("hadamard", "random_orthogonal"),
                    "fdn_lines": (10, 18),
                },
            },
        ],
    },
    "music-reverb-v1": {
        "description": "Mix-oriented augmentation for music tagging/separation tasks.",
        "archetypes": [
            {
                "name": "plate",
                "weight": 0.32,
                "params": {
                    "rt60": (0.70, 2.20),
                    "pre_delay_ms": (6.0, 22.0),
                    "wet": (0.24, 0.50),
                    "dry": (0.62, 0.92),
                    "damping": (0.34, 0.62),
                    "width": (1.00, 1.46),
                    "fdn_matrix": ("householder", "random_orthogonal", "circulant"),
                    "fdn_lines": (8, 16),
                },
            },
            {
                "name": "chamber",
                "weight": 0.38,
                "params": {
                    "rt60": (1.30, 3.10),
                    "pre_delay_ms": (12.0, 42.0),
                    "wet": (0.28, 0.60),
                    "dry": (0.54, 0.88),
                    "damping": (0.30, 0.58),
                    "width": (1.08, 1.56),
                    "fdn_matrix": ("hadamard", "random_orthogonal"),
                    "fdn_lines": (10, 20),
                },
            },
            {
                "name": "long_hall",
                "weight": 0.30,
                "params": {
                    "rt60": (2.60, 6.00),
                    "pre_delay_ms": (24.0, 80.0),
                    "wet": (0.34, 0.72),
                    "dry": (0.40, 0.78),
                    "damping": (0.22, 0.52),
                    "width": (1.20, 1.80),
                    "fdn_matrix": ("random_orthogonal", "elliptic", "tv_unitary"),
                    "fdn_lines": (12, 24),
                },
            },
        ],
    },
    "drums-room-v1": {
        "description": "Transient-preserving drum-room augmentation with short reflections.",
        "archetypes": [
            {
                "name": "tight_room",
                "weight": 0.52,
                "params": {
                    "rt60": (0.22, 0.95),
                    "pre_delay_ms": (0.0, 10.0),
                    "wet": (0.12, 0.34),
                    "dry": (0.84, 1.00),
                    "damping": (0.52, 0.88),
                    "width": (0.90, 1.16),
                    "fdn_matrix": ("hadamard", "householder"),
                    "fdn_lines": (6, 12),
                },
            },
            {
                "name": "studio_room",
                "weight": 0.33,
                "params": {
                    "rt60": (0.62, 1.70),
                    "pre_delay_ms": (4.0, 18.0),
                    "wet": (0.22, 0.46),
                    "dry": (0.68, 0.96),
                    "damping": (0.40, 0.78),
                    "width": (0.98, 1.34),
                    "fdn_matrix": ("householder", "circulant", "random_orthogonal"),
                    "fdn_lines": (8, 16),
                },
            },
            {
                "name": "arena",
                "weight": 0.15,
                "params": {
                    "rt60": (1.50, 3.20),
                    "pre_delay_ms": (12.0, 34.0),
                    "wet": (0.28, 0.58),
                    "dry": (0.52, 0.86),
                    "damping": (0.28, 0.62),
                    "width": (1.08, 1.66),
                    "fdn_matrix": ("random_orthogonal", "elliptic"),
                    "fdn_lines": (10, 20),
                },
            },
        ],
    },
}


@dataclass(slots=True)
class AugmentationPlan:
    """One deterministic render plan row for dataset augmentation."""

    index: int
    source_index: int
    source_id: str
    split: str
    label: str
    tags: tuple[str, ...]
    infile: Path
    outfile: Path
    dry_copy_outfile: Path | None
    profile: str
    archetype: str
    variant_index: int
    seed: int
    config: RenderConfig
    source_metadata: dict[str, Any]


@dataclass(slots=True)
class AugmentationBuild:
    """Expanded augmentation manifest with resolved render plans."""

    dataset_name: str
    profile: str
    seed: int
    variants_per_input: int
    output_root: Path
    write_analysis: bool
    copy_dry: bool
    plans: list[AugmentationPlan]


def augmentation_profile_names() -> list[str]:
    """Return sorted augmentation profile names."""
    return sorted(_AUGMENTATION_PROFILES)


def augmentation_profiles() -> dict[str, dict[str, Any]]:
    """Return copy of profile definitions for CLI docs/help output."""
    copied: dict[str, dict[str, Any]] = {}
    for key, profile in _AUGMENTATION_PROFILES.items():
        copied[key] = {
            "description": str(profile["description"]),
            "archetypes": [dict(item) for item in profile["archetypes"]],
        }
    return copied


def build_augmentation_manifest_template() -> dict[str, Any]:
    """Return a JSON-serializable augmentation manifest template."""
    return {
        "version": AUGMENT_MANIFEST_VERSION,
        "dataset_name": "verbx_augmented_set",
        "profile": "asr-reverb-v1",
        "seed": 20260314,
        "variants_per_input": 4,
        "output_root": "augmented_out",
        "write_analysis": False,
        "default_options": {
            "engine": "algo",
            "repeat": 1,
            "output_subtype": "float32",
            "normalize_stage": "none",
            "output_peak_norm": "input",
        },
        "jobs": [
            {
                "id": "utt_0001",
                "infile": "data/clean/utt_0001.wav",
                "split": "train",
                "label": "speaker_a",
                "tags": ["speech", "clean"],
                "variants": 6,
                "options": {"rt60": 1.1},
                "metadata": {"speaker_id": "spk_a", "language": "en"},
            }
        ],
    }


def build_augmentation_plans(
    *,
    payload: dict[str, Any],
    manifest_path: Path,
    output_root_override: Path | None = None,
    copy_dry: bool = False,
    verify_split_isolation: bool = True,
) -> AugmentationBuild:
    """Expand augmentation manifest into deterministic render plans."""
    jobs_raw = payload.get("jobs")
    if not isinstance(jobs_raw, list) or len(jobs_raw) == 0:
        raise ValueError("Augmentation manifest requires non-empty top-level 'jobs' array.")

    profile = _normalize_profile_name(str(payload.get("profile", "asr-reverb-v1")))
    if profile not in _AUGMENTATION_PROFILES:
        options = ", ".join(augmentation_profile_names())
        raise ValueError(
            f"Unknown augmentation profile '{profile}'. Supported profiles: {options}."
        )

    seed = _parse_positive_int(payload.get("seed", 20260314), context="seed")
    variants_default = _parse_positive_int(
        payload.get("variants_per_input", 4),
        context="variants_per_input",
    )
    if variants_default > 500:
        raise ValueError("variants_per_input must be <= 500.")
    write_analysis = bool(payload.get("write_analysis", False))

    output_root = (
        output_root_override.resolve()
        if output_root_override is not None
        else _resolve_output_root(payload.get("output_root"), manifest_path)
    )
    dataset_name = _sanitize_token(str(payload.get("dataset_name", "verbx_augmented_set")))
    default_options = payload.get("default_options", {})
    if not isinstance(default_options, dict):
        raise ValueError("default_options must be an object when provided.")

    plans: list[AugmentationPlan] = []
    source_id_splits: dict[str, set[str]] = {}
    infile_splits: dict[str, set[str]] = {}
    global_index = 1
    for source_index, job_raw in enumerate(jobs_raw, start=1):
        if not isinstance(job_raw, dict):
            raise ValueError(f"jobs[{source_index - 1}] must be an object.")

        infile = _resolve_input_path(job_raw.get("infile"), manifest_path, source_index)
        source_id = _sanitize_token(
            str(job_raw.get("id", job_raw.get("source_id", infile.stem))),
            fallback=f"source_{source_index:05d}",
        )
        split = _sanitize_token(str(job_raw.get("split", "train")), fallback="train")
        label = str(job_raw.get("label", ""))
        tags = _parse_tags(job_raw.get("tags"))
        variants = _parse_positive_int(
            job_raw.get("variants", variants_default),
            context=f"jobs[{source_index - 1}].variants",
        )
        if variants > 500:
            raise ValueError(f"jobs[{source_index - 1}].variants must be <= 500.")
        job_options = job_raw.get("options", {})
        if not isinstance(job_options, dict):
            raise ValueError(f"jobs[{source_index - 1}].options must be an object.")
        source_metadata_raw = job_raw.get("metadata", {})
        source_metadata = (
            dict(source_metadata_raw) if isinstance(source_metadata_raw, dict) else {}
        )
        source_id_splits.setdefault(source_id, set()).add(split)
        infile_key = str(infile.resolve())
        infile_splits.setdefault(infile_key, set()).add(split)
        if verify_split_isolation:
            source_splits = source_id_splits[source_id]
            if len(source_splits) > 1:
                joined = ", ".join(sorted(source_splits))
                raise ValueError(
                    "Split isolation violation: "
                    f"source id '{source_id}' appears in splits {joined}. "
                    "Set verify_split_isolation=False to allow this."
                )
            file_splits = infile_splits[infile_key]
            if len(file_splits) > 1:
                joined = ", ".join(sorted(file_splits))
                raise ValueError(
                    f"Split isolation violation: infile '{infile}' appears in splits {joined}. "
                    "Set verify_split_isolation=False to allow this."
                )

        base_name = f"{source_index:05d}_{source_id}"
        dry_copy_outfile = (
            output_root / split / f"{base_name}__dry{infile.suffix.lower()}"
            if copy_dry
            else None
        )

        for variant_index in range(variants):
            variant_seed = int(seed + (source_index * 100_003) + variant_index)
            rng = np.random.default_rng(variant_seed)
            archetype, sampled_options = _sample_profile_options(profile=profile, rng=rng)
            merged_options: dict[str, Any] = {}
            merged_options.update(_augmentation_render_defaults())
            merged_options.update(sampled_options)
            merged_options.update(default_options)
            merged_options.update(job_options)

            out_suffix = _normalize_output_extension(
                merged_options.get("output_ext", payload.get("output_ext", ".wav"))
            )
            outfile = (
                output_root
                / split
                / f"{base_name}__{archetype}__a{variant_index + 1:03d}{out_suffix}"
            )

            config = _render_config_from_options(merged_options)
            config.progress = False
            config.frames_out = None
            if write_analysis:
                config.silent = False
                analysis_name = f"{base_name}__{archetype}__a{variant_index + 1:03d}.analysis.json"
                config.analysis_out = str(output_root / "analysis" / analysis_name)
            else:
                config.silent = True
                config.analysis_out = None

            plans.append(
                AugmentationPlan(
                    index=global_index,
                    source_index=source_index,
                    source_id=source_id,
                    split=split,
                    label=label,
                    tags=tags,
                    infile=infile,
                    outfile=outfile,
                    dry_copy_outfile=dry_copy_outfile,
                    profile=profile,
                    archetype=archetype,
                    variant_index=variant_index + 1,
                    seed=variant_seed,
                    config=config,
                    source_metadata=source_metadata,
                )
            )
            global_index += 1

    return AugmentationBuild(
        dataset_name=dataset_name,
        profile=profile,
        seed=seed,
        variants_per_input=variants_default,
        output_root=output_root,
        write_analysis=write_analysis,
        copy_dry=copy_dry,
        plans=plans,
    )


def render_config_snapshot(config: RenderConfig) -> dict[str, Any]:
    """Return compact config snapshot for augmentation metadata export."""
    return {
        "engine": str(config.engine),
        "rt60": float(config.rt60),
        "pre_delay_ms": float(config.pre_delay_ms),
        "wet": float(config.wet),
        "dry": float(config.dry),
        "damping": float(config.damping),
        "width": float(config.width),
        "fdn_matrix": str(config.fdn_matrix),
        "fdn_lines": int(config.fdn_lines),
        "output_subtype": str(config.output_subtype),
        "normalize_stage": str(config.normalize_stage),
        "output_peak_norm": str(config.output_peak_norm),
    }


def _augmentation_render_defaults() -> dict[str, Any]:
    return {
        "engine": "algo",
        "repeat": 1,
        "output_subtype": "float32",
        "normalize_stage": "none",
        "output_peak_norm": "input",
        "target_lufs": None,
        "target_peak_dbfs": None,
        "limiter": False,
        "progress": False,
    }


def _sample_profile_options(
    *,
    profile: str,
    rng: np.random.Generator,
) -> tuple[str, dict[str, Any]]:
    profile_obj = _AUGMENTATION_PROFILES[profile]
    archetypes = profile_obj["archetypes"]
    weights = np.asarray([float(item["weight"]) for item in archetypes], dtype=np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        weights = np.ones_like(weights) / float(max(1, weights.shape[0]))
    else:
        weights /= weight_sum
    idx = int(rng.choice(len(archetypes), p=weights))
    archetype = archetypes[idx]
    sampled: dict[str, Any] = {}
    for key, spec in archetype["params"].items():
        sampled[key] = _sample_param_value(spec, rng=rng)
    if "wet" in sampled and "dry" in sampled:
        sampled["wet"] = float(np.clip(float(sampled["wet"]), 0.0, 1.0))
        sampled["dry"] = float(np.clip(float(sampled["dry"]), 0.0, 1.0))
        if float(sampled["wet"]) <= 1e-9 and float(sampled["dry"]) <= 1e-9:
            sampled["dry"] = 1.0
    if "rt60" in sampled:
        sampled["rt60"] = float(np.clip(float(sampled["rt60"]), 0.1, 120.0))
    return str(archetype["name"]), sampled


def _sample_param_value(spec: Any, *, rng: np.random.Generator) -> Any:
    if isinstance(spec, tuple):
        if len(spec) == 2 and all(isinstance(item, (int, float)) for item in spec):
            low = float(spec[0])
            high = float(spec[1])
            if high < low:
                low, high = high, low
            if low.is_integer() and high.is_integer() and int(high - low) >= 1:
                return int(rng.integers(int(low), int(high) + 1))
            return float(rng.uniform(low, high))
        if len(spec) > 0:
            return spec[int(rng.integers(0, len(spec)))]
    if isinstance(spec, list):
        if len(spec) == 0:
            return None
        return spec[int(rng.integers(0, len(spec)))]
    return spec


def _resolve_output_root(raw: Any, manifest_path: Path) -> Path:
    if raw is None:
        return (manifest_path.parent / "augmented_out").resolve()
    token = str(raw).strip()
    if token == "":
        return (manifest_path.parent / "augmented_out").resolve()
    path = Path(token)
    if path.is_absolute():
        return path.resolve()
    return (manifest_path.parent / path).resolve()


def _resolve_input_path(raw: Any, manifest_path: Path, source_index: int) -> Path:
    token = str(raw).strip()
    if token == "":
        raise ValueError(f"jobs[{source_index - 1}] is missing required infile.")
    path = Path(token)
    resolved = path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()
    if not resolved.exists():
        raise ValueError(f"jobs[{source_index - 1}] infile not found: {resolved}")
    return resolved


def _normalize_output_extension(raw: Any) -> str:
    token = str(raw).strip().lower()
    if token == "":
        token = ".wav"
    if not token.startswith("."):
        token = f".{token}"
    if token not in _AUDIO_EXTENSIONS:
        options = ", ".join(sorted(_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported output_ext '{token}'. Supported extensions: {options}.")
    return token


def _render_config_from_options(options: dict[str, Any]) -> RenderConfig:
    fields = RenderConfig.__dataclass_fields__.keys()
    filtered = {key: value for key, value in options.items() if key in fields}
    pre_delay_note = filtered.get("pre_delay_note")
    bpm = filtered.get("bpm")
    if isinstance(pre_delay_note, str):
        fallback_ms = float(filtered.get("pre_delay_ms", 20.0))
        resolved_bpm = float(bpm) if isinstance(bpm, (float, int)) else None
        filtered["pre_delay_ms"] = parse_pre_delay_ms(pre_delay_note, resolved_bpm, fallback_ms)
    return RenderConfig(**filtered)


def _normalize_profile_name(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


def _sanitize_token(value: str, *, fallback: str = "item") -> str:
    token = _TOKEN_SAFE_RE.sub("_", str(value).strip())
    token = token.strip("._-")
    token = re.sub(r"_+", "_", token)
    return token if token != "" else fallback


def _parse_positive_int(raw: Any, *, context: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer.") from exc
    if value < 1:
        raise ValueError(f"{context} must be >= 1.")
    return int(value)


def _parse_tags(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        return ()
    tags: list[str] = []
    for value in raw:
        token = _sanitize_token(str(value), fallback="")
        if token != "":
            tags.append(token)
    deduped = sorted(set(tags))
    return tuple(deduped)
