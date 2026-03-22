"""Public Python API for programmatic verbx workflows.

This module exposes a small, stable surface for automation pipelines and
notebooks that should not need to shell out to the CLI.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.analysis.framewise import write_framewise_csv
from verbx.config import RenderConfig
from verbx.core.pipeline import run_render_pipeline
from verbx.io.audio import read_audio, validate_audio_path
from verbx.ir.generator import IRGenConfig, generate_ir as _generate_ir, write_ir_artifacts


__all__ = ["analyze_file", "generate_ir", "render_file"]


def render_file(
    infile: str | Path,
    outfile: str | Path,
    *,
    config: RenderConfig | None = None,
    **config_overrides: Any,
) -> dict[str, Any]:
    """Render one file with ``verbx`` and return the structured report.

    Args:
        infile: Input audio file path.
        outfile: Output audio file path.
        config: Optional base ``RenderConfig``.
        **config_overrides: Optional ``RenderConfig`` field overrides.
    """
    runtime_config = _resolve_render_config(config=config, overrides=config_overrides)
    return run_render_pipeline(Path(infile), Path(outfile), runtime_config)


def generate_ir(
    out_ir: str | Path,
    *,
    config: IRGenConfig | None = None,
    write_metadata: bool = True,
) -> dict[str, Any]:
    """Generate an impulse response and write audio/metadata artifacts.

    Returns a summary payload with output path, sample rate, and metadata.
    """
    out_path = Path(out_ir)
    ir_config = config if config is not None else IRGenConfig()
    audio, sr, meta = _generate_ir(ir_config)
    write_ir_artifacts(
        out_path,
        audio,
        sr,
        meta,
        silent=not write_metadata,
    )
    payload: dict[str, Any] = {
        "out_ir": str(out_path),
        "sample_rate": int(sr),
        "metadata": meta,
    }
    if write_metadata:
        payload["metadata_path"] = str(out_path.with_suffix(f"{out_path.suffix}.ir.meta.json"))
    return payload


def analyze_file(
    infile: str | Path,
    *,
    include_loudness: bool = False,
    include_edr: bool = False,
    ambi_order: int | None = None,
    ambi_normalization: str = "auto",
    ambi_channel_order: str = "auto",
    json_out: str | Path | None = None,
    frames_out: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze one audio file and return metrics.

    Optional JSON and framewise CSV artifacts mirror the CLI ``analyze`` outputs.
    """
    path = Path(infile)
    validate_audio_path(str(path))
    audio, sr = read_audio(str(path))

    analyzer = AudioAnalyzer()
    metrics = analyzer.analyze(
        audio,
        sr,
        include_loudness=include_loudness,
        include_edr=include_edr,
        ambi_order=ambi_order,
        ambi_normalization=ambi_normalization.strip().lower(),
        ambi_channel_order=ambi_channel_order.strip().lower(),
    )

    payload: dict[str, Any] = {
        "infile": str(path),
        "sample_rate": int(sr),
        "channels": int(audio.shape[1]),
        "metrics": metrics,
    }

    if json_out is not None:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["json_out"] = str(json_path)

    if frames_out is not None:
        frames_path = Path(frames_out)
        frames_path.parent.mkdir(parents=True, exist_ok=True)
        write_framewise_csv(frames_path, audio, sr)
        payload["frames_out"] = str(frames_path)

    return payload


def _resolve_render_config(*, config: RenderConfig | None, overrides: dict[str, Any]) -> RenderConfig:
    """Return a render config with optional field overrides."""
    base = config if config is not None else RenderConfig()
    if not overrides:
        return base

    valid_fields = set(RenderConfig.__dataclass_fields__.keys())
    unknown = sorted(set(overrides) - valid_fields)
    if unknown:
        bad = ", ".join(unknown)
        raise ValueError(f"Unknown RenderConfig override(s): {bad}")

    return replace(base, **overrides)
