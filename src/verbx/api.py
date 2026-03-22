"""Stable Python API surface for programmatic verbx workflows.

This module exposes a small, versioned façade over core pipeline functions so
notebook/research/integration users can call verbx directly from Python without
shelling out to the CLI.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.config import RenderConfig
from verbx.core.pipeline import run_render_pipeline
from verbx.io.audio import read_audio, validate_audio_path
from verbx.ir.generator import IRGenConfig, generate_ir as _generate_ir_core

AudioArray = npt.NDArray[np.float64]

_RENDER_CONFIG_FIELD_NAMES = {field.name for field in fields(RenderConfig)}
_IR_CONFIG_FIELD_NAMES = {field.name for field in fields(IRGenConfig)}


def _build_render_config(
    config: RenderConfig | None = None,
    **options: Any,
) -> RenderConfig:
    if config is not None and len(options) > 0:
        raise ValueError("Pass either 'config' or keyword options, not both.")
    if config is not None:
        return config
    unknown = sorted(set(options) - _RENDER_CONFIG_FIELD_NAMES)
    if len(unknown) > 0:
        listed = ", ".join(unknown)
        raise ValueError(f"Unsupported RenderConfig option(s): {listed}")
    return RenderConfig(**options)


def _build_ir_config(config: IRGenConfig | None = None, **options: Any) -> IRGenConfig:
    if config is not None and len(options) > 0:
        raise ValueError("Pass either 'config' or keyword options, not both.")
    if config is not None:
        return config
    unknown = sorted(set(options) - _IR_CONFIG_FIELD_NAMES)
    if len(unknown) > 0:
        listed = ", ".join(unknown)
        raise ValueError(f"Unsupported IRGenConfig option(s): {listed}")
    return IRGenConfig(**options)


def render_file(
    infile: str | Path,
    outfile: str | Path,
    *,
    config: RenderConfig | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Render one input file to an output file and return the pipeline report."""
    resolved_config = _build_render_config(config, **options)
    return run_render_pipeline(Path(infile), Path(outfile), resolved_config)


def analyze_file(
    infile: str | Path,
    *,
    include_loudness: bool = False,
    include_edr: bool = False,
    ambi_order: int | None = None,
    ambi_normalization: str = "auto",
    ambi_channel_order: str = "auto",
) -> dict[str, float]:
    """Analyze one audio file and return the metric payload as a dict."""
    validate_audio_path(str(infile))
    audio, sr = read_audio(str(infile))
    analyzer = AudioAnalyzer()
    return analyzer.analyze(
        audio,
        sr,
        include_loudness=include_loudness,
        include_edr=include_edr,
        ambi_order=ambi_order,
        ambi_normalization=ambi_normalization,
        ambi_channel_order=ambi_channel_order,
    )


def generate_ir(
    *,
    config: IRGenConfig | None = None,
    **options: Any,
) -> tuple[AudioArray, int, dict[str, Any]]:
    """Generate an impulse response in-memory and return audio, sr, and metadata."""
    resolved_config = _build_ir_config(config, **options)
    return _generate_ir_core(resolved_config)


__all__ = [
    "analyze_file",
    "generate_ir",
    "render_file",
]
