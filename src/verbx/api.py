"""Stable Python API surface for common verbx workflows.

This module intentionally exposes a small, versioned façade over internals:

- ``render_file`` for one-shot audio rendering.
- ``generate_ir`` for synthetic IR creation + artifact writing.
- ``analyze_file`` for offline metric extraction.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.config import RenderConfig
from verbx.core.pipeline import run_render_pipeline
from verbx.io.audio import read_audio, validate_audio_path
from verbx.ir.generator import IRGenConfig, write_ir_artifacts
from verbx.ir.generator import generate_ir as _generate_ir

T = TypeVar("T")


def render_file(
    infile: str | Path,
    outfile: str | Path,
    *,
    config: RenderConfig | None = None,
    **render_options: Any,
) -> dict[str, Any]:
    """Render one input file to one output file and return the render report."""
    resolved = _with_overrides(config if config is not None else RenderConfig(), render_options)
    return run_render_pipeline(Path(infile), Path(outfile), resolved)


def generate_ir(
    outfile: str | Path,
    *,
    config: IRGenConfig | None = None,
    write_metadata: bool = True,
    **ir_options: Any,
) -> dict[str, Any]:
    """Generate an IR, write it to disk, and return generated artifact details."""
    resolved = _with_overrides(config if config is not None else IRGenConfig(), ir_options)
    ir_audio, sr, meta = _generate_ir(resolved)
    out_path = Path(outfile)
    write_ir_artifacts(
        out_path=out_path,
        audio=ir_audio,
        sr=sr,
        meta=meta,
        silent=not write_metadata,
    )
    return {
        "outfile": str(out_path),
        "sample_rate": sr,
        "samples": int(ir_audio.shape[0]),
        "channels": int(ir_audio.shape[1]),
        "metadata_path": (
            str(out_path.with_suffix(f"{out_path.suffix}.ir.meta.json")) if write_metadata else None
        ),
        "metadata": meta,
    }


def analyze_file(
    infile: str | Path,
    *,
    include_loudness: bool = False,
    include_edr: bool = False,
    ambi_order: int | None = None,
    ambi_normalization: str = "auto",
    ambi_channel_order: str = "auto",
) -> dict[str, float]:
    """Analyze one audio file and return canonical metrics."""
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


def _with_overrides(base: T, overrides: dict[str, Any]) -> T:
    if len(overrides) == 0:
        return base
    return replace(base, **overrides)
