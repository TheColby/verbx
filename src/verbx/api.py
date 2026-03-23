"""Stable Python API wrappers for common verbx workflows.

This module provides a small surface intended for notebook/research integration
without shelling out to the CLI.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.config import RenderConfig
from verbx.core.pipeline import run_render_pipeline
from verbx.io.audio import read_audio, validate_audio_path
from verbx.ir.generator import (
    IRGenConfig,
    generate_or_load_cached_ir,
    write_ir_artifacts,
)


def render_file(
    infile: str | Path,
    outfile: str | Path,
    *,
    config: RenderConfig | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Render a file with optional ``RenderConfig`` overrides.

    Example:
        >>> render_file("in.wav", "out.wav", engine="algo", rt60=2.5, wet=0.3, dry=0.7)
    """
    in_path = Path(infile)
    out_path = Path(outfile)
    validate_audio_path(in_path)
    base = asdict(config) if config is not None else asdict(RenderConfig())
    for key, value in overrides.items():
        if key in base and value is not None:
            base[key] = value
    resolved = RenderConfig(**base)
    return run_render_pipeline(in_path, out_path, resolved)


def analyze_file(
    infile: str | Path,
    *,
    include_loudness: bool = True,
    include_edr: bool = False,
    ambi_order: int | None = None,
    ambi_normalization: str = "auto",
    ambi_channel_order: str = "auto",
) -> dict[str, float]:
    """Analyze a file and return metric payload."""
    in_path = Path(infile)
    validate_audio_path(in_path)
    audio, sr = read_audio(str(in_path))
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
    outfile: str | Path,
    *,
    config: IRGenConfig | None = None,
    cache_dir: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Generate (or cache-load) an IR and optionally write it to ``outfile``."""
    out_path = Path(outfile)
    cfg = IRGenConfig() if config is None else config
    resolved_cache = Path(cache_dir) if cache_dir is not None else Path(".verbx_cache/ir")
    audio, sr, meta, _cache_path, _cache_hit = generate_or_load_cached_ir(
        config=cfg,
        cache_dir=resolved_cache,
    )
    if write:
        write_ir_artifacts(
            out_path,
            audio=audio,
            sr=sr,
            meta=meta,
            silent=True,
        )
    return {
        "outfile": str(out_path),
        "sample_rate": int(sr),
        "channels": int(audio.shape[1]) if audio.ndim == 2 else 1,
        "frames": int(audio.shape[0]),
        "meta": meta,
    }
