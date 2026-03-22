"""Stable Python API surface for file render, IR generation, and analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.config import RenderConfig
from verbx.core.pipeline import run_render_pipeline
from verbx.io.audio import read_audio, validate_audio_path
from verbx.ir.generator import IRGenConfig, generate_or_load_cached_ir, write_ir_artifacts


def render_file(
    infile: str | Path,
    outfile: str | Path,
    *,
    config: RenderConfig | None = None,
    **render_options: Any,
) -> dict[str, Any]:
    """Render one input file and write an output file.

    Parameters
    ----------
    infile:
        Input audio path.
    outfile:
        Rendered output audio path.
    config:
        Optional ``RenderConfig`` instance. If omitted, one is built from
        ``render_options``.
    render_options:
        ``RenderConfig`` keyword fields used only when ``config`` is omitted.
    """
    in_path = Path(infile)
    out_path = Path(outfile)

    if config is not None and len(render_options) > 0:
        msg = "Provide either config or render_options, not both."
        raise ValueError(msg)

    render_config = config if config is not None else RenderConfig(**render_options)
    return run_render_pipeline(in_path, out_path, render_config)


def analyze_file(
    infile: str | Path,
    *,
    include_loudness: bool = False,
    include_edr: bool = False,
    ambi_order: int | None = None,
    ambi_normalization: str = "auto",
    ambi_channel_order: str = "auto",
) -> dict[str, float]:
    """Analyze one file and return canonical scalar metrics."""
    path = Path(infile)
    validate_audio_path(str(path))
    audio, sr = read_audio(str(path))
    analyzer = AudioAnalyzer()
    return analyzer.analyze(
        audio,
        sr=sr,
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
    cache_dir: str | Path = ".verbx_cache/irs",
    write_meta: bool = True,
    **ir_options: Any,
) -> dict[str, Any]:
    """Generate an IR, write it to disk, and return metadata.

    Returns a payload containing output path, sample-rate, metadata, and
    cache provenance.
    """
    out_path = Path(outfile)
    cache_path = Path(cache_dir)

    if config is not None and len(ir_options) > 0:
        msg = "Provide either config or ir_options, not both."
        raise ValueError(msg)

    ir_config = config if config is not None else IRGenConfig(**ir_options)
    audio, sr, meta, cached_path, cache_hit = generate_or_load_cached_ir(ir_config, cache_path)
    write_ir_artifacts(out_path, audio, sr, meta, silent=not write_meta)

    return {
        "outfile": str(out_path),
        "sample_rate": int(sr),
        "meta": meta,
        "cache_hit": bool(cache_hit),
        "cache_path": str(cached_path),
        "meta_path": (
            str(out_path.with_suffix(f"{out_path.suffix}.ir.meta.json"))
            if write_meta
            else None
        ),
    }
