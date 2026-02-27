"""Typer CLI for verbx."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.config import EngineName, IRNormalize, RenderConfig
from verbx.core.pipeline import run_render_pipeline
from verbx.io.audio import read_audio, validate_audio_path
from verbx.logging import configure_logging
from verbx.presets.default_presets import preset_names

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    help="Extreme reverb CLI with scalable DSP architecture.",
)
console = Console()


@app.command()
def render(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    outfile: Path = typer.Argument(..., resolve_path=True),
    engine: EngineName = typer.Option("auto", "--engine", help="Engine: conv, algo, or auto."),
    rt60: float = typer.Option(60.0, "--rt60", min=0.1),
    wet: float = typer.Option(0.8, "--wet", min=0.0, max=1.0),
    dry: float = typer.Option(0.2, "--dry", min=0.0, max=1.0),
    repeat: int = typer.Option(1, "--repeat", min=1),
    freeze: bool = typer.Option(False, "--freeze", help="Enable freeze segment mode."),
    start: float | None = typer.Option(None, "--start", min=0.0),
    end: float | None = typer.Option(None, "--end", min=0.0),
    pre_delay_ms: float = typer.Option(20.0, "--pre-delay-ms", min=0.0),
    damping: float = typer.Option(0.45, "--damping", min=0.0, max=1.0),
    width: float = typer.Option(1.0, "--width", min=0.0, max=2.0),
    mod_depth_ms: float = typer.Option(2.0, "--mod-depth-ms", min=0.0),
    mod_rate_hz: float = typer.Option(0.1, "--mod-rate-hz", min=0.0),
    ir: Path | None = typer.Option(None, "--ir", exists=True, readable=True, resolve_path=True),
    ir_normalize: IRNormalize = typer.Option("peak", "--ir-normalize"),
    tail_limit: float | None = typer.Option(None, "--tail-limit", min=0.0),
    threads: int | None = typer.Option(None, "--threads", min=1),
    partition_size: int = typer.Option(16_384, "--partition-size", min=256),
    block_size: int = typer.Option(4096, "--block-size", min=256),
    analysis_out: str | None = typer.Option(None, "--analysis-out"),
    silent: bool = typer.Option(False, "--silent", help="Disable analysis JSON + console output."),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    """Render input audio with algorithmic or convolution reverb."""
    config = RenderConfig(
        engine=engine,
        rt60=rt60,
        pre_delay_ms=pre_delay_ms,
        damping=damping,
        width=width,
        mod_depth_ms=mod_depth_ms,
        mod_rate_hz=mod_rate_hz,
        wet=wet,
        dry=dry,
        repeat=repeat,
        freeze=freeze,
        start=start,
        end=end,
        block_size=block_size,
        ir=None if ir is None else str(ir),
        ir_normalize=ir_normalize,
        tail_limit=tail_limit,
        threads=threads,
        partition_size=partition_size,
        analysis_out=analysis_out,
        silent=silent,
        progress=progress,
    )

    configure_logging(verbose=not config.silent)

    try:
        report = run_render_pipeline(infile=infile, outfile=outfile, config=config)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc

    if config.silent:
        return

    table = Table(title="Render Summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("engine", str(report.get("engine", "unknown")))
    table.add_row("sample_rate", str(report.get("sample_rate", "")))
    table.add_row("channels", str(report.get("channels", "")))
    table.add_row("input_samples", str(report.get("input_samples", "")))
    table.add_row("output_samples", str(report.get("output_samples", "")))
    table.add_row("analysis_json", str(report.get("analysis_path", "")))
    console.print(table)


@app.command()
def analyze(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    json_out: Path | None = typer.Option(None, "--json-out", resolve_path=True),
) -> None:
    """Analyze an audio file and print a summary table."""
    validate_audio_path(str(infile))
    audio, sr = read_audio(str(infile))
    analyzer = AudioAnalyzer()
    metrics = analyzer.analyze(audio, sr)

    table = Table(title=f"Analysis: {infile.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for key in sorted(metrics):
        table.add_row(key, f"{metrics[key]:.6f}")
    console.print(table)

    if json_out is not None:
        payload = {"sample_rate": sr, "channels": audio.shape[1], "metrics": metrics}
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.command()
def suggest(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
) -> None:
    """Suggest practical render defaults from input analysis."""
    validate_audio_path(str(infile))
    audio, sr = read_audio(str(infile))
    analyzer = AudioAnalyzer()
    metrics = analyzer.analyze(audio, sr)

    duration = metrics["duration"]
    dynamic = metrics["dynamic_range"]
    flatness = metrics["spectral_flatness"]

    suggested_rt60 = float(np.clip(duration * 1.8, 25.0, 120.0))
    suggested_wet = float(np.clip(0.55 + (dynamic / 60.0), 0.4, 0.95))
    suggested_dry = float(np.clip(1.0 - (suggested_wet * 0.85), 0.05, 0.85))
    suggested_engine = "conv" if flatness < 0.12 else "algo"

    table = Table(title=f"Suggested Parameters: {infile.name}")
    table.add_column("Parameter", style="green")
    table.add_column("Suggested Value", style="white")
    table.add_row("engine", suggested_engine)
    table.add_row("rt60", f"{suggested_rt60:.2f}")
    table.add_row("wet", f"{suggested_wet:.3f}")
    table.add_row("dry", f"{suggested_dry:.3f}")
    table.add_row("repeat", "2" if duration < 15.0 else "1")
    table.add_row("pre-delay-ms", "25.0")
    table.add_row("damping", "0.45")
    table.add_row("width", "1.15")
    table.add_row("mod-depth-ms", "2.0")
    table.add_row("mod-rate-hz", "0.08")
    console.print(table)


@app.command(name="presets")
def list_presets() -> None:
    """Print available presets."""
    names = preset_names()
    table = Table(title="Available Presets")
    table.add_column("Preset", style="green")
    for name in names:
        table.add_row(name)
    console.print(table)


if __name__ == "__main__":
    app()
