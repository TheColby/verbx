"""Compare command module."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import soundfile as sf
import typer
from rich.console import Console
from rich.table import Table

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.commands.common import processing_status, write_json_atomic
from verbx.io.audio import read_audio, validate_audio_path

console = Console()


def compare(
    file_a: Annotated[Path, typer.Argument(..., exists=True, readable=True, resolve_path=True)],
    file_b: Annotated[Path, typer.Argument(..., exists=True, readable=True, resolve_path=True)],
    json_out: Annotated[
        Path | None,
        typer.Option(
            "--json-out",
            resolve_path=True,
            help="Optional path to write the comparison report as JSON.",
        ),
    ] = None,
    lufs: Annotated[
        bool, typer.Option("--lufs", help="Include LUFS/true-peak/LRA metrics.")
    ] = False,
    room: Annotated[
        bool,
        typer.Option(
            "--room",
            help="Include room size and acoustic property estimates for both files.",
        ),
    ] = False,
) -> None:
    """Side-by-side comparison of two audio files."""
    try:
        with processing_status("Compare audio files"):
            validate_audio_path(str(file_a))
            validate_audio_path(str(file_b))
            audio_a, sr_a = read_audio(str(file_a))
            audio_b, sr_b = read_audio(str(file_b))
            analyzer = AudioAnalyzer()
            metrics_a = analyzer.analyze(audio_a, sr_a, include_loudness=lufs, include_room=room)
            metrics_b = analyzer.analyze(audio_b, sr_b, include_loudness=lufs, include_room=room)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    all_keys = sorted(set(metrics_a) | set(metrics_b))
    table = Table(title=f"Compare: {file_a.name} vs {file_b.name}")
    table.add_column("Metric", style="cyan")
    table.add_column(file_a.name, justify="right")
    table.add_column(file_b.name, justify="right")
    table.add_column("Delta (B - A)", justify="right", style="yellow")
    for key in all_keys:
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)
        str_a = (
            f"{val_a:.6f}" if isinstance(val_a, float) else str(val_a)
        ) if val_a is not None else "-"
        str_b = (
            f"{val_b:.6f}" if isinstance(val_b, float) else str(val_b)
        ) if val_b is not None else "-"
        if isinstance(val_a, float) and isinstance(val_b, float):
            str_delta = f"{val_b - val_a:+.6f}"
        elif val_a is not None and val_b is not None:
            str_delta = "-"
        else:
            str_delta = "-"
        table.add_row(key, str_a, str_b, str_delta)
    console.print(table)
    if sr_a != sr_b:
        console.print(
            f"[yellow]Warning:[/yellow] sample rates differ: "
            f"{file_a.name}={sr_a} Hz, {file_b.name}={sr_b} Hz"
        )

    if json_out is not None:
        payload: dict[str, Any] = {
            "schema": "compare-report-v1",
            "file_a": str(file_a),
            "file_b": str(file_b),
            "sample_rate_a": int(sr_a),
            "sample_rate_b": int(sr_b),
            "channels_a": int(audio_a.shape[1]),
            "channels_b": int(audio_b.shape[1]),
            "metrics_a": metrics_a,
            "metrics_b": metrics_b,
            "delta": {
                key: metrics_b[key] - metrics_a[key]  # type: ignore[operator]
                for key in all_keys
                if key in metrics_a
                and key in metrics_b
                and isinstance(metrics_a[key], float)
                and isinstance(metrics_b[key], float)
            },
        }
        try:
            write_json_atomic(json_out.resolve(), payload)
            console.print(f"[dim]Comparison report written to {json_out.resolve()}[/dim]")
        except (OSError, RuntimeError, ValueError) as exc:
            raise typer.BadParameter(f"Failed to write --json-out: {exc}") from exc
