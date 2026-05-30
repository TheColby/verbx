"""Analyze command module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import soundfile as sf
import typer
from rich.console import Console
from rich.table import Table

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.analysis.framewise import write_framewise_csv
from verbx.commands.common import processing_status
from verbx.config import AmbiChannelOrder, AmbiNormalization
from verbx.io.audio import read_audio, validate_audio_path

console = Console()


def analyze(
    infile: Annotated[Path, typer.Argument(..., exists=True, readable=True, resolve_path=True)],
    json_out: Annotated[Path | None, typer.Option("--json-out", resolve_path=True)] = None,
    lufs: Annotated[
        bool, typer.Option("--lufs", help="Include LUFS/true-peak/LRA metrics.")
    ] = False,
    edr: Annotated[
        bool,
        typer.Option(
            "--edr",
            help="Include EDR (Energy Decay Relief) summary metrics.",
        ),
    ] = False,
    frames_out: Annotated[
        Path | None, typer.Option("--frames-out", resolve_path=True)
    ] = None,
    ambi_order: Annotated[
        int,
        typer.Option(
            "--ambi-order",
            min=0,
            max=7,
            help="Enable Ambisonics spatial metrics for the given order.",
        ),
    ] = 0,
    ambi_normalization: Annotated[
        AmbiNormalization,
        typer.Option(
            "--ambi-normalization",
            help="Ambisonics normalization convention for analysis mode.",
        ),
    ] = "auto",
    channel_order: Annotated[
        AmbiChannelOrder,
        typer.Option(
            "--channel-order",
            help="Ambisonics channel order convention for analysis mode.",
        ),
    ] = "auto",
    room: Annotated[
        bool,
        typer.Option(
            "--room",
            help=(
                "Estimate room size, dimensions, absorption, critical distance, "
                "and class from the signal's reverberant decay characteristics. "
                "Works best on reverberant recordings or rendered impulse responses."
            ),
        ),
    ] = False,
) -> None:
    """Analyze an audio file and print a summary table."""
    _validate_analyze_call(infile, json_out, frames_out)
    try:
        with processing_status("Analyze audio"):
            validate_audio_path(str(infile))
            audio, sr = read_audio(str(infile))
            analyzer = AudioAnalyzer()
            metrics = analyzer.analyze(
                audio,
                sr,
                include_loudness=lufs,
                include_edr=edr,
                include_room=room,
                ambi_order=int(ambi_order) if int(ambi_order) > 0 else None,
                ambi_normalization=str(ambi_normalization).strip().lower(),
                ambi_channel_order=str(channel_order).strip().lower(),
            )
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title=f"Analysis: {infile.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for key in sorted(metrics):
        value = metrics[key]
        table.add_row(key, f"{value:.6f}" if isinstance(value, float) else str(value))
    console.print(table)

    if json_out is not None:
        payload = {"sample_rate": sr, "channels": audio.shape[1], "metrics": metrics}
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if frames_out is not None:
        write_framewise_csv(frames_out, audio, sr)


def _validate_analyze_call(infile: Path, json_out: Path | None, frames_out: Path | None) -> None:
    """Validate analyze command output paths."""
    if json_out is not None and infile.resolve() == json_out.resolve():
        msg = "--json-out must be different from input file."
        raise typer.BadParameter(msg)
    if frames_out is not None and infile.resolve() == frames_out.resolve():
        msg = "--frames-out must be different from input file."
        raise typer.BadParameter(msg)
