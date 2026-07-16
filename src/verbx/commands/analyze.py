"""Analyze command module."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import soundfile as sf
import typer
from rich.console import Console
from rich.table import Table

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.analysis.framewise import write_framewise_csv
from verbx.commands.common import processing_status, write_json_atomic
from verbx.config import AmbiChannelOrder, AmbiNormalization
from verbx.io.audio import read_audio, validate_audio_path

console = Console()


def analyze(
    infile: Annotated[Path, typer.Argument(..., exists=True, readable=True, resolve_path=True)],
    json_out: Annotated[Path | None, typer.Option("--json-out", resolve_path=True)] = None,
    reverb: Annotated[
        bool,
        typer.Option(
            "--reverb/--no-reverb",
            help=(
                "Include peak-aligned RT60/EDT/T20/T30, clarity, definition, "
                "center-time, DRR, confidence, and early-IACC metrics."
            ),
        ),
    ] = True,
    input_kind: Annotated[
        str,
        typer.Option(
            "--input-kind",
            help="Reverb-analysis source model: auto, ir, or program.",
        ),
    ] = "auto",
    direct_window_ms: Annotated[
        float,
        typer.Option(
            "--direct-window-ms",
            min=0.1,
            max=100.0,
            help="Direct-sound integration window used for the DRR estimate.",
        ),
    ] = 2.5,
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
                include_reverb=reverb,
                reverb_input_kind=input_kind,
                reverb_direct_window_ms=direct_window_ms,
                ambi_order=int(ambi_order) if int(ambi_order) > 0 else None,
                ambi_normalization=str(ambi_normalization).strip().lower(),
                ambi_channel_order=str(channel_order).strip().lower(),
            )
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title=f"Analysis: {infile.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    ordered_keys = sorted(metrics, key=lambda key: (not key.startswith("reverb_"), key))
    for key in ordered_keys:
        value = metrics[key]
        table.add_row(key, f"{value:.6f}" if isinstance(value, float) else str(value))
    console.print(table)

    if json_out is not None:
        payload = {
            "schema": "analyze-report-v1",
            "source": {
                "path": str(infile.resolve()),
                "sample_rate_hz": int(sr),
                "channels": int(audio.shape[1]),
                "frames": int(audio.shape[0]),
                "duration_seconds": float(audio.shape[0] / max(1, int(sr))),
            },
            "analysis": {
                "reverb": bool(reverb),
                "input_kind": str(input_kind).strip().lower(),
                "direct_window_ms": float(direct_window_ms),
                "loudness": bool(lufs),
                "edr": bool(edr),
                "room": bool(room),
                "ambi_order": int(ambi_order),
                "ambi_normalization": str(ambi_normalization),
                "channel_order": str(channel_order),
            },
            # Retained for compatibility with the pre-v1 report shape.
            "sample_rate": int(sr),
            "channels": int(audio.shape[1]),
            "metrics": metrics,
        }
        write_json_atomic(json_out.resolve(), payload)
        console.print(f"[dim]Analysis report written to {json_out.resolve()}[/dim]")

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
    if json_out is not None and frames_out is not None and json_out.resolve() == frames_out.resolve():
        msg = "--json-out and --frames-out must use different paths."
        raise typer.BadParameter(msg)
