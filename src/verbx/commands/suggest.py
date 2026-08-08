"""Suggest command module."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import numpy as np
import soundfile as sf
import typer
from rich.console import Console
from rich.table import Table

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.commands.common import processing_status, write_json_atomic
from verbx.io.audio import read_audio, validate_audio_path

console = Console()


def _confidence_and_rationale(
    *, duration: float, dynamic: float, flatness: float, engine: str
) -> tuple[float, str]:
    """Explain the deterministic heuristic behind a suggested preset.

    This is deliberately a confidence in the *heuristic's evidence*, not a
    claim of perceptual similarity or a neural-model probability.  It gives
    callers an auditable bridge to the future suggestive AI tier.
    """
    duration_evidence = float(np.clip(duration / 2.0, 0.0, 1.0))
    dynamic_evidence = float(np.clip(dynamic / 30.0, 0.0, 1.0))
    flatness_distance = float(np.clip(abs(flatness - 0.12) / 0.12, 0.0, 1.0))
    confidence = float(np.clip(0.35 + 0.30 * duration_evidence + 0.20 * dynamic_evidence + 0.15 * flatness_distance, 0.0, 0.95))
    texture = "low spectral flatness" if engine == "conv" else "broadband spectral texture"
    rationale = (
        f"Selected {engine} from {texture}; RT60 follows the {duration:.2f}s "
        f"program duration and wet mix follows the {dynamic:.2f} dB dynamic range."
    )
    return confidence, rationale


def suggest(
    infile: Annotated[Path, typer.Argument(..., exists=True, readable=True, resolve_path=True)],
    pin: Annotated[
        Path | None,
        typer.Option(
            "--pin",
            resolve_path=True,
            help="Write suggested parameters as a JSON preset file.",
        ),
    ] = None,
) -> None:
    """Suggest practical render defaults from input analysis."""
    try:
        with processing_status("Analyze for suggestions"):
            validate_audio_path(str(infile))
            audio, sr = read_audio(str(infile))
            analyzer = AudioAnalyzer()
            metrics = analyzer.analyze(audio, sr)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    duration = float(metrics.get("duration", 0.0))
    dynamic = float(metrics.get("dynamic_range", 0.0))
    flatness = float(metrics.get("spectral_flatness", 0.0))

    suggested_rt60 = float(np.clip(duration * 1.8, 25.0, 120.0))
    suggested_wet = float(np.clip(0.55 + (dynamic / 60.0), 0.4, 0.95))
    suggested_dry = float(np.clip(1.0 - (suggested_wet * 0.85), 0.05, 0.85))
    suggested_engine = "conv" if flatness < 0.12 else "algo"
    confidence, rationale = _confidence_and_rationale(
        duration=duration,
        dynamic=dynamic,
        flatness=flatness,
        engine=suggested_engine,
    )

    table = Table(title=f"Suggested Parameters: {infile.name}")
    table.add_column("Parameter", style="green")
    table.add_column("Suggested Value", style="white")
    table.add_row("engine", suggested_engine)
    table.add_row("confidence", f"{confidence:.0%} (heuristic)")
    table.add_row("rationale", rationale)
    table.add_row("rt60", f"{suggested_rt60:.2f}")
    table.add_row("wet", f"{suggested_wet:.3f}")
    table.add_row("dry", f"{suggested_dry:.3f}")
    table.add_row("repeat", "2" if duration < 15.0 else "1")
    table.add_row("target-lufs", "-18.0")
    table.add_row("target-peak-dbfs", "-1.0")
    table.add_row("normalize-stage", "post")
    table.add_row("shimmer", "off")
    table.add_row("duck", "off")
    console.print(table)

    if pin is not None:
        pinned: dict[str, Any] = {
            "engine": suggested_engine,
            "rt60": round(suggested_rt60, 4),
            "wet": round(suggested_wet, 4),
            "dry": round(suggested_dry, 4),
            "repeat": 2 if duration < 15.0 else 1,
            "target_lufs": -18.0,
            "target_peak_dbfs": -1.0,
            "normalize_stage": "post",
            "shimmer": False,
            "duck": False,
            "suggestion": {
                "schema": "suggestion-rationale-v1",
                "method": "deterministic-feature-heuristic",
                "confidence": round(confidence, 4),
                "rationale": rationale,
            },
        }
        try:
            write_json_atomic(pin.resolve(), pinned)
            console.print(f"[dim]Preset pinned to {pin.resolve()}[/dim]")
        except (OSError, RuntimeError, ValueError) as exc:
            raise typer.BadParameter(f"Failed to write --pin: {exc}") from exc
