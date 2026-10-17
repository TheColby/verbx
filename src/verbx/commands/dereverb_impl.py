# ruff: noqa: B008
"""Extracted ``dereverb`` command implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
import typer
from rich.console import Console
from rich.table import Table

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.commands.common import processing_status as _processing_status
from verbx.commands.common import write_json_atomic as _write_json_atomic
from verbx.commands.validators import ensure_distinct_paths as _ensure_distinct_paths
from verbx.commands.validators import validate_output_audio_path as _validate_output_audio_path
from verbx.config import OutputSubtype
from verbx.core.dereverb import (
    DereverbConfig,
    apply_dereverb,
    parse_dereverb_window_weights,
    run_dereverb_benchmark,
)
from verbx.io.audio import read_audio, validate_audio_path, write_audio

console = Console()


def dereverb_impl(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    outfile: Path = typer.Argument(..., resolve_path=True),
    mode: Literal["wiener", "spectral_sub"] = typer.Option(
        "wiener",
        "--mode",
        help="Dereverberation mode: wiener or spectral_sub.",
    ),
    strength: float = typer.Option(
        0.65,
        "--strength",
        min=0.0,
        max=2.0,
        help="Suppression strength (higher removes more tail but can add artifacts).",
    ),
    floor: float = typer.Option(
        0.08,
        "--floor",
        min=0.0,
        max=1.0,
        help="Spectral floor to preserve ambience and avoid musical noise.",
    ),
    window_ms: float = typer.Option(
        42.67,
        "--window-ms",
        min=2.0,
        help="STFT analysis window size in milliseconds.",
    ),
    hop_ms: float = typer.Option(
        10.67,
        "--hop-ms",
        min=1.0,
        help="STFT hop size in milliseconds.",
    ),
    tail_ms: float = typer.Option(
        220.0,
        "--tail-ms",
        min=1.0,
        help="Late-field smoothing horizon in milliseconds.",
    ),
    pre_emphasis: float = typer.Option(
        0.0,
        "--pre-emphasis",
        min=0.0,
        max=0.98,
        help="Optional high-frequency emphasis before dereverberation.",
    ),
    mix: float = typer.Option(
        1.0,
        "--mix",
        min=0.0,
        max=1.0,
        help="Wet mix of dereverberated output (1.0 = fully processed).",
    ),
    window_type: str = typer.Option(
        "hann",
        "--window-type",
        help=(
            "Analysis window family "
            "(hann, hamming, blackman, kaiser, dpss, tukey, chebwin, and many more)."
        ),
    ),
    synthesis_window_type: str | None = typer.Option(
        None,
        "--synthesis-window-type",
        help="Optional synthesis window family. Defaults to --window-type.",
    ),
    window_symmetric: bool = typer.Option(
        False,
        "--window-symmetric/--window-periodic",
        help="Use symmetric windows instead of periodic STFT windows.",
    ),
    window_alpha: float = typer.Option(0.5, "--window-alpha", min=0.0),
    window_beta: float = typer.Option(14.0, "--window-beta", min=0.0),
    window_std: float = typer.Option(2.5, "--window-std", min=1e-6),
    window_power: float = typer.Option(1.5, "--window-power", min=1e-6),
    window_atten_db: float = typer.Option(100.0, "--window-atten-db", min=1e-3),
    window_nbar: int = typer.Option(4, "--window-nbar", min=2),
    window_nw: float = typer.Option(2.5, "--window-nw", min=1e-3),
    window_tau: float = typer.Option(3.0, "--window-tau", min=1e-6),
    window_weights: str | None = typer.Option(
        None,
        "--window-weights",
        help="Optional comma-separated weights for general_cosine windows.",
    ),
    out_subtype: OutputSubtype = typer.Option(
        "auto",
        "--out-subtype",
        help="Output subtype: auto, float32, float64, pcm16, pcm24, pcm32.",
    ),
    json_out: Path | None = typer.Option(
        None,
        "--json-out",
        resolve_path=True,
        help="Optional path for detailed dereverb metrics JSON.",
    ),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress summary table output."),
    benchmark: bool = typer.Option(
        False,
        "--benchmark",
        help=(
            "Run a synthetic quality benchmark (SNR, spectral distance) without processing "
            "any files. --json-out writes the benchmark report when provided."
        ),
    ),
    benchmark_rt60: float = typer.Option(
        1.2,
        "--benchmark-rt60",
        min=0.1,
        max=60.0,
        help="Simulated RT60 in seconds for the synthetic benchmark IR.",
    ),
    benchmark_sr: int = typer.Option(
        24000,
        "--benchmark-sr",
        min=8000,
        help="Sample rate for synthetic benchmark signal.",
    ),
) -> None:
    """Suppress late reverberation from an existing audio recording."""
    try:
        parsed_window_weights = parse_dereverb_window_weights(window_weights)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if benchmark:
        configs = [
            DereverbConfig(
                mode="wiener",
                strength=float(strength),
                floor=float(floor),
                window_ms=float(window_ms),
                hop_ms=float(hop_ms),
                tail_ms=float(tail_ms),
                analysis_window=str(window_type),
                synthesis_window=synthesis_window_type,
                window_symmetric=bool(window_symmetric),
                window_alpha=float(window_alpha),
                window_beta=float(window_beta),
                window_std=float(window_std),
                window_power=float(window_power),
                window_atten_db=float(window_atten_db),
                window_nbar=int(window_nbar),
                window_nw=float(window_nw),
                window_tau=float(window_tau),
                window_weights=parsed_window_weights,
            ),
            DereverbConfig(
                mode="spectral_sub",
                strength=float(strength),
                floor=float(floor),
                window_ms=float(window_ms),
                hop_ms=float(hop_ms),
                tail_ms=float(tail_ms),
                analysis_window=str(window_type),
                synthesis_window=synthesis_window_type,
                window_symmetric=bool(window_symmetric),
                window_alpha=float(window_alpha),
                window_beta=float(window_beta),
                window_std=float(window_std),
                window_power=float(window_power),
                window_atten_db=float(window_atten_db),
                window_nbar=int(window_nbar),
                window_nw=float(window_nw),
                window_tau=float(window_tau),
                window_weights=parsed_window_weights,
            ),
        ]
        with _processing_status("Run dereverb benchmark", enabled=not quiet):
            report = run_dereverb_benchmark(
                sr=int(benchmark_sr),
                rt60=float(benchmark_rt60),
                configs=configs,
            )
        if json_out is not None:
            try:
                json_out.parent.mkdir(parents=True, exist_ok=True)
                _write_json_atomic(json_out.resolve(), report)
            except (OSError, RuntimeError, ValueError) as exc:
                raise typer.BadParameter(f"Failed to write --json-out: {exc}") from exc
        if not quiet:
            table = Table(title=f"Dereverb Benchmark (RT60={benchmark_rt60}s, SR={benchmark_sr})")
            table.add_column("Mode", style="cyan")
            table.add_column("SNR (dB)", justify="right")
            table.add_column("SNR Δ (dB)", justify="right")
            table.add_column("Spectral Dist (Hz)", justify="right")
            table.add_column("RMS Δ (dB)", justify="right")
            results = report.get("results", [])
            if isinstance(results, list):
                for r in results:
                    table.add_row(
                        str(r["mode"]),
                        f"{r['snr_db']:.2f}",
                        f"{r['snr_improvement_db']:+.2f}",
                        f"{r['spectral_dist_hz']:.1f}",
                        f"{r['rms_delta_db']:+.2f}",
                    )
            console.print(table)
            console.print(
                f"[dim]Baseline SNR (reverberant vs clean): "
                f"{report['snr_reverberant_db']:.2f} dB[/dim]"
            )
            if json_out is not None:
                console.print(f"[dim]Benchmark report written to {json_out.resolve()}[/dim]")
        return

    _ensure_distinct_paths(infile, outfile, "INFILE", "OUTFILE")
    _validate_output_audio_path(outfile, out_subtype)
    if json_out is not None and infile.resolve() == json_out.resolve():
        msg = "--json-out must be different from INFILE."
        raise typer.BadParameter(msg)
    if json_out is not None and outfile.resolve() == json_out.resolve():
        msg = "--json-out must be different from OUTFILE."
        raise typer.BadParameter(msg)
    if hop_ms >= window_ms:
        msg = "--hop-ms must be smaller than --window-ms."
        raise typer.BadParameter(msg)

    subtype_map = {
        "auto": None,
        "float32": "FLOAT",
        "float64": "DOUBLE",
        "pcm16": "PCM_16",
        "pcm24": "PCM_24",
        "pcm32": "PCM_32",
    }
    resolved_subtype = subtype_map[str(out_subtype)]
    try:
        with _processing_status("Dereverberate audio", enabled=True):
            validate_audio_path(str(infile))
            audio, sr = read_audio(str(infile))
            config = DereverbConfig(
                mode=mode,
                strength=float(strength),
                floor=float(floor),
                window_ms=float(window_ms),
                hop_ms=float(hop_ms),
                tail_ms=float(tail_ms),
                pre_emphasis=float(pre_emphasis),
                mix=float(mix),
                analysis_window=str(window_type),
                synthesis_window=synthesis_window_type,
                window_symmetric=bool(window_symmetric),
                window_alpha=float(window_alpha),
                window_beta=float(window_beta),
                window_std=float(window_std),
                window_power=float(window_power),
                window_atten_db=float(window_atten_db),
                window_nbar=int(window_nbar),
                window_nw=float(window_nw),
                window_tau=float(window_tau),
                window_weights=parsed_window_weights,
            )
            processed = apply_dereverb(audio, sr, config)
            write_audio(str(outfile), processed, sr, subtype=resolved_subtype)
            analyzer = AudioAnalyzer()
            input_metrics = analyzer.analyze(audio, sr, include_loudness=True)
            output_metrics = analyzer.analyze(processed, sr, include_loudness=True)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    suppression_db = float(
        20.0
        * np.log10(
            (float(output_metrics.get("rms", 1e-12)) + 1e-12)
            / (float(input_metrics.get("rms", 1e-12)) + 1e-12)
        )
    )
    payload: dict[str, Any] = {
        "schema": "dereverb-report-v1",
        "command": "dereverb",
        "status": "ok",
        "input": str(infile),
        "output": str(outfile),
        "sample_rate": int(sr),
        "channels": int(audio.shape[1]),
        "config": {
            "mode": mode,
            "strength": float(strength),
            "floor": float(floor),
            "window_ms": float(window_ms),
            "hop_ms": float(hop_ms),
            "tail_ms": float(tail_ms),
            "pre_emphasis": float(pre_emphasis),
            "mix": float(mix),
            "window_type": str(window_type),
            "synthesis_window_type": synthesis_window_type,
            "window_symmetric": bool(window_symmetric),
            "window_alpha": float(window_alpha),
            "window_beta": float(window_beta),
            "window_std": float(window_std),
            "window_power": float(window_power),
            "window_atten_db": float(window_atten_db),
            "window_nbar": int(window_nbar),
            "window_nw": float(window_nw),
            "window_tau": float(window_tau),
            "window_weights": list(parsed_window_weights),
            "out_subtype": out_subtype,
        },
        "metrics": {
            "input": input_metrics,
            "output": output_metrics,
            "rms_delta_db": suppression_db,
        },
    }
    if json_out is not None:
        _write_json_atomic(json_out.resolve(), payload)

    if quiet:
        return

    table = Table(title=f"Dereverb: {infile.name} → {outfile.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    table.add_row(
        "duration_s",
        f"{input_metrics.get('duration', 0.0):.3f}",
        f"{output_metrics.get('duration', 0.0):.3f}",
    )
    table.add_row(
        "rms",
        f"{input_metrics.get('rms', 0.0):.6f}",
        f"{output_metrics.get('rms', 0.0):.6f}",
    )
    table.add_row(
        "dynamic_range_db",
        f"{input_metrics.get('dynamic_range', 0.0):.3f}",
        f"{output_metrics.get('dynamic_range', 0.0):.3f}",
    )
    table.add_row(
        "spectral_centroid_hz",
        f"{input_metrics.get('spectral_centroid', 0.0):.2f}",
        f"{output_metrics.get('spectral_centroid', 0.0):.2f}",
    )
    table.add_row("rms_delta_db", "-", f"{suppression_db:.3f}")
    console.print(table)
    if json_out is not None:
        console.print(f"[dim]Detailed JSON written to {json_out.resolve()}[/dim]")
