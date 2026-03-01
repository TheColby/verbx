"""Typer CLI for verbx."""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
import typer
from rich.console import Console
from rich.table import Table

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.analysis.framewise import write_framewise_csv
from verbx.config import (
    DeviceName,
    EngineName,
    IRMatrixLayout,
    IRMode,
    IRNormalize,
    NormalizeStage,
    OutputPeakNorm,
    OutputSubtype,
    RenderConfig,
)
from verbx.core.batch_scheduler import (
    BatchJobResult,
    BatchJobSpec,
    BatchSchedulePolicy,
    estimate_job_cost,
    order_jobs,
    run_parallel_batch,
)
from verbx.core.pipeline import run_render_pipeline
from verbx.core.tempo import parse_pre_delay_ms
from verbx.io.audio import read_audio, validate_audio_path
from verbx.ir.fitting import (
    IRFitCandidate,
    IRFitScore,
    IRFitTarget,
    build_ir_fit_candidates,
    derive_ir_fit_target,
    score_ir_candidate,
)
from verbx.ir.generator import IRGenConfig, generate_or_load_cached_ir, write_ir_artifacts
from verbx.ir.metrics import analyze_ir
from verbx.ir.shaping import apply_ir_shaping
from verbx.ir.tuning import analyze_audio_for_tuning, parse_frequency_hz
from verbx.logging import configure_logging
from verbx.presets.default_presets import preset_names

IRFileFormat = Literal["auto", "wav", "flac", "aiff", "aif", "ogg", "caf"]

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    help="Extreme reverb CLI with scalable DSP architecture.",
)
ir_app = typer.Typer(help="Impulse response workflows.")
cache_app = typer.Typer(help="IR cache inspection and cleanup.")
batch_app = typer.Typer(help="Batch manifest generation and rendering.")

app.add_typer(ir_app, name="ir")
app.add_typer(cache_app, name="cache")
app.add_typer(batch_app, name="batch")

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
    pre_delay: str | None = typer.Option(None, "--pre-delay"),
    bpm: float | None = typer.Option(None, "--bpm", min=1.0),
    damping: float = typer.Option(0.45, "--damping", min=0.0, max=1.0),
    width: float = typer.Option(1.0, "--width", min=0.0, max=2.0),
    mod_depth_ms: float = typer.Option(2.0, "--mod-depth-ms", min=0.0),
    mod_rate_hz: float = typer.Option(0.1, "--mod-rate-hz", min=0.0),
    beast_mode: int = typer.Option(
        1,
        "--beast-mode",
        min=1,
        max=100,
        help=(
            "Scales core reverb parameters by an intensity multiplier (1-100) "
            "to push denser, longer, freeze-like tails."
        ),
    ),
    ir: Path | None = typer.Option(None, "--ir", exists=True, readable=True, resolve_path=True),
    self_convolve: bool = typer.Option(
        False,
        "--self-convolve",
        help=(
            "Use INFILE as its own IR and force fast partitioned convolution "
            "(equivalent to --engine conv --ir INFILE)."
        ),
    ),
    ir_normalize: IRNormalize = typer.Option("peak", "--ir-normalize"),
    ir_matrix_layout: IRMatrixLayout = typer.Option("output-major", "--ir-matrix-layout"),
    tail_limit: float | None = typer.Option(None, "--tail-limit", min=0.0),
    threads: int | None = typer.Option(None, "--threads", min=1),
    device: DeviceName = typer.Option(
        "auto",
        "--device",
        help="Compute device preference: auto, cpu, cuda, or mps (Apple Silicon).",
    ),
    partition_size: int = typer.Option(16_384, "--partition-size", min=256),
    ir_gen: bool = typer.Option(False, "--ir-gen"),
    ir_gen_mode: IRMode = typer.Option("hybrid", "--ir-gen-mode"),
    ir_gen_length: float = typer.Option(60.0, "--ir-gen-length", min=0.1),
    ir_gen_seed: int = typer.Option(0, "--ir-gen-seed"),
    ir_gen_cache_dir: str = typer.Option(".verbx_cache/irs", "--ir-gen-cache-dir"),
    block_size: int = typer.Option(4096, "--block-size", min=256),
    target_lufs: float | None = typer.Option(None, "--target-lufs"),
    target_peak_dbfs: float | None = typer.Option(None, "--target-peak-dbfs"),
    true_peak: bool = typer.Option(True, "--true-peak/--sample-peak"),
    limiter: bool = typer.Option(True, "--limiter/--no-limiter"),
    normalize_stage: NormalizeStage = typer.Option("post", "--normalize-stage"),
    repeat_target_lufs: float | None = typer.Option(None, "--repeat-target-lufs"),
    repeat_target_peak_dbfs: float | None = typer.Option(None, "--repeat-target-peak-dbfs"),
    out_subtype: OutputSubtype = typer.Option(
        "auto",
        "--out-subtype",
        help="Output file subtype. Use float32 for 32-bit float WAV/AIFF where supported.",
    ),
    output_peak_norm: OutputPeakNorm = typer.Option(
        "none",
        "--output-peak-norm",
        help=(
            "Final peak normalization mode: none, input peak match, explicit target, or full-scale."
        ),
    ),
    output_peak_target_dbfs: float | None = typer.Option(
        None,
        "--output-peak-target-dbfs",
        help="Target dBFS used when --output-peak-norm target is selected.",
    ),
    shimmer: bool = typer.Option(False, "--shimmer"),
    shimmer_semitones: float = typer.Option(12.0, "--shimmer-semitones"),
    shimmer_mix: float = typer.Option(0.25, "--shimmer-mix", min=0.0, max=1.0),
    shimmer_feedback: float = typer.Option(0.35, "--shimmer-feedback", min=0.0, max=0.98),
    shimmer_highcut: float | None = typer.Option(10_000.0, "--shimmer-highcut", min=10.0),
    shimmer_lowcut: float | None = typer.Option(300.0, "--shimmer-lowcut", min=10.0),
    duck: bool = typer.Option(False, "--duck"),
    duck_attack: float = typer.Option(20.0, "--duck-attack", min=0.1),
    duck_release: float = typer.Option(350.0, "--duck-release", min=0.1),
    bloom: float = typer.Option(0.0, "--bloom", min=0.0),
    lowcut: float | None = typer.Option(None, "--lowcut", min=10.0),
    highcut: float | None = typer.Option(None, "--highcut", min=10.0),
    tilt: float = typer.Option(0.0, "--tilt"),
    frames_out: str | None = typer.Option(None, "--frames-out"),
    analysis_out: str | None = typer.Option(None, "--analysis-out"),
    silent: bool = typer.Option(False, "--silent", help="Disable analysis JSON + console output."),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    """Render input audio with algorithmic or convolution reverb."""
    resolved_pre_delay_ms = parse_pre_delay_ms(pre_delay, bpm, pre_delay_ms)

    config = RenderConfig(
        engine=engine,
        rt60=rt60,
        pre_delay_ms=resolved_pre_delay_ms,
        pre_delay_note=pre_delay,
        bpm=bpm,
        damping=damping,
        width=width,
        mod_depth_ms=mod_depth_ms,
        mod_rate_hz=mod_rate_hz,
        beast_mode=beast_mode,
        wet=wet,
        dry=dry,
        repeat=repeat,
        freeze=freeze,
        start=start,
        end=end,
        block_size=block_size,
        ir=None if ir is None else str(ir),
        self_convolve=self_convolve,
        ir_normalize=ir_normalize,
        ir_matrix_layout=ir_matrix_layout,
        tail_limit=tail_limit,
        threads=threads,
        device=device,
        partition_size=partition_size,
        ir_gen=ir_gen,
        ir_gen_mode=ir_gen_mode,
        ir_gen_length=ir_gen_length,
        ir_gen_seed=ir_gen_seed,
        ir_gen_cache_dir=ir_gen_cache_dir,
        target_lufs=target_lufs,
        target_peak_dbfs=target_peak_dbfs,
        use_true_peak=true_peak,
        limiter=limiter,
        normalize_stage=normalize_stage,
        repeat_target_lufs=repeat_target_lufs,
        repeat_target_peak_dbfs=repeat_target_peak_dbfs,
        output_subtype=out_subtype,
        output_peak_norm=output_peak_norm,
        output_peak_target_dbfs=output_peak_target_dbfs,
        shimmer=shimmer,
        shimmer_semitones=shimmer_semitones,
        shimmer_mix=shimmer_mix,
        shimmer_feedback=shimmer_feedback,
        shimmer_highcut=shimmer_highcut,
        shimmer_lowcut=shimmer_lowcut,
        duck=duck,
        duck_attack=duck_attack,
        duck_release=duck_release,
        bloom=bloom,
        lowcut=lowcut,
        highcut=highcut,
        tilt=tilt,
        frames_out=frames_out,
        analysis_out=analysis_out,
        silent=silent,
        progress=progress,
    )

    _validate_render_call(infile, outfile, config)
    configure_logging(verbose=not config.silent)

    try:
        report = run_render_pipeline(infile=infile, outfile=outfile, config=config)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if config.silent:
        return

    table = Table(title="Render Summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("requested_engine", str(report.get("effective", {}).get("engine_requested", "")))
    table.add_row("engine", str(report.get("engine", "unknown")))
    table.add_row("requested_device", str(report.get("effective", {}).get("device_requested", "")))
    table.add_row("device", str(report.get("effective", {}).get("device_resolved", "")))
    table.add_row(
        "device_platform",
        str(report.get("effective", {}).get("device_platform_resolved", "")),
    )
    table.add_row("compute_backend", str(report.get("effective", {}).get("compute_backend", "")))
    table.add_row("ir_used", str(report.get("effective", {}).get("ir_used")))
    table.add_row(
        "self_convolve",
        str(report.get("effective", {}).get("self_convolve", False)),
    )
    table.add_row("sample_rate", str(report.get("sample_rate", "")))
    table.add_row("channels", str(report.get("channels", "")))
    table.add_row("input_samples", str(report.get("input_samples", "")))
    table.add_row("output_samples", str(report.get("output_samples", "")))
    table.add_row(
        "tail_padding_s",
        str(report.get("effective", {}).get("tail_padding_seconds", "")),
    )
    table.add_row("beast_mode", str(report.get("effective", {}).get("beast_mode", 1)))
    table.add_row("out_subtype", str(report.get("effective", {}).get("output_subtype", "")))
    table.add_row("output_peak_norm", str(report.get("effective", {}).get("output_peak_norm", "")))
    table.add_row(
        "output_peak_target_dbfs",
        str(report.get("effective", {}).get("output_peak_target_dbfs", "")),
    )
    table.add_row("streaming_mode", str(report.get("effective", {}).get("streaming_mode", "")))
    config_report = report.get("config", {})
    if isinstance(config_report, dict):
        for key in (
            "rt60",
            "pre_delay_ms",
            "beast_mode",
            "wet",
            "dry",
            "repeat",
            "ir_matrix_layout",
            "self_convolve",
            "damping",
            "width",
            "block_size",
        ):
            if key in config_report:
                table.add_row(key, str(config_report[key]))
    table.add_row("analysis_json", str(report.get("analysis_path", "")))
    if "frames_path" in report:
        table.add_row("frames_csv", str(report.get("frames_path")))
    if "ir_runtime" in report:
        runtime = report["ir_runtime"]
        if isinstance(runtime, dict):
            table.add_row("ir_runtime_path", str(runtime.get("ir_path", "")))
            table.add_row("ir_cache_hit", str(runtime.get("cache_hit", False)))
    console.print(table)


@app.command()
def analyze(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    json_out: Path | None = typer.Option(None, "--json-out", resolve_path=True),
    lufs: bool = typer.Option(False, "--lufs", help="Include LUFS/true-peak/LRA metrics."),
    frames_out: Path | None = typer.Option(None, "--frames-out", resolve_path=True),
) -> None:
    """Analyze an audio file and print a summary table."""
    _validate_analyze_call(infile, json_out, frames_out)
    try:
        validate_audio_path(str(infile))
        audio, sr = read_audio(str(infile))
        analyzer = AudioAnalyzer()
        metrics = analyzer.analyze(audio, sr, include_loudness=lufs)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title=f"Analysis: {infile.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for key in sorted(metrics):
        table.add_row(key, f"{metrics[key]:.6f}")
    console.print(table)

    if json_out is not None:
        payload = {"sample_rate": sr, "channels": audio.shape[1], "metrics": metrics}
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if frames_out is not None:
        write_framewise_csv(frames_out, audio, sr)


@app.command()
def suggest(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
) -> None:
    """Suggest practical render defaults from input analysis."""
    try:
        validate_audio_path(str(infile))
        audio, sr = read_audio(str(infile))
        analyzer = AudioAnalyzer()
        metrics = analyzer.analyze(audio, sr)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

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
    table.add_row("target-lufs", "-18.0")
    table.add_row("target-peak-dbfs", "-1.0")
    table.add_row("normalize-stage", "post")
    table.add_row("shimmer", "off")
    table.add_row("duck", "off")
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


@ir_app.command("gen")
def ir_gen(
    out_ir: Path = typer.Argument(..., resolve_path=True),
    out_format: IRFileFormat = typer.Option("auto", "--format"),
    mode: IRMode = typer.Option("hybrid", "--mode"),
    length: float = typer.Option(60.0, "--length", min=0.1),
    sr: int = typer.Option(48_000, "--sr", min=8_000),
    channels: int = typer.Option(2, "--channels", min=1),
    seed: int = typer.Option(0, "--seed"),
    rt60: float | None = typer.Option(None, "--rt60", min=0.1),
    rt60_low: float | None = typer.Option(None, "--rt60-low", min=0.1),
    rt60_high: float | None = typer.Option(None, "--rt60-high", min=0.1),
    damping: float = typer.Option(0.4, "--damping", min=0.0, max=1.0),
    lowcut: float | None = typer.Option(None, "--lowcut", min=10.0),
    highcut: float | None = typer.Option(None, "--highcut", min=10.0),
    tilt: float = typer.Option(0.0, "--tilt"),
    normalize: Literal["none", "peak", "rms"] = typer.Option("peak", "--normalize"),
    peak_dbfs: float = typer.Option(-1.0, "--peak-dbfs"),
    target_lufs: float | None = typer.Option(None, "--target-lufs"),
    true_peak: bool = typer.Option(True, "--true-peak/--sample-peak"),
    er_count: int = typer.Option(24, "--er-count", min=0),
    er_max_delay_ms: float = typer.Option(90.0, "--er-max-delay-ms", min=1.0),
    er_decay_shape: str = typer.Option("exp", "--er-decay-shape"),
    er_stereo_width: float = typer.Option(1.0, "--er-stereo-width", min=0.0, max=2.0),
    er_room: float = typer.Option(1.0, "--er-room", min=0.1),
    diffusion: float = typer.Option(0.5, "--diffusion", min=0.0, max=1.0),
    mod_depth_ms: float = typer.Option(1.5, "--mod-depth-ms", min=0.0),
    mod_rate_hz: float = typer.Option(0.12, "--mod-rate-hz", min=0.0),
    density: float = typer.Option(1.0, "--density", min=0.01),
    tuning: str = typer.Option("A4=440", "--tuning"),
    modal_count: int = typer.Option(48, "--modal-count", min=1),
    modal_q_min: float = typer.Option(5.0, "--modal-q-min", min=0.5),
    modal_q_max: float = typer.Option(60.0, "--modal-q-max", min=0.5),
    modal_spread_cents: float = typer.Option(5.0, "--modal-spread-cents", min=0.0),
    modal_low_hz: float = typer.Option(80.0, "--modal-low-hz", min=20.0),
    modal_high_hz: float = typer.Option(12_000.0, "--modal-high-hz", min=50.0),
    fdn_lines: int = typer.Option(8, "--fdn-lines", min=1),
    fdn_matrix: str = typer.Option("hadamard", "--fdn-matrix"),
    fdn_stereo_inject: float = typer.Option(1.0, "--fdn-stereo-inject", min=0.0, max=1.0),
    f0: str | None = typer.Option(None, "--f0", help="e.g. 64, 64Hz, or 64 Hz"),
    analyze_input: Path | None = typer.Option(
        None,
        "--analyze-input",
        exists=True,
        readable=True,
        resolve_path=True,
        help="Input audio to estimate fundamentals/harmonics for IR tuning",
    ),
    harmonic_align_strength: float = typer.Option(
        0.75, "--harmonic-align-strength", min=0.0, max=1.0
    ),
    resonator: bool = typer.Option(
        False,
        "--resonator/--no-resonator",
        help="Enable Modalys-inspired physical modal-bank late-tail coloration.",
    ),
    resonator_mix: float = typer.Option(0.35, "--resonator-mix", min=0.0, max=1.0),
    resonator_modes: int = typer.Option(32, "--resonator-modes", min=1),
    resonator_q_min: float = typer.Option(8.0, "--resonator-q-min", min=0.5),
    resonator_q_max: float = typer.Option(90.0, "--resonator-q-max", min=0.5),
    resonator_low_hz: float = typer.Option(50.0, "--resonator-low-hz", min=20.0),
    resonator_high_hz: float = typer.Option(9000.0, "--resonator-high-hz", min=30.0),
    resonator_late_start_ms: float = typer.Option(80.0, "--resonator-late-start-ms", min=0.0),
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    """Generate an IR file with deterministic caching."""
    _validate_ir_gen_call(
        out_ir=out_ir,
        out_format=out_format,
        rt60=rt60,
        rt60_low=rt60_low,
        rt60_high=rt60_high,
        modal_q_min=modal_q_min,
        modal_q_max=modal_q_max,
        modal_low_hz=modal_low_hz,
        modal_high_hz=modal_high_hz,
        resonator_q_min=resonator_q_min,
        resonator_q_max=resonator_q_max,
        resonator_low_hz=resonator_low_hz,
        resonator_high_hz=resonator_high_hz,
    )

    f0_hz: float | None = None
    harmonic_targets_hz: tuple[float, ...] = ()

    if f0 is not None:
        try:
            f0_hz = parse_frequency_hz(f0)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    if analyze_input is not None:
        est_f0, harmonics = analyze_audio_for_tuning(analyze_input)
        if f0_hz is None:
            f0_hz = est_f0
        harmonic_targets_hz = tuple(harmonics)

    cfg = IRGenConfig(
        mode=mode,
        length=length,
        sr=sr,
        channels=channels,
        seed=seed,
        rt60=rt60,
        rt60_low=rt60_low,
        rt60_high=rt60_high,
        damping=damping,
        lowcut=lowcut,
        highcut=highcut,
        tilt=tilt,
        normalize=normalize,
        peak_dbfs=peak_dbfs,
        target_lufs=target_lufs,
        true_peak=true_peak,
        er_count=er_count,
        er_max_delay_ms=er_max_delay_ms,
        er_decay_shape=er_decay_shape,
        er_stereo_width=er_stereo_width,
        er_room=er_room,
        diffusion=diffusion,
        mod_depth_ms=mod_depth_ms,
        mod_rate_hz=mod_rate_hz,
        density=density,
        tuning=tuning,
        modal_count=modal_count,
        modal_q_min=modal_q_min,
        modal_q_max=modal_q_max,
        modal_spread_cents=modal_spread_cents,
        modal_low_hz=modal_low_hz,
        modal_high_hz=modal_high_hz,
        fdn_lines=fdn_lines,
        fdn_matrix=fdn_matrix,
        fdn_stereo_inject=fdn_stereo_inject,
        f0_hz=f0_hz,
        harmonic_targets_hz=harmonic_targets_hz,
        harmonic_align_strength=harmonic_align_strength,
        resonator=resonator,
        resonator_mix=resonator_mix,
        resonator_modes=resonator_modes,
        resonator_q_min=resonator_q_min,
        resonator_q_max=resonator_q_max,
        resonator_low_hz=resonator_low_hz,
        resonator_high_hz=resonator_high_hz,
        resonator_late_start_ms=resonator_late_start_ms,
    )

    resolved_out_ir = _resolve_ir_output_path(out_ir, out_format)

    try:
        audio, out_sr, meta, cache_path, cache_hit = generate_or_load_cached_ir(
            cfg,
            cache_dir=Path(cache_dir),
        )
        write_ir_artifacts(resolved_out_ir, audio, out_sr, meta, silent=silent)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if silent:
        return

    table = Table(title="IR Generation")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("mode", mode)
    table.add_row("out_ir", str(resolved_out_ir))
    table.add_row("format", out_format)
    table.add_row("cache_path", str(cache_path))
    table.add_row("cache_hit", str(cache_hit))
    table.add_row("duration_s", f"{audio.shape[0] / out_sr:.2f}")
    table.add_row("channels", str(audio.shape[1]))
    if f0_hz is not None:
        table.add_row("f0_hz", f"{f0_hz:.3f}")
    if analyze_input is not None:
        table.add_row("analyze_input", str(analyze_input))
        table.add_row("harmonics_detected", str(len(harmonic_targets_hz)))
    table.add_row("resonator", str(resonator))
    if resonator:
        table.add_row("resonator_mix", f"{resonator_mix:.3f}")
        table.add_row("resonator_modes", str(resonator_modes))
        table.add_row(
            "resonator_band_hz",
            f"{resonator_low_hz:.1f}-{resonator_high_hz:.1f}",
        )
        table.add_row("resonator_q", f"{resonator_q_min:.2f}-{resonator_q_max:.2f}")
        table.add_row("resonator_late_start_ms", f"{resonator_late_start_ms:.2f}")
    console.print(table)


@ir_app.command("analyze")
def ir_analyze(
    ir_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    json_out: Path | None = typer.Option(None, "--json-out", resolve_path=True),
) -> None:
    """Analyze an impulse response."""
    _validate_ir_analyze_call(ir_file, json_out)
    try:
        audio, sr = sf.read(str(ir_file), always_2d=True, dtype="float32")
        metrics = analyze_ir(np.asarray(audio, dtype=np.float32), int(sr))
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title=f"IR Analysis: {ir_file.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    for key in [
        "duration_seconds",
        "peak_dbfs",
        "rms_dbfs",
        "rt60_estimate_seconds",
        "early_late_ratio_db",
        "stereo_coherence",
    ]:
        value = metrics.get(key)
        if isinstance(value, float):
            table.add_row(key, f"{value:.6f}")
    decay_points = metrics.get("decay_curve_db", [])
    point_count = len(decay_points) if isinstance(decay_points, list) else 0
    table.add_row("decay_curve_points", str(point_count))
    console.print(table)

    if json_out is not None:
        payload = {"file": str(ir_file), "sample_rate": int(sr), "metrics": metrics}
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@ir_app.command("process")
def ir_process(
    in_ir: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_ir: Path = typer.Argument(..., resolve_path=True),
    damping: float = typer.Option(0.4, "--damping", min=0.0, max=1.0),
    lowcut: float | None = typer.Option(None, "--lowcut", min=10.0),
    highcut: float | None = typer.Option(None, "--highcut", min=10.0),
    tilt: float = typer.Option(0.0, "--tilt"),
    normalize: Literal["none", "peak", "rms"] = typer.Option("peak", "--normalize"),
    peak_dbfs: float = typer.Option(-1.0, "--peak-dbfs"),
    target_lufs: float | None = typer.Option(None, "--target-lufs"),
    true_peak: bool = typer.Option(True, "--true-peak/--sample-peak"),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    """Process an existing IR through shaping/targeting chain."""
    _validate_ir_process_call(in_ir, out_ir)
    try:
        audio, sr = sf.read(str(in_ir), always_2d=True, dtype="float32")
        processed = apply_ir_shaping(
            np.asarray(audio, dtype=np.float32),
            sr=int(sr),
            damping=damping,
            lowcut=lowcut,
            highcut=highcut,
            tilt=tilt,
            normalize=normalize,
            peak_dbfs=peak_dbfs,
            target_lufs=target_lufs,
            use_true_peak=true_peak,
        )

        meta = {"source": str(in_ir), "metrics": analyze_ir(processed, int(sr))}
        write_ir_artifacts(out_ir, processed, int(sr), meta, silent=silent)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@ir_app.command("fit")
def ir_fit(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_ir: Path = typer.Argument(..., resolve_path=True),
    top_k: int = typer.Option(3, "--top-k", min=1),
    base_mode: IRMode = typer.Option("hybrid", "--base-mode"),
    length: float = typer.Option(60.0, "--length", min=0.1),
    seed: int = typer.Option(0, "--seed"),
    candidate_pool: int = typer.Option(12, "--candidate-pool", min=1),
    fit_workers: int = typer.Option(0, "--fit-workers", min=0, help="0 = auto"),
    analyze_tuning: bool = typer.Option(True, "--analyze-tuning/--no-analyze-tuning"),
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
) -> None:
    """Analyze source audio, score candidate IRs, and write top-k results."""
    _validate_output_audio_path(out_ir, "auto")
    try:
        audio, sr = read_audio(str(infile))
        analyzer = AudioAnalyzer()
        metrics = analyzer.analyze(audio, sr)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    pool_size = max(top_k, candidate_pool)
    target_profile = derive_ir_fit_target(metrics, sr)

    f0_hz: float | None = None
    harmonics: tuple[float, ...] = ()
    if analyze_tuning:
        try:
            f0_est, harmonic_est = analyze_audio_for_tuning(infile, max_harmonics=12)
            f0_hz = f0_est
            harmonics = tuple(harmonic_est)
        except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError):
            f0_hz = None
            harmonics = ()

    candidates = build_ir_fit_candidates(
        base_mode=base_mode,
        length=length,
        sr=sr,
        channels=max(1, min(2, audio.shape[1])),
        seed=seed,
        pool_size=pool_size,
        target=target_profile,
        f0_hz=f0_hz,
        harmonic_targets_hz=harmonics,
    )

    cache_root = Path(cache_dir)
    scored = _score_fit_candidates(
        candidates=candidates,
        target=target_profile,
        cache_dir=cache_root,
        fit_workers=fit_workers,
    )

    selected = sorted(
        scored,
        key=lambda item: item.score.score,
        reverse=True,
    )[:top_k]

    created: list[str] = []
    for rank, item in enumerate(selected, start=1):
        target_path = (
            out_ir
            if top_k == 1
            else out_ir.with_name(f"{out_ir.stem}_{rank:02d}{out_ir.suffix}")
        )
        meta = dict(item.meta)
        meta["fit"] = {
            "rank": rank,
            "score": item.score.score,
            "strategy": item.candidate.strategy,
            "target": asdict(target_profile),
            "errors": asdict(item.score),
            "detail_metrics": item.detail_metrics,
        }
        cached_audio, _ = sf.read(str(item.cache_path), always_2d=True, dtype="float32")
        write_ir_artifacts(
            target_path,
            np.asarray(cached_audio, dtype=np.float32),
            item.sr,
            meta,
            silent=False,
        )
        created.append(str(target_path))

    table = Table(title="IR Fit")
    table.add_column("Field", style="green")
    table.add_column("Value", style="white")
    table.add_row("input", str(infile))
    table.add_row("top_k", str(top_k))
    table.add_row("candidate_pool", str(pool_size))
    table.add_row("target_rt60", f"{target_profile.rt60_seconds:.2f}")
    table.add_row("target_early_late_db", f"{target_profile.early_late_ratio_db:.2f}")
    table.add_row("target_coherence", f"{target_profile.stereo_coherence:.3f}")
    if f0_hz is not None:
        table.add_row("detected_f0_hz", f"{f0_hz:.3f}")
    if selected:
        table.add_row("best_score", f"{selected[0].score.score:.5f}")
        table.add_row("best_strategy", selected[0].candidate.strategy)
    table.add_row("outputs", "\n".join(created))
    console.print(table)


@cache_app.command("info")
def cache_info(
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
) -> None:
    """Show cache statistics."""
    root = Path(cache_dir)
    if root.exists() and not root.is_dir():
        msg = f"Cache path is not a directory: {root}"
        raise typer.BadParameter(msg)
    wavs = sorted(root.glob("*.wav"))
    metas = sorted(root.glob("*.meta.json"))
    total_bytes = (
        sum(path.stat().st_size for path in root.glob("*") if path.is_file())
        if root.exists()
        else 0
    )

    table = Table(title="Cache Info")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("cache_dir", str(root))
    table.add_row("wav_files", str(len(wavs)))
    table.add_row("meta_files", str(len(metas)))
    table.add_row("size_mb", f"{total_bytes / (1024 * 1024):.3f}")
    console.print(table)


@cache_app.command("clear")
def cache_clear(
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
) -> None:
    """Clear IR cache directory."""
    root = Path(cache_dir)
    if root.exists() and not root.is_dir():
        msg = f"Cache path is not a directory: {root}"
        raise typer.BadParameter(msg)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    console.print(f"Cleared cache: {root}")


@batch_app.command("template")
def batch_template() -> None:
    """Print a batch manifest template as JSON."""
    template = {
        "version": "0.4",
        "jobs": [
            {
                "infile": "input.wav",
                "outfile": "output.wav",
                "options": {
                    "engine": "auto",
                    "rt60": 60.0,
                    "wet": 0.8,
                    "dry": 0.2,
                    "repeat": 1,
                },
            }
        ],
    }
    typer.echo(json.dumps(template, indent=2))


@batch_app.command("render")
def batch_render(
    manifest: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    jobs: int = typer.Option(0, "--jobs", min=0, help="0 = auto"),
    schedule: BatchSchedulePolicy = typer.Option("longest-first", "--schedule"),
    retries: int = typer.Option(0, "--retries", min=0),
    continue_on_error: bool = typer.Option(False, "--continue-on-error/--fail-fast"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Render jobs from manifest.json."""
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON manifest: {exc}"
        raise typer.BadParameter(msg) from exc
    if not isinstance(payload, dict) or "jobs" not in payload:
        raise typer.BadParameter("Manifest must contain a top-level 'jobs' array")

    job_list = payload["jobs"]
    if not isinstance(job_list, list):
        raise typer.BadParameter("jobs must be a list")

    prepared_jobs: list[BatchJobSpec] = []

    for idx, job in enumerate(job_list, start=1):
        if not isinstance(job, dict):
            raise typer.BadParameter(f"jobs[{idx - 1}] must be an object")
        infile = Path(str(job.get("infile", "")))
        outfile = Path(str(job.get("outfile", "")))
        options = job.get("options", {})
        if not isinstance(options, dict):
            raise typer.BadParameter(f"jobs[{idx - 1}].options must be an object")

        try:
            render_config = _render_config_from_options(options)
        except (TypeError, ValueError) as exc:
            msg = f"jobs[{idx - 1}] has invalid options: {exc}"
            raise typer.BadParameter(msg) from exc
        _validate_batch_job_paths(infile, outfile, idx)
        prepared_jobs.append(
            BatchJobSpec(
                index=idx,
                infile=infile,
                outfile=outfile,
                config=render_config,
                estimated_cost=estimate_job_cost(infile, render_config),
            )
        )

    if dry_run:
        ordered = order_jobs(prepared_jobs, schedule)
        for job in ordered:
            console.print(
                "[dry-run] job "
                f"{job.index}: {job.infile} -> {job.outfile} "
                f"(cost={job.estimated_cost:.2f}, schedule={schedule})"
            )
        return

    max_workers = int(os.cpu_count() or 1) if jobs == 0 else int(jobs)
    max_workers = max(1, min(max_workers, len(prepared_jobs)))

    def runner(job: BatchJobSpec) -> None:
        run_render_pipeline(infile=job.infile, outfile=job.outfile, config=job.config)

    def on_result(result: BatchJobResult) -> None:
        if result.success:
            console.print(
                f"rendered job {result.index}: {result.outfile} "
                f"(attempts={result.attempts}, {result.duration_seconds:.2f}s)"
            )
        else:
            console.print(
                f"failed job {result.index}: {result.outfile} "
                f"(attempts={result.attempts}) {result.error}"
            )

    try:
        run_parallel_batch(
            jobs=prepared_jobs,
            max_workers=max_workers,
            schedule=schedule,
            retries=retries,
            continue_on_error=continue_on_error,
            runner=runner,
            on_result=on_result,
        )
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _render_config_from_options(options: dict[str, Any]) -> RenderConfig:
    fields = RenderConfig.__dataclass_fields__.keys()
    filtered = {key: value for key, value in options.items() if key in fields}

    pre_delay_note = filtered.get("pre_delay_note")
    bpm = filtered.get("bpm")
    if isinstance(pre_delay_note, str):
        fallback_ms = float(filtered.get("pre_delay_ms", 20.0))
        resolved_bpm = float(bpm) if isinstance(bpm, (float, int)) else None
        filtered["pre_delay_ms"] = parse_pre_delay_ms(pre_delay_note, resolved_bpm, fallback_ms)

    return RenderConfig(**filtered)


class _ScoredFitCandidate:
    def __init__(
        self,
        *,
        candidate: IRFitCandidate,
        score: IRFitScore,
        detail_metrics: dict[str, float],
        sr: int,
        meta: dict[str, Any],
        cache_path: Path,
    ) -> None:
        self.candidate = candidate
        self.score = score
        self.detail_metrics = detail_metrics
        self.sr = int(sr)
        self.meta = meta
        self.cache_path = cache_path


def _score_fit_candidates(
    *,
    candidates: list[IRFitCandidate],
    target: IRFitTarget,
    cache_dir: Path,
    fit_workers: int,
) -> list[_ScoredFitCandidate]:
    worker_count = int(os.cpu_count() or 1) if fit_workers == 0 else fit_workers
    worker_count = max(1, min(worker_count, len(candidates)))

    def evaluate(candidate: IRFitCandidate) -> _ScoredFitCandidate:
        audio, sr, meta, cache_path, _ = generate_or_load_cached_ir(
            candidate.config,
            cache_dir=cache_dir,
        )
        score, detail_metrics = score_ir_candidate(ir_audio=audio, sr=sr, target=target)
        return _ScoredFitCandidate(
            candidate=candidate,
            score=score,
            detail_metrics=detail_metrics,
            sr=sr,
            meta=meta,
            cache_path=cache_path,
        )

    if worker_count == 1:
        return [evaluate(candidate) for candidate in candidates]

    scored: list[_ScoredFitCandidate] = []
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="verbx-fit") as pool:
        futures: list[Future[_ScoredFitCandidate]] = [
            pool.submit(evaluate, candidate) for candidate in candidates
        ]
        for fut in as_completed(futures):
            scored.append(fut.result())
    return scored


def _resolve_ir_output_path(out_ir: Path, out_format: IRFileFormat) -> Path:
    """Resolve output IR path based on explicit format switch."""
    if out_format == "auto":
        return out_ir if out_ir.suffix else out_ir.with_suffix(".wav")

    suffix = ".aiff" if out_format == "aiff" else f".{out_format}"
    return out_ir.with_suffix(suffix)


def _validate_render_call(infile: Path, outfile: Path, config: RenderConfig) -> None:
    _ensure_distinct_paths(infile, outfile, "INFILE", "OUTFILE")
    _validate_output_audio_path(outfile, config.output_subtype)

    if config.self_convolve:
        if config.ir is not None:
            msg = "Use either --ir or --self-convolve, not both."
            raise typer.BadParameter(msg)
        if config.ir_gen:
            msg = "Use either --ir-gen or --self-convolve, not both."
            raise typer.BadParameter(msg)
        if config.engine == "algo":
            msg = "--self-convolve is only valid with --engine conv or --engine auto."
            raise typer.BadParameter(msg)

    if (
        config.engine == "conv"
        and config.ir is None
        and not config.ir_gen
        and not config.self_convolve
    ):
        msg = "Convolution render requires --ir PATH, --ir-gen, or --self-convolve."
        raise typer.BadParameter(msg)

    if config.ir is not None and not Path(config.ir).exists():
        msg = f"IR file not found: {config.ir}"
        raise typer.BadParameter(msg)

    if config.wet == 0.0 and config.dry == 0.0:
        msg = "At least one of --wet or --dry must be non-zero."
        raise typer.BadParameter(msg)

    if config.freeze:
        if config.start is None or config.end is None:
            msg = "--freeze requires both --start and --end."
            raise typer.BadParameter(msg)
        if config.end <= config.start:
            msg = "--end must be greater than --start when --freeze is enabled."
            raise typer.BadParameter(msg)
    elif config.start is not None or config.end is not None:
        msg = "--start/--end are only valid when --freeze is enabled."
        raise typer.BadParameter(msg)

    if config.output_peak_norm == "target" and config.output_peak_target_dbfs is None:
        msg = "--output-peak-norm target requires --output-peak-target-dbfs."
        raise typer.BadParameter(msg)
    if config.output_peak_norm != "target" and config.output_peak_target_dbfs is not None:
        msg = "--output-peak-target-dbfs is only valid with --output-peak-norm target."
        raise typer.BadParameter(msg)

    if config.ir_gen and config.ir is not None:
        msg = "Use either --ir or --ir-gen, not both."
        raise typer.BadParameter(msg)


def _validate_analyze_call(infile: Path, json_out: Path | None, frames_out: Path | None) -> None:
    if json_out is not None and infile.resolve() == json_out.resolve():
        msg = "--json-out must be different from input file."
        raise typer.BadParameter(msg)
    if frames_out is not None and infile.resolve() == frames_out.resolve():
        msg = "--frames-out must be different from input file."
        raise typer.BadParameter(msg)


def _validate_ir_gen_call(
    out_ir: Path,
    out_format: IRFileFormat,
    rt60: float | None,
    rt60_low: float | None,
    rt60_high: float | None,
    modal_q_min: float,
    modal_q_max: float,
    modal_low_hz: float,
    modal_high_hz: float,
    resonator_q_min: float,
    resonator_q_max: float,
    resonator_low_hz: float,
    resonator_high_hz: float,
) -> None:
    resolved = _resolve_ir_output_path(out_ir, out_format)
    _validate_output_audio_path(resolved, "auto")

    if rt60 is not None and (rt60_low is not None or rt60_high is not None):
        msg = "Use either --rt60 or --rt60-low/--rt60-high, not both."
        raise typer.BadParameter(msg)
    if (rt60_low is None) != (rt60_high is None):
        msg = "Both --rt60-low and --rt60-high must be provided together."
        raise typer.BadParameter(msg)
    if rt60_low is not None and rt60_high is not None and rt60_low > rt60_high:
        msg = "--rt60-low must be <= --rt60-high."
        raise typer.BadParameter(msg)
    if modal_q_min > modal_q_max:
        msg = "--modal-q-min must be <= --modal-q-max."
        raise typer.BadParameter(msg)
    if modal_low_hz >= modal_high_hz:
        msg = "--modal-low-hz must be < --modal-high-hz."
        raise typer.BadParameter(msg)
    if resonator_q_min > resonator_q_max:
        msg = "--resonator-q-min must be <= --resonator-q-max."
        raise typer.BadParameter(msg)
    if resonator_low_hz >= resonator_high_hz:
        msg = "--resonator-low-hz must be < --resonator-high-hz."
        raise typer.BadParameter(msg)


def _validate_ir_process_call(in_ir: Path, out_ir: Path) -> None:
    _ensure_distinct_paths(in_ir, out_ir, "IN_IR", "OUT_IR")
    _validate_output_audio_path(out_ir, "auto")


def _validate_ir_analyze_call(ir_file: Path, json_out: Path | None) -> None:
    if json_out is not None and ir_file.resolve() == json_out.resolve():
        msg = "--json-out must be different from input IR file."
        raise typer.BadParameter(msg)


def _validate_batch_job_paths(infile: Path, outfile: Path, idx: int) -> None:
    if infile.resolve() == outfile.resolve():
        msg = f"jobs[{idx - 1}] infile and outfile must be different."
        raise typer.BadParameter(msg)
    if not infile.exists():
        msg = f"jobs[{idx - 1}] infile not found: {infile}"
        raise typer.BadParameter(msg)
    _validate_output_audio_path(outfile, "auto")


def _ensure_distinct_paths(in_path: Path, out_path: Path, in_label: str, out_label: str) -> None:
    if in_path.resolve() == out_path.resolve():
        msg = f"{in_label} and {out_label} must be different paths."
        raise typer.BadParameter(msg)


def _validate_output_audio_path(path: Path, out_subtype_mode: str) -> None:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "":
        msg = f"Output path must include an audio file extension: {path}"
        raise typer.BadParameter(msg)

    format_map = {
        "wav": "WAV",
        "flac": "FLAC",
        "aif": "AIFF",
        "aiff": "AIFF",
        "ogg": "OGG",
        "caf": "CAF",
        "au": "AU",
    }
    fmt = format_map.get(suffix)
    if fmt is None:
        msg = f"Unsupported output audio extension: .{suffix}"
        raise typer.BadParameter(msg)

    subtype_map = {
        "auto": None,
        "float32": "FLOAT",
        "float64": "DOUBLE",
        "pcm16": "PCM_16",
        "pcm24": "PCM_24",
        "pcm32": "PCM_32",
    }
    subtype = subtype_map.get(out_subtype_mode)
    if out_subtype_mode not in subtype_map:
        msg = f"Unsupported --out-subtype value: {out_subtype_mode}"
        raise typer.BadParameter(msg)

    if subtype is None:
        if not sf.check_format(fmt):
            msg = f"SoundFile cannot write format '{fmt}' for output path {path}"
            raise typer.BadParameter(msg)
    else:
        if not sf.check_format(fmt, subtype):
            msg = f"Subtype '{subtype}' is not supported for format '{fmt}'"
            raise typer.BadParameter(msg)


if __name__ == "__main__":
    app()
