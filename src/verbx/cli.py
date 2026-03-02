"""Typer CLI entrypoint for verbx.

Commands are grouped by workflow:
- top-level render/analyze/suggest/presets
- ``ir`` synthesis/inspection
- ``cache`` management
- ``batch`` orchestration
"""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

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


class LuckyIRProcessConfig(TypedDict):
    """Typed config payload for ``ir process --lucky`` randomization."""

    damping: float
    lowcut: float | None
    highcut: float | None
    tilt: float
    normalize: Literal["none", "peak", "rms"]
    peak_dbfs: float
    target_lufs: float | None
    true_peak: bool

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
    lucky: int | None = typer.Option(
        None,
        "--lucky",
        min=1,
        max=500,
        help=(
            "Generate N wild random renders from one input using randomized parameters. "
            "Outputs are written to --lucky-out-dir (or OUTFILE parent by default)."
        ),
    ),
    lucky_out_dir: Path | None = typer.Option(
        None,
        "--lucky-out-dir",
        resolve_path=True,
        help="Output directory used when --lucky is enabled.",
    ),
    lucky_seed: int | None = typer.Option(
        None,
        "--lucky-seed",
        help="Optional deterministic seed for --lucky render generation.",
    ),
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
    _validate_lucky_call(config, lucky, lucky_out_dir)
    configure_logging(verbose=not config.silent)

    if lucky is not None:
        try:
            info = sf.info(str(infile))
            duration_seconds = (
                float(info.frames) / float(info.samplerate) if info.samplerate > 0 else 0.0
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc

        out_dir = outfile.parent if lucky_out_dir is None else lucky_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        seed = _resolve_lucky_seed(lucky_seed)

        lucky_rows: list[dict[str, str]] = []
        for idx in range(lucky):
            rng = np.random.default_rng(seed + idx)
            lucky_config = _build_lucky_config(
                base=config,
                rng=rng,
                input_duration_seconds=duration_seconds,
            )
            lucky_config.progress = config.progress
            lucky_out = out_dir / f"{outfile.stem}.lucky_{idx + 1:03d}{outfile.suffix}"

            try:
                report = run_render_pipeline(infile=infile, outfile=lucky_out, config=lucky_config)
            except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
                raise typer.BadParameter(str(exc)) from exc

            lucky_rows.append(
                {
                    "index": str(idx + 1),
                    "outfile": str(lucky_out),
                    "engine": str(report.get("engine", "unknown")),
                    "rt60": f"{float(lucky_config.rt60):.2f}",
                    "repeat": str(int(lucky_config.repeat)),
                    "beast": str(int(lucky_config.beast_mode)),
                }
            )

        if not config.silent:
            summary = Table(title=f"Lucky Render Batch ({lucky} outputs)")
            summary.add_column("#", style="cyan", justify="right")
            summary.add_column("outfile", style="white")
            summary.add_column("engine", style="green")
            summary.add_column("rt60", justify="right")
            summary.add_column("repeat", justify="right")
            summary.add_column("beast", justify="right")
            for row in lucky_rows:
                summary.add_row(
                    row["index"],
                    row["outfile"],
                    row["engine"],
                    row["rt60"],
                    row["repeat"],
                    row["beast"],
                )
            console.print(summary)
        return

    try:
        report = run_render_pipeline(infile=infile, outfile=outfile, config=config)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if config.silent:
        return

    _print_render_summary(report)


@app.command()
def analyze(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    json_out: Path | None = typer.Option(None, "--json-out", resolve_path=True),
    lufs: bool = typer.Option(False, "--lufs", help="Include LUFS/true-peak/LRA metrics."),
    edr: bool = typer.Option(
        False,
        "--edr",
        help="Include EDR (Energy Decay Relief) summary metrics.",
    ),
    frames_out: Path | None = typer.Option(None, "--frames-out", resolve_path=True),
) -> None:
    """Analyze an audio file and print a summary table."""
    _validate_analyze_call(infile, json_out, frames_out)
    try:
        validate_audio_path(str(infile))
        audio, sr = read_audio(str(infile))
        analyzer = AudioAnalyzer()
        metrics = analyzer.analyze(audio, sr, include_loudness=lufs, include_edr=edr)
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
    lucky: int | None = typer.Option(
        None,
        "--lucky",
        min=1,
        max=500,
        help=(
            "Generate N randomized IR files from one base setup. "
            "Outputs are written to --lucky-out-dir (or OUT_IR parent by default)."
        ),
    ),
    lucky_out_dir: Path | None = typer.Option(
        None,
        "--lucky-out-dir",
        resolve_path=True,
        help="Output directory used when --lucky is enabled.",
    ),
    lucky_seed: int | None = typer.Option(
        None,
        "--lucky-seed",
        help="Optional deterministic seed for --lucky IR generation.",
    ),
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
    _validate_generic_lucky_call(lucky, lucky_out_dir)

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

    if lucky is not None:
        out_dir = resolved_out_ir.parent if lucky_out_dir is None else lucky_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        seed_value = _resolve_lucky_seed(lucky_seed)

        rows: list[dict[str, str]] = []
        for idx in range(lucky):
            rng = np.random.default_rng(seed_value + idx)
            lucky_cfg = _build_lucky_ir_gen_config(cfg, rng=rng)
            lucky_out = (
                out_dir / f"{resolved_out_ir.stem}.lucky_{idx + 1:03d}{resolved_out_ir.suffix}"
            )
            try:
                audio, out_sr, meta, cache_path, cache_hit = generate_or_load_cached_ir(
                    lucky_cfg,
                    cache_dir=Path(cache_dir),
                )
                write_ir_artifacts(lucky_out, audio, out_sr, meta, silent=silent)
            except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
                raise typer.BadParameter(str(exc)) from exc

            rows.append(
                {
                    "index": str(idx + 1),
                    "out_ir": str(lucky_out),
                    "mode": lucky_cfg.mode,
                    "length_s": f"{float(lucky_cfg.length):.2f}",
                    "rt60": (
                        f"{float(lucky_cfg.rt60):.2f}"
                        if lucky_cfg.rt60 is not None
                        else (
                            f"{float(lucky_cfg.rt60_low or 0.0):.2f}-"
                            f"{float(lucky_cfg.rt60_high or 0.0):.2f}"
                        )
                    ),
                    "cache_hit": str(cache_hit),
                    "cache_path": str(cache_path),
                }
            )

        if not silent:
            table = Table(title=f"Lucky IR Generation Batch ({lucky} outputs)")
            table.add_column("#", style="cyan", justify="right")
            table.add_column("out_ir", style="white")
            table.add_column("mode", style="green")
            table.add_column("length_s", justify="right")
            table.add_column("rt60", justify="right")
            table.add_column("cache_hit", justify="right")
            for row in rows:
                table.add_row(
                    row["index"],
                    row["out_ir"],
                    row["mode"],
                    row["length_s"],
                    row["rt60"],
                    row["cache_hit"],
                )
            console.print(table)
        return

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
    lucky: int | None = typer.Option(
        None,
        "--lucky",
        min=1,
        max=500,
        help=(
            "Generate N randomized processed IR files from one input IR. "
            "Outputs are written to --lucky-out-dir (or OUT_IR parent by default)."
        ),
    ),
    lucky_out_dir: Path | None = typer.Option(
        None,
        "--lucky-out-dir",
        resolve_path=True,
        help="Output directory used when --lucky is enabled.",
    ),
    lucky_seed: int | None = typer.Option(
        None,
        "--lucky-seed",
        help="Optional deterministic seed for --lucky IR processing.",
    ),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    """Process an existing IR through shaping/targeting chain."""
    _validate_ir_process_call(in_ir, out_ir)
    _validate_generic_lucky_call(lucky, lucky_out_dir)
    try:
        audio, sr = sf.read(str(in_ir), always_2d=True, dtype="float32")
        base_audio = np.asarray(audio, dtype=np.float32)
        sr_i = int(sr)
        if lucky is None:
            processed = apply_ir_shaping(
                base_audio,
                sr=sr_i,
                damping=damping,
                lowcut=lowcut,
                highcut=highcut,
                tilt=tilt,
                normalize=normalize,
                peak_dbfs=peak_dbfs,
                target_lufs=target_lufs,
                use_true_peak=true_peak,
            )

            meta = {"source": str(in_ir), "metrics": analyze_ir(processed, sr_i)}
            write_ir_artifacts(out_ir, processed, sr_i, meta, silent=silent)
            return

        out_dir = out_ir.parent if lucky_out_dir is None else lucky_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        seed_value = _resolve_lucky_seed(lucky_seed)

        rows: list[dict[str, str]] = []
        for idx in range(lucky):
            rng = np.random.default_rng(seed_value + idx)
            cfg = _build_lucky_ir_process_config(
                damping=damping,
                lowcut=lowcut,
                highcut=highcut,
                tilt=tilt,
                normalize=normalize,
                peak_dbfs=peak_dbfs,
                target_lufs=target_lufs,
                true_peak=true_peak,
                rng=rng,
                sr=sr_i,
            )
            lucky_out = out_dir / f"{out_ir.stem}.lucky_{idx + 1:03d}{out_ir.suffix}"
            processed = apply_ir_shaping(
                base_audio,
                sr=sr_i,
                damping=cfg["damping"],
                lowcut=cfg["lowcut"],
                highcut=cfg["highcut"],
                tilt=cfg["tilt"],
                normalize=cfg["normalize"],
                peak_dbfs=cfg["peak_dbfs"],
                target_lufs=cfg["target_lufs"],
                use_true_peak=cfg["true_peak"],
            )

            meta = {
                "source": str(in_ir),
                "lucky": {"index": idx + 1, **cfg},
                "metrics": analyze_ir(processed, sr_i),
            }
            write_ir_artifacts(lucky_out, processed, sr_i, meta, silent=silent)
            rows.append(
                {
                    "index": str(idx + 1),
                    "out_ir": str(lucky_out),
                    "normalize": cfg["normalize"],
                    "tilt": f"{float(cfg['tilt']):.2f}",
                    "damping": f"{float(cfg['damping']):.2f}",
                    "target_lufs": (
                        f"{float(cfg['target_lufs']):.2f}"
                        if cfg["target_lufs"] is not None
                        else "none"
                    ),
                }
            )

        if not silent:
            table = Table(title=f"Lucky IR Process Batch ({lucky} outputs)")
            table.add_column("#", style="cyan", justify="right")
            table.add_column("out_ir", style="white")
            table.add_column("normalize", style="green")
            table.add_column("tilt", justify="right")
            table.add_column("damping", justify="right")
            table.add_column("target_lufs", justify="right")
            for row in rows:
                table.add_row(
                    row["index"],
                    row["out_ir"],
                    row["normalize"],
                    row["tilt"],
                    row["damping"],
                    row["target_lufs"],
                )
            console.print(table)
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
    lucky: int | None = typer.Option(
        None,
        "--lucky",
        min=1,
        max=500,
        help=(
            "For each manifest job, generate N wild random render variants. "
            "Outputs are written to --lucky-out-dir (or each job OUTFILE parent by default)."
        ),
    ),
    lucky_out_dir: Path | None = typer.Option(
        None,
        "--lucky-out-dir",
        resolve_path=True,
        help="Output directory used when --lucky is enabled.",
    ),
    lucky_seed: int | None = typer.Option(
        None,
        "--lucky-seed",
        help="Optional deterministic seed for --lucky batch generation.",
    ),
) -> None:
    """Render jobs from manifest.json."""
    _validate_generic_lucky_call(lucky, lucky_out_dir)
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
    prepared_index = 1
    lucky_seed_value = _resolve_lucky_seed(lucky_seed) if lucky is not None else 0

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
        if lucky is None:
            prepared_jobs.append(
                BatchJobSpec(
                    index=prepared_index,
                    infile=infile,
                    outfile=outfile,
                    config=render_config,
                    estimated_cost=estimate_job_cost(infile, render_config),
                )
            )
            prepared_index += 1
            continue

        try:
            info = sf.info(str(infile))
            duration_seconds = (
                float(info.frames) / float(info.samplerate) if info.samplerate > 0 else 0.0
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            msg = f"jobs[{idx - 1}] failed to inspect infile: {exc}"
            raise typer.BadParameter(msg) from exc

        out_dir = outfile.parent if lucky_out_dir is None else lucky_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for variant_idx in range(lucky):
            rng = np.random.default_rng(lucky_seed_value + ((idx - 1) * lucky) + variant_idx)
            lucky_config = _build_lucky_config(
                base=render_config,
                rng=rng,
                input_duration_seconds=duration_seconds,
            )
            lucky_config.progress = False
            lucky_config.analysis_out = None
            lucky_config.frames_out = None
            lucky_out = out_dir / f"{outfile.stem}.lucky_{variant_idx + 1:03d}{outfile.suffix}"
            prepared_jobs.append(
                BatchJobSpec(
                    index=prepared_index,
                    infile=infile,
                    outfile=lucky_out,
                    config=lucky_config,
                    estimated_cost=estimate_job_cost(infile, lucky_config),
                )
            )
            prepared_index += 1

    if not prepared_jobs:
        raise typer.BadParameter("Manifest has no jobs to render")

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
    """Build ``RenderConfig`` from manifest options with safe field filtering."""
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
    """Internal transport object for IR-fit ranking results."""

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
    """Evaluate IR-fit candidates serially or in parallel."""
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


def _print_render_summary(report: dict[str, Any]) -> None:
    """Print the standard single-render summary table."""
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


def _build_lucky_config(
    base: RenderConfig,
    rng: np.random.Generator,
    input_duration_seconds: float,
) -> RenderConfig:
    """Build one randomized high-intensity render config for ``--lucky`` mode."""
    cfg = RenderConfig(**asdict(base))

    cfg.engine = cast(EngineName, rng.choice(np.array(["algo", "conv", "auto"], dtype=object)))
    cfg.beast_mode = int(rng.integers(1, 7))
    cfg.rt60 = float(rng.uniform(0.8, 12.0))
    cfg.pre_delay_ms = float(rng.uniform(0.0, 500.0))
    cfg.damping = float(rng.uniform(0.0, 1.0))
    cfg.width = float(rng.uniform(0.0, 2.0))
    cfg.mod_depth_ms = float(rng.uniform(0.0, 25.0))
    cfg.mod_rate_hz = float(rng.uniform(0.02, 2.0))
    cfg.wet = float(rng.uniform(0.55, 1.0))
    cfg.dry = float(rng.uniform(0.0, 0.6))
    cfg.repeat = int(rng.integers(1, 4))
    cfg.partition_size = int(rng.choice(np.array([4096, 8192, 16384, 32768], dtype=np.int32)))
    cfg.block_size = int(rng.choice(np.array([1024, 2048, 4096, 8192], dtype=np.int32)))
    cfg.progress = False
    cfg.frames_out = None
    cfg.analysis_out = None

    cfg.freeze = bool(input_duration_seconds > 1.5 and rng.random() < 0.33)
    if cfg.freeze:
        max_start = max(0.0, input_duration_seconds - 0.25)
        start = float(rng.uniform(0.0, max_start))
        max_len = max(0.2, min(4.0, input_duration_seconds * 0.45))
        end = float(min(input_duration_seconds, start + rng.uniform(0.2, max_len)))
        if end <= start:
            end = min(input_duration_seconds, start + 0.2)
        cfg.start = start
        cfg.end = end
    else:
        cfg.start = None
        cfg.end = None

    cfg.target_lufs = float(rng.uniform(-30.0, -10.0)) if rng.random() < 0.45 else None
    cfg.target_peak_dbfs = float(rng.uniform(-9.0, -0.6)) if rng.random() < 0.7 else None
    cfg.use_true_peak = bool(rng.random() < 0.8)
    cfg.normalize_stage = cast(
        NormalizeStage,
        rng.choice(np.array(["none", "post", "per-pass"], dtype=object)),
    )
    if cfg.normalize_stage == "per-pass":
        cfg.repeat_target_lufs = (
            float(rng.uniform(-30.0, -12.0)) if rng.random() < 0.55 else cfg.target_lufs
        )
        cfg.repeat_target_peak_dbfs = (
            float(rng.uniform(-10.0, -0.8)) if rng.random() < 0.6 else cfg.target_peak_dbfs
        )
    else:
        cfg.repeat_target_lufs = None
        cfg.repeat_target_peak_dbfs = None

    cfg.shimmer = bool(rng.random() < 0.42)
    cfg.shimmer_semitones = float(rng.choice(np.array([7.0, 12.0, 19.0, 24.0], dtype=np.float32)))
    cfg.shimmer_mix = float(rng.uniform(0.05, 0.85))
    cfg.shimmer_feedback = float(rng.uniform(0.05, 0.95))
    cfg.shimmer_lowcut = float(rng.uniform(80.0, 600.0))
    cfg.shimmer_highcut = float(rng.uniform(2_000.0, 16_000.0))

    cfg.duck = bool(rng.random() < 0.45)
    cfg.duck_attack = float(rng.uniform(2.0, 120.0))
    cfg.duck_release = float(rng.uniform(80.0, 1200.0))
    cfg.bloom = float(rng.uniform(0.0, 6.0))
    cfg.tilt = float(rng.uniform(-8.0, 8.0))

    if rng.random() < 0.6:
        cfg.lowcut = float(rng.uniform(20.0, 500.0))
    else:
        cfg.lowcut = None
    if rng.random() < 0.8:
        min_high = (cfg.lowcut + 300.0) if cfg.lowcut is not None else 800.0
        cfg.highcut = float(rng.uniform(min_high, 20_000.0))
    else:
        cfg.highcut = None

    # Ensure convolution path has a valid IR source whenever selected.
    if cfg.engine in {"conv", "auto"} and rng.random() < 0.75:
        mode = int(rng.integers(0, 3))
        if mode == 0 and base.ir is not None:
            cfg.ir = base.ir
            cfg.ir_gen = False
            cfg.self_convolve = False
        elif mode == 1:
            cfg.self_convolve = True
            cfg.ir = None
            cfg.ir_gen = False
        else:
            cfg.ir_gen = True
            cfg.ir = None
            cfg.self_convolve = False
            cfg.ir_gen_mode = cast(
                IRMode,
                rng.choice(np.array(["hybrid", "fdn", "modal", "stochastic"], dtype=object)),
            )
            cfg.ir_gen_length = float(rng.uniform(5.0, 60.0))
            cfg.ir_gen_seed = int(rng.integers(0, 2_147_483_647))
    else:
        cfg.ir = None
        cfg.ir_gen = False
        cfg.self_convolve = False
        if cfg.engine == "conv":
            cfg.engine = "algo"

    cfg.tail_limit = float(rng.uniform(2.0, 40.0)) if rng.random() < 0.4 else None
    return cfg


def _build_lucky_ir_gen_config(
    base: IRGenConfig,
    rng: np.random.Generator,
) -> IRGenConfig:
    """Build one randomized IR generation config for ``ir gen --lucky``."""
    cfg = IRGenConfig(**asdict(base))

    cfg.mode = cast(
        IRMode,
        rng.choice(np.array(["hybrid", "fdn", "stochastic", "modal"], dtype=object)),
    )
    cfg.seed = int(rng.integers(0, 2_147_483_647))
    cfg.length = float(np.clip(base.length * rng.uniform(0.5, 2.5), 0.1, 120.0))
    cfg.rt60 = float(np.clip((base.rt60 or 12.0) * rng.uniform(0.4, 2.5), 0.1, 180.0))
    cfg.rt60_low = None
    cfg.rt60_high = None

    cfg.damping = float(rng.uniform(0.0, 1.0))
    cfg.diffusion = float(rng.uniform(0.05, 1.0))
    cfg.density = float(rng.uniform(0.1, 2.2))
    cfg.mod_depth_ms = float(rng.uniform(0.0, 18.0))
    cfg.mod_rate_hz = float(rng.uniform(0.02, 1.2))

    cfg.lowcut = float(rng.uniform(20.0, 500.0)) if rng.random() < 0.6 else None
    if rng.random() < 0.8:
        min_high = (cfg.lowcut + 300.0) if cfg.lowcut is not None else 800.0
        cfg.highcut = float(rng.uniform(min_high, (cfg.sr * 0.48)))
    else:
        cfg.highcut = None
    cfg.tilt = float(rng.uniform(-8.0, 8.0))

    cfg.normalize = cast(
        Literal["none", "peak", "rms"],
        rng.choice(np.array(["none", "peak", "rms"], dtype=object)),
    )
    cfg.peak_dbfs = float(rng.uniform(-12.0, -0.5))
    cfg.target_lufs = float(rng.uniform(-32.0, -12.0)) if rng.random() < 0.45 else None
    cfg.true_peak = bool(rng.random() < 0.7)

    cfg.er_count = int(rng.integers(0, 96))
    cfg.er_max_delay_ms = float(rng.uniform(5.0, 180.0))
    cfg.er_decay_shape = cast(str, rng.choice(np.array(["exp", "linear", "sqrt"], dtype=object)))
    cfg.er_stereo_width = float(rng.uniform(0.0, 2.0))
    cfg.er_room = float(rng.uniform(0.1, 3.0))

    cfg.modal_count = int(rng.integers(8, 128))
    cfg.modal_q_min = float(rng.uniform(0.8, 20.0))
    cfg.modal_q_max = float(rng.uniform(max(cfg.modal_q_min + 0.5, 5.0), 120.0))
    cfg.modal_spread_cents = float(rng.uniform(0.0, 40.0))
    cfg.modal_low_hz = float(rng.uniform(30.0, 400.0))
    cfg.modal_high_hz = float(rng.uniform(max(cfg.modal_low_hz + 200.0, 1200.0), cfg.sr * 0.48))

    cfg.fdn_lines = int(rng.choice(np.array([4, 6, 8, 10, 12, 16], dtype=np.int32)))
    cfg.fdn_matrix = cast(
        str,
        rng.choice(np.array(["hadamard", "householder", "random_orthogonal"], dtype=object)),
    )
    cfg.fdn_stereo_inject = float(rng.uniform(0.0, 1.0))

    cfg.harmonic_align_strength = float(rng.uniform(0.0, 1.0))
    cfg.resonator = bool(rng.random() < 0.45)
    cfg.resonator_mix = float(rng.uniform(0.0, 0.95))
    cfg.resonator_modes = int(rng.integers(8, 80))
    cfg.resonator_q_min = float(rng.uniform(0.8, 24.0))
    cfg.resonator_q_max = float(rng.uniform(max(cfg.resonator_q_min + 0.5, 6.0), 140.0))
    cfg.resonator_low_hz = float(rng.uniform(20.0, 250.0))
    cfg.resonator_high_hz = float(
        rng.uniform(max(cfg.resonator_low_hz + 500.0, 1500.0), cfg.sr * 0.48)
    )
    cfg.resonator_late_start_ms = float(rng.uniform(0.0, 400.0))
    return cfg


def _build_lucky_ir_process_config(
    *,
    damping: float,
    lowcut: float | None,
    highcut: float | None,
    tilt: float,
    normalize: Literal["none", "peak", "rms"],
    peak_dbfs: float,
    target_lufs: float | None,
    true_peak: bool,
    rng: np.random.Generator,
    sr: int,
) -> LuckyIRProcessConfig:
    """Build one randomized shaping config for ``ir process --lucky``."""
    lucky_lowcut = lowcut if lowcut is not None else float(rng.uniform(20.0, 500.0))
    lucky_lowcut = float(np.clip(lucky_lowcut * rng.uniform(0.5, 2.0), 20.0, sr * 0.45))

    if highcut is None:
        min_high = min(sr * 0.48, lucky_lowcut + 100.0)
        lucky_highcut = float(rng.uniform(max(min_high, 400.0), sr * 0.49))
    else:
        lucky_highcut = float(np.clip(highcut * rng.uniform(0.5, 1.7), 200.0, sr * 0.49))
    if lucky_highcut <= lucky_lowcut:
        lucky_highcut = min(sr * 0.49, lucky_lowcut + 200.0)

    modes = np.array(["none", "peak", "rms"], dtype=object)
    normalize_mode = (
        cast(Literal["none", "peak", "rms"], rng.choice(modes))
        if rng.random() < 0.85
        else normalize
    )
    return {
        "damping": float(np.clip(damping * rng.uniform(0.4, 2.2), 0.0, 1.0)),
        "lowcut": lucky_lowcut if rng.random() < 0.8 else None,
        "highcut": lucky_highcut if rng.random() < 0.9 else None,
        "tilt": float(np.clip(tilt + rng.uniform(-6.0, 6.0), -12.0, 12.0)),
        "normalize": normalize_mode,
        "peak_dbfs": float(np.clip(peak_dbfs + rng.uniform(-8.0, 2.0), -18.0, -0.1)),
        "target_lufs": (
            float(np.clip((target_lufs or -22.0) + rng.uniform(-10.0, 10.0), -36.0, -8.0))
            if rng.random() < 0.6
            else None
        ),
        "true_peak": bool(rng.random() < 0.7 if true_peak else rng.random() < 0.4),
    }


def _resolve_lucky_seed(lucky_seed: int | None) -> int:
    """Resolve deterministic seed for lucky-mode batches."""
    if lucky_seed is not None:
        return int(lucky_seed)
    return int(np.random.default_rng().integers(0, 2_147_483_647))


def _resolve_ir_output_path(out_ir: Path, out_format: IRFileFormat) -> Path:
    """Resolve output IR path based on explicit format switch."""
    if out_format == "auto":
        return out_ir if out_ir.suffix else out_ir.with_suffix(".wav")

    suffix = ".aiff" if out_format == "aiff" else f".{out_format}"
    return out_ir.with_suffix(suffix)


def _validate_lucky_call(
    config: RenderConfig,
    lucky: int | None,
    lucky_out_dir: Path | None,
) -> None:
    """Validate lucky-mode options for randomized batch rendering."""
    _validate_generic_lucky_call(lucky, lucky_out_dir)
    if config.analysis_out is not None:
        msg = "Do not use --analysis-out with --lucky (analysis files are per-output by default)."
        raise typer.BadParameter(msg)
    if config.frames_out is not None:
        msg = "Do not use --frames-out with --lucky."
        raise typer.BadParameter(msg)


def _validate_generic_lucky_call(lucky: int | None, lucky_out_dir: Path | None) -> None:
    """Validate generic lucky-mode options shared by multiple commands."""
    if lucky is None:
        return
    if lucky < 1:
        msg = "--lucky must be >= 1."
        raise typer.BadParameter(msg)
    if lucky_out_dir is not None and lucky_out_dir.exists() and not lucky_out_dir.is_dir():
        msg = f"--lucky-out-dir is not a directory: {lucky_out_dir}"
        raise typer.BadParameter(msg)


def _validate_render_call(infile: Path, outfile: Path, config: RenderConfig) -> None:
    """Validate render CLI arguments before pipeline execution."""
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
    """Validate analyze command output paths."""
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
    """Validate IR generation options and output path constraints."""
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
    """Validate IR process command path arguments."""
    _ensure_distinct_paths(in_ir, out_ir, "IN_IR", "OUT_IR")
    _validate_output_audio_path(out_ir, "auto")


def _validate_ir_analyze_call(ir_file: Path, json_out: Path | None) -> None:
    """Validate IR analyze optional output path."""
    if json_out is not None and ir_file.resolve() == json_out.resolve():
        msg = "--json-out must be different from input IR file."
        raise typer.BadParameter(msg)


def _validate_batch_job_paths(infile: Path, outfile: Path, idx: int) -> None:
    """Validate one batch job's input/output paths."""
    if infile.resolve() == outfile.resolve():
        msg = f"jobs[{idx - 1}] infile and outfile must be different."
        raise typer.BadParameter(msg)
    if not infile.exists():
        msg = f"jobs[{idx - 1}] infile not found: {infile}"
        raise typer.BadParameter(msg)
    _validate_output_audio_path(outfile, "auto")


def _ensure_distinct_paths(in_path: Path, out_path: Path, in_label: str, out_label: str) -> None:
    """Ensure input and output paths are not identical."""
    if in_path.resolve() == out_path.resolve():
        msg = f"{in_label} and {out_label} must be different paths."
        raise typer.BadParameter(msg)


def _validate_output_audio_path(path: Path, out_subtype_mode: str) -> None:
    """Validate output extension and requested SoundFile subtype support."""
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
