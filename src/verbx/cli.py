"""Typer CLI for verbx."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
import typer
from rich.console import Console
from rich.table import Table

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.analysis.framewise import write_framewise_csv
from verbx.config import EngineName, IRMode, IRNormalize, NormalizeStage, RenderConfig
from verbx.core.pipeline import run_render_pipeline
from verbx.core.tempo import parse_pre_delay_ms
from verbx.io.audio import read_audio, validate_audio_path
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
    ir: Path | None = typer.Option(None, "--ir", exists=True, readable=True, resolve_path=True),
    ir_normalize: IRNormalize = typer.Option("peak", "--ir-normalize"),
    tail_limit: float | None = typer.Option(None, "--tail-limit", min=0.0),
    threads: int | None = typer.Option(None, "--threads", min=1),
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
    validate_audio_path(str(infile))
    audio, sr = read_audio(str(infile))
    analyzer = AudioAnalyzer()
    metrics = analyzer.analyze(audio, sr, include_loudness=lufs)

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
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    """Generate an IR file with deterministic caching."""
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
    )

    resolved_out_ir = _resolve_ir_output_path(out_ir, out_format)

    audio, out_sr, meta, cache_path, cache_hit = generate_or_load_cached_ir(
        cfg,
        cache_dir=Path(cache_dir),
    )
    write_ir_artifacts(resolved_out_ir, audio, out_sr, meta, silent=silent)

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
    console.print(table)


@ir_app.command("analyze")
def ir_analyze(
    ir_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    json_out: Path | None = typer.Option(None, "--json-out", resolve_path=True),
) -> None:
    """Analyze an impulse response."""
    audio, sr = sf.read(str(ir_file), always_2d=True, dtype="float32")
    metrics = analyze_ir(np.asarray(audio, dtype=np.float32), int(sr))

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


@ir_app.command("fit")
def ir_fit(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_ir: Path = typer.Argument(..., resolve_path=True),
    top_k: int = typer.Option(3, "--top-k", min=1),
    base_mode: IRMode = typer.Option("hybrid", "--base-mode"),
    length: float = typer.Option(60.0, "--length", min=0.1),
    seed: int = typer.Option(0, "--seed"),
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
) -> None:
    """Analyze input and synthesize top-k candidate IRs."""
    audio, sr = read_audio(str(infile))
    analyzer = AudioAnalyzer()
    metrics = analyzer.analyze(audio, sr)

    rt60 = float(np.clip(metrics["duration"] * 1.4, 6.0, 120.0))
    modes: list[IRMode] = [base_mode, "hybrid", "fdn", "stochastic", "modal"]

    created: list[str] = []
    for idx in range(top_k):
        mode = modes[idx % len(modes)]
        cfg = IRGenConfig(
            mode=mode,
            length=length,
            sr=sr,
            channels=min(2, audio.shape[1]),
            seed=seed + idx,
            rt60=rt60,
            damping=0.45,
            normalize="peak",
        )
        ir_audio, ir_sr, meta, _, _ = generate_or_load_cached_ir(cfg, cache_dir=Path(cache_dir))

        if top_k == 1:
            target = out_ir
        else:
            target = out_ir.with_name(f"{out_ir.stem}_{idx + 1:02d}{out_ir.suffix}")

        write_ir_artifacts(target, ir_audio, ir_sr, meta, silent=False)
        created.append(str(target))

    table = Table(title="IR Fit")
    table.add_column("Field", style="green")
    table.add_column("Value", style="white")
    table.add_row("input", str(infile))
    table.add_row("top_k", str(top_k))
    table.add_row("estimated_rt60", f"{rt60:.2f}")
    table.add_row("outputs", "\n".join(created))
    console.print(table)


@cache_app.command("info")
def cache_info(
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
) -> None:
    """Show cache statistics."""
    root = Path(cache_dir)
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
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    console.print(f"Cleared cache: {root}")


@batch_app.command("template")
def batch_template() -> None:
    """Print a batch manifest template as JSON."""
    template = {
        "version": "0.3",
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
    jobs: int = typer.Option(1, "--jobs", min=1),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Render jobs from manifest.json."""
    _ = jobs  # v0.3: parsed and accepted; processing remains sequential for determinism.

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "jobs" not in payload:
        raise typer.BadParameter("Manifest must contain a top-level 'jobs' array")

    job_list = payload["jobs"]
    if not isinstance(job_list, list):
        raise typer.BadParameter("jobs must be a list")

    for idx, job in enumerate(job_list, start=1):
        if not isinstance(job, dict):
            raise typer.BadParameter(f"jobs[{idx - 1}] must be an object")
        infile = Path(str(job.get("infile", "")))
        outfile = Path(str(job.get("outfile", "")))
        options = job.get("options", {})
        if not isinstance(options, dict):
            raise typer.BadParameter(f"jobs[{idx - 1}].options must be an object")

        render_config = _render_config_from_options(options)

        if dry_run:
            console.print(f"[dry-run] job {idx}: {infile} -> {outfile}")
            continue

        run_render_pipeline(infile=infile, outfile=outfile, config=render_config)
        console.print(f"rendered job {idx}: {outfile}")


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


def _resolve_ir_output_path(out_ir: Path, out_format: IRFileFormat) -> Path:
    """Resolve output IR path based on explicit format switch."""
    if out_format == "auto":
        return out_ir if out_ir.suffix else out_ir.with_suffix(".wav")

    suffix = ".aiff" if out_format == "aiff" else f".{out_format}"
    return out_ir.with_suffix(suffix)


if __name__ == "__main__":
    app()
