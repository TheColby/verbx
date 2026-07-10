# ruff: noqa: B008
"""Extracted IR command implementations.

These implementations intentionally reuse a few legacy CLI helpers via lazy
imports from ``verbx.cli`` so command behavior stays identical while the public
CLI surface continues moving out of the monolithic entrypoint file.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import soundfile as sf
import typer
from rich.table import Table

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.commands.validators import validate_output_audio_path
from verbx.config import IRMode, IRMorphMismatchPolicy
from verbx.io.audio import read_audio
from verbx.ir.fitting import build_ir_fit_candidates, derive_ir_fit_target
from verbx.ir.generator import write_ir_artifacts
from verbx.ir.metrics import analyze_ir
from verbx.ir.morph import IRMorphConfig, generate_or_load_cached_morphed_ir
from verbx.ir.shaping import apply_ir_shaping
from verbx.ir.sofa import extract_sofa_ir, read_sofa_info
from verbx.ir.trace import (
    generate_trace_ir,
    parse_dxf_room_outline,
    parse_trace_vector,
    write_trace_report,
)
from verbx.ir.tuning import analyze_audio_for_tuning


def _cli() -> Any:
    from verbx import cli as cli_module

    return cli_module


def ir_analyze_impl(
    ir_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    json_out: Path | None = typer.Option(None, "--json-out", resolve_path=True),
) -> None:
    """Analyze an impulse response."""
    cli_module = _cli()
    cli_module._validate_ir_analyze_call(ir_file, json_out)
    try:
        with cli_module._processing_status("Analyze IR"):
            audio, sr = sf.read(str(ir_file), always_2d=True, dtype="float64")
            metrics = analyze_ir(np.asarray(audio, dtype=np.float64), int(sr))
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
    cli_module.console.print(table)

    if json_out is not None:
        payload = {"file": str(ir_file), "sample_rate": int(sr), "metrics": metrics}
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ir_sofa_info_impl(
    sofa_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    json_out: Path | None = typer.Option(None, "--json-out", resolve_path=True),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    """Inspect SOFA metadata and dimensions."""
    cli_module = _cli()
    try:
        with cli_module._processing_status("Read SOFA metadata", enabled=not silent):
            info = read_sofa_info(sofa_file)
    except (ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    payload = {
        "path": info.path,
        "conventions": info.conventions,
        "version": info.version,
        "data_ir_shape": list(info.data_ir_shape),
        "sample_rate_hz": int(info.sample_rate_hz),
        "source_position_shape": (
            None if info.source_position_shape is None else list(info.source_position_shape)
        ),
        "listener_position_shape": (
            None if info.listener_position_shape is None else list(info.listener_position_shape)
        ),
        "receiver_position_shape": (
            None if info.receiver_position_shape is None else list(info.receiver_position_shape)
        ),
        "emitter_position_shape": (
            None if info.emitter_position_shape is None else list(info.emitter_position_shape)
        ),
        "dimension_labels": list(info.dimension_labels),
    }

    if not silent:
        table = Table(title=f"SOFA Info: {Path(info.path).name}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("conventions", info.conventions)
        table.add_row("version", info.version)
        table.add_row("sample_rate_hz", str(int(info.sample_rate_hz)))
        table.add_row("data_ir_shape", str(info.data_ir_shape))
        if len(info.dimension_labels) > 0:
            table.add_row("dimension_labels", ", ".join(info.dimension_labels))
        if info.source_position_shape is not None:
            table.add_row("source_position_shape", str(info.source_position_shape))
        if info.listener_position_shape is not None:
            table.add_row("listener_position_shape", str(info.listener_position_shape))
        if info.receiver_position_shape is not None:
            table.add_row("receiver_position_shape", str(info.receiver_position_shape))
        if info.emitter_position_shape is not None:
            table.add_row("emitter_position_shape", str(info.emitter_position_shape))
        cli_module.console.print(table)

    if json_out is not None:
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ir_sofa_extract_impl(
    sofa_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_ir: Path = typer.Argument(..., resolve_path=True),
    measurement_index: int = typer.Option(
        0,
        "--measurement-index",
        min=0,
        help="Measurement index for SOFA Data/IR extraction (first axis in strict modes).",
    ),
    emitter_index: int = typer.Option(
        0,
        "--emitter-index",
        min=0,
        help="Emitter index for rank-4 Data/IR extraction (strict mode).",
    ),
    target_sr: int | None = typer.Option(
        None,
        "--target-sr",
        min=1,
        help="Optional output sample rate target for extracted IR.",
    ),
    normalize: Literal["none", "peak", "rms"] = typer.Option(
        "peak",
        "--normalize",
        help="Normalization for extracted IR matrix.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict/--best-effort",
        help="Strict expects Data/IR rank 3 (M,R,N) or 4 (M,R,E,N).",
    ),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    """Extract SOFA FIR data to a WAV matrix for convolution workflows."""
    cli_module = _cli()
    validate_output_audio_path(out_ir, "auto")
    try:
        with cli_module._processing_status("Extract SOFA IR", enabled=not silent):
            audio, sr, meta = extract_sofa_ir(
                sofa_file,
                measurement_index=int(measurement_index),
                emitter_index=int(emitter_index),
                target_sr=None if target_sr is None else int(target_sr),
                normalize=str(normalize),
                strict=bool(strict),
            )
            write_ir_artifacts(out_ir, audio, sr, meta, silent=silent)
    except (ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if silent:
        return

    table = Table(title=f"SOFA Extract: {out_ir.name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("source", str(sofa_file))
    table.add_row("out_ir", str(out_ir))
    table.add_row("sample_rate_hz", str(int(sr)))
    table.add_row("shape", str(tuple(int(v) for v in audio.shape)))
    table.add_row("normalize", str(normalize))
    table.add_row("strict", str(bool(strict)))
    sample_rate_action = str(meta.get("sample_rate_action", "none"))
    table.add_row("sample_rate_action", sample_rate_action)
    cli_module.console.print(table)


def ir_trace_impl(
    dxf_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_ir: Path = typer.Argument(..., resolve_path=True),
    source: str = typer.Option(..., "--source"),
    listener: str = typer.Option(..., "--listener"),
    height: float = typer.Option(3.0, "--height", min=0.1),
    material: str = typer.Option("studio", "--material"),
    rays: int = typer.Option(50_000, "--rays", min=1),
    length: float = typer.Option(4.0, "--length", min=0.05),
    target_sr: int = typer.Option(48_000, "--target-sr", min=8_000),
    seed: int = typer.Option(0, "--seed"),
    json_out: Path | None = typer.Option(None, "--json-out", resolve_path=True),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    """Generate an experimental room IR from a constrained DXF room outline."""
    cli_module = _cli()
    validate_output_audio_path(out_ir, "auto")
    try:
        source_pos = parse_trace_vector(source, label="--source")
        listener_pos = parse_trace_vector(listener, label="--listener")
        with cli_module._processing_status("Trace DXF room", enabled=not silent):
            geometry = parse_dxf_room_outline(dxf_file, height_m=float(height))
            audio, report = generate_trace_ir(
                geometry=geometry,
                source_pos_m=source_pos,
                listener_pos_m=listener_pos,
                material=str(material),
                rays=int(rays),
                length_s=float(length),
                sr=int(target_sr),
                seed=int(seed),
            )
            meta = {
                "mode": "trace",
                "source": str(dxf_file),
                "trace_report": report,
            }
            write_ir_artifacts(out_ir, audio, int(target_sr), meta, silent=silent)
            report_path = (
                out_ir.with_suffix(f"{out_ir.suffix}.trace.json")
                if json_out is None
                else json_out
            )
            write_trace_report(report_path, report)
    except (ValueError, RuntimeError, OSError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if silent:
        return

    room = report["geometry"]["room"]
    trace = report["trace"]
    table = Table(title="IR Trace")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("schema", str(report["schema"]))
    table.add_row("experimental", str(bool(report["experimental"])))
    table.add_row("out_ir", str(out_ir))
    table.add_row("report", str(report_path))
    table.add_row("room_dims_m", str(room["room_dims_m"]))
    table.add_row("material", str(report["material"]["default"]))
    table.add_row("rays", str(trace["rays"]))
    table.add_row("sample_rate", str(trace["target_sr"]))
    table.add_row("length_s", f"{float(trace['length_s']):.3f}")
    table.add_row("estimated_rt60_s", f"{float(trace['estimated_rt60_s']):.3f}")
    table.add_row("reflection_count", str(trace["reflection_count"]))
    warnings = report["geometry"].get("warnings", [])
    if isinstance(warnings, list) and len(warnings) > 0:
        table.add_row("warnings", "\n".join(str(item) for item in warnings))
    cli_module.console.print(table)


def ir_process_impl(
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
    cli_module = _cli()
    cli_module._validate_ir_process_call(in_ir, out_ir)
    cli_module._validate_generic_lucky_call(lucky, lucky_out_dir)
    try:
        with cli_module._processing_status("Load IR for processing", enabled=not silent):
            audio, sr = sf.read(str(in_ir), always_2d=True, dtype="float64")
            base_audio = np.asarray(audio, dtype=np.float64)
            sr_i = int(sr)
        if lucky is None:
            with cli_module._processing_status("Process IR", enabled=not silent):
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
        seed_value = cli_module._resolve_lucky_seed(lucky_seed)

        rows: list[dict[str, str]] = []
        with cli_module._BatchStatusBar(
            total=lucky,
            label="Lucky IR processing",
            enabled=not silent,
        ) as status:
            for idx in range(lucky):
                rng = np.random.default_rng(seed_value + idx)
                cfg = cli_module._build_lucky_ir_process_config(
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
                status.advance(detail=f"seed={seed_value + idx}")

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
            cli_module.console.print(table)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def ir_morph_impl(
    ir_a: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    ir_b: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_ir: Path = typer.Argument(..., resolve_path=True),
    mode: str = typer.Option(
        "equal-power",
        "--mode",
        help="Morph mode: linear, equal-power, spectral, or envelope-aware.",
    ),
    alpha: float = typer.Option(0.5, "--alpha", min=0.0, max=1.0),
    early_ms: float = typer.Option(
        80.0,
        "--early-ms",
        min=0.0,
        help="Early/late split used by split/envelope-aware morphing (ms).",
    ),
    early_alpha: float | None = typer.Option(
        None,
        "--early-alpha",
        min=0.0,
        max=1.0,
        help="Optional alpha override for early-reflection region.",
    ),
    late_alpha: float | None = typer.Option(
        None,
        "--late-alpha",
        min=0.0,
        max=1.0,
        help="Optional alpha override for late-tail region.",
    ),
    align_decay: bool = typer.Option(
        True,
        "--align-decay/--no-align-decay",
        help="Align decay profiles before morphing for stable RT trajectories.",
    ),
    phase_coherence: float = typer.Option(
        0.75,
        "--phase-coherence",
        min=0.0,
        max=1.0,
        help="Phase-coherence safeguard strength for spectral morphing.",
    ),
    spectral_smooth_bins: int = typer.Option(
        3,
        "--spectral-smooth-bins",
        min=0,
        max=128,
        help="Frequency smoothing radius (FFT bins) used by spectral modes.",
    ),
    mismatch_policy: IRMorphMismatchPolicy = typer.Option(
        "coerce",
        "--mismatch-policy",
        help=(
            "Mismatch behavior for sample-rate/channel/duration differences: "
            "coerce (align) or strict (fail)."
        ),
    ),
    target_sr: int | None = typer.Option(
        None,
        "--target-sr",
        min=1,
        help="Optional target sample rate for morph processing and output.",
    ),
    cache_dir: str = typer.Option(".verbx_cache/ir_morph", "--cache-dir"),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    """Morph two IR files with cache-backed Track D processing."""
    cli_module = _cli()
    cli_module._validate_ir_morph_call(
        ir_a=ir_a,
        ir_b=ir_b,
        out_ir=out_ir,
        mode=mode,
        early_alpha=early_alpha,
        late_alpha=late_alpha,
        mismatch_policy=mismatch_policy,
        cache_dir=cache_dir,
    )

    cfg = IRMorphConfig(
        mode=cast(
            Literal["linear", "equal-power", "spectral", "envelope-aware"],
            cli_module.validate_ir_morph_mode_name(mode),
        ),
        alpha=float(alpha),
        early_ms=float(early_ms),
        early_alpha=None if early_alpha is None else float(early_alpha),
        late_alpha=None if late_alpha is None else float(late_alpha),
        align_decay=bool(align_decay),
        phase_coherence=float(phase_coherence),
        spectral_smooth_bins=int(spectral_smooth_bins),
        mismatch_policy=cast(
            IRMorphMismatchPolicy,
            cli_module.normalize_ir_morph_mismatch_policy_name(mismatch_policy),
        ),
    )

    try:
        with cli_module._processing_status("Morph IRs", enabled=not silent):
            audio, sr, meta, cache_path, cache_hit = generate_or_load_cached_morphed_ir(
                ir_a_path=ir_a,
                ir_b_path=ir_b,
                config=cfg,
                cache_dir=Path(cache_dir),
                target_sr=None if target_sr is None else int(target_sr),
            )
            write_ir_artifacts(out_ir, audio, sr, meta, silent=silent)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if silent:
        return

    table = Table(title="IR Morph")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("mode", cfg.mode)
    table.add_row("alpha", f"{cfg.alpha:.3f}")
    table.add_row("early_ms", f"{cfg.early_ms:.2f}")
    table.add_row("mismatch_policy", str(cfg.mismatch_policy))
    table.add_row("out_ir", str(out_ir))
    table.add_row("cache_path", str(cache_path))
    table.add_row("cache_hit", str(cache_hit))
    table.add_row("sample_rate", str(int(sr)))
    table.add_row("channels", str(int(audio.shape[1])))
    table.add_row("duration_s", f"{float(audio.shape[0]) / float(sr):.3f}")
    quality = meta.get("quality", {})
    if isinstance(quality, dict):
        drift = quality.get("rt60_drift_s")
        if drift is not None:
            table.add_row("rt60_drift_s", f"{float(drift):.4f}")
        spectral = quality.get("spectral_distance_db")
        if spectral is not None:
            table.add_row("spectral_distance_db", f"{float(spectral):.4f}")
    cli_module.console.print(table)


def ir_fit_impl(
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
    cli_module = _cli()
    validate_output_audio_path(out_ir, "auto")
    try:
        with cli_module._processing_status("Analyze source for IR fit"):
            audio, sr = read_audio(str(infile))
            analyzer = AudioAnalyzer()
            metrics = analyzer.analyze(audio, sr)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    pool_size = max(top_k, candidate_pool)
    numeric_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }
    target_profile = derive_ir_fit_target(numeric_metrics, sr)

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
    scored = cli_module._score_fit_candidates(
        candidates=candidates,
        target=target_profile,
        cache_dir=cache_root,
        fit_workers=fit_workers,
        show_progress=True,
    )

    selected = sorted(
        scored,
        key=lambda item: item.score.score,
        reverse=True,
    )[:top_k]

    created: list[str] = []
    with cli_module._BatchStatusBar(
        total=len(selected),
        label="Write fitted IRs",
        enabled=True,
    ) as status:
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
            cached_audio, _ = sf.read(str(item.cache_path), always_2d=True, dtype="float64")
            write_ir_artifacts(
                target_path,
                np.asarray(cached_audio, dtype=np.float64),
                item.sr,
                meta,
                silent=False,
            )
            created.append(str(target_path))
            status.advance(detail=f"rank={rank}")

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
    cli_module.console.print(table)
