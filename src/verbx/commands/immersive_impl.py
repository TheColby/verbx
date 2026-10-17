# ruff: noqa: B008
"""Extracted ``immersive`` command implementations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console
from rich.table import Table

from verbx.commands.batch_impl import (
    render_config_from_options,
    validate_batch_job_paths,
)
from verbx.commands.common import processing_status as _processing_status
from verbx.core.immersive import (
    QueueWorkerConfig,
    build_qc_gates,
    evaluate_immersive_qc,
    generate_immersive_handoff_package,
    run_file_queue_worker,
    summarize_file_queue,
    validate_layout_hint,
)
from verbx.core.pipeline import run_render_pipeline
from verbx.core.schema_versions import IMMERSIVE_QUEUE_VERSION
from verbx.io.audio import read_audio

console = Console()


def immersive_template_impl() -> None:
    """Print an immersive scene handoff template as JSON."""
    template = {
        "scene_name": "feature_episode_01",
        "sample_rate": 48_000,
        "bed": {
            "name": "bed_main",
            "path": "renders/bed_7p1p2.wav",
            "layout": "7.1.2",
            "render_options": {"wet": 0.75, "rt60": 4.5},
        },
        "objects": [
            {
                "id": "obj_001",
                "name": "lead_vox",
                "path": "renders/obj_lead_vox.wav",
                "layout": "mono",
                "start_s": 0.0,
                "gain_db": 0.0,
                "x": 0.05,
                "y": 0.0,
                "z": 0.0,
                "render_options": {"wet": 0.45, "rt60": 2.8},
            }
        ],
        "policy": {
            "mode": "bed-safe",
            "max_bed_wet": 0.85,
            "max_object_wet": 0.6,
            "max_object_rt60": 12.0,
            "downmix_max_delta_db": 4.0,
        },
        "qc_gates": {
            "target_lufs": -18.0,
            "lufs_tolerance": 3.0,
            "max_true_peak_dbfs": -1.0,
            "max_fold_down_delta_db": 4.0,
            "min_channel_occupancy": 0.34,
            "occupancy_threshold_dbfs": -45.0,
        },
        "deliverables": {
            "adm_sidecar": True,
            "object_stem_manifest": True,
            "qa_bundle": True,
        },
    }
    typer.echo(json.dumps(template, indent=2))


def immersive_handoff_impl(
    scene_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_dir: Path = typer.Argument(..., resolve_path=True),
    strict: bool = typer.Option(
        True,
        "--strict/--warn-only",
        help="Fail if policy/QC errors are detected.",
    ),
) -> None:
    """Generate immersive handoff sidecars and deliverable manifests."""
    try:
        payload = json.loads(scene_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid scene JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("Scene file must contain a JSON object.")

    try:
        with _processing_status("Build immersive handoff package"):
            summary = generate_immersive_handoff_package(
                scene=cast(dict[str, Any], payload),
                out_dir=out_dir,
                strict=strict,
            )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    outputs_raw = summary.get("outputs", {})
    outputs = outputs_raw if isinstance(outputs_raw, dict) else {}
    policy_raw = summary.get("policy", {})
    policy = policy_raw if isinstance(policy_raw, dict) else {}
    qa_raw = summary.get("qa_summary", {})
    qa = qa_raw if isinstance(qa_raw, dict) else {}
    validation_raw = summary.get("validation", {})
    validation = validation_raw if isinstance(validation_raw, dict) else {}
    validation_errors = validation.get("errors", [])
    validation_warnings = validation.get("warnings", [])
    failed_tracks = qa.get("failed_tracks", [])

    table = Table(title="Immersive Handoff")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("scene", str(summary.get("scene_name", "")))
    table.add_row("strict", str(bool(summary.get("strict", False))))
    table.add_row("qa_all_pass", str(bool(qa.get("all_pass", False))))
    table.add_row("policy_mode", str(policy.get("mode", "")))
    table.add_row("policy_errors", str(len(policy.get("errors", []))))
    table.add_row("policy_warnings", str(len(policy.get("warnings", []))))
    table.add_row("validation_errors", str(len(validation_errors)))
    table.add_row("validation_warnings", str(len(validation_warnings)))
    table.add_row("failed_tracks", str(len(failed_tracks)))
    table.add_row(
        "outputs",
        "\n".join(str(path) for path in outputs.values()) if len(outputs) > 0 else "(none)",
    )
    console.print(table)


def immersive_qc_impl(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    layout: str = typer.Option(
        "auto",
        "--layout",
        help=(
            "Channel layout hint: auto, mono, stereo, lcr, 5.1, 7.1, "
            "7.1.2, 7.1.4, 7.2.4, 8.0, 16.0, 64.4"
        ),
    ),
    target_lufs: float = typer.Option(-18.0, "--target-lufs"),
    lufs_tolerance: float = typer.Option(3.0, "--lufs-tolerance", min=0.0),
    max_true_peak_dbfs: float = typer.Option(-1.0, "--max-true-peak-dbfs"),
    max_fold_down_delta_db: float = typer.Option(4.0, "--max-fold-down-delta-db", min=0.0),
    min_channel_occupancy: float = typer.Option(
        0.34,
        "--min-channel-occupancy",
        min=0.0,
        max=1.0,
    ),
    occupancy_threshold_dbfs: float = typer.Option(
        -45.0,
        "--occupancy-threshold-dbfs",
    ),
    json_out: Path | None = typer.Option(
        None,
        "--json-out",
        resolve_path=True,
        help="Optional output path for QC JSON payload.",
    ),
    fail_on_violation: bool = typer.Option(
        False,
        "--fail-on-violation",
        help="Exit with code 2 when any QC gate fails.",
    ),
) -> None:
    """Run immersive QC gates for loudness/true-peak/fold-down/occupancy."""
    try:
        with _processing_status("Run immersive QC"):
            audio, sr = read_audio(str(infile))
            resolved_layout = validate_layout_hint(layout)
            gates = build_qc_gates(
                {
                    "target_lufs": target_lufs,
                    "lufs_tolerance": lufs_tolerance,
                    "max_true_peak_dbfs": max_true_peak_dbfs,
                    "max_fold_down_delta_db": max_fold_down_delta_db,
                    "min_channel_occupancy": min_channel_occupancy,
                    "occupancy_threshold_dbfs": occupancy_threshold_dbfs,
                }
            )
            report = evaluate_immersive_qc(
                audio=audio,
                sr=sr,
                label=infile.stem,
                layout=resolved_layout,
                gates=gates,
            )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    metrics_raw = report.get("metrics", {})
    metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
    passes_raw = report.get("passes", {})
    passes = passes_raw if isinstance(passes_raw, dict) else {}

    table = Table(title="Immersive QC")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("file", str(infile))
    table.add_row("layout", str(report.get("layout", "")))
    table.add_row("channels", str(report.get("channels", "")))
    table.add_row("integrated_lufs", f"{float(metrics.get('integrated_lufs', 0.0)):.2f}")
    table.add_row("true_peak_dbfs", f"{float(metrics.get('true_peak_dbfs', 0.0)):.2f}")
    table.add_row("fold_down_delta_db", f"{float(metrics.get('fold_down_delta_db', 0.0)):.2f}")
    table.add_row("channel_occupancy", f"{float(metrics.get('channel_occupancy', 0.0)):.3f}")
    table.add_row("loudness_gate", str(bool(passes.get("loudness", False))))
    table.add_row("true_peak_gate", str(bool(passes.get("true_peak", False))))
    table.add_row("fold_down_gate", str(bool(passes.get("fold_down_delta", False))))
    table.add_row("occupancy_gate", str(bool(passes.get("channel_occupancy", False))))
    table.add_row("all_pass", str(bool(report.get("pass", False))))
    console.print(table)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if fail_on_violation and (not bool(report.get("pass", False))):
        raise typer.Exit(code=2)


def immersive_queue_template_impl() -> None:
    """Print a file-backed immersive queue template as JSON."""
    template = {
        "version": IMMERSIVE_QUEUE_VERSION,
        "backend": "file",
        "jobs": [
            {
                "id": "job_0001",
                "infile": "input.wav",
                "outfile": "renders/output.wav",
                "max_retries": 1,
                "options": {
                    "engine": "algo",
                    "rt60": 3.0,
                    "wet": 0.7,
                    "dry": 0.3,
                    "progress": False,
                },
            }
        ],
    }
    typer.echo(json.dumps(template, indent=2))


def immersive_queue_status_impl(
    queue_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
) -> None:
    """Show file-queue state summary."""
    try:
        status = summarize_file_queue(queue_file)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title="Immersive Queue Status")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("queue", str(status.get("queue_path", "")))
    table.add_row("state_root", str(status.get("state_root", "")))
    table.add_row("total_jobs", str(status.get("total_jobs", 0)))
    table.add_row("success_jobs", str(status.get("success_jobs", 0)))
    table.add_row("failed_jobs", str(status.get("failed_jobs", 0)))
    table.add_row("claimed_jobs", str(status.get("claimed_jobs", 0)))
    table.add_row("pending_jobs", str(status.get("pending_jobs", 0)))
    console.print(table)


def immersive_queue_worker_impl(
    queue_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    worker_id: str | None = typer.Option(
        None,
        "--worker-id",
        help="Worker identifier. Defaults to host PID-based value.",
    ),
    heartbeat_dir: Path = typer.Option(
        Path(".verbx_queue_heartbeats"),
        "--heartbeat-dir",
        resolve_path=True,
        help="Directory for per-worker heartbeat JSON files.",
    ),
    poll_ms: int = typer.Option(800, "--poll-ms", min=50),
    max_jobs: int = typer.Option(0, "--max-jobs", min=0, help="0 = run until queue drain"),
    stale_claim_seconds: float = typer.Option(120.0, "--stale-claim-seconds", min=1.0),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--fail-fast"),
    fail_if_any_failed: bool = typer.Option(
        False,
        "--fail-if-any-failed",
        help="Exit with code 2 if any queue jobs end in failed state.",
    ),
) -> None:
    """Run one distributed queue worker for immersive batch execution."""
    resolved_worker_id = (
        worker_id
        if worker_id is not None and str(worker_id).strip() != ""
        else f"worker_{os.getpid()}"
    )

    config = QueueWorkerConfig(
        worker_id=resolved_worker_id,
        heartbeat_dir=heartbeat_dir,
        poll_ms=poll_ms,
        max_jobs=max_jobs,
        stale_claim_seconds=stale_claim_seconds,
        continue_on_error=continue_on_error,
    )

    def runner(job: Any) -> None:
        if not isinstance(job, dict):
            raise ValueError("queue runner received invalid job payload")
        infile = Path(str(job.get("infile", "")))
        outfile = Path(str(job.get("outfile", "")))
        options_raw = job.get("options")
        options = options_raw if isinstance(options_raw, dict) else {}
        render_config = render_config_from_options(options)
        validate_batch_job_paths(infile, outfile, 1)
        run_render_pipeline(infile=infile, outfile=outfile, config=render_config)

    try:
        with _processing_status("Run immersive queue worker"):
            summary = run_file_queue_worker(
                queue_path=queue_file,
                runner=runner,
                config=config,
            )
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title="Immersive Queue Worker")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("worker_id", str(summary.get("worker_id", "")))
    table.add_row("queue_path", str(summary.get("queue_path", "")))
    table.add_row("state_root", str(summary.get("state_root", "")))
    table.add_row("processed", str(summary.get("processed", 0)))
    table.add_row("success", str(summary.get("success", 0)))
    table.add_row("retried", str(summary.get("retried", 0)))
    table.add_row("failed", str(summary.get("failed", 0)))
    table.add_row("last_error", str(summary.get("last_error", "")))
    console.print(table)

    if fail_if_any_failed and int(summary.get("failed", 0)) > 0:
        raise typer.Exit(code=2)
