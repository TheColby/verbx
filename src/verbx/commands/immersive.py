# ruff: noqa: B008
"""Immersive command wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer


def _forward(name: str, params: dict[str, Any]) -> None:
    from verbx import cli as cli_module

    return cli_module.get_command_impl(name)(**params)


def immersive_template() -> None:
    _forward("_immersive_template_impl", {})


def immersive_handoff(
    scene_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_dir: Path = typer.Argument(..., resolve_path=True),
    strict: bool = typer.Option(
        True,
        "--strict/--warn-only",
        help="Fail if policy/QC errors are detected.",
    ),
) -> None:
    _forward("_immersive_handoff_impl", dict(locals()))


def immersive_qc(
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
    _forward("_immersive_qc_impl", dict(locals()))


def immersive_queue_template() -> None:
    _forward("_immersive_queue_template_impl", {})


def immersive_queue_status(
    queue_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
) -> None:
    _forward("_immersive_queue_status_impl", dict(locals()))


def immersive_queue_worker(
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
    _forward("_immersive_queue_worker_impl", dict(locals()))
