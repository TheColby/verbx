# ruff: noqa: B008
"""Batch command wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from verbx.core.batch_scheduler import BatchSchedulePolicy


def _forward(name: str, params: dict[str, Any]) -> None:
    from verbx import cli as cli_module

    return cli_module.get_command_impl(name)(**params)


def batch_template() -> None:
    _forward("_batch_template_impl", {})


def batch_augment_template() -> None:
    _forward("_batch_augment_template_impl", {})


def batch_augment_profiles(
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit profile definitions as JSON instead of a table.",
    ),
) -> None:
    _forward("_batch_augment_profiles_impl", dict(locals()))


def batch_augment(
    manifest: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    output_root: Path | None = typer.Option(
        None,
        "--output-root",
        resolve_path=True,
        help="Override output root directory from manifest.",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Optional profile override (for quick profile A/B against one manifest).",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Optional deterministic seed override.",
    ),
    variants_per_input: int | None = typer.Option(
        None,
        "--variants-per-input",
        min=1,
        max=500,
        help="Optional variants-per-source override.",
    ),
    write_analysis: bool | None = typer.Option(
        None,
        "--write-analysis/--no-write-analysis",
        help="Override manifest write_analysis behavior.",
    ),
    copy_dry: bool = typer.Option(
        False,
        "--copy-dry/--no-copy-dry",
        help="Copy clean source files into output tree (paired dry/wet dataset layout).",
    ),
    verify_split_isolation: bool = typer.Option(
        True,
        "--verify-split-isolation/--allow-split-overlap",
        help=(
            "Require one source id/input file to belong to exactly one split "
            "(prevents train/val/test leakage)."
        ),
    ),
    jobs: int = typer.Option(0, "--jobs", min=0, help="0 = auto"),
    schedule: BatchSchedulePolicy = typer.Option("longest-first", "--schedule"),
    retries: int = typer.Option(0, "--retries", min=0),
    continue_on_error: bool = typer.Option(False, "--continue-on-error/--fail-fast"),
    fail_if_any_failed: bool = typer.Option(
        True,
        "--fail-if-any-failed/--allow-failed",
        help="Exit non-zero when any augmentation render fails.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    jsonl_out: Path | None = typer.Option(
        None,
        "--jsonl-out",
        resolve_path=True,
        help=(
            "Path for dataset metadata JSONL "
            "(default: <output_root>/augmentation_manifest.jsonl)."
        ),
    ),
    summary_out: Path | None = typer.Option(
        None,
        "--summary-out",
        resolve_path=True,
        help="Path for run summary JSON (default: <output_root>/augmentation_summary.json).",
    ),
    dataset_card_out: Path | None = typer.Option(
        None,
        "--dataset-card-out",
        resolve_path=True,
        help="Optional Markdown dataset-card path for ML dataset documentation.",
    ),
    metrics_csv_out: Path | None = typer.Option(
        None,
        "--metrics-csv-out",
        resolve_path=True,
        help="Optional CSV path for per-output acoustic features.",
    ),
    metrics_include_loudness: bool = typer.Option(
        False,
        "--metrics-include-loudness/--metrics-fast",
        help="Include LUFS/true-peak/LRA in metrics CSV export (slower).",
    ),
    qa_bundle_out: Path | None = typer.Option(
        None,
        "--qa-bundle-out",
        resolve_path=True,
        help=(
            "Optional QA bundle JSON path (default: <output_root>/augmentation_qa_bundle.json)."
        ),
    ),
    baseline_summary: Path | None = typer.Option(
        None,
        "--baseline-summary",
        exists=True,
        readable=True,
        resolve_path=True,
        help="Optional prior augmentation summary/QA bundle JSON used for regeneration deltas.",
    ),
    provenance_hash: bool = typer.Option(
        False,
        "--provenance-hash/--no-provenance-hash",
        help="Emit deterministic provenance hash over manifest payload + source signatures.",
    ),
) -> None:
    _forward("_batch_augment_impl", dict(locals()))


def batch_render(
    manifest: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    jobs: int = typer.Option(0, "--jobs", min=0, help="0 = auto"),
    schedule: BatchSchedulePolicy = typer.Option("longest-first", "--schedule"),
    retries: int = typer.Option(0, "--retries", min=0),
    continue_on_error: bool = typer.Option(False, "--continue-on-error/--fail-fast"),
    checkpoint_file: Path | None = typer.Option(
        None,
        "--checkpoint-file",
        resolve_path=True,
        help="Optional checkpoint file used to persist per-job completion state.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from --checkpoint-file and skip already completed jobs.",
    ),
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
    progress_json: Path | None = typer.Option(
        None,
        "--progress-json",
        resolve_path=True,
        help=(
            "Append one JSONL line per completed job to this file. "
            "Each line contains index, outfile, success, duration_seconds, and error."
        ),
    ),
) -> None:
    _forward("_batch_render_impl", dict(locals()))
