# ruff: noqa: B008
"""Extracted ``batch`` command implementations.

A few legacy helpers still live in ``verbx.cli`` (shared with the render and
IR paths); they are reached through lazy wrappers so command behavior stays
identical while the monolithic entrypoint keeps shrinking.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf
import typer
from rich.console import Console
from rich.table import Table

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.commands.common import checkpoint_success_outfiles as _checkpoint_success_outfiles
from verbx.commands.common import load_batch_checkpoint as _load_batch_checkpoint
from verbx.commands.common import sha256_file as _sha256_file
from verbx.commands.common import write_csv_atomic as _write_csv_atomic
from verbx.commands.common import write_json_atomic as _write_json_atomic
from verbx.commands.common import write_jsonl_atomic as _write_jsonl_atomic
from verbx.commands.common import write_text_atomic as _write_text_atomic
from verbx.commands.validators import validate_output_audio_path as _validate_output_audio_path
from verbx.config import RenderConfig
from verbx.core.augmentation import (
    AugmentationBuild,
    augmentation_profile_names,
    augmentation_profiles,
    build_augmentation_manifest_template,
    build_augmentation_plans,
    render_config_snapshot,
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
from verbx.core.schema_versions import (
    AUGMENT_QA_BUNDLE_VERSION,
    AUGMENT_SUMMARY_VERSION,
    BATCH_CHECKPOINT_VERSION,
    BATCH_MANIFEST_VERSION,
)
from verbx.core.tempo import parse_pre_delay_ms
from verbx.io.audio import read_audio

console = Console()


def _cli() -> Any:
    """Lazy accessor for legacy helpers still living in ``verbx.cli``."""
    from verbx import cli as cli_module

    return cli_module


def _BatchStatusBar(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    return _cli()._BatchStatusBar(*args, **kwargs)


def _build_lucky_config(*args: Any, **kwargs: Any) -> Any:
    return _cli()._build_lucky_config(*args, **kwargs)


def _resolve_lucky_seed(*args: Any, **kwargs: Any) -> Any:
    return _cli()._resolve_lucky_seed(*args, **kwargs)


def _validate_generic_lucky_call(*args: Any, **kwargs: Any) -> Any:
    return _cli()._validate_generic_lucky_call(*args, **kwargs)


def render_config_from_options(options: dict[str, Any]) -> RenderConfig:
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


def validate_batch_job_paths(infile: Path, outfile: Path, idx: int) -> None:
    """Validate one batch job's input/output paths."""
    if infile.resolve() == outfile.resolve():
        msg = f"jobs[{idx - 1}] infile and outfile must be different."
        raise typer.BadParameter(msg)
    if not infile.exists():
        msg = f"jobs[{idx - 1}] infile not found: {infile}"
        raise typer.BadParameter(msg)
    _validate_output_audio_path(outfile, "auto")


def batch_template_impl() -> None:
    """Print a batch manifest template as JSON."""
    template = {
        "version": BATCH_MANIFEST_VERSION,
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


def batch_augment_template_impl() -> None:
    """Print an AI/data-augmentation manifest template as JSON."""
    template = build_augmentation_manifest_template()
    template["profiles"] = {
        name: {"description": str(data.get("description", ""))}
        for name, data in augmentation_profiles().items()
    }
    typer.echo(json.dumps(template, indent=2))


def batch_augment_profiles_impl(
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit profile definitions as JSON instead of a table.",
    ),
) -> None:
    """List built-in augmentation profiles and archetypes."""
    profiles = augmentation_profiles()
    if as_json:
        typer.echo(json.dumps(profiles, indent=2, sort_keys=True))
        return

    table = Table(title="Batch Augment Profiles")
    table.add_column("Profile", style="cyan")
    table.add_column("Archetypes", style="magenta")
    table.add_column("Description", style="white")
    for name in augmentation_profile_names():
        data = profiles.get(name, {})
        archetypes = data.get("archetypes", [])
        archetype_names = ", ".join(
            str(item.get("name", "")) for item in archetypes if isinstance(item, dict)
        )
        table.add_row(
            name,
            archetype_names,
            str(data.get("description", "")),
        )
    console.print(table)


def batch_augment_impl(
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
            "Path for dataset metadata JSONL (default: <output_root>/augmentation_manifest.jsonl)."
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
        help=("Optional QA bundle JSON path (default: <output_root>/augmentation_qa_bundle.json)."),
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
    """Render a deterministic augmentation dataset for AI research workflows."""
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("Augmentation manifest must be a JSON object.")

    if profile is not None:
        payload["profile"] = str(profile)
    if seed is not None:
        payload["seed"] = int(seed)
    if variants_per_input is not None:
        payload["variants_per_input"] = int(variants_per_input)
    if write_analysis is not None:
        payload["write_analysis"] = bool(write_analysis)

    try:
        build = build_augmentation_plans(
            payload=cast(dict[str, Any], payload),
            manifest_path=manifest,
            output_root_override=output_root,
            copy_dry=copy_dry,
            verify_split_isolation=verify_split_isolation,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if len(build.plans) == 0:
        raise typer.BadParameter("No augmentation plans were generated from manifest.")

    for plan in build.plans:
        if plan.infile.resolve() == plan.outfile.resolve():
            msg = (
                f"Augmentation source #{plan.source_index} resolved identical input/output path: "
                f"{plan.infile}"
            )
            raise typer.BadParameter(msg)
        if not plan.infile.exists():
            raise typer.BadParameter(f"Augmentation infile not found: {plan.infile}")
        _validate_output_audio_path(plan.outfile, str(plan.config.output_subtype))

    if dry_run:
        _print_augmentation_dry_run(build=build, schedule=schedule)
        return

    _prepare_augmentation_output_dirs(build=build)
    _copy_augmentation_dry_sources(build=build)

    prepared_jobs = [
        BatchJobSpec(
            index=plan.index,
            infile=plan.infile,
            outfile=plan.outfile,
            config=plan.config,
            estimated_cost=estimate_job_cost(plan.infile, plan.config),
        )
        for plan in build.plans
    ]

    max_workers = int(os.cpu_count() or 1) if jobs == 0 else int(jobs)
    max_workers = max(1, min(max_workers, len(prepared_jobs)))
    results_by_index: dict[int, BatchJobResult] = {}

    def runner(job: BatchJobSpec) -> None:
        run_render_pipeline(infile=job.infile, outfile=job.outfile, config=job.config)

    def on_result(result: BatchJobResult) -> None:
        results_by_index[int(result.index)] = result
        if result.success:
            console.print(
                f"augmented {result.index}: {result.outfile} "
                f"(attempts={result.attempts}, {result.duration_seconds:.2f}s)"
            )
        else:
            console.print(
                f"augment-failed {result.index}: {result.outfile} "
                f"(attempts={result.attempts}) {result.error}"
            )

    with _BatchStatusBar(total=len(prepared_jobs), label="Batch augment", enabled=True) as status:

        def on_result_with_status(result: BatchJobResult) -> None:
            on_result(result)
            status.advance(detail=f"job={result.index}")

        try:
            run_parallel_batch(
                jobs=prepared_jobs,
                max_workers=max_workers,
                schedule=schedule,
                retries=retries,
                continue_on_error=continue_on_error,
                runner=runner,
                on_result=on_result_with_status,
            )
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc

    out_root = build.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = (
        jsonl_out.resolve()
        if jsonl_out is not None
        else (out_root / "augmentation_manifest.jsonl").resolve()
    )
    summary_path = (
        summary_out.resolve()
        if summary_out is not None
        else (out_root / "augmentation_summary.json").resolve()
    )
    qa_bundle_path = (
        qa_bundle_out.resolve()
        if qa_bundle_out is not None
        else (out_root / "augmentation_qa_bundle.json").resolve()
    )
    records = _build_augmentation_records(build=build, results_by_index=results_by_index)
    _write_jsonl_atomic(jsonl_path, records)
    summary_payload = _build_augmentation_summary(
        build=build,
        records=records,
        manifest_path=manifest,
        jsonl_path=jsonl_path,
        schedule=schedule,
        jobs=max_workers,
    )
    baseline_payload: dict[str, Any] | None = None
    if baseline_summary is not None:
        baseline_payload = _load_optional_summary_payload(baseline_summary)
        summary_payload["baseline_summary"] = str(baseline_summary.resolve())

    qa_metric_rows: list[dict[str, Any]] = []
    if metrics_csv_out is not None:
        metrics_path = metrics_csv_out.resolve()
        metrics_rows = _build_augmentation_metrics_rows(
            records=records,
            include_loudness=metrics_include_loudness,
        )
        _write_csv_atomic(metrics_path, metrics_rows)
        summary_payload["metrics_csv"] = str(metrics_path)
        summary_payload["metrics_rows"] = len(metrics_rows)
        summary_payload["metrics_include_loudness"] = bool(metrics_include_loudness)
        qa_metric_rows = metrics_rows

    if len(qa_metric_rows) == 0:
        qa_metric_rows = _build_augmentation_metrics_rows(
            records=records,
            include_loudness=False,
        )
    qa_bundle_payload = _build_augmentation_qa_bundle(
        build=build,
        records=records,
        metrics_rows=qa_metric_rows,
        baseline_summary=baseline_payload,
    )
    _write_json_atomic(qa_bundle_path, qa_bundle_payload)
    summary_payload["qa_bundle"] = str(qa_bundle_path)
    summary_payload["qa_bundle_version"] = str(qa_bundle_payload.get("version", ""))

    if provenance_hash:
        provenance = _compute_augmentation_provenance_hash(
            build=build,
            manifest=manifest,
            records=records,
        )
        summary_payload["provenance_hash"] = str(provenance.get("sha256", ""))
        summary_payload["provenance"] = provenance

    if dataset_card_out is not None:
        dataset_card_path = dataset_card_out.resolve()
        card = _build_augmentation_dataset_card(
            build=build,
            summary=summary_payload,
            manifest_path=manifest,
            summary_path=summary_path,
            records=records,
        )
        _write_text_atomic(dataset_card_path, card)
        summary_payload["dataset_card"] = str(dataset_card_path)

    _write_json_atomic(summary_path, summary_payload)
    _print_augmentation_summary_table(summary_payload, summary_path=summary_path)

    failed = int(summary_payload.get("failed", 0))
    if fail_if_any_failed and failed > 0:
        raise typer.Exit(code=2)


def batch_render_impl(
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
    """Render jobs from manifest.json."""
    if resume and checkpoint_file is None:
        msg = "--resume requires --checkpoint-file."
        raise typer.BadParameter(msg)
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
            render_config = render_config_from_options(options)
        except (TypeError, ValueError) as exc:
            msg = f"jobs[{idx - 1}] has invalid options: {exc}"
            raise typer.BadParameter(msg) from exc
        validate_batch_job_paths(infile, outfile, idx)
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

    checkpoint_payload: dict[str, Any] | None = None
    resumed_outfiles: set[str] = set()
    if checkpoint_file is not None:
        if resume:
            checkpoint_payload = _load_batch_checkpoint(checkpoint_file)
            resumed_outfiles = _checkpoint_success_outfiles(checkpoint_payload)
            if len(resumed_outfiles) > 0:
                before = len(prepared_jobs)
                prepared_jobs = [
                    job
                    for job in prepared_jobs
                    if str(job.outfile.resolve()) not in resumed_outfiles
                ]
                skipped = before - len(prepared_jobs)
                if skipped > 0:
                    console.print(f"resuming batch: skipped {skipped} completed jobs")
        else:
            checkpoint_payload = {
                "version": BATCH_CHECKPOINT_VERSION,
                "manifest": str(manifest.resolve()),
                "results": [],
            }

    if len(prepared_jobs) == 0:
        console.print("batch complete: no pending jobs after resume filtering")
        return

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

    if progress_json is not None:
        try:
            progress_json.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise typer.BadParameter(f"Cannot create --progress-json parent: {exc}") from exc

    def on_result(result: BatchJobResult) -> None:
        if checkpoint_file is not None and checkpoint_payload is not None:
            checkpoint_payload.setdefault("results", [])
            assert isinstance(checkpoint_payload["results"], list)
            checkpoint_payload["results"].append(
                {
                    "index": int(result.index),
                    "outfile": str(result.outfile.resolve()),
                    "success": bool(result.success),
                    "attempts": int(result.attempts),
                    "duration_seconds": float(result.duration_seconds),
                    "estimated_cost": float(result.estimated_cost),
                    "error": result.error,
                }
            )
            _write_json_atomic(checkpoint_file, checkpoint_payload)
        if progress_json is not None:
            line = json.dumps(
                {
                    "index": int(result.index),
                    "outfile": str(result.outfile.resolve()),
                    "success": bool(result.success),
                    "duration_seconds": float(result.duration_seconds),
                    "error": result.error,
                }
            )
            try:
                with progress_json.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass  # progress reporting is best-effort
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

    with _BatchStatusBar(total=len(prepared_jobs), label="Batch render", enabled=True) as status:

        def on_result_with_status(result: BatchJobResult) -> None:
            on_result(result)
            status.advance(detail=f"job={result.index}")

        try:
            run_parallel_batch(
                jobs=prepared_jobs,
                max_workers=max_workers,
                schedule=schedule,
                retries=retries,
                continue_on_error=continue_on_error,
                runner=runner,
                on_result=on_result_with_status,
            )
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc


def _build_augmentation_metrics_rows(
    *,
    records: list[dict[str, Any]],
    include_loudness: bool,
) -> list[dict[str, Any]]:
    """Extract per-output metrics rows from augmentation records."""
    analyzer = AudioAnalyzer()
    rows: list[dict[str, Any]] = []
    for record in records:
        if not bool(record.get("success", False)):
            continue
        outfile = Path(str(record.get("outfile", "")))
        base: dict[str, Any] = {
            "augment_index": int(record.get("augment_index", 0)),
            "source_id": str(record.get("source_id", "")),
            "split": str(record.get("split", "")),
            "label": str(record.get("label", "")),
            "archetype": str(record.get("archetype", "")),
            "outfile": str(outfile),
        }
        try:
            audio, sr = read_audio(str(outfile))
            metrics = analyzer.analyze(
                audio,
                sr,
                include_loudness=include_loudness,
                include_edr=False,
            )
            base["sample_rate"] = int(sr)
            base["samples"] = int(audio.shape[0])
            base["channels"] = int(audio.shape[1])
            for key in sorted(metrics):
                base[key] = float(metrics[key])
            base["metrics_error"] = ""
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            base["metrics_error"] = str(exc)
        rows.append(base)
    return rows


def _build_augmentation_dataset_card(
    *,
    build: AugmentationBuild,
    summary: dict[str, Any],
    manifest_path: Path,
    summary_path: Path,
    records: list[dict[str, Any]],
) -> str:
    """Generate a compact markdown dataset card for AI/data workflows."""
    split_counts = cast(dict[str, Any], summary.get("split_counts", {}))
    label_counts = cast(dict[str, Any], summary.get("label_counts", {}))
    archetype_counts = cast(dict[str, Any], summary.get("archetype_counts", {}))
    lines = [
        "# Dataset Card",
        "",
        "## Identity",
        f"- dataset_name: `{build.dataset_name}`",
        f"- profile: `{build.profile}`",
        f"- seed: `{build.seed}`",
        f"- generated_utc: `{datetime.now(UTC).isoformat()}`",
        f"- manifest_path: `{manifest_path.resolve()}`",
        f"- summary_json: `{summary_path.resolve()}`",
        f"- qa_bundle_json: `{summary.get('qa_bundle', '(none)')}`",
        f"- output_root: `{build.output_root.resolve()}`",
        "",
        "## Render Summary",
        f"- planned: `{summary.get('planned', 0)}`",
        f"- success: `{summary.get('success', 0)}`",
        f"- failed: `{summary.get('failed', 0)}`",
        f"- variants_per_input: `{build.variants_per_input}`",
        f"- provenance_hash: `{summary.get('provenance_hash', '(disabled)')}`",
        "",
        "## Split Distribution",
    ]
    if split_counts:
        for split in sorted(split_counts):
            lines.append(f"- {split}: `{split_counts[split]}`")
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "## Label Distribution",
        ]
    )
    if label_counts:
        for label in sorted(label_counts):
            lines.append(f"- {label}: `{label_counts[label]}`")
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "## Archetype Distribution",
        ]
    )
    if archetype_counts:
        for archetype in sorted(archetype_counts):
            lines.append(f"- {archetype}: `{archetype_counts[archetype]}`")
    else:
        lines.append("- (none)")

    rt60_values: list[float] = []
    wet_values: list[float] = []
    damping_values: list[float] = []
    fdn_lines_values: list[int] = []
    fdn_matrices: set[str] = set()
    for row in records:
        config = row.get("render_config", {})
        if not isinstance(config, dict):
            continue
        rt60 = config.get("rt60")
        wet = config.get("wet")
        damping = config.get("damping")
        fdn_lines = config.get("fdn_lines")
        fdn_matrix = config.get("fdn_matrix")
        if isinstance(rt60, (int, float)):
            rt60_values.append(float(rt60))
        if isinstance(wet, (int, float)):
            wet_values.append(float(wet))
        if isinstance(damping, (int, float)):
            damping_values.append(float(damping))
        if isinstance(fdn_lines, (int, float)):
            fdn_lines_values.append(int(fdn_lines))
        if isinstance(fdn_matrix, str) and fdn_matrix != "":
            fdn_matrices.add(fdn_matrix)

    lines.extend(
        [
            "",
            "## Parameter Envelope",
        ]
    )
    if rt60_values:
        lines.append(f"- rt60_s_min_max: `{min(rt60_values):.3f} .. {max(rt60_values):.3f}`")
    if wet_values:
        lines.append(f"- wet_min_max: `{min(wet_values):.3f} .. {max(wet_values):.3f}`")
    if damping_values:
        lines.append(f"- damping_min_max: `{min(damping_values):.3f} .. {max(damping_values):.3f}`")
    if fdn_lines_values:
        lines.append(f"- fdn_lines_min_max: `{min(fdn_lines_values)} .. {max(fdn_lines_values)}`")
    if fdn_matrices:
        lines.append(f"- fdn_matrices: `{', '.join(sorted(fdn_matrices))}`")
    if len(lines) > 0 and lines[-1] == "## Parameter Envelope":
        lines.append("- (no parameter metadata available)")

    lines.extend(
        [
            "",
            "## Notes",
            "- This card is auto-generated by `verbx batch augment`.",
            (
                "- Keep this file with `augmentation_manifest.jsonl` and summary JSON "
                "for reproducibility."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _print_augmentation_dry_run(
    *,
    build: AugmentationBuild,
    schedule: BatchSchedulePolicy,
) -> None:
    """Print dry-run summary for augmentation plan expansion."""
    table = Table(title="Batch Augment Dry-Run")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("dataset_name", build.dataset_name)
    table.add_row("profile", build.profile)
    table.add_row("seed", str(build.seed))
    table.add_row("variants_per_input", str(build.variants_per_input))
    table.add_row("write_analysis", str(build.write_analysis))
    table.add_row("copy_dry", str(build.copy_dry))
    table.add_row("schedule", str(schedule))
    table.add_row("plans", str(len(build.plans)))
    table.add_row("output_root", str(build.output_root))
    console.print(table)

    ordered = order_jobs(
        [
            BatchJobSpec(
                index=plan.index,
                infile=plan.infile,
                outfile=plan.outfile,
                config=plan.config,
                estimated_cost=estimate_job_cost(plan.infile, plan.config),
            )
            for plan in build.plans
        ],
        schedule,
    )
    for job in ordered:
        match_plan = build.plans[job.index - 1]
        console.print(
            "[dry-run] augment "
            f"{job.index}: {job.infile} -> {job.outfile} "
            f"(profile={match_plan.profile}, archetype={match_plan.archetype}, "
            f"split={match_plan.split}, label={match_plan.label}, seed={match_plan.seed})"
        )


def _copy_augmentation_dry_sources(*, build: AugmentationBuild) -> list[Path]:
    """Copy clean source files to output tree (optional paired dataset mode)."""
    copied: list[Path] = []
    seen: set[str] = set()
    for plan in build.plans:
        target = plan.dry_copy_outfile
        if target is None:
            continue
        key = str(target.resolve())
        if key in seen:
            continue
        seen.add(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan.infile, target)
        copied.append(target)
    return copied


def _prepare_augmentation_output_dirs(*, build: AugmentationBuild) -> None:
    """Ensure batch augment output directories exist before parallel rendering."""
    for plan in build.plans:
        plan.outfile.parent.mkdir(parents=True, exist_ok=True)
        if plan.config.analysis_out is not None:
            Path(plan.config.analysis_out).parent.mkdir(parents=True, exist_ok=True)


def _build_augmentation_records(
    *,
    build: AugmentationBuild,
    results_by_index: dict[int, BatchJobResult],
) -> list[dict[str, Any]]:
    """Build machine-friendly per-variant metadata rows."""
    records: list[dict[str, Any]] = []
    for plan in build.plans:
        result = results_by_index.get(plan.index)
        success = bool(result.success) if result is not None else False
        duration = float(result.duration_seconds) if result is not None else 0.0
        attempts = int(result.attempts) if result is not None else 0
        error = None if result is None else result.error
        outfile_rel = (
            str(plan.outfile.relative_to(build.output_root))
            if plan.outfile.is_relative_to(build.output_root)
            else str(plan.outfile)
        )
        record = {
            "augment_index": int(plan.index),
            "source_index": int(plan.source_index),
            "source_id": plan.source_id,
            "split": plan.split,
            "label": plan.label,
            "tags": list(plan.tags),
            "infile": str(plan.infile.resolve()),
            "outfile": str(plan.outfile.resolve()),
            "outfile_relative": outfile_rel,
            "profile": plan.profile,
            "archetype": plan.archetype,
            "variant_index": int(plan.variant_index),
            "seed": int(plan.seed),
            "render_config": render_config_snapshot(plan.config),
            "success": success,
            "attempts": attempts,
            "duration_seconds": duration,
            "error": error,
            "source_metadata": dict(plan.source_metadata),
        }
        if plan.dry_copy_outfile is not None:
            dry_rel = (
                str(plan.dry_copy_outfile.relative_to(build.output_root))
                if plan.dry_copy_outfile.is_relative_to(build.output_root)
                else str(plan.dry_copy_outfile)
            )
            record["dry_outfile"] = str(plan.dry_copy_outfile.resolve())
            record["dry_outfile_relative"] = dry_rel
        records.append(record)
    return records


def _build_augmentation_summary(
    *,
    build: AugmentationBuild,
    records: list[dict[str, Any]],
    manifest_path: Path,
    jsonl_path: Path,
    schedule: BatchSchedulePolicy,
    jobs: int,
) -> dict[str, Any]:
    """Build summary payload for augmentation run."""
    total = len(records)
    success = int(sum(1 for row in records if bool(row.get("success", False))))
    failed = int(total - success)
    split_counts: dict[str, int] = {}
    split_source_ids: dict[str, set[str]] = {}
    split_label_counts: dict[str, dict[str, int]] = {}
    label_counts: dict[str, int] = {}
    archetype_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    for row in records:
        split = str(row.get("split", ""))
        label = str(row.get("label", ""))
        source_id = str(row.get("source_id", ""))
        archetype = str(row.get("archetype", ""))
        split_counts[split] = int(split_counts.get(split, 0) + 1)
        if split != "" and label != "":
            split_map = split_label_counts.setdefault(split, {})
            split_map[label] = int(split_map.get(label, 0) + 1)
        if split != "" and source_id != "":
            split_source_ids.setdefault(split, set()).add(source_id)
        if label != "":
            label_counts[label] = int(label_counts.get(label, 0) + 1)
        if archetype != "":
            archetype_counts[archetype] = int(archetype_counts.get(archetype, 0) + 1)
        tags = row.get("tags", [])
        if isinstance(tags, list):
            for tag in tags:
                token = str(tag)
                if token == "":
                    continue
                tag_counts[token] = int(tag_counts.get(token, 0) + 1)
    return {
        "version": AUGMENT_SUMMARY_VERSION,
        "mode": "batch-augment",
        "dataset_name": build.dataset_name,
        "profile": build.profile,
        "seed": int(build.seed),
        "variants_per_input": int(build.variants_per_input),
        "write_analysis": bool(build.write_analysis),
        "copy_dry": bool(build.copy_dry),
        "manifest_path": str(manifest_path.resolve()),
        "output_root": str(build.output_root.resolve()),
        "metadata_jsonl": str(jsonl_path.resolve()),
        "schedule": str(schedule),
        "jobs": int(jobs),
        "planned": int(total),
        "success": int(success),
        "failed": int(failed),
        "unique_sources": len({str(row.get("source_id", "")) for row in records}),
        "source_split_counts": {
            split: len(source_ids) for split, source_ids in sorted(split_source_ids.items())
        },
        "split_counts": split_counts,
        "split_label_counts": {
            split: dict(sorted(label_map.items()))
            for split, label_map in sorted(split_label_counts.items())
        },
        "label_counts": label_counts,
        "archetype_counts": archetype_counts,
        "tag_counts": tag_counts,
    }


def _load_optional_summary_payload(path: Path) -> dict[str, Any]:
    """Load optional JSON summary payload for regeneration delta reporting."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"Invalid baseline summary JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("--baseline-summary must contain a JSON object.")
    return cast(dict[str, Any], payload)


def _build_augmentation_qa_bundle(
    *,
    build: AugmentationBuild,
    records: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
    baseline_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build augmentation QA bundle with split-quality and class-balance deltas."""
    split_label_counts = _extract_split_label_counts(records)
    split_quality = _summarize_split_quality(metrics_rows)
    successful = int(sum(1 for row in records if bool(row.get("success", False))))
    failed = int(len(records) - successful)

    baseline_split_label_counts = (
        _extract_baseline_split_label_counts(baseline_summary)
        if baseline_summary is not None
        else {}
    )
    class_balance_delta = (
        _compute_class_balance_delta(
            current=split_label_counts,
            baseline=baseline_split_label_counts,
        )
        if baseline_summary is not None
        else None
    )

    return {
        "version": AUGMENT_QA_BUNDLE_VERSION,
        "mode": "batch-augment-qa",
        "dataset_name": build.dataset_name,
        "profile": build.profile,
        "seed": int(build.seed),
        "planned": len(records),
        "success": int(successful),
        "failed": int(failed),
        "split_label_counts": split_label_counts,
        "split_quality": split_quality,
        "baseline_present": baseline_summary is not None,
        "class_balance_delta": class_balance_delta,
    }


def _extract_split_label_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Extract split->label->count table from augmentation metadata rows."""
    counts: dict[str, dict[str, int]] = {}
    for row in records:
        split = str(row.get("split", ""))
        label = str(row.get("label", ""))
        if split == "" or label == "":
            continue
        split_map = counts.setdefault(split, {})
        split_map[label] = int(split_map.get(label, 0) + 1)
    return {split: dict(sorted(label_map.items())) for split, label_map in sorted(counts.items())}


def _extract_baseline_split_label_counts(payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Extract split-label table from baseline summary or QA bundle payload."""
    direct = payload.get("split_label_counts")
    if isinstance(direct, dict):
        return _normalize_split_label_counts(direct)
    qa_raw = payload.get("class_balance")
    if isinstance(qa_raw, dict):
        nested = qa_raw.get("split_label_counts")
        if isinstance(nested, dict):
            return _normalize_split_label_counts(nested)
    return {}


def _normalize_split_label_counts(payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    normalized: dict[str, dict[str, int]] = {}
    for split, labels_raw in payload.items():
        if not isinstance(labels_raw, dict):
            continue
        labels: dict[str, int] = {}
        for label, count_raw in labels_raw.items():
            try:
                labels[str(label)] = int(count_raw)
            except (TypeError, ValueError):
                continue
        if labels:
            normalized[str(split)] = dict(sorted(labels.items()))
    return dict(sorted(normalized.items()))


def _compute_class_balance_delta(
    *,
    current: dict[str, dict[str, int]],
    baseline: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Compute split-level and global class-balance deltas against baseline."""
    split_delta: dict[str, dict[str, dict[str, float | int]]] = {}
    for split in sorted(set(current) | set(baseline)):
        cur_labels = current.get(split, {})
        base_labels = baseline.get(split, {})
        cur_total = int(sum(cur_labels.values()))
        base_total = int(sum(base_labels.values()))
        label_delta: dict[str, dict[str, float | int]] = {}
        for label in sorted(set(cur_labels) | set(base_labels)):
            cur_count = int(cur_labels.get(label, 0))
            base_count = int(base_labels.get(label, 0))
            cur_ratio = float(cur_count / cur_total) if cur_total > 0 else 0.0
            base_ratio = float(base_count / base_total) if base_total > 0 else 0.0
            label_delta[label] = {
                "current_count": cur_count,
                "baseline_count": base_count,
                "count_delta": int(cur_count - base_count),
                "current_ratio": cur_ratio,
                "baseline_ratio": base_ratio,
                "ratio_delta": float(cur_ratio - base_ratio),
            }
        split_delta[split] = label_delta

    current_global = _collapse_split_label_counts(current)
    baseline_global = _collapse_split_label_counts(baseline)
    global_total = int(sum(current_global.values()))
    baseline_total = int(sum(baseline_global.values()))
    global_delta: dict[str, dict[str, float | int]] = {}
    for label in sorted(set(current_global) | set(baseline_global)):
        cur_count = int(current_global.get(label, 0))
        base_count = int(baseline_global.get(label, 0))
        cur_ratio = float(cur_count / global_total) if global_total > 0 else 0.0
        base_ratio = float(base_count / baseline_total) if baseline_total > 0 else 0.0
        global_delta[label] = {
            "current_count": cur_count,
            "baseline_count": base_count,
            "count_delta": int(cur_count - base_count),
            "current_ratio": cur_ratio,
            "baseline_ratio": base_ratio,
            "ratio_delta": float(cur_ratio - base_ratio),
        }
    return {
        "split_label_delta": split_delta,
        "global_label_delta": global_delta,
    }


def _collapse_split_label_counts(counts: dict[str, dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for label_map in counts.values():
        for label, count in label_map.items():
            merged[label] = int(merged.get(label, 0) + int(count))
    return merged


def _summarize_split_quality(metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize numeric quality metrics per split from augmentation metrics rows."""
    fields = (
        "rt60_estimate_seconds",
        "early_late_ratio_db",
        "stereo_width",
        "stereo_coherence",
        "spectral_centroid",
        "spectral_flatness",
        "dynamic_range",
        "lufs_integrated",
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in metrics_rows:
        split = str(row.get("split", ""))
        if split == "":
            continue
        grouped.setdefault(split, []).append(row)
    summary: dict[str, Any] = {}
    for split, rows in sorted(grouped.items()):
        split_summary: dict[str, Any] = {}
        for field in fields:
            stats = _summarize_metrics_field(rows, field)
            if stats is not None:
                split_summary[field] = stats
        errors = int(sum(1 for row in rows if str(row.get("metrics_error", "")).strip() != ""))
        split_summary["rows"] = len(rows)
        split_summary["metrics_errors"] = int(errors)
        summary[split] = split_summary
    return summary


def _summarize_metrics_field(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, float] | None:
    values: list[float] = []
    for row in rows:
        raw = row.get(field)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    if len(values) == 0:
        return None
    vec = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(vec)),
        "max": float(np.max(vec)),
        "mean": float(np.mean(vec)),
    }


def _compute_augmentation_provenance_hash(
    *,
    build: AugmentationBuild,
    manifest: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute deterministic augmentation provenance hash for registry workflows."""
    manifest_payload: dict[str, Any] | None = None
    try:
        parsed = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            manifest_payload = cast(dict[str, Any], parsed)
    except (OSError, json.JSONDecodeError):
        manifest_payload = None
    if manifest_payload is None:
        manifest_canonical = manifest.read_bytes()
    else:
        manifest_canonical = json.dumps(
            manifest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    source_paths = sorted({str(plan.infile.resolve()) for plan in build.plans})
    source_signatures: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    digest.update(manifest_canonical)
    for source in source_paths:
        path = Path(source)
        info = sf.info(str(path))
        signature = {
            "path": source,
            "sha256": _sha256_file(path),
            "frames": int(info.frames),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "format": str(info.format),
            "subtype": str(info.subtype),
        }
        source_signatures.append(signature)
        digest.update(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(str(len(records)).encode("utf-8"))

    return {
        "schema": "augmentation-provenance-v1",
        "sha256": digest.hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_canonical).hexdigest(),
        "source_count": len(source_signatures),
        "sources": source_signatures,
    }


def _print_augmentation_summary_table(
    summary: dict[str, Any],
    *,
    summary_path: Path,
) -> None:
    """Print concise augmentation run summary."""
    table = Table(title="Batch Augment Summary")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    for key in (
        "dataset_name",
        "profile",
        "seed",
        "variants_per_input",
        "planned",
        "success",
        "failed",
        "unique_sources",
        "output_root",
        "metadata_jsonl",
    ):
        table.add_row(key, str(summary.get(key, "")))
    metrics_csv = summary.get("metrics_csv")
    if metrics_csv:
        table.add_row("metrics_csv", str(metrics_csv))
    qa_bundle = summary.get("qa_bundle")
    if qa_bundle:
        table.add_row("qa_bundle", str(qa_bundle))
    dataset_card = summary.get("dataset_card")
    if dataset_card:
        table.add_row("dataset_card", str(dataset_card))
    provenance_hash = summary.get("provenance_hash")
    if provenance_hash:
        table.add_row("provenance_hash", str(provenance_hash))
    table.add_row("summary_json", str(summary_path.resolve()))
    console.print(table)
