# ruff: noqa: B008
"""Streaming corpus augmentation command and worker helpers."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import librosa
import numpy as np
import soundfile as sf
import typer
from rich.console import Console
from rich.table import Table

from verbx.commands.common import checkpoint_success_outfiles as _checkpoint_success_outfiles
from verbx.commands.common import load_batch_checkpoint as _load_batch_checkpoint
from verbx.commands.common import write_json_atomic as _write_json_atomic
from verbx.config import OutputSubtype
from verbx.core.algo_reverb import AlgoReverbConfig, AlgoReverbEngine
from verbx.io.audio import read_audio


def _BatchStatusBar(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    from verbx import cli as cli_module

    legacy: Any = cli_module
    return legacy._BatchStatusBar(*args, **kwargs)


console = Console()
_CORPUS_AUDIO_EXTENSIONS = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".caf"}


def batch_corpus_generate(
    source: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        resolve_path=True,
        help="Input audio file or folder to augment.",
    ),
    output_root: Path = typer.Option(
        ...,
        "--output-root",
        resolve_path=True,
        help="Output directory where generated variants are written.",
    ),
    variants_per_input: int = typer.Option(
        16,
        "--variants-per-input",
        min=1,
        help=(
            "Number of generated variants per input file. No practical hard limit is imposed; "
            "this can be set to very large values for massive corpus generation runs."
        ),
    ),
    seed: int = typer.Option(
        20260322,
        "--seed",
        help="Deterministic random seed for reproducible corpus generation.",
    ),
    recurse: bool = typer.Option(
        True,
        "--recurse/--no-recurse",
        help="Recurse into nested folders when source is a directory.",
    ),
    time_shift_min_ms: float = typer.Option(
        -80.0,
        "--time-shift-min-ms",
        help="Minimum time-shift in milliseconds (negative = earlier).",
    ),
    time_shift_max_ms: float = typer.Option(
        80.0,
        "--time-shift-max-ms",
        help="Maximum time-shift in milliseconds.",
    ),
    pitch_shift_min_semitones: float = typer.Option(
        -2.5,
        "--pitch-shift-min-semitones",
        help="Minimum pitch shift in semitones.",
    ),
    pitch_shift_max_semitones: float = typer.Option(
        2.5,
        "--pitch-shift-max-semitones",
        help="Maximum pitch shift in semitones.",
    ),
    reverb_wet_min: float = typer.Option(
        0.12,
        "--reverb-wet-min",
        min=0.0,
        max=1.0,
        help="Minimum sampled wet mix for algorithmic reverb.",
    ),
    reverb_wet_max: float = typer.Option(
        0.48,
        "--reverb-wet-max",
        min=0.0,
        max=1.0,
        help="Maximum sampled wet mix for algorithmic reverb.",
    ),
    reverb_rt60_min: float = typer.Option(
        0.25,
        "--reverb-rt60-min",
        min=0.05,
        help="Minimum sampled algorithmic reverb RT60 in seconds.",
    ),
    reverb_rt60_max: float = typer.Option(
        4.50,
        "--reverb-rt60-max",
        min=0.05,
        help="Maximum sampled algorithmic reverb RT60 in seconds.",
    ),
    reverb_pre_delay_min_ms: float = typer.Option(
        0.0,
        "--reverb-pre-delay-min-ms",
        help="Minimum sampled reverb pre-delay in milliseconds.",
    ),
    reverb_pre_delay_max_ms: float = typer.Option(
        42.0,
        "--reverb-pre-delay-max-ms",
        help="Maximum sampled reverb pre-delay in milliseconds.",
    ),
    output_subtype: OutputSubtype = typer.Option(
        "float32",
        "--output-subtype",
        help="Output subtype for generated corpus audio files.",
    ),
    execution_profile: Literal["cpu-balanced", "cpu-throughput", "cpu-safe"] = typer.Option(
        "cpu-balanced", "--execution-profile", help="CPU worker selection profile."
    ),
    jobs: int = typer.Option(
        0,
        "--jobs",
        min=0,
        help="Worker threads (0 selects automatically). Number of worker threads used per input file when generating variants.",
    ),
    retries: int = typer.Option(
        0,
        "--retries",
        min=0,
        help="Retry attempts per failed output variant before marking it failed.",
    ),
    checkpoint_file: Path | None = typer.Option(
        None,
        "--checkpoint-file",
        resolve_path=True,
        help="Optional JSON checkpoint path for resume-safe corpus generation.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from --checkpoint-file and skip already completed outputs.",
    ),
    num_shards: int = typer.Option(
        1,
        "--num-shards",
        min=1,
        help="Total shard count for distributed corpus generation.",
    ),
    shard_index: int = typer.Option(
        0,
        "--shard-index",
        min=0,
        help="Zero-based shard index to execute from --num-shards.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Only print generation plan and totals without rendering files.",
    ),
) -> None:
    """Generate augmented audio corpora from one file or a directory of files."""
    cpu_count = os.cpu_count() or 1
    effective_jobs = jobs or (
        1
        if execution_profile == "cpu-safe"
        else cpu_count
        if execution_profile == "cpu-throughput"
        else max(1, cpu_count // 2)
    )
    if source.is_file():
        files = [source]
    else:
        pattern = "**/*" if recurse else "*"
        files = [
            path
            for path in sorted(source.glob(pattern))
            if path.is_file() and path.suffix.lower() in _CORPUS_AUDIO_EXTENSIONS
        ]
    if len(files) == 0:
        raise typer.BadParameter(f"No audio files found under: {source}")
    if shard_index >= num_shards:
        raise typer.BadParameter("--shard-index must be smaller than --num-shards.")
    if resume and checkpoint_file is None:
        raise typer.BadParameter("--resume requires --checkpoint-file.")

    if time_shift_min_ms > time_shift_max_ms:
        raise typer.BadParameter("--time-shift-min-ms must be <= --time-shift-max-ms.")
    if pitch_shift_min_semitones > pitch_shift_max_semitones:
        raise typer.BadParameter(
            "--pitch-shift-min-semitones must be <= --pitch-shift-max-semitones."
        )
    if reverb_wet_min > reverb_wet_max:
        raise typer.BadParameter("--reverb-wet-min must be <= --reverb-wet-max.")
    if reverb_rt60_min > reverb_rt60_max:
        raise typer.BadParameter("--reverb-rt60-min must be <= --reverb-rt60-max.")
    if reverb_pre_delay_min_ms > reverb_pre_delay_max_ms:
        raise typer.BadParameter("--reverb-pre-delay-min-ms must be <= --reverb-pre-delay-max-ms.")

    sharded_files = [path for index, path in enumerate(files) if index % num_shards == shard_index]
    if len(sharded_files) == 0:
        raise typer.BadParameter(
            "No input files selected for this shard; adjust --num-shards/--shard-index."
        )
    total_outputs = len(sharded_files) * int(variants_per_input)
    manifest_path = output_root / "corpus_generation_manifest.jsonl"
    summary_path = output_root / "corpus_generation_summary.json"
    checkpoint_path = checkpoint_file.resolve() if checkpoint_file is not None else None

    if dry_run:
        console.print(
            "[dry-run] corpus-generate "
            f"inputs={len(sharded_files)} variants_per_input={variants_per_input} "
            f"outputs={total_outputs} shard={shard_index}/{num_shards} "
            f"jobs={jobs} retries={retries}"
        )
        return

    output_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)
    started_timer = time.perf_counter()
    resumed_outputs: set[str] = set()
    checkpoint_payload: dict[str, Any] = {"version": "0.1", "results": []}
    if checkpoint_path is not None and resume:
        checkpoint_payload = _load_batch_checkpoint(checkpoint_path)
        resumed_outputs = _checkpoint_success_outfiles(checkpoint_payload)
    checkpoint_records = checkpoint_payload.setdefault("results", [])
    if not isinstance(checkpoint_records, list):
        checkpoint_records = []
        checkpoint_payload["results"] = checkpoint_records

    manifest_mode = "a" if resume and manifest_path.exists() else "w"
    manifest_lock = Lock()
    checkpoint_lock = Lock()
    read_seconds = 0.0
    process_seconds = 0.0
    write_seconds = 0.0
    generated = 0
    resumed_skipped = 0
    failed = 0
    total_attempts = 0
    retried_outputs = 0

    with manifest_path.open(manifest_mode, encoding="utf-8") as manifest_handle:
        with _BatchStatusBar(total=total_outputs, label="Corpus generate", enabled=True) as status:
            for file_index, infile in enumerate(sharded_files):
                read_started = time.perf_counter()
                audio, sr = read_audio(str(infile))
                read_seconds += time.perf_counter() - read_started
                rel_parent = infile.parent.relative_to(source) if source.is_dir() else Path("")
                out_dir = (output_root / rel_parent / infile.stem).resolve()
                out_dir.mkdir(parents=True, exist_ok=True)
                variant_indices = list(range(variants_per_input))
                with ThreadPoolExecutor(max_workers=effective_jobs) as executor:
                    futures: dict[Future[dict[str, Any]], tuple[int, Path]] = {}
                    for variant_index in variant_indices:
                        outfile = (
                            out_dir
                            / f"{infile.stem}.aug_{variant_index:08d}{infile.suffix.lower()}"
                        ).resolve()
                        if str(outfile) in resumed_outputs:
                            resumed_skipped += 1
                            status.advance(detail=f"{infile.name}#{variant_index} resumed")
                            continue
                        futures[
                            executor.submit(
                                _generate_corpus_variant,
                                infile=infile,
                                outfile=outfile,
                                audio=audio,
                                sr=sr,
                                source_index=file_index,
                                variant_index=variant_index,
                                seed=seed,
                                time_shift_min_ms=time_shift_min_ms,
                                time_shift_max_ms=time_shift_max_ms,
                                pitch_shift_min_semitones=pitch_shift_min_semitones,
                                pitch_shift_max_semitones=pitch_shift_max_semitones,
                                reverb_rt60_min=reverb_rt60_min,
                                reverb_rt60_max=reverb_rt60_max,
                                reverb_wet_min=reverb_wet_min,
                                reverb_wet_max=reverb_wet_max,
                                reverb_pre_delay_min_ms=reverb_pre_delay_min_ms,
                                reverb_pre_delay_max_ms=reverb_pre_delay_max_ms,
                                output_subtype=str(output_subtype),
                                retries=int(retries),
                            )
                        ] = (variant_index, outfile)
                    for future in as_completed(futures):
                        variant_index, outfile = futures[future]
                        try:
                            record = future.result()
                            generated += 1
                            process_seconds += float(record["process_seconds"])
                            write_seconds += float(record["write_seconds"])
                            attempts = int(record.get("attempts", 1))
                            total_attempts += attempts
                            if attempts > 1:
                                retried_outputs += 1
                            with manifest_lock:
                                manifest_handle.write(f"{json.dumps(record, sort_keys=True)}\n")
                                manifest_handle.flush()
                            if checkpoint_path is not None:
                                with checkpoint_lock:
                                    checkpoint_records.append(
                                        {
                                            "outfile": record["output"],
                                            "success": True,
                                            "timestamp": datetime.now(UTC).isoformat(),
                                        }
                                    )
                                    _write_json_atomic(checkpoint_path, checkpoint_payload)
                        except Exception as exc:  # pragma: no cover - defensive write-path guard
                            failed += 1
                            if checkpoint_path is not None:
                                with checkpoint_lock:
                                    checkpoint_records.append(
                                        {
                                            "outfile": str(outfile),
                                            "success": False,
                                            "error": str(exc),
                                            "timestamp": datetime.now(UTC).isoformat(),
                                        }
                                    )
                                    _write_json_atomic(checkpoint_path, checkpoint_payload)
                            status.advance(detail=f"{infile.name}#{variant_index} failed")
                            continue
                        attempts = int(record.get("attempts", 1))
                        retry_note = "" if attempts <= 1 else f" retries={attempts - 1}"
                        status.advance(detail=f"{infile.name}#{variant_index}{retry_note}")
    elapsed_seconds = max(1e-9, time.perf_counter() - started_timer)
    outputs_per_second = float(generated) / elapsed_seconds
    completed_total = generated + failed
    attempts_per_output = (
        float(total_attempts) / float(completed_total) if completed_total > 0 else 0.0
    )
    summary = {
        "mode": "batch-corpus-generate",
        "source": str(source),
        "output_root": str(output_root),
        "seed": int(seed),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "inputs": len(sharded_files),
        "variants_per_input": int(variants_per_input),
        "generated_outputs": int(generated),
        "resumed_skipped": int(resumed_skipped),
        "failed": int(failed),
        "jobs": int(jobs),
        "execution_profile": execution_profile,
        "effective_jobs": effective_jobs,
        "read_seconds": read_seconds,
        "process_seconds": process_seconds,
        "write_seconds": write_seconds,
        "retries": int(retries),
        "total_attempts": int(total_attempts),
        "retried_outputs": int(retried_outputs),
        "attempts_per_output": float(attempts_per_output),
        "elapsed_seconds": float(elapsed_seconds),
        "outputs_per_second": float(outputs_per_second),
        "num_shards": int(num_shards),
        "shard_index": int(shard_index),
        "manifest_jsonl": str(manifest_path),
    }
    if checkpoint_path is not None:
        summary["checkpoint_file"] = str(checkpoint_path)
    _write_json_atomic(summary_path, summary)
    console.print(
        f"Generated {generated} files across {len(sharded_files)} inputs "
        f"(resumed_skipped={resumed_skipped}, failed={failed})."
    )
    rates_table = Table(title="Corpus Generation Throughput")
    rates_table.add_column("Metric")
    rates_table.add_column("Value")
    rates_table.add_row("elapsed_seconds", f"{elapsed_seconds:.2f}")
    rates_table.add_row("outputs_per_second", f"{outputs_per_second:.2f}")
    rates_table.add_row("total_attempts", str(total_attempts))
    rates_table.add_row("retried_outputs", str(retried_outputs))
    rates_table.add_row("attempts_per_output", f"{attempts_per_output:.2f}")
    console.print(rates_table)
    console.print(f"Manifest: {manifest_path}")
    console.print(f"Summary: {summary_path}")
    if failed:
        raise typer.Exit(code=1)


def _time_shift_audio(*, audio: np.ndarray, sr: int, shift_ms: float) -> np.ndarray:
    """Shift audio in time while preserving original shape."""
    x = np.asarray(audio, dtype=np.float64)
    shift_samples = round((shift_ms / 1000.0) * float(sr))
    if shift_samples == 0 or x.shape[0] == 0:
        return x.copy()

    out = np.zeros_like(x, dtype=np.float64)
    if shift_samples > 0:
        if shift_samples >= x.shape[0]:
            return out
        out[shift_samples:, :] = x[:-shift_samples, :]
        return out

    abs_shift = abs(shift_samples)
    if abs_shift >= x.shape[0]:
        return out
    out[:-abs_shift, :] = x[abs_shift:, :]
    return out


def _generate_corpus_variant(
    *,
    infile: Path,
    outfile: Path,
    audio: np.ndarray,
    sr: int,
    source_index: int,
    variant_index: int,
    seed: int,
    time_shift_min_ms: float,
    time_shift_max_ms: float,
    pitch_shift_min_semitones: float,
    pitch_shift_max_semitones: float,
    reverb_rt60_min: float,
    reverb_rt60_max: float,
    reverb_wet_min: float,
    reverb_wet_max: float,
    reverb_pre_delay_min_ms: float,
    reverb_pre_delay_max_ms: float,
    output_subtype: str,
    retries: int,
) -> dict[str, Any]:
    """Render one corpus variant and return its manifest metadata."""
    max_attempts = max(1, int(retries) + 1)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            process_started = time.perf_counter()
            seed_seq = np.random.SeedSequence(
                [int(seed), int(source_index), int(variant_index), int(attempt)]
            )
            rng = np.random.default_rng(seed_seq)
            shift_ms = float(rng.uniform(time_shift_min_ms, time_shift_max_ms))
            pitch_steps = float(rng.uniform(pitch_shift_min_semitones, pitch_shift_max_semitones))
            rt60 = float(rng.uniform(reverb_rt60_min, reverb_rt60_max))
            wet = float(rng.uniform(reverb_wet_min, reverb_wet_max))
            pre_delay_ms = float(rng.uniform(reverb_pre_delay_min_ms, reverb_pre_delay_max_ms))
            damping = float(rng.uniform(0.35, 0.82))
            width = float(rng.uniform(0.90, 1.35))

            shifted = _time_shift_audio(audio=audio, sr=sr, shift_ms=shift_ms)
            pitched = _pitch_shift_audio(shifted, sr=sr, semitones=pitch_steps)
            reverb = AlgoReverbEngine(
                AlgoReverbConfig(
                    rt60=rt60,
                    pre_delay_ms=pre_delay_ms,
                    damping=damping,
                    width=width,
                    wet=wet,
                    dry=max(0.0, 1.0 - wet),
                )
            )
            processed = reverb.process(pitched, sr)
            process_seconds = time.perf_counter() - process_started
            write_started = time.perf_counter()
            sf.write(
                str(outfile),
                processed,
                sr,
                subtype={
                    "auto": None,
                    "float32": "FLOAT",
                    "float64": "DOUBLE",
                    "pcm16": "PCM_16",
                    "pcm24": "PCM_24",
                    "pcm32": "PCM_32",
                }[output_subtype],
            )
            return {
                "process_seconds": process_seconds,
                "write_seconds": time.perf_counter() - write_started,
                "source": str(infile),
                "output": str(outfile),
                "source_index": source_index,
                "variant_index": variant_index,
                "sample_rate": sr,
                "attempts": int(attempt),
                "augment": {
                    "time_shift_ms": shift_ms,
                    "pitch_shift_semitones": pitch_steps,
                    "reverb_rt60": rt60,
                    "reverb_wet": wet,
                    "reverb_pre_delay_ms": pre_delay_ms,
                    "reverb_damping": damping,
                    "reverb_width": width,
                },
            }
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            continue

    assert last_error is not None
    raise RuntimeError(
        f"corpus variant failed after {max_attempts} attempts: {last_error}"
    ) from last_error


def _pitch_shift_audio(audio: np.ndarray, *, sr: int, semitones: float) -> np.ndarray:
    """Pitch shift each channel while preserving signal length."""
    x = np.asarray(audio, dtype=np.float64)
    if abs(semitones) < 1e-9:
        return x.copy()
    out = np.zeros_like(x, dtype=np.float64)
    for ch in range(x.shape[1]):
        shifted = librosa.effects.pitch_shift(x[:, ch], sr=sr, n_steps=semitones)
        if shifted.shape[0] != x.shape[0]:
            shifted = librosa.util.fix_length(shifted, size=x.shape[0])
        out[:, ch] = np.asarray(shifted, dtype=np.float64)
    return out
