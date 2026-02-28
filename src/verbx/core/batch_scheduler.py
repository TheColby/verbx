"""Parallel batch scheduler with job ordering and retry handling."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import soundfile as sf

from verbx.config import RenderConfig

BatchSchedulePolicy = Literal["fifo", "shortest-first", "longest-first"]


@dataclass(slots=True)
class BatchJobSpec:
    """Prepared batch job with runtime estimate used for scheduling."""

    index: int
    infile: Path
    outfile: Path
    config: RenderConfig
    estimated_cost: float


@dataclass(slots=True)
class BatchJobResult:
    """Result for one scheduled batch job."""

    index: int
    outfile: Path
    success: bool
    attempts: int
    duration_seconds: float
    estimated_cost: float
    error: str | None = None


BatchRunner = Callable[[BatchJobSpec], None]
BatchResultCallback = Callable[[BatchJobResult], None]


def estimate_job_cost(infile: Path, config: RenderConfig) -> float:
    """Estimate relative render cost for ordering."""
    try:
        info = sf.info(str(infile))
        duration_seconds = (
            float(info.frames) / float(info.samplerate) if info.samplerate > 0 else 0.0
        )
    except (RuntimeError, TypeError, ValueError):
        duration_seconds = 0.0

    is_convolution = config.engine == "conv" or (config.engine == "auto" and config.ir is not None)
    base = max(0.25, duration_seconds)
    repeat_factor = max(1.0, float(config.repeat))
    tail_factor = 1.0 if is_convolution else float(1.0 + min(3.0, config.rt60 / 60.0))
    ir_gen_factor = 1.3 if config.ir_gen else 1.0
    return float(base * repeat_factor * tail_factor * ir_gen_factor)


def order_jobs(
    jobs: list[BatchJobSpec],
    schedule: BatchSchedulePolicy,
) -> list[BatchJobSpec]:
    """Order jobs according to selected policy."""
    if schedule == "fifo":
        return sorted(jobs, key=lambda item: item.index)
    if schedule == "shortest-first":
        return sorted(jobs, key=lambda item: (item.estimated_cost, item.index))
    if schedule == "longest-first":
        return sorted(jobs, key=lambda item: (-item.estimated_cost, item.index))
    msg = f"Unsupported schedule policy: {schedule}"
    raise ValueError(msg)


def run_parallel_batch(
    *,
    jobs: list[BatchJobSpec],
    max_workers: int,
    schedule: BatchSchedulePolicy,
    retries: int,
    continue_on_error: bool,
    runner: BatchRunner,
    on_result: BatchResultCallback | None = None,
) -> list[BatchJobResult]:
    """Run batch jobs in parallel with policy ordering and retries."""
    if not jobs:
        return []

    ordered = order_jobs(jobs, schedule)
    worker_count = max(1, min(max_workers, len(ordered)))
    results: list[BatchJobResult] = []

    futures: dict[Future[BatchJobResult], BatchJobSpec] = {}
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="verbx-batch-v04") as pool:
        for job in ordered:
            fut = pool.submit(_run_job_with_retries, job, retries, runner)
            futures[fut] = job

        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            if on_result is not None:
                on_result(result)

            if (not result.success) and (not continue_on_error):
                for pending in futures:
                    if not pending.done():
                        pending.cancel()
                msg = (
                    f"Batch job {result.index} failed after {result.attempts} "
                    f"attempt(s): {result.error}"
                )
                raise RuntimeError(msg)

    return sorted(results, key=lambda item: item.index)


def _run_job_with_retries(
    job: BatchJobSpec,
    retries: int,
    runner: BatchRunner,
) -> BatchJobResult:
    start = time.perf_counter()
    max_attempts = max(1, retries + 1)
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            runner(job)
            duration = float(time.perf_counter() - start)
            return BatchJobResult(
                index=job.index,
                outfile=job.outfile,
                success=True,
                attempts=attempt,
                duration_seconds=duration,
                estimated_cost=job.estimated_cost,
            )
        except Exception as exc:  # pragma: no cover - integration exercised through CLI tests
            last_error = str(exc)

    duration = float(time.perf_counter() - start)
    return BatchJobResult(
        index=job.index,
        outfile=job.outfile,
        success=False,
        attempts=max_attempts,
        duration_seconds=duration,
        estimated_cost=job.estimated_cost,
        error=last_error,
    )
