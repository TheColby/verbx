from __future__ import annotations

from pathlib import Path

from verbx.config import RenderConfig
from verbx.core.batch_scheduler import BatchJobSpec, order_jobs, run_parallel_batch


def _job(idx: int, cost: float) -> BatchJobSpec:
    return BatchJobSpec(
        index=idx,
        infile=Path(f"in_{idx}.wav"),
        outfile=Path(f"out_{idx}.wav"),
        config=RenderConfig(),
        estimated_cost=cost,
    )


def test_order_jobs_longest_first() -> None:
    jobs = [_job(1, 2.0), _job(2, 5.0), _job(3, 1.0)]
    ordered = order_jobs(jobs, "longest-first")
    assert [job.index for job in ordered] == [2, 1, 3]


def test_parallel_batch_retries_then_success() -> None:
    jobs = [_job(1, 1.0)]
    attempts = {"count": 0}

    def runner(job: BatchJobSpec) -> None:
        _ = job
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("transient")

    results = run_parallel_batch(
        jobs=jobs,
        max_workers=1,
        schedule="fifo",
        retries=1,
        continue_on_error=False,
        runner=runner,
    )

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].attempts == 2
