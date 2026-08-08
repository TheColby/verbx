from __future__ import annotations

from pathlib import Path


def _ci_workflow() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / ".github/workflows/ci.yml").read_text(encoding="utf-8")


def test_performance_job_blocks_on_benchmark_regression() -> None:
    workflow = _ci_workflow()
    job = workflow.split("  perf-baseline:", maxsplit=1)[1].split(
        "  native-parity:", maxsplit=1
    )[0]

    assert "scripts/benchmark_render_baseline.py" in job
    assert "--fail-on-regression" in job
    assert "if: always()" in job


def test_native_job_builds_and_blocks_on_structural_parity() -> None:
    workflow = _ci_workflow()
    job = workflow.split("  native-parity:", maxsplit=1)[1]

    assert "scripts/compare_native_render_parity.py" in job
    assert "--build-native" in job
    assert "--strict-structural" in job
    assert "native_parity_report.json" in job
