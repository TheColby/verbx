from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts/benchmark_render_baseline.py"
    spec = importlib.util.spec_from_file_location("benchmark_render_baseline", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _result(module, *, name: str = "fixture", elapsed: float = 1.0):
    return module.ScenarioResult(
        name=name,
        elapsed_seconds=elapsed,
        output_seconds=2.0,
        realtime_factor=2.0,
        output_channels=1,
        output_samples=32_000,
    )


def test_evaluate_accepts_result_within_reference_and_budget() -> None:
    module = _load_module()

    report, has_regression = module._evaluate(
        results=[_result(module, elapsed=1.5)],
        baseline={
            "scenarios": {
                "fixture": {
                    "reference_seconds": 1.0,
                    "max_seconds": 2.0,
                }
            }
        },
        compare_threshold=2.0,
    )

    assert has_regression is False
    assert report["summary"] == {
        "scenario_count": 1,
        "regression_count": 0,
        "missing_baseline_count": 0,
        "has_regression": False,
    }
    assert report["scenarios"][0]["baseline_missing"] is False


def test_evaluate_flags_reference_ratio_regression() -> None:
    module = _load_module()

    report, has_regression = module._evaluate(
        results=[_result(module, elapsed=2.01)],
        baseline={"scenarios": {"fixture": {"reference_seconds": 1.0}}},
        compare_threshold=2.0,
    )

    assert has_regression is True
    assert report["summary"]["regression_count"] == 1
    assert report["scenarios"][0]["ratio_to_reference"] == 2.01
    assert report["scenarios"][0]["regression"] is True


def test_evaluate_flags_absolute_budget_regression() -> None:
    module = _load_module()

    report, has_regression = module._evaluate(
        results=[_result(module, elapsed=1.01)],
        baseline={"scenarios": {"fixture": {"max_seconds": 1.0}}},
        compare_threshold=3.0,
    )

    assert has_regression is True
    assert report["scenarios"][0]["over_budget"] is True


def test_evaluate_flags_missing_scenario_baseline() -> None:
    module = _load_module()

    report, has_regression = module._evaluate(
        results=[_result(module)],
        baseline={"scenarios": {}},
        compare_threshold=3.0,
    )

    assert has_regression is True
    assert report["summary"]["missing_baseline_count"] == 1
    assert report["scenarios"][0]["baseline_missing"] is True
    assert report["scenarios"][0]["regression"] is True
