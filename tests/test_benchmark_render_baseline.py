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


def test_warmup_runs_every_path_before_fresh_measurements(tmp_path, monkeypatch) -> None:
    module = _load_module()
    calls = []

    def scenario(name):
        def run(directory):
            calls.append((name, directory.name))
            # Warmups simulate expensive first-use compilation. Only the second
            # render's throughput should enter the baseline comparison.
            return _result(module, name=name, elapsed=50 if directory.name == "first_render" else 1)

        return run

    names = [
        "_scenario_algo_short",
        "_scenario_algo_long_tail",
        "_scenario_conv_matrix",
        "_scenario_conv_target_sr_192k",
    ]
    for name in names:
        monkeypatch.setattr(module, name, scenario(name))
    measured, first = module._run_scenarios(tmp_path)
    assert calls == [(name, "first_render") for name in names] + [
        (name, "measured") for name in names
    ]
    assert all(item.elapsed_seconds == 50 for item in first)
    assert all(item.elapsed_seconds == 1 for item in measured)


def test_warmed_regression_still_fails_and_first_render_is_reported(tmp_path, monkeypatch) -> None:
    import argparse
    import json

    module = _load_module()
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"scenarios": {"fixture": {"max_seconds": 2}}}))
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            baseline=baseline,
            json_out=output,
            compare_threshold=3,
            fail_on_regression=True,
        ),
    )
    monkeypatch.setattr(
        module,
        "_run_scenarios",
        lambda tmp: (
            [_result(module, elapsed=3)],
            [_result(module, elapsed=50)],
        ),
    )
    assert module.main() == 1
    report = json.loads(output.read_text())
    assert report["scenarios"][0]["elapsed_seconds"] == 3
    assert report["scenarios"][0]["over_budget"] is True
    assert report["first_render_scenarios"][0]["elapsed_seconds"] == 50
    assert report["timing_policy"]["comparison"] == "warmed_render"


def test_slow_first_render_does_not_fail_warmed_throughput(tmp_path, monkeypatch) -> None:
    import argparse
    import json

    module = _load_module()
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"scenarios": {"fixture": {"max_seconds": 2}}}))
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            baseline=baseline,
            json_out=output,
            compare_threshold=3,
            fail_on_regression=True,
        ),
    )
    monkeypatch.setattr(
        module,
        "_run_scenarios",
        lambda tmp: (
            [_result(module, elapsed=1)],
            [_result(module, elapsed=50)],
        ),
    )
    assert module.main() == 0
    report = json.loads(output.read_text())
    assert report["summary"]["has_regression"] is False
    assert report["first_render_scenarios"][0]["elapsed_seconds"] == 50
