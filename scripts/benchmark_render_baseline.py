#!/usr/bin/env python3
"""Run deterministic render micro-benchmarks and compare against baseline budgets."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from verbx.config import RenderConfig
from verbx.core.pipeline import run_render_pipeline

ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class ScenarioResult:
    """One benchmark scenario result."""

    name: str
    elapsed_seconds: float
    output_seconds: float
    realtime_factor: float
    output_channels: int
    output_samples: int


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float64), int(sr), subtype="DOUBLE")


def _scenario_algo_short(tmp: Path) -> ScenarioResult:
    sr = 16_000
    n = int(1.0 * sr)
    t = np.arange(n, dtype=np.float64) / float(sr)
    x = (0.22 * np.sin(2.0 * np.pi * 180.0 * t)).astype(np.float64)[:, np.newaxis]
    x[200:900, 0] += 0.15

    infile = tmp / "algo_short_in.wav"
    outfile = tmp / "algo_short_out.wav"
    _write_wav(infile, x, sr)

    config = RenderConfig(
        engine="algo",
        rt60=4.0,
        fdn_lines=8,
        allpass_stages=4,
        wet=0.82,
        dry=0.18,
        silent=True,
        progress=False,
        normalize_stage="none",
        output_peak_norm="none",
    )
    start = time.perf_counter()
    report = run_render_pipeline(infile=infile, outfile=outfile, config=config)
    elapsed = max(1e-9, time.perf_counter() - start)
    out_samples = int(report.get("output_samples", 0))
    out_channels = int(report.get("channels", 0))
    out_seconds = float(out_samples) / float(sr)
    return ScenarioResult(
        name="algo_short_room",
        elapsed_seconds=float(elapsed),
        output_seconds=out_seconds,
        realtime_factor=out_seconds / float(elapsed),
        output_channels=out_channels,
        output_samples=out_samples,
    )


def _scenario_algo_long_tail(tmp: Path) -> ScenarioResult:
    sr = 2_000
    x = np.zeros((1_000, 1), dtype=np.float64)
    x[0, 0] = 0.8

    infile = tmp / "algo_long_tail_in.wav"
    outfile = tmp / "algo_long_tail_out.wav"
    _write_wav(infile, x, sr)

    config = RenderConfig(
        engine="algo",
        rt60=130.0,
        fdn_lines=4,
        allpass_stages=2,
        wet=0.9,
        dry=0.1,
        silent=True,
        progress=False,
        normalize_stage="none",
        output_peak_norm="none",
    )
    start = time.perf_counter()
    report = run_render_pipeline(infile=infile, outfile=outfile, config=config)
    elapsed = max(1e-9, time.perf_counter() - start)
    out_samples = int(report.get("output_samples", 0))
    out_channels = int(report.get("channels", 0))
    out_seconds = float(out_samples) / float(sr)
    return ScenarioResult(
        name="algo_long_tail_130s",
        elapsed_seconds=float(elapsed),
        output_seconds=out_seconds,
        realtime_factor=out_seconds / float(elapsed),
        output_channels=out_channels,
        output_samples=out_samples,
    )


def _scenario_conv_matrix(tmp: Path) -> ScenarioResult:
    sr = 16_000
    n = int(0.8 * sr)
    x = np.zeros((n, 2), dtype=np.float64)
    x[120:420, 0] = 0.5
    x[200:500, 1] = -0.45

    ir = np.zeros((192, 4), dtype=np.float64)
    ir[0, 0] = 1.0
    ir[0, 3] = 1.0
    ir[12, 1] = 0.45
    ir[19, 2] = 0.35

    infile = tmp / "conv_matrix_in.wav"
    irfile = tmp / "conv_matrix_ir.wav"
    outfile = tmp / "conv_matrix_out.wav"
    _write_wav(infile, x, sr)
    _write_wav(irfile, ir, sr)

    config = RenderConfig(
        engine="conv",
        ir=str(irfile),
        ir_route_map="full",
        wet=1.0,
        dry=0.0,
        silent=True,
        progress=False,
        normalize_stage="none",
        output_peak_norm="none",
        input_layout="stereo",
        output_layout="stereo",
    )
    start = time.perf_counter()
    report = run_render_pipeline(infile=infile, outfile=outfile, config=config)
    elapsed = max(1e-9, time.perf_counter() - start)
    out_samples = int(report.get("output_samples", 0))
    out_channels = int(report.get("channels", 0))
    out_seconds = float(out_samples) / float(sr)
    return ScenarioResult(
        name="conv_matrix_stereo",
        elapsed_seconds=float(elapsed),
        output_seconds=out_seconds,
        realtime_factor=out_seconds / float(elapsed),
        output_channels=out_channels,
        output_samples=out_samples,
    )


def _load_baseline(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _evaluate(
    *,
    results: list[ScenarioResult],
    baseline: dict[str, Any],
    compare_threshold: float,
) -> tuple[dict[str, Any], bool]:
    baseline_scenarios = baseline.get("scenarios", {})
    if not isinstance(baseline_scenarios, dict):
        baseline_scenarios = {}

    scenario_payload: list[dict[str, Any]] = []
    has_regression = False
    for item in results:
        baseline_entry = baseline_scenarios.get(item.name, {})
        reference = None
        budget = None
        if isinstance(baseline_entry, dict):
            ref_raw = baseline_entry.get("reference_seconds")
            budget_raw = baseline_entry.get("max_seconds")
            if isinstance(ref_raw, (int, float)) and float(ref_raw) > 0.0:
                reference = float(ref_raw)
            if isinstance(budget_raw, (int, float)) and float(budget_raw) > 0.0:
                budget = float(budget_raw)

        ratio_to_reference = None if reference is None else item.elapsed_seconds / reference
        over_budget = None if budget is None else item.elapsed_seconds > budget
        regression = (
            ratio_to_reference is not None and ratio_to_reference > float(compare_threshold)
        ) or (over_budget is True)
        has_regression = has_regression or bool(regression)

        scenario_payload.append(
            {
                **asdict(item),
                "baseline_reference_seconds": reference,
                "baseline_max_seconds": budget,
                "ratio_to_reference": ratio_to_reference,
                "compare_threshold": float(compare_threshold),
                "over_budget": bool(over_budget) if over_budget is not None else None,
                "regression": bool(regression),
            }
        )

    summary = {
        "scenario_count": len(results),
        "regression_count": int(sum(1 for row in scenario_payload if row["regression"])),
        "has_regression": bool(has_regression),
    }
    return {"scenarios": scenario_payload, "summary": summary}, bool(has_regression)


def _render_markdown_table(report: dict[str, Any]) -> str:
    scenarios = report.get("scenarios", [])
    if not isinstance(scenarios, list) or len(scenarios) == 0:
        return "No scenarios executed."
    lines = [
        "| Scenario | elapsed_s | output_s | realtime_x | ratio_to_ref | regression |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in scenarios:
        if not isinstance(row, dict):
            continue
        ratio = row.get("ratio_to_reference")
        ratio_text = "-" if ratio is None else f"{float(ratio):.3f}"
        lines.append(
            "| {name} | {elapsed:.3f} | {out:.3f} | {rt:.2f} | {ratio} | {reg} |".format(
                name=str(row.get("name", "")),
                elapsed=float(row.get("elapsed_seconds", 0.0)),
                out=float(row.get("output_seconds", 0.0)),
                rt=float(row.get("realtime_factor", 0.0)),
                ratio=ratio_text,
                reg="yes" if bool(row.get("regression", False)) else "no",
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "docs/benchmarks/ci_baseline.json",
        help="Baseline JSON with per-scenario reference/max seconds.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "docs/benchmarks/latest_report.json",
        help="Output report JSON path.",
    )
    parser.add_argument(
        "--compare-threshold",
        type=float,
        default=3.0,
        help="Regression threshold multiplier against baseline reference seconds.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when regression is detected.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare_threshold = float(max(1.0, args.compare_threshold))
    baseline = _load_baseline(args.baseline)

    with tempfile.TemporaryDirectory(prefix="verbx_bench_") as tmp_dir:
        tmp = Path(tmp_dir)
        results = [
            _scenario_algo_short(tmp),
            _scenario_algo_long_tail(tmp),
            _scenario_conv_matrix(tmp),
        ]

    evaluated, has_regression = _evaluate(
        results=results,
        baseline=baseline,
        compare_threshold=compare_threshold,
    )
    report = {
        "tool": "scripts/benchmark_render_baseline.py",
        "compare_threshold": compare_threshold,
        "baseline_path": str(args.baseline),
        "baseline_version": baseline.get("version") if isinstance(baseline, dict) else None,
        **evaluated,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(_render_markdown_table(report))
    print(f"Report written: {args.json_out}")

    if args.fail_on_regression and has_regression:
        print("Regression detected.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
