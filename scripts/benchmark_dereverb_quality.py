#!/usr/bin/env python3
"""Objective quality benchmark harness for the verbx dereverb engine.

Runs PESQ-inspired (Bark-weighted SNR), STOI-approximation, and MCD (ASR
WER proxy) metrics across multiple reverb conditions and reports per-metric
pass/fail against configurable thresholds.

Usage::

    python scripts/benchmark_dereverb_quality.py
    python scripts/benchmark_dereverb_quality.py --json-out dereverb_quality.json
    python scripts/benchmark_dereverb_quality.py \\
        --snr-min 1.0 --stoi-min 0.25 --bark-snr-min 1.0 --mcd-max 12.0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verbx.core.dereverb import DereverbConfig, run_dereverb_benchmark  # noqa: E402


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Scenario:
    label: str
    rt60: float
    duration_s: float
    sr: int


SCENARIOS: list[Scenario] = [
    Scenario(label="short_room",      rt60=0.3,  duration_s=3.0, sr=24000),
    Scenario(label="medium_hall",     rt60=0.8,  duration_s=4.0, sr=24000),
    Scenario(label="long_reverb",     rt60=2.0,  duration_s=5.0, sr=24000),
    Scenario(label="extreme_tail",    rt60=4.0,  duration_s=5.0, sr=16000),
]

DEFAULT_CONFIGS: list[DereverbConfig] = [
    DereverbConfig(mode="wiener",      strength=0.65),
    DereverbConfig(mode="spectral_sub", strength=0.65),
]

# Default pass/fail thresholds
DEFAULT_SNR_MIN_DB: float = 1.0          # minimum SNR improvement over reverberant baseline
DEFAULT_BARK_SNR_MIN_DB: float = 1.0     # minimum Bark-weighted SNR improvement
DEFAULT_STOI_MIN: float = 0.25           # minimum STOI approximation score [0,1]
DEFAULT_MCD_MAX_DB: float = 15.0         # maximum MCD (lower = better spectral match)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _run_scenario(
    scenario: Scenario,
    configs: list[DereverbConfig],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    report = run_dereverb_benchmark(
        sr=scenario.sr,
        duration_s=scenario.duration_s,
        rt60=scenario.rt60,
        configs=configs,
    )

    snr_min = thresholds["snr_min_db"]
    bark_min = thresholds["bark_snr_min_db"]
    stoi_min = thresholds["stoi_min"]
    mcd_max = thresholds["mcd_max_db"]

    results_with_verdict: list[dict[str, Any]] = []
    scenario_pass = True
    for result in report["results"]:
        r = dict(result)
        snr_imp = float(r.get("snr_improvement_db", 0.0))
        bark_imp = float(r.get("bark_snr_improvement_db", 0.0))
        stoi = float(r.get("stoi_approx", 0.0))
        mcd = float(r.get("mcd_db", 0.0))

        verdict: dict[str, bool] = {
            "snr_improvement_pass": snr_imp >= snr_min,
            "bark_snr_improvement_pass": bark_imp >= bark_min,
            "stoi_pass": stoi >= stoi_min,
            "mcd_pass": mcd <= mcd_max,
        }
        r["verdict"] = verdict
        r["pass"] = all(verdict.values())
        if not r["pass"]:
            scenario_pass = False
        results_with_verdict.append(r)

    return {
        "scenario": scenario.label,
        "rt60": scenario.rt60,
        "sr": scenario.sr,
        "duration_s": scenario.duration_s,
        "baseline": {
            "snr_reverberant_db": report.get("snr_reverberant_db"),
            "bark_snr_reverberant_db": report.get("bark_snr_reverberant_db"),
            "stoi_reverberant": report.get("stoi_reverberant"),
            "mcd_reverberant_db": report.get("mcd_reverberant_db"),
        },
        "results": results_with_verdict,
        "pass": scenario_pass,
    }


def run_benchmark(
    scenarios: list[Scenario] | None = None,
    configs: list[DereverbConfig] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    if scenarios is None:
        scenarios = SCENARIOS
    if configs is None:
        configs = DEFAULT_CONFIGS
    if thresholds is None:
        thresholds = {
            "snr_min_db": DEFAULT_SNR_MIN_DB,
            "bark_snr_min_db": DEFAULT_BARK_SNR_MIN_DB,
            "stoi_min": DEFAULT_STOI_MIN,
            "mcd_max_db": DEFAULT_MCD_MAX_DB,
        }

    scenario_reports: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_reports.append(_run_scenario(scenario, configs, thresholds))

    overall_pass = all(s["pass"] for s in scenario_reports)
    return {
        "schema": "dereverb-quality-benchmark-v1",
        "thresholds": thresholds,
        "scenarios": scenario_reports,
        "pass": overall_pass,
        "total_scenarios": len(scenario_reports),
        "passed_scenarios": sum(1 for s in scenario_reports if s["pass"]),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Objective quality benchmark harness for verbx dereverb.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--json-out",
        metavar="PATH",
        help="Write JSON report to this file (default: print to stdout)",
    )
    p.add_argument(
        "--snr-min",
        type=float,
        default=DEFAULT_SNR_MIN_DB,
        metavar="DB",
        help="Minimum required SNR improvement over reverberant baseline (dB)",
    )
    p.add_argument(
        "--bark-snr-min",
        type=float,
        default=DEFAULT_BARK_SNR_MIN_DB,
        metavar="DB",
        help="Minimum required Bark-weighted SNR improvement (dB, PESQ-style)",
    )
    p.add_argument(
        "--stoi-min",
        type=float,
        default=DEFAULT_STOI_MIN,
        metavar="SCORE",
        help="Minimum required STOI approximation score [0, 1]",
    )
    p.add_argument(
        "--mcd-max",
        type=float,
        default=DEFAULT_MCD_MAX_DB,
        metavar="DB",
        help="Maximum allowed mel-cepstral distortion (dB, ASR WER proxy; lower = better)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    thresholds = {
        "snr_min_db": args.snr_min,
        "bark_snr_min_db": args.bark_snr_min,
        "stoi_min": args.stoi_min,
        "mcd_max_db": args.mcd_max,
    }

    print("Running dereverb quality benchmark...", flush=True)
    report = run_benchmark(thresholds=thresholds)

    json_str = json.dumps(report, indent=2)

    if args.json_out:
        Path(args.json_out).write_text(json_str, encoding="utf-8")
        print(f"Report written to {args.json_out}")
    else:
        print(json_str)

    # Print summary table
    print(f"\nScenarios: {report['passed_scenarios']}/{report['total_scenarios']} passed")
    for s in report["scenarios"]:
        status = "PASS" if s["pass"] else "FAIL"
        print(f"  [{status}] {s['scenario']} (RT60={s['rt60']}s)")
        for r in s["results"]:
            mode_status = "PASS" if r["pass"] else "FAIL"
            print(
                f"         [{mode_status}] mode={r['mode']}"
                f"  SNR_imp={r.get('snr_improvement_db', '?'):+.2f}dB"
                f"  bark_imp={r.get('bark_snr_improvement_db', '?'):+.2f}dB"
                f"  STOI={r.get('stoi_approx', '?'):.3f}"
                f"  MCD={r.get('mcd_db', '?'):.1f}dB"
            )

    if not report["pass"]:
        print("\nBenchmark FAILED — one or more scenarios did not meet thresholds.", file=sys.stderr)
        return 1

    print("\nBenchmark PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
