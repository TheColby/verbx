# Render Performance Baseline

This directory tracks a small deterministic benchmark suite for CI comparison.

## What it measures

`scripts/benchmark_render_baseline.py` runs four representative scenarios:

- `algo_short_room`: short algorithmic render (production-style room settings).
- `algo_long_tail_130s`: long-tail algorithmic regression fixture (`RT60=130s`).
- `conv_matrix_stereo`: matrix-routed stereo convolution regression fixture.
- `conv_target_sr_192k`: convolution render with integrated `--target-sr 192000`
  sample-rate conversion path.

The benchmark emits:

- wall-clock elapsed seconds,
- output duration seconds,
- realtime factor (`output_seconds / elapsed_seconds`),
- comparison against baseline reference/budget values.

## Baseline file

`ci_baseline.json` defines per-scenario:

- `reference_seconds`: expected reference runtime.
- `max_seconds`: soft budget threshold for obvious regressions.

## Run locally

```bash
uv run python scripts/benchmark_render_baseline.py \
  --baseline docs/benchmarks/ci_baseline.json \
  --json-out docs/benchmarks/latest_report.json \
  --compare-threshold 3.0
```

To fail locally on detected regressions:

```bash
uv run python scripts/benchmark_render_baseline.py \
  --baseline docs/benchmarks/ci_baseline.json \
  --json-out docs/benchmarks/latest_report.json \
  --compare-threshold 3.0 \
  --fail-on-regression
```

## CI integration

The `perf-baseline` CI job runs this script on `ubuntu-latest` and uploads the
JSON report artifact for trend inspection and release triage.
