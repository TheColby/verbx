# IR Morph QA Guide

`verbx ir morph-sweep` is the Track D regression command for batch IR morph
analysis across deterministic alpha timelines.

## What it solves

- Generates one morphed IR per alpha step.
- Emits machine-readable QA artifacts (`CSV` + summary `JSON`).
- Supports long-run robustness via retries, checkpointing, and resume.
- Reuses deterministic cache artifacts for faster reruns and reproducibility.

## Quick start

```bash
verbx ir morph-sweep ir_a.wav ir_b.wav out/morph_sweep \
  --alpha-start 0.0 \
  --alpha-end 1.0 \
  --alpha-steps 9 \
  --workers 4 \
  --retries 1 \
  --checkpoint-file out/morph_sweep.checkpoint.json
```

## Explicit timeline example

```bash
verbx ir morph-sweep ir_a.wav ir_b.wav out/morph_sweep \
  --alpha 0.0 \
  --alpha 0.1 \
  --alpha 0.25 \
  --alpha 0.5 \
  --alpha 0.75 \
  --alpha 1.0 \
  --qa-json-out out/custom_summary.json \
  --qa-csv-out out/custom_metrics.csv
```

## Failure and resume workflow

```bash
# first pass
verbx ir morph-sweep ir_a.wav ir_b.wav out/morph_sweep \
  --alpha-start 0 --alpha-end 1 --alpha-steps 17 \
  --checkpoint-file out/morph.checkpoint.json \
  --retries 2 \
  --continue-on-error \
  --allow-failed

# resume pass
verbx ir morph-sweep ir_a.wav ir_b.wav out/morph_sweep \
  --alpha-start 0 --alpha-end 1 --alpha-steps 17 \
  --checkpoint-file out/morph.checkpoint.json \
  --resume
```

## Artifacts

- Sweep outputs:
  - `<out_dir>/<prefix>_<index>_aNNN.wav`
- Metrics CSV:
  - default `<out_dir>/morph_sweep_metrics.csv`
- Summary JSON:
  - default `<out_dir>/morph_sweep_summary.json`
- Optional checkpoint JSON:
  - path from `--checkpoint-file`

## Key metrics

The per-step CSV includes objective quality indicators:

- `rt60_drift_s`
- `early_late_drift_db`
- `spectral_distance_db`
- `interchannel_coherence_delta`

Use these fields to build regression gates and detect perceptual drift across
future code changes.
