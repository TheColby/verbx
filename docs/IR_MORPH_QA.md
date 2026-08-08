# IR Morph Quality Assurance

`verbx ir morph-sweep` renders a repeatable sequence between two impulse responses and records enough evidence to compare that sequence across revisions. It is intended for regression work, parameter studies, and review sessions in which one isolated morph is not sufficient. The command does not pronounce a render perceptually correct. It produces audio, measurements, and provenance that make an informed decision easier to repeat.

## What the Sweep Produces

A sweep evaluates an ordered alpha timeline. Alpha zero corresponds to the first IR, alpha one corresponds to the second, and intermediate values describe the selected morph algorithm's path between them. For each value, verbx writes a morphed WAV and one row of quality measurements. It also writes a CSV for detailed comparison and a JSON summary for automation.

The command supports parallel workers, bounded retries, checkpointing, and resume. Those controls matter when the endpoint IRs are long or highly multichannel: an interrupted study can continue without discarding successful results, and repeated inputs can reuse deterministic cache artifacts.

## A Uniform Timeline

The shortest useful experiment samples the interval uniformly. Nine steps include both endpoints and seven intermediate states:

```bash
verbx ir morph-sweep ir_a.wav ir_b.wav out/morph_sweep \
  --alpha-start 0.0 \
  --alpha-end 1.0 \
  --alpha-steps 9 \
  --workers 4 \
  --retries 1 \
  --checkpoint-file out/morph_sweep.checkpoint.json
```

Listen to the endpoints first. A smooth numerical trajectory cannot rescue an incorrect source IR, channel mismatch, or unintended normalization. Then audition the sequence in order and at matched monitoring level. Discontinuities are often easier to hear as a progression than as unrelated files.

## A Deliberate Timeline

Uniform sampling is convenient, but it can miss a narrow transition. Repeat `--alpha` to concentrate renders near a region where timbre, decay, or spatial image changes quickly:

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

An explicit timeline is also useful when alpha values correspond to scene cues, edit points, or a control curve used elsewhere. Preserve the list with the test assets so that a later run compares the same states rather than merely the same endpoints.

## Checkpoint and Resume

Long studies should write a checkpoint outside the output filenames themselves. The first command below records each completed job and continues after an isolated failure. The second reads that checkpoint and skips completed outputs:

```bash
# Initial pass
verbx ir morph-sweep ir_a.wav ir_b.wav out/morph_sweep \
  --alpha-start 0 --alpha-end 1 --alpha-steps 17 \
  --checkpoint-file out/morph.checkpoint.json \
  --retries 2 \
  --continue-on-error \
  --allow-failed

# Resume after correcting the failure or restoring the worker
verbx ir morph-sweep ir_a.wav ir_b.wav out/morph_sweep \
  --alpha-start 0 --alpha-end 1 --alpha-steps 17 \
  --checkpoint-file out/morph.checkpoint.json \
  --resume
```

`--allow-failed` is appropriate for exploratory batches in which partial evidence is useful. It should not quietly weaken a release gate. In CI, prefer a nonzero exit when any required render fails, and archive the checkpoint only as diagnostic evidence.

## Output Artifacts

The output directory is a small experiment record, not merely a folder of audio. Its contents have separate jobs:

- `<out_dir>/<prefix>_<index>_aNNN.wav` contains the rendered IR for one alpha value.

- `<out_dir>/morph_sweep_metrics.csv` contains one row per planned render, including status, timing, cache state, audio properties, and quality measurements.

- `<out_dir>/morph_sweep_summary.json` records the command mode, endpoints, alpha list, scheduling choices, completion counts, artifact paths, and aggregate statistics.

- The optional checkpoint JSON at `--checkpoint-file` records recoverable progress for a later `--resume` run.

Keep the CSV and summary beside any listening notes. A WAV sequence without its alpha values and mode is difficult to reproduce, while measurements without the corresponding sound invite conclusions detached from audibility.

## Reading the Measurements

Each metric compares the output with a target interpolated from measurements of the two endpoints. Smaller drift is generally desirable, but none of the quantities is a universal acceptance threshold.

- `rt60_target_s` and `rt60_out_s` report target and measured reverberation time in seconds. `rt60_drift_s` is their absolute difference in seconds.

- `early_late_target_db` and `early_late_out_db` report the target and measured early-to-late energy ratio in decibels. `early_late_drift_db` is the absolute difference in decibels.

- `spectral_distance_db` measures distance from the interpolated endpoint spectrum in decibels. Inspect frequency-localized problems with a separate spectral plot rather than treating this scalar as a complete tonal diagnosis.

- `interchannel_coherence_target` and `interchannel_coherence_out` are unitless coherence scores. `interchannel_coherence_delta` is their absolute difference. A small delta preserves the target statistic; it does not prove correct localization or a convincing immersive field.

A missing value can be meaningful. Very short, silent, or pathological IRs may not support a stable decay estimate. Do not convert missing measurements to zero, because zero drift and unavailable evidence are different conditions.

## Building a Regression Gate

A useful gate begins with a known fixture pair and limits chosen from repeated good renders, not arbitrary round numbers. Measure several accepted runs, study normal variation, then set tolerances wide enough to ignore harmless numerical noise and narrow enough to catch the artifact under investigation. Keep separate gates for channel count, sample rate, completion, and objective drift so that a routing failure is not hidden inside an aggregate score.

After the automated gate passes, audition at least the endpoints, midpoint, and the states nearest the largest reported drift. For multichannel material, include the intended layout and at least one required fold-down. The final question is not whether every curve is smooth. It is whether the morph behaves continuously, preserves the intended spatial and decay relationships, and remains usable in the production context that motivated the test.
