# JSON Schemas and Field Contracts

_Status: public alpha (`0.7.x`)_

This document defines practical field contracts for machine-generated manifests
and automation payloads used by `verbx`.

---

## 1) `batch render` manifest

Top-level object:

```json
{
  "version": "0.5",
  "jobs": [
    {
      "infile": "input.wav",
      "outfile": "out.wav",
      "options": {
        "engine": "algo",
        "rt60": 2.5,
        "wet": 0.3,
        "dry": 0.7
      }
    }
  ]
}
```

Required:
- `version` (string)
- `jobs` (array)
- per-job: `infile`, `outfile`, `options`

---

## 2) `batch augment` manifest

Top-level object:

```json
{
  "version": "0.7",
  "dataset_name": "dataset_name",
  "profile": "asr-reverb-v1",
  "seed": 20260322,
  "variants_per_input": 4,
  "output_root": "augmented_out",
  "jobs": [
    {
      "id": "utt_0001",
      "infile": "dry.wav",
      "split": "train",
      "label": "speaker_a",
      "tags": ["speech"],
      "options": {
        "rt60": 0.35,
        "wet": 0.25,
        "dry": 0.9
      }
    }
  ]
}
```

Required:
- `version`, `dataset_name`, `profile`, `jobs`
- per-job: `id`, `infile`

Optional but recommended:
- `seed`, `variants_per_input`, `output_root`, `split`, `label`, `tags`

---

## 3) `--automation-file` JSON

Top-level object:

```json
{
  "version": "1",
  "points": [
    {"target": "wet", "time_ms": 0, "value": 0.2},
    {"target": "wet", "time_ms": 5000, "value": 0.8}
  ]
}
```

Required:
- `points` array of objects with:
  - `target` (string)
  - `time_ms` (number, >= 0)
  - `value` (number)

---

## 4) `batch corpus-generate` summary JSON

Key fields written to `corpus_generation_summary.json`:

- identity: `mode`, `source`, `output_root`, `seed`
- scale: `inputs`, `variants_per_input`, `generated_outputs`
- resilience: `retries`, `total_attempts`, `retried_outputs`, `failed`, `resumed_skipped`
- throughput: `elapsed_seconds`, `outputs_per_second`, `attempts_per_output`
- sharding: `num_shards`, `shard_index`
- artifacts: `manifest_jsonl`, optional `checkpoint_file`

---

## Validation note

During alpha, schemas are contract docs (not yet published as JSON-Schema files).
CLI validations remain the source of truth.
