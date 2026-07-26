# JSON Schemas: Manifests and Automation

This document defines the structured JSON contracts used by `verbx batch` and render automation.

- Batch render manifest schema id: `verbx-batch-manifest-v0.5`
- Batch augmentation manifest schema id: `verbx-augment-manifest-v0.7`
- Automation file schema id: `verbx-automation-v0.7`

> These are documentation schemas (stable contract docs). They are designed to mirror current parser behavior.

---

## 1) Batch Render Manifest (`verbx-batch-manifest-v0.5`)

Use with:

```bash
verbx batch render manifest.json --jobs 8
```

Minimal valid shape:

```json
{
  "version": "0.5",
  "jobs": [
    {
      "infile": "input.wav",
      "outfile": "output.wav",
      "options": {
        "engine": "auto",
        "rt60": 60.0,
        "wet": 0.8,
        "dry": 0.2,
        "repeat": 1
      }
    }
  ]
}
```

### Required fields

- `jobs` (array): list of batch jobs.
- For each `jobs[]` item:
  - `infile` (string path)
  - `outfile` (string path)
  - `options` (object; any `RenderConfig` key/value)

### Notes

- `version` is recommended for compatibility tracking.
- `options` is validated against `RenderConfig` semantics at runtime.

---

## 2) Batch Augmentation Manifest (`verbx-augment-manifest-v0.7`)

Use with:

```bash
verbx batch augment augment.json --jobs 8
```

Template shape:

```json
{
  "version": "0.7",
  "dataset_name": "verbx_augmented_set",
  "profile": "asr-reverb-v1",
  "seed": 20260314,
  "variants_per_input": 4,
  "output_root": "augmented_out",
  "write_analysis": false,
  "default_options": {
    "engine": "algo",
    "repeat": 1,
    "output_subtype": "float32",
    "normalize_stage": "none",
    "output_peak_norm": "input"
  },
  "jobs": [
    {
      "id": "utt_0001",
      "infile": "data/clean/utt_0001.wav",
      "split": "train",
      "label": "speaker_a",
      "tags": ["speech", "clean"],
      "variants": 6,
      "options": {"rt60": 1.1},
      "metadata": {"speaker_id": "spk_a", "language": "en"}
    }
  ]
}
```

### Required fields

- `jobs` (non-empty array)
- For each `jobs[]` item:
  - `infile` (string path)

### Common optional fields

Top-level:
- `version` (string)
- `dataset_name` (string)
- `profile` (string; one of built-in profiles)
- `seed` (positive integer)
- `variants_per_input` (positive integer, max 500)
- `output_root` (string path)
- `write_analysis` (boolean)
- `default_options` (object; merged into each variant)

Per-job:
- `id`, `source_id` (string)
- `split` (string; defaults to `train`)
- `label` (string)
- `tags` (array of strings)
- `variants` (positive integer, max 500)
- `options` (object)
- `metadata` (object)

---

## 3) Render Automation File (`verbx-automation-v0.7`)

Use with:

```bash
verbx render in.wav out.wav --automation-file automation.json
```

Minimal shape:

```json
{
  "mode": "block",
  "block_ms": 20.0,
  "lanes": [
    {
      "target": "wet",
      "points": [[0.0, 0.2], [2.0, 0.8], [5.0, 0.4]],
      "curve": "linear"
    }
  ]
}
```

### Top-level fields

- `mode` (string): `block` or `sample`
- `block_ms` (number): block size in milliseconds when block mode is active
- `lanes` (array): one or more automation lanes

### Lane fields (baseline)

- `target` (string): automation target name (`wet`, `dry`, `rt60`, etc.)
- `points` (array): control points as `[time_seconds, value]`
- `curve` (string, optional): interpolation mode (for example `linear`)

### Lane fields (feature-vector lanes)

Feature-vector lanes are also allowed and are normalized by feature-lane parsing logic.
A lane still requires `target`, and includes a source descriptor such as feature source, optional weighting, and bounds/clamp information.

---

## 4) Contract Stability

- These schema ids are maintained for the current public-alpha patch line (`0.7.x`).
- If a breaking manifest/automation format change is introduced, increment the schema id and document migration notes in this file and `CHANGELOG.md`.
