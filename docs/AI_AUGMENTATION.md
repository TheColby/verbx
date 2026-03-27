# AI Research and Data Augmentation

`verbx` includes a deterministic batch augmentation workflow for ML datasets.

## What this solves

- Generate many reverberant variants per clean source with fixed random seeds.
- Keep train/val/test split and label metadata attached to every rendered file.
- Export machine-friendly run artifacts (`JSONL` manifest + summary JSON).
- Emit split-level QA bundle artifacts (quality summaries + class-balance tables).
- Export optional dataset-card Markdown and per-output metrics CSV artifacts.
- Enforce split-isolation by default to reduce accidental train/val/test leakage.
- Optionally copy dry sources into the output tree for paired clean/wet training.

## Commands

### 1) Emit a manifest template

```bash
verbx batch augment-template > augment_manifest.json
```

### 1b) Inspect built-in profile families

```bash
verbx batch augment-profiles
verbx batch augment-profiles --json
```

### 2) Run augmentation

```bash
verbx batch augment augment_manifest.json \
  --jobs 8 \
  --copy-dry \
  --dataset-card-out out/DATASET_CARD.md \
  --metrics-csv-out out/augmentation_metrics.csv \
  --qa-bundle-out out/augmentation_qa_bundle.json \
  --provenance-hash \
  --summary-out out/augmentation_summary.json \
  --jsonl-out out/augmentation_manifest.jsonl
```

### 3) Dry-run planning only

```bash
verbx batch augment augment_manifest.json --dry-run
```

### 4) Direct corpus generation from file/folder

Use this when you want fast audio corpus expansion without authoring a manifest first.
Each variant applies randomized time shift, pitch shift, and algorithmic reverb inside
user-provided limits.

```bash
verbx batch corpus-generate data/clean \
  --output-root out/corpus \
  --variants-per-input 128 \
  --time-shift-min-ms -60 --time-shift-max-ms 60 \
  --pitch-shift-min-semitones -2 --pitch-shift-max-semitones 2 \
  --reverb-rt60-min 0.3 --reverb-rt60-max 3.8 \
  --reverb-wet-min 0.1 --reverb-wet-max 0.5 \
  --seed 20260322
```

Tip: `--variants-per-input` accepts very large values for long-running corpus generation.
Use `--dry-run` first to validate total output counts before launching a large job.

## Manifest format

```json
{
  "version": "0.7.1",
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

## Built-in profiles

- `asr-reverb-v1`: speech robustness (booth/room/hall-like ranges).
- `music-reverb-v1`: music training (plate/chamber/long-hall ranges).
- `drums-room-v1`: transient-preserving drum-room ranges.

## Output artifacts

- Rendered audio:
  - `<output_root>/<split>/<source>__<archetype>__aNNN.wav`
- Optional copied dry files:
  - `<output_root>/<split>/<source>__dry.wav`
- Metadata JSONL:
  - one row per rendered variant with `source_id`, `split`, `label`, `tags`,
    sampled `render_config`, deterministic `seed`, and success/error status
- Summary JSON:
  - aggregate counts (`planned/success/failed`) plus split/label/archetype/tag counts
- QA bundle JSON:
  - split-level metric summaries and split/global class-balance tables
  - when `--baseline-summary` is provided, includes regeneration deltas
- Optional dataset card:
  - markdown file documenting generation settings and distribution summary
- Optional metrics CSV:
  - per-output scalar features for QA, filtering, and dataset debugging
- Optional provenance hash:
  - deterministic manifest+input signature for external dataset registries

## Reproducibility notes

- Identical manifest + seed + inputs => identical augmentation plans.
- Use `--profile`, `--seed`, or `--variants-per-input` to override manifest values
  without editing files.
- Default split guard enforces one source ID/input path per split. Use
  `--allow-split-overlap` only when intentional cross-split reuse is required.
- Use `--baseline-summary <previous_summary.json>` to compare class-balance drift
  between regeneration runs.
