# verbx

`verbx` is a production-grade Python CLI for extreme reverb workflows.
It includes long-tail algorithmic and convolution engines, freeze/repeat chains,
loudness-aware output targeting, and synthetic IR generation with caching.

## Status

Current implementation level: **v0.3**

- Prompt 1: scaffolding and architecture
- Prompt 2: functional DSP render path
- Prompt 3: loudness/peak + shimmer/ambient controls
- Prompt 4: IR factory, cache, batch, tempo sync, framewise analysis

## Can I Use `verbx` Without Hatch?

Yes.

Hatch is convenient, but optional. You can use `verbx` with plain `pip`, a virtualenv,
`pipx`, or directly via `python -m verbx.cli`.

## Features

- CLI-only architecture (Typer + Rich)
- Algorithmic reverb (FDN + diffusion topology)
- Partitioned FFT convolution (long IR friendly)
- Freeze segment looping + repeat chaining
- Loudness/peak controls (LUFS, sample peak, true-peak approximation)
- Ambient controls (shimmer, ducking, bloom, tilt EQ)
- Synthetic IR generation (`fdn`, `stochastic`, `modal`, `hybrid`)
- Deterministic IR cache with metadata sidecars
- Batch rendering manifests
- Tempo-synced note parsing (`--pre-delay 1/8D --bpm 120`)
- Framewise CSV analysis exports

## Requirements

- Python 3.11+
- `libsndfile` available on system (required by `soundfile`)

## Installation and Quick Start

### Option A: Hatch (recommended for contributors)

```bash
hatch env create
hatch run verbx --help
```

### Option B: Plain virtualenv + pip (no Hatch)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
verbx --help
```

### Option C: pipx (isolated app install)

```bash
pipx install .
verbx --help
```

### Option D: Run module directly (no console-script install)

```bash
python -m pip install typer rich numpy scipy soundfile librosa pyloudnorm
PYTHONPATH=src python -m verbx.cli --help
```

## Choosing How To Run `verbx`

### Hatch

Pros:

- Best match for this repository's contributor workflow (`hatch run lint`, `typecheck`, `test`)
- Reproducible project environment defined in `pyproject.toml`
- No need to manually remember command variants for QA checks

Cons:

- Requires installing Hatch
- Dependency resolution/install is usually slower than `uv`

Best for:

- Contributors working on `verbx` itself
- CI/local parity with documented project scripts

### `uv`

Pros:

- Very fast environment creation and dependency install
- Supports project run flows (`uv run ...`) and pip-compatible flows (`uv pip ...`)
- Good for users who already standardize on `uv` across projects

Cons:

- Repo scripts are authored under Hatch env scripts, so command names differ
- Team docs and CI in this repo are Hatch-first

Best for:

- Power users prioritizing speed
- Local dev where you prefer `uv` tooling conventions

Example (`uv`-native):

```bash
uv sync --extra dev
uv run verbx --help
```

Example (pip-compatible with `uv`):

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
verbx --help
```

### Plain `venv` + `pip`

Pros:

- Universal, standard Python workflow
- No extra package manager required

Cons:

- More manual steps
- Slower installs than `uv`

Best for:

- Environments where only standard Python tooling is allowed

### `pipx`

Pros:

- Isolated app install without polluting global Python packages
- Simple for command-line usage only

Cons:

- Less convenient for editing/testing local source changes
- Not ideal for contributor QA loops

Best for:

- End users who only want to run `verbx` commands

### Direct `python -m verbx.cli`

Pros:

- Quickest way to run source without creating a console script entry point
- Useful for debugging local module execution

Cons:

- Requires setting `PYTHONPATH=src` (or equivalent path setup)
- Easier to drift from normal installed usage
- Not ideal as a primary workflow

Best for:

- Fast local checks and debugging
- Situations where you intentionally avoid install steps

Recommendation:

- Use Hatch for contributing to this repo.
- Use `uv` if you want the same result with faster dependency operations.
- Use `venv` + `pip` for maximal portability.
- Use `python -m verbx.cli` for quick local debugging only.

## Quick Start Recipes

### 1) First render (algorithmic)

```bash
verbx render input.wav output.wav --engine algo --rt60 80 --wet 0.85 --dry 0.15
```

### 2) Convolution render with external IR

```bash
verbx render input.wav output.wav --engine conv --ir hall_ir.wav --partition-size 16384
```

### 3) Freeze + repeat chain

```bash
verbx render input.wav output.wav --freeze --start 2.0 --end 4.0 --repeat 3
```

### 4) Loudness and peak-targeted render

```bash
verbx render input.wav output.wav \
  --target-lufs -18 \
  --target-peak-dbfs -1 \
  --true-peak \
  --normalize-stage post
```

### 5) Shimmer + ambient controls

```bash
verbx render input.wav output.wav \
  --shimmer --shimmer-semitones 12 --shimmer-mix 0.35 \
  --duck --duck-attack 15 --duck-release 250 \
  --bloom 2.0 --tilt 1.5
```

### 6) Tempo-synced pre-delay

```bash
verbx render input.wav output.wav --pre-delay 1/8D --bpm 120
```

### 7) Framewise analysis CSV during render

```bash
verbx render input.wav output.wav --frames-out reports/output_frames.csv
```

### 8) Auto-generate cached IR during render

```bash
verbx render input.wav output.wav \
  --ir-gen --ir-gen-mode hybrid --ir-gen-length 120 --ir-gen-seed 7
```

## CLI Command Cookbook

### Global help

```bash
verbx --help
```

### Core commands

```bash
verbx render INFILE OUTFILE [options]
verbx analyze INFILE [--lufs] [--json-out report.json] [--frames-out frames.csv]
verbx suggest INFILE
verbx presets
```

### `render` examples

```bash
# high-density algorithmic tail
verbx render in.wav out.wav --engine algo --rt60 120 --damping 0.5 --width 1.2

# convolution with IR normalization and tail cap
verbx render in.wav out.wav --engine conv --ir plate.wav --ir-normalize peak --tail-limit 45

# per-pass normalization strategy
verbx render in.wav out.wav --repeat 4 --normalize-stage per-pass --repeat-target-lufs -20

# output sample-peak strategy
verbx render in.wav out.wav --target-peak-dbfs -2 --sample-peak

# disable limiter
verbx render in.wav out.wav --no-limiter
```

### `analyze` examples

```bash
verbx analyze in.wav
verbx analyze in.wav --lufs
verbx analyze in.wav --json-out reports/in_analysis.json
verbx analyze in.wav --frames-out reports/in_frames.csv
```

### `ir` command group

```bash
verbx ir gen OUT_IR.wav [options]
verbx ir analyze IR.wav
verbx ir process IN_IR.wav OUT_IR.wav [options]
verbx ir fit INPUT.wav OUT_IR.wav --top-k 5
```

#### Generate IR examples

```bash
# hybrid, long tail
verbx ir gen IRs/hybrid_120.wav --mode hybrid --length 120 --seed 42

# format switch (overrides extension)
verbx ir gen IRs/custom_ir --mode hybrid --length 120 --format flac

# explicit harmonic anchor
verbx ir gen IRs/modal_64hz.wav --mode modal --f0 "64 Hz"

# auto-tune using source audio fundamentals/harmonics
verbx ir gen IRs/tuned_from_source.wav --mode hybrid --analyze-input source.wav

# extended control example
verbx ir gen IRs/cinematic.wav \
  --mode hybrid --length 180 --rt60 95 --damping 0.45 \
  --er-count 32 --er-max-delay-ms 120 --er-room 1.3 \
  --diffusion 0.7 --density 1.2 --tilt 1.5 --lowcut 80 --highcut 12000
```

#### Analyze/process/fit IR examples

```bash
verbx ir analyze IRs/hybrid_120.wav --json-out reports/hybrid_120_analysis.json

verbx ir process IRs/hybrid_120.wav IRs/hybrid_120_dark.wav \
  --lowcut 120 --highcut 7000 --tilt -1.0 --normalize peak

verbx ir fit input.wav IRs/fitted.wav --top-k 3
```

### Cache command group

```bash
verbx cache info
verbx cache clear
```

### Batch command group

```bash
verbx batch template > manifest.json
verbx batch render manifest.json --jobs 4
verbx batch render manifest.json --jobs 4 --dry-run
```

## Pregenerated IRs and Audio Examples

### Pregenerated long IRs (60s–360s)

- Folder: [IRs](IRs/)
- Details: [IRs/README.md](IRs/README.md)

Included examples:

- `ir_hybrid_60s.flac`
- `ir_fdn_90s.flac`
- `ir_stochastic_120s.flac`
- `ir_modal_180s.flac`
- `ir_hybrid_240s.flac`
- `ir_modal_360s.flac`

Each IR includes a sidecar metadata file:

- `<name>.ir.meta.json`

### Short audio demos

- [Dry click](examples/audio/dry_click.wav)
- [Hybrid IR (short)](examples/audio/hybrid_ir_short.wav)
- [Dry click reverbed](examples/audio/dry_click_reverbed.wav)

## Generate 25 IRs With Varying Parameters

### Python script

```bash
./scripts/generate_ir_bank.py \
  --out IRs/generated_25 \
  --count 25 \
  --sr 12000 \
  --channels 2 \
  --format flac
```

### Bash script (CLI-driven)

```bash
./scripts/generate_ir_bank.sh IRs/generated_25_cli 25 flac
```

Scripts:

- [scripts/generate_ir_bank.py](scripts/generate_ir_bank.py)
- [scripts/generate_ir_bank.sh](scripts/generate_ir_bank.sh)

## Development

### Lint / type-check / tests

With Hatch:

```bash
hatch run lint
hatch run typecheck
hatch run test
```

Without Hatch:

```bash
ruff check .
pyright
pytest
```

## Project Layout

- `src/verbx/cli.py`: command routing and UX
- `src/verbx/core/`: DSP engines, pipeline, loudness, shimmer, ambient, tempo
- `src/verbx/analysis/`: global + framewise feature extraction
- `src/verbx/io/`: audio I/O and progress utilities
- `src/verbx/ir/`: synthetic IR generation modes, shaping, metrics, tuning
- `tests/`: automated tests
- `docs/`: deeper guides

## Additional Docs

- [IR synthesis guide](docs/IR_SYNTHESIS.md)

## Roadmap

- v0.4: framewise modulation analysis, advanced IR fitting heuristics, parallel batch scheduler
