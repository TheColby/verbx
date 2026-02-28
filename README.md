# verbx

`verbx` is a production-grade Python CLI for extreme reverb processing workflows.
It supports long-tail algorithmic reverb, partitioned convolution, freeze processing,
repeat chaining, and rich input/output analysis.

## Status

This repository now includes a **functional v0.3 DSP implementation** with typed, modular architecture.

## Features

- Typer-based CLI with commands:
  - `render`
  - `analyze`
  - `presets`
- Typed engine abstraction for algorithmic and convolution reverbs
- FDN-based algorithmic extreme reverb with pre-diffusion and damping
- Partitioned FFT convolution for long impulse responses
- Freeze segment loop mode and repeat-pass safety conditioning
- Loudness/peak targeting: LUFS normalization, sample peak and true-peak ceiling
- Shimmer enhancement, ducking, bloom, and tilt EQ controls
- IR synthesis factory (`verbx ir`) with deterministic cache and metadata
- Batch rendering manifests and cache management commands
- Tempo-synced pre-delay note parsing (`--pre-delay 1/8D --bpm 120`)
- Expanded analysis module with time/spectral/stereo metrics
- Soundfile-based audio I/O, block iteration, normalization and limiting helpers
- Rich logging and real render progress stages (read/process/write/analyze)
- Ruff, Pyright, and Pytest integration
- GitHub Actions CI

## Requirements

- Python 3.11+
- [Hatch](https://hatch.pypa.io/)

## Installation

```bash
hatch env create
hatch run verbx --help
```

## Example Commands

```bash
# Show CLI help
hatch run verbx --help

# Render with algorithmic reverb
hatch run verbx render input.wav output.wav --engine algo --rt60 90 --wet 0.85 --dry 0.15

# Render with convolution reverb
hatch run verbx render input.wav output.wav --engine conv --ir hall_ir.wav --partition-size 16384

# Render with freeze and repeat chain
hatch run verbx render input.wav output.wav --freeze --start 2.0 --end 4.0 --repeat 3

# Render with LUFS/true-peak targets and ambient processing
hatch run verbx render input.wav output.wav --target-lufs -18 --target-peak-dbfs -1 --true-peak --shimmer --duck --bloom 2.0 --tilt 1.5

# Analyze file
hatch run verbx analyze input.wav

# Analyze with LUFS/true-peak/LRA metrics
hatch run verbx analyze input.wav --lufs

# Generate and analyze long IRs
hatch run verbx ir gen irs/hybrid_120.wav --mode hybrid --length 120 --seed 42
hatch run verbx ir analyze irs/hybrid_120.wav

# Auto-generate cached IR during render
hatch run verbx render input.wav output.wav --ir-gen --ir-gen-mode hybrid --ir-gen-length 120 --ir-gen-seed 7

# Batch workflow
hatch run verbx batch template > manifest.json
hatch run verbx batch render manifest.json --jobs 4

# Suggest settings
hatch run verbx suggest input.wav

# List presets
hatch run verbx presets
```

## Development

```bash
hatch run lint
hatch run typecheck
hatch run test
```

## Project Layout

- `src/verbx/cli.py`: Typer app and command wiring
- `src/verbx/core/`: Reverb engines and processing helpers
- `src/verbx/analysis/`: Time and spectral feature extraction
- `src/verbx/io/`: Audio I/O and progress reporting
- `src/verbx/ir/`: IR generation modes, shaping, metrics, and cache orchestration
- `tests/`: CLI and module-level tests

## IR Guide

- [IR synthesis recipes](docs/IR_SYNTHESIS.md)

## Audio Examples

- [Dry click](examples/audio/dry_click.wav)
- [Hybrid IR (short)](examples/audio/hybrid_ir_short.wav)
- [Dry click reverbed](examples/audio/dry_click_reverbed.wav)

## Pregenerated IRs

- [IRs folder](IRs/) with several long IRs in the 60s–360s range.
- [IRs README](IRs/README.md) describes included files and metadata sidecars.

### Generate 25 varied IRs

```bash
# Python generator
./scripts/generate_ir_bank.py --out IRs/generated_25 --count 25 --sr 12000 --channels 2 --format flac

# Bash generator (CLI-driven)
./scripts/generate_ir_bank.sh IRs/generated_25_cli 25 flac

# Direct IR generation with explicit output format
hatch run verbx ir gen IRs/my_ir --mode hybrid --length 120 --format wav

# Tune IR with explicit fundamental or input analysis
hatch run verbx ir gen IRs/my_tuned_ir.wav --mode modal --f0 "64 Hz"
hatch run verbx ir gen IRs/my_auto_tuned_ir.wav --mode hybrid --analyze-input source.wav
```

## Roadmap

- v0.4: framewise modulation analysis, advanced IR fitting heuristics, parallel batch scheduler
