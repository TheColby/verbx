# verbx

`verbx` is a production-grade Python CLI for extreme reverb processing workflows.
It supports long-tail algorithmic reverb, partitioned convolution, freeze processing,
repeat chaining, and rich input/output analysis.

## Status

This repository now includes a **functional v0.2 DSP implementation** with typed, modular architecture.

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
- `tests/`: CLI and module-level tests

## Roadmap

- v0.3: IR synthesis factory, caching, batch workflows, tempo sync
