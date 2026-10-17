# AGENTS.md

This file contains context and guidelines for agents working on the `verbx` project.

## Project Description
`verbx` is a Python 3.11+ CLI tool for extreme audio reverberation, managed with Hatch.

## Core Features
- **DSP Architecture**:
    - Numba-optimized Feedback Delay Network (FDN) algorithmic reverb.
    - Uniform Partitioned Convolution (UPC) for efficient streaming.
- **Processing**:
    - 'Freeze' mode for infinite sustain using cross-faded loops.
    - Automatic reverb tail flushing.
- **Audio Analysis**:
    - RT60 estimation (T30) via energy decay.
    - Spectral contrast.

## Dependencies
- `typer`: CLI framework.
- `rich`: Terminal formatting.
- `numpy`, `scipy`: Numerical processing.
- `soundfile`: Audio I/O.
- `librosa`: Audio analysis.
- `numba`: JIT compilation for performance.

## Constraints & Guidelines
- **No GUI**: The project focuses solely on CLI functionality.
- **Typing**: Code requires strong typing (`pyright` compatible).
- **Linting/Formatting**: Uses `ruff`.
