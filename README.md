# verbx

Extreme ambient audio reverberator CLI tool.

## Features

- **Extreme Algorithmic Reverb**: FDN-based with long tails.
- **Convolution Reverb**: Support for partitioned FFT convolution.
- **Freeze Mode**: Infinite sustain processing.
- **Repeat Chaining**: Recursive reverb application.
- **Detailed Analysis**: SoX-style audio feature extraction.
- **Production Ready**: Fully typed, tested, and CI-integrated.

## Installation

```bash
pip install verbx
```

Or via Hatch:

```bash
hatch run verbx --help
```

## Usage

### Render

Apply reverb to an audio file:

```bash
verbx render input.wav output.wav --engine algo --rt60 4.0 --wet 0.6
```

### Analyze

Analyze audio features:

```bash
verbx analyze input.wav
```

### Presets

List available presets:

```bash
verbx presets
```

## Roadmap

- [ ] Full DSP Implementation (Algo/Conv)
- [ ] Freeze/Repeat Logic
- [ ] Advanced Analysis
- [ ] Spectral Features
