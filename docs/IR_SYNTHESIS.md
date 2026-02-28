# IR Synthesis (v0.3)

`verbx` v0.3 adds deterministic impulse-response synthesis with four modes:

- `fdn`: Algorithmic FDN-derived IR
- `stochastic`: Noise-shaped diffuse tail
- `modal`: Decaying modal resonator bank
- `hybrid`: Early reflections + blended stochastic/modal/fdn tail

## Quick Start

```bash
# Generate a 120s hybrid IR
hatch run verbx ir gen irs/wash_120.wav --mode hybrid --length 120 --seed 42 --rt60 90

# Override output container format via switch
hatch run verbx ir gen irs/wash_120 --mode hybrid --length 120 --format flac

# Force musical anchor (fundamental)
hatch run verbx ir gen irs/wash_120.wav --mode modal --length 120 --f0 "64 Hz"

# Auto-tune IR to an input file's detected fundamentals/harmonics
hatch run verbx ir gen irs/tuned_from_input.wav --mode hybrid --analyze-input source.wav

# Analyze generated IR
hatch run verbx ir analyze irs/wash_120.wav

# Use generated IR directly in render
hatch run verbx render input.wav output.wav --engine conv --ir irs/wash_120.wav

# Auto-generate and cache IR during render
hatch run verbx render input.wav output.wav --ir-gen --ir-gen-mode hybrid --ir-gen-length 120 --ir-gen-seed 7
```

## Recipes

### Wash 120s

```bash
hatch run verbx ir gen irs/wash_120.wav \
  --mode hybrid --length 120 --rt60 95 --damping 0.45 \
  --diffusion 0.7 --density 1.2 --er-count 32 --er-max-delay-ms 120 \
  --normalize peak --peak-dbfs -1
```

### Cinematic Hybrid

```bash
hatch run verbx ir gen irs/cinematic_hybrid.wav \
  --mode hybrid --length 75 --rt60 60 --tilt 1.5 --lowcut 80 --highcut 12000 \
  --er-room 1.3 --er-stereo-width 1.2 --mod-depth-ms 2.0 --mod-rate-hz 0.1
```

### Pitched Modal

```bash
hatch run verbx ir gen irs/pitched_modal.wav \
  --mode modal --length 45 --rt60 35 --seed 99 --tuning A4=432 \
  --modal-count 64 --modal-q-min 7 --modal-q-max 90 \
  --modal-low-hz 60 --modal-high-hz 9000 --modal-spread-cents 8
```

## Cache

The IR cache stores reproducible artifacts by parameter hash:

- `.verbx_cache/irs/<hash>.wav`
- `.verbx_cache/irs/<hash>.meta.json`

Commands:

```bash
hatch run verbx cache info
hatch run verbx cache clear
```

## Batch

```bash
hatch run verbx batch template > manifest.json
hatch run verbx batch render manifest.json --jobs 4
```

## Tempo Sync

Render supports note-based pre-delay parsing:

```bash
hatch run verbx render in.wav out.wav --pre-delay 1/8D --bpm 120
```

Supported forms: `1/4`, `1/8D`, `1/8T`, or raw seconds (`0.125`).
