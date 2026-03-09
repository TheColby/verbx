# IR Synthesis (v0.6.0 + v0.7 extensions)

`verbx` includes deterministic impulse-response synthesis with four modes:

- `fdn`: Algorithmic FDN-derived IR
- `stochastic`: Noise-shaped diffuse tail
- `modal`: Decaying modal resonator bank
- `hybrid`: Early reflections + blended stochastic/modal/fdn tail

Related v0.7 IR features:

- cache-backed IR morphing (`verbx ir morph`)
- render-time IR blend/morph (`verbx render --ir-blend ...`)
- morph quality metadata (RT drift and spectral-distance diagnostics)

## Roadmap Alignment (v0.7 Completion Program)

IR workflows map directly to `README.md` Stream R3 milestones:

- `R3.1 cache determinism`: canonical cache keys and metadata compatibility checks
  across sample-rate/channel-layout variants.
- `R3.2 operational QA`: deterministic morph/blend diagnostic artifacts for batch
  acceptance and regression checks.
- `R3.3 failure safety`: explicit mismatch policy plus retry/resume hardening for
  long morph batches.

When updating `ir morph` or render-time `--ir-blend` behavior, keep this guide,
the CLI switch tables in `README.md`, and `docs/REFERENCES.md` source links in
lock-step.

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

# Add Modalys-inspired late-tail resonator coloration
hatch run verbx ir gen irs/resonated_120.wav --mode hybrid --length 120 \
  --resonator --resonator-mix 0.4 --resonator-modes 28 \
  --resonator-low-hz 70 --resonator-high-hz 8000

# Analyze generated IR
hatch run verbx ir analyze irs/wash_120.wav

# Morph two IRs into a hybrid target (cache-backed)
hatch run verbx ir morph irs/hall_A.wav irs/hall_B.wav irs/hall_AB.wav \
  --mode envelope-aware --alpha 0.6 --early-ms 80 --align-decay

# Use generated IR directly in render
hatch run verbx render input.wav output.wav --engine conv --ir irs/wash_120.wav

# Auto-generate and cache IR during render
hatch run verbx render input.wav output.wav --ir-gen --ir-gen-mode hybrid --ir-gen-length 120 --ir-gen-seed 7

# Render-time IR blending (Track D)
hatch run verbx render input.wav output.wav --engine conv --ir irs/hall_A.wav \
  --ir-blend irs/hall_B.wav --ir-blend-mix 0.5 --ir-blend-mode spectral
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

### Resonator-Colored Hybrid (Modalys-inspired)

```bash
hatch run verbx ir gen irs/resonator_hybrid.wav \
  --mode hybrid --length 90 --rt60 70 --seed 33 \
  --resonator --resonator-mix 0.38 --resonator-modes 24 \
  --resonator-q-min 10 --resonator-q-max 120 \
  --resonator-low-hz 80 --resonator-high-hz 7000 \
  --resonator-late-start-ms 90 --f0 "64 Hz"
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
