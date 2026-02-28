# Pregenerated IRs

This folder contains pregenerated synthetic impulse responses from `verbx`.

## Included examples (60s to 360s)

- `ir_hybrid_60s.flac`
- `ir_fdn_90s.flac`
- `ir_stochastic_120s.flac`
- `ir_modal_180s.flac`
- `ir_hybrid_240s.flac`
- `ir_modal_360s.flac`

Each IR has a metadata sidecar:

- `<name>.ir.meta.json`

## Generate 25 varied IRs

Python:

```bash
./scripts/generate_ir_bank.py --out IRs/generated_25 --count 25 --sr 12000 --channels 2 --format flac
```

Bash (CLI-driven):

```bash
./scripts/generate_ir_bank.sh IRs/generated_25_cli 25 flac
```

Direct CLI format switch:

```bash
hatch run verbx ir gen IRs/custom_ir --mode hybrid --length 120 --format aiff
```

Tuning options:

```bash
# explicit musical anchor
hatch run verbx ir gen IRs/custom_tuned.wav --mode modal --f0 "64 Hz"

# infer fundamentals/harmonics from input audio
hatch run verbx ir gen IRs/custom_from_input.wav --mode hybrid --analyze-input dry_guitar.wav
```
