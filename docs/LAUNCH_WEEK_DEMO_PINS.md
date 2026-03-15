# Launch Week Demo Pins (v0.7.0)

_Updated: 2026-03-15_

This document pins demo commands and expected output fingerprints for launch-week
reproducibility.

## Environment Assumptions

- Checkout at tag `v0.7.0`.
- Python 3.11.
- `libsndfile` available.
- Command execution from repository root.

## Pinned Demo Commands

```bash
# 1) Cathedral drums
verbx render examples/audio/realistic_drums_dry.wav examples/audio/extreme_cathedral_drums.wav \
  --engine algo --rt60 8.0 --wet 0.85 --dry 0.25 --pre-delay-ms 45 \
  --fdn-lines 16 --fdn-matrix hadamard --lowcut 80 --highcut 12000

# 2) Shimmer music
verbx render examples/audio/realistic_music_dry.wav examples/audio/extreme_shimmer_music.wav \
  --engine algo --rt60 6.0 --wet 0.8 --dry 0.3 \
  --shimmer --shimmer-semitones 12 --shimmer-mix 0.35 --shimmer-feedback 0.65 \
  --pre-delay-ms 30 --fdn-lines 16

# 3) Plate speech
verbx render examples/audio/realistic_speech_dry.wav examples/audio/extreme_plate_speech.wav \
  --engine algo --rt60 1.8 --wet 0.7 --dry 0.4 \
  --fdn-matrix circulant --lowcut 200 --highcut 6000 --pre-delay-ms 12

# 4) Frozen music
verbx render examples/audio/realistic_music_dry.wav examples/audio/extreme_frozen_music.wav \
  --engine algo --rt60 30.0 --wet 0.95 --dry 0.1 \
  --fdn-lines 32 --pre-delay-ms 60 --fdn-matrix hadamard
```

## Expected Outputs

| Output file | Expected SHA256 | Format | Sample rate | Channels | Duration (s) |
|---|---|---|---:|---:|---:|
| `examples/audio/extreme_cathedral_drums.wav` | `68399ee6fd477f3428d0c9a18c12c3f92197105d1cc597e76053940258e419fd` | WAV PCM16 | 24000 | 2 | 13.545 |
| `examples/audio/extreme_shimmer_music.wav` | `b12998cd442ee5df33f354c87cf89c22595515ad7beb54754bde7bf7d6ce1d89` | WAV PCM16 | 24000 | 2 | 17.162 |
| `examples/audio/extreme_plate_speech.wav` | `096880ecc33d3b4c856419423ba056109476d0ea6bb99fc020086fa6661df55b` | WAV PCM16 | 24000 | 2 | 7.322 |
| `examples/audio/extreme_frozen_music.wav` | `9fb5e26159dd666a4f4da28e3da1db29b8de213df61ea39978db3257e82d5081` | WAV PCM16 | 24000 | 2 | 35.570 |

## Quick Verification

```bash
shasum -a 256 \
  examples/audio/extreme_cathedral_drums.wav \
  examples/audio/extreme_shimmer_music.wav \
  examples/audio/extreme_plate_speech.wav \
  examples/audio/extreme_frozen_music.wav
```

If any hash differs, capture:

- command used
- `verbx version`
- `verbx doctor --json-out doctor.json`
- output from `verbx render ... --repro-bundle`
