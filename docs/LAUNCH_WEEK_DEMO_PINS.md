# Launch Week Demo Pins (v0.7.6 example-pack refresh)

_Updated: 2026-04-02_

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
  --fdn-lines 16 --fdn-matrix hadamard --lowcut 80 --highcut 12000 \
  --target-sr 48000 --out-subtype pcm24 --output-peak-norm target --output-peak-target-dbfs -2

# 2) Shimmer music
verbx render examples/audio/realistic_music_dry.wav examples/audio/extreme_shimmer_music.wav \
  --engine algo --rt60 6.0 --wet 0.8 --dry 0.3 \
  --shimmer --shimmer-semitones 12 --shimmer-mix 0.35 --shimmer-feedback 0.65 \
  --pre-delay-ms 30 --fdn-lines 16 \
  --target-sr 48000 --out-subtype pcm24 --output-peak-norm target --output-peak-target-dbfs -2

# 3) Plate speech
verbx render examples/audio/realistic_speech_dry.wav examples/audio/extreme_plate_speech.wav \
  --engine algo --rt60 1.8 --wet 0.7 --dry 0.4 \
  --fdn-matrix circulant --lowcut 200 --highcut 6000 --pre-delay-ms 12 \
  --target-sr 48000 --out-subtype pcm24 --output-peak-norm target --output-peak-target-dbfs -2

# 4) Frozen music
verbx render examples/audio/realistic_music_dry.wav examples/audio/extreme_frozen_music.wav \
  --engine algo --rt60 30.0 --wet 0.95 --dry 0.1 \
  --fdn-lines 32 --pre-delay-ms 60 --fdn-matrix hadamard \
  --target-sr 48000 --out-subtype pcm24 --output-peak-norm target --output-peak-target-dbfs -2
```

## Expected Outputs

| Output file | Expected SHA256 | Format | Sample rate | Channels | Duration (s) |
|---|---|---|---:|---:|---:|
| `examples/audio/extreme_cathedral_drums.wav` | `6b39278ab6963080ac4fe8cf742472bfe4d339140f95b83ae20feaf1c2c79894` | WAV PCM24 | 48000 | 2 | 12.475 |
| `examples/audio/extreme_shimmer_music.wav` | `180cf52c52aecd0d3b0f13220c6d20761662e91a64eccad9f23613e5fbc59e73` | WAV PCM24 | 48000 | 2 | 13.674 |
| `examples/audio/extreme_plate_speech.wav` | `1b193f61dde112fdfe49e44e359936356701a802d47ad2a3100e6f0f0d8ade1b` | WAV PCM24 | 48000 | 2 | 7.321 |
| `examples/audio/extreme_frozen_music.wav` | `5c55b46b5935d4f1e03711548a8ecfef7cb6d01ae6cca3ba6e72e19ce66e7ddd` | WAV PCM24 | 48000 | 2 | 35.570 |

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
