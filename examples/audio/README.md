# Audio Example Pack

This folder contains tiny demo assets for quick listening and verification.

## Included files

- `dry_click.wav`: one-shot dry click reference.
- `dry_click_reverbed.wav`: click rendered with reverb for sanity-check playback.
- `hybrid_ir_short.wav`: short IR asset used in quick convolution demos.

Realistic example set (all stereo, 24 kHz, PCM16):

- `realistic_speech_dry.wav`
- `realistic_speech_room.wav`
- `realistic_music_dry.wav`
- `realistic_music_hall.wav`
- `realistic_drums_dry.wav`
- `realistic_drums_room.wav`
- `realistic_examples.meta.json` (generation parameters and render snapshots)

Extreme example set — showcasing verbx's signature sound design range:

- `extreme_cathedral_drums.wav`: drums through 8s Hadamard FDN cathedral reverb (16-line, 45ms pre-delay)
- `extreme_shimmer_music.wav`: music through 6s algorithmic reverb with octave shimmer and 0.65 feedback
- `extreme_plate_speech.wav`: speech through circulant FDN plate simulation (1.8s RT60, bandlimited 200–6000 Hz)
- `extreme_frozen_music.wav`: music through 30s Hadamard FDN with near-infinite tail (32-line, 60ms pre-delay)

Commands used to generate these (from repo root):

```bash
verbx render examples/audio/realistic_drums_dry.wav examples/audio/extreme_cathedral_drums.wav \
  --engine algo --rt60 8.0 --wet 0.85 --dry 0.25 --pre-delay-ms 45 \
  --fdn-lines 16 --fdn-matrix hadamard --lowcut 80 --highcut 12000

verbx render examples/audio/realistic_music_dry.wav examples/audio/extreme_shimmer_music.wav \
  --engine algo --rt60 6.0 --wet 0.8 --dry 0.3 \
  --shimmer --shimmer-semitones 12 --shimmer-mix 0.35 --shimmer-feedback 0.65 \
  --pre-delay-ms 30 --fdn-lines 16

verbx render examples/audio/realistic_speech_dry.wav examples/audio/extreme_plate_speech.wav \
  --engine algo --rt60 1.8 --wet 0.7 --dry 0.4 \
  --fdn-matrix circulant --lowcut 200 --highcut 6000 --pre-delay-ms 12

verbx render examples/audio/realistic_music_dry.wav examples/audio/extreme_frozen_music.wav \
  --engine algo --rt60 30.0 --wet 0.95 --dry 0.1 \
  --fdn-lines 32 --pre-delay-ms 60 --fdn-matrix hadamard
```

## Regenerate realistic examples

From repo root:

```bash
uv run python scripts/generate_realistic_audio_examples.py
```

Optional:

```bash
uv run python scripts/generate_realistic_audio_examples.py --skip-renders
```
