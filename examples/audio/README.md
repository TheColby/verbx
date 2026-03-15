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

Experimental music tradition example set — eight demos from the avant-garde/experimental canon:

- `lucier_sitting_room.wav`: speech → 7-pass room resonance accumulation (Alvin Lucier)
- `eno_discreet_music.wav`: music → 12s ambient tail, low damping (Brian Eno)
- `oliveros_deep_listening.wav`: music → 18s cave-scale resonance, 32-line FDN (Pauline Oliveros)
- `fripp_frippertronics.wav`: music → octave shimmer tape-loop, feedback=0.78 (Fripp/Eno)
- `mbv_shoegaze.wav`: music → dense shimmer wash, circulant FDN (My Bloody Valentine)
- `reich_phase_drums.wav`: drums → tight 0.7s room, circulant diffusion (Steve Reich)
- `radigue_drone.wav`: music → 45s near-infinite sustain, wet=0.97 (Eliane Radigue)
- `feldman_sparse_room.wav`: music → 3.8s room, allpass diffusion, low wet (Morton Feldman)

Commands used to generate these (from repo root):

```bash
verbx render examples/audio/realistic_speech_dry.wav examples/audio/lucier_sitting_room.wav \
  --engine algo --rt60 4.5 --wet 1.0 --dry 0.0 \
  --fdn-lines 16 --fdn-matrix hadamard --repeat 7 --lowcut 60

verbx render examples/audio/realistic_music_dry.wav examples/audio/eno_discreet_music.wav \
  --engine algo --rt60 12.0 --wet 0.92 --dry 0.08 \
  --fdn-lines 16 --fdn-matrix hadamard --pre-delay-ms 35 --damping 0.25 --lowcut 50

verbx render examples/audio/realistic_music_dry.wav examples/audio/oliveros_deep_listening.wav \
  --engine algo --rt60 18.0 --wet 0.95 --dry 0.10 \
  --fdn-lines 32 --fdn-matrix hadamard --pre-delay-ms 55 --damping 0.15 --lowcut 30

verbx render examples/audio/realistic_music_dry.wav examples/audio/fripp_frippertronics.wav \
  --engine algo --rt60 8.0 --wet 0.82 --dry 0.28 \
  --fdn-lines 16 --fdn-matrix hadamard \
  --shimmer --shimmer-semitones 12 --shimmer-mix 0.45 --shimmer-feedback 0.78 \
  --pre-delay-ms 25

verbx render examples/audio/realistic_music_dry.wav examples/audio/mbv_shoegaze.wav \
  --engine algo --rt60 5.0 --wet 0.88 --dry 0.22 \
  --fdn-lines 16 --fdn-matrix circulant \
  --shimmer --shimmer-semitones 12 --shimmer-mix 0.55 --shimmer-feedback 0.72 \
  --pre-delay-ms 8 --lowcut 80

verbx render examples/audio/realistic_drums_dry.wav examples/audio/reich_phase_drums.wav \
  --engine algo --rt60 0.7 --wet 0.55 --dry 0.50 \
  --fdn-lines 8 --fdn-matrix circulant --pre-delay-ms 18 --damping 0.6 --lowcut 60

verbx render examples/audio/realistic_music_dry.wav examples/audio/radigue_drone.wav \
  --engine algo --rt60 45.0 --wet 0.97 --dry 0.05 \
  --fdn-lines 32 --fdn-matrix hadamard --damping 0.10 --lowcut 20

verbx render examples/audio/realistic_music_dry.wav examples/audio/feldman_sparse_room.wav \
  --engine algo --rt60 3.8 --wet 0.52 --dry 0.52 \
  --fdn-lines 8 --fdn-matrix circulant --pre-delay-ms 30 --damping 0.50 --allpass-stages 4
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
