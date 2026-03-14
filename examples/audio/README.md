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

## Regenerate realistic examples

From repo root:

```bash
uv run python scripts/generate_realistic_audio_examples.py
```

Optional:

```bash
uv run python scripts/generate_realistic_audio_examples.py --skip-renders
```
