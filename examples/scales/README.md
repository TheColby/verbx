# Scala tuning examples

These Scala `.scl` files are inputs for microtonal synthetic impulse-response
generation. The included `19edo.scl` divides an octave into 19 equal steps.

```bash
verbx ir gen /tmp/verbx-19edo.wav \
  --mode hybrid --length 8 --rt60 4.5 \
  --scala-file examples/scales/19edo.scl \
  --scala-root-hz 220 --scala-low-hz 100 --scala-high-hz 8000 \
  --scala-strength 0.7 --scala-bandwidth-cents 24 --scala-gain-db 5

verbx render examples/audio/realistic_music_dry.wav /tmp/verbx-19edo-demo.wav \
  --engine conv --ir /tmp/verbx-19edo.wav --wet 0.4 --dry 1
```

The IR metadata sidecar records the scale description, content hash, root
mapping, resolved target frequencies, and emphasis parameters. Edit the scale
or any tuning option to obtain a distinct deterministic cache entry.
