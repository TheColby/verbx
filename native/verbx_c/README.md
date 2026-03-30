# verbx-c

`verbx-c` is the native C executable track for `verbx` `v0.8`.

This directory now contains the first functional native render path:

- native build configuration
- standalone `verbx-c` executable target
- core CLI command dispatch (`help`, `version`, `doctor`, `render`)
- mono/stereo WAV read/write
- first native offline algorithmic reverb path with float64 internal processing
- offline lifecycle shell: read -> render -> tail-finalize -> write

The Python application in `src/verbx/` remains the released implementation for
`v0.7.x`. The native track is where the executable rewrite starts.

## Build

With a plain C compiler:

```bash
./scripts/build_verbx_c.sh
./build/native/verbx_c/verbx-c version
```

With CMake:

```bash
cmake -S native/verbx_c -B build/native/verbx_c
cmake --build build/native/verbx_c
./build/native/verbx_c/verbx-c doctor
```

## What Works Today

The current native binary is intentionally narrow, but it is no longer a stub.

- commands: `help`, `version`, `doctor`, `render`
- channels: mono and stereo
- input WAV formats:
  - PCM16
  - PCM24
  - PCM32
  - float32
  - float64
- output WAV formats:
  - `pcm16`
  - `float32`
  - `float64`
- processing precision: internal `float64`
- render model: deterministic offline file render
- tail handling: threshold/hold based trim with a short click-safe fade into
  exact zeros

Example:

```bash
./build/native/verbx_c/verbx-c render in.wav out.wav \
  --rt60 4.0 \
  --wet 0.85 \
  --dry 0.15 \
  --pre-delay-ms 25 \
  --damping 0.5 \
  --tail-threshold-db -100 \
  --tail-hold-ms 10 \
  --out-format float32
```

The current DSP is a foundational Schroeder/Moorer-style offline reverb core,
not yet the full Python FDN engine.

## Immediate goals

- stabilize the native error model, logging model, and offline process contract
- broaden native render coverage and add deterministic parity fixtures
- port the higher-order FDN/automation render core in small, testable pieces
- keep regression parity with the `v0.7.x` Python renderer during the migration
