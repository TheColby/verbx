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
The canonical feature/gap matrix lives in
[`docs/NATIVE_PARITY.md`](../../docs/NATIVE_PARITY.md).

## Build

With a plain C compiler:

```bash
./scripts/build_verbx_c.sh
./build/native/verbx_c/verbx-c version
```

Useful build-script options:

```bash
./scripts/build_verbx_c.sh --clean --doctor
./scripts/build_verbx_c.sh --print-path
CC=clang CFLAGS="-O2" ./scripts/build_verbx_c.sh
```

Install the native binary and man page into a local prefix:

```bash
scripts/install_verbx_c.sh --prefix "$HOME/.local" --doctor
```

With CMake:

```bash
cmake -S native/verbx_c -B build/native/verbx_c
cmake --build build/native/verbx_c
./build/native/verbx_c/verbx-c doctor
```

For machine-readable diagnostics:

```bash
./build/native/verbx_c/verbx-c doctor --json-out native-doctor.json
```

After installing with `scripts/install_verbx_c.sh`, the native man page is
available as `man verbx-c` when the install prefix's man path is visible.

## What Works Today

The current native binary is intentionally narrow, but it is no longer a stub.

- commands: `help`, `version`, `doctor`, `render`
- doctor reports: human-readable stdout plus optional
  `native-doctor-report-v1` JSON via `doctor --json-out`
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
- plug-in foundation parameters: RT60 coarse/fine mapping supports `0.01s` to
  `360s`, with Freeze and Reverse represented as explicit mode parameters
- tail handling: threshold/hold based trim with a short click-safe fade into
  exact zeros
- peak-safe output: optional render-level scaling with `--peak-safe` and
  `--peak-ceiling-db`
- model selection: `--model fdn|spring|plate`; spring and plate use separate
  deterministic delay/diffusion tunings within the native offline core

Example:

```bash
./build/native/verbx_c/verbx-c render in.wav out.wav \
  --model plate \
  --rt60 4.0 \
  --wet 0.85 \
  --dry 0.15 \
  --pre-delay-ms 25 \
  --damping 0.5 \
  --tail-threshold-db -100 \
  --tail-hold-ms 10 \
  --peak-safe \
  --peak-ceiling-db -1 \
  --out-format float32 \
  --json-out native-report.json
```

The current DSP is a foundational Schroeder/Moorer-style offline reverb core,
not yet the full Python FDN engine.

## Immediate goals

- stabilize the native error model, logging model, and offline process contract
- broaden native render coverage against
  `tests/fixtures/native_render_parity_contract.json`
- keep `docs/NATIVE_PARITY.md` updated whenever native scope changes
- port the higher-order FDN/automation render core in small, testable pieces
- keep regression parity with the `v0.7.x` Python renderer during the migration

## Parity Contract

The first native parity target is intentionally narrow: deterministic offline
render only, mono/stereo WAV IO, `rt60`, `wet`, `dry`, `pre-delay`, damping,
tail threshold/hold/metric, peak-safe output, and output subtype selection. The checked-in
contract at `tests/fixtures/native_render_parity_contract.json` defines the
fixture names, accepted formats, deferred features, and metric tolerances that
the `v0.8` native track should satisfy before broadening the DSP surface.

`verbx-c render --json-out report.json` writes a `native-render-report-v1`
payload with the render contract, frame counts, output format, tail settings,
and input/output peak-safety metrics.

Run the Python/native comparison harness with:

```bash
uv run python scripts/compare_native_render_parity.py \
  --build-native \
  --report build/native_render_parity_report.json
```

The script can build `verbx-c`, renders every contract
fixture through both the Python reference and native candidate, and emits a
machine-readable metric report. Add `--strict` when using it as a failing gate.
