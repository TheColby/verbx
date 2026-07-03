# Native Parity Matrix

This document tracks the `verbx-c` native executable during the `v0.8` hybrid
transition. Python `verbx` remains the default public-alpha CLI until native
parity is proven by tests and fixtures.

## Release Shape

Chosen `v0.8` shape: **hybrid wrapper phase before full replacement**.

- `verbx-c` ships as an opt-in native executable.
- Python `verbx` remains the default command.
- Native expansion must land behind deterministic tests and update this matrix.
- Broad replacement is deferred until the parity contract passes for the chosen
  feature slice.

## Current Status

| Area | Python `verbx` | Native `verbx-c` | Status | Verification |
| --- | --- | --- | --- | --- |
| CLI entrypoint | Full public CLI | `help`, `version`, `doctor`, `render` | Partial native slice | `uv run pytest tests/test_native_scaffold.py` |
| Offline algorithmic render | Full FDN feature surface | Foundational Schroeder/Moorer-style core | Not equivalent | `scripts/compare_native_render_parity.py` |
| WAV input | libsndfile-backed broad format support | Mono/stereo WAV PCM16/24/32 and float32/float64 | Narrow parity | `tests/test_native_scaffold.py` |
| WAV output | Broad libsndfile output support | `pcm16`, `float32`, `float64` WAV | Narrow parity | `tests/test_native_scaffold.py` |
| Render controls | Hundreds of CLI options | `rt60`, `wet`, `dry`, pre-delay, damping, tail controls, peak-safe output | Narrow parity | `tests/fixtures/native_render_parity_contract.json` |
| Tail handling | Python tail-stop semantics plus long-render safeguards | Threshold/hold tail trim with exact-zero ending | In progress | Native scaffold tests and parity report |
| Peak-safe output | Limiter/normalization/output peak controls | `--peak-safe --peak-ceiling-db` render-level scaling | Implemented for native slice | `test_native_render_peak_safe_scales_float_output_to_ceiling` |
| Machine-readable reports | Render, realtime, dereverb, doctor, compare reports | `native-render-report-v1`, `native-doctor-report-v1` | Implemented for native slice | Native scaffold tests |
| Build ergonomics | Python packaging/install flow | `scripts/build_verbx_c.sh` with `--clean`, `--doctor`, `--print-path` | Implemented | `test_native_build_script_exposes_ergonomic_flags` |
| Install ergonomics | Python install helper, man pages, Homebrew formula | `scripts/install_verbx_c.sh`, `verbx-c(1)` | Local install implemented | `test_native_install_script_installs_binary_and_man_page` |
| Realtime | Full Python realtime path | Not implemented | Deferred | Roadmap only |
| Convolution | Python convolution engine | Not implemented | Deferred | Roadmap only |
| Dereverb | Python DSP dereverb | Not implemented | Deferred | Roadmap only |
| IR tools | Python IR synth/morph/library tools | Not implemented | Deferred | Roadmap only |
| Batch/immersive workflows | Python workflow commands | Not implemented | Deferred | Roadmap only |
| Presets | Built-in/generated Python preset bank | Not implemented | Deferred | Roadmap only |

## Native Parity Contract

The checked-in contract is:

```text
tests/fixtures/native_render_parity_contract.json
```

Run the comparison harness:

```bash
uv run python scripts/compare_native_render_parity.py \
  --build-native \
  --report build/native_render_parity_report.json
```

The report is allowed to fail metric tolerances while the native DSP remains a
foundational core. The harness itself must continue to run and emit JSON so the
gap is measurable.

## Before Expanding Native Scope

For each new native feature slice:

1. Add the control or behavior to `tests/fixtures/native_render_parity_contract.json`.
2. Add or update native scaffold coverage in `tests/test_native_scaffold.py`.
3. Update `scripts/compare_native_render_parity.py` when the comparison payload changes.
4. Update this matrix with the new status and verification command.
5. Keep Python `verbx` as the default CLI unless the full replacement decision is revisited.

## Deferred Until After The First `v0.8` Slice

- Native realtime audio.
- Native convolution engine.
- Native dereverb.
- Native IR synthesis, morphing, and library workflows.
- Full Python FDN parity, including automation, shimmer, freeze, repeat, and
  generated preset coverage.
- Homebrew/tap integration for installing `verbx-c` directly.
- CI release packaging checks for native artifacts.
