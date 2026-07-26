# verbx v0.9.2 and Next 4 Weeks

_Execution checklist derived from `README.md`, `CHANGELOG.md`, and
`docs/ROADMAP.md`._

_Last updated: 2026-07-26._

---

## Goal

Ship `v0.9.2` as a focused hardening release, then use its enforced quality
gates as the baseline for the next native and physical-acoustics increments.
New feature breadth remains secondary to compatibility, repeatability, and
measured validation.

## v0.9.2 Release Slice

### Python Core

- [x] Add immutable typed section views for engine, execution, tail, and output
  settings without breaking the flat `RenderConfig` constructor or reports.
- [x] Extract the shared numeric/choice parser cluster from `src/verbx/cli.py`.
- [x] Enforce streaming/in-memory parity for the convolution path.
- [x] Split nonlinearity and spatial-coupling responsibilities out of
  `algo_reverb.py`.
- [ ] Split the remaining delay-kernel responsibility out of `algo_reverb.py`.

### Quality Gates

- [x] Fail CI when a render benchmark exceeds its checked-in budget.
- [x] Fail CI when a benchmark scenario lacks a baseline.
- [x] Preserve benchmark reports as CI artifacts even on failure.
- [x] Build `verbx-c` and enforce the checked-in structural parity contract in
  a dedicated CI job.

### Native and Plug-in Readiness

- [x] Preserve source duration when native tail completion truncates an active
  or dry-only render.
- [x] Preserve a deterministic source-duration floor for silent native input.
- [x] Replace the foundational native `fdn` comb bank with a bounded
  allocation-free eight-line Hadamard FDN slice.
- [ ] Extend that slice toward Python-reference modulation, multiband, and
  automation parity.
- [ ] Implement bounded-lookahead reverse processing.
- [ ] Complete signed AUv3 and representative DAW-host validation.
- [ ] Add multichannel layouts and host compatibility certification.

### Physical-Acoustics Validation

- [x] Expose deterministic image-source paths before tap aggregation.
- [x] Check in an analytic rectangular-room reference corpus.
- [x] Validate direct and first-order distance, delay, and per-material gain.
- [ ] Add measured anechoic/convolution references with documented capture
  provenance.
- [x] Retain DXF tracing as experimental until measured-reference validation
  and per-layer material assignment are complete.

## Week 1: Release Closure

- [ ] Run the full Python test, lint, and type-check suites.
- [x] Run native C tests and the strict structural parity job locally.
- [x] Confirm generated CLI reference and launch examples match the shipped
  interface.
- [x] Review release notes for research-grade and experimental claims.
- [ ] Tag and publish `v0.9.2` only after all required checks pass.

## Week 2: Architecture

- [x] Move one cohesive helper cluster out of `cli.py`.
- [x] Add convolution streaming/in-memory parity coverage.
- [x] Define the native higher-order FDN state and coefficient contract in
  `docs/NATIVE_FDN_CONTRACT.md`.
- [x] Measure and record current Python/native DSP deltas before changing the
  native loop.

## Week 3: Native DSP and Hosts

- [x] Port one bounded FDN slice with allocation-free sample processing.
- [ ] Add impulse, reset, automation, and sanitizer coverage for that slice.
- [ ] Validate AU/VST3 loading and audible wet-tail output in supported hosts.
- [ ] Record host, OS, architecture, sample rate, block size, and result.

## Week 4: Evidence Review

- [ ] Compare benchmark trends against the `v0.9.1` baseline.
- [ ] Review ISM timing against the first measured reference.
- [x] Retain DXF tracing as experimental pending measured evidence.
- [ ] Select the next release from evidence: native FDN parity, physical-room
  validation, or further stabilization.

## Exit Criteria

- `v0.9.2` metadata and release notes agree.
- Python tests, lint, and type checking pass.
- Performance and native structural parity are blocking CI checks.
- Native render never shortens output below its deterministic source floor.
- ISM timing has a checked-in analytic reference and a clear measured-reference
  follow-up.
- No new production claim is made for experimental DXF, SDN, neural, or modal
  workflows without corresponding validation evidence.
