# verbx v0.9.1 and Next 4 Weeks

_Execution checklist derived from `README.md`, `CHANGELOG.md`, and
`docs/ROADMAP.md`._

_Last updated: 2026-07-23._

---

## Goal

Ship `v0.9.1` as a focused hardening release, then use its enforced quality
gates as the baseline for the next native and physical-acoustics increments.
New feature breadth remains secondary to compatibility, repeatability, and
measured validation.

## v0.9.1 Release Slice

### Python Core

- [x] Add immutable typed section views for engine, execution, tail, and output
  settings without breaking the flat `RenderConfig` constructor or reports.
- [ ] Continue extracting implementation helpers from `src/verbx/cli.py`.
- [ ] Enforce streaming/in-memory parity for the convolution path.
- [ ] Split the remaining delay-kernel, nonlinearity, and spatial-coupling
  responsibilities out of `algo_reverb.py`.

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
- [ ] Replace the foundational native Schroeder/Moorer loop with the
  higher-order FDN used by the Python reference.
- [ ] Implement bounded-lookahead reverse processing.
- [ ] Complete signed AUv3 and representative DAW-host validation.
- [ ] Add multichannel layouts and host compatibility certification.

### Physical-Acoustics Validation

- [x] Expose deterministic image-source paths before tap aggregation.
- [x] Check in an analytic rectangular-room reference corpus.
- [x] Validate direct and first-order distance, delay, and per-material gain.
- [ ] Add measured anechoic/convolution references with documented capture
  provenance.
- [ ] Make an explicit graduate/retain-experimental decision for DXF tracing.

## Week 1: Release Closure

- [ ] Run the full Python test, lint, and type-check suites.
- [ ] Run native C tests and the strict structural parity job locally.
- [ ] Confirm generated CLI and user-guide outputs still match the shipped
  interface.
- [ ] Review release notes for research-grade and experimental claims.
- [ ] Tag and publish `v0.9.1` only after all required checks pass.

## Week 2: Architecture

- [ ] Move one cohesive helper cluster out of `cli.py`.
- [ ] Add convolution streaming/in-memory parity coverage.
- [ ] Define the native higher-order FDN state and coefficient contract.
- [ ] Measure current Python/native DSP deltas before changing the native loop.

## Week 3: Native DSP and Hosts

- [ ] Port one bounded FDN slice with allocation-free realtime behavior.
- [ ] Add impulse, reset, automation, and sanitizer coverage for that slice.
- [ ] Validate AU/VST3 loading and audible wet-tail output in supported hosts.
- [ ] Record host, OS, architecture, sample rate, block size, and result.

## Week 4: Evidence Review

- [ ] Compare benchmark trends against the `v0.9.1` baseline.
- [ ] Review ISM timing against the first measured reference.
- [ ] Decide whether DXF tracing advances, remains experimental, or pauses.
- [ ] Select the next release from evidence: native FDN parity, physical-room
  validation, or further stabilization.

## Exit Criteria

- `v0.9.1` metadata and release notes agree.
- Python tests, lint, and type checking pass.
- Performance and native structural parity are blocking CI checks.
- Native render never shortens output below its deterministic source floor.
- ISM timing has a checked-in analytic reference and a clear measured-reference
  follow-up.
- No new production claim is made for experimental DXF, SDN, neural, or modal
  workflows without corresponding validation evidence.
