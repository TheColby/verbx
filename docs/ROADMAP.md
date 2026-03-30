# verbx Roadmap

_Last updated: 2026-03-30. Maintained with `README.md` and `CHANGELOG.md`._

---

## 1. Release Posture

**Current release:** `v0.7.6`
**Status:** public alpha (research-grade)
**Versioning policy:** semantic (`0.7.x` patch line during alpha)

verbx currently ships dual-engine reverb, deterministic automation/feature
control, immersive QC/handoff, reproducibility tooling, f64 internal DSP, and
experimental dereverberation workflows.

---

## 2. v0.7.6 Current Patch Line

Current patch-line status:

- [x] Runtime/package metadata aligned to `v0.7.6`.
- [x] Tail completion, proxy streaming, dereverb QA, release-health tooling, and IR library work shipped in `v0.7.6`.
- [x] Land the next focused `0.7.x` patch feature set and promote it from `Unreleased` into `CHANGELOG.md`.
- [x] Room size estimator integrated into analysis engine (`verbx analyze --room`, `verbx compare --room`, `AudioAnalyzer.analyze(include_room=True)`).

## 3. v0.7.5 Feature Pack (Completed)

Requested feature set 1-10 is implemented and tested:

- [x] 1. Tail completion controls (`--tail-stop-threshold-db`, `--tail-stop-hold-ms`, `--tail-stop-metric`)
- [x] 2. Algorithmic long-render proxy streaming path (`--algo-stream`)
- [x] 3. Large-output container controls (`--output-container auto|wav|w64|rf64`)
- [x] 4. Matrix morphing between FDN families (`--fdn-matrix-morph-to`, `--fdn-matrix-morph-seconds`)
- [x] 5. Per-band control lanes for RT60 and crossovers (`fdn-rt60-*`, `fdn-xover-*-hz`)
- [x] 6. Geometry-based early reflections (`--er-geometry` and room/source/listener controls)
- [x] 7. Dedicated dereverberation command (`verbx dereverb`)
- [x] 8. Auto-fit profile heuristics (`--auto-fit speech|music|drums|ambient`)
- [x] 9. Multichannel shimmer spatial decorrelation (`--shimmer-spatial` + spread/delay controls)
- [x] 10. Optional CUDA acceleration for algorithmic proxy path (`--algo-gpu-proxy --device cuda`)

---

## 4. v0.8 Native Executable Program

`v0.8` is the native C executable line. The Python implementation remains the
released/public-alpha tool during the transition.

### 4.1 Foundation

- [x] Land native source tree and build entrypoint (`native/verbx_c/`, `scripts/build_verbx_c.sh`).
- [x] Establish standalone executable identity (`verbx-c`) and minimal CLI surface.
- [ ] Define native error model, logging model, and deterministic offline process contract.

### 4.2 Audio Runtime

- [x] Implement mono/stereo WAV read in C with float64 decode for PCM16/24/32 and float32/float64 inputs.
- [x] Implement native WAV write for `pcm16`, `float32`, and `float64`.
- [x] Port analysis-free offline render lifecycle: read -> process -> tail finalize -> write.
- [ ] Mirror current Python tail-stop semantics and sample-rate policy deterministically.

### 4.3 DSP Port

- [x] Port a first native offline late-field core (pre-delay, combs, allpass diffusion, tail finalization).
- [ ] Replace the foundational Schroeder/Moorer core with the higher-order FDN loop used by `v0.7.x`.
- [ ] Port damping, width, pre-delay, freeze, repeat, and normalization in controlled phases.
- [ ] Define parity corpus against `v0.7.x` Python outputs before feature expansion.

### 4.4 Productization

- [ ] Decide whether `verbx-c` remains a transition binary or replaces `verbx` at release.
- [ ] Add native packaging/release flow (install script, Homebrew, man pages, CI).
- [ ] Document feature parity and feature gaps continuously during the migration.

---

## 5. Remaining 0.7.x Priorities

- [x] Expand `verbx dereverb` objective quality validation (PESQ/STOI/ASR WER-style benchmark harness).
- [x] Broaden algorithmic proxy-stream eligibility while preserving deterministic parity checks.
- [x] Add CI/hardware coverage for CUDA and Apple Silicon acceleration paths.
- [x] Tighten public alpha packaging/release health checks across PyPI and Homebrew channels.

---

## 6. Known Constraints (Alpha)

- Offline-first architecture; real-time plugin hosting is not in `0.7.x` scope.
- Very long tails remain compute-heavy; throughput depends on partition/block settings and hardware.
- CUDA acceleration currently benefits convolution-heavy paths most.
- Render-time sample-rate conversion is deterministic and offline-oriented.

---

## 7. Maintenance Rule

When a roadmap item is completed:

1. Update this roadmap immediately.
2. Update `CHANGELOG.md` in the same change.
3. Update `README.md` command/docs references in the same change.
