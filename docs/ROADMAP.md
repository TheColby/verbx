# verbx Roadmap

_Last updated: 2026-03-27. Maintained with `README.md` and `CHANGELOG.md`._

---

## 1. Release Posture

**Current release:** `v0.7.3`
**Status:** public alpha
**Versioning policy:** semantic (`0.7.x` patch line during alpha)

verbx is now feature-complete enough for public alpha distribution: dual-engine
rendering, deterministic automation/feature control, immersive QC/handoff,
reproducibility tooling, and f64 internal DSP.

---

## 2. Public Alpha Launch Checklist (v0.7.0)

- [x] Tag and publish `v0.7.0` release artifacts.
- [x] Verify install flows on macOS/Linux (`uv`, `pip+venv`, `install.sh`).
- [x] Confirm man pages install and render correctly (`man verbx`, `man verbx-render`).
- [x] Run release smoke matrix:
  - [x] Algorithmic render smoke test
  - [x] Convolution render smoke test
  - [x] IR generation + IR morph smoke test
  - [x] Batch augment smoke test
  - [x] Immersive handoff + QC smoke test
- [x] Publish public-alpha notes with known limitations and support channels.
- [x] Pin demo commands and expected outputs for launch week reproducibility.
- [x] Add Homebrew distribution path (formula + release-sync automation).

---

## 3. Musical Demo Set (Public Alpha)

These artist-inspired workflows are part of launch-critical documentation and
must remain in `README.md` recipe sections:

- Alvin Lucier / **I Am Sitting in a Room** (iterative room resonance accumulation)
- Brian Eno / **Discreet Music** (long-tail ambient loopbed)
- Pauline Oliveros / **Deep Listening** (extended drone-space rendering)
- Frippertronics-style tape-loop accumulation
- Shoegaze reverse-wash (freeze + shimmer)

Launch deliverables should include short before/after clips for each workflow.

---

## 4. v0.7.x Tracks (Post-Launch Alpha Hardening)

### 4.1 Stability and Diagnostics

- [x] Expand regression fixtures for long-tail renders (>120 s RT60).
- [x] Add deterministic golden tests for feature-vector lane behavior.
- [x] Add performance baseline report (`docs/benchmarks/`) for CI comparison.
- [x] Improve failure messaging for multichannel routing misconfiguration.

### 4.2 API and Integration

- [x] Ship a stable `verbx.api` Python surface (`render_file`, `generate_ir`, `analyze_file`).
- [x] Add minimal notebook examples for research and dataset workflows.
- [x] Add structured JSON schema docs for manifests and automation files.

### 4.3 Spatial and Interop

- [x] Harden layout/route validation for large immersive buses.
- [x] Add explicit examples for `7.2.4`, `8.0`, `16.0`, and `64.4` routing.
- [x] Evaluate SOFA import path feasibility for a future `0.7.x` patch.

### 4.4 Documentation Quality

- [x] Keep CLI tables in `README.md` synchronized with actual CLI help text.
- [x] Keep equation formatting and variable definitions consistent in DSP sections.
- [x] Keep launch examples mirrored across README, man pages, and cookbook.

---

## 5. Known Constraints (Alpha)

- Batch/offline-first architecture; real-time plugin hosting is not in `0.7.x` scope.
- Long-tail renders are compute-heavy by design; throughput depends strongly on
  partition/block settings and hardware.
- GPU acceleration currently targets convolution-heavy workloads.

---

## 6. Maintenance Rule

When a roadmap item is completed:

1. Update this roadmap checkbox immediately.
2. Update `CHANGELOG.md` under `Unreleased` (or release section if tagged).
3. Update `README.md` command/docs references in the same change.
