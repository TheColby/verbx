# verbx Roadmap

_Last updated: 2026-03-28. Maintained with `README.md` and `CHANGELOG.md`._

---

## 1. Release Posture

**Current release:** `v0.7.5`  
**Status:** public alpha (research-grade)  
**Versioning policy:** semantic (`0.7.x` patch line during alpha)

verbx currently ships dual-engine reverb, deterministic automation/feature
control, immersive QC/handoff, reproducibility tooling, f64 internal DSP, and
experimental dereverberation workflows.

---

## 2. v0.7.5 Feature Pack (Completed)

Requested feature set 1-10 is now implemented and tested:

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

## 3. Remaining 0.7.x Priorities

- [ ] Expand `verbx dereverb` objective quality validation (PESQ/STOI/ASR WER-style benchmark harness).
- [ ] Broaden algorithmic proxy-stream eligibility while preserving deterministic parity checks.
- [ ] Add CI/hardware coverage for CUDA and Apple Silicon acceleration paths.
- [ ] Tighten public alpha packaging/release health checks across PyPI and Homebrew channels.

---

## 4. Known Constraints (Alpha)

- Offline-first architecture; real-time plugin hosting is not in `0.7.x` scope.
- Very long tails remain compute-heavy; throughput depends on partition/block settings and hardware.
- CUDA acceleration currently benefits convolution-heavy paths most.
- Render-time sample-rate conversion is deterministic and offline-oriented.

---

## 5. Maintenance Rule

When a roadmap item is completed:

1. Update this roadmap immediately.
2. Update `CHANGELOG.md` in the same change.
3. Update `README.md` command/docs references in the same change.
