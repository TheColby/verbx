# verbx Roadmap

_Last updated: 2026-04-03. Maintained with `README.md` and `CHANGELOG.md`._

---

## 1. Release Posture

**Current release:** `v0.7.7`
**Status:** public alpha (research-grade)
**Versioning policy:** semantic (`0.7.x` patch line during alpha)

verbx currently ships dual-engine reverb, deterministic automation/feature
control, immersive QC/handoff, reproducibility tooling, f64 internal DSP,
experimental dereverberation workflows, a room size estimator, and initial
CLI-selectable realtime duplex auditioning.

---

## 2. v0.7.7 Current Patch Line — Structural Refactor

Patch line opened 2026-03-30. Items below are the active focus.

- [x] Runtime/package metadata aligned to `v0.7.7`.
- [x] `estimate_room_size` decomposed into six public pipeline stages (`extract_edr_rt60`, `infer_absorption`, `estimate_volume`, `project_dimensions`, `score_confidence`, `classify_room`).
- [x] FDN matrix operations extracted to `src/verbx/core/fdn_matrix.py` (all `build_*` and `apply_sparse_pair_mix` functions now independently importable and testable).
- [x] Pyright suppressions documented with rationale; `reportUnknownLambdaType` removed; remaining suppressions scoped with TODO for `0.7.7` follow-up.
- [x] Replace `dict[str, Any]` render reports in `pipeline.py` with typed `RenderReport` mapping objects while preserving CLI/test compatibility.
- [x] Extract algorithmic proxy IR generation into `src/verbx/core/algo_proxy.py` so offline streaming and realtime monitoring share one implementation.
- [x] Add an initial command-module split under `src/verbx/commands/` with `realtime.py` as the first standalone command surface.
- [x] Add initial realtime duplex monitoring with CLI-selectable input/output devices and algorithmic-proxy or convolution live engines.
- [x] Update README, CLI reference, and release/support docs for the refactor and realtime command surface.
- [ ] Decompose `cli.py` (8 376 lines) into per-command submodules under `src/verbx/commands/`.
- [ ] Decompose `RenderConfig` (162 fields) into composed sub-configs (`FDNConfig`, `AutomationConfig`, `SpatialConfig`, `StreamingConfig`).
- [x] Decompose `run_render_pipeline` (~640 lines) into explicit pipeline stages.
- [ ] Add dedicated unit tests for `automation.py`, `convolution_reverb.py`, `feature_vector.py`, `immersive.py`.
- [ ] Wire benchmark scripts into CI as blocking quality-regression gates.
- [ ] Enforce streaming/in-memory parity at the test level (extend `test_proxy_stream_parity.py` to cover convolution path).
- [ ] Decompose `algo_reverb.py` remaining methods into sub-modules (delay kernel, nonlinearity, spatial coupling).

## 2a. v0.7.6 Patch Line (Completed)

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
- [x] Define native error model, logging model, and deterministic offline process contract.
- [ ] Decide whether realtime audio belongs in the native line immediately or remains Python-only during transition.

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

## 6. Physically Modelled Room Acoustics

_Priority track opened 2026-03-31. Informs both the Python alpha line and the v0.8 native engine._

Current verbx reverb is parametric (FDN) and convolution-based.  Neither
engine derives its character from an explicit physical room model.  This
section tracks the work needed to add first-class physics-driven simulation.

### 6.1 Foundation — Room Geometry Model

- [x] Define `RoomGeometry` dataclass: dimensions (L × W × H), wall materials
  per face, source and listener positions (mirrors existing `--er-geometry`
  arguments but made first-class and reusable across engines).
- [x] Validate geometry against Bolt region criteria; emit warnings for
  pathological aspect ratios.
- [x] Add `verbx room-model` sub-command for geometry inspection and
  dimension-from-RT60 inversion (wraps existing `room_size.py` stages).

### 6.2 Image Source Method (ISM) — Full Response

The current `apply_image_source_early_reflections` only generates early
reflections up to a fixed order.  The full ISM response includes the
diffuse tail derived from image-source density.

- [ ] Extend ISM engine to configurable reflection order (1–6) with
  frequency-dependent wall absorption per material.
- [ ] Compute diffuse energy onset time from echo density; hand off to FDN
  at the Schroeder frequency transition.
- [ ] Expose as `--engine ism` and as a two-stage `ism+fdn` hybrid that uses
  ISM for the early field and FDN for the late diffuse field.
- [ ] Add parity corpus against measured anechoic + convolution references.

### 6.3 Scattering Delay Networks (SDN)

verbx's FDN matrix already contains an `sdn_hybrid` matrix type but without
a true SDN room topology.  SDN explicitly models wall scattering nodes.

- [ ] Implement full SDN room model: one scattering node per wall face,
  delay lines from source → nodes → listener, inter-node coupling matrix
  derived from room geometry.
- [ ] Derive delay-line lengths analytically from room dimensions and
  source/listener positions.
- [ ] Map SDN absorption coefficients from `RoomGeometry` wall materials.
- [ ] Validate against known RT60 and early-reflection timing benchmarks.
- [ ] Expose as `--engine sdn` with `--engine sdn+ism` hybrid option.

### 6.4 Geometry-to-FDN Parameter Derivation

For users who prefer the existing FDN engine but want physically grounded
parameter choices:

- [ ] Auto-derive FDN delay-line lengths from room dimensions (modal spacing
  from room geometry; prime-ratio delays from volume and aspect ratio).
- [ ] Auto-derive per-band RT60 targets from Sabine/Eyring with
  frequency-dependent absorption from material library.
- [ ] Auto-derive pre-delay from direct-path travel time (source → listener
  distance at speed of sound).
- [ ] Expose as `--preset room:<L>x<W>x<H>/<material>` shorthand.

### 6.5 Room Acoustics Material Library

- [ ] Add `src/verbx/ir/materials.py`: frequency-dependent absorption
  coefficient table for ~20 common materials (concrete, drywall, glass,
  carpet, acoustic foam, wood panel, etc.) drawn from published Sabine data.
- [ ] Expose via `--er-material` and the `RoomGeometry` model.
- [ ] Include Sabine data citations and units in module docstring.

### 6.6 Room Size Estimation from Recordings ✅

_Already shipped in v0.7.6._

`verbx analyze --room` estimates room volume, dimensions (W × D × H), mean
absorption, critical distance, and acoustic class directly from any
reverberant recording or rendered IR, using Sabine/Eyring inversion of
EDR-derived RT60 values.  The estimator exposes six independently callable
pipeline stages (`extract_edr_rt60`, `infer_absorption`, `estimate_volume`,
`project_dimensions`, `score_confidence`, `classify_room`) refined in
v0.7.7.

---

## 7. AI / Neural Architecture Track

_Informed by: Steinmetz et al., "Audio Signal Processing in the Artificial
Intelligence Era: Challenges and Directions," JAES Vol. 73, 2025
(MERL TR2025-116).  Items below are research-track goals; none are
committed to a specific patch line yet._

### 7.1 Neural Reverb Parameter Estimation

The paper identifies ML-based automatic audio effect parameter control
(including artificial reverberation, ref. [92]) as an active research
direction.  For verbx this means:

- [ ] `verbx suggest --match <reference.wav>`: neural network estimates
  optimal FDN/convolution parameters (RT60, pre-delay, damping, wet/dry)
  to match the acoustic character of a reference recording.
- [ ] Build training harness: synthetic room IRs → Sabine-derived labels →
  small CNN/TCN regressor per parameter group.
- [ ] Evaluate with perceptual similarity metrics (FAD, deep feature cosine
  distance) against held-out reference recordings.

### 7.2 Differentiable DSP (DDSP) FDN

The paper advocates combining classical DSP with neural networks via
differentiable signal processing (DDSP, ref. [12, 57]).

- [ ] Make the FDN processing graph differentiable (PyTorch/JAX mode):
  delay-line read, feedback matrix multiply, one-pole filter, wet/dry mix
  all as autograd-compatible operations.
- [ ] Enable gradient-based parameter optimisation: given input + target,
  minimize a perceptual loss through the differentiable FDN.
- [ ] Provide `verbx fit --target <ref.wav> --engine algo` CLI entry point.
- [ ] Hybrid loss: multi-scale STFT + deep feature distance (VGGish or
  music-domain embedding).

### 7.3 Grey-Box Neural FDN

The paper highlights grey-box neural models (refs. [98, 99]) as superior to
pure black-box approaches for audio effects while retaining interpretability.

- [ ] Add a lightweight residual neural correction layer on top of the
  physical FDN (small TCN operating on the FDN wet output).
- [ ] Train residual to minimise artefacts and spectral colouration vs.
  measured IRs without replacing the interpretable physical parameters.
- [ ] Keep all physical parameters user-visible; neural correction is an
  optional `--neural-correction` flag.

### 7.4 ML-Based Dereverberation

The paper covers ML approaches to source separation in reverberant
environments (Section 3.5, Section 3.6).  Current `verbx dereverb` is
entirely DSP-based (spectral subtraction + Wiener).

- [ ] Add an optional neural dereverberation backend: small causal TCN/LSTM
  trained on simulated reverberant/anechoic pairs.
- [ ] Constrain model to < 5 ms algorithmic latency (paper's hearing-aid
  target) for future real-time applicability.
- [ ] Benchmark against existing DSP dereverb using the existing
  `bark_snr_db`, `stoi_approx`, `mcd_db` harness in `benchmark_dereverb_quality.py`.

### 7.5 Perceptual Evaluation Infrastructure

The paper identifies PESQ, PEAQ, and FAD limitations and recommends
deep feature losses and differentiable reference-free metrics.

- [ ] Integrate `torchaudio-squim` as an optional reference-free quality
  estimator alongside existing PESQ-proxy metrics.
- [ ] Add Fréchet Audio Distance (FAD) gate to the dereverb benchmark harness.
- [ ] Add `verbx analyze --perceptual` flag that runs all quality estimators
  and returns a unified score dict (LUFS, BARK-SNR, STOI-approx, FAD-ref).

### 7.6 Automation Level / AI Interaction Tiers

The paper proposes four AI interaction tiers (automatic, independent,
suggestive, insightive).  verbx's `--auto-fit` is currently insightive
(provides starting parameters; user retains control).  Future:

- [ ] **Suggestive tier**: `verbx suggest` emits ranked parameter proposals
  with confidence scores and a human-readable rationale for each suggestion.
- [ ] **Independent tier**: `verbx auto-render` runs end-to-end with
  sensible defaults, zero required arguments, and a self-descriptive report.
- [ ] Ensure all AI-derived parameter changes are logged, reversible, and
  fully auditable in the analysis JSON output.

### 7.7 High Sample Rate and Rate-Agnostic Processing

The paper flags 96 kHz support and rate-agnostic model design as open
problems.

- [ ] Profile and optimise verbx DSP kernels at 88.2 / 96 kHz; identify
  bottlenecks in FDN matrix ops and convolution engine.
- [ ] Investigate implicit neural representation (INR) of impulse responses
  for sample-rate-independent IR interpolation (paper ref. [42]).
- [ ] Add 96 kHz test fixtures to the CI matrix.

---

## 8. Valhalla-Inspired Algorithm Research

_Study track: document specific algorithmic techniques from the Valhalla
DSP reverb family and assess which are missing or under-developed in verbx._

- [ ] **Dense diffusion networks**: Valhalla uses very-high-order allpass
  cascades and nested allpass structures for extreme diffusion density.
  Assess verbx allpass chain depth and add `--allpass-stages` guidance notes.
- [ ] **Modulated delay lines with interpolated read**: Valhalla employs
  high-resolution fractional delay interpolation and band-limited modulation
  to eliminate metallic artefacts.  Audit verbx's current modulation
  implementation (`mod_depth_ms`, `mod_rate_hz`) and identify interpolation
  order gaps.
- [ ] **Per-line crossover filters**: Valhalla splits each FDN delay line
  into frequency bands with independent gains — beyond verbx's current
  three-band crossover.  Add per-line EQ post-filter capability.
- [ ] **Pre-echo / smear controls**: Valhalla exposes "Size", "Diffusion", and
  "Pre-delay" as independent perceptual controls rather than direct DSP
  parameters.  Map these to verbx's parameter space and expose as macro
  shortcuts.
- [ ] **Shimmer / pitch-shift algorithms**: Compare Valhalla Shimmer's
  pitch-shifting approach against verbx's current `ShimmerProcessor`
  (librosa-based phase vocoder).  Investigate time-domain PSOLA or
  STFT-based pitch shifting for lower latency and artefact profile.
- [ ] **Room mode resonances**: Valhalla Room explicitly models low-frequency
  room modes as resonant filter banks.  Cross-reference with the physically
  modelled room work in Section 6.

---

## 9. Known Constraints (Alpha)

- Offline-first architecture; real-time plugin hosting is not in `0.7.x` scope.
- Very long tails remain compute-heavy; throughput depends on partition/block settings and hardware.
- CUDA acceleration currently benefits convolution-heavy paths most.
- Render-time sample-rate conversion is deterministic and offline-oriented.

---

## 10. Maintenance Rule

When a roadmap item is completed:

1. Update this roadmap immediately.
2. Update `CHANGELOG.md` in the same change.
3. Update `README.md` command/docs references in the same change.
