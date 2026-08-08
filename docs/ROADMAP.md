# verbx Roadmap

_Last updated: 2026-07-26. Maintained with `README.md`, `CHANGELOG.md`, and the generated user guide outputs._

---

## 1. Release Posture

**Current release:** `v0.9.9`
**Status:** public alpha (research-grade)
**Versioning policy:** semantic prerelease (`0.x` during public alpha)

verbx currently ships dual-engine reverb, deterministic automation/feature
control, immersive QC/handoff, reproducibility tooling, f64 internal DSP,
experimental dereverberation workflows, a room size estimator, and initial
CLI-selectable realtime duplex auditioning.

---

## 2. v0.9.9 Plug-in A/B Comparison

- [x] Add full parameter-state A/B snapshots to the plug-in processor.
- [x] Surface A/B recall controls in the editor.
- [x] Verify bidirectional A/B recall in the JUCE interaction smoke test.

`v0.9.9` adds a direct reverb-comparison workflow without adding audio-thread
allocations or changing the host automation contract.

## 2a. v0.9.8 Preset Browser Filtering

- [x] Add live name filtering to the in-editor preset browser.
- [x] Preserve program IDs when the visible list is filtered.
- [x] Cover filter, selection, and reset behavior in the JUCE interaction
  smoke test.

`v0.9.8` makes the large native program library practical to scan during a
session without changing host program semantics.

## 2a. v0.9.7 In-Editor Preset Browser

- [x] Add an editor preset browser backed by the host-visible program bank.
- [x] Keep the browser synchronized with processor program changes.
- [x] Verify program/model recall through the JUCE interaction smoke test.

`v0.9.7` makes the 260-program library directly usable from the plug-in
surface, while retaining normal AU/VST3 host program behavior.

## 2a. v0.9.6 Model-Aware Plug-in Controls

- [x] Make physical character controls contextual to the active model.
- [x] Keep model switching host-automatable while keeping irrelevant controls
  out of the active Perform surface.
- [x] Cover Spring, Plate, and Algorithmic visibility behavior in the JUCE
  interaction smoke test.

`v0.9.6` refines the physical-model workflow so the editor presents one clear
character control at a time without narrowing the host automation surface.

## 2a. v0.9.5 Physical-Model Character Controls

- [x] Add host-automatable Spring Tension and Plate Brightness parameters.
- [x] Apply character controls inside bounded realtime Spring and Plate DSP
  paths with callback-safe smoothing.
- [x] Surface both controls in the plug-in editor and test their host state.

`v0.9.5` deepens physical-model control without changing the plug-in's
allocation-free realtime contract.

## 2a. v0.9.4 Native Physical-Model Plug-in Slice

- [x] Expose host-automatable Algorithmic, Spring, and Plate realtime models.
- [x] Add bounded serial spring-tank and parallel dispersive plate topologies
  to the shared native realtime core.
- [x] Wire Spring and Plate preset families to their matching models and cover
  host-state, C realtime, and JUCE editor interactions.

`v0.9.4` is a focused native plug-in expansion. The realtime physical models
remain intentionally bounded approximations; offline modal FE remains the
high-detail sound-design path.

## 2a. v0.9.3 Streaming Parity and CLI Decomposition Slice

- [x] Enforce full-pipeline convolution streaming/in-memory parity for peak
  and RMS tail completion.
- [x] Fix partial-partition streaming tail loss for non-aligned source lengths.
- [x] Move generic numeric and choice parsers into the shared command validator
  module while retaining CLI compatibility aliases.

`v0.9.3` remains a focused hardening release. It closes a sample-level
streaming correctness gap before further native or physical-acoustics breadth.

## 2a. v0.9.1 Stabilization and Validation Slice

- [x] Expose composed, typed views over the monolithic render configuration
  while retaining CLI and report compatibility.
- [x] Promote render-performance regression detection to a blocking CI gate.
- [x] Tighten the deterministic native-render behavior covered by the parity
  contract.
- [x] Add an analytic rectangular-room ISM reference corpus covering direct
  and first-order path distance, sample timing, and material-dependent gain.

`v0.9.1` is intentionally a hardening release. Broader SDN, neural, and
arbitrary-CAD expansion remains gated on repeatable evaluation evidence.

---

## 3. v0.9 Physical Room Slice (Completed)

- [x] Ship `--engine ism-fdn` for a rectangular image-source early field fed
  into the established FDN late field.
- [x] Extend ISM to reflection orders `0..6` with deterministic, material-aware
  per-surface reflectivity.
- [x] Preserve resolved room dimensions, source/listener positions, wall
  materials, warnings, and ISM order in render-report provenance.
- [x] Add direct acoustic and end-to-end CLI regression coverage.

The native `verbx-c` executable does not claim ISM/FDN parity in `v0.9`; the
physical room path remains Python-reference functionality until the native FDN
port and parity contract can support the same scene model.

## 3a. Experimental Electro-Mechanical Modal FE (Completed)

- [x] Add `--electromechanical-solver modal-fe` alongside the fast default
  proxy voice for `--algo-model spring|plate`.
- [x] Solve bounded lumped-mass spring chains, optional inter-spring coupling,
  and structured mass-lumped clamped plate grids as deterministic modal IRs.
- [x] Expose mesh, retained-mode, coupling, loss, material, and pickup controls
  and document the governing generalized eigenproblem.

This is an offline research/sound-design solver, not a calibrated commercial
hardware emulation. Native `verbx-c` remains proxy-only for this feature.

---

## 3. v0.7.8 Model and Stability Slice (Completed)

- [x] Add explicit algorithmic model selection: `fdn`, `spring`, and `plate`.
- [x] Add deterministic spring/plate topology defaults while preserving RT60,
  damping, width, modulation, automation, proxy rendering, and report output.
- [x] Add `classic_spring` and `bright_plate` reference presets.
- [x] Extend `verbx-c render` with `--model fdn|spring|plate` and report the
  selected native model in `native-render-report-v1`.
- [x] Add Python and native regression coverage proving finite, distinct model
  tails and native JSON reporting.

The remaining `0.7.x` stabilization work is intentionally narrow: complete
the `cli.py` helper extraction, compose `RenderConfig`, and promote benchmark
and streaming-parity checks to CI gates before another major DSP expansion.

---

## 4. v0.7.7 Current Patch Line – Structural Refactor

Patch line opened 2026-03-30. Items below are the active focus.

- [x] Runtime/package metadata aligned to `v0.7.7`.
- [x] `estimate_room_size` decomposed into six public pipeline stages (`extract_edr_rt60`, `infer_absorption`, `estimate_volume`, `project_dimensions`, `score_confidence`, `classify_room`).
- [x] FDN matrix operations extracted to `src/verbx/core/fdn_matrix.py` (all `build_*` and `apply_sparse_pair_mix` functions now independently importable and testable).
- [x] Pyright suppressions documented with rationale; `reportUnknownLambdaType` removed; remaining suppressions scoped with TODO for `0.7.7` follow-up.
- [x] Replace `dict[str, Any]` render reports in `pipeline.py` with typed `RenderReport` mapping objects while preserving CLI/test compatibility.
- [x] Extract algorithmic proxy IR generation into `src/verbx/core/algo_proxy.py` so offline streaming and realtime monitoring share one implementation.
- [x] Add an initial command-module split under `src/verbx/commands/` with `realtime.py` as the first standalone command surface.
- [x] Continue command-module split by moving onboarding/diagnostic commands into `src/verbx/commands/system.py`.
- [x] Continue command-module split by moving preset inspection and cache commands into `src/verbx/commands/`.
- [x] Continue command-module split by moving `analyze`, `compare`, and `suggest` into dedicated command modules plus shared command helpers.
- [x] Add initial realtime duplex monitoring with CLI-selectable input/output devices and algorithmic-proxy or convolution live engines.
- [x] Update README, CLI reference, and release/support docs for the refactor and realtime command surface.
- [x] Move all CLI command entrypoints into per-command submodules under `src/verbx/commands/`.
- [x] Continue shrinking `cli.py` by migrating a cohesive parser and
  choice-validation helper cluster out of the legacy entrypoint module.
- [x] Decompose `RenderConfig` (162 fields) into composed sub-config snapshots (`FDNConfig`, `AutomationConfig`, `SpatialConfig`, `StreamingConfig`) while preserving the flat constructor and serialized preset compatibility.
- [x] Decompose `run_render_pipeline` (~640 lines) into explicit pipeline stages.
- [x] Add dedicated unit tests for `automation.py`, `convolution_reverb.py`, `feature_vector.py`, `immersive.py`.
- [x] Wire benchmark scripts into CI as blocking quality-regression gates.
- [x] Enforce streaming/in-memory parity at the test level (extend
  `test_proxy_stream_parity.py` to cover convolution peak/RMS tail paths).
- [x] Extract FDN nonlinearity and spatial-coupling helpers into dedicated,
  directly tested modules.
- [x] Extract pure FDN delay-layout and fractional-read helpers into
  `fdn_delays.py`, retaining engine compatibility wrappers and direct contracts.

## 4a. v0.7.6 Patch Line (Completed)

- [x] Runtime/package metadata aligned to `v0.7.6`.
- [x] Tail completion, proxy streaming, dereverb QA, release-health tooling, and IR library work shipped in `v0.7.6`.
- [x] Land the next focused `0.7.x` patch feature set and promote it from `Unreleased` into `CHANGELOG.md`.
- [x] Room size estimator integrated into analysis engine (`verbx analyze --room`, `verbx compare --room`, `AudioAnalyzer.analyze(include_room=True)`).

## 5. v0.7.5 Feature Pack (Completed)

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

## 6. v0.8 Native Executable Program

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
- [x] Add explicit native `--tail-limit` control and report it in `native-render-report-v1`; model-derived DSP padding is now bounded consistently with Python's configured tail limit.

### 4.3 DSP Port

- [x] Port a first native offline late-field core (pre-delay, combs, allpass diffusion, tail finalization).
- [x] Replace the foundational `fdn` comb bank with a bounded eight-line
  Hadamard FDN slice.
- [ ] Extend the bounded native FDN to the Python reference's modulation,
  multiband, automation, and advanced topology surface.
- [ ] Port damping, width, pre-delay, freeze, repeat, and normalization in controlled phases.
- [x] Add native peak-safe output with deterministic peak/gain reporting.
- [x] Define the first narrow parity contract in `tests/fixtures/native_render_parity_contract.json`.
- [x] Generate Python/native metric comparisons from that contract before feature expansion (`scripts/compare_native_render_parity.py`).
- [x] Emit `native-render-report-v1` JSON from `verbx-c render --json-out` for
  machine-readable native support bundles.

### 4.4 Productization

- [x] Decide whether `verbx-c` remains a transition binary or replaces `verbx`
  at release: `v0.8` is a hybrid wrapper phase before full replacement.
- [x] Document the chosen `v0.8` parity scope in `README.md` and this roadmap.
- [ ] Add native packaging/release flow:
  - [x] install script
  - [x] man page
  - [ ] Homebrew formula/tap integration for `verbx-c`
  - [ ] CI packaging/release check
- [x] Improve native build/doctor ergonomics with build-script flags and
  `native-doctor-report-v1` JSON diagnostics.
- [x] Document feature parity and feature gaps continuously during the migration
  in `docs/NATIVE_PARITY.md`.

Chosen `v0.8` release shape:

- Ship `verbx-c` as an opt-in native executable, not as a replacement for the
  Python `verbx` command.
- Support the deterministic offline render slice first: mono/stereo WAV input,
  `pcm16`/`float32`/`float64` output, `rt60`, `wet`, `dry`, pre-delay, damping,
  tail threshold/hold/metric, peak-safe output, and render/doctor JSON reports.
- Keep Python as the default public-alpha CLI for realtime, dereverb,
  convolution, IR workflows, batch, immersive utilities, presets, and the full
  FDN feature surface.
- Require native feature expansion to pass the checked-in parity contract and
  comparison harness before broadening scope.

---

## 7. Remaining 0.7.x Priorities

- [x] Expand `verbx dereverb` objective quality validation (PESQ/STOI/ASR WER-style benchmark harness).
- [x] Broaden algorithmic proxy-stream eligibility while preserving deterministic parity checks.
- [x] Add CI/hardware coverage for CUDA and Apple Silicon acceleration paths.
- [x] Tighten public alpha packaging/release health checks across PyPI and Homebrew channels.

---

## 8. Physically Modelled Room Acoustics

_Priority track opened 2026-03-31. Informs both the Python alpha line and the v0.8 native engine._

Current verbx reverb is parametric (FDN) and convolution-based.  Neither
engine derives its character from an explicit physical room model.  This
section tracks the work needed to add first-class physics-driven simulation.

### 6.1 Foundation – Room Geometry Model

- [x] Define `RoomGeometry` dataclass: dimensions (L × W × H), wall materials
  per face, source and listener positions (mirrors existing `--er-geometry`
  arguments but made first-class and reusable across engines).
- [x] Validate geometry against Bolt region criteria; emit warnings for
  pathological aspect ratios.
- [x] Add `verbx room-model` sub-command for geometry inspection and
  dimension-from-RT60 inversion (wraps existing `room_size.py` stages).

### 6.2 Image Source Method (ISM) – Full Response

The shipped `ism-fdn` engine generates deterministic image-source paths at
orders 0–6 and hands the material-aware early field to the established FDN
late field. A pure full-response ISM engine and an echo-density-derived
Schroeder handoff remain research work.

- [x] Extend ISM early-field generation to configurable reflection order
  (0–6) with material-dependent wall absorption.
- [ ] Compute diffuse energy onset time from echo density and derive the FDN
  handoff at the Schroeder transition instead of using the current hybrid
  boundary.
- [x] Expose the two-stage early-ISM/late-FDN path as `--engine ism-fdn`.
- [ ] Add a pure `--engine ism` full-response mode only after its computational
  bounds and validation contract are defined.
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

- [x] Auto-derive FDN delay-line lengths from room dimensions (modal spacing
  from room geometry; prime-ratio delays from volume and aspect ratio).
- [x] Auto-derive per-band RT60 targets from Sabine/Eyring with
  frequency-dependent absorption from material library.
- [x] Auto-derive pre-delay from direct-path travel time (source → listener
  distance at speed of sound).
- [x] Expose as `--preset room:<L>x<W>x<H>/<material>` shorthand.

### 6.5 Room Acoustics Material Library

- [x] Add `src/verbx/ir/materials.py`: frequency-dependent absorption
  coefficient table for ~20 common materials (concrete, drywall, glass,
  carpet, acoustic foam, wood panel, etc.) drawn from published Sabine data.
- [x] Expose the table through `verbx ir trace --material` reports and the
  `RoomGeometry` mean-absorption model.
- [x] Include coefficient units/frequency bands in the module docstring and
  JSON report payload.
- [ ] Add per-surface/per-layer material assignment once DXF layer import lands.

### 6.6 CAD / DXF Ray-Tracing IR Import

Goal: import constrained architectural CAD geometry and generate an impulse
response that can feed the existing convolution engine, without claiming full
architectural-acoustics accuracy in the first slice.

- [x] Define supported geometry subset for MVP: clean DXF room boundaries,
  planar wall/floor/ceiling surfaces, closed volumes or closed 2D plans with
  explicit height, and optional layer-name material hints.
- [x] Add experimental command shape:
  `verbx ir trace ROOM.dxf OUT_IR.wav --source x,y,z --listener x,y,z --rays N
  --length S --material default:NAME --target-sr SR`.
- [x] Build DXF ingest/normalization stage that converts CAD entities into
  `RoomGeometry`-compatible planes/triangles and emits warnings for open
  boundaries, non-manifold geometry, unsupported curves, or missing scale units.
- [x] Generate deterministic early reflections with image-source/ray-hit timing
  and amplitude, then synthesize late decay from stochastic ray energy histograms.
- [x] Support frequency-dependent material absorption/scattering for the default
  material profile in `trace-report-v1`.
- [ ] Add per-layer material overrides once DXF layer import lands.
- [x] Write `trace-report-v1` JSON with geometry stats, material assignment,
  direct path, reflection counts, ray budget, estimated RT60, and warnings.
- [x] Output an IR WAV usable by `verbx render --engine conv --ir OUT_IR.wav`.
- [x] Keep first implementation experimental and scoped to demoable room-like
  DXF files; robust arbitrary CAD cleanup remains a later 2+ month milestone.
- [x] Retain DXF tracing as experimental after the `v0.9.2` evidence review;
  measured-reference validation and per-layer materials are still required
  before graduation.

Estimated effort:

- Prototype: 1-2 weeks for simple DXF ingest plus plausible early reflections.
- Useful MVP: 3-5 weeks for DXF-to-IR, reports, docs, and fixtures.
- Good acoustic tool: 6-10 weeks for materials, stochastic tails, validation,
  and plots.

### 6.7 Room Size Estimation from Recordings ✅

_Already shipped in v0.7.6._

`verbx analyze --room` estimates room volume, dimensions (W × D × H), mean
absorption, critical distance, and acoustic class directly from any
reverberant recording or rendered IR, using Sabine/Eyring inversion of
EDR-derived RT60 values.  The estimator exposes six independently callable
pipeline stages (`extract_edr_rt60`, `infer_absorption`, `estimate_volume`,
`project_dimensions`, `score_confidence`, `classify_room`) refined in
v0.7.7.

---

## Next 30 Days Priorities (Snapshot)

- Lock a first-pass `verbx.api` module and document stability guarantees.
- Publish at least two notebook examples (`render`, `ir`, `analyze`).
- Add immersive routing docs for `7.2.4` and `16.0` as baseline layouts.
- Complete one end-to-end schema example for automation + manifest JSON.

---

## 9. AI / Neural Architecture Track

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

## 10. Valhalla-Inspired Algorithm Research

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
  into frequency bands with independent gains – beyond verbx's current
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

## 11. Known Constraints (Alpha)

- The Python `0.7.x` line remains offline/realtime-CLI focused. The `v0.8`
  native track now includes a usable mono/stereo AUv2/AUv3/VST3/standalone
  plug-in slice. The true AUv3 extension is embedded, sandbox-entitled, signed,
  and PlugInKit-registered through its containing app; broad host certification
  is still incomplete.
- Native quality modes now execute allocation-free wet-path oversampling at
  Host, 2x, 4x, or the smallest integer factor at or above 192 kHz. Higher-order
  resampling filters, true lookahead reverse processing, and multichannel
  plug-in layouts remain parity work.
- Very long tails remain compute-heavy; throughput depends on partition/block settings and hardware.
- CUDA acceleration currently benefits convolution-heavy paths most.
- Render-time sample-rate conversion is deterministic and offline-oriented.

---

## 12. Maintenance Rule

When a roadmap item is completed:

1. Update this roadmap immediately.
2. Update `CHANGELOG.md` in the same change.
3. Update `README.md` command/docs references in the same change.
