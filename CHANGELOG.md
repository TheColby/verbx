# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Changed
- Native AU/VST3 host validation now negotiates a stereo bus layout and
  requires measurable wet-tail energy instead of checking only finite samples.
- The macOS installer now unregisters an ad-hoc AUv3 extension so it cannot
  shadow the validated AUv2 component in Logic. AUv3 hosting is enabled with
  an explicit Apple code-signing identity, with a development-only ad-hoc
  override.
- Made the native editor responsive to DAW window negotiation: it now opens at
  a host-safe 1280x720, can shrink to 800x450, and no longer forces a fixed
  aspect-ratio constrainer that could prevent Audacity or Logic from opening
  the plug-in window.
- Rebuilt the native plug-in editor as the approved full-screen 16:9 spatial
  console: DXF geometry theater, loudness meters, image/ray panels, live decay
  analyzer, nine parameter cards including separate RT60 coarse/fine controls,
  quality/mode controls, and expert status cards. The README now carries a real
  capture of the compiled implementation.
- Native plug-in metadata now identifies Colby Leider as the author/vendor,
  uses the stable `com.colbyleider.verbx` bundle identifier, and the installer
  fully signs and verifies copied macOS bundles before refreshing Audio Unit
  discovery. An explicit `--reset-plugin-cache` recovery path backs up stale
  Apple cache files before forcing a clean host scan.
- macOS AU/VST3/standalone builds now default to universal `arm64+x86_64`
  binaries with a macOS 12 deployment target, preventing plug-in disappearance
  when a universal DAW or scanner runs under Rosetta.
- `run_render_pipeline()` is now split into explicit streaming and in-memory
  stage helpers, reducing the amount of orchestration living in a single branchy
  function.
- CLI decomposition continues: `presets` and `cache` now live under
  `src/verbx/commands/`, following the earlier `realtime`, `room_model`, and
  `system` extractions.
- `analyze`, `compare`, and `suggest` now live in dedicated command modules,
  and shared CLI status/JSON helpers were moved into
  `src/verbx/commands/common.py`.
- All remaining CLI entrypoints now register from command modules
  (`render`, `dereverb`, `ir`, `batch`, and `immersive` families), leaving
  `src/verbx/cli.py` as the shrinking implementation/helper layer instead of
  the public command-definition surface.
- Command-module split status: the next safe post-IR slice is shared validators;
  path/output-audio validation now lives in `src/verbx/commands/validators.py`,
  with render, batch, and IR workflows continuing to use compatibility aliases
  while the remaining helper clusters are extracted.
- Realtime stabilization now reports device-selection and backend stream-open
  failures with viable device hints, attempted sample rate/block/channel
  settings, and recovery guidance instead of terse backend errors.
- Render/realtime/dereverb report payloads are more consistent:
  `verbx realtime --json-out` now writes a `realtime-report-v1` session report,
  and dereverb reports include shared `command` and `status` metadata.
- Week 3 output-delivery hardening now rejects contradictory limiter
  threshold/ceiling settings and explicit container/extension mismatches before
  rendering, avoiding silent limiter collapse or mislabeled W64/RF64 files.
- Added Week 3 stabilization presets and quickstart/docs examples for
  room-modelled renders, dereverb cleanup, limiter-safe delivery, and bounded
  long-tail W64 output.

### Added
- Added an opt-in JUCE host smoke executable that discovers AU/VST3 bundles,
  instantiates VERBX, creates its editor, and processes an impulse so host-load
  regressions can be diagnosed independently of DAW caches.
- Added a true macOS AUv3 app extension: the JUCE AUv3 wrapper now builds with
  standard CMake generators, links through `NSExtensionMain`, and embeds as
  `VERBX.app/Contents/PlugIns/VERBX.appex`. The installer nested-signs and
  registers the extension with PlugInKit under Colby Leider.
- Expanded `./install.sh` into a complete per-user installer for Python runtime
  extras, man pages, `verbx-c`, Release Audio Unit/VST3 bundles, and the JUCE
  standalone app, with pinned JUCE download, offline source, custom destination,
  component-skip, existing-artifact, and dry-run controls.
- Promoted the native JUCE plug-in from pass-through scaffold to an audible,
  allocation-free mono/stereo Schroeder reverb with the complete initial
  12-parameter control dock, effective-RT60 display, Freeze, reverse-style
  swell, and overlaid realtime spectrum analyzer.
- Added persistent realtime DSP state, wet-tail/reset tests, sanitizer coverage,
  20 ms automation smoothing, and successful standalone, AU, and VST3 build
  validation.
- `src/verbx/core/room_geometry.py` with a reusable `RoomGeometry` dataclass,
  direct-path/pre-delay metrics, aspect-ratio analysis, and Bolt-style warnings
  for physically grounded room workflows.
- Low-latency realtime dereverb modes for `verbx realtime`:
  `--live-mode dereverb` for standalone cleanup and
  `--live-mode dereverb-reverb` for dereverb feeding the live reverb engine,
  with dedicated spectral controls for mode, strength, floor, window/hop,
  tail tracking, attenuation clamp, stereo linking, and gain trim.
- Optional algorithmic comb-cloud coloration for `verbx render` via
  `--comb-cloud`, with controls for bank size, feedback, mix, seed, and custom
  delay lists.
- Consolidated documentation build outputs: `docs/USERGUIDE.md` and
  `USERGUIDE.pdf`, generated from README plus user-facing guides/tips.
- `verbx room-model`, a new CLI sub-command for inspecting explicit room
  geometry or inferring a plausible rectangular room from RT60 plus an
  absorption/material assumption.
- Dynamic room-derived render presets via
  `--preset room:<width>x<depth>x<height>/<material>`, which apply a geometry
  baseline (`er_geometry`, direct-path pre-delay, absorption/material, room-size
  macro, and size-scaled FDN density) without taking away explicit CLI
  overrides.
- Expanded the built-in reverb preset bank to 293 presets by generating 280
  deterministic style/space combinations such as `warm_chamber`,
  `shimmer_cathedral`, `cinematic_scoring_stage`, and `infinite_cavern`.
- Native `verbx-c` process-contract reporting and tail metric selection:
  `verbx-c doctor` now surfaces the deterministic offline lifecycle and exit
  contract, and `verbx-c render` supports `--tail-metric peak|rms`.
- Native `verbx-c render` now supports peak-safe output via `--peak-safe` and
  `--peak-ceiling-db`, with input/output peak and gain values reported in the
  deterministic render summary.
- Native `verbx-c render --json-out report.json` now writes a
  `native-render-report-v1` support bundle with render settings, frame counts,
  output format, tail metric, and peak-safety analysis fields.
- Native doctor/build ergonomics improved: `verbx-c doctor --json-out` writes a
  `native-doctor-report-v1` diagnostics bundle, and
  `scripts/build_verbx_c.sh` now supports `--clean`, `--doctor`, `--print-path`,
  `CC`, `CFLAGS`, and `LDFLAGS`.
- Added native install packaging primitives: `scripts/install_verbx_c.sh`
  installs `verbx-c` plus the new `verbx-c(1)` man page into a chosen prefix.
- Added `docs/NATIVE_PARITY.md` as the canonical native feature/gap matrix for
  the `v0.8` hybrid transition.
- Added experimental `verbx ir trace`, a constrained ASCII DXF room-outline to
  IR prototype that writes stereo IR WAV files plus `trace-report-v1` geometry,
  reflection, ray-budget, and metric reports.
- Added `src/verbx/ir/materials.py` with 20 frequency-dependent room-material
  profiles; `verbx ir trace --material` now validates names and records
  octave-band absorption plus scattering metadata in `trace-report-v1`.
- Documented the chosen `v0.8` release shape as a hybrid transition: `verbx-c`
  ships as an opt-in native render/doctor binary while Python `verbx` remains
  the default public-alpha CLI until broader parity is proven.
- `scripts/compare_native_render_parity.py`, a contract-driven Python/native
  render comparison harness that builds `verbx-c`, renders deterministic
  fixtures from `tests/fixtures/native_render_parity_contract.json`, and emits
  machine-readable metric reports for the `v0.8` parity track.

## [0.7.7] - 2026-03-31

### Changed
- **`room_size.py` decomposed into six public pipeline stages**: `extract_edr_rt60`,
  `infer_absorption`, `estimate_volume`, `project_dimensions`, `score_confidence`,
  `classify_room` — each independently callable, testable, and replaceable.
  `estimate_room_size` is now a thin orchestrator that calls them in sequence.
- **FDN matrix operations extracted to `src/verbx/core/fdn_matrix.py`**: all
  `build_*` matrix builders (`hadamard`, `random_orthogonal`, `circulant`,
  `elliptic`, `sdn_hybrid`, `householder`), graph topology helpers
  (`build_graph_edges`, `build_graph_pairings`, `build_sparse_pairings`), and
  `apply_sparse_pair_mix` / `build_sparse_mix_matrix` are now module-level
  functions importable without instantiating `AlgoReverbEngine`.
- **Pyright suppressions documented**: all six suppressions in `pyproject.toml`
  now carry rationale comments; `reportUnknownLambdaType` removed; remaining
  four carry explicit TODO markers for narrowing to per-file scope once
  `pipeline.py` report types are strengthened.

### Added
- **Roadmap sections 6–8**: physically modelled room acoustics track (ISM full
  response, SDN engine, geometry-to-FDN parameter derivation, material library),
  AI/neural architecture track (DDSP FDN, grey-box neural FDN, neural
  dereverberation, perceptual evaluation infrastructure — informed by Steinmetz
  et al. MERL TR2025-116 JAES 2025), and Valhalla-inspired algorithm research
  track (diffusion density, per-line crossover filters, room mode resonances).

### Added
- **Room size estimator** (`verbx.analysis.room_size`): estimates room volume,
  dimensions, mean absorption, critical distance, and acoustic class from any
  reverberant recording or rendered IR.  Uses Sabine/Eyring reverberation
  formulas with EDR-derived RT60 measurements and automatic absorption inference
  from the spectral decay shape.  Exposed via:
  - `AudioAnalyzer.analyze(include_room=True)` — adds `room_*` prefixed keys
    to the flat metrics dict
  - `verbx analyze --room` — prints room metrics in the analysis table and
    includes them in `--json-out` output
  - `verbx compare --room` — shows room metrics side-by-side with string-valued
    fields (class, method, confidence) displayed as labels and numeric fields
    shown with delta

### Previously added
- Began the `v0.8` native executable track with a standalone C11 scaffold under
  `native/verbx_c/`.
- Added a functional native `verbx-c` render path with mono/stereo WAV
  read/write, deterministic offline lifecycle orchestration, and a foundational
  float64 algorithmic reverb core in C.
- Added `scripts/build_verbx_c.sh` for one-command native builds and
  `tests/test_native_scaffold.py` as compile, CLI, and WAV round-trip coverage
  for the native executable track.

## [0.7.6] - 2026-03-29

### Changed
- Tail completion now trims excess post-decay silence instead of preserving full
  padded render length.
- Tail finalization now applies a short click-safe raised-cosine fade before the
  hard-zero hold segment to avoid end-of-file clicks.
- **Algorithmic proxy-stream eligibility broadened**: `--algo-stream` now works
  with `--target-sr` (sample-rate conversion handled via pre-resampling to a
  temp file) and with `--lowcut` / `--highcut` / `--tilt` (EQ applied as a
  deterministic post-stream pass).  All time-varying controls (modulation,
  shimmer, FDN time-variance, matrix morphing) remain on the standard path.
  Parity verified against the in-memory render path.

### Added
- Tail completion regression tests for in-memory and streaming write paths
  (`tests/test_tail_completion.py`).
- Large curated IR library generator (`scripts/generate_ir_library.py`) and
  generated folder-sorted library under `IRs/library/` (varying lengths across
  tiny/short/medium/long buckets, all four synthesis modes, deterministic
  manifest and metadata sidecars).
- **Dereverb objective quality benchmark harness** (`scripts/benchmark_dereverb_quality.py`):
  multi-scenario CLI tool with PESQ-inspired Bark-weighted SNR, STOI
  approximation, and mel-cepstral distortion (MCD / ASR WER proxy) metrics,
  JSON output, and configurable pass/fail thresholds.
- Three new objective quality metrics in `verbx.core.dereverb`:
  `_bark_weighted_snr_db` (PESQ-style), `_stoi_approx` (intelligibility proxy),
  `_mcd_db` (mel-cepstral distortion / ASR WER proxy). All reported in
  `run_dereverb_benchmark()` results (schema bumped to `dereverb-benchmark-v2`).
- **CI hardware coverage**: `ci-gpu.yml` now runs automatically on push to
  `main` and weekly (cron), uses `macos-14` (guaranteed Apple Silicon / M1) for
  MPS tests, adds a Python 3.12 matrix entry, and runs the full core test suite
  on Apple Silicon. `ci.yml` adds a `test-macos` job on `macos-14`.
- **Release health checks** (`scripts/check_release_health.py`): pre-release
  script verifying version tag / `pyproject.toml` consistency, CHANGELOG entry
  presence, and Homebrew formula pin. Integrated into `release.yml` before and
  after the build step, plus a `verify-pypi-install` job that smoke-tests the
  published package from PyPI after upload.

## [0.7.5] - 2026-03-28

### Added
- New deterministic dereverberation command:
  - `verbx dereverb INFILE OUTFILE`
  with `wiener` and `spectral_sub` modes, configurable STFT/tail controls,
  subtype selection, and optional JSON reporting.
- Tail write-completion controls:
  - `--tail-stop-threshold-db`
  - `--tail-stop-hold-ms`
  - `--tail-stop-metric`
- Algorithmic long-render proxy streaming:
  - `--algo-stream`
  - `--algo-proxy-ir-max-seconds`
  - `--algo-gpu-proxy` (CUDA path)
- Output container controls:
  - `--output-container auto|wav|w64|rf64`
  including auto-upgrade logic for large outputs.
- Matrix morph controls for algorithmic FDN:
  - `--fdn-matrix-morph-to`
  - `--fdn-matrix-morph-seconds`
- New automation/control targets for multiband decay behavior:
  - `fdn-rt60-low`, `fdn-rt60-mid`, `fdn-rt60-high`
  - `fdn-xover-low-hz`, `fdn-xover-high-hz`
- Geometry-based early-reflection pre-stage:
  - `--er-geometry`
  - `--er-room-dims-m`, `--er-source-pos-m`, `--er-listener-pos-m`
  - `--er-absorption`, `--er-material`
- Auto-fit heuristic profiles:
  - `--auto-fit none|speech|music|drums|ambient`
- Multichannel shimmer decorrelation controls:
  - `--shimmer-spatial`
  - `--shimmer-spread-cents`
  - `--shimmer-decorrelation-ms`
- Expanded regression coverage for new render/dereverb pathways and CLI
  validation for `w64` output extension support.
- Explicit unsafe oscillation controls for algorithmic renders:
  - `--unsafe-self-oscillate`
  - `--unsafe-loop-gain`
  including analysis metadata tagging via `compute_backend=...-unsafeosc`.

### Changed
- Version metadata, release docs, and roadmap updated to `v0.7.5`.
- README CLI quick-reference expanded with new render and dereverb options.
- Roadmap restructured around completed `v0.7.5` feature pack and remaining
  `0.7.x` priorities.

### Fixed
- Corrected `packaging/homebrew/verbx.rb` SHA256 to match the published
  `v0.7.4` GitHub source tarball.

## [0.7.4] - 2026-03-27

### Added
- Autogenerated CLI help reference at `docs/CLI_REFERENCE.md` via
  `scripts/generate_cli_reference.py`.
- Shared schema/version constants module at `src/verbx/core/schema_versions.py`
  to eliminate duplicated payload-version literals.
- Launch-example parity checker (`scripts/check_launch_examples.py`) plus
  canonical command source (`docs/LAUNCH_EXAMPLES_CANONICAL.txt`) and CI
  enforcement.
- SOFA interoperability feasibility note (`docs/SOFA_FEASIBILITY.md`) capturing
  recommended `0.7.x` import/extract scope and constraints.
- SOFA MVP command surface:
  - `verbx ir sofa-info`
  - `verbx ir sofa-extract`
  for deterministic FIR-matrix extraction into existing convolution workflows.
- Optional dependency extra `verbx[sofa]` (h5py-backed SOFA parsing path).
- 192 kHz resample-path benchmark scenario (`conv_target_sr_192k`) in
  `scripts/benchmark_render_baseline.py` and CI baseline docs.

### Changed
- CI now checks CLI reference freshness (`scripts/generate_cli_reference.py --check`).
- CI now checks launch-example parity (`scripts/check_launch_examples.py --check`).
- `verbx render` now supports `--target-sr` for integrated sample-rate
  conversion during render (for example, rendering directly to 192 kHz output
  without pre-resampling source audio).
- Documentation sweep updated README, schema reference, AI augmentation guide,
  cookbook, and man pages with `--target-sr` usage and high-resolution float
  render examples.
- Release workflow now enforces explicit publish/sync policy gates:
  - `RELEASE_REQUIRE_PYPI` (default `true`)
  - `RELEASE_REQUIRE_HOMEBREW` (default `true`)
- GitHub Release creation now runs only after build/publish/sync jobs complete.
- Render RT60 default is now sourced from a shared constant
  (`RT60_DEFAULT_SECONDS`) across config and CLI.
- Schema/docs defaults were aligned with runtime behavior:
  `rt60=60.0`, `pre_delay_ms=20.0`, `shimmer_feedback=0.35`, and expanded
  `fdn_matrix` option set in `docs/SCHEMA_REFERENCE.md`.
- Convolution route validation now hard-fails ambiguous large-bus auto-mapping
  cases (`16.0`/`64.4` with mono/channel-matched IR) unless `--ir-route-map`
  is explicitly set.
- Convolution route aliases for `16.0` and `64.4` were expanded and
  backward-compatible legacy aliases were retained.
- Render CLI pre-validation now mirrors large-bus route-map ambiguity checks for
  earlier actionable failure messages.
- Man pages now advertise `verbx 0.7.4` metadata consistently and use
  research-grade positioning text.
- README DSP sections now include explicit variable definitions/notation for FDN
  and partitioned-convolution equations.

## [0.7.3] - 2026-03-26

### Added
- Stable Python API surface at `verbx.api`
  (`render_file`, `generate_ir`, `analyze_file`, `read_audio`, `write_audio`).
- Package-level API re-exports via `verbx.__init__` for library workflows.
- Immersive layout label support for large buses (`16.0` and `64.4`).
- Structured schema reference documentation in `docs/SCHEMA_REFERENCE.md`.
- Research notebook example for dataset workflows:
  `examples/dataset_augmentation.ipynb`.
- Public Homebrew tap repository published at `TheColby/homebrew-verbx`.

### Changed
- Bumped package/version metadata and release docs to `v0.7.3`.
- Homebrew maintainer guidance now documents excluded fragile build dependencies
  (`numba`, `llvmlite`, `scikit-learn`) and required runtime deps (`numpy`,
  `scipy`).
- Packaging metadata now consistently describes verbx as research-grade.

## [0.7.2] - 2026-03-17

### Added
- Command-level status bars now cover render execution paths, including standard
  single renders and `--lucky` render batches.
- `quickstart --smoke-test` and `doctor --render-smoke-test` now display
  explicit status bars while processing.

### Changed
- Status reporting is now consistent across processing-oriented CLI commands,
  making long-running tasks visibly active unless quiet/silent/progress-disabled
  modes are selected.

## [0.7.1] - 2026-03-16

### Added
- Cross-platform install verification workflow
  (`.github/workflows/install-verify.yml`) that checks `uv`, `pip+venv`, and
  `scripts/install.sh` + man-page rendering on macOS and Linux.
- Public-alpha support/limitations notes in `docs/PUBLIC_ALPHA_NOTES.md`.
- Launch-week pinned demo command/hash matrix in
  `docs/LAUNCH_WEEK_DEMO_PINS.md`.
- Long-tail regression coverage for algorithmic renders with `RT60 > 120s`.
- Deterministic golden tests for feature-vector target-source lane behavior.
- Benchmark reporting harness (`scripts/benchmark_render_baseline.py`) plus
  baseline docs in `docs/benchmarks/`.

### Changed
- Raised RT60 upper bounds to 3600 seconds across control-target specs, runtime
  automation clamps, and CLI validation for render/IR workflows.
- Updated release docs and roadmap status for current public-alpha readiness.
- Normalized version labeling to semantic `0.7.x` notation across package/docs.
- Upgraded GitHub Actions workflows to current Node24-compatible action majors.
- Hardened release publish flow with explicit auth mode resolution:
  token, trusted publishing, or documented skip.
- Improved multichannel routing error diagnostics with explicit channel math and
  route/layout remediation hints.
- Added CI `perf-baseline` job that publishes benchmark JSON artifacts for
  comparison across runs.

### Fixed
- `batch augment` now pre-creates output/analysis directories before parallel
  rendering, so runs succeed even when `--copy-dry` is not enabled.
- `scripts/install.sh` now handles empty pip-argument arrays correctly under
  strict shell settings (notably Bash 3.2 on macOS runners).

## [0.7.0] - 2026-03-14

_Public alpha release._

### Added
- Deterministic time-varying automation lanes (`--automation-file`,
  `--automation-point`) with block/sample modes, smoothing, clamps, and trace
  export.
- Feature-vector control lanes (`--feature-vector-lane`) with weighted, curved,
  hysteretic mapping and deterministic signatures.
- External feature-guide ingest (`--feature-guide`) with deterministic alignment
  policy (`align`/`strict`) and guide-alignment metadata.
- Composable target-source feature mapping graphs (`source=target:<target>`) with
  acyclic topological evaluation and deterministic mapping signatures.
- Track C controls for algorithmic FDN behavior, including
  `fdn-rt60-tilt` and `fdn-tonal-correction-strength`.
- Track D IR morph workflows, cache-backed blending, and render-time IR blend
  control (`ir-blend-alpha`).
- Immersive interoperability workflows (`immersive handoff`, `immersive qc`,
  distributed queue worker/status/template commands).
- `verbx version` command for explicit CLI/package version reporting.
- Release-readiness project docs: `CONTRIBUTING.md`, `SECURITY.md`, and
  `CODE_OF_CONDUCT.md`.
- GitHub Actions CI (lint, typecheck, test) and tag-based release workflow with
  optional PyPI publish.

### Changed
- Internal processing paths standardized on `float64` for render, automation,
  and analysis math.
- Render summary now surfaces output audio feature/statistics by default unless
  reduced verbosity/quiet modes are selected.
- Installer and man-page coverage expanded and documented.
- Version scheme normalized to semantic minor (`0.7.x`).

### Fixed
- Deterministic control-plane and lane-validation diagnostics strengthened across
  automation and feature-lane workflows.

## [0.6.0] - 2026-03-08

### Added
- Advanced FDN structures and controls, including topology expansion and
  Ambisonics-focused spatial workflows.
- Additional CLI/documentation coverage for production-oriented reverb design.

## [0.5.0] - 2026-03-02

### Added
- v0.5 feature set covering expanded render workflows, testing growth, and
  installer/man-page foundations.

## [0.4.0] - 2026-02-28

### Added
- Control and modulation expansion, stronger batch semantics, and operational
  reliability improvements.

## [0.3.0] - 2026-02-27

### Added
- IR generation/analysis/processing command group and deterministic IR cache
  workflows.

## [0.2.0] - 2026-02-26

### Added
- Broader convolution and algorithmic tuning controls, plus expanded analysis
  utilities.

## [0.1.0] - 2026-02-24

### Added
- Initial dual-engine reverberation CLI (`render`, `analyze`, `suggest`,
  `presets`) with typed config and deterministic analysis reporting.
