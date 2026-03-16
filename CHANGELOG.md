# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]
### Added
_No changes yet._

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
