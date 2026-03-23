# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]
### Added
- Compatibility matrix doc for install/runtime channels (`docs/COMPATIBILITY_MATRIX.md`).
- Stable Python API wrappers in `verbx.api` (`render_file`, `generate_ir`, `analyze_file`).
- Schema contract docs for batch manifests and automation payloads (`docs/SCHEMAS.md`).

### Changed
- `batch corpus-generate` now reports throughput and retry metrics in summary output
  (`elapsed_seconds`, `outputs_per_second`, `total_attempts`, retry stats).
- `batch corpus-generate` adds `--retries` for per-variant retry ergonomics.
- Install verification CI now exercises npm launcher onboarding flow.
- `quickstart` includes npm onboarding + readiness command path.
- Immersive QC now reports an explicit `layout_channels` gate and expected channel count.
- Added docs-sync tests to keep key README command flags aligned with CLI help text.

## [0.7.3] - 2026-03-23

### Added
- Homebrew formula artifact in-repo at `packaging/homebrew/verbx.rb`.
- Homebrew maintainer guide (`docs/HOMEBREW.md`) and refresh helper script
  (`scripts/refresh_homebrew_formula.sh`).
- Optional release automation to sync formula updates to tap repo
  (`sync-homebrew-tap` job in `.github/workflows/release.yml`).

### Changed
- README installation section now includes Homebrew install paths.
- npm launcher now uses platform-native `PYTHONPATH` delimiter handling for
  cross-platform execution.
- `quickstart` now includes an npm installation + readiness verification
  workflow (`quickstart --verify --strict` and `doctor --render-smoke-test`).

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
