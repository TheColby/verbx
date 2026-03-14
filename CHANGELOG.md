# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- `verbx version` command for explicit CLI/package version reporting.
- Release-readiness project docs: `CONTRIBUTING.md`, `SECURITY.md`, and
  `CODE_OF_CONDUCT.md`.
- GitHub release workflow for tag-based artifact build, validation, and optional
  PyPI publish.

### Changed
- Launch metadata aligned to `v0.7.0` across package metadata and man pages.
- Roadmap section renamed to clarify post-launch `v0.7.x` scope.

## [0.7.0] - 2026-03-13

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

### Changed
- Internal processing paths standardized on `float64` for render, automation,
  and analysis math.
- Render summary now surfaces output audio feature/statistics by default unless
  reduced verbosity/quiet modes are selected.
- Installer and man-page coverage expanded and documented.

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
