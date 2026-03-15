# Public Alpha Notes (v0.7.0)

_Updated: 2026-03-15_

`verbx` v0.7.0 is a public alpha focused on robust offline rendering,
reproducibility, and production-oriented DSP control.

## Scope

- Dual-engine render path (`algo`, `conv`, `auto`) with deterministic control
  plane behavior.
- Track C/D automation and IR morph workflows.
- Immersive handoff/QC tooling and AI augmentation workflows.

## Known Limitations

- Offline-first architecture: no real-time plugin host in `0.7.x`.
- Very long tails are intentionally compute-heavy; render time depends strongly
  on hardware and block/partition settings.
- GPU acceleration is most effective for convolution-heavy workloads.
- FOA (`--ambi-order 1`) is the validated Ambisonics path for this alpha.
- Cross-platform floating-point and library differences can introduce tiny
  low-level output deltas for long renders; use reproducibility bundles when
  comparing runs.

## Support Channels

- Bugs/regressions: [GitHub Issues](https://github.com/TheColby/verbx/issues)
- Security reports: [GitHub Security Advisory submission](https://github.com/TheColby/verbx/security/advisories/new)
- Release artifacts and announcements: [GitHub Releases](https://github.com/TheColby/verbx/releases)

## What To Include In Bug Reports

- Exact CLI command and full stderr/stdout text.
- `verbx doctor --json-out doctor.json` output.
- `--repro-bundle` output from the failing render.
- OS, Python version, and commit/tag.
