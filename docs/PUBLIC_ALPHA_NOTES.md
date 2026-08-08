# Public Alpha Notes (v0.9.9)

_Updated: 2026-08-05_

`verbx` v0.9.9 is a public alpha focused on robust offline rendering,
reproducibility, advanced DSP control, structural cleanup, and initial
realtime duplex auditioning.

## Scope

The supported alpha surface is summarized below. Each item describes shipped behavior, not a promise of parity across every command, host, or spatial format.

- Dual-engine render path (`algo`, `conv`, `auto`) with deterministic control
  plane behavior.
- Track C/D automation and IR morph workflows.
- Immersive handoff/QC tooling and AI augmentation workflows.
- Initial realtime duplex monitoring (`verbx realtime`) with selectable input
  and output devices, using direct convolution or an algorithmic proxy IR.

## Known Limitations

The limits below define where testing and support remain narrower than the complete design vocabulary described elsewhere in the book.

- Realtime support is intentionally narrow: `verbx realtime` is for live
  auditioning, not for full offline feature parity or plugin-host workflows.
- Very long tails are intentionally compute-heavy; render time depends strongly
  on hardware and block/partition settings.
- GPU acceleration is most effective for convolution-heavy workloads.
- FOA (`--ambi-order 1`) is the validated Ambisonics path for this alpha.
- Cross-platform floating-point and library differences can introduce tiny
  low-level output deltas for long renders; use reproducibility bundles when
  comparing runs.
- Realtime device support depends on the optional `sounddevice` backend and the
  host audio stack. Confirm device visibility with `verbx realtime --list-devices`.

## Support Channels

Use the channel that matches the report so release discussion, security information, and ordinary defects do not become mixed together.

- Bugs/regressions: [GitHub Issues](https://github.com/TheColby/verbx/issues)
- Security reports: [GitHub Security Advisory submission](https://github.com/TheColby/verbx/security/advisories/new)
- Release artifacts and announcements: [GitHub Releases](https://github.com/TheColby/verbx/releases)

## What To Include In Bug Reports

A useful report should let another person reproduce the failure without guessing at the environment or processing state. Include the following evidence when it is available and lawful to share.

- Exact CLI command and full stderr/stdout text.
- `verbx doctor --json-out doctor.json` output.
- `--repro-bundle` output from the failing render.
- For realtime issues: selected `--input-device`, `--output-device`, sample
  rate, block size, and whether `--engine algo` or `--engine conv` was used.
- OS, Python version, and commit/tag.
