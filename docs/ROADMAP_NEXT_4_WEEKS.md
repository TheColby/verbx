# verbx Next 4 Weeks

_Execution checklist derived from the active roadmap in `README.md`,
`CHANGELOG.md`, and `docs/ROADMAP.md`._

_Last updated: 2026-04-16._

---

## Goal

Use this document as the short-horizon execution checklist for the current
`0.7.x` stabilization phase, the initial `v0.8` native parity track, and one
bounded research prototype.

---

## Priority Order

- [ ] 1. Stabilize Python `0.7.x` render and realtime behavior.
- [ ] 2. Finish CLI, docs, and test consolidation.
- [ ] 3. Push native render parity on a narrow deterministic slice.

## Active Focus

Use these as the working top-line priorities for the current cycle:

- [ ] Stabilize Python `0.7.x` render and realtime behavior.
- [ ] Finish CLI, docs, and test consolidation.
- [ ] Push native render parity on a narrow, deterministic feature slice.

---

## Track 1: Shipping Work

### Week 1

- [x] Continue shrinking `src/verbx/cli.py` by extracting implementation-heavy command handlers into `src/verbx/commands/`.
- [x] Add focused regression coverage for the extracted IR command registry path.
- [x] Re-sync generated command documentation:
  - [x] `docs/CLI_REFERENCE.md`
  - [x] `docs/USERGUIDE.md`
  - [x] `USERGUIDE.pdf`
- [ ] Identify the next safe extraction slice after IR (`realtime`, shared validators, or remaining helper clusters).
- [ ] Add a short “command-module split status” note to `CHANGELOG.md` when the next slice lands.

### Week 2

- [ ] Harden realtime UX:
  - [ ] clearer startup summaries
  - [ ] better device/input-output error reporting
  - [ ] safer defaults for live dereverb and freeze/infinite-style reverb
- [ ] Add consistent machine-readable analysis/report outputs anywhere render, realtime, and dereverb still diverge.
- [ ] Close the remaining “looks hung” and “surprising default” render flows with fail-fast validation or clearer status output.

### Week 3

- [ ] Finalize limiter and output-delivery ergonomics:
  - [ ] limiter behavior sanity pass
  - [ ] peak and ceiling handling review
  - [ ] output subtype/container validation review
  - [ ] long-render safeguard review
- [ ] Expand presets so new room-model, dereverb, and limiter features are represented in ready-to-run examples.
- [ ] Run a release-readiness docs/examples pass across `README.md`, launch examples, and generated guides.

### Week 4

- [ ] Freeze the Python CLI surface enough to support native parity work cleanly.
- [ ] Prioritize bug fixes over new flag growth for the release window.
- [ ] Cut a focused `0.7.x` stabilization release.

---

## Track 2: Native Parity Work

### Week 1

- [ ] Define the minimum `verbx-c` parity target for offline render.
- [ ] Write down the parity contract for:
  - [ ] input/output formats
  - [ ] tail handling
  - [ ] core algorithmic controls
- [ ] Add a small parity matrix in tests or fixtures so native work has a concrete target.

### Week 2

- [ ] Expand native render flags for the most-used offline slice first:
  - [ ] `rt60`
  - [ ] `wet`
  - [ ] `dry`
  - [ ] subtype/format selection
  - [ ] tail metrics
  - [ ] peak-safe output
- [ ] Add golden-file or metric-based Python/native comparisons for a deterministic fixture set.

### Week 3

- [ ] Add analysis/report parity where feasible so native runs can emit a comparable JSON support bundle.
- [ ] Improve native doctor/build ergonomics and packaging scripts.

### Week 4

- [ ] Decide the `v0.8` release shape:
  - [ ] native render only
  - [ ] native render + doctor + analysis
  - [ ] hybrid wrapper phase before full replacement
- [ ] Document the chosen `v0.8` parity scope in `README.md` and `docs/ROADMAP.md`.

---

## Track 3: Research Bets

### Week 1

- [ ] Pick exactly one topology-expansion prototype for this cycle.
- [ ] Prefer next-generation FDN topology work over starting both geometry and neural branches at once.
- [ ] Write down success criteria for the chosen prototype before implementation starts.

### Week 2

- [ ] Build one prototype behind an explicit experimental flag.
- [ ] Candidate directions:
  - [ ] denser delay-feedback matrix variants
  - [ ] geometry-to-FDN parameter derivation improvements
  - [ ] intelligibility-aware dereverb scoring

### Week 3

- [ ] Build evaluation harnesses before broadening scope:
  - [ ] render benchmarks
  - [ ] repeatable presets
  - [ ] objective metrics
  - [ ] example corpus

### Week 4

- [ ] Decide whether the prototype:
  - [ ] graduates into `0.7.x`
  - [ ] waits for `0.8`
  - [ ] remains experimental only

---

## Recommended Sequence

- [ ] Stabilize Python `0.7.x` render and realtime behavior.
- [ ] Finish CLI, docs, and test consolidation.
- [ ] Push native render parity on a narrow, deterministic feature slice.
- [ ] Defer broader research expansion until the three priorities above are stable.

---

## Exit Criteria

### Shipping

- [ ] `src/verbx/cli.py` is meaningfully smaller and mostly orchestration plus shared compatibility helpers.
- [ ] Generated docs are in sync with the shipped CLI surface.
- [ ] New realtime, dereverb, limiter, and long-tail behaviors have focused regression coverage.

### Native

- [ ] Native render parity is defined against a stable fixture set.
- [ ] Python/native comparison runs produce repeatable results for the chosen slice.

### Research

- [ ] Prototype is behind a flag.
- [ ] Prototype has an evaluation harness.
- [ ] Prototype has a go/no-go decision by the end of Week 4.
