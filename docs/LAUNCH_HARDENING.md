# Launch Hardening: Top 5 Likely Public Complaints

This document captures the five most likely announcement-day complaints and the concrete mitigations now implemented in `verbx`.

## 1) "I installed it, but I still don't know if it actually works"

### Complaint pattern
- Users can run `verbx --help`, but still don't trust that rendering works end-to-end.

### Mitigation
- `verbx quickstart --smoke-test`
- `verbx quickstart --smoke-test --strict`
- `verbx quickstart --smoke-test --json-out launch_smoke.json`

### What it does
- Generates a tiny synthetic test input.
- Runs a real algorithmic render.
- Reports pass/fail, frame counts, and output path.

---

## 2) "Doctor says my environment is fine, but rendering still fails"

### Complaint pattern
- Environment checks can pass while pipeline execution still fails due to runtime/runtime-IO edges.

### Mitigation
- `verbx doctor --render-smoke-test`
- `verbx doctor --render-smoke-test --strict`
- `verbx doctor --render-smoke-test --smoke-out-dir out/doctor_smoke`

### What it does
- Runs diagnostics and a real smoke render in one command.
- Supports strict non-zero exit for CI gates.

---

## 3) "When render fails, I can't share a useful bug report"

### Complaint pattern
- Users see an error in terminal but have no structured failure artifact to attach.

### Mitigation
- `verbx render INFILE OUTFILE --failure-report-out failure.json`

### What it does
- On execution failure, writes a structured JSON report with:
  - error type/message
  - resolved input/output paths
  - full render config snapshot
  - runtime diagnostics (platform/deps/device resolution)

---

## 4) "Dry-run helps, but I still can't budget disk/compute before launch"

### Complaint pattern
- Users need rough size planning for long tails and large batches.

### Mitigation
- `verbx render INFILE OUTFILE --dry-run`

### What it does
- Dry-run output now includes:
  - `estimated_output_duration_s`
  - `estimated_output_size_mb`

This gives practical preflight planning data before writing audio.

---

## 5) "I need reproducibility artifacts for demos, audits, and issue triage"

### Complaint pattern
- Teams need deterministic support artifacts for launch demos and incident reports.

### Mitigation
- `verbx render INFILE OUTFILE --repro-bundle`
- `verbx render INFILE OUTFILE --repro-bundle-out out/repro.json`

### What it does
- Writes a reproducibility bundle with:
  - input/output signatures and hashes
  - effective render configuration snapshot
  - deterministic run signature

---

## Recommended launch-day preflight

```bash
# 1) Verify environment + real smoke render
verbx doctor --render-smoke-test --strict --json-out doctor_launch.json

# 2) Validate expensive render settings before runtime
verbx render in.wav out.wav --engine algo --rt60 120 --dry-run

# 3) Run final render with reproducibility + failure capture artifacts
verbx render in.wav out.wav \
  --engine algo --rt60 120 \
  --repro-bundle-out out/repro.json \
  --failure-report-out out/failure.json
```
