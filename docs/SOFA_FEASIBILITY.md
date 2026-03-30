# SOFA Import Feasibility (v0.7.x Evaluation)

_Date: March 27, 2026_

## Status

`v0.7.6` ships a narrow MVP:

- `verbx ir sofa-info FILE.sofa`
- `verbx ir sofa-extract FILE.sofa OUT.wav`

This implements the recommended first step from this feasibility analysis:
deterministic extraction of SOFA FIR data into explicit WAV matrices for the
existing convolution engine.

## Summary

SOFA import is technically feasible in `verbx`, but should land as a scoped
interoperability feature in a future `0.7.x` patch (not as a broad, implicit
"load any SOFA and hope" path).

Recommended first scope:

- Support **FIR-based SimpleFreeFieldHRIR / SingleRoomMIMO** style datasets.
- Convert selected SOFA views into explicit `M x N` IR matrices compatible with
  existing `--ir-matrix-layout` and `--ir-route-map` behavior.
- Preserve deterministic render metadata (source file hash, selected listener
  position, emitter index, and conversion options) in analysis JSON.

## Why It Fits `verbx`

- Convolution engine already supports general matrix routing and long IR tails.
- Internal DSP is `float64`, which is appropriate for high-order FIR datasets.
- Existing schema/provenance infrastructure can capture deterministic SOFA
  conversion context.

## Constraints and Risks

1. **Convention diversity:** SOFA is a container + convention family, not one
   single structure. Different datasets expose different required fields.
2. **Coordinate interpretation:** listener/source orientation and units can
   drift across datasets; a silent mismatch gives valid but wrong routing.
3. **Channel explosion:** HOA or dense directional sets can exceed practical
   realtime/offline budgets if mapped naively.
4. **Resampling policy:** sample-rate mismatch must be deterministic and
   explicit (`strict` vs `coerce` style, mirroring IR blend policies).
5. **Dependency footprint:** adding HDF5 stack must remain optional and
   isolated from baseline install paths.

## Proposed Integration Shape

### CLI Surface (future)

- `verbx ir sofa-info FILE.sofa`
  - Prints convention, sample rate, dimensions, emitter/listener counts,
    available positions, and FIR length.
- `verbx ir sofa-extract FILE.sofa OUT.wav [options]`
  - Exports selected FIR set to an explicit WAV matrix for reuse with existing
    convolution render flow.
  - Options:
    - `--listener-index`, `--emitter-index`, `--azimuth`, `--elevation`
    - `--sample-rate` (optional deterministic resample)
    - `--layout-map` (maps output channels to known layout tokens)
    - `--strict` (fail on unsupported convention/metadata mismatch)

### Data Model

- Keep SOFA parsing and extraction in `verbx.ir` domain.
- Treat exported WAV matrix as canonical render input for `verbx render`.
- Store extraction metadata sidecar JSON for reproducibility.

## Dependency Strategy

- Optional extra only (example): `pip install "verbx[sofa]"`.
- Preferred lightweight read path:
  - `h5py` for container access
  - small in-repo convention validator for supported conventions
- No hard dependency on SOFA stack for default install.

## Validation Plan (future patch)

1. Golden fixture tests using tiny SOFA samples:
   - convention detection
   - index/position selection
   - deterministic WAV export hash
2. Route parity tests:
   - extracted matrix renders match expected channel counts/layouts
3. Failure-path tests:
   - unsupported convention
   - missing required variables
   - strict mismatch behavior

## Recommendation

Proceed with a narrow SOFA MVP focused on **deterministic extraction to explicit
IR matrices**, then expand convention coverage incrementally. This matches
`verbx` architecture and avoids destabilizing the current public-alpha surface.

## References

- SOFA project: <https://www.sofaconventions.org/>
- AES69 (SOFA standard family): <https://www.aes.org/publications/standards/search.cfm?docID=99>
