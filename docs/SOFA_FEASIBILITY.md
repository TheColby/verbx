# SOFA Import Feasibility (0.7.x)

_Updated: 2026-03-23_

## Verdict

**Feasible in a phased rollout.**

SOFA import is technically compatible with `verbx` convolution routing but should
be staged behind strict validation and explicit scope boundaries in alpha.

## Why it is feasible

- Convolution path already supports explicit symbolic layouts and matrix routing.
- Existing IR ingestion + channel-aware processing can host SOFA-derived impulse data.
- Current QA/reporting stack can surface validation issues during import/convert.

## Risks to control

- SOFA files vary widely in coordinate conventions and metadata completeness.
- Large HRTF datasets can exceed practical memory/CPU budgets for naïve conversion.
- Ambiguous listener/source geometry can lead to perceptual errors if silently coerced.

## Proposed phases

1. **Phase A (read-only inspection)**
   - Parse and validate core SOFA metadata.
   - Emit conversion report without rendering.
2. **Phase B (offline conversion)**
   - Convert constrained SOFA subsets to explicit IR matrix artifacts.
   - Require explicit output layout selection and sample-rate policy.
3. **Phase C (production defaults)**
   - Promote stable subset to first-class docs/examples.
   - Add CI regression fixtures for supported SOFA variants.

## Out of scope for 0.7.x

- Real-time SOFA convolution plugin hosting.
- Automatic perceptual optimization across arbitrary SOFA corpora.
