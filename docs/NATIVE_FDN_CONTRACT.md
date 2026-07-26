# Native Higher-Order FDN Contract

_Status: implementation contract for the next bounded `verbx-c` DSP slice._

## Scope

The first native FDN slice replaces the four parallel combs in
`native/verbx_c/src/algo_reverb.c` for the `fdn` model only. Spring and plate
retain their current tunings until separate parity fixtures exist. The slice
does not claim automation, modulation, multiband, shimmer, freeze, or
multichannel parity.

## Fixed State

The realtime state is allocated once during initialization:

- eight delay buffers with base lengths `31, 37, 41, 43, 47, 53, 59, 67 ms`
- one write index per delay line
- one damping-filter state per delay line
- one DC-block input/output state pair per delay line
- one pre-delay buffer and index
- fixed eight-element read, filtered, feedback, and output scratch vectors

Processing performs no allocation, locking, file access, or coefficient
construction.

## Coefficients

For delay length \(d_i\) samples, sample rate \(f_s\), and requested decay
\(T_{60}\), the line feedback gain is:

\[
g_i = \min\left(0.995,\ 10^{-3(d_i/f_s)/T_{60}}\right)
\]

The first slice uses the normalized order-eight Hadamard matrix
\(H_8 / \sqrt{8}\). The matrix is orthonormal, so decay is controlled by the
per-line gains rather than an unbounded matrix norm. Damping follows the Python
reference convention: a larger user damping value produces stronger
high-frequency attenuation in the feedback loop.

## Sample Step

For each input sample:

1. Advance the pre-delay.
2. Read the eight delay outputs.
3. Apply the per-line damping filter and DC blocker.
4. Multiply the filtered vector by the normalized Hadamard matrix.
5. Write the pre-delayed input injection plus `g_i * feedback_i` to each line.
6. Produce wet output from a normalized signed sum of the line outputs.
7. Apply the existing dry/wet mix and tail finalization.

Stereo uses independent state with the existing deterministic right-channel
delay scale. Reset clears every buffer, index, and filter state.

## Acceptance Gates

- Existing native build, doctor, WAV, tail-floor, and JSON-report tests pass.
- AddressSanitizer and UndefinedBehaviorSanitizer report no failures.
- Impulse output is finite and deterministic after reset.
- The wet path allocates zero bytes after initialization.
- `scripts/compare_native_render_parity.py --strict-structural` passes.
- Metric deltas improve or are explicitly re-baselined with listening and
  measurement evidence; structural success alone is not DSP parity.

## Pre-Port Baseline

Measured on 2026-07-26 with the checked-in parity fixture:

| Scenario | Duration delta | Peak delta | RMS delta |
|---|---:|---:|---:|
| `mono_impulse_short_tail` | 27.19 ms | 0.18286 | 26.58 dB |
| `stereo_impulse_mixed_tail` | 89.89 ms | 0.67609 | 0.12 dB |

The structural contract passed for sample rate, channel count, finite samples,
and exact-zero tail completion. The aggregate metric contract did not pass, as
expected for the foundational Schroeder/Moorer implementation.

## First-Slice Result

The bounded eight-line slice landed on 2026-07-26. Native C tests and strict
structural parity pass. The aggregate metric contract remains intentionally
open: mono RMS delta improved from 26.58 dB to 13.06 dB, while duration and
stereo RMS/peak deltas show that Python-reference diffusion, output scaling,
and tail behavior still need dedicated parity work.
