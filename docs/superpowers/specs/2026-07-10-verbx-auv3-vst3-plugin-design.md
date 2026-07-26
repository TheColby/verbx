# VERBX AUv3/VST3 Plug-in Design

Date: 2026-07-10
Status: Approved design direction

## Summary

VERBX will become a full-screen spatial reverb plug-in for AU/AUv3 and VST3 hosts. The approved product direction is a spatial architecture instrument that stays close to the existing dark high-fidelity GUI mockup: geometry and room behavior are the signature, while the front panel still feels like a serious studio processor with loudness, imaging, spectrum, macro controls, expert tabs, and clear realtime status.

The first implementation path is a JUCE-style native plug-in shell backed by a shared realtime DSP core extracted from the existing native work. The initial plug-in slice should be deterministic and low-risk: realtime algorithmic reverb with spatial/geometry controls, stable host parameters, and clear status/error reporting. Heavy DXF/CAD ray tracing and generated IR workflows are designed as offline/precompute features that feed safe realtime assets into the plug-in.

## Goals

- Ship a credible AU/AUv3 and VST3 direction without attempting full Python CLI parity in the first slice.
- Make the full-screen editor feel like a flagship spatial reverb instrument, not a generic utility panel.
- Keep the realtime audio callback deterministic, bounded, and free of file I/O, allocation-heavy work, and geometry parsing.
- Default to high fidelity: host-rate I/O with internal processing targeting 192 kHz and 32-bit float.
- Expose a stable host automation surface with performance-first controls and deeper expert controls.
- Make reverb times playable across extreme ranges with RT60 coarse/fine controls and logarithmic mapping from 0.01 seconds to 360 seconds.

## Non-Goals For The First Slice

- Full Python render/realtime/dereverb/convolution parity.
- Realtime DXF parsing or live ray tracing inside the audio callback.
- Exposing every CLI option as a primary front-panel control.
- Replacing the CLI; the plug-in should share native primitives where practical, not absorb all offline workflows.
- Pro Tools/AAX support. This can be evaluated later after VST3/AU foundations are stable.

## Product Direction

The approved UI direction is "Spatial Architecture Instrument, closer to the GUI mockup, occupying an entire screen." The center of the editor is the Spatial Decay Theater: a large room/geometry view with source/listener placement, ray/early-reflection visualization, room shell, and decay behavior. Around it, the plug-in keeps the mockup's studio-console anatomy:

- Top bar: VERBX wordmark, preset browser, A/B, engine mode, bypass/live state.
- Left panel: loudness, true peak, safety limiter, ducking status, meter bank.
- Center: full-screen spatial theater plus primary macro controls.
- Right panels: goniometer/correlation and space/material summary.
- Wide spectrum/decay strip: live decay spectrum, EDR-style view, CPU and latency status.
- Expert tabs: FDN and Diffusion, Shimmer and Color, Dynamics and Tone, Spatial and Geometry.
- Footer: host rate, internal quality target, block size, latency, CPU, version, and warnings.
- Mode buttons: Freeze/Infinite and Reverse Reverb should be visible performance controls, not buried in a menu.

The front panel should feel performance-simple. The host automation surface can be deeper, but the editor should not overwhelm users with hundreds of equal-weight controls.

## Audio Architecture

The recommended build path is a JUCE-style native plug-in shell plus a shared native DSP core.

### Host Shell

The plug-in shell owns:

- AU/AUv3 and VST3 lifecycle.
- Bus layout negotiation.
- Parameter registration and host automation names.
- Preset state serialization.
- Editor lifecycle.
- Latency reporting.
- Host-rate audio I/O.

The shell must not own core DSP algorithms beyond adaptation and orchestration.

### Engine Adapter

The adapter connects host blocks to the realtime DSP core:

- Accept host-rate input buffers.
- Apply fixed or bounded block adaptation.
- Smooth parameter changes.
- Map normalized host parameters to DSP values.
- Oversample toward the internal quality target when needed.
- Report effective latency.
- Push lock-free telemetry snapshots to the editor.

### DSP Core

The realtime DSP core should start with the narrow deterministic feature slice:

- Algorithmic reverb using FDN/diffuser primitives.
- Pre-delay, size, RT60, damping, diffusion, width, wet, dry.
- Matrix selection and line count where stable.
- Modulation rate/depth where realtime safe.
- Ducking and limiter controls where already deterministic.
- Freeze/infinite reverb as a dedicated mode.
- Reverse reverb as a dedicated mode button with bounded lookahead/buffer behavior.
- Spatial blend controls that influence early/late behavior without requiring live ray tracing.

The DSP core must be independent enough to test outside a DAW host.

## Sample Rate And Precision

The plug-in cannot force the DAW project sample rate. Host I/O stays at the DAW's current rate. VERBX defaults to:

- Internal quality target: 192 kHz.
- Processing precision: 32-bit float.
- Oversampling: enabled as needed to approach the 192 kHz internal target.

The quality control should expose:

- Host
- 2x
- 4x
- Target 192 kHz

Target 192 kHz is enabled by default. If the CPU cost is too high, the plug-in should display an explicit status message and allow the user to lower quality rather than silently changing behavior.

Optional double-precision processing can be added later for selected internal accumulators or offline render paths, but it is not required for the first plug-in slice.

## RT60 Control Model

RT60 uses paired controls:

- `rt60_coarse`: normalized host parameter mapped logarithmically from 0.01 seconds to 360 seconds.
- `rt60_fine`: bipolar normalized host parameter centered at zero, applied as a log-space trim around the coarse value.
- Effective RT60: the final clamped value shown in seconds.

The proposed fine range is plus or minus 20 percent around the coarse value. This keeps fine adjustment useful at short, medium, and very long decay times.

Freeze or infinite reverb is a separate mode parameter. It must not be represented as "RT60 at maximum."

Reverse reverb is also a separate mode parameter. It should be exposed as a visible button near Freeze/Infinite and should not change the meaning of RT60. The implementation must make its added latency explicit because reverse tails require buffering or pre-rendered/reordered tail segments.

The mapping should be deterministic and testable. A suitable form is:

```text
coarse_seconds = exp(lerp(log(0.01), log(360.0), normalized_coarse))
fine_ratio = exp(log(1.20) * bipolar_fine)
effective_rt60 = clamp(coarse_seconds * fine_ratio, 0.01, 360.0)
```

## Parameter Surface

The plug-in should avoid exposing all current CLI-era controls as first-class front-panel controls. Use three layers:

- Performance layer: roughly 24 to 32 primary controls, including the main macro row, quality, freeze, limiter/duck status, and spatial blend controls.
- Expert layer: roughly 60 to 80 controls organized into tabs for FDN, color, dynamics, tone, and geometry.
- Exhaustive/automation layer: later advanced view for stable additional parameters once their realtime semantics are mature.

The initial main-page macro set is:

- Pre-delay
- Room size
- RT60 coarse
- RT60 fine
- Damping
- Width
- Diffusion
- Wet
- Dry
- Reverse mode button
- Freeze/Infinite mode button

RT60 coarse may be visually dominant. RT60 fine can be a smaller companion knob or trim ring near the coarse control.

## Geometry And DXF Strategy

Geometry is core to the product identity, but heavy geometry work should not run in the realtime callback.

The first spatial implementation should include:

- Room dimensions.
- Material profile.
- Source/listener placement.
- Layout target such as stereo, surround, or 7.2.4.
- Ray/geometry blend parameter.
- Imported profile metadata display where available.

DXF/CAD support should be treated as an offline/precompute pipeline:

- Import DXF outside the audio callback.
- Validate geometry.
- Generate a compact spatial profile, early-reflection model, or IR asset.
- Store deterministic hashes and metadata for recall.
- Load only validated, bounded assets into realtime processing.

This preserves the distinctive spatial identity while keeping the plug-in stable in DAWs.

## State, Presets, And Assets

Preset state should be host-recallable and deterministic. It should include:

- All registered parameters.
- Engine mode.
- Quality target.
- Geometry profile reference.
- Asset hash/version.
- UI page/tab state where appropriate.

Large external assets, such as imported DXF files or generated IRs, should not be embedded blindly into host state. Store references plus compact metadata, and show clear missing-asset errors when needed.

## Error Handling

The plug-in should fail clearly rather than surprise the user.

Visible status/error cases include:

- Unsupported or changed channel layout.
- Missing geometry asset or invalid asset hash.
- Invalid DXF-derived profile.
- Quality target too expensive for current CPU/block size.
- Oversampling unavailable at the current host sample rate.
- Internal latency changed and requires host notification.
- Parameter value clamped by safety rules.

The footer/status strip is the primary place for warnings. Severe errors should degrade safely to bypass or a known-good default rather than producing unstable audio.

## Testing Strategy

Testing starts below the DAW host:

- Parameter mapping tests for RT60 coarse/fine across the full 0.01s to 360s range.
- DSP unit tests for deterministic blocks and no NaN/Inf output.
- Golden audio snapshots at 44.1, 48, 96, and 192 kHz host rates.
- Oversampling/latency reporting tests.
- Preset serialization and recall tests.
- Geometry profile validation tests.
- Stress tests for extreme RT60, freeze mode, reverse mode, wet/dry edges, and CPU-heavy quality settings.

DAW validation comes after the standalone core tests:

- VST3 load/save/automation smoke tests.
- AU/AUv3 load/save/automation smoke tests.
- Bypass, latency compensation, sample-rate change, and block-size change checks.
- Visual telemetry checks to confirm the editor never blocks or mutates the audio callback.

## Implementation Sequence

1. Define the plug-in parameter manifest, including RT60 coarse/fine and quality target.
2. Extract or wrap the native realtime DSP core behind a testable C/C++ interface.
3. Build a minimal plug-in shell with audio pass-through, state, parameters, and latency reporting.
4. Connect the deterministic reverb core with oversampling and smoothing.
5. Implement the full-screen editor skeleton based on the approved spatial-console direction.
6. Add telemetry taps and visual panels.
7. Add geometry profile data structures and validation.
8. Add offline DXF/profile import as a bounded asset workflow.
9. Expand expert controls only after their realtime semantics are stable.

## Open Decisions For The Implementation Plan

- Whether the first Apple target should be AUv3 only or AUv2 plus AUv3 for broader desktop DAW compatibility.
- Whether JUCE is adopted directly or the repo first creates a neutral native core with a later JUCE wrapper.
- Which DAW hosts form the initial smoke-test matrix.
- Whether the first UI implementation uses custom JUCE drawing, a webview-like prototype, or a hybrid staged approach.

These are implementation-plan decisions, not blockers for the approved design direction.
