# VERBX AUv3/VST3 Plug-in Handbook

This handbook describes the VERBX plug-in direction, the native foundation that
already exists, and the operational practices required to turn that foundation
into a dependable AU, AUv3, VST3, and standalone product. It is deliberately
honest about maturity. The repository contains a tested parameter manifest, a
realtime context boundary, a guarded C++17/JUCE shell, state serialization, and
a realtime spectrum-overlay component, a complete initial control dock, and an
allocation-free mono/stereo Schroeder reverb with real wet-path oversampling.
It is a usable native engine, but not yet the final multichannel architecture.
The full-screen
image below is the approved visual target and a live design-prototype capture,
not a screenshot of a shipping binary.

![VERBX full-screen plug-in design](assets/verbx_plugin_fullscreen.png)

The next image is an actual capture of the compiled JUCE standalone editor.
Its input is muted by the host, so the post-DSP trace sits at the analyzer floor;
the logarithmic frequency grid and overlaid display are production C++ rather
than browser-prototype artwork.

![VERBX native realtime spectrum analyzer](assets/verbx_plugin_native_analyzer.jpg)

The next image is generated directly from the compiled editor interaction
smoke test. Expert mode keeps every continuous parameter available as both a
dial and a precision fader, retains the realtime analyzer, and adds five
four-way selector banks. These selectors write existing host state rather than
creating hidden or decorative settings.

![VERBX compiled Expert control matrix](assets/verbx_plugin_expert.png)

## 1. Product Intent

VERBX is designed as a spatial architecture instrument rather than a generic
reverb with a decorative room picture. Geometry, decay, imaging, dynamics, and
quality status should be readable at a glance. The front panel must remain
playable even though the underlying engine has a deep parameter surface. That
leads to three control layers:

- A performance layer for the controls a musician or mixer reaches for during
  playback: pre-delay, room size, RT60 coarse and fine, damping, width,
  diffusion, wet, dry, Freeze, Reverse, and quality.
- An expert layer that currently provides linked precision control and safe
  macros for quality, width, decay, mix routing, and tail character; future FDN
  topology, modulation, dynamics, and geometry controls land only after those
  behaviors are realtime-safe and stable.
- An automation layer for parameters that a host can recall and automate even
  when they are not continuously visible on the main page.

The plug-in must never imply that visual complexity is equivalent to acoustic
accuracy. A geometry display is useful only when it corresponds to validated
room metadata, an early-reflection model, or a deterministic imported asset.
The display should state whether it is showing a parametric room, a DXF-derived
profile, a measured impulse response, or an illustrative preview.

### Recommended Companion Text

Will C. Pirkle's *[Designing Audio Effect Plugins in C++: For AAX, AU, and VST3
with DSP Theory](https://www.routledge.com/Designing-Audio-Effect-Plugins-in-C-For-AAX-AU-and-VST3-with-DSP-Theory/Pirkle/p/book/9781138591899)*,
2nd edition (Routledge, 2019), is the recommended companion text for this
handbook. Its treatment of plug-in anatomy, an API-independent processing core,
host wrappers, parameters, GUI design, delay structures, reverberation, and
dynamics provides a useful engineering vocabulary for the boundaries described
here. VERBX does not adopt Pirkle's example framework wholesale: its C realtime
contract, JUCE adapter, parameter manifest, state format, analyzer telemetry,
and validation matrix remain repository-specific designs. Readers should use
the book for the broader implementation discipline and this handbook for the
exact VERBX contracts.

## 2. What Exists In The Repository

The first foundation slice is intentionally narrow and testable:

- `verbx_c_core` is a reusable C11 static library target.
- `plugin_params.h` defines twelve stable initial parameters and the quality
  choices.
- `plugin_params.c` implements deterministic clamping and logarithmic RT60
  coarse/fine mapping from 0.01 seconds to 360 seconds.
- `plugin_realtime.h` defines host configuration, realtime parameters, status,
  context lifecycle, latency accessors, and processing entry points.
- `plugin_realtime.c` validates host configuration, allocates persistent state
  only during preparation, and provides bounded mono/stereo Schroeder processing
  with pre-delay, room scale, RT60, damping, diffusion, width, wet/dry, Freeze,
  and a zero-lookahead reverse-style swell. Continuous controls use 20 ms
  smoothing inside the native state so host automation does not zipper. Host,
  2x, 4x, and Target quality modes execute that wet network at the reported
  internal rate without callback allocation.
- `native/verbx_plugin` contains the guarded JUCE shell for AU, AUv3, VST3, and
  standalone targets.
- The processor caches atomic parameter pointers during construction so the
  callback reads values without rebuilding strings or searching parameter maps.
- The resizable editor includes a post-DSP spectrum overlay. A fixed SPSC ring
  carries mono output snapshots off the callback; the message thread performs
  an 8192-point Hann FFT at 30 visual frames per second, with logarithmic
  frequency spacing, release smoothing, and a decaying peak trace.
- Perform and Expert are native JUCE pages. Expert contains nine linked rotary
  controls, nine precision faders, the live analyzer, and twenty selector
  buttons; all write the existing APVTS host state.
- The complete initial twelve-parameter surface is attached to host automation,
  with compact musical units and a live effective-RT60 readout.

This boundary is valuable even before the reverb DSP is connected. Host code,
parameter identity, state recall, bus negotiation, callback constraints, and
latency reporting can be stabilized independently of the sound engine.

## 3. Building The Plug-in Foundation

Default repository configuration does not require JUCE:

```bash
cmake -S native/verbx_plugin -B build/native/verbx_plugin
```

The configure output should say that `VERBX_ENABLE_JUCE_PLUGIN` is off. This is
the supported path for contributors who only need the C core or Python CLI.

When JUCE is installed as a discoverable CMake package, enable the actual host
targets:

```bash
cmake -S native/verbx_plugin -B build/native/verbx_plugin-juce \
  -DVERBX_ENABLE_JUCE_PLUGIN=ON
cmake --build build/native/verbx_plugin-juce --config Release
```

For a JUCE source checkout, add
`-DVERBX_JUCE_SOURCE_DIR=/path/to/JUCE` to the configure command.

The enabled target requests AU, AUv3, VST3, and Standalone formats. A successful
compile is only the beginning of validation. The resulting formats must be
scanned, instantiated, automated, saved, reopened, and stress-tested in their
actual hosts before compatibility is claimed.

## 4. Signal And Ownership Boundaries

The plug-in has three architectural layers. The host shell owns format-specific
lifecycle, buses, state, parameters, editor creation, and host latency
notification. The adapter maps host blocks and normalized automation into the
native realtime contract. The DSP core owns audio behavior and must remain
testable without a DAW.

![Algorithmic signal topology](assets/userguide_figures/01_signal_flow.png)

The audio callback is a hard boundary. It must not parse DXF, open files, update
the preset browser, allocate variable-sized containers, rebuild parameter IDs,
take UI locks, or wait for background work. Large assets are prepared away from
the callback and swapped through bounded, versioned handles. Telemetry flows in
the opposite direction through compact snapshots that the editor may poll.

## 5. Precision And Sample Rate

The DAW controls project sample rate. VERBX cannot force a host session to 192
kHz. The default quality choice, Target 192 kHz, selects the smallest integer
factor whose internal rate reaches or exceeds 192 kHz. At a 48 kHz host rate
this is 4x/192 kHz; at 96 kHz it is 2x/192 kHz; at 44.1 kHz it is 5x/220.5 kHz;
and at 192 kHz or above it does not increase the rate. Host, 2x, and 4x select
their exact factors. The wet network uses causal linear interpolation and
box-filter decimation, while the dry path retains the original host samples.
Quality changes allocate and prepare away from the callback, then cross the
processor boundary through a nonblocking atomic guard.

The initial callback contract uses 32-bit float because that is the common
native exchange type for AU/VST processing and keeps bandwidth bounded. This
does not forbid double-precision accumulators, offline float64 rendering, or a
future double-precision host path. Precision claims should always distinguish
host buffer format, internal state precision, oversampled processing rate, and
file-render output format.

![Sample-rate cost](assets/userguide_figures/36_sample_rate_cost.png)

## 6. RT60 Coarse And Fine

One linear knob cannot offer useful control from 0.01 seconds to 360 seconds.
The coarse control therefore maps normalized automation logarithmically:

```text
coarse = exp(log(0.01) + normalized * (log(360.0) - log(0.01)))
fine_ratio = exp(log(1.20) * bipolar_fine)
effective = clamp(coarse * fine_ratio, 0.01, 360.0)
```

The fine control provides a plus-or-minus 20 percent log-space trim. It remains
musically useful around a short booth, a medium hall, and an enormous ambient
tail. The effective value shown by the editor is authoritative; the coarse knob
position alone is not.

Freeze is separate from maximum RT60. Freeze changes the energy behavior of the
network and needs its own smoothing and safety semantics. Reverse is also a
separate mode because it introduces a fundamentally different envelope and may
require buffering and reported latency.

![RT60 decay families](assets/userguide_figures/03_rt60_decay_families.png)

## 7. Initial Parameter Reference

The first manifest contains twelve parameters. Their IDs are intended to remain
stable once released because DAW automation and saved sessions depend on them.

| Parameter ID | User meaning | Initial range | Default | Realtime note |
| --- | --- | --- | --- | --- |
| `pre_delay_ms` | Gap before the reverberant field | 0 to 1000 ms | 18 ms | Delay changes require smoothing or crossfade |
| `room_size` | Macro geometry/scale control | 0 to 1 | 0.72 | Must not resize unbounded memory in callback |
| `rt60_coarse` | Logarithmic decay position | 0 to 1 | 0.50 | Maps to 0.01 to 360 seconds |
| `rt60_fine` | Bipolar log trim | –1 to 1 | 0 | Applies about plus/minus 20 percent |
| `damping` | High-frequency decay loss | 0 to 0.98 | 0.41 | Coefficients need stable interpolation |
| `width` | Stereo/spatial spread | 0 to 2 | 1.35 | Check mono and correlation behavior |
| `diffusion` | Echo-density macro | 0 to 1 | 0.65 | Structural changes may need a safe transition |
| `wet` | Processed contribution | 0 to 1 | 0.62 | Use a deliberate mix law |
| `dry` | Direct contribution | 0 to 1 | 0.78 | Preserve gain staging and bypass behavior |
| `freeze` | Infinite/sustaining mode | off/on | off | Separate energy-state transition |
| `reverse` | Reverse-envelope mode | off/on | off | Buffering and latency must be explicit |
| `quality_mode` | Internal rate policy | Host/2x/4x/Target | Target | Reprepare outside callback when needed |

## 8. State, Presets, And Session Recall

The JUCE shell serializes its parameter tree into host state. A production
version also needs an explicit schema version, migration rules, asset identity,
and diagnostics for partial restoration. A preset should not silently become a
different sound because a geometry file moved or an IR cache was regenerated.

Small deterministic data belongs in host state. Large DXF files, measured IRs,
and generated spatial assets should normally be referenced by a stable identity
plus a content hash. The state can embed a compact fallback profile when that is
small enough. Missing assets should produce a visible warning and a safe known
sound, not silence, noise, or an unexplained default room.

## 9. Geometry And DXF

DXF import is an offline preparation workflow. Parsing, topology repair, ray
generation, acceleration-structure construction, and IR synthesis must not run
inside the audio callback. The realtime side consumes a validated bounded
profile: room dimensions, material identifiers, source/listener transforms,
early-reflection taps, spatial metadata, and optional prepared IR partitions.

The visual theater should show provenance. Labels such as `Parametric Room`,
`Imported DXF Profile`, `Measured IR`, or `Illustrative View` prevent the user
from confusing a decorative image with a physical simulation. Geometry edits
can be staged in the editor and committed through a background preparation job;
the audio thread receives the finished immutable result through a safe swap.

![Material absorption map](assets/userguide_figures/55_material_absorption_map.png)

## 10. Latency

Total plug-in latency may include block adaptation, oversampling filters,
lookahead dynamics, convolution partitions, and reverse-reverb buffering. The
processor must report the exact stable value to the host whenever a mode changes
the processing graph. If latency cannot change safely during playback, the UI
should state that the change will apply on transport stop or reprepare.

![Realtime latency components](assets/userguide_figures/02_realtime_latency.png)

The current realtime reverb and causal oversampling path report zero frames:
they do not buffer future host samples. That value must change when a
latency-producing filter, convolution partition, or bounded reverse-lookahead
stage lands. The live status display distinguishes host and internal rates,
factor, block size, and algorithmic latency. Device I/O and end-to-end monitored
latency remain separate measurements.

## 11. Realtime Safety

The callback should perform bounded arithmetic, pointer traversal, atomic loads,
and DSP over memory prepared in advance. It should avoid filesystem access,
logging, UI calls, mutex acquisition, heap allocation, and operations whose
runtime grows unpredictably with external assets. Denormal handling, NaN/Inf
containment, and explicit channel/block bounds are part of the audio contract.

![Realtime callback budget](assets/userguide_figures/82_realtime_callback_budget.png)

Parameter smoothing is not cosmetic. Abrupt pre-delay, damping, mix, and matrix
changes can click or destabilize feedback. Each parameter needs a declared
transition strategy: sample ramp, coefficient interpolation, dual-engine
crossfade, transport-gated reprepare, or intentionally discrete switch.

## 12. Freeze And Reverse

Freeze should enter and leave with controlled energy. The production algorithm
must define whether input injection stops, decay gains approach unity, damping
continues, modulation remains active, and limiter protection stays engaged. A
freeze button is not permission for unbounded gain.

![Infinite reverb behavior](assets/userguide_figures/24_infinite_reverb.png)

Reverse needs an explicit latency model. A true reverse response requires future
context, pre-rendered material, or a bounded capture window. The UI should show
the active window and reported latency. If a low-latency approximation is
offered, it must be named as an approximation rather than presented as identical
to offline reverse rendering.

![Reverse reverb envelope](assets/userguide_figures/33_reverse_reverb_envelope.png)

## 13. Loudness, Ducking, And Limiting

The limiter is a safety layer, not a substitute for stable feedback design.
Meters should distinguish input, wet return, output, gain reduction, true peak,
and any safety attenuation. Ducking should expose detector source, attack,
release, range, and whether the dry path participates.

Host bypass semantics matter. A hard host bypass may skip internal tails, while
an effect bypass parameter can preserve or drain them. The product must define
both behaviors and test them in every supported format.

## 14. Bus Layouts And Spatial Formats

The foundation accepts matched mono or stereo main buses. Production expansion
should proceed through explicit layouts rather than accepting arbitrary channel
counts. Stereo, 5.1, 7.1, 7.1.4, and ambisonic modes have different routing,
normalization, and host metadata requirements. A layout is supported only after
processing, state recall, metering, and host scanning have been validated.

![Speaker layout coverage](assets/userguide_figures/72_speaker_layout_coverage.png)

## 15. Editor And Accessibility

The full-screen design is dense, so hierarchy is essential. Keyboard focus,
screen-reader labels, scalable text, high-contrast status colors, and a compact
window mode should be designed alongside custom drawing. A parameter remains
usable when its decorative visualization is disabled.

Telemetry should be rate-limited and decoupled from audio. The editor reads
snapshots; it does not interrogate mutable DSP structures. Closing the editor
must not change the sound or CPU behavior of the processor.

The implemented spectrum overlay follows that rule: the callback only writes
post-DSP mono samples into a fixed lock-free ring and drops new analyzer samples
if the display falls behind. The editor drains that ring, windows and transforms
8192 samples, smooths the dB response, and paints the fill and peak paths at 30
Hz. No FFT, path allocation, repaint, or UI lock occurs on the audio thread.

### Expert Control Matrix

Select **Expert** in the editor header to replace the visual performance
console with a dense precision workspace. The top row contains nine rotary
controls and the center matrix contains nine linked horizontal faders. Each
dial/fader pair is attached to the same APVTS parameter, so moving either
control updates host automation, the other control, saved state, and the DSP.
The spectrum analyzer remains visible while editing.

The five selector banks each provide four native buttons:

- **Quality** writes Host, 2x, 4x, or Target 192 kHz policy.
- **Width Matrix** writes calibrated Mono, Natural, Wide, or Ultra width.
- **Decay Range** writes logarithmic Tight, Room, or Hall RT60 values; Freeze
  also writes the separate Freeze state.
- **Mix Routing** writes matched Dry, Insert, Parallel, or Send dry/wet pairs.
- **Tail Character** writes matched damping/diffusion pairs for Clean, Warm,
  Dark, or Air behavior.

Selector highlighting follows current host parameter values. If automation or
manual editing creates a value that does not exactly match a macro, the bank
clears its highlight rather than claiming a preset that is no longer active.
This makes the selector state descriptive, not authoritative.

Numeric entry uses the units shown by the control. Enter pre-delay in
milliseconds, RT60 directly in seconds, and Room Size, RT60 Fine, Damping,
Width, Diffusion, Wet, and Dry as percentages. RT60 seconds are inverted through
the logarithmic 0.01-to-360-second mapping; for example, typing `4.8 s` produces
an effective 4.8-second decay rather than treating 4.8 as a normalized value.
RT60 Fine uses its displayed plus-or-minus 20 percent scale. Click a dial arc
for immediate positioning, drag vertically for precision, use the mouse wheel
for increments, or double-click to restore the declared parameter default.

## 16. Compatibility Claims

The repository currently provides build scaffolding, not a published host
compatibility matrix. A format name in CMake means that JUCE can generate that
target when dependencies and platform tools are available. It does not mean the
binary has passed scanning and session-recall tests in every DAW.

Compatibility statements should name plug-in format, CPU architecture,
operating-system version, host/version, sample rate, block size, bus layout,
and validation date. Results from a standalone build do not substitute for an
AUv3 sandbox test or VST3 host scan.

## 17. Validation Commands

The native foundation can be checked without JUCE:

```bash
cmake -S native/verbx_c -B build/native/verbx_c-plan
cmake --build build/native/verbx_c-plan
ctest --test-dir build/native/verbx_c-plan --output-on-failure
uv run pytest tests/test_native_scaffold.py -q
./scripts/build_verbx_c.sh --clean --doctor
cmake -S native/verbx_plugin -B build/native/verbx_plugin-plan
```

These checks verify the C boundary, direct-build alignment, native render
regressions, and JUCE-disabled configure path. They do not compile or validate a
JUCE-enabled plug-in when JUCE is absent.

## 18. Reading The Operational Cards

The remainder of this handbook is organized as one-page cards. They are not
compatibility certifications or fixed presets. They are repeatable starting
points, automation studies, validation procedures, and troubleshooting drills.
Record the plug-in build, host, sample rate, block size, channel layout, and
asset hashes whenever a card is used for a formal test.

## 19. Production Starting-Point Cards

### 19.1 Lead vocal

**Production card: Lead vocal in Tight room**

- Intent: preserve consonants and front-of-mix intelligibility.
- Space character: dense early cues and a controlled short tail.
- Starting macros: room size 0.28, damping 0.20, diffusion 0.52, wet 0.28.
- Pre-delay working range: 18 to 45 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry lead vocal at a calibrated monitoring level. Establish the direct image first, then raise the wet return until dense early cues and a controlled short tail is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch sibilance and center stability. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Lead vocal in Studio chamber**

- Intent: preserve consonants and front-of-mix intelligibility.
- Space character: a smooth useful chamber with moderate width.
- Starting macros: room size 0.42, damping 0.34, diffusion 0.64, wet 0.38.
- Pre-delay working range: 18 to 45 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry lead vocal at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a smooth useful chamber with moderate width is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch sibilance and center stability. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Lead vocal in Scoring stage**

- Intent: preserve consonants and front-of-mix intelligibility.
- Space character: clear source distance and a broad late field.
- Starting macros: room size 0.58, damping 0.46, diffusion 0.72, wet 0.44.
- Pre-delay working range: 18 to 45 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry lead vocal at a calibrated monitoring level. Establish the direct image first, then raise the wet return until clear source distance and a broad late field is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch sibilance and center stability. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Lead vocal in Concert hall**

- Intent: preserve consonants and front-of-mix intelligibility.
- Space character: a long integrated decay with stable imaging.
- Starting macros: room size 0.68, damping 0.52, diffusion 0.78, wet 0.50.
- Pre-delay working range: 18 to 45 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry lead vocal at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a long integrated decay with stable imaging is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch sibilance and center stability. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Lead vocal in Stone cathedral**

- Intent: preserve consonants and front-of-mix intelligibility.
- Space character: slow spectral decay and monumental scale.
- Starting macros: room size 0.78, damping 0.38, diffusion 0.84, wet 0.58.
- Pre-delay working range: 18 to 45 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry lead vocal at a calibrated monitoring level. Establish the direct image first, then raise the wet return until slow spectral decay and monumental scale is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch sibilance and center stability. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Lead vocal in Plate-like field**

- Intent: preserve consonants and front-of-mix intelligibility.
- Space character: fast diffusion with less geometric localization.
- Starting macros: room size 0.48, damping 0.62, diffusion 0.88, wet 0.46.
- Pre-delay working range: 18 to 45 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry lead vocal at a calibrated monitoring level. Establish the direct image first, then raise the wet return until fast diffusion with less geometric localization is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch sibilance and center stability. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Lead vocal in Reverse chamber**

- Intent: preserve consonants and front-of-mix intelligibility.
- Space character: a bounded reverse envelope with explicit latency.
- Starting macros: room size 0.50, damping 0.45, diffusion 0.70, wet 0.52.
- Pre-delay working range: 18 to 45 ms.
- Modes: Reverse on; Freeze off.

Begin with the dry lead vocal at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a bounded reverse envelope with explicit latency is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch sibilance and center stability. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Lead vocal in Frozen architecture**

- Intent: preserve consonants and front-of-mix intelligibility.
- Space character: a sustained field entered and exited safely.
- Starting macros: room size 0.72, damping 0.44, diffusion 0.82, wet 0.66.
- Pre-delay working range: 18 to 45 ms.
- Modes: Reverse off; Freeze prepared/on.

Begin with the dry lead vocal at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a sustained field entered and exited safely is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch sibilance and center stability. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

### 19.2 Spoken word

**Production card: Spoken word in Tight room**

- Intent: add believable room cues without masking language.
- Space character: dense early cues and a controlled short tail.
- Starting macros: room size 0.28, damping 0.20, diffusion 0.52, wet 0.28.
- Pre-delay working range: 8 to 28 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry spoken word at a calibrated monitoring level. Establish the direct image first, then raise the wet return until dense early cues and a controlled short tail is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check breaths, plosives, and noise-floor lift. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Spoken word in Studio chamber**

- Intent: add believable room cues without masking language.
- Space character: a smooth useful chamber with moderate width.
- Starting macros: room size 0.42, damping 0.34, diffusion 0.64, wet 0.38.
- Pre-delay working range: 8 to 28 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry spoken word at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a smooth useful chamber with moderate width is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check breaths, plosives, and noise-floor lift. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Spoken word in Scoring stage**

- Intent: add believable room cues without masking language.
- Space character: clear source distance and a broad late field.
- Starting macros: room size 0.58, damping 0.46, diffusion 0.72, wet 0.44.
- Pre-delay working range: 8 to 28 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry spoken word at a calibrated monitoring level. Establish the direct image first, then raise the wet return until clear source distance and a broad late field is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check breaths, plosives, and noise-floor lift. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Spoken word in Concert hall**

- Intent: add believable room cues without masking language.
- Space character: a long integrated decay with stable imaging.
- Starting macros: room size 0.68, damping 0.52, diffusion 0.78, wet 0.50.
- Pre-delay working range: 8 to 28 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry spoken word at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a long integrated decay with stable imaging is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check breaths, plosives, and noise-floor lift. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Spoken word in Stone cathedral**

- Intent: add believable room cues without masking language.
- Space character: slow spectral decay and monumental scale.
- Starting macros: room size 0.78, damping 0.38, diffusion 0.84, wet 0.58.
- Pre-delay working range: 8 to 28 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry spoken word at a calibrated monitoring level. Establish the direct image first, then raise the wet return until slow spectral decay and monumental scale is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check breaths, plosives, and noise-floor lift. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Spoken word in Plate-like field**

- Intent: add believable room cues without masking language.
- Space character: fast diffusion with less geometric localization.
- Starting macros: room size 0.48, damping 0.62, diffusion 0.88, wet 0.46.
- Pre-delay working range: 8 to 28 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry spoken word at a calibrated monitoring level. Establish the direct image first, then raise the wet return until fast diffusion with less geometric localization is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check breaths, plosives, and noise-floor lift. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Spoken word in Reverse chamber**

- Intent: add believable room cues without masking language.
- Space character: a bounded reverse envelope with explicit latency.
- Starting macros: room size 0.50, damping 0.45, diffusion 0.70, wet 0.52.
- Pre-delay working range: 8 to 28 ms.
- Modes: Reverse on; Freeze off.

Begin with the dry spoken word at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a bounded reverse envelope with explicit latency is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check breaths, plosives, and noise-floor lift. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Spoken word in Frozen architecture**

- Intent: add believable room cues without masking language.
- Space character: a sustained field entered and exited safely.
- Starting macros: room size 0.72, damping 0.44, diffusion 0.82, wet 0.66.
- Pre-delay working range: 8 to 28 ms.
- Modes: Reverse off; Freeze prepared/on.

Begin with the dry spoken word at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a sustained field entered and exited safely is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check breaths, plosives, and noise-floor lift. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

### 19.3 Drum kit

**Production card: Drum kit in Tight room**

- Intent: build size while preserving transient geometry.
- Space character: dense early cues and a controlled short tail.
- Starting macros: room size 0.28, damping 0.20, diffusion 0.52, wet 0.28.
- Pre-delay working range: 4 to 24 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry drum kit at a calibrated monitoring level. Establish the direct image first, then raise the wet return until dense early cues and a controlled short tail is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check kick definition and snare tail density. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Drum kit in Studio chamber**

- Intent: build size while preserving transient geometry.
- Space character: a smooth useful chamber with moderate width.
- Starting macros: room size 0.42, damping 0.34, diffusion 0.64, wet 0.38.
- Pre-delay working range: 4 to 24 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry drum kit at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a smooth useful chamber with moderate width is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check kick definition and snare tail density. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Drum kit in Scoring stage**

- Intent: build size while preserving transient geometry.
- Space character: clear source distance and a broad late field.
- Starting macros: room size 0.58, damping 0.46, diffusion 0.72, wet 0.44.
- Pre-delay working range: 4 to 24 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry drum kit at a calibrated monitoring level. Establish the direct image first, then raise the wet return until clear source distance and a broad late field is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check kick definition and snare tail density. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Drum kit in Concert hall**

- Intent: build size while preserving transient geometry.
- Space character: a long integrated decay with stable imaging.
- Starting macros: room size 0.68, damping 0.52, diffusion 0.78, wet 0.50.
- Pre-delay working range: 4 to 24 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry drum kit at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a long integrated decay with stable imaging is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check kick definition and snare tail density. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Drum kit in Stone cathedral**

- Intent: build size while preserving transient geometry.
- Space character: slow spectral decay and monumental scale.
- Starting macros: room size 0.78, damping 0.38, diffusion 0.84, wet 0.58.
- Pre-delay working range: 4 to 24 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry drum kit at a calibrated monitoring level. Establish the direct image first, then raise the wet return until slow spectral decay and monumental scale is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check kick definition and snare tail density. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Drum kit in Plate-like field**

- Intent: build size while preserving transient geometry.
- Space character: fast diffusion with less geometric localization.
- Starting macros: room size 0.48, damping 0.62, diffusion 0.88, wet 0.46.
- Pre-delay working range: 4 to 24 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry drum kit at a calibrated monitoring level. Establish the direct image first, then raise the wet return until fast diffusion with less geometric localization is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check kick definition and snare tail density. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Drum kit in Reverse chamber**

- Intent: build size while preserving transient geometry.
- Space character: a bounded reverse envelope with explicit latency.
- Starting macros: room size 0.50, damping 0.45, diffusion 0.70, wet 0.52.
- Pre-delay working range: 4 to 24 ms.
- Modes: Reverse on; Freeze off.

Begin with the dry drum kit at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a bounded reverse envelope with explicit latency is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check kick definition and snare tail density. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Drum kit in Frozen architecture**

- Intent: build size while preserving transient geometry.
- Space character: a sustained field entered and exited safely.
- Starting macros: room size 0.72, damping 0.44, diffusion 0.82, wet 0.66.
- Pre-delay working range: 4 to 24 ms.
- Modes: Reverse off; Freeze prepared/on.

Begin with the dry drum kit at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a sustained field entered and exited safely is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check kick definition and snare tail density. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

### 19.4 Piano

**Production card: Piano in Tight room**

- Intent: support sustain without blurring note attacks.
- Space character: dense early cues and a controlled short tail.
- Starting macros: room size 0.28, damping 0.20, diffusion 0.52, wet 0.28.
- Pre-delay working range: 12 to 40 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry piano at a calibrated monitoring level. Establish the direct image first, then raise the wet return until dense early cues and a controlled short tail is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, listen for low-mid modal buildup. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Piano in Studio chamber**

- Intent: support sustain without blurring note attacks.
- Space character: a smooth useful chamber with moderate width.
- Starting macros: room size 0.42, damping 0.34, diffusion 0.64, wet 0.38.
- Pre-delay working range: 12 to 40 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry piano at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a smooth useful chamber with moderate width is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, listen for low-mid modal buildup. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Piano in Scoring stage**

- Intent: support sustain without blurring note attacks.
- Space character: clear source distance and a broad late field.
- Starting macros: room size 0.58, damping 0.46, diffusion 0.72, wet 0.44.
- Pre-delay working range: 12 to 40 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry piano at a calibrated monitoring level. Establish the direct image first, then raise the wet return until clear source distance and a broad late field is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, listen for low-mid modal buildup. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Piano in Concert hall**

- Intent: support sustain without blurring note attacks.
- Space character: a long integrated decay with stable imaging.
- Starting macros: room size 0.68, damping 0.52, diffusion 0.78, wet 0.50.
- Pre-delay working range: 12 to 40 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry piano at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a long integrated decay with stable imaging is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, listen for low-mid modal buildup. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Piano in Stone cathedral**

- Intent: support sustain without blurring note attacks.
- Space character: slow spectral decay and monumental scale.
- Starting macros: room size 0.78, damping 0.38, diffusion 0.84, wet 0.58.
- Pre-delay working range: 12 to 40 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry piano at a calibrated monitoring level. Establish the direct image first, then raise the wet return until slow spectral decay and monumental scale is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, listen for low-mid modal buildup. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Piano in Plate-like field**

- Intent: support sustain without blurring note attacks.
- Space character: fast diffusion with less geometric localization.
- Starting macros: room size 0.48, damping 0.62, diffusion 0.88, wet 0.46.
- Pre-delay working range: 12 to 40 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry piano at a calibrated monitoring level. Establish the direct image first, then raise the wet return until fast diffusion with less geometric localization is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, listen for low-mid modal buildup. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Piano in Reverse chamber**

- Intent: support sustain without blurring note attacks.
- Space character: a bounded reverse envelope with explicit latency.
- Starting macros: room size 0.50, damping 0.45, diffusion 0.70, wet 0.52.
- Pre-delay working range: 12 to 40 ms.
- Modes: Reverse on; Freeze off.

Begin with the dry piano at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a bounded reverse envelope with explicit latency is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, listen for low-mid modal buildup. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Piano in Frozen architecture**

- Intent: support sustain without blurring note attacks.
- Space character: a sustained field entered and exited safely.
- Starting macros: room size 0.72, damping 0.44, diffusion 0.82, wet 0.66.
- Pre-delay working range: 12 to 40 ms.
- Modes: Reverse off; Freeze prepared/on.

Begin with the dry piano at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a sustained field entered and exited safely is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, listen for low-mid modal buildup. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

### 19.5 Acoustic guitar

**Production card: Acoustic guitar in Tight room**

- Intent: add depth without combing the direct image.
- Space character: dense early cues and a controlled short tail.
- Starting macros: room size 0.28, damping 0.20, diffusion 0.52, wet 0.28.
- Pre-delay working range: 10 to 32 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry acoustic guitar at a calibrated monitoring level. Establish the direct image first, then raise the wet return until dense early cues and a controlled short tail is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check pick articulation and mono fold-down. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Acoustic guitar in Studio chamber**

- Intent: add depth without combing the direct image.
- Space character: a smooth useful chamber with moderate width.
- Starting macros: room size 0.42, damping 0.34, diffusion 0.64, wet 0.38.
- Pre-delay working range: 10 to 32 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry acoustic guitar at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a smooth useful chamber with moderate width is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check pick articulation and mono fold-down. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Acoustic guitar in Scoring stage**

- Intent: add depth without combing the direct image.
- Space character: clear source distance and a broad late field.
- Starting macros: room size 0.58, damping 0.46, diffusion 0.72, wet 0.44.
- Pre-delay working range: 10 to 32 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry acoustic guitar at a calibrated monitoring level. Establish the direct image first, then raise the wet return until clear source distance and a broad late field is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check pick articulation and mono fold-down. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Acoustic guitar in Concert hall**

- Intent: add depth without combing the direct image.
- Space character: a long integrated decay with stable imaging.
- Starting macros: room size 0.68, damping 0.52, diffusion 0.78, wet 0.50.
- Pre-delay working range: 10 to 32 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry acoustic guitar at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a long integrated decay with stable imaging is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check pick articulation and mono fold-down. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Acoustic guitar in Stone cathedral**

- Intent: add depth without combing the direct image.
- Space character: slow spectral decay and monumental scale.
- Starting macros: room size 0.78, damping 0.38, diffusion 0.84, wet 0.58.
- Pre-delay working range: 10 to 32 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry acoustic guitar at a calibrated monitoring level. Establish the direct image first, then raise the wet return until slow spectral decay and monumental scale is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check pick articulation and mono fold-down. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Acoustic guitar in Plate-like field**

- Intent: add depth without combing the direct image.
- Space character: fast diffusion with less geometric localization.
- Starting macros: room size 0.48, damping 0.62, diffusion 0.88, wet 0.46.
- Pre-delay working range: 10 to 32 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry acoustic guitar at a calibrated monitoring level. Establish the direct image first, then raise the wet return until fast diffusion with less geometric localization is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check pick articulation and mono fold-down. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Acoustic guitar in Reverse chamber**

- Intent: add depth without combing the direct image.
- Space character: a bounded reverse envelope with explicit latency.
- Starting macros: room size 0.50, damping 0.45, diffusion 0.70, wet 0.52.
- Pre-delay working range: 10 to 32 ms.
- Modes: Reverse on; Freeze off.

Begin with the dry acoustic guitar at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a bounded reverse envelope with explicit latency is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check pick articulation and mono fold-down. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Acoustic guitar in Frozen architecture**

- Intent: add depth without combing the direct image.
- Space character: a sustained field entered and exited safely.
- Starting macros: room size 0.72, damping 0.44, diffusion 0.82, wet 0.66.
- Pre-delay working range: 10 to 32 ms.
- Modes: Reverse off; Freeze prepared/on.

Begin with the dry acoustic guitar at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a sustained field entered and exited safely is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check pick articulation and mono fold-down. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

### 19.6 Electric guitar

**Production card: Electric guitar in Tight room**

- Intent: place the cabinet in a designed environment.
- Space character: dense early cues and a controlled short tail.
- Starting macros: room size 0.28, damping 0.20, diffusion 0.52, wet 0.28.
- Pre-delay working range: 6 to 30 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry electric guitar at a calibrated monitoring level. Establish the direct image first, then raise the wet return until dense early cues and a controlled short tail is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch upper-mid glare in the return. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Electric guitar in Studio chamber**

- Intent: place the cabinet in a designed environment.
- Space character: a smooth useful chamber with moderate width.
- Starting macros: room size 0.42, damping 0.34, diffusion 0.64, wet 0.38.
- Pre-delay working range: 6 to 30 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry electric guitar at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a smooth useful chamber with moderate width is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch upper-mid glare in the return. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Electric guitar in Scoring stage**

- Intent: place the cabinet in a designed environment.
- Space character: clear source distance and a broad late field.
- Starting macros: room size 0.58, damping 0.46, diffusion 0.72, wet 0.44.
- Pre-delay working range: 6 to 30 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry electric guitar at a calibrated monitoring level. Establish the direct image first, then raise the wet return until clear source distance and a broad late field is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch upper-mid glare in the return. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Electric guitar in Concert hall**

- Intent: place the cabinet in a designed environment.
- Space character: a long integrated decay with stable imaging.
- Starting macros: room size 0.68, damping 0.52, diffusion 0.78, wet 0.50.
- Pre-delay working range: 6 to 30 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry electric guitar at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a long integrated decay with stable imaging is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch upper-mid glare in the return. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Electric guitar in Stone cathedral**

- Intent: place the cabinet in a designed environment.
- Space character: slow spectral decay and monumental scale.
- Starting macros: room size 0.78, damping 0.38, diffusion 0.84, wet 0.58.
- Pre-delay working range: 6 to 30 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry electric guitar at a calibrated monitoring level. Establish the direct image first, then raise the wet return until slow spectral decay and monumental scale is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch upper-mid glare in the return. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Electric guitar in Plate-like field**

- Intent: place the cabinet in a designed environment.
- Space character: fast diffusion with less geometric localization.
- Starting macros: room size 0.48, damping 0.62, diffusion 0.88, wet 0.46.
- Pre-delay working range: 6 to 30 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry electric guitar at a calibrated monitoring level. Establish the direct image first, then raise the wet return until fast diffusion with less geometric localization is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch upper-mid glare in the return. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Electric guitar in Reverse chamber**

- Intent: place the cabinet in a designed environment.
- Space character: a bounded reverse envelope with explicit latency.
- Starting macros: room size 0.50, damping 0.45, diffusion 0.70, wet 0.52.
- Pre-delay working range: 6 to 30 ms.
- Modes: Reverse on; Freeze off.

Begin with the dry electric guitar at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a bounded reverse envelope with explicit latency is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch upper-mid glare in the return. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Electric guitar in Frozen architecture**

- Intent: place the cabinet in a designed environment.
- Space character: a sustained field entered and exited safely.
- Starting macros: room size 0.72, damping 0.44, diffusion 0.82, wet 0.66.
- Pre-delay working range: 6 to 30 ms.
- Modes: Reverse off; Freeze prepared/on.

Begin with the dry electric guitar at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a sustained field entered and exited safely is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch upper-mid glare in the return. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

### 19.7 Strings

**Production card: Strings in Tight room**

- Intent: extend bow sustain and ensemble width.
- Space character: dense early cues and a controlled short tail.
- Starting macros: room size 0.28, damping 0.20, diffusion 0.52, wet 0.28.
- Pre-delay working range: 16 to 55 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry strings at a calibrated monitoring level. Establish the direct image first, then raise the wet return until dense early cues and a controlled short tail is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check section localization and high decay. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Strings in Studio chamber**

- Intent: extend bow sustain and ensemble width.
- Space character: a smooth useful chamber with moderate width.
- Starting macros: room size 0.42, damping 0.34, diffusion 0.64, wet 0.38.
- Pre-delay working range: 16 to 55 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry strings at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a smooth useful chamber with moderate width is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check section localization and high decay. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Strings in Scoring stage**

- Intent: extend bow sustain and ensemble width.
- Space character: clear source distance and a broad late field.
- Starting macros: room size 0.58, damping 0.46, diffusion 0.72, wet 0.44.
- Pre-delay working range: 16 to 55 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry strings at a calibrated monitoring level. Establish the direct image first, then raise the wet return until clear source distance and a broad late field is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check section localization and high decay. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Strings in Concert hall**

- Intent: extend bow sustain and ensemble width.
- Space character: a long integrated decay with stable imaging.
- Starting macros: room size 0.68, damping 0.52, diffusion 0.78, wet 0.50.
- Pre-delay working range: 16 to 55 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry strings at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a long integrated decay with stable imaging is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check section localization and high decay. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Strings in Stone cathedral**

- Intent: extend bow sustain and ensemble width.
- Space character: slow spectral decay and monumental scale.
- Starting macros: room size 0.78, damping 0.38, diffusion 0.84, wet 0.58.
- Pre-delay working range: 16 to 55 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry strings at a calibrated monitoring level. Establish the direct image first, then raise the wet return until slow spectral decay and monumental scale is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check section localization and high decay. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Strings in Plate-like field**

- Intent: extend bow sustain and ensemble width.
- Space character: fast diffusion with less geometric localization.
- Starting macros: room size 0.48, damping 0.62, diffusion 0.88, wet 0.46.
- Pre-delay working range: 16 to 55 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry strings at a calibrated monitoring level. Establish the direct image first, then raise the wet return until fast diffusion with less geometric localization is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check section localization and high decay. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Strings in Reverse chamber**

- Intent: extend bow sustain and ensemble width.
- Space character: a bounded reverse envelope with explicit latency.
- Starting macros: room size 0.50, damping 0.45, diffusion 0.70, wet 0.52.
- Pre-delay working range: 16 to 55 ms.
- Modes: Reverse on; Freeze off.

Begin with the dry strings at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a bounded reverse envelope with explicit latency is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check section localization and high decay. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Strings in Frozen architecture**

- Intent: extend bow sustain and ensemble width.
- Space character: a sustained field entered and exited safely.
- Starting macros: room size 0.72, damping 0.44, diffusion 0.82, wet 0.66.
- Pre-delay working range: 16 to 55 ms.
- Modes: Reverse off; Freeze prepared/on.

Begin with the dry strings at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a sustained field entered and exited safely is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check section localization and high decay. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

### 19.8 Synth pad

**Production card: Synth pad in Tight room**

- Intent: turn sustained harmony into an evolving field.
- Space character: dense early cues and a controlled short tail.
- Starting macros: room size 0.28, damping 0.20, diffusion 0.52, wet 0.28.
- Pre-delay working range: 0 to 40 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry synth pad at a calibrated monitoring level. Establish the direct image first, then raise the wet return until dense early cues and a controlled short tail is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch feedback energy and stereo correlation. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Synth pad in Studio chamber**

- Intent: turn sustained harmony into an evolving field.
- Space character: a smooth useful chamber with moderate width.
- Starting macros: room size 0.42, damping 0.34, diffusion 0.64, wet 0.38.
- Pre-delay working range: 0 to 40 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry synth pad at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a smooth useful chamber with moderate width is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch feedback energy and stereo correlation. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Synth pad in Scoring stage**

- Intent: turn sustained harmony into an evolving field.
- Space character: clear source distance and a broad late field.
- Starting macros: room size 0.58, damping 0.46, diffusion 0.72, wet 0.44.
- Pre-delay working range: 0 to 40 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry synth pad at a calibrated monitoring level. Establish the direct image first, then raise the wet return until clear source distance and a broad late field is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch feedback energy and stereo correlation. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Synth pad in Concert hall**

- Intent: turn sustained harmony into an evolving field.
- Space character: a long integrated decay with stable imaging.
- Starting macros: room size 0.68, damping 0.52, diffusion 0.78, wet 0.50.
- Pre-delay working range: 0 to 40 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry synth pad at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a long integrated decay with stable imaging is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch feedback energy and stereo correlation. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Synth pad in Stone cathedral**

- Intent: turn sustained harmony into an evolving field.
- Space character: slow spectral decay and monumental scale.
- Starting macros: room size 0.78, damping 0.38, diffusion 0.84, wet 0.58.
- Pre-delay working range: 0 to 40 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry synth pad at a calibrated monitoring level. Establish the direct image first, then raise the wet return until slow spectral decay and monumental scale is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch feedback energy and stereo correlation. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Synth pad in Plate-like field**

- Intent: turn sustained harmony into an evolving field.
- Space character: fast diffusion with less geometric localization.
- Starting macros: room size 0.48, damping 0.62, diffusion 0.88, wet 0.46.
- Pre-delay working range: 0 to 40 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry synth pad at a calibrated monitoring level. Establish the direct image first, then raise the wet return until fast diffusion with less geometric localization is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch feedback energy and stereo correlation. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Synth pad in Reverse chamber**

- Intent: turn sustained harmony into an evolving field.
- Space character: a bounded reverse envelope with explicit latency.
- Starting macros: room size 0.50, damping 0.45, diffusion 0.70, wet 0.52.
- Pre-delay working range: 0 to 40 ms.
- Modes: Reverse on; Freeze off.

Begin with the dry synth pad at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a bounded reverse envelope with explicit latency is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch feedback energy and stereo correlation. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Synth pad in Frozen architecture**

- Intent: turn sustained harmony into an evolving field.
- Space character: a sustained field entered and exited safely.
- Starting macros: room size 0.72, damping 0.44, diffusion 0.82, wet 0.66.
- Pre-delay working range: 0 to 40 ms.
- Modes: Reverse off; Freeze prepared/on.

Begin with the dry synth pad at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a sustained field entered and exited safely is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, watch feedback energy and stereo correlation. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

### 19.9 Percussion

**Production card: Percussion in Tight room**

- Intent: create rhythmic depth around short impulses.
- Space character: dense early cues and a controlled short tail.
- Starting macros: room size 0.28, damping 0.20, diffusion 0.52, wet 0.28.
- Pre-delay working range: 2 to 22 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry percussion at a calibrated monitoring level. Establish the direct image first, then raise the wet return until dense early cues and a controlled short tail is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check early reflections against tempo. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Percussion in Studio chamber**

- Intent: create rhythmic depth around short impulses.
- Space character: a smooth useful chamber with moderate width.
- Starting macros: room size 0.42, damping 0.34, diffusion 0.64, wet 0.38.
- Pre-delay working range: 2 to 22 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry percussion at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a smooth useful chamber with moderate width is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check early reflections against tempo. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Percussion in Scoring stage**

- Intent: create rhythmic depth around short impulses.
- Space character: clear source distance and a broad late field.
- Starting macros: room size 0.58, damping 0.46, diffusion 0.72, wet 0.44.
- Pre-delay working range: 2 to 22 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry percussion at a calibrated monitoring level. Establish the direct image first, then raise the wet return until clear source distance and a broad late field is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check early reflections against tempo. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Percussion in Concert hall**

- Intent: create rhythmic depth around short impulses.
- Space character: a long integrated decay with stable imaging.
- Starting macros: room size 0.68, damping 0.52, diffusion 0.78, wet 0.50.
- Pre-delay working range: 2 to 22 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry percussion at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a long integrated decay with stable imaging is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check early reflections against tempo. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Percussion in Stone cathedral**

- Intent: create rhythmic depth around short impulses.
- Space character: slow spectral decay and monumental scale.
- Starting macros: room size 0.78, damping 0.38, diffusion 0.84, wet 0.58.
- Pre-delay working range: 2 to 22 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry percussion at a calibrated monitoring level. Establish the direct image first, then raise the wet return until slow spectral decay and monumental scale is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check early reflections against tempo. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Percussion in Plate-like field**

- Intent: create rhythmic depth around short impulses.
- Space character: fast diffusion with less geometric localization.
- Starting macros: room size 0.48, damping 0.62, diffusion 0.88, wet 0.46.
- Pre-delay working range: 2 to 22 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry percussion at a calibrated monitoring level. Establish the direct image first, then raise the wet return until fast diffusion with less geometric localization is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check early reflections against tempo. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Percussion in Reverse chamber**

- Intent: create rhythmic depth around short impulses.
- Space character: a bounded reverse envelope with explicit latency.
- Starting macros: room size 0.50, damping 0.45, diffusion 0.70, wet 0.52.
- Pre-delay working range: 2 to 22 ms.
- Modes: Reverse on; Freeze off.

Begin with the dry percussion at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a bounded reverse envelope with explicit latency is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check early reflections against tempo. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Percussion in Frozen architecture**

- Intent: create rhythmic depth around short impulses.
- Space character: a sustained field entered and exited safely.
- Starting macros: room size 0.72, damping 0.44, diffusion 0.82, wet 0.66.
- Pre-delay working range: 2 to 22 ms.
- Modes: Reverse off; Freeze prepared/on.

Begin with the dry percussion at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a sustained field entered and exited safely is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, check early reflections against tempo. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

### 19.10 Field recording

**Production card: Field recording in Tight room**

- Intent: recontextualize a scene without losing its anchors.
- Space character: dense early cues and a controlled short tail.
- Starting macros: room size 0.28, damping 0.20, diffusion 0.52, wet 0.28.
- Pre-delay working range: 0 to 60 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry field recording at a calibrated monitoring level. Establish the direct image first, then raise the wet return until dense early cues and a controlled short tail is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, compare spectral floor and spatial plausibility. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Field recording in Studio chamber**

- Intent: recontextualize a scene without losing its anchors.
- Space character: a smooth useful chamber with moderate width.
- Starting macros: room size 0.42, damping 0.34, diffusion 0.64, wet 0.38.
- Pre-delay working range: 0 to 60 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry field recording at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a smooth useful chamber with moderate width is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, compare spectral floor and spatial plausibility. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Field recording in Scoring stage**

- Intent: recontextualize a scene without losing its anchors.
- Space character: clear source distance and a broad late field.
- Starting macros: room size 0.58, damping 0.46, diffusion 0.72, wet 0.44.
- Pre-delay working range: 0 to 60 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry field recording at a calibrated monitoring level. Establish the direct image first, then raise the wet return until clear source distance and a broad late field is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, compare spectral floor and spatial plausibility. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Field recording in Concert hall**

- Intent: recontextualize a scene without losing its anchors.
- Space character: a long integrated decay with stable imaging.
- Starting macros: room size 0.68, damping 0.52, diffusion 0.78, wet 0.50.
- Pre-delay working range: 0 to 60 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry field recording at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a long integrated decay with stable imaging is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, compare spectral floor and spatial plausibility. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Field recording in Stone cathedral**

- Intent: recontextualize a scene without losing its anchors.
- Space character: slow spectral decay and monumental scale.
- Starting macros: room size 0.78, damping 0.38, diffusion 0.84, wet 0.58.
- Pre-delay working range: 0 to 60 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry field recording at a calibrated monitoring level. Establish the direct image first, then raise the wet return until slow spectral decay and monumental scale is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, compare spectral floor and spatial plausibility. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Field recording in Plate-like field**

- Intent: recontextualize a scene without losing its anchors.
- Space character: fast diffusion with less geometric localization.
- Starting macros: room size 0.48, damping 0.62, diffusion 0.88, wet 0.46.
- Pre-delay working range: 0 to 60 ms.
- Modes: Reverse off; Freeze off.

Begin with the dry field recording at a calibrated monitoring level. Establish the direct image first, then raise the wet return until fast diffusion with less geometric localization is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, compare spectral floor and spatial plausibility. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Field recording in Reverse chamber**

- Intent: recontextualize a scene without losing its anchors.
- Space character: a bounded reverse envelope with explicit latency.
- Starting macros: room size 0.50, damping 0.45, diffusion 0.70, wet 0.52.
- Pre-delay working range: 0 to 60 ms.
- Modes: Reverse on; Freeze off.

Begin with the dry field recording at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a bounded reverse envelope with explicit latency is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, compare spectral floor and spatial plausibility. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

**Production card: Field recording in Frozen architecture**

- Intent: recontextualize a scene without losing its anchors.
- Space character: a sustained field entered and exited safely.
- Starting macros: room size 0.72, damping 0.44, diffusion 0.82, wet 0.66.
- Pre-delay working range: 0 to 60 ms.
- Modes: Reverse off; Freeze prepared/on.

Begin with the dry field recording at a calibrated monitoring level. Establish the direct image first, then raise the wet return until a sustained field entered and exited safely is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.

For this source, compare spectral floor and spatial plausibility. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.

Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.


\newpage

## 20. Automation Study Cards

### 20.1 Pre-Delay

**Automation card: Pre-Delay: Slow rise**

- Host parameter: `pre_delay_ms`.
- Motion: move from the lower setting to the upper setting over eight or more bars.
- Primary observation: separates the direct event from the room onset.
- Transition requirement: use a ramp or delay-line crossfade.

Write the slow rise move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement reveals zipper noise and coefficient discontinuities. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Pre-Delay: Slow fall**

- Host parameter: `pre_delay_ms`.
- Motion: return gradually toward the dry or compact state.
- Primary observation: separates the direct event from the room onset.
- Transition requirement: use a ramp or delay-line crossfade.

Write the slow fall move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests whether stored energy decays naturally. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Pre-Delay: Tempo pulse**

- Host parameter: `pre_delay_ms`.
- Motion: alternate two musically useful values on a bar or phrase boundary.
- Primary observation: separates the direct event from the room onset.
- Transition requirement: use a ramp or delay-line crossfade.

Write the tempo pulse move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests repeatability and transition timing. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Pre-Delay: Scene switch**

- Host parameter: `pre_delay_ms`.
- Motion: change once at an arrangement boundary and hold.
- Primary observation: separates the direct event from the room onset.
- Transition requirement: use a ramp or delay-line crossfade.

Write the scene switch move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests state recall and discrete transition behavior. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

### 20.2 Room Size

**Automation card: Room Size: Slow rise**

- Host parameter: `room_size`.
- Motion: move from the lower setting to the upper setting over eight or more bars.
- Primary observation: changes perceived scale and reflection spacing.
- Transition requirement: stage structural changes outside the callback.

Write the slow rise move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement reveals zipper noise and coefficient discontinuities. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Room Size: Slow fall**

- Host parameter: `room_size`.
- Motion: return gradually toward the dry or compact state.
- Primary observation: changes perceived scale and reflection spacing.
- Transition requirement: stage structural changes outside the callback.

Write the slow fall move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests whether stored energy decays naturally. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Room Size: Tempo pulse**

- Host parameter: `room_size`.
- Motion: alternate two musically useful values on a bar or phrase boundary.
- Primary observation: changes perceived scale and reflection spacing.
- Transition requirement: stage structural changes outside the callback.

Write the tempo pulse move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests repeatability and transition timing. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Room Size: Scene switch**

- Host parameter: `room_size`.
- Motion: change once at an arrangement boundary and hold.
- Primary observation: changes perceived scale and reflection spacing.
- Transition requirement: stage structural changes outside the callback.

Write the scene switch move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests state recall and discrete transition behavior. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

### 20.3 RT60 Coarse

**Automation card: RT60 Coarse: Slow rise**

- Host parameter: `rt60_coarse`.
- Motion: move from the lower setting to the upper setting over eight or more bars.
- Primary observation: moves through the full logarithmic decay range.
- Transition requirement: display the effective seconds value.

Write the slow rise move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement reveals zipper noise and coefficient discontinuities. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: RT60 Coarse: Slow fall**

- Host parameter: `rt60_coarse`.
- Motion: return gradually toward the dry or compact state.
- Primary observation: moves through the full logarithmic decay range.
- Transition requirement: display the effective seconds value.

Write the slow fall move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests whether stored energy decays naturally. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: RT60 Coarse: Tempo pulse**

- Host parameter: `rt60_coarse`.
- Motion: alternate two musically useful values on a bar or phrase boundary.
- Primary observation: moves through the full logarithmic decay range.
- Transition requirement: display the effective seconds value.

Write the tempo pulse move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests repeatability and transition timing. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: RT60 Coarse: Scene switch**

- Host parameter: `rt60_coarse`.
- Motion: change once at an arrangement boundary and hold.
- Primary observation: moves through the full logarithmic decay range.
- Transition requirement: display the effective seconds value.

Write the scene switch move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests state recall and discrete transition behavior. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

### 20.4 RT60 Fine

**Automation card: RT60 Fine: Slow rise**

- Host parameter: `rt60_fine`.
- Motion: move from the lower setting to the upper setting over eight or more bars.
- Primary observation: trims decay proportionally around the coarse value.
- Transition requirement: keep zero as the exact neutral point.

Write the slow rise move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement reveals zipper noise and coefficient discontinuities. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: RT60 Fine: Slow fall**

- Host parameter: `rt60_fine`.
- Motion: return gradually toward the dry or compact state.
- Primary observation: trims decay proportionally around the coarse value.
- Transition requirement: keep zero as the exact neutral point.

Write the slow fall move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests whether stored energy decays naturally. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: RT60 Fine: Tempo pulse**

- Host parameter: `rt60_fine`.
- Motion: alternate two musically useful values on a bar or phrase boundary.
- Primary observation: trims decay proportionally around the coarse value.
- Transition requirement: keep zero as the exact neutral point.

Write the tempo pulse move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests repeatability and transition timing. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: RT60 Fine: Scene switch**

- Host parameter: `rt60_fine`.
- Motion: change once at an arrangement boundary and hold.
- Primary observation: trims decay proportionally around the coarse value.
- Transition requirement: keep zero as the exact neutral point.

Write the scene switch move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests state recall and discrete transition behavior. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

### 20.5 Damping

**Automation card: Damping: Slow rise**

- Host parameter: `damping`.
- Motion: move from the lower setting to the upper setting over eight or more bars.
- Primary observation: changes high-frequency persistence.
- Transition requirement: interpolate stable filter coefficients.

Write the slow rise move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement reveals zipper noise and coefficient discontinuities. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Damping: Slow fall**

- Host parameter: `damping`.
- Motion: return gradually toward the dry or compact state.
- Primary observation: changes high-frequency persistence.
- Transition requirement: interpolate stable filter coefficients.

Write the slow fall move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests whether stored energy decays naturally. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Damping: Tempo pulse**

- Host parameter: `damping`.
- Motion: alternate two musically useful values on a bar or phrase boundary.
- Primary observation: changes high-frequency persistence.
- Transition requirement: interpolate stable filter coefficients.

Write the tempo pulse move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests repeatability and transition timing. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Damping: Scene switch**

- Host parameter: `damping`.
- Motion: change once at an arrangement boundary and hold.
- Primary observation: changes high-frequency persistence.
- Transition requirement: interpolate stable filter coefficients.

Write the scene switch move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests state recall and discrete transition behavior. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

### 20.6 Width

**Automation card: Width: Slow rise**

- Host parameter: `width`.
- Motion: move from the lower setting to the upper setting over eight or more bars.
- Primary observation: changes lateral energy and correlation.
- Transition requirement: monitor mono compatibility during movement.

Write the slow rise move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement reveals zipper noise and coefficient discontinuities. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Width: Slow fall**

- Host parameter: `width`.
- Motion: return gradually toward the dry or compact state.
- Primary observation: changes lateral energy and correlation.
- Transition requirement: monitor mono compatibility during movement.

Write the slow fall move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests whether stored energy decays naturally. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Width: Tempo pulse**

- Host parameter: `width`.
- Motion: alternate two musically useful values on a bar or phrase boundary.
- Primary observation: changes lateral energy and correlation.
- Transition requirement: monitor mono compatibility during movement.

Write the tempo pulse move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests repeatability and transition timing. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Width: Scene switch**

- Host parameter: `width`.
- Motion: change once at an arrangement boundary and hold.
- Primary observation: changes lateral energy and correlation.
- Transition requirement: monitor mono compatibility during movement.

Write the scene switch move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests state recall and discrete transition behavior. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

### 20.7 Diffusion

**Automation card: Diffusion: Slow rise**

- Host parameter: `diffusion`.
- Motion: move from the lower setting to the upper setting over eight or more bars.
- Primary observation: changes echo-density buildup.
- Transition requirement: crossfade when topology must change.

Write the slow rise move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement reveals zipper noise and coefficient discontinuities. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Diffusion: Slow fall**

- Host parameter: `diffusion`.
- Motion: return gradually toward the dry or compact state.
- Primary observation: changes echo-density buildup.
- Transition requirement: crossfade when topology must change.

Write the slow fall move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests whether stored energy decays naturally. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Diffusion: Tempo pulse**

- Host parameter: `diffusion`.
- Motion: alternate two musically useful values on a bar or phrase boundary.
- Primary observation: changes echo-density buildup.
- Transition requirement: crossfade when topology must change.

Write the tempo pulse move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests repeatability and transition timing. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Diffusion: Scene switch**

- Host parameter: `diffusion`.
- Motion: change once at an arrangement boundary and hold.
- Primary observation: changes echo-density buildup.
- Transition requirement: crossfade when topology must change.

Write the scene switch move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests state recall and discrete transition behavior. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

### 20.8 Wet

**Automation card: Wet: Slow rise**

- Host parameter: `wet`.
- Motion: move from the lower setting to the upper setting over eight or more bars.
- Primary observation: sets processed contribution.
- Transition requirement: choose and document the mix law.

Write the slow rise move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement reveals zipper noise and coefficient discontinuities. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Wet: Slow fall**

- Host parameter: `wet`.
- Motion: return gradually toward the dry or compact state.
- Primary observation: sets processed contribution.
- Transition requirement: choose and document the mix law.

Write the slow fall move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests whether stored energy decays naturally. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Wet: Tempo pulse**

- Host parameter: `wet`.
- Motion: alternate two musically useful values on a bar or phrase boundary.
- Primary observation: sets processed contribution.
- Transition requirement: choose and document the mix law.

Write the tempo pulse move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests repeatability and transition timing. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Wet: Scene switch**

- Host parameter: `wet`.
- Motion: change once at an arrangement boundary and hold.
- Primary observation: sets processed contribution.
- Transition requirement: choose and document the mix law.

Write the scene switch move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests state recall and discrete transition behavior. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

### 20.9 Dry

**Automation card: Dry: Slow rise**

- Host parameter: `dry`.
- Motion: move from the lower setting to the upper setting over eight or more bars.
- Primary observation: sets direct contribution.
- Transition requirement: preserve bypass and gain staging.

Write the slow rise move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement reveals zipper noise and coefficient discontinuities. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Dry: Slow fall**

- Host parameter: `dry`.
- Motion: return gradually toward the dry or compact state.
- Primary observation: sets direct contribution.
- Transition requirement: preserve bypass and gain staging.

Write the slow fall move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests whether stored energy decays naturally. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Dry: Tempo pulse**

- Host parameter: `dry`.
- Motion: alternate two musically useful values on a bar or phrase boundary.
- Primary observation: sets direct contribution.
- Transition requirement: preserve bypass and gain staging.

Write the tempo pulse move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests repeatability and transition timing. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Dry: Scene switch**

- Host parameter: `dry`.
- Motion: change once at an arrangement boundary and hold.
- Primary observation: sets direct contribution.
- Transition requirement: preserve bypass and gain staging.

Write the scene switch move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests state recall and discrete transition behavior. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

### 20.10 Freeze

**Automation card: Freeze: Slow rise**

- Host parameter: `freeze`.
- Motion: move from the lower setting to the upper setting over eight or more bars.
- Primary observation: changes network energy behavior.
- Transition requirement: use a debounced, smoothed mode transition.

Write the slow rise move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement reveals zipper noise and coefficient discontinuities. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Freeze: Slow fall**

- Host parameter: `freeze`.
- Motion: return gradually toward the dry or compact state.
- Primary observation: changes network energy behavior.
- Transition requirement: use a debounced, smoothed mode transition.

Write the slow fall move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests whether stored energy decays naturally. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Freeze: Tempo pulse**

- Host parameter: `freeze`.
- Motion: alternate two musically useful values on a bar or phrase boundary.
- Primary observation: changes network energy behavior.
- Transition requirement: use a debounced, smoothed mode transition.

Write the tempo pulse move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests repeatability and transition timing. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Freeze: Scene switch**

- Host parameter: `freeze`.
- Motion: change once at an arrangement boundary and hold.
- Primary observation: changes network energy behavior.
- Transition requirement: use a debounced, smoothed mode transition.

Write the scene switch move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests state recall and discrete transition behavior. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

### 20.11 Reverse

**Automation card: Reverse: Slow rise**

- Host parameter: `reverse`.
- Motion: move from the lower setting to the upper setting over eight or more bars.
- Primary observation: changes the envelope and buffering model.
- Transition requirement: report added latency before activation.

Write the slow rise move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement reveals zipper noise and coefficient discontinuities. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Reverse: Slow fall**

- Host parameter: `reverse`.
- Motion: return gradually toward the dry or compact state.
- Primary observation: changes the envelope and buffering model.
- Transition requirement: report added latency before activation.

Write the slow fall move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests whether stored energy decays naturally. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Reverse: Tempo pulse**

- Host parameter: `reverse`.
- Motion: alternate two musically useful values on a bar or phrase boundary.
- Primary observation: changes the envelope and buffering model.
- Transition requirement: report added latency before activation.

Write the tempo pulse move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests repeatability and transition timing. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Reverse: Scene switch**

- Host parameter: `reverse`.
- Motion: change once at an arrangement boundary and hold.
- Primary observation: changes the envelope and buffering model.
- Transition requirement: report added latency before activation.

Write the scene switch move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests state recall and discrete transition behavior. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

### 20.12 Quality

**Automation card: Quality: Slow rise**

- Host parameter: `quality_mode`.
- Motion: move from the lower setting to the upper setting over eight or more bars.
- Primary observation: selects the internal rate policy.
- Transition requirement: apply through a safe reprepare boundary.

Write the slow rise move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement reveals zipper noise and coefficient discontinuities. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Quality: Slow fall**

- Host parameter: `quality_mode`.
- Motion: return gradually toward the dry or compact state.
- Primary observation: selects the internal rate policy.
- Transition requirement: apply through a safe reprepare boundary.

Write the slow fall move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests whether stored energy decays naturally. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Quality: Tempo pulse**

- Host parameter: `quality_mode`.
- Motion: alternate two musically useful values on a bar or phrase boundary.
- Primary observation: selects the internal rate policy.
- Transition requirement: apply through a safe reprepare boundary.

Write the tempo pulse move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests repeatability and transition timing. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

**Automation card: Quality: Scene switch**

- Host parameter: `quality_mode`.
- Motion: change once at an arrangement boundary and hold.
- Primary observation: selects the internal rate policy.
- Transition requirement: apply through a safe reprepare boundary.

Write the scene switch move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.

Listen at the beginning, during the transition, and after the value settles. This movement tests state recall and discrete transition behavior. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.

Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.


\newpage

## 21. Quality And Latency Cards

**Quality card 1: 44100 Hz, Host, 64 frames**

- Host rate: 44100 Hz.
- Quality policy: Host: no intentional internal rate multiplication.
- Expected internal-rate contract: 44100 Hz.
- Host block duration: 1.451 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 2: 44100 Hz, Host, 512 frames**

- Host rate: 44100 Hz.
- Quality policy: Host: no intentional internal rate multiplication.
- Expected internal-rate contract: 44100 Hz.
- Host block duration: 11.610 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 3: 44100 Hz, 2x, 64 frames**

- Host rate: 44100 Hz.
- Quality policy: 2x: twice the host rate.
- Expected internal-rate contract: 88200 Hz.
- Host block duration: 1.451 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 4: 44100 Hz, 2x, 512 frames**

- Host rate: 44100 Hz.
- Quality policy: 2x: twice the host rate.
- Expected internal-rate contract: 88200 Hz.
- Host block duration: 11.610 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 5: 44100 Hz, 4x, 64 frames**

- Host rate: 44100 Hz.
- Quality policy: 4x: four times the host rate.
- Expected internal-rate contract: 176400 Hz.
- Host block duration: 1.451 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 6: 44100 Hz, 4x, 512 frames**

- Host rate: 44100 Hz.
- Quality policy: 4x: four times the host rate.
- Expected internal-rate contract: 176400 Hz.
- Host block duration: 11.610 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 7: 44100 Hz, Target 192 kHz, 64 frames**

- Host rate: 44100 Hz.
- Quality policy: Target 192 kHz: the smallest integer factor reaching at least 192 kHz.
- Expected internal-rate contract: 220500 Hz.
- Host block duration: 1.451 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 8: 44100 Hz, Target 192 kHz, 512 frames**

- Host rate: 44100 Hz.
- Quality policy: Target 192 kHz: the smallest integer factor reaching at least 192 kHz.
- Expected internal-rate contract: 220500 Hz.
- Host block duration: 11.610 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 9: 48000 Hz, Host, 64 frames**

- Host rate: 48000 Hz.
- Quality policy: Host: no intentional internal rate multiplication.
- Expected internal-rate contract: 48000 Hz.
- Host block duration: 1.333 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 10: 48000 Hz, Host, 512 frames**

- Host rate: 48000 Hz.
- Quality policy: Host: no intentional internal rate multiplication.
- Expected internal-rate contract: 48000 Hz.
- Host block duration: 10.667 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 11: 48000 Hz, 2x, 64 frames**

- Host rate: 48000 Hz.
- Quality policy: 2x: twice the host rate.
- Expected internal-rate contract: 96000 Hz.
- Host block duration: 1.333 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 12: 48000 Hz, 2x, 512 frames**

- Host rate: 48000 Hz.
- Quality policy: 2x: twice the host rate.
- Expected internal-rate contract: 96000 Hz.
- Host block duration: 10.667 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 13: 48000 Hz, 4x, 64 frames**

- Host rate: 48000 Hz.
- Quality policy: 4x: four times the host rate.
- Expected internal-rate contract: 192000 Hz.
- Host block duration: 1.333 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 14: 48000 Hz, 4x, 512 frames**

- Host rate: 48000 Hz.
- Quality policy: 4x: four times the host rate.
- Expected internal-rate contract: 192000 Hz.
- Host block duration: 10.667 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 15: 48000 Hz, Target 192 kHz, 64 frames**

- Host rate: 48000 Hz.
- Quality policy: Target 192 kHz: the smallest integer factor reaching at least 192 kHz.
- Expected internal-rate contract: 192000 Hz.
- Host block duration: 1.333 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 16: 48000 Hz, Target 192 kHz, 512 frames**

- Host rate: 48000 Hz.
- Quality policy: Target 192 kHz: the smallest integer factor reaching at least 192 kHz.
- Expected internal-rate contract: 192000 Hz.
- Host block duration: 10.667 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 17: 96000 Hz, Host, 64 frames**

- Host rate: 96000 Hz.
- Quality policy: Host: no intentional internal rate multiplication.
- Expected internal-rate contract: 96000 Hz.
- Host block duration: 0.667 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 18: 96000 Hz, Host, 512 frames**

- Host rate: 96000 Hz.
- Quality policy: Host: no intentional internal rate multiplication.
- Expected internal-rate contract: 96000 Hz.
- Host block duration: 5.333 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 19: 96000 Hz, 2x, 64 frames**

- Host rate: 96000 Hz.
- Quality policy: 2x: twice the host rate.
- Expected internal-rate contract: 192000 Hz.
- Host block duration: 0.667 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 20: 96000 Hz, 2x, 512 frames**

- Host rate: 96000 Hz.
- Quality policy: 2x: twice the host rate.
- Expected internal-rate contract: 192000 Hz.
- Host block duration: 5.333 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 21: 96000 Hz, 4x, 64 frames**

- Host rate: 96000 Hz.
- Quality policy: 4x: four times the host rate.
- Expected internal-rate contract: 384000 Hz.
- Host block duration: 0.667 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 22: 96000 Hz, 4x, 512 frames**

- Host rate: 96000 Hz.
- Quality policy: 4x: four times the host rate.
- Expected internal-rate contract: 384000 Hz.
- Host block duration: 5.333 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 23: 96000 Hz, Target 192 kHz, 64 frames**

- Host rate: 96000 Hz.
- Quality policy: Target 192 kHz: the smallest integer factor reaching at least 192 kHz.
- Expected internal-rate contract: 192000 Hz.
- Host block duration: 0.667 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 24: 96000 Hz, Target 192 kHz, 512 frames**

- Host rate: 96000 Hz.
- Quality policy: Target 192 kHz: the smallest integer factor reaching at least 192 kHz.
- Expected internal-rate contract: 192000 Hz.
- Host block duration: 5.333 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 25: 192000 Hz, Host, 64 frames**

- Host rate: 192000 Hz.
- Quality policy: Host: no intentional internal rate multiplication.
- Expected internal-rate contract: 192000 Hz.
- Host block duration: 0.333 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 26: 192000 Hz, Host, 512 frames**

- Host rate: 192000 Hz.
- Quality policy: Host: no intentional internal rate multiplication.
- Expected internal-rate contract: 192000 Hz.
- Host block duration: 2.667 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 27: 192000 Hz, 2x, 64 frames**

- Host rate: 192000 Hz.
- Quality policy: 2x: twice the host rate.
- Expected internal-rate contract: 384000 Hz.
- Host block duration: 0.333 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 28: 192000 Hz, 2x, 512 frames**

- Host rate: 192000 Hz.
- Quality policy: 2x: twice the host rate.
- Expected internal-rate contract: 384000 Hz.
- Host block duration: 2.667 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 29: 192000 Hz, 4x, 64 frames**

- Host rate: 192000 Hz.
- Quality policy: 4x: four times the host rate.
- Expected internal-rate contract: 768000 Hz.
- Host block duration: 0.333 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 30: 192000 Hz, 4x, 512 frames**

- Host rate: 192000 Hz.
- Quality policy: 4x: four times the host rate.
- Expected internal-rate contract: 768000 Hz.
- Host block duration: 2.667 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 31: 192000 Hz, Target 192 kHz, 64 frames**

- Host rate: 192000 Hz.
- Quality policy: Target 192 kHz: the smallest integer factor reaching at least 192 kHz.
- Expected internal-rate contract: 192000 Hz.
- Host block duration: 0.333 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

**Quality card 32: 192000 Hz, Target 192 kHz, 512 frames**

- Host rate: 192000 Hz.
- Quality policy: Target 192 kHz: the smallest integer factor reaching at least 192 kHz.
- Expected internal-rate contract: 192000 Hz.
- Host block duration: 2.667 ms before device and plug-in latency.

Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.

Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.

Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.


\newpage

## 22. Host Validation Cards

### 22.1 Scan and instantiate

**Validation card: Standalone: Scan and instantiate**

- Surface: the JUCE standalone wrapper.
- Goal: confirm the format scans, loads, and creates a stable processor/editor pair.
- Context emphasis: device setup and callback behavior without DAW compensation.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the scan and instantiate procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: Desktop AU: Scan and instantiate**

- Surface: an Audio Unit host.
- Goal: confirm the format scans, loads, and creates a stable processor/editor pair.
- Context emphasis: Apple scanning, state, buses, and latency notification.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the scan and instantiate procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: AUv3: Scan and instantiate**

- Surface: an AUv3-capable sandboxed host.
- Goal: confirm the format scans, loads, and creates a stable processor/editor pair.
- Context emphasis: sandbox lifecycle, resources, and compact-window behavior.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the scan and instantiate procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: VST3: Scan and instantiate**

- Surface: a VST3 host.
- Goal: confirm the format scans, loads, and creates a stable processor/editor pair.
- Context emphasis: component/controller state, scanning, and automation identity.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the scan and instantiate procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

### 22.2 Parameter automation

**Validation card: Standalone: Parameter automation**

- Surface: the JUCE standalone wrapper.
- Goal: write, read, trim, suspend, and replay every exposed parameter.
- Context emphasis: device setup and callback behavior without DAW compensation.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the parameter automation procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: Desktop AU: Parameter automation**

- Surface: an Audio Unit host.
- Goal: write, read, trim, suspend, and replay every exposed parameter.
- Context emphasis: Apple scanning, state, buses, and latency notification.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the parameter automation procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: AUv3: Parameter automation**

- Surface: an AUv3-capable sandboxed host.
- Goal: write, read, trim, suspend, and replay every exposed parameter.
- Context emphasis: sandbox lifecycle, resources, and compact-window behavior.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the parameter automation procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: VST3: Parameter automation**

- Surface: a VST3 host.
- Goal: write, read, trim, suspend, and replay every exposed parameter.
- Context emphasis: component/controller state, scanning, and automation identity.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the parameter automation procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

### 22.3 State recall

**Validation card: Standalone: State recall**

- Surface: the JUCE standalone wrapper.
- Goal: save, close, reopen, and compare all parameters and asset identities.
- Context emphasis: device setup and callback behavior without DAW compensation.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the state recall procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: Desktop AU: State recall**

- Surface: an Audio Unit host.
- Goal: save, close, reopen, and compare all parameters and asset identities.
- Context emphasis: Apple scanning, state, buses, and latency notification.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the state recall procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: AUv3: State recall**

- Surface: an AUv3-capable sandboxed host.
- Goal: save, close, reopen, and compare all parameters and asset identities.
- Context emphasis: sandbox lifecycle, resources, and compact-window behavior.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the state recall procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: VST3: State recall**

- Surface: a VST3 host.
- Goal: save, close, reopen, and compare all parameters and asset identities.
- Context emphasis: component/controller state, scanning, and automation identity.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the state recall procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

### 22.4 Latency compensation

**Validation card: Standalone: Latency compensation**

- Surface: the JUCE standalone wrapper.
- Goal: measure impulse alignment and compare it with the reported frame count.
- Context emphasis: device setup and callback behavior without DAW compensation.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the latency compensation procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: Desktop AU: Latency compensation**

- Surface: an Audio Unit host.
- Goal: measure impulse alignment and compare it with the reported frame count.
- Context emphasis: Apple scanning, state, buses, and latency notification.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the latency compensation procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: AUv3: Latency compensation**

- Surface: an AUv3-capable sandboxed host.
- Goal: measure impulse alignment and compare it with the reported frame count.
- Context emphasis: sandbox lifecycle, resources, and compact-window behavior.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the latency compensation procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: VST3: Latency compensation**

- Surface: a VST3 host.
- Goal: measure impulse alignment and compare it with the reported frame count.
- Context emphasis: component/controller state, scanning, and automation identity.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the latency compensation procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

### 22.5 Bus negotiation

**Validation card: Standalone: Bus negotiation**

- Surface: the JUCE standalone wrapper.
- Goal: exercise supported mono/stereo layouts and reject unsupported layouts clearly.
- Context emphasis: device setup and callback behavior without DAW compensation.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the bus negotiation procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: Desktop AU: Bus negotiation**

- Surface: an Audio Unit host.
- Goal: exercise supported mono/stereo layouts and reject unsupported layouts clearly.
- Context emphasis: Apple scanning, state, buses, and latency notification.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the bus negotiation procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: AUv3: Bus negotiation**

- Surface: an AUv3-capable sandboxed host.
- Goal: exercise supported mono/stereo layouts and reject unsupported layouts clearly.
- Context emphasis: sandbox lifecycle, resources, and compact-window behavior.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the bus negotiation procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: VST3: Bus negotiation**

- Surface: a VST3 host.
- Goal: exercise supported mono/stereo layouts and reject unsupported layouts clearly.
- Context emphasis: component/controller state, scanning, and automation identity.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the bus negotiation procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

### 22.6 Transport changes

**Validation card: Standalone: Transport changes**

- Surface: the JUCE standalone wrapper.
- Goal: start, stop, loop, seek, and change tempo without corrupting the tail.
- Context emphasis: device setup and callback behavior without DAW compensation.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the transport changes procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: Desktop AU: Transport changes**

- Surface: an Audio Unit host.
- Goal: start, stop, loop, seek, and change tempo without corrupting the tail.
- Context emphasis: Apple scanning, state, buses, and latency notification.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the transport changes procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: AUv3: Transport changes**

- Surface: an AUv3-capable sandboxed host.
- Goal: start, stop, loop, seek, and change tempo without corrupting the tail.
- Context emphasis: sandbox lifecycle, resources, and compact-window behavior.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the transport changes procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: VST3: Transport changes**

- Surface: a VST3 host.
- Goal: start, stop, loop, seek, and change tempo without corrupting the tail.
- Context emphasis: component/controller state, scanning, and automation identity.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the transport changes procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

### 22.7 Sample-rate changes

**Validation card: Standalone: Sample-rate changes**

- Surface: the JUCE standalone wrapper.
- Goal: reprepare at each supported host rate without stale buffers or status.
- Context emphasis: device setup and callback behavior without DAW compensation.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the sample-rate changes procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: Desktop AU: Sample-rate changes**

- Surface: an Audio Unit host.
- Goal: reprepare at each supported host rate without stale buffers or status.
- Context emphasis: Apple scanning, state, buses, and latency notification.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the sample-rate changes procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: AUv3: Sample-rate changes**

- Surface: an AUv3-capable sandboxed host.
- Goal: reprepare at each supported host rate without stale buffers or status.
- Context emphasis: sandbox lifecycle, resources, and compact-window behavior.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the sample-rate changes procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: VST3: Sample-rate changes**

- Surface: a VST3 host.
- Goal: reprepare at each supported host rate without stale buffers or status.
- Context emphasis: component/controller state, scanning, and automation identity.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the sample-rate changes procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

### 22.8 Editor lifecycle

**Validation card: Standalone: Editor lifecycle**

- Surface: the JUCE standalone wrapper.
- Goal: open, resize, close, and reopen the editor while audio remains unchanged.
- Context emphasis: device setup and callback behavior without DAW compensation.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the editor lifecycle procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: Desktop AU: Editor lifecycle**

- Surface: an Audio Unit host.
- Goal: open, resize, close, and reopen the editor while audio remains unchanged.
- Context emphasis: Apple scanning, state, buses, and latency notification.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the editor lifecycle procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: AUv3: Editor lifecycle**

- Surface: an AUv3-capable sandboxed host.
- Goal: open, resize, close, and reopen the editor while audio remains unchanged.
- Context emphasis: sandbox lifecycle, resources, and compact-window behavior.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the editor lifecycle procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

**Validation card: VST3: Editor lifecycle**

- Surface: a VST3 host.
- Goal: open, resize, close, and reopen the editor while audio remains unchanged.
- Context emphasis: component/controller state, scanning, and automation identity.
- Status: protocol only until a dated result is recorded.

Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the editor lifecycle procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.

Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.

Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.


\newpage

## 23. Troubleshooting Cards

**Troubleshooting card 1: The plug-in does not appear after scanning**

- Symptom: The plug-in does not appear after scanning.
- Likely causes: the binary is in the wrong format location, failed validation, or was built for the wrong architecture.
- First recovery: inspect the host scan log, confirm architecture and format, then rescan a clean build.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 2: The editor opens but audio is dry**

- Symptom: The editor opens but audio is dry.
- Likely causes: wet gain is down, routing state is stale, or the prepared DSP context failed.
- First recovery: confirm Wet/Dry values, live internal-rate status, and host logs before treating this as a scanner fault.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 3: Automation recalls the wrong control**

- Symptom: Automation recalls the wrong control.
- Likely causes: a parameter ID or version changed after a session was saved.
- First recovery: compare manifest IDs and restore stable identifiers; never repair this by reordering blindly.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 4: A preset opens with missing geometry**

- Symptom: A preset opens with missing geometry.
- Likely causes: the referenced DXF/profile asset moved or its hash changed.
- First recovery: locate the exact asset, verify its hash, or use the stored bounded fallback profile.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 5: CPU rises sharply at 48 kHz**

- Symptom: CPU rises sharply at 48 kHz.
- Likely causes: Target 192 kHz implies a 4x internal-rate goal.
- First recovery: compare Host and 2x modes, increase block size, and record the quality tradeoff.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 6: CPU rises at 192 kHz**

- Symptom: CPU rises at 192 kHz.
- Likely causes: the host is already processing a very high sample rate.
- First recovery: use Host mode and reduce expensive topology before changing musical controls.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 7: The host reports no latency**

- Symptom: The host reports no latency.
- Likely causes: the current causal reverb/oversampling graph is intentionally zero-lookahead or a later buffering graph did not notify the host.
- First recovery: measure with an impulse and compare the result with the status accessor.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 8: Reverse feels late**

- Symptom: Reverse feels late.
- Likely causes: reverse processing requires a capture or lookahead window.
- First recovery: verify the declared reverse window and host delay compensation.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 9: Freeze gets louder over time**

- Symptom: Freeze gets louder over time.
- Likely causes: feedback energy is above a stable bound or input injection remains active.
- First recovery: disengage safely, lower the stored energy, and inspect freeze gain and limiter reduction.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 10: Freeze clicks when toggled**

- Symptom: Freeze clicks when toggled.
- Likely causes: the mode changes coefficients or injection abruptly.
- First recovery: add a state transition ramp or dual-path crossfade and retest at full-scale impulses.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 11: The tail disappears on bypass**

- Symptom: The tail disappears on bypass.
- Likely causes: the host uses hard bypass and skips processing.
- First recovery: offer a tail-preserving effect bypass parameter and document the host bypass behavior.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 12: The image collapses in mono**

- Symptom: The image collapses in mono.
- Likely causes: width or decorrelation has produced excessive anti-correlation.
- First recovery: reduce width, check early/late balance, and validate the mono sum.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 13: The output contains denormal CPU spikes**

- Symptom: The output contains denormal CPU spikes.
- Likely causes: very small feedback values are entering slow floating-point paths.
- First recovery: enable denormal protection and test long decays into digital silence.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 14: The output contains NaN or Inf**

- Symptom: The output contains NaN or Inf.
- Likely causes: unstable feedback, invalid coefficients, or corrupted state reached the DSP.
- First recovery: mute safely, capture diagnostics, clamp inputs, and fix the originating invariant.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 15: Changing room size causes a click**

- Symptom: Changing room size causes a click.
- Likely causes: delay topology changed without a transition.
- First recovery: prepare the new network off-thread and crossfade bounded states.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 16: Changing quality interrupts playback**

- Symptom: Changing quality interrupts playback.
- Likely causes: the internal processing graph requires reprepare.
- First recovery: apply quality changes at a declared safe boundary and show pending status.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 17: Meters freeze when the editor closes**

- Symptom: Meters freeze when the editor closes.
- Likely causes: telemetry ownership is incorrectly coupled to the editor.
- First recovery: keep DSP telemetry independent and let the editor subscribe only while visible.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 18: Closing the editor changes CPU**

- Symptom: Closing the editor changes CPU.
- Likely causes: visualization work or DSP ownership is attached to editor lifetime.
- First recovery: separate processor state from the editor and repeat the lifecycle test.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 19: A DAW project reopens silently**

- Symptom: A DAW project reopens silently.
- Likely causes: state restoration failed or a required asset was unavailable.
- First recovery: load a safe default audibly, show a blocking status, and preserve diagnostic metadata.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 20: Wet and dry at unity clip**

- Symptom: Wet and dry at unity clip.
- Likely causes: the mix law sums correlated paths above full scale.
- First recovery: choose equal-power or gain-compensated behavior and expose output safety metering.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 21: Pre-delay automation flanges**

- Symptom: Pre-delay automation flanges.
- Likely causes: a moving delay is being read without an intentional interpolation strategy.
- First recovery: crossfade read heads or constrain automation to safe transitions.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 22: High damping sounds unstable**

- Symptom: High damping sounds unstable.
- Likely causes: filter coefficients approach an unsafe limit at the active internal rate.
- First recovery: bound the coefficient domain and test every quality mode.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 23: The host rejects the channel layout**

- Symptom: The host rejects the channel layout.
- Likely causes: input and output buses are mismatched or unsupported.
- First recovery: request matched mono/stereo for the foundation and log the rejected layout clearly.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

**Troubleshooting card 24: The screenshot and editor differ**

- Symptom: The screenshot and editor differ.
- Likely causes: the screenshot is the visual design target while the JUCE editor remains a scaffold.
- First recovery: use the maturity statement and track UI implementation separately from DSP readiness.

Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.

Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.

After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.


\newpage

## 24. Preset Design Cards

### 24.1 Rooms

**Preset card: Rooms / Intimate**

- Family identity: natural early cues and controlled short decay.
- Variant direction: keep pre-delay and width restrained; prioritize direct connection.
- Macro anchors: room size 0.20, damping 0.40, diffusion 0.55.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Rooms / Open**

- Family identity: natural early cues and controlled short decay.
- Variant direction: increase width and early/late separation while preserving center focus.
- Macro anchors: room size 0.20, damping 0.40, diffusion 0.55.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Rooms / Dark**

- Family identity: natural early cues and controlled short decay.
- Variant direction: increase damping and reduce high-frequency persistence.
- Macro anchors: room size 0.20, damping 0.40, diffusion 0.55.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Rooms / Infinite**

- Family identity: natural early cues and controlled short decay.
- Variant direction: prepare a safe Freeze transition and conservative output protection.
- Macro anchors: room size 0.20, damping 0.40, diffusion 0.55.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

### 24.2 Chambers

**Preset card: Chambers / Intimate**

- Family identity: dense useful depth around voices and instruments.
- Variant direction: keep pre-delay and width restrained; prioritize direct connection.
- Macro anchors: room size 0.38, damping 0.52, diffusion 0.68.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Chambers / Open**

- Family identity: dense useful depth around voices and instruments.
- Variant direction: increase width and early/late separation while preserving center focus.
- Macro anchors: room size 0.38, damping 0.52, diffusion 0.68.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Chambers / Dark**

- Family identity: dense useful depth around voices and instruments.
- Variant direction: increase damping and reduce high-frequency persistence.
- Macro anchors: room size 0.38, damping 0.52, diffusion 0.68.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Chambers / Infinite**

- Family identity: dense useful depth around voices and instruments.
- Variant direction: prepare a safe Freeze transition and conservative output protection.
- Macro anchors: room size 0.38, damping 0.52, diffusion 0.68.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

### 24.3 Halls

**Preset card: Halls / Intimate**

- Family identity: integrated long decay with stable source localization.
- Variant direction: keep pre-delay and width restrained; prioritize direct connection.
- Macro anchors: room size 0.58, damping 0.64, diffusion 0.76.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Halls / Open**

- Family identity: integrated long decay with stable source localization.
- Variant direction: increase width and early/late separation while preserving center focus.
- Macro anchors: room size 0.58, damping 0.64, diffusion 0.76.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Halls / Dark**

- Family identity: integrated long decay with stable source localization.
- Variant direction: increase damping and reduce high-frequency persistence.
- Macro anchors: room size 0.58, damping 0.64, diffusion 0.76.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Halls / Infinite**

- Family identity: integrated long decay with stable source localization.
- Variant direction: prepare a safe Freeze transition and conservative output protection.
- Macro anchors: room size 0.58, damping 0.64, diffusion 0.76.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

### 24.4 Plates

**Preset card: Plates / Intimate**

- Family identity: fast diffusion and bright sustained density.
- Variant direction: keep pre-delay and width restrained; prioritize direct connection.
- Macro anchors: room size 0.44, damping 0.72, diffusion 0.86.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Plates / Open**

- Family identity: fast diffusion and bright sustained density.
- Variant direction: increase width and early/late separation while preserving center focus.
- Macro anchors: room size 0.44, damping 0.72, diffusion 0.86.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Plates / Dark**

- Family identity: fast diffusion and bright sustained density.
- Variant direction: increase damping and reduce high-frequency persistence.
- Macro anchors: room size 0.44, damping 0.72, diffusion 0.86.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Plates / Infinite**

- Family identity: fast diffusion and bright sustained density.
- Variant direction: prepare a safe Freeze transition and conservative output protection.
- Macro anchors: room size 0.44, damping 0.72, diffusion 0.86.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

### 24.5 Architectures

**Preset card: Architectures / Intimate**

- Family identity: geometry-led spaces with explicit source/listener context.
- Variant direction: keep pre-delay and width restrained; prioritize direct connection.
- Macro anchors: room size 0.66, damping 0.56, diffusion 0.78.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Architectures / Open**

- Family identity: geometry-led spaces with explicit source/listener context.
- Variant direction: increase width and early/late separation while preserving center focus.
- Macro anchors: room size 0.66, damping 0.56, diffusion 0.78.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Architectures / Dark**

- Family identity: geometry-led spaces with explicit source/listener context.
- Variant direction: increase damping and reduce high-frequency persistence.
- Macro anchors: room size 0.66, damping 0.56, diffusion 0.78.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Architectures / Infinite**

- Family identity: geometry-led spaces with explicit source/listener context.
- Variant direction: prepare a safe Freeze transition and conservative output protection.
- Macro anchors: room size 0.66, damping 0.56, diffusion 0.78.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

### 24.6 Experimental

**Preset card: Experimental / Intimate**

- Family identity: reverse, freeze, and exaggerated spatial behavior.
- Variant direction: keep pre-delay and width restrained; prioritize direct connection.
- Macro anchors: room size 0.76, damping 0.48, diffusion 0.82.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Experimental / Open**

- Family identity: reverse, freeze, and exaggerated spatial behavior.
- Variant direction: increase width and early/late separation while preserving center focus.
- Macro anchors: room size 0.76, damping 0.48, diffusion 0.82.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Experimental / Dark**

- Family identity: reverse, freeze, and exaggerated spatial behavior.
- Variant direction: increase damping and reduce high-frequency persistence.
- Macro anchors: room size 0.76, damping 0.48, diffusion 0.82.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

**Preset card: Experimental / Infinite**

- Family identity: reverse, freeze, and exaggerated spatial behavior.
- Variant direction: prepare a safe Freeze transition and conservative output protection.
- Macro anchors: room size 0.76, damping 0.48, diffusion 0.82.
- Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.

Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.

Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.

Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.


\newpage

## 25. Parameter Interaction Cards

**Interaction card 1: Pre-Delay with Room Size**

- Parameters: `pre_delay_ms` and `room_size`.
- First role: separates the direct event from the room onset.
- Second role: changes perceived scale and reflection spacing.
- Transition rules: use a ramp or delay-line crossfade; stage structural changes outside the callback.

Hold Room Size at its default and sweep Pre-Delay through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 2: Pre-Delay with RT60 Coarse**

- Parameters: `pre_delay_ms` and `rt60_coarse`.
- First role: separates the direct event from the room onset.
- Second role: moves through the full logarithmic decay range.
- Transition rules: use a ramp or delay-line crossfade; display the effective seconds value.

Hold RT60 Coarse at its default and sweep Pre-Delay through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 3: Pre-Delay with RT60 Fine**

- Parameters: `pre_delay_ms` and `rt60_fine`.
- First role: separates the direct event from the room onset.
- Second role: trims decay proportionally around the coarse value.
- Transition rules: use a ramp or delay-line crossfade; keep zero as the exact neutral point.

Hold RT60 Fine at its default and sweep Pre-Delay through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 4: Pre-Delay with Damping**

- Parameters: `pre_delay_ms` and `damping`.
- First role: separates the direct event from the room onset.
- Second role: changes high-frequency persistence.
- Transition rules: use a ramp or delay-line crossfade; interpolate stable filter coefficients.

Hold Damping at its default and sweep Pre-Delay through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 5: Pre-Delay with Width**

- Parameters: `pre_delay_ms` and `width`.
- First role: separates the direct event from the room onset.
- Second role: changes lateral energy and correlation.
- Transition rules: use a ramp or delay-line crossfade; monitor mono compatibility during movement.

Hold Width at its default and sweep Pre-Delay through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 6: Pre-Delay with Diffusion**

- Parameters: `pre_delay_ms` and `diffusion`.
- First role: separates the direct event from the room onset.
- Second role: changes echo-density buildup.
- Transition rules: use a ramp or delay-line crossfade; crossfade when topology must change.

Hold Diffusion at its default and sweep Pre-Delay through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 7: Pre-Delay with Wet**

- Parameters: `pre_delay_ms` and `wet`.
- First role: separates the direct event from the room onset.
- Second role: sets processed contribution.
- Transition rules: use a ramp or delay-line crossfade; choose and document the mix law.

Hold Wet at its default and sweep Pre-Delay through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 8: Pre-Delay with Dry**

- Parameters: `pre_delay_ms` and `dry`.
- First role: separates the direct event from the room onset.
- Second role: sets direct contribution.
- Transition rules: use a ramp or delay-line crossfade; preserve bypass and gain staging.

Hold Dry at its default and sweep Pre-Delay through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 9: Pre-Delay with Freeze**

- Parameters: `pre_delay_ms` and `freeze`.
- First role: separates the direct event from the room onset.
- Second role: changes network energy behavior.
- Transition rules: use a ramp or delay-line crossfade; use a debounced, smoothed mode transition.

Hold Freeze at its default and sweep Pre-Delay through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 10: Pre-Delay with Reverse**

- Parameters: `pre_delay_ms` and `reverse`.
- First role: separates the direct event from the room onset.
- Second role: changes the envelope and buffering model.
- Transition rules: use a ramp or delay-line crossfade; report added latency before activation.

Hold Reverse at its default and sweep Pre-Delay through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 11: Pre-Delay with Quality**

- Parameters: `pre_delay_ms` and `quality_mode`.
- First role: separates the direct event from the room onset.
- Second role: selects the internal rate policy.
- Transition rules: use a ramp or delay-line crossfade; apply through a safe reprepare boundary.

Hold Quality at its default and sweep Pre-Delay through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 12: Room Size with RT60 Coarse**

- Parameters: `room_size` and `rt60_coarse`.
- First role: changes perceived scale and reflection spacing.
- Second role: moves through the full logarithmic decay range.
- Transition rules: stage structural changes outside the callback; display the effective seconds value.

Hold RT60 Coarse at its default and sweep Room Size through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 13: Room Size with RT60 Fine**

- Parameters: `room_size` and `rt60_fine`.
- First role: changes perceived scale and reflection spacing.
- Second role: trims decay proportionally around the coarse value.
- Transition rules: stage structural changes outside the callback; keep zero as the exact neutral point.

Hold RT60 Fine at its default and sweep Room Size through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 14: Room Size with Damping**

- Parameters: `room_size` and `damping`.
- First role: changes perceived scale and reflection spacing.
- Second role: changes high-frequency persistence.
- Transition rules: stage structural changes outside the callback; interpolate stable filter coefficients.

Hold Damping at its default and sweep Room Size through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 15: Room Size with Width**

- Parameters: `room_size` and `width`.
- First role: changes perceived scale and reflection spacing.
- Second role: changes lateral energy and correlation.
- Transition rules: stage structural changes outside the callback; monitor mono compatibility during movement.

Hold Width at its default and sweep Room Size through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 16: Room Size with Diffusion**

- Parameters: `room_size` and `diffusion`.
- First role: changes perceived scale and reflection spacing.
- Second role: changes echo-density buildup.
- Transition rules: stage structural changes outside the callback; crossfade when topology must change.

Hold Diffusion at its default and sweep Room Size through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 17: Room Size with Wet**

- Parameters: `room_size` and `wet`.
- First role: changes perceived scale and reflection spacing.
- Second role: sets processed contribution.
- Transition rules: stage structural changes outside the callback; choose and document the mix law.

Hold Wet at its default and sweep Room Size through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 18: Room Size with Dry**

- Parameters: `room_size` and `dry`.
- First role: changes perceived scale and reflection spacing.
- Second role: sets direct contribution.
- Transition rules: stage structural changes outside the callback; preserve bypass and gain staging.

Hold Dry at its default and sweep Room Size through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 19: Room Size with Freeze**

- Parameters: `room_size` and `freeze`.
- First role: changes perceived scale and reflection spacing.
- Second role: changes network energy behavior.
- Transition rules: stage structural changes outside the callback; use a debounced, smoothed mode transition.

Hold Freeze at its default and sweep Room Size through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 20: Room Size with Reverse**

- Parameters: `room_size` and `reverse`.
- First role: changes perceived scale and reflection spacing.
- Second role: changes the envelope and buffering model.
- Transition rules: stage structural changes outside the callback; report added latency before activation.

Hold Reverse at its default and sweep Room Size through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 21: Room Size with Quality**

- Parameters: `room_size` and `quality_mode`.
- First role: changes perceived scale and reflection spacing.
- Second role: selects the internal rate policy.
- Transition rules: stage structural changes outside the callback; apply through a safe reprepare boundary.

Hold Quality at its default and sweep Room Size through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 22: RT60 Coarse with RT60 Fine**

- Parameters: `rt60_coarse` and `rt60_fine`.
- First role: moves through the full logarithmic decay range.
- Second role: trims decay proportionally around the coarse value.
- Transition rules: display the effective seconds value; keep zero as the exact neutral point.

Hold RT60 Fine at its default and sweep RT60 Coarse through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 23: RT60 Coarse with Damping**

- Parameters: `rt60_coarse` and `damping`.
- First role: moves through the full logarithmic decay range.
- Second role: changes high-frequency persistence.
- Transition rules: display the effective seconds value; interpolate stable filter coefficients.

Hold Damping at its default and sweep RT60 Coarse through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 24: RT60 Coarse with Width**

- Parameters: `rt60_coarse` and `width`.
- First role: moves through the full logarithmic decay range.
- Second role: changes lateral energy and correlation.
- Transition rules: display the effective seconds value; monitor mono compatibility during movement.

Hold Width at its default and sweep RT60 Coarse through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 25: RT60 Coarse with Diffusion**

- Parameters: `rt60_coarse` and `diffusion`.
- First role: moves through the full logarithmic decay range.
- Second role: changes echo-density buildup.
- Transition rules: display the effective seconds value; crossfade when topology must change.

Hold Diffusion at its default and sweep RT60 Coarse through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 26: RT60 Coarse with Wet**

- Parameters: `rt60_coarse` and `wet`.
- First role: moves through the full logarithmic decay range.
- Second role: sets processed contribution.
- Transition rules: display the effective seconds value; choose and document the mix law.

Hold Wet at its default and sweep RT60 Coarse through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 27: RT60 Coarse with Dry**

- Parameters: `rt60_coarse` and `dry`.
- First role: moves through the full logarithmic decay range.
- Second role: sets direct contribution.
- Transition rules: display the effective seconds value; preserve bypass and gain staging.

Hold Dry at its default and sweep RT60 Coarse through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 28: RT60 Coarse with Freeze**

- Parameters: `rt60_coarse` and `freeze`.
- First role: moves through the full logarithmic decay range.
- Second role: changes network energy behavior.
- Transition rules: display the effective seconds value; use a debounced, smoothed mode transition.

Hold Freeze at its default and sweep RT60 Coarse through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 29: RT60 Coarse with Reverse**

- Parameters: `rt60_coarse` and `reverse`.
- First role: moves through the full logarithmic decay range.
- Second role: changes the envelope and buffering model.
- Transition rules: display the effective seconds value; report added latency before activation.

Hold Reverse at its default and sweep RT60 Coarse through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 30: RT60 Coarse with Quality**

- Parameters: `rt60_coarse` and `quality_mode`.
- First role: moves through the full logarithmic decay range.
- Second role: selects the internal rate policy.
- Transition rules: display the effective seconds value; apply through a safe reprepare boundary.

Hold Quality at its default and sweep RT60 Coarse through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 31: RT60 Fine with Damping**

- Parameters: `rt60_fine` and `damping`.
- First role: trims decay proportionally around the coarse value.
- Second role: changes high-frequency persistence.
- Transition rules: keep zero as the exact neutral point; interpolate stable filter coefficients.

Hold Damping at its default and sweep RT60 Fine through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 32: RT60 Fine with Width**

- Parameters: `rt60_fine` and `width`.
- First role: trims decay proportionally around the coarse value.
- Second role: changes lateral energy and correlation.
- Transition rules: keep zero as the exact neutral point; monitor mono compatibility during movement.

Hold Width at its default and sweep RT60 Fine through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 33: RT60 Fine with Diffusion**

- Parameters: `rt60_fine` and `diffusion`.
- First role: trims decay proportionally around the coarse value.
- Second role: changes echo-density buildup.
- Transition rules: keep zero as the exact neutral point; crossfade when topology must change.

Hold Diffusion at its default and sweep RT60 Fine through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 34: RT60 Fine with Wet**

- Parameters: `rt60_fine` and `wet`.
- First role: trims decay proportionally around the coarse value.
- Second role: sets processed contribution.
- Transition rules: keep zero as the exact neutral point; choose and document the mix law.

Hold Wet at its default and sweep RT60 Fine through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 35: RT60 Fine with Dry**

- Parameters: `rt60_fine` and `dry`.
- First role: trims decay proportionally around the coarse value.
- Second role: sets direct contribution.
- Transition rules: keep zero as the exact neutral point; preserve bypass and gain staging.

Hold Dry at its default and sweep RT60 Fine through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 36: RT60 Fine with Freeze**

- Parameters: `rt60_fine` and `freeze`.
- First role: trims decay proportionally around the coarse value.
- Second role: changes network energy behavior.
- Transition rules: keep zero as the exact neutral point; use a debounced, smoothed mode transition.

Hold Freeze at its default and sweep RT60 Fine through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 37: RT60 Fine with Reverse**

- Parameters: `rt60_fine` and `reverse`.
- First role: trims decay proportionally around the coarse value.
- Second role: changes the envelope and buffering model.
- Transition rules: keep zero as the exact neutral point; report added latency before activation.

Hold Reverse at its default and sweep RT60 Fine through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 38: RT60 Fine with Quality**

- Parameters: `rt60_fine` and `quality_mode`.
- First role: trims decay proportionally around the coarse value.
- Second role: selects the internal rate policy.
- Transition rules: keep zero as the exact neutral point; apply through a safe reprepare boundary.

Hold Quality at its default and sweep RT60 Fine through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 39: Damping with Width**

- Parameters: `damping` and `width`.
- First role: changes high-frequency persistence.
- Second role: changes lateral energy and correlation.
- Transition rules: interpolate stable filter coefficients; monitor mono compatibility during movement.

Hold Width at its default and sweep Damping through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 40: Damping with Diffusion**

- Parameters: `damping` and `diffusion`.
- First role: changes high-frequency persistence.
- Second role: changes echo-density buildup.
- Transition rules: interpolate stable filter coefficients; crossfade when topology must change.

Hold Diffusion at its default and sweep Damping through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 41: Damping with Wet**

- Parameters: `damping` and `wet`.
- First role: changes high-frequency persistence.
- Second role: sets processed contribution.
- Transition rules: interpolate stable filter coefficients; choose and document the mix law.

Hold Wet at its default and sweep Damping through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 42: Damping with Dry**

- Parameters: `damping` and `dry`.
- First role: changes high-frequency persistence.
- Second role: sets direct contribution.
- Transition rules: interpolate stable filter coefficients; preserve bypass and gain staging.

Hold Dry at its default and sweep Damping through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 43: Damping with Freeze**

- Parameters: `damping` and `freeze`.
- First role: changes high-frequency persistence.
- Second role: changes network energy behavior.
- Transition rules: interpolate stable filter coefficients; use a debounced, smoothed mode transition.

Hold Freeze at its default and sweep Damping through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 44: Damping with Reverse**

- Parameters: `damping` and `reverse`.
- First role: changes high-frequency persistence.
- Second role: changes the envelope and buffering model.
- Transition rules: interpolate stable filter coefficients; report added latency before activation.

Hold Reverse at its default and sweep Damping through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 45: Damping with Quality**

- Parameters: `damping` and `quality_mode`.
- First role: changes high-frequency persistence.
- Second role: selects the internal rate policy.
- Transition rules: interpolate stable filter coefficients; apply through a safe reprepare boundary.

Hold Quality at its default and sweep Damping through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 46: Width with Diffusion**

- Parameters: `width` and `diffusion`.
- First role: changes lateral energy and correlation.
- Second role: changes echo-density buildup.
- Transition rules: monitor mono compatibility during movement; crossfade when topology must change.

Hold Diffusion at its default and sweep Width through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 47: Width with Wet**

- Parameters: `width` and `wet`.
- First role: changes lateral energy and correlation.
- Second role: sets processed contribution.
- Transition rules: monitor mono compatibility during movement; choose and document the mix law.

Hold Wet at its default and sweep Width through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 48: Width with Dry**

- Parameters: `width` and `dry`.
- First role: changes lateral energy and correlation.
- Second role: sets direct contribution.
- Transition rules: monitor mono compatibility during movement; preserve bypass and gain staging.

Hold Dry at its default and sweep Width through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 49: Width with Freeze**

- Parameters: `width` and `freeze`.
- First role: changes lateral energy and correlation.
- Second role: changes network energy behavior.
- Transition rules: monitor mono compatibility during movement; use a debounced, smoothed mode transition.

Hold Freeze at its default and sweep Width through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 50: Width with Reverse**

- Parameters: `width` and `reverse`.
- First role: changes lateral energy and correlation.
- Second role: changes the envelope and buffering model.
- Transition rules: monitor mono compatibility during movement; report added latency before activation.

Hold Reverse at its default and sweep Width through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 51: Width with Quality**

- Parameters: `width` and `quality_mode`.
- First role: changes lateral energy and correlation.
- Second role: selects the internal rate policy.
- Transition rules: monitor mono compatibility during movement; apply through a safe reprepare boundary.

Hold Quality at its default and sweep Width through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 52: Diffusion with Wet**

- Parameters: `diffusion` and `wet`.
- First role: changes echo-density buildup.
- Second role: sets processed contribution.
- Transition rules: crossfade when topology must change; choose and document the mix law.

Hold Wet at its default and sweep Diffusion through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 53: Diffusion with Dry**

- Parameters: `diffusion` and `dry`.
- First role: changes echo-density buildup.
- Second role: sets direct contribution.
- Transition rules: crossfade when topology must change; preserve bypass and gain staging.

Hold Dry at its default and sweep Diffusion through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 54: Diffusion with Freeze**

- Parameters: `diffusion` and `freeze`.
- First role: changes echo-density buildup.
- Second role: changes network energy behavior.
- Transition rules: crossfade when topology must change; use a debounced, smoothed mode transition.

Hold Freeze at its default and sweep Diffusion through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 55: Diffusion with Reverse**

- Parameters: `diffusion` and `reverse`.
- First role: changes echo-density buildup.
- Second role: changes the envelope and buffering model.
- Transition rules: crossfade when topology must change; report added latency before activation.

Hold Reverse at its default and sweep Diffusion through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 56: Diffusion with Quality**

- Parameters: `diffusion` and `quality_mode`.
- First role: changes echo-density buildup.
- Second role: selects the internal rate policy.
- Transition rules: crossfade when topology must change; apply through a safe reprepare boundary.

Hold Quality at its default and sweep Diffusion through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 57: Wet with Dry**

- Parameters: `wet` and `dry`.
- First role: sets processed contribution.
- Second role: sets direct contribution.
- Transition rules: choose and document the mix law; preserve bypass and gain staging.

Hold Dry at its default and sweep Wet through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 58: Wet with Freeze**

- Parameters: `wet` and `freeze`.
- First role: sets processed contribution.
- Second role: changes network energy behavior.
- Transition rules: choose and document the mix law; use a debounced, smoothed mode transition.

Hold Freeze at its default and sweep Wet through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 59: Wet with Reverse**

- Parameters: `wet` and `reverse`.
- First role: sets processed contribution.
- Second role: changes the envelope and buffering model.
- Transition rules: choose and document the mix law; report added latency before activation.

Hold Reverse at its default and sweep Wet through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 60: Wet with Quality**

- Parameters: `wet` and `quality_mode`.
- First role: sets processed contribution.
- Second role: selects the internal rate policy.
- Transition rules: choose and document the mix law; apply through a safe reprepare boundary.

Hold Quality at its default and sweep Wet through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 61: Dry with Freeze**

- Parameters: `dry` and `freeze`.
- First role: sets direct contribution.
- Second role: changes network energy behavior.
- Transition rules: preserve bypass and gain staging; use a debounced, smoothed mode transition.

Hold Freeze at its default and sweep Dry through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 62: Dry with Reverse**

- Parameters: `dry` and `reverse`.
- First role: sets direct contribution.
- Second role: changes the envelope and buffering model.
- Transition rules: preserve bypass and gain staging; report added latency before activation.

Hold Reverse at its default and sweep Dry through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 63: Dry with Quality**

- Parameters: `dry` and `quality_mode`.
- First role: sets direct contribution.
- Second role: selects the internal rate policy.
- Transition rules: preserve bypass and gain staging; apply through a safe reprepare boundary.

Hold Quality at its default and sweep Dry through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 64: Freeze with Reverse**

- Parameters: `freeze` and `reverse`.
- First role: changes network energy behavior.
- Second role: changes the envelope and buffering model.
- Transition rules: use a debounced, smoothed mode transition; report added latency before activation.

Hold Reverse at its default and sweep Freeze through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 65: Freeze with Quality**

- Parameters: `freeze` and `quality_mode`.
- First role: changes network energy behavior.
- Second role: selects the internal rate policy.
- Transition rules: use a debounced, smoothed mode transition; apply through a safe reprepare boundary.

Hold Quality at its default and sweep Freeze through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

**Interaction card 66: Reverse with Quality**

- Parameters: `reverse` and `quality_mode`.
- First role: changes the envelope and buffering model.
- Second role: selects the internal rate policy.
- Transition rules: report added latency before activation; apply through a safe reprepare boundary.

Hold Quality at its default and sweep Reverse through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.

Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.

Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.


\newpage

## 26. Monitoring And Audition Cards

### 26.1 Lead vocal

**Audition card: Lead vocal on Nearfield monitors**

- Source: Lead vocal.
- Monitoring context: Nearfield monitors.
- Judgment goal: judge center focus, depth layers, and low-mid buildup at a calibrated position.
- Useful pre-delay range: 18 to 45 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to preserve consonants and front-of-mix intelligibility. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, judge center focus, depth layers, and low-mid buildup at a calibrated position. For lead vocal, watch sibilance and center stability. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Lead vocal on Headphones**

- Source: Lead vocal.
- Monitoring context: Headphones.
- Judgment goal: inspect modulation, tail texture, and left/right discontinuities without room masking.
- Useful pre-delay range: 18 to 45 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to preserve consonants and front-of-mix intelligibility. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, inspect modulation, tail texture, and left/right discontinuities without room masking. For lead vocal, watch sibilance and center stability. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Lead vocal on Mono sum**

- Source: Lead vocal.
- Monitoring context: Mono sum.
- Judgment goal: expose anti-correlation, combing, and source loss caused by excessive width.
- Useful pre-delay range: 18 to 45 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to preserve consonants and front-of-mix intelligibility. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, expose anti-correlation, combing, and source loss caused by excessive width. For lead vocal, watch sibilance and center stability. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Lead vocal on Low-level playback**

- Source: Lead vocal.
- Monitoring context: Low-level playback.
- Judgment goal: test whether the space remains legible without relying on loudness.
- Useful pre-delay range: 18 to 45 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to preserve consonants and front-of-mix intelligibility. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, test whether the space remains legible without relying on loudness. For lead vocal, watch sibilance and center stability. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

### 26.2 Spoken word

**Audition card: Spoken word on Nearfield monitors**

- Source: Spoken word.
- Monitoring context: Nearfield monitors.
- Judgment goal: judge center focus, depth layers, and low-mid buildup at a calibrated position.
- Useful pre-delay range: 8 to 28 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to add believable room cues without masking language. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, judge center focus, depth layers, and low-mid buildup at a calibrated position. For spoken word, check breaths, plosives, and noise-floor lift. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Spoken word on Headphones**

- Source: Spoken word.
- Monitoring context: Headphones.
- Judgment goal: inspect modulation, tail texture, and left/right discontinuities without room masking.
- Useful pre-delay range: 8 to 28 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to add believable room cues without masking language. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, inspect modulation, tail texture, and left/right discontinuities without room masking. For spoken word, check breaths, plosives, and noise-floor lift. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Spoken word on Mono sum**

- Source: Spoken word.
- Monitoring context: Mono sum.
- Judgment goal: expose anti-correlation, combing, and source loss caused by excessive width.
- Useful pre-delay range: 8 to 28 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to add believable room cues without masking language. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, expose anti-correlation, combing, and source loss caused by excessive width. For spoken word, check breaths, plosives, and noise-floor lift. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Spoken word on Low-level playback**

- Source: Spoken word.
- Monitoring context: Low-level playback.
- Judgment goal: test whether the space remains legible without relying on loudness.
- Useful pre-delay range: 8 to 28 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to add believable room cues without masking language. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, test whether the space remains legible without relying on loudness. For spoken word, check breaths, plosives, and noise-floor lift. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

### 26.3 Drum kit

**Audition card: Drum kit on Nearfield monitors**

- Source: Drum kit.
- Monitoring context: Nearfield monitors.
- Judgment goal: judge center focus, depth layers, and low-mid buildup at a calibrated position.
- Useful pre-delay range: 4 to 24 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to build size while preserving transient geometry. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, judge center focus, depth layers, and low-mid buildup at a calibrated position. For drum kit, check kick definition and snare tail density. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Drum kit on Headphones**

- Source: Drum kit.
- Monitoring context: Headphones.
- Judgment goal: inspect modulation, tail texture, and left/right discontinuities without room masking.
- Useful pre-delay range: 4 to 24 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to build size while preserving transient geometry. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, inspect modulation, tail texture, and left/right discontinuities without room masking. For drum kit, check kick definition and snare tail density. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Drum kit on Mono sum**

- Source: Drum kit.
- Monitoring context: Mono sum.
- Judgment goal: expose anti-correlation, combing, and source loss caused by excessive width.
- Useful pre-delay range: 4 to 24 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to build size while preserving transient geometry. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, expose anti-correlation, combing, and source loss caused by excessive width. For drum kit, check kick definition and snare tail density. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Drum kit on Low-level playback**

- Source: Drum kit.
- Monitoring context: Low-level playback.
- Judgment goal: test whether the space remains legible without relying on loudness.
- Useful pre-delay range: 4 to 24 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to build size while preserving transient geometry. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, test whether the space remains legible without relying on loudness. For drum kit, check kick definition and snare tail density. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

### 26.4 Piano

**Audition card: Piano on Nearfield monitors**

- Source: Piano.
- Monitoring context: Nearfield monitors.
- Judgment goal: judge center focus, depth layers, and low-mid buildup at a calibrated position.
- Useful pre-delay range: 12 to 40 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to support sustain without blurring note attacks. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, judge center focus, depth layers, and low-mid buildup at a calibrated position. For piano, listen for low-mid modal buildup. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Piano on Headphones**

- Source: Piano.
- Monitoring context: Headphones.
- Judgment goal: inspect modulation, tail texture, and left/right discontinuities without room masking.
- Useful pre-delay range: 12 to 40 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to support sustain without blurring note attacks. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, inspect modulation, tail texture, and left/right discontinuities without room masking. For piano, listen for low-mid modal buildup. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Piano on Mono sum**

- Source: Piano.
- Monitoring context: Mono sum.
- Judgment goal: expose anti-correlation, combing, and source loss caused by excessive width.
- Useful pre-delay range: 12 to 40 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to support sustain without blurring note attacks. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, expose anti-correlation, combing, and source loss caused by excessive width. For piano, listen for low-mid modal buildup. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Piano on Low-level playback**

- Source: Piano.
- Monitoring context: Low-level playback.
- Judgment goal: test whether the space remains legible without relying on loudness.
- Useful pre-delay range: 12 to 40 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to support sustain without blurring note attacks. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, test whether the space remains legible without relying on loudness. For piano, listen for low-mid modal buildup. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

### 26.5 Acoustic guitar

**Audition card: Acoustic guitar on Nearfield monitors**

- Source: Acoustic guitar.
- Monitoring context: Nearfield monitors.
- Judgment goal: judge center focus, depth layers, and low-mid buildup at a calibrated position.
- Useful pre-delay range: 10 to 32 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to add depth without combing the direct image. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, judge center focus, depth layers, and low-mid buildup at a calibrated position. For acoustic guitar, check pick articulation and mono fold-down. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Acoustic guitar on Headphones**

- Source: Acoustic guitar.
- Monitoring context: Headphones.
- Judgment goal: inspect modulation, tail texture, and left/right discontinuities without room masking.
- Useful pre-delay range: 10 to 32 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to add depth without combing the direct image. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, inspect modulation, tail texture, and left/right discontinuities without room masking. For acoustic guitar, check pick articulation and mono fold-down. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Acoustic guitar on Mono sum**

- Source: Acoustic guitar.
- Monitoring context: Mono sum.
- Judgment goal: expose anti-correlation, combing, and source loss caused by excessive width.
- Useful pre-delay range: 10 to 32 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to add depth without combing the direct image. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, expose anti-correlation, combing, and source loss caused by excessive width. For acoustic guitar, check pick articulation and mono fold-down. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Acoustic guitar on Low-level playback**

- Source: Acoustic guitar.
- Monitoring context: Low-level playback.
- Judgment goal: test whether the space remains legible without relying on loudness.
- Useful pre-delay range: 10 to 32 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to add depth without combing the direct image. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, test whether the space remains legible without relying on loudness. For acoustic guitar, check pick articulation and mono fold-down. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

### 26.6 Electric guitar

**Audition card: Electric guitar on Nearfield monitors**

- Source: Electric guitar.
- Monitoring context: Nearfield monitors.
- Judgment goal: judge center focus, depth layers, and low-mid buildup at a calibrated position.
- Useful pre-delay range: 6 to 30 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to place the cabinet in a designed environment. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, judge center focus, depth layers, and low-mid buildup at a calibrated position. For electric guitar, watch upper-mid glare in the return. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Electric guitar on Headphones**

- Source: Electric guitar.
- Monitoring context: Headphones.
- Judgment goal: inspect modulation, tail texture, and left/right discontinuities without room masking.
- Useful pre-delay range: 6 to 30 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to place the cabinet in a designed environment. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, inspect modulation, tail texture, and left/right discontinuities without room masking. For electric guitar, watch upper-mid glare in the return. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Electric guitar on Mono sum**

- Source: Electric guitar.
- Monitoring context: Mono sum.
- Judgment goal: expose anti-correlation, combing, and source loss caused by excessive width.
- Useful pre-delay range: 6 to 30 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to place the cabinet in a designed environment. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, expose anti-correlation, combing, and source loss caused by excessive width. For electric guitar, watch upper-mid glare in the return. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Electric guitar on Low-level playback**

- Source: Electric guitar.
- Monitoring context: Low-level playback.
- Judgment goal: test whether the space remains legible without relying on loudness.
- Useful pre-delay range: 6 to 30 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to place the cabinet in a designed environment. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, test whether the space remains legible without relying on loudness. For electric guitar, watch upper-mid glare in the return. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

### 26.7 Strings

**Audition card: Strings on Nearfield monitors**

- Source: Strings.
- Monitoring context: Nearfield monitors.
- Judgment goal: judge center focus, depth layers, and low-mid buildup at a calibrated position.
- Useful pre-delay range: 16 to 55 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to extend bow sustain and ensemble width. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, judge center focus, depth layers, and low-mid buildup at a calibrated position. For strings, check section localization and high decay. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Strings on Headphones**

- Source: Strings.
- Monitoring context: Headphones.
- Judgment goal: inspect modulation, tail texture, and left/right discontinuities without room masking.
- Useful pre-delay range: 16 to 55 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to extend bow sustain and ensemble width. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, inspect modulation, tail texture, and left/right discontinuities without room masking. For strings, check section localization and high decay. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Strings on Mono sum**

- Source: Strings.
- Monitoring context: Mono sum.
- Judgment goal: expose anti-correlation, combing, and source loss caused by excessive width.
- Useful pre-delay range: 16 to 55 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to extend bow sustain and ensemble width. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, expose anti-correlation, combing, and source loss caused by excessive width. For strings, check section localization and high decay. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Strings on Low-level playback**

- Source: Strings.
- Monitoring context: Low-level playback.
- Judgment goal: test whether the space remains legible without relying on loudness.
- Useful pre-delay range: 16 to 55 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to extend bow sustain and ensemble width. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, test whether the space remains legible without relying on loudness. For strings, check section localization and high decay. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

### 26.8 Synth pad

**Audition card: Synth pad on Nearfield monitors**

- Source: Synth pad.
- Monitoring context: Nearfield monitors.
- Judgment goal: judge center focus, depth layers, and low-mid buildup at a calibrated position.
- Useful pre-delay range: 0 to 40 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to turn sustained harmony into an evolving field. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, judge center focus, depth layers, and low-mid buildup at a calibrated position. For synth pad, watch feedback energy and stereo correlation. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Synth pad on Headphones**

- Source: Synth pad.
- Monitoring context: Headphones.
- Judgment goal: inspect modulation, tail texture, and left/right discontinuities without room masking.
- Useful pre-delay range: 0 to 40 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to turn sustained harmony into an evolving field. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, inspect modulation, tail texture, and left/right discontinuities without room masking. For synth pad, watch feedback energy and stereo correlation. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Synth pad on Mono sum**

- Source: Synth pad.
- Monitoring context: Mono sum.
- Judgment goal: expose anti-correlation, combing, and source loss caused by excessive width.
- Useful pre-delay range: 0 to 40 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to turn sustained harmony into an evolving field. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, expose anti-correlation, combing, and source loss caused by excessive width. For synth pad, watch feedback energy and stereo correlation. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Synth pad on Low-level playback**

- Source: Synth pad.
- Monitoring context: Low-level playback.
- Judgment goal: test whether the space remains legible without relying on loudness.
- Useful pre-delay range: 0 to 40 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to turn sustained harmony into an evolving field. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, test whether the space remains legible without relying on loudness. For synth pad, watch feedback energy and stereo correlation. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

### 26.9 Percussion

**Audition card: Percussion on Nearfield monitors**

- Source: Percussion.
- Monitoring context: Nearfield monitors.
- Judgment goal: judge center focus, depth layers, and low-mid buildup at a calibrated position.
- Useful pre-delay range: 2 to 22 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to create rhythmic depth around short impulses. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, judge center focus, depth layers, and low-mid buildup at a calibrated position. For percussion, check early reflections against tempo. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Percussion on Headphones**

- Source: Percussion.
- Monitoring context: Headphones.
- Judgment goal: inspect modulation, tail texture, and left/right discontinuities without room masking.
- Useful pre-delay range: 2 to 22 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to create rhythmic depth around short impulses. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, inspect modulation, tail texture, and left/right discontinuities without room masking. For percussion, check early reflections against tempo. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Percussion on Mono sum**

- Source: Percussion.
- Monitoring context: Mono sum.
- Judgment goal: expose anti-correlation, combing, and source loss caused by excessive width.
- Useful pre-delay range: 2 to 22 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to create rhythmic depth around short impulses. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, expose anti-correlation, combing, and source loss caused by excessive width. For percussion, check early reflections against tempo. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Percussion on Low-level playback**

- Source: Percussion.
- Monitoring context: Low-level playback.
- Judgment goal: test whether the space remains legible without relying on loudness.
- Useful pre-delay range: 2 to 22 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to create rhythmic depth around short impulses. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, test whether the space remains legible without relying on loudness. For percussion, check early reflections against tempo. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

### 26.10 Field recording

**Audition card: Field recording on Nearfield monitors**

- Source: Field recording.
- Monitoring context: Nearfield monitors.
- Judgment goal: judge center focus, depth layers, and low-mid buildup at a calibrated position.
- Useful pre-delay range: 0 to 60 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to recontextualize a scene without losing its anchors. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, judge center focus, depth layers, and low-mid buildup at a calibrated position. For field recording, compare spectral floor and spatial plausibility. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Field recording on Headphones**

- Source: Field recording.
- Monitoring context: Headphones.
- Judgment goal: inspect modulation, tail texture, and left/right discontinuities without room masking.
- Useful pre-delay range: 0 to 60 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to recontextualize a scene without losing its anchors. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, inspect modulation, tail texture, and left/right discontinuities without room masking. For field recording, compare spectral floor and spatial plausibility. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Field recording on Mono sum**

- Source: Field recording.
- Monitoring context: Mono sum.
- Judgment goal: expose anti-correlation, combing, and source loss caused by excessive width.
- Useful pre-delay range: 0 to 60 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to recontextualize a scene without losing its anchors. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, expose anti-correlation, combing, and source loss caused by excessive width. For field recording, compare spectral floor and spatial plausibility. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

**Audition card: Field recording on Low-level playback**

- Source: Field recording.
- Monitoring context: Low-level playback.
- Judgment goal: test whether the space remains legible without relying on loudness.
- Useful pre-delay range: 0 to 60 ms.

Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to recontextualize a scene without losing its anchors. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.

In this monitoring context, test whether the space remains legible without relying on loudness. For field recording, compare spectral floor and spatial plausibility. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.

Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.


\newpage

## 27. Asset Lifecycle Cards

### 27.1 DXF room shell

**Asset card: DXF room shell: Import**

- Asset: DXF room shell.
- Lifecycle stage: Import.
- Stage objective: read and normalize the external representation away from the audio callback.
- Identity fields: geometry topology, units, transforms, and source/listener coordinates.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: DXF room shell: Validate**

- Asset: DXF room shell.
- Lifecycle stage: Validate.
- Stage objective: reject malformed, unbounded, unsupported, or ambiguous content with field-specific errors.
- Identity fields: geometry topology, units, transforms, and source/listener coordinates.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: DXF room shell: Prepare and cache**

- Asset: DXF room shell.
- Lifecycle stage: Prepare and cache.
- Stage objective: produce an immutable realtime-ready representation with a deterministic key.
- Identity fields: geometry topology, units, transforms, and source/listener coordinates.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: DXF room shell: Recall**

- Asset: DXF room shell.
- Lifecycle stage: Recall.
- Stage objective: resolve the exact asset by identity and degrade safely when it is unavailable.
- Identity fields: geometry topology, units, transforms, and source/listener coordinates.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

### 27.2 Early-reflection profile

**Asset card: Early-reflection profile: Import**

- Asset: Early-reflection profile.
- Lifecycle stage: Import.
- Stage objective: read and normalize the external representation away from the audio callback.
- Identity fields: bounded tap times, gains, directions, and profile version.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Early-reflection profile: Validate**

- Asset: Early-reflection profile.
- Lifecycle stage: Validate.
- Stage objective: reject malformed, unbounded, unsupported, or ambiguous content with field-specific errors.
- Identity fields: bounded tap times, gains, directions, and profile version.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Early-reflection profile: Prepare and cache**

- Asset: Early-reflection profile.
- Lifecycle stage: Prepare and cache.
- Stage objective: produce an immutable realtime-ready representation with a deterministic key.
- Identity fields: bounded tap times, gains, directions, and profile version.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Early-reflection profile: Recall**

- Asset: Early-reflection profile.
- Lifecycle stage: Recall.
- Stage objective: resolve the exact asset by identity and degrade safely when it is unavailable.
- Identity fields: bounded tap times, gains, directions, and profile version.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

### 27.3 Measured impulse response

**Asset card: Measured impulse response: Import**

- Asset: Measured impulse response.
- Lifecycle stage: Import.
- Stage objective: read and normalize the external representation away from the audio callback.
- Identity fields: sample rate, channels, trim, normalization, provenance, and checksum.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Measured impulse response: Validate**

- Asset: Measured impulse response.
- Lifecycle stage: Validate.
- Stage objective: reject malformed, unbounded, unsupported, or ambiguous content with field-specific errors.
- Identity fields: sample rate, channels, trim, normalization, provenance, and checksum.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Measured impulse response: Prepare and cache**

- Asset: Measured impulse response.
- Lifecycle stage: Prepare and cache.
- Stage objective: produce an immutable realtime-ready representation with a deterministic key.
- Identity fields: sample rate, channels, trim, normalization, provenance, and checksum.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Measured impulse response: Recall**

- Asset: Measured impulse response.
- Lifecycle stage: Recall.
- Stage objective: resolve the exact asset by identity and degrade safely when it is unavailable.
- Identity fields: sample rate, channels, trim, normalization, provenance, and checksum.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

### 27.4 Generated impulse response

**Asset card: Generated impulse response: Import**

- Asset: Generated impulse response.
- Lifecycle stage: Import.
- Stage objective: read and normalize the external representation away from the audio callback.
- Identity fields: generator version, seed, parameters, output format, and checksum.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Generated impulse response: Validate**

- Asset: Generated impulse response.
- Lifecycle stage: Validate.
- Stage objective: reject malformed, unbounded, unsupported, or ambiguous content with field-specific errors.
- Identity fields: generator version, seed, parameters, output format, and checksum.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Generated impulse response: Prepare and cache**

- Asset: Generated impulse response.
- Lifecycle stage: Prepare and cache.
- Stage objective: produce an immutable realtime-ready representation with a deterministic key.
- Identity fields: generator version, seed, parameters, output format, and checksum.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Generated impulse response: Recall**

- Asset: Generated impulse response.
- Lifecycle stage: Recall.
- Stage objective: resolve the exact asset by identity and degrade safely when it is unavailable.
- Identity fields: generator version, seed, parameters, output format, and checksum.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

### 27.5 Material library

**Asset card: Material library: Import**

- Asset: Material library.
- Lifecycle stage: Import.
- Stage objective: read and normalize the external representation away from the audio callback.
- Identity fields: stable material IDs, absorption bands, scattering values, and revision.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Material library: Validate**

- Asset: Material library.
- Lifecycle stage: Validate.
- Stage objective: reject malformed, unbounded, unsupported, or ambiguous content with field-specific errors.
- Identity fields: stable material IDs, absorption bands, scattering values, and revision.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Material library: Prepare and cache**

- Asset: Material library.
- Lifecycle stage: Prepare and cache.
- Stage objective: produce an immutable realtime-ready representation with a deterministic key.
- Identity fields: stable material IDs, absorption bands, scattering values, and revision.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Material library: Recall**

- Asset: Material library.
- Lifecycle stage: Recall.
- Stage objective: resolve the exact asset by identity and degrade safely when it is unavailable.
- Identity fields: stable material IDs, absorption bands, scattering values, and revision.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

### 27.6 Preset bank

**Asset card: Preset bank: Import**

- Asset: Preset bank.
- Lifecycle stage: Import.
- Stage objective: read and normalize the external representation away from the audio callback.
- Identity fields: parameter schema, author metadata, tags, asset references, and migration version.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Preset bank: Validate**

- Asset: Preset bank.
- Lifecycle stage: Validate.
- Stage objective: reject malformed, unbounded, unsupported, or ambiguous content with field-specific errors.
- Identity fields: parameter schema, author metadata, tags, asset references, and migration version.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Preset bank: Prepare and cache**

- Asset: Preset bank.
- Lifecycle stage: Prepare and cache.
- Stage objective: produce an immutable realtime-ready representation with a deterministic key.
- Identity fields: parameter schema, author metadata, tags, asset references, and migration version.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Preset bank: Recall**

- Asset: Preset bank.
- Lifecycle stage: Recall.
- Stage objective: resolve the exact asset by identity and degrade safely when it is unavailable.
- Identity fields: parameter schema, author metadata, tags, asset references, and migration version.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

### 27.7 Telemetry configuration

**Asset card: Telemetry configuration: Import**

- Asset: Telemetry configuration.
- Lifecycle stage: Import.
- Stage objective: read and normalize the external representation away from the audio callback.
- Identity fields: meter rates, history lengths, visualization channels, and safety bounds.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Telemetry configuration: Validate**

- Asset: Telemetry configuration.
- Lifecycle stage: Validate.
- Stage objective: reject malformed, unbounded, unsupported, or ambiguous content with field-specific errors.
- Identity fields: meter rates, history lengths, visualization channels, and safety bounds.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Telemetry configuration: Prepare and cache**

- Asset: Telemetry configuration.
- Lifecycle stage: Prepare and cache.
- Stage objective: produce an immutable realtime-ready representation with a deterministic key.
- Identity fields: meter rates, history lengths, visualization channels, and safety bounds.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: Telemetry configuration: Recall**

- Asset: Telemetry configuration.
- Lifecycle stage: Recall.
- Stage objective: resolve the exact asset by identity and degrade safely when it is unavailable.
- Identity fields: meter rates, history lengths, visualization channels, and safety bounds.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

### 27.8 HRTF or SOFA set

**Asset card: HRTF or SOFA set: Import**

- Asset: HRTF or SOFA set.
- Lifecycle stage: Import.
- Stage objective: read and normalize the external representation away from the audio callback.
- Identity fields: convention, receiver/emitter indices, coordinate system, sample rate, and license.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: HRTF or SOFA set: Validate**

- Asset: HRTF or SOFA set.
- Lifecycle stage: Validate.
- Stage objective: reject malformed, unbounded, unsupported, or ambiguous content with field-specific errors.
- Identity fields: convention, receiver/emitter indices, coordinate system, sample rate, and license.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: HRTF or SOFA set: Prepare and cache**

- Asset: HRTF or SOFA set.
- Lifecycle stage: Prepare and cache.
- Stage objective: produce an immutable realtime-ready representation with a deterministic key.
- Identity fields: convention, receiver/emitter indices, coordinate system, sample rate, and license.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

**Asset card: HRTF or SOFA set: Recall**

- Asset: HRTF or SOFA set.
- Lifecycle stage: Recall.
- Stage objective: resolve the exact asset by identity and degrade safely when it is unavailable.
- Identity fields: convention, receiver/emitter indices, coordinate system, sample rate, and license.

Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.

Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.

Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.


\newpage

## 28. Release Readiness Cards

### 28.1 Scanning

**Release card: macOS AU: Scanning**

- Target: macOS AU.
- Target scope: desktop Audio Unit hosts and Apple validation tooling.
- Readiness area: clean install, discovery, validation logs, duplicate IDs, and rescan behavior.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: macOS AUv3: Scanning**

- Target: macOS AUv3.
- Target scope: sandboxed extension lifecycle and AUv3-capable hosts.
- Readiness area: clean install, discovery, validation logs, duplicate IDs, and rescan behavior.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: VST3: Scanning**

- Target: VST3.
- Target scope: VST3 scanning, component/controller state, and supported desktop hosts.
- Readiness area: clean install, discovery, validation logs, duplicate IDs, and rescan behavior.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

### 28.2 State

**Release card: macOS AU: State**

- Target: macOS AU.
- Target scope: desktop Audio Unit hosts and Apple validation tooling.
- Readiness area: project recall, preset migration, asset identity, defaults, and corrupted-state recovery.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: macOS AUv3: State**

- Target: macOS AUv3.
- Target scope: sandboxed extension lifecycle and AUv3-capable hosts.
- Readiness area: project recall, preset migration, asset identity, defaults, and corrupted-state recovery.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: VST3: State**

- Target: VST3.
- Target scope: VST3 scanning, component/controller state, and supported desktop hosts.
- Readiness area: project recall, preset migration, asset identity, defaults, and corrupted-state recovery.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

### 28.3 Audio

**Release card: macOS AU: Audio**

- Target: macOS AU.
- Target scope: desktop Audio Unit hosts and Apple validation tooling.
- Readiness area: silence, impulses, full-scale signals, long tails, NaN/Inf containment, and channel integrity.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: macOS AUv3: Audio**

- Target: macOS AUv3.
- Target scope: sandboxed extension lifecycle and AUv3-capable hosts.
- Readiness area: silence, impulses, full-scale signals, long tails, NaN/Inf containment, and channel integrity.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: VST3: Audio**

- Target: VST3.
- Target scope: VST3 scanning, component/controller state, and supported desktop hosts.
- Readiness area: silence, impulses, full-scale signals, long tails, NaN/Inf containment, and channel integrity.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

### 28.4 Automation

**Release card: macOS AU: Automation**

- Target: macOS AU.
- Target scope: desktop Audio Unit hosts and Apple validation tooling.
- Readiness area: all write/read modes, undo, copy, parameter identity, and editor synchronization.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: macOS AUv3: Automation**

- Target: macOS AUv3.
- Target scope: sandboxed extension lifecycle and AUv3-capable hosts.
- Readiness area: all write/read modes, undo, copy, parameter identity, and editor synchronization.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: VST3: Automation**

- Target: VST3.
- Target scope: VST3 scanning, component/controller state, and supported desktop hosts.
- Readiness area: all write/read modes, undo, copy, parameter identity, and editor synchronization.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

### 28.5 Latency

**Release card: macOS AU: Latency**

- Target: macOS AU.
- Target scope: desktop Audio Unit hosts and Apple validation tooling.
- Readiness area: reported frames, impulse measurement, mode changes, compensation, and transport boundaries.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: macOS AUv3: Latency**

- Target: macOS AUv3.
- Target scope: sandboxed extension lifecycle and AUv3-capable hosts.
- Readiness area: reported frames, impulse measurement, mode changes, compensation, and transport boundaries.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: VST3: Latency**

- Target: VST3.
- Target scope: VST3 scanning, component/controller state, and supported desktop hosts.
- Readiness area: reported frames, impulse measurement, mode changes, compensation, and transport boundaries.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

### 28.6 Performance

**Release card: macOS AU: Performance**

- Target: macOS AU.
- Target scope: desktop Audio Unit hosts and Apple validation tooling.
- Readiness area: CPU, memory, denormals, editor cost, quality modes, and long-session stability.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: macOS AUv3: Performance**

- Target: macOS AUv3.
- Target scope: sandboxed extension lifecycle and AUv3-capable hosts.
- Readiness area: CPU, memory, denormals, editor cost, quality modes, and long-session stability.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: VST3: Performance**

- Target: VST3.
- Target scope: VST3 scanning, component/controller state, and supported desktop hosts.
- Readiness area: CPU, memory, denormals, editor cost, quality modes, and long-session stability.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

### 28.7 Editor

**Release card: macOS AU: Editor**

- Target: macOS AU.
- Target scope: desktop Audio Unit hosts and Apple validation tooling.
- Readiness area: resize, scale, accessibility, keyboard focus, reopen, telemetry, and headless processing.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: macOS AUv3: Editor**

- Target: macOS AUv3.
- Target scope: sandboxed extension lifecycle and AUv3-capable hosts.
- Readiness area: resize, scale, accessibility, keyboard focus, reopen, telemetry, and headless processing.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: VST3: Editor**

- Target: VST3.
- Target scope: VST3 scanning, component/controller state, and supported desktop hosts.
- Readiness area: resize, scale, accessibility, keyboard focus, reopen, telemetry, and headless processing.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

### 28.8 Distribution

**Release card: macOS AU: Distribution**

- Target: macOS AU.
- Target scope: desktop Audio Unit hosts and Apple validation tooling.
- Readiness area: bundle contents, architecture slices, signing, notarization, installer, and uninstall.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: macOS AUv3: Distribution**

- Target: macOS AUv3.
- Target scope: sandboxed extension lifecycle and AUv3-capable hosts.
- Readiness area: bundle contents, architecture slices, signing, notarization, installer, and uninstall.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: VST3: Distribution**

- Target: VST3.
- Target scope: VST3 scanning, component/controller state, and supported desktop hosts.
- Readiness area: bundle contents, architecture slices, signing, notarization, installer, and uninstall.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

### 28.9 Diagnostics

**Release card: macOS AU: Diagnostics**

- Target: macOS AU.
- Target scope: desktop Audio Unit hosts and Apple validation tooling.
- Readiness area: status messages, support bundle, crash context, asset hashes, and privacy review.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: macOS AUv3: Diagnostics**

- Target: macOS AUv3.
- Target scope: sandboxed extension lifecycle and AUv3-capable hosts.
- Readiness area: status messages, support bundle, crash context, asset hashes, and privacy review.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: VST3: Diagnostics**

- Target: VST3.
- Target scope: VST3 scanning, component/controller state, and supported desktop hosts.
- Readiness area: status messages, support bundle, crash context, asset hashes, and privacy review.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

### 28.10 Documentation

**Release card: macOS AU: Documentation**

- Target: macOS AU.
- Target scope: desktop Audio Unit hosts and Apple validation tooling.
- Readiness area: build status, supported hosts, limitations, examples, screenshots, and release notes.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: macOS AUv3: Documentation**

- Target: macOS AUv3.
- Target scope: sandboxed extension lifecycle and AUv3-capable hosts.
- Readiness area: build status, supported hosts, limitations, examples, screenshots, and release notes.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

**Release card: VST3: Documentation**

- Target: VST3.
- Target scope: VST3 scanning, component/controller state, and supported desktop hosts.
- Readiness area: build status, supported hosts, limitations, examples, screenshots, and release notes.
- Evidence required: dated environment, build commit, result, and retained logs/artifacts.

Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.

A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.

After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.


\newpage

## 29. Spatial Bus Validation Cards

### 29.1 Mono

**Bus card: Mono: Algorithmic**

- Layout: Mono: one matched input and output channel with no hidden stereo assumptions.
- Processing mode: Algorithmic: the core FDN/diffuser path with matched input/output layout.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: Mono: Reverse**

- Layout: Mono: one matched input and output channel with no hidden stereo assumptions.
- Processing mode: Reverse: the bounded reverse window, channel alignment, and reported latency.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: Mono: Freeze**

- Layout: Mono: one matched input and output channel with no hidden stereo assumptions.
- Processing mode: Freeze: sustained energy, channel stability, exit behavior, and safety limiting.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: Mono: Geometry or IR**

- Layout: Mono: one matched input and output channel with no hidden stereo assumptions.
- Processing mode: Geometry or IR: prepared spatial assets, channel metadata, and deterministic fallback.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

### 29.2 Stereo

**Bus card: Stereo: Algorithmic**

- Layout: Stereo: matched left/right buses, stable center, width, correlation, and mono fold-down.
- Processing mode: Algorithmic: the core FDN/diffuser path with matched input/output layout.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: Stereo: Reverse**

- Layout: Stereo: matched left/right buses, stable center, width, correlation, and mono fold-down.
- Processing mode: Reverse: the bounded reverse window, channel alignment, and reported latency.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: Stereo: Freeze**

- Layout: Stereo: matched left/right buses, stable center, width, correlation, and mono fold-down.
- Processing mode: Freeze: sustained energy, channel stability, exit behavior, and safety limiting.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: Stereo: Geometry or IR**

- Layout: Stereo: matched left/right buses, stable center, width, correlation, and mono fold-down.
- Processing mode: Geometry or IR: prepared spatial assets, channel metadata, and deterministic fallback.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

### 29.3 5.1

**Bus card: 5.1: Algorithmic**

- Layout: 5.1: explicit L/R/C/LFE/Ls/Rs routing and a declared LFE policy.
- Processing mode: Algorithmic: the core FDN/diffuser path with matched input/output layout.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: 5.1: Reverse**

- Layout: 5.1: explicit L/R/C/LFE/Ls/Rs routing and a declared LFE policy.
- Processing mode: Reverse: the bounded reverse window, channel alignment, and reported latency.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: 5.1: Freeze**

- Layout: 5.1: explicit L/R/C/LFE/Ls/Rs routing and a declared LFE policy.
- Processing mode: Freeze: sustained energy, channel stability, exit behavior, and safety limiting.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: 5.1: Geometry or IR**

- Layout: 5.1: explicit L/R/C/LFE/Ls/Rs routing and a declared LFE policy.
- Processing mode: Geometry or IR: prepared spatial assets, channel metadata, and deterministic fallback.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

### 29.4 7.1

**Bus card: 7.1: Algorithmic**

- Layout: 7.1: side/rear separation, normalization, and host channel-order verification.
- Processing mode: Algorithmic: the core FDN/diffuser path with matched input/output layout.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: 7.1: Reverse**

- Layout: 7.1: side/rear separation, normalization, and host channel-order verification.
- Processing mode: Reverse: the bounded reverse window, channel alignment, and reported latency.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: 7.1: Freeze**

- Layout: 7.1: side/rear separation, normalization, and host channel-order verification.
- Processing mode: Freeze: sustained energy, channel stability, exit behavior, and safety limiting.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: 7.1: Geometry or IR**

- Layout: 7.1: side/rear separation, normalization, and host channel-order verification.
- Processing mode: Geometry or IR: prepared spatial assets, channel metadata, and deterministic fallback.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

### 29.5 7.1.4

**Bus card: 7.1.4: Algorithmic**

- Layout: 7.1.4: bed plus height routing, elevation behavior, and immersive meter coverage.
- Processing mode: Algorithmic: the core FDN/diffuser path with matched input/output layout.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: 7.1.4: Reverse**

- Layout: 7.1.4: bed plus height routing, elevation behavior, and immersive meter coverage.
- Processing mode: Reverse: the bounded reverse window, channel alignment, and reported latency.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: 7.1.4: Freeze**

- Layout: 7.1.4: bed plus height routing, elevation behavior, and immersive meter coverage.
- Processing mode: Freeze: sustained energy, channel stability, exit behavior, and safety limiting.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: 7.1.4: Geometry or IR**

- Layout: 7.1.4: bed plus height routing, elevation behavior, and immersive meter coverage.
- Processing mode: Geometry or IR: prepared spatial assets, channel metadata, and deterministic fallback.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

### 29.6 First-order ambisonics

**Bus card: First-order ambisonics: Algorithmic**

- Layout: First-order ambisonics: ACN/SN3D ordering, rotation behavior, and decoder-independent energy.
- Processing mode: Algorithmic: the core FDN/diffuser path with matched input/output layout.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: First-order ambisonics: Reverse**

- Layout: First-order ambisonics: ACN/SN3D ordering, rotation behavior, and decoder-independent energy.
- Processing mode: Reverse: the bounded reverse window, channel alignment, and reported latency.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: First-order ambisonics: Freeze**

- Layout: First-order ambisonics: ACN/SN3D ordering, rotation behavior, and decoder-independent energy.
- Processing mode: Freeze: sustained energy, channel stability, exit behavior, and safety limiting.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

**Bus card: First-order ambisonics: Geometry or IR**

- Layout: First-order ambisonics: ACN/SN3D ordering, rotation behavior, and decoder-independent energy.
- Processing mode: Geometry or IR: prepared spatial assets, channel metadata, and deterministic fallback.
- Foundation status: mono/stereo only; larger layouts are future validation protocols.
- Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.

Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.

Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.

Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.


\newpage

## 30. Parameter Signal-Test Cards

### 30.1 Pre-Delay

**Signal-test card: Pre-Delay with Digital silence**

- Parameter: `pre_delay_ms`: separates the direct event from the room onset.
- Signal: Digital silence.
- Observation goal: detect denormals, stale buffers, uninitialized state, and noise-floor growth.
- Required transition behavior: use a ramp or delay-line crossfade.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the digital silence first with Pre-Delay at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to detect denormals, stale buffers, uninitialized state, and noise-floor growth. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Pre-Delay with Single-sample impulse**

- Parameter: `pre_delay_ms`: separates the direct event from the room onset.
- Signal: Single-sample impulse.
- Observation goal: reveal latency, early reflections, channel routing, and deterministic tail shape.
- Required transition behavior: use a ramp or delay-line crossfade.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the single-sample impulse first with Pre-Delay at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to reveal latency, early reflections, channel routing, and deterministic tail shape. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Pre-Delay with Full-scale alternating impulses**

- Parameter: `pre_delay_ms`: separates the direct event from the room onset.
- Signal: Full-scale alternating impulses.
- Observation goal: stress headroom, limiter response, mode transitions, and sign symmetry.
- Required transition behavior: use a ramp or delay-line crossfade.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the full-scale alternating impulses first with Pre-Delay at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to stress headroom, limiter response, mode transitions, and sign symmetry. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Pre-Delay with 80 Hz sine**

- Parameter: `pre_delay_ms`: separates the direct event from the room onset.
- Signal: 80 Hz sine.
- Observation goal: expose low-frequency decay, modulation, modal buildup, and channel phase differences.
- Required transition behavior: use a ramp or delay-line crossfade.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 80 hz sine first with Pre-Delay at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose low-frequency decay, modulation, modal buildup, and channel phase differences. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Pre-Delay with 8 kHz sine**

- Parameter: `pre_delay_ms`: separates the direct event from the room onset.
- Signal: 8 kHz sine.
- Observation goal: expose damping, interpolation, aliasing, and high-frequency stability.
- Required transition behavior: use a ramp or delay-line crossfade.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 8 khz sine first with Pre-Delay at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose damping, interpolation, aliasing, and high-frequency stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Pre-Delay with Pink noise burst**

- Parameter: `pre_delay_ms`: separates the direct event from the room onset.
- Signal: Pink noise burst.
- Observation goal: show broadband spectral decay, gain behavior, and early/late balance.
- Required transition behavior: use a ramp or delay-line crossfade.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the pink noise burst first with Pre-Delay at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to show broadband spectral decay, gain behavior, and early/late balance. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Pre-Delay with Dry speech phrase**

- Parameter: `pre_delay_ms`: separates the direct event from the room onset.
- Signal: Dry speech phrase.
- Observation goal: test intelligibility, sibilance, plosives, pre-delay, and ducking behavior.
- Required transition behavior: use a ramp or delay-line crossfade.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the dry speech phrase first with Pre-Delay at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test intelligibility, sibilance, plosives, pre-delay, and ducking behavior. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Pre-Delay with Transient drum loop**

- Parameter: `pre_delay_ms`: separates the direct event from the room onset.
- Signal: Transient drum loop.
- Observation goal: test attack preservation, density buildup, tempo interaction, and peak safety.
- Required transition behavior: use a ramp or delay-line crossfade.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the transient drum loop first with Pre-Delay at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test attack preservation, density buildup, tempo interaction, and peak safety. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Pre-Delay with Sustained harmonic pad**

- Parameter: `pre_delay_ms`: separates the direct event from the room onset.
- Signal: Sustained harmonic pad.
- Observation goal: test modulation smoothness, correlation, freeze energy, and long-term stability.
- Required transition behavior: use a ramp or delay-line crossfade.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sustained harmonic pad first with Pre-Delay at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test modulation smoothness, correlation, freeze energy, and long-term stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Pre-Delay with Sixty-second tail capture**

- Parameter: `pre_delay_ms`: separates the direct event from the room onset.
- Signal: Sixty-second tail capture.
- Observation goal: test memory stability, decay completion, noise floor, and repeatability.
- Required transition behavior: use a ramp or delay-line crossfade.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sixty-second tail capture first with Pre-Delay at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test memory stability, decay completion, noise floor, and repeatability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

### 30.2 Room Size

**Signal-test card: Room Size with Digital silence**

- Parameter: `room_size`: changes perceived scale and reflection spacing.
- Signal: Digital silence.
- Observation goal: detect denormals, stale buffers, uninitialized state, and noise-floor growth.
- Required transition behavior: stage structural changes outside the callback.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the digital silence first with Room Size at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to detect denormals, stale buffers, uninitialized state, and noise-floor growth. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Room Size with Single-sample impulse**

- Parameter: `room_size`: changes perceived scale and reflection spacing.
- Signal: Single-sample impulse.
- Observation goal: reveal latency, early reflections, channel routing, and deterministic tail shape.
- Required transition behavior: stage structural changes outside the callback.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the single-sample impulse first with Room Size at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to reveal latency, early reflections, channel routing, and deterministic tail shape. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Room Size with Full-scale alternating impulses**

- Parameter: `room_size`: changes perceived scale and reflection spacing.
- Signal: Full-scale alternating impulses.
- Observation goal: stress headroom, limiter response, mode transitions, and sign symmetry.
- Required transition behavior: stage structural changes outside the callback.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the full-scale alternating impulses first with Room Size at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to stress headroom, limiter response, mode transitions, and sign symmetry. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Room Size with 80 Hz sine**

- Parameter: `room_size`: changes perceived scale and reflection spacing.
- Signal: 80 Hz sine.
- Observation goal: expose low-frequency decay, modulation, modal buildup, and channel phase differences.
- Required transition behavior: stage structural changes outside the callback.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 80 hz sine first with Room Size at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose low-frequency decay, modulation, modal buildup, and channel phase differences. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Room Size with 8 kHz sine**

- Parameter: `room_size`: changes perceived scale and reflection spacing.
- Signal: 8 kHz sine.
- Observation goal: expose damping, interpolation, aliasing, and high-frequency stability.
- Required transition behavior: stage structural changes outside the callback.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 8 khz sine first with Room Size at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose damping, interpolation, aliasing, and high-frequency stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Room Size with Pink noise burst**

- Parameter: `room_size`: changes perceived scale and reflection spacing.
- Signal: Pink noise burst.
- Observation goal: show broadband spectral decay, gain behavior, and early/late balance.
- Required transition behavior: stage structural changes outside the callback.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the pink noise burst first with Room Size at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to show broadband spectral decay, gain behavior, and early/late balance. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Room Size with Dry speech phrase**

- Parameter: `room_size`: changes perceived scale and reflection spacing.
- Signal: Dry speech phrase.
- Observation goal: test intelligibility, sibilance, plosives, pre-delay, and ducking behavior.
- Required transition behavior: stage structural changes outside the callback.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the dry speech phrase first with Room Size at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test intelligibility, sibilance, plosives, pre-delay, and ducking behavior. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Room Size with Transient drum loop**

- Parameter: `room_size`: changes perceived scale and reflection spacing.
- Signal: Transient drum loop.
- Observation goal: test attack preservation, density buildup, tempo interaction, and peak safety.
- Required transition behavior: stage structural changes outside the callback.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the transient drum loop first with Room Size at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test attack preservation, density buildup, tempo interaction, and peak safety. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Room Size with Sustained harmonic pad**

- Parameter: `room_size`: changes perceived scale and reflection spacing.
- Signal: Sustained harmonic pad.
- Observation goal: test modulation smoothness, correlation, freeze energy, and long-term stability.
- Required transition behavior: stage structural changes outside the callback.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sustained harmonic pad first with Room Size at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test modulation smoothness, correlation, freeze energy, and long-term stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Room Size with Sixty-second tail capture**

- Parameter: `room_size`: changes perceived scale and reflection spacing.
- Signal: Sixty-second tail capture.
- Observation goal: test memory stability, decay completion, noise floor, and repeatability.
- Required transition behavior: stage structural changes outside the callback.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sixty-second tail capture first with Room Size at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test memory stability, decay completion, noise floor, and repeatability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

### 30.3 RT60 Coarse

**Signal-test card: RT60 Coarse with Digital silence**

- Parameter: `rt60_coarse`: moves through the full logarithmic decay range.
- Signal: Digital silence.
- Observation goal: detect denormals, stale buffers, uninitialized state, and noise-floor growth.
- Required transition behavior: display the effective seconds value.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the digital silence first with RT60 Coarse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to detect denormals, stale buffers, uninitialized state, and noise-floor growth. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Coarse with Single-sample impulse**

- Parameter: `rt60_coarse`: moves through the full logarithmic decay range.
- Signal: Single-sample impulse.
- Observation goal: reveal latency, early reflections, channel routing, and deterministic tail shape.
- Required transition behavior: display the effective seconds value.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the single-sample impulse first with RT60 Coarse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to reveal latency, early reflections, channel routing, and deterministic tail shape. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Coarse with Full-scale alternating impulses**

- Parameter: `rt60_coarse`: moves through the full logarithmic decay range.
- Signal: Full-scale alternating impulses.
- Observation goal: stress headroom, limiter response, mode transitions, and sign symmetry.
- Required transition behavior: display the effective seconds value.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the full-scale alternating impulses first with RT60 Coarse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to stress headroom, limiter response, mode transitions, and sign symmetry. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Coarse with 80 Hz sine**

- Parameter: `rt60_coarse`: moves through the full logarithmic decay range.
- Signal: 80 Hz sine.
- Observation goal: expose low-frequency decay, modulation, modal buildup, and channel phase differences.
- Required transition behavior: display the effective seconds value.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 80 hz sine first with RT60 Coarse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose low-frequency decay, modulation, modal buildup, and channel phase differences. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Coarse with 8 kHz sine**

- Parameter: `rt60_coarse`: moves through the full logarithmic decay range.
- Signal: 8 kHz sine.
- Observation goal: expose damping, interpolation, aliasing, and high-frequency stability.
- Required transition behavior: display the effective seconds value.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 8 khz sine first with RT60 Coarse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose damping, interpolation, aliasing, and high-frequency stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Coarse with Pink noise burst**

- Parameter: `rt60_coarse`: moves through the full logarithmic decay range.
- Signal: Pink noise burst.
- Observation goal: show broadband spectral decay, gain behavior, and early/late balance.
- Required transition behavior: display the effective seconds value.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the pink noise burst first with RT60 Coarse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to show broadband spectral decay, gain behavior, and early/late balance. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Coarse with Dry speech phrase**

- Parameter: `rt60_coarse`: moves through the full logarithmic decay range.
- Signal: Dry speech phrase.
- Observation goal: test intelligibility, sibilance, plosives, pre-delay, and ducking behavior.
- Required transition behavior: display the effective seconds value.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the dry speech phrase first with RT60 Coarse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test intelligibility, sibilance, plosives, pre-delay, and ducking behavior. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Coarse with Transient drum loop**

- Parameter: `rt60_coarse`: moves through the full logarithmic decay range.
- Signal: Transient drum loop.
- Observation goal: test attack preservation, density buildup, tempo interaction, and peak safety.
- Required transition behavior: display the effective seconds value.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the transient drum loop first with RT60 Coarse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test attack preservation, density buildup, tempo interaction, and peak safety. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Coarse with Sustained harmonic pad**

- Parameter: `rt60_coarse`: moves through the full logarithmic decay range.
- Signal: Sustained harmonic pad.
- Observation goal: test modulation smoothness, correlation, freeze energy, and long-term stability.
- Required transition behavior: display the effective seconds value.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sustained harmonic pad first with RT60 Coarse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test modulation smoothness, correlation, freeze energy, and long-term stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Coarse with Sixty-second tail capture**

- Parameter: `rt60_coarse`: moves through the full logarithmic decay range.
- Signal: Sixty-second tail capture.
- Observation goal: test memory stability, decay completion, noise floor, and repeatability.
- Required transition behavior: display the effective seconds value.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sixty-second tail capture first with RT60 Coarse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test memory stability, decay completion, noise floor, and repeatability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

### 30.4 RT60 Fine

**Signal-test card: RT60 Fine with Digital silence**

- Parameter: `rt60_fine`: trims decay proportionally around the coarse value.
- Signal: Digital silence.
- Observation goal: detect denormals, stale buffers, uninitialized state, and noise-floor growth.
- Required transition behavior: keep zero as the exact neutral point.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the digital silence first with RT60 Fine at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to detect denormals, stale buffers, uninitialized state, and noise-floor growth. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Fine with Single-sample impulse**

- Parameter: `rt60_fine`: trims decay proportionally around the coarse value.
- Signal: Single-sample impulse.
- Observation goal: reveal latency, early reflections, channel routing, and deterministic tail shape.
- Required transition behavior: keep zero as the exact neutral point.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the single-sample impulse first with RT60 Fine at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to reveal latency, early reflections, channel routing, and deterministic tail shape. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Fine with Full-scale alternating impulses**

- Parameter: `rt60_fine`: trims decay proportionally around the coarse value.
- Signal: Full-scale alternating impulses.
- Observation goal: stress headroom, limiter response, mode transitions, and sign symmetry.
- Required transition behavior: keep zero as the exact neutral point.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the full-scale alternating impulses first with RT60 Fine at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to stress headroom, limiter response, mode transitions, and sign symmetry. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Fine with 80 Hz sine**

- Parameter: `rt60_fine`: trims decay proportionally around the coarse value.
- Signal: 80 Hz sine.
- Observation goal: expose low-frequency decay, modulation, modal buildup, and channel phase differences.
- Required transition behavior: keep zero as the exact neutral point.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 80 hz sine first with RT60 Fine at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose low-frequency decay, modulation, modal buildup, and channel phase differences. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Fine with 8 kHz sine**

- Parameter: `rt60_fine`: trims decay proportionally around the coarse value.
- Signal: 8 kHz sine.
- Observation goal: expose damping, interpolation, aliasing, and high-frequency stability.
- Required transition behavior: keep zero as the exact neutral point.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 8 khz sine first with RT60 Fine at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose damping, interpolation, aliasing, and high-frequency stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Fine with Pink noise burst**

- Parameter: `rt60_fine`: trims decay proportionally around the coarse value.
- Signal: Pink noise burst.
- Observation goal: show broadband spectral decay, gain behavior, and early/late balance.
- Required transition behavior: keep zero as the exact neutral point.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the pink noise burst first with RT60 Fine at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to show broadband spectral decay, gain behavior, and early/late balance. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Fine with Dry speech phrase**

- Parameter: `rt60_fine`: trims decay proportionally around the coarse value.
- Signal: Dry speech phrase.
- Observation goal: test intelligibility, sibilance, plosives, pre-delay, and ducking behavior.
- Required transition behavior: keep zero as the exact neutral point.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the dry speech phrase first with RT60 Fine at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test intelligibility, sibilance, plosives, pre-delay, and ducking behavior. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Fine with Transient drum loop**

- Parameter: `rt60_fine`: trims decay proportionally around the coarse value.
- Signal: Transient drum loop.
- Observation goal: test attack preservation, density buildup, tempo interaction, and peak safety.
- Required transition behavior: keep zero as the exact neutral point.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the transient drum loop first with RT60 Fine at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test attack preservation, density buildup, tempo interaction, and peak safety. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Fine with Sustained harmonic pad**

- Parameter: `rt60_fine`: trims decay proportionally around the coarse value.
- Signal: Sustained harmonic pad.
- Observation goal: test modulation smoothness, correlation, freeze energy, and long-term stability.
- Required transition behavior: keep zero as the exact neutral point.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sustained harmonic pad first with RT60 Fine at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test modulation smoothness, correlation, freeze energy, and long-term stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: RT60 Fine with Sixty-second tail capture**

- Parameter: `rt60_fine`: trims decay proportionally around the coarse value.
- Signal: Sixty-second tail capture.
- Observation goal: test memory stability, decay completion, noise floor, and repeatability.
- Required transition behavior: keep zero as the exact neutral point.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sixty-second tail capture first with RT60 Fine at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test memory stability, decay completion, noise floor, and repeatability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

### 30.5 Damping

**Signal-test card: Damping with Digital silence**

- Parameter: `damping`: changes high-frequency persistence.
- Signal: Digital silence.
- Observation goal: detect denormals, stale buffers, uninitialized state, and noise-floor growth.
- Required transition behavior: interpolate stable filter coefficients.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the digital silence first with Damping at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to detect denormals, stale buffers, uninitialized state, and noise-floor growth. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Damping with Single-sample impulse**

- Parameter: `damping`: changes high-frequency persistence.
- Signal: Single-sample impulse.
- Observation goal: reveal latency, early reflections, channel routing, and deterministic tail shape.
- Required transition behavior: interpolate stable filter coefficients.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the single-sample impulse first with Damping at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to reveal latency, early reflections, channel routing, and deterministic tail shape. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Damping with Full-scale alternating impulses**

- Parameter: `damping`: changes high-frequency persistence.
- Signal: Full-scale alternating impulses.
- Observation goal: stress headroom, limiter response, mode transitions, and sign symmetry.
- Required transition behavior: interpolate stable filter coefficients.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the full-scale alternating impulses first with Damping at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to stress headroom, limiter response, mode transitions, and sign symmetry. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Damping with 80 Hz sine**

- Parameter: `damping`: changes high-frequency persistence.
- Signal: 80 Hz sine.
- Observation goal: expose low-frequency decay, modulation, modal buildup, and channel phase differences.
- Required transition behavior: interpolate stable filter coefficients.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 80 hz sine first with Damping at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose low-frequency decay, modulation, modal buildup, and channel phase differences. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Damping with 8 kHz sine**

- Parameter: `damping`: changes high-frequency persistence.
- Signal: 8 kHz sine.
- Observation goal: expose damping, interpolation, aliasing, and high-frequency stability.
- Required transition behavior: interpolate stable filter coefficients.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 8 khz sine first with Damping at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose damping, interpolation, aliasing, and high-frequency stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Damping with Pink noise burst**

- Parameter: `damping`: changes high-frequency persistence.
- Signal: Pink noise burst.
- Observation goal: show broadband spectral decay, gain behavior, and early/late balance.
- Required transition behavior: interpolate stable filter coefficients.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the pink noise burst first with Damping at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to show broadband spectral decay, gain behavior, and early/late balance. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Damping with Dry speech phrase**

- Parameter: `damping`: changes high-frequency persistence.
- Signal: Dry speech phrase.
- Observation goal: test intelligibility, sibilance, plosives, pre-delay, and ducking behavior.
- Required transition behavior: interpolate stable filter coefficients.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the dry speech phrase first with Damping at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test intelligibility, sibilance, plosives, pre-delay, and ducking behavior. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Damping with Transient drum loop**

- Parameter: `damping`: changes high-frequency persistence.
- Signal: Transient drum loop.
- Observation goal: test attack preservation, density buildup, tempo interaction, and peak safety.
- Required transition behavior: interpolate stable filter coefficients.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the transient drum loop first with Damping at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test attack preservation, density buildup, tempo interaction, and peak safety. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Damping with Sustained harmonic pad**

- Parameter: `damping`: changes high-frequency persistence.
- Signal: Sustained harmonic pad.
- Observation goal: test modulation smoothness, correlation, freeze energy, and long-term stability.
- Required transition behavior: interpolate stable filter coefficients.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sustained harmonic pad first with Damping at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test modulation smoothness, correlation, freeze energy, and long-term stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Damping with Sixty-second tail capture**

- Parameter: `damping`: changes high-frequency persistence.
- Signal: Sixty-second tail capture.
- Observation goal: test memory stability, decay completion, noise floor, and repeatability.
- Required transition behavior: interpolate stable filter coefficients.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sixty-second tail capture first with Damping at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test memory stability, decay completion, noise floor, and repeatability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

### 30.6 Width

**Signal-test card: Width with Digital silence**

- Parameter: `width`: changes lateral energy and correlation.
- Signal: Digital silence.
- Observation goal: detect denormals, stale buffers, uninitialized state, and noise-floor growth.
- Required transition behavior: monitor mono compatibility during movement.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the digital silence first with Width at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to detect denormals, stale buffers, uninitialized state, and noise-floor growth. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Width with Single-sample impulse**

- Parameter: `width`: changes lateral energy and correlation.
- Signal: Single-sample impulse.
- Observation goal: reveal latency, early reflections, channel routing, and deterministic tail shape.
- Required transition behavior: monitor mono compatibility during movement.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the single-sample impulse first with Width at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to reveal latency, early reflections, channel routing, and deterministic tail shape. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Width with Full-scale alternating impulses**

- Parameter: `width`: changes lateral energy and correlation.
- Signal: Full-scale alternating impulses.
- Observation goal: stress headroom, limiter response, mode transitions, and sign symmetry.
- Required transition behavior: monitor mono compatibility during movement.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the full-scale alternating impulses first with Width at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to stress headroom, limiter response, mode transitions, and sign symmetry. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Width with 80 Hz sine**

- Parameter: `width`: changes lateral energy and correlation.
- Signal: 80 Hz sine.
- Observation goal: expose low-frequency decay, modulation, modal buildup, and channel phase differences.
- Required transition behavior: monitor mono compatibility during movement.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 80 hz sine first with Width at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose low-frequency decay, modulation, modal buildup, and channel phase differences. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Width with 8 kHz sine**

- Parameter: `width`: changes lateral energy and correlation.
- Signal: 8 kHz sine.
- Observation goal: expose damping, interpolation, aliasing, and high-frequency stability.
- Required transition behavior: monitor mono compatibility during movement.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 8 khz sine first with Width at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose damping, interpolation, aliasing, and high-frequency stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Width with Pink noise burst**

- Parameter: `width`: changes lateral energy and correlation.
- Signal: Pink noise burst.
- Observation goal: show broadband spectral decay, gain behavior, and early/late balance.
- Required transition behavior: monitor mono compatibility during movement.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the pink noise burst first with Width at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to show broadband spectral decay, gain behavior, and early/late balance. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Width with Dry speech phrase**

- Parameter: `width`: changes lateral energy and correlation.
- Signal: Dry speech phrase.
- Observation goal: test intelligibility, sibilance, plosives, pre-delay, and ducking behavior.
- Required transition behavior: monitor mono compatibility during movement.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the dry speech phrase first with Width at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test intelligibility, sibilance, plosives, pre-delay, and ducking behavior. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Width with Transient drum loop**

- Parameter: `width`: changes lateral energy and correlation.
- Signal: Transient drum loop.
- Observation goal: test attack preservation, density buildup, tempo interaction, and peak safety.
- Required transition behavior: monitor mono compatibility during movement.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the transient drum loop first with Width at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test attack preservation, density buildup, tempo interaction, and peak safety. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Width with Sustained harmonic pad**

- Parameter: `width`: changes lateral energy and correlation.
- Signal: Sustained harmonic pad.
- Observation goal: test modulation smoothness, correlation, freeze energy, and long-term stability.
- Required transition behavior: monitor mono compatibility during movement.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sustained harmonic pad first with Width at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test modulation smoothness, correlation, freeze energy, and long-term stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Width with Sixty-second tail capture**

- Parameter: `width`: changes lateral energy and correlation.
- Signal: Sixty-second tail capture.
- Observation goal: test memory stability, decay completion, noise floor, and repeatability.
- Required transition behavior: monitor mono compatibility during movement.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sixty-second tail capture first with Width at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test memory stability, decay completion, noise floor, and repeatability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

### 30.7 Diffusion

**Signal-test card: Diffusion with Digital silence**

- Parameter: `diffusion`: changes echo-density buildup.
- Signal: Digital silence.
- Observation goal: detect denormals, stale buffers, uninitialized state, and noise-floor growth.
- Required transition behavior: crossfade when topology must change.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the digital silence first with Diffusion at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to detect denormals, stale buffers, uninitialized state, and noise-floor growth. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Diffusion with Single-sample impulse**

- Parameter: `diffusion`: changes echo-density buildup.
- Signal: Single-sample impulse.
- Observation goal: reveal latency, early reflections, channel routing, and deterministic tail shape.
- Required transition behavior: crossfade when topology must change.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the single-sample impulse first with Diffusion at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to reveal latency, early reflections, channel routing, and deterministic tail shape. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Diffusion with Full-scale alternating impulses**

- Parameter: `diffusion`: changes echo-density buildup.
- Signal: Full-scale alternating impulses.
- Observation goal: stress headroom, limiter response, mode transitions, and sign symmetry.
- Required transition behavior: crossfade when topology must change.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the full-scale alternating impulses first with Diffusion at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to stress headroom, limiter response, mode transitions, and sign symmetry. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Diffusion with 80 Hz sine**

- Parameter: `diffusion`: changes echo-density buildup.
- Signal: 80 Hz sine.
- Observation goal: expose low-frequency decay, modulation, modal buildup, and channel phase differences.
- Required transition behavior: crossfade when topology must change.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 80 hz sine first with Diffusion at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose low-frequency decay, modulation, modal buildup, and channel phase differences. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Diffusion with 8 kHz sine**

- Parameter: `diffusion`: changes echo-density buildup.
- Signal: 8 kHz sine.
- Observation goal: expose damping, interpolation, aliasing, and high-frequency stability.
- Required transition behavior: crossfade when topology must change.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 8 khz sine first with Diffusion at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose damping, interpolation, aliasing, and high-frequency stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Diffusion with Pink noise burst**

- Parameter: `diffusion`: changes echo-density buildup.
- Signal: Pink noise burst.
- Observation goal: show broadband spectral decay, gain behavior, and early/late balance.
- Required transition behavior: crossfade when topology must change.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the pink noise burst first with Diffusion at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to show broadband spectral decay, gain behavior, and early/late balance. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Diffusion with Dry speech phrase**

- Parameter: `diffusion`: changes echo-density buildup.
- Signal: Dry speech phrase.
- Observation goal: test intelligibility, sibilance, plosives, pre-delay, and ducking behavior.
- Required transition behavior: crossfade when topology must change.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the dry speech phrase first with Diffusion at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test intelligibility, sibilance, plosives, pre-delay, and ducking behavior. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Diffusion with Transient drum loop**

- Parameter: `diffusion`: changes echo-density buildup.
- Signal: Transient drum loop.
- Observation goal: test attack preservation, density buildup, tempo interaction, and peak safety.
- Required transition behavior: crossfade when topology must change.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the transient drum loop first with Diffusion at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test attack preservation, density buildup, tempo interaction, and peak safety. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Diffusion with Sustained harmonic pad**

- Parameter: `diffusion`: changes echo-density buildup.
- Signal: Sustained harmonic pad.
- Observation goal: test modulation smoothness, correlation, freeze energy, and long-term stability.
- Required transition behavior: crossfade when topology must change.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sustained harmonic pad first with Diffusion at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test modulation smoothness, correlation, freeze energy, and long-term stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Diffusion with Sixty-second tail capture**

- Parameter: `diffusion`: changes echo-density buildup.
- Signal: Sixty-second tail capture.
- Observation goal: test memory stability, decay completion, noise floor, and repeatability.
- Required transition behavior: crossfade when topology must change.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sixty-second tail capture first with Diffusion at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test memory stability, decay completion, noise floor, and repeatability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

### 30.8 Wet

**Signal-test card: Wet with Digital silence**

- Parameter: `wet`: sets processed contribution.
- Signal: Digital silence.
- Observation goal: detect denormals, stale buffers, uninitialized state, and noise-floor growth.
- Required transition behavior: choose and document the mix law.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the digital silence first with Wet at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to detect denormals, stale buffers, uninitialized state, and noise-floor growth. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Wet with Single-sample impulse**

- Parameter: `wet`: sets processed contribution.
- Signal: Single-sample impulse.
- Observation goal: reveal latency, early reflections, channel routing, and deterministic tail shape.
- Required transition behavior: choose and document the mix law.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the single-sample impulse first with Wet at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to reveal latency, early reflections, channel routing, and deterministic tail shape. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Wet with Full-scale alternating impulses**

- Parameter: `wet`: sets processed contribution.
- Signal: Full-scale alternating impulses.
- Observation goal: stress headroom, limiter response, mode transitions, and sign symmetry.
- Required transition behavior: choose and document the mix law.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the full-scale alternating impulses first with Wet at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to stress headroom, limiter response, mode transitions, and sign symmetry. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Wet with 80 Hz sine**

- Parameter: `wet`: sets processed contribution.
- Signal: 80 Hz sine.
- Observation goal: expose low-frequency decay, modulation, modal buildup, and channel phase differences.
- Required transition behavior: choose and document the mix law.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 80 hz sine first with Wet at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose low-frequency decay, modulation, modal buildup, and channel phase differences. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Wet with 8 kHz sine**

- Parameter: `wet`: sets processed contribution.
- Signal: 8 kHz sine.
- Observation goal: expose damping, interpolation, aliasing, and high-frequency stability.
- Required transition behavior: choose and document the mix law.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 8 khz sine first with Wet at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose damping, interpolation, aliasing, and high-frequency stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Wet with Pink noise burst**

- Parameter: `wet`: sets processed contribution.
- Signal: Pink noise burst.
- Observation goal: show broadband spectral decay, gain behavior, and early/late balance.
- Required transition behavior: choose and document the mix law.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the pink noise burst first with Wet at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to show broadband spectral decay, gain behavior, and early/late balance. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Wet with Dry speech phrase**

- Parameter: `wet`: sets processed contribution.
- Signal: Dry speech phrase.
- Observation goal: test intelligibility, sibilance, plosives, pre-delay, and ducking behavior.
- Required transition behavior: choose and document the mix law.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the dry speech phrase first with Wet at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test intelligibility, sibilance, plosives, pre-delay, and ducking behavior. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Wet with Transient drum loop**

- Parameter: `wet`: sets processed contribution.
- Signal: Transient drum loop.
- Observation goal: test attack preservation, density buildup, tempo interaction, and peak safety.
- Required transition behavior: choose and document the mix law.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the transient drum loop first with Wet at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test attack preservation, density buildup, tempo interaction, and peak safety. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Wet with Sustained harmonic pad**

- Parameter: `wet`: sets processed contribution.
- Signal: Sustained harmonic pad.
- Observation goal: test modulation smoothness, correlation, freeze energy, and long-term stability.
- Required transition behavior: choose and document the mix law.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sustained harmonic pad first with Wet at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test modulation smoothness, correlation, freeze energy, and long-term stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Wet with Sixty-second tail capture**

- Parameter: `wet`: sets processed contribution.
- Signal: Sixty-second tail capture.
- Observation goal: test memory stability, decay completion, noise floor, and repeatability.
- Required transition behavior: choose and document the mix law.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sixty-second tail capture first with Wet at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test memory stability, decay completion, noise floor, and repeatability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

### 30.9 Dry

**Signal-test card: Dry with Digital silence**

- Parameter: `dry`: sets direct contribution.
- Signal: Digital silence.
- Observation goal: detect denormals, stale buffers, uninitialized state, and noise-floor growth.
- Required transition behavior: preserve bypass and gain staging.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the digital silence first with Dry at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to detect denormals, stale buffers, uninitialized state, and noise-floor growth. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Dry with Single-sample impulse**

- Parameter: `dry`: sets direct contribution.
- Signal: Single-sample impulse.
- Observation goal: reveal latency, early reflections, channel routing, and deterministic tail shape.
- Required transition behavior: preserve bypass and gain staging.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the single-sample impulse first with Dry at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to reveal latency, early reflections, channel routing, and deterministic tail shape. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Dry with Full-scale alternating impulses**

- Parameter: `dry`: sets direct contribution.
- Signal: Full-scale alternating impulses.
- Observation goal: stress headroom, limiter response, mode transitions, and sign symmetry.
- Required transition behavior: preserve bypass and gain staging.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the full-scale alternating impulses first with Dry at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to stress headroom, limiter response, mode transitions, and sign symmetry. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Dry with 80 Hz sine**

- Parameter: `dry`: sets direct contribution.
- Signal: 80 Hz sine.
- Observation goal: expose low-frequency decay, modulation, modal buildup, and channel phase differences.
- Required transition behavior: preserve bypass and gain staging.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 80 hz sine first with Dry at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose low-frequency decay, modulation, modal buildup, and channel phase differences. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Dry with 8 kHz sine**

- Parameter: `dry`: sets direct contribution.
- Signal: 8 kHz sine.
- Observation goal: expose damping, interpolation, aliasing, and high-frequency stability.
- Required transition behavior: preserve bypass and gain staging.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 8 khz sine first with Dry at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose damping, interpolation, aliasing, and high-frequency stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Dry with Pink noise burst**

- Parameter: `dry`: sets direct contribution.
- Signal: Pink noise burst.
- Observation goal: show broadband spectral decay, gain behavior, and early/late balance.
- Required transition behavior: preserve bypass and gain staging.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the pink noise burst first with Dry at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to show broadband spectral decay, gain behavior, and early/late balance. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Dry with Dry speech phrase**

- Parameter: `dry`: sets direct contribution.
- Signal: Dry speech phrase.
- Observation goal: test intelligibility, sibilance, plosives, pre-delay, and ducking behavior.
- Required transition behavior: preserve bypass and gain staging.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the dry speech phrase first with Dry at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test intelligibility, sibilance, plosives, pre-delay, and ducking behavior. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Dry with Transient drum loop**

- Parameter: `dry`: sets direct contribution.
- Signal: Transient drum loop.
- Observation goal: test attack preservation, density buildup, tempo interaction, and peak safety.
- Required transition behavior: preserve bypass and gain staging.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the transient drum loop first with Dry at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test attack preservation, density buildup, tempo interaction, and peak safety. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Dry with Sustained harmonic pad**

- Parameter: `dry`: sets direct contribution.
- Signal: Sustained harmonic pad.
- Observation goal: test modulation smoothness, correlation, freeze energy, and long-term stability.
- Required transition behavior: preserve bypass and gain staging.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sustained harmonic pad first with Dry at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test modulation smoothness, correlation, freeze energy, and long-term stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Dry with Sixty-second tail capture**

- Parameter: `dry`: sets direct contribution.
- Signal: Sixty-second tail capture.
- Observation goal: test memory stability, decay completion, noise floor, and repeatability.
- Required transition behavior: preserve bypass and gain staging.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sixty-second tail capture first with Dry at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test memory stability, decay completion, noise floor, and repeatability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

### 30.10 Freeze

**Signal-test card: Freeze with Digital silence**

- Parameter: `freeze`: changes network energy behavior.
- Signal: Digital silence.
- Observation goal: detect denormals, stale buffers, uninitialized state, and noise-floor growth.
- Required transition behavior: use a debounced, smoothed mode transition.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the digital silence first with Freeze at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to detect denormals, stale buffers, uninitialized state, and noise-floor growth. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Freeze with Single-sample impulse**

- Parameter: `freeze`: changes network energy behavior.
- Signal: Single-sample impulse.
- Observation goal: reveal latency, early reflections, channel routing, and deterministic tail shape.
- Required transition behavior: use a debounced, smoothed mode transition.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the single-sample impulse first with Freeze at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to reveal latency, early reflections, channel routing, and deterministic tail shape. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Freeze with Full-scale alternating impulses**

- Parameter: `freeze`: changes network energy behavior.
- Signal: Full-scale alternating impulses.
- Observation goal: stress headroom, limiter response, mode transitions, and sign symmetry.
- Required transition behavior: use a debounced, smoothed mode transition.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the full-scale alternating impulses first with Freeze at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to stress headroom, limiter response, mode transitions, and sign symmetry. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Freeze with 80 Hz sine**

- Parameter: `freeze`: changes network energy behavior.
- Signal: 80 Hz sine.
- Observation goal: expose low-frequency decay, modulation, modal buildup, and channel phase differences.
- Required transition behavior: use a debounced, smoothed mode transition.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 80 hz sine first with Freeze at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose low-frequency decay, modulation, modal buildup, and channel phase differences. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Freeze with 8 kHz sine**

- Parameter: `freeze`: changes network energy behavior.
- Signal: 8 kHz sine.
- Observation goal: expose damping, interpolation, aliasing, and high-frequency stability.
- Required transition behavior: use a debounced, smoothed mode transition.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 8 khz sine first with Freeze at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose damping, interpolation, aliasing, and high-frequency stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Freeze with Pink noise burst**

- Parameter: `freeze`: changes network energy behavior.
- Signal: Pink noise burst.
- Observation goal: show broadband spectral decay, gain behavior, and early/late balance.
- Required transition behavior: use a debounced, smoothed mode transition.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the pink noise burst first with Freeze at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to show broadband spectral decay, gain behavior, and early/late balance. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Freeze with Dry speech phrase**

- Parameter: `freeze`: changes network energy behavior.
- Signal: Dry speech phrase.
- Observation goal: test intelligibility, sibilance, plosives, pre-delay, and ducking behavior.
- Required transition behavior: use a debounced, smoothed mode transition.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the dry speech phrase first with Freeze at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test intelligibility, sibilance, plosives, pre-delay, and ducking behavior. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Freeze with Transient drum loop**

- Parameter: `freeze`: changes network energy behavior.
- Signal: Transient drum loop.
- Observation goal: test attack preservation, density buildup, tempo interaction, and peak safety.
- Required transition behavior: use a debounced, smoothed mode transition.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the transient drum loop first with Freeze at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test attack preservation, density buildup, tempo interaction, and peak safety. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Freeze with Sustained harmonic pad**

- Parameter: `freeze`: changes network energy behavior.
- Signal: Sustained harmonic pad.
- Observation goal: test modulation smoothness, correlation, freeze energy, and long-term stability.
- Required transition behavior: use a debounced, smoothed mode transition.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sustained harmonic pad first with Freeze at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test modulation smoothness, correlation, freeze energy, and long-term stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Freeze with Sixty-second tail capture**

- Parameter: `freeze`: changes network energy behavior.
- Signal: Sixty-second tail capture.
- Observation goal: test memory stability, decay completion, noise floor, and repeatability.
- Required transition behavior: use a debounced, smoothed mode transition.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sixty-second tail capture first with Freeze at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test memory stability, decay completion, noise floor, and repeatability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

### 30.11 Reverse

**Signal-test card: Reverse with Digital silence**

- Parameter: `reverse`: changes the envelope and buffering model.
- Signal: Digital silence.
- Observation goal: detect denormals, stale buffers, uninitialized state, and noise-floor growth.
- Required transition behavior: report added latency before activation.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the digital silence first with Reverse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to detect denormals, stale buffers, uninitialized state, and noise-floor growth. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Reverse with Single-sample impulse**

- Parameter: `reverse`: changes the envelope and buffering model.
- Signal: Single-sample impulse.
- Observation goal: reveal latency, early reflections, channel routing, and deterministic tail shape.
- Required transition behavior: report added latency before activation.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the single-sample impulse first with Reverse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to reveal latency, early reflections, channel routing, and deterministic tail shape. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Reverse with Full-scale alternating impulses**

- Parameter: `reverse`: changes the envelope and buffering model.
- Signal: Full-scale alternating impulses.
- Observation goal: stress headroom, limiter response, mode transitions, and sign symmetry.
- Required transition behavior: report added latency before activation.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the full-scale alternating impulses first with Reverse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to stress headroom, limiter response, mode transitions, and sign symmetry. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Reverse with 80 Hz sine**

- Parameter: `reverse`: changes the envelope and buffering model.
- Signal: 80 Hz sine.
- Observation goal: expose low-frequency decay, modulation, modal buildup, and channel phase differences.
- Required transition behavior: report added latency before activation.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 80 hz sine first with Reverse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose low-frequency decay, modulation, modal buildup, and channel phase differences. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Reverse with 8 kHz sine**

- Parameter: `reverse`: changes the envelope and buffering model.
- Signal: 8 kHz sine.
- Observation goal: expose damping, interpolation, aliasing, and high-frequency stability.
- Required transition behavior: report added latency before activation.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 8 khz sine first with Reverse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose damping, interpolation, aliasing, and high-frequency stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Reverse with Pink noise burst**

- Parameter: `reverse`: changes the envelope and buffering model.
- Signal: Pink noise burst.
- Observation goal: show broadband spectral decay, gain behavior, and early/late balance.
- Required transition behavior: report added latency before activation.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the pink noise burst first with Reverse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to show broadband spectral decay, gain behavior, and early/late balance. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Reverse with Dry speech phrase**

- Parameter: `reverse`: changes the envelope and buffering model.
- Signal: Dry speech phrase.
- Observation goal: test intelligibility, sibilance, plosives, pre-delay, and ducking behavior.
- Required transition behavior: report added latency before activation.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the dry speech phrase first with Reverse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test intelligibility, sibilance, plosives, pre-delay, and ducking behavior. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Reverse with Transient drum loop**

- Parameter: `reverse`: changes the envelope and buffering model.
- Signal: Transient drum loop.
- Observation goal: test attack preservation, density buildup, tempo interaction, and peak safety.
- Required transition behavior: report added latency before activation.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the transient drum loop first with Reverse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test attack preservation, density buildup, tempo interaction, and peak safety. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Reverse with Sustained harmonic pad**

- Parameter: `reverse`: changes the envelope and buffering model.
- Signal: Sustained harmonic pad.
- Observation goal: test modulation smoothness, correlation, freeze energy, and long-term stability.
- Required transition behavior: report added latency before activation.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sustained harmonic pad first with Reverse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test modulation smoothness, correlation, freeze energy, and long-term stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Reverse with Sixty-second tail capture**

- Parameter: `reverse`: changes the envelope and buffering model.
- Signal: Sixty-second tail capture.
- Observation goal: test memory stability, decay completion, noise floor, and repeatability.
- Required transition behavior: report added latency before activation.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sixty-second tail capture first with Reverse at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test memory stability, decay completion, noise floor, and repeatability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

### 30.12 Quality

**Signal-test card: Quality with Digital silence**

- Parameter: `quality_mode`: selects the internal rate policy.
- Signal: Digital silence.
- Observation goal: detect denormals, stale buffers, uninitialized state, and noise-floor growth.
- Required transition behavior: apply through a safe reprepare boundary.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the digital silence first with Quality at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to detect denormals, stale buffers, uninitialized state, and noise-floor growth. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Quality with Single-sample impulse**

- Parameter: `quality_mode`: selects the internal rate policy.
- Signal: Single-sample impulse.
- Observation goal: reveal latency, early reflections, channel routing, and deterministic tail shape.
- Required transition behavior: apply through a safe reprepare boundary.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the single-sample impulse first with Quality at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to reveal latency, early reflections, channel routing, and deterministic tail shape. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Quality with Full-scale alternating impulses**

- Parameter: `quality_mode`: selects the internal rate policy.
- Signal: Full-scale alternating impulses.
- Observation goal: stress headroom, limiter response, mode transitions, and sign symmetry.
- Required transition behavior: apply through a safe reprepare boundary.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the full-scale alternating impulses first with Quality at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to stress headroom, limiter response, mode transitions, and sign symmetry. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Quality with 80 Hz sine**

- Parameter: `quality_mode`: selects the internal rate policy.
- Signal: 80 Hz sine.
- Observation goal: expose low-frequency decay, modulation, modal buildup, and channel phase differences.
- Required transition behavior: apply through a safe reprepare boundary.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 80 hz sine first with Quality at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose low-frequency decay, modulation, modal buildup, and channel phase differences. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Quality with 8 kHz sine**

- Parameter: `quality_mode`: selects the internal rate policy.
- Signal: 8 kHz sine.
- Observation goal: expose damping, interpolation, aliasing, and high-frequency stability.
- Required transition behavior: apply through a safe reprepare boundary.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the 8 khz sine first with Quality at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to expose damping, interpolation, aliasing, and high-frequency stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Quality with Pink noise burst**

- Parameter: `quality_mode`: selects the internal rate policy.
- Signal: Pink noise burst.
- Observation goal: show broadband spectral decay, gain behavior, and early/late balance.
- Required transition behavior: apply through a safe reprepare boundary.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the pink noise burst first with Quality at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to show broadband spectral decay, gain behavior, and early/late balance. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Quality with Dry speech phrase**

- Parameter: `quality_mode`: selects the internal rate policy.
- Signal: Dry speech phrase.
- Observation goal: test intelligibility, sibilance, plosives, pre-delay, and ducking behavior.
- Required transition behavior: apply through a safe reprepare boundary.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the dry speech phrase first with Quality at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test intelligibility, sibilance, plosives, pre-delay, and ducking behavior. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Quality with Transient drum loop**

- Parameter: `quality_mode`: selects the internal rate policy.
- Signal: Transient drum loop.
- Observation goal: test attack preservation, density buildup, tempo interaction, and peak safety.
- Required transition behavior: apply through a safe reprepare boundary.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the transient drum loop first with Quality at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test attack preservation, density buildup, tempo interaction, and peak safety. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Quality with Sustained harmonic pad**

- Parameter: `quality_mode`: selects the internal rate policy.
- Signal: Sustained harmonic pad.
- Observation goal: test modulation smoothness, correlation, freeze energy, and long-term stability.
- Required transition behavior: apply through a safe reprepare boundary.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sustained harmonic pad first with Quality at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test modulation smoothness, correlation, freeze energy, and long-term stability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

**Signal-test card: Quality with Sixty-second tail capture**

- Parameter: `quality_mode`: selects the internal rate policy.
- Signal: Sixty-second tail capture.
- Observation goal: test memory stability, decay completion, noise floor, and repeatability.
- Required transition behavior: apply through a safe reprepare boundary.

Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the sixty-second tail capture first with Quality at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.

Automate one slow transition and one abrupt host change while the signal is active. The test is intended to test memory stability, decay completion, noise floor, and repeatability. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.

Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.


\newpage

## 31. Parameter Regression-Triage Cards

### 31.1 Pre-Delay

**Triage card: Pre-Delay: Audio safety**

- Parameter: `pre_delay_ms`: separates the direct event from the room onset.
- Failure class: Audio safety.
- Scope: clicks, instability, non-finite output, runaway gain, channel corruption, or unreported latency.
- Expected transition: use a ramp or delay-line crossfade.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For audio safety, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `pre_delay_ms` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Pre-Delay: State and automation**

- Parameter: `pre_delay_ms`: separates the direct event from the room onset.
- Failure class: State and automation.
- Scope: wrong recall, parameter-ID drift, host-write mismatch, migration loss, or format disagreement.
- Expected transition: use a ramp or delay-line crossfade.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For state and automation, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `pre_delay_ms` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Pre-Delay: User-facing behavior**

- Parameter: `pre_delay_ms`: separates the direct event from the room onset.
- Failure class: User-facing behavior.
- Scope: misleading value text, stale UI, unclear status, inaccessible control, or undocumented transition.
- Expected transition: use a ramp or delay-line crossfade.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For user-facing behavior, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `pre_delay_ms` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

### 31.2 Room Size

**Triage card: Room Size: Audio safety**

- Parameter: `room_size`: changes perceived scale and reflection spacing.
- Failure class: Audio safety.
- Scope: clicks, instability, non-finite output, runaway gain, channel corruption, or unreported latency.
- Expected transition: stage structural changes outside the callback.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For audio safety, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `room_size` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Room Size: State and automation**

- Parameter: `room_size`: changes perceived scale and reflection spacing.
- Failure class: State and automation.
- Scope: wrong recall, parameter-ID drift, host-write mismatch, migration loss, or format disagreement.
- Expected transition: stage structural changes outside the callback.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For state and automation, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `room_size` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Room Size: User-facing behavior**

- Parameter: `room_size`: changes perceived scale and reflection spacing.
- Failure class: User-facing behavior.
- Scope: misleading value text, stale UI, unclear status, inaccessible control, or undocumented transition.
- Expected transition: stage structural changes outside the callback.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For user-facing behavior, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `room_size` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

### 31.3 RT60 Coarse

**Triage card: RT60 Coarse: Audio safety**

- Parameter: `rt60_coarse`: moves through the full logarithmic decay range.
- Failure class: Audio safety.
- Scope: clicks, instability, non-finite output, runaway gain, channel corruption, or unreported latency.
- Expected transition: display the effective seconds value.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For audio safety, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `rt60_coarse` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: RT60 Coarse: State and automation**

- Parameter: `rt60_coarse`: moves through the full logarithmic decay range.
- Failure class: State and automation.
- Scope: wrong recall, parameter-ID drift, host-write mismatch, migration loss, or format disagreement.
- Expected transition: display the effective seconds value.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For state and automation, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `rt60_coarse` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: RT60 Coarse: User-facing behavior**

- Parameter: `rt60_coarse`: moves through the full logarithmic decay range.
- Failure class: User-facing behavior.
- Scope: misleading value text, stale UI, unclear status, inaccessible control, or undocumented transition.
- Expected transition: display the effective seconds value.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For user-facing behavior, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `rt60_coarse` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

### 31.4 RT60 Fine

**Triage card: RT60 Fine: Audio safety**

- Parameter: `rt60_fine`: trims decay proportionally around the coarse value.
- Failure class: Audio safety.
- Scope: clicks, instability, non-finite output, runaway gain, channel corruption, or unreported latency.
- Expected transition: keep zero as the exact neutral point.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For audio safety, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `rt60_fine` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: RT60 Fine: State and automation**

- Parameter: `rt60_fine`: trims decay proportionally around the coarse value.
- Failure class: State and automation.
- Scope: wrong recall, parameter-ID drift, host-write mismatch, migration loss, or format disagreement.
- Expected transition: keep zero as the exact neutral point.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For state and automation, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `rt60_fine` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: RT60 Fine: User-facing behavior**

- Parameter: `rt60_fine`: trims decay proportionally around the coarse value.
- Failure class: User-facing behavior.
- Scope: misleading value text, stale UI, unclear status, inaccessible control, or undocumented transition.
- Expected transition: keep zero as the exact neutral point.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For user-facing behavior, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `rt60_fine` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

### 31.5 Damping

**Triage card: Damping: Audio safety**

- Parameter: `damping`: changes high-frequency persistence.
- Failure class: Audio safety.
- Scope: clicks, instability, non-finite output, runaway gain, channel corruption, or unreported latency.
- Expected transition: interpolate stable filter coefficients.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For audio safety, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `damping` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Damping: State and automation**

- Parameter: `damping`: changes high-frequency persistence.
- Failure class: State and automation.
- Scope: wrong recall, parameter-ID drift, host-write mismatch, migration loss, or format disagreement.
- Expected transition: interpolate stable filter coefficients.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For state and automation, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `damping` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Damping: User-facing behavior**

- Parameter: `damping`: changes high-frequency persistence.
- Failure class: User-facing behavior.
- Scope: misleading value text, stale UI, unclear status, inaccessible control, or undocumented transition.
- Expected transition: interpolate stable filter coefficients.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For user-facing behavior, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `damping` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

### 31.6 Width

**Triage card: Width: Audio safety**

- Parameter: `width`: changes lateral energy and correlation.
- Failure class: Audio safety.
- Scope: clicks, instability, non-finite output, runaway gain, channel corruption, or unreported latency.
- Expected transition: monitor mono compatibility during movement.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For audio safety, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `width` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Width: State and automation**

- Parameter: `width`: changes lateral energy and correlation.
- Failure class: State and automation.
- Scope: wrong recall, parameter-ID drift, host-write mismatch, migration loss, or format disagreement.
- Expected transition: monitor mono compatibility during movement.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For state and automation, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `width` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Width: User-facing behavior**

- Parameter: `width`: changes lateral energy and correlation.
- Failure class: User-facing behavior.
- Scope: misleading value text, stale UI, unclear status, inaccessible control, or undocumented transition.
- Expected transition: monitor mono compatibility during movement.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For user-facing behavior, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `width` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

### 31.7 Diffusion

**Triage card: Diffusion: Audio safety**

- Parameter: `diffusion`: changes echo-density buildup.
- Failure class: Audio safety.
- Scope: clicks, instability, non-finite output, runaway gain, channel corruption, or unreported latency.
- Expected transition: crossfade when topology must change.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For audio safety, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `diffusion` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Diffusion: State and automation**

- Parameter: `diffusion`: changes echo-density buildup.
- Failure class: State and automation.
- Scope: wrong recall, parameter-ID drift, host-write mismatch, migration loss, or format disagreement.
- Expected transition: crossfade when topology must change.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For state and automation, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `diffusion` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Diffusion: User-facing behavior**

- Parameter: `diffusion`: changes echo-density buildup.
- Failure class: User-facing behavior.
- Scope: misleading value text, stale UI, unclear status, inaccessible control, or undocumented transition.
- Expected transition: crossfade when topology must change.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For user-facing behavior, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `diffusion` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

### 31.8 Wet

**Triage card: Wet: Audio safety**

- Parameter: `wet`: sets processed contribution.
- Failure class: Audio safety.
- Scope: clicks, instability, non-finite output, runaway gain, channel corruption, or unreported latency.
- Expected transition: choose and document the mix law.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For audio safety, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `wet` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Wet: State and automation**

- Parameter: `wet`: sets processed contribution.
- Failure class: State and automation.
- Scope: wrong recall, parameter-ID drift, host-write mismatch, migration loss, or format disagreement.
- Expected transition: choose and document the mix law.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For state and automation, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `wet` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Wet: User-facing behavior**

- Parameter: `wet`: sets processed contribution.
- Failure class: User-facing behavior.
- Scope: misleading value text, stale UI, unclear status, inaccessible control, or undocumented transition.
- Expected transition: choose and document the mix law.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For user-facing behavior, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `wet` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

### 31.9 Dry

**Triage card: Dry: Audio safety**

- Parameter: `dry`: sets direct contribution.
- Failure class: Audio safety.
- Scope: clicks, instability, non-finite output, runaway gain, channel corruption, or unreported latency.
- Expected transition: preserve bypass and gain staging.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For audio safety, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `dry` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Dry: State and automation**

- Parameter: `dry`: sets direct contribution.
- Failure class: State and automation.
- Scope: wrong recall, parameter-ID drift, host-write mismatch, migration loss, or format disagreement.
- Expected transition: preserve bypass and gain staging.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For state and automation, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `dry` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Dry: User-facing behavior**

- Parameter: `dry`: sets direct contribution.
- Failure class: User-facing behavior.
- Scope: misleading value text, stale UI, unclear status, inaccessible control, or undocumented transition.
- Expected transition: preserve bypass and gain staging.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For user-facing behavior, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `dry` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

### 31.10 Freeze

**Triage card: Freeze: Audio safety**

- Parameter: `freeze`: changes network energy behavior.
- Failure class: Audio safety.
- Scope: clicks, instability, non-finite output, runaway gain, channel corruption, or unreported latency.
- Expected transition: use a debounced, smoothed mode transition.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For audio safety, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `freeze` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Freeze: State and automation**

- Parameter: `freeze`: changes network energy behavior.
- Failure class: State and automation.
- Scope: wrong recall, parameter-ID drift, host-write mismatch, migration loss, or format disagreement.
- Expected transition: use a debounced, smoothed mode transition.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For state and automation, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `freeze` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Freeze: User-facing behavior**

- Parameter: `freeze`: changes network energy behavior.
- Failure class: User-facing behavior.
- Scope: misleading value text, stale UI, unclear status, inaccessible control, or undocumented transition.
- Expected transition: use a debounced, smoothed mode transition.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For user-facing behavior, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `freeze` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

### 31.11 Reverse

**Triage card: Reverse: Audio safety**

- Parameter: `reverse`: changes the envelope and buffering model.
- Failure class: Audio safety.
- Scope: clicks, instability, non-finite output, runaway gain, channel corruption, or unreported latency.
- Expected transition: report added latency before activation.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For audio safety, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `reverse` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Reverse: State and automation**

- Parameter: `reverse`: changes the envelope and buffering model.
- Failure class: State and automation.
- Scope: wrong recall, parameter-ID drift, host-write mismatch, migration loss, or format disagreement.
- Expected transition: report added latency before activation.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For state and automation, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `reverse` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Reverse: User-facing behavior**

- Parameter: `reverse`: changes the envelope and buffering model.
- Failure class: User-facing behavior.
- Scope: misleading value text, stale UI, unclear status, inaccessible control, or undocumented transition.
- Expected transition: report added latency before activation.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For user-facing behavior, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `reverse` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

### 31.12 Quality

**Triage card: Quality: Audio safety**

- Parameter: `quality_mode`: selects the internal rate policy.
- Failure class: Audio safety.
- Scope: clicks, instability, non-finite output, runaway gain, channel corruption, or unreported latency.
- Expected transition: apply through a safe reprepare boundary.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For audio safety, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `quality_mode` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Quality: State and automation**

- Parameter: `quality_mode`: selects the internal rate policy.
- Failure class: State and automation.
- Scope: wrong recall, parameter-ID drift, host-write mismatch, migration loss, or format disagreement.
- Expected transition: apply through a safe reprepare boundary.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For state and automation, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `quality_mode` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

**Triage card: Quality: User-facing behavior**

- Parameter: `quality_mode`: selects the internal rate policy.
- Failure class: User-facing behavior.
- Scope: misleading value text, stale UI, unclear status, inaccessible control, or undocumented transition.
- Expected transition: apply through a safe reprepare boundary.

Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.

Classify severity by outcome rather than by code size. For user-facing behavior, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `quality_mode` diverges from its contract.

Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.


\newpage

## 32. Closing Checklist

Before calling a VERBX plug-in build production-ready, confirm:

- the JUCE-enabled targets compile for every claimed architecture and format;
- parameter IDs and state migrations are frozen and tested;
- the realtime DSP replaces pass-through and has deterministic golden tests;
- callback code is allocation-free, lock-free, bounded, and NaN-safe;
- latency is measured and reported for every processing mode;
- sample-rate, block-size, layout, bypass, and transport changes are tested;
- Freeze and Reverse have explicit energy, transition, and latency semantics;
- geometry and IR assets are prepared outside the callback and recalled by hash;
- the production editor implements the visual target accessibly and without affecting sound;
- every host compatibility claim names a dated tested environment; and
- installers, signatures, notarization, scanning, crash recovery, and support bundles are complete.

The current stateful oversampled reverb makes these obligations visible and testable. The next major milestone is bounded-lookahead reverse processing with exact host latency notification, followed by multichannel layouts and a dated compatibility matrix.
