#!/usr/bin/env python3
"""Generate the long-form AUv3/VST3 plug-in handbook source."""

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "docs" / "PLUGIN_GUIDE.md"


INTRODUCTION = r"""# VERBX AUv3/VST3 Plug-in Handbook

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
"""


SOURCES = [
    ("Lead vocal", "preserve consonants and front-of-mix intelligibility", "18 to 45 ms", "watch sibilance and center stability"),
    ("Spoken word", "add believable room cues without masking language", "8 to 28 ms", "check breaths, plosives, and noise-floor lift"),
    ("Drum kit", "build size while preserving transient geometry", "4 to 24 ms", "check kick definition and snare tail density"),
    ("Piano", "support sustain without blurring note attacks", "12 to 40 ms", "listen for low-mid modal buildup"),
    ("Acoustic guitar", "add depth without combing the direct image", "10 to 32 ms", "check pick articulation and mono fold-down"),
    ("Electric guitar", "place the cabinet in a designed environment", "6 to 30 ms", "watch upper-mid glare in the return"),
    ("Strings", "extend bow sustain and ensemble width", "16 to 55 ms", "check section localization and high decay"),
    ("Synth pad", "turn sustained harmony into an evolving field", "0 to 40 ms", "watch feedback energy and stereo correlation"),
    ("Percussion", "create rhythmic depth around short impulses", "2 to 22 ms", "check early reflections against tempo"),
    ("Field recording", "recontextualize a scene without losing its anchors", "0 to 60 ms", "compare spectral floor and spatial plausibility"),
]

SPACES = [
    ("Tight room", 0.28, 0.20, 0.52, 0.28, "dense early cues and a controlled short tail"),
    ("Studio chamber", 0.42, 0.34, 0.64, 0.38, "a smooth useful chamber with moderate width"),
    ("Scoring stage", 0.58, 0.46, 0.72, 0.44, "clear source distance and a broad late field"),
    ("Concert hall", 0.68, 0.52, 0.78, 0.50, "a long integrated decay with stable imaging"),
    ("Stone cathedral", 0.78, 0.38, 0.84, 0.58, "slow spectral decay and monumental scale"),
    ("Plate-like field", 0.48, 0.62, 0.88, 0.46, "fast diffusion with less geometric localization"),
    ("Reverse chamber", 0.50, 0.45, 0.70, 0.52, "a bounded reverse envelope with explicit latency"),
    ("Frozen architecture", 0.72, 0.44, 0.82, 0.66, "a sustained field entered and exited safely"),
]

PARAMETERS = [
    ("Pre-Delay", "pre_delay_ms", "separates the direct event from the room onset", "use a ramp or delay-line crossfade"),
    ("Room Size", "room_size", "changes perceived scale and reflection spacing", "stage structural changes outside the callback"),
    ("RT60 Coarse", "rt60_coarse", "moves through the full logarithmic decay range", "display the effective seconds value"),
    ("RT60 Fine", "rt60_fine", "trims decay proportionally around the coarse value", "keep zero as the exact neutral point"),
    ("Damping", "damping", "changes high-frequency persistence", "interpolate stable filter coefficients"),
    ("Width", "width", "changes lateral energy and correlation", "monitor mono compatibility during movement"),
    ("Diffusion", "diffusion", "changes echo-density buildup", "crossfade when topology must change"),
    ("Wet", "wet", "sets processed contribution", "choose and document the mix law"),
    ("Dry", "dry", "sets direct contribution", "preserve bypass and gain staging"),
    ("Freeze", "freeze", "changes network energy behavior", "use a debounced, smoothed mode transition"),
    ("Reverse", "reverse", "changes the envelope and buffering model", "report added latency before activation"),
    ("Quality", "quality_mode", "selects the internal rate policy", "apply through a safe reprepare boundary"),
]

SHAPES = [
    ("Slow rise", "move from the lower setting to the upper setting over eight or more bars", "reveals zipper noise and coefficient discontinuities"),
    ("Slow fall", "return gradually toward the dry or compact state", "tests whether stored energy decays naturally"),
    ("Tempo pulse", "alternate two musically useful values on a bar or phrase boundary", "tests repeatability and transition timing"),
    ("Scene switch", "change once at an arrangement boundary and hold", "tests state recall and discrete transition behavior"),
]

SAMPLE_RATES = [44100, 48000, 96000, 192000]
QUALITY_MODES = [
    ("Host", "no intentional internal rate multiplication"),
    ("2x", "twice the host rate"),
    ("4x", "four times the host rate"),
    ("Target 192 kHz", "the smallest integer factor reaching at least 192 kHz"),
]
BLOCK_SIZES = [64, 512]

VALIDATION_AREAS = [
    ("Scan and instantiate", "confirm the format scans, loads, and creates a stable processor/editor pair"),
    ("Parameter automation", "write, read, trim, suspend, and replay every exposed parameter"),
    ("State recall", "save, close, reopen, and compare all parameters and asset identities"),
    ("Latency compensation", "measure impulse alignment and compare it with the reported frame count"),
    ("Bus negotiation", "exercise supported mono/stereo layouts and reject unsupported layouts clearly"),
    ("Transport changes", "start, stop, loop, seek, and change tempo without corrupting the tail"),
    ("Sample-rate changes", "reprepare at each supported host rate without stale buffers or status"),
    ("Editor lifecycle", "open, resize, close, and reopen the editor while audio remains unchanged"),
]

HOST_CONTEXTS = [
    ("Standalone", "the JUCE standalone wrapper", "device setup and callback behavior without DAW compensation"),
    ("Desktop AU", "an Audio Unit host", "Apple scanning, state, buses, and latency notification"),
    ("AUv3", "an AUv3-capable sandboxed host", "sandbox lifecycle, resources, and compact-window behavior"),
    ("VST3", "a VST3 host", "component/controller state, scanning, and automation identity"),
]

TROUBLESHOOTING = [
    ("The plug-in does not appear after scanning", "the binary is in the wrong format location, failed validation, or was built for the wrong architecture", "inspect the host scan log, confirm architecture and format, then rescan a clean build"),
    ("The editor opens but audio is dry", "wet gain is down, routing state is stale, or the prepared DSP context failed", "confirm Wet/Dry values, live internal-rate status, and host logs before treating this as a scanner fault"),
    ("Automation recalls the wrong control", "a parameter ID or version changed after a session was saved", "compare manifest IDs and restore stable identifiers; never repair this by reordering blindly"),
    ("A preset opens with missing geometry", "the referenced DXF/profile asset moved or its hash changed", "locate the exact asset, verify its hash, or use the stored bounded fallback profile"),
    ("CPU rises sharply at 48 kHz", "Target 192 kHz implies a 4x internal-rate goal", "compare Host and 2x modes, increase block size, and record the quality tradeoff"),
    ("CPU rises at 192 kHz", "the host is already processing a very high sample rate", "use Host mode and reduce expensive topology before changing musical controls"),
    ("The host reports no latency", "the current causal reverb/oversampling graph is intentionally zero-lookahead or a later buffering graph did not notify the host", "measure with an impulse and compare the result with the status accessor"),
    ("Reverse feels late", "reverse processing requires a capture or lookahead window", "verify the declared reverse window and host delay compensation"),
    ("Freeze gets louder over time", "feedback energy is above a stable bound or input injection remains active", "disengage safely, lower the stored energy, and inspect freeze gain and limiter reduction"),
    ("Freeze clicks when toggled", "the mode changes coefficients or injection abruptly", "add a state transition ramp or dual-path crossfade and retest at full-scale impulses"),
    ("The tail disappears on bypass", "the host uses hard bypass and skips processing", "offer a tail-preserving effect bypass parameter and document the host bypass behavior"),
    ("The image collapses in mono", "width or decorrelation has produced excessive anti-correlation", "reduce width, check early/late balance, and validate the mono sum"),
    ("The output contains denormal CPU spikes", "very small feedback values are entering slow floating-point paths", "enable denormal protection and test long decays into digital silence"),
    ("The output contains NaN or Inf", "unstable feedback, invalid coefficients, or corrupted state reached the DSP", "mute safely, capture diagnostics, clamp inputs, and fix the originating invariant"),
    ("Changing room size causes a click", "delay topology changed without a transition", "prepare the new network off-thread and crossfade bounded states"),
    ("Changing quality interrupts playback", "the internal processing graph requires reprepare", "apply quality changes at a declared safe boundary and show pending status"),
    ("Meters freeze when the editor closes", "telemetry ownership is incorrectly coupled to the editor", "keep DSP telemetry independent and let the editor subscribe only while visible"),
    ("Closing the editor changes CPU", "visualization work or DSP ownership is attached to editor lifetime", "separate processor state from the editor and repeat the lifecycle test"),
    ("A DAW project reopens silently", "state restoration failed or a required asset was unavailable", "load a safe default audibly, show a blocking status, and preserve diagnostic metadata"),
    ("Wet and dry at unity clip", "the mix law sums correlated paths above full scale", "choose equal-power or gain-compensated behavior and expose output safety metering"),
    ("Pre-delay automation flanges", "a moving delay is being read without an intentional interpolation strategy", "crossfade read heads or constrain automation to safe transitions"),
    ("High damping sounds unstable", "filter coefficients approach an unsafe limit at the active internal rate", "bound the coefficient domain and test every quality mode"),
    ("The host rejects the channel layout", "input and output buses are mismatched or unsupported", "request matched mono/stereo for the foundation and log the rejected layout clearly"),
    ("The screenshot and editor differ", "the screenshot is the visual design target while the JUCE editor remains a scaffold", "use the maturity statement and track UI implementation separately from DSP readiness"),
]

PRESET_FAMILIES = [
    ("Rooms", "natural early cues and controlled short decay", 0.20, 0.40, 0.55),
    ("Chambers", "dense useful depth around voices and instruments", 0.38, 0.52, 0.68),
    ("Halls", "integrated long decay with stable source localization", 0.58, 0.64, 0.76),
    ("Plates", "fast diffusion and bright sustained density", 0.44, 0.72, 0.86),
    ("Architectures", "geometry-led spaces with explicit source/listener context", 0.66, 0.56, 0.78),
    ("Experimental", "reverse, freeze, and exaggerated spatial behavior", 0.76, 0.48, 0.82),
]

PRESET_VARIANTS = [
    ("Intimate", "keep pre-delay and width restrained; prioritize direct connection"),
    ("Open", "increase width and early/late separation while preserving center focus"),
    ("Dark", "increase damping and reduce high-frequency persistence"),
    ("Infinite", "prepare a safe Freeze transition and conservative output protection"),
]

MONITOR_CONTEXTS = [
    ("Nearfield monitors", "judge center focus, depth layers, and low-mid buildup at a calibrated position"),
    ("Headphones", "inspect modulation, tail texture, and left/right discontinuities without room masking"),
    ("Mono sum", "expose anti-correlation, combing, and source loss caused by excessive width"),
    ("Low-level playback", "test whether the space remains legible without relying on loudness"),
]

ASSET_TYPES = [
    ("DXF room shell", "geometry topology, units, transforms, and source/listener coordinates"),
    ("Early-reflection profile", "bounded tap times, gains, directions, and profile version"),
    ("Measured impulse response", "sample rate, channels, trim, normalization, provenance, and checksum"),
    ("Generated impulse response", "generator version, seed, parameters, output format, and checksum"),
    ("Material library", "stable material IDs, absorption bands, scattering values, and revision"),
    ("Preset bank", "parameter schema, author metadata, tags, asset references, and migration version"),
    ("Telemetry configuration", "meter rates, history lengths, visualization channels, and safety bounds"),
    ("HRTF or SOFA set", "convention, receiver/emitter indices, coordinate system, sample rate, and license"),
]

ASSET_STAGES = [
    ("Import", "read and normalize the external representation away from the audio callback"),
    ("Validate", "reject malformed, unbounded, unsupported, or ambiguous content with field-specific errors"),
    ("Prepare and cache", "produce an immutable realtime-ready representation with a deterministic key"),
    ("Recall", "resolve the exact asset by identity and degrade safely when it is unavailable"),
]

RELEASE_AREAS = [
    ("Scanning", "clean install, discovery, validation logs, duplicate IDs, and rescan behavior"),
    ("State", "project recall, preset migration, asset identity, defaults, and corrupted-state recovery"),
    ("Audio", "silence, impulses, full-scale signals, long tails, NaN/Inf containment, and channel integrity"),
    ("Automation", "all write/read modes, undo, copy, parameter identity, and editor synchronization"),
    ("Latency", "reported frames, impulse measurement, mode changes, compensation, and transport boundaries"),
    ("Performance", "CPU, memory, denormals, editor cost, quality modes, and long-session stability"),
    ("Editor", "resize, scale, accessibility, keyboard focus, reopen, telemetry, and headless processing"),
    ("Distribution", "bundle contents, architecture slices, signing, notarization, installer, and uninstall"),
    ("Diagnostics", "status messages, support bundle, crash context, asset hashes, and privacy review"),
    ("Documentation", "build status, supported hosts, limitations, examples, screenshots, and release notes"),
]

RELEASE_TARGETS = [
    ("macOS AU", "desktop Audio Unit hosts and Apple validation tooling"),
    ("macOS AUv3", "sandboxed extension lifecycle and AUv3-capable hosts"),
    ("VST3", "VST3 scanning, component/controller state, and supported desktop hosts"),
]

BUS_LAYOUTS = [
    ("Mono", "one matched input and output channel with no hidden stereo assumptions"),
    ("Stereo", "matched left/right buses, stable center, width, correlation, and mono fold-down"),
    ("5.1", "explicit L/R/C/LFE/Ls/Rs routing and a declared LFE policy"),
    ("7.1", "side/rear separation, normalization, and host channel-order verification"),
    ("7.1.4", "bed plus height routing, elevation behavior, and immersive meter coverage"),
    ("First-order ambisonics", "ACN/SN3D ordering, rotation behavior, and decoder-independent energy"),
]

BUS_MODES = [
    ("Algorithmic", "the core FDN/diffuser path with matched input/output layout"),
    ("Reverse", "the bounded reverse window, channel alignment, and reported latency"),
    ("Freeze", "sustained energy, channel stability, exit behavior, and safety limiting"),
    ("Geometry or IR", "prepared spatial assets, channel metadata, and deterministic fallback"),
]

TEST_SIGNALS = [
    ("Digital silence", "detect denormals, stale buffers, uninitialized state, and noise-floor growth"),
    ("Single-sample impulse", "reveal latency, early reflections, channel routing, and deterministic tail shape"),
    ("Full-scale alternating impulses", "stress headroom, limiter response, mode transitions, and sign symmetry"),
    ("80 Hz sine", "expose low-frequency decay, modulation, modal buildup, and channel phase differences"),
    ("8 kHz sine", "expose damping, interpolation, aliasing, and high-frequency stability"),
    ("Pink noise burst", "show broadband spectral decay, gain behavior, and early/late balance"),
    ("Dry speech phrase", "test intelligibility, sibilance, plosives, pre-delay, and ducking behavior"),
    ("Transient drum loop", "test attack preservation, density buildup, tempo interaction, and peak safety"),
    ("Sustained harmonic pad", "test modulation smoothness, correlation, freeze energy, and long-term stability"),
    ("Sixty-second tail capture", "test memory stability, decay completion, noise floor, and repeatability"),
]

TRIAGE_CLASSES = [
    ("Audio safety", "clicks, instability, non-finite output, runaway gain, channel corruption, or unreported latency"),
    ("State and automation", "wrong recall, parameter-ID drift, host-write mismatch, migration loss, or format disagreement"),
    ("User-facing behavior", "misleading value text, stale UI, unclear status, inaccessible control, or undocumented transition"),
]


def page_break(lines: list[str]) -> None:
    lines.extend(["", r"\newpage", ""])


def card(lines: list[str], title: str, metadata: list[str], paragraphs: list[str]) -> None:
    lines.extend([f"**{title}**", ""])
    lines.extend(f"- {item}" for item in metadata)
    lines.append("")
    for paragraph in paragraphs:
        lines.extend([paragraph, ""])
    page_break(lines)


def production_cards(lines: list[str]) -> None:
    lines.extend(["## 19. Production Starting-Point Cards", ""])
    for source_index, source in enumerate(SOURCES):
        source_name, purpose, predelay, watch = source
        lines.extend([f"### 19.{source_index + 1} {source_name}", ""])
        for space in SPACES:
            space_name, size, damping, diffusion, wet, character = space
            reverse = space_name == "Reverse chamber"
            freeze = space_name == "Frozen architecture"
            title = f"Production card: {source_name} in {space_name}"
            metadata = [
                f"Intent: {purpose}.",
                f"Space character: {character}.",
                f"Starting macros: room size {size:.2f}, damping {damping:.2f}, diffusion {diffusion:.2f}, wet {wet:.2f}.",
                f"Pre-delay working range: {predelay}.",
                f"Modes: Reverse {'on' if reverse else 'off'}; Freeze {'prepared/on' if freeze else 'off'}.",
            ]
            paragraphs = [
                f"Begin with the dry {source_name.lower()} at a calibrated monitoring level. Establish the direct image first, then raise the wet return until {character} is audible without replacing the source. The initial macro values are coordinates, not a preset guarantee. Adjust RT60 coarse by ear, use RT60 fine for the last proportional correction, and read the effective seconds display rather than inferring time from knob angle.",
                f"For this source, {watch}. Compare the processed signal in stereo and mono, then bypass at matched loudness. If the room becomes impressive only when it is louder, correct gain before evaluating tone. Check the first reflection region separately from the late field: pre-delay and early geometry govern separation, while damping, diffusion, and RT60 govern how the tail occupies the arrangement.",
                "Record host rate, quality mode, block size, layout, effective RT60, reported latency, and peak reduction. If an imported geometry profile is involved, record its content hash. Save a user preset only after reopening the session and confirming that parameters, mode buttons, and assets recall identically.",
            ]
            card(lines, title, metadata, paragraphs)


def automation_cards(lines: list[str]) -> None:
    lines.extend(["## 20. Automation Study Cards", ""])
    for parameter_index, parameter in enumerate(PARAMETERS):
        label, key, meaning, transition = parameter
        lines.extend([f"### 20.{parameter_index + 1} {label}", ""])
        for shape in SHAPES:
            shape_name, motion, risk = shape
            title = f"Automation card: {label}: {shape_name}"
            metadata = [
                f"Host parameter: `{key}`.",
                f"Motion: {motion}.",
                f"Primary observation: {meaning}.",
                f"Transition requirement: {transition}.",
            ]
            paragraphs = [
                f"Write the {shape_name.lower()} move in the host, close the editor, and play it twice. The second pass must match the first because automation belongs to the processor, not the visible UI. Reopen the editor and verify that the visual control follows without feeding values back into the host. Repeat after saving and reopening the project.",
                f"Listen at the beginning, during the transition, and after the value settles. This movement {risk}. Use an impulse, sustained tone, and representative music because each exposes a different failure mode. Inspect output for clicks, NaN/Inf, unexpected gain, channel asymmetry, and a latency change that was not reported.",
                "Document whether the control is continuously smooth, crossfaded, or intentionally discrete. If a safe implementation requires reprepare, the host-visible value may update immediately while audio waits for a declared boundary; the status strip must make that pending state clear. Automation compatibility is not complete until touch, latch, read, trim, undo, and session recall have all been exercised.",
            ]
            card(lines, title, metadata, paragraphs)


def quality_cards(lines: list[str]) -> None:
    lines.extend(["## 21. Quality And Latency Cards", ""])
    scenarios = list(product(SAMPLE_RATES, QUALITY_MODES, BLOCK_SIZES))
    for index, (rate, quality, block) in enumerate(scenarios, start=1):
        quality_name, quality_meaning = quality
        target_rate = {
            "Host": rate,
            "2x": rate * 2,
            "4x": rate * 4,
            "Target 192 kHz": rate * max(1, (192000 + rate - 1) // rate),
        }[quality_name]
        block_ms = 1000.0 * block / rate
        title = f"Quality card {index}: {rate} Hz, {quality_name}, {block} frames"
        metadata = [
            f"Host rate: {rate} Hz.",
            f"Quality policy: {quality_name}: {quality_meaning}.",
            f"Expected internal-rate contract: {target_rate} Hz.",
            f"Host block duration: {block_ms:.3f} ms before device and plug-in latency.",
        ]
        paragraphs = [
            "Prepare the processor at this exact rate and maximum block size. Confirm that the internal-rate accessor matches the policy and that no multiplication overflow or invalid mode is accepted. Process zero, one, nominal, and maximum-length blocks. Then vary the actual callback length below the declared maximum to model hosts that use nonuniform final blocks.",
            "Measure CPU with the editor closed and open. Separate the active resampling cost from FDN topology, modulation, telemetry, and drawing. A high quality mode may be intentionally expensive, but it must not silently fall back. Confirm the live internal-rate and factor status after each change. If the system cannot sustain the target, show a warning and let the user choose a lower mode.",
            "Measure algorithmic plug-in latency with an impulse and compare it with the reported frame count. Do not add device input/output latency to the value reported to the DAW. For live monitoring, separately estimate end-to-end latency from device buffers, host safety buffers, block duration, and plug-in processing. Save all measurements with architecture, operating system, host version, and build commit.",
        ]
        card(lines, title, metadata, paragraphs)


def validation_cards(lines: list[str]) -> None:
    lines.extend(["## 22. Host Validation Cards", ""])
    for area_index, area in enumerate(VALIDATION_AREAS):
        area_name, goal = area
        lines.extend([f"### 22.{area_index + 1} {area_name}", ""])
        for context in HOST_CONTEXTS:
            context_name, surface, emphasis = context
            title = f"Validation card: {context_name}: {area_name}"
            metadata = [
                f"Surface: {surface}.",
                f"Goal: {goal}.",
                f"Context emphasis: {emphasis}.",
                "Status: protocol only until a dated result is recorded.",
            ]
            paragraphs = [
                f"Start from a clean build and a new host project. Record the exact binary path, architecture, signature state, plug-in format, host version, and VERBX commit. Perform the {area_name.lower()} procedure first at 48 kHz with a moderate block, then repeat at the lowest-latency and highest-quality settings intended for support.",
                "Use deterministic test audio and save the host project before changing anything else. Reopen it in a fresh host process. Compare parameter values, effective RT60, quality mode, Freeze/Reverse state, reported latency, bus layout, and asset identity. Capture the host scan or validation log when behavior differs from standalone.",
                "Mark the result pass, fail, or blocked. A pass names the tested environment and does not imply universal compatibility. A fail includes the smallest reproduction and whether the processor degraded to safe pass-through, bypass, silence, or an error. A blocked result states the missing tool, entitlement, SDK, device, or host access without converting absence of evidence into a compatibility claim.",
            ]
            card(lines, title, metadata, paragraphs)


def troubleshooting_cards(lines: list[str]) -> None:
    lines.extend(["## 23. Troubleshooting Cards", ""])
    for index, issue in enumerate(TROUBLESHOOTING, start=1):
        symptom, likely, recovery = issue
        title = f"Troubleshooting card {index}: {symptom}"
        metadata = [
            f"Symptom: {symptom}.",
            f"Likely causes: {likely}.",
            f"First recovery: {recovery}.",
        ]
        paragraphs = [
            "Preserve evidence before changing the system. Record the build commit, plug-in format, host/version, sample rate, block size, bus layout, quality mode, effective RT60, asset hashes, and the exact action that triggered the problem. Save scanner output, host logs, and a minimal project when available; do not include private project audio unless it is necessary and authorized.",
            "Reduce the case methodically. Disable imported assets, use matched mono or stereo, return automation to static values, select Host quality, and test a short RT60 with conservative wet/dry gain. Change one condition at a time. If pass-through succeeds but reverb fails, the shell and bus path are probably intact; if scanning fails, DSP parameter changes are unlikely to matter.",
            "After the root cause is fixed, add a regression below the DAW whenever possible. Parameter mapping, prepare validation, block bounds, state serialization, and deterministic DSP belong in automated tests. Keep the host reproduction as a format-specific smoke test. Update the visible status message so the next failure is diagnosable without a debugger.",
        ]
        card(lines, title, metadata, paragraphs)


def preset_cards(lines: list[str]) -> None:
    lines.extend(["## 24. Preset Design Cards", ""])
    for family_index, family in enumerate(PRESET_FAMILIES):
        family_name, identity, size, damping, diffusion = family
        lines.extend([f"### 24.{family_index + 1} {family_name}", ""])
        for variant in PRESET_VARIANTS:
            variant_name, direction = variant
            title = f"Preset card: {family_name} / {variant_name}"
            metadata = [
                f"Family identity: {identity}.",
                f"Variant direction: {direction}.",
                f"Macro anchors: room size {size:.2f}, damping {damping:.2f}, diffusion {diffusion:.2f}.",
                "Required metadata: schema, build, author, description, tags, quality policy, layout, and asset identity.",
            ]
            paragraphs = [
                "Design the preset at a calibrated level with at least speech, drums, harmonic music, and an impulse. Set coarse RT60 for the family scale, then use fine RT60 to place the decay precisely. Match output loudness before comparing variants. A useful preset should communicate its intent without depending on a hidden gain advantage.",
                "Decide whether quality mode belongs to the preset or remains a user/global preference. If it is stored, explain the CPU implication. Freeze and Reverse must be deliberately authored, never inherited accidentally from the previous preset. Geometry and IR assets need content hashes and a clear fallback behavior when unavailable.",
                "Save, reload, and compare the preset in standalone plus each supported plug-in format. Confirm that host automation remains mapped to the same parameter IDs after preset changes. Add search tags that describe source, scale, material, brightness, motion, and special modes. Do not publish a compatibility or physical-accuracy claim that the preset metadata cannot support.",
            ]
            card(lines, title, metadata, paragraphs)


def interaction_cards(lines: list[str]) -> None:
    lines.extend(["## 25. Parameter Interaction Cards", ""])
    for index, (left, right) in enumerate(combinations(PARAMETERS, 2), start=1):
        left_label, left_key, left_meaning, left_transition = left
        right_label, right_key, right_meaning, right_transition = right
        title = f"Interaction card {index}: {left_label} with {right_label}"
        metadata = [
            f"Parameters: `{left_key}` and `{right_key}`.",
            f"First role: {left_meaning}.",
            f"Second role: {right_meaning}.",
            f"Transition rules: {left_transition}; {right_transition}.",
        ]
        paragraphs = [
            f"Hold {right_label} at its default and sweep {left_label} through a conservative, musical range. Return to default, then reverse the roles. Finally test a diagonal move in which both values rise and an opposing move in which one rises while the other falls. This separates each control's independent contribution from the interaction that users will actually hear in automation and preset changes.",
            "Use an impulse to reveal timing and topology, sustained noise to reveal spectral balance, speech to reveal intelligibility, and music to reveal masking. Match output loudness between states. Watch correlation, peak level, limiter reduction, effective RT60, latency, and CPU. If the pair changes a structural property, verify that the adapter stages or crossfades the new state instead of mutating unbounded DSP storage in the callback.",
            "Save all four corners plus the center as host states and reopen them in a fresh process. Parameter identity must remain stable regardless of registration order. If one control is discrete, host automation should not imply a smooth intermediate sound that the DSP cannot provide. Document clamping and pending-reprepare behavior in both the parameter text and status strip.",
        ]
        card(lines, title, metadata, paragraphs)


def monitoring_cards(lines: list[str]) -> None:
    lines.extend(["## 26. Monitoring And Audition Cards", ""])
    for source_index, source in enumerate(SOURCES):
        source_name, purpose, predelay, watch = source
        lines.extend([f"### 26.{source_index + 1} {source_name}", ""])
        for context_name, context_goal in MONITOR_CONTEXTS:
            title = f"Audition card: {source_name} on {context_name}"
            metadata = [
                f"Source: {source_name}.",
                f"Monitoring context: {context_name}.",
                f"Judgment goal: {context_goal}.",
                f"Useful pre-delay range: {predelay}.",
            ]
            paragraphs = [
                f"Choose a short phrase that contains both sparse and dense moments. Calibrate the dry path, add the plug-in at unity host gain, and set a moderate room before comparing. The artistic objective is to {purpose}. Toggle matched-loudness bypass and keep the transport loop long enough to hear the tail complete rather than restarting it at every bar.",
                f"In this monitoring context, {context_goal}. For {source_name.lower()}, {watch}. Check the direct onset, early-reflection zone, modal buildup, late decay, and final noise floor as separate listening events. Repeat with the editor closed to ensure visual telemetry does not alter processing or CPU enough to change the result.",
                "Record monitor chain, playback level, room correction, host rate, block size, quality mode, bus layout, and effective RT60. A headphone finding does not automatically predict speaker translation, and a mono pass does not prove immersive routing. Promote a preset only after it survives at least two complementary monitoring contexts and a reopened-session check.",
            ]
            card(lines, title, metadata, paragraphs)


def asset_cards(lines: list[str]) -> None:
    lines.extend(["## 27. Asset Lifecycle Cards", ""])
    for asset_index, asset in enumerate(ASSET_TYPES):
        asset_name, asset_fields = asset
        lines.extend([f"### 27.{asset_index + 1} {asset_name}", ""])
        for stage_name, stage_goal in ASSET_STAGES:
            title = f"Asset card: {asset_name}: {stage_name}"
            metadata = [
                f"Asset: {asset_name}.",
                f"Lifecycle stage: {stage_name}.",
                f"Stage objective: {stage_goal}.",
                f"Identity fields: {asset_fields}.",
            ]
            paragraphs = [
                "Perform this stage on a worker or management thread, never from the realtime callback. Bound file size, element count, channel count, duration, and memory before allocating the prepared representation. Normalize paths and units without losing the original provenance. Error messages should identify the invalid field and preserve enough context for a support bundle.",
                "Compute a deterministic content hash after canonicalization and include a schema/version identifier. Cache keys must include every input that changes realtime output. Two assets with the same display name are not interchangeable. When preparation succeeds, publish an immutable handle through a bounded swap; retire the previous handle only after the audio thread can no longer reference it.",
                "Test missing, moved, corrupted, oversized, unsupported, and version-mismatched cases. Host state should remain loadable even when the external asset does not. The processor may use a compact embedded fallback or safe default, but it must show that substitution. Never conceal an asset failure behind an unrelated room preset or silently regenerate nondeterministic content.",
            ]
            card(lines, title, metadata, paragraphs)


def release_cards(lines: list[str]) -> None:
    lines.extend(["## 28. Release Readiness Cards", ""])
    for area_index, area in enumerate(RELEASE_AREAS):
        area_name, area_scope = area
        lines.extend([f"### 28.{area_index + 1} {area_name}", ""])
        for target_name, target_scope in RELEASE_TARGETS:
            title = f"Release card: {target_name}: {area_name}"
            metadata = [
                f"Target: {target_name}.",
                f"Target scope: {target_scope}.",
                f"Readiness area: {area_scope}.",
                "Evidence required: dated environment, build commit, result, and retained logs/artifacts.",
            ]
            paragraphs = [
                "Run this gate from a clean release candidate rather than a developer build directory. Confirm architecture, optimization, symbols policy, bundle identity, version, and parameter schema. Use a new host project plus a reopened representative project. Record both success and the exact unsupported configurations so release notes can be precise.",
                "A pass requires observable evidence: scanner output, deterministic render comparison, state round-trip, measured latency, performance trace, accessibility result, installer receipt, or documentation review as appropriate. A warning is not a pass unless it is explicitly accepted and documented. A failure must block the compatibility claim even if another format succeeds on the same machine.",
                "After fixing a failure, rerun the focused reproduction and the neighboring gates most likely to regress. Keep host-specific checks above a larger automated native test base so common DSP and state failures are found before format testing. Archive the final matrix with the release tag and publish only the subset of environments actually validated.",
            ]
            card(lines, title, metadata, paragraphs)


def bus_cards(lines: list[str]) -> None:
    lines.extend(["## 29. Spatial Bus Validation Cards", ""])
    for layout_index, layout in enumerate(BUS_LAYOUTS):
        layout_name, layout_contract = layout
        lines.extend([f"### 29.{layout_index + 1} {layout_name}", ""])
        for mode_name, mode_contract in BUS_MODES:
            title = f"Bus card: {layout_name}: {mode_name}"
            metadata = [
                f"Layout: {layout_name}: {layout_contract}.",
                f"Processing mode: {mode_name}: {mode_contract}.",
                "Foundation status: mono/stereo only; larger layouts are future validation protocols.",
                "Required probes: per-channel impulses, correlated program, decorrelated program, silence, and full-scale safety input.",
            ]
            paragraphs = [
                "Ask the host for the exact layout and verify the channel order exposed to the processor. Reject unsupported or mismatched buses before processing. Run a labeled impulse through each input separately and inspect every output, reported channel label, meter, and serialized layout field. Hidden stereo assumptions are especially dangerous when expanding to center, surround, height, or ambisonic channels.",
                "Measure gain and latency per channel. Confirm that wet/dry, bypass, Freeze, Reverse, limiter, and telemetry follow the declared routing policy. Fold down through an agreed reference path where relevant, but do not use a fold-down to conceal incorrect native output routing. LFE injection, height decorrelation, and ambisonic normalization require explicit policies rather than inherited defaults.",
                "Save and reopen the host project, then change to another supported layout and back. Reprepare must release or resize channel-dependent state outside the callback. If the requested layout cannot be restored, show a clear status and select a safe behavior. Do not claim a spatial format from a visually correct meter alone; channel-accurate audio and metadata are the authority.",
            ]
            card(lines, title, metadata, paragraphs)


def test_vector_cards(lines: list[str]) -> None:
    lines.extend(["## 30. Parameter Signal-Test Cards", ""])
    for parameter_index, parameter in enumerate(PARAMETERS):
        label, key, meaning, transition = parameter
        lines.extend([f"### 30.{parameter_index + 1} {label}", ""])
        for signal_name, signal_goal in TEST_SIGNALS:
            title = f"Signal-test card: {label} with {signal_name}"
            metadata = [
                f"Parameter: `{key}`: {meaning}.",
                f"Signal: {signal_name}.",
                f"Observation goal: {signal_goal}.",
                f"Required transition behavior: {transition}.",
            ]
            paragraphs = [
                f"Prepare matched mono and stereo contexts at 48 kHz, then repeat the focused failure case at 44.1, 96, and 192 kHz. Process the {signal_name.lower()} first with {label} at default, then at minimum, midpoint, and maximum or each discrete choice. Capture output, status, effective RT60, latency, peak, and channel count. The same input and state must produce the same output within the declared numerical tolerance.",
                f"Automate one slow transition and one abrupt host change while the signal is active. The test is intended to {signal_goal}. Listen and inspect samples around the transition. A click, non-finite value, unexplained latency jump, channel mismatch, or value-dependent memory allocation is a defect even when musical program material masks it.",
                "Run with the editor closed and open, then save and reopen the host project before repeating the endpoint cases. Compare standalone, AU/AUv3, and VST3 only after the native result is stable. Store small deterministic outputs as golden fixtures where practical; store metrics and hashes for long outputs. Add the smallest failing case to automated coverage and keep the host project as a format smoke test.",
            ]
            card(lines, title, metadata, paragraphs)


def triage_cards(lines: list[str]) -> None:
    lines.extend(["## 31. Parameter Regression-Triage Cards", ""])
    for parameter_index, parameter in enumerate(PARAMETERS):
        label, key, meaning, transition = parameter
        lines.extend([f"### 31.{parameter_index + 1} {label}", ""])
        for triage_name, triage_scope in TRIAGE_CLASSES:
            title = f"Triage card: {label}: {triage_name}"
            metadata = [
                f"Parameter: `{key}`: {meaning}.",
                f"Failure class: {triage_name}.",
                f"Scope: {triage_scope}.",
                f"Expected transition: {transition}.",
            ]
            paragraphs = [
                "Freeze the failing build and capture a minimal deterministic reproduction before changing code. Record format, host, architecture, sample rate, block size, layout, quality mode, previous and current parameter values, transport state, reported latency, and whether the editor was open. Preserve the smallest audio fixture, state blob, automation lane, or screenshot needed to demonstrate the regression.",
                f"Classify severity by outcome rather than by code size. For {triage_name.lower()}, determine whether the failure can damage audio, corrupt a project, break recall, defeat automation, or merely reduce clarity. Compare the C manifest, realtime context, JUCE parameter object, serialized state, and displayed value to find the first boundary where `{key}` diverges from its contract.",
                "Fix the lowest shared layer that owns the invariant, then add a native regression before repeating format-specific smoke tests. Verify minimum, default, maximum, out-of-range, automated, saved, and reopened cases. Update status text or documentation when the behavior is intentionally constrained. Close triage only when the original project is recovered or a migration path is documented and tested.",
            ]
            card(lines, title, metadata, paragraphs)


def build_markdown() -> str:
    lines = [INTRODUCTION.rstrip(), ""]
    production_cards(lines)
    automation_cards(lines)
    quality_cards(lines)
    validation_cards(lines)
    troubleshooting_cards(lines)
    preset_cards(lines)
    interaction_cards(lines)
    monitoring_cards(lines)
    asset_cards(lines)
    release_cards(lines)
    bus_cards(lines)
    test_vector_cards(lines)
    triage_cards(lines)
    lines.extend(
        [
            "## 32. Closing Checklist",
            "",
            "Before calling a VERBX plug-in build production-ready, confirm:",
            "",
            "- the JUCE-enabled targets compile for every claimed architecture and format;",
            "- parameter IDs and state migrations are frozen and tested;",
            "- the realtime DSP replaces pass-through and has deterministic golden tests;",
            "- callback code is allocation-free, lock-free, bounded, and NaN-safe;",
            "- latency is measured and reported for every processing mode;",
            "- sample-rate, block-size, layout, bypass, and transport changes are tested;",
            "- Freeze and Reverse have explicit energy, transition, and latency semantics;",
            "- geometry and IR assets are prepared outside the callback and recalled by hash;",
            "- the production editor implements the visual target accessibly and without affecting sound;",
            "- every host compatibility claim names a dated tested environment; and",
            "- installers, signatures, notarization, scanning, crash recovery, and support bundles are complete.",
            "",
            "The current stateful oversampled reverb makes these obligations visible and testable. The next major milestone is bounded-lookahead reverse processing with exact host latency notification, followed by multichannel layouts and a dated compatibility matrix.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_markdown(), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
