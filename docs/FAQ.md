# Frequently Asked Questions

This appendix answers the questions most likely to arise while installing, running,
measuring, and integrating verbx. Commands assume a shell in which `verbx` is on the
path. Run `verbx COMMAND --help` whenever the installed version is the final authority
for an option or default.

| If you need to... | Start with... |
|---|---|
| Verify the installation | `verbx doctor --strict` |
| Render a first example | `verbx render in.wav out.wav --preset warm-hall` |
| Inspect an audio file | `verbx analyze out.wav --lufs --edr --room --json-out analysis.json` |
| Find audio devices | `verbx realtime --list-devices` |
| List the preset library | `verbx presets` |
| Diagnose a failed render | `verbx render ... --failure-report-out failure.json` |

## Installation and First Launch

**Q1. How do I verify that verbx is installed correctly?**

Run `verbx version`, followed by `verbx doctor --strict`. For a stronger end-to-end
check, use `verbx doctor --render-smoke-test`; it creates and processes a small test
signal so the command path, audio dependencies, and render pipeline are exercised
together.

**Q2. What should I do if the shell says `verbx: command not found`?**

Open a new terminal after installation and confirm that the installation directory is
on `PATH`. If verbx was installed into a virtual environment, activate that environment
before running the command. `python -m verbx version` can help distinguish an importable
package from a missing console-script path.

**Q3. Should I install with Homebrew or from the repository?**

Use Homebrew for the simplest supported macOS command-line installation. Use a source
checkout when developing verbx, running the complete test suite, building plug-ins, or
changing documentation. Do not mix two active installations while diagnosing version
or plug-in discovery problems.

**Q4. What does `install.sh` install?**

The repository installer builds the available command-line and native targets and
places supported plug-in bundles in the standard per-user macOS locations. It does not
control a DAW's validation cache, so the host may still require a rescan or restart.
Read the installer summary carefully because unavailable SDKs or formats are reported
rather than silently fabricated.

**Q5. Why does a newly installed plug-in not appear immediately in my DAW?**

Quit and reopen the DAW, request a full plug-in rescan, and verify the bundle with the
host's plug-in manager. On macOS, also confirm that the architecture matches the host
and that quarantine, signing, or validation did not reject the bundle. The manufacturer
name is **Colby Leider**.

**Q6. Can installation diagnostics be saved for support or automation?**

Yes. Use `verbx doctor --json-out doctor.json`. Add `--strict` when a warning should
produce a failing exit status in continuous integration, and add
`--render-smoke-test --smoke-out-dir smoke` when a reproducible audio artifact is useful.

## Choosing and Shaping Reverb

**Q7. When should I choose algorithmic reverb instead of convolution?**

Choose algorithmic reverb for continuously variable decay, modulation, freeze, unusual
feedback structures, and low-memory realtime operation. Choose convolution when a
specific measured or designed impulse response is the sound. A hybrid workflow can use
convolution for early reflections and an algorithmic network for the late field.

**Q8. What does `--engine auto` do?**

Auto selects an engine from the supplied inputs and requested controls. An explicit IR
normally favors convolution; algorithm-only controls favor the algorithmic engine. Use
`--engine algo` or `--engine conv` when reproducibility requires an unambiguous choice.

**Q9. What does RT60 mean?**

RT60 is the estimated time, in seconds, for reverberant energy to decay by 60 dB under
the model's assumptions. It describes a decay slope, not necessarily the exact file
length. Noise floors, frequency dependence, modulation, gates, and freeze can make the
perceived tail differ from the nominal value.

**Q10. Why is RT60 controlled logarithmically?**

Useful decay times span several orders of magnitude, from tight rooms to minute-long
sound-design fields. A logarithmic control gives short decays enough precision while
still reaching extreme values. The plug-in combines coarse and fine controls so broad
changes and exact adjustments are both practical.

**Q11. What is the difference between pre-delay and early reflections?**

Pre-delay is the initial dry-to-reverberant gap. Early reflections are discrete arrivals
that follow and convey room size, source position, and surface geometry. Increasing
pre-delay can preserve articulation without changing the late decay, while changing the
early-reflection pattern changes spatial character.

**Q12. How do damping, filtering, and diffusion differ?**

Damping changes decay as a function of frequency. Input or output filters shape what
enters or leaves the reverberator. Diffusion changes echo density and the audibility of
individual repetitions. A dark but sparse tail and a bright but highly diffuse tail are
therefore both possible.

## Rendering and Output

**Q13. What is a good first render command?**

Start with:

```bash
verbx render in.wav out.wav --engine algo --rt60 2.4 --wet 0.35 --dry 0.8
```

Then change one parameter at a time. Add `--json-out out.report.json` when the settings
and processing summary should travel with the audio.

**Q14. Why can a render with an extremely long RT60 appear to hang?**

The renderer may be generating and writing a tail proportional to the requested decay.
An RT60 of 3,600 seconds is an hour-scale process, not an ordinary room render. Use
freeze for intentional indefinite sustain, or choose a bounded output duration and
watch the current processing status rather than treating elapsed time as a completion
estimate.

**Q15. What is the difference between a very long decay and freeze?**

A very long RT60 still decays. Freeze deliberately approaches lossless recirculation and
is intended for sustained textures. Freeze output must still be bounded during offline
rendering; otherwise there is no natural endpoint at which a complete file can be
written.

**Q16. Can verbx write 32-bit and 64-bit floating-point WAV files?**

Yes, where the selected backend supports the requested subtype. Float32 is the normal
interchange choice; float64 preserves additional numerical precision at the cost of
larger files and reduced compatibility with some software. File precision does not by
itself guarantee that every internal stage or host uses the same precision.

**Q17. How should I prevent clipping in wet renders?**

Leave headroom in both dry and wet paths, especially when long tails accumulate energy.
Use the limiter deliberately rather than as a substitute for gain staging, and inspect
peak, true-peak, and loudness measurements after rendering. A normalized impulse
response can still produce overload when dense material is convolved with it.

**Q18. Which machine-readable render outputs are available?**

Use `--json-out` for the processing report, `--analysis-out` for analysis results,
`--frames-out` for frame-level data, `--repro-bundle` for reproducibility material, and
`--failure-report-out` for structured failure context. `--dry-run` is useful before a
large render because it validates and reports without committing to the full process.

## Realtime Audio and Latency

**Q19. How do I list realtime input and output devices?**

Run `verbx realtime --list-devices`. Then pass the displayed device name or identifier
with `--input-device` and `--output-device`. If a device disappears, close applications
that may hold it exclusively and list devices again.

**Q20. What does end-to-end realtime latency include?**

It includes input conversion and driver buffering, verbx input buffering, algorithm or
partitioned-convolution delay, output buffering, driver scheduling, and output
conversion. The audio block size is only one component. Measure round-trip latency with
a physical loopback when sample-accurate compensation matters.

**Q21. How can I reduce realtime latency?**

Choose a smaller `--block-size`, use an algorithmic engine or a smaller convolution
partition, avoid unnecessary channel expansion, and select a low-latency device driver.
Reduce one setting at a time while checking for xruns. The smallest accepted value is
not necessarily the most reliable value under a real session load.

**Q22. Why do I hear clicks or dropouts at a small block size?**

The processor or driver is missing deadlines. Increase the block size, close competing
audio applications, simplify the routing, reduce expensive analysis displays, or use a
less demanding engine. Stable uninterrupted audio is more important than a nominally
lower buffer that repeatedly underruns.

**Q23. Can realtime mode dereverberate and add reverb in one pass?**

Yes. Select `--live-mode dereverb-reverb` and configure both stages. This is more
computationally demanding than either stage alone, so establish a stable block size with
each stage separately before combining them.

**Q24. How do I save a realtime session report?**

Add `--json-out realtime.json`. The report records the resolved devices, stream format,
buffer choices, mode, and available runtime counters so a session can be compared with
a later run or attached to a bug report.

## AUv3 and VST3 Plug-ins

**Q25. Which sample rate and bit depth does the plug-in use?**

The DAW negotiates the plug-in's sample rate and processing format. The project's
192 kHz, 32-bit float default describes its preferred high-resolution offline policy,
not a request that overrides the host session. A 48 kHz DAW session therefore runs the
plug-in at 48 kHz.

**Q26. Why can a plug-in be discovered but fail to load?**

Discovery only proves that the host found a bundle. Loading can still fail because of
architecture, signing, deployment-target, factory-registration, bus-layout, or state
initialization errors. Check the DAW validation log and run the repository's plug-in
validation commands before changing the GUI.

**Q27. Why does the plug-in appear under Colby Leider rather than verbx?**

DAWs commonly group effects by manufacturer. **Colby Leider** is the manufacturer and
author metadata; **verbx** is the product name. Both fields are intentional and should
remain stable so saved sessions and host catalogs do not split into duplicate entries.

**Q28. What is on the Expert page?**

The Expert page exposes the detailed network, damping, modulation, spatial, dynamics,
and quality controls that would overwhelm the primary performance page. Start from a
preset, make broad changes on the main page, and enter Expert only when a specific
technical adjustment is needed.

**Q29. Does the realtime spectrum analyzer change the sound?**

No. It is a metering overlay fed from the audio path, not an audio processor. Disabling
or reducing its refresh rate should leave the rendered signal unchanged while reducing
GUI work on constrained systems.

**Q30. How should RT60 automation behave in the plug-in?**

The coarse control moves logarithmically across the full 0.01–360 second performance
range, while the fine control trims the current value. Hosts should automate the
normalized parameter smoothly; abrupt jumps in a feedback network may still require
internal smoothing to avoid clicks or unstable energy changes.

## Analysis and Dereverberation

**Q31. How do I extract reverb metrics from an audio file?**

Run:

```bash
verbx analyze recording.wav --reverb --lufs --edr --room --json-out analysis.json
```

Use `--input-kind ir` for a measured impulse response, `--input-kind program` for music
or speech, or `--input-kind auto` when verbx should infer the interpretation.

**Q32. Why do RT60 estimates differ between an impulse response and music?**

An impulse response exposes a direct decay curve. Music and speech contain overlapping
events, changing spectra, dynamics processing, and background noise, so their estimates
are inferential. Treat program-material measurements as descriptive evidence rather
than the same laboratory quantity obtained from a clean excitation and response.

**Q33. What do EDR and frame-level outputs add?**

An energy-decay relief shows decay across both time and frequency, revealing bands that
ring longer than a single broadband RT60 suggests. Frame output preserves local
measurements for plotting, quality control, or downstream research instead of reducing
the file to one summary number.

**Q34. Can verbx estimate room properties from an ordinary recording?**

It can report model-based room indicators with `--room`, but these are estimates, not a
replacement for geometry, calibrated excitation, microphone positions, and atmospheric
conditions. Confidence should decrease when the source is dense, dynamically processed,
or recorded in a noisy environment.

**Q35. What can dereverberation realistically remove?**

Dereverberation can reduce diffuse late energy and improve intelligibility, but it
cannot perfectly recover information masked by reflections or noise. Aggressive settings
may create pumping, spectral holes, or speech artifacts. Compare bypassed and processed
signals at matched loudness and preserve the original recording.

**Q36. How do I start realtime dereverberation safely?**

First verify devices with `verbx realtime --list-devices`. Then run realtime with
`--live-mode dereverb`, conservative reduction, and a stable block size. Monitor speech
transients, noise pumping, CPU deadline misses, and the saved JSON report before adding
a reverb stage.

## Impulse Responses, Geometry, and Spatial Audio

**Q37. How do I discover the impulse-response tools?**

Run `verbx ir --help`. The group includes generation, analysis, SOFA inspection and
extraction, geometry tracing, processing, morphing, sweep creation, and fitting. Run the
selected subcommand with `--help` because its required files and coordinate conventions
are more specific than the top-level summary.

**Q38. Can verbx generate reverberation from a DXF room model?**

The geometry workflow can trace supported DXF room descriptions for early-reflection
and IR work, subject to the documented parser and material limitations. Validate scale,
closed surfaces, source position, receiver position, normals, and units before trusting
the acoustical result. A CAD drawing is not automatically a complete acoustic model.

**Q39. What is a hybrid geometry-plus-algorithmic reverb?**

It uses traced or measured early arrivals for spatial identity and an algorithmic late
field for dense, controllable decay. The handoff must match level, spectrum, direction,
and time so the two parts sound like one room rather than an echo pattern followed by a
separate tail.

**Q40. What is SOFA used for?**

SOFA stores spatially indexed acoustic measurements such as head-related or room impulse
responses. Use `verbx ir sofa-info` to inspect a file and `verbx ir sofa-extract` to
select responses. Confirm coordinate conventions and sample rate before convolution.

**Q41. Does verbx support surround and immersive workflows?**

Yes. The CLI includes multichannel routing, ambisonic analysis, matrix convolution, and
immersive template, handoff, quality-control, and queue workflows. Preserve explicit
channel labels and ordering throughout; a valid channel count does not prove that every
speaker is mapped to the intended position.

**Q42. Does verbx create a finished Dolby Atmos master?**

verbx can prepare, render, inspect, and hand off spatial assets, but final Dolby Atmos
authoring and delivery still belong in a licensed renderer and DAW workflow that meets
the destination specification. Treat verbx reports as technical companions, not as a
replacement for required Dolby validation or deliverables.

## Presets, Reproducibility, and Troubleshooting

**Q43. How do I browse the large preset library?**

Run `verbx presets` to list presets, `verbx presets --show NAME` to inspect one, and
`verbx presets --validate NAME` to validate its resolved data. Start from a descriptive
family rather than auditioning hundreds in arbitrary order, then compare at matched
loudness.

**Q44. How do I make a render reproducible?**

Pin the verbx version, engine, preset or complete option set, random seed where used,
input-file checksum, and output format. Save `--json-out` and `--repro-bundle` artifacts
with the audio. Reproducibility includes the resolved defaults, not merely the switches
typed on the command line.

**Q45. How can I compare two processing choices objectively?**

Use `verbx compare` for supported measurements, then perform a level-matched listening
comparison. Keep the source, trim, output format, and analysis settings identical. A
lower error metric or longer RT60 is not automatically the musically preferred result.

**Q46. What information belongs in a useful bug report?**

Include `verbx version`, operating system and architecture, the exact command, the
smallest shareable input, expected and observed behavior, exit status, doctor JSON,
processing or failure JSON, and relevant DAW validation logs. Remove private paths or
audio only after preserving enough structure to reproduce the defect.

**Q47. When should I clear or inspect the cache?**

Use `verbx cache --help` when stale generated assets, unexpectedly high disk use, or a
version transition points to cached data. Inspect before deleting whenever possible.
Clearing a cache may make the next run slower, but it should not be required for ordinary
parameter changes.

**Q48. What is the safest way to explore extreme settings?**

Lower monitor gain, keep a limiter after the experimental path, render a short excerpt,
use a bounded duration, and save dry-run or failure reports. Feedback, freeze, modulation,
and long decay can accumulate energy unexpectedly. Preserve the dry original and move
from conservative values toward the extreme rather than starting at the maximum.
