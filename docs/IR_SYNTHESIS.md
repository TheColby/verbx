# IR Synthesis — A Dual-Layer Reference

> Version: v0.7.4 (public alpha)

This document is written for two readers at once. The plain-English sections
give you enough to use the tool effectively without reading source code. The
technical sections go deep — modal decay math, FDN topology, stochastic density
shaping, morph blend modes — and assume you know what a Schroeder allpass is and
roughly why it matters. Skip what you don't need.

---

## 1. What Is an Impulse Response?

### Plain English

Clap your hands once in a cathedral. The sound you hear after that single clap
— all those echoes, the wash, the way it slowly dies away — that is the room's
impulse response. It is the room's acoustic fingerprint.

Convolution reverb works by capturing that fingerprint as a digital audio file
and mathematically "stamping" it onto whatever audio you feed in. The result
sounds like the audio was recorded in that room. It is not simulation, it is
multiplication in the frequency domain — and it sounds convincing because it
uses real (or carefully synthesized) physics.

verbx does not require you to record a real room. It synthesizes IRs from
scratch, with deterministic parameters, so you get the same result every time
from the same seed.

### For DSP Engineers

An impulse response h[n] completely characterizes a linear time-invariant (LTI)
system. In the context of room acoustics, convolution reverb computes:

    y[n] = (x * h)[n] = sum_k x[k] * h[n - k]

which in practice is implemented as OLA or OLS partitioned convolution to keep
block latency manageable. verbx's IR synthesis produces h[n] directly, bypassing
any physical measurement chain. The goal is parametric control rather than
physical accuracy.

All internal DSP runs in float64. Output subtype (float32, float64, PCM-16/24)
is a container choice at export time and does not affect synthesis precision.
The output is always normalized (peak or RMS/LUFS depending on config) before
writing.

RT60 is defined in the Sabine sense: the time for the sound pressure level to
decay by 60 dB. Internally verbx resolves RT60 into per-mode envelope time
constants via:

    tau = RT60 / ln(10^6) = RT60 / 13.816

which gives the exponential decay envelope e^(-t/tau) used across modes.

---

## 2. The Four Synthesis Modes

### 2a. `fdn` — Feedback Delay Network

#### What it sounds like

Dense, smooth, algorithmic reverb. Think classic hardware reverb units from the
1980s — less "room" character, more "sound in space." Works well for clean
ambience that sits behind a mix without calling attention to itself. The
modulation options add subtle pitch shimmer that keeps it from sounding static.

#### How it works

The FDN is the workhorse of algorithmic reverb. The architecture is a set of N
delay lines connected in a feedback loop through a mixing matrix M:

    s[n] = M * s[n - d] + b * x[n]
    y[n] = c^T * s[n]

where d is the vector of delay lengths (in samples), b is the input gains, and
c is the output tap weights. For the loop to be stable and lossless, M must be
unitary or near-unitary — verbx supports Hadamard matrices (default) and
random orthogonal matrices. The Hadamard construction at order N is efficient
(O(N log N) multiply) and guarantees perfect echo density growth.

Delay line lengths are chosen to be mutually coprime (no shared factors) to
maximize echo density. The default `fdn_lines = 8` gives 8 delay lines; going
to 16 significantly increases diffusion at the cost of memory and compute.

The FDN includes optional time-varying modulation (`fdn_tv_rate_hz`,
`fdn_tv_depth`): each delay line length is sinusoidally modulated at an
individually offset rate derived from the master rate. This breaks up metallic
resonances at the cost of LTI purity — the output is no longer strictly an IR
of a fixed system, but the artifact is perceptually benign at low modulation
depths.

Three-band RT60 control (`fdn_rt60_low`, `fdn_rt60_mid`, `fdn_rt60_high`) is
implemented as per-band absorption shelving filters inserted in each feedback
loop — similar in spirit to the Jot-Chaigne absorptive FDN. Low-frequency RT60
is typically longer than mid in real rooms; setting `fdn_rt60_low` slightly
above `fdn_rt60_mid` makes the tail feel more physical.

Graph topology options (`ring`, `random`) control how the N delay lines
interconnect before the full mixing matrix is applied. Ring is the classic
single-loop structure; random graph with degree k introduces k cross-connections
per node and increases early diffusion at the cost of slightly less predictable
echo pattern.

The cascade FDN option (`fdn_cascade`) chains two FDN stages at different delay
scales, with the second stage fed by the first at a ratio of `fdn_cascade_delay_scale`.
This is loosely analogous to Griesinger's nested FDN and significantly increases
low-frequency density without proportionally increasing the delay buffer size.

**Known limitation:** The FDN IR is rendered by impulse-exciting the network and
reading N * length_samples samples of output. For very long IRs (120s at 48kHz)
this is about 5.76e9 operations — non-trivial. Sparse mode (`fdn_sparse`) reduces
this by zeroing low-magnitude delay connections, with controllable degree.

---

### 2b. `stochastic` — Noise-Shaped Diffuse Tail

#### What it sounds like

Pure reverb wash with no discrete echoes and no tonal coloration. Sounds like
the late tail of a large space — a long, smooth decay. Useful as a bed when you
want pure texture without any pitched character. Can sound a bit thin on its own
(there is no direct sound energy), but as a component in hybrid mode it provides
exactly what it promises: texture.

#### How it works

Stochastic synthesis generates band-limited noise shaped by a smooth exponential
envelope. The core idea is that a diffuse reverb tail, at sufficient echo density,
is perceptually indistinguishable from exponentially decaying noise filtered to
match the room's frequency-dependent absorption. This is the statistical basis
for Gardner's observation that above roughly 2000 echoes/second the ear cannot
resolve individual reflections.

The implementation generates a Gaussian noise vector of the full IR length,
applies a per-channel decorrelation (different seed offsets per channel), then
multiplies by an RT60 envelope:

    h[n] = noise[n] * e^(-n / (sr * tau))

where tau = RT60 / ln(10^6). The `diffusion` parameter controls a frequency-
domain shaping pass: values above 0.5 increasingly smooth the spectral envelope
toward white (flat), while lower values allow more spectral irregularity to
persist, giving a slightly colored tail that can sound more like a real room.

The `density` parameter scales the RMS of the noise before the envelope is
applied. It does not change the decay shape, only the apparent "thickness" of
the tail. At density > 1.0 the tail can clip before normalization, which is
intentional — normalization re-scales — but very high density values will pull
the effective noise floor up and reduce dynamic range in the output.

Band control (`rt60_low`, `rt60_high`): when these differ, the stochastic tail
interpolates between two exponential envelopes across frequency using a
crossover filter. This approximates frequency-dependent absorption without a
full filter-per-sample approach.

---

### 2c. `modal` — Decaying Modal Resonator Bank

#### What it sounds like

Pitched, tonal, resonant — like a metal plate, a wine glass, or a very live
tiled room. Modes ring at specific frequencies, giving reverb a musical quality
that can either blend into pitched content or clash with it depending on tuning.
Excellent for designed reverbs on melodic material when you want the space to
have harmonic relationship to the source.

#### How it works

Each mode is an exponentially decaying sinusoid:

    h_k[n] = A_k * sin(2*pi*f_k*n/sr + phi_k) * e^(-n/tau_k)

with amplitude A_k, frequency f_k, phase phi_k, and time constant tau_k. The
bank sums N such modes:

    h[n] = (1/sqrt(N)) * sum_{k=1}^{N} h_k[n] + epsilon * noise_bed[n]

The 1/sqrt(N) normalization keeps total energy approximately constant regardless
of modal count, which is the right thing to do (amplitude does not add linearly
for uncorrelated modes — RMS does).

Frequencies are sampled log-uniformly between `modal_low_hz` and `modal_high_hz`:

    f_k = exp(U[log(f_low), log(f_high)])

This is the correct distribution for musical pitch perception — equal intervals
on a log scale correspond to equal musical intervals. Linear uniform sampling
would cluster modes near `modal_high_hz` perceptually.

The Q-to-tau relationship follows standard resonator theory:

    tau_q = Q / (pi * f_k)

but verbx blends this with the global RT60 target:

    tau_rt = RT60 / 6.91
    tau_k = min(max(tau_q, 0.01), max(2 * tau_rt, 0.03))

This clipping prevents individual modes from ringing dramatically longer than
the RT60 budget, which would otherwise create audible individual tones at the
end of the tail — a problem that sounds terrible and is hard to fix after the
fact. In practice you can push `modal_q_max` to 200 or higher if you want
those long-ringing tones intentionally (bells, chimes), just know the IR will
have frequency-domain spikes that may interact poorly with the source material
unless tuned deliberately.

Harmonic alignment (`--f0`, `--analyze-input`) nudges mode frequencies toward
integer multiples of a fundamental using `tune_frequency_to_targets()`. The
`align_strength` parameter (0–1) linearly interpolates each mode's sampled
frequency toward the nearest harmonic target. At strength 1.0, all modes
collapse to exact harmonics — a fully pitched comb. At 0.0, no alignment.
Values around 0.5–0.7 give a nice blend of harmonic coloration without the
artificiality of a pure comb filter.

An air noise bed at amplitude 0.02 * rms is added to prevent the IR from
being spectrally empty between modal peaks. Without it, convolution with
broadband content can reveal the gaps as spectral dips.

**Stereo:** each mode is panned independently using equal-power panning:
L = sqrt(0.5 * (1 - pan)), R = sqrt(0.5 * (1 + pan)). For channels > 2,
additional channels receive the mono sum scaled by 1/ch. Not ideal for
surround, but functional for stereo and tolerable for LCR.

---

### 2d. `hybrid` — The Mode You Will Actually Use

#### What it sounds like

Combines discrete early echoes (the "slap" you hear in the first 50–120ms)
with a blended late tail. The early reflections give spatial cues — your brain
uses them to estimate room size and shape. The late tail does the wash. Together
they sound more like a real room than any single mode alone.

In practice, hybrid is what you actually want for most work. The other modes
are useful for sound design extremes (very pure algorithmic color, pitched modal
effects) or as components you mix yourself. Hybrid handles general-purpose
reverb production well.

#### How it works

Hybrid generation follows a four-step pipeline:

1. Generate early reflections (see Section 3)
2. Generate three late-tail components independently: stochastic (seed + 11),
   modal (seed + 17, half the modal count), FDN (seed + 23)
3. Blend the three tails with fixed weights:

       ir_tail = 0.55 * stoch + 0.25 * modal + 0.20 * fdn

4. Add early reflections by overlap-add into the first er_max_delay_ms samples

The seed offsets (+11, +17, +23) are not magic numbers — they just ensure the
three RNGs produce independent sequences from the same user seed. The blending
weights (0.55/0.25/0.20) were chosen empirically: stochastic dominates because
it produces the smoothest late tail, modal adds tonal character, FDN adds
algorithmic density structure. These weights are fixed in the current release;
future versions may expose them as parameters.

The post-blend pipeline applies globally: harmonic alignment, optional Modalys
resonator layer, then IR shaping (filters, normalization). So tuning flags like
`--f0` and `--resonator` work across all modes, not just modal.

---

## 3. Early Reflections

### Perceptual Role

Early reflections — the first 5–80ms or so after the direct sound — are the
most important part of reverb for spatial perception. They are what lets you
tell a bathroom from a concert hall even before the tail arrives. They carry
IACC (interaural cross-correlation) cues that localize you in space, and their
density and timing pattern gives a strong sense of room size.

The direct sound is always present at sample 0 (amplitude 1.0 per channel) as
a convolution anchor — this is the Kronecker delta that initiates the impulse
response and ensures the convolution engine sees a properly calibrated gain
reference.

### Stochastic Cloud Implementation

Early reflections in verbx are a sparse, deterministic stochastic cloud rather
than a geometric ray-tracing approximation. This is a deliberate choice: geometric
computation would require room dimensions and surface materials as inputs, adding
complexity with limited benefit for non-realistic spaces.

The generator (`generate_early_reflections`) places `er_count` taps into a
buffer of length `er_max_delay_ms` samples. Each tap's delay is sampled uniformly
over [1, max_delay_samples]:

    delay_k ~ U[1, floor((er_max_delay_ms / 1000) * sr)]

Amplitude follows one of three decay laws indexed by `er_decay_shape`:

    linear:  amp = 1 - (delay / max_delay)
    sqrt:    amp = sqrt(max(1e-6, 1 - delay / max_delay))
    exp:     amp = exp(-3.2 * delay / max_delay)

The exponential law (default) corresponds to a room with absorption coefficient
around 0.2 uniformly distributed, which is plausible for a moderately live space.
The sqrt law gives a slower initial roll-off and is useful for very live spaces
like stairwells. Linear is rarely what you want but is there for completeness.

The raw decay amplitude is then multiplied by a uniform random jitter in
[0.35, 1.0] and by `er_room` (clipped to [0.1, 3.0]):

    amp_k = decay(delay_k) * er_room * U[0.35, 1.0]

The `er_room` parameter is loosely a room-size proxy: values > 1.0 increase
reflection amplitude, making the space feel more reverberant; values < 1.0
create a drier early field.

Stereo width is applied as pan spread per tap. Pan is sampled U[-1, 1] and
scaled by `er_stereo_width`. Left/right amplitudes use a simple linear pan law
(not equal-power) because the taps are sparse enough that the distinction
matters less than in dense diffuse material:

    L_k = amp_k * 0.5 * (2 - max(0, pan_k))
    R_k = amp_k * 0.5 * (2 + min(0, pan_k))

For er_stereo_width = 0.0, all taps are center-panned (mono ER). For 1.0,
full ±1.0 pan range. For 2.0, panning can exceed the unit range, intentionally
widening beyond the speaker base — useful for envelopment effects but beware
mono compatibility.

One thing I would change in a future version: the tap delays should ideally be
constrained to avoid clustering (e.g., via Poisson process sampling or minimum
inter-tap spacing). Uniform sampling occasionally produces bunches of taps
at similar delays that create audible comb-filter artifacts on transient material.
For now, setting `er_count` to 16–24 rather than 64 reduces this risk.

---

## 4. IR Morphing

### What It Does

Morphing takes two existing IRs — say, a small room and a large hall — and
interpolates between them to create a new IR that has characteristics of both.
You control the blend position with `--alpha` (0.0 = first IR, 1.0 = second IR,
0.5 = midpoint). The cache means repeated morphs at the same parameters are
instantaneous.

This is useful in production when you want to automate a sense of space expanding
or contracting across a section of a piece, or when you have two reference IRs
and want something "between" them tonally.

### Blend Modes

Three blend modes are available via `--mode`:

**`linear`**

Simple time-domain weighted sum:

    h_out[n] = (1 - alpha) * h_A[n] + alpha * h_B[n]

Fast but acoustically naive. If A and B have different RT60 values, the
crossfaded IR will have a decay shape that is neither A's nor B's — it will
be a linear interpolation of amplitudes, which for exponential decays means
the instantaneous decay rate changes non-monotonically with alpha. At alpha=0.5
between a 1s and 4s RT60, you get approximately a 2s decay, but the shape
has a kink where the two envelopes cross. For subtle blends (alpha < 0.2
or > 0.8) this is usually fine.

**`spectral`**

Blend in the frequency domain:

    H_out[k] = (1 - alpha) * H_A[k] + alpha * H_B[k]

where H_A, H_B are the FFTs of the respective IRs (zero-padded to avoid
circular convolution artifacts on the way back). This preserves the spectral
shape of each source more faithfully than time-domain blending. Particularly
useful when the two IRs have different frequency-dependent coloration (one
bright, one dark) and you want to smoothly interpolate that color.

Phase handling: the implementation blends complex spectra, not magnitude-only.
If A and B have substantially different phase responses (which real IRs almost
always do), the interpolated phase can produce notches in the output. The
`--align-decay` flag pre-aligns the energy onset of both IRs before blending,
which partially mitigates this.

**`envelope-aware`**

The most sophisticated option and the one I would recommend as a starting point
for serious work. It estimates the temporal energy envelope of each IR, blends
the envelopes separately, then re-modulates the blended spectral content. This
is loosely analogous to the approach in Cross-Synthesis of exponential decays.

Specifically:

1. Compute short-time energy envelope E_A[t] and E_B[t] via squared Hilbert
   analytic signal or RMS-over-window (current implementation uses the latter
   at 10ms windows for speed)
2. Interpolate envelopes: E_out[t] = (1-alpha)*E_A[t] + alpha*E_B[t]
3. Generate late tail by blending at the spectral level with envelope correction:
   scale H_out[k] such that the resulting time-domain signal matches E_out

Morph quality diagnostics are reported in the metadata: RT60 drift (|RT60_out
- expected_interpolated_RT60| in seconds) and spectral distance (log-spectral
deviation in dB between H_out and the target interpolated spectrum). Use these
to detect degenerate morphs before committing them to a batch render.

---

## 5. The Cache

### Why It Matters

Generating a 120-second hybrid IR at 48kHz involves several million floating-
point operations across three independently synthesized late tails plus early
reflections. That takes a couple of seconds on a modern CPU. If you are running
a batch of 50 renders that all use the same IR parameters, regenerating it every
time is wasteful and slows iteration.

The cache stores both the audio and a metadata sidecar, so a cache hit is just
a file read — negligible latency.

### How the Hash Key Works

The cache key is a 16-character hex prefix of SHA-256 applied to the
JSON-serialized `IRGenConfig` dataclass:

    payload = asdict(config)
    payload["_schema"] = "verbx-ir-v0.4"
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    key = sha256(text.encode("utf-8")).hexdigest()[:16]

Every field of `IRGenConfig` contributes to the hash — mode, length, sr,
channels, seed, rt60, all FDN topology flags, all modal parameters, all
resonator settings, tuning, normalization target, everything. Changing any
single field produces a different hash and therefore a different cache entry.

The schema version string `"verbx-ir-v0.4"` is a namespace guard. When the
IRGenConfig schema changes in a breaking way (field added, removed, or renamed),
bump this version string. All existing cache entries will become unreachable
(not deleted, just miss) and will be regenerated on next access. This is
intentional: it avoids silent stale-cache bugs at the cost of some wasted disk
space during version transitions.

Cache files live at:

    .verbx_cache/irs/<hash>.wav
    .verbx_cache/irs/<hash>.meta.json

The metadata JSON contains the full config, version, seed, and IR metrics
(RT60 measured, spectral centroid, etc.) as computed by `analyze_ir()`.

Management:

```bash
hatch run verbx cache info    # show size, entry count, oldest/newest
hatch run verbx cache clear   # remove all cached IR files
```

Note: `cache clear` is non-reversible. If you have long IRs cached that took
significant time to generate, consider archiving them before clearing.

---

## 6. Quick Start Recipes

### Wash / Ambient Bed (120s)

For pad layers, atmospheric beds, tape-delay washout. Long RT60, high diffusion,
gentle early reflections. Peak normalization at -1 dBFS leaves headroom for
downstream summing.

```bash
hatch run verbx ir gen irs/wash_120.wav \
  --mode hybrid --length 120 --rt60 95 --damping 0.45 \
  --diffusion 0.7 --density 1.2 --er-count 32 --er-max-delay-ms 120 \
  --normalize peak --peak-dbfs -1
```

### Cinematic Hybrid

Good starting point for score and sound design work. Moderate length, gentle
low-frequency roll-off, subtle modulation to avoid static digital quality.
`--tilt 1.5` tilts the spectrum slightly darker (positive tilt = more low end).

```bash
hatch run verbx ir gen irs/cinematic_hybrid.wav \
  --mode hybrid --length 75 --rt60 60 --tilt 1.5 --lowcut 80 --highcut 12000 \
  --er-room 1.3 --er-stereo-width 1.2 --mod-depth-ms 2.0 --mod-rate-hz 0.1
```

### Pitched Modal — Tuned Resonance

For pitched instruments where you want the room to harmonize with the source.
Set `--f0` to the root note of your key. High Q range (7–90) gives a mix of
short and long-ringing modes. Increase `--modal-count` for denser resonance.

```bash
hatch run verbx ir gen irs/pitched_modal.wav \
  --mode modal --length 45 --rt60 35 --seed 99 --tuning A4=432 \
  --modal-count 64 --modal-q-min 7 --modal-q-max 90 \
  --modal-low-hz 60 --modal-high-hz 9000 --modal-spread-cents 8
```

### Resonator-Colored Hybrid (Modalys-Inspired)

The Modalys resonator layer sits on top of the hybrid tail starting at
`--resonator-late-start-ms`. Good for adding physical resonance character
to an otherwise smooth hybrid — think the body resonance of a large wooden
instrument or a tuned metal plate. `--resonator-mix 0.38` is a conservative
setting; go to 0.6+ for obviously colored effects.

```bash
hatch run verbx ir gen irs/resonator_hybrid.wav \
  --mode hybrid --length 90 --rt60 70 --seed 33 \
  --resonator --resonator-mix 0.38 --resonator-modes 24 \
  --resonator-q-min 10 --resonator-q-max 120 \
  --resonator-low-hz 80 --resonator-high-hz 7000 \
  --resonator-late-start-ms 90 --f0 "64 Hz"
```

### Batch Render

```bash
hatch run verbx batch template > manifest.json
hatch run verbx batch render manifest.json --jobs 4
```

### Tempo-Synced Pre-Delay

```bash
hatch run verbx render in.wav out.wav --pre-delay 1/8D --bpm 120
```

Supported note forms: `1/4`, `1/8D` (dotted), `1/8T` (triplet), or raw seconds
(`0.125`).

### Morph Two IRs

```bash
hatch run verbx ir morph irs/hall_A.wav irs/hall_B.wav irs/hall_AB.wav \
  --mode envelope-aware --alpha 0.6 --early-ms 80 --align-decay
```

### Auto-Tune IR to Source Analysis

Analyzes the source file's detected fundamentals and harmonic content, then
aligns modal frequencies accordingly. Useful when you want the reverb to
reinforce the harmonic content of a specific instrument recording.

```bash
hatch run verbx ir gen irs/tuned_from_input.wav \
  --mode hybrid --analyze-input source.wav
```

---

## 7. Choosing Parameters for Specific Acoustic Spaces

This section is opinionated. These are starting points, not prescriptions.

### Small Live Room (drum room, bathroom)

Characteristics: short RT60 (0.3–0.8s), dense early reflections with hard
flutter risk, bright spectral character, strong IACC asymmetry.

```bash
--mode hybrid --length 3 --rt60 0.5 \
--er-count 48 --er-max-delay-ms 30 --er-decay-shape exp \
--er-room 0.8 --damping 0.6 --diffusion 0.4 \
--highcut 14000 --lowcut 100
```

Keep `er_count` high and `er_max_delay_ms` short. The dense early cloud in
a small room is the defining character — do not cut corners here. Low `--rt60`
with moderate `--damping` will give you that tight, controlled sound.

### Medium Hall (recital hall, 300–800 seats)

The "bread and butter" reverb space. RT60 around 1.5–2.0s mid, slightly
longer at low frequencies.

```bash
--mode hybrid --length 6 --rt60 1.8 \
--fdn-rt60-low 2.2 --fdn-rt60-mid 1.8 --fdn-rt60-high 1.2 \
--er-count 28 --er-max-delay-ms 60 --er-room 1.0 \
--damping 0.35 --diffusion 0.6 --tilt 0.5
```

The three-band RT60 control is worth using here. Real halls have longer bass
decay — it is the signature of mass and volume. Skipping this makes algorithmic
reverb sound synthetic quickly.

### Cathedral / Large Church

Long RT60, extreme low-frequency prolongation, relatively sparse early
reflections (the room geometry produces long inter-reflection paths), heavy
high-frequency absorption.

```bash
--mode hybrid --length 30 --rt60 7 \
--fdn-rt60-low 9.5 --fdn-rt60-high 4.5 \
--er-count 18 --er-max-delay-ms 110 --er-room 1.2 \
--damping 0.5 --diffusion 0.75 --density 0.9 \
--lowcut 40 --highcut 10000 --tilt 2.0
```

High `--tilt` darkens the spectrum to match the heavy air absorption over
long distances. With a 30-second IR the cache becomes practically essential.

### Metal Plate / Physical Resonator

Not a room at all — a physical resonator with well-defined modal structure.
The modal mode is the right tool here, not hybrid.

```bash
--mode modal --length 8 --rt60 5 \
--modal-count 96 --modal-q-min 40 --modal-q-max 400 \
--modal-low-hz 200 --modal-high-hz 16000 \
--modal-spread-cents 3 --f0 "220 Hz"
```

High Q values (40–400) give the long, singing ring of plate reverb. The very
narrow `--modal-spread-cents 3` keeps modes tightly clustered around harmonic
targets for a more pitched character. Lower Q (5–15) at wider spread approaches
spring reverb.

### Infinite / Frozen Pad

Not physically realistic, obviously, but useful as a creative tool — the IR
never really decays. Very long length with high RT60 relative to length.

```bash
--mode stochastic --length 120 --rt60 180 \
--diffusion 0.85 --density 0.8 --damping 0.15
```

The RT60 exceeding the IR length means the decay envelope never reaches -60 dB
within the buffer — the tail is essentially flat. Combined with high diffusion
you get a dense, white-ish wash. Works well convolved with a long pad or a
heavily sustained note.

---

## 8. Expert Reference: Implementation Notes, Limitations, Future Work

### Precision

All synthesis arrays are `np.float64` throughout. The `AudioArray` type alias
is `npt.NDArray[np.float64]`. Normalization and shaping are also f64. Conversion
to f32 or integer PCM happens only at the `sf.write()` call in
`write_ir_artifacts()` and `generate_or_load_cached_ir()`. If you are loading
cached IRs programmatically, always request `dtype="float64"` in `sf.read()` —
this is what the library does internally, but third-party code may not.

### Determinism Guarantees

Given identical `IRGenConfig`, `generate_ir()` will produce bit-identical output
across runs on the same platform and NumPy version. The RNG throughout is
`numpy.random.default_rng(seed)` (PCG64), not the legacy `np.random` global
state. Cross-platform determinism (x86 vs. ARM) is not formally guaranteed
because of potential FPU differences in transcendental function evaluation
(specifically `np.exp` and `np.sin` on long arrays). In practice, for seeds
used in production it has not been an issue, but do not rely on cross-platform
bit-identity for regression tests.

### Hybrid Blend Weights

The fixed weights `(0.55 stoch, 0.25 modal, 0.20 fdn)` in hybrid mode are the
result of listening tests with speech and music material. They are hard-coded.
If you need different weights, the cleanest approach is to generate the three
components separately (using the three individual modes with appropriate seeds)
and mix them externally. This is more verbose but gives full control.

### The Resonator Layer

`apply_modalys_resonator_layer()` is not a faithful Modalys physical simulation
— the name is aspirational. It is a second modal bank applied as a late-tail
coloration on top of the synthesized IR, with a soft gate that only activates
after `resonator_late_start_ms`. The gate is not hard-gated but uses a smooth
fade-in envelope over approximately 20ms to avoid clicks. The resonator bank
shares the same harmonic alignment infrastructure as the main modal mode, so
`--f0` and `--analyze-input` targets propagate through to it.

### Known Limitations

**No perceptual diffusion modeling.** The FDN diffusion is purely algorithmic;
there is no Schroeder allpass diffuser chain in the current release. Adding one
(e.g., a series of 4–6 allpass sections before the FDN input) would improve
early echo density at short RT60 settings. This is on the roadmap.

**Uniform reflection time sampling.** As noted in Section 3, uniform tap-time
sampling in the early reflections module occasionally produces clusters. A
Poisson-disk or minimum-interval constraint would give more physically plausible
tap spacing.

**Spectral morph phase interpolation.** The `spectral` blend mode blends complex
spectra, meaning phase interpolation is linear in the complex plane, which can
produce magnitude notches at the midpoint. A magnitude-only blend with phase
selection (or phase interpolation via circular statistics) would be more robust.

**FDN IR length.** Very long FDN IRs (60s+) are slow to generate because the
FDN must be simulated sample-by-sample (or block-by-block). There is no shortcut
for a time-varying FDN — you cannot analytically compute its IR without running
the network. The cache makes this practical for production use, but the first
generation of a long FDN IR takes wall-clock time proportional to IR length.

**Channel count > 2.** The modal panner for channels > 2 distributes all
extra channels as mono sum / ch, which is not proper ambisonic or surround
panning. Do not use verbx-synthesized IRs for surround production without
verifying the channel assignments match your downstream decoder expectations.

### Roadmap Alignment (v0.7.4)

From the R3 milestones:

- `R3.1 cache determinism`: cross-sample-rate cache lookup with canonical
  resampling. An IR generated at 96kHz should be retrievable (with a quality
  warning) by a 48kHz render job without full regeneration.

- `R3.2 operational QA`: morph diagnostic artifacts — small reference renders
  that can be compared against golden files in CI. The morph metadata already
  includes RT60 drift and spectral distance; R3.2 formalizes the acceptance
  thresholds.

- `R3.3 failure safety`: retry/resume logic for long morph batches that fail
  mid-job. Currently a failed morph job leaves partial output; R3.3 will add
  a checkpoint file and graceful resume.

When updating `ir morph` or render-time `--ir-blend` behavior, keep this
document, the CLI switch tables in `README.md`, and `docs/REFERENCES.md`
source links in sync.

### References

For the DSP foundations underlying this implementation:

- Jot, J.-M. & Chaigne, A. (1991). Digital delay networks for designing
  artificial reverberators. AES 90th Convention.
- Schlecht, S. J. & Habets, E. A. P. (2017). On lossless feedback delay
  networks. IEEE Transactions on Signal Processing.
- Valimaki, V. et al. (2012). Fifty years of artificial reverberation.
  IEEE Transactions on Audio, Speech, and Language Processing.
- Gardner, W. G. (1992). A realtime multichannel room simulation system.
  JASA.
- Griesinger, D. (1996). Spaciousness and envelopment in musical acoustics.
  AES 101st Convention.

See `docs/REFERENCES.md` for full citation list including Modalys and
convolution engine references.
