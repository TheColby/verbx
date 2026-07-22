# IR Synthesis – A Dual-Layer Reference

> Version: v0.7.7 (public alpha)

This document is written for two readers at once. The plain-English sections
give you enough to use the tool effectively without reading source code. The
technical sections go deep – modal decay math, FDN topology, stochastic density
shaping, morph blend modes – and assume you know what a Schroeder allpass is and
roughly why it matters. Skip what you don't need.

---

## 1. What Is an Impulse Response?

### Plain English

Clap your hands once in a cathedral. The sound you hear after that single clap
— all those echoes, the wash, the way it slowly dies away – that is the room's
impulse response. It is the room's acoustic fingerprint.

Convolution reverb works by capturing that fingerprint as a digital audio file
and mathematically "stamping" it onto whatever audio you feed in. The result
sounds like the audio was recorded in that room. It is not simulation, it is
multiplication in the frequency domain – and it sounds convincing because it
uses real (or carefully synthesized) physics.

verbx does not require you to record a real room. It synthesizes IRs from
scratch, with deterministic parameters, so you get the same result every time
from the same seed.

### For DSP Engineers

An impulse response $h[n]$ completely characterizes a linear time-invariant (LTI)
system. In the context of room acoustics, convolution reverb computes:

$$
y[n] = (x * h)[n] = \sum_k x[k]\,h[n-k]
$$

which in practice is implemented as OLA or OLS partitioned convolution to keep
block latency manageable. verbx's IR synthesis produces $h[n]$ directly, bypassing
any physical measurement chain. The goal is parametric control rather than
physical accuracy.

All internal DSP runs in float64. Output subtype (float32, float64, PCM-16/24)
is a container choice at export time and does not affect synthesis precision.
The output is always normalized (peak or RMS/LUFS depending on config) before
writing.

RT60 is defined in the Sabine sense: the time for the sound pressure level to
decay by 60 dB. Internally verbx resolves $T_{60}$ into per-mode envelope time
constants via:

$$
\tau = \frac{T_{60}}{\ln(10^6)} = \frac{T_{60}}{13.816}
$$

which gives the exponential decay envelope $e^{-t/\tau}$ used across modes.

---

## 2. The Four Synthesis Modes

### 2a. `fdn` – Feedback Delay Network

#### What it sounds like

Dense, smooth, algorithmic reverb. Think classic hardware reverb units from the
1980s – less "room" character, more "sound in space." Works well for clean
ambience that sits behind a mix without calling attention to itself. The
modulation options add subtle pitch shimmer that keeps it from sounding static.

#### How it works

The FDN is the workhorse of algorithmic reverb. The architecture is a set of $N$
delay lines connected in a feedback loop through a mixing matrix $M$:

$$
\begin{aligned}
s[n] &= M\,s[n-d] + b\,x[n] \\
y[n] &= c^{\mathsf T}s[n]
\end{aligned}
$$

where $d$ is the vector of delay lengths (in samples), $b$ is the input gains, and
$c$ is the output tap weights. For the loop to be stable and lossless, $M$ must be
unitary or near-unitary – verbx supports Hadamard matrices (default) and
random orthogonal matrices. The Hadamard construction at order $N$ is efficient
($\mathcal{O}(N \log N)$ multiply) and guarantees perfect echo density growth.

Delay line lengths are chosen to be mutually coprime (no shared factors) to
maximize echo density. The default `fdn_lines = 8` gives 8 delay lines; going
to 16 significantly increases diffusion at the cost of memory and compute.

The FDN includes optional time-varying modulation (`fdn_tv_rate_hz`,
`fdn_tv_depth`): each delay line length is sinusoidally modulated at an
individually offset rate derived from the master rate. This breaks up metallic
resonances at the cost of LTI purity – the output is no longer strictly an IR
of a fixed system, but the artifact is perceptually benign at low modulation
depths.

Three-band RT60 control (`fdn_rt60_low`, `fdn_rt60_mid`, `fdn_rt60_high`) is
implemented as per-band absorption shelving filters inserted in each feedback
loop – similar in spirit to the Jot-Chaigne absorptive FDN. Low-frequency RT60
is typically longer than mid in real rooms; setting `fdn_rt60_low` slightly
above `fdn_rt60_mid` makes the tail feel more physical.

Graph topology options (`ring`, `random`) control how the $N$ delay lines
interconnect before the full mixing matrix is applied. Ring is the classic
single-loop structure; random graph with degree $k$ introduces $k$ cross-connections
per node and increases early diffusion at the cost of slightly less predictable
echo pattern.

The cascade FDN option (`fdn_cascade`) chains two FDN stages at different delay
scales, with the second stage fed by the first at a ratio of `fdn_cascade_delay_scale`.
This is loosely analogous to Griesinger's nested FDN and significantly increases
low-frequency density without proportionally increasing the delay buffer size.

**Known limitation:** The FDN IR is rendered by impulse-exciting the network and
reading $N \times L$ output samples, where $L$ is the requested IR length. For very long IRs (120s at 48kHz)
this is about 5.76e9 operations – non-trivial. Sparse mode (`fdn_sparse`) reduces
this by zeroing low-magnitude delay connections, with controllable degree.

---

### 2b. `stochastic` – Noise-Shaped Diffuse Tail

#### What it sounds like

Pure reverb wash with no discrete echoes and no tonal coloration. Sounds like
the late tail of a large space – a long, smooth decay. Useful as a bed when you
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

$$
h[n] = \operatorname{noise}[n] \, e^{-n/(f_s \tau)}
$$

where $\tau = \frac{T_{60}}{\ln(10^6)}$. The `diffusion` parameter controls a frequency-
domain shaping pass: values above 0.5 increasingly smooth the spectral envelope
toward white (flat), while lower values allow more spectral irregularity to
persist, giving a slightly colored tail that can sound more like a real room.

The `density` parameter scales the RMS of the noise before the envelope is
applied. It does not change the decay shape, only the apparent "thickness" of
the tail. At density > 1.0 the tail can clip before normalization, which is
intentional – normalization re-scales – but very high density values will pull
the effective noise floor up and reduce dynamic range in the output.

Band control (`rt60_low`, `rt60_high`): when these differ, the stochastic tail
interpolates between two exponential envelopes across frequency using a
crossover filter. This approximates frequency-dependent absorption without a
full filter-per-sample approach.

---

### 2c. `modal` – Decaying Modal Resonator Bank

#### What it sounds like

Pitched, tonal, resonant – like a metal plate, a wine glass, or a very live
tiled room. Modes ring at specific frequencies, giving reverb a musical quality
that can either blend into pitched content or clash with it depending on tuning.
Excellent for designed reverbs on melodic material when you want the space to
have harmonic relationship to the source.

#### How it works

Each mode is an exponentially decaying sinusoid:

$$
h_k[n] = A_k \sin\!\left(2\pi f_k n/f_s + \phi_k\right) e^{-n/\tau_k}
$$

with amplitude $A_k$, frequency $f_k$, phase $\phi_k$, and time constant $\tau_k$. The
bank sums $N$ such modes:

$$
h[n] = \frac{1}{\sqrt{N}} \sum_{k=1}^{N} h_k[n] + \epsilon \,\operatorname{noise\_bed}[n]
$$

The $1/\sqrt{N}$ normalization keeps total energy approximately constant regardless
of modal count, which is the right thing to do (amplitude does not add linearly
for uncorrelated modes – RMS does).

Frequencies are sampled log-uniformly between `modal_low_hz` and `modal_high_hz`:

$$
f_k = \exp\!\left(U[\log(f_{\mathrm{low}}), \log(f_{\mathrm{high}})]\right)
$$

This is the correct distribution for musical pitch perception – equal intervals
on a log scale correspond to equal musical intervals. Linear uniform sampling
would cluster modes near `modal_high_hz` perceptually.

The $Q$-to-$\tau$ relationship follows standard resonator theory:

$$
\tau_q = \frac{Q}{\pi f_k}
$$

but verbx blends this with the global RT60 target:

$$
\begin{aligned}
\tau_{\mathrm{RT}} &= \frac{T_{60}}{6.91} \\
\tau_k &= \min\!\left(\max(\tau_q, 0.01), \max(2\tau_{\mathrm{RT}}, 0.03)\right)
\end{aligned}
$$

This clipping prevents individual modes from ringing dramatically longer than
the RT60 budget, which would otherwise create audible individual tones at the
end of the tail – a problem that sounds terrible and is hard to fix after the
fact. In practice you can push `modal_q_max` to 200 or higher if you want
those long-ringing tones intentionally (bells, chimes), just know the IR will
have frequency-domain spikes that may interact poorly with the source material
unless tuned deliberately.

Harmonic alignment (`--f0`, `--analyze-input`) nudges mode frequencies toward
integer multiples of a fundamental using `tune_frequency_to_targets()`. The
`align_strength` parameter (0–1) linearly interpolates each mode's sampled
frequency toward the nearest harmonic target. At strength 1.0, all modes
collapse to exact harmonics – a fully pitched comb. At 0.0, no alignment.
Values around 0.5–0.7 give a nice blend of harmonic coloration without the
artificiality of a pure comb filter.

#### Scala scales and microtonal emphasis

`--scala-file SCALE.scl` replaces the ordinary harmonic target list with pitch
classes parsed from a Scala scale. verbx accepts cents entries such as `63.1579`,
integer ratios such as `2`, and fractional ratios such as `7/4`. Lines beginning
with `!`, and text following `!` on a data line, are comments. The final declared
pitch defines the repeat interval, so octave, tritave, and other non-octave
scales use the same path.

Let $r_d$ be the ratio of scale degree $d$, $r_p$ the repeat interval, $f_r$ the
root frequency, and $d_r$ the selected root degree. The resolved frequency for
degree $d$ in register $k$ is

$$
f_{d,k} = f_r\frac{r_d}{r_{d_r}}r_p^k.
$$

verbx expands this lattice only between `--scala-low-hz` and
`--scala-high-hz`, then removes targets above 0.49 times the sample rate. If a
high-division scale exceeds `--scala-max-targets`, targets are selected at even
positions on the logarithmically ordered list so low and high registers remain
represented. This is a deterministic DSP budget, not a random pitch omission.

The resolved frequencies serve two related purposes. First, modal frequencies
are pulled toward the nearest scale target by `--scala-strength`. Second, after
the selected synthesis mode is complete, a parallel constant-Q filter bank
reinforces those bands in every mode. For a center frequency $f_c$ and bandwidth
$b$ cents, the approximate filter quality is

$$
Q = \frac{f_c}{f_c\left(2^{b/2400}-2^{-b/2400}\right)}.
$$

The filtered bank is RMS-normalized before `--scala-gain-db` and
`--scala-strength` are applied. Final IR normalization still runs afterward,
which bounds export headroom but does not erase the relative spectral emphasis.
Narrow widths produce clearly pitched resonances; broader widths produce a
gentler spectral affinity. Start at 20 to 35 cents, 3 to 6 dB, and strength 0.5
to 0.75 for musical material.

| Option | Default | Meaning |
|---|---:|---|
| `--scala-file` | none | UTF-8 Scala `.scl` file |
| `--scala-root-hz` | 440 Hz | Frequency assigned to the selected root degree |
| `--scala-root-degree` | 0 | Zero-based degree treated as the root |
| `--scala-low-hz` | modal low limit | Lowest generated target |
| `--scala-high-hz` | modal high limit | Highest target before Nyquist clamping |
| `--scala-strength` | 1.0 | Modal attraction and emphasis blend, 0 to 1 |
| `--scala-bandwidth-cents` | 25 cents | Constant-Q emphasis width |
| `--scala-gain-db` | 4 dB | Emphasis-bank gain before final normalization |
| `--scala-max-targets` | 128 | Maximum filter and tuning target count |

The parser rejects malformed counts, nonpositive ratios, unsorted degrees,
invalid root degrees, and empty post-Nyquist ranges before synthesis. Scala
tuning cannot be combined with `--analyze-input` or `--f0` because those options
would define a competing frequency target set. Use `--scala-root-hz` instead.

For complete import examples and a musical treatment of consonance through
time, root mapping, register, scale cardinality, non-octave periods,
transposition, harmonic rhythm, orchestration, spatialization, and changing
harmony, continue with [Microtonal Workflows, Scala Import, and Scale-Tuned
Reverberation](MICROTONAL_SCALA_WORKFLOWS.md).

An air noise bed at amplitude $0.02\,r_{\mathrm{RMS}}$ is added to prevent the IR from
being spectrally empty between modal peaks. Without it, convolution with
broadband content can reveal the gaps as spectral dips.

**Stereo:** each mode is panned independently using equal-power panning:
$L = \sqrt{0.5(1-p)}$ and $R = \sqrt{0.5(1+p)}$, where $p$ is pan position.
For $C>2$ channels, additional channels receive the mono sum scaled by $1/C$. Not ideal for
surround, but functional for stereo and tolerable for LCR.

---

### 2d. `hybrid` – The Mode You Will Actually Use

#### What it sounds like

Combines discrete early echoes (the "slap" you hear in the first 50–120ms)
with a blended late tail. The early reflections give spatial cues – your brain
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

$$
h_{\mathrm{tail}} = 0.55\,h_{\mathrm{stoch}} + 0.25\,h_{\mathrm{modal}} + 0.20\,h_{\mathrm{FDN}}
$$

4. Add early reflections by overlap-add into the first er_max_delay_ms samples

The seed offsets (+11, +17, +23) are not magic numbers – they just ensure the
three RNGs produce independent sequences from the same user seed. The blending
weights (0.55/0.25/0.20) were chosen empirically: stochastic dominates because
it produces the smoothest late tail, modal adds tonal character, FDN adds
algorithmic density structure. These weights are fixed in the current release;
future versions may expose them as parameters.

The post-blend pipeline applies globally: harmonic alignment, Scala constant-Q
emphasis, optional Modalys resonator layer, then IR shaping (filters,
normalization). Tuning flags such as `--f0`, `--scala-file`, and `--resonator`
therefore work across all modes, not just modal.

---

## 3. Early Reflections

### Perceptual Role

Early reflections – the first 5–80ms or so after the direct sound – are the
most important part of reverb for spatial perception. They are what lets you
tell a bathroom from a concert hall even before the tail arrives. They carry
IACC (interaural cross-correlation) cues that localize you in space, and their
density and timing pattern gives a strong sense of room size.

The direct sound is always present at sample 0 (amplitude 1.0 per channel) as
a convolution anchor – this is the Kronecker delta that initiates the impulse
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

$$
d_k \sim U\!\left[1, \left\lfloor \frac{t_{\max} f_s}{1000} \right\rfloor \right]
$$

Amplitude follows one of three decay laws indexed by `er_decay_shape`:

$$
\begin{aligned}
\text{linear:}\quad & amp = 1 - \left(\frac{\mathrm{delay}}{\mathrm{max\_delay}}\right) \\
\text{sqrt:}\quad & amp = \sqrt{\max\!\left(10^{-6}, 1 - \frac{\mathrm{delay}}{\mathrm{max\_delay}}\right)} \\
\text{exp:}\quad & amp = \exp\!\left(-3.2 \frac{\mathrm{delay}}{\mathrm{max\_delay}}\right)
\end{aligned}
$$

Here $d_k$ is tap $k$'s delay in samples, $t_{\max}$ is the `er_max_delay_ms`
control, and $f_s$ is sample rate in hertz. The exponential law (default)
corresponds to a room with absorption coefficient
around 0.2 uniformly distributed, which is plausible for a moderately live space.
The sqrt law gives a slower initial roll-off and is useful for very live spaces
like stairwells. Linear is rarely what you want but is there for completeness.

The raw decay amplitude is then multiplied by a uniform random jitter in
[0.35, 1.0] and by `er_room` (clipped to [0.1, 3.0]):

$$
a_k = q(d_k)\,r_{\mathrm{ER}}\,U[0.35, 1.0]
$$

Here $a_k$ is tap amplitude, $q(d_k)$ is the selected decay law, and
$r_{\mathrm{ER}}$ is the `er_room` value. The `er_room` parameter is loosely a room-size proxy: values > 1.0 increase
reflection amplitude, making the space feel more reverberant; values < 1.0
create a drier early field.

Stereo width is applied as pan spread per tap. Pan $p_k$ is sampled from
$U[\text{–}1, 1]$ and
scaled by `er_stereo_width`. Left/right amplitudes use a simple linear pan law
(not equal-power) because the taps are sparse enough that the distinction
matters less than in dense diffuse material:

$$
\begin{aligned}
L_k &= 0.5\,a_k\left(2 - \max(0, p_k)\right) \\
R_k &= 0.5\,a_k\left(2 + \min(0, p_k)\right)
\end{aligned}
$$

For er_stereo_width = 0.0, all taps are center-panned (mono ER). For 1.0,
full ±1.0 pan range. For 2.0, panning can exceed the unit range, intentionally
widening beyond the speaker base – useful for envelopment effects but beware
mono compatibility.

One thing I would change in a future version: the tap delays should ideally be
constrained to avoid clustering (e.g., via Poisson process sampling or minimum
inter-tap spacing). Uniform sampling occasionally produces bunches of taps
at similar delays that create audible comb-filter artifacts on transient material.
For now, setting `er_count` to 16–24 rather than 64 reduces this risk.

---

## 4. IR Morphing

### What It Does

Morphing takes two existing IRs – say, a small room and a large hall – and
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

$$
h_{\mathrm{out}}[n] = (1-\alpha) h_A[n] + \alpha h_B[n]
$$

Fast but acoustically naive. If $A$ and $B$ have different RT60 values, the
crossfaded IR will have a decay shape that is neither $A$'s nor $B$'s – it will
be a linear interpolation of amplitudes, which for exponential decays means
the instantaneous decay rate changes non-monotonically with $\alpha$. At
$\alpha=0.5$ between a 1s and 4s RT60, you get approximately a 2s decay, but the shape
has a kink where the two envelopes cross. For subtle blends ($\alpha<0.2$
or $\alpha>0.8$) this is usually fine.

**`spectral`**

Blend in the frequency domain:

$$
H_{\mathrm{out}}[k] = (1-\alpha) H_A[k] + \alpha H_B[k]
$$

where $H_A$ and $H_B$ are the FFTs of the respective IRs (zero-padded to avoid
circular convolution artifacts on the way back). This preserves the spectral
shape of each source more faithfully than time-domain blending. Particularly
useful when the two IRs have different frequency-dependent coloration (one
bright, one dark) and you want to smoothly interpolate that color.

Phase handling: the implementation blends complex spectra, not magnitude-only.
If $A$ and $B$ have substantially different phase responses (which real IRs almost
always do), the interpolated phase can produce notches in the output. The
`--align-decay` flag pre-aligns the energy onset of both IRs before blending,
which partially mitigates this.

**`envelope-aware`**

The most sophisticated option and the one I would recommend as a starting point
for serious work. It estimates the temporal energy envelope of each IR, blends
the envelopes separately, then re-modulates the blended spectral content. This
is loosely analogous to the approach in Cross-Synthesis of exponential decays.

Specifically:

1. Compute short-time energy envelopes $E_A[t]$ and $E_B[t]$ via squared Hilbert
   analytic signal or RMS-over-window (current implementation uses the latter
   at 10ms windows for speed)
2. Interpolate envelopes:

   $$
   E_{\mathrm{out}}[t] = (1-\alpha) E_A[t] + \alpha E_B[t]
   $$

3. Generate late tail by blending at the spectral level with envelope correction:
   scale $H_{\mathrm{out}}[k]$ such that the resulting time-domain signal matches
   $E_{\mathrm{out}}[t]$

Morph quality diagnostics are reported in the metadata: RT60 drift
($|T_{60,\mathrm{out}}-T_{60,\mathrm{expected}}|$ in seconds) and spectral distance
(log-spectral deviation in dB between $H_{\mathrm{out}}$ and the target interpolated spectrum). Use these
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
a file read – negligible latency.

### How the Hash Key Works

The cache key is a 16-character hex prefix of SHA-256 applied to the
JSON-serialized `IRGenConfig` dataclass:

    payload = asdict(config)
    payload["_schema"] = "verbx-ir-v0.5"
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    key = sha256(text.encode("utf-8")).hexdigest()[:16]

Every field of `IRGenConfig` contributes to the hash – `mode`, `length`, `sr`,
`channels`, `seed`, `rt60`, all FDN topology flags, all modal parameters, all
resonator settings, tuning, normalization target, everything. Changing any
single field produces a different hash and therefore a different cache entry.

The schema version string `"verbx-ir-v0.5"` is a namespace guard. When the
IRGenConfig schema changes in a breaking way (field added, removed, or renamed),
bump this version string. All existing cache entries will become unreachable
(not deleted, just miss) and will be regenerated on next access. This is
intentional: it avoids silent stale-cache bugs at the cost of some wasted disk
space during version transitions.

Cache files live at:

    .verbx_cache/irs/<hash>.wav
    .verbx_cache/irs/<hash>.meta.json

The metadata JSON contains the full config, version, seed, and IR metrics
(RT60 measured, spectral centroid, etc.) as computed by `analyze_ir()`. For a
Scala-tuned IR it also preserves the source filename, description, SHA-256
content hash, root frequency and degree, all resolved targets, filter width,
gain, strength, and target budget. The source hash is part of the cache identity,
so editing a scale in place cannot silently reuse the previous IR.

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
gentle early reflections. Peak normalization at –1 dBFS leaves headroom for
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

### Pitched Modal – Tuned Resonance

For pitched instruments where you want the room to harmonize with the source.
Set `--f0` to the root note of your key. High $Q$ range (7–90) gives a mix of
short and long-ringing modes. Increase `--modal-count` for denser resonance.

```bash
hatch run verbx ir gen irs/pitched_modal.wav \
  --mode modal --length 45 --rt60 35 --seed 99 --tuning A4=432 \
  --modal-count 64 --modal-q-min 7 --modal-q-max 90 \
  --modal-low-hz 60 --modal-high-hz 9000 --modal-spread-cents 8
```

### Microtonal Hybrid from a Scala Scale

Scale-tuned reverberation can range from a nearly invisible spectral affinity
to an exposed bank of sympathetic resonances. The examples below keep Scala
import offline and deterministic: each command generates an ordinary WAV IR
that can be auditioned through convolution, reused in a DAW, or archived with
its metadata sidecar. They also demonstrate why scale, root, degree mapping,
bandwidth, register, synthesis mode, and source material must be considered as
one musical design.

The included scale library provides three contrasting starting points:

- `19edo.scl` divides the octave into 19 equal steps.
- `5_limit_major.scl` uses a seven-degree, 5-limit just-intonation collection.
- `bohlen_pierce_13edo.scl` divides the $3/1$ tritave into 13 equal steps rather
  than repeating at the octave.

#### Example 1: A subtle 19-EDO harmonic halo

This first recipe maps degree 0 of 19-EDO to 220 Hz and expands the lattice
through the useful audio range. Moderate strength, 36-cent bandwidth, and 3 dB
of emphasis make the tuning more likely to be heard as room color than as a
separate resonator. The hybrid mode retains diffuse energy between scale bands,
which lets chromatic and continuously pitched source material move without
feeling quantized.

```bash
mkdir -p irs renders

hatch run verbx ir gen irs/19edo_halo.wav \
  --mode hybrid --length 18 --rt60 7 --seed 19 \
  --scala-file examples/scales/19edo.scl \
  --scala-root-hz 220 --scala-root-degree 0 \
  --scala-low-hz 140 --scala-high-hz 9000 \
  --scala-strength 0.48 --scala-bandwidth-cents 36 \
  --scala-gain-db 3 --scala-max-targets 128

hatch run verbx render examples/audio/realistic_music_dry.wav \
  renders/19edo_halo.wav --engine conv --ir irs/19edo_halo.wav \
  --wet 0.28 --dry 1
```

Listen first at the stated wet level, then solo the wet return. In context, the
tail should feel unusually coherent without producing obvious isolated notes.
Soloed, the 19-step lattice should be easier to hear in sustained harmonics and
broadband attacks. Raising `--scala-low-hz` keeps the reverb from inventing a
strong bass pedal beneath harmonically mobile music.

#### Example 2: Exposed 5-limit just-intonation resonance

The next design uses a sparse 5-limit scale and modal synthesis. Ratios such as
$5/4$, $3/2$, and $5/3$ generate exact frequency relationships to the mapped
root. Narrower 14-cent bands, stronger attraction, and a lower target ceiling
turn the IR into an audible resonant object. Dry clicks and percussion are good
test sources because their broad attacks excite many modes at once.

```bash
hatch run verbx ir gen irs/just_modal_A3.wav \
  --mode modal --length 12 --rt60 8 --seed 51 \
  --modal-count 72 --modal-q-min 18 --modal-q-max 110 \
  --scala-file examples/scales/5_limit_major.scl \
  --scala-root-hz 220 --scala-root-degree 0 \
  --scala-low-hz 80 --scala-high-hz 5000 \
  --scala-strength 0.88 --scala-bandwidth-cents 14 \
  --scala-gain-db 8 --scala-max-targets 96

hatch run verbx render examples/audio/dry_click.wav \
  renders/just_modal_click.wav --engine conv \
  --ir irs/just_modal_A3.wav --wet 1 --dry 0

hatch run verbx render examples/audio/realistic_drums_dry.wav \
  renders/just_modal_drums.wav --engine conv \
  --ir irs/just_modal_A3.wav --wet 0.42 --dry 1
```

On the click, identify the lowest stable pitch and then follow upper modes as
they decay at different rates. On drums, listen for a pitched residue after the
transient. If the result sounds like an added chord rather than reverberation,
reduce strength toward 0.65, increase bandwidth toward 24 cents, or change from
`modal` to `hybrid`. If the kick drum pulls the perceived root too strongly,
raise `--scala-low-hz` instead of removing low frequencies from the source.

#### Example 3: A non-octave Bohlen–Pierce field

Bohlen–Pierce tuning repeats at the $3/1$ tritave. Its scale degrees therefore
do not return to equivalent pitch classes at each octave. This example uses a
hybrid tail, a restrained lower limit, and a 7.5-second RT60 so the non-octave
relationships can overlap without creating an indefinitely accumulating drone.

```bash
hatch run verbx ir gen irs/bohlen_pierce_space.wav \
  --mode hybrid --length 16 --rt60 7.5 --seed 313 \
  --scala-file examples/scales/bohlen_pierce_13edo.scl \
  --scala-root-hz 130.8128 --scala-root-degree 0 \
  --scala-low-hz 90 --scala-high-hz 7800 \
  --scala-strength 0.72 --scala-bandwidth-cents 26 \
  --scala-gain-db 5 --scala-max-targets 104

hatch run verbx render examples/audio/realistic_speech_dry.wav \
  renders/bohlen_pierce_speech.wav --engine conv \
  --ir irs/bohlen_pierce_space.wav --wet 0.34 --dry 1

hatch run verbx render examples/audio/realistic_music_dry.wav \
  renders/bohlen_pierce_music.wav --engine conv \
  --ir irs/bohlen_pierce_space.wav --wet 0.34 --dry 1
```

Speech reveals the scale as a changing coloration because vowels excite
different subsets of the lattice. Harmonic music reveals disagreement between
octave-repeating source partials and tritave-repeating resonances. Compare the
two renders at matched level: the same IR may read as timbre on speech and as a
second harmonic system on sustained music.

#### Example 4: Rotate the root degree without changing the reference pitch

`--scala-root-degree` decides which scale degree receives
`--scala-root-hz`. The following matched-seed pair assigns 220 Hz first to the
implicit $1/1$ and then to degree 5 of 19-EDO. Every non-tuning parameter is held
constant, so the comparison isolates degree rotation rather than random-tail
variation.

```bash
hatch run verbx ir gen irs/19edo_degree_0.wav \
  --mode hybrid --length 14 --rt60 6 --seed 1905 \
  --scala-file examples/scales/19edo.scl \
  --scala-root-hz 220 --scala-root-degree 0 \
  --scala-low-hz 100 --scala-high-hz 8000 \
  --scala-strength 0.68 --scala-bandwidth-cents 22 --scala-gain-db 5

hatch run verbx ir gen irs/19edo_degree_5.wav \
  --mode hybrid --length 14 --rt60 6 --seed 1905 \
  --scala-file examples/scales/19edo.scl \
  --scala-root-hz 220 --scala-root-degree 5 \
  --scala-low-hz 100 --scala-high-hz 8000 \
  --scala-strength 0.68 --scala-bandwidth-cents 22 --scala-gain-db 5

hatch run verbx render examples/audio/realistic_music_dry.wav \
  renders/19edo_degree_0.wav --engine conv --ir irs/19edo_degree_0.wav \
  --wet 0.38 --dry 1

hatch run verbx render examples/audio/realistic_music_dry.wav \
  renders/19edo_degree_5.wav --engine conv --ir irs/19edo_degree_5.wav \
  --wet 0.38 --dry 1
```

The second IR does not merely transpose the first by five ordinary semitones.
It maps a different degree ratio to the same physical reference frequency, then
reconstructs the complete lattice around that mapping. Compare sustained notes,
cadences, and silence after phrase endings. The changed relationship is often
most audible after the dry source stops.

#### Example 5: Separate tuning from random topology in an A/B test

A rigorous comparison needs an untuned control. Use the same mode, seed, length,
RT60, filters, and channel count, then change only the Scala options. The first
IR below establishes the control; the second adds a broad, moderate 5-limit
affinity.

```bash
hatch run verbx ir gen irs/control_seed_808.wav \
  --mode hybrid --length 20 --rt60 9 --seed 808 \
  --lowcut 70 --highcut 11000

hatch run verbx ir gen irs/just_seed_808.wav \
  --mode hybrid --length 20 --rt60 9 --seed 808 \
  --lowcut 70 --highcut 11000 \
  --scala-file examples/scales/5_limit_major.scl \
  --scala-root-hz 196 --scala-root-degree 0 \
  --scala-low-hz 100 --scala-high-hz 9000 \
  --scala-strength 0.55 --scala-bandwidth-cents 32 --scala-gain-db 4

hatch run verbx render examples/audio/realistic_music_dry.wav \
  renders/control_seed_808.wav --engine conv --ir irs/control_seed_808.wav \
  --wet 0.4 --dry 1

hatch run verbx render examples/audio/realistic_music_dry.wav \
  renders/just_seed_808.wav --engine conv --ir irs/just_seed_808.wav \
  --wet 0.4 --dry 1

hatch run verbx analyze renders/control_seed_808.wav \
  --json-out renders/control_seed_808.analysis.json

hatch run verbx analyze renders/just_seed_808.wav \
  --json-out renders/just_seed_808.analysis.json
```

Level-match before judging the pair. Use the JSON reports to check that a level
or decay difference is not masquerading as a tuning preference. Then describe
whether the scale-conditioned version changes perceived consonance, roughness,
brightness, pitch stability, or the boundary between source and room.

#### Example 6: Prepare a rooted bank for realtime or DAW use

A static convolution IR cannot follow chord symbols automatically, but a small
rooted library can be generated before a performance or mix. The command below
shows one member of such a bank. Repeat it with the same seed and different
root frequencies, preserving a filename that states the mapping.

```bash
hatch run verbx ir gen irs/ji_root_C3_130.8128.wav \
  --mode hybrid --length 10 --rt60 4.8 --seed 5150 \
  --scala-file examples/scales/5_limit_major.scl \
  --scala-root-hz 130.8128 --scala-root-degree 0 \
  --scala-low-hz 90 --scala-high-hz 7500 \
  --scala-strength 0.6 --scala-bandwidth-cents 28 --scala-gain-db 4

hatch run verbx realtime --engine conv \
  --ir irs/ji_root_C3_130.8128.wav --block-size 128
```

For section changes, overlap two wet returns and use an equal-power crossfade
long enough to preserve the old IR's early field. An abrupt replacement cuts
off the previous convolution state and can sound like a gate. In a DAW, place
each rooted IR on a separate return and automate sends or return gains. In an
offline CLI workflow, render sections with complete tails and assemble them
afterward.

The realtime latency is the same as for any other IR with the same partition
and block settings. Scala parsing, target expansion, and filter-bank
construction do not run on the callback thread. The metadata sidecar records
the scale hash and resolved targets, so an IR bank remains auditable even if two
source files happen to share a filename.

### Resonator-Colored Hybrid (Modalys-Inspired)

The Modalys resonator layer sits on top of the hybrid tail starting at
`--resonator-late-start-ms`. Good for adding physical resonance character
to an otherwise smooth hybrid – think the body resonance of a large wooden
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
a small room is the defining character – do not cut corners here. Low `--rt60`
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
decay – it is the signature of mass and volume. Skipping this makes algorithmic
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

Not a room at all – a physical resonator with well-defined modal structure.
The modal mode is the right tool here, not hybrid.

```bash
--mode modal --length 8 --rt60 5 \
--modal-count 96 --modal-q-min 40 --modal-q-max 400 \
--modal-low-hz 200 --modal-high-hz 16000 \
--modal-spread-cents 3 --f0 "220 Hz"
```

High $Q$ values (40–400) give the long, singing ring of plate reverb. The very
narrow `--modal-spread-cents 3` keeps modes tightly clustered around harmonic
targets for a more pitched character. Lower $Q$ (5–15) at wider spread approaches
spring reverb.

### Infinite / Frozen Pad

Not physically realistic, obviously, but useful as a creative tool – the IR
never really decays. Very long length with high RT60 relative to length.

```bash
--mode stochastic --length 120 --rt60 180 \
--diffusion 0.85 --density 0.8 --damping 0.15
```

The RT60 exceeding the IR length means the decay envelope never reaches –60 dB
within the buffer – the tail is essentially flat. Combined with high diffusion
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
for a time-varying FDN – you cannot analytically compute its IR without running
the network. The cache makes this practical for production use, but the first
generation of a long FDN IR takes wall-clock time proportional to IR length.

**Channel count > 2.** The modal panner for channels > 2 distributes all
extra channels as mono sum / ch, which is not proper ambisonic or surround
panning. Do not use verbx-synthesized IRs for surround production without
verifying the channel assignments match your downstream decoder expectations.

### Roadmap Alignment (v0.7.5)

From the R3 milestones:

- `R3.1 cache determinism`: cross-sample-rate cache lookup with canonical
  resampling. An IR generated at 96kHz should be retrievable (with a quality
  warning) by a 48kHz render job without full regeneration.

- `R3.2 operational QA`: morph diagnostic artifacts – small reference renders
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


- Gardner, W. G. (1992). A realtime multichannel room simulation system.
  JASA.
- Griesinger, D. (1996). Spaciousness and envelopment in musical acoustics.
  AES 101st Convention.
- Jot, J.-M. & Chaigne, A. (1991). Digital delay networks for designing
  artificial reverberators. AES 90th Convention.
- Schlecht, S. J. & Habets, E. A. P. (2017). On lossless feedback delay
  networks. IEEE Transactions on Signal Processing.
- Valimaki, V. et al. (2012). Fifty years of artificial reverberation.
  IEEE Transactions on Audio, Speech, and Language Processing.


See `docs/REFERENCES.md` for full citation list including Modalys and
convolution engine references.
