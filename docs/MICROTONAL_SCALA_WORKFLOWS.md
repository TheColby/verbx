# Microtonal Workflows, Scala Import, and Scale-Tuned Reverberation

A conventional reverb is usually designed to avoid obvious pitch. Its modes
are dense, irregular, or modulated so the tail supports many notes without
advertising a key. Scale-tuned reverberation takes a different position: the
decay field may have a harmonic vocabulary of its own. Resonances can reinforce
the tuning of a work, create a stable spectral halo around selected degrees, or
make out-of-scale material sound deliberately tense.

verbx imports Scala `.scl` files during synthetic IR generation. It expands the
scale over a bounded frequency range, uses the resolved frequencies as modal
targets, and applies a controlled constant-Q emphasis bank. The resulting WAV
is an ordinary impulse response. It can therefore be used offline, in verbx
realtime convolution, in a DAW convolution plug-in, or in a reproducible audio
dataset without parsing Scala data in the audio callback.

This is **scale-conditioned reverberation**, not pitch correction. It does not
detect each incoming note, retune the source, or change the IR as harmony moves.
The fixed IR behaves more like a sympathetic resonator whose preferred
frequencies were derived from a tuning system.

## What a Scala File Describes

The Scala scale format is a compact text representation of one repeating pitch
collection. Blank lines and text beginning with `!` are ignored. The first
content line is a description, the second is the number of following pitch
entries, and the remaining lines define strictly increasing degrees above an
implicit ratio of $1/1$.

An entry containing a decimal point is interpreted as cents. An integer is a
whole-number ratio, and a fraction is an explicit ratio. The final entry is the
period of repetition. It is often $2/1$, or 1200 cents, but it can instead be a
tritave, stretched octave, or another interval.

```text
! 19edo.scl
19-tone equal temperament
19
63.1578947
126.3157895
189.4736842
252.6315789
315.7894737
378.9473684
442.1052632
505.2631579
568.4210526
631.5789474
694.7368421
757.8947368
821.0526316
884.2105263
947.3684211
1010.5263158
1073.6842105
1136.8421053
2/1
```

Scala keyboard mappings use the separate `.kbm` format. verbx currently reads
`.scl`, not `.kbm`. Root mapping is supplied explicitly with
`--scala-root-hz` and `--scala-root-degree`, which keeps the generated IR
independent of MIDI note numbering.

## How verbx Maps Degrees to Frequencies

Let $r_d$ be the ratio for degree $d$, $r_p$ the repeat-period ratio, $f_r$ the
frequency assigned to the selected root, and $d_r$ the selected root degree.
For register index $k$, verbx resolves

$$
f_{d,k}=f_r\frac{r_d}{r_{d_r}}r_p^k.
$$

Only targets between `--scala-low-hz` and `--scala-high-hz` are retained, and
targets too close to Nyquist are rejected. If a high-division scale creates
more targets than `--scala-max-targets`, verbx samples evenly through the
logarithmically ordered list. This preserves low-to-high coverage while bounding
the number of filters and modal attractors.

`--scala-root-degree` is zero based and refers to the implicit unison plus the
listed degrees before the final period. Degree zero is therefore the implicit
$1/1$. Assigning 220 Hz to degree zero places the unison lattice on 220 Hz;
assigning 220 Hz to another degree rotates the same scale around that reference.

## Tuning Mathematics for Reverb Designers

Microtonal work becomes easier when ratios, cents, and frequencies are treated
as different views of the same interval. A ratio $r$ converts to cents by

$$
c(r)=1200\log_2(r),
$$

and a cent value $c$ converts back to a ratio by

$$
r(c)=2^{c/1200}.
$$

Equal divisions of a period use one repeated step. If a period ratio $r_p$ is
divided into $N$ equal parts, degree $d$ has ratio

$$
r_d=r_p^{d/N}.
$$

For ordinary $N$-EDO, $r_p=2$; for 13 equal divisions of the Bohlen–Pierce
tritave, $r_p=3$. The distinction is structural. An octave-period scale repeats
its pitch classes when frequency doubles, while a tritave-period scale repeats
when frequency triples. Convolution exposes that difference because every
source partial encounters a fixed lattice extending through many registers.

Just-intonation scales specify small-integer ratios instead of equal steps. A
$3/2$ perfect fifth above 220 Hz resolves to 330 Hz, while a $5/4$ major third
resolves to 275 Hz. Those exact values matter when tuned resonances are narrow.
If a source partial lies at frequency $f_s$ and the corresponding IR resonance
lies at $f_r$, their beating rate is approximately

$$
f_b=\lvert f_s-f_r\rvert.
$$

A 1 Hz difference produces a slow pulse; a 12 Hz difference can produce
obvious roughness; a much larger difference may be heard as separate spectral
components. The perceptual boundary is frequency dependent and affected by
level, duration, masking, and the spectra of both source and return. This is why
one fixed cent tolerance cannot describe every register equally well.

### Bandwidth, quality factor, and selectivity

verbx expresses resonance width in cents because that is musically portable
across frequency. If the full bandwidth is $b$ cents around center frequency
$f_c$, the approximate linear bandwidth is

$$
\Delta f=f_c\left(2^{b/2400}-2^{-b/2400}\right),
$$

and the corresponding quality factor is

$$
Q=\frac{f_c}{\Delta f}.
$$

The same cent width therefore occupies more hertz in the treble than in the
bass while preserving approximately equal pitch selectivity. Narrow bands
produce long, identifiable ringing and reveal small tuning differences. Broad
bands tolerate vibrato, bends, ensemble intonation, and detuned partials, but
they also reduce the identity of individual scale degrees.

Bandwidth should be chosen with the source, not in isolation. A fixed-pitch
synthesizer can support narrow bands; a choir, string ensemble, or analog
oscillator often benefits from a wider field. Percussive noise may excite every
band strongly enough that even moderate widths create an unmistakably pitched
tail.

## A Signal Model for Scale-Conditioned Decay

For input $x[n]$ and IR $h[n]$, convolution produces

$$
y[n]=\sum_m x[m]h[n-m].
$$

A useful conceptual decomposition writes the tuned IR as

$$
h[n]=(1-\alpha)h_u[n]+\alpha h_s[n],
$$

where $h_u[n]$ is the underlying untuned or diffuse response, $h_s[n]$ is the
scale-conditioned component, and $\alpha$ represents the effective tuning
strength. This equation is conceptual rather than an exact statement of every
internal normalization stage, but it explains the audible continuum. Near
$\alpha=0$, the result behaves like an ordinary room with slight color. Near
$\alpha=1$, the tail behaves more like a resonator bank.

The tuned component may be approximated as a sum of damped modes:

$$
h_s[n]=\sum_{k=1}^{K} A_k e^{-n/\tau_k}
\sin\!\left(2\pi f_k n/f_s+\phi_k\right),
$$

where each $f_k$ is attracted toward a resolved Scala target. The input does
not activate every mode equally. In the frequency domain,

$$
Y(e^{j\omega})=X(e^{j\omega})H(e^{j\omega}),
$$

so a resonance becomes prominent only when the source supplies energy near it.
This explains why the same IR can sound gently colored on one instrument and
strongly pitched on another.

The tail also acts as a memory. If note $i$ ends at time $t_i$ with decay
constant $\tau_i$, its residual amplitude at later time $t$ is proportional to

$$
a_i(t)=a_i(t_i)e^{-(t-t_i)/\tau_i}.
$$

The harmony heard at time $t$ therefore includes the present source plus
weighted remnants of earlier events. Tuning and RT60 cannot be separated
musically: the scale selects what persists, while decay time determines how
many prior events remain available to interact.

## Psychoacoustics of Tuned Reverberation

### Fusion, beating, and roughness

Two nearby components may fuse into one fluctuating tone, produce audible
beats, or separate into distinct pitches. Tuned reverb can place the dry source
and wet resonance on either side of those perceptual regimes. Narrow bands and
high wet level emphasize discrepancies; broad bands and diffuse energy promote
fusion. Slow beating can sound like life or motion, while dense beating across
many partials can make a tail grainy, unstable, or harsh.

The relationship changes over time. During the attack, the direct source often
masks the return. After the source stops, the resonance is exposed and its
pitch may appear to “resolve” even though the system is linear and time
invariant. This temporal unmasking is one reason tuned reverberation can affect
phrase endings more strongly than note beginnings.

### Auditory masking and register

Strong source energy can hide nearby wet resonances during a phrase. A sparse
arrangement or a rest reveals them. Low resonances need little gain to imply a
pedal because the bass contains fewer competing modes and anchors pitch
strongly. Midrange resonances interact with fundamentals and low partials, so
they most readily alter perceived harmony. High resonances often read as sheen,
metal, or air unless they form a sparse and sustained pattern.

Masking also explains why soloing the wet return is necessary but insufficient.
The solo reveals the IR's vocabulary; the full mix reveals whether that
vocabulary survives, supports, or interferes with the source. Decisions should
be made in both conditions at matched playback level.

### Pitch salience versus spatial plausibility

Ordinary room reverberation contains modes, but a dense, irregular distribution
usually prevents one scale from dominating. Pulling modes toward a pitch
lattice increases pitch salience and can reduce the impression of a neutral
physical room. There is no universal optimum. Production work may favor a
plausible space with subtle affinity, while installation, electroacoustic, or
experimental work may intentionally cross into an audible instrument.

Three controls move the result along this continuum:

1. Increase `--scala-strength` to align and blend more strongly.
2. Increase `--scala-gain-db` to expose the conditioned bands.
3. Decrease `--scala-bandwidth-cents` to make each band more selective.

Changing all three at once makes diagnosis difficult. Move one coordinate at a
time, level-match, and preserve the previous IR for comparison.

## How Each Synthesis Mode Expresses a Scale

The same `.scl` file does not produce the same musical behavior in every mode.

| Mode | Scale mechanism | Typical impression | Best first use |
|---|---|---|---|
| `fdn` | Conditioned bands color a recursively mixed late field | Smooth space with a persistent spectral center | General musical reverb that must remain cohesive |
| `stochastic` | Constant-Q emphasis shapes a noise-derived decay | Diffuse, breath-like spectrum with weak note identity | Vocals, speech, pads, and subtle alternate-tuning color |
| `modal` | Resonator frequencies are attracted toward scale targets, then emphasized | Clearly pitched object or sympathetic resonator | Percussion, impulses, drones, and exposed sound design |
| `hybrid` | Early field and multiple late families combine before conditioning | Balanced room cues plus audible harmonic affinity | Default composition and production workflow |

`modal` is the most revealing diagnostic mode because individual resonances are
easy to hear. It is also the easiest mode to overstate. `stochastic` can hide
scale identity when bands are broad or gain is low, but that restraint may be
exactly right for a vocal mix. `hybrid` is usually the safest starting point
because untuned energy fills gaps between targets and maintains a continuous
tail.

The selected mode should match the artistic metaphor. If the desired result is
“a room sympathetic to this scale,” begin with `hybrid` or `fdn`. If it is “an
instrument made from this scale,” begin with `modal`. If it is “air colored by
this scale,” begin with `stochastic`.

## Designing and Auditing a Scala File

A scale file is small enough to inspect manually, and it should be. Before
generation, verify:

1. The description identifies the scale and source clearly.
2. The declared count equals the number of pitch rows.
3. Every degree lies strictly above the implicit $1/1$ and below or at the
   final period.
4. Degrees are strictly increasing.
5. Decimal cents and ratios express the intended values.
6. The last entry is the intended repeat period, not accidentally the last
   pitch class below it.
7. The file is UTF-8 text and comments begin with `!`.

Do not infer tuning from a filename alone. A file called `just_major.scl` may
use a different seventh, omit a degree, or repeat at a nonstandard period. The
SHA-256 stored by verbx identifies the actual bytes used in synthesis.

For a new scale, begin with a narrow frequency range and print or inspect the
resolved metadata. Confirm that the selected root appears where expected and
that the highest targets remain below Nyquist. Then expand the range. This
staged procedure catches root-degree and period mistakes before a long IR is
rendered.

The bundled examples provide useful parser contrasts:

- `examples/scales/19edo.scl` uses cents and an octave period.
- `examples/scales/5_limit_major.scl` uses integer ratios and an octave period.
- `examples/scales/bohlen_pierce_13edo.scl` uses cents with a $3/1$ period.

They are demonstrations, not canonical definitions of every performance
practice associated with those tuning families.

## First Complete Workflow

Generate the tuned IR once, inspect it, and then use it in as many renders as
needed:

```bash
verbx ir gen irs/19edo_hybrid.wav \
  --mode hybrid --length 18 --rt60 7 --seed 19 \
  --scala-file examples/scales/19edo.scl \
  --scala-root-hz 220 --scala-root-degree 0 \
  --scala-low-hz 90 --scala-high-hz 10000 \
  --scala-strength 0.65 --scala-bandwidth-cents 22 \
  --scala-gain-db 5 --scala-max-targets 128

verbx ir analyze irs/19edo_hybrid.wav \
  --json-out irs/19edo_hybrid.analysis.json

verbx render source.wav tuned_space.wav \
  --engine conv --ir irs/19edo_hybrid.wav --wet 0.35 --dry 1
```

For low-latency auditioning, load the generated IR into realtime convolution:

```bash
verbx realtime \
  --engine conv --ir irs/19edo_hybrid.wav --block-size 128
```

Scala parsing, target expansion, and filter construction happen during `ir
gen`, not in the realtime callback. Realtime latency is therefore determined by
the audio device, host buffers, block size, safety buffering, and convolution
partitioning rather than the number of degrees in the source scale.

## The Controls as Musical Decisions

| Control | Musical question | Conservative start | More audible effect |
|---|---|---:|---:|
| `--scala-root-hz` | Where is the tuning lattice anchored? | Match the work's tuning reference | Shift by a scale degree or structural bass frequency |
| `--scala-root-degree` | Which degree receives the reference frequency? | `0` | Rotate the modal hierarchy around another degree |
| `--scala-low-hz` | Should bass modes participate? | 100 Hz | 35–70 Hz for drones and installation work |
| `--scala-high-hz` | How much brightness carries pitch identity? | 6–10 kHz | 12–16 kHz for bright, exposed resonances |
| `--scala-strength` | How strongly do modes approach the scale? | 0.45–0.70 | 0.85–1.00 for an unmistakably tuned object |
| `--scala-bandwidth-cents` | Are resonances selective or forgiving? | 20–35 cents | 5–15 cents for narrow ringing bands |
| `--scala-gain-db` | How far does the lattice emerge from the diffuse tail? | 3–6 dB | 8–12 dB for special effects |
| `--scala-max-targets` | How dense may the target lattice become? | 96–128 | More targets for high-division scales and wide ranges |

Strength and gain are related but not identical. Strength controls modal
attraction and the blend of the emphasis layer. Gain controls the level of the
filtered bank before final normalization. Bandwidth controls how much nearby
pitch motion is accepted. A narrow, strong setting sounds like a bank of
sympathetic strings; a broad, moderate setting sounds more like a room whose
color gently favors the scale.

## Musical Implications of a Tuned Decay Field

### Consonance becomes time dependent

A dry pitch can end while its scale-related energy remains in the tail. The
next note is therefore heard against a memory of previous notes. A melodic
interval that is locally consonant may produce roughness when the older decay
overlaps it, while a dissonant attack may resolve into a stable resonant field.
Scale-tuned reverb makes harmonic rhythm partly a function of RT60.

Longer decay is not merely “more” of the same tuning. It increases the number
of prior events simultaneously represented by the room. Slow music can expose
individual degrees; fast music can accumulate the complete scale into a
spectral aggregate. When testing a design, vary tempo before changing the scale
or declaring the tuning ineffective.

### The reverb can confirm or contradict the ensemble

When performers and IR share a tuning, resonant energy tends to reinforce
stable scale positions. Near misses, expressive intonation, vibrato, and pitch
bends pass through and may beat against narrow bands. This can make intonation
audible in a productive way, but it can also make a flexible vocal or string
line feel constrained.

Using a different scale in the IR creates a second harmonic layer. A
12-tone-equal-tempered source through a just-intonation IR may produce slow
beating around nominally equivalent intervals. A diatonic source through a
19-EDO IR can make chromatic inflections leave different spectral residues.
The result is not automatically dissonant; bandwidth, root, register, source
spectrum, and decay determine whether the difference reads as color, chorus,
or conflict.

### Register controls whether tuning is heard as harmony or timbre

Low-frequency targets are sparse and individually audible. They can imply a
fundamental, pedal, or room mode even when the source does not sustain that
pitch. Midrange targets interact strongly with musical fundamentals and lower
partials. High-frequency targets are more likely to be heard as sheen,
brightness, or metallic identity than as named pitch.

For a mix that must remain harmonically agile, begin the tuned range above the
bass and use broader bands. For an installation, drone, or resonant percussion
piece, extending the lattice downward can make the reverb function as an
instrumental voice.

### Scale cardinality changes texture

A five- or seven-degree scale produces a relatively sparse lattice and leaves
audible gaps. Equal divisions with 19, 31, or 53 degrees create increasing
spectral density. A larger pitch count does not guarantee a smoother tail:
narrow filters can still expose beating and local clusters, while the target
budget may omit some expanded degrees. Consider cardinality together with
bandwidth and range.

Non-octave scales are especially distinctive because spectral relationships do
not repeat at powers of two. A tritave-period scale can cause familiar octave
equivalence to drift across registers. This may be musically compelling on
bells, percussion, synthetic tones, and inharmonic sources, but it can feel
unstable on octave-doubled orchestration unless that instability is intended.

### The source spectrum determines which degrees awaken

An IR cannot emphasize energy that the source does not excite. A flute-like
tone may illuminate only a few bands; noise, cymbals, consonants, and distorted
sources can excite nearly the entire lattice. Transients reveal the IR itself,
while sustained tones reveal the interaction between source partials and tuned
resonances.

Test at least four source classes: an impulse or click, a chromatic sine sweep,
a sustained harmonic sound, and broadband musical material. This separates the
design of the resonator from the accident of one orchestration.

### Spatialization can distribute harmonic function

A stereo or multichannel design need not use one identical tuned field in every
channel. Related IRs can place complementary degree sets, roots, or bandwidths
around the listener. Keep enough common energy to maintain one space, and avoid
hard channel-specific peaks that disappear under downmixing. Treat each IR as
part of a spatial harmony, then check mono, stereo, binaural, and speaker-array
translations at matched level.

## Source-Aware Orchestration

The scale file defines available resonances, but orchestration determines which
ones receive energy. Designing the IR without considering the source is like
specifying sympathetic strings without asking what will excite them.

### Voice and choir

Vowels contain moving formants and a harmonic series whose exact frequencies
follow the sung fundamental. Consonants provide broadband transients that can
excite the entire tuned field. A narrow IR may reinforce stable vowels while
making consonant tails unexpectedly pitched. For solo voice, begin above the
lowest expected fundamental and use 25–45-cent bandwidth. For choir, broaden
further unless exposing intonation differences is part of the piece.

Test vibrato at the slowest and fastest expected rates. If the dry pitch moves
outside a narrow resonance while the tail remains fixed, the result can alternate
between reinforcement and attenuation. That may sound luminous, unstable, or
simply out of tune depending on context. A broad hybrid field generally follows
human pitch variation more gracefully than a high-$Q$ modal bank.

### Strings and winds

String spectra change with bow position, pressure, mute, register, and
articulation. Sul ponticello playing supplies abundant upper-partial energy and
can reveal high Scala targets dramatically; a soft fundamental-rich tone may
activate only a few low modes. Glissandi create a useful diagnostic because
resonances brighten as partials cross the fixed lattice.

Wind instruments differ in harmonic structure and noise content. Clarinet-like
odd-partial emphasis, flute-like breath energy, and brass brightness illuminate
different degree subsets even on the same written pitch. Test representative
registers independently. A tuning that flatters a low horn line may turn a high
reed texture into a dense whistle bank.

### Piano, harp, and fixed-pitch instruments

Fixed-pitch instruments make tuning differences easy to reproduce. They also
carry long natural decays, so instrument resonance and room resonance overlap.
Before increasing RT60, compare the dry decay with the wet-only decay. If the
source already sustains for several seconds, a shorter tuned IR may create more
clarity than a long one.

Pedal behavior matters. A sustained piano resonance can feed the reverb with
partials from many earlier notes, producing a two-stage memory: strings retain
the harmony, then the IR retains the strings. Sparse textures expose this
beautifully; dense pedal can turn even a seven-degree IR into broadband
accumulation.

### Percussion and impulsive excitation

Clicks, woodblocks, snare attacks, and mallet transients approximate broad
excitation and make the IR's pitch set audible. A modal Scala IR can transform
an unpitched attack into a pitched gesture. This is one of the clearest ways to
treat reverberation as orchestration rather than ambience.

The apparent pitch may depend more on modal amplitude and damping than on the
formal root. If one low target dominates, adjust `--scala-low-hz`, use a higher
root, or reduce modal emphasis before changing the scale itself. Preserve peak
headroom because simultaneous excitation of many resonances can create a large
wet transient followed by a deceptively quiet tail.

### Electronics, distortion, and noise

An ideal sine excites only a narrow part of the IR. Saturation or distortion
adds harmonics that can awaken many more degrees. If tuning seems absent on a
pure oscillator, compare the same line after gentle saturation. Conversely, if
a distorted source makes the field overwhelming, generate a version with a
lower high-frequency limit rather than merely reducing global wet level.

Noise reveals the transfer function almost directly. Filtered noise can audit
one register at a time, while full-band noise exposes the complete lattice.
Because noise is not harmonically committed, the tuned return may become the
primary pitch-bearing layer.

## Composition and Production Strategies

### Stable halo

Use the composition's tuning, moderate strength, 25–40-cent bandwidth, and a
high-pass-limited target range. The reverb supports pitch identity while the
dry source remains dominant. This is a practical starting point for vocals,
chamber textures, and tuned percussion.

### Harmonic shadow

Root the IR on a structural pitch that is absent from the current passage. As
the tail accumulates, it implies a latent pedal or tonal region. Automate the
wet level rather than switching IRs abruptly so the shadow enters as memory.

### Controlled contradiction

Generate two IRs from related but nonidentical scales. Render identical source
material through each and crossfade between the wet returns. Because an IR is
time invariant, the transition remains reproducible and can be shaped without
changing the dry performance.

```bash
verbx ir gen irs/scale_a.wav \
  --mode hybrid --length 14 --rt60 6 --seed 41 \
  --scala-file scales/scale_a.scl --scala-root-hz 220 \
  --scala-strength 0.6 --scala-bandwidth-cents 28

verbx ir gen irs/scale_b.wav \
  --mode hybrid --length 14 --rt60 6 --seed 41 \
  --scala-file scales/scale_b.scl --scala-root-hz 220 \
  --scala-strength 0.6 --scala-bandwidth-cents 28

verbx ir morph irs/scale_a.wav irs/scale_b.wav irs/transition.wav \
  --mode equal-power --alpha 0.5
```

Using the same seed and synthesis parameters isolates the scale as the main
experimental variable. Morphing the two final IRs produces a fixed intermediate
response; it does not create time-varying tuning inside one render.

### Percussive resonator

Use `modal`, strong alignment, narrow bands, and a lower target ceiling. Short
impulses then excite pitched decays much like struck bars or strings. Preserve
headroom because many resonances can sum during dense attacks.

### Diffuse microtonal room

Use `hybrid` or `stochastic`, strength below 0.6, 30–60-cent bandwidth, and a
moderate gain. This leaves enough untuned energy to bind the tail together and
reduces the impression of a parallel synthesizer.

## Worked Harmonic-Time Scenarios

### One chord, several decay windows

Begin with one sustained sonority followed by silence. Render it through three
IRs sharing scale, root, seed, and spectral limits but using RT60 values of 1.5,
5, and 15 seconds. The short response behaves primarily as color. The middle
response lets individual degrees become legible after release. The long
response turns the chord into a formal background for whatever follows.

This comparison demonstrates why scale-tuned reverb cannot be specified by
tuning alone. The same frequency lattice may function as timbre, cadence, pedal,
or independent layer solely because decay changes the listening window.

### Diatonic phrase against a just field

Use a 5-limit scale rooted on the phrase's structural center, then play or
render a 12-EDO phrase through it. Stable notes may align closely with IR
targets, while tempered thirds and sevenths create slow discrepancies. Use
broad bands first. Narrow the bands only after deciding whether the resulting
beats articulate tension meaningfully.

Then transpose the dry phrase without changing the IR. This converts the field
from confirmation to contradiction. Finally, transpose both source and root by
the same ratio. The three conditions distinguish absolute resonance placement
from interval structure.

### Modulation between two roots

Prepare two IRs from the same scale, seed, mode, RT60, and bandwidth, but assign
different root frequencies. Put each on a separate wet return. Before the
harmonic modulation, fade the destination return in while the source still
occupies the old region. Let both tails coexist briefly, then fade the old
return after its important decay has passed.

The overlap is not merely technical concealment. It creates a harmonic corridor
in which both resonant memories coexist. A short crossfade reads as a spatial
switch; a long crossfade can become a composed modulation. Keep the dry signal
stable so the listener can attribute the transformation to the field rather
than to level movement.

### Scale mutation with a fixed root

Generate matched IRs from two scales sharing the same root and period but
differing in selected degrees. The common degrees preserve continuity while
changed degrees alter the decay's internal harmony. Equal-power IR morphing
creates one static midpoint; parallel returns with automated gains create a
time-varying transition.

Use an impulse to confirm that the two endpoint IRs have comparable onset and
energy. Then test sustained material. A difference that is subtle on the
impulse may become obvious after repeated harmonic excitation.

### Non-octave form

With a tritave-period IR, orchestrate successive registers rather than treating
octave duplication as neutral. A bass note and its octave may excite different
relations to the field, while a frequency tripling returns to the scale's
periodic structure. Alternate octave and tritave displacement to make the
reverb reveal the difference after each attack.

Non-octave design is especially effective when the source itself is inharmonic
or when register functions structurally. Bells, metallic percussion, granular
spectra, and synthesized partial sets can cooperate with the field without
implying conventional octave-based harmony.

### Silence as harmonic exposure

Compose rests long enough to hear the tuned return without the direct source.
The attack may conceal differences that become clear only after 300 ms or one
second. A rest can therefore operate as a harmonic microscope. Compare early
silence, where room cues dominate, with late silence, where the narrowest and
longest modes survive.

When a phrase ending sounds wrong, do not inspect only its final chord. The tail
contains weighted remnants of earlier notes. Shorten RT60, reduce low-frequency
participation, or alter harmonic rhythm before changing the final sonority.

## Transposition, Modulation, and Changing Harmony

A scale-conditioned IR has a fixed root. To transpose the resonant field,
regenerate it with a different `--scala-root-hz`. For a frequency ratio $r$,
the transposed root is

$$
f_r' = r f_r.
$$

For $n$ equal-tempered semitones, $r=2^{n/12}$. For a Scala degree, use the
degree's ratio directly. Regeneration is deterministic and cached, so a small
library of roots can be prepared before a session.

Do not switch long IRs at arbitrary note boundaries and expect an inaudible
transition. The old convolution state contains the previous tuning. Safer
strategies include overlapping wet returns, equal-power crossfades longer than
the important early field, or rendering sections separately with complete
tails. A future dynamic resonator could track harmony continuously, but that is
a different DSP architecture from static convolution.

## Stereo, Surround, and Immersive Deployment

Scale tuning adds a harmonic dimension to spatial routing. A multichannel field
can repeat one lattice everywhere, distribute related lattices, or separate
diffuse and pitched responsibilities between layers.

### Shared-field strategy

Use one multichannel IR or closely matched IRs with the same scale, root, and
conditioning. This produces the strongest sense of one coherent room. Vary
early reflections and decorrelation while keeping the late spectral center
shared. It is the safest strategy for film, vocal, and ensemble work where
spatial continuity matters more than hearing individual degrees move.

### Complementary-degree strategy

Divide a scale into overlapping subsets and assign them to different returns.
For example, front channels can carry stable structural degrees while side and
rear returns emphasize tensions or upper extensions. Always retain common
diffuse energy; completely disjoint narrow lattices can make head movement or
downmixing expose holes.

This strategy is compositional, not a claim of physical-room realism. Document
which degrees belong to each layer and test every fold-down. A stereo or mono
sum may combine subsets into a denser spectrum than any individual speaker
reveals.

### Height as spectral perspective

Height returns often tolerate brighter target ranges because listeners can
associate elevated energy with air or halo. Rather than sending the complete
low-frequency lattice overhead, use a higher `--scala-low-hz`, moderate gain,
and enough diffuse content to avoid isolated whistles. Keep bass and structural
root information in the bed unless a deliberately disembodied image is wanted.

### Moving objects and fixed tuned fields

An object can move through a fixed bed or field return while retaining one
scale-conditioned decay. If several location-specific IRs use different roots,
movement also becomes harmonic transformation. Crossfade states rather than
hard-switching them, and distinguish panning automation from IR-state
automation in the session record.

### Translation checks

Evaluate at least speaker-array, binaural, stereo, and mono versions. Listen for
degree cancellation, exaggerated shared bands, center buildup, and a height
layer that disappears when folded down. Use matched loudness and include both
sustained and transient material. A spatial design is not complete because its
native layout sounds impressive.

## Measurement and Analytical Listening

Standard room metrics do not fully describe scale conditioning. RT60, EDT, and
clarity remain useful, but they should be paired with frequency-target evidence.

### Target-to-neighbor contrast

For each resolved frequency $f_k$, compare energy in a narrow target band with
energy in adjacent off-target bands. A simple contrast measure is

$$
C_k=10\log_{10}\!\left(\frac{E_{\mathrm{target},k}}
{E_{\mathrm{neighbor},k}+\epsilon}\right),
$$

where $\epsilon$ prevents division by zero. Report the distribution of $C_k$
rather than only its mean. One dominant low mode can otherwise hide weak upper
conditioning.

### Decay by target degree

Filter the IR around selected degrees and fit decay slopes separately. Two
targets can have similar peak magnitude but different persistence. Degree-wise
decay is musically important because late harmony is governed by the modes that
survive, not merely those that begin loudly.

### Matched-seed controls

Generate an untuned and tuned IR with identical mode, seed, length, RT60,
channel count, and broad spectral shaping. Render the same source through both
and loudness-match the results. This isolates conditioning from random topology
and gross level. Repeat across several seeds before making a general claim;
one favorable random realization is an anecdote, not a robust result.

### A listening vocabulary

Use consistent descriptors so notes can be compared across sessions:

- **Pitch salience:** how clearly the return suggests stable pitch.
- **Roughness:** how strongly close components produce rapid fluctuation.
- **Fusion:** whether source and return form one object.
- **Harmonic drag:** whether old tail energy resists a new harmony.
- **Spectral halo:** broad affinity without distinct notes.
- **Resonator identity:** perception of the return as an instrument.
- **Register bias:** concentration of the effect in bass, middle, or treble.
- **Decay revoicing:** change in apparent harmony as modes disappear.

Collect notes during full-mix, wet-only, and post-release listening. The three
conditions answer different questions and should not be collapsed into one
preference score.

## Reproducible Scale Libraries

Treat the `.scl` file as source data, not merely a preset name. Two files with
the same filename can contain different degrees. verbx records the scale
description, content SHA-256, root mapping, resolved frequencies, frequency
limits, strength, bandwidth, gain, and target budget in the IR metadata. The
content hash also participates in cache identity.

For a production or research corpus, retain:

- the original `.scl` file and its license or source note;
- the exact `verbx ir gen` command or configuration;
- the generated IR and metadata sidecar;
- an analysis JSON report;
- the source audio identity and render command;
- listening notes describing tuning, root, register, and perceived interaction.

When generating machine-learning data, split scale families before expanding
roots, seeds, RT60 values, and source files. Variants from one scale should not
leak across training, validation, and test sets if the experiment claims
generalization to unseen tunings. Include untuned controls and level-matched
ablation renders so a model cannot solve the task from loudness alone.

## Scale-Tuned Reverb for Audio AI and Data Augmentation

Scale-conditioned IRs can test whether an audio model is robust to structured,
musically meaningful coloration rather than only generic room decay. They are
useful for source separation, pitch estimation, transcription, instrument
recognition, dereverberation, acoustic-scene classification, and generative
audio evaluation, but only when the data split prevents tuning leakage.

### Define the experimental factor

Decide what “tuning variation” means before generating files. Possible factors
include scale family, degree count, period ratio, root mapping, bandwidth,
strength, target range, or the relation between source tuning and IR tuning.
Changing all of them together produces variety but weak scientific evidence.

A controlled factorial design might vary:

| Factor | Example levels |
|---|---|
| Scale family | 19-EDO, 5-limit just collection, 13-EDT Bohlen–Pierce |
| Root relation | matched, one degree displaced, unrelated |
| Strength | 0, 0.4, 0.7, 1.0 |
| Bandwidth | 12, 30, 60 cents |
| RT60 | 1.5, 5, 12 seconds |
| Source class | speech, monophonic music, polyphonic music, percussion, noise |

The strength-zero condition is essential. It reveals whether performance
changes come from reverberation itself or from the scale-conditioned component.

### Split by lineage, not rendered filename

One `.scl` file can generate hundreds of roots, seeds, RT60 values, and source
combinations. Randomly splitting those WAV files allows nearly identical tuning
lattices into training and evaluation sets. Instead, assign the parent scale or
scale family to a split first, then generate descendants inside that split.

For tests of unseen roots but known interval structure, keep scale files shared
and split root mappings. For tests of unseen tuning systems, keep complete scale
families out of training. State which generalization claim the split supports.

### Preserve dry and wet evidence

For supervised dereverberation or augmentation, retain the dry source, IR,
wet-only convolution, final mixture, and exact gain relation. A useful linear
mixture is

$$
y[n]=g_d x[n]+g_w(x*h)[n],
$$

where $g_d$ and $g_w$ are documented dry and wet gains. If normalization runs
after mixing, record it because it changes the direct-to-reverberant relation.

Do not let filenames encode the target class if a data loader or model can see
them. Store scale identity and parameters in a manifest. Hash the source,
Scala file, IR, and output so regenerated corpora can be audited.

### Avoid shortcut learning

A classifier may appear to recognize a tuning while actually recognizing
loudness, spectral tilt, IR length, one source instrument, or one random seed.
Use matched-seed controls, loudness matching, balanced source classes, and
multiple roots. Include adversarial controls where scale is held constant but
gain and RT60 change, and where room statistics remain constant while the scale
changes.

For dereverberation, evaluate whether the model removes the tail without
flattening legitimate source harmonics near Scala targets. A model trained only
on untuned rooms may interpret stable tuned decay as part of the instrument; a
model trained too aggressively on tuned IRs may suppress sustained musical
partials.

### Evaluation beyond one aggregate score

Report task metrics separately by scale family, root relation, source class,
strength, bandwidth, and RT60. Add target-band residual measurements for
dereverberation and pitch error for transcription. A single mean can hide a
model that performs well on diffuse fields and fails on sparse modal ones.

Listen to representative errors. Scale-conditioned tails can create musically
important artifacts that generic perceptual scores underweight, including a
wrong residual pedal, beating around a held note, or late energy that changes
apparent chord quality.

## Failure Modes and Corrections

| Symptom | Likely cause | Correction |
|---|---|---|
| Tail sounds like unrelated sine tones | Strength or gain too high; bandwidth too narrow | Lower strength, broaden bands, or use `hybrid` |
| Bass implies the wrong harmony | Root mapping or low limit conflicts with the work | Verify root degree and raise `--scala-low-hz` |
| Effect disappears in a dense mix | Too little source energy near targets or bands too broad | Solo the return, inspect spectrum, then increase gain modestly |
| Vocal vibrato sounds trapped | Resonances are too selective | Broaden bandwidth or move tuning emphasis above the vocal fundamentals |
| High-division scale sounds incomplete | Target budget is truncating the lattice | Increase `--scala-max-targets` or narrow the frequency range |
| Realtime launch seems slow | IR is being generated in the session path | Pre-generate and cache the IR before performance |
| A scale edit seems ignored | An old IR file was loaded | Regenerate and verify the recorded SHA-256 |
| Downmix becomes phasey or hollow | Channel variants are too spectrally independent | Increase shared diffuse energy and validate mono/stereo folds |

## A Focused Listening Exercise

Choose one dry sustained chord and one short percussion phrase. Generate four
IRs with the same mode, seed, length, RT60, root, and scale:

1. An untuned control.
2. A broad setting at 50 cents and strength 0.4.
3. A moderate setting at 25 cents and strength 0.65.
4. A narrow setting at 10 cents and strength 0.9.

Loudness-match the wet returns. For each source, describe pitch stability,
roughness, decay continuity, apparent register, and whether the reverb is heard
as space, timbre, harmony, or a separate instrument. Repeat after transposing
the root by one scale degree. The exercise reveals a central principle:
scale-tuned reverberation is not one effect but a continuum between diffuse
space and resonant composition.

## Extended Laboratory and Composition Studies

### Study 1: Cents, ratios, and audible beating

Create three two-note source files whose upper frequencies differ slightly
around one target ratio. Render them through a narrow modal IR. Measure the
frequency difference, predict the beat rate, and compare prediction with the
wet-only decay. Repeat one octave higher while preserving the same cent offset.
Explain why the hertz difference changes.

### Study 2: Bandwidth as performance tolerance

Generate IRs at 10, 25, 50, and 100 cents with all other parameters fixed.
Process a stable oscillator, a singer or string tone with vibrato, and a
glissando. Identify the point at which tuning changes from selective resonance
to broad coloration for each source.

### Study 3: Scale cardinality and modal density

Compare five-, seven-, 12-, 19-, and 31-degree octave-period scales over the
same frequency range and target budget. Keep seed, mode, RT60, strength, and
gain fixed. Count resolved targets, inspect spectral spacing, and describe when
individual pitch identity gives way to a continuous field.

### Study 4: Octave and non-octave periodicity

Process the same sequence through an octave-period IR and a tritave-period IR.
Transpose the source by $2/1$, then by $3/1$. Determine which transposition
preserves the relation to each field. Compose a short passage in which the
reverb, rather than the dry line, reveals the periodic structure.

### Study 5: Harmonic memory and tempo

Use one chord progression at three tempi and three RT60 settings. Do not change
the scale or wet gain. Mark where prior harmonies remain audible under later
ones. Find a tempo-decay combination that supports continuity and another that
creates deliberate harmonic drag.

### Study 6: Root-degree rotation

Hold `--scala-root-hz` fixed while changing `--scala-root-degree`. Compare the
resolved target lists before listening. Render one source through each IR and
explain why rotation is not equivalent to an ordinary equal-tempered
transposition.

### Study 7: Orchestration as excitation

Render click, speech, bowed string, piano, cymbal, sine, and noise through one
IR. Use wet-only outputs and matched levels. Create a table of which registers
and degrees each source excites most strongly, then orchestrate a one-minute
study that reveals the lattice gradually without changing the IR.

### Study 8: Two-root modulation

Build two rooted IRs with identical seeds. Place them on parallel returns and
compose a transition using a short crossfade, a long crossfade, and a period of
intentional overlap. Compare whether each version sounds like a key change,
room change, or emergence of a second instrument.

### Study 9: Spatial harmony

Assign a shared diffuse field to all channels and complementary tuned layers to
front, side, rear, and height groups. Produce speaker-array, binaural, stereo,
and mono renders. Document which harmonic relationships survive each
translation and revise the layout to reduce cancellation or buildup.

### Study 10: Matched-seed perceptual test

Generate an untuned control and three conditioned IRs from the same seed.
Loudness-match randomized renders and conduct a blinded listening test. Ask
participants to rate pitch salience, roughness, fusion, and spatial plausibility.
Report individual responses as well as means; tuned reverb can divide listeners
according to attention and musical experience.

### Study 11: Dereverberation stress test

Create matched dry, ordinary-reverb, and scale-tuned-reverb examples. Run the
same dereverberation configuration on both wet conditions. Compare target-band
residuals, source-harmonic damage, RT60 reduction, and listening quality. Explain
whether the estimator treats tuned decay as room energy or musical sustain.

### Study 12: A composed resonant architecture

Design a five-minute work in which at least three formal sections use the same
dry instrumental palette but different relationships to a fixed Scala IR.
Section one should confirm the field, section two should contradict it, and
section three should use silence to expose accumulated decay. Submit the scale,
commands, IR metadata, analysis JSON, score or timeline, and a commentary on
how reverberation carries form.

These studies progress from isolated variables to complete musical design. In
every case, retain an untuned or matched control and distinguish what the
measurement demonstrates from what the listening interpretation suggests.
