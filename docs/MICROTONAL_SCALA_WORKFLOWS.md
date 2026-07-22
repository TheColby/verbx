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
