# verbx Extreme Workflow Cookbook (100 Recipes)

A high-intensity command library for experimental and large-space reverb design in `verbx`. This is not a beginner tutorial — it is a working reference for sound designers, composers, and engineers who want to push the tool into territory where things get genuinely strange. Some of these sounds are useful. Some are purely destructive. All of them are instructive.

---

## Before You Start

**What should `in.wav` be?**

Almost anything works, but the most interesting results come from material with transient information — percussive hits, short melodic phrases, spoken words, field recordings with attack. Sustained drones and pads tend to disappear into long tails. Heavily compressed material loses dynamic articulation in the freeze and bloom passes. A single piano note, a snare hit, a short spoken syllable, a plucked string — these are ideal starting points.

For multichannel recipes (sections 8), replace `in.wav` with `in_5p1.wav` or `in_7p1.wav` as noted. These should be properly interleaved multichannel files, not summed stereo.

**How to read the commands**

Every recipe is a single shell command. Arguments with `--` are named flags. Numeric values follow directly after the flag name. File paths are relative to wherever you run the command from. If you want to run these from a different directory, adjust the paths accordingly.

The output folder must exist before you run anything:

```bash
mkdir -p out
```

**What do the output files mean?**

Each output file is named with a three-digit index and a short descriptor. The index matches the recipe number. The descriptor is a rough hint at what parameter dominated the result. Output files are standard WAV unless `--out-subtype` specifies otherwise. Loudness normalization is applied at the output stage unless you explicitly disable it.

**A note on headphones vs. speakers**

Wide-field and shimmer recipes will behave very differently on headphones vs. speakers. Multichannel outputs need a properly configured monitoring chain. Several of the self-convolution recipes produce subsonic energy that is inaudible on small speakers but will stress subwoofers and some headphone amplifiers. You have been warned.

---

## Building Blocks

Before diving into the extremes, here are the six commands worth understanding deeply. Everything else in this cookbook is a variation or combination of these.

```bash
# 1. Simplest possible algo reverb — 10 seconds of decay, mostly wet
verbx render in.wav out/block_01_algo.wav --engine algo --rt60 10 --wet 0.9 --dry 0.1

# 2. Convolution with a real IR — the cleanest path to a convincing space
verbx render in.wav out/block_02_conv.wav --engine conv --ir hall.wav

# 3. Freeze — grabs a slice of the audio and holds it as an infinite pad
verbx render in.wav out/block_03_freeze.wav --freeze --start 1 --end 2

# 4. Shimmer — pitch-shifted feedback that creates the classic ethereal octave shimmer
verbx render in.wav out/block_04_shimmer.wav --engine algo --shimmer --shimmer-semitones 12

# 5. Self-convolution — uses the audio as its own impulse response (things escalate fast)
verbx render in.wav out/block_05_self.wav --self-convolve

# 6. Lucky mode — randomizes parameters and generates multiple variations at once
verbx render in.wav out/block_06_lucky.wav --lucky 5 --lucky-out-dir out/lucky_block
```

These six commands are the grammar. The 100 recipes below are the sentences.

---

## Public Alpha Musical Reference Workflows

These five recipes are the canonical musical examples used in the public alpha
launch narrative. They are maintained in `README.md` and mirrored here so they
do not drift.

**1) Alvin Lucier / I Am Sitting in a Room (iterative room resonance)**
```bash
verbx render voice.wav lucier_7pass.wav --engine algo --rt60 4.5 \
  --wet 1.0 --dry 0.0 --repeat 7 --fdn-lines 16 --fdn-matrix hadamard --lowcut 60
```

**2) Brian Eno / Discreet Music (ambient loopbed)**
```bash
verbx render input.wav eno_ambient.wav --engine algo --rt60 12.0 \
  --wet 0.92 --dry 0.08 --damping 0.25 --pre-delay-ms 35 \
  --fdn-lines 16 --fdn-matrix hadamard --lowcut 50 \
  --target-lufs -22 --target-peak-dbfs -2
```

**3) Pauline Oliveros / Deep Listening (extended drone-space)**
```bash
verbx render drone.wav deep_listening.wav --engine algo --rt60 18.0 \
  --wet 0.95 --dry 0.10 --fdn-lines 32 --fdn-matrix hadamard \
  --pre-delay-ms 55 --damping 0.15 --lowcut 30 \
  --target-lufs -24 --target-peak-dbfs -2
```

**4) Frippertronics-style tape-loop accumulation**
```bash
verbx render guitar.wav frippertronics.wav --engine algo --rt60 8.0 \
  --wet 0.82 --dry 0.28 --fdn-lines 16 --fdn-matrix hadamard \
  --shimmer --shimmer-semitones 12 --shimmer-mix 0.45 --shimmer-feedback 0.78 \
  --pre-delay-ms 25 --target-peak-dbfs -2
```

**5) Shoegaze reverse-wash (freeze + shimmer)**
```bash
verbx render guitar.wav shoegaze.wav --engine algo \
  --freeze --start 1.0 --end 2.4 \
  --shimmer --shimmer-semitones 12 --shimmer-mix 0.55 --shimmer-feedback 0.72 \
  --rt60 5.0 --wet 0.88 --dry 0.22 --fdn-matrix circulant --lowcut 80 \
  --width 1.4 --target-peak-dbfs -2
```

---

## How to Listen Critically

This section sits between the building blocks and the recipes because the way you listen matters as much as the settings you use.

**For algorithmic reverbs (section 1):** Listen for the density of the diffusion network. A well-tuned algo reverb should feel like a continuous fog rather than a series of distinct echoes. Listen for colorations in the decay — most algo engines have a characteristic tone that emerges around 3-6 seconds of tail. High `--beast-mode` values will push the feedback networks toward instability; you will hear this as a shift from smooth decay to a kind of churning or fluttering texture.

**For freeze and repeat chains (section 2):** The transition point between the frozen segment and the incoming signal is the interesting moment. Listen for phase artifacts at loop boundaries, for build-up of low-mid energy across repeats, and for how the freeze interacts with the original signal's wet/dry balance. Repeat chains tend to accumulate energy — listen for how normalization (or its absence) shapes the perceived loudness arc.

**For convolution (section 3):** The first 50ms of a convolution reverb contains the early reflections, which define the perceived size of the space. The tail defines the character. Listen to them separately by shortening and extending `--tail-limit`. A good IR should feel like moving the source into a real space; a bad IR will sound like filtering. Partition size affects latency but not sound quality in offline rendering — in these recipes it primarily affects CPU allocation.

**For self-convolution (section 4):** This is mathematically interesting and sonically extreme. The audio is convolved with itself, which has the effect of squaring the spectral envelope. Peaks in the original spectrum become dramatically exaggerated. Quiet passages become very quiet. Loud passages become overwhelming. The result is often described as "the audio eating itself." Listen for the spectral imbalance this creates, and for how `--tilt` can be used to correct or exaggerate it.

**For shimmer, duck, and bloom (section 5):** Shimmer is additive — it adds pitched content that was not there before. Duck is subtractive — it removes the reverb tail while the dry signal is present. Bloom is temporal — it delays the onset of the reverb buildup. Listen to each in isolation before combining them. When all three are active simultaneously (recipe 50), the result is layered enough that you should listen to it at least three times: once for the dry signal, once for the reverb tail, once for the shimmer layer.

**For self-generated IRs (section 7):** The most useful skill when working with synthetic IRs is learning to distinguish the IR's character from the reverb engine's character. Process the same input with different IRs and compare. A stochastic IR will produce a different kind of density than an FDN IR even if the RT60 is identical.

---

## Section 1: Algorithmic Extremes (Recipes 1-10)

Algorithmic reverb builds synthetic spaces using networks of delay lines, all-pass filters, and feedback matrices — no recorded impulse required. This makes it fast to iterate and infinitely malleable, but it also means the "space" it simulates is fundamentally fictional. That is its power. In film post, algo reverbs are used for sci-fi environments, supernatural spaces, and any situation where the reverb needs to be emotionally correct rather than acoustically accurate. In electronic music, they are the texture underneath ambient pads, the wash behind a snare, the infinite hallway a note disappears into.

The `--beast-mode` flag increases the density and complexity of the internal diffusion network. Low values (1-4) are broadly useful. Values above 8 begin to produce audible coloration. Values above 12 are genuinely unstable on some inputs and will produce results that range from magnificent to unusable.

---

**Recipe 1**
```bash
verbx render in.wav out/001_algo_long.wav --engine algo --rt60 120 --wet 0.95 --dry 0.1 --beast-mode 6
```
_What it sounds like:_ A two-minute reverb tail that utterly swallows the source. The original signal is barely audible.

_DSP note:_ RT60 of 120 seconds means the energy takes two full minutes to decay by 60dB. Combined with `--beast-mode 6`, the diffusion network is running at higher complexity, producing a very smooth, dense tail. At `--wet 0.95`, you are hearing almost exclusively the reverberant field.

---

**Recipe 2**
```bash
verbx render in.wav out/002_algo_dark.wav --engine algo --rt60 90 --damping 0.9 --wet 0.9 --dry 0.15
```
_What it sounds like:_ A very long, very dark reverb — like shouting into a large cave and hearing the sound transform into low-frequency mush.

_DSP note:_ `--damping 0.9` applies frequency-dependent attenuation to the high frequencies on each feedback pass through the reverb network. High damping values simulate absorptive surfaces (heavy drapes, acoustic foam, water). The result is a reverb that loses its high-frequency content exponentially faster than its low-frequency content.

---

**Recipe 3**
```bash
verbx render in.wav out/003_algo_wide.wav --engine algo --rt60 75 --width 2.0 --wet 0.85 --dry 0.2
```
_What it sounds like:_ An expansive stereo reverb where the tail feels wider than the speakers. Sounds placed at center appear to spread outward as they decay.

_DSP note:_ `--width 2.0` applies a Mid-Side matrix transform to the stereo output of the reverb, scaling the Side component. Values above 1.0 exaggerate the stereo difference signal. This can produce phantom imaging that extends beyond the speaker baseline. Listen on headphones for the full effect — and be aware that mono compatibility degrades as width increases.

---

**Recipe 4**
```bash
verbx render in.wav out/004_algo_mod_slow.wav --engine algo --rt60 80 --mod-depth-ms 12 --mod-rate-hz 0.05
```
_What it sounds like:_ A subtly breathing reverb, almost alive. The tail slowly warps over time. Works well for sustained material.

_DSP note:_ The modulation LFO at 0.05 Hz has a period of 20 seconds. It is modulating the delay lengths in the reverb network by up to 12ms, which creates gentle pitch and timing variations in the late reverb field. This is the mechanism behind the characteristic "swimming" quality of classic hardware reverbs like the Lexicon 224.

---

**Recipe 5**
```bash
verbx render in.wav out/005_algo_mod_fast.wav --engine algo --rt60 65 --mod-depth-ms 8 --mod-rate-hz 1.2
```
_What it sounds like:_ A chorusing, slightly seasick reverb. The tail has a metallic shimmer that is different from the shimmer flag — this one is caused by interference patterns in the modulated delays.

_DSP note:_ At 1.2 Hz with 8ms depth, the modulation is fast enough to produce audible sideband frequencies. The interaction between the modulation rate and the delay network produces beating artifacts that some ears read as pleasant chorus and others read as instability. At higher depths it crosses into vibrato territory.

---

**Recipe 6**
```bash
verbx render in.wav out/006_algo_predelay_cloud.wav --engine algo --rt60 70 --pre-delay-ms 220 --wet 0.9
```
_What it sounds like:_ The source plays, there is a noticeable gap of silence, then a massive reverb cloud arrives. Very dramatic. Works extremely well with vocals and solo instruments.

_DSP note:_ 220ms pre-delay is well above the Haas threshold (~30ms), meaning the brain perceives the reverb tail as a distinct echo event rather than as spatial information. This is a widely used production technique for creating "space" without burying the source in reverb. The gap also allows transient information from the dry signal to remain fully articulate.

---

**Recipe 7**
```bash
verbx render in.wav out/007_algo_splash.wav --engine algo --rt60 45 --beast-mode 10 --repeat 2
```
_What it sounds like:_ A dense, complex reverb splash that repeats twice. The beast-mode density combined with repetition creates a layered wash.

_DSP note:_ `--repeat 2` runs the render pipeline twice and stacks the outputs. At `--beast-mode 10`, the diffusion network is at high complexity, so each pass through adds a distinct but related layer of dense reflection patterns.

---

**Recipe 8**
```bash
verbx render in.wav out/008_algo_massive.wav --engine algo --rt60 180 --wet 1.0 --dry 0.0 --beast-mode 12
```
_What it sounds like:_ The source disappears completely. What remains is three minutes of dense, evolving reverb that has almost no relationship to what went in. Use headphones. This one is genuinely terrifying at volume.

_DSP note:_ `--wet 1.0 --dry 0.0` eliminates the direct signal path entirely. Combined with a 180-second RT60 and beast-mode 12, the reverb network is operating at high feedback complexity with an enormous decay envelope. The output is essentially a synthetic texture derived from the input's early moments. On spectrally complex input, this is often beautiful. On input with strong low-frequency content, it will rumble and build in unexpected ways.

> **Expert note:** At RT60 values this extreme, the reverb network is operating at the edge of stability. The internal feedback matrix is nominally convergent, but floating-point accumulation over very long tails can still produce low-level artifacts that surface in the noise floor of spectral analysis tools. In verbx this occurs in 64-bit (`float64`) internal DSP, which substantially lowers numerical drift versus single-precision paths.

---

**Recipe 9**
```bash
verbx render in.wav out/009_algo_air.wav --engine algo --rt60 50 --damping 0.2 --highcut 18000 --tilt 3
```
_What it sounds like:_ A bright, airy reverb. The tail sparkles. Good on acoustic instruments and vocals where you want presence, not darkness.

_DSP note:_ Low damping (0.2) preserves high-frequency content across feedback passes, producing a reverb tail that retains treble energy. `--tilt 3` applies a gentle high-shelf tilt to the output, further emphasizing the air frequencies above 4kHz.

---

**Recipe 10**
```bash
verbx render in.wav out/010_algo_mud.wav --engine algo --rt60 110 --lowcut 30 --highcut 1800 --tilt -5
```
_What it sounds like:_ A thick, dark, low-mid reverb that eliminates almost all high-frequency content. Like being inside a concrete room full of water.

_DSP note:_ The combined effect of `--highcut 1800` and `--tilt -5` aggressively attenuates frequencies above 1.8kHz. The result is a reverb that exists almost entirely in the 30-1800Hz band. Long RT60 in this band produces the characteristic "mud" of untreated room reverb. Useful for horror sound design, industrial textures, and sub-bass reinforcement.

---

## How to Listen: Freeze and Repeat

Freeze is distinct from reverb — it is not a space simulation, it is a temporal manipulation. When you freeze a segment, you are extracting a window of audio and playing it back as a loop. The quality of the loop (smoothness, phase coherence, the presence or absence of click artifacts) depends on the nature of the source material at the freeze point. Sustained tones freeze cleanly. Transients at the freeze boundary create artifacts that can be used deliberately.

Repeat chains stack the entire render output and re-process it. This accumulates artifacts, saturates the frequency response, and gradually transforms the material. Listen to how each repeat changes the spectral balance, and notice whether normalization is fighting the accumulation or allowing it.

---

## Section 2: Freeze and Repeat Chains (Recipes 11-20)

Freeze workflows are central to ambient music production and sound design for picture. The ability to hold a moment of audio indefinitely — creating a synthetic drone from a single phrase, or sustaining a room ambience between cuts — is one of the more practically useful things `verbx` does. Repeat chains go further, iterating the reverb process multiple times and creating layered, progressively transformed textures. In scoring, this is used to build tension by accumulating reverberant energy. In experimental music, it is used to watch a source material destroy itself through repeated convolution.

---

**Recipe 11**
```bash
verbx render in.wav out/011_freeze_short.wav --freeze --start 2 --end 3 --repeat 3 --engine algo
```
_What it sounds like:_ A one-second window of the source audio frozen and repeated three times through an algo reverb. You will hear the source phrase briefly, then a looped, reverberant version of it.

_DSP note:_ `--start 2 --end 3` extracts the audio between the 2-second and 3-second marks. This window is looped internally and then passed through the reverb engine. The algo engine processes the frozen loop, so the reverb is applied to the looped content rather than to the original recording.

---

**Recipe 12**
```bash
verbx render in.wav out/012_freeze_wide.wav --freeze --start 4 --end 6 --repeat 4 --width 1.8
```
_What it sounds like:_ A two-second freeze, widened dramatically, repeated four times. The stereo image expands with each repeat.

_DSP note:_ Width processing at 1.8 applied to a frozen loop will amplify any stereo differences present in the extracted window. If the source is mono, the width processing will apply decorrelation to create artificial stereo spread, which accumulates across the four repeat passes.

---

**Recipe 13**
```bash
verbx render in.wav out/013_freeze_beast.wav --freeze --start 1 --end 2.2 --repeat 5 --beast-mode 14
```
_What it sounds like:_ A beast-mode freeze — dense, complex, with five passes of increasingly saturated reverb. The original source becomes unrecognizable by pass three.

_DSP note:_ Beast-mode 14 pushes the diffusion network into high-complexity territory. Applied to a freeze loop, each repeat pass processes an already-reverberant signal through a high-density network, compounding the diffusion and creating what is effectively reverb-of-reverb.

---

**Recipe 14**
```bash
verbx render in.wav out/014_freeze_dark.wav --freeze --start 3 --end 5 --repeat 3 --damping 0.85 --tilt -4
```
_What it sounds like:_ A dark, muffled freeze. The repeated passes become progressively more low-frequency dominated. Good for horror, industrial, or ominous ambient textures.

_DSP note:_ High damping combined with negative tilt stacks two separate high-frequency attenuation mechanisms. Damping reduces HF within the reverb feedback loop; tilt applies a static EQ shelf to the output. Over three repeat passes, the cumulative HF loss is significant.

---

**Recipe 15**
```bash
verbx render in.wav out/015_freeze_shimmer.wav --freeze --start 2.5 --end 4 --repeat 2 --shimmer
```
_What it sounds like:_ A frozen segment with shimmering pitched overtones that build across two repeat passes. The shimmer adds an octave-up harmonic that makes the loop feel larger and more cinematic.

_DSP note:_ The shimmer algorithm pitch-shifts the reverb signal up by 12 semitones (default) and feeds it back into the reverb input with a configurable mix level. Applied to a freeze loop, this creates a self-sustaining harmonic layer that re-enters the loop on each repeat.

---

**Recipe 16**
```bash
verbx render in.wav out/016_repeat_algo.wav --engine algo --rt60 55 --repeat 6 --normalize-stage per-pass
```
_What it sounds like:_ Six passes of algo reverb, each normalized independently. The loudness stays consistent but the texture becomes progressively denser and more complex.

_DSP note:_ `--normalize-stage per-pass` applies normalization at the output of each repeat iteration before passing it into the next. This prevents the accumulation from causing clipping, but it also means the normalization gain changes between passes, which can introduce subtle level-stepping artifacts on very short repeat segments.

---

**Recipe 17**
```bash
verbx render in.wav out/017_repeat_conv.wav --engine conv --ir hall.wav --repeat 4 --normalize-stage per-pass
```
_What it sounds like:_ Four convolution passes with a hall IR. The space expands with each pass — by pass four, the room sounds enormous.

_DSP note:_ Convolving the output of a convolution with the same IR is mathematically equivalent to squaring the transfer function of the impulse response (approximately). This causes the early reflections to multiply and the tail to extend non-linearly. The result is a reverb that grows beyond what the original IR's RT60 would suggest.

---

**Recipe 18**
```bash
verbx render in.wav out/018_repeat_duck.wav --engine algo --repeat 5 --duck --duck-attack 5 --duck-release 800
```
_What it sounds like:_ The dry signal punches through cleanly, then the reverb blooms afterward. Across five repeats, the duck-and-bloom pattern becomes a rhythmic feature of the texture.

_DSP note:_ Duck is a sidechain-style gain reduction applied to the wet signal when the dry signal is above a threshold. `--duck-attack 5` means the attenuation engages very quickly (5ms), protecting the transient. `--duck-release 800` means the attenuation releases slowly (800ms), giving the reverb a long fade-in after each transient. Combined with repeat, this creates a forward-moving texture that maintains rhythmic articulation.

---

**Recipe 19**
```bash
verbx render in.wav out/019_repeat_bloom.wav --engine algo --repeat 4 --bloom 4.5 --wet 0.95
```
_What it sounds like:_ The reverb builds slowly rather than appearing instantly. Over four repeats, the bloom tail becomes the dominant sound — a slow, rising wash.

_DSP note:_ Bloom delays the onset of the reverb density curve. A bloom value of 4.5 means the reverb takes 4.5 seconds to reach its full diffusion density. On repeat pass one, you hear a slow build. On subsequent passes, the input to each pass already contains reverberant energy, which the bloom curve then delays further — producing increasingly stretched decay envelopes.

---

**Recipe 20**
```bash
verbx render in.wav out/020_repeat_floor.wav --engine algo --rt60 140 --repeat 3 --target-lufs -26
```
_What it sounds like:_ Three passes of a very long reverb, normalized to a broadcast-appropriate loudness level. Useful as a bed or texture layer.

_DSP note:_ `--target-lufs -26` is roughly the integrated loudness target for film broadcast (-24 LUFS) plus a 2dB headroom buffer. Applying this to a three-pass, 140-second reverb tail ensures the output is at a known loudness level suitable for mixing into a larger context without additional gain staging.

---

## How to Listen: Convolution Modes

Convolution reverb is multiplication in the frequency domain — the spectrum of the input is multiplied by the spectrum of the impulse response. This makes it inherently linear and phase-coherent in a way that algorithmic reverbs are not. What you hear in a convolution reverb is literally what was captured in the IR, faithfully reproduced. The quality of the output is bounded by the quality of the IR. A noisy or poorly captured IR will produce a noisy reverb. A pristine IR from a great acoustic space will produce something that sounds like the real thing.

The interesting edge cases come from using IRs that are not captured from real spaces — synthetic IRs, self-convolution, matrix IRs for multichannel formats. In these cases, convolution stops being a space simulator and becomes a spectral transformation tool.

---

## Section 3: Convolution Heavy Modes (Recipes 21-30)

Convolution is the workhorse of post-production reverb. It is what music supervisors use when they need a score to sound like it was performed in a specific church. It is what sound designers use when they need a gunshot to sound like it happened in a parking garage rather than a foley stage. The goal is accuracy and plausibility, and when it works well, it is invisible.

The recipes in this section push convolution into less conventional territory — matrix IRs for multichannel, extremely large or small partition sizes, truncated tails, and combination with other processing.

---

**Recipe 21**
```bash
verbx render in.wav out/021_conv_hall.wav --engine conv --ir hall.wav --partition-size 32768
```
_What it sounds like:_ A clean hall reverb with a large FFT partition. Sounds like a real concert hall.

_DSP note:_ Partition-based convolution divides the IR into blocks that are convolved separately and summed. Larger partition sizes reduce CPU load at the cost of increased latency (in real-time contexts). In offline rendering, the partition size primarily affects memory allocation and processing chunk size — a partition of 32768 samples at 48kHz represents about 682ms per block.

---

**Recipe 22**
```bash
verbx render in.wav out/022_conv_plate.wav --engine conv --ir plate.wav --ir-normalize peak
```
_What it sounds like:_ A plate reverb character — dense, bright, fast-building. Classic on vocals and snare.

_DSP note:_ `--ir-normalize peak` normalizes the impulse response to its peak sample value before convolution. This ensures consistent output levels regardless of how the IR was captured. Without normalization, the loudness of the convolution output depends entirely on the amplitude of the IR, which varies between recordings.

---

**Recipe 23**
```bash
verbx render in.wav out/023_conv_church.wav --engine conv --ir church.wav --tail-limit 150
```
_What it sounds like:_ A church reverb capped at 150 seconds. Long, but not infinite.

_DSP note:_ `--tail-limit 150` truncates the IR at 150 seconds and applies a fade-out at the truncation point. Church IRs can run very long (some cathedrals have RT60s exceeding 10 seconds), and limiting the tail reduces both file size and processing time without significantly affecting the perceptual character of the early reverb.

---

**Recipe 24**
```bash
verbx render in.wav out/024_conv_mono_to_all.wav --engine conv --ir mono_ir.wav --wet 0.9 --dry 0.2
```
_What it sounds like:_ A stereo input processed with a mono IR. The reverb will be mono-width, which can sound narrower than expected.

_DSP note:_ When a mono IR is used on a stereo input, `verbx` applies the IR to each channel independently. The resulting reverb is correlated (identical in both channels) rather than decorrelated. This is technically correct but may sound narrower than a stereo IR. Some engineers prefer this for dialogue processing where reverb width should match the source.

---

**Recipe 25**
```bash
verbx render in.wav out/025_conv_surround.wav --engine conv --ir matrix_5p1.wav --ir-matrix-layout output-major
```
_What it sounds like:_ A multichannel convolution where each output channel gets its own IR. Requires a multichannel IR file.

_DSP note:_ Output-major matrix layout means the IR file is organized so that the first group of channels represents the IR for output channel 1 from all inputs, the second group for output channel 2, and so on. This is the format used when you want each output speaker to have a distinct reverb tail derived from all inputs.

---

**Recipe 26**
```bash
verbx render in.wav out/026_conv_input_major.wav --engine conv --ir matrix_5p1.wav --ir-matrix-layout input-major
```
_What it sounds like:_ The same multichannel convolution, but with the IR channels interpreted differently. The output will be different from recipe 25 even with the same IR file.

_DSP note:_ Input-major layout means the IR file is organized so that the first group of channels represents the IR for input channel 1 to all outputs, the second group for input channel 2, etc. The two layout modes are equivalent for symmetric IRs but produce different results for asymmetric multichannel IRs. Which is "correct" depends on how the IR was captured and exported.

---

**Recipe 27**
```bash
verbx render in.wav out/027_conv_fast.wav --engine conv --ir hall.wav --partition-size 65536 --normalize-stage none
```
_What it sounds like:_ A hall reverb with the largest partition size and no output normalization. Results in faster offline processing but potentially louder or quieter output depending on the IR.

_DSP note:_ Disabling normalization with `--normalize-stage none` passes the raw convolution output to the file. If the IR has a high amplitude and the input is also loud, the output can clip. This is intentional in certain mastering and sound design contexts where you want the raw convolution result without any gain adjustment.

---

**Recipe 28**
```bash
verbx render in.wav out/028_conv_dense.wav --engine conv --ir hall.wav --repeat 3 --beast-mode 5
```
_What it sounds like:_ Three convolution passes with beast-mode density. The hall becomes impossibly dense — more reflection than room.

_DSP note:_ Beast-mode applied to a convolution engine increases the density of the convolution kernel by internal upsampling and interpolation of the IR. Combined with three repeat passes, the result is a reverb that has far more dense reflections than the original hall IR would suggest.

---

**Recipe 29**
```bash
verbx render in.wav out/029_conv_trimmed.wav --engine conv --ir hall.wav --tail-limit 20
```
_What it sounds like:_ A very short hall reverb — you get the early reflections and room character but almost none of the tail. Sounds like a large room with very absorptive surfaces.

_DSP note:_ Truncating the IR at 20 seconds cuts most of the reverberation tail while preserving the crucial early reflection structure that defines the perceived size and shape of the space. This is useful when you want spatial information but not acoustic congestion in a mix.

---

**Recipe 30**
```bash
verbx render in.wav out/030_conv_fulltail.wav --engine conv --ir hall.wav --output-peak-norm input
```
_What it sounds like:_ A full hall reverb normalized so that its peak matches the peak of the input file. This ensures a predictable output level.

_DSP note:_ `--output-peak-norm input` measures the peak amplitude of the input file and applies gain to the output so that both share the same peak level. This is a reference-based normalization approach, useful for comparative listening where you want multiple output files at matched levels.

---

## Section 4: Self-Convolution and Feedback Smear (Recipes 31-40)

Self-convolution is where things get genuinely experimental. Using the audio as its own impulse response creates a spectral feedback loop — the signal's frequency peaks are amplified and its nulls are deepened, producing a transformation that is related to but distinct from the source. It is not a reverb in any conventional sense. It is more like the audio reflecting itself into a new shape.

This section is primarily useful for experimental sound design, academic study, and extreme texture generation. In production contexts, the output is rarely usable without subsequent processing, but as a raw material generator it is remarkable. The sounds here have appeared in horror films, industrial music, and electroacoustic composition.

---

**Recipe 31**
```bash
verbx render in.wav out/031_self_base.wav --self-convolve --normalize-stage none
```
_What it sounds like:_ The source convolved with itself. Spectral peaks become dominant. The result sounds like an exaggerated, smeared version of the original.

_DSP note:_ Self-convolution in the frequency domain squares the magnitude spectrum and doubles the phase. The squared magnitude means that any spectral peak becomes a more prominent peak, and any spectral trough becomes a deeper trough. The phase doubling causes phase alignment differences that create comb filtering when compared with the original. No normalization means the output level will vary dramatically depending on the input spectrum.

---

**Recipe 32**
```bash
verbx render in.wav out/032_self_beast.wav --self-convolve --beast-mode 20 --normalize-stage none
```
_What it sounds like:_ Self-convolution at extreme beast-mode. Do not play this loud without listening at low volume first.

_DSP note:_ Beast-mode 20 is at the upper end of the useful range. Applied to self-convolution, it adds multiple additional feedback paths through the diffusion network, each of which processes the already-self-convolved signal. The cumulative effect can produce output levels orders of magnitude above the input. `--normalize-stage none` means these levels are uncontrolled.

> **Expert note:** Self-convolution at high beast-mode values is computationally similar to running a feedback network with gain slightly above unity. In a real-time context this would cause unbounded growth (infinite clipping). In offline rendering, the process terminates when the tail reaches the end of the computed buffer, so the output is finite, but the energy distribution within that buffer may be extremely non-uniform. Spectral analysis of the output is informative — you will see the input's spectral peaks raised to extreme prominence.

---

**Recipe 33**
```bash
verbx render in.wav out/033_self_longtail.wav --self-convolve --tail-limit 200 --partition-size 32768
```
_What it sounds like:_ A very long self-convolution tail. The smeared version of the source continues for over three minutes.

_DSP note:_ Self-convolution produces a signal whose duration is approximately twice the duration of the input. `--tail-limit 200` extends the computed output to 200 seconds, allowing the self-convolved tail to fully develop. The large partition size handles the long convolution kernel efficiently.

---

**Recipe 34**
```bash
verbx render in.wav out/034_self_bright.wav --self-convolve --tilt 5 --highcut 18000
```
_What it sounds like:_ A self-convolution biased toward high frequencies. Sounds metallic, glassy, sometimes resembling granular synthesis.

_DSP note:_ Self-convolution tends to bias toward whichever spectral region has the highest energy in the source. `--tilt 5` pre-emphasizes the high frequencies before the convolution operation, which means the squared spectrum will have an even stronger high-frequency component in the output. The hard cut at 18kHz prevents aliased content from contaminating the output.

---

**Recipe 35**
```bash
verbx render in.wav out/035_self_dark.wav --self-convolve --tilt -6 --highcut 2500
```
_What it sounds like:_ A dark self-convolution. The result is dominated by low-to-mid frequencies. Sounds like a monster or machine dreaming.

_DSP note:_ `--tilt -6` and `--highcut 2500` together severely limit the bandwidth before and after the self-convolution. The squared low-frequency spectrum produces a dense, rumbling texture with very little high-frequency content. Apply EQ and saturation downstream to shape this into something usable.

---

**Recipe 36**
```bash
verbx render in.wav out/036_self_duck.wav --self-convolve --duck --duck-attack 3 --duck-release 600
```
_What it sounds like:_ Self-convolution with the dry signal punching through the self-convolved wash. The articulation of the original is preserved within the smeared texture.

_DSP note:_ Ducking applied to self-convolution uses the dry signal as the sidechain to attenuate the self-convolved output. This is unusual because the "wet" signal being ducked is derived from the same source as the sidechain. The result depends heavily on the timing relationship between the dry transients and the smeared output. Fast attack (3ms) means the duck engages almost immediately with each transient.

---

**Recipe 37**
```bash
verbx render in.wav out/037_self_shimmer.wav --self-convolve --shimmer --shimmer-mix 0.55
```
_What it sounds like:_ Self-convolution with an octave-up harmonic layer. Sounds enormous and harmonically complex.

_DSP note:_ Shimmer applied to the output of self-convolution adds a pitch-shifted copy of the already-spectrally-extreme self-convolved signal. At 0.55 mix, the shimmer layer is almost as loud as the self-convolved base. The harmonic content of the shimmer is a pitch-shifted version of the squared spectrum, which can create unusual harmonic relationships.

---

**Recipe 38**
```bash
verbx render in.wav out/038_self_repeat.wav --self-convolve --repeat 3 --normalize-stage per-pass
```
_What it sounds like:_ Three passes of self-convolution. The first pass is already extreme; the second and third are increasingly abstract.

_DSP note:_ Each repeat pass uses the output of the previous pass as the input to the next self-convolution. Mathematically, this raises the original spectrum to the power of 2^n (where n is the pass number). By pass three, the spectral peaks are at 2^3 = 8 times their original prominence (in log scale, +18dB relative to the original spectral balance). Per-pass normalization prevents clipping between passes.

---

**Recipe 39**
```bash
verbx render in.wav out/039_self_loud.wav --self-convolve --target-lufs -16 --target-peak-dbfs -1
```
_What it sounds like:_ Self-convolution normalized to streaming-loudness standards. Useful when the self-convolve output needs to be delivered at a calibrated level.

_DSP note:_ `-16 LUFS` is approximately the target for music streaming platforms. Combined with `-1 dBFS` true peak headroom, this configuration takes the unpredictable amplitude of self-convolution output and brings it to a usable, deliverable level. The nature of the sound is unchanged; only the gain is adjusted.

---

**Recipe 40**
```bash
verbx render in.wav out/040_self_huge.wav --self-convolve --beast-mode 40 --repeat 2
```
_What it sounds like:_ Beast-mode 40, two repeat passes of self-convolution. This is at the theoretical limit of what the engine can compute in a reasonable time. The output is unpredictable. Back up your input first.

_DSP note:_ Beast-mode 40 is at the edge of numerical stability for most input signals. Combined with two repeat passes of self-convolution, the internal state of the engine accumulates floating-point error at a rate that can produce sub-audible noise floors that are louder than the intended output. This is extreme use and is documented here for completeness, not as a recommended workflow.

---

## Section 5: Shimmer, Ducking, Bloom, and Tilt (Recipes 41-50)

These four tools are the texture controls. Shimmer adds. Duck subtracts. Bloom delays and builds. Tilt tilts. Each has a clear sonic function that can be grasped immediately, but the interesting work happens at the intersections — when shimmer feeds into a bloomed reverb, or when duck is combined with slow attack times and long tails.

In film scoring, shimmer is used to signal memory, transcendence, or altered states. Duck is ubiquitous in dialogue-heavy mixes where reverb would otherwise obscure speech. Bloom simulates the acoustic behavior of large spaces where diffuse reflections arrive after the direct sound. Tilt shapes the perceived brightness of a space to match the scene.

---

**Recipe 41**
```bash
verbx render in.wav out/041_shimmer_oct.wav --engine algo --shimmer --shimmer-semitones 12 --shimmer-mix 0.35
```
_What it sounds like:_ The classic shimmer reverb — an octave above the source, feeding back into the reverb tail. Recognizable from ambient music, film scores, and certain synthesizer patches.

_DSP note:_ Shimmer works by pitch-shifting the reverb output up by the specified number of semitones and feeding it back into the reverb input with the specified mix level. At 12 semitones (one octave), the feedback loop creates a self-sustaining harmonic layer that grows proportionally to the reverb's RT60. At 0.35 mix, it is present but not dominant.

---

**Recipe 42**
```bash
verbx render in.wav out/042_shimmer_double.wav --engine algo --shimmer --shimmer-semitones 24 --shimmer-feedback 0.8
```
_What it sounds like:_ Two octaves up, with very high feedback. The shimmer dominates the reverb completely. Sounds angelic or catastrophic depending on the source material.

_DSP note:_ `--shimmer-feedback 0.8` means the pitch-shifted signal is re-injected at 80% of its output level on each pass through the feedback loop. At two octaves, this produces an extremely bright, self-sustaining shimmer that can outlast the reverb tail itself. On input with strong low-frequency content, the two-octave shift brings bass frequencies into the midrange, which can produce unexpectedly clear harmonic content in the shimmer.

---

**Recipe 43**
```bash
verbx render in.wav out/043_shimmer_fifth.wav --engine algo --shimmer --shimmer-semitones 7 --shimmer-mix 0.5
```
_What it sounds like:_ A shimmer a perfect fifth above the source. Less obviously "shimmer-y" than the octave version, more harmonically complex. Try this on minor-key source material for interesting results.

_DSP note:_ Seven semitones is a perfect fifth in equal temperament. The pitch ratio is 2^(7/12) ≈ 1.498, very close to the just-intonation ratio of 3:2 (1.5). The slight deviation from pure tuning creates slow beating between the shimmer and source harmonics, which contributes to the characteristic "movement" of shimmer reverb.

---

**Recipe 44**
```bash
verbx render in.wav out/044_duck_hard.wav --engine algo --duck --duck-attack 2 --duck-release 900
```
_What it sounds like:_ The reverb ducks hard and fast when the source is present, then blooms slowly afterward. Classic dialogue-reverb behavior.

_DSP note:_ 2ms attack means the sidechain engages almost instantaneously — essentially every transient in the source will cause the reverb to duck. 900ms release means the reverb takes nearly a second to fully return after the source falls below the threshold. This is a deliberate choice to keep the reverb audible only in pauses and phrase endings, which is standard practice in narrative dialogue mixing.

---

**Recipe 45**
```bash
verbx render in.wav out/045_duck_soft.wav --engine algo --duck --duck-attack 60 --duck-release 180
```
_What it sounds like:_ A gentle, subtle duck. The reverb barely moves. Good for music where you want slight movement without obvious pumping.

_DSP note:_ 60ms attack is slow enough that transients pass through before the duck engages, which means percussive events do not cause sudden reverb dips. 180ms release is fast enough that the reverb returns before the next transient arrives (assuming material above ~160 BPM). The result is a slow, gentle breathing motion in the reverb rather than a clearly audible pumping effect.

---

**Recipe 46**
```bash
verbx render in.wav out/046_bloom_long.wav --engine algo --bloom 5 --rt60 90
```
_What it sounds like:_ A reverb that takes five seconds to reach full density. For the first few seconds after the source, you hear a thin, distant reflection; then the full reverb gradually fills in.

_DSP note:_ Bloom shapes the density envelope of the early reverb diffusion. A bloom value of 5 seconds means the all-pass diffuser network is fed input progressively over the first five seconds rather than all at once. The mathematical model is an exponential ramp of the diffusion feedback coefficient from 0 to its nominal value over the bloom duration.

---

**Recipe 47**
```bash
verbx render in.wav out/047_bloom_shimmer.wav --engine algo --bloom 3 --shimmer --shimmer-mix 0.4
```
_What it sounds like:_ A reverb that builds slowly into a shimmer cloud. The shimmer layer arrives as the reverb density fills in. Very cinematic.

_DSP note:_ Because shimmer feeds into the reverb input, the bloom envelope affects how the shimmer builds as well. In the first three seconds, there is very little shimmer because there is very little reverb density to pitch-shift. As the density grows, the shimmer grows with it. This creates a natural swell that feels less mechanical than fixed shimmer.

---

**Recipe 48**
```bash
verbx render in.wav out/048_tilt_up.wav --engine algo --tilt 6 --lowcut 120
```
_What it sounds like:_ A bright, clear reverb with the low frequencies removed. Useful on source material where the reverb tail would otherwise accumulate low-mid energy.

_DSP note:_ `--tilt 6` applies a +6dB/octave high-shelf tilt to the reverb output, progressively boosting frequencies as they increase. Combined with `--lowcut 120`, which removes everything below 120Hz, the result is a reverb that exists almost entirely in the mid-to-high frequency range.

---

**Recipe 49**
```bash
verbx render in.wav out/049_tilt_down.wav --engine algo --tilt -6 --highcut 4500
```
_What it sounds like:_ A dark reverb with a strong low-frequency bias. The tail rumbles and warms. Good for thunder, drums, and low-frequency sound design.

_DSP note:_ `--tilt -6` applies a -6dB/octave downward slope, attenuating frequencies as they increase. Combined with `--highcut 4500`, which removes everything above 4.5kHz, the result is a reverb confined to the bass and low-mid range.

---

**Recipe 50**
```bash
verbx render in.wav out/050_combo_extreme.wav --engine algo --duck --bloom 4 --tilt 4 --shimmer --beast-mode 8
```
_What it sounds like:_ Everything at once. Duck, bloom, shimmer, tilt, and high beast-mode. This is not a subtle reverb. It is a feature.

_DSP note:_ The signal chain here is complex: the source is passed into the algo engine with beast-mode 8 diffusion, then bloom shapes the density buildup over 4 seconds, then shimmer adds a pitch-shifted layer into the reverb feedback, then tilt tilts the output toward brightness, then duck dynamically attenuates the entire wet signal when the dry source is present. Each of these processes interacts with the others. You will likely need to A/B this against a simpler configuration to hear what each element is contributing.

> **Expert note:** When combining duck and bloom, be aware that the duck sidechain is typically keyed from the dry input, not the wet output. This means the duck engages based on the source dynamics, while the bloom is shaping the reverb density based on elapsed time. In cases where the source has a slow attack (strings, pads), the duck may not engage strongly enough to prevent reverb buildup during the bloom phase, resulting in a more forward reverb than expected. Adjust duck threshold to compensate.

---

## Section 6: Loudness and Output Format (Recipes 51-60)

Loudness normalization is one of those topics that sounds boring until you work in a context where it matters — and then it is everything. Streaming platforms, broadcast, cinema, and immersive audio all have different integrated loudness targets, true peak limits, and gating rules. This section covers the practical configurations for common delivery requirements, as well as the edge cases where you want no normalization at all.

Output format matters more than most engineers think. Float32 preserves more dynamic range than PCM24 in intermediate processing stages. Float64 is belt-and-suspenders overkill for most work but useful when accumulating multiple processing passes. These recipes cover the full range.

---

**Recipe 51**
```bash
verbx render in.wav out/051_lufs_24.wav --target-lufs -24 --target-peak-dbfs -2
```
_What it sounds like:_ Output normalized to -24 LUFS integrated loudness with -2 dBFS peak limit. This is a conservative broadcast target.

_DSP note:_ Integrated LUFS measurement gates silence and weights frequencies according to the ITU-R BS.1770 loudness model (which de-emphasizes very low and very high frequencies relative to mid frequencies). The measurement reflects perceived loudness rather than signal amplitude. `-24 LUFS` is the US broadcast standard; the European standard is `-23 LUFS`.

---

**Recipe 52**
```bash
verbx render in.wav out/052_lufs_18.wav --target-lufs -18 --target-peak-dbfs -1 --true-peak
```
_What it sounds like:_ Music streaming target loudness with true peak limiting. Appropriate for Spotify, Apple Music, and similar platforms.

_DSP note:_ True peak measurement uses oversampled analysis (typically 4x) to detect inter-sample peaks that sample-level measurement would miss. An inter-sample peak of -1 dBFS can produce a true peak well above 0 dBFS after digital-to-analog conversion, causing clipping in the playback chain. `--true-peak` engages a true peak limiter that accounts for this.

---

**Recipe 53**
```bash
verbx render in.wav out/053_sample_peak.wav --target-peak-dbfs -0.5 --sample-peak
```
_What it sounds like:_ Peak normalized to -0.5 dBFS at the sample level. This is a basic peak normalization without loudness weighting.

_DSP note:_ Sample peak normalization finds the single loudest sample in the file and applies uniform gain to bring it to the specified level. It ignores loudness perception entirely — a file that is -0.5 dBFS peak could be at -35 LUFS (very quiet) or -8 LUFS (very loud) depending on its dynamic range. Use this when you need predictable amplitude, not predictable loudness.

---

**Recipe 54**
```bash
verbx render in.wav out/054_per_pass.wav --repeat 4 --normalize-stage per-pass --repeat-target-lufs -22
```
_What it sounds like:_ Four repeat passes, each normalized to -22 LUFS before the next pass begins. The loudness remains consistent across passes but the texture accumulates.

_DSP note:_ `--repeat-target-lufs -22` sets the per-pass loudness target independently from any output-stage loudness target. This is useful for repeat chains where you want to prevent energy accumulation from causing clipping in intermediate passes while still having the final output normalized differently at the output stage.

---

**Recipe 55**
```bash
verbx render in.wav out/055_no_limiter.wav --engine algo --rt60 70 --no-limiter
```
_What it sounds like:_ Reverb output with no peak limiting. The output can exceed 0 dBFS. You need a float-format output file or downstream clipping control for this to be useful.

_DSP note:_ The `--no-limiter` flag bypasses the output-stage limiter and leaves the raw reverb output at its natural amplitude. On inputs with strong transients, the reverb energy can accumulate to levels well above 0 dBFS. This is only appropriate for intermediate processing stages where the output will be further processed before final delivery.

---

**Recipe 56**
```bash
verbx render in.wav out/056_float32.wav --engine conv --ir hall.wav --out-subtype float32
```
_What it sounds like:_ Hall convolution rendered to 32-bit float WAV. Transparent and suitable for further processing.

_DSP note:_ Float32 encoding provides approximately 24 bits of dynamic range at any given magnitude, with the advantage that the encoding range is not fixed — levels above 0 dBFS are representable without clipping. This makes float32 ideal for intermediate processing files. The cost is slightly larger files than PCM24 for equivalent audible quality.

---

**Recipe 57**
```bash
verbx render in.wav out/057_float64.wav --engine conv --ir hall.wav --out-subtype float64
```
_What it sounds like:_ Hall convolution rendered to 64-bit double float WAV. Identical in audible quality to float32 for most material.

_DSP note:_ Float64 provides approximately 52 bits of mantissa precision. The audible difference from float32 is essentially nonexistent for any material that remains in a normal dynamic range. Float64 is primarily useful for accumulation-heavy operations (many repeat passes, self-convolution chains) where floating-point rounding errors in float32 could accumulate to audible levels. For single-pass renders, float32 is sufficient.

---

**Recipe 58**
```bash
verbx render in.wav out/058_pcm24.wav --engine conv --ir hall.wav --out-subtype pcm24
```
_What it sounds like:_ Hall convolution rendered to 24-bit integer PCM. The standard delivery format for most professional audio workflows.

_DSP note:_ PCM24 provides a fixed dynamic range of approximately 144dB theoretical, 120dB practical. It cannot represent values above 0 dBFS without clipping. Unlike float formats, clipping in PCM24 is hard — the value wraps or clips at the encoding limit. For final delivery files, PCM24 is usually appropriate. For intermediate processing, prefer float32 or float64.

---

**Recipe 59**
```bash
verbx render in.wav out/059_peak_input.wav --engine algo --output-peak-norm input
```
_What it sounds like:_ The reverb output is gain-adjusted so its peak matches the peak of the input file. Useful for comparative listening.

_DSP note:_ `--output-peak-norm input` measures the peak sample amplitude of the input file (`in.wav`) and applies the gain necessary to bring the output to the same peak level. This is reference-relative normalization — the output level is defined by the input level rather than by an absolute target. Useful when batch-processing variations and wanting all outputs at consistent relative levels.

---

**Recipe 60**
```bash
verbx render in.wav out/060_peak_target.wav --engine algo --output-peak-norm target --output-peak-target-dbfs -9
```
_What it sounds like:_ The reverb output is peak-normalized to a specific absolute level of -9 dBFS.

_DSP note:_ `--output-peak-norm target` enables absolute peak normalization, and `--output-peak-target-dbfs -9` sets the target. -9 dBFS is a common bus-level target for mixing — it provides substantial headroom for subsequent processing while keeping the signal well above the noise floor. This is different from LUFS normalization: -9 dBFS peak does not guarantee any particular loudness.

---

## Section 7: Synthetic IR Generation (Recipes 61-70)

Generating synthetic impulse responses is where `verbx` overlaps with acoustic modeling. Rather than measuring a real space, these commands synthesize IRs from mathematical models — Feedback Delay Networks, stochastic noise models, and modal synthesis. The results range from plausible-but-not-real-sounding rooms to completely abstract spectral transformers.

The practical value of synthetic IRs is flexibility: you can tune the RT60, the modal density, the fundamental resonant frequency, and the reflection pattern, all without access to a real space. For science fiction and fantasy sound design, this is invaluable. For electronic music, it allows creating reverbs that are tuned to the key of the track. For theatrical foley, it allows matching a reverb exactly to the visual environment.

The workflow in recipes 61-70 forms a complete IR generation and processing pipeline: generate, process, analyze, then use.

---

**Recipe 61**
```bash
verbx ir gen out/061_ir_hybrid.wav --mode hybrid --length 120 --seed 61
```
_What it sounds like:_ An IR for a synthetic room with hybrid early-reflection and statistical-tail modeling. Sounds plausible but not identifiably real.

_DSP note:_ Hybrid mode combines a deterministic early-reflection model (which generates discrete room reflections based on a shoe-box room approximation) with a statistical late-reverberation model (which uses a filtered noise model for the tail). The seed ensures reproducibility — the same seed and parameters will always produce the same IR.

---

**Recipe 62**
```bash
verbx ir gen out/062_ir_fdn.wav --mode fdn --length 180 --fdn-lines 12 --seed 62
```
_What it sounds like:_ An FDN-derived IR. Dense, smooth, mathematically perfect in its diffusion. Does not sound like a real room but sounds like a very convincing synthetic space.

_DSP note:_ Feedback Delay Networks use a matrix of delay lines with a feedback matrix (usually unitary, to preserve energy). 12 FDN lines produces a dense, highly decorrelated late-reverb tail. The Hadamard or Householder matrix typically used for the feedback connections ensures maximum decorrelation with minimal computation. The resulting IR captures the FDN's impulse response rather than running the FDN in real time.

---

**Recipe 63**
```bash
verbx ir gen out/063_ir_stochastic.wav --mode stochastic --length 240 --density 1.8 --seed 63
```
_What it sounds like:_ A stochastic IR — the late reverb tail is synthesized from a filtered and shaped noise burst. Density 1.8 produces a very dense, continuous tail.

_DSP note:_ Stochastic IR synthesis models the late-field reverberation as an exponentially decaying noise process, shaped by a target spectral envelope. Density 1.8 controls the number of reflections per unit time in the early part of the stochastic model. Higher density values produce smoother, more "diffuse" reverb, while lower values produce grainier, more textured tails.

---

**Recipe 64**
```bash
verbx ir gen out/064_ir_modal.wav --mode modal --length 90 --modal-count 96 --seed 64
```
_What it sounds like:_ A modal IR — synthesized from 96 resonant modes. Depending on the source material, you will hear clear resonant frequencies in the reverb tail.

_DSP note:_ Modal synthesis generates an IR as a sum of decaying sinusoids, each at a different frequency (mode) with a different amplitude and decay rate. 96 modes is relatively dense — individual modes may not be distinguishable. At lower mode counts (8-16), the individual resonances are clearly audible and the "room" sounds like a resonator rather than a space.

---

**Recipe 65**
```bash
verbx ir gen out/065_ir_tuned.wav --mode modal --length 120 --f0 64Hz --seed 65
```
_What it sounds like:_ A modal IR tuned to a fundamental frequency of 64Hz (roughly a C2). The reverb will reinforce pitches harmonically related to C2.

_DSP note:_ Setting `--f0 64Hz` anchors the modal synthesis so that the lowest-frequency mode is at 64Hz, and subsequent modes are placed at harmonically related frequencies (or near-harmonic distributions, depending on the mode generation algorithm). This creates a reverb that colors the source material based on its harmonic relationship to C2. For music in C, this is complementary; for music in other keys, it can create tension or harmonic muddiness.

---

**Recipe 66**
```bash
verbx ir gen out/066_ir_from_input.wav --mode hybrid --analyze-input in.wav --seed 66
```
_What it sounds like:_ A synthetic IR derived from the spectral and dynamic characteristics of the input file itself. The reverb will have a character that is somehow "related" to the source.

_DSP note:_ `--analyze-input` extracts the spectral envelope, dynamic range, and tempo/transient information from the input file and uses this to shape the IR synthesis parameters. The result is an IR that resonates sympathetically with the source material. This is useful for creating reverbs that feel "organic" to a specific piece of music or audio.

---

**Recipe 67**
```bash
verbx ir gen out/067_ir_resonator.wav --mode hybrid --resonator --resonator-mix 0.6 --seed 67
```
_What it sounds like:_ A hybrid IR with additional resonator modes mixed in at 60%. The tail has clear resonant peaks that color whatever is processed through it.

_DSP note:_ The resonator adds a set of high-Q bandpass filters to the IR synthesis chain. `--resonator-mix 0.6` blends the resonator output with the base hybrid IR at 60%. The resonant frequencies are determined by the mode generation algorithm and the seed. This is closer to a comb filter or spring reverb character than a room reverb.

---

**Recipe 68**
```bash
verbx ir process out/067_ir_resonator.wav out/068_ir_processed.wav --tilt -4 --normalize peak
```
_What it sounds like:_ The resonator IR processed with a dark tilt and peak normalized. Ready to use as a dark, resonant convolution IR.

_DSP note:_ IR processing applies DSP to the impulse response itself rather than to the output of the reverb. Tilting an IR by -4dB/octave means the convolution will produce a dark, bass-heavy reverb regardless of what the source material sounds like. Peak normalizing the IR ensures consistent output levels when the IR is used in recipe 70.

---

**Recipe 69**
```bash
verbx ir analyze out/068_ir_processed.wav --json-out out/069_ir_analysis.json
```
_What it sounds like:_ This produces no audio — it generates a JSON analysis report of the processed IR.

_DSP note:_ IR analysis typically measures RT20, RT30, RT60 (derived from the decay curve), early decay time, clarity (C50, C80), and spectral centroid as a function of time. This data is useful for predicting how the IR will behave as a reverb and for comparing multiple IRs quantitatively. The JSON output can be imported into analysis or visualization tools.

---

**Recipe 70**
```bash
verbx render in.wav out/070_ir_render.wav --engine conv --ir out/068_ir_processed.wav --repeat 2
```
_What it sounds like:_ The input processed with the custom IR we generated and processed in recipes 67-68, repeated twice. This completes the synthetic IR pipeline.

_DSP note:_ This is the payoff of the IR generation workflow. The convolution engine uses the processed synthetic IR as if it were a measured room IR. Two repeat passes mean the input is convolved twice with the resonant, dark-tilted synthetic IR, producing a dense, colored reverb with the resonator's character amplified.

---

## How to Listen: Spatial and Multichannel

Multichannel listening requires a properly calibrated monitoring chain. Without it, you cannot judge the spatial balance of a 5.1 or 7.1 render — you are listening to a subset of the output. If you are working on multichannel material without a multichannel monitor system, use a downmix for reference, but always verify critical decisions on proper playback.

For the algorithmic multichannel recipes, pay attention to the relationship between the center channel and the surrounds. In ambience and reverb for picture, the surrounds are typically at a lower level than the fronts. Over-reverberant surrounds are one of the most common mixing errors in immersive audio.

---

## Section 8: Multichannel and Spatial Processing (Recipes 71-80)

Multichannel reverb is a workflow unto itself. The spatial distribution of reverb energy — which sounds arrive from which directions, how the early reflections establish the size and shape of the space, how the late field wraps around the listener — is as important as the RT60 or the dry/wet balance. Bad multichannel reverb sounds like stereo reverb playing from four speakers. Good multichannel reverb sounds like being inside a space.

In cinema mixing, the surround channels carry ambient reverb and environmental information; the front channels carry the action and dialogue. The LFE carries bass and sub-bass reverb energy. The workflow for managing these independently in convolution is through matrix IRs — a different impulse response for each output channel.

---

**Recipe 71**
```bash
verbx render in_5p1.wav out/071_5p1_conv.wav --engine conv --ir ir_5p1_matrix.wav --ir-matrix-layout output-major
```
_What it sounds like:_ A 5.1 convolution reverb with a properly configured matrix IR. Each output channel gets its own reverb character derived from the appropriate IR channel.

_DSP note:_ Output-major layout: the IR file contains IR channels ordered as (IR for output L from all inputs), (IR for output R from all inputs), (IR for output C from all inputs), and so on. This is the standard layout for most multichannel IR libraries captured with separate microphone positions.

---

**Recipe 72**
```bash
verbx render in_7p1.wav out/072_7p1_conv.wav --engine conv --ir ir_7p1_matrix.wav --ir-matrix-layout output-major
```
_What it sounds like:_ The same as recipe 71 but for 7.1 — eight output channels. The additional side channels extend the envelopment around the listener.

_DSP note:_ 7.1 requires an IR with at least 8 channels (or a matrix of 8 output IRs). The side-surround channels in 7.1 typically receive IRs with slightly later direct energy and more diffuse early reflections than the rear surrounds, creating a front-to-back sense of depth.

---

**Recipe 73**
```bash
verbx render in_5p1.wav out/073_5p1_algo.wav --engine algo --rt60 85 --width 1.6
```
_What it sounds like:_ A 5.1 algorithmic reverb. Faster to compute than convolution and more flexible for creative work.

_DSP note:_ Algorithmic multichannel reverb applies the reverb engine independently or jointly across the output channels, depending on the engine's internal architecture. Width at 1.6 in a 5.1 context affects the spread between left and right channels but does not directly control the front-to-back depth, which is managed by the engine's channel assignment logic.

---

**Recipe 74**
```bash
verbx render in_7p1.wav out/074_7p1_algo_beast.wav --engine algo --beast-mode 12 --repeat 3
```
_What it sounds like:_ High-density algo reverb on a 7.1 input, with three repeat passes. Very dense, enveloping. Not for dialogue — this is for music or texture.

_DSP note:_ Beast-mode 12 in multichannel increases the diffusion network complexity proportionally across channels. Three repeat passes in 7.1 produce a very large output file. Ensure you have sufficient storage before running this recipe.

---

**Recipe 75**
```bash
verbx render in_5p1.wav out/075_5p1_freeze.wav --engine algo --freeze --start 3 --end 4.5 --repeat 2
```
_What it sounds like:_ A frozen 5.1 pad extracted from the 3-4.5 second range, repeated twice. Creates a multichannel ambient drone.

_DSP note:_ Freeze in multichannel mode extracts the specified window from all channels simultaneously, maintaining the spatial coherence of the frozen segment. The frozen loop, when processed through the algo reverb, produces a reverberant pad that preserves the original spatial distribution of the source material.

---

**Recipe 76**
```bash
verbx render in_7p1.wav out/076_7p1_shimmer.wav --engine algo --shimmer --shimmer-mix 0.45
```
_What it sounds like:_ Shimmer on 7.1. The pitch-shifted layer fills the entire surround field. Very immersive, slightly overwhelming on material with strong low-frequency content.

_DSP note:_ The shimmer feedback path in multichannel wraps around all channels. The pitch-shifted output from any given channel feeds back into all channels based on the internal routing matrix. This means shimmer in multichannel is not simply the stereo shimmer experience multiplied — it has a distinct spatial character because the feedback paths are longer and more complex.

---

**Recipe 77**
```bash
verbx render in_5p1.wav out/077_5p1_target.wav --target-lufs -23 --target-peak-dbfs -2
```
_What it sounds like:_ 5.1 loudness normalization to the European broadcast standard (-23 LUFS) with -2 dBFS peak limit.

_DSP note:_ Integrated LUFS for multichannel uses the channel-weighting defined in ITU-R BS.1770, which applies -1.5 dB attenuation to surround channels (Ls, Rs) in the loudness measurement. This means a 5.1 mix at -23 LUFS measured will have higher actual SPL in the surrounds than a stereo mix at the same measured loudness value. Be aware of this when comparing multichannel and stereo versions.

---

**Recipe 78**
```bash
verbx render in_7p1.wav out/078_7p1_fullscale.wav --output-peak-norm full-scale --out-subtype float32
```
_What it sounds like:_ 7.1 output normalized to full scale in float32 format. Every bit of dynamic range used.

_DSP note:_ `--output-peak-norm full-scale` applies gain to bring the loudest sample in the entire multichannel file to 0 dBFS. In float32, this is representable without clipping. This is an intermediate processing format — do not deliver float32 full-scale as a final output without confirming the downstream system can handle it.

---

**Recipe 79**
```bash
verbx render in_5p1.wav out/079_5p1_tailcap.wav --engine conv --ir ir_5p1_matrix.wav --tail-limit 12
```
_What it sounds like:_ A 5.1 convolution reverb with a 12-second tail cap. Short enough that the surround field does not overwhelm the dry signal.

_DSP note:_ 12 seconds is generous for most indoor spaces but short for cathedrals and large outdoors captures. Truncating at 12 seconds in multichannel reduces processing time significantly for long IRs. The fade-out applied at the truncation point should be long enough to avoid a click but short enough not to audibly shorten the perceived tail.

---

**Recipe 80**
```bash
verbx render in_7p1.wav out/080_7p1_longtail.wav --engine conv --ir ir_7p1_matrix.wav --tail-limit 240
```
_What it sounds like:_ A 7.1 convolution with a four-minute tail cap. This is a large file and a long render. The surround field will sustain for an extremely long time.

_DSP note:_ 240-second tail-limit on an 8-channel 48kHz float32 file is approximately 7.5GB of output. Monitor disk space. The long tail is primarily useful for theatrical contexts where the reverb needs to sustain through a long pause or for ambisonic processing where the tail contributes to a sense of space even at very low levels.

---

## Section 9: Analysis, Suggestion, and Batch Tools (Recipes 81-90)

The analysis and batch tools are where `verbx` starts to function as a production pipeline rather than a single-render tool. These commands are less dramatic than the extremes in sections 1-8, but they are often more useful in practice. Understanding what your audio contains before applying reverb is good engineering. Batching a large set of renders with known parameters is how you scale up to feature-length projects.

The `suggest` command is worth studying — it uses analysis of the input to generate parameter recommendations, which can be a useful starting point for creative decisions or a sanity check on aggressive settings.

---

**Recipe 81**
```bash
verbx render in.wav out/081_predelay_8d.wav --pre-delay 1/8D --bpm 96 --engine algo
```
_What it sounds like:_ The reverb is delayed by a dotted eighth note at 96 BPM (approximately 469ms). The reverb arrives on the off-beat, which can create a rhythmic push-pull effect.

_DSP note:_ Tempo-synced pre-delay uses the BPM value to calculate the delay in milliseconds. A dotted eighth at 96 BPM = (60000 / 96) * 1.5 / 2 ≈ 469ms. This pre-delay duration means the reverb onset aligns with the dotted eighth note position. At 96 BPM, this falls between beats in a way that complements the rhythm without fighting it.

---

**Recipe 82**
```bash
verbx render in.wav out/082_predelay_triplet.wav --pre-delay 1/16T --bpm 132 --engine algo
```
_What it sounds like:_ Pre-delay synced to a sixteenth-note triplet at 132 BPM. The reverb arrives in an unusual rhythmic position — syncopated and slightly off-center.

_DSP note:_ A sixteenth-note triplet at 132 BPM = (60000 / 132) / 4 * (2/3) ≈ 75.8ms. This is within the Haas fusion zone (above 30ms), so it may or may not be perceived as a distinct echo — the perception depends on the source material and the reverb level.

---

**Recipe 83**
```bash
verbx analyze in.wav --lufs --edr --json-out out/083_analysis.json
```
_What it sounds like:_ No audio output. Produces a JSON file with integrated loudness (LUFS) and early decay rate (EDR) measurements of the input file.

_DSP note:_ EDR (Energy Decay Relief) is a time-frequency representation of how energy decays across the frequency spectrum. It is calculated from the short-time power spectral density as a function of time. High EDR at a specific frequency means that frequency decays more slowly than others — this identifies resonant problems in room recordings and helps predict where synthetic reverbs will be tonally uneven.

---

**Recipe 84**
```bash
verbx analyze in.wav --frames-out out/084_frames.csv --edr
```
_What it sounds like:_ No audio output. Produces a CSV file with per-frame EDR measurements. Suitable for import into data analysis or visualization tools.

_DSP note:_ Per-frame analysis provides time-resolved measurement at the frame rate defined by the analysis window size (typically 512 or 1024 samples). The CSV format is useful for plotting or further analysis in Python, R, or MATLAB. The EDR frames show how the energy decay distribution changes over time — useful for identifying time-varying resonances in long recordings.

---

**Recipe 85**
```bash
verbx suggest in.wav
```
_What it sounds like:_ No audio output. Prints parameter suggestions to the console based on analysis of the input file.

_DSP note:_ The suggestion engine analyzes the input's dynamic range, spectral centroid, transient density, and estimated RT60 (if the input contains room ambience), then recommends `--rt60`, `--damping`, `--pre-delay-ms`, and `--wet`/`--dry` values appropriate to the input's character. These are starting points, not definitive recommendations. Use them to anchor your creative decisions, then adjust from there.

---

**Recipe 86**
```bash
verbx batch template > out/086_manifest.json
```
_What it sounds like:_ No audio output. Generates a template batch manifest JSON file.

_DSP note:_ The manifest file defines a list of render jobs with all parameters specified. The template contains example entries for each engine type and common parameter configurations. Edit the manifest to define your actual batch before running recipes 87-88.

---

**Recipe 87**
```bash
verbx batch render manifest.json --jobs 8 --schedule longest-first --retries 1
```
_What it sounds like:_ Executes all renders defined in manifest.json using 8 parallel workers, processing the longest renders first, with one retry on failure.

_DSP note:_ `--schedule longest-first` is a work-stealing heuristic that prioritizes long renders early so that parallel workers remain active throughout the batch. If you schedule shortest-first, you may complete many small renders quickly but then have only one long render running at the end, leaving 7 workers idle. Longest-first minimizes total wall-clock time for heterogeneous batches.

---

**Recipe 88**
```bash
verbx batch render manifest.json --jobs 4 --schedule shortest-first --dry-run
```
_What it sounds like:_ No audio output. Dry run — prints what would be rendered without executing anything.

_DSP note:_ Always run `--dry-run` first on a new manifest to verify that all paths are correct and all parameters are valid before committing to a long batch render. The dry-run output shows the resolved parameters for each job, including any defaults that will be applied, which makes it easy to spot configuration errors.

---

**Recipe 89**
```bash
verbx cache info
```
_What it sounds like:_ No audio output. Prints information about the convolution cache — what is cached, how large, when last accessed.

_DSP note:_ `verbx` caches partitioned IR FFTs to avoid recomputing them on repeated renders with the same IR. The cache is particularly valuable for batch renders where the same IR is used across many jobs. Cache entries include the IR file path, partition size, sample rate, and the cached FFT data. Stale cache entries (from modified IRs) are detected by file modification time.

---

**Recipe 90**
```bash
verbx cache clear
```
_What it sounds like:_ No audio output. Clears the convolution cache.

_DSP note:_ Clearing the cache forces recomputation of IR FFTs on the next render. Do this after modifying IR files that you have previously used, or when disk space is constrained. The cache is stored in a platform-specific location (typically in the user's application data directory). Cache clearing does not affect output files or manifests.

---

## Section 10: Lucky-Mode Wildcards (Recipes 91-100)

Lucky mode is what happens when you hand the parameter space to the engine and say "find something interesting." It randomizes settings within configured bounds and generates a specified number of variations in a single run. It is not random in the degenerate sense — the parameter space is constrained to avoid unlistenable results — but it covers territory you would not explore by hand.

The practical value of lucky mode is speed of discovery. When you do not know what you want, running lucky with a good seed can surface directions you would not have chosen deliberately. The seed is important: fix it for reproducibility, vary it for exploration. Documenting seeds that produce interesting results is worth doing — a single seed number represents an entire family of related variations.

Lucky mode is also useful at the end of a project when you think you are done. Run a lucky set on your primary source material. One of the variations will be better than what you have. This is not always true. But it is true often enough to make the practice worthwhile.

---

**Recipe 91**
```bash
verbx render in.wav out/lucky.wav --lucky 5 --lucky-out-dir out/lucky_01
```
_What it sounds like:_ Five random variations of the input with randomized parameters, written to the `out/lucky_01` directory.

_DSP note:_ Without `--lucky-seed`, each run produces different results. Lucky mode randomizes within safe parameter ranges — RT60 is bounded to prevent extremely short or extremely long tails, beast-mode is bounded to avoid instability, wet/dry ratios are biased toward useful mixes. The five files are numbered and can be compared directly.

---

**Recipe 92**
```bash
verbx render in.wav out/lucky.wav --lucky 10 --lucky-out-dir out/lucky_02 --lucky-seed 2026
```
_What it sounds like:_ Ten variations with a fixed seed. Running this command again will produce identical outputs. Use this when you want to share a specific lucky set with collaborators.

_DSP note:_ The seed initializes the pseudo-random number generator used for parameter selection. A fixed seed ensures deterministic parameter sampling, meaning the same `--lucky-seed` on the same input file will always produce the same output files. Vary the seed to explore different families of variations.

---

**Recipe 93**
```bash
verbx render in.wav out/lucky.wav --lucky 25 --lucky-out-dir out/lucky_03 --device auto
```
_What it sounds like:_ Twenty-five variations with automatic GPU/CPU device selection for rendering. If a GPU is available, it will be used; otherwise falls back to CPU.

_DSP note:_ `--device auto` allows `verbx` to select the most efficient compute device. For convolution-heavy lucky sets, GPU acceleration can significantly reduce render time. For algo-only sets, the difference is less pronounced. The device selection is made once at the start of the batch and applied to all 25 renders.

---

**Recipe 94**
```bash
verbx render in.wav out/lucky.wav --lucky 50 --lucky-out-dir out/lucky_04 --no-progress
```
_What it sounds like:_ Fifty variations with progress output suppressed. This is designed for batch environments where terminal output is logged.

_DSP note:_ `--no-progress` disables the progress bar and per-file status output. In CI/CD pipelines, Docker containers, or other environments where terminal control codes cause issues in log output, suppressing progress is important for log readability. Output errors and warnings are still printed.

---

**Recipe 95**
```bash
verbx render in.wav out/lucky.wav --lucky 8 --lucky-out-dir out/lucky_05 --engine algo
```
_What it sounds like:_ Eight algorithmic reverb variations. Lucky mode constrained to the algo engine — no convolution.

_DSP note:_ Constraining lucky mode to a specific engine narrows the parameter space and produces a more consistent set of variations. All eight files will share the same engine character (algo diffusion) but vary in RT60, modulation, beast-mode, pre-delay, and other algo-specific parameters.

---

**Recipe 96**
```bash
verbx render in.wav out/lucky.wav --lucky 8 --lucky-out-dir out/lucky_06 --engine conv --ir hall.wav
```
_What it sounds like:_ Eight convolution variations using the hall IR. Lucky mode varies the convolution parameters — partition size, tail limit, repeat, normalization — rather than the IR itself.

_DSP note:_ When lucky mode is constrained to the conv engine with a specific IR, the randomization applies to the processing around the convolution: pre-delay, wet/dry balance, tail truncation, repeat count, and loudness. The hall IR's character is consistent across all eight; the variations represent different ways of using that IR.

---

**Recipe 97**
```bash
verbx render in.wav out/lucky.wav --lucky 12 --lucky-out-dir out/lucky_07 --self-convolve
```
_What it sounds like:_ Twelve self-convolution variations. Wild. Some will be unusable; some will be remarkable. The variance is higher in self-convolve lucky mode than in any other configuration.

_DSP note:_ Self-convolve lucky mode varies the beast-mode level, tilt, tail limit, shimmer parameters, and repeat count within ranges that produce interesting results rather than simply clipped noise. That said, the signal path for self-convolution is inherently less stable than algo or conv modes, so the variance between "good" and "bad" outputs is higher. Plan to audition all 12.

---

**Recipe 98**
```bash
verbx render in.wav out/lucky.wav --lucky 15 --lucky-out-dir out/lucky_08 --target-lufs -20
```
_What it sounds like:_ Fifteen variations, each normalized to -20 LUFS. All outputs are at a consistent loudness, making A/B comparison easier.

_DSP note:_ Applying `--target-lufs` to lucky mode normalizes each output independently after rendering. This is useful for comparative listening because the loudness differences between variations are removed, allowing you to evaluate the texture and character rather than the level. -20 LUFS is a comfortable listening level for extended auditioning.

---

**Recipe 99**
```bash
verbx render in.wav out/lucky.wav --lucky 30 --lucky-out-dir out/lucky_09 --out-subtype float32
```
_What it sounds like:_ Thirty float32 variations. The float format preserves headroom for downstream processing — useful if you plan to layer or process the lucky outputs further.

_DSP note:_ Using float32 for lucky output means you do not need to worry about clipping during the render — the format accommodates values above 0 dBFS. This is particularly useful with self-convolve and beast-mode combinations in lucky mode, where the amplitude is less predictable. Process and normalize at the final stage of your signal chain rather than at the lucky render stage.

---

**Recipe 100**
```bash
verbx render in.wav out/lucky.wav --lucky 100 --lucky-out-dir out/lucky_10 --lucky-seed 404
```
_What it sounds like:_ One hundred variations, fixed seed 404. The largest single lucky run in this cookbook. Expect a long render time and a large directory.

_DSP note:_ A 100-variation lucky set with a fixed seed is a complete exploration of one corner of the parameter space. With seed 404, the parameter sampling will explore the same 100 points in the parameter space every time. This is useful as a reference set — a known, reproducible collection of 100 reverb treatments for a given input. At 48kHz stereo float32, even 60-second outputs will total several gigabytes. Plan accordingly.

> **Expert note:** The seed 404 is not special. But it is documented here as a reference: if you run recipe 100 on the same input file and do not get the same output, something has changed in the parameter sampling algorithm, the random number generator implementation, or the engine version. Treat the output of seed 404 as a regression test for reproducibility.

---

## Notes for Large Runs

Start with a small lucky count (3-5) to confirm runtime and output size before committing to a 50 or 100-variation run.

Use `--no-progress` in log-friendly batch environments. The progress bar uses terminal control codes that corrupt plain-text logs.

Fix `--lucky-seed` whenever reproducibility matters. Document the seed alongside the source file and any post-processing applied.

Monitor storage. Long tails, high repeat counts, and multichannel outputs in float format accumulate quickly. A batch of 100 stereo float32 files at 3 minutes each is approximately 34GB at 48kHz.

The `verbx cache info` and `verbx cache clear` commands (recipes 89-90) are useful before and after large batch runs. Clear the cache if disk space is constrained; leave it in place if you will be reusing the same IRs across multiple batches.

When a run produces something you want to keep, immediately rename the output file out of the numbered sequence. The output directory will be overwritten if you rerun the batch with the same output path.
