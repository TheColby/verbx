<p align="center">
  <img src="docs/assets/verbx_logo.png" width="420" />
</p>

# verbx

**Colossal 64-bit spatial audio reverberator, accelerated with CUDA and Metal.**

`verbx` is a research-grade Python CLI for creating reverb effects that range from subtle room placement to cathedral-scale tails 3600 seconds long. It handles the complete reverb workflow: ingesting and generating impulse responses, processing audio through two independent engines, controlling every parameter with time-varying automation, delivering loudness-targeted multichannel output, reducing late-room smear with deterministic dereverberation, producing reproducible analysis artifacts at every step, and now previewing spaces in realtime from CLI-selectable audio devices.

You can batch reverberate a directory of audio files to create lush Dolby Atmos beds. or use it as part of your corpus-augmentation workflow for audio AI projects. 

Under the hood, everything runs in 64-bit floating point. The algorithmic engine is built around a configurable Feedback Delay Network with eight matrix families, multiband decay, and optional time-varying behavior. The convolution engine uses partitioned FFT with optional CUDA acceleration and full M-input-to-N-output matrix routing. Both engines share the same diffusion, shimmer, ducking, freeze, loudness, and spatial controls.

The latest `v0.7.7` work also starts to bridge pure parametric design with
explicit acoustics. There is now a reusable room-geometry model for dimensions,
materials, source/listener placement, Bolt-style proportion warnings, and RT60
to rectangular-room inversion via `verbx room-model`.

This is not a "set RT60 and go" tool. The parameter surface is wide by design. Most users start with three flags and expand from there.

For AI workflows, `verbx` is also a strong command-line tool for deterministic audio data augmentation and voice-model robustness testing. You can generate reproducible reverberant variants for ASR/TTS/speaker pipelines, keep split-safe metadata, and batch large render sets from manifests.

```bash
# A room that no physical building has ever had. RT60 = 120 seconds.
verbx render voice.wav out.wav \
  --engine algo --rt60 120 --wet 0.99 --dry 0.01 \
  --fdn-lines 32 --fdn-matrix tv_unitary --fdn-tv-rate-hz 0.30 \
  --shimmer --shimmer-semitones 12 --shimmer-mix 0.45 \
  --bloom 2.8 --tilt 2.0
```

---

## Instant Sonic Gratification

If you want immediate results with minimal decision-making, run this:

```bash
git clone https://github.com/TheColby/verbx.git && cd verbx && \
./install.sh --prefix "$HOME/.local" && \
verbx render ../in.wav out.wav --engine algo --rt60 120 --wet 0.99 --dry 0.01
```

That gives you an absurd long-tail render immediately. If `verbx` is not on PATH after install, run:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Quick Start

If `verbx` is already installed, this is the fastest way to start:

```bash
verbx render input.wav output.wav --engine algo --rt60 2.5 --wet 0.3 --dry 0.7
```

This applies a natural-sounding 2.5-second algorithmic reverb. Output is written to `output.wav`, with analysis at `output.wav.analysis.json`.

Need a live preview instead of an offline bounce?

```bash
verbx realtime --engine algo --input-device "Built-in Microphone" \
  --output-device "Headphones" --rt60 20 --freeze --shimmer \
  --fdn-matrix tv-unitary --fdn-tv-rate-hz 0.35 --fdn-tv-depth 0.12 \
  --lowcut 120 --highcut 9000 --tilt 1.5 --duration 10
```

Initial realtime mode runs either direct convolution from `--ir` or an
algorithmic proxy IR rendered once and monitored through the streaming
convolver. It is meant for auditioning spaces and tails, not yet for the full
automation/batch feature set. Realtime `--freeze` is an honest approximation:
it renders a long self-sustaining proxy tail for live auditioning rather than
reusing the offline segment-freeze operator.

If you need to install first on macOS:

```bash
brew tap thecolby/verbx
brew install thecolby/verbx/verbx
verbx quickstart
```

Need starting settings for your file?

```bash
verbx suggest input.wav
verbx quickstart
```

### Five Runnable Examples

```bash
# 1. Natural room — voice or piano in a medium hall
verbx render in.wav hall.wav --engine algo --rt60 2.0 --wet 0.25 --dry 0.8 --pre-delay-ms 18

# 2. Convolution with a real IR — character follows the space you measured
verbx render in.wav conv.wav --engine conv --ir hall_ir.wav --partition-size 16384

# 3. Shimmer pad — pitch-shifted ambient wash, good for synths
verbx render in.wav shimmer.wav --engine algo --rt60 12 --wet 0.85 \
  --shimmer --shimmer-semitones 12 --shimmer-mix 0.35 --bloom 2.0

# 4. Broadcast loudness target — -23 LUFS, -1 dBTP true peak
verbx render in.wav broadcast.wav --target-lufs -23 --true-peak --target-peak-dbfs -1

# 5. Extreme ambient — 90-second tail, slow evolution, near-frozen
verbx render in.wav ambient.wav --engine algo --rt60 90 --wet 0.92 \
  --fdn-matrix tv_unitary --fdn-tv-rate-hz 0.08 --bloom 2.0 --tilt 0.8
```

```bash
# Output-definition presets (default is HD)
verbx render in.wav out_hd.wav --engine conv --ir hall_ir.wav
verbx render in.wav out_md.wav --engine conv --ir hall_ir.wav --quality-preset md
verbx render in.wav out_sd.wav --engine conv --ir hall_ir.wav --quality-preset sd
```

## Full Installation Instructions

**With uv (fastest):**

```bash
git clone https://github.com/TheColby/verbx.git && cd verbx
uv sync && uv run verbx --help
```

For realtime audio device support:

```bash
uv sync --extra realtime
```

**With pip + venv:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e . && verbx --help
```

For realtime audio device support:

```bash
pip install -e ".[realtime]"
```

**With the install script (installs man pages too):**

```bash
./install.sh --prefix "$HOME/.local"
verbx --help && man verbx-render
man verbx-dereverb
```

**With Homebrew (macOS):**

```bash
brew tap thecolby/verbx
brew install thecolby/verbx/verbx
verbx version
```

Official tap repository: `TheColby/homebrew-verbx`.

For local maintainer testing, you can also install from the in-repo formula:

```bash
brew install --build-from-source ./packaging/homebrew/verbx.rb
```

**Requirements:** Python 3.11+, `libsndfile` on system path. Optional: `numba` (faster algorithmic path), `cupy` (CUDA convolution), `h5py` (SOFA import/extract via `verbx ir sofa-*`), `sounddevice` (realtime input/output via `verbx realtime`).

Homebrew maintainer details: [`docs/HOMEBREW.md`](docs/HOMEBREW.md)

If `verbx` is not found after install, add `~/.local/bin` to your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.zshrc or ~/.bashrc
```

## Python API (Research Workflows)

Use `verbx` as a library when you need notebook/pipeline integration:

```python
from verbx.api import analyze_file, generate_ir, render_file
from verbx.config import RenderConfig
from verbx.ir import IRGenConfig

report = render_file("dry.wav", "wet.wav", RenderConfig(engine="algo", rt60=2.5, wet=0.7))
ir_audio, ir_sr, ir_meta = generate_ir(IRGenConfig(mode="fdn", duration=3.0, sr=48000))
metrics = analyze_file("wet.wav", include_loudness=True)
```

## Audio Examples

Rendered examples are included in [`examples/audio/`](examples/audio/). The pack is now delivered at 48 kHz, PCM24. Most examples are stereo; the utility click and short hybrid IR files are mono. The shimmer-heavy examples were re-rendered at this higher rate specifically to remove the grit from the older 24 kHz / PCM16 pack.

GitHub repository README pages do not provide reliable inline audio controls. The `Play`
links below open each asset directly in the browser's native media player with one click.

### Utility and verification files

| File | Play | Description |
|------|------|-------------|
| [`dry_click.wav`](examples/audio/dry_click.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/dry_click.wav) | One-shot dry click reference for sanity checks |
| [`dry_click_reverbed.wav`](examples/audio/dry_click_reverbed.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/dry_click_reverbed.wav) | Reverberated click for immediate A/B verification |
| [`hybrid_ir_short.wav`](examples/audio/hybrid_ir_short.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/hybrid_ir_short.wav) | Short hybrid IR asset used in quick convolution demos |

### Realistic dry/wet example pairs

| File | Play | Description |
|------|------|-------------|
| [`realistic_speech_dry.wav`](examples/audio/realistic_speech_dry.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/realistic_speech_dry.wav) | Dry speech source used for room and plate examples |
| [`realistic_speech_room.wav`](examples/audio/realistic_speech_room.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/realistic_speech_room.wav) | Natural speech room render |
| [`realistic_music_dry.wav`](examples/audio/realistic_music_dry.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/realistic_music_dry.wav) | Dry music source used for ambient and shimmer examples |
| [`realistic_music_hall.wav`](examples/audio/realistic_music_hall.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/realistic_music_hall.wav) | Natural concert-hall style music render, re-tuned for a cleaner less congested tail |
| [`realistic_drums_dry.wav`](examples/audio/realistic_drums_dry.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/realistic_drums_dry.wav) | Dry drum source used for room and cathedral examples |
| [`realistic_drums_room.wav`](examples/audio/realistic_drums_room.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/realistic_drums_room.wav) | Natural drum room render |

### Extreme range demos

| File | Play | Description | Key settings |
|------|------|-------------|--------------|
| [`extreme_cathedral_drums.wav`](examples/audio/extreme_cathedral_drums.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/extreme_cathedral_drums.wav) | Drums → 8s Hadamard FDN cathedral | `--rt60 8.0 --fdn-lines 16 --fdn-matrix hadamard` |
| [`extreme_shimmer_music.wav`](examples/audio/extreme_shimmer_music.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/extreme_shimmer_music.wav) | Music → 6s reverb with octave shimmer | `--shimmer --shimmer-semitones 12 --shimmer-feedback 0.65` |
| [`extreme_plate_speech.wav`](examples/audio/extreme_plate_speech.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/extreme_plate_speech.wav) | Speech → circulant FDN plate simulation | `--rt60 1.8 --fdn-matrix circulant --lowcut 200 --highcut 6000` |
| [`extreme_frozen_music.wav`](examples/audio/extreme_frozen_music.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/extreme_frozen_music.wav) | Music → 30s near-infinite tail (32-line FDN) | `--rt60 30.0 --fdn-lines 32 --wet 0.95` |

### Experimental music tradition demos

Eight examples drawn from the experimental and avant-garde music tradition, each isolating a
different reverb behavior or aesthetic.

| File | Play | Inspiration | What to listen for |
|------|------|-------------|-------------------|
| [`lucier_sitting_room.wav`](examples/audio/lucier_sitting_room.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/lucier_sitting_room.wav) | Alvin Lucier — *I Am Sitting in a Room* | Speech run through the room 7× until only resonant frequencies survive |
| [`eno_discreet_music.wav`](examples/audio/eno_discreet_music.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/eno_discreet_music.wav) | Brian Eno — *Discreet Music* / Ambient series | 12s tail swallowing the source into a continuous wash |
| [`oliveros_deep_listening.wav`](examples/audio/oliveros_deep_listening.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/oliveros_deep_listening.wav) | Pauline Oliveros — *Deep Listening* | 18s cave-scale resonance, very low damping, 32-line FDN |
| [`fripp_frippertronics.wav`](examples/audio/fripp_frippertronics.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/fripp_frippertronics.wav) | Robert Fripp — Frippertronics tape-loop | Octave shimmer with 0.78 feedback accumulating over 8s |
| [`mbv_shoegaze.wav`](examples/audio/mbv_shoegaze.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/mbv_shoegaze.wav) | My Bloody Valentine — *Loveless* wall of sound | Dense shimmer wash (mix 0.55) through circulant FDN |
| [`reich_phase_drums.wav`](examples/audio/reich_phase_drums.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/reich_phase_drums.wav) | Steve Reich — phase minimalism | Tight 0.7s room on percussion, circulant diffusion |
| [`radigue_drone.wav`](examples/audio/radigue_drone.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/radigue_drone.wav) | Eliane Radigue — *ADNOS* / drone electronics | 45s near-infinite sustain, 32-line Hadamard, wet 0.97 |
| [`feldman_sparse_room.wav`](examples/audio/feldman_sparse_room.wav) | [Play](https://cdn.jsdelivr.net/gh/TheColby/verbx@main/examples/audio/feldman_sparse_room.wav) | Morton Feldman — late period | 3.8s room, low wet (0.52), allpass diffusion, contemplative space |

Dry source files are in the same directory. See [`examples/audio/README.md`](examples/audio/README.md) for the full render commands.

---

## Public Alpha Launch Notes

Current public alpha release: **v0.7.7**.

- `verbx` is currently research-grade software (public alpha), not production-certified.
- Confirm your environment with `verbx quickstart --verify --strict` and `verbx doctor`.
- Verify one algorithmic render and one convolution render before batch usage.
- For live monitoring, verify `verbx realtime --list-devices` before relying on realtime auditioning.
- For reproducible reports and bug submissions, attach `--repro-bundle` outputs and `verbx doctor --json-out doctor.json`.
- For demo-ready outputs, keep `--true-peak --target-peak-dbfs -1` enabled when files will be transcoded.
- Public alpha scope, known limitations, and support paths:
  [`docs/PUBLIC_ALPHA_NOTES.md`](docs/PUBLIC_ALPHA_NOTES.md)
- Launch-week pinned demo commands and expected SHA256 outputs:
  [`docs/LAUNCH_WEEK_DEMO_PINS.md`](docs/LAUNCH_WEEK_DEMO_PINS.md)
- Canonical launch-example parity check:
  `python scripts/check_launch_examples.py --check`
- PyPI publish auth setup for maintainers:
  [`docs/PYPI_PUBLISH_SETUP.md`](docs/PYPI_PUBLISH_SETUP.md)

---

## v0.8 Native Track

`v0.8` is the planned native C executable line. The released tool remains the
Python implementation in `v0.7.x`, but the native rewrite has now started with
an executable scaffold under [`native/verbx_c/README.md`](native/verbx_c/README.md).

Current native status:

- standalone `verbx-c` executable target
- portable C11 build path via `scripts/build_verbx_c.sh`
- implemented commands: `help`, `version`, `doctor`, `render`
- mono/stereo WAV input: PCM16/24/32 and float32/float64
- mono/stereo WAV output: `pcm16`, `float32`, `float64`
- deterministic offline render lifecycle in C: read -> process -> tail-finalize -> write
- explicit native process/error contract surfaced in `verbx-c doctor`
- native tail-stop metric selection: `--tail-metric peak|rms`
- foundational native algorithmic reverb core with float64 internal processing

Example native smoke test:

```bash
./scripts/build_verbx_c.sh
./build/native/verbx_c/verbx-c doctor
./build/native/verbx_c/verbx-c render in.wav out.wav --rt60 3.5 --out-format float32
```

---

## Announcement Channels

- Release announcements: [github.com/TheColby/verbx/releases](https://github.com/TheColby/verbx/releases)
- Homebrew tap updates: [github.com/TheColby/homebrew-verbx](https://github.com/TheColby/homebrew-verbx)
- Homebrew project news/blog: [brew.sh/blog](https://brew.sh/blog/)

Note: Homebrew blog posts cover Homebrew project releases and ecosystem updates; third-party tap formula launches are announced by the tap/project maintainers.

---

## What Is Reverb? (and Why Does verbx Sound Different)

When sound leaves a source in a physical space, it arrives at a listener via multiple paths: the direct path, early reflections from nearby surfaces, and a dense late diffuse field that gradually decays as energy is absorbed by materials and air. The perceptual result — the sense of space — depends on the timing, density, and spectral character of that decay. A bathroom might have an RT60 around 0.5 seconds. A large concert hall is typically 1.5–2.5 seconds. A cathedral can reach 8–12 seconds. The human auditory system is acutely sensitive to these cues and uses them to infer room size, distance from source, and surface hardness. This is why reverb affects not just the sound of a recording, but the perceived physicality of it.

In digital audio production, reverb is synthesized one of two ways. Algorithmic reverbs construct the room response from digital signal processing structures — delay networks, filters, and feedback topologies — shaped to produce the statistical properties of a real room without simulating any specific one. Convolution reverbs play back a recorded or synthesized impulse response, which captures everything about a real space in a single linear filter. Each approach has genuine advantages: algorithmic is controllable, computationally efficient at extreme lengths, and creates spaces that do not physically exist; convolution is realistic and reproducible from measured spaces.

Most reverb tools top out at RT60 values between 10 and 30 seconds. verbx is designed for extreme decay lengths — up to 3600 seconds — without the numerical instability that typically kills long algorithmic tails. The key is the Feedback Delay Network design: 64-bit internal precision everywhere, per-line gain calibration from the exact RT60-to-gain formula, and a choice of eight feedback matrix families that let you control tail diffusion and decay coloration independently from decay time. At 120 seconds of RT60, you are not simulating any physical space — you are synthesizing a temporal dimension that does not exist acoustically. That's the point. Beyond the algorithmic side, the convolution engine supports true `M x N` matrix routing for multichannel spaces, and the IR synthesis toolchain generates IRs up to 3600 seconds in four modes with deterministic caching so the same seed always produces the same space.

The Schroeder frequency is often approximated as:

$$
f_s \approx 2000\sqrt{\frac{T_{60}}{V}}
$$

where $f_s$ is in hertz, $T_{60}$ is RT60 in seconds, and $V$ is room volume in `m^3`. This is the threshold below which modal behavior dominates over diffuse statistics. For very long tails and large virtual spaces, this boundary sits low in the frequency range, meaning the modal structure of the FDN matters more, not less, than in short-room design. verbx exposes direct control over that structure: matrix type, delay line count, per-band RT60 targets, and time-varying decorrelation rates, so you can design long tails that remain spectrally coherent rather than metallic or ringing.

---

## Core Concepts

### Algorithmic vs. Convolution

**For beginners:** Algorithmic reverb synthesizes the space from scratch using delay networks and filters. It does not need an external file, responds instantly to parameter changes, and can produce decay times no physical room could sustain. Convolution reverb applies a pre-recorded impulse response — a measurement of what a specific room does to a click — to your audio. The result sounds like the space where the IR was recorded.

**For experts:** The algorithmic engine in verbx uses a Schroeder allpass diffusion stage feeding a fully coupled N-line FDN with configurable feedback matrix. Convolution uses uniformly-partitioned overlap-save FFT with optional CUDA acceleration via CuPy. The two engines share the same pre-delay, shimmer, freeze, ducking, bloom, tilt, loudness, and spatial stages. Use `--engine auto` and verbx selects based on whether an IR is present.

Choose algorithmic when you want extreme lengths, animated or time-varying decay, spaces that do not exist, low storage overhead. Choose convolution when you want: the character of a specific real or designed space, exact linear reproduction of an IR, or multichannel matrix routing from a measured space.

### RT60

**For beginners:** RT60 is roughly how long the reverb tail takes to fade away — specifically, how many seconds until the level drops by 60 dB (about a factor of 1000 in amplitude). A small bathroom is around 0.5 seconds. A bedroom is 0.3–0.8 seconds. A concert hall is 1.5–2.5 seconds. A cathedral reaches 5–12 seconds. verbx handles up to 3600 seconds. If the tail sounds too long and washes over everything, reduce RT60. If it sounds too dry and cut-off, increase it.

**For experts:** RT60 drives per-line gain calibration in the FDN:

$$
g_i = 10^{-\frac{3d_i}{T_{60}}}
$$

where $d_i$ is delay-line $i$ duration in seconds and $T_{60}$ is the target decay time. Shorter delays require gains closer to $1.0$ for the same RT60 target. Multiband RT60 (`--fdn-rt60-low`, `--fdn-rt60-mid`, `--fdn-rt60-high`) applies this formula per band with crossovers at `--fdn-xover-low-hz` and `--fdn-xover-high-hz`. The `--fdn-rt60-tilt` parameter applies a Jot-style frequency-dependent decay skew around the broadband target without requiring explicit per-band values. For analysis, use `verbx analyze --edr` to compute frequency-dependent RT estimates via backward Schroeder integration of the output.

### Impulse Responses

**For beginners:** An impulse response (IR) is a recording of what a space does to a single perfect click. When you convolve your audio with an IR, your audio sounds like it was played in that space. verbx can use IR files from external libraries, or generate its own synthetic IRs in four modes. You do not need an IR to use verbx — the algorithmic engine works without one.

**For experts:** verbx IR synthesis runs in four modes. `fdn` constructs a tail from the same FDN core used in algorithmic rendering, with configurable matrix family and decay parameters. `stochastic` generates exponentially-decayed filtered noise, shaped to match an RT60 curve. `modal` synthesizes a bank of tuned resonators — useful for musically-pitched spaces or physically-inspired objects. `hybrid` combines FDN late field with stochastic early reflections and optional modal resonator coloration. All modes use deterministic content-hash caching so repeated generation with the same parameters retrieves from cache rather than recomputing. The cache is keyed on mode, all synthesis parameters, seed, sample rate, channels, and length.

### Wet/Dry Mix

**For beginners:** `--wet` controls how much reverb you hear; `--dry` controls how much of the original unprocessed signal you keep. Most reverb uses are parallel — you blend the two. Start with `--wet 0.2 --dry 0.8` for subtle room feel and increase wet for more spaciousness. A setting of `--wet 1.0 --dry 0.0` is fully wet with no dry signal — often used in freeze or ambient texture work where you want the reverb itself as the sound.

**For experts:** verbx allows wet values above 1.0 for deliberate creative overdriving of the wet bus prior to the final mix. This is intentional and distinct from a gain error — it allows the reverb field to dominate with headroom for the loudness and limiter stages downstream to manage levels. Both `--wet` and `--dry` are valid automation targets: you can write time-varying lanes that sweep wet depth over the duration of a render, useful for automating reverb throws or level-responsive gating.

---

## The Engines

### Algorithmic Engine (`--engine algo`)

The algorithmic engine synthesizes reverb without an impulse response file. It is well suited for extreme tail lengths, evolving or modulated spaces, and creative applications where physical accuracy is not the goal.

**What it sounds like:** Smooth, dense, fully controllable. At short RT60 values (under 3 seconds) it behaves like a believable room. As RT60 increases past 20–30 seconds, it transitions into something entirely non-physical — a sustained shimmer of harmonic energy that can evolve slowly over minutes. The matrix family is the main texture control: Hadamard produces a more uniform, neutral tail; `tv_unitary` adds slow decorrelation motion; `graph` with ring topology sounds regular and periodic; `random` sounds unpredictable.

**Signal flow:**

```
input
  └─ pre-delay (z^-N_pre)
       └─ allpass diffusion (K stages)
            └─ FDN feedback loop
                 ├─ delay bank (N lines, z^-N_i)
                 ├─ per-line conditioning D_i(z)  [damping + DC block]
                 ├─ RT60 gain G  [diagonal, per-line]
                 ├─ feedback matrix M  [orthonormal family]
                 ├─ optional DFM micro-delays
                 └─ optional link filter
            └─ wet projection
  └─ dry signal
       └─ wet/dry mix → shimmer → bloom/tilt/EQ → loudness → output
```

Delay notation: `z^-N` means an integer-sample delay of $N$ samples.

**FDN mechanics:** At each sample, the FDN reads from $N$ delay lines, applies per-line damping and DC blocking, multiplies by the gain diagonal $\mathbf{G}$, multiplies by the feedback matrix $\mathbf{M}$, adds the injected excitation from the diffusion stage, and writes back to the delays. The matrix $\mathbf{M}$ must be orthonormal (or nearly so) to preserve energy over long tails; verbx orthonormalizes all matrix families before use. The state update is:

$$
\mathbf{y}[n] = \mathbf{D}\!\left(\mathbf{x}_{\mathrm{fb}}[n]\right)
$$

$$
\mathbf{x}_{\mathrm{fb}}[n+1] = \mathbf{G}\mathbf{M}\mathbf{y}[n] + \mathbf{u}[n]
$$

where:

- $n$ is the discrete-time sample index.
- $\mathbf{x}_{\mathrm{fb}}[n]$ is the feedback-state vector before loop conditioning.
- $\mathbf{y}[n]$ is the conditioned state after $\mathbf{D}(\cdot)$.
- $\mathbf{D}(\cdot)$ is per-line loop conditioning (damping + DC blocking).
- $\mathbf{G}$ is the diagonal RT60 gain matrix with entries $g_i$.
- $\mathbf{M}$ is the orthonormal feedback mixing matrix.
- $\mathbf{u}[n]$ is the post-diffusion excitation injected into the loop.

**FDN gain calibration:** For delay line $i$ with period $d_i$ seconds and target decay $T_{60}$:

$$
g_i = 10^{-\frac{3d_i}{T_{60}}}
$$

Shorter delay lines require gains closer to 1.0. This is computed per line so different delay lengths in the same network all decay toward the same target RT60.

**Matrix families:**

| Matrix | Sound character | Math note |
|---|---|---|
| `hadamard` | Even, neutral density | `N x N` Walsh-Hadamard; valid for power-of-2 line counts |
| `householder` | Similar to Hadamard, slightly more uniform | Householder reflection matrix |
| `random_orthogonal` | Unpredictable coloration | QR decomposition of random normal matrix |
| `circulant` | Periodic, regular resonance | Diagonalized by DFT; controlled frequency-domain structure |
| `elliptic` | Weighted energy distribution | Elliptic rotation-based coupling |
| `tv_unitary` | Slowly evolving, reduced metallic ringing | Time-varying orthonormal update at `--fdn-tv-rate-hz` Hz |
| `graph` | Topology-controlled pair mixing | Staged edge interactions over ring/path/star/random graph |
| `sdn_hybrid` | Geometry-inspired directional scatter | Scattering delay network coupling approach |

**Key parameters:**

| Parameter | Range | What it does | Expert note |
|---|---|---|---|
| `--rt60` | 0.1–3600 | Decay time target (seconds) | Drives per-line gain via `g_i = 10^(-3 d_i / T60)` |
| `--fdn-lines` | 2–64 | Number of delay lines | Higher line counts increase tail density; above 32 the returns diminish |
| `--fdn-matrix` | see above | Feedback mixing topology | Controls tail texture and energy diffusion pattern |
| `--allpass-stages` | 0–16 | Early diffusion stages | 4–10 is typical; 0 disables diffusion entirely |
| `--allpass-gain` | ±0.99 | Allpass coefficient | Per-stage or broadcast; must stay inside unit circle |
| `--damping` | 0–1 | HF rolloff in feedback loop | Higher values darken the tail faster |
| `--fdn-rt60-tilt` | -1 to 1 | Low/high decay skew | Positive = longer lows, shorter highs |
| `--fdn-link-filter` | none/lowpass/highpass | In-loop spectral shaping | Shapes the spectral flow on feedback edges |
| `--fdn-tv-rate-hz` | 0–5 | Time-varying matrix update rate | Active only with `tv_unitary`; slow rates reduce ringing |
| `--mod-depth-ms` | 0–10 | Delay modulation depth | Small values suppress metallic resonances |
| `--width` | 0–2 | Stereo spread | Increases decorrelation between channels |
| `--fdn-sparse` | flag | Sparse pair-mixing topology | Higher apparent order at lower compute cost |
| `--fdn-cascade` | flag | Nested FDN injection | Secondary network feeds early density into primary |
| `--unsafe-self-oscillate` | flag | UNSAFE above-unity feedback mode | Algorithmic engine only; for intentional self-oscillation |
| `--unsafe-loop-gain` | 0.01–1.25 | UNSAFE feedback gain scale | Use `>1.0` to drive oscillation |

---

### Convolution Engine (`--engine conv`)

The convolution engine filters audio through an impulse response. Use it when you want the character of a specific space — measured or synthesized — applied exactly.

**What it sounds like:** The output has the exact spectral and temporal character of the IR. A measured cathedral IR makes everything sound like it was played in that cathedral. A verbx-generated hybrid IR sounds like a designed space tuned to your specifications. Self-convolution (`--self-convolve`) smears a sound with its own spectral envelope — a different kind of effect.

**Partitioned convolution:** For long IRs, direct time-domain convolution is impractical. verbx uses uniformly-partitioned overlap-save convolution in the frequency domain:

$$
Y_k(\omega) = \sum_{p=0}^{P-1} X_{k-p}(\omega)\,H_p(\omega)
$$

where:

- $k$ is the current processing frame index.
- $\omega$ is frequency-bin index in the FFT domain.
- $P$ is the number of IR partitions.
- $X_{k-p}(\omega)$ is the stored input spectrum for frame $k-p$.
- $H_p(\omega)$ is the precomputed spectrum of IR partition $p$.
- $Y_k(\omega)$ is the accumulated output spectrum for frame $k$.

`--partition-size` controls the partition length: larger partitions reduce per-block FFT overhead but increase latency and peak memory. 16384–65536 samples is a practical range for offline rendering. With CuPy installed and `--device cuda`, the FFT multiply accumulation runs on GPU.

**Streaming vs. in-memory:** verbx automatically uses streaming convolution (low peak RAM) when the render is simple: engine conv, no repeat, no freeze, no normalization stages, no post-processing effects. All other combinations fall back to full-buffer processing. If RAM is a concern for very long IRs, keep the render chain minimal.

**Multichannel routing:** For $M$ input channels and $N$ output channels:

$$
y_o[n] = \sum_{i=0}^{M-1} \left(x_i * h_{i,o}\right)[n]
$$

where:

- $M$ is input-channel count and $N$ is output-channel count.
- $x_i[n]$ is input channel $i$.
- $h_{i,o}[n]$ is the IR from input channel $i$ to output channel $o$.
- $y_o[n]$ is output channel $o$.
- $*$ denotes linear convolution.

The IR file must contain $M \times N$ channels packed in output-major order (channel index $oM + i$) or input-major order ($iN + o$). Set `--ir-matrix-layout output-major` or `input-major` accordingly. Wrong packing order produces valid audio but semantically incorrect routing; verify with `verbx analyze` on the output.

**Key parameters:**

| Parameter | Range/values | What it does | Expert note |
|---|---|---|---|
| `--ir` | file path | Impulse response to apply | WAV, FLAC, AIFF, OGG, CAF all supported |
| `--partition-size` | 1024–131072 | FFT block size | Larger = more throughput, higher latency |
| `--tail-limit` | seconds | Cap IR tail at this length | Useful to bound compute on long IRs in batch |
| `--ir-normalize` | peak/rms/none | IR level normalization | `peak` is safest for predictable headroom |
| `--ir-matrix-layout` | output-major/input-major | Multichannel channel packing | Must match how the IR was created |
| `--ir-blend` | file path | Second IR for render-time blending | Repeatable; blend alpha automatable |
| `--ir-blend-mode` | linear/equal-power/spectral/envelope-aware | Blend algorithm | `envelope-aware` preserves early reflection character independently from late tail |
| `--self-convolve` | flag | Input file is its own IR | Spectral smearing / texture effect |
| `--device` | auto/cpu/cuda/mps | Compute backend | CUDA requires CuPy; MPS uses Apple Silicon profile |

---

## Impulse Response Synthesis

verbx generates its own IRs in four synthesis modes. The complete parameter reference is in [docs/IR_SYNTHESIS.md](docs/IR_SYNTHESIS.md). The IR toolchain is accessible via `verbx ir gen`, or triggered inline during render with `--ir-gen`.

A larger curated IR set is available in [`IRs/library/`](IRs/library/) with
folder-sorted buckets by length (`tiny`, `short`, `medium`, `long`) and mode
(`fdn`, `stochastic`, `modal`, `hybrid`), plus deterministic metadata in
`IRs/library/manifest.json`.

**Synthesis modes:**

| Mode | Character | Best for |
|---|---|---|
| `fdn` | Smooth, configurable, FDN-consistent | Spaces that match your algorithmic render topology |
| `stochastic` | Diffuse, noise-shaped, natural-sounding | Neutral halls, rooms, generic reverbs |
| `modal` | Resonant, tonal, pitched | Metal objects, tuned rooms, experimental textures |
| `hybrid` | Stochastic early + FDN late + optional modal resonator | Most general use; strong default |

**Common use case table:**

| Goal | Mode | Key flags |
|---|---|---|
| Neutral hall, natural decay | `stochastic` | `--rt60 3.0 --damping 0.4` |
| Match your FDN render topology | `fdn` | Same `--fdn-lines`, `--fdn-matrix` as render |
| Musical, pitched resonances | `modal` | `--f0 "64 Hz" --modal-count 40` |
| General cinematic space | `hybrid` | `--length 120 --seed 42` |
| Analyze and match audio source | `hybrid` | `--analyze-input source.wav` |

Generated IRs are cached by content hash + parameters. Repeated calls with the same settings return from cache instantly.

```bash
verbx ir gen my_space.wav --mode hybrid --length 120 --rt60 8.0 --seed 42
verbx ir analyze my_space.wav --json-out my_space_analysis.json
verbx ir morph space_A.wav space_B.wav blended.wav --mode equal-power --alpha 0.5
```

---

## Effects and Post-Processing

### Shimmer

Shimmer pitch-shifts the reverb tail (typically up an octave) and blends it back into the wet signal. The result is a bright, harmonically rich coloration that works well on pads, sustained notes, and ambient textures. The `--shimmer-feedback` parameter is the one most people get wrong: above around 0.85, the feedback loop builds exponentially. This is not a bug — it is the intended mechanism for extreme infinite-rise textures — but it requires either a tail limit, loudness targeting, or deliberate management to avoid runaway gain.

Safe mode clamps `--shimmer-feedback` to `0.98`. For intentional self-oscillation in the algorithmic path, enable `--unsafe-self-oscillate` and use `--unsafe-loop-gain > 1.0` (for example `1.03`).

```bash
--shimmer --shimmer-semitones 12 --shimmer-mix 0.35 --shimmer-feedback 0.70
--shimmer-lowcut 300 --shimmer-highcut 12000    # control frequency range of shimmer layer
```

### Freeze / Repeat

`--freeze` locks onto a segment of audio (defined by `--start` and `--end` in seconds) and loops it through the reverb engine with an equal-power crossfade at loop boundaries. This produces sustained, near-static textures. `--repeat N` runs the full render chain N times sequentially, each pass using the output of the previous as input — an iterative reprocessing that progressively imprints the room resonance on the source. Classic application: Alvin Lucier's "I Am Sitting in a Room" technique.

Use `--output-peak-norm input` with repeat chains to keep levels stable across passes.

### Ducking

`--duck` is the reverb effect most mix engineers do not use until they hear it. It attenuates the reverb output while the source signal is active, then lets the tail bloom in the gaps. The effect keeps the dry signal clean and articulate while the reverb is still long and spacious. Especially effective on drums, vocals, and anything with rhythmic transients.

```bash
--duck --duck-attack 15 --duck-release 250
```

Attack controls how quickly the reverb ducks when signal appears; release controls how quickly it recovers. Shorter release times give a punchier, more gated feel.

### Bloom

`--bloom N` emphasizes the slow build-up phase of the wet field, creating a cinematic swell effect where the reverb tail rises rather than immediately decaying. Values between 1.5 and 3.0 are perceptible as a rise before the decay plateau. Higher values push into dramatic orchestral-swell territory. It operates on the spectral envelope of the wet output and is distinct from simple gain automation.

### Tilt EQ

`--tilt N` applies a broadband spectral tilt to the wet field. Positive values (try 1.0–3.0) brighten the reverb tail; negative values darken it. This is a post-wet control, so it does not affect the dry signal or the decay mathematics — it only shapes the perceptual tone of the reverb output. Combine with `--lowcut` and `--highcut` for more specific frequency management.

---

## Spatial and Surround

For most uses, stereo output is all you need. Multichannel processing becomes relevant when you are delivering to a surround format, working in Ambisonics, or routing reverb through a spatial bus.

**Channel layouts:**

| Layout | Channels | Use case |
|---|---|---|
| `mono` | 1 | Mono sources or mono IR processing |
| `stereo` | 2 | Standard stereo output |
| `LCR` | 3 | Left/Center/Right film format |
| `5.1` | 6 | Standard surround |
| `7.1` | 8 | Expanded surround |
| `7.1.2` | 10 | Surround with overhead pair |
| `7.1.4` | 12 | Full Atmos bed format |
| `7.2.4` | 13 | 7-bed + dual-LFE + 4-top layout |
| `8.0` | 8 | 8-channel bed without dedicated LFE |
| `16.0` | 16 | Large-format discrete bed |
| `64.4` | 68 | High-density immersive bed + top layer |

Use `--input-layout` and `--output-layout` to declare channel semantics explicitly. Without them, verbx uses channel count alone, which can produce ambiguous routing for formats above stereo.

For large immersive outputs (`16.0`, `64.4`), set `--ir-route-map` explicitly when the IR is mono or channel-matched to the input. Recommended defaults:

- `--ir-route-map broadcast` for mono/channel-matched IRs
- `--ir-route-map full` for matrix-packed `M x N` IRs

Other formats are also easy to support: the routing and DSP paths already operate on arbitrary channel counts, and new symbolic layout names are straightforward to add when you need explicit semantics.

**Ambisonics:** verbx supports First-Order Ambisonics (FOA) with ACN channel ordering and SN3D/N3D/FuMa normalization. Use `--ambi-order 1` to declare FOA mode. `--ambi-encode-from stereo` encodes a stereo input into FOA before processing; `--ambi-decode-to stereo` decodes back out after. `--ambi-rotate-yaw-deg` applies rotation in the Ambisonics domain — useful for spatial orientation of the reverb field relative to a listener position. FUMA is FOA-only; ACN with SN3D is the standard workflow for most Ambisonics toolchains.

**IR matrix routing for surround:** If your IR file contains `M x N` channels (for $M$ input and $N$ output channels), declare the packing order with `--ir-matrix-layout`. Output-major packing stores all inputs for output 0 first, then all inputs for output 1, etc. (channel index $oM + i$). Input-major stores all outputs for input 0 first (channel index $iN + o$). A 5.1 input to 5.1 output full-matrix IR has 36 channels; a diagonal (same IR per channel) has 6. The routing is explicit: verbx does not guess.

---

## Loudness and Metering

Most audio delivered for broadcast, streaming, or film needs to hit a loudness target. EBU R128 / ITU-R BS.1770 defines integrated loudness in LUFS (Loudness Units relative to Full Scale). The practical difference between targeting -23 LUFS for broadcast and -14 LUFS for streaming can be over 9 dB of apparent level — enough to sound completely wrong in one context if mastered for the other.

verbx has a full loudness pipeline:

- **`--target-lufs N`** measures integrated loudness and scales to the target. Applies after the reverb processing stage.
- **`--target-peak-dbfs N`** enforces a peak ceiling. Use with `--true-peak` for inter-sample peak checking (required for formats that will be transcoded, as codec interpolation can raise peaks above the stored sample values). Sample peak (`--sample-peak`) is sufficient for archival.
- **`--output-peak-norm [input|target|full-scale]`** is a final-stage peak fit applied after all other processing: `input` matches the input file's peak, `target` uses an explicit dBFS value, `full-scale` normalizes to near 0 dBFS.
- **Soft limiter:** enabled by default as a final safety stage. Disable with `--no-limiter` when you want to pass raw dynamics to a downstream limiter in your chain.

The loudness and peak stages are intentionally separate because they serve different goals. Loudness targeting is about program-level normalization. Peak ceiling is about short-term safety. Do not conflate them.

True-peak detection uses oversampled measurement (ITU-R BS.1770). The difference between a sample peak of -0.1 dBFS and a true peak of +0.4 dBFS is invisible in sample-domain inspection but will cause clipping in AAC, MP3, and most streaming codecs. Use `--true-peak --target-peak-dbfs -1` for any output that will be transcoded.

---

## Automation and Modulation

The automation system lets you change reverb parameters over the duration of a render — wet depth, RT60, room size, decay tilt, IR blend position, and more — without editing the audio manually. This is useful for: reverb throws (sudden wet increase on a vocal), automated room size sweeps during a sound design cue, feature-reactive ducking where loudness in the source drives reverb depth, and batch augmentation where different parameter curves are applied to each variant.

**Automation lanes via file or inline points:**
```bash
# Sweep RT60 from 0.8s to 10s over 15 seconds
verbx render in.wav out.wav --engine algo \
  --automation-point "rt60:0.0:0.8:linear" \
  --automation-point "rt60:15.0:10.0:linear"

# From a JSON automation file
verbx render in.wav out.wav --automation-file automation.json
```

**Automation targets include:** `wet`, `dry`, `gain-db`, `rt60`, `damping`, `room-size`, `room-size-macro`, `clarity-macro`, `warmth-macro`, `envelopment-macro`, `fdn-rt60-tilt`, `fdn-tonal-correction-strength`, `ir-blend-alpha` (requires `--ir-blend`).

**Feature vector lanes** drive automation from frame-level audio analysis of the source signal. A "feature vector" is a time-series of per-frame descriptors extracted from audio: loudness, transient strength, spectral centroid, spectral flatness, harmonic ratio, MFCCs, formant spread, rhythm pulse, and more. You map these onto render targets with weight, curve shape (linear, smoothstep, power), and hysteresis:

```bash
# Wet depth tracks loudness and transients in the source
verbx render in.wav out.wav --engine conv --ir hall.wav \
  --feature-vector-lane "target=wet,source=loudness_norm,weight=0.70,curve=smoothstep,combine=replace" \
  --feature-vector-lane "target=wet,source=transient_strength,weight=0.30,curve=power,curve_amount=1.4,combine=add" \
  --feature-vector-trace-out trace.csv
```

Use `--feature-guide GUIDE.wav` to drive feature extraction from a separate audio file rather than the render input — a sidechain-style workflow.

**Modulation bus** provides LFO and envelope sources for simple periodic or input-reactive variation without needing a full automation file:
```bash
--mod-target mix --mod-min 0.1 --mod-max 0.9 \
  --mod-source "lfo:sine:0.07:1.0*0.7" \
  --mod-source "env:20:350*0.4"
```

Source syntax: `lfo:<shape>:<rate_hz>[:depth[:phase_deg]][*weight]` | `env[:attack_ms[:release_ms]][*weight]` | `audio-env:<path>[:attack_ms[:release_ms]][*weight]` | `const:<value>[*weight]`

---

## CLI Reference

Canonical, autogenerated help snapshots live in
[`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md). This README section is a
curated quick-reference for common switches.

### verbx render

`verbx render INFILE OUTFILE [options]`

#### Engine and Room Behavior

| Switch | Range | What it does | Expert note |
|---|---|---|---|
| `--engine` | algo/conv/auto | Reverb engine | `auto` picks `conv` if IR present, else `algo` |
| `--rt60` | 0.1–3600 | Decay time (seconds) | Per-line gain via `g_i = 10^(-3 d_i / T60)` |
| `--wet` | 0–∞ | Wet signal level | Values >1.0 overdrive wet bus intentionally |
| `--dry` | 0–1 | Dry signal level | |
| `--pre-delay-ms` | 0–500 | Reverb onset delay (ms) | |
| `--pre-delay` | e.g. `1/8D` | Musical note-value pre-delay | Requires `--bpm` |
| `--bpm` | float | Tempo for note-based pre-delay | |
| `--damping` | 0–1 | HF decay rate in feedback | Higher = darker tail |
| `--width` | 0–2 | Stereo decorrelation | |
| `--allpass-stages` | 0–16 | Diffusion stage count | |
| `--allpass-gain` | ±0.99 | Per-stage allpass coefficient | Comma-separated per-stage list accepted |
| `--fdn-lines` | 2–64 | Delay line count | |
| `--fdn-matrix` | see table above | Feedback matrix family | |
| `--fdn-tv-rate-hz` | 0–5 | TV-unitary update rate | `tv_unitary` only |
| `--fdn-tv-depth` | 0–1 | TV-unitary blend depth | `tv_unitary` only |
| `--fdn-matrix-morph-to` | matrix family | Target matrix for gradual morphing | Morphs from `--fdn-matrix` to target |
| `--fdn-matrix-morph-seconds` | seconds | Matrix morph duration | Requires `--fdn-matrix-morph-to` |
| `--fdn-dfm-delays-ms` | float | DFM micro-delay size | One value or one per line |
| `--fdn-sparse` | flag | Sparse pair-mixing topology | Exclusive with `tv_unitary` and `graph` |
| `--fdn-sparse-degree` | 1–8 | Pair-mixing stages | |
| `--fdn-link-filter` | none/lowpass/highpass | In-loop spectral shaping | |
| `--fdn-link-filter-hz` | Hz | Link filter cutoff | |
| `--fdn-rt60-tilt` | -1 to 1 | Low/high RT skew | Positive = longer lows |
| `--fdn-tonal-correction-strength` | 0–1 | Decay-color equalization | Track C control |
| `--fdn-cascade` | flag | Nested FDN injection | |
| `--fdn-graph-topology` | ring/path/star/random | Graph topology | `graph` matrix only |
| `--fdn-spatial-coupling-mode` | none/adjacent/front_rear/bed_top/all_to_all | Channel wet-bus coupling | |
| `--fdn-nonlinearity` | none/tanh/softclip | In-loop saturation | Keep blend low: 0.05–0.25 |
| `--beast-mode` | 1–100 | Parameter multiplier | 2–5 for heavier ambience, 10+ for extreme |

#### RT60 and Multiband Decay

| Switch | Range | What it does |
|---|---|---|
| `--fdn-rt60-low` | seconds | Low-band RT60 target |
| `--fdn-rt60-mid` | seconds | Mid-band RT60 target |
| `--fdn-rt60-high` | seconds | High-band RT60 target |
| `--fdn-xover-low-hz` | Hz | Low/mid crossover |
| `--fdn-xover-high-hz` | Hz | Mid/high crossover |

#### Convolution and IR Routing

| Switch | Values | What it does |
|---|---|---|
| `--ir` | file path | External IR for convolution |
| `--ir-normalize` | peak/rms/none | IR normalization before convolution |
| `--ir-matrix-layout` | output-major/input-major | Multichannel IR channel packing |
| `--ir-route-map` | auto/diagonal/broadcast/full | Channel routing strategy |
| `--partition-size` | int | FFT partition size |
| `--tail-limit` | seconds | Cap convolution tail |
| `--self-convolve` | flag | Use input as its own IR |
| `--ir-blend` | file path | Blend a second IR at render time (repeatable) |
| `--ir-blend-mix` | 0–1 | Blend coefficient(s) |
| `--ir-blend-mode` | linear/equal-power/spectral/envelope-aware | Blend algorithm |
| `--ir-gen` | flag | Auto-generate IR before render |
| `--ir-gen-mode` | fdn/stochastic/modal/hybrid | IR synthesis mode |
| `--ir-gen-length` | seconds | Generated IR duration |
| `--ir-gen-seed` | int | Deterministic seed |

#### Spatial

| Switch | Values | What it does |
|---|---|---|
| `--input-layout` | auto/mono/stereo/LCR/5.1/7.1/7.1.2/7.1.4/7.2.4/8.0/16.0/64.4 | Input channel semantics |
| `--output-layout` | auto/mono/stereo/LCR/5.1/7.1/7.1.2/7.1.4/7.2.4/8.0/16.0/64.4 | Output channel semantics |
| `--ambi-order` | 0–7 | Ambisonics order (1 = FOA) |
| `--ambi-normalization` | auto/sn3d/n3d/fuma | Normalization convention |
| `--channel-order` | auto/acn/fuma | Channel ordering convention |
| `--ambi-encode-from` | none/mono/stereo | Encode to FOA before render |
| `--ambi-decode-to` | none/stereo | Decode from Ambisonics after render |
| `--ambi-rotate-yaw-deg` | degrees | Yaw rotation in Ambisonics domain |

#### Effects

| Switch | Values | What it does |
|---|---|---|
| `--shimmer` | flag | Enable shimmer (pitch-shifted reverb coloration) |
| `--shimmer-semitones` | semitones | Pitch shift amount |
| `--shimmer-mix` | 0–1 | Shimmer blend |
| `--shimmer-feedback` | 0–1.25 | Shimmer feedback (>0.85 = rising; >0.98 requires unsafe mode) |
| `--unsafe-self-oscillate` | flag | UNSAFE: allow above-unity feedback in algorithmic mode |
| `--unsafe-loop-gain` | 0.01–1.25 | UNSAFE algorithmic loop-gain scale (`>1.0` for self-oscillation) |
| `--shimmer-spatial` | flag | Enable multichannel shimmer decorrelation | Useful for immersive beds |
| `--shimmer-spread-cents` | cents | Per-channel shimmer detune spread | Used with `--shimmer-spatial` |
| `--shimmer-decorrelation-ms` | ms | Per-channel shimmer delay spread | Used with `--shimmer-spatial` |
| `--duck` | flag | Enable sidechain ducking |
| `--duck-attack` | ms | Ducking attack time |
| `--duck-release` | ms | Ducking release time |
| `--bloom` | 0–5 | Wet field build-up emphasis |
| `--lowcut` | Hz | Post-wet high-pass filter |
| `--highcut` | Hz | Post-wet low-pass filter |
| `--tilt` | dB/oct | Post-wet spectral tilt |
| `--freeze` | flag | Loop a segment through the engine |
| `--start` | seconds | Freeze segment start |
| `--end` | seconds | Freeze segment end |
| `--repeat` | int | Repeat render passes |

#### Loudness and Output

| Switch | Values | What it does |
|---|---|---|
| `--target-lufs` | LUFS | Integrated loudness target |
| `--target-peak-dbfs` | dBFS | Peak ceiling |
| `--true-peak` / `--sample-peak` | flag | Peak detection mode |
| `--limiter` / `--no-limiter` | flag | Final safety limiter |
| `--normalize-stage` | none/post/per-pass | When normalization applies |
| `--output-peak-norm` | none/input/target/full-scale | Final peak fit |
| `--quality-preset` | sd/md/hd | Output-definition preset (`sd`=44.1 kHz PCM16, `md`=48 kHz PCM24, `hd`=192 kHz float32 default) |
| `--out-subtype` | auto/float32/float64/pcm16/pcm24/pcm32 | Output file bit depth (overrides preset subtype) |
| `--target-sr` | Hz | Render/output sample-rate conversion (overrides preset sample rate) |
| `--output-container` | auto/wav/w64/rf64 | Output container selection | `auto` upgrades long WAV renders to W64 |
| `--tail-stop-threshold-db` | dBFS | Tail detector threshold for write completion | Lower = longer retained tail |
| `--tail-stop-hold-ms` | ms | Explicit final zero-hold duration | Click-safe fade-out plus hard-zero ending |
| `--tail-stop-metric` | peak/rms | Tail detector metric | RMS is smoother, peak is stricter |

When `--target-sr` differs from the input file rate, `verbx render` performs
deterministic internal resampling and writes the output at the requested rate.

#### Execution and Reporting

| Switch | Values | What it does |
|---|---|---|
| `--device` | auto/cpu/cuda/mps | Compute backend |
| `--threads` | int | CPU thread count hint |
| `--algo-stream` | flag | Algorithmic proxy-IR streaming mode | Memory-friendly for very long renders |
| `--algo-proxy-ir-max-seconds` | seconds | Maximum proxy-IR duration | Used with `--algo-stream` |
| `--algo-gpu-proxy` | flag | Route algo proxy through CUDA convolution | Requires `--algo-stream --device cuda` |
| `--dry-run` | flag | Validate config without writing audio |
| `--auto-fit` | none/speech/music/drums/ambient | Apply profile-derived starting values | Respects explicit CLI overrides |
| `--preset` | name or `room:WxDxH/material` | Apply named preset or geometry-derived room baseline |
| `--lucky N` | int | Generate N randomized variants |
| `--frames-out` | path | Per-frame metrics CSV |
| `--analysis-out` | path | JSON analysis report path |
| `--repro-bundle` | flag | Write reproducibility bundle |
| `--quiet` | flag | Suppress console summary |
| `--silent` | flag | Suppress all output including analysis JSON |

---

### Early Reflection Geometry (Render)

| Switch | Values | What it does |
|---|---|---|
| `--er-geometry` | flag | Enable first-order image-source early reflections before main engine |
| `--er-room-dims-m` | `L,W,H` | Room dimensions (meters) |
| `--er-source-pos-m` | `x,y,z` | Source position (meters) |
| `--er-listener-pos-m` | `x,y,z` | Listener position (meters) |
| `--er-absorption` | 0.0–0.99 | Wall absorption coefficient |
| `--er-material` | anechoic/dead/studio/hall/stone | Preset absorption profile |

---

### verbx ir

```bash
verbx ir gen OUT_IR.wav [options]          # synthesize an IR
verbx ir analyze IR_FILE.wav              # measure RT60, EDT, spectral decay
verbx ir process IN_IR.wav OUT_IR.wav     # shape existing IR (EQ, normalize, tilt)
verbx ir morph IR_A.wav IR_B.wav OUT.wav  # blend two IRs
verbx ir morph-sweep IR_A.wav IR_B.wav OUT_DIR  # alpha-timeline sweep with QA artifacts
verbx ir fit INFILE.wav OUT_IR.wav        # fit an IR to match source audio
verbx ir sofa-info FILE.sofa              # inspect SOFA conventions/dimensions
verbx ir sofa-extract FILE.sofa OUT.wav   # extract FIR matrix for convolution renders
```

**`ir gen` key flags:** `--mode [fdn|stochastic|modal|hybrid]`, `--length`, `--rt60`, `--damping`, `--seed`, `--sr`, `--channels`, `--er-count`, `--diffusion`, `--fdn-lines`, `--fdn-matrix`, `--resonator`, `--resonator-mix`, `--analyze-input`, `--harmonic-align-strength`, `--f0`

**`ir morph` key flags:** `--mode [linear|equal-power|spectral|envelope-aware]`, `--alpha`, `--early-ms`, `--early-alpha`, `--late-alpha`, `--align-decay`, `--phase-coherence`, `--mismatch-policy [coerce|strict]`

**`ir morph-sweep` key flags:** Same as morph plus `--alpha-start`, `--alpha-end`, `--alpha-steps`, `--workers`, `--retries`, `--checkpoint-file`, `--resume`, `--qa-json-out`, `--qa-csv-out`

**`ir sofa-extract` key flags:** `--measurement-index`, `--emitter-index`, `--target-sr`, `--normalize [none|peak|rms]`, `--strict`

---

### verbx realtime

`verbx realtime [options]`

Realtime mode is currently a preview/audition path. `--engine conv` streams a
real IR directly; `--engine algo` renders a static proxy IR once, then runs the
live monitor through the streaming convolution engine. That means you get live
device routing and stable tails today, without pretending the full offline
automation surface is callback-safe yet. In other words: convolution settings
act live, while algorithmic settings shape the startup proxy IR that the live
convolver uses for the session.

**Transport and device routing**

| Switch | Values | What it does |
|---|---|---|
| `--engine` | auto/conv/algo | Live engine mode. `auto` chooses convolution when `--ir` is present, else algorithmic proxy |
| `--ir` | file path | IR source for realtime convolution |
| `--input-device` | index or substring | Select live input device |
| `--output-device` | index or substring | Select live output device |
| `--list-devices` | flag | Print available realtime devices and exit |
| `--sample-rate` | Hz | Live stream sample rate |
| `--block-size` | samples | Driver callback block size |
| `--partition-size` | samples | Convolution partition size used in the live processor |
| `--input-channels` | int | Processor input width. Defaults to mono/stereo, or to the length of `--input-channel-map` |
| `--input-channel-map` | comma-separated 1-based ints | Select and reorder hardware input channels, for example `1,3` or `1,3,5,7` |
| `--output-channels` | int | Processor output width. Defaults to processor width, or to the length of `--output-channel-map` |
| `--output-channel-map` | comma-separated 1-based ints | Select and reorder hardware output channels that receive processor outputs |
| `--duration` | seconds | Stop automatically after N seconds; omit for Ctrl-C run |
| `--quiet` | flag | Reduce console output |

**Realtime mix and proxy-room controls**

| Switch | Values | What it does |
|---|---|---|
| `--wet` / `--dry` | 0–1 | Live wet/dry mix in the convolver |
| `--rt60` | seconds | Algorithmic proxy decay time |
| `--pre-delay-ms` | ms | Algorithmic proxy pre-delay |
| `--damping` | 0–1 | Algorithmic proxy damping |
| `--width` | 0–2 | Algorithmic proxy stereo width |
| `--mod-depth-ms` / `--mod-rate-hz` | ms / Hz | Proxy delay modulation depth and rate |
| `--freeze` | flag | Realtime algo only: approximate infinite sustain via a long self-sustaining proxy tail |
| `--algo-proxy-ir-max-seconds` | seconds | Upper bound on startup proxy IR render length |
| `--lowcut` / `--highcut` / `--tilt` | Hz / dB tilt | Shape the startup proxy IR spectrum before live convolution |

**FDN topology and feedback options**

| Switch | Values | What it does |
|---|---|---|
| `--fdn-lines` | 1–64 | Proxy FDN line count |
| `--fdn-matrix` | hadamard/householder/random_orthogonal/circulant/elliptic/tv_unitary/graph/sdn_hybrid | Proxy matrix family |
| `--fdn-tv-rate-hz` / `--fdn-tv-depth` | Hz / amount | Time-varying matrix motion for supported FDNs |
| `--fdn-dfm-delays-ms` | comma-separated ms | Delay-feedback modulation taps |
| `--fdn-sparse` / `--fdn-sparse-degree` | flag / int | Sparse feedback wiring and degree |
| `--fdn-cascade` and friends | flag / scalars | Enable cascaded/nested FDN behavior |
| `--fdn-rt60-low` / `--mid` / `--high` | seconds | Multiband RT60 targets |
| `--fdn-rt60-tilt` | -1 to 1 | Tilt the decay profile across bands |
| `--fdn-link-filter*` | mode / Hz / mix | Filter energy in the feedback links |
| `--fdn-graph-topology` / `--fdn-graph-degree` / `--fdn-graph-seed` | topology / int / int | Graph-based FDN layout controls |
| `--fdn-matrix-morph-to` / `--fdn-matrix-morph-seconds` | matrix / seconds | Morph between matrix families during proxy synthesis |
| `--fdn-spatial-coupling-mode` / `--strength` | mode / 0–1 | Immersive cross-cluster coupling |
| `--fdn-nonlinearity*` | mode / amount / drive | Nonlinear feedback coloration |

**Diffusion, shimmer, and perceptual macros**

| Switch | Values | What it does |
|---|---|---|
| `--allpass-stages` | 0–64 | Diffusion depth |
| `--allpass-gain` | float or comma-separated list | Shared or per-stage diffusion coefficient(s) |
| `--allpass-delays-ms` | comma-separated ms | Custom allpass delay times |
| `--comb-delays-ms` | comma-separated ms | Custom FDN/comb delay times |
| `--shimmer` and `--shimmer-*` | flag / scalars | Startup proxy shimmer block with pitch, mix, feedback, filters, spatial spread |
| `--room-size-macro` / `--clarity-macro` / `--warmth-macro` / `--envelopment-macro` | -1 to 1 | Jot-inspired perceptual macro controls |
| `--algo-decorrelation-front` / `--rear` / `--top` | 0–1 | Extra proxy decorrelation for immersive layouts |
| `--unsafe-self-oscillate` / `--unsafe-loop-gain` | flag / scalar | Deliberately allow runaway feedback behavior when you really mean it |

Notes:

- When `--engine conv` is used with `--ir`, algorithmic proxy flags are rejected instead of being silently ignored.
- Realtime `--freeze` is not the offline segment-freeze processor. It is a live-preview approximation built on a long self-sustaining proxy IR.
- Channel maps are 1-based hardware channel numbers. If you pass `--input-channel-map 1,3`, processor input 1 comes from hardware input 1 and processor input 2 comes from hardware input 3.
- Channel-count switches must match the length of the corresponding channel map when both are provided.
- The autogenerated exhaustive help for every switch lives in [`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md).

Example:

```bash
verbx realtime --engine algo \
  --input-device "Built-in Microphone" \
  --output-device "Headphones" \
  --sample-rate 48000 --block-size 256 \
  --input-channel-map 1,3 --output-channel-map 1,2,5,6 \
  --rt60 24 --freeze --shimmer \
  --fdn-matrix tv-unitary --fdn-tv-rate-hz 0.35 --fdn-tv-depth 0.12 \
  --fdn-graph-topology star --fdn-sparse --fdn-cascade \
  --lowcut 120 --highcut 9000 --tilt 1.5
```

---

### verbx analyze

`verbx analyze INFILE [options]`

Outputs loudness, peak, spectral, and decay metrics. Key flags:

| Switch | What it produces |
|---|---|
| `--lufs` | Integrated LUFS, true peak, LRA |
| `--edr` | Frequency-dependent RT60 estimates via Schroeder backward integration |
| `--frames-out path` | Per-frame CSV with time-varying descriptors |
| `--json-out path` | Full metric payload in JSON |
| `--ambi-order N` | Ambisonics spatial metrics for HOA assets |

---

### verbx room-model

```bash
verbx room-model --dims-m 6,8,3
verbx room-model --rt60 1.6 --material hall --json-out room.json
```

Use this when you want a physically grounded sanity check before rendering.
`verbx room-model` either inspects an explicit rectangular room geometry or
infers one from RT60 plus an absorption/material assumption. It reports volume,
surface area, direct-path pre-delay, aspect ratios, Bolt-style proportion
warnings, and writes JSON when requested.

If you already know the dimensions and want to jump straight to a render, you
can skip the inspection step and use the matching render shorthand:

```bash
verbx render in.wav out.wav --preset room:6x8x3/hall
```

| Switch | Values | What it does |
|---|---|---|
| `--dims-m` | `width,depth,height` | Inspect an explicit rectangular room |
| `--rt60` | seconds | Infer room dimensions from RT60 plus absorption |
| `--absorption` | 0.01–0.99 | Override the mean absorption used for RT60 inversion |
| `--material` | preset name | Use a wall material preset when `--absorption` is omitted |
| `--source-pos-m` | `x,y,z` meters | Source position inside the room |
| `--listener-pos-m` | `x,y,z` meters | Listener position inside the room |
| `--json-out` | path | Write the full geometry payload as JSON |

---

### verbx dereverb

`verbx dereverb INFILE OUTFILE [options]`

Deterministic spectral late-tail suppression for existing recordings.

| Switch | Values | What it does |
|---|---|---|
| `--mode` | wiener/spectral_sub | Suppression algorithm |
| `--strength` | 0–2 | Reverberant suppression amount |
| `--floor` | 0–1 | Residual floor to reduce musical-noise artifacts |
| `--window-ms` | ms | STFT analysis window |
| `--hop-ms` | ms | STFT hop size (must be smaller than window) |
| `--tail-ms` | ms | Late-field smoothing horizon |
| `--pre-emphasis` | 0–0.98 | Optional HF emphasis before suppression |
| `--mix` | 0–1 | Blend of processed output |
| `--out-subtype` | auto/float32/float64/pcm16/pcm24/pcm32 | Output encoding |
| `--json-out` | path | Write structured dereverb report |

---

### verbx batch

```bash
verbx batch template > manifest.json           # generate manifest skeleton
verbx batch render manifest.json --jobs 8      # parallel render
verbx batch augment-template > augment.json    # generate augmentation manifest
verbx batch augment-profiles                   # list built-in profiles
verbx batch augment augment.json --jobs 8      # generate training dataset
```

**Batch render flags:** `--jobs`, `--schedule [fifo|shortest-first|longest-first]`, `--retries`, `--continue-on-error`, `--checkpoint-file`, `--resume`, `--dry-run`

**Batch augment flags:** Built-in profiles `asr-reverb-v1`, `music-reverb-v1`, `drums-room-v1`. Key flags: `--copy-dry`, `--dataset-card-out`, `--metrics-csv-out`, `--qa-bundle-out`, `--provenance-hash`, `--verify-split-isolation`

---

### Other Commands

```bash
verbx suggest INFILE      # analysis-driven starter settings for your specific audio
verbx realtime --list-devices   # list selectable live audio devices
verbx realtime --engine algo --input-device 0 --output-device 3   # live preview
verbx render in.wav out.wav --preset room:6x8x3/hall   # geometry-derived room baseline
verbx room-model --rt60 1.8 --material hall   # infer a plausible room geometry
verbx dereverb INFILE OUTFILE   # suppress late reverberation from an existing recording
verbx presets             # list built-in presets
verbx presets --show cathedral_extreme   # inspect preset parameters
verbx quickstart          # copy-paste workflows for first-run scenarios
verbx quickstart --verify --strict       # startup readiness check (useful before demos)
verbx doctor              # platform/acceleration diagnostics
verbx doctor --json-out doctor.json      # machine-readable diagnostics for issue reports
verbx version             # package version string
verbx cache info          # inspect IR cache
verbx cache clear         # clear IR cache
```

---

## Recipes

### Beginner Recipes

**Subtle room glue — keeps everything sounding like it was recorded together:**
```bash
verbx render mix_bus.wav glued.wav --engine algo --rt60 0.8 --wet 0.15 --dry 0.9 --pre-delay-ms 12
```

**Natural vocal hall — spacious without washing the lyrics:**
```bash
verbx render vocals.wav vocals_hall.wav --engine algo \
  --rt60 2.2 --wet 0.28 --dry 0.78 --pre-delay-ms 22 --lowcut 200 --highcut 10000
```

**Drums with ducking — tail blooms between hits, never clutters transients:**
```bash
verbx render drums.wav drums_room.wav --engine algo \
  --rt60 1.4 --wet 0.55 --dry 0.6 --duck --duck-attack 10 --duck-release 180
```

**Convolution from a free IR library — real space character:**
```bash
verbx render piano.wav piano_conv.wav --engine conv --ir hall_ir.wav --ir-normalize peak --wet 0.5 --dry 0.7
```

**Tempo-synced pre-delay — reverb onset lines up with the beat:**
```bash
verbx render snare.wav snare_delay.wav --engine algo --pre-delay 1/8D --bpm 128 --rt60 1.8 --wet 0.45
```

**Loudness-safe delivery — hits -16 LUFS with -1 dBTP ceiling:**
```bash
verbx render master.wav delivered.wav --engine algo --rt60 2.0 --wet 0.2 \
  --target-lufs -16 --true-peak --target-peak-dbfs -1
```

---

### Production Recipes

**Broadcast dialogue room — natural placement, EBU R128 compliant:**
```bash
verbx render dialogue.wav dialogue_room.wav --engine conv \
  --ir small_room_ir.wav --wet 0.25 --dry 0.85 --pre-delay-ms 8 \
  --lowcut 150 --highcut 9000 --target-lufs -23 --true-peak --target-peak-dbfs -1
```

**Film score hall — wide, clear, cinematic:**
```bash
verbx render strings.wav strings_hall.wav \
  --engine conv --ir large_hall_ir.wav \
  --wet 0.65 --dry 0.55 --pre-delay 1/16 --bpm 72 \
  --width 1.2 --bloom 1.8 --tilt 1.0 \
  --lowcut 80 --target-lufs -20 --target-peak-dbfs -1.5
```

**Gated drum space — 1980s aesthetic, punchy tail that cuts off:**
```bash
verbx render drums.wav drums_gated.wav --engine conv \
  --ir plate_short.wav --ir-normalize peak --tail-limit 1.2 \
  --wet 0.75 --dry 0.4 --highcut 9000 --target-peak-dbfs -1
```

**Dub chamber send — high-wet parallel texture, bandwidth controlled:**
```bash
verbx render snare_send.wav dub_chamber.wav --engine conv \
  --ir spring_ir.wav --repeat 2 --wet 0.95 --dry 0.05 \
  --lowcut 180 --highcut 4500 --tilt -2.0 --output-peak-norm input
```

**Sparse hall for piano or choir — depth without obscuring articulation:**
```bash
verbx render piano.wav piano_hall.wav --engine conv --ir hall_ir.wav \
  --pre-delay 1/16 --bpm 60 --wet 0.55 --dry 0.7 \
  --lowcut 120 --highcut 11000 --target-lufs -20 --target-peak-dbfs -1
```

**Cathedral vocal/organ — long, immersive, cinematic:**
```bash
verbx render choir.wav choir_cathedral.wav --engine conv \
  --ir cathedral_ir.wav --wet 0.82 --dry 0.35 --rt60 90 \
  --lowcut 70 --highcut 10000 --target-lufs -21 --true-peak --target-peak-dbfs -1
```

**Track D IR blend — morphing between two hall characters during render:**
```bash
verbx render in.wav morphed.wav --engine conv --ir hall_A.wav \
  --ir-blend hall_B.wav --ir-blend-mix 0.6 --ir-blend-mode envelope-aware \
  --ir-blend-early-ms 60 --automation-point "ir-blend-alpha:0.0:0.0" \
  --automation-point "ir-blend-alpha:30.0:1.0"
```

**AI dataset batch — augmentation with split isolation and metrics:**
```bash
verbx batch augment augment_manifest.json --profile asr-reverb-v1 \
  --jobs 8 --copy-dry --verify-split-isolation \
  --metrics-csv-out out/metrics.csv --dataset-card-out out/DATASET_CARD.md \
  --qa-bundle-out out/qa_bundle.json --provenance-hash
```

---

### Experimental Recipes

Eight approaches from the experimental music tradition. Rendered demos for all of these are
in [`examples/audio/`](examples/audio/).

---

**Alvin Lucier — *I Am Sitting in a Room*** (iterative room resonance accumulation)

Each pass imprints the room's modal resonances more deeply. After 12–20 passes, only the
resonant frequencies of the virtual room survive — the original speech is gone.

```bash
mkdir passes && cp voice.wav passes/pass_00.wav && current="passes/pass_00.wav"
for i in $(seq 1 20); do
  next=$(printf "passes/pass_%02d.wav" "$i")
  verbx render "$current" "$next" --engine algo --rt60 4.5 \
    --wet 1.0 --dry 0.0 --fdn-lines 16 --fdn-matrix hadamard \
    --lowcut 60 --no-progress
  current="$next"
done
# Quick single-command version (7 passes baked in):
verbx render voice.wav lucier_7pass.wav --engine algo --rt60 4.5 \
  --wet 1.0 --dry 0.0 --repeat 7 --fdn-lines 16 --fdn-matrix hadamard --lowcut 60
```

---

**Brian Eno — *Discreet Music* / Ambient series** (endless ambient tail)

Decay so long the source dissolves. The wet signal becomes the room's breath.

```bash
verbx render input.wav eno_ambient.wav --engine algo --rt60 12.0 \
  --wet 0.92 --dry 0.08 --damping 0.25 --pre-delay-ms 35 \
  --fdn-lines 16 --fdn-matrix hadamard --lowcut 50 \
  --target-lufs -22 --target-peak-dbfs -2
```

---

**Pauline Oliveros — *Deep Listening*** (cave-scale resonance)

Inspired by Oliveros's work in underground cisterns. Very low damping lets every frequency
sustain; 32-line FDN produces the lateral complexity of stone architecture.

```bash
verbx render drone.wav deep_listening.wav --engine algo --rt60 18.0 \
  --wet 0.95 --dry 0.10 --fdn-lines 32 --fdn-matrix hadamard \
  --pre-delay-ms 55 --damping 0.15 --lowcut 30 \
  --target-lufs -24 --target-peak-dbfs -2
# For a 240-second synthesized IR version:
verbx render drone.wav deep_ir.wav --ir-gen --ir-gen-mode hybrid \
  --ir-gen-length 240 --ir-gen-seed 108 --engine conv \
  --wet 0.9 --dry 0.15 --target-lufs -24 --target-peak-dbfs -2
```

---

**Robert Fripp / Eno — Frippertronics tape-loop accumulation**

Shimmer feedback builds over each block. At 0.78, the octave layer accumulates like a tape
recirculation loop growing denser with each pass.

```bash
verbx render guitar.wav frippertronics.wav --engine algo --rt60 8.0 \
  --wet 0.82 --dry 0.28 --fdn-lines 16 --fdn-matrix hadamard \
  --shimmer --shimmer-semitones 12 --shimmer-mix 0.45 --shimmer-feedback 0.78 \
  --pre-delay-ms 25 --target-peak-dbfs -2
# Iterative version — 12 passes with gradual timbral drift:
mkdir fripp && cp guitar.wav fripp/pass_00.wav && current="fripp/pass_00.wav"
for i in $(seq 1 12); do
  next=$(printf "fripp/pass_%02d.wav" "$i")
  verbx render "$current" "$next" --engine algo --rt60 8.0 \
    --wet 0.82 --dry 0.12 --shimmer --shimmer-semitones 12 \
    --shimmer-feedback 0.78 --no-progress
  current="$next"
done
```

---

**Shoegaze / My Bloody Valentine — wall of sound** (dense shimmer wash)

Freeze a guitar sustain, then bury it in octave shimmer and a circulant FDN. The circulant
matrix produces the smeared, tonally undifferentiated density that defines the genre.

```bash
verbx render guitar.wav shoegaze.wav --engine algo \
  --freeze --start 1.0 --end 2.4 \
  --shimmer --shimmer-semitones 12 --shimmer-mix 0.55 --shimmer-feedback 0.72 \
  --rt60 5.0 --wet 0.88 --dry 0.22 --fdn-matrix circulant --lowcut 80 \
  --width 1.4 --target-peak-dbfs -2
```

---

**Steve Reich — phase minimalism** (tight rhythmic room)

Short RT60 with a circulant diffusion matrix keeps individual hits distinct while adding
spatial depth. The circulant matrix's circular delay structure creates subtle comb filtering
that complements phase-shifted rhythmic material.

```bash
verbx render percussion.wav reich_room.wav --engine algo --rt60 0.7 \
  --wet 0.55 --dry 0.50 --fdn-lines 8 --fdn-matrix circulant \
  --pre-delay-ms 18 --damping 0.6 --lowcut 60
```

---

**Eliane Radigue — *ADNOS* / drone electronics** (near-infinite sustain)

At RT60=45s with wet=0.97, the dry signal is almost entirely subsumed. Radigue's aesthetic
is about sound that has been in the room so long it has become the room.

```bash
verbx render drone.wav radigue.wav --engine algo --rt60 45.0 \
  --wet 0.97 --dry 0.05 --fdn-lines 32 --fdn-matrix hadamard \
  --damping 0.10 --lowcut 20 --target-lufs -28 --target-peak-dbfs -2
```

---

**Morton Feldman — late period** (contemplative sparse space)

Feldman's late works often feature long silences and isolated events in large, reflective
spaces. Medium RT60, restrained wet level, allpass diffusion, no shimmer.

```bash
verbx render piano.wav feldman.wav --engine algo --rt60 3.8 \
  --wet 0.52 --dry 0.52 --fdn-lines 8 --fdn-matrix circulant \
  --pre-delay-ms 30 --damping 0.50 --allpass-stages 4 \
  --target-lufs -26 --target-peak-dbfs -2
```

---

**Self-convolution texture smear** — signal convolved with itself:
```bash
verbx render input.wav self_convolved.wav --self-convolve \
  --beast-mode 12 --partition-size 16384 --normalize-stage none
```

**Feature-reactive reverb depth** — wet depth tracks source loudness in real time:
```bash
verbx render in.wav reactive.wav --engine conv --ir hall.wav \
  --feature-vector-lane "target=wet,source=loudness_norm,weight=0.70,curve=smoothstep,combine=replace" \
  --feature-vector-lane "target=wet,source=transient_strength,weight=0.30,combine=add" \
  --feature-vector-frame-ms 40 --feature-vector-trace-out trace.csv
```

**Lucky mode exploration** — 12 randomized wild variants from one source:
```bash
verbx render in.wav out/lucky.wav --lucky 12 --lucky-out-dir out/lucky_set --lucky-seed 2026 --no-progress
```

---

### Workflow Recipes

**Batch parallel render from a manifest:**
```bash
verbx batch template > manifest.json      # edit manifest.json with your jobs
verbx batch render manifest.json --jobs 8 --schedule longest-first --retries 1 \
  --checkpoint-file manifest.checkpoint.json
# interrupted? resume from where it stopped:
verbx batch render manifest.json --jobs 8 --resume --checkpoint-file manifest.checkpoint.json
```

**IR sweep QA — morph between two IRs with quality metrics:**
```bash
verbx ir morph-sweep ir_a.wav ir_b.wav out/sweep \
  --alpha-start 0.0 --alpha-end 1.0 --alpha-steps 9 \
  --workers 4 --retries 1 --mismatch-policy strict \
  --checkpoint-file out/sweep.checkpoint.json \
  --qa-json-out out/sweep_summary.json --qa-csv-out out/sweep_metrics.csv
```

**Generate a bank of 25 varied IRs:**
```bash
./scripts/generate_ir_bank.sh IRs/bank_25 25 flac
# or with explicit Python control:
./scripts/generate_ir_bank.py --out IRs/bank_25 --count 25 --sr 48000 --channels 2 --format flac
```

**Generate a large folder-sorted IR library (varying lengths):**
```bash
uv run python scripts/generate_ir_library.py \
  --out IRs/library --sr 12000 --channels 2 --format flac --seeds-per-shape 1
```

**Pre-render validation — catch config errors before a long job:**
```bash
verbx render long_input.wav output.wav --engine algo --rt60 180 --fdn-lines 32 --dry-run
# prints resolved config, estimated output duration, device selection — no audio written
```

---

## Performance and Acceleration

**CPU (default):** All processing. Algorithmic FDN path benefits from `numba` when installed — install with `pip install numba` and verbx uses JIT-compiled inner loops automatically. Check with `verbx doctor`.

**Apple Silicon (MPS):** `--device mps` uses the MPS profile for the algorithmic path. The convolution FFT runs on CPU (NumPy/SciPy). Threading helps: `--threads 8` is a good starting point for M-series chips. Apple Silicon is well-suited for the algorithmic engine; the memory bandwidth advantage shows on high line count FDN renders.

**CUDA:** `--device cuda` enables GPU-accelerated partitioned FFT convolution via CuPy. Install with `pip install cupy-cuda12x` (match your CUDA version). The algorithmic engine does not benefit from CUDA — it runs on CPU regardless. CUDA acceleration is most valuable for long-IR convolution with large files. If CuPy is unavailable, verbx falls back to CPU silently.

**Block size and partition size:** `--block-size` controls the algorithmic engine's internal block size — larger blocks can improve throughput at the cost of responsiveness per block. `--partition-size` controls convolution FFT partition length — the main tuning knob for convolution throughput. Larger partitions reduce per-block overhead but increase peak memory. For offline rendering, 16384–65536 is a good range. For very long IRs (120s+), larger partition sizes (65536) often give better throughput.

**Streaming convolution** engages automatically for simple conv renders (no normalization, no post-effects, no freeze, `--repeat 1`). Peak RAM use scales with partition size rather than IR length in this mode.

**Numba:** When installed, verbx automatically JIT-compiles the FDN inner loop for the algorithmic engine. First render with a new configuration takes a few extra seconds to compile; subsequent renders at the same parameters are significantly faster. To verify it is active: `verbx doctor --json-out doctor.json` and check `numba_available`.

---

## DSP Architecture

For contributors and people who want to understand the signal chain in code.

**Module map:**

| Path | Contents |
|---|---|
| `src/verbx/cli.py` | Command routing, CLI surface, option validation |
| `src/verbx/core/algo_reverb.py` | Algorithmic FDN engine |
| `src/verbx/core/conv_reverb.py` | Partitioned FFT convolution engine |
| `src/verbx/core/pipeline.py` | Render orchestration, stage ordering |
| `src/verbx/core/loudness.py` | LUFS targeting, true-peak, limiter |
| `src/verbx/core/shimmer.py` | Shimmer, bloom, ducking, tilt |
| `src/verbx/core/tempo.py` | Note-value pre-delay parsing |
| `src/verbx/analysis/` | Frame extraction, EDR, Ambisonics metrics |
| `src/verbx/ir/` | IR synthesis modes, shaping, morphing, fitting, cache |
| `src/verbx/io/` | Audio I/O, progress reporting |

**Signal chain (algorithmic engine):**

```
input audio
  │
  ├─ [dry path] ──────────────────────────────────────────┐
  │                                                        │
  └─ pre-delay (z^-N)                                      │
       └─ allpass diffusion (stages 1..K)                  │
            └─ FDN core                                     │
                 ├─ delay bank (lines 1..N)                │
                 ├─ loop conditioning D(z)                 │
                 ├─ RT60 gain matrix G                     │
                 ├─ feedback matrix M [orthonormal]        │
                 ├─ [optional] DFM micro-delays            │
                 ├─ [optional] link filter                 │
                 └─ [optional] in-loop nonlinearity        │
            └─ wet projection                              │
                 └─ shimmer / bloom / duck / tilt / EQ ───┤
                                                           │
  wet/dry mix ◄──────────────────────────────────────────┘
       └─ loudness stage (LUFS / peak / limiter)
            └─ final peak normalization
                 └─ audio write
                      └─ analysis JSON + frames CSV
```

Notation: `z^-N` denotes an integer-sample delay of $N$ samples, $K$ is
allpass-stage count, and $N$ (in `lines 1..N`) is FDN delay-line count.

**Precision:** All DSP — FDN state updates, FFT operations, allpass filters, automation curves, feature vectors, analysis metrics — runs in `float64` internally. Output is downcast at write time according to `--out-subtype`. `verbx render` defaults to HD output (`192000 Hz`, `float32`) unless overridden by `--quality-preset`, `--target-sr`, or `--out-subtype`.

**Key design decisions:**
- Per-line gain calibration (not global feedback gain) lets all delay lines, regardless of length, track the same RT60 target. This is essential for stable long tails.
- Orthonormalization of all matrix families before use prevents energy accumulation in high-feedback topologies.
- Automation evaluation uses a slew limiter and deadband guard in addition to smoothing to prevent abrupt control jumps and high-frequency control chatter in block-mode evaluation.
- The IR cache uses a content hash (audio samples + metadata) rather than file path, so the same IR content at a different path still hits cache.

---

## Contributors

- Colby Leider (creator and maintainer)
- Full contributors graph: [github.com/TheColby/verbx/graphs/contributors](https://github.com/TheColby/verbx/graphs/contributors)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. Quick start:

```bash
hatch env create
hatch run lint      # ruff check
hatch run typecheck # pyright strict
hatch run test      # pytest
```

`uv` alternative:
```bash
uv sync --extra dev
uv run ruff check . && uv run pyright && uv run pytest
```

Report security issues via [SECURITY.md](SECURITY.md). See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## References and Further Reading

Full bibliography: [docs/REFERENCES.md](docs/REFERENCES.md)

Key papers:

- **Schroeder (1962)** — "Natural sounding artificial reverberation." The foundational work on allpass and comb filter reverb structures that forms the basis for most algorithmic reverb design.
- **Jot & Chaigne (1991)** — "Digital delay networks for designing artificial reverberators." Introduced the Feedback Delay Network in its modern form; directly informs the gain calibration formula used in verbx.
- **Jot (1992)** — "An analysis/synthesis approach to real-time artificial reverberation." Extends FDN theory to frequency-dependent decay, the basis for multiband RT60 control.
- **Smith (1985)** — "A new approach to digital reverberation using closed waveguide networks." Scattering Delay Networks — a physical wave propagation model distinct from the FDN approach; informs the `sdn_hybrid` matrix family.
- **Valimaki et al. (2012)** — "Fifty years of artificial reverberation." Survey paper; an accessible overview of the full history of algorithmic reverb from Schroeder to modern approaches.
- **Gardner (1998)** — "Reverberation algorithms." Practical implementation guide covering partitioned convolution, early reflections, and late field design.

Additional guides in `docs/`:
- [Autogenerated CLI reference](docs/CLI_REFERENCE.md) — machine-generated `--help` snapshots for all command groups
- [IR synthesis guide](docs/IR_SYNTHESIS.md) — complete parameter reference for all synthesis modes
- [AI augmentation guide](docs/AI_AUGMENTATION.md) — dataset generation workflow documentation
- [Schema reference](docs/SCHEMA_REFERENCE.md) — JSON/CSV formats for manifests and automation
- [Dataset augmentation notebook](examples/dataset_augmentation.ipynb) — Python API workflow for ML pipelines
- [IR morph QA guide](docs/IR_MORPH_QA.md) — morph-sweep QA artifacts and CI integration
- [Benchmark baseline guide](docs/benchmarks/README.md) — CI/runtime comparison workflow
- [Extreme cookbook](docs/EXTREME_COOKBOOK.md) — 100 additional workflow examples
- [SOFA interoperability note](docs/SOFA_FEASIBILITY.md) — shipped `sofa-info` / `sofa-extract` workflow and current constraints
- [Launch example parity checker](scripts/check_launch_examples.py) — verifies canonical launch commands stay mirrored across docs/man pages

---

## License

See [LICENSE](LICENSE).

v0.7.7 — current release (public alpha). See [CHANGELOG.md](CHANGELOG.md) for version history.
