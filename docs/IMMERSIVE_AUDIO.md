# Immersive Reverb, Surround Sound, and Dolby Atmos

Immersive audio is not simply stereo with more loudspeakers. It is a collection of related production models that describe space in different ways: as fixed output channels, as a listener-centered sound field, or as audio objects accompanied by position metadata. Reverb exposes every difference among those models. A tail can remain tied to a channel, rotate as a scene, spread through a fixed bed, or follow an object whose final loudspeaker feeds are calculated only during rendering.

This chapter explains those distinctions and turns them into practical verbx workflows. It covers conventional surround, height layouts, Ambisonics, binaural monitoring, Dolby Atmos beds and objects, reverb routing, DAW handoff, deliverables, and quality control. It also states the present product boundary plainly: **verbx can generate and process discrete multichannel audio, Ambisonic material, matrix-routed convolution, and machine-readable handoff evidence; it does not currently author Dolby object metadata or create an ADM BWF, DAMF, or IMF IAB master.** Use an Atmos-capable DAW and the Dolby Atmos Renderer for that final authoring stage.

The goal is not to prescribe one spatial aesthetic. It is to make the signal path legible enough that a mix decision remains a mix decision, rather than becoming an accidental consequence of channel order, renderer behavior, bass management, or a mistaken assumption about what a file contains.

## 1. Three Ways to Describe Space

### 1.1 Channel-based audio

A channel-based signal assigns each waveform to a named destination such as Left, Center, Right, Left Surround, or Left Top Front. Stereo, 5.1, 7.1, and a discrete 7.1.4 speaker feed are channel-based formats. Their strengths are directness and predictability: if the playback system honors the labels, the Left Surround signal reaches the left-surround path. Their limitation is that the mix is bound to the assumed loudspeaker geometry. Translating a 7.1.4 print to headphones or a smaller room requires a downmix or a separate renderer.

Channel-based reverb is often the most stable choice for an enveloping late field. A diffuse return can occupy side, rear, and height channels while dry anchors remain in front. The return does not need to carry a moving trajectory; it needs a carefully designed covariance pattern, spectral balance, and decay envelope.

### 1.2 Scene-based audio

Scene-based systems describe a sound field around a listening point. Ambisonics is the most familiar example. Instead of storing one channel per loudspeaker, it stores spherical-harmonic components. A decoder then maps those components to the available loudspeaker array or to binaural headphones. First-Order Ambisonics (FOA) uses four components; higher orders increase spatial resolution at the cost of more channels and more demanding capture, processing, and decoding.

A scene can be rotated without reauthoring every loudspeaker feed. This makes Ambisonics useful for virtual reality, 360-degree video, game ambience, acoustic documentation, and portable reverberant fields. It is not the same as Dolby Atmos object metadata. Converting an Ambisonic scene into an Atmos session requires a render or a deliberate object/bed authoring step, not a channel-label rename.

### 1.3 Object-based audio

An object combines an audio signal with metadata describing where and how it should render. Position, divergence or size, and binaural behavior may vary over time. The final loudspeaker feeds are computed by a renderer for the target endpoint. That target might be a 7.1.4 room, 5.1, stereo, a soundbar, or binaural headphones.

Object audio is especially valuable when a sound must occupy a precise or moving location that a fixed bed cannot address independently. It is not automatically better for every reverb return. A dense, statistically diffuse late field often belongs in a bed because it should envelop rather than announce a point source. A featured reverse tail, rotating shimmer, or isolated reflection can be an excellent object because its trajectory is part of the composition.

## 2. Reading Surround and Height Layout Names

The notation `7.1.4` counts three groups: seven ear-level full-range channels, one low-frequency-effects channel, and four height channels. It says nothing by itself about whether the signal is a monitoring render, a DAW bus, a codec output, or an authored bed. Context matters.

| Name | Nominal contents | Typical role |
|---|---:|---|
| `2.0` | Left, Right | Stereo delivery or monitoring |
| `5.1` | Five full-range channels plus LFE | Broadcast, film, home-theater surround |
| `7.1` | Seven full-range channels plus LFE | Expanded side/rear surround |
| `7.1.2` | 7.1 plus two height channels | Standard Dolby Atmos bed; also a fixed-channel immersive bus |
| `7.1.4` | 7.1 plus four height channels | Common Atmos monitoring and speaker-render layout; **not** the default Atmos bed |
| `7.2.4` | Seven ear-level, two LFE, four height | Installation or custom discrete layout; not a standard Atmos bed name |
| FOA | Four spherical-harmonic components | Portable first-order scene representation |

The distinction is easier to see spatially. Figure 1 compares the ten fixed channels of a 7.1.2 bed with a 7.1.4 monitoring array. The Renderer may distribute bed and object energy among all available loudspeakers; the presence of four physical top loudspeakers does not turn the source bed into 7.1.4.

![A schematic comparison of a 7.1.2 Atmos bed and a 7.1.4 monitoring layout.](assets/immersive_audio/01_bed_vs_monitor_layout.png)

**Figure: Bed channels versus monitor channels.** The left plan shows the standard 7.1.2 bed concept: eight channels at or around ear level, including LFE, and two top-middle channels. The right plan shows a common 7.1.4 monitoring arrangement with separate front-top and rear-top loudspeakers. The positions are conceptual rather than installation specifications; use the room-design guidance appropriate to the renderer, room, and delivery contract.

### 2.1 Channel order is part of the format

For a standard Dolby 7.1.2 bed, Dolby documents the SMPTE order as `L, R, C, LFE, Ls, Rs, Lrs, Rrs, Ltm, Rtm`. Channel count alone cannot establish that order. A ten-channel WAV could contain the same signals in a different sequence and sound dramatically wrong while still opening successfully.

verbx currently uses the symbolic `7.1.2` labels `L, R, C, LFE, Ls, Rs, Lrs, Rrs, Ltf, Rtf`. Therefore, when preparing a verbx `7.1.2` file for an Atmos bed, **verify and explicitly map channels 9 and 10 to the Renderer’s `Ltm` and `Rtm` inputs**. The labels are not interchangeable merely because the channel count agrees. This is a handoff requirement, not a sonic preference.

Use a spoken channel-identification file or short, non-overlapping impulses before trusting a complex mix. A successful import proves only that the container is readable; it does not prove semantic routing.

## 3. Dolby Atmos Architecture

Dolby Atmos separates audio into beds and objects, then delegates endpoint-specific channel generation to a Renderer. According to Dolby’s current professional documentation, a typical music workflow uses one 7.1.2 bed and may use up to 118 objects within a system supporting up to 128 input paths. The exact available count can depend on how stereo objects and beds consume paths.

The bed carries fixed-channel material. It is appropriate for stems whose spatial role is broad and stable: an orchestral room, diffuse crowd, ensemble ambience, or reverberant field. Objects carry audio plus metadata. They are appropriate for a featured source or effect whose authored position, size, motion, or binaural behavior must survive downstream rendering.

Figure 2 shows why an Atmos master cannot be reduced to a 7.1.4 WAV. The bed, object signals, and metadata enter the Renderer separately. It can then derive multiple channel and binaural re-renders from one authored master.

![A signal-flow diagram showing a 7.1.2 bed, audio objects, and metadata entering the Dolby Atmos Renderer and producing multiple endpoint renders.](assets/immersive_audio/02_atmos_renderer_architecture.png)

**Figure: Dolby Atmos bed/object rendering architecture.** Fixed-channel bed audio, object audio with trajectories, and session metadata remain distinct until the Renderer. The Renderer combines them according to the target endpoint, producing outputs such as 7.1.4, 5.1, stereo, or binaural. A conventional multichannel WAV represents one channel render; it does not carry the complete object-based authoring model shown here.

### 3.1 Why the 7.1.2 bed does not reach every speaker independently

The standard bed provides a stable immersive foundation, but it does not offer separate fixed channels for every possible width or top-front/top-rear location. When a sound needs an independently addressable location outside the bed’s fixed destinations, author it as an object. This is one reason a hybrid mix is normal: the bed supplies continuity and envelopment while objects supply location-specific detail.

### 3.2 Objects are not moving tracks by default

An object may remain stationary for an entire piece. The useful property is not motion itself; it is that the Renderer receives a position and can translate it to each endpoint. Excessive motion can shrink perceived scale, distract from musical phrasing, or produce unstable binaural images. Reserve motion for an audible purpose.

### 3.3 The Renderer is part of the instrument

In object-based production, monitoring is already a render. A pan decision heard over 7.1.4 loudspeakers and the corresponding binaural result are two interpretations of the same metadata. Neither should be treated as a disposable preview. The production process must listen through the intended render paths early enough that translation problems can inform the mix.

## 4. Reverb as an Immersive Layer

Reverb contributes several perceptually distinct cues:

- Early reflections communicate apparent room size, source distance, surface placement, and localization.
- The late field communicates envelopment, decay, density, spectral absorption, and the persistence of the acoustic environment.
- Direction-dependent decay communicates architecture: a tall reflective space may retain energy above the listener while side energy damps sooner.
- Interchannel covariance controls whether the field feels broad, stable, phasey, collapsed, or detached.
- Modulation and diffusion determine whether discrete echoes fuse into a field or remain audible as patterns.

Treating all of those cues as one twelve-channel send leaves musical control on the table. Figure 3 presents a more useful topology. It divides the effect into early definition, a diffuse bed return, and an optional designed object return.

![A hybrid immersive-reverb signal-flow diagram with separate early-reflection, diffuse-bed, and object-effect branches.](assets/immersive_audio/03_hybrid_reverb_topology.png)

**Figure: Hybrid immersive-reverb topology.** A dry source feeds three sends with different perceptual jobs. Early reflections reinforce front and side localization; the diffuse late field enters a stable 7.1.2 bed return; and a special reverse, freeze, or moving tail is returned as a mono or stereo object. The Renderer combines those paths for loudspeaker and binaural endpoints. This separation prevents decorative motion from destabilizing the entire room impression.

### 4.1 Front channels and the center anchor

The center channel is powerful because it can create a stable screen or stage anchor. It is also easy to overload. Sending a highly correlated reverb return equally to Left, Center, and Right can build a narrow, level-heavy front image. For dialogue and lead vocals, begin with less late-field energy in Center than in Left/Right and surrounds. Preserve intelligibility with front-weighted early reflections, predelay, ducking, or spectral shaping rather than by eliminating all spatial response.

### 4.2 Side and rear channels

Side channels often carry the strongest envelopment cue because they stimulate lateral perception. Rear channels can extend depth, but excessive rear level can pull attention behind the listener or make the room feel detached from the source. A robust late field is usually decorrelated rather than copied. Identical tails in every channel may collapse during downmixing and can produce comb filtering or unstable images.

### 4.3 Height channels

Height should communicate vertical extent, overhead reflections, or a deliberate compositional gesture. Simply duplicating the bed return into top channels raises level without necessarily creating elevation. Consider later or spectrally different overhead energy, independent decorrelation, and a controlled height-to-bed ratio. In many natural rooms, the top field can be smoother and less transient-rich than the direct and early-reflection field.

Four-top monitoring gives a renderer more ways to express front-to-rear height movement, but a standard 7.1.2 bed still supplies only two fixed top-middle channels. Use objects if a return must be explicitly top-front, top-rear, or moving along the ceiling.

### 4.4 LFE is not bass management

The LFE channel is a creative effects channel, not the destination for all low-frequency content. Bass management is a monitoring-system function that redirects low frequencies from main channels according to crossover settings. A large reverb tail can already reach the subwoofer through bass management without being sent to LFE.

As a default, keep diffuse reverb out of LFE unless there is a clear artistic or delivery reason. Uncontrolled low-frequency decay consumes headroom, masks rhythm, and may translate unpredictably across systems. If LFE reverb is intentional, high-pass or band-limit it, monitor the LFE channel directly, and check the full-range re-render separately from a bass-managed room.

### 4.5 Decorrelation without disconnection

Immersive width requires difference among channels, but random difference is not automatically coherent. Useful decorrelation preserves a shared decay envelope and tonal identity while varying delay, modulation phase, diffusion state, or allpass structure. The goal is a single room observed from many directions, not twelve unrelated reverbs.

verbx exposes front, rear, and top decorrelation controls so these regions can be tuned independently. Start conservatively, then compare the discrete layout, stereo fold-down, and binaural render. If the immersive version sounds expansive but the stereo version becomes hollow, the channel relationships are too antagonistic.

## 5. What verbx Can Produce Today

The following boundary is intentionally explicit.

| Capability | Current status | Meaning for an Atmos workflow |
|---|---|---|
| Discrete multichannel WAVE rendering | Supported | Produce bed candidates, speaker prints, wet stems, and channel-identification files |
| Named input/output layouts | Supported | Declare routing intent for common layouts rather than relying only on channel count |
| Matrix convolution | Supported | Route each input independently to one or more output channels with an explicitly packed IR matrix |
| Algorithmic spatial decorrelation | Supported | Shape front, rear, and top late-field relationships |
| FOA encode, rotate, and stereo decode | Supported with documented constraints | Prepare or process a scene-based field before a separate Atmos authoring stage |
| Higher-order Ambisonic validation and channel-order metadata | Supported where documented by the CLI | Keep scene conventions explicit during handoff |
| JSON analysis and handoff manifests | Supported | Record channels, routing, settings, and QC evidence alongside audio |
| Dolby object trajectories and size automation | Not currently authored | Create these in an Atmos-capable DAW |
| Per-object Dolby binaural metadata | Not currently authored | Set and audition this in the DAW/Renderer |
| Native ADM BWF, DAMF, or IMF IAB master | Not currently written | Export or record the master through the Dolby Atmos Renderer or an integrated authoring environment |
| Dolby certification or delivery approval | Not implied | Follow the current distributor, label, broadcaster, studio, and Dolby requirements |

A JSON sidecar can be extremely useful, but it is documentation rather than embedded Dolby metadata. Renaming a sidecar or adding the letters “ADM” to a filename does not create the required BWF metadata chunk. Likewise, a 7.1.4 WAV is a speaker-channel render, not an editable object master.

## 6. Practical verbx Recipes

The commands below prepare audio for immersive sessions. They do not bypass the authoring and Renderer stages.

### 6.1 Create a 7.1.2 wet-bed candidate

Render a discrete ten-channel return, then explicitly map the final two channels during DAW import:

```bash
verbx render source.wav wet_712.wav \
  --engine algo \
  --output-layout 7.1.2 \
  --rt60 2.8 \
  --predelay 28 \
  --algo-decorrelation-front 0.22 \
  --algo-decorrelation-rear 0.48 \
  --algo-decorrelation-top 0.62 \
  --wet 1.0 --dry 0.0 \
  --json-out wet_712.analysis.json
```

On import, confirm `L, R, C, LFE, Ls, Rs, Lrs, Rrs` and remap verbx `Ltf, Rtf` to the intended Atmos bed `Ltm, Rtm` inputs. Do not infer the map from file width alone.

### 6.2 Create a 7.1.4 speaker print

Use this when a twelve-channel file is specifically required for monitoring, installation playback, or comparison. Do not call it an Atmos master:

```bash
verbx render source.wav wet_714_print.wav \
  --engine algo \
  --output-layout 7.1.4 \
  --rt60 3.6 \
  --algo-decorrelation-front 0.20 \
  --algo-decorrelation-rear 0.52 \
  --algo-decorrelation-top 0.70 \
  --wet 1.0 --dry 0.0 \
  --json-out wet_714_print.analysis.json
```

Name the file with `_print`, `_rerender`, or another unambiguous label so nobody mistakes it for a bed/object master.

### 6.3 Prepare a featured object-return stem

A special tail can remain mono or stereo until Atmos authoring. This preserves the DAW’s ability to attach object metadata:

```bash
verbx render vocal_throw.wav vocal_reverse_object.wav \
  --engine algo \
  --reverse \
  --rt60 6.5 \
  --predelay 90 \
  --wet 1.0 --dry 0.0 \
  --output-layout mono \
  --json-out vocal_reverse_object.analysis.json
```

Import the result, assign it to an object, then author the static position or trajectory while listening through the Renderer. A stereo effect may consume two object paths; use stereo only when its internal width is essential.

### 6.4 Process an FOA field

For a scene-based intermediate, keep ordering and normalization explicit:

```bash
verbx render stereo_room.wav room_foa.wav \
  --ambi-order 1 \
  --channel-order acn \
  --ambi-encode-from stereo \
  --ambi-rotate-yaw-deg 25 \
  --output-layout auto \
  --json-out room_foa.analysis.json
```

FOA is useful when rotation and portable scene decoding matter more than sharply isolated positions. To use it in Atmos, decode or spatialize it through a compatible plug-in or DAW workflow and audition the resulting bed/object assignment. Do not relabel the four channels as an Atmos bed.

### 6.5 Use a full matrix IR

A true multichannel room response can encode direction-dependent coupling. For $M$ inputs and $N$ outputs, a full matrix contains $M N$ impulse responses. The file packing order must match `--ir-matrix-layout`:

```bash
verbx render input_51.wav room_matrix_714.wav \
  --input-layout 5.1 \
  --output-layout 7.1.4 \
  --ir concert_hall_6x12.wav \
  --ir-route-map full \
  --ir-matrix-layout output-major \
  --wet 1.0 --dry 0.0 \
  --json-out room_matrix_714.analysis.json
```

Before using a large matrix in a mix, test one input at a time. Confirm that front impulses create the intended early and late energy, rear inputs do not swap, top outputs are truly top outputs, and no LFE row is populated unintentionally.

## 7. Bed, Object, and Hybrid Reverb Strategies

### 7.1 Bed return

A bed return is the safest default for a shared acoustic environment. Route multiple sources to one immersive reverb, preserve a coherent decay law, and use channel-dependent decorrelation to create width. This approach conserves object paths and keeps the room stable as sources move.

For orchestral or ensemble work, the bed can establish the hall while dry and early-reflection cues retain stage placement. For dialogue, a subtle bed can match production ambience or establish scene scale without attaching the entire room to a moving actor object.

### 7.2 Object return

An object return is useful when the reverb itself is a featured event. Examples include a freeze that rises above the listener, a reverse tail that approaches from the rear, a single reflection that traces a wall, or a transition wash that grows from near to far. Render the audio effect in verbx, then perform the object authoring in the Atmos environment.

Avoid making every source’s complete late field a separate moving object. The result can consume paths, complicate metadata, and make the perceived room chase the dry source. In real rooms, direct sound localizes strongly while the late field becomes increasingly diffuse.

### 7.3 Hybrid return

The hybrid strategy uses a shared bed for the ordinary room and one or more objects for exceptional cues. It scales well because the bed carries statistical density while objects remain sparse and meaningful. It also translates more gracefully: if a dramatic object changes character in binaural, the foundational room remains intact.

### 7.4 Stem architecture

A practical immersive session might contain:

- `Room_ER_front`: early reflections emphasizing front and side channels.
- `Room_late_712`: diffuse 7.1.2 bed return.
- `Vocal_throw_mono`: mono special-effect stem assigned to an object.
- `Orchestra_halo_stereo`: stereo height effect assigned as a stereo object or deliberately spread into the bed.
- `LFE_effect`: separately managed, intentional LFE-only content if required.
- `Room_QC`: channel-identification impulses and the associated JSON report, excluded from the final program.

Use names that describe both acoustic function and spatial role. “BigVerb12” is less useful than “HallLate_712Bed” when the session is reopened six months later.

## 8. DAW and Renderer Handoff

### 8.1 Integrated-renderer workflow

In an integrated environment such as Logic Pro, assign tracks and returns to the bed or to objects, place Atmos-aware processing in the correct part of the signal path, and monitor through the project’s Dolby Atmos plug-in. Apple documents an important signal-flow distinction: processing before the Atmos plug-in operates on the bed path, while processing after it affects monitoring and channel-based bounces rather than the exported object master. Verify the current application behavior rather than assuming an ordinary surround-master insert model.

Export an ADM BWF only from the supported spatial-audio export path. Apple describes that export as containing bed audio, object audio, object pan automation, and metadata. A normal multichannel bounce is a channel-based render and serves a different purpose.

### 8.2 External-renderer workflow

With a separate Dolby Atmos Renderer, route each bed channel and object path to its assigned Renderer input. Confirm sample rate, frame rate, synchronization, input assignment, and monitoring return before mixing. Record or export the required master through the Renderer workflow, then generate re-renders and QC material from that same authoritative master.

### 8.3 Handoff package

A strong handoff includes more than audio:

- Clearly named mono, stereo, and multichannel WAV files.
- A channel-order document for every file wider than stereo.
- Sample rate, bit depth, start time, frame rate where applicable, and exact duration.
- A verbx JSON report containing processing parameters and measured properties.
- Dry references when the receiving mixer may need to revise the effect.
- A short text note identifying bed candidates, object candidates, and speaker prints.
- Channel-identification impulses or spoken IDs for nonstandard layouts.
- The expected Atmos authoring environment and Renderer version, when contractually relevant.

Do not place a speaker print and an object-master candidate in the same folder with nearly identical names. Make semantic differences visible at a glance.

## 9. Monitoring and Translation

No single playback path proves an immersive master. A 7.1.4 room reveals image placement and envelopment; 5.1 reveals whether height-dependent balances survive; stereo reveals center buildup, phase cancellation, and overall hierarchy; binaural reveals renderer-dependent externalization, elevation, and near/far behavior.

Figure 4 frames monitoring as a loop. The same authoritative master feeds each re-render. Differences that damage the musical intent return to the source session for correction; they are not patched independently into unrelated exports.

![A quality-control loop connecting a source session and reference Renderer to 7.1.4, 5.1, stereo, and binaural endpoints.](assets/immersive_audio/04_translation_qc_loop.png)

**Figure: Immersive translation and quality-control loop.** The authored beds, objects, and automation feed one reference Renderer. Four representative endpoints reveal different failure modes: 7.1.4 tests the intended room image, 5.1 exposes reliance on height, stereo tests hierarchy and phase interaction, and binaural tests headphone localization and metadata behavior. Observations return to the source session so the master remains the single source of truth.

### 9.1 Loudspeaker-room checks

An immersive room should be calibrated, time-aligned, and level-matched according to the applicable standard and facility procedure. Check that monitoring bass management is understood and repeatable. A subwoofer response problem can be misdiagnosed as an LFE-mix problem; a mistimed height speaker can be misdiagnosed as poor elevation metadata.

Walk the room after the reference-position pass. The mix need not be identical in every seat, but catastrophic localization jumps, comb filtering, or disappearing returns reveal fragile channel relationships.

### 9.2 Stereo checks

Listen for excessive front-center buildup, loss of ambience, hollow tonal balance, and transient smearing. A diffuse field can become quieter in stereo because decorrelated channels partially cancel or because the renderer applies conservative downmix coefficients. Correct the spatial design, not merely the stereo file, unless the delivery contract explicitly calls for a separately mastered stereo version.

### 9.3 Binaural checks

Binaural rendering depends on the Renderer’s head-related transfer functions and object metadata. A height cue that is obvious over loudspeakers may become subtle over headphones; a large diffuse bed may sound internalized; a near object may become distractingly intimate. Check several good headphones and avoid making decisions from a single consumer spatializer layered on top of the reference Renderer.

When supported, per-object binaural modes such as Off, Near, Mid, and Far affect perceived distance. These modes are metadata decisions, not substitutions for acoustic predelay, direct-to-reverberant ratio, spectral damping, or early-reflection design.

### 9.4 Renderer and consumer-device differences

Soundbars, televisions, headphones, mobile devices, and theatrical systems may use different rendering strategies. The correct response is not to optimize for every device independently. Establish a defensible reference master, test the required official re-renders, and investigate only repeatable failures that compromise intent.

## 10. Ambisonics and Atmos Are Complementary

Ambisonics represents a listener-centered field; Atmos represents beds and objects for endpoint rendering. Both can describe immersion, but their coordinate systems and production assumptions differ.

| Question | Ambisonics | Dolby Atmos |
|---|---|---|
| What is stored? | Spherical-harmonic scene components | Bed channels, object audio, and metadata |
| Primary strength | Rotation and layout-independent scene decoding | Authored object placement plus broad endpoint ecosystem |
| Spatial resolution | Increases with Ambisonic order | Depends on object/bed authoring and endpoint renderer |
| Typical reverb use | Portable room field or environmental ambience | Bed ambience plus optional object effects |
| Headphone output | Binaural decode of the scene | Renderer binaural output with object metadata behavior |
| Conversion | Requires decoding or compatible spatialization | Requires an authoring decision, not relabeling |

An Ambisonic room recording can become an excellent Atmos ambience after deliberate decoding and routing. One option is to decode it to a 7.1.2 bed. Another is to use an Ambisonic-capable spatializer that feeds an Atmos-compatible path. A third is to separate salient directional events as objects while retaining the diffuse field in the bed. Choose based on the scene, not on the desire to maximize the number of moving icons.

## 11. Deliverables and File Types

The final file format determines what remains editable and what metadata travels with the audio.

### 11.1 Multichannel WAVE

A conventional multichannel WAVE file carries discrete PCM channels and container metadata. It can be a bed candidate, a loudspeaker print, a re-render, an installation master, or a stem. It does not inherently carry Atmos object trajectories. Channel labels and order must be documented and verified.

### 11.2 ADM BWF

An ADM BWF combines PCM audio with Audio Definition Model metadata in a Broadcast Wave container. Dolby and Apple describe it as an interchange master capable of carrying bed and object audio with the associated metadata. It is not a consumer listening file and should not be confused with a normal multichannel bounce. Dolby notes that changes to an imported ADM BWF are generally made in the originating authoring application rather than edited as if it were a native Renderer session.

### 11.3 DAMF

A Dolby Atmos Master File set is the Renderer’s native master representation, commonly involving `.atmos`, `.audio`, and `.metadata` files. It supports continuing work in the Renderer-oriented ecosystem and can be preferable when an editable Renderer master is required.

### 11.4 IMF IAB

IMF Immersive Audio Bitstream packages are associated with interoperable mastering and distribution workflows. Use them only when required by the delivery specification and generated through the appropriate toolchain.

Figure 5 marks the practical boundary. verbx creates audio assets and evidence on the left; the Atmos DAW and Renderer author spatial metadata and master formats on the right.

![A delivery diagram separating verbx-generated audio and reports from Atmos metadata authoring and master-file generation.](assets/immersive_audio/05_delivery_boundary.png)

**Figure: verbx-to-Atmos delivery boundary.** verbx can create discrete stems, Ambisonic scenes, matrix-convolved outputs, and reports that make a handoff reproducible. The receiving Atmos session assigns beds and objects, writes trajectories and binaural metadata, renders endpoints, and produces the required ADM BWF, DAMF, or IMF IAB master. The arrow represents explicit DAW import and mapping, not an automatic file conversion.

## 12. Immersive Quality-Control Checklist

### 12.1 Before authoring

- Confirm every source file’s sample rate, bit depth, duration, start point, and channel count.
- Read the verbx JSON report and compare it with the filename and handoff note.
- Run a channel-identification file through the exact import route.
- Verify whether a wide file is a bed candidate, a speaker print, an Ambisonic scene, or a matrix IR output.
- Confirm that no limiter, normalizer, or loudness stage was applied accidentally to a wet stem.

### 12.2 During Atmos authoring

- Verify 7.1.2 bed order as `L, R, C, LFE, Ls, Rs, Lrs, Rrs, Ltm, Rtm` at the Renderer boundary.
- Solo every bed channel and object input at least once.
- Confirm mono versus stereo object assignments and path consumption.
- Inspect object position, size, motion, and binaural metadata for unintended automation.
- Check whether master-bus processing affects the bed, the monitor render, the exported master, or some combination.
- Monitor both full-range and LFE paths without conflating LFE with bass management.

### 12.3 Before delivery

- Verify or record the final master through the authoritative Renderer workflow.
- Play the master from beginning to end in the Renderer rather than checking only a DAW bounce.
- Check all contractually required re-renders, including stereo and binaural where applicable.
- Measure loudness and true peak on the deliverables required by the recipient; do not assume one target for every platform.
- Confirm sync, frame rate, program start, tail duration, and end-of-file behavior.
- Compare the delivered master’s metadata and duration with the source session.
- Preserve the source session, Renderer master, verbx reports, and delivery notes together.

## 13. Creative Spatial-Reverb Recipes

### 13.1 Natural hall around a front ensemble

Keep direct sound and the strongest early reflections in the front stage. Build the late field primarily in side, rear, and top bed channels with moderate decorrelation. Use less Center late energy than Left/Right unless the production calls for a tightly anchored acoustic. Confirm that stereo retains warmth without turning cloudy.

### 13.2 Intimate vocal with an overhead halo

Keep the vocal itself anchored. Send a filtered, predelayed wet stem to a stereo object or to carefully controlled top energy. The halo should respond to phrasing rather than run continuously at the same level. Duck the return during consonants and let it recover into rests. In binaural, compare Near, Mid, and Far metadata choices with the acoustic predelay; they solve different distance cues.

### 13.3 Percussion room with stable transients

Use short front and side early reflections to establish room size. Feed the late field after enough predelay to preserve attacks. Increase rear/top decorrelation more than front decorrelation. Keep low-frequency decay shorter than midrange decay, and avoid LFE unless a specific impact effect requires it.

### 13.4 Reverse transition over the listener

Render a mono or stereo reverse-reverb stem in verbx. Author it as an object that moves only during the transition, then resolves into a stable bed or dry source. Check the movement over 7.1.4 and binaural; simplify the trajectory if headphone localization becomes erratic.

### 13.5 Exterior-to-interior scene change

Automate the acoustic architecture rather than merely raising wet level. Begin with sparse, low-density exterior reflections. Introduce enclosed early reflections as the threshold is crossed, then lengthen and darken the late field. Keep the shared environment in the bed and reserve objects for reflections tied to visible architectural events.

### 13.6 Infinite or frozen field

A frozen field can occupy a bed when it represents an environment, or an object when it behaves as a featured musical entity. Watch low-frequency accumulation and interchannel correlation. A stationary, diffuse freeze often feels larger than a freeze that continuously circles the listener.

## 14. Common Failure Modes

### “The height disappears in stereo.”

Height is not a separate axis in two-channel playback. The Renderer must express it through spectral, timing, level, and binaural cues. Strengthen the compositional hierarchy and test the reference binaural render; do not simply add top channels to the stereo mix.

### “The 7.1.4 file imported, so the Atmos master is finished.”

It is a channel-based speaker print. It does not contain editable object trajectories or the complete Atmos master metadata. Import stems into an Atmos authoring environment and create the required master there.

### “The top-front and top-rear channels are silent in my bed.”

A standard 7.1.2 bed supplies top-middle channels, not four independently addressable top channels. The Renderer maps the bed into the monitoring layout. Use objects for independently authored top-front or top-rear content.

### “The room sounds huge over speakers but phasey on headphones.”

Reduce antagonistic decorrelation, inspect copied channels, and compare the Renderer’s binaural output without an additional consumer spatializer. Keep the shared decay envelope coherent even when channel details differ.

### “The subwoofer is full of reverb although LFE is empty.”

That can be normal bass management. Full-range bed channels contain low frequencies that the monitor controller redirects below its crossover. Solo LFE and inspect the monitoring configuration before changing the mix.

### “Channels 9 and 10 sound too far forward.”

Check the handoff map. verbx’s current symbolic `7.1.2` utility layout labels those channels `Ltf/Rtf`, whereas a Dolby bed expects `Ltm/Rtm`. Map them explicitly and verify with identification signals.

### “The ADM BWF does not reflect my last monitor-chain EQ.”

Determine where the processing sits relative to the Atmos authoring and Renderer path. Monitoring or post-render processing may affect a channel bounce without changing the exported object master. Use the DAW’s official signal-flow documentation and re-export from the authoritative session.

## 15. Recommended Official Reading

The following primary documentation is arranged alphabetically by organization:

- **Apple.** [Export a spatial audio project as an ADM BWF file in Logic Pro](https://support.apple.com/guide/logicpro/export-a-spatial-audio-project-dolby-atmos-lgcp258ed132/mac). Explains the difference between an object-based spatial export and a conventional channel bounce.
- **Apple.** [Monitor a Spatial Audio mix in Logic Pro](https://support.apple.com/en-lamr/guide/logicpro/lgcp179f27c1/mac). Describes speaker, stereo, and binaural monitoring formats.
- **Apple.** [Use beds and objects in Logic Pro](https://support.apple.com/en-mide/guide/logicpro/lgcp73f9b9ac/mac). Covers integrated bed/object assignment and path limits.
- **Dolby.** [Dolby Atmos Renderer](https://professional.dolby.com/product/dolby-atmos-content-creation/dolby-atmos-renderer/). Product-level description of bed/object rendering, master creation, and re-renders.
- **Dolby.** [How do I QC my Dolby Atmos mix?](https://professionalsupport.dolby.com/s/article/How-do-I-QC-my-Dolby-Atmos-mix). Emphasizes master verification in the Renderer and delivery-specific requirements.
- **Dolby.** [Overview of Dolby Atmos Master File Formats](https://professionalsupport.dolby.com/s/article/Overview-of-Dolby-Atmos-Master-File-Formats). Compares ADM BWF, DAMF, and IMF IAB roles.
- **Dolby.** [What are Beds and Objects in Dolby Atmos?](https://professionalsupport.dolby.com/s/article/What-are-Beds-and-Objects-in-Dolby-Atmos). Defines the default 7.1.2 bed and object workflow.
- **Dolby.** [What channel order should be used for assigning bed audio to the Renderer?](https://professionalsupport.dolby.com/s/article/What-channel-order-should-be-used-for-assigning-bed-audio-to-the-Renderer). Gives the SMPTE 7.1.2 bed order used at the Renderer boundary.
- **International Telecommunication Union.** [Recommendation ITU-R BS.2051: Advanced sound system for programme production](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.2051-2-201807-S%21%21PDF-E.pdf). Defines reference loudspeaker-layout concepts for advanced sound systems.

Specifications, application behavior, and distributor requirements change. Treat these links and the recipient’s current delivery document as authoritative for a production, and treat this chapter as an engineering and creative workflow guide.
