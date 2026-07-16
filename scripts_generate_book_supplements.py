#!/usr/bin/env python3
"""Generate the introductory diagram chapter and educational appendix."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path
from urllib.parse import quote_plus

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "docs" / "assets" / "intro_block_diagrams"

DIAGRAMS = (
    ("The complete verbx workflow", ("Audio or IR", "Choose engine", "Shape space", "Measure", "WAV + JSON"), "Every workflow begins with an identified source and ends with both an audible artifact and machine-readable evidence."),
    ("Algorithmic render path", ("Input frames", "Early reflections", "Diffusers", "FDN late field", "Output stage"), "The algorithmic engine separates arrival cues, density growth, sustained decay, and delivery processing."),
    ("Convolution render path", ("Input frames", "IR prepare", "FFT partitions", "Matrix routing", "Output stage"), "Partitioned convolution trades a small fixed scheduling delay for efficient processing of long and multichannel impulse responses."),
    ("Realtime callback boundary", ("Audio device", "Input buffer", "Realtime DSP", "Output buffer", "Audio device"), "The callback path must remain bounded and allocation-free; preparation, file access, and reporting belong outside it."),
    ("End-to-end latency stack", ("Input safety", "Host block", "DSP delay", "Output safety", "Acoustic arrival"), "Perceived realtime latency is the sum of device buffers, host scheduling, algorithmic lookahead, and physical propagation."),
    ("Reverb event anatomy", ("Direct sound", "Pre-delay", "Early field", "Density build", "Late decay"), "A useful listening model treats reverberation as an ordered event rather than a single wet signal."),
    ("RT60 control hierarchy", ("Coarse RT60", "Fine trim", "Band tilt", "Freeze logic", "Effective tail"), "Coarse and fine controls establish time scale while frequency-dependent decay and freeze determine how energy persists."),
    ("FDN feedback loop", ("Delay bank", "Damping filters", "Mixing matrix", "Feedback gain", "Delay bank"), "Delay lengths create modes; filters set color; the matrix disperses energy; calibrated gain establishes decay."),
    ("Impulse-response lifecycle", ("Capture or synthesize", "Inspect", "Trim", "Normalize", "Version + hash"), "An impulse response is an auditable asset whose provenance and preparation affect every convolution result."),
    ("Dereverberation path", ("Time-frequency analysis", "Room estimate", "Suppression mask", "Artifact guard", "Resynthesis"), "Dereverberation balances late-energy reduction against speech, transient, and ambience preservation."),
    ("Dynamics around reverb", ("Dry detector", "Ducking envelope", "Wet return", "Limiter", "Delivery meter"), "Ducking creates foreground space before a limiter protects the combined output and meters verify the result."),
    ("Spatial routing", ("Source channels", "Bus layout", "Room matrix", "Decoder", "Speakers or binaural"), "Channel identity must remain explicit from source through room processing and final reproduction."),
    ("Automation contract", ("Timeline", "Validate points", "Interpolate", "Apply at boundary", "Report effective state"), "Automation is safest when parsed and validated off-thread, then applied deterministically at block boundaries."),
    ("Analysis evidence chain", ("Rendered audio", "Metrics", "Thresholds", "Warnings", "JSON report"), "Analysis becomes operational evidence when measurements, policy thresholds, warnings, and provenance share one report."),
    ("Preset lifecycle", ("Design intent", "Parameter state", "Audition matrix", "Serialize", "Recall test"), "A preset is more than values: it needs intent, compatibility metadata, calibrated auditioning, and deterministic recall."),
    ("Plug-in host contract", ("Host state", "Parameter manifest", "Realtime core", "Bus negotiation", "Host output"), "AUv3 and VST3 integration wrap the same DSP in host-specific state, layout, and lifecycle contracts."),
    ("Regression workflow", ("Golden fixture", "Render candidate", "Compare audio + JSON", "Triage boundary", "Archive evidence"), "Deterministic fixtures make audible and numerical drift discoverable before release."),
    ("Learning loop", ("Predict", "Render", "Measure", "Listen", "Revise model"), "The textbook exercises use repeated prediction, experiment, measurement, and critical listening to connect controls with perception."),
)

PROJECTS = (
    ("Calibrate a Natural Room", "RT60, pre-delay, early/late balance", "Create three believable rooms for speech, percussion, and strings from one dry recording."),
    ("Map the Reverb Event", "direct sound, early reflections, late decay", "Annotate arrival times and energy regions in an impulse response and in a rendered waveform."),
    ("Coarse and Fine RT60 Laboratory", "logarithmic control and parameter resolution", "Measure how coarse and fine controls distribute useful adjustment over 0.01 to 360 seconds."),
    ("Frequency-Dependent Decay", "multiband RT60 and damping", "Design warm, neutral, and bright tails with equal broadband RT60 but different band decay."),
    ("Pre-Delay and Apparent Distance", "depth cues and source separation", "Hold loudness constant while testing whether pre-delay changes foreground placement."),
    ("Early-Reflection Geometry", "room dimensions and directional cues", "Compare compact, wide, and tall reflection patterns using the same late field."),
    ("Diffusion Build-Up", "echo density and allpass structure", "Find the transition from discrete echoes to a smooth field for three source types."),
    ("FDN Matrix Listening Test", "feedback matrices and modal texture", "Blind-compare matrix families under matched delay lengths, RT60, and output level."),
    ("Delay-Line Distribution", "mode spacing and metallic coloration", "Construct prime, randomized, and clustered delay sets and relate spectra to listening notes."),
    ("Compose with Infinite Sustain", "freeze as harmonic memory, voice-leading, and density control", "Compose a miniature in which captured tails become sustained harmonic material rather than background ambience."),
    ("Reverse-Reverb Phrase Study", "anacrusis, envelope reversal, and transition timing", "Compose phrase-leading reverse tails whose arrivals articulate pickups, cadences, and formal boundaries."),
    ("Shimmer Canon and Harmonic Field", "pitch-shifted feedback, canonic imitation, and spectral register", "Write a canonic miniature in which shimmer intervals extend a limited pitch collection into a controlled harmonic field."),
    ("Ducked-Reverb Rhythmic Counterpoint", "sidechain rhythm, negative space, and call-and-response", "Compose a groove in which the wet return answers the dry material with clearly perceived rhythmic counterpoint."),
    ("Limiter Transparency", "lookahead, knee, release, and true peak", "Protect an extreme wet mix while minimizing transient flattening and pumping."),
    ("Convolution Partition Plan", "latency versus throughput", "Select partition sizes for live monitoring, music production, and offline archival rendering."),
    ("Impulse-Response Capture", "sweeps, deconvolution, and noise", "Plan and document a repeatable room capture with calibration and failure checks."),
    ("IR Editing and Provenance", "trim, normalize, resample, and hash", "Prepare an IR library while preserving source identity and reversible processing metadata."),
    ("Morphing-Space Miniature", "IR interpolation, formal trajectory, and perceptual continuity", "Compose a miniature whose formal sections are defined by a continuous journey between contrasting spaces."),
    ("Dereverberation Tradeoff", "late-energy suppression and artifacts", "Build a strength sweep and select an operating point from measured and blind-listening evidence."),
    ("Speech Intelligibility Study", "DRR, clarity, and spectral masking", "Compare natural reverb, dereverberation, and re-reverberation for spoken material."),
    ("Realtime Latency Audit", "device, host, DSP, and acoustic delay", "Measure round-trip latency and reconcile it with the predicted buffer stack."),
    ("Realtime Failure Injection", "xruns, overload, and recovery", "Stress block size and quality modes while recording dropout thresholds and recovery behavior."),
    ("Spatial Automation as Musical Form", "timeline interpolation, depth dramaturgy, and deterministic state", "Compose a work in which automated depth, width, and decay create the principal large-scale form."),
    ("Compose a Reverb-Preset Etude", "semantic preset design, variation, and timbral orchestration", "Design a related preset family and compose an etude whose sections reveal each preset's distinct musical role."),
    ("Loudness-Matched Evaluation", "LUFS, peak, and perceptual bias", "Compare six reverb designs only after integrated loudness and peak constraints are matched."),
    ("Stereo Width Without Phase Damage", "correlation and mono compatibility", "Increase envelopment while maintaining a documented mono fold-down target."),
    ("Multichannel Bus Verification", "channel identity and matrix routing", "Use labeled impulses to prove every input-to-output route in a surround configuration."),
    ("Ambisonic Rotational Counterpoint", "scene representation, spatial imitation, and decoding", "Compose a multi-voice texture whose contrapuntal identities remain clear while their spatial trajectories rotate and exchange positions."),
    ("Room-Model Inversion", "Sabine estimates and geometric constraints", "Infer plausible dimensions and absorption from RT60, then explain non-unique solutions."),
    ("Ray-Traced Early Field Proposal", "CAD geometry, materials, and validation", "Specify a DXF-to-reflection prototype with assumptions, error bounds, and listening tests."),
    ("Machine-Readable Evidence", "analysis JSON and schema stability", "Design a report consumed by a regression script without parsing human progress text."),
    ("Native Plug-in Parity Slice", "C++ DSP, host state, and deterministic comparison", "Port one narrow feature and prove parity against the Python reference."),
    ("Original Spatial Composition", "original composition, spatial form, and integrated engineering practice", "Create an original multichannel work in which reverberation is indispensable to the composition's harmony, rhythm, gesture, or form."),
    ("Harmonic Pedals from Decay", "crossfaded tails, harmonic memory, and voice-leading", "Compose a harmonic sequence in which each decaying room becomes a pedal tone beneath the next sonority."),
    ("Tempo-Synchronized Decay Canon", "tempo ratios, RT60, meter, and imitation", "Write a canon whose tail lengths articulate changing subdivisions and metric relationships."),
    ("Antiphonal Virtual Architecture", "call-and-response, early reflections, and contrasting rooms", "Compose for two virtual ensembles whose phrases and rooms answer one another across an imagined architecture."),
    ("Multiband Tail Orchestration", "register, frequency-dependent decay, and instrumental balance", "Orchestrate low, middle, and high registers with different decay profiles that function as independent musical layers."),
    ("Percussive Reflection Groove", "early-reflection timing, transient rhythm, and groove", "Build a percussion study in which discrete early reflections supply notated offbeats, flams, and polyrhythms."),
    ("Vocal Space Dramaturgy", "narrative distance, intimacy, and spatial automation", "Compose a song or spoken scene whose emotional arc is carried by changing vocal distance and enclosure."),
    ("Dub Send Performance", "reverb throws, feedback, mute gestures, and live form", "Perform a mix in which manually played sends and returns become an improvising instrumental part."),
    ("Silence and Cadential Decay", "rests, phrase endings, and composed resonance", "Write a sequence of cadences in which measured silence and audible decay complete, delay, or contradict closure."),
    ("Room-Swap Theme and Variations", "variation form, orchestration, and acoustic identity", "Create a theme and four variations whose musical characters emerge from contrasting room designs."),
    ("Site-Specific IR Portrait", "impulse-response capture, resonance, and place", "Capture or document a real acoustic and compose a portrait that responds to its measured modes, clarity, and decay."),
    ("Modulated-Reverb Timbre Study", "modulation rate, depth, pitch stability, and color", "Compose a timbral etude in which reverb modulation changes orchestral color without obscuring pitch or pulse."),
    ("Spatial Fugue Across Rooms", "fugal process, room identity, and moving perspective", "Write a fugue whose entries retain contrapuntal identity while migrating among distinct acoustic spaces."),
    ("Adaptive Reverb Cue", "interactive states, seamless transitions, and narrative space", "Compose a modular cue whose reverb responds coherently to three imagined game or installation states."),
    ("Reverb-Orchestrated Song", "sectional contrast, vocal clarity, and send orchestration", "Produce an original song whose verse, chorus, and bridge use distinct but related spatial orchestrations."),
    ("Portfolio Capstone: Spatial Composition Cycle", "portfolio curation, extended form, and production delivery", "Create a multi-movement spatial composition cycle that synthesizes at least three earlier compositional techniques."),
)

COMPOSITION_PROJECTS = {
    10: {
        "procedure": "Write a two- to three-minute piece in three connected sections. Introduce a compact motif or chord field, capture at least three freeze states at structurally important moments, and release or filter them so each frozen layer changes the harmony or orchestration. Print separate dry, ordinary-wet, and frozen-return stems; leave deliberate rests so listeners can hear how the room memory develops. Test every sustained state for peak growth and spectral crowding before the final render.",
        "evidence": "Submit a stereo or multichannel master, dry and return stems, a one-page formal or score map marking every capture and release, exact commands or plug-in automation, presets, analysis JSON, and a labeled spectrogram or energy plot. Identify the pitch content retained by each frozen state and document the safety settings that keep it stable.",
        "questions": "When does a frozen tail become a musical voice rather than an effect? How does each capture alter the perceived harmony, density, or meter? Which register accumulates fastest, and what compositional decision solves that buildup without merely turning the return down?",
        "completion": "The piece must make at least three clearly intentional freeze gestures that participate in voice-leading or form. A listener should be able to describe the harmonic function of the sustained material, and the full-duration render must remain stable and unclipped.",
        "extension": "Adapt the piece for realtime performance, assigning capture, release, damping, and clear actions to playable controls; document a recovery gesture for an accidental or unstable freeze.",
    },
    11: {
        "procedure": "Choose a tempo and compose a 60- to 90-second solo, vocal, or chamber etude with at least four reverse-reverb anticipations. Place one before an opening entrance, one across an internal phrase boundary, one at a cadence, and one that deliberately misdirects the listener. Align each swell to the destination attack in beats or milliseconds, then vary duration, spectral emphasis, and wet level without moving the performed note.",
        "evidence": "Submit the score or annotated timeline, dry and processed masters, isolated reverse-return stems, exact commands or automation, and a table relating tempo subdivisions to swell durations. Include analysis JSON and one labeled waveform showing the destination transient and at least two reverse envelopes.",
        "questions": "Which swells sound like pickups, crescendos, breaths, or causal impossibilities? How early can the destination be predicted without weakening surprise? Does the reverse gesture reinforce the meter, create a cross-rhythm, or suspend pulse?",
        "completion": "At least four reverse events must have distinct phrase functions, land accurately, and preserve the destination transient. The musical result must still read as a coherent composition when the reverse-return stem is muted, while the processed version must reveal a stronger formal design.",
        "extension": "Recompose one passage so the reverse tail supplies the only audible preparation for a silent or omitted destination note.",
    },
    12: {
        "procedure": "Limit the source to a five- to seven-note pitch collection and write a two-minute canon for two or more voices. Choose two shimmer intervals, such as an octave and a fifth, and assign each a formal role. Introduce voices one at a time, automate feedback and damping to move from sparse imitation to a shared harmonic cloud, and reserve the brightest register for a single structural climax.",
        "evidence": "Submit a notated score or piano-roll reduction, dry voices, shimmer returns, final master, pitch-interval map, presets or automation, analysis JSON, and a labeled time-frequency plot. Mark every point at which the feedback path creates a pitch not present in the performed material.",
        "questions": "Does the shimmer field preserve the canon or dissolve it? Which generated partials function as harmony, orchestration, or noise? How do feedback, damping, and source register change the perceived consonance of the same transposition interval?",
        "completion": "The piece must demonstrate intelligible imitation, purposeful harmonic expansion, and a controlled spectral climax without runaway high-frequency energy. Both shimmer intervals must be identifiable by ear or convincingly justified in the analysis.",
        "extension": "Create a second realization using a non-octave interval and revise the source pitch collection so the feedback field remains harmonically coherent.",
    },
    13: {
        "procedure": "Compose a 24- to 32-bar groove with a dry foreground line and a reverb return designed as its answering voice. Build three sections using release times related to contrasting subdivisions, such as eighth notes, dotted eighths, and quarter-note triplets. Shape the source rhythm and its rests so the return articulates syncopations rather than merely filling gaps; retain one passage where ducking is bypassed for comparison.",
        "evidence": "Submit a tempo map, rhythmic reduction of dry attacks and wet responses, dry and return stems, bypass comparison, final master, exact ducking and reverb settings, analysis JSON, and a labeled envelope plot covering at least two bars.",
        "questions": "Where does the return behave like echo, sustain, or a separate contrapuntal layer? Which release relationship strengthens the groove, and which creates metric ambiguity? How much source silence is required before the wet rhythm becomes independently legible?",
        "completion": "The wet return must produce three audibly different rhythmic roles while the foreground remains intelligible. At least one section must create a convincing call-and-response pattern that disappears when ducking is bypassed.",
        "extension": "Arrange the return as a second percussion part for live players, then compare the performed and signal-processed counterpoints.",
    },
    18: {
        "procedure": "Select three contrasting impulse responses that suggest intimate, public, and impossible spaces. Compose a two- to four-minute miniature with an opening, transition, and arrival whose spatial trajectory is as deliberate as its harmonic trajectory. Build at least five reproducible morph states, include one continuous transition lasting a complete phrase, and avoid changing orchestration during one decisive morph so the space itself carries the form.",
        "evidence": "Submit the score or formal timeline, source and final audio, isolated wet stem, all IRs with provenance and hashes, morph settings, analysis JSON, and a labeled plot of morph position against musical time. Provide short listening annotations for every state and discontinuity check.",
        "questions": "Can listeners identify formal change before they identify the new room? Which acoustic attributes interpolate smoothly, and which appear to switch categories? Does the same harmony acquire a different function when its apparent architecture changes?",
        "completion": "The spatial trajectory must create a perceptible beginning, transformation, and destination without relying only on level fades. IR provenance must be complete, all five states reproducible, and no unintended click or timbral discontinuity may remain.",
        "extension": "Reverse the spatial journey while preserving the notes and durations, then explain which aspects of the musical form also seem reversed.",
    },
    23: {
        "procedure": "Compose a three- to five-minute work whose primary form is a trajectory through foreground, middle distance, enclosure, and release. Automate at least four spatial variables, including RT60, pre-delay, wet/dry balance, and width or position, but introduce them in stages so their musical functions remain audible. Render the same automation at multiple block sizes and preserve a fixed seed wherever the engine permits.",
        "evidence": "Submit the master, dry and wet stems, score or formal map, automation data, preset state, exact renders at two or more block sizes, analysis JSON, and a labeled multivariable timeline. Include null or difference measurements and listening notes for any block-boundary discrepancy.",
        "questions": "Which spatial parameter acts most like orchestration, harmony, dynamics, or tempo? Can a listener hear the formal return when the notes do not recur? Where does smooth automation obscure a boundary that a discrete change would articulate more effectively?",
        "completion": "The composition must contain at least four clearly perceived spatial states and one convincing formal return created principally by automation. Repeated renders must satisfy the documented reproducibility tolerance or explain every divergence.",
        "extension": "Produce a second version in which performers, rather than a fixed timeline, trigger the same spatial form from musical cues.",
    },
    24: {
        "procedure": "Design a family of six related reverb presets organized around musical functions such as pulse, veil, chamber, bloom, fracture, and horizon. Compose a 90-second etude in six short sections using the same source ensemble throughout; each section must make one preset necessary to its articulation, harmony, or texture. Constrain shared parameters so the family sounds related, and vary only the dimensions needed for each role.",
        "evidence": "Submit the six presets with semantic descriptions, the score or session timeline, dry and wet stems, final master, a parameter-difference table, recall-test results, analysis JSON, and one labeled comparison figure. Include a blind naming trial showing whether listeners can connect sound and intended role.",
        "questions": "Which preset names predict musical behavior rather than generic mood? What invariant makes the collection sound like one family? Can a preset remain distinctive when source register, density, or tempo changes?",
        "completion": "All six presets must recall deterministically, remain distinguishable under loudness-matched listening, and perform a different compositional function in the etude. Names and descriptions must allow another musician to choose an appropriate preset without seeing its parameter values.",
        "extension": "Orchestrate the etude for a contrasting ensemble and revise no more than two parameters per preset to preserve the family identity.",
    },
    28: {
        "procedure": "Write a two- to four-minute three- or four-voice contrapuntal texture and encode it as a first-order Ambisonic scene. Give each voice a recurring spatial motif, then compose one imitative rotation, one exchange of positions, one stationary pedal against moving voices, and one global scene rotation. Decode to at least two playback formats and retain a binaural or stereo reference.",
        "evidence": "Submit the score, FOA master, decoded versions, isolated voices, trajectory diagram, channel-order and normalization settings, exact commands, analysis JSON, and labeled level or localization checks. Document every intended spatial invariant before comparing decoders.",
        "questions": "Which contrapuntal identities survive rotation because of timbre, rhythm, register, or location? When does motion clarify imitation, and when does it compete with voice-leading? What remains invariant across decoders, and what is reproduction-dependent?",
        "completion": "The spatial motifs and at least three contrapuntal events must remain intelligible in both decodes, channel conventions must be verified, and global rotation must preserve the documented level relationships within tolerance.",
        "extension": "Create a performance version in which one spatial voice follows a live conductor or performer while the remaining trajectories stay deterministic.",
    },
    33: {
        "procedure": "Compose and produce an original five- to eight-minute work for stereo, binaural, Ambisonic, or loudspeaker-array presentation. State a compositional thesis explaining how reverberation governs at least two domains among harmony, rhythm, gesture, orchestration, memory, and large-scale form. Develop the work through sketches, controlled listening comparisons, and a final production that integrates algorithmic or convolution reverb, automation, dynamics, and spatial routing without treating the room as a last-stage decoration.",
        "evidence": "Submit the final master and required playback derivatives, score or detailed formal map, dry and processed stems, presets, automation, impulse responses with provenance, exact commands, analysis JSON, plots, technical rider, and a reproducibility manifest with software revision and hashes. Include a five- to eight-minute presentation or written commentary that connects specific audible moments to the compositional thesis.",
        "questions": "What musical information exists only because of the designed acoustic? Where does the room act as instrument, ensemble, transition, or memory? Which technical constraint produced the strongest creative decision, and which apparently impressive effect was removed because it weakened the piece?",
        "completion": "The work must be compositionally complete, technically deliverable, and reproducible by another student. Independent listeners must be able to identify at least two intended spatial-form events, and the submitted evidence must demonstrate stable levels, correct routing, and deliberate control of accumulated energy.",
        "extension": "Prepare a concert or installation realization for a different reproduction system and write an adaptation note distinguishing musical invariants from venue-specific decisions.",
    },
    34: {
        "procedure": "Write a two- to three-minute progression of at least six sonorities. Render or automate each chord so its tail crosses into the next harmony, then decide whether the retained tones act as common tones, suspensions, dissonant pedals, or harmonic foreshadowing. Include one passage where the dry attack disappears and the listener hears only overlapping decays; control register and damping so the accumulated harmony remains intentional.",
        "evidence": "Submit the score or harmonic reduction, dry attacks, isolated tails, final master, exact settings, analysis JSON, and a labeled pitch-versus-time map showing the six transitions. Mark every retained pitch and explain whether it belongs to the departing chord, arriving chord, or neither.",
        "questions": "When does a decaying chord stop belonging to its source? Which crossings strengthen voice-leading, and which create unwanted harmonic mud? Can the room anticipate a harmony before any dry instrument plays it?",
        "completion": "All six transitions must have a stated harmonic function, and at least three must be distinguishable in blind listening. The tails must remain stable, pitched enough to support the analysis, and free of accidental low-mid accumulation.",
        "extension": "Reharmonize the piece while preserving every reverb print, treating the original tails as fixed contrapuntal material.",
    },
    35: {
        "procedure": "Choose a tempo and derive three decay targets from beat subdivisions or multiples, such as an eighth note, dotted quarter, and two measures. Compose a two-minute canon that moves through the three time scales, then introduce one metric modulation while preserving an audible decay ratio. Compare mathematically exact RT60 values with nearby values selected by ear and keep level constant.",
        "evidence": "Submit the score, tempo map, decay calculations, dry and processed stems, exact automation or commands, analysis JSON, and a labeled timeline connecting entries, beat values, and measured decay. Include one click-referenced excerpt for timing verification and one musical master without click.",
        "questions": "Do listeners hear tail duration as meter, articulation, or texture? Which ratios remain perceptible without a click? When does mathematically synchronized decay sound less musical than a deliberately offset value?",
        "completion": "The canon must present three clearly contrasting time scales and one convincing metric transition. Calculated and measured values must be documented, and the chosen musical deviations must be justified by listening rather than convenience.",
        "extension": "Repeat the canon at a new tempo while preserving proportional decay relationships, then compare whether the same ratios retain their character.",
    },
    36: {
        "procedure": "Compose a three-minute antiphonal work for two source groups. Give each group a contrasting virtual room, early-reflection pattern, and distance, then write six exchanges that evolve from literal call-and-response to overlap and interruption. Include one moment when the rooms exchange ensembles and one shared cadence in a third, unifying space.",
        "evidence": "Submit the score, isolated ensemble stems, isolated room returns, final master, room diagrams, presets or commands, analysis JSON, and a labeled formal map of the six exchanges. Document apparent direction, pre-delay, and early-to-late balance for every architectural state.",
        "questions": "Does architecture clarify ensemble identity as strongly as orchestration? What changes when a phrase moves into the other group's room? How does the shared final space alter the perceived social or dramatic relationship?",
        "completion": "Listeners must identify the two virtual ensembles and the room exchange without seeing the session. The shared cadence must sound like a meaningful architectural convergence, not a global wetness increase.",
        "extension": "Adapt the work for two live groups placed in a real venue and redesign the virtual rooms around the venue's measured response.",
    },
    37: {
        "procedure": "Compose for three registral layers using contrasting low-, mid-, and high-frequency decay. Begin with solo introductions that establish each layer, then combine them in a dense central section and redistribute the tails during a coda. Keep broadband RT60 or overall loudness sufficiently controlled that the form is created by spectral decay rather than simple level growth.",
        "evidence": "Submit a score or register chart, dry and band-isolated wet stems, final master, multiband settings, analysis JSON, and labeled decay curves for all three regions. Include a short orchestration note explaining which instruments leave spectral room for which tails.",
        "questions": "Which register reads as sustain, halo, or masking? Can a high-frequency tail articulate form without sounding merely bright? How does source orchestration need to change when the room occupies part of the spectrum?",
        "completion": "All three decay layers must perform distinct musical roles and remain audible in the combined section. Measured curves must support the intended hierarchy, and the final mix must avoid uncontrolled low-frequency carryover.",
        "extension": "Invert the decay orchestration so the shortest and longest bands exchange roles while the written notes remain unchanged.",
    },
    38: {
        "procedure": "Record or synthesize a dry kit of at least five transient sounds. Configure early reflections to create flams, offbeats, tuplets, or short polyrhythms, then compose a 32-bar study in which those reflections provide an audible secondary percussion part. Use four sections: establishment, displacement, densification, and release; reserve the late tail for color rather than rhythmic definition.",
        "evidence": "Submit the dry one-shots, notated attack-and-reflection grid, dry and early-field stems, final master, exact settings, analysis JSON, and a labeled waveform over two representative bars. Distinguish performed attacks from generated reflections in the notation.",
        "questions": "Which reflections fuse into timbre, and which separate into rhythm? How does source articulation change perceived reflection timing? At what density does the virtual percussion part lose groove and become diffuse ambience?",
        "completion": "The reflection part must create at least three reproducible rhythmic figures and remain intelligible without excessive late reverb. A listener must be able to tap or transcribe one generated pattern from the wet stem alone.",
        "extension": "Arrange the generated reflection part for a live percussionist and compare timing, groove, and timbre with the signal-processed version.",
    },
    39: {
        "procedure": "Write and record a three- to four-minute song, monologue, or dramatic scene with at least five vocal distance states. Map those states to a narrative arc that includes intimacy, withdrawal, confrontation, memory, and return. Coordinate pre-delay, early reflections, damping, width, and wet level with performance and arrangement rather than automating every parameter together.",
        "evidence": "Submit the text or score, dry vocal, wet return, final mix, five-state cue sheet, automation, presets, analysis JSON, and a labeled distance timeline. Include intelligibility notes from at least two listeners and identify every word intentionally obscured or exposed.",
        "questions": "Which acoustic cue most strongly changes the speaker's apparent psychological distance? When does a large room imply power rather than remoteness? Can a return to intimacy be heard without reducing overall loudness?",
        "completion": "All five narrative states must be perceptually distinct while essential text remains intelligible. Spatial changes must coincide with dramatic decisions, and the final return must recall the opening state without simply copying it.",
        "extension": "Create a binaural version in which apparent location and head-relative perspective reinforce the same dramatic arc.",
    },
    40: {
        "procedure": "Prepare four contrasting stems and perform a three- to five-minute live dub mix using reverb sends, mutes, feedback, damping, freeze, and return level as playable gestures. Define a cue structure but leave room for improvisation. Rehearse safe maximum feedback, create at least eight intentional throws, and include one recovery from a deliberately over-dense texture.",
        "evidence": "Submit the source stems, unedited performance, final edited version if used, control map, cue sheet, presets, analysis JSON, and a labeled gesture timeline. Document peak and spectral behavior around the densest throw and explain the recovery strategy.",
        "questions": "Which controls behave most like instrumental articulation? How does performance timing differ from drawing automation? What makes a throw feel responsive to the source rather than pasted onto it?",
        "completion": "The performance must contain eight identifiable send gestures, one controlled feedback build, and one musical recovery without clipping or panic muting. The cue structure must be repeatable while allowing meaningful variation between takes.",
        "extension": "Perform the piece with another musician controlling source arrangement while you control the spatial instrument; document the shared cue vocabulary.",
    },
    41: {
        "procedure": "Compose twelve short phrase endings for one ensemble. Vary harmonic closure, articulation, RT60, damping, and the notated silence that follows so the room completes, delays, questions, or erases each cadence. Sequence the endings into a two- to three-minute form in which silence has measured duration and at least one decay is interrupted before its natural end.",
        "evidence": "Submit the score with notated rests and tail cues, dry phrases, wet returns, final sequence, decay measurements, analysis JSON, and a labeled comparison of all twelve endings. Include listener judgments of closure strength without revealing the intended categories.",
        "questions": "Where does the cadence end: at the attack, release, threshold crossing, or next entrance? Can an unresolved harmony feel closed because its room disappears? How much silence is needed for a long decay to become structural rather than decorative?",
        "completion": "At least four distinct closure functions must be recognized by listeners, and every rest must be compositionally justified. The interrupted decay must sound intentional and alter the formal reading of its phrase.",
        "extension": "Perform the sequence in a reverberant real room and rewrite the notated silences to incorporate its natural decay.",
    },
    42: {
        "procedure": "Compose a concise theme and four variations for the same core ensemble. Assign each variation a room identity such as chamber, hall, plate-like studio, or impossible synthetic space. Preserve the theme's notes in the first pass, then reorchestrate articulation, register, density, and tempo only where the room demands a musical response.",
        "evidence": "Submit the theme and four scores or timelines, dry common-source comparison, final variations, presets or IRs, analysis JSON, and a parameter-and-orchestration matrix. Include a blind test asking listeners to match each room with its intended variation character.",
        "questions": "How much musical character comes from room identity before notes change? Which reorchestrations are genuinely required by decay or clarity? Does an impossible room suggest formal possibilities unavailable to an acoustic model?",
        "completion": "All four variations must remain recognizably related while presenting distinct acoustic characters. At least three room-to-character matches must exceed chance in the listening test, and every orchestration change must cite an acoustic reason.",
        "extension": "Create a fifth variation by morphing continuously through the four spaces while holding orchestration fixed.",
    },
    43: {
        "procedure": "Capture a room impulse response or select a thoroughly documented local IR, then analyze its resonances, clarity, noise floor, and decay. Compose a three- to five-minute portrait that features at least two measured traits: one may be embraced as material and one may be countered through orchestration or timing. Include an unprocessed field recording or contextual sound if appropriate and ethically permissible.",
        "evidence": "Submit the raw and prepared IR, provenance, photographs or diagram when permitted, hashes, analysis JSON, measured plots, score or timeline, dry and convolved stems, and final master. Explain every edit made to the IR and retain a reversible preparation chain.",
        "questions": "Which audible traits belong uniquely to this place? Does convolution preserve identity when the source is unrelated to the site? What ethical, practical, or historical context should accompany an acoustic portrait?",
        "completion": "The composition must make two measured site traits musically perceptible, and the IR provenance must permit another student to reproduce the result. Preparation may improve usability but must not erase the documented identity of the capture.",
        "extension": "Present the piece in the captured location or simulate that return, then analyze the interaction between real and convolved acoustics.",
    },
    44: {
        "procedure": "Choose two sustained and two transient sources, then compose a two-minute timbral etude with three modulation regimes: nearly static, gently animated, and deliberately unstable. Control modulation rate, depth, diffusion, and damping independently; introduce each regime in isolation before layering them. Preserve one pitched passage as a reference for detecting unwanted chorusing or intonation drift.",
        "evidence": "Submit source stems, wet-only stems, final master, modulation settings, analysis JSON, and labeled spectrogram and stereo-correlation plots. Include listening notes that distinguish motion, pitch change, density, and width rather than grouping them as a single impression.",
        "questions": "When does modulation read as room movement, ensemble size, chorus, or pitch instability? Which source envelopes reveal modulation most clearly? Can motion increase without widening the image?",
        "completion": "The three regimes must be perceptually distinct and technically stable, and the reference passage must retain acceptable pitch clarity. The composition must use modulation to articulate form rather than cycling continuously without purpose.",
        "extension": "Map modulation depth to a performer-controlled gesture and compare deterministic automation with embodied timing.",
    },
    45: {
        "procedure": "Write a three- or four-voice fugue or fugal exposition in which each subject entry begins in a recognizable room. As episodes develop, migrate voices between rooms, exchange apparent foreground and background, and include one stretto whose density is clarified by spatial separation. End with either convergence into one room or permanent architectural divergence.",
        "evidence": "Submit the score, isolated voices and room returns, final master, room-assignment map, routing or automation, analysis JSON, and a labeled trajectory diagram. Mark subject, answer, countersubject, episodes, migrations, and the spatial strategy used in the stretto.",
        "questions": "Can room identity function like instrumentation in a fugue? Which migrations preserve voice identity, and which deliberately destabilize it? Does spatial separation clarify counterpoint better than level or register changes?",
        "completion": "Listeners must follow at least three subject entries and identify the major room migration. The stretto must remain contrapuntally intelligible, and the ending's architectural decision must support the harmonic and formal outcome.",
        "extension": "Decode the work for a contrasting playback format and revise only spatial routing, not notes, to restore the intended contrapuntal clarity.",
    },
    46: {
        "procedure": "Design three interactive states for an imagined game, installation, or performance: exploration, tension, and transformation. Compose modular stems that can recombine vertically and horizontally, then create reverb states and transition rules that preserve tails across state changes without exposing clicks or implausible room resets. Demonstrate at least six non-linear transitions, including a rapid reversal.",
        "evidence": "Submit all stems, state diagram, transition rules, preset and automation data, six transition renders, final demonstration, analysis JSON, and a labeled latency and tail-continuity report. Identify which parameters interpolate, switch, freeze, or wait for a musical boundary.",
        "questions": "What acoustic information must persist for a world to feel continuous? When should a narrative state override physical plausibility? Which transition sounds convincing in isolation but fails after repeated interactive use?",
        "completion": "All six transitions must be click-free, musically coherent, and reproducible in arbitrary order. The rapid reversal must preserve intelligible state identity, and no transition may depend on a single fixed timeline.",
        "extension": "Build a simple live controller or scripted simulator that chooses states unpredictably and record a five-minute emergent performance.",
    },
    47: {
        "procedure": "Write and produce an original song with verse, chorus, bridge, and final return. Design a related spatial orchestration for each section using send hierarchy, pre-delay, decay, width, ducking, and selective effects rather than one global preset. Keep the lead intelligible, give at least two supporting instruments independent depth roles, and make the final return transform an earlier space.",
        "evidence": "Submit the song chart or score, dry multitrack or grouped stems, wet returns, final mix, section-by-section send map, presets and automation, analysis JSON, and labeled loudness, clarity, and correlation comparisons. Include a vocal-up and instrumental reference for translation checks.",
        "questions": "Which section becomes larger because of arrangement, and which because of room design? How does pre-delay preserve lyrical focus at the largest moment? What makes the final return feel remembered rather than repeated?",
        "completion": "Verse, chorus, bridge, and final return must have distinct but related spatial identities. The vocal must remain intelligible, mono compatibility must meet the stated target, and the spatial orchestration must survive at least two playback systems.",
        "extension": "Create an alternate mix that reverses the expected scale hierarchy, making the verse enormous and chorus intimate without weakening the song's form.",
    },
    48: {
        "procedure": "Create an eight- to twelve-minute cycle of at least three connected movements. Synthesize three or more earlier techniques, including one harmonic or rhythmic process, one spatial-form process, and one realtime or interactive process. Establish recurring musical material and a recurring acoustic identity, then transform both across the cycle. Workshop sketches, conduct blind listening checks, and prepare a final concert, installation, or release-ready master.",
        "evidence": "Submit the complete cycle, movement stems, score or detailed formal document, presets, automation, IR provenance, exact commands, analysis JSON, technical rider, playback derivatives, and a reproducibility manifest with hashes. Include a portfolio commentary connecting specific audible moments to earlier exercises and explaining revisions made after listening tests.",
        "questions": "What unifies the cycle when rooms, media, or techniques change? Which spatial process became genuinely compositional, and which remained production support? How does the final movement reinterpret acoustic material introduced earlier?",
        "completion": "The cycle must be compositionally complete, technically deliverable, and reproducible by another student. Independent listeners must recognize at least two recurring spatial ideas, all playback formats must pass routing and level checks, and the commentary must demonstrate substantive revision rather than feature accumulation.",
        "extension": "Adapt the cycle to a new venue or medium and document which musical-spatial relationships are invariant and which require recomposition.",
    },
}

COMPOSITION_PROJECT_FIELDS = {
    "procedure",
    "evidence",
    "questions",
    "completion",
    "extension",
}

MUSIC_EXPANSION = (
    ("Sacred resonance and ritual", "Guillaume de Machaut", "Messe de Nostre Dame", "c. 1365", "polyphonic mass", "Track how long stone-room sustain joins successive voices without erasing their independent entries."),
    ("Sacred resonance and ritual", "Gregorio Allegri", "Miserere mei, Deus", "c. 1638", "choral work", "Compare distant solo-group height with the grounded main choir and listen for architecture acting as an antiphonal mixer."),
    ("Sacred resonance and ritual", "J. S. Bach", "St Matthew Passion", "1727", "oratorio", "Use contrasting recordings to separate compositional double-choir space from venue and microphone perspective."),
    ("Sacred resonance and ritual", "Anton Bruckner", "Locus iste", "1869", "motet", "Notice how rests expose the room and how harmonic arrivals briefly convert decay into part of the chord."),
    ("Sacred resonance and ritual", "Arvo Pärt", "Passio", "1982", "ECM New Series recording", "Study how narrow dynamics, sustained tones, and silence make the recording acoustic an active formal layer."),
    ("Sacred resonance and ritual", "John Tavener", "The Protecting Veil", "1988", "cello and strings", "Listen for solo-cello presence floating ahead of a slowly changing orchestral halo."),
    ("Orchestral depth and concert-hall scale", "Gustav Mahler", "Symphony No. 2", "1894", "symphony", "Map offstage and distant events against close orchestral attacks, especially during the final movement's expanding scale."),
    ("Orchestral depth and concert-hall scale", "Claude Debussy", "La mer", "1905", "orchestral work", "Follow how orchestral color, hall bloom, and dynamic shaping create depth without fixed foreground/background roles."),
    ("Orchestral depth and concert-hall scale", "Gustav Holst", "Neptune, the Mystic", "1916", "The Planets", "The disappearing offstage chorus is a model for level, distance, filtering, and decay acting as one continuous gesture."),
    ("Orchestral depth and concert-hall scale", "Maurice Ravel", "Daphnis et Chloé, Suite No. 2", "1913", "ballet suite", "Compare transient woodwind detail with the broad chorus and string field at the sunrise climax."),
    ("Orchestral depth and concert-hall scale", "Benjamin Britten", "War Requiem", "1962", "requiem", "Separate chamber, orchestral, and distant ceremonial layers by apparent location as well as orchestration."),
    ("Orchestral depth and concert-hall scale", "György Ligeti", "Lontano", "1967", "orchestral work", "Dense micropolyphony provides a reference for diffusion build-up whose source identities remain barely perceptible."),
    ("Jazz rooms and engineered intimacy", "Miles Davis", "So What", "1959", "Kind of Blue", "Use the studio's restrained ambience to study depth that does not announce itself as an effect."),
    ("Jazz rooms and engineered intimacy", "Bill Evans Trio", "Waltz for Debby", "1961", "Waltz for Debby", "Audience, room, and trio occupy one acoustic scene; note how venue sound supplies scale without obscuring touch."),
    ("Jazz rooms and engineered intimacy", "John Coltrane", "A Love Supreme, Part I: Acknowledgement", "1965", "A Love Supreme", "Listen for cohesive room tone around forceful close detail and a rhythm section with stable depth."),
    ("Jazz rooms and engineered intimacy", "Miles Davis", "In a Silent Way / It's About That Time", "1969", "In a Silent Way", "Edits, electric timbres, and sustained studio space turn ambience into continuity across assembled performances."),
    ("Jazz rooms and engineered intimacy", "Alice Coltrane", "Journey in Satchidananda", "1971", "Journey in Satchidananda", "Harp resonance, drone, percussion, and studio ambience create layered sustain with distinct spectral zones."),
    ("Jazz rooms and engineered intimacy", "Jan Garbarek and The Hilliard Ensemble", "Parce mihi Domine", "1994", "Officium", "The saxophone and vocal ensemble demonstrate how one large church can preserve separate source distances."),
    ("Studio architecture and iconic production", "Roy Orbison", "In Dreams", "1963", "In Dreams", "Orchestral build and chamber-like return increase scale while Orbison's voice remains the fixed perceptual anchor."),
    ("Studio architecture and iconic production", "Patsy Cline", "Crazy", "1961", "Showcase", "A controlled plate-or-chamber halo supports vocal sustain without moving the singer behind the ensemble."),
    ("Studio architecture and iconic production", "Pink Floyd", "Echoes", "1971", "Meddle", "Treat the work as a catalog of synthetic distance, feedback, panning, and long-form transitions between acoustic worlds."),
    ("Studio architecture and iconic production", "David Bowie", "Heroes", "1977", "Heroes", "The vocal microphone-gate arrangement links performance intensity to apparent room size and should be studied dynamically."),
    ("Studio architecture and iconic production", "Prince and the Revolution", "Purple Rain", "1984", "Purple Rain", "Arena scale comes from performance, audience, delays, and long returns whose energy is carefully limited around the lead."),
    ("Studio architecture and iconic production", "U2", "A Sort of Homecoming", "1984", "The Unforgettable Fire", "Diffuse guitar and drum spaces show how modulation and ambience can bind a mix while preserving propulsion."),
    ("Ambient continuums and decaying form", "Tangerine Dream", "Phaedra", "1974", "Phaedra", "Sequenced pulses and evolving electronic clouds offer a test for separating repeated attacks from a continuously moving field."),
    ("Ambient continuums and decaying form", "Laraaji", "Meditation No. 1", "1980", "Ambient 3: Day of Radiance", "Bright struck transients accumulate into a harmonically dense tail that tests damping and feedback stability."),
    ("Ambient continuums and decaying form", "Biosphere", "Poa Alpina", "1997", "Substrata", "Environmental scale is implied through filtered layers and quiet returns rather than a single identifiable room."),
    ("Ambient continuums and decaying form", "Stars of the Lid", "Requiem for Dying Mothers, Part 2", "2001", "The Tired Sounds of Stars of the Lid", "Slow orchestral layers demonstrate how several long envelopes can overlap without uncontrolled low-mid buildup."),
    ("Ambient continuums and decaying form", "William Basinski", "d|p 1.1", "2002", "The Disintegration Loops", "Loop decay makes spectral loss and accumulating noise part of form, useful for comparing finite tail and degradation models."),
    ("Ambient continuums and decaying form", "Tim Hecker", "In the Fog II", "2011", "Ravedeath, 1972", "Distortion, organ-like sustain, and unstable depth show why reverb color cannot be separated from source processing."),
    ("Dub, trip-hop, and vocal depth", "Public Enemy", "Welcome to the Terrordome", "1990", "Fear of a Black Planet", "Dense sampling demonstrates selective short-space placement where a global long tail would destroy rhythmic hierarchy."),
    ("Dub, trip-hop, and vocal depth", "Massive Attack", "Teardrop", "1998", "Mezzanine", "The vocal occupies an intimate center while percussion and tonal beds imply a larger dark enclosure."),
    ("Dub, trip-hop, and vocal depth", "Portishead", "Roads", "1994", "Dummy", "Sparse arrangement makes the scale and automation of vocal and string ambience unusually exposed."),
    ("Dub, trip-hop, and vocal depth", "Radiohead", "Pyramid Song", "2001", "Amnesiac", "Piano resonance, voice, strings, and delayed percussion establish multiple depth planes that converge near climaxes."),
    ("Dub, trip-hop, and vocal depth", "James Blake", "Limit to Your Love", "2010", "James Blake", "Extreme contrast between dry intimacy, sub-bass energy, and expanding vocal space is a lesson in spectral headroom."),
    ("Dub, trip-hop, and vocal depth", "Frank Ocean", "Seigfried", "2016", "Blonde", "Shifting voice layers and quiet environmental tails create psychological distance with very low effect visibility."),
    ("Screen, installation, and game worlds", "Bernard Herrmann", "Scene d'Amour", "1958", "Vertigo", "Orchestral surges and scoring-stage depth link harmonic recurrence with a powerful sense of spatial return."),
    ("Screen, installation, and game worlds", "György Ligeti", "Lux Aeterna", "1966", "2001: A Space Odyssey soundtrack context", "The choral field demonstrates how dense voices and long room decay can become an apparently source-less texture."),
    ("Screen, installation, and game worlds", "Vangelis", "Blade Runner Blues", "1982", "Blade Runner", "Synth sustain, saxophone, city-like ambience, and delay construct a space larger than any visible room."),
    ("Screen, installation, and game worlds", "Angelo Badalamenti", "Laura Palmer's Theme", "1990", "Twin Peaks", "Foreground keyboard attacks and a dark orchestral halo show how reverberation can signal emotional scale changes."),
    ("Screen, installation, and game worlds", "John Williams", "Theme from Jurassic Park", "1993", "Jurassic Park", "The scoring-stage image balances monumental brass and strings with intelligible inner detail at very high density."),
    ("Screen, installation, and game worlds", "Martin O'Donnell and Michael Salvatori", "A Walk in the Woods", "2001", "Halo: Original Soundtrack", "Choral and synthetic layers demonstrate a game-space aesthetic that remains readable under repeated playback and combat effects."),
    ("Screen, installation, and game worlds", "Austin Wintory", "Nascence", "2012", "Journey", "The score's gradual spatial expansion offers a model for automating envelopment as a narrative parameter."),
    ("Contemporary immersive production", "SOPHIE", "Is It Cold in the Water?", "2018", "Oil of Every Pearl's Un-Insides", "Hyper-detailed dry events and enormous synthetic depth shifts reveal the expressive value of abrupt spatial discontinuity."),
    ("Contemporary immersive production", "Billie Eilish", "when the party's over", "2018", "WHEN WE ALL FALL ASLEEP, WHERE DO WE GO?", "Close vocal layers and restrained long returns show how arrangement can make subtle reverb feel immense."),
    ("Contemporary immersive production", "Jon Hopkins", "Luminous Beings", "2018", "Singularity", "Pulsing detail and expanding harmonic clouds invite analysis of modulation, stereo correlation, and long-tail density."),
    ("Contemporary immersive production", "Floating Points, Pharoah Sanders, and the London Symphony Orchestra", "Movement 6", "2021", "Promises", "A recurring electronic figure, saxophone, and orchestra share a slowly changing field without losing their independent depths."),
    ("Contemporary immersive production", "Caroline Polachek", "Billions", "2022", "Desire, I Want to Turn Into You", "Layered voice, percussion, and choral scale demonstrate precise contrast between intimate articulation and spectacular bloom."),
)


def _svg(title: str, nodes: tuple[str, ...]) -> str:
    width, height = 1200, 300
    box_w, box_h, gap = 188, 76, 42
    total = len(nodes) * box_w + (len(nodes) - 1) * gap
    x0 = (width - total) // 2
    y = 130
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#123431"/></marker></defs>',
        '<rect width="1200" height="300" rx="26" fill="#f6f0e3"/>',
        f'<text x="60" y="55" font-family="Georgia,serif" font-size="30" font-weight="700" fill="#123431">{escape(title)}</text>',
        '<path d="M60 76 H1140" stroke="#d59a22" stroke-width="5"/>',
    ]
    colors = ("#267f9a", "#d59a22", "#c75b3c", "#29a391", "#7151a3")
    for i, node in enumerate(nodes):
        x = x0 + i * (box_w + gap)
        parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="14" fill="#fff" stroke="{colors[i % len(colors)]}" stroke-width="4"/>')
        words = node.split()
        split = max(1, len(words) // 2)
        lines = (" ".join(words[:split]), " ".join(words[split:])) if len(words) > 2 else (node,)
        start_y = y + 34 - (len(lines) - 1) * 10
        for j, line in enumerate(lines):
            parts.append(f'<text x="{x + box_w/2}" y="{start_y + j*25}" text-anchor="middle" font-family="Georgia,serif" font-size="20" fill="#123431">{escape(line)}</text>')
        if i < len(nodes) - 1:
            x1 = x + box_w + 7
            x2 = x + box_w + gap - 7
            parts.append(f'<path d="M{x1} {y + box_h/2} H{x2}" stroke="#123431" stroke-width="3" marker-end="url(#arrow)"/>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Georgia Bold.ttf" if bold else "Georgia.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def _png(path: Path, title: str, nodes: tuple[str, ...]) -> None:
    scale = 2
    width, height = 1200 * scale, 300 * scale
    image = Image.new("RGB", (width, height), "#f6f0e3")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((2, 2, width - 3, height - 3), radius=26 * scale, outline="#d8cbb5", width=2 * scale)
    draw.text((60 * scale, 30 * scale), title, font=_font(30 * scale, True), fill="#123431")
    draw.line((60 * scale, 76 * scale, 1140 * scale, 76 * scale), fill="#d59a22", width=5 * scale)
    box_w, box_h, gap = 188 * scale, 76 * scale, 42 * scale
    total = len(nodes) * box_w + (len(nodes) - 1) * gap
    x0 = (width - total) // 2
    y = 130 * scale
    colors = ("#267f9a", "#d59a22", "#c75b3c", "#29a391", "#7151a3")
    for index, node in enumerate(nodes):
        x = x0 + index * (box_w + gap)
        draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=14 * scale, fill="white", outline=colors[index % len(colors)], width=4 * scale)
        words = node.split()
        split = max(1, len(words) // 2)
        lines = (" ".join(words[:split]), " ".join(words[split:])) if len(words) > 2 else (node,)
        font = _font(18 * scale)
        line_height = 23 * scale
        top = y + (box_h - len(lines) * line_height) // 2
        for line_index, line in enumerate(lines):
            bounds = draw.textbbox((0, 0), line, font=font)
            draw.text((x + (box_w - (bounds[2] - bounds[0])) // 2, top + line_index * line_height), line, font=font, fill="#123431")
        if index < len(nodes) - 1:
            x1, x2, mid = x + box_w + 6 * scale, x + box_w + gap - 8 * scale, y + box_h // 2
            draw.line((x1, mid, x2, mid), fill="#123431", width=3 * scale)
            draw.polygon(((x2, mid), (x2 - 10 * scale, mid - 6 * scale), (x2 - 10 * scale, mid + 6 * scale)), fill="#123431")
    image.save(path, optimize=True)


def generate_diagrams() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# System Orientation Through Block Diagrams",
        "",
        "This chapter is a visual map of the signal paths, state boundaries, and evidence loops used throughout verbx. Read each diagram from left to right, then use the paragraph beneath it to identify what may happen in realtime, what must be prepared off-thread, and what should be preserved in a report.",
        "",
    ]
    for number, (title, nodes, description) in enumerate(DIAGRAMS, 1):
        filename = f"{number:02d}_{title.lower().replace(' ', '_').replace('-', '_')}.svg"
        (ASSET_DIR / filename).write_text(_svg(title, nodes), encoding="utf-8")
        png_filename = filename.removesuffix(".svg") + ".png"
        _png(ASSET_DIR / png_filename, title, nodes)
        lines.extend((
            f"## {title}", "",
            f"The figure below introduces **{title}**. {description} Read it from left to right before using the detailed boundary notes beneath the visual.", "",
            f"![{title}.](assets/intro_block_diagrams/{png_filename})", "",
            f"**Figure: {title}.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.", "",
        ))
    (ROOT / "docs" / "INTRODUCTORY_BLOCK_DIAGRAMS.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def generate_projects() -> None:
    if len(PROJECTS) != 48:
        raise ValueError(f"Expected 48 educational projects, found {len(PROJECTS)}")
    invalid_numbers = sorted(set(COMPOSITION_PROJECTS) - set(range(1, len(PROJECTS) + 1)))
    if invalid_numbers:
        raise ValueError(f"Composition project numbers are out of range: {invalid_numbers}")
    for number, details in COMPOSITION_PROJECTS.items():
        missing = sorted(COMPOSITION_PROJECT_FIELDS - set(details))
        if missing:
            raise ValueError(f"Composition project {number} is missing fields: {missing}")

    lines = [
        "# Educational Exercises and Project Assignments",
        "",
        "These projects form a progressive laboratory course in reverberation, spatial audio, realtime systems, critical listening, musical composition, and reproducible audio engineering. Twenty-four projects are explicitly composition-centered, and every technical exercise may be adapted to original musical material. Each assignment is scoped for one book page; instructors may treat it as a weekly laboratory, combine adjacent projects, or select a focused sequence.",
        "",
        "## Assessment Framework",
        "",
        "Evaluate every submission on prediction, method, evidence, listening judgment, reproducibility, and clarity. Unless a project states otherwise, students should retain source audio, exact commands, parameter or preset state, analysis JSON, plots, and a short reflection distinguishing observation from interpretation.",
        "",
    ]
    for number, (title, concepts, brief) in enumerate(PROJECTS, 1):
        composition = COMPOSITION_PROJECTS.get(number)
        design_method = (
            "State the intended musical function before processing, make an initial sketch, and preserve an unprocessed version so every spatial decision can be evaluated as part of the composition."
            if composition
            else "Begin with a written prediction that names the expected audible and measurable changes. Use controlled source material and alter one principal variable at a time before combining controls."
        )
        procedure = (
            composition["procedure"]
            if composition
            else "Establish a dry baseline and one conservative reference render. Create at least five documented variants spanning the useful range, including one deliberately poor or unstable case when safe. Loudness-match comparisons, preserve deterministic seeds where applicable, and record effective settings rather than relying on command history alone."
        )
        evidence = (
            composition["evidence"]
            if composition
            else "Submit the source and rendered excerpts, exact runnable commands, preset or automation files, analysis JSON, a compact comparison table, and one labeled figure. Include listening conditions, sample rate, channel layout, block or partition size, software revision, and any warnings produced by verbx."
        )
        questions = (
            composition["questions"]
            if composition
            else "Which prediction was confirmed? Which result contradicted the model? What changed perceptually before a standard metric changed? Where does the preferred setting sit relative to a technical failure boundary? Name one confound and design a follow-up that isolates it."
        )
        completion = (
            composition["completion"]
            if composition
            else "Another student must be able to reproduce the central result from the submitted materials. Conclusions must cite both measured evidence and level-matched critical listening; screenshots alone are not evidence."
        )
        extension = (
            composition["extension"]
            if composition
            else "Repeat the decisive comparison with a contrasting source, room, sample rate, or reproduction layout and explain which conclusions generalize."
        )
        lines.extend((
            "\\newpage", "", f"## Project {number}: {title}", "",
            *(["**Project mode:** Musical composition and production.", ""] if composition else []),
            f"**Central concepts:** {concepts}.", "",
            f"**Design brief.** {brief} {design_method}", "",
            f"**Procedure.** {procedure}", "",
            f"**Evidence package.** {evidence}", "",
            f"**Questions for the report.** {questions}", "",
            f"**Completion standard.** {completion}", "",
            f"**Extension.** {extension}", "",
            "```{=latex}",
            "\\vfill",
            f"\\verbxFigureLead{{Project {number} laboratory cycle: {title}}}",
            f"\\verbxAssignmentPlate{{{number}}}{{{title}}}",
            f"\\verbxFigureCaption{{Project {number} laboratory cycle: {title}}}",
            "```", "",
        ))
    (ROOT / "docs" / "HOMEWORK_ASSIGNMENTS.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def generate_music_expansion() -> None:
    lines = [
        "## Expanded Listening Canon",
        "",
        "The following forty-eight additions broaden the appendix across sacred, orchestral, jazz, popular, ambient, screen, game, and contemporary immersive practices. Piece and album titles are italicized throughout. Links open a stable YouTube catalog search so readers can select an authorized or territorially available recording; recording-specific citations in the preceding section remain the preferred references where supplied.",
        "",
    ]
    active_category = None
    for category, creator, title, year, album, note in MUSIC_EXPANSION:
        if category != active_category:
            lines.extend((f"### {category.title()}", ""))
            active_category = category
        query = quote_plus(f"{creator} {title} official")
        album_text = f" from *{album}*" if album.lower() != title.lower() else f" on *{album}*"
        lines.extend((
            f"**{creator}, *{title}* ({year}).**{album_text}. {note} [YouTube catalog search](https://www.youtube.com/results?search_query={query}).",
            "",
        ))
    (ROOT / "docs" / "MUSICAL_PIECES_EXPANSION.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def normalize_music_typography() -> None:
    """Keep legacy work titles italicized in both Markdown and the PDF."""

    path = ROOT / "docs" / "MUSICAL_PIECES_APPENDIX.md"
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(?m)^\*\*(?P<creator>[^,\n]+),\s+(?P<title>(?!\*)[^\n]+?)\s+"
        r"\((?P<date>[^)]+)\)\.\*\*"
    )
    text = pattern.sub(
        lambda match: (
            f"**{match.group('creator')}, *{match.group('title')}* "
            f"({match.group('date')}).**"
        ),
        text,
    )
    text = text.replace(
        "- Aphex Twin. “#3” (“Rhubarb”), from *Selected Ambient Works Volume II*.",
        "- Aphex Twin. *#3 (Rhubarb)*, from *Selected Ambient Works Volume II*.",
    )
    replacements = {
        "**Brian Eno, *1/1 from Ambient 1: Music for Airports* (1978).**":
            "**Brian Eno, *1/1*, from *Ambient 1: Music for Airports* (1978).**",
        "**Aphex Twin, *Rhubarb* (1994).**":
            "**Aphex Twin, *#3 (Rhubarb)* (1994).**",
        "**Jonathan Harvey, *Mortuos Plango, *Vivos Voco** (1980).**":
            "**Jonathan Harvey, *Mortuos Plango, Vivos Voco* (1980).**",
        "**Luigi Nono, *Prometeo* (1981–1984).**":
            "**Luigi Nono, *Prometeo: Tragedia dell'ascolto* (1981–1984).**",
        "[Lovely Music notes for Lucier's I am sitting in a room]":
            "[Lovely Music notes for Lucier's *I am sitting in a room*]",
        "[Prometeo record]":
            "[*Prometeo: Tragedia dell'ascolto* record]",
        "[Le Encantadas]":
            "[*Le Encantadas o le avventure nel mare delle meraviglie*]",
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    path.write_text(text, encoding="utf-8")


def validate_music_typography() -> None:
    """Reject music entries whose work or album title escaped italics."""

    appendix = (ROOT / "docs" / "MUSICAL_PIECES_APPENDIX.md").read_text(encoding="utf-8")
    expansion = (ROOT / "docs" / "MUSICAL_PIECES_EXPANSION.md").read_text(encoding="utf-8")
    malformed: list[str] = []
    for path, text in (("appendix", appendix), ("expansion", expansion)):
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.startswith("**") and not re.match(
                r"^\*\*.+?, \*[^*]+\*(?:, from \*[^*]+\*)? \([^)]+\)\.\*\*",
                line,
            ):
                malformed.append(f"{path}:{line_number}: {line}")
            if line.startswith("- ") and "YouTube" in line and not re.search(r"\. \*[^*]+\*", line):
                malformed.append(f"{path}:{line_number}: {line}")
    if malformed:
        details = "\n".join(malformed)
        raise ValueError(f"Music titles must be italicized:\n{details}")


def main() -> None:
    normalize_music_typography()
    generate_diagrams()
    generate_projects()
    generate_music_expansion()
    validate_music_typography()
    print("Wrote introductory diagrams, 48 listening entries, and 48 educational projects")


if __name__ == "__main__":
    main()
