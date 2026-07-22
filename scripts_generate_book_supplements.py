#!/usr/bin/env python3
"""Generate the introductory diagram chapter and educational appendix."""

from __future__ import annotations

import importlib.util
import re
from html import escape
from pathlib import Path
from urllib.parse import quote_plus

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "docs" / "assets" / "intro_block_diagrams"
COMPOSITION_YEAR_SPEC = importlib.util.spec_from_file_location(
    "scripts_normalize_composition_years_for_supplements",
    ROOT / "scripts_normalize_composition_years.py",
)
assert (
    COMPOSITION_YEAR_SPEC is not None
    and COMPOSITION_YEAR_SPEC.loader is not None
)
COMPOSITION_YEARS = importlib.util.module_from_spec(COMPOSITION_YEAR_SPEC)
COMPOSITION_YEAR_SPEC.loader.exec_module(COMPOSITION_YEARS)

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

MUSIC_EXPANSION_CORE = (
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
    ("Early music, organ, and monumental acoustics", "Hildegard von Bingen", "O vis aeternitatis", "c. 1151", "A Feather on the Breath of God", "Follow the monophonic line into the room and distinguish melodic continuation from architectural sustain."),
    ("Early music, organ, and monumental acoustics", "Pérotin", "Viderunt omnes", "c. 1198", "Perotin", "Hear how sustained lower voices and measured upper parts use cathedral decay as a harmonic joining mechanism."),
    ("Early music, organ, and monumental acoustics", "Heinrich Schütz", "Saul, Saul, was verfolgst du mich?", "1650", "Symphoniae sacrae III", "Antiphonal calls, rests, and sharply profiled consonants reveal how a large room can become part of rhetorical timing."),
    ("Early music, organ, and monumental acoustics", "J. S. Bach", "Passacaglia and Fugue in C minor, BWV 582", "c. 1710", "", "Track the repeated bass through registration changes and ask when the building reinforces structure rather than merely adding grandeur."),
    ("Early music, organ, and monumental acoustics", "Olivier Messiaen", "Apparition de l'église éternelle", "1932", "", "The slow crescendo and decrescendo turn organ, low-frequency room modes, and silence into one monumental envelope."),
    ("Early music, organ, and monumental acoustics", "Charlemagne Palestine", "Schlingen-Blängen", "1974", "Schlingen-Blängen", "Sustained organ overtones expose beating, modal reinforcement, and the point at which the room seems to continue the instrument."),
    ("Opera, theater, and staged distance", "Claudio Monteverdi", "L'Orfeo: Toccata and Prologue", "1607", "L'Orfeo", "Compare ceremonial brass projection with the prologue's vocal presence and note how contrasting spaces establish the drama before the plot advances."),
    ("Opera, theater, and staged distance", "Wolfgang Amadeus Mozart", "Don Giovanni: Commendatore Scene", "1787", "Don Giovanni", "The confrontation uses register, orchestration, stage position, and hall scale to turn apparent distance into supernatural authority."),
    ("Opera, theater, and staged distance", "Richard Wagner", "Parsifal: Transformation Music", "1882", "Parsifal", "Listen for orchestral continuity making the imagined room appear to change size while the musical pulse remains suspended."),
    ("Opera, theater, and staged distance", "Luciano Berio", "Coro", "1975–1976", "Coro", "Singers paired with instruments create a distributed ensemble in which local timbres and collective resonance continually exchange roles."),
    ("Opera, theater, and staged distance", "Unsuk Chin", "Alice in Wonderland", "2007", "Alice in Wonderland", "Rapid changes of scale and orchestral perspective demonstrate how abrupt spatial contrast can support theatrical surrealism."),
    ("Opera, theater, and staged distance", "Michel van der Aa", "Sunken Garden", "2013", "Sunken Garden", "Live voices, orchestra, electronics, and film-derived space invite a comparison between visible location and heard location."),
    ("Resonant instruments and chamber scale", "Claude Debussy", "La cathédrale engloutie", "1910", "Préludes, Book I", "Pedal, register, and dynamics make a piano imply architecture; identify which depth cues exist before any recorded-room contribution."),
    ("Resonant instruments and chamber scale", "John Cage", "In a Landscape", "1948", "In a Landscape", "A limited pitch field and gentle attack make pedal resonance and room decay audible as part of the piece's continuity."),
    ("Resonant instruments and chamber scale", "Morton Feldman", "Rothko Chapel", "1971", "Rothko Chapel", "Sparse attacks, voices, viola, and percussion leave the listener enough time to hear each resonance change the color of the next event."),
    ("Resonant instruments and chamber scale", "George Crumb", "The Phantom Gondolier", "1972", "Makrokosmos, Volume I", "Extended piano techniques and vocalized sound create several apparent distances inside one instrument."),
    ("Resonant instruments and chamber scale", "Ryuichi Sakamoto", "andata", "2017", "async", "Piano, mechanical detail, and electronic atmosphere balance tactile proximity against a broad, unstable background field."),
    ("Resonant instruments and chamber scale", "Kali Malone", "Spectacle of Ritual", "2019", "The Sacrificial Code", "Organ tuning and patient voice leading expose slow interference patterns that can be mistaken for modulation or moving reflections."),
    ("Experimental voice, installation, and feedback", "Pauline Oliveros, Stuart Dempster, and Panaiotis", "Lear", "1989", "Deep Listening", "The Fort Worden cistern's extraordinary decay turns each performed sound into material for the next gesture."),
    ("Experimental voice, installation, and feedback", "Meredith Monk", "Gotham Lullaby", "1981", "Dolmen Music", "Layered vocal resonance demonstrates how timbre, vowel, and recording space can create depth without conventional orchestration."),
    ("Experimental voice, installation, and feedback", "Janet Cardiff", "The Forty Part Motet", "2001", "sound installation", "Forty individually recorded voices on forty loudspeakers let the listener move physically through Tallis's counterpoint."),
    ("Experimental voice, installation, and feedback", "Maryanne Amacher", "Head Rhythm 1 and Plaything 2", "1999", "Sound Characters (Making the Third Ear)", "High-level spectral interactions challenge the boundary between reproduced sound, room response, and listener-generated auditory phenomena."),
    ("Experimental voice, installation, and feedback", "Ellen Fullman and Okkyung Lee", "The Air Around Her, Part I", "2018", "The Air Around Her", "Long strings and cello activate architectural scale through sustained partials, beating, and slowly changing excitation."),
    ("Experimental voice, installation, and feedback", "Éliane Radigue", "Jetsun Mila", "1986", "Jetsun Mila", "Nearly static electronic layers reward close listening to minute spectral drift and the playback room's own contribution."),
    ("Soul, gospel, country, and vocal presence", "Sam Cooke", "A Change Is Gonna Come", "1964", "Ain't That Good News", "The lead remains intimate while strings and chamber-like depth expand the emotional horizon around it."),
    ("Soul, gospel, country, and vocal presence", "Aretha Franklin", "Amazing Grace", "1972", "Amazing Grace", "The live church recording joins solo voice, congregation, choir, and room response into one continuously changing performance system."),
    ("Soul, gospel, country, and vocal presence", "Daniel Lanois", "The Maker", "1989", "Acadie", "Percussion, voice, guitar, and dark ambience occupy distinct layers that remain legible despite a strongly atmospheric mix."),
    ("Soul, gospel, country, and vocal presence", "Chris Isaak", "Wicked Game", "1989", "Heart Shaped World", "A close vocal and highly spacious guitar show how two sources can imply different rooms without breaking the song's unity."),
    ("Soul, gospel, country, and vocal presence", "Emmylou Harris", "Where Will I Be", "1995", "Wrecking Ball", "Diffuse production enlarges the voice and instruments while preserving consonants, pulse, and a stable emotional foreground."),
    ("Soul, gospel, country, and vocal presence", "Weyes Blood", "Movies", "2019", "Titanic Rising", "The arrangement moves from intimate electronic detail toward orchestral bloom, making depth growth part of the song's form."),
    ("Heavy music, post-rock, and overwhelming scale", "Swans", "The Sound", "1996", "Soundtracks for the Blind", "Long crescendos and saturated layers test whether a large field can grow without losing its central pulse and harmonic direction."),
    ("Heavy music, post-rock, and overwhelming scale", "Sigur Rós", "Svefn-g-englar", "1999", "Ágætis byrjun", "Bowed guitar, voice, and spacious drums demonstrate how slow attacks and long tails can remain rhythmically articulate."),
    ("Heavy music, post-rock, and overwhelming scale", "Godspeed You! Black Emperor", "Storm", "2000", "Lift Your Skinny Fists Like Antennas to Heaven", "Ensemble accumulation and changing recording perspective make apparent room size rise with musical intensity."),
    ("Heavy music, post-rock, and overwhelming scale", "Boris", "Flood III", "2000", "Flood", "Distortion and extended decay fuse into a dense continuum whose clarity depends on spectral allocation rather than dryness."),
    ("Heavy music, post-rock, and overwhelming scale", "Sunn O)))", "It Took the Night to Believe", "2005", "Black One", "Low-frequency sustain and enormous apparent volume reveal how room modes, distortion, and feedback can become orchestration."),
    ("Heavy music, post-rock, and overwhelming scale", "Deafheaven", "Dream House", "2013", "Sunbather", "Fast drums and wide guitar fields provide a difficult test of transient preservation inside a bright, high-density wash."),
    ("Dub techno and electronic depth", "Basic Channel", "Quadrant Dub I", "1994", "Quadrant Dub", "Filtered chords and delay create a room whose boundaries are defined by repetition, noise, and spectral erosion rather than early reflections."),
    ("Dub techno and electronic depth", "Rhythm & Sound with Cornel Campbell", "King in My Empire", "2001", "King in My Empire", "Voice and rhythm remain centered while echoes recede into a dark, slowly modulated field."),
    ("Dub techno and electronic depth", "Deepchord Presents Echospace", "First Point of Aries", "2007", "The Coldest Season", "Layered delays, hiss, and diffuse chords make depth emerge from many low-contrast events rather than one obvious tail."),
    ("Dub techno and electronic depth", "Actress", "Hubble", "2010", "Splazsh", "Unstable transients and murky perspective show how partial obscurity can function as groove and timbral identity."),
    ("Dub techno and electronic depth", "Andy Stott", "Numb", "2012", "Luxury Problems", "A distant voice and heavy low-frequency structure demonstrate how spectral separation can support radically different apparent distances."),
    ("Dub techno and electronic depth", "Kelly Lee Owens", "Arthur", "2017", "Kelly Lee Owens", "Pulses, voice fragments, and expanding returns create movement by changing envelopment while the center remains restrained."),
    ("Soundscape composition and environmental scale", "Luc Ferrari", "Presque rien No. 1: Le Lever du jour au bord de la mer", "1970", "Presque Rien", "A recorded place unfolds as composition, making source distance, perspective, and time of day primary musical parameters."),
    ("Soundscape composition and environmental scale", "R. Murray Schafer", "Music for Wilderness Lake", "1979", "environmental work", "Trombonists positioned around a lake use propagation delay and landscape reflection as literal parts of ensemble timing."),
    ("Soundscape composition and environmental scale", "Annea Lockwood", "A Sound Map of the Hudson River", "1982", "A Sound Map of the Hudson River", "Changing water textures and recording perspectives construct a long-form portrait whose spaces are real yet editorially composed."),
    ("Soundscape composition and environmental scale", "Hildegard Westerkamp", "Kits Beach Soundwalk", "1989", "Transformations", "Voice, city, water, and microscopic shoreline sounds shift scale through recording perspective and studio transformation."),
    ("Soundscape composition and environmental scale", "Chris Watson", "Vatnajökull", "2003", "Weather Report", "Time-compressed glacier recordings turn environmental change into an eighteen-minute spectral and spatial trajectory."),
    ("Soundscape composition and environmental scale", "Jana Winderen", "Aquaculture", "2010", "Energy Field", "Hydrophones reveal underwater activity whose unfamiliar source cues require the listener to infer scale from spectrum and motion."),
)

MUSIC_EXPANSION_ADDITIONS = (
    ("Sacred resonance and ritual", "Giovanni Gabrieli", "In ecclesiis", "c. 1615", "Sacrae symphoniae II", "Separated vocal and instrumental choirs turn architectural distance into contrapuntal punctuation, with rests leaving the basilica-sized response fully exposed."),
    ("Sacred resonance and ritual", "Wolfgang Amadeus Mozart", "Ave verum corpus, K. 618", "1791", "motet", "Compact phrases and soft orchestration reveal how a modest sacred work can acquire scale from blended choral onset and restrained decay."),
    ("Sacred resonance and ritual", "Hector Berlioz", "Grande Messe des morts", "1837", "Requiem", "Widely separated brass groups and massed forces make direction, propagation, and long decay part of the written ceremonial drama."),
    ("Sacred resonance and ritual", "Sergei Rachmaninoff", "All-Night Vigil", "1915", "choral cycle", "Low bass fundamentals, close semitone motion, and sustained vowels test whether a room supports harmonic fusion without masking inner voices."),
    ("Sacred resonance and ritual", "Igor Stravinsky", "Symphony of Psalms", "1930", "choral symphony", "Dry rhythmic blocks alternate with resonant choral planes, making articulation and collective bloom equally important to the work's ritual character."),
    ("Sacred resonance and ritual", "Olivier Messiaen", "Et exspecto resurrectionem mortuorum", "1964", "orchestral work", "Gongs, winds, and silence exploit long architectural decay as measured continuation, especially when attacks are allowed to vanish completely before the next event."),
    ("Orchestral depth and concert-hall scale", "Hector Berlioz", "Symphonie fantastique", "1830", "symphony", "Extreme orchestral contrasts expose how hall return can enlarge distant bells and brass while preserving the nervous detail of strings and percussion."),
    ("Orchestral depth and concert-hall scale", "Charles Ives", "The Unanswered Question", "1906; revised later", "orchestral work", "Spatially separated strings, trumpet, and winds demonstrate that distance can distinguish simultaneous musical roles more clearly than timbre alone."),
    ("Orchestral depth and concert-hall scale", "Ralph Vaughan Williams", "Fantasia on a Theme by Thomas Tallis", "1910", "string orchestra", "Divided string groups and antiphonal placement create nested depth planes whose apparent boundaries expand and contract with register and dynamics."),
    ("Orchestral depth and concert-hall scale", "Ottorino Respighi", "Pines of Rome", "1924", "symphonic poem", "Offstage instruments, organ pedal, recorded birds, and monumental brass make the final spatial expansion a model of cumulative depth orchestration."),
    ("Orchestral depth and concert-hall scale", "Witold Lutosławski", "Livre pour orchestre", "1968", "orchestral work", "Controlled aleatory distributes transient detail across the ensemble, offering a demanding test for clarity during rapid changes in density and apparent width."),
    ("Orchestral depth and concert-hall scale", "Kaija Saariaho", "Orion", "2002", "orchestral work", "Spectral orchestration and sustained resonance make instrumental color appear to move through foreground, haze, and distant radiance."),
    ("Jazz rooms and engineered intimacy", "Duke Ellington", "Diminuendo and Crescendo in Blue", "1937", "orchestral jazz work", "Compare studio and live versions to hear how audience, stage leakage, and room excitation transform the same long-form crescendo."),
    ("Jazz rooms and engineered intimacy", "Thelonious Monk", "'Round Midnight", "1944", "jazz standard", "Sparse attacks and deliberate rests reveal the recording room around piano resonance while leaving harmonic ambiguity intact."),
    ("Jazz rooms and engineered intimacy", "Charles Mingus", "Goodbye Pork Pie Hat", "1959", "Mingus Ah Um", "Closely voiced horns, bass, and drums share a warm studio field whose cohesion depends on decay color more than conspicuous wetness."),
    ("Jazz rooms and engineered intimacy", "Keith Jarrett", "The Köln Concert, Part I", "1975", "The Köln Concert", "Piano, performer sounds, audience, and hall response form one continuous document in which resonant sustain guides improvisational pacing."),
    ("Jazz rooms and engineered intimacy", "Pat Metheny Group", "Are You Going with Me?", "1982", "Offramp", "Sustained guitar, synth, percussion, and long returns demonstrate how electronic jazz can maintain a stable center inside a panoramic field."),
    ("Jazz rooms and engineered intimacy", "Nils Petter Molvær", "Khmer", "1997", "Khmer", "Muted trumpet remains physically present against electronically extended drums and atmosphere, creating deliberate tension between intimacy and landscape."),
    ("Studio architecture and iconic production", "The Ronettes", "Be My Baby", "1963", "Presenting the Fabulous Ronettes Featuring Veronica", "Chamber reverberation, layered percussion, and dense orchestration show how a mono production can imply enormous depth through spectral and dynamic hierarchy."),
    ("Studio architecture and iconic production", "The Beatles", "A Day in the Life", "1967", "Sgt. Pepper's Lonely Hearts Club Band", "Contrasting vocal spaces and orchestral transitions make studio acoustics an explicit formal boundary between otherwise discontinuous sections."),
    ("Studio architecture and iconic production", "Led Zeppelin", "When the Levee Breaks", "1971", "Led Zeppelin IV", "The stairwell drum sound is a canonical lesson in distant microphones, compression, pre-delay, and a room response that becomes the groove's principal instrument."),
    ("Studio architecture and iconic production", "Kate Bush", "Running Up That Hill (A Deal with God)", "1985", "Hounds of Love", "Gated and layered drum ambience supports a driving pulse while voice and synthesizers occupy independently controlled depth planes."),
    ("Studio architecture and iconic production", "My Bloody Valentine", "To Here Knows When", "1991", "Loveless", "Reverse envelopes, diffuse guitars, and obscured attacks replace ordinary room cues with a continuously suspended synthetic space."),
    ("Studio architecture and iconic production", "Björk", "Jóga", "1997", "Homogenic", "Close voice, electronic percussion, strings, and volcanic-scale ambience demonstrate how abrupt depth contrast can intensify a song without weakening diction."),
    ("Ambient continuums and decaying form", "Brian Eno", "Discreet Music", "1975", "Discreet Music", "Slow tape-system evolution makes feedback, delay, and gradual spectral change audible as compositional process rather than decorative ambience."),
    ("Ambient continuums and decaying form", "Harold Budd and Brian Eno", "First Light", "1980", "The Plateaux of Mirror", "Soft piano attacks enter a diffuse electronic field whose decay blurs clock time while preserving each note's harmonic consequence."),
    ("Ambient continuums and decaying form", "Aphex Twin", "#3 (Rhubarb)", "1994", "Selected Ambient Works Volume II", "A nearly static harmonic loop reveals minute changes in tail color, noise, and beating that become perceptually large over repeated listening."),
    ("Ambient continuums and decaying form", "Gas", "Königsforst 1", "1998", "Königsforst", "Buried orchestral loops and a distant pulse make the virtual environment feel deep even when no discrete source has a stable location."),
    ("Ambient continuums and decaying form", "Loscil", "Endless Falls", "2010", "Endless Falls", "Rain, low-frequency beds, and restrained tonal events offer a study in separating environmental texture from synthesized late-field energy."),
    ("Ambient continuums and decaying form", "Sarah Davachi", "For Voice", "2020", "Cantus, Descant", "Organ-like sustain and narrow-band beating expose how playback-room modes interact with slowly changing electronic spectra."),
    ("Dub, trip-hop, and vocal depth", "Lee “Scratch” Perry and the Upsetters", "Dub Revolution", "1973", "Blackboard Jungle Dub", "Mutes, spring-like splashes, filtering, and feedback demonstrate that the wet return can function as an improvising answer rather than a fixed background."),
    ("Dub, trip-hop, and vocal depth", "Augustus Pablo", "King Tubby Meets Rockers Uptown", "1974", "King Tubbys Meets Rockers Uptown", "Melodica phrases and rhythm fragments trigger sharply timed echoes whose decay becomes counterpoint inside the groove."),
    ("Dub, trip-hop, and vocal depth", "Grace Jones", "Nightclubbing", "1981", "Nightclubbing", "Dry rhythmic authority coexists with selective vocal and instrumental depth, illustrating how restraint makes occasional spatial expansion more powerful."),
    ("Dub, trip-hop, and vocal depth", "Cocteau Twins", "Heaven or Las Vegas", "1990", "Heaven or Las Vegas", "Layered guitar and voice use modulation and bright diffusion to create a wide field that remains buoyant instead of collapsing into masking."),
    ("Dub, trip-hop, and vocal depth", "Tricky", "Hell Is Round the Corner", "1995", "Maxinquaye", "Whisper-close voices sit against a darker sampled environment, making low-level ambience and spectral contrast carry psychological distance."),
    ("Dub, trip-hop, and vocal depth", "FKA twigs", "cellophane", "2019", "Magdalene", "An exposed vocal, piano resonance, and carefully delayed blooms demonstrate how a nearly empty arrangement can support radical changes in emotional scale."),
    ("Screen, installation, and game worlds", "Jerry Goldsmith", "Main Title from Alien", "1979", "Alien", "Orchestral timbre and scoring-stage depth suggest a vast hostile environment before image-specific effects define the fictional space."),
    ("Screen, installation, and game worlds", "Wendy Carlos and Rachel Elkind", "Main Title from The Shining", "1980", "The Shining", "Electronic sustain, chant-like material, and low-frequency bloom create architectural dread through slow spectral motion rather than realistic room simulation."),
    ("Screen, installation, and game worlds", "Akira Yamaoka", "Theme of Laura", "2001", "Silent Hill 2", "Guitar, drums, and grainy atmosphere balance song-like immediacy with the unstable environmental identity of a psychological game world."),
    ("Screen, installation, and game worlds", "Jeremy Soule", "Secunda", "2011", "The Elder Scrolls V: Skyrim", "Quiet piano and orchestral haze are designed to coexist with changing gameplay ambience, offering a model for depth that tolerates interruption and repetition."),
    ("Screen, installation, and game worlds", "Hildur Guðnadóttir", "Bridge of Death", "2019", "Chernobyl", "Cello-derived textures, noise, and industrial resonance blur the line between score, environment, and imagined physical danger."),
    ("Contemporary immersive production", "Hans Zimmer", "Paul's Dream", "2021", "Dune", "Voice, percussion, synthetic low end, and immense multichannel scale demonstrate how immersive reverberation can imply geography and ritual simultaneously."),
    ("Contemporary immersive production", "Nicolas Jaar", "Space Is Only Noise If You Can See", "2011", "Space Is Only Noise", "Speech, pulses, close detail, and oblique room cues make negative space and changing perspective central to the track's identity."),
    ("Contemporary immersive production", "Oneohtrix Point Never", "Chrome Country", "2013", "R Plus Seven", "Synthetic choir, bright transients, and sudden depth changes turn familiar reverberant signs into unstable digital architecture."),
    ("Contemporary immersive production", "Holly Herndon", "Chorus", "2014", "Platform", "Distributed vocal fragments and machine-cut ambience expose the expressive boundary between spatial continuity and frame-level interruption."),
    ("Contemporary immersive production", "Björk", "Black Lake", "2015", "Vulnicura", "Voice, strings, electronics, and long-form spatial expansion create a dramatic landscape whose scale changes with the emotional argument."),
    ("Contemporary immersive production", "Arooj Aftab", "Mohabbat", "2021", "Vulture Prince", "The voice remains poised and intimate while harp, strings, and subtle returns construct a deep, slow-moving nocturnal field."),
    ("Contemporary immersive production", "Beyoncé", "VIRGO'S GROOVE", "2022", "Renaissance", "Dense vocal arrangement, groove, and immersive release formats invite comparison of front-focused clarity with enveloping production detail."),
    ("Early music, organ, and monumental acoustics", "Guillaume Dufay", "Nuper rosarum flores", "1436", "motet", "Proportional form and ceremonial placement invite listeners to separate claims about architectural symbolism from directly audible spatial organization."),
    ("Early music, organ, and monumental acoustics", "Johannes Ockeghem", "Missa prolationum", "c. 1470", "cyclic mass", "Canonic voices at different mensural relationships require a resonant acoustic that blends sonority without erasing independent temporal layers."),
    ("Early music, organ, and monumental acoustics", "Thomas Tallis", "If ye love me", "c. 1540", "motet", "Clear syllabic writing provides a controlled reference for consonant intelligibility and harmonic warmth in contrasting church acoustics."),
    ("Early music, organ, and monumental acoustics", "Charles-Marie Widor", "Symphony for Organ No. 5: Toccata", "1879", "organ symphony", "Repeated manual figures and sustained pedal energy expose low-frequency decay, registration-dependent masking, and the building's contribution to apparent power."),
    ("Early music, organ, and monumental acoustics", "César Franck", "Choral No. 3 in A minor", "1890", "Trois chorals", "Dense chromatic harmony and organ sustain reveal when reverberation reinforces phrase architecture and when it obscures harmonic rhythm."),
    ("Early music, organ, and monumental acoustics", "György Ligeti", "Volumina", "1962", "", "Clusters, changing wind pressure, and massive spectral blocks activate the organ and room as one nonlinear, slowly decaying instrument."),
    ("Opera, theater, and staged distance", "Henry Purcell", "Dido and Aeneas: Dido's Lament", "1689", "Dido and Aeneas", "A grounded bass and exposed vocal line let hall decay extend grief without displacing the singer from the dramatic foreground."),
    ("Opera, theater, and staged distance", "Giuseppe Verdi", "Messa da Requiem: Dies irae", "1874", "Messa da Requiem", "Explosive chorus and bass drum alternate with near-silence, making room recovery time and dynamic headroom essential to the theatrical effect."),
    ("Opera, theater, and staged distance", "Giacomo Puccini", "Tosca: Te Deum", "1900", "Tosca", "Processional layers, bells, chorus, organ, and solo voice build a convincing vertical hierarchy within a single stage and pit perspective."),
    ("Opera, theater, and staged distance", "Benjamin Britten", "Peter Grimes: Four Sea Interludes", "1945", "Peter Grimes", "Orchestral depth and changing spectral weather translate an unseen environment into the psychological space surrounding the drama."),
    ("Opera, theater, and staged distance", "Kaija Saariaho", "L'Amour de loin", "2000", "L'Amour de loin", "Orchestra and electronics mediate literal and emotional distance, making unattainable proximity the opera's central spatial condition."),
    ("Opera, theater, and staged distance", "George Benjamin", "Written on Skin", "2012", "Written on Skin", "Transparent orchestration and sharply controlled resonance permit sudden shifts between narrated distance, immediate violence, and suspended time."),
    ("Resonant instruments and chamber scale", "Ludwig van Beethoven", "Piano Sonata No. 14 in C-sharp minor, Op. 27 No. 2", "1801", "piano sonata", "Repeated accompaniment and long pedal invite comparison between notated resonance, instrument decay, and the room's added harmonic blur."),
    ("Resonant instruments and chamber scale", "Frédéric Chopin", "Prelude in E minor, Op. 28 No. 4", "1839", "prelude", "Sparse attacks and changing inner voices make even a short room response consequential to harmonic pacing and cadential silence."),
    ("Resonant instruments and chamber scale", "Maurice Ravel", "Ondine", "1908", "Gaspard de la nuit", "Rapid figuration, pedal, and register create a liquid halo whose clarity depends on balancing instrumental resonance against room decay."),
    ("Resonant instruments and chamber scale", "Tōru Takemitsu", "Rain Tree Sketch II", "1992", "piano work", "Isolated sonorities and carefully weighted silence allow every decay to function as orchestration and formal punctuation."),
    ("Resonant instruments and chamber scale", "John Luther Adams", "The Farthest Place", "2001", "piano work", "Layered resonance and spacious pacing make the piano suggest an environment while retaining the physical detail of hammer and string."),
    ("Resonant instruments and chamber scale", "Nils Frahm", "Says", "2013", "Spaces", "A repeating synthesizer figure grows into a wide live field, providing a clear example of envelopment increasing while musical material remains economical."),
    ("Experimental voice, installation, and feedback", "Steve Reich", "Come Out", "1966", "tape work", "Phase divergence transforms one recorded voice from intelligible speech into rhythm, spectrum, and a virtual field of multiplying sources."),
    ("Experimental voice, installation, and feedback", "Cathy Berberian", "Stripsody", "1966", "solo vocal work", "Rapid shifts among vocal characters test whether acoustic support preserves theatrical articulation without homogenizing timbre."),
    ("Experimental voice, installation, and feedback", "Joan La Barbara", "Circular Song", "1975", "Voice Is the Original Instrument", "Continuous breath, multiphonics, and resonant vocal technique complicate the distinction between source sustain and added reverberation."),
    ("Experimental voice, installation, and feedback", "Robert Ashley", "Automatic Writing", "1979", "Automatic Writing", "Whispered speech, electronics, and close-room ambiguity create an unstable boundary between private utterance and environmental sound."),
    ("Experimental voice, installation, and feedback", "Trevor Wishart", "Vox 5", "1986", "VOX Cycle", "Continuous transformation between voice and synthetic resonance offers a reference for spaces whose apparent material changes with the source."),
    ("Experimental voice, installation, and feedback", "Pamela Z", "An In", "1998", "A Delay Is Better", "Live voice, looping, and delay build contrapuntal layers whose intelligibility depends on precise timing and selective spectral space."),
    ("Soul, gospel, country, and vocal presence", "Elvis Presley", "Blue Moon", "1954", "Elvis Presley", "A distant, wavering vocal return contrasts with sparse accompaniment, making early studio echo part of the singer's vulnerable persona."),
    ("Soul, gospel, country, and vocal presence", "The Staple Singers", "I'll Take You There", "1972", "Be Altitude: Respect Yourself", "Economical groove and vocal call-and-response show how a compact ambience can support communal energy without softening rhythmic edges."),
    ("Soul, gospel, country, and vocal presence", "Dolly Parton", "I Will Always Love You", "1974", "Jolene", "A direct lead vocal and restrained instrumental field demonstrate how subtle depth can preserve narrative intimacy at emotional peaks."),
    ("Soul, gospel, country, and vocal presence", "Bruce Springsteen", "I'm on Fire", "1984", "Born in the U.S.A.", "Quiet voice, short percussion, and nocturnal ambience create psychological distance through low-level detail rather than spectacular decay."),
    ("Soul, gospel, country, and vocal presence", "Cowboy Junkies", "Sweet Jane", "1988", "The Trinity Session", "A single ambisonic microphone in a resonant church captures performers and architecture as one scene, making placement inseparable from balance."),
    ("Soul, gospel, country, and vocal presence", "Lucinda Williams", "Right in Time", "1998", "Car Wheels on a Gravel Road", "Voice and band combine dryness, slap-like depth, and controlled sustain to keep lyrical presence inside a textured roots production."),
    ("Heavy music, post-rock, and overwhelming scale", "Black Sabbath", "Black Sabbath", "1970", "Black Sabbath", "Rain, tolling bell, sparse guitar, and long decay establish dread before the ensemble's weight arrives, making environment part of the riff's meaning."),
    ("Heavy music, post-rock, and overwhelming scale", "Joy Division", "Atmosphere", "1980", "Licht und Blindheit", "Distant drums, baritone voice, and synthetic haze turn production depth into the song's defining emotional architecture."),
    ("Heavy music, post-rock, and overwhelming scale", "Talk Talk", "After the Flood", "1991", "Laughing Stock", "Organ, guitar distortion, room tone, and dynamic restraint create scale through accumulated resonance rather than continuous loudness."),
    ("Heavy music, post-rock, and overwhelming scale", "Slowdive", "When the Sun Hits", "1993", "Souvlaki", "Vocal intimacy and broad guitar wash demonstrate how midrange density can remain emotionally legible when attacks and tails overlap."),
    ("Heavy music, post-rock, and overwhelming scale", "Mogwai", "Mogwai Fear Satan", "1997", "Mogwai Young Team", "A long crescendo moves from exposed instrumental room to saturated mass, revealing how decay strategy must change with level and density."),
    ("Heavy music, post-rock, and overwhelming scale", "Cult of Luna", "Vicarious Redemption", "2013", "Vertikal", "Layered guitars, electronics, and drums construct industrial scale while alternating narrow pressure with wide, high-density release."),
    ("Dub techno and electronic depth", "Maurizio", "M4", "1995", "M-Series", "A restrained pulse and slowly changing chord return demonstrate how filtering and feedback can imply vast depth with very few events."),
    ("Dub techno and electronic depth", "Porter Ricks", "Port Gentil", "1996", "Biokinetics", "Submerged transients and pressure-like modulation make the field feel physical even when ordinary source and room boundaries disappear."),
    ("Dub techno and electronic depth", "Pole", "Fahren", "1998", "CD 1", "Clicks, bass, and unstable delay reveal how faults and residual noise can articulate the edges of a synthetic acoustic."),
    ("Dub techno and electronic depth", "Burial", "Archangel", "2007", "Untrue", "Pitch-shifted voice, rain-like noise, and displaced percussion create an urban depth field from fragments rather than continuous reverberation."),
    ("Dub techno and electronic depth", "Yagya", "Rigning One", "2009", "Rigning", "Rain and soft chord pulses maintain multiple time scales, allowing listeners to compare environmental continuity with musical decay."),
    ("Dub techno and electronic depth", "Vladislav Delay", "Kuopio", "2012", "Kuopio", "Irregular echo, clipped events, and broad low-frequency motion challenge assumptions that depth requires smooth or naturalistic late fields."),
    ("Soundscape composition and environmental scale", "Pierre Schaeffer", "Étude aux chemins de fer", "1948", "Cinq études de bruits", "Recorded trains are reorganized so perspective, mechanical rhythm, and captured environment become compositional parameters independent of their original scene."),
    ("Soundscape composition and environmental scale", "Iannis Xenakis", "Bohor", "1962", "electroacoustic work", "Dense metallic spectra and sustained energy create a virtual enclosure whose scale changes with playback level, speaker spacing, and room response."),
    ("Soundscape composition and environmental scale", "Francisco López", "La Selva", "1998", "La Selva", "Rainforest recordings remove visual explanation and foreground the listener's changing inference of distance, density, and biological activity."),
    ("Soundscape composition and environmental scale", "Toshiya Tsunoda", "Scenery from a Lighthouse", "2003", "Scenery from a Lighthouse", "Vibration and environmental detail expose transmission through structures, making contact resonance as important as airborne perspective."),
    ("Soundscape composition and environmental scale", "Peter Cusack", "Sounds from Dangerous Places", "2012", "Sounds from Dangerous Places", "Recordings from environmentally damaged sites ask how audible place, documentation, and political context alter the interpretation of acoustic space."),
    ("Soundscape composition and environmental scale", "Lawrence English", "Cruel Optimism", "2017", "Cruel Optimism", "Field recordings, voice, and dense electronic layers move between recognizable environment and overwhelming abstraction without a fixed listening distance."),
)

_MUSIC_CATEGORY_ORDER = {
    category: index
    for index, category in enumerate(dict.fromkeys(entry[0] for entry in MUSIC_EXPANSION_CORE))
}
MUSIC_EXPANSION = tuple(
    sorted(
        (*MUSIC_EXPANSION_CORE, *MUSIC_EXPANSION_ADDITIONS),
        key=lambda entry: _MUSIC_CATEGORY_ORDER[entry[0]],
    )
)

MUSIC_STUDY_PROMPTS = (
    "Rebuild one 20–30 second spatial gesture from {title} with original material, then preserve the exact render command and JSON report.",
    "Match the apparent depth of {title} with convolution and algorithmic engines at equal loudness; compare pre-delay, direct-to-reverberant ratio, and high-band decay.",
    "Create dry, early-only, late-only, and complete reconstructions of one cue from {title}; identify which path carries its spatial identity.",
    "Analyze a lawful excerpt or an original reconstruction of {title} with `verbx analyze`; compare audible decay with broadband and frequency-dependent measurements.",
    "Translate the spatial premise of {title} to mono, stereo, and surround; document which distance and envelopment cues survive each layout.",
    "Automate only one spatial parameter across a short form inspired by {title}; test whether the form remains legible without level automation.",
)

MUSIC_PRIMARY_SOURCES = {
    ("Annea Lockwood", "A Sound Map of the Hudson River"): (
        "Composer work catalog",
        "https://www.annealockwood.com/compositions/",
    ),
    ("Chris Watson", "Vatnajökull"): (
        "Artist release notes",
        "https://chriswatson.net/2003/08/02/to47-weather-report/",
    ),
    ("Ellen Fullman and Okkyung Lee", "The Air Around Her, Part I"): (
        "Long String Instrument documentation",
        "https://www.ellenfullman.com/about",
    ),
    ("Hildegard Westerkamp", "Kits Beach Soundwalk"): (
        "Composer work page",
        "https://www.hildegardwesterkamp.ca/sound/comp/3/kitsbeach/",
    ),
    ("Jana Winderen", "Aquaculture"): (
        "Artist release notes",
        "https://www.janawinderen.com/releases/energy-field",
    ),
    ("Janet Cardiff", "The Forty Part Motet"): (
        "Artist installation record",
        "https://cardiffmiller.com/installations/the-forty-part-motet/",
    ),
    ("Meredith Monk", "Gotham Lullaby"): (
        "Artist recording catalog",
        "https://meredithmonk.org/store/cds",
    ),
    ("Pauline Oliveros, Stuart Dempster, and Panaiotis", "Lear"): (
        "Pauline Oliveros discography",
        "https://www.paulineoliveros.us/disco.html",
    ),
}


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
    if len(MUSIC_EXPANSION) != 192:
        raise ValueError(
            f"Expected 192 expanded listening entries, found {len(MUSIC_EXPANSION)}"
        )

    lines = [
        "## Expanded Listening Canon",
        "",
        "The following 192 additions broaden the appendix across sacred, orchestral, jazz, popular, ambient, theater, installation, environmental, game, and contemporary immersive practices. Piece and album titles are italicized throughout. Each entry pairs a listening cue with a practical study prompt. Links open a stable YouTube catalog search so readers can select an authorized or territorially available recording; recording-specific citations in the preceding section and the primary documentation below remain the preferred references where supplied.",
        "",
        "### Curated Listening Routes",
        "",
        "- **Architecture and choir.** Begin with *Spem in alium*, *Miserere mei, Deus*, *Passio*, *The Forty Part Motet*, and *Kits Beach Soundwalk* to compare written spatial design, recorded architecture, installation, and narrated place.",
        "- **Rhythm and short space.** Move from *Mystery Train* through *King Tubby Meets Rockers Uptown*, *Quadrant Dub I*, *Welcome to the Terrordome*, and *Dream House* to hear when reflection becomes pulse, syncopation, or masking.",
        "- **Long decay and near-stasis.** Compare *Apparition de l'église éternelle*, *Schlingen-Blängen*, *Deep Listening*, *Jetsun Mila*, *The Sacrificial Code*, and *#3 (Rhubarb)* at matched monitoring level.",
        "- **Voice and psychological distance.** Trace the changing foreground in *A Change Is Gonna Come*, *Amazing Grace*, *Where Will I Be*, *Roads*, *cellophane*, and *when the party's over*.",
        "- **Movement and immersion.** Study *Kontakte*, *Persephassa*, *Répons*, *Le Encantadas o le avventure nel mare delle meraviglie*, *Movement 6*, and *Billions* with explicit attention to trajectory and envelopment.",
        "- **Environment as form.** Hear *Presque rien No. 1*, *Music for Wilderness Lake*, *A Sound Map of the Hudson River*, *Kits Beach Soundwalk*, *Vatnajökull*, and *Aquaculture* as composed relationships among listener, place, microphone, and playback room.",
        "",
    ]
    active_category = None
    category_index = 0
    for category, creator, title, year, album, note in MUSIC_EXPANSION:
        if category != active_category:
            lines.extend((f"### {category.title()}", ""))
            active_category = category
            category_index = 0
        query = quote_plus(f"{creator} {title} official")
        if not album:
            album_text = ""
        elif album.lower() != title.lower():
            album_text = f" from *{album}*"
        else:
            album_text = f" on *{album}*"
        source = MUSIC_PRIMARY_SOURCES.get((creator, title))
        source_text = f" [{source[0]}]({source[1]})." if source else ""
        study_prompt = MUSIC_STUDY_PROMPTS[category_index % len(MUSIC_STUDY_PROMPTS)].format(
            title=f"*{title}*"
        )
        category_index += 1
        lines.extend((
            f"**{creator}, *{title}* ({year}).**{album_text}{'.' if album_text else ''} {note} [YouTube catalog search](https://www.youtube.com/results?search_query={query}).{source_text}",
            "",
            f"> **Study prompt.** {study_prompt}",
            "",
        ))

    lines.extend((
        "## Selected Primary Documentation",
        "",
        "These creator and institutional pages document works whose site, installation, release, or recording history is especially important to the listening note. The list is alphabetical by creator.",
        "",
    ))
    for (creator, title), (label, url) in sorted(MUSIC_PRIMARY_SOURCES.items()):
        lines.append(f"- {creator}. *{title}*. [{label}]({url}).")
    lines.extend((
        "",
        "\\newpage",
        "",
        "## Appendix A Listening Record",
        "",
        "Use one record for each focused comparison. The aim is to preserve enough evidence that another reader can repeat the listening conditions, understand the inference, and rebuild the spatial principle without copying the recording.",
        "",
        "- **Source and version.** Record the work, performer, release, mix or remaster, playback service or physical medium, and the exact passage studied. Note when a catalog search led to a different recording than the one described in the entry.",
        "- **Playback conditions.** Record sample rate, channel layout, loudspeaker or headphone system, listening level, room, normalization state, and whether any host spatialization or room correction was active.",
        "- **Spatial event map.** Mark direct attacks, first audible reflections, echo clusters, late-field blooms, gates, feedback gestures, movement, and silence. Use elapsed time in seconds so observations can be checked without reproducing lyrics or score text.",
        "- **Measured evidence.** Save `verbx analyze` JSON for lawful source material or for the reconstruction. Separate measured quantities from perceptual descriptions, and do not present program-material RT60 estimates as calibrated room measurements.",
        "- **Reconstruction.** Preserve the dry source, exact command or plug-in state, random seed where relevant, output format, and one deliberately contrasting render. State which property was held constant during the comparison.",
        "- **Listening judgment.** Describe what the reverberation contributes to rhythm, form, harmony, orchestration, drama, or place. End by naming one parameter that mattered less than expected and one interaction that mattered more.",
        "",
        "**Two-pass reconstruction protocol.**",
        "",
        "1. **Establish the baseline.** Trim a short original source, set conservative monitor gain, and render a dry reference. Write a prediction for apparent size, distance, decay color, and movement before choosing an engine or preset.",
        "2. **Build controlled layers.** Create early-only and late-only versions before the complete reconstruction. Change one parameter family at a time, level-match every comparison, and keep at least one intentionally wrong version that makes the role of the target cue easier to hear.",
        "3. **Test without labels.** Randomize the dry reference, reconstruction, and contrasting render. Ask at least one listener to rank distance, clarity, envelopment, and musical fit without seeing the settings. Record disagreement rather than averaging it away.",
        "4. **Archive the evidence.** Save audio hashes, commands or plug-in state, reports, routing notes, and a short conclusion. If the study uses a commercial recording for listening, archive only bibliographic and timing information unless distribution rights permit more.",
        "",
        "A complete record should fit on this page. It is evidence for critical listening, not a claim that one preset or metric explains the artistic result. The strongest conclusion names the audible interaction, the controlled comparison that exposed it, and the limit beyond which the reconstruction stopped serving the music.",
        "",
    ))
    markdown = "\n".join(lines).rstrip() + "\n"
    markdown, _ = COMPOSITION_YEARS.normalize_composition_years(
        markdown, COMPOSITION_YEARS.composition_catalog()
    )
    (ROOT / "docs" / "MUSICAL_PIECES_EXPANSION.md").write_text(
        markdown, encoding="utf-8"
    )


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
            if (
                line.startswith("**")
                and not line.startswith("**Study prompt.**")
                and not re.fullmatch(r"\*\*[^*]+\.\*\*", line)
                and not re.match(
                    r"^\*\*.+?, \*[^*]+\* \([^)]+\)"
                    r"(?:, from \*[^*]+\*(?: \([^)]+\))?)?\.\*\*",
                    line,
                )
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
    print("Wrote introductory diagrams, 96 listening entries, and 48 educational projects")


if __name__ == "__main__":
    main()
