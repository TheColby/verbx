#!/usr/bin/env python3
"""Generate the introductory diagram chapter and educational appendix."""

from __future__ import annotations

from html import escape
from pathlib import Path
import re
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
    ("Freeze Stability Study", "infinite sustain and energy control", "Run a two-minute freeze test and document spectral accumulation, drift, and peak growth."),
    ("Reverse-Reverb Design", "envelope reversal and transition timing", "Build phrase-leading reverse tails that arrive musically without masking the destination transient."),
    ("Shimmer Feedback Safety", "pitch shifting inside feedback", "Map the boundary between harmonic bloom and unstable high-frequency accumulation."),
    ("Ducking as Orchestration", "sidechain dynamics and intelligibility", "Create three release laws that reveal different rhythmic subdivisions after a vocal phrase."),
    ("Limiter Transparency", "lookahead, knee, release, and true peak", "Protect an extreme wet mix while minimizing transient flattening and pumping."),
    ("Convolution Partition Plan", "latency versus throughput", "Select partition sizes for live monitoring, music production, and offline archival rendering."),
    ("Impulse-Response Capture", "sweeps, deconvolution, and noise", "Plan and document a repeatable room capture with calibration and failure checks."),
    ("IR Editing and Provenance", "trim, normalize, resample, and hash", "Prepare an IR library while preserving source identity and reversible processing metadata."),
    ("IR Morphing Atlas", "interpolation and perceptual continuity", "Create a five-point trajectory between contrasting spaces and identify discontinuities."),
    ("Dereverberation Tradeoff", "late-energy suppression and artifacts", "Build a strength sweep and select an operating point from measured and blind-listening evidence."),
    ("Speech Intelligibility Study", "DRR, clarity, and spectral masking", "Compare natural reverb, dereverberation, and re-reverberation for spoken material."),
    ("Realtime Latency Audit", "device, host, DSP, and acoustic delay", "Measure round-trip latency and reconcile it with the predicted buffer stack."),
    ("Realtime Failure Injection", "xruns, overload, and recovery", "Stress block size and quality modes while recording dropout thresholds and recovery behavior."),
    ("Automation Reproducibility", "timeline interpolation and deterministic state", "Render the same automation in several block sizes and compare audio and JSON outputs."),
    ("Preset Taxonomy", "semantic naming and controlled variation", "Design a coherent twelve-preset family whose labels predict audible differences."),
    ("Loudness-Matched Evaluation", "LUFS, peak, and perceptual bias", "Compare six reverb designs only after integrated loudness and peak constraints are matched."),
    ("Stereo Width Without Phase Damage", "correlation and mono compatibility", "Increase envelopment while maintaining a documented mono fold-down target."),
    ("Multichannel Bus Verification", "channel identity and matrix routing", "Use labeled impulses to prove every input-to-output route in a surround configuration."),
    ("Ambisonic Rotation Exercise", "scene representation and decoding", "Rotate a first-order scene, decode it twice, and verify spatial and level invariants."),
    ("Room-Model Inversion", "Sabine estimates and geometric constraints", "Infer plausible dimensions and absorption from RT60, then explain non-unique solutions."),
    ("Ray-Traced Early Field Proposal", "CAD geometry, materials, and validation", "Specify a DXF-to-reflection prototype with assumptions, error bounds, and listening tests."),
    ("Machine-Readable Evidence", "analysis JSON and schema stability", "Design a report consumed by a regression script without parsing human progress text."),
    ("Native Plug-in Parity Slice", "C++ DSP, host state, and deterministic comparison", "Port one narrow feature and prove parity against the Python reference."),
    ("Capstone Spatial Production", "integrated artistic and engineering practice", "Deliver a multichannel production, reproducibility bundle, technical report, and critical reflection."),
)

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
            f"![Block diagram {number}: {title}.](assets/intro_block_diagrams/{png_filename})", "",
            f"**Block diagram {number}.** {description} The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.", "",
        ))
    (ROOT / "docs" / "INTRODUCTORY_BLOCK_DIAGRAMS.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def generate_projects() -> None:
    lines = [
        "# Educational Exercises and Project Assignments",
        "",
        "These projects form a progressive laboratory course in reverberation, spatial audio, realtime systems, critical listening, and reproducible audio engineering. Each assignment is scoped for one book page; instructors may treat it as a weekly laboratory, combine adjacent projects, or select a focused sequence.",
        "",
        "## Assessment Framework",
        "",
        "Evaluate every submission on prediction, method, evidence, listening judgment, reproducibility, and clarity. Unless a project states otherwise, students should retain source audio, exact commands, parameter or preset state, analysis JSON, plots, and a short reflection distinguishing observation from interpretation.",
        "",
    ]
    for number, (title, concepts, brief) in enumerate(PROJECTS, 1):
        lines.extend((
            "\\newpage", "", f"## Project {number}: {title}", "",
            f"**Central concepts:** {concepts}.", "",
            f"**Design brief.** {brief} Begin with a written prediction that names the expected audible and measurable changes. Use controlled source material and alter one principal variable at a time before combining controls.", "",
            "**Procedure.** Establish a dry baseline and one conservative reference render. Create at least five documented variants spanning the useful range, including one deliberately poor or unstable case when safe. Loudness-match comparisons, preserve deterministic seeds where applicable, and record effective settings rather than relying on command history alone.", "",
            "**Evidence package.** Submit the source and rendered excerpts, exact runnable commands, preset or automation files, analysis JSON, a compact comparison table, and one labeled figure. Include listening conditions, sample rate, channel layout, block or partition size, software revision, and any warnings produced by verbx.", "",
            "**Questions for the report.** Which prediction was confirmed? Which result contradicted the model? What changed perceptually before a standard metric changed? Where does the preferred setting sit relative to a technical failure boundary? Name one confound and design a follow-up that isolates it.", "",
            "**Completion standard.** Another student must be able to reproduce the central result from the submitted materials. Conclusions must cite both measured evidence and level-matched critical listening; screenshots alone are not evidence.", "",
            "**Extension.** Repeat the decisive comparison with a contrasting source, room, sample rate, or reproduction layout and explain which conclusions generalize.", "",
            "```{=latex}",
            "\\vfill",
            f"\\verbxAssignmentPlate{{{number}}}{{{title}}}",
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
        r"(?m)^\*\*(?P<creator>.+?),\s+(?P<title>(?!\*)[^\n]+?)\s+"
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
    path.write_text(text, encoding="utf-8")


def main() -> None:
    normalize_music_typography()
    generate_diagrams()
    generate_projects()
    generate_music_expansion()
    print("Wrote introductory diagrams, 48 listening entries, and 33 educational projects")


if __name__ == "__main__":
    main()
