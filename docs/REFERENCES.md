# Research Papers and References

Focused bibliography for extreme reverberation DSP and reverberation research in general (algorithmic reverb, FDN, convolution/IR, late-field modeling, and room reverberation metrics).

Pruned from the broader bibliography on 2026-03-01 to keep only reverb-centric papers.

Total entries: 1,002 (102 curated annotated entries + 900 extended Crossref entries)

Within each topical section, entries are alphabetized by first-author surname. Records
without credited authors sort under "Unknown authors."

---

## Where to Start

If you are new to reverb DSP and want to understand the intellectual lineage of this tool,
start with these seven papers, listed alphabetically by first author. The notes suggest a
useful reading sequence; everything else in the bibliography builds on, extends, or argues
with what these establish.

**1. Jot and Chaigne (1997) — Maximally diffusive yet efficient FDN** ([entry 95](#entry-95))

The pivot point between classic Schroeder-style structures and modern FDN design. Jot formalized the feedback delay network as a matrix + delay system, gave a clean stability condition, and showed how to design the feedback matrix for maximum diffusion. This paper is the reason every serious reverb plugin since the late 1990s has an "FDN" somewhere inside it.

**2. Rocchesso and Smith (1997) — Circulant and elliptic FDN families** ([entry 96](#entry-96))

Published the same year as Jot-Chaigne and essential reading alongside it. Rocchesso and Smith worked out the eigenvalue structure of feedback matrices, explained why some matrix designs produce audible resonances, and gave concrete construction rules for matrices that avoid them. The circulant family they describe remains a go-to for large-order FDNs.

**3. Schlecht and Habets (2015) — Time-varying feedback matrices** ([entry 39](#entry-39))

Identifies the modal ringing problem that plagues fixed-matrix FDNs (you hear it as metallic "flutter" on transients) and proposes time-varying rotation of the feedback matrix as the solution. This is the theoretical grounding for verbx's reactive-control stream. Dense math, but the payoff is real.

**4. Schlecht and Habets (2019) — Dense Reverberation with Delay Feedback Matrices** ([entry 29](#entry-29))

The most recent of the mandatory reads. Introduces the delay-feedback matrix concept, which lets you pack feedback directly into the delay topology rather than separating the two. Yields higher echo density for a given compute budget. The paper that shapes verbx's next-generation topology expansion.

**5. Schroeder (1961) — Further Progress with Colorless Artificial Reverberation** ([entry 97](#entry-97))

The immediate follow-up, published the same year. Refines the allpass topology and introduces the parallel-comb / series-allpass architecture that became the canonical "Schroeder reverberator." The coloration problems it leaves unsolved are exactly what motivated everything in the FDN section below.

**6. Schroeder and Logan (1961) — "Colorless" artificial reverberation** ([entry 98](#entry-98))

This is the paper that launched a thousand reverb plugins. Schroeder identified the two fundamental primitives — the comb filter and the allpass filter — and showed how to combine them to produce a plausible diffuse tail without coloring the sound. Almost every algorithmic reverb written in the six decades since has either implemented these ideas directly or consciously departed from them. Read this before anything else.

**7. Valimaki et al. (2012) — Fifty Years of Artificial Reverberation** ([entry 59](#entry-59))

A historical survey connecting mechanical, algorithmic, convolution, and perceptual reverberation.

---

## Key Results Reference

Quick-lookup table of the equations you will cite most often during development. These are not derivations — follow the source links for those.

| Result | Formula | Source | Notes |
|---|---|---|---|
| **Sabine equation** | $T_{60}=0.161V/A$, where $A=\sum_i S_i\alpha_i$ | Sabine (1900), summarized in [RA2](#entry-ra2), [RA3](#entry-ra3), and [RA5](#entry-ra5) | Assumes perfectly diffuse field. Breaks down in rooms with non-uniform absorption or very low average absorption coefficient. Over-predicts $T_{60}$ in dead rooms. |
| **Eyring correction** | $T_{60}=0.161V/[-S\ln(1-\bar{\alpha})]$ | Eyring (1930), see [RA2](#entry-ra2) and [RA5](#entry-ra5) | More accurate when average absorption is high ($\bar{\alpha}>0.3$). Reduces to Sabine in the limit of low absorption. |
| **FDN gain calibration** | $g_i = 10^{-3d_i/T_{60}}$ per delay line, where $d_i$ is delay length in seconds | Jot and Chaigne (1997), entry [95](#entry-95); Schlecht and Habets (2015), entry [39](#entry-39) | Applied per-band when using frequency-dependent absorption filters on the delay outputs. This is the central calibration formula for matching a target RT60. |
| **EDT definition** | Early Decay Time = time for first 10 dB of decay on the energy decay curve, extrapolated to 60 dB | ISO 3382-1; summarized in entry [80](#entry-80) | EDT correlates better with perceived liveness than RT60 in spaces with non-exponential decay. |
| **C80 (Clarity)** | $C_{80} = 10 \log_{10}\!\left(\frac{\int_{0}^{80\,\mathrm{ms}} h^{2}(t)\,dt}{\int_{80\,\mathrm{ms}}^{\infty} h^{2}(t)\,dt}\right)\,\mathrm{dB}$ | ISO 3382-1; see entry [80](#entry-80) | Ratio of early to late energy, 80 ms threshold. Positive values indicate clear/direct sound; negative values indicate reverberant/muddy. |
| **D50 (Definition)** | $D_{50} = \frac{\int_{0}^{50\,\mathrm{ms}} h^{2}(t)\,dt}{\int_{0}^{\infty} h^{2}(t)\,dt}$ | ISO 3382-1; see entry [80](#entry-80) | Fraction of total energy arriving in first 50 ms. Ranges 0-1; higher values correlate with better speech intelligibility. Uses 50 ms threshold versus C80's 80 ms. |

---

## Roadmap Citation Index (v0.7.0 Completion Program)

### By stream

- `Stream R1 (reactive control hardening)`:
  [Schlecht and Habets (2015) - time-varying FDN matrices](#ref-fdn-tv-matrix-2015),
  [Valimaki et al. (2012) - fifty years of artificial reverberation](#ref-reverb-survey-2012)
- `Stream R2 (perceptual/Jot-inspired calibration)`:
  [Jot and Chaigne (1997) - maximally diffusive FDN](#ref-jot-chaigne-1997),
  [Valimaki et al. (2012) - fifty years of artificial reverberation](#ref-reverb-survey-2012)
- `Stream R3 (IR morph productionization)`:
  [Valimaki et al. (2012) - fifty years of artificial reverberation](#ref-reverb-survey-2012)
- `Stream R4 (next-generation topology expansion)`:
  [Rocchesso and Smith (1997) - circulant/elliptic FDN families](#ref-fdn-circulant-elliptic-1997),
  [Schlecht and Habets (2019) - delay feedback matrices](#ref-fdn-delay-feedback-2019)

### Core canonical links


- [Rocchesso and Smith (1997) - circulant/elliptic FDN families](#ref-fdn-circulant-elliptic-1997)
- [Schlecht and Habets (2015) - time-varying FDN matrices](#ref-fdn-tv-matrix-2015)
- [Schlecht and Habets (2019) - delay feedback matrices](#ref-fdn-delay-feedback-2019)
- [Valimaki et al. (2012) - fifty years of artificial reverberation](#ref-reverb-survey-2012)


---

## Section 1: Foundational Works

The papers every reverb developer must read. These define the vocabulary, the fundamental structures, and the problems that all subsequent work attempts to solve.

<a id="entry-95"></a><a id="ref-jot-chaigne-1997"></a>
**[F4]** Jot, J.-M.; Chaigne, A. (1997). Maximally diffusive yet efficient feedback delay networks for artificial reverberation. *IEEE Signal Processing Letters*. DOI: [10.1109/97.623041](https://doi.org/10.1109/97.623041)

> Annotation: The paper that unified the FDN framework and gave practitioners the tools to actually design good reverbs rather than accidentally stumbling into them. Provides the stability condition, the diffuseness criterion, and the RT60 calibration formula that verbx uses directly.

**[F8]** Polack, Jean-Dominique (2025). Revisiting reverberation. *Acta Acustica*. DOI: [10.1051/aacus/2025010](https://doi.org/10.1051/aacus/2025010)

> Annotation: A senior researcher's reassessment of the theoretical assumptions behind classical reverberation theory. Worth reading for the perspective it offers on what the Sabine/Eyring/FDN frameworks do and do not capture.

**[F7]** Reiss, Joshua D.; McPherson, Andrew P. (2026). Reverberation. *Audio Effects*. DOI: [10.1201/9781003593942-10](https://doi.org/10.1201/9781003593942-10)

> Annotation: A current textbook chapter that surveys the state of practical reverb algorithm design. Useful as a pedagogical introduction before diving into the primary literature above.

<a id="entry-96"></a><a id="ref-fdn-circulant-elliptic-1997"></a>
**[F3]** Rocchesso, D.; Smith, J.O. (1997). Circulant and elliptic feedback delay networks for artificial reverberation. *IEEE Transactions on Speech and Audio Processing*. DOI: [10.1109/89.554269](https://doi.org/10.1109/89.554269)

> Annotation: Works out the eigenvalue structure of feedback matrices in detail and explains exactly why naive matrix choices cause audible modal ringing. Introduces the circulant family of matrices, which remain a practical default for large FDN orders. Indispensable alongside Jot-Chaigne.

<a id="entry-97"></a>
**[F2]** Schroeder, M.R. (1961). Further Progress with Colorless Artificial Reverberation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.1936895](https://doi.org/10.1121/1.1936895)

> Annotation: The companion paper published the same year, refining the allpass topology. Together with the IRE paper above, this establishes the canonical Schroeder reverberator that hardware units of the 1970s and virtually every software reverb through the 1990s was built on.

<a id="entry-98"></a>
**[F1]** Schroeder, M.R.; Logan, B.F. (1961). "Colorless" artificial reverberation. *IRE Transactions on Audio*. DOI: [10.1109/tau.1961.1166351](https://doi.org/10.1109/tau.1961.1166351)

> Annotation: The founding document of algorithmic reverberation. Introduces parallel comb filters followed by series allpass sections as a way to produce high echo density without coloring the spectrum. Every subsequent algorithmic reverb either implements this architecture or explicitly departs from it. Read this first.

**[F9]** Theodore, Michael (2004). Altiverb Reverberation Software. *Computer Music Journal*. DOI: [10.1162/comj.2004.28.2.101](https://doi.org/10.1162/comj.2004.28.2.101)

> Annotation: A practitioner's account of the first commercially successful high-quality convolution reverb plugin. Important context for understanding where the convolution approach succeeded over algorithmic approaches, and what trade-offs it brought.

**[F6]** Unknown authors (1955). Artificial reverberation. *Journal of the Institution of Electrical Engineers*. DOI: [10.1049/jiee-3.1955.0135](https://doi.org/10.1049/jiee-3.1955.0135)

> Annotation: Pre-Schroeder documentation of early tape and mechanical approaches to artificial reverberation. Useful historical context for understanding what Schroeder's 1961 papers were responding to.

**[F10]** Unknown authors (2021). The Role of Modal Excitation in Colorless Reverberation. *2021 24th International Conference on Digital Audio Effects (DAFx)*. DOI: [10.23919/dafx51585.2021.9768217](https://doi.org/10.23919/dafx51585.2021.9768217)

> Annotation: Returns to Schroeder's original "colorless" criterion and examines it through the lens of modal analysis. Explains the precise mechanism by which poorly designed delay networks produce coloration, which is the problem verbx's time-varying matrix stream addresses.

<a id="entry-59"></a><a id="ref-reverb-survey-2012"></a>
**[F5]** Valimaki, V.; Parker, J.D.; Savioja, L.; et al. (2012). Fifty Years of Artificial Reverberation. *IEEE Transactions on Audio, Speech, and Language Processing*. DOI: [10.1109/tasl.2012.2189567](https://doi.org/10.1109/tasl.2012.2189567)

> Annotation: The definitive survey of the field from Schroeder's original papers through convolution and perceptual approaches. The single most useful reference document for a reverb developer. If you understand everything in this paper you understand the intellectual foundation of the entire discipline.

---

## Section 2: Feedback Delay Networks

FDN theory, stability conditions, matrix design, and time-varying extensions. This is the core algorithmic section for verbx.

**[FDN5]** Koontz, Warren L. (2013). Artificial reverberation using multi-port acoustic elements. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4854696](https://doi.org/10.1121/1.4854696)

> Annotation: Proposes multi-port acoustic element networks as an alternative FDN-like architecture. Useful as a conceptual reference for exploring topologies beyond the standard scattering-matrix formulation.

**[FDN2]** Schlecht, S.J.; Habets, E.A.P. (2015). Simulation of Room Reverberation Using a Feedback Delay Network. *Acoustics Australia*. DOI: [10.1007/s40857-015-0005-8](https://doi.org/10.1007/s40857-015-0005-8)

> Annotation: Companion paper to [FDN1], covering implementation details of the full FDN simulation pipeline including frequency-dependent absorption and the gain calibration workflow. More accessible than the JASA paper; read this alongside it.

<a id="entry-39"></a><a id="ref-fdn-tv-matrix-2015"></a>
**[FDN1]** Schlecht, S.J.; Habets, E.A.P. (2015). Time-varying feedback matrices in feedback delay networks and their application in artificial reverberation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4928394](https://doi.org/10.1121/1.4928394)

> Annotation: Proves that slowly rotating the feedback matrix breaks up the fixed modal structure that causes metallic flutter in fixed-matrix FDNs, without destabilizing the network or significantly changing the statistical RT60. The theoretical basis for Stream R1.

<a id="entry-29"></a><a id="ref-fdn-delay-feedback-2019"></a>
**[FDN3]** Schlecht, S.J.; Habets, E.A.P. (2019). Dense Reverberation with Delay Feedback Matrices. *2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)*. DOI: [10.1109/waspaa.2019.8937284](https://doi.org/10.1109/waspaa.2019.8937284)

> Annotation: Introduces delay-feedback matrices, which embed the feedback network directly into the delay topology rather than treating the delay and feedback stages as separate. Achieves higher echo density per unit of computation. This is the architectural foundation for verbx's Stream R4 topology expansion.

**[FDN4]** Unknown authors (2024). Active Acoustics with a Phase Cancelling Modal Reverberator. *Journal of the Audio Engineering Society*. DOI: [10.17743/jaes.2022.0171](https://doi.org/10.17743/jaes.2022.0171)

> Annotation: Applies modal reverberator theory to active acoustic systems. Useful for understanding how FDN structures can be used to synthesize or suppress specific room modes, which informs edge cases in verbx's reactive control design.

---

## Section 3: Convolution and IR Processing

Partitioned FFT convolution, impulse response measurement, and IR morphing. The theoretical backbone for verbx's Stream R3.

**[CV3]** Arcas, Kevin; Chaigne, Antoine (2010). On the quality of plate reverberation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2009.07.013](https://doi.org/10.1016/j.apacoust.2009.07.013)

> Annotation: Analyzes plate reverb physically and identifies the spectral characteristics that distinguish high-quality plate IRs from poor ones. Useful as a reference when evaluating morphed or synthesized IRs against physical ground truth.

**[CV1]** Ducasse, Éric; Rodriguez, Samuel; Bonnet, Marc (2026). Early-reverberation imaging functions for bounded elastic domains. *Acta Acustica*. DOI: [10.1051/aacus/2025069](https://doi.org/10.1051/aacus/2025069)

> Annotation: Develops imaging functions for early reflections from bounded elastic surfaces. Relevant to physically-grounded IR construction and to understanding what the early-reflection part of a measured IR actually encodes about room geometry.

**[CV4]** Mechel, Fridolin (2012). Reverberation with Mirror Sources. *Room Acoustical Fields*. DOI: [10.1007/978-3-642-22356-3_17](https://doi.org/10.1007/978-3-642-22356-3_17)

> Annotation: Textbook chapter on image-source methods for computing room impulse responses. The image-source approach is the classical alternative to FDN for early reflections; understanding it illuminates the trade-offs in hybrid reverb architectures.

**[CV5]** Schneiderwind, Christian; De Sena, Enzo; Neidhardt, Annika (2026). Perceptual effects of modified late reverberation and reverberation time in auditory augmented reality in two rooms. *Acta Acustica*. DOI: [10.1051/aacus/2026012](https://doi.org/10.1051/aacus/2026012)

> Annotation: Examines what happens perceptually when you modify the late tail of a measured IR independently of the early part. Directly relevant to IR morphing workflows — establishes the perceptual independence of early and late portions and the threshold above which manipulation becomes detectable.

**[CV2]** Unknown authors (2008). Room Impulse Responses, Decay Curves and Reverberation Times. *Formulas of Acoustics*. DOI: [10.1007/978-3-540-76833-3_262](https://doi.org/10.1007/978-3-540-76833-3_262)

> Annotation: Reference-book treatment of the mathematical relationship between the room impulse response and the energy decay curve. Essential grounding for anyone implementing EDC analysis or RT60 estimation from measured IRs.

---

## Section 4: Room Acoustics Fundamentals

RT60, energy decay curves, Sabine/Eyring theory, and the physics of reverberant fields. These papers establish the physical quantities that verbx's algorithms attempt to match, measure, and explain to users. They are the backbone references for verbx's analysis JSON, room-model warnings, preset calibration, and documentation around when a single reverberation-time value is or is not enough.

**[RA14]** Berger, Russ (2011). From Sports Arena to Sanctuary: Taming a Texas-Sized Reverberation Time. *Acoustics Today*. DOI: [10.1121/1.3576192](https://doi.org/10.1121/1.3576192)

> Annotation: Practitioner case study of a space with exceptionally long RT60. Context entry useful for anchoring intuition about what multi-second reverberation actually sounds like in a physical space. It helps translate abstract values such as 6, 10, or more seconds of decay into practical consequences for speech, music, crowd noise, and treatment strategy, which is useful when documenting verbx's extreme-tail and infinite-style modes.

**[RA20]** Clow, S (2023). Sound Insulation Testing - Sensitivity to Reverberation Time: T30 v T20. *Autumn Conference Acoustics 2003*. DOI: [10.25144/18162](https://doi.org/10.25144/18162)

> Context entry. Compares T20 and T30 evaluation windows for building acoustics measurements. Peripheral to reverb synthesis but relevant to RT60 measurement methodology. It supports exposing multiple decay-window estimates in analysis because different windows can disagree when the noise floor, curvature, or early decay behavior contaminates a simple full-tail fit.

**[RA15]** Diaz, Cesar; Pedrero, Antonio (2005). The reverberation time of furnished rooms in dwellings. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2004.12.002](https://doi.org/10.1016/j.apacoust.2004.12.002)

> Annotation: Measurement study of RT60 in typical furnished domestic rooms. Useful reference for the short-RT60 end of verbx's operating range. This anchors small-room presets in measured domestic conditions rather than only studio or concert-hall ideals, and it is helpful for explaining why living-room reverberation usually needs early-reflection realism and spectral coloration more than a long, obvious tail.

**[RA18]** Dunham, J (2023). Comparing Measured Sound Strength to Theory as a Function of Reverberation Time and Room Volume. *Auditorium Acoustics 2023*. DOI: [10.25144/15977](https://doi.org/10.25144/15977)

> Context entry. Empirical validation of the relationship between room volume, RT60, and sound strength. Background context. This is relevant to gain staging and apparent distance because a longer or larger room is not only a longer tail; it also changes perceived loudness, source support, and the balance between direct and reverberant energy.

**[RA9]** Fay, Michael W. (2019). Reverberation time slope ratio. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.5136901](https://doi.org/10.1121/1.5136901)

> Annotation: Proposes the slope ratio as a diagnostic for non-exponential decay curves, which arise whenever a room has coupled sub-spaces or highly non-uniform absorption. Relevant to detecting cases where a single RT60 number is insufficient. This is useful for future analysis reports because it suggests a compact way to flag curved or multi-stage decay behavior instead of silently fitting one straight line to a tail that does not behave exponentially.

**[RA12]** Granzotto, Nicola; Caniato, Marco (2023). Influence of not homogeneous absorption on reverberation time. *2023 Immersive and 3D Audio: from Architecture to Automotive (I3DA)*. DOI: [10.1109/i3da57090.2023.10289246](https://doi.org/10.1109/i3da57090.2023.10289246)

> Annotation: Quantifies the error in RT60 prediction when absorption is spatially non-uniform. This is the regime where Sabine and Eyring both fail. Relevant to understanding the boundaries of the physical models verbx targets, especially for rooms with heavy treatment on one surface, audience absorption, curtains, or irregular furnishing. It motivates warnings around overconfident physical estimates when absorption cannot reasonably be treated as evenly distributed.

**[RA13]** Jones, Douglas R. (2011). Reverberation and Time Response. *Sound of Worship*. DOI: [10.1016/b978-0-240-81339-4.00021-1](https://doi.org/10.1016/b978-0-240-81339-4.00021-1)

> Annotation: Applied treatment of reverberation in large worship spaces. Context entry for understanding design targets at extreme RT60 values that verbx is built to reproduce. It provides a real architectural setting for long-decay presets, where the goal is not merely "more tail" but a musically and verbally usable balance among warmth, clarity, spaciousness, and intelligibility.

<a id="entry-ra2"></a>
**[RA2]** Kuttruff, Heinrich; Vorländer, Michael (2024). Reverberation and steady-state energy density. *Room Acoustics*. DOI: [10.1201/9781003389873-5](https://doi.org/10.1201/9781003389873-5)

> Annotation: Current edition textbook chapter on statistical room acoustics theory. The definitive modern treatment of Sabine and Eyring theory, with careful attention to the diffuse-field assumption and where it fails. This is the main reference for deciding when verbx should present a physical room estimate confidently and when it should warn that geometry, absorption distribution, or source/listener placement makes the model only approximate.

**[RA11]** Nowoswiat, Artur; Olechowska, Marcelina (2016). Investigation Studies on the Application of Reverberation Time. *Archives of Acoustics*. DOI: [10.1515/aoa-2016-0002](https://doi.org/10.1515/aoa-2016-0002)

> Annotation: Empirical study comparing different RT60 measurement protocols and their agreement across room types. Useful calibration context. It supports treating RT60 as a measured quantity with method-dependent uncertainty, which matters for verbx when comparing T20, T30, EDT, and fitted full-tail estimates in automated analysis output.

**[RA7]** Pan, Jie (2004). Invited Review Paper: The Physics of Reverberation. *Building Acoustics*. DOI: [10.1260/1351010041494746](https://doi.org/10.1260/1351010041494746)

> Annotation: A clear physics-first treatment of how reverberant fields form, covering both the modal and geometric views. Useful for developers who want to understand why the simple RT60 model is an approximation. It is a good conceptual anchor for the split between early reflections, modal coloration, and late diffuse decay, which maps directly onto verbx controls such as early level, density, modulation, and tail shaping.

**[RA10]** Prato, Andrea; Casassa, Federico; Schiavi, Alessandro (2016). Reverberation time measurements in non-diffuse acoustic field by the modal reverberation time. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2016.03.041](https://doi.org/10.1016/j.apacoust.2016.03.041)

> Annotation: Addresses RT60 measurement in non-diffuse fields by separating individual modal contributions. Relevant context for understanding why blind estimation algorithms struggle in small rooms. The modal framing also helps explain low-frequency artifacts in synthesized or measured IRs: a short room can have individual resonances that dominate perceived decay even when the broadband RT60 looks plausible.

**[RA16]** Schnitta, Bonnie (2013). Achieving optimal reverberation time in a room, using newly patented tuning tubes. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4801071](https://doi.org/10.1121/1.4801071)

> Context entry. Describes physical treatment of reverberation time; tangential to algorithmic approaches. It remains useful as a reminder that algorithmic controls model only part of a larger acoustic design problem: real spaces use absorbers, resonators, diffusers, and tunable devices to change decay behavior in ways that can be uneven across frequency and position.

**[RA17]** Stephenson, UM (2023). On the Influence of Ceiling and Audience Profile on the Reverberation Time and Other Room Acoustical Parameters. *Auditorium Acoustics 2008*. DOI: [10.25144/17526](https://doi.org/10.25144/17526)

> Context entry. Geometric acoustics study of audience and ceiling geometry effects on RT60. Background context for physical room modeling. It reinforces that audience layout and architectural geometry can change decay and clarity in ways that are not captured by volume and average absorption alone, which matters for any future room-parameter import or venue-style preset work.

**[RA8]** Tohyama, Mikio (2011). Reverberation Sound in Rooms. *Signals and Communication Technology*. DOI: [10.1007/978-3-642-20122-6_11](https://doi.org/10.1007/978-3-642-20122-6_11)

> Annotation: Signal-processing perspective on room reverberation, bridging acoustic theory and DSP formulations. Useful for practitioners who are more comfortable with z-domain analysis than with acoustic physics. This helps connect room-acoustic quantities to filters, feedback structures, and impulse-response manipulation, making it a natural reference for the implementation side of algorithmic reverb and dereverberation.

<a id="entry-ra3"></a>
**[RA3]** Unknown authors (2002). Reverberation and steady state energy density. *Room Acoustics*. DOI: [10.1201/9781482286632-9](https://doi.org/10.1201/9781482286632-9)

> Annotation: Earlier edition of the Kuttruff treatment. Included because some derivations presented here are clearer than in the updated edition. It is useful as a companion reading source for implementation notes because the older presentation makes several intermediate assumptions more explicit, which helps when translating textbook acoustics into deterministic code paths and test fixtures.

<a id="entry-80"></a>
**[RA1]** Unknown authors (2008). Room Impulse Responses, Decay Curves and Reverberation Times. *Formulas of Acoustics*. DOI: [10.1007/978-3-540-76833-3_262](https://doi.org/10.1007/978-3-540-76833-3_262)

> Annotation: The primary reference for the EDT, C80, and D50 definitions given in the Key Results Reference table above. The formulas here are the ones used in verbx's metric computation layer. It also provides the bridge between an impulse response as raw samples and an energy decay curve as an interpretable acoustic object, which is why it is the practical starting point for validating analysis JSON, preset reports, and user-facing room clarity summaries.

<a id="entry-ra5"></a>
**[RA5]** Unknown authors (2010). Deriving the Reverberation Time Equation. *Acoustics and Psychoacoustics*. DOI: [10.1016/b978-0-240-52175-6.00014-4](https://doi.org/10.1016/b978-0-240-52175-6.00014-4)

> Annotation: Companion derivation to [RA4], covering the wideband case. Read before [RA4]. It gives the simplest path from room volume and absorption to a single reverberation-time estimate, making it a useful baseline for docs, examples, and tests before adding frequency-dependent materials or per-band correction curves.

**[RA4]** Unknown authors (2010). Deriving the Reverberation Time Equation for Different Frequencies and Surfaces. *Acoustics and Psychoacoustics*. DOI: [10.1016/b978-0-240-52175-6.00015-6](https://doi.org/10.1016/b978-0-240-52175-6.00015-6)

> Annotation: Pedagogical derivation of frequency-dependent RT60, showing explicitly how absorption coefficients vary with frequency and how this produces frequency-dependent decay. The basis for verbx's per-band RT60 calibration. This reference is especially important for explaining why low, mid, and high bands should not be forced to share the same decay target, and why a preset can be technically correct only if its spectral tail shape matches the intended room material behavior.

**[RA19]** Walker, R (2024). Microprocessor Controlled Equipment for the Measurement of Reverberation Time. *Acoustics '82*. DOI: [10.25144/23148](https://doi.org/10.25144/23148)

> Context entry. Historical document on early automated RT60 measurement hardware. Included for completeness. It gives useful historical perspective on why reverberation-time measurement became standardized around repeatable excitation, decay capture, and fitted windows, which parallels verbx's emphasis on deterministic analysis output rather than subjective labeling alone.

**[RA6]** Xiang, Ning (2020). Generalization of Sabine's reverberation theory. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0001806](https://doi.org/10.1121/10.0001806)

> Annotation: Extends the Sabine framework to non-diffuse rooms using statistical energy analysis. Useful for understanding where the simple T60 prediction fails and what a more complete physical model needs to account for. For verbx, this supports the design choice to expose diagnostics rather than just a number: non-diffuse behavior can make the decay curve bend, split, or depend strongly on source and receiver position.

---

## Section 5: Spatial Audio and Ambisonics

**[SP2]** Betlehem, T.; Poletti, M.A. (2009). Sound field reproduction around a scatterer in reverberation. *2009 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2009.4959527](https://doi.org/10.1109/icassp.2009.4959527)

> Annotation: Addresses the problem of reproducing a reverberant sound field in the presence of a scattering object. Relevant context for spatial rendering of reverb in immersive systems. It highlights that the listener, playback setup, and objects in the field can alter the reproduced reverberant image, which is important background for Ambisonic, binaural, or multichannel render modes.

**[SP1]** Corey, J (2010). Spatial Attributes and Reverberation. *Audio Production and Critical Listening*. DOI: [10.1016/b978-0-240-81295-3.00003-4](https://doi.org/10.1016/b978-0-240-81295-3.00003-4)

> Annotation: Textbook chapter on how reverberation interacts with spatial audio perception, covering apparent source width, listener envelopment, and their relationship to early vs. late energy. Grounding for verbx's spatial output design. It is especially useful for separating "longer" from "larger": a reverb can increase depth, width, and immersion without simply increasing RT60, so spatial presets need independent control over early reflections, late-field decorrelation, and wet image spread.

**[SP3]** Ren Gang; Shivaswamy, S. H.; Roessner, S.; et al. (2013). An additional "Depth" of reverberation helps content stand out: Media content emphasis using audio reverberation effect. *2013 IEEE International Conference on Consumer Electronics (ICCE)*. DOI: [10.1109/icce.2013.6486811](https://doi.org/10.1109/icce.2013.6486811)

> Annotation: Explores the use of selective reverberation as a production tool for creating depth and emphasis in consumer audio. Useful framing for understanding how reverb functions as a mixing tool rather than purely a simulation. For verbx documentation, this supports creative features such as ducking, depth automation, and source-specific wetness because those controls are production decisions as much as acoustic-model decisions.

**[SP4]** Zhan Haoke; Cai Zhiming; Yuan Bingcheng (2008). Space-time ODN suppressing reverberation method. *2008 9th International Conference on Signal Processing*. DOI: [10.1109/icosp.2008.4697671](https://doi.org/10.1109/icosp.2008.4697671)

> Context entry. Dereverberation from a spatial signal processing perspective. Included for conceptual contrast with synthesis approaches. It gives a useful inverse-problem counterpoint to verbx's render path: the same spatial cues that help a reverb feel enveloping can be exploited or suppressed when the goal is clarity, separation, or reduced late energy.

---

## Section 6: Perceptual Studies

How humans perceive reverberation: thresholds of detection, preference curves, and the perceptual dimensions of reverberant spaces. These papers inform which parameters verbx needs to control precisely and which it can treat as second-order. They are also the references behind documentation language that distinguishes measured correctness from perceptual usefulness, since a numerically accurate RT60 does not always guarantee the listener hears the intended space.

**[PE6]** Blasinski, Lukasz; Felcyn, Jan; Kocinski, Jedrzej (2024). Perception of reverberation length in rooms with reverberation enhancement systems. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0026485](https://doi.org/10.1121/10.0026485)

> Annotation: Studies the just-noticeable difference for RT60 changes in rooms with active enhancement systems. Directly relevant to calibrating the resolution of verbx's RT60 control. It helps determine sensible CLI step sizes, UI increments, and preset spacing so the tool does not imply precision that listeners cannot reliably perceive, while still preserving fine control for measurement and regression testing.

**[PE9]** Erkelens, Jan S.; Heusdens, Richard (2011). A statistical room impulse response model with frequency dependent reverberation time for single-microphone late reverberation suppression. *Interspeech 2011*. DOI: [10.21437/interspeech.2011-82](https://doi.org/10.21437/interspeech.2011-82)

> Annotation: Derives a statistical model of the RIR late field that explicitly captures frequency-dependent decay. The model's assumptions are directly relevant to how verbx represents the decay filter bank. It is also a useful cross-reference between perceptual and statistical modeling: the late field can be compactly parameterized, but the usefulness of that parameterization depends on how well it predicts audible late reverberation.

**[PE2]** Javed, Hamza A.; Naylor, Patrick A. (2015). An extended reverberation decay tail metric as a measure of perceived late reverberation. *2015 23rd European Signal Processing Conference (EUSIPCO)*. DOI: [10.1109/eusipco.2015.7362546](https://doi.org/10.1109/eusipco.2015.7362546)

> Annotation: Proposes a metric that correlates more closely with the perceived extent of late reverberation than RT60 alone. Useful for evaluating whether a verbx preset sounds as long as its nominal RT60 suggests it should. This matters for long-tail and infinite-style modes, where the perceptual impression of lingering energy can diverge from the fitted decay time because spectral balance, masking, and late-field texture all influence audibility.

**[PE7]** Ninounakis, TJ; Davies, WJ (2024). The Perception of Small Changes in Reverberation Time within Recording Studio Control Rooms. *Reproduced Sound 1998*. DOI: [10.25144/18947](https://doi.org/10.25144/18947)

> Annotation: Measures perceptual thresholds for RT60 change in a professional monitoring environment. Complements [PE6] with studio-context data. This is valuable because control-room listening emphasizes small changes, translation, and repeatable monitoring conditions, so it gives a stricter perceptual reference than casual playback or large-room listening.

**[PE8]** Strother, TEN; Davies, WJ (2023). Is There a Relationship Between Pitch Perception and Individual Preference of Concert Hall Reverberation Time?. *Acoustics 2023*. DOI: [10.25144/16625](https://doi.org/10.25144/16625)

> Context entry. Investigates individual differences in preferred RT60. Peripheral to algorithm design; relevant to preset design philosophy. It supports keeping verbx presets adjustable rather than presenting one "correct" concert-hall or studio value, because preference can depend on listener, source material, pitch range, musical style, and listening task.

**[PE4]** Tsilfidis, Alexandros; Mourjopoulos, John (2009). Perceptually-motivated selective suppression of late reverberation. *2009 16th International Conference on Digital Signal Processing*. DOI: [10.1109/icdsp.2009.5201165](https://doi.org/10.1109/icdsp.2009.5201165)

> Annotation: Earlier version of the work above; contains some derivations that the 2011 paper abbreviates. It is useful as a supplementary implementation reference because conference versions often expose the signal model and algorithmic tradeoffs more directly, making it easier to compare perceptual late-tail suppression with verbx's deterministic render and realtime controls.

**[PE3]** Tsilfidis, Alexandros; Mourjopoulos, John (2011). Blind single-channel suppression of late reverberation based on perceptual reverberation modeling. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.3533690](https://doi.org/10.1121/1.3533690)

> Annotation: Develops a perceptual model of late reverberation and uses it to drive a suppression algorithm. The perceptual model itself, especially how the ear weights late energy, is the relevant contribution for verbx's design. It helps frame dereverberation and reverb reduction as selective control of perceptually harmful energy rather than blanket tail removal, which is important for low-latency real-time behavior.

**[PE5]** Watkins, Anthony J. (2005). Listening in real-room reverberation: Effects of extrinsic context. *Auditory Signal Processing*. DOI: [10.1007/0-387-27045-0_52](https://doi.org/10.1007/0-387-27045-0_52)

> Annotation: Psychoacoustic study on how listeners adapt to reverberation over time and how prior context affects the perception of a reverberant signal. Explains why very long reverb tails can become perceptually transparent under sustained exposure. This is important for preset evaluation because an effect that sounds dramatic in isolation may feel natural after adaptation, while a subtle change may be obvious in an A/B comparison.

**[PE1]** Zahorik, Pavel (2004). Perceptual scaling of room reverberation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4784535](https://doi.org/10.1121/1.4784535)

> Annotation: Maps the perceptual space of reverberation using multidimensional scaling, identifying the primary perceptual dimensions, roughly decay time and reverberance character. Fundamental to understanding which physical parameters predict subjective quality. It is a key reason verbx should expose grouped controls and presets around perceptual outcomes such as clarity, bloom, distance, and room size rather than only offering raw acoustic variables.

---

## Section 7: Machine Learning and Blind Estimation

CRNN-based RT60 estimation, neural dereverberation, and related machine learning approaches. Relevant to verbx's analysis pipeline, especially future work around blind room inference, automatic preset suggestion, dereverb assistance, and confidence scoring. These entries also provide baseline methods that can be compared against deterministic estimators before verbx grows heavier neural dependencies.

**[ML6]** Ai, Yang; Wang, Xin; Yamagishi, Junichi; et al. (2020). Reverberation Modeling for Source-Filter-Based Neural Vocoder. *Interspeech 2020*. DOI: [10.21437/interspeech.2020-1613](https://doi.org/10.21437/interspeech.2020-1613)

> Annotation: Embeds a reverberation model inside a neural vocoder. Demonstrates how differentiable reverb models can be combined with neural speech synthesis. Relevant to any future verbx neural extensions. It is not part of the current deterministic core, but it frames a possible future where reverb parameters remain interpretable while being optimized or predicted inside a learned speech pipeline.

**[ML8]** Andrijasevic, Andrea (2020). Effect of phoneme variations on blind reverberation time estimation. *Acta Acustica*. DOI: [10.1051/aacus/2020001](https://doi.org/10.1051/aacus/2020001)

> Annotation: Examines how the phonetic content of speech affects the accuracy of blind RT60 estimators. Useful for understanding failure modes and the range of conditions over which blind estimators can be trusted. It supports surfacing caveats in analysis JSON, since voiced vowels, fricatives, pauses, and transient-rich material may provide very different evidence for decay fitting.

**[ML5]** Choi, Yeonjong; Xie, Chao; Toda, Tomoki (2023). Reverberation-Controllable Voice Conversion Using Reverberation Time Estimator. *INTERSPEECH 2023*. DOI: [10.21437/interspeech.2023-1356](https://doi.org/10.21437/interspeech.2023-1356)

> Annotation: Uses an RT60 estimator as a conditioning signal inside a voice conversion network to control the reverberation of synthesized speech. Demonstrates an end-to-end neural approach to RT60-conditioned synthesis that is architecturally adjacent to verbx's generative modes. The key design lesson is that estimated acoustics can become controllable latent variables, opening a path from analysis to automatic matching, transfer, or preset generation.

**[ML1]** Deng, Shuwen; Mack, Wolfgang; Habets, Emanuel A.P. (2020). Online Blind Reverberation Time Estimation Using CRNNs. *Interspeech 2020*. DOI: [10.21437/interspeech.2020-2156](https://doi.org/10.21437/interspeech.2020-2156)

> Annotation: Demonstrates that a convolutional-recurrent neural network can estimate RT60 reliably from a single speech channel in real time. Sets the state of the art for the blind estimation problem that verbx's analysis mode needs to solve. It is a useful benchmark for future low-latency analysis because it treats estimation as an online problem, where window size, model state, and update cadence all affect end-to-end responsiveness.

**[ML7]** Kodrasi, Ina; Bourlard, Herve (2018). Single-channel Late Reverberation Power Spectral Density Estimation Using Denoising Autoencoders. *Interspeech 2018*. DOI: [10.21437/interspeech.2018-1660](https://doi.org/10.21437/interspeech.2018-1660)

> Annotation: Uses a denoising autoencoder to estimate the power spectral density of the late reverberant field. The estimated late-field PSD is then usable for suppression or analysis. Methodologically relevant to verbx's analysis pipeline because late-field energy is exactly the component that must be separated from direct sound, noise, and early reflections when reporting dereverb strength or residual tail.

**[ML4]** Lollmann, Heinrich W.; Brendel, Andreas; Kellermann, Walter (2018). Efficient ML-Estimator for Blind Reverberation Time Estimation. *2018 26th European Signal Processing Conference (EUSIPCO)*. DOI: [10.23919/eusipco.2018.8553001](https://doi.org/10.23919/eusipco.2018.8553001)

> Annotation: Maximum-likelihood formulation for blind RT60 estimation with a focus on computational efficiency. A strong non-neural baseline to compare CRNN approaches against. This is important for the 0.7.x stabilization track because deterministic, lightweight estimators are easier to ship, test, and run in CLI or realtime contexts than model-backed systems with larger dependencies.

**[ML11]** Ma, Changxue; Shi, Guangji (2012). Reverberation time estimation based on multidelay acoustic echo cancellation. *2012 International Conference on Audio, Language and Image Processing*. DOI: [10.1109/icalip.2012.6376617](https://doi.org/10.1109/icalip.2012.6376617)

> Context entry. Exploits an existing echo cancellation framework to derive RT60 estimates. Alternative estimation strategy for completeness. It is useful as a reminder that reverberation estimates can emerge from adjacent real-time systems, not only from offline IR analysis, which may matter if verbx later integrates with capture, monitoring, or conferencing workflows.

**[ML2]** Mack, Wolfgang; Deng, Shuwen; Habets, Emanuel A.P. (2020). Single-Channel Blind Direct-to-Reverberation Ratio Estimation Using Masking. *Interspeech 2020*. DOI: [10.21437/interspeech.2020-2171](https://doi.org/10.21437/interspeech.2020-2171)

> Annotation: Companion paper to [ML1], estimating the direct-to-reverberant ratio rather than RT60. DRR is the perceptually critical parameter for how "wet" a signal sounds, distinct from how long the tail is. For verbx, this points toward richer analysis output: RT60 can describe decay duration, but DRR and early-to-late balance describe distance, clarity, and apparent source presence.

**[ML13]** Malik, Hafiz; Farid, Hany (2010). Audio forensics from acoustic reverberation. *2010 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2010.5495479](https://doi.org/10.1109/icassp.2010.5495479)

> Context entry. Forensic identification of recording environments from reverberation. Companion study to [ML12]. This supports the broader idea that reverberation carries scene information, so analysis features in verbx should preserve details useful for matching, comparison, and provenance rather than collapsing everything into a single wet/dry description.

**[ML12]** Malik, Hafiz; Zhao, Hong (2012). Recording environment identification using acoustic reverberation. *2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2012.6288258](https://doi.org/10.1109/icassp.2012.6288258)

> Context entry. Uses reverberation signatures to fingerprint recording environments. Tangential but provides perspective on what makes each room's impulse response distinctive. It helps explain why room matching is more than matching RT60: spectral coloration, early reflection pattern, and decay structure can identify a space even when broad statistics look similar.

**[ML10]** Prodeus, Arkadiy (2015). Parameter optimization of the single channel late reverberation suppression technique. *2015 IEEE 35th International Conference on Electronics and Nanotechnology (ELNANO)*. DOI: [10.1109/elnano.2015.7146889](https://doi.org/10.1109/elnano.2015.7146889)

> Context entry. Optimization of suppression filter parameters in a statistical dereverberation framework. Earlier work by the same author as [ML9]. It provides background for tuning dereverb options such as reduction amount, smoothing, and artifact control, where overly aggressive settings may improve measured tail reduction while hurting speech naturalness.

**[ML9]** Prodeus, Arkadiy (2017). Late reverberation reduction and blind reverberation time measurement for automatic speech recognition. *2017 IEEE First Ukraine Conference on Electrical and Computer Engineering (UKRCON)*. DOI: [10.1109/ukrcon.2017.8100319](https://doi.org/10.1109/ukrcon.2017.8100319)

> Context entry. Combines blind RT60 estimation with late-reverb suppression for ASR. Context for understanding the estimation-then-suppression pipeline architecture. It is relevant to verbx's real-time dereverberation path because it treats measurement and processing as a coupled loop: estimation quality affects suppression strength, and suppression artifacts affect downstream speech recognition or intelligibility.

**[ML3]** Xiong, Feifei; Goetze, Stefan; Kollmeier, Birger; et al. (2019). Joint Estimation of Reverberation Time and Early-To-Late Reverberation Ratio From Single-Channel Speech Signals. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*. DOI: [10.1109/taslp.2018.2877894](https://doi.org/10.1109/taslp.2018.2877894)

> Annotation: Estimates RT60 and early-to-late ratio simultaneously from a single microphone. The joint approach avoids the error propagation that plagues sequential estimators and is worth evaluating for verbx's blind analysis mode. It also suggests a product-level direction: when verbx reports room character, it should estimate related acoustic quantities together and expose confidence or consistency checks instead of presenting each metric as isolated truth.

**[ML14]** Zheng, Chenxi; Chan, Wai-Yip (2013). Late reverberation suppression using MMSE modulation spectral estimation. *Interspeech 2013*. DOI: [10.21437/interspeech.2013-718](https://doi.org/10.21437/interspeech.2013-718)

> Context entry. MMSE estimator operating in the modulation spectrum domain for late reverb suppression. Useful for understanding the modulation-domain perspective on reverberation. It adds another lens for dereverberation design because late reverb smears temporal modulation, so suppression can be evaluated not only by decay curves but also by how well it restores speech envelope contrast.

---

## Section 8: Speech and Room Acoustics

Papers on speech intelligibility in reverberant environments, dereverberation for ASR, and vocal effects in reverberant spaces. These are more peripherally related to verbx's core synthesis and analysis functions, but they matter for judging whether a reverb is useful, harmful, or stylistically appropriate. They support future intelligibility-aware modes, speech-oriented presets, QA metrics, and dereverb settings that prioritize clarity instead of only reducing measured tail energy.

**[SR8]** Arai, Takayuki (2006). Preprocessing speech against reverberation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4781210](https://doi.org/10.1121/1.4781210)

> Context entry. Signal processing approaches to pre-filtering speech before transmission through reverberant environments. Inverse problem perspective on the synthesis work. It is conceptually useful for verbx because adding reverb and preparing speech to survive reverb are mirror-image tasks, both concerned with how temporal envelopes and spectral cues survive a room response.

**[SR13]** Barnett, P (2024). Study of Word Score Test Results to Determine the Robust Components of Speech Subject to Noise and Reverberation. *Reproduced Sound 1998*. DOI: [10.25144/18937](https://doi.org/10.25144/18937)

> Context entry. Word intelligibility scoring under combined noise and reverberation. Included for completeness. It is relevant to future test design because real-world speech problems often involve both noise and late reflection, so an evaluation corpus that varies only reverb may miss important interaction effects.

**[SR14]** Bottalico, Pasquale; Graetzer, Simone; Hunter, Eric (2016). Effect of reverberation time on vocal fatigue. *Speech Prosody 2016*. DOI: [10.21437/speechprosody.2016-101](https://doi.org/10.21437/speechprosody.2016-101)

> Context entry. Studies how room RT60 affects talker vocal effort and fatigue. Peripheral to reverb synthesis; human factors context. It reminds us that room acoustics affect performers and speakers as well as listeners; for simulated environments, this can matter when designing presets for narration, voiceover, classroom, worship, or rehearsal contexts.

**[SR19]** Choi, Jae; Kim, Jeunghun; Kang, Shin Jae; et al. (2015). Reverberation-robust acoustic indoor localization. *Interspeech 2015*. DOI: [10.21437/interspeech.2015-678](https://doi.org/10.21437/interspeech.2015-678)

> Context entry. Room localization from reverberant acoustic signals. Tangential to synthesis; relevant to analysis. It reinforces that reverberation contains spatial and environmental information, so analysis tools can potentially infer room properties, device position, or capture context when enough cues survive in the signal.

**[SR1]** Habets, Emanuel A. P. (2010). Speech Dereverberation Using Statistical Reverberation Models. *Signals and Communication Technology*. DOI: [10.1007/978-1-84996-056-4_3](https://doi.org/10.1007/978-1-84996-056-4_3)

> Annotation: Comprehensive chapter on statistical approaches to single-channel dereverberation. The statistical models of the late reverberant field described here inform verbx's internal representations of the reverberant tail. It is the central speech-dereverb reference in this bibliography because it connects room impulse response assumptions, late-tail statistics, suppression filters, and the practical artifacts that appear when trying to remove reverberation from a single channel.

**[SR11]** Hodoshima, Nao (2019). Effects of Urgent Speech and Congruent/Incongruent Text on Speech Intelligibility in Noise and Reverberation. *Interspeech 2019*. DOI: [10.21437/interspeech.2019-1902](https://doi.org/10.21437/interspeech.2019-1902)

> Context entry. Cognitive factors in speech intelligibility under reverberation. Peripheral to DSP design. It broadens the interpretation of "clarity" beyond acoustics alone, showing that urgency, text congruence, attention, and linguistic context can interact with room effects in ways that purely signal-level metrics may miss.

**[SR10]** Hodoshima, Nao (2024). Effects of talker and playback rate of reverberation-induced speech on speech intelligibility of older adults. *Interspeech 2024*. DOI: [10.21437/interspeech.2024-1721](https://doi.org/10.21437/interspeech.2024-1721)

> Context entry. Speech intelligibility study focusing on older listeners in reverberant conditions. Peripheral to DSP design; population-specific psychoacoustic data. It is useful for accessibility-oriented thinking because reverberation that feels acceptable to one listener group may disproportionately reduce intelligibility for another, especially when playback rate and talker characteristics vary.

**[SR12]** Hodoshima, Nao; Arai, Takayuki; Kurisu, Kiyohiro (2012). Intelligibility of speech spoken in noise/reverberation for older adults in reverberant environments. *Interspeech 2012*. DOI: [10.21437/interspeech.2012-415](https://doi.org/10.21437/interspeech.2012-415)

> Context entry. Earlier Hodoshima et al. study on older listener intelligibility. Part of a research series. It helps establish continuity across population-specific speech studies and supports documenting that dereverb or clarity presets may be especially valuable for accessibility, education, and public-address style material.

**[SR6]** Kocinski, Jedrzej; Ozimek, Edward (2016). Speech Recognition in an Enclosure with a Long Reverberation Time. *Archives of Acoustics*. DOI: [10.1515/aoa-2016-0025](https://doi.org/10.1515/aoa-2016-0025)

> Context entry. Empirical study of intelligibility at long RT60 values. Relevant to understanding perceptual consequences of the extreme reverberation times verbx targets. It provides speech-focused grounding for why enormous tails should often be paired with ducking, predelay, spectral shaping, or clarity-preserving options when the source is spoken word rather than sustained music.

**[SR5]** Leclere, Thibaud; Lavandier, Mathieu; Culling, John F. (2015). Speech intelligibility prediction in reverberation: Towards an integrated model of speech transmission, spatial unmasking, and binaural de-reverberation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4921028](https://doi.org/10.1121/1.4921028)

> Context entry. Integrated model of speech intelligibility in reverberation incorporating binaural effects. Background context for understanding the perceptual impact of long RT60 on speech. It shows that spatial hearing and binaural unmasking can change intelligibility outcomes, which matters when comparing mono, stereo, binaural, and multichannel reverberation or dereverberation modes.

**[SR22]** Li, Xuan; Ma, Xiaochuan; Hou, Chaohuan (2008). Broadband DOA Estimation by Beamforming with Suppression of Reverberation. *2008 Congress on Image and Signal Processing*. DOI: [10.1109/cisp.2008.550](https://doi.org/10.1109/cisp.2008.550)

> Context entry. Broadband DOA estimation compensating for reverberation. Spatial processing context. It supports the broader analysis roadmap around multichannel and spatial features, where reverberation is not just a nuisance but a structured part of the observed sound field.

**[SR4]** Nishiura, Takanobu; Fukumori, Takahiro (2011). Suitable Reverberation Criteria for Distant-talking Speech Recognition. *Speech Technologies*. DOI: [10.5772/18937](https://doi.org/10.5772/18937)

> Context entry. Identifies which room acoustics metrics best predict ASR accuracy. Establishes C80 and DRR as more predictive than RT60 alone for intelligibility in ASR contexts. This is a strong reason for verbx analysis reports to include clarity and direct/reverberant metrics alongside RT60, especially when users are processing dialogue, conferencing audio, lectures, or datasets for recognition systems.

**[SR20]** Pang, Cheng; Zhang, Jie; Liu, Hong (2015). Direction of arrival estimation based on reverberation weighting and noise error estimator. *Interspeech 2015*. DOI: [10.21437/interspeech.2015-681](https://doi.org/10.21437/interspeech.2015-681)

> Context entry. DOA estimation using reverberation structure. Spatial analysis context; peripheral to verbx. It is useful background for multichannel analysis because direction estimates can be biased or stabilized by reverberant energy depending on the method, array geometry, and late-field assumptions.

**[SR17]** Peddinti, Vijayaditya; Chen, Guoguo; Povey, Daniel; et al. (2015). Reverberation robust acoustic modeling using i-vectors with time delay neural networks. *Interspeech 2015*. DOI: [10.21437/interspeech.2015-527](https://doi.org/10.21437/interspeech.2015-527)

> Context entry. ASR robustness to reverberation using TDNN-based acoustic models. Neural ASR context; tangential to synthesis. It supports data-generation and benchmarking use cases where verbx could produce controlled reverberant examples for testing whether recognition models remain stable across room conditions.

**[SR23]** Peng, Jianxin (2015). Chinese Syllable and Phoneme Identification in Noise and Reverberation. *Archives of Acoustics*. DOI: [10.2478/aoa-2014-0052](https://doi.org/10.2478/aoa-2014-0052)

> Context entry. Language-specific phoneme intelligibility under reverberation. Peripheral to DSP design. It is included because reverberation can affect phoneme classes differently, and language-specific corpora may reveal clarity failures that broad acoustic metrics or English-only tests do not capture.

**[SR16]** Petkov, Petko N.; Braunschweiler, Norbert; Stylianou, Yannis (2016). Automated Pause Insertion for Improved Intelligibility Under Reverberation. *Interspeech 2016*. DOI: [10.21437/interspeech.2016-960](https://doi.org/10.21437/interspeech.2016-960)

> Context entry. Modifies speech rhythm to improve intelligibility in reverberant conditions. Peripheral to synthesis. It is useful for product thinking because not every reverb-related clarity problem must be solved by tail suppression; spacing, pauses, articulation, and source preprocessing can also improve intelligibility in reverberant playback.

**[SR15]** Petkov, Petko N.; Stylianou, Yannis (2016). Generalizing Steady State Suppression for Enhanced Intelligibility Under Reverberation. *Interspeech 2016*. DOI: [10.21437/interspeech.2016-1026](https://doi.org/10.21437/interspeech.2016-1026)

> Context entry. Signal processing method for intelligibility enhancement in reverberant conditions. Included for completeness. It provides another example of improving intelligibility without necessarily reconstructing a perfectly dry signal, which is an important distinction for practical dereverberation controls that trade naturalness against clarity.

**[SR18]** Raut, Chandra Kant; Nishimoto, Takuya; Sagayama, Shigeki (2005). Model adaptation by state splitting of HMM for long reverberation. *Interspeech 2005*. DOI: [10.21437/interspeech.2005-157](https://doi.org/10.21437/interspeech.2005-157)

> Context entry. HMM adaptation for reverberant ASR. Historical ASR context. It shows the long-running nature of the reverberant speech problem and gives older-model context for why modern neural systems still benefit from controlled room simulation and reverberation-aware training data.

**[SR7]** Rennies, Jan; Brand, Thomas; Kollmeier, Birger (2011). Prediction of binaural intelligibility level differences in reverberation. *Interspeech 2011*. DOI: [10.21437/interspeech.2011-10](https://doi.org/10.21437/interspeech.2011-10)

> Context entry. Binaural intelligibility model predictions in reverberant conditions. Background reference for spatial dereverberation contexts. It supports the idea that a spatial reverb or dereverb mode should preserve useful interaural cues where possible, because apparent clarity can depend on binaural separation and not only on mono energy decay.

**[SR2]** Saijo, Kohei; Wichern, Gordon; Germain, Francois G.; et al. (2024). Enhanced Reverberation as Supervision for Unsupervised Speech Separation. *Interspeech 2024*. DOI: [10.21437/interspeech.2024-1241](https://doi.org/10.21437/interspeech.2024-1241)

> Context entry. Uses added reverberation as a data augmentation strategy for training speech separation models. Relevant for understanding how verbx might be used as a data synthesis tool. It supports a secondary use case where verbx is not only an effects processor but also a controllable corpus generator for robust speech, separation, or ASR experiments.

**[SR21]** Saric, Zoran; Jovicic, Slobodan (2003). Adaptive beamforming in room with reverberation. *8th European Conference on Speech Communication and Technology (Eurospeech 2003)*. DOI: [10.21437/eurospeech.2003-218](https://doi.org/10.21437/eurospeech.2003-218)

> Context entry. Beamforming in reverberant rooms. Spatial processing context. It provides a classic multichannel counterpoint to single-channel dereverberation, reminding readers that spatial filtering can reject some reverberant energy but is constrained by room reflections, array aperture, and source movement.

**[SR3]** Sehr, Armin; Kellermann, Walter (2008). New Results for Feature-Domain Reverberation Modeling. *2008 Hands-Free Speech Communication and Microphone Arrays*. DOI: [10.1109/hscma.2008.4538713](https://doi.org/10.1109/hscma.2008.4538713)

> Context entry. Feature-domain reverberation model for robust ASR. Peripheral to synthesis but relevant to understanding how reverb distorts feature representations. It helps explain why dereverberation quality cannot be judged only by waveform similarity: downstream feature spaces may amplify or ignore different kinds of smearing, coloration, and late-energy bias.

**[SR9]** Tsilfidis, Alexandros; Mourjopoulos, John (2011). Blind single-channel suppression of late reverberation based on perceptual reverberation modeling. *The Journal of the Acoustical Society of America*. (Also listed as [PE3])

> Annotation: Cross-listed perceptual dereverberation reference. In the speech section, its main value is showing that late-tail suppression can be guided by audibility and speech clarity rather than by raw energy reduction alone. This keeps the bibliography connected across perceptual modeling, dereverb design, and real-time speech use cases.

**[SR25]** Unknown authors (1995). Fichera, I (2020). the Accuracy of Reverberation Time Predictions for General Teaching Spaces. *Acoustics 2020*. DOI: [10.25144/13333](https://doi.org/10.25144/13333)

> Context entry. Validation of RT60 prediction formulas in classroom environments. Context for applied room acoustics. It connects reverberation prediction to teaching spaces, where clarity and learning outcomes make acoustic estimates socially consequential rather than merely technical.

**[SR24]** Unknown authors (2008). Reverberation Assessment in Audioband Speech Signals for Telepresence Systems. *Proceedings of the International Conference on Signal Processing and Multimedia Applications*. DOI: [10.5220/0001936602570262](https://doi.org/10.5220/0001936602570262)

> Context entry. Reverberation metrics for telepresence audio quality assessment. Peripheral context. It is relevant to practical deployments where users care about communication quality, echoic coloration, fatigue, and perceived professionalism rather than architectural realism alone.

---

## Section 9: Underwater and Non-Room Reverberation

Papers whose "reverberation" subject matter refers to underwater acoustics, seabed scattering, structural media, musicology, or other domains unrelated to architectural room acoustics. Included because they were in the source bibliography; they have no direct bearing on verbx's current design. Retained for bibliographic completeness, search disambiguation, and to make clear which references should not be used as evidence for room-reverb DSP decisions.

**[UW2]** Chotiros, Nicholas (2012). Non-Rayleigh reverberation statistics. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4708991](https://doi.org/10.1121/1.4708991)

> Context entry. Statistical characterization of underwater backscatter. Retained for completeness. It is useful mostly as a vocabulary boundary marker: similar statistical language appears in room acoustics, but the physical scattering mechanisms and operating assumptions are different from architectural reverberation.

**[UW9]** Guo, Yongqiang; Che, Weiqiu (2010). Reverberation-Ray Matrix Analysis of Acoustic Waves in Multilayered Anisotropic Structures. *Acoustic Waves*. DOI: [10.5772/9788](https://doi.org/10.5772/9788)

> Context entry. Reverberation-ray matrix method for layered elastic media. Structural acoustics domain; retained for completeness. It is another example where "reverberation" describes repeated wave interaction in a medium, but the target domain is not listener-room perception or audio-effect synthesis.

**[UW5]** Harrison, Chris (2009). Reverberation versus time or reverberation versus range? A definitive relationship. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.3248705](https://doi.org/10.1121/1.3248705)

> Context entry. Underwater reverberation decay modeling. Retained for completeness. Its time-versus-range framing belongs to sonar and propagation analysis, not room impulse response synthesis, so it is cataloged here as an adjacent but non-authoritative use of the word reverberation.

**[UW3]** Katsnelson, Boris; Petnikov, Valery; Lynch, James (2011). Low-Frequency Bottom Reverberation in Shallow Water. *Fundamentals of Shallow Water Acoustics*. DOI: [10.1007/978-1-4419-9777-7_6](https://doi.org/10.1007/978-1-4419-9777-7_6)

> Context entry. Shallow water bottom reverberation. Domain is underwater acoustics; retained for completeness only. It documents a non-room meaning of late returned acoustic energy and helps prevent accidental overextension of underwater decay or scattering models into verbx's room-oriented DSP.

**[UW6]** LePage, Kevin D. (2009). High-frequency broadband coherent reverberation predictions for the reverberation modeling workshop. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.3248711](https://doi.org/10.1121/1.3248711)

> Context entry. Underwater high-frequency reverberation modeling workshop proceedings. Retained for completeness. It is useful only for bibliography hygiene and terminology separation; verbx should continue to rely on room-acoustic and speech-acoustic references for implementation decisions.

**[UW4]** Oh (2015). Low-Frequency Normal Mode Reverberation Model. *The Journal of the Acoustical Society of Korea*. DOI: [10.7776/ask.2015.34.3.184](https://doi.org/10.7776/ask.2015.34.3.184)

> Context entry. Normal mode model for low-frequency underwater reverberation. Domain is underwater acoustics; retained for completeness only. The normal-mode framing is physically interesting but not directly applicable to ordinary room reverberation controls, except as a broad reminder that low-frequency propagation can be strongly modal in many acoustic systems.

**[UW10]** Paul, Joanna (2025). Amplification, Reverberation, and Distortion. *Music as Classical Reception*. DOI: [10.1093/9780198958024.003.0012](https://doi.org/10.1093/9780198958024.003.0012)

> Context entry. Musicological analysis of reverberation as aesthetic phenomenon. Not DSP; retained for completeness. It may be useful for cultural or aesthetic framing, but it should not be cited for algorithm design, measurement formulas, or real-time processing behavior.

**[UW1]** Preston, John R.; Abraham, Douglas A.; Yang, Jie (2014). Non-stationary reverberation observations from the shallow water TREX13 reverberation experiments using the FORA triplet array. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4900079](https://doi.org/10.1121/1.4900079)

> Context entry. Shallow-water acoustic reverberation measurements. Domain is underwater acoustics; retained for completeness only. The term "reverberation" here refers to environmental backscatter and propagation effects rather than enclosed-room decay, so it should not be used to justify verbx room modeling choices.

**[UW8]** Tang, Dajun (2013). Issues in reverberation modeling. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4831447](https://doi.org/10.1121/1.4831447)

> Context entry. Underwater reverberation modeling issues review. Retained for completeness. It remains in the bibliography to avoid silently dropping source material, but it should be treated as out of scope for room-size, RT60, dereverb, and algorithmic reverb documentation.

**[UW7]** Thorsos, Eric I.; Perkins, John S. (2007). Reverberation modeling issues highlighted by the first Reverberation Modeling Workshop. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2943047](https://doi.org/10.1121/1.2943047)

> Context entry. Summary of the first underwater acoustic reverberation modeling workshop. Retained for completeness. It helps identify a separate modeling community with different validation tasks, datasets, and physical assumptions than the architectural and production-audio problems verbx addresses.

---

---

---

## Section 10: Extended Crossref Literature Index

This unannotated discovery index adds 900 Crossref-derived references to the 102 hand-curated entries above, bringing the guide bibliography to 1,002 total entries.

The entries below are intentionally not treated as vetted design authority. They are included to make the PDF a much broader literature map for reverberation, dereverberation, spatial audio, room acoustics, and related acoustic measurement work. Use the annotated sections above for canonical implementation guidance.

Source: Crossref Works API metadata, generated May 22, 2026.

Discovery queries:

- Room reverberation and room acoustics: `room acoustics reverberation time RT60`
- Artificial reverberation and FDNs: `artificial reverberation feedback delay network`
- Impulse responses and convolution reverb: `room impulse response convolution reverberation`
- Speech dereverberation and clarity: `speech dereverberation reverberation direct-to-reverberant ratio`
- Spatial audio and Ambisonics: `spatial audio ambisonics reverberation auralization`
- Perceptual reverberation: `perception reverberation listener envelopment clarity`
- Auralization and virtual acoustics: `auralization virtual acoustics reverberation`
- Acoustic measurement and decay metrics: `energy decay curve reverberation time acoustic measurement`
- Acoustic simulation and image-source models: `image source method room acoustic simulation reverberation`
- Late reverberation modeling: `late reverberation statistical model acoustic signal processing`

### Extended Bibliography Entries


#### Room reverberation and room acoustics

**[XREF0459]** Abd-Elbasseer, Mohamed; Kh Mohamed, Hatem (2021). Performance Evaluation and Effectiveness of the Reverberation Room. *Sound&Vibration*. DOI: [10.32604/sv.2021.09417](https://doi.org/10.32604/sv.2021.09417)

**[XREF0696]** Abolhasannejad, Vahideh; Golmohammadi, Rostam; Aliabadi, Mohsen; et al. (2018). An image-based method for non-contact and dynamic room acoustics analysis. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2018.05.018](https://doi.org/10.1016/j.apacoust.2018.05.018)

**[XREF0462]** Accolti, Ernesto; di Sciascio, Fernando (2017). Room acoustics: Idealized field and real field considerations. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0000795](https://doi.org/10.1121/2.0000795)

**[XREF0251]** Accolti, Ernesto; di Sciascio, Fernando (2021). On the use of time windows for the determination of sound strength parameter G from uncalibrated room impulse responses measurements. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2021.108023](https://doi.org/10.1016/j.apacoust.2021.108023)

**[XREF0871]** Accolti, Ernesto; Gimenez, Javier; Vorländer, Michael (2023). Uncertainties of room acoustics simulation due to directivity data of musical instruments. *Unspecified venue*. DOI: [10.36227/techrxiv.20858596.v3](https://doi.org/10.36227/techrxiv.20858596.v3)

**[XREF0873]** Accolti, Ernesto; Gimenez, Javier; Vorländer, Michael (2023). Uncertainties of room acoustics simulation due to directivity data of musical instruments. *Unspecified venue*. DOI: [10.36227/techrxiv.20858596.v2](https://doi.org/10.36227/techrxiv.20858596.v2)

**[XREF0900]** Accolti, Ernesto; Gimenez, Javier; Vorländer, Michael (2023). Uncertainties of room acoustics simulation due to directivity data of musical instruments. *Unspecified venue*. DOI: [10.36227/techrxiv.20858596](https://doi.org/10.36227/techrxiv.20858596)

**[XREF0314]** Achdjian, Hossep; Bustillo, Julien; Arciniegas, Andres; et al. (2018). Contact surface fraction evaluation between aluminium and polymer using acoustic reverberation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2018.07.016](https://doi.org/10.1016/j.apacoust.2018.07.016)

**[XREF0788]** Adelman-Larsen, Niels W. (2017). Amplified music and variable acoustics—Shorter reverberation times at low frequencies. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4988599](https://doi.org/10.1121/1.4988599)

**[XREF0087]** Adelman-Larsen, Niels Werner; Jeong, Cheol-Ho; Støfringsdal, Bård (2018). Investigation on acceptable reverberation time at various frequency bands in halls that present amplified music. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2017.07.005](https://doi.org/10.1016/j.apacoust.2017.07.005)

**[XREF0579]** Ågren, Anders (1992). The design and evaluation of a hemi-anechoic engine test room. *Applied Acoustics*. DOI: [10.1016/0003-682x(92)90024-m](https://doi.org/10.1016/0003-682x(92)90024-m)

**[XREF0890]** Ahnert, Wolfgang; Noy, Dirk (2022). Room Acoustics and Sound System Design. *Sound Reinforcement for Audio Engineers*. DOI: [10.4324/9781003220268-2](https://doi.org/10.4324/9781003220268-2)

**[XREF0876]** Aitchison, G. J. (1965). Measurement of Reverberation Time. *American Journal of Physics*. DOI: [10.1119/1.1971726](https://doi.org/10.1119/1.1971726)

**[XREF0564]** Aknin, Achille; Badeau, Roland (2021). Stochastic Reverberation Model with a Frequency Dependent Attenuation. *2021 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)*. DOI: [10.1109/waspaa52581.2021.9632792](https://doi.org/10.1109/waspaa52581.2021.9632792)

**[XREF0057]** Al-bayyar, Zinah; Kitapci, Kivanc (2022). Effects of reverberation time and sound source composition on sense of place constructs. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2021.108427](https://doi.org/10.1016/j.apacoust.2021.108427)

**[XREF0515]** Alarcão, D.; Coelho, J. L. Bento (2003). Lambertian Enclosures — A First Step towards Fast Room Acoustics Simulation. *Building Acoustics*. DOI: [10.1260/135101003765184807](https://doi.org/10.1260/135101003765184807)

**[XREF0701]** Alarcão, Diogo; Inácio, Octávio (2022). Determination of room acoustic parameters using spherical beamforming – The example of Lisbon’s Garrett Hall. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2022.108734](https://doi.org/10.1016/j.apacoust.2022.108734)

**[XREF0635]** Alary, B.; Politis, A. (2022). A Dataset for Location- and Direction-Dependent Reverberation Analysis. *Proceedings of the 10th Convention of the European Acoustics Association Forum Acusticum 2023*. DOI: [10.61782/fa.2023.0833](https://doi.org/10.61782/fa.2023.0833)

**[XREF0770]** Aletta, Francesco; Kang, Jian (2020). Historical Acoustics: Relationships between People and Sound over Time. *Acoustics*. DOI: [10.3390/acoustics2010009](https://doi.org/10.3390/acoustics2010009)

**[XREF0573]** Algargoosh, Alaa; Navvab, Mojtaba; Granzow, John (2022). A method for analyzing room modal response using auralization. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2022.108859](https://doi.org/10.1016/j.apacoust.2022.108859)

**[XREF0678]** Allen, J. B. (1982). Effects of small room reverberation on subjective preference. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2019478](https://doi.org/10.1121/1.2019478)

**[XREF0773]** Allen, Jont Brandon (1979). Method and apparatus for cancelling room reverberation and noise pickup. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.382584](https://doi.org/10.1121/1.382584)

**[XREF0150]** Alpkocak, Adil; Sis, Malik (2010). Computing Impulse Response of Room Acoustics Using the Ray-Tracing Method in Time Domain. *Archives of Acoustics*. DOI: [10.2478/v10168-010-0039-8](https://doi.org/10.2478/v10168-010-0039-8)

**[XREF0239]** Álvarez, Julián; Graus, Ramon; Martín-Nieva, Helena (2025). A historical overview of Higini Arau’s proposal for calculating reverberation time (1988). *Proceedings of the 11th Convention of the European Acoustics Association Forum Acusticum / EuroNoise 2025*. DOI: [10.61782/fa.2025.0012](https://doi.org/10.61782/fa.2025.0012)

**[XREF0472]** Amador, Emmanuel; Miry, Celine; Bouyge, Nicolas (2014). Compatible susceptibility measurements in fully anechoic room and reverberation chamber. *2014 International Symposium on Electromagnetic Compatibility*. DOI: [10.1109/emceurope.2014.6931024](https://doi.org/10.1109/emceurope.2014.6931024)

**[XREF0347]** Andrieu, Guillaume (2020). Frequency and time-domain performance assessment of vibrating intrinsic reverberation chambers. *Electromagnetic Reverberation Chambers: Recent advances and innovative applications*. DOI: [10.1049/sbew544e_ch3](https://doi.org/10.1049/sbew544e_ch3)

**[XREF0802]** Aralikatti, Rohith; Boeddeker, Christoph; Wichern, Gordon; et al. (2023). Reverberation as Supervision For Speech Separation. *ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp49357.2023.10095022](https://doi.org/10.1109/icassp49357.2023.10095022)

**[XREF0062]** Arau, Higini (1997). Variation of the Reverberation Time of Places of Public Assembly with Audience Size. *Building Acoustics*. DOI: [10.1177/1351010x9700400202](https://doi.org/10.1177/1351010x9700400202)

**[XREF0628]** Arau-Puchades, Higini (2012). Sound Pressure Levels in Rooms: A Study of Steady State Intensity, Total Sound Level, Reverberation Distance, a New Discussion of Steady State Intensity and other Experimental Formulae. *Building Acoustics*. DOI: [10.1260/1351-010x.19.3.205](https://doi.org/10.1260/1351-010x.19.3.205)

**[XREF0803]** Arau-Puchades, Higini (2012). The Refurbishment of the Orchestra Rehearsal Room of the Great Theater of Liceu. *Building Acoustics*. DOI: [10.1260/1351-010x.19.1.45](https://doi.org/10.1260/1351-010x.19.1.45)

**[XREF0427]** Arau-Puchades, Higini; Berardi, Umberto (2013). The reverberation radius in an enclosure with asymmetrical absorption distribution. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4800909](https://doi.org/10.1121/1.4800909)

**[XREF0042]** Aretz, Marc; Orlowski, Raf (2009). Sound strength and reverberation time in small concert halls. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2009.02.001](https://doi.org/10.1016/j.apacoust.2009.02.001)

**[XREF0253]** Aretz, Marc; Orlowski, Raf (2010). Balancing Sound Strength and Reverberation Time in Small Concert Halls by Means of Variable Acoustics. *Noise & Vibration Worldwide*. DOI: [10.1260/0957-4565.41.8.11](https://doi.org/10.1260/0957-4565.41.8.11)

**[XREF0157]** Aretz, Marc; Orlowski, Raf (2011). Balancing sound strength and reverberation time in small concert halls by means of variable acoustics. *Noise Notes*. DOI: [10.1260/1475-4738.10.1.3](https://doi.org/10.1260/1475-4738.10.1.3)

**[XREF0126]** Arnela, Marc; Burbano-Escolà, Ricardo; Scoczynski Ribeiro, Rodrigo; et al. (2025). Reverberation time and random-incidence sound absorption measured in the audible and ultrasonic ranges with an omnidirectional parametric loudspeaker. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2024.110414](https://doi.org/10.1016/j.apacoust.2024.110414)

**[XREF0721]** Atkins, Joshua; West, James E. (2015). Spatial Audio and Room Acoustics. *Acoustics, Information, and Communication*. DOI: [10.1007/978-3-319-05660-9_1](https://doi.org/10.1007/978-3-319-05660-9_1)

**[XREF0879]** Austin, Beth; Cheer, Jordan (2025). Experimental implementation of an actively controlled tuned vibration absorber using piezoelectric actuators. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002056](https://doi.org/10.1121/2.0002056)

**[XREF0008]** Avis, MR; Darlington, P (2024). Modifying Low Frequency Room Acoustics 1: Local Active De-Reverberation. *RS 11*. DOI: [10.25144/20135](https://doi.org/10.25144/20135)

**[XREF0487]** Aydin, Aybar; Baran, Vlad; Zhang, Kathleen Ying-Ying; et al. (2024). A real-world Lexicon 960L reverberation chamber: Simulating a hardware reverberation unit in virtual acoustics. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0027047](https://doi.org/10.1121/10.0027047)

**[XREF0358]** Ayr, Ubaldo; Martellotta, Francesco; Rospi, Gianluca (2017). A method for the low frequency qualification of reverberation test rooms using a validated finite element model. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2016.09.001](https://doi.org/10.1016/j.apacoust.2016.09.001)

**[XREF0448]** Azad, H (2023). Ali Qapu: Persian Historical Music Room. *Auditorium Acoustics 2008*. DOI: [10.25144/17495](https://doi.org/10.25144/17495)

**[XREF0426]** Azad, Hassan; Meyer, Julie; Siebein, Gary; et al. (2019). The Effects of Adding Pyramidal and Convex Diffusers on Room Acoustic Parameters in a Small Non-Diffuse Room. *Acoustics*. DOI: [10.3390/acoustics1030037](https://doi.org/10.3390/acoustics1030037)

**[XREF0711]** Baade, Peter K.; Maling, Jr., George C. (1998). Technical note: Reverberation room qualification using multitone signals. *Noise Control Engineering Journal*. DOI: [10.3397/1.2828451](https://doi.org/10.3397/1.2828451)

**[XREF0017]** Baker, M. (1974). An automatic reverberation time measuring system. *Applied Acoustics*. DOI: [10.1016/0003-682x(74)90035-8](https://doi.org/10.1016/0003-682x(74)90035-8)

**[XREF0388]** Balint, Jamilla (2023). Measuring Sound Absorption: The Hundred-Year Debate on the Reverberation Chamber Method. *Acoustics Today*. DOI: [10.1121/at.2023.19.3.13](https://doi.org/10.1121/at.2023.19.3.13)

**[XREF0404]** Bamba, A.; Joseph, W.; Plets, D.; et al. (2011). Assessment of reverberation time by two measurement systems for room electromagnetics analysis. *2011 IEEE International Symposium on Antennas and Propagation (APSURSI)*. DOI: [10.1109/aps.2011.5997191](https://doi.org/10.1109/aps.2011.5997191)

**[XREF0421]** Barbar, Steve (2023). Electronic architecture—Improving room acoustics using time variant electro-acoustic systems. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0018330](https://doi.org/10.1121/10.0018330)

**[XREF0207]** Barbaresi, L.; Pettoni Possenti, V.; Scrosati, C. (2024). On the influence of reference reverberation time on façade sound insulation measurements. *Proceedings of the 10th Convention of the European Acoustics Association Forum Acusticum 2023*. DOI: [10.61782/fa.2023.0319](https://doi.org/10.61782/fa.2023.0319)

**[XREF0445]** Barron, M (2023). Auditorium Acoustics - a Room Acoustician's Perspective. *Reproduced Sound 2007*. DOI: [10.25144/17744](https://doi.org/10.25144/17744)

**[XREF0670]** Barron, M (2024). Current Developments in Analogue Acoustic Modelling. *Room Acoustics with Emphasis on Electroacoustics 1979*. DOI: [10.25144/23453](https://doi.org/10.25144/23453)

**[XREF0254]** Barron, Randall F. (2002). Room Acoustics. *Industrial Noise Control and Acoustics*. DOI: [10.1201/9780203910085-7](https://doi.org/10.1201/9780203910085-7)

**[XREF0380]** Bartel, Thomas W (1977). Interactive computer program for the determination of reverberation time. *Unspecified venue*. DOI: [10.6028/nbs.ir.77-1383](https://doi.org/10.6028/nbs.ir.77-1383)

**[XREF0073]** Bartel, Thomas W.; Magrab, Edward B. (1976). Measurement of reverberation time in the National Bureau of Standards reverberation room. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2003434](https://doi.org/10.1121/1.2003434)

**[XREF0736]** Bastien, Corto; Ernoult, Augustin; Fritz, Claudia (2025). Modeling the intonation profile of a recorder. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002181](https://doi.org/10.1121/2.0002181)

**[XREF0219]** Battaglia, Paul (2017). Speech recognition in reverberation and background chatter. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0000668](https://doi.org/10.1121/2.0000668)

**[XREF0872]** Baumeister, K. J. (1986). Reverberation Effects on Directionality and Response of Stationary Monopole and Dipole Sources in a Wind Tunnel. *Journal of Vibration and Acoustics*. DOI: [10.1115/1.3269307](https://doi.org/10.1115/1.3269307)

**[XREF0031]** Bean, L.W. (1989). Direct measurement of reverberation time with a sound level meter. *Applied Acoustics*. DOI: [10.1016/0003-682x(89)90056-x](https://doi.org/10.1016/0003-682x(89)90056-x)

**[XREF0009]** Belanger, Zackery (2021). Beyond reverberation time: Toward a new geometric approach to room acoustics. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0007826](https://doi.org/10.1121/10.0007826)

**[XREF0059]** Benedetto, G.; Brosio, E.; Spagnolo, R. (1981). The effect of stationary diffusers in the measurement of sound absorption coefficients in a reverberation room: An experimental study. *Applied Acoustics*. DOI: [10.1016/0003-682x(81)90044-x](https://doi.org/10.1016/0003-682x(81)90044-x)

**[XREF0306]** Benedetto, Giuliana; Spagnolo, Renato (1983). A method for correcting the reverberation times of enclosures as a function of humidity and temperature. *Applied Acoustics*. DOI: [10.1016/0003-682x(83)90013-0](https://doi.org/10.1016/0003-682x(83)90013-0)

**[XREF0595]** Berdahl, Edgar; Niemeyer, Gunter; Smith III, Julius O. (2010). A physically motivated room reverberation enhancement system that is stable in any (passive) room.. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.3383487](https://doi.org/10.1121/1.3383487)

**[XREF0076]** Berzborn, Marco; Vorländer, Michael (2025). Stochastic Variational Inference of Directional Decay Times in a Reverberation Room. *Proceedings of the 11th Convention of the European Acoustics Association Forum Acusticum / EuroNoise 2025*. DOI: [10.61782/fa.2025.0830](https://doi.org/10.61782/fa.2025.0830)

**[XREF0798]** Besnier, Philippe (2022). Statistics of electromagnetic reverberation chambers and their simulation through time domain modeling. *Advanced Time Domain Modeling for Electrical Engineering*. DOI: [10.1049/sbew550e_ch14](https://doi.org/10.1049/sbew550e_ch14)

**[XREF0714]** Bietz, Heinrich; Wittstock, Volker (2025). Combining absorption coefficient measurements in reverberation rooms with different volumes. *Proceedings of the 11th Convention of the European Acoustics Association Forum Acusticum / EuroNoise 2025*. DOI: [10.61782/fa.2025.0211](https://doi.org/10.61782/fa.2025.0211)

**[XREF0084]** Bilbao, Stefan; Hamilton, Brian (2016). Passive time-domain numerical designs for room acoustics simulation. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0000460](https://doi.org/10.1121/2.0000460)

**[XREF0743]** Bilbao, Stefan; Hamilton, Brian (2017). Directional source modeling in wave-based room acoustics simulation. *2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)*. DOI: [10.1109/waspaa.2017.8170007](https://doi.org/10.1109/waspaa.2017.8170007)

**[XREF0419]** Bilbao, Stefan; Hamilton, Brian (2019). Passive volumetric time domain simulation for room acoustics applications. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.5095876](https://doi.org/10.1121/1.5095876)

**[XREF0005]** Billon, A.; Picaut, J.; Sakout, A. (2008). Prediction of the reverberation time in high absorbent room using a modified-diffusion model. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2007.03.001](https://doi.org/10.1016/j.apacoust.2007.03.001)

**[XREF0033]** Bjor, Ole-Herman; Hognestad, Hårek (1979). A method for automatic reverberation time measurement. *Applied Acoustics*. DOI: [10.1016/0003-682x(79)90034-3](https://doi.org/10.1016/0003-682x(79)90034-3)

**[XREF0731]** Blacodon, Daniel; Bulté, Jean (2013). De-reverberation of a closed test section of a wind tunnel with a multi microphones cepstral method. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4801387](https://doi.org/10.1121/1.4801387)

**[XREF0412]** Blanck, Michael W. (1976). Qualification of a 300 Cubic Metre Reverberation Room. *Noise Control Engineering*. DOI: [10.3397/1.2832041](https://doi.org/10.3397/1.2832041)

**[XREF0289]** Blasinski, L.; Kociński, J. (2024). Perception of reverberation length in rooms with Active Acoustics Enhancement Systems. *Proceedings of the 10th Convention of the European Acoustics Association Forum Acusticum 2023*. DOI: [10.61782/fa.2023.0235](https://doi.org/10.61782/fa.2023.0235)

**[XREF0870]** Boning, W; Bassuet, A (2023). From the Sound Up Reverse-Engineering Room Shapes from Acoustic Signatures. *Auditorium Acoustics 2015*. DOI: [10.25144/16948](https://doi.org/10.25144/16948)

**[XREF0887]** Boning, W; Bassuet, A (2023). From the Sound Up Reverse-Engineering Room Shapes from Acoustic Signatures. *Auditorium Acoustics 2015*. DOI: [10.25144/16185](https://doi.org/10.25144/16185)

**[XREF0719]** Boothroyd, Arthur (2003). Hearing aids and room acoustics. *The Hearing Journal*. DOI: [10.1097/01.hj.0000292914.14697.ba](https://doi.org/10.1097/01.hj.0000292914.14697.ba)

**[XREF0363]** Bork, I (2023). Simulation and Measurement of Auditorium Acoustics - the Round Robins on Room Acoustical Simulation. *Auditorium Acoustics 2002*. DOI: [10.25144/18317](https://doi.org/10.25144/18317)

**[XREF0543]** Borkowski, Bartłomiej; Pluta, Marek (2015). Automated Measurement System for Room Acoustics – an Initial Feasibility Study. *Archives of Acoustics*. DOI: [10.1515/aoa-2015-0043](https://doi.org/10.1515/aoa-2015-0043)

**[XREF0015]** Bottalico, Pasquale; Graetzer, Simone; Hunter, Eric J. (2016). Speech accommodation to room acoustics: Reverberation time and clarity. *Journal of the Acoustical Society of America*. DOI: [10.1121/1.4949779](https://doi.org/10.1121/1.4949779)

**[XREF0374]** Boucher, Matthew; Rychtarikova, Monika; Zelem, Lukas; et al. (2019). Reverberation time and audibility in phased geometrical acoustics using plane or spherical wave reflection coefficients. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.5101840](https://doi.org/10.1121/1.5101840)

**[XREF0425]** Boucher, Matthew A.; Rychtarikova, Monika; Zelem, Lukáš; et al. (2019). Reverberation time and audibility in phased geometrical acoustics using plane or spherical wave reflection coefficients. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.5095862](https://doi.org/10.1121/1.5095862)

**[XREF0860]** Bovo, R; Ciorba, A; Abenante, L; et al. (2023). Effects of Classroom Noise &#38; Reverberation on the Speech Perception of Bilingual Children Learning in Their Second Language. *Spring Conference Acoustics 2008*. DOI: [10.25144/17598](https://doi.org/10.25144/17598)

**[XREF0138]** Braat-Eggen, Ella; Poll, Marijke Keus v.d.; Hornikx, Maarten; et al. (2019). Auditory distraction in open-plan study environments: Effects of background speech and reverberation time on a collaboration task. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2019.04.038](https://doi.org/10.1016/j.apacoust.2019.04.038)

**[XREF0585]** Bradley, David; Adelgren, Jacob; Müller-Trapet, Markus; et al. (2013). Effect of boundary diffusers in a reverberation chamber: Preliminary investigation. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4862775](https://doi.org/10.1121/1.4862775)

**[XREF0378]** Bradley, David T.; Müller-Trapet, Markus; Adelgren, Jacob; et al. (2014). Comparison of Hanging Panels and Boundary Diffusers in a Reverberation Chamber. *Building Acoustics*. DOI: [10.1260/1351-010x.21.2.145](https://doi.org/10.1260/1351-010x.21.2.145)

**[XREF0772]** Bradley, David T.; Wang, Lily M. (2009). Quantifying the Double Slope Effect in Coupled Volume Room Systems. *Building Acoustics*. DOI: [10.1260/135101009788913275](https://doi.org/10.1260/135101009788913275)

**[XREF0334]** Bradley, J.S. (2011). Review of objective room acoustics measures and future needs. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2011.04.004](https://doi.org/10.1016/j.apacoust.2011.04.004)

**[XREF0170]** Bradley, JS (2023). Using Room Acoustics Measures to Understand a Large Room and Sound Reinforcement System. *Auditorium Acoustics 2011*. DOI: [10.25144/16844](https://doi.org/10.25144/16844)

**[XREF0665]** Brandão, Eric; Santos, Edna S.O.; Melo, Viviane S.G.; et al. (2022). On the performance investigation of distinct algorithms for room acoustics simulation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2021.108484](https://doi.org/10.1016/j.apacoust.2021.108484)

**[XREF0460]** Brandewie, Eugene; Zahorik, Pavel (2012). Adaptation to Room Acoustics Using the Modified Rhyme Test. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4738498](https://doi.org/10.1121/1.4738498)

**[XREF0866]** Branski, Adam; Kocan-Krawczyk, Anna; Predka, Edyta (2017). An Influence of the Wall Acoustic Impedance on the Room Acoustics. The Exact Solution. *Archives of Acoustics*. DOI: [10.1515/aoa-2017-0070](https://doi.org/10.1515/aoa-2017-0070)

**[XREF0238]** Bredvei, Ø; Tronstad, TV; Nielsen, JL (2023). Telepresence Room Acoustics. *Auditorium Acoustics 2008*. DOI: [10.25144/17518](https://doi.org/10.25144/17518)

**[XREF0615]** Bridger, Joseph; Stewart, Noral; Farbo, Aaron (2007). Lessons learned from room acoustics design and investigation (especially from using EASE room acoustics software) in our consulting practice. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4782185](https://doi.org/10.1121/1.4782185)

**[XREF0855]** Brinkmann, Fabian; Aspöck, Lukas; Ackermann, David; et al. (2021). A benchmark for room acoustical simulation. Concept and database. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2020.107867](https://doi.org/10.1016/j.apacoust.2020.107867)

**[XREF0494]** Brixen, Eddy B. (2011). Room Acoustics Measures. *Audio Metering*. DOI: [10.1016/b978-0-240-81467-4.10031-0](https://doi.org/10.1016/b978-0-240-81467-4.10031-0)

**[XREF0839]** Brooks, Bennett M. (2005). The feasibility of noise and reverberation control for good acoustics in classrooms. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4788388](https://doi.org/10.1121/1.4788388)

**[XREF0287]** Broyles, Jonathan Michael; Rusk, Zane Tyler (2023). Predicting the reverberation time of concert halls by use of a random forest regression model. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0001751](https://doi.org/10.1121/2.0001751)

**[XREF0752]** Brutti, Alessio; Matassoni, Marco (2014). On the use of Early-To-Late Reverberation ratio for ASR in reverberant environments. *2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2014.6854481](https://doi.org/10.1109/icassp.2014.6854481)

**[XREF0437]** Buen, A; Strand, L (2023). Room Acoustics in the Scene Ii in the New Oslo Opera and the Ridehuset - Two Variable Acoustics Coupled Space Venues. *Auditorium Acoustics 2008*. DOI: [10.25144/17482](https://doi.org/10.25144/17482)

**[XREF0636]** Buggè, Valentina; Shtrepi, Louena; Nocerino, Giovanni; et al. (2025). Integrating design, acoustic performance and fabrication: the case of the acoustic optimization of a meeting room. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002236](https://doi.org/10.1121/2.0002236)

**[XREF0575]** Bullard, Chad; Lentz, Jennifer (2024). Auditory brightness in room reverberation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0027603](https://doi.org/10.1121/10.0027603)

**[XREF0055]** Bullard, Chad; Lentz, Jennifer J. (2024). Auditory brightness perception changes in room reverberation. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002237](https://doi.org/10.1121/2.0002237)

**[XREF0710]** Burd, A N (1974). Room Acoustics. *Physics Bulletin*. DOI: [10.1088/0031-9112/25/8/033](https://doi.org/10.1088/0031-9112/25/8/033)

**[XREF0151]** Burgess, M.A.; Utley, W.A. (1985). Reverberation times in British living rooms. *Applied Acoustics*. DOI: [10.1016/0003-682x(85)90055-6](https://doi.org/10.1016/0003-682x(85)90055-6)

**[XREF0318]** Byoung-gi Lee; JongSuk Choi; Daijin Kim; et al. (2010). Verification of sound source localization in reverberation room and its real time adaptation using visual information. *2010 IEEE Workshop on Advanced Robotics and its Social Impacts*. DOI: [10.1109/arso.2010.5679699](https://doi.org/10.1109/arso.2010.5679699)

**[XREF0156]** Cabrera, Densil; Xun, Jianyang; Guski, Martin (2016). Calculating Reverberation Time from Impulse Responses: A Comparison of Software Implementations. *Acoustics Australia*. DOI: [10.1007/s40857-016-0055-6](https://doi.org/10.1007/s40857-016-0055-6)

**[XREF0093]** Campbell, Murray (2001). Reverberation time. *Oxford Music Online*. DOI: [10.1093/gmo/9781561592630.article.23282](https://doi.org/10.1093/gmo/9781561592630.article.23282)

**[XREF0410]** Campo, Nicola; Rissone, Paolo; Toderi, Marco (2000). Adaptive pyramid tracing: a new technique for room acoustics. *Applied Acoustics*. DOI: [10.1016/s0003-682x(99)00072-9](https://doi.org/10.1016/s0003-682x(99)00072-9)

**[XREF0539]** Canfield-Dafilou, Elliot K.; Yusuf, Suhail; Rau, Mark (2025). Acoustics of sympathetic strings in a sarangi. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002280](https://doi.org/10.1121/2.0002280)

**[XREF0193]** Caniato, Marco; Bettarello, Federica (2025). From Theory to Reality: Assessing Predictive Models for Reverberation Time in Built Environments. *Proceedings of the 11th Convention of the European Acoustics Association Forum Acusticum / EuroNoise 2025*. DOI: [10.61782/fa.2025.0418](https://doi.org/10.61782/fa.2025.0418)

**[XREF0857]** Cansu, Nicole; Öhlund Wistbacka, Greta; Holmqvist-Jämsén, Sofia; et al. (2025). Speaker’s comfort and vocal effort in different room acoustic conditions: A controlled field experiment in a university lecture room. *Building Acoustics*. DOI: [10.1177/1351010x251364493](https://doi.org/10.1177/1351010x251364493)

**[XREF0647]** Cao, Feng; Zhang, Xuegang; Han, Jing; et al. (2021). Experimental Analysis of Statistical Property of Low Frequency Reverberation Envelope in Shallow Water. *2021 OES China Ocean Acoustics (COA)*. DOI: [10.1109/coa50123.2021.9520013](https://doi.org/10.1109/coa50123.2021.9520013)

**[XREF0407]** Cao, Huanzhi; Cai, Zhiming (2016). A binning method based on split-beam array to suppress reverberation. *2016 IEEE/OES China Ocean Acoustics (COA)*. DOI: [10.1109/coa.2016.7535667](https://doi.org/10.1109/coa.2016.7535667)

**[XREF0527]** Cardoso Soares, Murilo; Brandão Carneiro, Eric; Aizik Tenenbaum, Roberto; et al. (2022). Low-frequency room acoustical simulation of a small room with BEM and complex-valued surface impedances. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2021.108570](https://doi.org/10.1016/j.apacoust.2021.108570)

**[XREF0444]** Carey, William M.; Pierce, Allan D. (2012). Sound speed, pulse spreading and reverberation in muddy bubbly sediments. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.3693533](https://doi.org/10.1121/1.3693533)

**[XREF0514]** Castañeda, Eduardo Méndez (1994). Computer-based system for reverberation room design. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.411050](https://doi.org/10.1121/1.411050)

**[XREF0291]** Cecchi, S.; Romoli, L.; Carini, A.; et al. (2014). A multichannel and multiple position adaptive room response equalizer in warped domain: Real-time implementation and performance evaluation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2014.02.011](https://doi.org/10.1016/j.apacoust.2014.02.011)

**[XREF0748]** Cerdá, S.; Giménez, A.; Romero, J.; et al. (2009). Room acoustical parameters: A factor analysis approach. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2008.01.001](https://doi.org/10.1016/j.apacoust.2008.01.001)

**[XREF0850]** Cha, Changhyok; Lee, Hyojin; Jeong, Daeup (2024). Measurements of sound absorption coefficients of raked audience seating in a rectangular scale model room. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2024.109872](https://doi.org/10.1016/j.apacoust.2024.109872)

**[XREF0376]** Chakraborty, Rupayan; Nadeu, Climent (2013). Real-time multi-microphone recognition of simultaneous sounds in a room environment. *2013 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2013.6639359](https://doi.org/10.1109/icassp.2013.6639359)

**[XREF0283]** Champagne, B.; Bedard, S.; Stephenne, A. (1996). Performance of time-delay estimation in the presence of room reverberation. *IEEE Transactions on Speech and Audio Processing*. DOI: [10.1109/89.486067](https://doi.org/10.1109/89.486067)

**[XREF0858]** Chang, Po-Rong; Lin, C. G.; Yeh, Bao-Fuh (1994). Inverse filtering of a loudspeaker and room acoustics using time-delay neural networks. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.409959](https://doi.org/10.1121/1.409959)

**[XREF0068]** Chen, Qiu; Ou, Dayi (2021). The effects of classroom reverberation time and traffic noise on English listening comprehension of Chinese university students. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2021.108082](https://doi.org/10.1016/j.apacoust.2021.108082)

**[XREF0022]** Chen, Yung-Hsiang; Chen, Kuo-Tsai; Chaing, Yan-Hua (1996). Plate-damping measurements in a single reverberation room. *Applied Acoustics*. DOI: [10.1016/0003-682x(95)00052-b](https://doi.org/10.1016/0003-682x(95)00052-b)

**[XREF0205]** Cheng, Jing-Jing; Zhang, Xiao-Pai; Liu, Yan; et al. (2017). Two Methods of Measuring Reverberation Time and Sound Absorption Coefficient in Reverberation Room. *Materials in Environmental Engineering*. DOI: [10.1515/9783110516623-135](https://doi.org/10.1515/9783110516623-135)

**[XREF0791]** Chin-Bing, Stanley A.; Murphy, Joseph E. (1994). Computer model simulations of reverberation from the ARSRP acoustics experiment. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.410736](https://doi.org/10.1121/1.410736)

**[XREF0220]** Cho, Hannah (2024). Neural Network Architectures for Simulating Time-Varying Room Acoustics. *2024 Conference on AI, Science, Engineering, and Technology (AIxSET)*. DOI: [10.1109/aixset62544.2024.00052](https://doi.org/10.1109/aixset62544.2024.00052)

**[XREF0786]** Cho, Wan-Ho; Ih, Jeong-Guon; Katsumata, Tomohisa; et al. (2018). Best practice for positioning sound absorbers at room surface. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2017.08.015](https://doi.org/10.1016/j.apacoust.2017.08.015)

**[XREF0295]** Chu, Kevin; Collins, Leslie; Mainsah, Boyla (2022). Suppressing reverberation in cochlear implant stimulus patterns using time-frequency masks based on phoneme groups. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0001698](https://doi.org/10.1121/2.0001698)

**[XREF0709]** Chu, Kevin; Throckmorton, Chandra; Collins, Leslie; et al. (2018). Using machine learning to mitigate the effects of reverberation and noise in cochlear implants. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0000905](https://doi.org/10.1121/2.0000905)

**[XREF0602]** Chu, W. T. (1978). Pure-tone measurements in a reverberation room. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2004370](https://doi.org/10.1121/1.2004370)

**[XREF0827]** Chu, W. T. (1982). Near and farfield transfer-function technique for reverberation room response studies. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2019759](https://doi.org/10.1121/1.2019759)

**[XREF0869]** Chu, W. T. (1985). Room response measurements in a reverberation chamber containing a rotating diffuser. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.392195](https://doi.org/10.1121/1.392195)

**[XREF0221]** Chu, W.T. (1990). Impulse-response and reverberation-decay measurements made by using a periodic pseudorandom sequence. *Applied Acoustics*. DOI: [10.1016/0003-682x(90)90018-p](https://doi.org/10.1016/0003-682x(90)90018-p)

**[XREF0792]** Chu, Wing T. (1998). Reverberation room qualification using the m-sequence as multitone signals. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.424496](https://doi.org/10.1121/1.424496)

**[XREF0479]** Chun, I; Rafaely, B; Joseph, P (2023). Investigation of the Broadband Spatial Correlation in a Large Reverberation Chamber. *Past, Present and Future Acoustics and EPSRC Day*. DOI: [10.25144/18211](https://doi.org/10.25144/18211)

**[XREF0140]** Ćirić, D.; Puljizević, M.; Pantić, A.; et al. (2022). Energy Decay Curve Deviation in the Absorption Coefficient Measurement in a Small Reverberation Room. *Proceedings of the 10th Convention of the European Acoustics Association Forum Acusticum 2023*. DOI: [10.61782/fa.2023.0891](https://doi.org/10.61782/fa.2023.0891)

**[XREF0699]** Ćirić, Dejan G.; Janković, Marko (2018). Correction of room impulse response truncation based on a nonlinear decay model. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2017.11.018](https://doi.org/10.1016/j.apacoust.2017.11.018)

**[XREF0815]** Ćirić, Dejan G.; Milošević, Miroslava A. (2005). Optimal Determination of the Truncation Point of Room Impulse Responses. *Building Acoustics*. DOI: [10.1260/1351010053499216](https://doi.org/10.1260/1351010053499216)

**[XREF0174]** Clarke, Joseph L. (2015). Catacoustic Enchantment: The Romantic Conception of Reverberation. *Grey Room*. DOI: [10.1162/grey_a_00175](https://doi.org/10.1162/grey_a_00175)

**[XREF0029]** Coleman, P; Jackson, PJB (2024). Planarity Analysis of Room Acoustics for Object-Based Reverberation. *24th International Conference on Sound and Vibration 2017, London Calling*. DOI: [10.25144/24452](https://doi.org/10.25144/24452)

**[XREF0759]** Cops, A.; Soubrier, D. (1988). Sound transmission loss of glass and windows in laboratories with different room design. *Applied Acoustics*. DOI: [10.1016/0003-682x(88)90061-8](https://doi.org/10.1016/0003-682x(88)90061-8)

**[XREF0021]** Cops, A.; Vanhaecht, J.; Leppens, K. (1995). Sound absorption in a reverberation room: Causes of discrepancies on measurement results. *Applied Acoustics*. DOI: [10.1016/0003-682x(95)00029-9](https://doi.org/10.1016/0003-682x(95)00029-9)

**[XREF0069]** Costa-Felix, Rodrigo P.B. (2016). Measurement Precision Under Repeatability Conditions of a Batch of Sound Power Assessment for Blenders in Reverberation Room. *Archives of Acoustics*. DOI: [10.1515/aoa-2016-0057](https://doi.org/10.1515/aoa-2016-0057)

**[XREF0370]** Cotana, Franco (2000). An improved room acoustic model. *Applied Acoustics*. DOI: [10.1016/s0003-682x(99)00074-2](https://doi.org/10.1016/s0003-682x(99)00074-2)

**[XREF0676]** Cowell, R (2024). Some Practical Aspects of Sound Conditioning. *Room Acoustics with Emphasis on Electroacoustics 1979*. DOI: [10.25144/23455](https://doi.org/10.25144/23455)

**[XREF0856]** COX, TJ (2024). Room Acoustics Diffusers: Pyramids and Wedges. *24th International Conference on Sound and Vibration 2017, London Calling*. DOI: [10.25144/24657](https://doi.org/10.25144/24657)

**[XREF0505]** Crighton, D. G.; Dowling, A. P.; Williams, J. E. Ffowcs; et al. (1992). Reverberation. *Modern Methods in Analytical Acoustics*. DOI: [10.1007/978-1-4471-0399-8_22](https://doi.org/10.1007/978-1-4471-0399-8_22)

**[XREF0584]** Cucharero, Jose; Hänninen, Tuomas; Lokki, Tapio (2019). Influence of Sound-Absorbing Material Placement on Room Acoustical Parameters. *Acoustics*. DOI: [10.3390/acoustics1030038](https://doi.org/10.3390/acoustics1030038)

**[XREF0468]** Culver, R.L.; McDaniel, S.T. (1991). Bistatic ocean surface reverberation simulation. *[Proceedings] ICASSP 91: 1991 International Conference on Acoustics, Speech, and Signal Processing*. DOI: [10.1109/icassp.1991.150715](https://doi.org/10.1109/icassp.1991.150715)

**[XREF0463]** D’Antonio, Peter; Cox, Trevor J. (2002). The development and use of the diffusion scattering coefficients in room modeling software: The effect of diffuse reflections on reverberation time. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4778120](https://doi.org/10.1121/1.4778120)

**[XREF0867]** Dance, S.M.; Shield, B.M. (1999). Modelling of sound fields in enclosed spaces with absorbent room surfaces. Part I: performance spaces. *Applied Acoustics*. DOI: [10.1016/s0003-682x(98)00064-4](https://doi.org/10.1016/s0003-682x(98)00064-4)

**[XREF0863]** Dance, S.M.; Shield, B.M. (2000). Modelling of sound fields in enclosed spaces with absorbent room surfaces Part II. Absorptive panels. *Applied Acoustics*. DOI: [10.1016/s0003-682x(00)00011-6](https://doi.org/10.1016/s0003-682x(00)00011-6)

**[XREF0826]** Dance, S.M; Shield, B.M (2000). Modelling of sound fields in enclosed spaces with absorbent room surfaces Part III. Barriers. *Applied Acoustics*. DOI: [10.1016/s0003-682x(00)00012-8](https://doi.org/10.1016/s0003-682x(00)00012-8)

**[XREF0846]** Darabundit, Champ C.; Chatziioannou, Vasileios; Scavone, Gary (2025). Mapping woodwind playability using measurement based physical models. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002112](https://doi.org/10.1121/2.0002112)

**[XREF0804]** Darlington, P; Avis, MR (2024). Modifying Low Frequency Room Acoustics 2: Global Control using Active Absorbers. *RS 11*. DOI: [10.25144/20137](https://doi.org/10.25144/20137)

**[XREF0762]** Davy, J.L. (1981). The relative variance of the transmission function of a reverberation room. *Journal of Sound and Vibration*. DOI: [10.1016/s0022-460x(81)80044-2](https://doi.org/10.1016/s0022-460x(81)80044-2)

**[XREF0742]** Davy, J.L. (1986). The ensemble variance of random noise in a reverberation room. *Journal of Sound and Vibration*. DOI: [10.1016/s0022-460x(86)80113-4](https://doi.org/10.1016/s0022-460x(86)80113-4)

**[XREF0574]** Davy, J.L.; Davern, W.A.; Dubout, P. (1989). Qualification of room diffusion for absorption measurements. *Applied Acoustics*. DOI: [10.1016/0003-682x(89)90092-3](https://doi.org/10.1016/0003-682x(89)90092-3)

**[XREF0197]** de Haas, Kevin; Schutte, Michael; Ewert, Stephan (2025). Real-time Virtual Environment and Room Acoustics Simulator. *Proceedings of the 11th Convention of the European Acoustics Association Forum Acusticum / EuroNoise 2025*. DOI: [10.61782/fa.2025.0729](https://doi.org/10.61782/fa.2025.0729)

**[XREF0691]** de la Prida, Daniel; Pedrero, Antonio; Azpicueta-Ruiz, Luis Antonio; et al. (2021). Listening tests in room acoustics: Comparison of overall difference protocols regarding operational power. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2021.108186](https://doi.org/10.1016/j.apacoust.2021.108186)

**[XREF0333]** de Leon, Phillip L.; Trevizo, Audrey L. (2007). Speaker Identification in the Presence of Room Reverberation. *2007 Biometrics Symposium*. DOI: [10.1109/bcc.2007.4430533](https://doi.org/10.1109/bcc.2007.4430533)

**[XREF0442]** de M. Prego, Thiago; de Lima, Amaro A.; Zambrano-Lopez, Rafael; et al. (2015). Blind estimators for reverberation time and direct-to-reverberant energy ratio using subband speech decomposition. *2015 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)*. DOI: [10.1109/waspaa.2015.7336954](https://doi.org/10.1109/waspaa.2015.7336954)

**[XREF0662]** Dedousis, G; Bakogiannis, K; Andreopoulou, A; et al. (2023). Room Acoustics Mismatches of Rehearsal Spaces and Concert Halls and Their Impact on Music Performance. *Auditorium Acoustics 2023*. DOI: [10.25144/15985](https://doi.org/10.25144/15985)

**[XREF0807]** Defrance, G; Daudet, L; Polack, J-D (2023). Characterizing Sound Sources for Room-Acoustical Measurements. *Auditorium Acoustics 2008*. DOI: [10.25144/17491](https://doi.org/10.25144/17491)

**[XREF0439]** Defrance, Guillaume; Polack, Jean-Dominique (2015). Estimating the Crossover Time Within Room Impulse Responses. *Acoustics, Information, and Communication*. DOI: [10.1007/978-3-319-05660-9_4](https://doi.org/10.1007/978-3-319-05660-9_4)

**[XREF0673]** Degraeve, Andy; Pissoort, Davy (2016). Study of the effectiveness of spatially EM-diverse redundant systems under reverberation room conditions. *2016 IEEE International Symposium on Electromagnetic Compatibility (EMC)*. DOI: [10.1109/isemc.2016.7571676](https://doi.org/10.1109/isemc.2016.7571676)

**[XREF0582]** Dekker, H. (1974). EDGE effect measurements in a reverberation room. *Journal of Sound and Vibration*. DOI: [10.1016/s0022-460x(74)80164-1](https://doi.org/10.1016/s0022-460x(74)80164-1)

**[XREF0214]** Delsasso, L. P.; Leonard, R. W.; Knudsen, V. O. (1966). Reverberation-Room Acoustics. II: Effects of Absorptive Floor, Suspended Ceiling Panels, and Diffusers on Speech and Music. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.1943131](https://doi.org/10.1121/1.1943131)

**[XREF0202]** Denison, Michael H.; Anderson, Brian E. (2018). Time reversal acoustics applied to rooms of various reverberation times. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.5080560](https://doi.org/10.1121/1.5080560)

**[XREF0569]** Deželak, Ferdinand; Čurović, Luka; Čudina, Mirko (2016). Determination of the sound energy level of a gunshot and its applications in room acoustics. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2015.12.001](https://doi.org/10.1016/j.apacoust.2015.12.001)

**[XREF0811]** Dibble, K (2024). Single Figure Transmission Loss Ratings for De-Mountable Room Partitions - Are They Workable?. *Acoustics '88*. DOI: [10.25144/21787](https://doi.org/10.25144/21787)

**[XREF0563]** Dick, David A.; Vigeant, Michelle C. (2016). A comparison of measured room acoustics metrics using a spherical microphone array and conventional methods. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2016.01.008](https://doi.org/10.1016/j.apacoust.2016.01.008)

**[XREF0830]** Didier, Paul; Van hoorickx, Cédric; Reynders, Edwin (2021). An optimization method for reverberation room design. *INTER-NOISE and NOISE-CON Congress and Conference Proceedings*. DOI: [10.3397/in-2021-2118](https://doi.org/10.3397/in-2021-2118)

**[XREF0312]** Didier, Paul; Van hoorickx, Cédric; Reynders, Edwin (2024). Reverberation Room Design Optimisation for Low-Frequency Diffuse Sound Absorption Testing. *Unspecified venue*. DOI: [10.2139/ssrn.4737589](https://doi.org/10.2139/ssrn.4737589)

**[XREF0371]** Didier, Paul; Van hoorickx, Cédric; Reynders, Edwin P.B. (2022). Numerical study of the accuracy and reproducibility of sound absorption measurements in reverberation rooms at low frequencies. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2022.109047](https://doi.org/10.1016/j.apacoust.2022.109047)

**[XREF0020]** Didier, Paul; Van hoorickx, Cédric; Reynders, Edwin P.B. (2025). Reverberation room design optimisation for low-frequency diffuse sound absorption testing. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2024.110311](https://doi.org/10.1016/j.apacoust.2024.110311)

**[XREF0232]** Diether, Salomon; Bruderer, Lukas; Streich, Andreas; et al. (2015). Efficient blind estimation of subband reverberation time from speech in non-diffuse environments. *2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2015.7178068](https://doi.org/10.1109/icassp.2015.7178068)

**[XREF0018]** Dikshit, H.K.; Dikshit, K.D. (1979). Holographic measurement of reverberation time. *Applied Acoustics*. DOI: [10.1016/0003-682x(79)90036-7](https://doi.org/10.1016/0003-682x(79)90036-7)

**[XREF0805]** Don; Davis, Carolyn (2010). Large Room Acoustics. *Audio Engineering Explained- for professional audio recording*. DOI: [10.1016/b978-0-240-81273-1.00017-7](https://doi.org/10.1016/b978-0-240-81273-1.00017-7)

**[XREF0775]** Dougherty, Robert P.; Fonseca, William D’Andrea; Gerges, Samir N. Y. (2008). Beamforming in Reflecting Environments: An Experiment in a Reverberation Chamber. *ASME 2008 Noise Control and Acoustics Division Conference*. DOI: [10.1115/ncad2008-73020](https://doi.org/10.1115/ncad2008-73020)

**[XREF0002]** Dragonetti, Raffaele; Ianniello, Carmine; Romano, Rosario A. (2009). Reverberation time measurement by the product of two room impulse responses. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2007.12.001](https://doi.org/10.1016/j.apacoust.2007.12.001)

**[XREF0129]** Drechsler, Stefan; Stephenson, Uwe M. (2013). The effect of edge caused diffusion on the reverberation time - A semi analytical approach. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4799960](https://doi.org/10.1121/1.4799960)

**[XREF0485]** Duangpummet, Suradej; Karnjana, Jessada; Kongprawechnon, Waree; et al. (2022). Blind estimation of speech transmission index and room acoustic parameters based on the extended model of room impulse response. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2021.108372](https://doi.org/10.1016/j.apacoust.2021.108372)

**[XREF0177]** Duanqi, Xiang; Zheng, Wang; Jinjing, Chen (1991). Acoustic design of a reverberation chamber. *Applied Acoustics*. DOI: [10.1016/0003-682x(91)90050-o](https://doi.org/10.1016/0003-682x(91)90050-o)

**[XREF0641]** Duarte, EAC; Moorhouse, A (2023). An Alternative to Reverberation Time Measurement for Sound Insulation Testing. *See it Hear!*. DOI: [10.25144/17635](https://doi.org/10.25144/17635)

**[XREF0581]** Dujourdy, H; Toulemonde, T (2023). Rebuilding of an Orchestra Rehearsal Room Comparison Between Objective and Perceptive Measurements for Room Acoustic Predictions. *Auditorium Acoustics 2015*. DOI: [10.25144/16142](https://doi.org/10.25144/16142)

**[XREF0006]** Dumortier, Baldwin; Vincent, Emmanuel (2014). Blind RT60 estimation robust across room sizes and source distances. *2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2014.6854592](https://doi.org/10.1109/icassp.2014.6854592)

**[XREF0755]** Dunbar, J. Y.; LeBel, C. J. (1951). Room Reverberation Study by Ultra Speed Recording. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.1917352](https://doi.org/10.1121/1.1917352)

**[XREF0263]** Dunham, Joshua R. (2023). Comparing measured sound strength to theory as a function of reverberation time and room volume. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0018160](https://doi.org/10.1121/10.0018160)

**[XREF0046]** Eargle, John M. (1994). Estimation of Room Constant When Room Volume and Reverberation Time are Known. *Electroacoustical Reference Data*. DOI: [10.1007/978-1-4615-2027-6_27](https://doi.org/10.1007/978-1-4615-2027-6_27)

**[XREF0092]** Eargle, John M. (1994). Estimation of Total Absorption When Room Volume and Reverberation Time are Known. *Electroacoustical Reference Data*. DOI: [10.1007/978-1-4615-2027-6_26](https://doi.org/10.1007/978-1-4615-2027-6_26)

**[XREF0779]** Eargle, John M. (1994). Optimum Reverberation Time as a Function of Frequency. *Electroacoustical Reference Data*. DOI: [10.1007/978-1-4615-2027-6_147](https://doi.org/10.1007/978-1-4615-2027-6_147)

**[XREF0075]** Eargle, John M. (1994). Optimum Reverberation Time as a Function of Room Volume and Usage. *Electroacoustical Reference Data*. DOI: [10.1007/978-1-4615-2027-6_146](https://doi.org/10.1007/978-1-4615-2027-6_146)

**[XREF0749]** Eastland, Grant C.; Buck, William C. (2016). Reverberation characterization inside an anechoic test chamber at the Weapon Sonar Test Facility at NUWC Division Keyport. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0000508](https://doi.org/10.1121/2.0000508)

**[XREF0215]** Eaton, James; Gaubitch, Nikolay D.; Naylor, Patrick A. (2013). Noise-robust reverberation time estimation using spectral decay distributions with reduced computational cost. *2013 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2013.6637629](https://doi.org/10.1109/icassp.2013.6637629)

**[XREF0540]** Ebbitt, Gordon L. (2000). Reverberation room for acoustical testing. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.428382](https://doi.org/10.1121/1.428382)

**[XREF0685]** Economou, P; Economou, C; Charalampous, P (2024). Beyond Sabine: Investigating the Acoustical Phenomenon of Reverberation using Room Modal Decay. *24th International Conference on Sound and Vibration 2017, London Calling*. DOI: [10.25144/24456](https://doi.org/10.25144/24456)

**[XREF0464]** Edwards, Nicholas A. (1983). Music performance acoustics and room shape: An investigation employing an images model of room acoustics. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2021178](https://doi.org/10.1121/1.2021178)

**[XREF0292]** Elkhateeb, Ahmed (2019). What should the reverberation inside a masjid be?. *Worship Sound Spaces*. DOI: [10.4324/9780429279782-6](https://doi.org/10.4324/9780429279782-6)

**[XREF0406]** Elliott, A. (2019). Measurement of in-room impact noise reduction. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2018.12.020](https://doi.org/10.1016/j.apacoust.2018.12.020)

**[XREF0666]** Ellis, Dale D.; Franklin, J. B. (1987). The Importance of Hybrid Ray Paths, Bottom Loss, and Facet Reflection on Ocean Bottom Reverberation. *Progress in Underwater Acoustics*. DOI: [10.1007/978-1-4613-1871-2_10](https://doi.org/10.1007/978-1-4613-1871-2_10)

**[XREF0420]** Ellis, Dale D.; Hefner, Brian T.; Tang, Dajun; et al. (2021). A look at reverberation and target echo on a vertical array during the Target and Reverberation Experiment. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0001521](https://doi.org/10.1121/2.0001521)

**[XREF0136]** Ellis, Dale D.; Yang, Jie (2021). Comparison of range-dependent reverberation model predictions with array data from the 2013 Target and Reverberation Experiment. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0001499](https://doi.org/10.1121/2.0001499)

**[XREF0440]** Ellison, S; Poletti, M (2023). Control of Room Acoustic Parameters by the Variable Room Acoustics System (Vras). *Improving the Listening Experience*. DOI: [10.25144/18059](https://doi.org/10.25144/18059)

**[XREF0606]** Ellison, S; Poletti, M (2024). Variable Room Acoustics System Philosophy and Application. *Reproduced Sound 2000*. DOI: [10.25144/18679](https://doi.org/10.25144/18679)

**[XREF0060]** Embrechts, Jean-Jacques (2013). The contributions of pairs of parallel surfaces in a simple analytical model of room reverberation. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4798979](https://doi.org/10.1121/1.4798979)

**[XREF0032]** Embrechts, Jean-Jacques (2014). A Geometrical Acoustics Approach Linking Surface Scattering and Reverberation in Room Acoustics. *Acta Acustica united with Acustica*. DOI: [10.3813/aaa.918766](https://doi.org/10.3813/aaa.918766)

**[XREF0111]** Erkelens, Jan S.; Heusdens, Richard (2010). Noise and late-reverberation suppression in time-varying acoustical environments. *2010 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2010.5495178](https://doi.org/10.1109/icassp.2010.5495178)

**[XREF0763]** Ermann, Michael (2005). Coupled Volumes: Secondary Room Reverberance and the Double-Sloped Decay of Concert Halls. *Building Acoustics*. DOI: [10.1260/135101005774353069](https://doi.org/10.1260/135101005774353069)

**[XREF0134]** Escolano, José; Spa, Carlos; Garriga, Adán; et al. (2013). Removal of afterglow effects in 2-D discrete-time room acoustics simulations. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2012.12.012](https://doi.org/10.1016/j.apacoust.2012.12.012)

**[XREF0841]** Evans, T. Brandon (2022). Echoes in the capitol, echoes in history: architectural acoustics, media archaeology, and the infrapolitics of reverberation. *Sound Studies*. DOI: [10.1080/20551940.2022.2058765](https://doi.org/10.1080/20551940.2022.2058765)

**[XREF0738]** Exton, P; Marshall, H (2023). The Room Acoustic Design of the Guangzhou Opera House. *Auditorium Acoustics 2011*. DOI: [10.25144/16853](https://doi.org/10.25144/16853)

**[XREF0065]** Farhat, Youssef; Bustillo, Julien; Achdjian, Hossep; et al. (2022). Acoustic reverberation time determination in solid medium. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2022.108958](https://doi.org/10.1016/j.apacoust.2022.108958)

**[XREF0116]** Fasllija, E.; Yilmazer, S. (2024). Measurement of the absorption coefficient of inhomogeneous Micro-Perforated Panels in a Small-Scale Reverberation Room. *Proceedings of the 10th Convention of the European Acoustics Association Forum Acusticum 2023*. DOI: [10.61782/fa.2023.0943](https://doi.org/10.61782/fa.2023.0943)

**[XREF0127]** Feistel, Stefan; Ahnert, Wolfgang (2025). Complex acoustic simulations in real time. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002187](https://doi.org/10.1121/2.0002187)

**[XREF0571]** Felis, Józef; Flach, Artur; Kamisiński, Tadeusz (2012). Testing of a Device for Positioning Measuring Microphones in Anechoic and Reverberation Chambers. *Archives of Acoustics*. DOI: [10.2478/v10168-012-0032-5](https://doi.org/10.2478/v10168-012-0032-5)

**[XREF0645]** Feuillade, C (2023). Babinet's Principle in Acoustics a Time-Domain Reappraisal. *Acoustics 2023*. DOI: [10.25144/16579](https://doi.org/10.25144/16579)

**[XREF0013]** Filho, José Nivaldo Sarinho; Thomazelli, Rodolfo; Masiero, Bruno Sanches (2025). Optimizing low-frequency reverberation in a critical listening room. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002068](https://doi.org/10.1121/2.0002068)

**[XREF0532]** Fink, Daniel (2024). A comprehensive new definition of Noise. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002057](https://doi.org/10.1121/2.0002057)

**[XREF0408]** Fleming, Patrick H. (2024). The Historical Building and Room Acoustics of the Stockholm Public Library (1925–28, 1931–32). *Acoustics*. DOI: [10.3390/acoustics6030041](https://doi.org/10.3390/acoustics6030041)

**[XREF0024]** Fothergill, L.C. (1982). An investigation of simple methods for assessing reverberation time. *Applied Acoustics*. DOI: [10.1016/0003-682x(82)90012-3](https://doi.org/10.1016/0003-682x(82)90012-3)

**[XREF0823]** Foulkes, Timothy (2013). Coping with curves in room design. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4801403](https://doi.org/10.1121/1.4801403)

**[XREF0783]** Frank, Scott D.; Ivakin, Anatoliy N. (2017). Estimating the influence of ice thickness and elasticity on long-range narrow-band reverberation in an Arctic environment. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0000595](https://doi.org/10.1121/2.0000595)

**[XREF0842]** Frank, Scott D.; Ivakin, Anatoliy N. (2018). Low-frequency reverberation estimates based on elastic parabolic equation solutions for free-surface and ice-covered Arctic environments. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0000990](https://doi.org/10.1121/2.0000990)

**[XREF0242]** Fuchs, H.V; Zha, X; Pommerer, M (2000). Qualifying freefield and reverberation rooms for frequencies below 100 Hz. *Applied Acoustics*. DOI: [10.1016/s0003-682x(99)00038-9](https://doi.org/10.1016/s0003-682x(99)00038-9)

**[XREF0757]** Furue, Yoshihiro (1990). Sound propagation from the inside to the outside of a room through an aperture. *Applied Acoustics*. DOI: [10.1016/0003-682x(90)90057-2](https://doi.org/10.1016/0003-682x(90)90057-2)

**[XREF0101]** G.S., Pushpalatha; Shivaputra; N., Mohan Kumar (2012). Spectrogram Study of Echo and Reverberation– A Novel Approach to Reduce Echo and Reverberation in Room Acoustics. *Lecture Notes of the Institute for Computer Sciences, Social Informatics and Telecommunications Engineering*. DOI: [10.1007/978-3-642-32573-1_36](https://doi.org/10.1007/978-3-642-32573-1_36)

**[XREF0070]** Galbrun, Laurent; Kitapci, Kivanc (2014). Accuracy of speech transmission index predictions based on the reverberation time and signal-to-noise ratio. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2014.02.001](https://doi.org/10.1016/j.apacoust.2014.02.001)

**[XREF0722]** Galbrun, Laurent; Kitapci, Kivanc (2016). Speech intelligibility of English, Polish, Arabic and Mandarin under different room acoustic conditions. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2016.07.003](https://doi.org/10.1016/j.apacoust.2016.07.003)

**[XREF0612]** Gan, Woon Siong (2021). Application of Time-Reversal Acoustics to Seismic Exploration. *Time Reversal Acoustics*. DOI: [10.1007/978-981-16-3235-8_13](https://doi.org/10.1007/978-981-16-3235-8_13)

**[XREF0724]** Gan, Woon Siong (2021). Application of Time-Reversal Acoustics to Signal Processing and Underwater Communications. *Time Reversal Acoustics*. DOI: [10.1007/978-981-16-3235-8_12](https://doi.org/10.1007/978-981-16-3235-8_12)

**[XREF0702]** Gan, Woon Siong (2021). Application of Time-Reversal Acoustics to Ultrasound Non-destructive Testing. *Time Reversal Acoustics*. DOI: [10.1007/978-981-16-3235-8_14](https://doi.org/10.1007/978-981-16-3235-8_14)

**[XREF0475]** Gan, Woon Siong (2021). Linear Time-Reversal Acoustics. *Time Reversal Acoustics*. DOI: [10.1007/978-981-16-3235-8_3](https://doi.org/10.1007/978-981-16-3235-8_3)

**[XREF0750]** Gan, Woon Siong (2021). Non-reciprocal Acoustics. *Time Reversal Acoustics*. DOI: [10.1007/978-981-16-3235-8_5](https://doi.org/10.1007/978-981-16-3235-8_5)

**[XREF0470]** Gan, Woon Siong (2021). Nonlinear Time-Reversal Acoustics. *Time Reversal Acoustics*. DOI: [10.1007/978-981-16-3235-8_4](https://doi.org/10.1007/978-981-16-3235-8_4)

**[XREF0511]** Gan, Woon Siong (2021). Time-Reversal Acoustics and Superresolution. *Time Reversal Acoustics*. DOI: [10.1007/978-981-16-3235-8_8](https://doi.org/10.1007/978-981-16-3235-8_8)

**[XREF0064]** Ganapathy, Sriram; Pelecanos, Jason; Omar, Mohamed Kamal (2011). Feature normalization for speaker verification in room reverberation. *2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2011.5947438](https://doi.org/10.1109/icassp.2011.5947438)

**[XREF0667]** Gang, Ren; Bocko, Mark F.; Headlam, Dave (2010). Reverberation features identification from music recordings using the discrete wavelet transform. *2010 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2010.5496094](https://doi.org/10.1109/icassp.2010.5496094)

**[XREF0521]** GAO, T. F.; Shang, E. C. (2006). The Optimum Source Depth Distribution for Reverberation Inversion in a Shallow-Water Waveguide. *Theoretical and Computational Acoustics 2005*. DOI: [10.1142/9789812772602_0006](https://doi.org/10.1142/9789812772602_0006)

**[XREF0599]** García-Barrios, Guillermo; Latorre Iglesias, Eduardo; Gutiérrez-Arriola, Juana M.; et al. (2023). Exploiting spatial diversity for increasing the robustness of sound source localization systems against reverberation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2022.109138](https://doi.org/10.1016/j.apacoust.2022.109138)

**[XREF0682]** Gastmeier, William J. (2024). Teaching concepts of acoustics in air—Part 3, reverberation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0027120](https://doi.org/10.1121/10.0027120)

**[XREF0777]** Ge, H. L.; Zhao, H. F.; Gong, X. Y.; et al. (2004). Bottom reflection phase shift parameter inversion from reverberation and propagation data. *Theoretical and Computational Acoustics 2003*. DOI: [10.1142/9789812702609_0010](https://doi.org/10.1142/9789812702609_0010)

**[XREF0723]** Gelfand, S. A. (1977). Room reverberation effects on recognition of some consonant features. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2016388](https://doi.org/10.1121/1.2016388)

**[XREF0325]** Gelvez-Barrera, Tatiana; Leclere, Quentin; Nicolas, Barbara; et al. (2025). Time-domain Beamforming for Room Acoustics Analysis based on Reverberant Field Estimation. *ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp49660.2025.10889527](https://doi.org/10.1109/icassp49660.2025.10889527)

**[XREF0203]** Gething, MR (2024). Computer Simulation of Reverberation from a One Dimensional Seabed. *Scattering Phenomena in Underwater Acoustics 1985*. DOI: [10.25144/23566](https://doi.org/10.25144/23566)

**[XREF0482]** Gibbs, B. (2010). Collected Papers in Building Acoustics: Room Acoustics and Environmental Noise. *Noise Control Engineering Journal*. DOI: [10.3397/1.3491149](https://doi.org/10.3397/1.3491149)

**[XREF0545]** Gilbert, E. (1982). Reverberation between concentric spheres. *IEEE Transactions on Acoustics, Speech, and Signal Processing*. DOI: [10.1109/tassp.1982.1163889](https://doi.org/10.1109/tassp.1982.1163889)

**[XREF0212]** Gilford, C.L.S. (1974). Room acoustics. *Applied Acoustics*. DOI: [10.1016/0003-682x(74)90019-x](https://doi.org/10.1016/0003-682x(74)90019-x)

**[XREF0542]** Gillespie, David (2025). Quantitative comparison of various cajon models. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002060](https://doi.org/10.1121/2.0002060)

**[XREF0847]** Giordano, Nicholas (2025). An approach to describing sound production at the mouth of a wind instrument. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002168](https://doi.org/10.1121/2.0002168)

**[XREF0650]** Giordano, Nicholas (2025). Structure of air flow near a woodwind tone hole. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002054](https://doi.org/10.1121/2.0002054)

**[XREF0154]** Giri, Ritwik; Seltzer, Michael L.; Droppo, Jasha; et al. (2015). Improving speech recognition in reverberation using a room-aware deep neural network and multi-task learning. *2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2015.7178925](https://doi.org/10.1109/icassp.2015.7178925)

**[XREF0359]** Giuliano, H.G.; Velis, A.G.; Méndez, A.M. (1996). The reverberation chamber at the Laboratorio de Acústica y Luminotecnia of the Comisión de Investigaciones Científicas. *Applied Acoustics*. DOI: [10.1016/0003-682x(95)00054-d](https://doi.org/10.1016/0003-682x(95)00054-d)

**[XREF0686]** Godfrey, Richard D.; Feth, Lawrence L.; Winch, Peter (2011). Benchmark measurements of noise, reverberation time, and an estimate of speech intelligibility in a representative operating room at Nationwide Children's Hospital in Columbus, Ohio. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.3654263](https://doi.org/10.1121/1.3654263)

**[XREF0864]** Goehle, Geoff; Cowen, Benjamin (2025). Out of background distribution detection for sonar imagery using variational autoencoding. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002222](https://doi.org/10.1121/2.0002222)

**[XREF0379]** Goertner, Jean A. (1980). Computer Model Predictions of Ocean Basin Reverberation for Large Underwater Explosions. *Bottom-Interacting Ocean Acoustics*. DOI: [10.1007/978-1-4684-9051-0_40](https://doi.org/10.1007/978-1-4684-9051-0_40)

**[XREF0681]** Goldhahn, Ryan; Hickman, Granger; Krolik, Jeffery L. (2007). Waveguide Invariant Reverberation Mitigation for Active Sonar. *2007 IEEE International Conference on Acoustics, Speech and Signal Processing - ICASSP '07*. DOI: [10.1109/icassp.2007.366392](https://doi.org/10.1109/icassp.2007.366392)

**[XREF0056]** Gómez Escobar, V.; Barrigón Morillas, J.M. (2015). Analysis of intelligibility and reverberation time recommendations in educational rooms. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2015.03.001](https://doi.org/10.1016/j.apacoust.2015.03.001)

**[XREF0201]** Gotz, Philipp; Tuna, Cagdas; Walther, Andreas; et al. (2022). Blind Reverberation Time Estimation in Dynamic Acoustic Conditions. *ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp43922.2022.9746457](https://doi.org/10.1109/icassp43922.2022.9746457)

**[XREF0100]** Gou, Wenbo; Liang, Hong; Li, Hui; et al. (2026). Shallow water reverberation suppression based on a time-frequency patch tensor model for broadband active sonar systems. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2025.111144](https://doi.org/10.1016/j.apacoust.2025.111144)

**[XREF0323]** Goydke, Hans (1997). New international standards for building and room acoustics. *Applied Acoustics*. DOI: [10.1016/s0003-682x(97)00045-5](https://doi.org/10.1016/s0003-682x(97)00045-5)

**[XREF0587]** Gramez, Abdelghani; Boubenider, Fouad (2017). Acoustic comfort evaluation for a conference room: A case study. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2016.11.014](https://doi.org/10.1016/j.apacoust.2016.11.014)

**[XREF0627]** Greenberg, J.E.; Zurek, P.M. (1991). Adaptive Beamformer Performance In Reverberation. *Final Program and Paper Summaries 1991 IEEE ASSP Workshop on Applications of Signal Processing to Audio and Acoustics*. DOI: [10.1109/aspaa.1991.634121](https://doi.org/10.1109/aspaa.1991.634121)

**[XREF0520]** Greenblatt, Aaron B.; Abel, Jonathan S.; Berners, David P. (2010). A hybrid reverberation crossfading technique. *2010 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2010.5495753](https://doi.org/10.1109/icassp.2010.5495753)

**[XREF0198]** Griesinger, D (2024). Further Investigation into the Loudness of Running Reverberation. *Opera and Concert Hall Acoustics 1995*. DOI: [10.25144/20019](https://doi.org/10.25144/20019)

**[XREF0367]** Gualtierotti; Climescu-Haulica; Pal (2002). Likelihood ratio detection of signals on reverberation noise. *IEEE International Conference on Acoustics Speech and Signal Processing*. DOI: [10.1109/icassp.2002.1006061](https://doi.org/10.1109/icassp.2002.1006061)

**[XREF0604]** Gualtierotti, A.F.; Climescu-Haulica, A.; Pal, M.D. (2002). Likelihood ratio detection of signals on reverberation noise. *IEEE International Conference on Acoustics Speech and Signal Processing*. DOI: [10.1109/icassp.2002.5744920](https://doi.org/10.1109/icassp.2002.5744920)

**[XREF0044]** Guidorzi, Paolo; Pompoli, Francesco; Bonfiglio, Paolo; et al. (2020). A newly developed low-cost 3D acoustic positioning system: Description and application in a reverberation room. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2019.107127](https://doi.org/10.1016/j.apacoust.2019.107127)

**[XREF0107]** Guillen, Ignacio; Llopis, Ana; Uris, Antonio (2002). Adequate Reverberation Time for Brass Band Auditoriums. *The International Journal of Acoustics and Vibration*. DOI: [10.20855/ijav.2002.7.2109](https://doi.org/10.20855/ijav.2002.7.2109)

**[XREF0727]** Gulsrud, TE (2023). Variation of Energy-Based Room Acoustic Parameters Withing a Multipurpose Hall. *Auditorium Acoustics 2002*. DOI: [10.25144/18335](https://doi.org/10.25144/18335)

**[XREF0446]** Guoqiang, Guo; Yixin, Yang; Chao, Sun (2008). Time Reversal Echo-to-reverberation Enhancement with Reverberation Nulling Constraints Based on Waveguide Invariant. *OCEANS 2008 - MTS/IEEE Kobe Techno-Ocean*. DOI: [10.1109/oceanskobe.2008.4531093](https://doi.org/10.1109/oceanskobe.2008.4531093)

**[XREF0544]** Hacıhabiboğlu, Hüseyin; Murtagh, Fionn (2008). Perceptual simplification for model-based binaural room auralisation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2007.02.006](https://doi.org/10.1016/j.apacoust.2007.02.006)

**[XREF0433]** Halmrast, T; Buen, A (2023). Simplified Room Acoustic Measurements. *Auditorium Acoustics 2008*. DOI: [10.25144/17523](https://doi.org/10.25144/17523)

**[XREF0102]** Hamilton, Brian (2021). 2pCA2 - Tutorial on finite-difference time-domain (FDTD) methods for room acoustics simulation. *Unspecified venue*. DOI: [10.26226/morressier.606f15dd30a2e980041f2468](https://doi.org/10.26226/morressier.606f15dd30a2e980041f2468)

**[XREF0469]** Hamilton, Brian (2021). Tutorial on finite-difference time-domain (FDTD) methods for room acoustics simulation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0004614](https://doi.org/10.1121/10.0004614)

**[XREF0338]** Hamilton, Brian; Bilbao, Stefan (2021). Time-domain modeling of wave-based room acoustics including viscothermal and relaxation effects in air. *JASA Express Letters*. DOI: [10.1121/10.0006298](https://doi.org/10.1121/10.0006298)

**[XREF0733]** Hansen, Colin H. (2007). Room Acoustics. *Handbook of Noise and Vibration Control*. DOI: [10.1002/9780470209707.ch103](https://doi.org/10.1002/9780470209707.ch103)

**[XREF0043]** Hanyu, Toshiki (2025). A novel algorithm for generating spatial reverberation based on room acoustics and room shape complexity. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0040014](https://doi.org/10.1121/10.0040014)

**[XREF0262]** Harkness, Edward L. (1972). An Architect's Guide for the Direct Design of Reverberation Time and Room Volume for the Concert Hall. *Architectural Science Review*. DOI: [10.1080/00038628.1972.9696312](https://doi.org/10.1080/00038628.1972.9696312)

**[XREF0874]** Harris, Richard W.; Goldstein, David P. (1979). Effects of Room Reverberation upon Hearing Aid Quality Judgments. *International Journal of Audiology*. DOI: [10.3109/00206097909081527](https://doi.org/10.3109/00206097909081527)

**[XREF0167]** Harrison, CH; Robins, AJ (2024). Reverberation Stimulation for Sonar Systems Assessment. *Recent Developments in Underwater Acoustics*. DOI: [10.25144/18845](https://doi.org/10.25144/18845)

**[XREF0508]** Harrison, Chris H. (2005). Fast Bistatic Signal-To-Reverberation-Ratio Calculation. *Journal of Computational Acoustics*. DOI: [10.1142/s0218396x05002669](https://doi.org/10.1142/s0218396x05002669)

**[XREF0252]** Hartmann, William (2024). The effect of reverberation time on the spatial relationships between interaural differences in a room. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0035047](https://doi.org/10.1121/10.0035047)

**[XREF0881]** Hashemgeloogerdi, Sahar; Braun, Sebastian (2020). Joint Beamforming and Reverberation Cancellation Using a Constrained Kalman Filter With Multichannel Linear Prediction. *ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp40776.2020.9053785](https://doi.org/10.1109/icassp40776.2020.9053785)

**[XREF0028]** Hastings, Mardi C.; Godfrey, Richard D. (1997). Reverberation Room Survey and Qualification According to ASTM C 423. *Noise Control and Acoustics*. DOI: [10.1115/imece1997-1032](https://doi.org/10.1115/imece1997-1032)

**[XREF0822]** Hazrati, Oldooz; Ghaffarzadegan, Shabnam; Hansen, John H.L. (2015). Leveraging automatic speech recognition in cochlear implants for improved speech intelligibility under reverberation. *2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2015.7178941](https://doi.org/10.1109/icassp.2015.7178941)

**[XREF0351]** Heap, NW; Oldham, DJ (2025). A Room Simulation System for Architectural Education. *Auditorium Acoustics and Electro-Acoustics 1982*. DOI: [10.25144/24861](https://doi.org/10.25144/24861)

**[XREF0280]** Heerema, Nelson; Hodgson, Murray (1999). Empirical models for predicting noise levels, reverberation times and fitting densities in industrial workrooms. *Applied Acoustics*. DOI: [10.1016/s0003-682x(98)00037-1](https://doi.org/10.1016/s0003-682x(98)00037-1)

**[XREF0522]** Herrmann, Björn (2025). eLife Assessment: Listening to the room: disrupting activity of dorsolateral prefrontal cortex impairs learning of room acoustics in human listeners. *Unspecified venue*. DOI: [10.7554/elife.107041.1.sa5](https://doi.org/10.7554/elife.107041.1.sa5)

**[XREF0236]** Hioka, Yusuke; Tang, Jen W.; Wan, Jacky (2016). Effect of adding artificial reverberation to speech-like masking sound. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2016.07.014](https://doi.org/10.1016/j.apacoust.2016.07.014)

**[XREF0235]** Hirayama, Takashi (1958). 3010) Reverberation Time of Coupled Room (No. 2)(Scientific Basis of Planning Building). *Transactions of the Architectural Institute of Japan*. DOI: [10.3130/aijsaxx.60.2.0_37](https://doi.org/10.3130/aijsaxx.60.2.0_37)

**[XREF0260]** Hirayama, Takashi; Torii, Masayuki; Yasuoka, Masato (1961). 3018) Reverberation Time of Coupled Room(Scientific Basis of Planning Building). *Transactions of the Architectural Institute of Japan*. DOI: [10.3130/aijsaxx.69.2.0_65](https://doi.org/10.3130/aijsaxx.69.2.0_65)

**[XREF0510]** Hjøbjerg, K (2024). Objective Room Acoustical Parameters using 2-CHANNEL Analysis. *Acoustics '86*. DOI: [10.25144/22165](https://doi.org/10.25144/22165)

**[XREF0851]** Hocquette, Lucas; Coleman, Philip; Roskam, Frederic (2025). Acoustic reflection parameterization based on the spatial decomposition method. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002037](https://doi.org/10.1121/2.0002037)

**[XREF0484]** Hodgson, M (2024). Dummy-Head Recording. *Room Acoustics with Emphasis on Electroacoustics 1979*. DOI: [10.25144/23454](https://doi.org/10.25144/23454)

**[XREF0184]** Hodgson, Murray (2001). Empirical Prediction of Speech Levels and Reverberation in Classrooms. *Building Acoustics*. DOI: [10.1260/1351010011501696](https://doi.org/10.1260/1351010011501696)

**[XREF0187]** Hodoshima, Nao; Arai, Takayuki; Svensson, Peter (2006). The effect of a preprocessing approach improving speech intelligibility in reverberation considering a public-address system and room acoustics. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4781481](https://doi.org/10.1121/1.4781481)

**[XREF0660]** Holmer, C. I. (1974). Introduction to Workshop Session on Reverberation Room Qualification. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.3437459](https://doi.org/10.1121/1.3437459)

**[XREF0265]** Honghu, Zhang; Jia, Yan; Jianxin, Peng (2019). Chinese speech intelligibility of elderly people in environments combining reverberation and noise. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2019.02.002](https://doi.org/10.1016/j.apacoust.2019.02.002)

**[XREF0473]** Hoover, K. Anthony; Ellison, Steve (2013). Electronically variable room acoustics - motivations and challenges. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4800309](https://doi.org/10.1121/1.4800309)

**[XREF0155]** Hopper, H; Thompson, D; Holland, K (2023). Reverberation Enhancement in Music Practice Rooms. *Acoustics 2011*. DOI: [10.25144/17063](https://doi.org/10.25144/17063)

**[XREF0474]** Hopper, Hugh; Thompson, David; Holland, Keith (2011). The effect of reverberation enhancement on the diffusion of the sound field. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.3664340](https://doi.org/10.1121/1.3664340)

**[XREF0658]** Hoshi, Kazuma; Hanyu, Toshiki (2014). Theoretical Modeling of Room Shape for Ray Tracing Simulation. *Building Acoustics*. DOI: [10.1260/1351-010x.21.1.21](https://doi.org/10.1260/1351-010x.21.1.21)

**[XREF0718]** Hou, Qiannan; Wu, Jinrong; Zhang, Jianlan; et al. (2016). Effect of pulse length on low frequency average reverberation intensity in shallow water waveguide. *2016 IEEE/OES China Ocean Acoustics (COA)*. DOI: [10.1109/coa.2016.7535830](https://doi.org/10.1109/coa.2016.7535830)

**[XREF0144]** Houtsma, Adrianus J. (2008). Noise and Reverberation Reduction in Post Chapel Activity Room. *Unspecified venue*. DOI: [10.21236/ada491523](https://doi.org/10.21236/ada491523)

**[XREF0524]** Howarth, M.J; Lam, Y.W (2000). An assessment of the accuracy of a hybrid room acoustics model with surface diffusion facility. *Applied Acoustics*. DOI: [10.1016/s0003-682x(99)00059-6](https://doi.org/10.1016/s0003-682x(99)00059-6)

**[XREF0178]** Huang, Wuqiong; Peng, Jianxin; Xie, Tinghui (2023). Study on Chinese Speech Intelligibility Under Different Low-Frequency Characteristics of Reverberation Time Using a Hybrid Method. *Archives of Acoustics*. DOI: [10.24425/aoa.2023.145229](https://doi.org/10.24425/aoa.2023.145229)

**[XREF0541]** Hudson, R.R. (1973). The reduction of camera fan noise in a photoprint room. *Applied Acoustics*. DOI: [10.1016/0003-682x(73)90013-3](https://doi.org/10.1016/0003-682x(73)90013-3)

**[XREF0161]** Huisman, WHT (2024). Reverberation and Attenuation by Trees: Measured and Modelled. *Spring Conference - Acoustics '89*. DOI: [10.25144/21624](https://doi.org/10.25144/21624)

**[XREF0336]** Hursky, Paul; Abawi, Ahmad T. (2013). Reverberation modeling on graphics processing units (GPUs). *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4800712](https://doi.org/10.1121/1.4800712)

**[XREF0169]** Hyrkas, Jeremy; Smyth, Tamara (2025). Vibrato suppression by time-varying delay and spectral magnitude demodulation. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002182](https://doi.org/10.1121/2.0002182)

**[XREF0001]** Ianniello, C. (1981). Walk away method versus reverberation time method in determining the room constant. *Applied Acoustics*. DOI: [10.1016/0003-682x(81)90010-4](https://doi.org/10.1016/0003-682x(81)90010-4)

**[XREF0233]** Iglehart, F. (2024). Proposed Revision of Reverberation Time by International Code Council for Classrooms with Children deaf/hard of Hearing. *Proceedings of the 10th Convention of the European Acoustics Association Forum Acusticum 2023*. DOI: [10.61782/fa.2023.0362](https://doi.org/10.61782/fa.2023.0362)

**[XREF0313]** Ikpekha, Oshoke Wil; Simms, Mark (2025). Effect of Acoustic Absorber Type and Size on Sound Absorption of Porous Materials in a Full-Scale Reverberation Chamber. *Acoustics*. DOI: [10.3390/acoustics7010003](https://doi.org/10.3390/acoustics7010003)

**[XREF0423]** Isakson, Marcia J.; Chotiros, Nicholas P.; Piper, James N.; et al. (2015). Acoustic scattering from a sandy seabed at the target and reverberation experiment 2013 (TREX13). *2015 IEEE/OES Acoustics in Underwater Geosciences Symposium (RIO Acoustics)*. DOI: [10.1109/rioacoustics.2015.7473615](https://doi.org/10.1109/rioacoustics.2015.7473615)

**[XREF0016]** Izumi, Yuto; Otani, Makoto (2021). Relation between Direction-of-Arrival distribution of reflected sounds in late reverberation and room characteristics: Geometrical acoustics investigation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2020.107805](https://doi.org/10.1016/j.apacoust.2020.107805)

**[XREF0758]** Jackson, Miranda; Scavone, Gary (2025). A study of impedance of brass instruments and mouthpieces. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002165](https://doi.org/10.1121/2.0002165)

**[XREF0344]** James, Adrian (2020). Uncertainty in Room Acoustics Measurements. *Uncertainty in Acoustics*. DOI: [10.1201/9780429470622-6](https://doi.org/10.1201/9780429470622-6)

**[XREF0657]** Jariwala, Rushi; Upadhyaya, Ishan; George, Nithin V. (2017). Robust equalizer design for adaptive room impulse response compensation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2017.04.004](https://doi.org/10.1016/j.apacoust.2017.04.004)

**[XREF0716]** Javed, Hamza A.; Cauchi, Benjamin; Doclo, Simon; et al. (2017). Measuring, modelling and predicting perceived reverberation. *2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2017.7952182](https://doi.org/10.1109/icassp.2017.7952182)

**[XREF0396]** Jemmott, Colin W.; Stevens, William K. (2011). The impact of reverberation on active sonar optimum frequency. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.3611429](https://doi.org/10.1121/1.3611429)

**[XREF0889]** Jeong, Cheol-Ho (2016). Kurtosis of room impulse responses as a diffuseness measure for reverberation chambers. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4949365](https://doi.org/10.1121/1.4949365)

**[XREF0279]** Jeong, Cheol Ho; Brunskog, Jonas; Jacobsen, Finn (2013). Room acoustic transition time based on reflection overlap. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4798991](https://doi.org/10.1121/1.4798991)

**[XREF0631]** Jeong, Daeup; Fricke, Fergus R. (2000). Frequency perception as a measure of room acoustic quality. *Applied Acoustics*. DOI: [10.1016/s0003-682x(99)00032-8](https://doi.org/10.1016/s0003-682x(99)00032-8)

**[XREF0882]** Jeong, Jeong-Ho (2018). Effect of Directivity and Position of Sound Source When Measure Reverberation Time for the Correction of Receiving Room. *Transactions of the Korean Society for Noise and Vibration Engineering*. DOI: [10.5050/ksnve.2018.28.4.388](https://doi.org/10.5050/ksnve.2018.28.4.388)

**[XREF0455]** Jin, Wenyu (2016). Adaptive reverberation cancelation for multizone soundfield reproduction using sparse methods. *2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2016.7471727](https://doi.org/10.1109/icassp.2016.7471727)

**[XREF0679]** Jobnes, AJ; Barnett, PW (2024). Assisted Resonance in Practice. *Room Acoustics with Emphasis on Electroacoustics 1979*. DOI: [10.25144/23450](https://doi.org/10.25144/23450)

**[XREF0790]** Johnson, Roger P. (1963). Effects of Room Geometry on Reverberation Measurements of Absorption Coefficients. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2142675](https://doi.org/10.1121/1.2142675)

**[XREF0795]** Johnston, William; Godakawela, Janith; Sharma, Bhisham (2025). Gradient fibro-porous materials for tailored sound absorption. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002148](https://doi.org/10.1121/2.0002148)

**[XREF0556]** Jones, Doug (2008). Small Room Acoustics. *Handbook for Sound Engineers*. DOI: [10.1016/b978-0-240-80969-4.50010-9](https://doi.org/10.1016/b978-0-240-80969-4.50010-9)

**[XREF0624]** Jones, Doug; Sumoro, Hadi (2025). Small Room Acoustics. *Handbook for Sound Engineers*. DOI: [10.4324/9781003430292-4](https://doi.org/10.4324/9781003430292-4)

**[XREF0045]** Jones, PE (2024). Revision of ISO R354: Measurement of Absorption Coefficients in a Reverberation Room. *Recent Advances in British and International Standardisation in Building Acoustics 1980*. DOI: [10.25144/23322](https://doi.org/10.25144/23322)

**[XREF0271]** Jordan, V.L. (1969). Room acoustics and architectural acoustics development in recent years. *Applied Acoustics*. DOI: [10.1016/0003-682x(69)90032-2](https://doi.org/10.1016/0003-682x(69)90032-2)

**[XREF0145]** Jot, J.-M. (1992). An analysis/synthesis approach to real-time artificial reverberation. *[Proceedings] ICASSP-92: 1992 IEEE International Conference on Acoustics, Speech, and Signal Processing*. DOI: [10.1109/icassp.1992.226080](https://doi.org/10.1109/icassp.1992.226080)

**[XREF0243]** Jot, Jean-Marc; Vandernoot, Guillaume; Warusfel, Olivier (1996). Analysis and synthesis of room reverberation in the time and frequency domains—Application to the restoration of room impulse responses corrupted by measurement noise.. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.415793](https://doi.org/10.1121/1.415793)

**[XREF0516]** Jungmann, Jan Ole; Mazur, Radoslaw; Mertins, Alfred (2014). Joint time-domain reshaping and frequency-domain equalization of room impulse responses. *2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2014.6854885](https://doi.org/10.1109/icassp.2014.6854885)

**[XREF0430]** Jungmann, Jan Ole; Mazur, Radoslaw; Mertins, Alfred (2015). Joint time- and frequency-domain reshaping of room impulse responses. *2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2015.7178066](https://doi.org/10.1109/icassp.2015.7178066)

**[XREF0787]** Kahle, Eckhard (2013). Room Acoustical Quality of Concert Halls: Perceptual Factors and Acoustic Criteria — Return from Experience. *Building Acoustics*. DOI: [10.1260/1351-010x.20.4.265](https://doi.org/10.1260/1351-010x.20.4.265)

**[XREF0896]** Kalinova, Klara (2023). The Application of Nanofibrous Resonant Membranes for Room Acoustics. *Nanomaterials*. DOI: [10.3390/nano13061115](https://doi.org/10.3390/nano13061115)

**[XREF0821]** Kamble, Madhu R.; Patil, Hemant A. (2019). Analysis of Reverberation via Teager Energy Features for Replay Spoof Speech Detection. *ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2019.8683830](https://doi.org/10.1109/icassp.2019.8683830)

**[XREF0820]** Kamisiński, Tadeusz; Brawata, Krzysztof; Pilch, Adam; et al. (2012). Test Signal Selection for Determining the Sound Scattering Coefficient in a Reverberation Chamber. *Archives of Acoustics*. DOI: [10.2478/v10168-012-0051-2](https://doi.org/10.2478/v10168-012-0051-2)

**[XREF0369]** Kanev, N. G. (2013). Reverberation in a trapezoidal room. *Acoustical Physics*. DOI: [10.1134/s1063771013050102](https://doi.org/10.1134/s1063771013050102)

**[XREF0131]** Kang, Shengxian; Mak, Cheuk Ming; Ou, Dayi; et al. (2023). Effects of speech intelligibility and reverberation time on the serial recall task in Chinese open-plan offices: A laboratory study. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2023.109378](https://doi.org/10.1016/j.apacoust.2023.109378)

**[XREF0189]** Karami, Mohsen; Shafieian, Masoumeh (2013). The Effect of Television Decors on the Change of Reverberation Time of the Studio. *Open Journal of Acoustics*. DOI: [10.4236/oja.2013.32005](https://doi.org/10.4236/oja.2013.32005)

**[XREF0195]** Karampatzakis, P (2023). Connection Between Iconography and Reverberation Performance of Byzantine Temples Experimental Investigation. *Auditorium Acoustics 2023*. DOI: [10.25144/16283](https://doi.org/10.25144/16283)

**[XREF0674]** Kates, James M. (2001). Room reverberation effects in hearing aid feedback cancellation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.1332379](https://doi.org/10.1121/1.1332379)

**[XREF0829]** Katz, Brian F. G. (2004). International Round Robin on Room Acoustical Impulse Response Analysis Software 2004. *Acoustics Research Letters Online*. DOI: [10.1121/1.1758239](https://doi.org/10.1121/1.1758239)

**[XREF0630]** Kauﬁnger, Philip G.; Gokani, Chirag A.; Hamilton, Mark F. (2025). Creative ways to study for an acoustics qualifying exam. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002106](https://doi.org/10.1121/2.0002106)

**[XREF0080]** Kawata, Makito; Tsuruta-Hamamura, Mariko; Hasegawa, Hiroshi (2023). Assessment of speech transmission index and reverberation time in standardized English as a foreign language test rooms. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2022.109093](https://doi.org/10.1016/j.apacoust.2022.109093)

**[XREF0434]** Kawata, Makito; Tsuruta-Hamamura, Mariko; Hasegawa, Hiroshi (2023). Influence of Test Room Acoustics on Non-Native Listeners’ Standardized Test Performance. *Acoustics*. DOI: [10.3390/acoustics5040066](https://doi.org/10.3390/acoustics5040066)

**[XREF0385]** Kawczinski, Kimberly; Fujioka, Takako; Canfield-Dafilou, Elliot K.; et al. (2020). Perceptual similarity and scaling of room reverberation features: Decay time and wet-dry ratio. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.5147643](https://doi.org/10.1121/1.5147643)

**[XREF0228]** Kelly, Ian J.; Boland, Francis M. (2014). Randomness and the reverberation time, RT60, of acoustic responses. *2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2014.6855204](https://doi.org/10.1109/icassp.2014.6855204)

**[XREF0098]** Kendrick, Paul; Shiers, Nicola; Conetta, Robert; et al. (2012). Blind estimation of reverberation time in classrooms and hospital wards. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2012.02.010](https://doi.org/10.1016/j.apacoust.2012.02.010)

**[XREF0761]** Kennedy, David R.; Saint, Lester D. (1968). Reverberation-Room Testing of Space-Vehicle Structures. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.1970504](https://doi.org/10.1121/1.1970504)

**[XREF0506]** Keränen, Jukka; Hongisto, Valtteri; Radun, Jenni (2026). Room Acoustic Differences Between Enclosed and Open Learning Spaces. *Acoustics*. DOI: [10.3390/acoustics8010017](https://doi.org/10.3390/acoustics8010017)

**[XREF0375]** Kessler, Ralph (2026). Immersive acoustics and reverberation. *Immersive Sound Volume II*. DOI: [10.4324/9780429297526-9](https://doi.org/10.4324/9780429297526-9)

**[XREF0534]** Kim, JungSu; Lee, JinHak; Choi, YoungJi; et al. (2012). The Effect of an Edge on the Measured Scattering Coefficients in a Reverberation Chamber Based on ISO 17497-1. *Building Acoustics*. DOI: [10.1260/1351-010x.19.1.13](https://doi.org/10.1260/1351-010x.19.1.13)

**[XREF0768]** King, Richard L.; Leonard, Brett; Howie, Will; et al. (2016). Real Rooms vs. Artificial Reverberation: An evaluation of actual source audio vs. artificial ambience. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0000515](https://doi.org/10.1121/2.0000515)

**[XREF0840]** Kinoshita, Keisuke; Nakatani, Tomohiro; Miyoshi, Masato (2010). Blind upmix of stereo music signals using multi-step linear prediction based reverberation extraction. *2010 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2010.5496234](https://doi.org/10.1109/icassp.2010.5496234)

**[XREF0386]** Kirszenstein, J. (1984). An image source computer model for room acoustics analysis and electroacoustic simulation. *Applied Acoustics*. DOI: [10.1016/0003-682x(84)90011-2](https://doi.org/10.1016/0003-682x(84)90011-2)

**[XREF0816]** Kistler, W.M. (1999). Time-slicing: a model for cerebellar function based on synchronization, reverberation, and time windows. *9th International Conference on Artificial Neural Networks: ICANN '99*. DOI: [10.1049/cp:19991165](https://doi.org/10.1049/cp:19991165)

**[XREF0797]** Kistler, Werner M. (2001). Time-slicing: A model for cerebellar function based on synchronization, reverberation, and time windows. *Neurocomputing*. DOI: [10.1016/s0925-2312(01)00497-0](https://doi.org/10.1016/s0925-2312(01)00497-0)

**[XREF0892]** Kiyohara, Kenji; Furuya, Ken’ichi; Kaneda, Yutaka (2002). Sweeping echoes perceived in a regularly shaped reverberation room. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.1433808](https://doi.org/10.1121/1.1433808)

**[XREF0397]** Klasco, Michael (1992). Computer modeling and auralization in room acoustics: An overview on computer models for room acoustics. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.404929](https://doi.org/10.1121/1.404929)

**[XREF0476]** Kleckner, Jeff A.; Kolano, Richard A. (2005). Reverberation Room Sound System Loudspeaker Selection. *SAE Technical Paper Series*. DOI: [10.4271/2005-01-2442](https://doi.org/10.4271/2005-01-2442)

**[XREF0765]** Knight, Derrick P. (2018). Testing the limits of reverberation room qualification standard AHRI 220. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.5036205](https://doi.org/10.1121/1.5036205)

**[XREF0181]** Knudsen, V. O.; Delsasso, L. P.; Leonard, R. W. (1966). Reverberation-Room Acoustics. I: Effects of Absorptive Floor, Suspended Ceiling Panels, and Diffusers on Decay Curves. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.1943130](https://doi.org/10.1121/1.1943130)

**[XREF0104]** Knudsen, V. O.; Delsasso, L. P.; Leonard, R. W. (1967). Reverberation-Room Acoustics—Effects of Various Boundary Conditions. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.1910703](https://doi.org/10.1121/1.1910703)

**[XREF0732]** Kobayashi, Y (2000). Joint time-frequency analysis on neuronal reverberation. *Neuroscience Research*. DOI: [10.1016/s0168-0102(00)81753-8](https://doi.org/10.1016/s0168-0102(00)81753-8)

**[XREF0570]** Kocan-Krawczyk, Anna; Brański, Adam (2018). A Rough Estimation of Acoustics of the Cuboidal Room with Impedance Walls. *Archives of Acoustics*. DOI: [10.24425/122370](https://doi.org/10.24425/122370)

**[XREF0862]** Kodrasi, Ina; Doclo, Simon (2017). Multi-Channel late reverberation power spectral density estimation based on nuclear norm minimization. *2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)*. DOI: [10.1109/waspaa.2017.8170003](https://doi.org/10.1109/waspaa.2017.8170003)

**[XREF0849]** Kodrasi, Ina; Doclo, Simon (2018). Joint Late Reverberation and Noise Power Spectral Density Estimation in a Spatially Homogeneous Noise Field. *2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2018.8462142](https://doi.org/10.1109/icassp.2018.8462142)

**[XREF0764]** Kolano, Richard; Abhyankar, Sanjay; Martin, Thomas (2013). Restoring and Upgrading of a Ford Motor Company Reverberation Room Test Suite. *SAE Technical Paper Series*. DOI: [10.4271/2013-01-1960](https://doi.org/10.4271/2013-01-1960)

**[XREF0610]** Kolano, Richard A.; Brown, Darren J. (2017). Upgrading a Large Reverberation Room to Meet AHRI 220. *SAE Technical Paper Series*. DOI: [10.4271/2017-01-1896](https://doi.org/10.4271/2017-01-1896)

**[XREF0888]** Kolano, Richard A.; Kleckner, Jeff A. (1997). Verification of a Miniature Reverberation Room for Sound Absorption Measurements Using Corner Microphone Technique. *SAE Technical Paper Series*. DOI: [10.4271/971895](https://doi.org/10.4271/971895)

**[XREF0588]** Kolano, Richard A.; Kleckner, Jeff A. (2003). Audio Engineering Principles for Reverberation Room Sound Systems. *SAE Technical Paper Series*. DOI: [10.4271/2003-01-1678](https://doi.org/10.4271/2003-01-1678)

**[XREF0137]** Kolarik, Andrew J; Pardhan, Shahina; Cirstea, Silvia; et al. (2013). Using Acoustic Information to Perceive Room Size: Effects of Blindness, Room Reverberation Time, and Stimulus. *Perception*. DOI: [10.1068/p7555](https://doi.org/10.1068/p7555)

**[XREF0301]** Korany, Noha (2008). Factors affecting sound coloration perceived due to room reverberation. *2008 National Radio Science Conference*. DOI: [10.1109/nrsc.2008.4542362](https://doi.org/10.1109/nrsc.2008.4542362)

**[XREF0672]** Kotarbińska, E. (1988). How to calculate the efficiency of an acoustic barrier in a flat room. *Applied Acoustics*. DOI: [10.1016/0003-682x(88)90010-2](https://doi.org/10.1016/0003-682x(88)90010-2)

**[XREF0596]** Koyama, Shoichi; Daudet, Laurent (2017). Comparison of reverberation models for sparse sound field decomposition. *2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)*. DOI: [10.1109/waspaa.2017.8170026](https://doi.org/10.1109/waspaa.2017.8170026)

**[XREF0416]** Kozlowski, Piotr Z. (2018). How to Adjust Room Acoustics to Multifunctional Use at Music Venues. *2018 Joint Conference - Acoustics*. DOI: [10.1109/acoustics.2018.8502383](https://doi.org/10.1109/acoustics.2018.8502383)

**[XREF0007]** Kraszewski, Jarosław (2012). Computing Reverberation Time in a 3D Room Model Using a Finite Difference Method Applied for the Diffusion Equation. *Archives of Acoustics*. DOI: [10.2478/v10168-012-0023-6](https://doi.org/10.2478/v10168-012-0023-6)

**[XREF0583]** Kulowski, A. (1982). Relationship between impulse response and other types of room acoustical responses. *Applied Acoustics*. DOI: [10.1016/0003-682x(82)90011-1](https://doi.org/10.1016/0003-682x(82)90011-1)

**[XREF0885]** Kulowski, Andrzej (2022). Do Leonardo Da Vinci’s Drawings, Room Acoustics And Radio Astronomy Have Anything In Common?. *Unspecified venue*. DOI: [10.21203/rs.3.rs-1206026/v1](https://doi.org/10.21203/rs.3.rs-1206026/v1)

**[XREF0619]** Kumar, Krishna; George, Nithin V. (2020). A generalized maximum correntropy criterion based robust sparse adaptive room equalization. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2019.107036](https://doi.org/10.1016/j.apacoust.2019.107036)

**[XREF0173]** Kuttruff, H (2023). Retrospective Room Acoustics. *Spring Conference Acoustics 2005*. DOI: [10.25144/17912](https://doi.org/10.25144/17912)

**[XREF0226]** Kuttruff, Heinrich (2002). Room Acoustics. *Unspecified venue*. DOI: [10.1201/9781482286632](https://doi.org/10.1201/9781482286632)

**[XREF0481]** Kuttruff, Heinrich (2016). Design considerations and design procedures. *Room Acoustics*. DOI: [10.1201/9781315372150-9](https://doi.org/10.1201/9781315372150-9)

**[XREF0431]** Kuttruff, Heinrich (2016). Electroacoustical systems in rooms. *Room Acoustics*. DOI: [10.1201/9781315372150-10](https://doi.org/10.1201/9781315372150-10)

**[XREF0095]** Kuttruff, Heinrich (2016). Geometrical room acoustics. *Room Acoustics*. DOI: [10.1201/9781315372150-4](https://doi.org/10.1201/9781315372150-4)

**[XREF0115]** Kuttruff, Heinrich (2016). Measuring techniques in room acoustics. *Room Acoustics*. DOI: [10.1201/9781315372150-8](https://doi.org/10.1201/9781315372150-8)

**[XREF0398]** Kuttruff, Heinrich (2016). Reflection and scattering. *Room Acoustics*. DOI: [10.1201/9781315372150-2](https://doi.org/10.1201/9781315372150-2)

**[XREF0004]** Kuttruff, Heinrich (2016). Reverberation and steady-state energy density. *Room Acoustics*. DOI: [10.1201/9781315372150-5](https://doi.org/10.1201/9781315372150-5)

**[XREF0234]** Kuttruff, Heinrich (2016). Room Acoustics. *Unspecified venue*. DOI: [10.1201/9781315372150](https://doi.org/10.1201/9781315372150)

**[XREF0671]** Kuttruff, Heinrich (2016). Some facts on sound waves, sources and hearing. *Room Acoustics*. DOI: [10.1201/9781315372150-1](https://doi.org/10.1201/9781315372150-1)

**[XREF0483]** Kuttruff, Heinrich (2016). Sound absorption and sound absorbers. *Room Acoustics*. DOI: [10.1201/9781315372150-6](https://doi.org/10.1201/9781315372150-6)

**[XREF0247]** Kuttruff, Heinrich (2016). Sound waves in a room. *Room Acoustics*. DOI: [10.1201/9781315372150-3](https://doi.org/10.1201/9781315372150-3)

**[XREF0096]** Kuttruff, Heinrich (2016). Subjective room acoustics. *Room Acoustics*. DOI: [10.1201/9781315372150-7](https://doi.org/10.1201/9781315372150-7)

**[XREF0891]** Kuttruff, Heinrich; Akustik, Institut für Technische; Aachen, Technische Hochschule; et al. (1973). Room Acoustics. *Unspecified venue*. DOI: [10.4324/9780203186237](https://doi.org/10.4324/9780203186237)

**[XREF0311]** Kuttruff, Heinrich; Mommertz, Eckard (2013). Room Acoustics. *Handbook of Engineering Acoustics*. DOI: [10.1007/978-3-540-69460-1_10](https://doi.org/10.1007/978-3-540-69460-1_10)

**[XREF0620]** Kuttruff, Heinrich; Vorländer, Michael (2024). Design considerations and design procedures. *Room Acoustics*. DOI: [10.1201/9781003389873-9](https://doi.org/10.1201/9781003389873-9)

**[XREF0531]** Kuttruff, Heinrich; Vorländer, Michael (2024). Electroacoustical systems in rooms. *Room Acoustics*. DOI: [10.1201/9781003389873-11](https://doi.org/10.1201/9781003389873-11)

**[XREF0114]** Kuttruff, Heinrich; Vorländer, Michael (2024). Geometrical room acoustics. *Room Acoustics*. DOI: [10.1201/9781003389873-4](https://doi.org/10.1201/9781003389873-4)

**[XREF0133]** Kuttruff, Heinrich; Vorländer, Michael (2024). Measuring techniques in room acoustics. *Room Acoustics*. DOI: [10.1201/9781003389873-8](https://doi.org/10.1201/9781003389873-8)

**[XREF0441]** Kuttruff, Heinrich; Vorländer, Michael (2024). Prediction models. *Room Acoustics*. DOI: [10.1201/9781003389873-10](https://doi.org/10.1201/9781003389873-10)

**[XREF0490]** Kuttruff, Heinrich; Vorländer, Michael (2024). Reflection and scattering. *Room Acoustics*. DOI: [10.1201/9781003389873-2](https://doi.org/10.1201/9781003389873-2)

**[XREF0299]** Kuttruff, Heinrich; Vorländer, Michael (2024). Room Acoustics. *Unspecified venue*. DOI: [10.1201/9781003389873](https://doi.org/10.1201/9781003389873)

**[XREF0812]** Kuttruff, Heinrich; Vorländer, Michael (2024). Some facts on sound waves, sources, and hearing. *Room Acoustics*. DOI: [10.1201/9781003389873-1](https://doi.org/10.1201/9781003389873-1)

**[XREF0623]** Kuttruff, Heinrich; Vorländer, Michael (2024). Sound absorption and sound absorbers. *Room Acoustics*. DOI: [10.1201/9781003389873-6](https://doi.org/10.1201/9781003389873-6)

**[XREF0300]** Kuttruff, Heinrich; Vorländer, Michael (2024). Sound waves in a room. *Room Acoustics*. DOI: [10.1201/9781003389873-3](https://doi.org/10.1201/9781003389873-3)

**[XREF0113]** Kuttruff, Heinrich; Vorländer, Michael (2024). Subjective room acoustics. *Room Acoustics*. DOI: [10.1201/9781003389873-7](https://doi.org/10.1201/9781003389873-7)

**[XREF0629]** Labia, Laura; Shtrepi, Louena; Astolfi, Arianna (2020). Improved Room Acoustics Quality in Meeting Rooms: Investigation on the Optimal Configurations of Sound-Absorptive and Sound-Diffusive Panels. *Acoustics*. DOI: [10.3390/acoustics2030025](https://doi.org/10.3390/acoustics2030025)

**[XREF0074]** Lai, Joseph C.S.; Qi, Dan (1993). Sound transmission loss measurements using the sound intensity technique Part 1: The effects of reverberation time. *Applied Acoustics*. DOI: [10.1016/0003-682x(93)90091-j](https://doi.org/10.1016/0003-682x(93)90091-j)

**[XREF0213]** Lam, Y.W. (1992). Room acoustics. *Applied Acoustics*. DOI: [10.1016/0003-682x(92)90060-6](https://doi.org/10.1016/0003-682x(92)90060-6)

**[XREF0405]** LAM, YW (2024). On the Parameters Controlling Diffusion Calculation in a Hybrid Computer Model for Room Acoustics Prediction. *Acoustics 94*. DOI: [10.25144/20247](https://doi.org/10.25144/20247)

**[XREF0695]** Lang, M. A.; Rennie, J. M. (1981). Qualification of a 94-Cubic Metre Reverberation Room Under ANS S1.21. *Noise Control Engineering*. DOI: [10.3397/1.2832187](https://doi.org/10.3397/1.2832187)

**[XREF0559]** Lautenbach, Margriet (2011). Diffusivity of diffusers in the reverberation room. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.3654428](https://doi.org/10.1121/1.3654428)

**[XREF0780]** Lautenbach, Margriet; Vercammen, Martijn (2013). Can we use the standard deviation of the reverberation time to describe diffusion in a reverberation chamber?. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4805676](https://doi.org/10.1121/1.4805676)

**[XREF0296]** Lautenbach, Margriet R. (2025). Room acoustics for classical halls, practice, theory and music: Nobody knows the sound of a C80. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002325](https://doi.org/10.1121/2.0002325)

**[XREF0061]** Lautenbach, Margriet R.; Vercammen, Martijn L. (2013). Can we use the standard deviation of the reverberation time to describe diffusion in a reverberation chamber?. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4800319](https://doi.org/10.1121/1.4800319)

**[XREF0465]** Lawless, MS; Vigeant, MC (2023). Investigation of the Effects of Room Acoustics Stimuli on Reward Regions in the Brain. *Auditorium Acoustics 2015*. DOI: [10.25144/16164](https://doi.org/10.25144/16164)

**[XREF0342]** LeBlanc, Lévy; Gauthier, Philippe-Aubert; Berry, Alain (2019). A method to virtually extend reverberation time of measured impulse responses without losing room coloration. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.5137186](https://doi.org/10.1121/1.5137186)

**[XREF0108]** Leccese, Francesco; Rocca, Michele; Salvadori, Giacomo (2018). Fast estimation of Speech Transmission Index using the Reverberation Time: Comparison between predictive equations for educational rooms of different sizes. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2018.05.019](https://doi.org/10.1016/j.apacoust.2018.05.019)

**[XREF0697]** Lee, Doheon; Cabrera, Densil (2009). Basic Considerations for Loudness-Based Analysis of Room Impulse Responses. *Building Acoustics*. DOI: [10.1260/135101009788066500](https://doi.org/10.1260/135101009788066500)

**[XREF0326]** Lee, Doheon; Cabrera, Densil (2010). Effect of listening level and background noise on the subjective decay rate of room impulse responses: Using time-varying loudness to model reverberance. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2010.04.005](https://doi.org/10.1016/j.apacoust.2010.04.005)

**[XREF0035]** Lee, Hyojin; Jeong, Daeup (2025). The effect of changes in the configuration of a slatted ceiling on reverberation and clarity in a room. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2025.110693](https://doi.org/10.1016/j.apacoust.2025.110693)

**[XREF0491]** LEE, Keunhwa; CHU, Youngmin; Seong, Woojae (2013). Geometrical Ray-Bundle Reverberation Modeling. *Journal of Computational Acoustics*. DOI: [10.1142/s0218396x13500112](https://doi.org/10.1142/s0218396x13500112)

**[XREF0341]** Lee, Sang-Kwon; Lee, Min-Sung (2004). Reverberation time measurement for an acoustic room with low value of BT by utilizing wavelet transform. *Journal of Sound and Vibration*. DOI: [10.1016/j.jsv.2003.10.020](https://doi.org/10.1016/j.jsv.2003.10.020)

**[XREF0286]** Lee, Sang-Kwon; Park, Jin-Ho (2002). Reverberation time measurement of a structure or a room using modified wavelet transform. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4779772](https://doi.org/10.1121/1.4779772)

**[XREF0649]** Leferink, Frank B. J.; Puylaert, Ben R. M. (1993). Accurate Shielding Effectiveness Measurements Using a Reverberation Room. *10th International Zurich Symposium and Technical Exhibition on Electromagnetic Compatibility*. DOI: [10.23919/emc.1993.10781188](https://doi.org/10.23919/emc.1993.10781188)

**[XREF0180]** Legrand, Olivier; Sornette, Didier (1990). Test of Sabine’s reverberation time in ergodic auditoriums within geometrical acoustics. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.399736](https://doi.org/10.1121/1.399736)

**[XREF0027]** Lehmann, Eric A.; Johansson, Anders M.; Nordholm, Sven (2007). Reverberation-Time Prediction Method for Room Impulse Responses Simulated with the Image-Source Model. *2007 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics*. DOI: [10.1109/aspaa.2007.4392980](https://doi.org/10.1109/aspaa.2007.4392980)

**[XREF0467]** Lehnert, Hilmar; Blauert, Jens (1992). Principles of binaural room simulation. *Applied Acoustics*. DOI: [10.1016/0003-682x(92)90049-x](https://doi.org/10.1016/0003-682x(92)90049-x)

**[XREF0230]** Leonard, Brett; King, Richard L.; Sikora, Grzegorz (2013). Interaction between critical listening environment acoustics and listener reverberation preference. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4799258](https://doi.org/10.1121/1.4799258)

**[XREF0162]** Leśna, P.; Skrodzka, E. (2010). Subjective Evaluation of Classroom Acoustics by Teenagers vs. Reverberation Time. *Acta Physica Polonica A*. DOI: [10.12693/aphyspola.118.115](https://doi.org/10.12693/aphyspola.118.115)

**[XREF0415]** Lewers, T. (1993). A combined beam tracing and radiatn exchange computer model of room acoustics. *Applied Acoustics*. DOI: [10.1016/0003-682x(93)90049-c](https://doi.org/10.1016/0003-682x(93)90049-c)

**[XREF0555]** Lewitz, JA (2024). Electroacoustics in 'Surround' Halls. *Room Acoustics with Emphasis on Electroacoustics 1979*. DOI: [10.25144/23452](https://doi.org/10.25144/23452)

**[XREF0833]** Li, Junfeng; Xia, Risheng; Yan, Yonghong (2012). A hybrid approach for simulation of room reverberation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4707996](https://doi.org/10.1121/1.4707996)

**[XREF0248]** Li, Song; Schlieper, Roman; Peissig, Jurgen (2019). A Hybrid Method for Blind Estimation of Frequency Dependent Reverberation Time Using Speech Signals. *ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2019.8682661](https://doi.org/10.1109/icassp.2019.8682661)

**[XREF0190]** Li, Yue; Meyer, Julie; Lokki, Tapio; et al. (2022). Benchmarking of finite-difference time-domain method and fast multipole boundary element method for room acoustics. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2022.108662](https://doi.org/10.1016/j.apacoust.2022.108662)

**[XREF0278]** Li, Yue; Meyer, Julie; Lokki, Tapio; et al. (2022). Corrigendum to “Benchmarking of finite-difference time-domain method and fast multipole boundary element method for room acoustics” [Appl. Acoust. 191 (2022) 108662]. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2022.108789](https://doi.org/10.1016/j.apacoust.2022.108789)

**[XREF0644]** Li, Zhiyu; Yue, Xinwen; Zhao, Shenghui; et al. (2026). Multimodal Deep Learning Method for Real-Time Spatial Room Impulse Response Computing. *ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp55912.2026.11463810](https://doi.org/10.1109/icassp55912.2026.11463810)

**[XREF0324]** Lim, Felicia; Naylor, Patrick A.; Thomas, Mark R. P.; et al. (2015). Acoustic blur kernel with sliding window for blind estimation of reverberation time. *2015 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)*. DOI: [10.1109/waspaa.2015.7336904](https://doi.org/10.1109/waspaa.2015.7336904)

**[XREF0164]** Lim, Felicia; Thomas, Mark R. P.; Tashev, Ivan J. (2015). Blur kernel estimation approach to blind reverberation time estimation. *2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2015.7177928](https://doi.org/10.1109/icassp.2015.7177928)

**[XREF0461]** LIM, H; Imran, M; Jeon, JY (2023). Spatial Decomposition and Beamforming for Predicting 3D Room Acoustics in Concert Halls. *Auditorium Acoustics 2015*. DOI: [10.25144/16143](https://doi.org/10.25144/16143)

**[XREF0597]** Lin, Hejie; Bengisu, Turgay; Mourelatos, Zissimos P. (2021). Room Acoustics and Acoustical Partitions. *Lecture Notes on Acoustics and Noise Control*. DOI: [10.1007/978-3-030-88213-6_10](https://doi.org/10.1007/978-3-030-88213-6_10)

**[XREF0185]** Liu, Xin; Jing, Xiaodong (2026). An inverse method for source identification in rectangular waveguides with reverberation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2025.111209](https://doi.org/10.1016/j.apacoust.2025.111209)

**[XREF0706]** Liu, Yangfan; Bolton, J. Stuart (2013). The use of equivalent source models for reduced order simulation in room acoustics. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4800912](https://doi.org/10.1121/1.4800912)

**[XREF0443]** Liu, Ye; Huang, Zhi-ping; Su, Shao-jing; et al. (2012). AR model whitening and signal detection based on GLD algorithm in the non-Gaussian reverberation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2012.05.002](https://doi.org/10.1016/j.apacoust.2012.05.002)

**[XREF0302]** Liu, Yuji; Tong, Feng; Zhong, Shuanglian; et al. (2021). Reverberation aware deep learning for environment tolerant microphone array DOA estimation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2021.108337](https://doi.org/10.1016/j.apacoust.2021.108337)

**[XREF0048]** Ljung, Robert; Kjellberg, Anders (2009). Long Reverberation Time Decreases Recall of Spoken Information. *Building Acoustics*. DOI: [10.1260/135101009790291273](https://doi.org/10.1260/135101009790291273)

**[XREF0656]** Lollmann, Heinrich W.; Vary, Peter (2009). A blind speech enhancement algorithm for the suppression of late reverberation and noise. *2009 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2009.4960502](https://doi.org/10.1109/icassp.2009.4960502)

**[XREF0165]** Lollmann, Heinrich W.; Vary, Peter (2011). Estimation of the frequency dependent reverberation time by means of warped filter-banks. *2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2011.5946402](https://doi.org/10.1109/icassp.2011.5946402)

**[XREF0390]** Lopez, Jose J.; Navarro, Juan M.; Carnicero, Diego; et al. (2013). Some comments about graphic processing unit (GPU) architectures applied to finite-difference time-domain (FDTD) room acoustics simulation: Present and future trends. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4800576](https://doi.org/10.1121/1.4800576)

**[XREF0708]** López, Matthew (2024). Reverberation. *Reverberation*. DOI: [10.5040/9780571395231.00000004](https://doi.org/10.5040/9780571395231.00000004)

**[XREF0730]** Lopez, Nicolas; Grenier, Yves; Richard, Gael; et al. (2014). Single channel reverberation suppression based on sparse linear prediction. *2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2014.6854591](https://doi.org/10.1109/icassp.2014.6854591)

**[XREF0767]** López, Paula Sánchez; Callens, Paul; Cernak, Milos (2021). A Universal Deep Room Acoustics Estimator. *2021 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)*. DOI: [10.1109/waspaa52581.2021.9632738](https://doi.org/10.1109/waspaa52581.2021.9632738)

**[XREF0601]** Lucus, Megan N.; Goshorn, Edward L.; Kemker, Brett E. (2010). Ambient noise levels and reverberation times in Mississippi school rooms. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.3556447](https://doi.org/10.1121/1.3556447)

**[XREF0895]** Luizard, P; Katz, BFG (2023). Coupled Volume Multi-Slope Room Impulse Responses: a Quantitative Analysis Method. *Auditorium Acoustics 2011*. DOI: [10.25144/16852](https://doi.org/10.25144/16852)

**[XREF0546]** Lunkov, Andrey A.; Mihnyuk, Aleksandr N.; Malykhin, Andrey Yu. (2016). Effect of internal waves on interference pattern of bottom reverberation. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0000160](https://doi.org/10.1121/2.0000160)

**[XREF0330]** Lüthi, G.; Desarnaulds, V. (2022). Reverberation time in sports halls: Analysis of a large database of in-situ measurements and simulations according to absorption positions. *Proceedings of the 10th Convention of the European Acoustics Association Forum Acusticum 2023*. DOI: [10.61782/fa.2023.0571](https://doi.org/10.61782/fa.2023.0571)

**[XREF0566]** Maa, Dah-You (1994). Sound field in a room and its active noise control. *Applied Acoustics*. DOI: [10.1016/0003-682x(94)90064-7](https://doi.org/10.1016/0003-682x(94)90064-7)

**[XREF0819]** Maas, Roland; Habets, Emanuel A.P.; Sehr, Armin; et al. (2012). On the application of reverberation suppression to robust speech recognition. *2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2012.6287875](https://doi.org/10.1109/icassp.2012.6287875)

**[XREF0725]** Maas, Roland; Thippur, Akshaya; Sehr, Armin; et al. (2013). An uncertainty decoding approach to noise- and reverberation-robust speech recognition. *2013 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2013.6639098](https://doi.org/10.1109/icassp.2013.6639098)

**[XREF0592]** Macadam, J.A. (1976). The measurement of sound radiation from room surfaces in lightweight buildings. *Applied Acoustics*. DOI: [10.1016/0003-682x(76)90002-5](https://doi.org/10.1016/0003-682x(76)90002-5)

**[XREF0639]** Maciejewski, Matthew (2026). Single-Microphone Audio Point Source Discriminative Localization from Reverberation Late Tail Estimation. *ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp55912.2026.11461520](https://doi.org/10.1109/icassp55912.2026.11461520)

**[XREF0663]** Mackenzie, RK; Mackenzie, CM (2024). Masking Sound: Current Research. *Room Acoustics with Emphasis on Electroacoustics 1979*. DOI: [10.25144/23456](https://doi.org/10.25144/23456)

**[XREF0810]** Maempel, Hans-Joachim; Jentsch, Matthias (2013). Auditory and Visual Contribution to Egocentric Distance and Room Size Perception. *Building Acoustics*. DOI: [10.1260/1351-010x.20.4.383](https://doi.org/10.1260/1351-010x.20.4.383)

**[XREF0825]** Maluski, Sophie; Gibbs, Barry (1998). Variation of Sound Level Difference in Dwellings Due to Room Modal Characteristics. *Building Acoustics*. DOI: [10.1177/1351010x9800500406](https://doi.org/10.1177/1351010x9800500406)

**[XREF0269]** Manik, Dhanesh N. (2017). Room Acoustics. *Vibro-Acoustics*. DOI: [10.1201/9781315156729-8](https://doi.org/10.1201/9781315156729-8)

**[XREF0717]** Mapp, Peter (1998). Studio and Control Room Acoustics. *Audio and Hi-Fi Handbook*. DOI: [10.1016/b978-0-08-054564-6.50007-0](https://doi.org/10.1016/b978-0-08-054564-6.50007-0)

**[XREF0593]** Marbjerg, Gerd; Brunskog, Jonas; Jeong, Cheol-Ho (2018). The difficulties of simulating the acoustics of an empty rectangular room with an absorbing ceiling. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2018.06.017](https://doi.org/10.1016/j.apacoust.2018.06.017)

**[XREF0051]** Marín, Albert; Giménez, Alicia (1996). Determination of the reverberation time in any point of a closed room by a physico-mathematical model based on geometric acoustics. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.416094](https://doi.org/10.1121/1.416094)

**[XREF0449]** Markovic, Milos; Geiger, Jurgen (2017). Reverberation-based feature extraction for acoustic scene classification. *2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2017.7952262](https://doi.org/10.1109/icassp.2017.7952262)

**[XREF0297]** Markovic, Milos; Olesen, Søren K.; Hammershoi, Dorte (2013). Three-dimensional point-cloud room model for room acoustics simulations. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4800237](https://doi.org/10.1121/1.4800237)

**[XREF0053]** Marsch, Jürgen; Pörschmann, Christoph (2000). Frequency dependent control of reverberation time for auditory virtual environments. *Applied Acoustics*. DOI: [10.1016/s0003-682x(99)00073-0](https://doi.org/10.1016/s0003-682x(99)00073-0)

**[XREF0128]** Martell-Villalpando, Jacques; Martínez-Borquez, Alejandro; Ibarra-Zarate, David I. (2025). Evaluating a scale model reverberation room for Sound Absorption Coefﬁcient measurements: A cost-effective alternative. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002026](https://doi.org/10.1121/2.0002026)

**[XREF0552]** Martin, N.; Mars, J.; Martin, J.; et al. (1995). A Capon's time-octave representation application in room acoustics. *IEEE Transactions on Signal Processing*. DOI: [10.1109/78.403343](https://doi.org/10.1109/78.403343)

**[XREF0875]** Martin, V.; Picinali, L. (2024). Comparing Online vs. Lab-based Experimental Approaches for the Perceptual Evaluation of Artificial Reverberation. *Proceedings of the 10th Convention of the European Acoustics Association Forum Acusticum 2023*. DOI: [10.61782/fa.2023.0312](https://doi.org/10.61782/fa.2023.0312)

**[XREF0099]** Masovic, Drasko; Oguc, Mete (2012). Low frequency measurements in building acoustics &#x2014; Analysis of reverberation time field Measurement results. *2012 20th Telecommunications Forum (TELFOR)*. DOI: [10.1109/telfor.2012.6419437](https://doi.org/10.1109/telfor.2012.6419437)

**[XREF0651]** Massé, Pierre; Carpentier, Thibaut; Warusfel, Olivier; et al. (2020). Denoising Directional Room Impulse Responses with Spatially Anisotropic Late Reverberation Tails. *Applied Sciences*. DOI: [10.3390/app10031033](https://doi.org/10.3390/app10031033)

**[XREF0848]** Matusiak, Ewa; Chatziioannou, Vasileios; van Walstijn, Maarten (2025). A refined bow-string interaction model considering hysteresis. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002120](https://doi.org/10.1121/2.0002120)

**[XREF0740]** May, Lloyd; Farzaneh, Nima; Das, Orchisama; et al. (2025). Comparison of Impulse Response Generation Methods for a Simple Shoebox-Shaped Room. *Acoustics*. DOI: [10.3390/acoustics7030056](https://doi.org/10.3390/acoustics7030056)

**[XREF0703]** McCammon, Diana F. (1993). Time and Angle Spreading from Rough Sediments. *Ocean Reverberation*. DOI: [10.1007/978-94-011-2078-4_17](https://doi.org/10.1007/978-94-011-2078-4_17)

**[XREF0626]** McDermott, Barbara; Allen, Jont (1976). Perceptual factors of small room reverberation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2003645](https://doi.org/10.1121/1.2003645)

**[XREF0669]** McDermott, Scott D. (2025). Accuracy of audio attenuation in 3D audio libraries. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002134](https://doi.org/10.1121/2.0002134)

**[XREF0209]** McLoughlin, Ian; Lee, Jeannie S.; Atmosukarto, Indriyati (2024). Single channel AI speech reverberation time modification for room dimension matching. *2024 IEEE International Symposium on Mixed and Augmented Reality Adjunct (ISMAR-Adjunct)*. DOI: [10.1109/ismar-adjunct64951.2024.00125](https://doi.org/10.1109/ismar-adjunct64951.2024.00125)

**[XREF0557]** McMullan, R. (1983). Room Acoustics. *Environmental Science in Building*. DOI: [10.1007/978-1-349-06279-9_11](https://doi.org/10.1007/978-1-349-06279-9_11)

**[XREF0558]** McMullan, R. (1989). Room Acoustics. *Environmental Science in Building*. DOI: [10.1007/978-1-349-19896-2_11](https://doi.org/10.1007/978-1-349-19896-2_11)

**[XREF0550]** McMullan, R. (1992). Room Acoustics. *Environmental Science in Building*. DOI: [10.1007/978-1-349-22169-1_11](https://doi.org/10.1007/978-1-349-22169-1_11)

**[XREF0553]** McMullan, Randall (1998). Room Acoustics. *Environmental Science in Building*. DOI: [10.1007/978-1-349-14811-0_11](https://doi.org/10.1007/978-1-349-14811-0_11)

**[XREF0533]** McMullan, Randall (2018). Room Acoustics. *Environmental Science in Building*. DOI: [10.1057/978-1-137-60545-0_10](https://doi.org/10.1057/978-1-137-60545-0_10)

**[XREF0365]** McQuillan, Jacob; van Walstijn, Maarten; Parker, Julian D.; et al. (2022). Experimental investigation of the driving mechanism in spring reverberation tanks. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0001662](https://doi.org/10.1121/2.0001662)

**[XREF0011]** Mealings, Kiri (2019). Validation of the SoundOut Room Acoustics Analyzer App for Classrooms: A New Method for Self-Assessment of Noise Levels and Reverberation Time in Schools. *Acoustics Australia*. DOI: [10.1007/s40857-019-00166-1](https://doi.org/10.1007/s40857-019-00166-1)

**[XREF0268]** Mealings, Kiri (2022). Classroom acoustics and cognition: A review of the effects of noise and reverberation on primary school children’s attention and memory. *Building Acoustics*. DOI: [10.1177/1351010x221104892](https://doi.org/10.1177/1351010x221104892)

**[XREF0003]** Meissner, Mirosław (2008). Influence of wall absorption on low-frequency dependence of reverberation time in room of irregular shape. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2007.02.004](https://doi.org/10.1016/j.apacoust.2007.02.004)

**[XREF0614]** Meissner, Mirosław (2013). Analytical and numerical study of acoustic intensity field in irregularly shaped room. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2012.11.009](https://doi.org/10.1016/j.apacoust.2012.11.009)

**[XREF0123]** Meissner, Mirosław (2017). Acoustics of small rectangular rooms: Analytical and numerical determination of reverberation parameters. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2017.01.020](https://doi.org/10.1016/j.apacoust.2017.01.020)

**[XREF0085]** Meissner, Mirosław; Zieliński, Tomasz G. (2022). Impact of Wall Impedance Phase Angle on Indoor Sound Field and Reverberation Parameters Derived from Room Impulse Response. *Archives of Acoustics*. DOI: [10.24425/aoa.2022.142008](https://doi.org/10.24425/aoa.2022.142008)

**[XREF0274]** Meissner, Miroslaw (2008). Low frequency evaluation of steady-state pressure distribution and reverberation time in two-room coupled system. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2935910](https://doi.org/10.1121/1.2935910)

**[XREF0310]** Meyer, Erwin; Neumann, Ernst-Georg (1972). Room Acoustics. *Physical and Applied Acoustics*. DOI: [10.1016/b978-0-12-493150-3.50008-4](https://doi.org/10.1016/b978-0-12-493150-3.50008-4)

**[XREF0345]** Meyer, J (2023). The St Michaelis Church in LÜNEBERG - an Example for Changing Room Acoustics. *Auditorium Acoustics 2008*. DOI: [10.25144/17498](https://doi.org/10.25144/17498)

**[XREF0343]** Meyer, Jürgen (2009). Foundations of Room Acoustics. *Acoustics and the Performance of Music*. DOI: [10.1007/978-0-387-09517-2_5](https://doi.org/10.1007/978-0-387-09517-2_5)

**[XREF0894]** Meynial, X; Lissek, H (2024). Active Reflectors for Room Acoustics. *The Legacy of the 20th Century and beyond 2000*. DOI: [10.25144/18803](https://doi.org/10.25144/18803)

**[XREF0457]** Meynial, X; Polack, J-D; Dodd, G; et al. (2024). All-Scale Room Acoustics Measurements with Midas. *Acoustics, Architecture and Auditoria 1992*. DOI: [10.25144/20720](https://doi.org/10.25144/20720)

**[XREF0687]** Mi, Huan; Kearney, Gavin; Daffern, Helena (2022). Impact Thresholds of Parameters of Binaural Room Impulse Responses (BRIRs) on Perceptual Reverberation. *Applied Sciences*. DOI: [10.3390/app12062823](https://doi.org/10.3390/app12062823)

**[XREF0782]** Miao, F. X.; Sun, Guojun; Pao, Y. H. (2009). Vibration Mode Analysis of Frames by the Method of Reverberation Ray Matrix. *Journal of Vibration and Acoustics*. DOI: [10.1115/1.3147127](https://doi.org/10.1115/1.3147127)

**[XREF0094]** Miguez, D; Farrell, O; Samami, K; et al. (2023). Application of Dynamic Substructuring and in Situ Blocked Force Method for Structure Borne Noise Prediction in a Reverberation Room. *Acoustics 2019*. DOI: [10.25144/15339](https://doi.org/10.25144/15339)

**[XREF0039]** Mijić, M. (1994). Gun-to-noise differences in the dwelling rooms reverberation time measurement. *Applied Acoustics*. DOI: [10.1016/0003-682x(94)90069-8](https://doi.org/10.1016/0003-682x(94)90069-8)

**[XREF0082]** Mikulski, Witold; Radosz, Jan (2011). Acoustics of Classrooms in Primary Schools - Results of the Reverberation Time and the Speech Transmission Index Assessments in Selected Buildings. *Archives of Acoustics*. DOI: [10.2478/v10168-011-0052-6](https://doi.org/10.2478/v10168-011-0052-6)

**[XREF0525]** Miles, Ronald N. (2020). Geometrical Room Acoustics. *Mechanical Engineering Series*. DOI: [10.1007/978-3-030-22676-3_8](https://doi.org/10.1007/978-3-030-22676-3_8)

**[XREF0537]** Miles, Ronald N. (2024). Geometrical Room Acoustics. *Mechanical Engineering Series*. DOI: [10.1007/978-3-031-33009-4_8](https://doi.org/10.1007/978-3-031-33009-4_8)

**[XREF0535]** Miller, Thomas E. (2015). Real time bottom reverberation simulation in deep and shallow ocean environments. *Unspecified venue*. DOI: [10.1575/1912/7723](https://doi.org/10.1575/1912/7723)

**[XREF0747]** Milo, Alessia; Einarsson, Jóhannes F.; Einarsson, Úlfur; et al. (2023). Treble Auralizer: a real time Web Audio Engine enabling 3DoF auralization of simulated room acoustics designs. *2023 Immersive and 3D Audio: from Architecture to Automotive (I3DA)*. DOI: [10.1109/i3da57090.2023.10289386](https://doi.org/10.1109/i3da57090.2023.10289386)

**[XREF0586]** Mironovs, Deniss (2025). Linking Speech Clarity, Reverberation, And Distance For Classroom Design Optimization. *Proceedings of the 11th Convention of the European Acoustics Association Forum Acusticum / EuroNoise 2025*. DOI: [10.61782/fa.2025.0521](https://doi.org/10.61782/fa.2025.0521)

**[XREF0655]** Missoni, Fulvio; Poole, Katarina C.; Picinali, Lorenzo; et al. (2025). Effects of auditory distance cues and reverberation on spatial perception and listening strategies. *npj Acoustics*. DOI: [10.1038/s44384-025-00027-4](https://doi.org/10.1038/s44384-025-00027-4)

**[XREF0613]** Mitra, Vikramjit; Tsiartas, Andreas; Shriberg, Elizabeth (2016). Noise and reverberation effects on depression detection from speech. *2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2016.7472788](https://doi.org/10.1109/icassp.2016.7472788)

**[XREF0877]** Miyoshi, M.; Kaneda, Y. (1988). Inverse filtering of room acoustics. *IEEE Transactions on Acoustics, Speech, and Signal Processing*. DOI: [10.1109/29.1509](https://doi.org/10.1109/29.1509)

**[XREF0608]** Mleczko, Dominik (2025). The influence of diffuser configuration in a reverberation room on sound absorption coefficient measurements. *MATERIAŁY Budowlane*. DOI: [10.15199/33.2025.08.25](https://doi.org/10.15199/33.2025.08.25)

**[XREF0072]** Mleczko, Dominik; Wszołek, Tadeusz (2019). Effect of Diffusing Elements in a Reverberation Room on the Results of Airborne Sound Insulation Laboratory Measurements. *Archives of Acoustics*. DOI: [10.24425/aoa.2019.129729](https://doi.org/10.24425/aoa.2019.129729)

**[XREF0726]** Mo, Dongpeng; Gao, Bo; Song, Wenhua; et al. (2021). Doppler Effect Analysis of Bottom Reverberation for a Moving Platform in Shallow Water. *2021 OES China Ocean Acoustics (COA)*. DOI: [10.1109/coa50123.2021.9519994](https://doi.org/10.1109/coa50123.2021.9519994)

**[XREF0223]** Mo, Fangshuo (2015). Reverberation decay functions for narrow bands obtained from filtered time-windowed room impulse responses. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4921287](https://doi.org/10.1121/1.4921287)

**[XREF0796]** Moller, H; Blasinski, L (2023). Room Acoustic Measurements in Halls with Electro-Acoustic Enhancement Systems. *Auditorium Acoustics 2023*. DOI: [10.25144/16002](https://doi.org/10.25144/16002)

**[XREF0853]** Mondet, Boris; Brunskog, Jonas; Jeong, Cheol-Ho; et al. (2020). From absorption to impedance: Enhancing boundary conditions in room acoustic simulations. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2019.04.034](https://doi.org/10.1016/j.apacoust.2019.04.034)

**[XREF0078]** Montoya, Juan C. (2017). Comparison and analysis of the methods defined by ASTM standard E2235-04, ISO 3382-2-2008, and EASE acoustical modeling software to determine reverberation time RT60 in ordinary rooms. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.5014159](https://doi.org/10.1121/1.5014159)

**[XREF0349]** Moore, J. E. (1978). Room Acoustics. *Design for Good Acoustics and Noise Control*. DOI: [10.1007/978-1-349-16035-8_4](https://doi.org/10.1007/978-1-349-16035-8_4)

**[XREF0828]** Mores, Robert; Bader, Rolf; Linke, Simon (2025). The khaen of the Hmong people in Northern Laos. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002077](https://doi.org/10.1121/2.0002077)

**[XREF0689]** Morgenstern, Hai; Rafaely, Boaz (2013). Enhanced spatial analysis of room acoustics using acoustic multiple-input multiple-output (MIMO) systems. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4798988](https://doi.org/10.1121/1.4798988)

**[XREF0186]** Mornington-West, A; Bray, JA (2024). Acoustic Reverberation Kit - a New Tool, to Assist in Acoustic Measurements. *Acoustics '84 (Microprocessor and Computer Applications in Acoustics)*. DOI: [10.25144/22563](https://doi.org/10.25144/22563)

**[XREF0261]** Möser, Michael (2004). Fundamentals of Room Acoustics. *Engineering Acoustics*. DOI: [10.1007/978-3-662-05391-1_7](https://doi.org/10.1007/978-3-662-05391-1_7)

**[XREF0372]** Möser, Michael (2009). Fundamentals of room acoustics. *Engineering Acoustics*. DOI: [10.1007/978-3-540-92723-5_7](https://doi.org/10.1007/978-3-540-92723-5_7)

**[XREF0117]** Moses, N. (1973). Electronically controlled high intensity reverberation chamber. *Applied Acoustics*. DOI: [10.1016/0003-682x(73)90006-6](https://doi.org/10.1016/0003-682x(73)90006-6)

**[XREF0688]** Mukae, Shunichi; Okuzono, Takeshi; Sakagami, Kimihiro (2022). On the Robustness and Efficiency of the Plane-Wave-Enriched FEM with Variable q-Approach on the 2D Room Acoustics Problem. *Acoustics*. DOI: [10.3390/acoustics4010004](https://doi.org/10.3390/acoustics4010004)

**[XREF0529]** Munshi, A.S. (1992). Equalizability of room acoustics. *[Proceedings] ICASSP-92: 1992 IEEE International Conference on Acoustics, Speech, and Signal Processing*. DOI: [10.1109/icassp.1992.226081](https://doi.org/10.1109/icassp.1992.226081)

**[XREF0143]** Murphy, Damian T.; Southern, Alex; Savioja, Lauri (2014). Source excitation strategies for obtaining impulse responses in finite difference time domain room acoustics simulation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2014.02.010](https://doi.org/10.1016/j.apacoust.2014.02.010)

**[XREF0883]** Murphy, DT; Howard, DM (2024). Room Acoustics Modelling using Digital Waveguide Mesh Structures. *RS 17 Measuring, Modelling or Muddling!*. DOI: [10.25144/18521](https://doi.org/10.25144/18521)

**[XREF0845]** Nagase, Ryudo; Oishi, Kunio; Furukawa, Toshihiro (2014). Performance of cepstrum-based deconvolution for DOA estimation in the presence of room reverberation. *2014 IEEE 3rd Global Conference on Consumer Electronics (GCCE)*. DOI: [10.1109/gcce.2014.7031207](https://doi.org/10.1109/gcce.2014.7031207)

**[XREF0050]** Nakamura, Satoshi; Shikano, Kiyohiro (1997). Room acoustics and reverberation: impact on hands-free recognition. *5th European Conference on Speech Communication and Technology (Eurospeech 1997)*. DOI: [10.21437/eurospeech.1997-629](https://doi.org/10.21437/eurospeech.1997-629)

**[XREF0384]** Nakatani, Tomohiro; Yoshioka, Takuya; Kinoshita, Keisuke; et al. (2009). Real-time speech enhancement in noisy reverberant multi-talker environments based on a location-independent room acoustics model. *2009 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2009.4959539](https://doi.org/10.1109/icassp.2009.4959539)

**[XREF0250]** Nam, Hyeonuk; Park, Yong-Hwa (2025). Coherence-based phonemic analysis on the effect of reverberation to practical automatic speech recognition. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2024.110233](https://doi.org/10.1016/j.apacoust.2024.110233)

**[XREF0634]** Namvar Arefi, Hossein; Ghiasi, Seyyed Mohammad Amin; Ghaffari, Seyyede Mahshid; et al. (2014). Conference Room Reverberation Time Correction Using Helmholtz Resonators Lined with Absorbers. *Shock and Vibration*. DOI: [10.1155/2014/472524](https://doi.org/10.1155/2014/472524)

**[XREF0040]** Nannariello, Joseph; Fricke, Fergus (1999). The prediction of reverberation time using neural network analysis. *Applied Acoustics*. DOI: [10.1016/s0003-682x(98)00081-4](https://doi.org/10.1016/s0003-682x(98)00081-4)

**[XREF0052]** Nannariello, Joseph; Fricke, Fergus (2002). The Prediction of Reverberation Time Using Optimal Neural Networks. *Building Acoustics*. DOI: [10.1260/135101002761035717](https://doi.org/10.1260/135101002761035717)

**[XREF0176]** Napolin, Julie Beth (2020). Reprise: Reverberation, Circumambience, and Form-Seeking Sound (Absalom, Absalom!). *The Fact of Resonance*. DOI: [10.5422/fordham/9780823288175.003.0008](https://doi.org/10.5422/fordham/9780823288175.003.0008)

**[XREF0852]** Nargolkar, Ishan; Vijayan, Kiran (2025). Coupled hydrodynamic-structural analysis on reuseable launch vehicle landing on barge. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002061](https://doi.org/10.1121/2.0002061)

**[XREF0208]** Nassiri, Parvin; Shalkouhi, Pedram Jafari (2011). Technical Note: Measurement of Background Noise Level and Prediction of Reverberation Time in University Classrooms. *The International Journal of Acoustics and Vibration*. DOI: [10.20855/ijav.2011.16.1273](https://doi.org/10.20855/ijav.2011.16.1273)

**[XREF0424]** Naylor, G.M. (1993). ODEON—Another hybrid room acoustical model. *Applied Acoustics*. DOI: [10.1016/0003-682x(93)90047-a](https://doi.org/10.1016/0003-682x(93)90047-a)

**[XREF0509]** Neal, MT; Vigeant, MC (2023). Subjective Study on Listener Envelopment using Hybrid Room Acoustics Simulation and Higher Order Ambisonics Reproduction. *Auditorium Acoustics 2015*. DOI: [10.25144/16165](https://doi.org/10.25144/16165)

**[XREF0600]** Nestoras, Christos; Dance, Stephen (2013). The Interrelationship between Room Acoustics Parameters as Measured in University Classrooms Using Four Source Configurations. *Building Acoustics*. DOI: [10.1260/1351-010x.20.1.43](https://doi.org/10.1260/1351-010x.20.1.43)

**[XREF0083]** Neubauer, Reinhard O. (2001). Estimation of Reverberation Time in Rectangular Rooms with Non-Uniformly Distributed Absorption Using a Modified Fitzroy Equation. *Building Acoustics*. DOI: [10.1260/1351010011501786](https://doi.org/10.1260/1351010011501786)

**[XREF0675]** Newell, Philip (2007). Room acoustics and means of control. *Recording Studio Design*. DOI: [10.1016/b978-0-240-52086-5.50010-2](https://doi.org/10.1016/b978-0-240-52086-5.50010-2)

**[XREF0664]** Newell, Philip (2012). Room Acoustics and Means of Control. *Recording Studio Design*. DOI: [10.1016/b978-0-240-52240-1.00004-1](https://doi.org/10.1016/b978-0-240-52240-1.00004-1)

**[XREF0196]** Newell, PR; Holland, KR; Hidley, T (2024). Control Room Reverberation Is Unwanted Noise. *Reproduced Sound 1994*. DOI: [10.25144/20276](https://doi.org/10.25144/20276)

**[XREF0517]** Nie, Ruixin; Liu, Xionghou; Sun, Chao; et al. (2021). Multi-ping Reverberation Suppression Combined with Spatial Continuity of Target Motion. *2021 OES China Ocean Acoustics (COA)*. DOI: [10.1109/coa50123.2021.9520050](https://doi.org/10.1109/coa50123.2021.9520050)

**[XREF0305]** Nielsen, Sofus Birkedal; Celestinos, Adrian (2008). Improving room acoustics at low frequencies with multiple loudspeakers and time based room correction. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2932925](https://doi.org/10.1121/1.2932925)

**[XREF0377]** Nikolic, I; Bjor, O-H (2023). Building and Room Acoustics Measurements with Sine-Sweep Technique. *Autumn Conference Acoustics 2003*. DOI: [10.25144/18144](https://doi.org/10.25144/18144)

**[XREF0659]** Nikunen, Joonas; Virtanen, Tuomas (2018). Estimation of Time-Varying Room Impulse Responses of Multiple Sound Sources from Observed Mixture and Isolated Source Signals. *2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2018.8462535](https://doi.org/10.1109/icassp.2018.8462535)

**[XREF0264]** Nimura, Tadamoto; Kido, Ken'iti (1956). Effect of Sound Reinforcement System in a Room on the Reverberation Time and Clearness. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.1918298](https://doi.org/10.1121/1.1918298)

**[XREF0019]** Nowoświat, Artur (2022). Impact of Temperature and Relative Humidity on Reverberation Time in a Reverberation Room. *Buildings*. DOI: [10.3390/buildings12081282](https://doi.org/10.3390/buildings12081282)

**[XREF0038]** Nowoświat, Artur; Olechowska, Marcelina (2016). Fast estimation of speech transmission index using the reverberation time. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2015.09.001](https://doi.org/10.1016/j.apacoust.2015.09.001)

**[XREF0088]** Nowoświat, Artur; Olechowska, Marcelina (2017). Estimation of Reverberation Time in Classrooms Using the Residual Minimization Method. *Archives of Acoustics*. DOI: [10.1515/aoa-2017-0065](https://doi.org/10.1515/aoa-2017-0065)

**[XREF0081]** Nowoświat, Artur; Olechowska, Marcelina (2022). Experimental Validation of the Model of Reverberation Time Prediction in a Room. *Buildings*. DOI: [10.3390/buildings12030347](https://doi.org/10.3390/buildings12030347)

**[XREF0103]** Nowoświat, Artur; Olechowska, Marcelina; Marchacz, Michał (2020). The effect of acoustical remedies changing the reverberation time for different frequencies in a dome used for worship: A case study. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2019.107143](https://doi.org/10.1016/j.apacoust.2019.107143)

**[XREF0049]** Nowoświat, Artur; Olechowska, Marcelina; Ślusarek, Jan (2016). Prediction of reverberation time using the residual minimization method. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2015.12.024](https://doi.org/10.1016/j.apacoust.2015.12.024)

**[XREF0120]** Nozaki, Kotoyo; Ikeda, Yusuke; Oikawa, Yasuhiro; et al. (2018). Blind reverberation energy estimation using exponential averaging with attack and release time constants for hearing aids. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2018.08.010](https://doi.org/10.1016/j.apacoust.2018.08.010)

**[XREF0625]** Nwankwo, Oliver O.; Szary, Marek L. (1996). Calibrating Reverberation Room for Accurate Material Sound Absorption Measurements. *SAE Technical Paper Series*. DOI: [10.4271/960191](https://doi.org/10.4271/960191)

**[XREF0728]** Oksanen, Sami; Parker, Julian; Politis, Archontis; et al. (2013). A directional diffuse reverberation model for excavated tunnels in rock. *2013 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2013.6637727](https://doi.org/10.1109/icassp.2013.6637727)

**[XREF0854]** Okubo, Hiroyuki; Otani, Masamichi; Ikezawa, Ryo; et al. (2001). A system for measuring the directional room acoustical parameters. *Applied Acoustics*. DOI: [10.1016/s0003-682x(00)00056-6](https://doi.org/10.1016/s0003-682x(00)00056-6)

**[XREF0683]** Okuzono, Takeshi; Otsuru, Toru; Sakagami, Kimihiro (2015). Applicability of an explicit time-domain finite-element method on room acoustics simulation. *Acoustical Science and Technology*. DOI: [10.1250/ast.36.377](https://doi.org/10.1250/ast.36.377)

**[XREF0578]** Okuzono, Takeshi; Otsuru, Toru; Tomiku, Reiji; et al. (2014). A finite-element method using dispersion reduced spline elements for room acoustics simulation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2013.12.010](https://doi.org/10.1016/j.apacoust.2013.12.010)

**[XREF0789]** Okuzono, Takeshi; Sakagami, Kimihiro (2015). A finite-element formulation for room acoustics simulation with microperforated panel sound absorbing structures: Verification with electro-acoustical equivalent circuit theory and wave theory. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2015.02.012](https://doi.org/10.1016/j.apacoust.2015.02.012)

**[XREF0272]** Okuzono, Takeshi; Yoshida, Takumi; Sakagami, Kimihiro (2021). Efficiency of room acoustic simulations with time-domain FEM including frequency-dependent absorbing boundary conditions: Comparison with frequency-domain FEM. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2021.108212](https://doi.org/10.1016/j.apacoust.2021.108212)

**[XREF0172]** Okuzono, Takeshi; Yoshida, Takumi; Sakagami, Kimihiro; et al. (2016). An explicit time-domain finite element method for room acoustics simulations: Comparison of the performance with implicit methods. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2015.10.027](https://doi.org/10.1016/j.apacoust.2015.10.027)

**[XREF0135]** Oldham, DJ (2024). Reverberation Reinforcement by Means of Electro-Acoustic Coupling. *Acoustics '79*. DOI: [10.25144/23470](https://doi.org/10.25144/23470)

**[XREF0071]** Olsson, Jörgen; Linderholt, Andreas; Jarnerö, Kirsi; et al. (2023). Incremental use of FFT as a solution for low BT-product reverberation time measurements. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2022.109191](https://doi.org/10.1016/j.apacoust.2022.109191)

**[XREF0838]** Olynyk, D.; Northwood, T. D. (1964). Comparison of Reverberation-Room and Impedance-Tube Absorption Measurements. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.1919339](https://doi.org/10.1121/1.1919339)

**[XREF0364]** Otshudi, L; Guilhot, JP; Charles, JL (2024). Overview of Techniques for Measuring Impulse Response in Room Acoustics. *Acoustics '88*. DOI: [10.25144/21794](https://doi.org/10.25144/21794)

**[XREF0414]** Ouis, D (1999). Scattering by a barrier in a room. *Applied Acoustics*. DOI: [10.1016/s0003-682x(98)00014-0](https://doi.org/10.1016/s0003-682x(98)00014-0)

**[XREF0216]** Özgenel, Çağlar Fırat; Sorguç, Arzu Gönenç (2012). A New Method of Curve Fitting for Calculation of Reverberation Time From Impulse Responses With Insufficient Length. *ASME 2012 Noise Control and Acoustics Division Conference*. DOI: [10.1115/ncad2012-1349](https://doi.org/10.1115/ncad2012-1349)

**[XREF0616]** Ozimek, E.; Rutkowski, L. (1989). Deformation of frequency modulated (FM) signals propagating in a room. *Applied Acoustics*. DOI: [10.1016/0003-682x(89)90055-8](https://doi.org/10.1016/0003-682x(89)90055-8)

**[XREF0700]** Pallet, D.S.; Bartel, T.W.; Voorhees, C.R. (1976). Recent Reverberation Room Qualification Studies at the National Bureau of Standards. *Noise Control Engineering*. DOI: [10.3397/1.2832038](https://doi.org/10.3397/1.2832038)

**[XREF0010]** Pallett, D.S.; Pierce, E.T.; Toth, D.D. (1976). A small-scale multi-purpose reverberation room. *Applied Acoustics*. DOI: [10.1016/0003-682x(76)90010-4](https://doi.org/10.1016/0003-682x(76)90010-4)

**[XREF0014]** Pan, Lili; Zhao, Yuezhe; Gao, Jianliang (2020). Factors influencing scattering coefficient measurement accuracy in scaled reverberation room. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2019.107072](https://doi.org/10.1016/j.apacoust.2019.107072)

**[XREF0340]** Papadakis, Nikos; Stavroulakis, Georgios E. (2015). Time domain finite element method for the calculation of impulse response of enclosed spaces. Room acoustics application. *AIP Conference Proceedings*. DOI: [10.1063/1.4939430](https://doi.org/10.1063/1.4939430)

**[XREF0859]** Papadopoulos, Christos I. (2001). Redistribution of the low frequency acoustic modes of a room: a finite element-based optimisation method. *Applied Acoustics*. DOI: [10.1016/s0003-682x(01)00002-0](https://doi.org/10.1016/s0003-682x(01)00002-0)

**[XREF0774]** Parada, Pablo Peso; Sharma, Dushyant; Naylor, Patrick A. (2014). Non-intrusive estimation of the level of reverberation in speech. *2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2014.6854497](https://doi.org/10.1109/icassp.2014.6854497)

**[XREF0844]** Passero, Carolina Reich Marcon; Zannin, Paulo Henrique Trombetta (2010). Statistical comparison of reverberation times measured by the integrated impulse response and interrupted noise methods, computationally simulated with ODEON software, and calculated by Sabine, Eyring and Arau-Puchades’ formulas. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2010.07.003](https://doi.org/10.1016/j.apacoust.2010.07.003)

**[XREF0642]** Patchett, Brian (2021). Modeling time reversal focusing amplitudes in a reverberation chamber using a modal summation approach.. *Unspecified venue*. DOI: [10.26226/morressier.606f15dd30a2e980041f240b](https://doi.org/10.26226/morressier.606f15dd30a2e980041f240b)

**[XREF0801]** Paulo, Joel Preto; Martins, Carlos Rodrigues; Bento Coelho, J.L. (2009). A hybrid MLS technique for room impulse response estimation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2008.07.007](https://doi.org/10.1016/j.apacoust.2008.07.007)

**[XREF0572]** Pawlak, Alan; Lee, Hyunkook (2026). Spatial segmentation of impulse response for room reflection analysis and auralization. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2026.111303](https://doi.org/10.1016/j.apacoust.2026.111303)

**[XREF0199]** Pearse, J; Healy, A (2023). Design Construction and Commissioning of a Reverberation Room. *Inter-Noise 2022*. DOI: [10.25144/14769](https://doi.org/10.25144/14769)

**[XREF0741]** Pearse, John; Healey, Aaron (2023). Design, construction and commissioning of a reverberation room. *INTER-NOISE and NOISE-CON Congress and Conference Proceedings*. DOI: [10.3397/in_2022_0287](https://doi.org/10.3397/in_2022_0287)

**[XREF0454]** Peer, Itai; Rafaely, Boaz; Zigel, Yaniv (2008). Reverberation matching for speaker recognition. *2008 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2008.4518738](https://doi.org/10.1109/icassp.2008.4518738)

**[XREF0054]** Peer, Itai; Rafaely, Boaz; Zigel, Yaniv (2008). Room Acoustics Parameters Affecting Speaker Recognition Degradation Under Reverberation. *2008 Hands-Free Speech Communication and Microphone Arrays*. DOI: [10.1109/hscma.2008.4538705](https://doi.org/10.1109/hscma.2008.4538705)

**[XREF0273]** Pei, Xingyuan; Ma, Xiaochuan; Li, Xuan; et al. (2024). Underwater reverberation suppression method by the symmetry of signal and centrosymmetric arrays. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2024.109986](https://doi.org/10.1016/j.apacoust.2024.109986)

**[XREF0793]** Pelzer, Soenke; Vorländer, Michael (2013). Inversion of a room acoustics model for the determination of acoustical surface properties in enclosed spaces. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4800297](https://doi.org/10.1121/1.4800297)

**[XREF0450]** Peng Jianxin; Peng, Jiang (2018). The Effects of the Noise and Reverberation on the Working Memory Span of Children. *Archives of Acoustics*. DOI: [10.24425/118087](https://doi.org/10.24425/118087)

**[XREF0818]** Pennock, Rachael; Rallapalli, Varsha; Souza, Pamela (2025). Language environment analysis precision in real-world environments. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002075](https://doi.org/10.1121/2.0002075)

**[XREF0030]** Pereira, Andreia; Gaspar, Anna; Godinho, Luís; et al. (2026). Influence of Sound Scattering on the Reverberation Time of a Shoebox Auditorium Using Room Acoustics Modelling. *Applied Sciences*. DOI: [10.3390/app16041960](https://doi.org/10.3390/app16041960)

**[XREF0158]** Petkov, Petko N.; Stylianou, Yannis (2017). Adaptive gain control and time warp for enhanced speech intelligibility under reverberation. *2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2017.7952244](https://doi.org/10.1109/icassp.2017.7952244)

**[XREF0638]** Petrick, Rico; Unoki, Masashi; Mittal, Anish; et al. (2008). A comprehensive study on the effects of room reverberation on fundamental frequency estimation. *Interspeech 2008*. DOI: [10.21437/interspeech.2008-30](https://doi.org/10.21437/interspeech.2008-30)

**[XREF0402]** Pierce, Allan D. (2019). Room Acoustics. *Acoustics*. DOI: [10.1007/978-3-030-11214-1_6](https://doi.org/10.1007/978-3-030-11214-1_6)

**[XREF0418]** Pihl, Jörgen (2011). Mid Frequency Bottom-Interacting Sound Propagation and Reverberation in the Baltic. *The Open Acoustics Journal*. DOI: [10.2174/1874837601104010001](https://doi.org/10.2174/1874837601104010001)

**[XREF0453]** Polack, Jean-Dominique (1993). Playing billiards in the concert hall: The mathematical foundations of geometrical room acoustics. *Applied Acoustics*. DOI: [10.1016/0003-682x(93)90054-a](https://doi.org/10.1016/0003-682x(93)90054-a)

**[XREF0403]** Poletti, M. A. (2011). Active Acoustic Systems for the Control of Room Acoustics. *Building Acoustics*. DOI: [10.1260/1351-010x.18.3-4.237](https://doi.org/10.1260/1351-010x.18.3-4.237)

**[XREF0785]** Porcinai, Emanuele; Lepa, Steffen; Klein, Paula; et al. (2025). Stage acoustics for chamber music ensembles: a laboratory study with auralised rooms. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002166](https://doi.org/10.1121/2.0002166)

**[XREF0781]** Pörschmann, Christoph; Stade, Philipp; Arend, Johannes M. (2017). Binaural auralization of proposed room modifications based on measured omnidirectional room impulse responses. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0000622](https://doi.org/10.1121/2.0000622)

**[XREF0159]** Postma, Bart N.J. (2013). A History of the Use of Time Intervals after the Direct Sound in Concert Hall Design before the Reverberation Formula of Sabine Became Generally Accepted. *Building Acoustics*. DOI: [10.1260/1351-010x.20.2.157](https://doi.org/10.1260/1351-010x.20.2.157)

**[XREF0502]** Postma, Barteld; Green, Evan; Kahle, Eckhard; et al. (2021). Pre-Sabine Room Acoustic Guidelines on Audience Rake, Stage Acoustics, and Dimension Ratios. *Acoustics*. DOI: [10.3390/acoustics3020017](https://doi.org/10.3390/acoustics3020017)

**[XREF0580]** Postma, Barteld N. J.; Katz, Brian F. G. (2020). Forum—Pre-Sabine room acoustic assumptions on reverberation and their influence on room acoustic design. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0001082](https://doi.org/10.1121/10.0001082)

**[XREF0362]** Prawda, K.; Schlecht, S.; Välimäki, V. (2022). Time Variance in Measured Room Impulse Responses. *Proceedings of the 10th Convention of the European Acoustics Association Forum Acusticum 2023*. DOI: [10.61782/fa.2023.0398](https://doi.org/10.61782/fa.2023.0398)

**[XREF0594]** Preston, John (2014). Some results from the very shallow water TREX13 reverberation experiments using the Five Octave Research Array triplet module. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4866141](https://doi.org/10.1121/1.4866141)

**[XREF0499]** Prinn, Albert; Badeau, Roland (2025). Verification of statistical wave field theory reverberation time predictions. *Unspecified venue*. DOI: [10.2139/ssrn.5865382](https://doi.org/10.2139/ssrn.5865382)

**[XREF0771]** Prinn, Albert; Xu, Zeyu; Habets, Emanuël (2025). Measures of validation for computational room acoustics. *Proceedings of the 11th Convention of the European Acoustics Association Forum Acusticum / EuroNoise 2025*. DOI: [10.61782/fa.2025.0554](https://doi.org/10.61782/fa.2025.0554)

**[XREF0307]** Prinn, Albert G. (2023). A Review of Finite Element Methods for Room Acoustics. *Acoustics*. DOI: [10.3390/acoustics5020022](https://doi.org/10.3390/acoustics5020022)

**[XREF0037]** Prinn, Albert G.; Badeau, Roland (2026). Verification of statistical wave field theory reverberation time predictions. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2026.111337](https://doi.org/10.1016/j.apacoust.2026.111337)

**[XREF0077]** Prinn, Albert G.; Tuna, Çağdaş; Walther, Andreas; et al. (2025). A study of the spatial non-uniformity of reverberation time at low frequencies. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2024.110220](https://doi.org/10.1016/j.apacoust.2024.110220)

**[XREF0843]** Prokofieva, E (2023). Effect of Presence of Furnishings in Designed and Measured Reverberation Time in School Spaces. *Inter-Noise 2022*. DOI: [10.25144/14780](https://doi.org/10.25144/14780)

**[XREF0067]** Putra, Jouvan Chandra Pratama; Rahmaniar, Irna; Rabiyanti (2018). Evaluation of reverberation time of class room. *AIP Conference Proceedings*. DOI: [10.1063/1.5042898](https://doi.org/10.1063/1.5042898)

**[XREF0217]** Putri, Tabita Febriawaty Kartika; Setyowati, Erni (2023). Reverberation Time Improvement in the Worship Room of Palihan Javanese Christian Church. *Journal of Architectural Research and Design Studies*. DOI: [10.20885/jars.vol7.iss2.art3](https://doi.org/10.20885/jars.vol7.iss2.art3)

**[XREF0119]** Puyana-Romero, Virginia; Núñez-Solano, Daniel; Hernández Molina, Ricardo; et al. (2020). Reverberation time measurements of a neonatal incubator. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2020.107374](https://doi.org/10.1016/j.apacoust.2020.107374)

**[XREF0692]** Puyana-Romero, Virginia; Nuñez-Solano, Daniel; Hernández-Molina, Ricardo; et al. (2025). Designing for Neonates’ Wellness: Differences in the Reverberation Time Between an Incubator Located in an Open Unit and in a Private Room of a NICU. *Buildings*. DOI: [10.3390/buildings15091411](https://doi.org/10.3390/buildings15091411)

**[XREF0690]** Qin, Feng; Chen, Jiangping; Chen, Zhijiu (2006). Acoustic characterization and prediction for fan-duct-plenum-room integrations. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2005.06.006](https://doi.org/10.1016/j.apacoust.2005.06.006)

**[XREF0498]** Qiu, Wenhao; Wang, Gang (2025). DRR-based acoustic detection model for estimating room shape. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2024.110216](https://doi.org/10.1016/j.apacoust.2024.110216)

**[XREF0206]** Qu, Sichao; Yang, Min; Xu, Yunfei; et al. (2023). Reverberation time control by acoustic metamaterials in a small room. *Building and Environment*. DOI: [10.1016/j.buildenv.2023.110753](https://doi.org/10.1016/j.buildenv.2023.110753)

**[XREF0837]** Raimond, A.; Watkins, A. J. (2009). Factors affecting a loudness asymmetry in real-room reverberation.. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.3249220](https://doi.org/10.1121/1.3249220)

**[XREF0834]** Raimond, Andrew; Watkins, Anthony (2012). Loudness asymmetry in real-room reverberation: cross-band effects. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4709017](https://doi.org/10.1121/1.4709017)

**[XREF0836]** Rajapaksha, Tilak; Qiu, Xiaojun; Cheng, Eva; et al. (2016). Geometrical room geometry estimation from room impulse responses. *2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2016.7471691](https://doi.org/10.1109/icassp.2016.7471691)

**[XREF0240]** Ratnam, Rama; Jones, Douglas L.; Wheeler, Bruce C.; et al. (2003). Online estimation of room reverberation time. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4780517](https://doi.org/10.1121/1.4780517)

**[XREF0654]** Rau, Mark; Scavone, Gary (2025). Vibration measurements comparing the contrabass and octobass. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002152](https://doi.org/10.1121/2.0002152)

**[XREF0148]** Ren, Gang; Bocko, Mark; Headlam, Dave (2010). Blind Deconvolution of Quadratic Time-Frequency Representations of Musical Signals for Reverberation Feature Extraction. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.3567151](https://doi.org/10.1121/1.3567151)

**[XREF0813]** Ren, Gang; Bocko, Mark; Headlam, Dave (2010). Statistical Spectrogram Modeling and Analysis for Blind Estimation of Room Acoustics from Musical Recordings. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.3560897](https://doi.org/10.1121/1.3560897)

**[XREF0486]** Reynders, Edwin P.B.; Van den Wyngaert, Jan; Verlinden, Marie; et al. (2024). Development and performance assessment of sound absorbing chandeliers for reverberation control and improved verbal communication in large rooms. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2024.109874](https://doi.org/10.1016/j.apacoust.2024.109874)

**[XREF0222]** Ribay, Guillemette; de Rosny, Julien; Fink, Mathias (2005). Time reversal of noise sources in a reverberation room. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.1886385](https://doi.org/10.1121/1.1886385)

**[XREF0607]** Rindel, Jens Holger (2017). Introduction to room acoustics. *Sound Insulation in Buildings*. DOI: [10.1201/9781351228206-5](https://doi.org/10.1201/9781351228206-5)

**[XREF0141]** Rittenschober, Thomas; Decloux, Antoine (2025). Robust 3D Localisation of Anomalies in Reverberation Time Signals. *Proceedings of the 11th Convention of the European Acoustics Association Forum Acusticum / EuroNoise 2025*. DOI: [10.61782/fa.2025.0754](https://doi.org/10.61782/fa.2025.0754)

**[XREF0182]** Rodríguez, Orlando Camargo (2023). Scattering and Reverberation. *Fundamentals of Underwater Acoustics*. DOI: [10.1007/978-3-031-31319-6_9](https://doi.org/10.1007/978-3-031-31319-6_9)

**[XREF0737]** Romanenko, G; Vorländer, M (2023). Employment of Spherical Wave Reflection Coefficient in Room Acoustics. *Research Symposium 2003*. DOI: [10.25144/18102](https://doi.org/10.25144/18102)

**[XREF0809]** Romero, J; Fazenda, B; Atmoko, H (2023). Characterisation of Small Room Acoustics for Audio Production. *Reproduced Sound 2009*. DOI: [10.25144/17458](https://doi.org/10.25144/17458)

**[XREF0132]** Ronsse, Lauren M.; Wang, Lily M. (2009). Effects of room reverberation time and receiver position on measured binaural room impulse responses.. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4783280](https://doi.org/10.1121/1.4783280)

**[XREF0808]** Rose, Jay (2008). Microphones and Room Acoustics. *Producing Great Sound for Film and Video*. DOI: [10.1016/b978-0-240-80970-0.50016-3](https://doi.org/10.1016/b978-0-240-80970-0.50016-3)

**[XREF0794]** Rutkowski, Leon (1996). A comparison of the frequency modulation transfer function with the modulation transfer function in a Room. *Applied Acoustics*. DOI: [10.1016/s0003-682x(96)00028-x](https://doi.org/10.1016/s0003-682x(96)00028-x)

**[XREF0591]** Rychtáriková, Monika; Bogaert, Tim van den; Vermeir, Gerrit; et al. (2011). Perceptual validation of virtual room acoustics: Sound localisation and speech understanding. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2010.11.012](https://doi.org/10.1016/j.apacoust.2010.11.012)

**[XREF0241]** Saarelma, J; Greco, G (2023). Sound Field Visualization using the Finite-Difference Time-Domain Method and Measured Spatial Room Impulse Responses. *Auditorium Acoustics 2015*. DOI: [10.25144/16155](https://doi.org/10.25144/16155)

**[XREF0429]** Saarelma, Jukka; Savioja, Lauri (2019). Spatial analysis of modal time evolution in room acoustics. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.5087997](https://doi.org/10.1121/1.5087997)

**[XREF0147]** Sadjadi, Seyed Omid; Boril, Hynek; Hansen, John H.L. (2012). A comparison of front-end compensation strategies for robust LVCSR under room reverberation and increased vocal effort. *2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2012.6288968](https://doi.org/10.1109/icassp.2012.6288968)

**[XREF0523]** Sadjadi, Seyed Omid; Hansen, John H.L. (2012). Blind reverberation mitigation for robust speaker identification. *2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2012.6288851](https://doi.org/10.1109/icassp.2012.6288851)

**[XREF0605]** Saji, Akira; Tanno, Keita; Huang, Jie (2011). Creation 3D sound by using HRTF with Room reverberation. *Principles And Applications Of Spatial Hearing*. DOI: [10.1142/9789814299312_0043](https://doi.org/10.1142/9789814299312_0043)

**[XREF0290]** Sakagami, Kimihiro; Uyama, Toru; Morimoto, Masayuki; et al. (2005). Prediction of the reverberation absorption coefficient of finite-size membrane absorbers. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2004.02.006](https://doi.org/10.1016/j.apacoust.2004.02.006)

**[XREF0091]** Sakamoto, Noriaki; Otsuru, Toru; Tomiku, Reiji; et al. (2018). Reproducibility of sound absorption and surface impedance of materials measured in a reverberation room using ensemble averaging technique with a pressure-velocity sensor and improved calibration. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2018.08.009](https://doi.org/10.1016/j.apacoust.2018.08.009)

**[XREF0086]** Samarasinghe, Prasanga N.; Abhayapala, Thushara D. (2017). Blind estimation of directional properties of room reverberation using a spherical microphone array. *2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2017.7952176](https://doi.org/10.1109/icassp.2017.7952176)

**[XREF0106]** Santos, Joao F.; Peters, Nils; Falk, Tiago H. (2013). Towards blind reverberation time estimation for non-speech signals. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4800598](https://doi.org/10.1121/1.4800598)

**[XREF0893]** Sarinho Filho, José N.; Thomazelli, Rodolfo; Masiero, Bruno (2025). Optimizing low-frequency reverberation in a critical listening room. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0037837](https://doi.org/10.1121/10.0037837)

**[XREF0832]** Sasaoka, Jun; Kawai, Keiji (2025). An attempt to apply the element-free Galerkin method to room acoustics simulation. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002034](https://doi.org/10.1121/2.0002034)

**[XREF0350]** Savioja, Lauri (2020). Simulation-Based Auralization of Room Acoustics. *Acoustics Today*. DOI: [10.1121/at.2020.16.4.48](https://doi.org/10.1121/at.2020.16.4.48)

**[XREF0089]** Schiildt, Christian; Handel, Peter (2013). Blind low-complexity estimation of reverberation time. *2013 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics*. DOI: [10.1109/waspaa.2013.6701875](https://doi.org/10.1109/waspaa.2013.6701875)

**[XREF0565]** Schneider, Martin; Kellermann, Walter (2013). Large-scale Multiple Input/Multiple Output system identification in room acoustics. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4801402](https://doi.org/10.1121/1.4801402)

**[XREF0192]** Schnitta, Bonnie (2013). Achieving optimal reverberation time in a room, using newly patented tuning tubes. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4806188](https://doi.org/10.1121/1.4806188)

**[XREF0153]** Schnitta, Bonnie; Mittendorf, Steve (2013). Reduction in reverberation time, resulting from a unique acoustic treatment behind the final surface layer of drywall. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4801405](https://doi.org/10.1121/1.4801405)

**[XREF0831]** Schoder, Stefan (2024). Physics-Informed Neural Networks for Modal Wave Field Predictions in 3D Room Acoustics. *Unspecified venue*. DOI: [10.20944/preprints202411.1848.v1](https://doi.org/10.20944/preprints202411.1848.v1)

**[XREF0200]** Schroeder, M. R.; Gerlach, R. (1974). Diffusion, Room Shape, and Absorber Location: Influence on Reverberation Time. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.3437322](https://doi.org/10.1121/1.3437322)

**[XREF0210]** Schroeder, M. R.; Gerlach, R. (1974). Diffusion, room shape and absorber location—influence on reverberation time. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.1903424](https://doi.org/10.1121/1.1903424)

**[XREF0817]** Schroeder, Manfred R. (1980). Acoustics in human communications: Room acoustics, music, and speech. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.384628](https://doi.org/10.1121/1.384628)

**[XREF0130]** Schuldt, Christian; Handel, Peter (2015). Noise robust integration for blind and non-blind reverberation time estimation. *2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2015.7177931](https://doi.org/10.1109/icassp.2015.7177931)

**[XREF0361]** Schultz, Theodore J. (1980). Reverberation time in occupied concert halls calculated from measured reverberation time in unoccupied halls. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2018427](https://doi.org/10.1121/1.2018427)

**[XREF0227]** Schultz, Theodore J. (1980). Room acoustics. *Applied Acoustics*. DOI: [10.1016/0003-682x(80)90032-8](https://doi.org/10.1016/0003-682x(80)90032-8)

**[XREF0878]** Schwarz, Andreas; Reindl, Klaus; Kellermann, Walter (2012). A two-channel reverberation suppression scheme based on blind signal separation and wiener filtering. *2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2012.6287830](https://doi.org/10.1109/icassp.2012.6287830)

**[XREF0609]** Sehr, Armin; Maas, Roland; Kellermann, Walter (2011). Frame-wise HMM adaptation using state-dependent reverberation estimates. *2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2011.5947600](https://doi.org/10.1109/icassp.2011.5947600)

**[XREF0751]** Sekiguchi, Katsuaki; Kimura, Sho (1991). Calculation of sound field in a room by finite sound ray integration method. *Applied Acoustics*. DOI: [10.1016/0003-682x(91)90053-h](https://doi.org/10.1016/0003-682x(91)90053-h)

**[XREF0183]** Senoussaoui, Mohammed; Santos, Joao F.; Falk, Tiago H. (2017). Speech temporal dynamics fusion approaches for noise-robust reverberation time estimation. *2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2017.7953217](https://doi.org/10.1109/icassp.2017.7953217)

**[XREF0353]** Shabtai, Noam R.; Rafaely, Boaz; Zigel, Yaniv (2011). The effect of reverberation on the performance of cepstral mean subtraction in speaker verification. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2010.09.009](https://doi.org/10.1016/j.apacoust.2010.09.009)

**[XREF0041]** Shalkouhi, P. Jafari (2009). Classroom Reverberation Time Based on the Sabine Equation. *Building Acoustics*. DOI: [10.1260/135101009790291255](https://doi.org/10.1260/135101009790291255)

**[XREF0411]** Shalkouhi, P. Jafari; Hodgson, Murray (2012). Comments on “Empirical Prediction of Speech Levels and Reverberation in Classrooms” [Build Acoust, 8(1), 1–14, 2001]. *Building Acoustics*. DOI: [10.1260/1351-010x.19.2.139](https://doi.org/10.1260/1351-010x.19.2.139)

**[XREF0677]** Sheaffer, Jonathan; Webb, Craig; Fazenda, Bruno M. (2013). Modelling binaural receivers in finite difference simulation of room acoustics. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4800195](https://doi.org/10.1121/1.4800195)

**[XREF0357]** Shelton, J; Dufournet, D (2024). Short Reverberation Time Measurement. *Windermere at Stratford*. DOI: [10.25144/18826](https://doi.org/10.25144/18826)

**[XREF0337]** Shen, Justin; Duraiswami, Ramani (2020). Data-driven feedback delay network construction for real-time virtual room acoustics. *Proceedings of the 15th International Audio Mostly Conference*. DOI: [10.1145/3411109.3411145](https://doi.org/10.1145/3411109.3411145)

**[XREF0715]** Shen, X (2004). Optimization of the locations of the loudspeaker and absorption material in a small room. *Applied Acoustics*. DOI: [10.1016/s0003-682x(04)00034-9](https://doi.org/10.1016/s0003-682x(04)00034-9)

**[XREF0712]** Shi, Liheng; Lin, Ju (2024). Renovation of the reverberation room at Ocean University of China based on ODEON simulation. *Sound & Vibration*. DOI: [10.59400/sv1680](https://doi.org/10.59400/sv1680)

**[XREF0142]** Shijin, Chen; Yuxuan, Zhang; Sheng, Yan; et al. (2021). A Novel Reverberation Mitigation Method Based On MIMO Sonar Space-Time Adaptive Processing. *2021 OES China Ocean Acoustics (COA)*. DOI: [10.1109/coa50123.2021.9519907](https://doi.org/10.1109/coa50123.2021.9519907)

**[XREF0622]** Shtrepi, Louena; Astolfi, Arianna; Rychtáriková, Monika (2014). The Influence of a Volume Scale-Factor on Scattering Coefficient Effects in Room Acoustics. *Building Acoustics*. DOI: [10.1260/1351-010x.21.2.153](https://doi.org/10.1260/1351-010x.21.2.153)

**[XREF0034]** Shtrepi, Louena; Prato, Andrea (2020). Towards a sustainable approach for sound absorption assessment of building materials: Validation of small-scale reverberation room measurements. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2020.107304](https://doi.org/10.1016/j.apacoust.2020.107304)

**[XREF0633]** Sibo, Li; Rui, Guo; Dianlun, Zhang; et al. (2024). Low False-Alarm Active Sonar Detection under Strong Reverberation. *2024 OES China Ocean Acoustics (COA)*. DOI: [10.1109/coa58979.2024.10723477](https://doi.org/10.1109/coa58979.2024.10723477)

**[XREF0294]** Silaban, Yessy Christanti; Setyowati, Erni; Hardiman, Gagoek (2018). The Effects of Acoustic Material as Absorption Material for Music Room Design by Using Reverberation Time Graph. *Advanced Science Letters*. DOI: [10.1166/asl.2018.13168](https://doi.org/10.1166/asl.2018.13168)

**[XREF0603]** Simon, Blake; Isakson, Marcia; Ballard, Megan (2018). Modeling acoustic wave propagation and reverberation in an ice covered environment using finite element analysis. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0000842](https://doi.org/10.1121/2.0000842)

**[XREF0066]** Sinal, Özgün; Yilmazer, Semiha (2018). Effects of perceived singing effort on classical singers’ reverberation time preferences towards music practice rooms. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2018.02.013](https://doi.org/10.1016/j.apacoust.2018.02.013)

**[XREF0497]** Skålevik, M (2023). Music, Room, Two Ears, Design and Paradigms. *Auditorium Acoustics 2023*. DOI: [10.25144/16029](https://doi.org/10.25144/16029)

**[XREF0680]** Skålevik, M (2023). Room Acoustic Parameters and Their Distribution over Concert Hall Seats. *Auditorium Acoustics 2008*. DOI: [10.25144/17515](https://doi.org/10.25144/17515)

**[XREF0026]** Skarlatos, Dimitris (1994). A time series approach on reverberation of large enclosures. *Applied Acoustics*. DOI: [10.1016/0003-682x(94)90002-7](https://doi.org/10.1016/0003-682x(94)90002-7)

**[XREF0760]** Sorenson, Steve (1997). Steady State Reverberation Time Measurement. *SAE Technical Paper Series*. DOI: [10.4271/972032](https://doi.org/10.4271/972032)

**[XREF0105]** Spa, Carlos; Garriga, Adan; Escolano, Jose (2010). Impedance boundary conditions for pseudo-spectral time-domain methods in room acoustics. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2009.11.015](https://doi.org/10.1016/j.apacoust.2009.11.015)

**[XREF0886]** Støfringsdal, B (2023). Integrated Room Acoustic and Electro-Acoustic Design - the Concert Venue at Rockheim, Norway. *Auditorium Acoustics 2011*. DOI: [10.25144/16821](https://doi.org/10.25144/16821)

**[XREF0452]** Staffeldt, Henrik (1993). Modelling of room acoustics and loudspeakers in JBL's complex array design program CADP2. *Applied Acoustics*. DOI: [10.1016/0003-682x(93)90050-g](https://doi.org/10.1016/0003-682x(93)90050-g)

**[XREF0109]** Stanzial, Domenico (1990). On the equation for reverberation time in acoustics. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2028829](https://doi.org/10.1121/1.2028829)

**[XREF0204]** Stauskis, V. J. (1998). The late reverberation time as a new criteria for the evaluation of hall acoustics. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.421707](https://doi.org/10.1121/1.421707)

**[XREF0704]** Steinmetz, Christian J.; Ithapu, Vamsi Krishna; Calamia, Paul (2021). Filtered Noise Shaping for Time Domain Room Impulse Response Estimation from Reverberant Speech. *2021 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)*. DOI: [10.1109/waspaa52581.2021.9632680](https://doi.org/10.1109/waspaa52581.2021.9632680)

**[XREF0270]** Stephenson, Uwe Martin (2006). Analytical derivation of a formula for the reduction of computation time by the voxel crossing technique used in room acoustical simulation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2006.01.005](https://doi.org/10.1016/j.apacoust.2006.01.005)

**[XREF0754]** Stewart, Noral D.; Montano, Walter (2025). Before the decibel, acoustical measurements – physical, aural, electrical. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002079](https://doi.org/10.1121/2.0002079)

**[XREF0023]** Strøm, S.; Krokstad, A.; Sørsdal, S.; et al. (1986). Design of room acoustics and a MCR reverberation system for Bjergsted concert hall in Stavanger. *Applied Acoustics*. DOI: [10.1016/0003-682x(86)90040-x](https://doi.org/10.1016/0003-682x(86)90040-x)

**[XREF0121]** Strasser, Helmut; Gruen, Kristina; Koch, Werner (2000). Office acoustics: Analyzing reverberation time and subjective evaluation. *Occupational Ergonomics*. DOI: [10.3233/oer-2000-2201](https://doi.org/10.3233/oer-2000-2201)

**[XREF0899]** SU, S (2023). Harnessing Generative AI Exploring the Synergy of Chatgpt and Grasshopper for Room Acoustic Design. *Acoustics 2023*. DOI: [10.25144/16620](https://doi.org/10.25144/16620)

**[XREF0194]** Sudarsono, Anugrah S.; Merthayasa, I. G. N.; Suprijanto (2015). Comparison between psycho-acoustics and physio-acoustic measurement to determine optimum reverberation time of pentatonic angklung music concert hall. *AIP Conference Proceedings*. DOI: [10.1063/1.4930753](https://doi.org/10.1063/1.4930753)

**[XREF0458]** Sumarac-Pavlovic, Dragana; Mijic, Miomir; Kurtovic, Husnija (2008). A simple impulse sound source for measurements in room acoustics. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2006.11.003](https://doi.org/10.1016/j.apacoust.2006.11.003)

**[XREF0698]** Summers, Jason (2011). Effects of surface scattering and room shape on the correspondence between statistical- and geometrical-acoustics model predictions.. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.3626897](https://doi.org/10.1121/1.3626897)

**[XREF0047]** Sun, Haitao; Sang, Xiangrui (2025). The Study on Scattering Coefficient Measurement Using a “Spark Train” in a Scaled Reverberation Room. *Acoustics Australia*. DOI: [10.1007/s40857-025-00372-0](https://doi.org/10.1007/s40857-025-00372-0)

**[XREF0276]** Szłapa, Piotr; Boroń, Marta; Zachara, Jolanta; et al. (2016). A Comparison of Handgun Shots, Balloon Bursts, and a Compressor Nozzle Hiss as Sound Sources for Reverberation Time Assessment. *Archives of Acoustics*. DOI: [10.1515/aoa-2016-0065](https://doi.org/10.1515/aoa-2016-0065)

**[XREF0652]** Szeląg, Agata; Zastawnik, Marcin (2025). Issues in the Design and Validation of Coupled Reverberation Roomsfor Testing Acoustic Insulation of Building Partitions. *Archives of Acoustics*. DOI: [10.24425/aoa.2025.153646](https://doi.org/10.24425/aoa.2025.153646)

**[XREF0554]** Szymanski, Jeff (2008). Small Room Acoustics. *Handbook for Sound Engineers*. DOI: [10.1016/b978-0-240-80969-4.50009-2](https://doi.org/10.1016/b978-0-240-80969-4.50009-2)

**[XREF0036]** Tahara, Y; Miyajima, T (1998). A New Approach to Optimum Reverberation Time Characteristics. *Applied Acoustics*. DOI: [10.1016/s0003-682x(97)00072-8](https://doi.org/10.1016/s0003-682x(97)00072-8)

**[XREF0489]** Talbot-Smith, Michael (1990). Room acoustics. *Broadcast Sound Technology*. DOI: [10.1016/b978-0-408-05442-3.50011-x](https://doi.org/10.1016/b978-0-408-05442-3.50011-x)

**[XREF0436]** Talbot-Smith, Michael (2023). Room acoustics. *Broadcast Sound Technology*. DOI: [10.4324/9781003460510-5](https://doi.org/10.4324/9781003460510-5)

**[XREF0611]** Talebzadeh, Arezoo (2024). Need for an inclusive approach in soundscape research. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002214](https://doi.org/10.1121/2.0002214)

**[XREF0125]** Talmon, Ronen; Habets, Emanuel A. P. (2013). Blind reverberation time estimation by intrinsic modeling of reverberant speech. *2013 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2013.6637628](https://doi.org/10.1109/icassp.2013.6637628)

**[XREF0224]** Tanaka, Toshiaki; Hama, Yoshinori; Nakamura, Yoshiyuki; et al. (2003). Time and spatial coherency of bottom reverberation. *The Journal of the Marine Acoustics Society of Japan*. DOI: [10.3135/jmasj.30.189](https://doi.org/10.3135/jmasj.30.189)

**[XREF0568]** Tardini, Virginia; Fratoni, Giulia; D’Orazio, Dario (2025). National Standards on Classroom Acoustics: Key descriptors and global perspectives. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002062](https://doi.org/10.1121/2.0002062)

**[XREF0693]** Thomas, Mark R. P.; Tashev, Ivan J.; Lim, Felicia; et al. (2014). Optimal beamforming as a time domain equalization problem with application to room acoustics. *2014 14th International Workshop on Acoustic Signal Enhancement (IWAENC)*. DOI: [10.1109/iwaenc.2014.6953341](https://doi.org/10.1109/iwaenc.2014.6953341)

**[XREF0166]** Thompson, Stephen C. (2025). Results from a time-domain clarinet model: articulation and changing notes. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002140](https://doi.org/10.1121/2.0002140)

**[XREF0661]** Thomson, DJ; Brooke, GH; Hamm, C; et al. (2023). A Spectral Decomposition Procedure for Determining the Fields at the Boundaries in a Pe-Based Reverberation Model. *Seabed and Sediment Acoustics 2015*. DOI: [10.25144/16039](https://doi.org/10.25144/16039)

**[XREF0549]** Thydal, Tobias; Pind, Finnur; Jeong, Cheol-Ho; et al. (2021). Experimental validation and uncertainty quantification in wave-based computational room acoustics. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2021.107939](https://doi.org/10.1016/j.apacoust.2021.107939)

**[XREF0897]** Tian, Jiayong (2010). Comparison Between Reverberation-Ray Matrix, Reverberation-Transfer Matrix, and Generalized Reverberation Matrix. *Wave Propagation in Materials for Modern Applications*. DOI: [10.5772/6847](https://doi.org/10.5772/6847)

**[XREF0246]** Tkaczyk, Viktoria (2015). The Shot Is Fired Unheard: Sigmund Exner and the Physiology of Reverberation. *Grey Room*. DOI: [10.1162/grey_a_00179](https://doi.org/10.1162/grey_a_00179)

**[XREF0244]** Todorovic, Dejan; Mihajlovic, Mirjana; Ristanovic, Ivana; et al. (2012). The influence of the reverberation time on the distribution of sound field in critical listening room at low frequencies. *2012 20th Telecommunications Forum (TELFOR)*. DOI: [10.1109/telfor.2012.6419436](https://doi.org/10.1109/telfor.2012.6419436)

**[XREF0729]** Tohyama, Mikio (2008). Room Transfer Function. *Handbook of Signal Processing in Acoustics*. DOI: [10.1007/978-0-387-30441-0_75](https://doi.org/10.1007/978-0-387-30441-0_75)

**[XREF0231]** Tohyama, Mikio (2020). Room reverberation theory and transfer function. *Acoustic Signals and Hearing*. DOI: [10.1016/b978-0-12-816391-7.00014-0](https://doi.org/10.1016/b978-0-12-816391-7.00014-0)

**[XREF0713]** Tomiku, Reiji; Sakamoto, Shinichi; Okamoto, Noriko; et al. (2014). Room Acoustics Simlation. *Computational Simulation in Architectural and Environmental Acoustics*. DOI: [10.1007/978-4-431-54454-8_6](https://doi.org/10.1007/978-4-431-54454-8_6)

**[XREF0466]** Toole, Floyd (2025). Room Acoustics and Acoustical Devices. *Sound Reproduction*. DOI: [10.4324/9781003477495-13](https://doi.org/10.4324/9781003477495-13)

**[XREF0646]** Toyoda, Emi; Sakamoto, Shinichi; Tachibana, Hideki (2004). Effects of room shape and diffusing treatment on the measurement of sound absorption coefficient in a reverberation room. *Acoustical Science and Technology*. DOI: [10.1250/ast.25.255](https://doi.org/10.1250/ast.25.255)

**[XREF0327]** Toyota, Yasuhisa; Komoda, Motoo; Beckmann, Daniel; et al. (2020). Reverberation Time, Other Metrics, and Underlying Goals. *Concert Halls by Nagata Acoustics*. DOI: [10.1007/978-3-030-42450-3_37](https://doi.org/10.1007/978-3-030-42450-3_37)

**[XREF0432]** Tronchin, Lamberto (2021). Variability of room acoustic parameters with thermo-hygrometric conditions. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2021.107933](https://doi.org/10.1016/j.apacoust.2021.107933)

**[XREF0285]** Tsui, Chung Y.; Voorhees, Carl R.; Yang, Jackson C.S. (1976). The design of small reverberation chambers for transmission loss measurement. *Applied Acoustics*. DOI: [10.1016/0003-682x(76)90015-3](https://doi.org/10.1016/0003-682x(76)90015-3)

**[XREF0492]** Tudoce, Juliette; Reserbat-Plantey, Antoine; Lewenstein, Maciej; et al. (2025). Quantum-inspired reverberation using Feedback Delay Network. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002128](https://doi.org/10.1121/2.0002128)

**[XREF0012]** Tzekakis, E. (1988). Reverberation time prediction software. *Applied Acoustics*. DOI: [10.1016/0003-682x(88)90072-2](https://doi.org/10.1016/0003-682x(88)90072-2)

**[XREF0898]** Ueno, Kanako; Kopco, Norbert; Shinn-Cunningham, Barbara (2006). Calibration of consonant perception in room reverberation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4781472](https://doi.org/10.1121/1.4781472)

**[XREF0480]** Unknown authors (1963). BS 3638: 1963. Method for the measurement of sound absorption coefficients (ISO) in a reverberation room. *Ultrasonics*. DOI: [10.1016/0041-624x(63)90182-3](https://doi.org/10.1016/0041-624x(63)90182-3)

**[XREF0778]** Unknown authors (1973). Digital time series analysis. *Applied Acoustics*. DOI: [10.1016/0003-682x(73)90017-0](https://doi.org/10.1016/0003-682x(73)90017-0)

**[XREF0179]** Unknown authors (1977). Chapter 8 Reverberation. *Elsevier Oceanography Series*. DOI: [10.1016/s0422-9894(08)70582-8](https://doi.org/10.1016/s0422-9894(08)70582-8)

**[XREF0422]** Unknown authors (1993). Auralisation in binaural room simulation—An example of implementation. *Applied Acoustics*. DOI: [10.1016/0003-682x(93)90068-h](https://doi.org/10.1016/0003-682x(93)90068-h)

**[XREF0090]** Unknown authors (1993). Auralisation of reverberation enhancement systems. *Applied Acoustics*. DOI: [10.1016/0003-682x(93)90073-f](https://doi.org/10.1016/0003-682x(93)90073-f)

**[XREF0536]** Unknown authors (1993). Computer simulation technique regarding distribution of early reflections for room acoustical design. *Applied Acoustics*. DOI: [10.1016/0003-682x(93)90070-m](https://doi.org/10.1016/0003-682x(93)90070-m)

**[XREF0245]** Unknown authors (1993). Electroacoustic simulation of listening room acoustics. *Applied Acoustics*. DOI: [10.1016/0003-682x(93)90061-a](https://doi.org/10.1016/0003-682x(93)90061-a)

**[XREF0139]** Unknown authors (1993). Simulating reverberation enhancement systems using the CATT-acoustic program. *Applied Acoustics*. DOI: [10.1016/0003-682x(93)90072-e](https://doi.org/10.1016/0003-682x(93)90072-e)

**[XREF0360]** Unknown authors (2002). Characterisation of subjective effects. *Room Acoustics*. DOI: [10.1201/9781482286632-11](https://doi.org/10.1201/9781482286632-11)

**[XREF0394]** Unknown authors (2002). Design considerations and design procedures. *Room Acoustics*. DOI: [10.1201/9781482286632-13](https://doi.org/10.1201/9781482286632-13)

**[XREF0356]** Unknown authors (2002). Electroacoustic installations in rooms. *Room Acoustics*. DOI: [10.1201/9781482286632-14](https://doi.org/10.1201/9781482286632-14)

**[XREF0079]** Unknown authors (2002). Geometrical room acoustics. *Room Acoustics*. DOI: [10.1201/9781482286632-8](https://doi.org/10.1201/9781482286632-8)

**[XREF0266]** Unknown authors (2002). Introduction. *Room Acoustics*. DOI: [10.1201/9781482286632-4](https://doi.org/10.1201/9781482286632-4)

**[XREF0097]** Unknown authors (2002). Measuring techniques in room acoustics. *Room Acoustics*. DOI: [10.1201/9781482286632-12](https://doi.org/10.1201/9781482286632-12)

**[XREF0321]** Unknown authors (2002). Reﬂection and scattering. *Room Acoustics*. DOI: [10.1201/9781482286632-6](https://doi.org/10.1201/9781482286632-6)

**[XREF0303]** Unknown authors (2002). Room Acoustics. *Dekker Mechanical Engineering*. DOI: [10.1201/9780203910085.ch7](https://doi.org/10.1201/9780203910085.ch7)

**[XREF0562]** Unknown authors (2002). Some facts on sound waves, sources and hearing. *Room Acoustics*. DOI: [10.1201/9781482286632-5](https://doi.org/10.1201/9781482286632-5)

**[XREF0391]** Unknown authors (2002). Sound absorption and sound absorbers. *Room Acoustics*. DOI: [10.1201/9781482286632-10](https://doi.org/10.1201/9781482286632-10)

**[XREF0598]** Unknown authors (2002). The sound ﬁeld in a closed space (wave theory). *Room Acoustics*. DOI: [10.1201/9781482286632-7](https://doi.org/10.1201/9781482286632-7)

**[XREF0865]** Unknown authors (2003). Reverberation unit. *Oxford Music Online*. DOI: [10.1093/gmo/9781561592630.article.j376900](https://doi.org/10.1093/gmo/9781561592630.article.j376900)

**[XREF0393]** Unknown authors (2004). Room acoustics. *Building Services Engineering*. DOI: [10.4324/9780203563434-18](https://doi.org/10.4324/9780203563434-18)

**[XREF0191]** Unknown authors (2006). Room acoustics. *Acoustics*. DOI: [10.1201/b16958-14](https://doi.org/10.1201/b16958-14)

**[XREF0328]** Unknown authors (2007). Room acoustics. *Building Services Engineering*. DOI: [10.4324/9780203962992-22](https://doi.org/10.4324/9780203962992-22)

**[XREF0399]** Unknown authors (2008). 4 Room acoustics. *Introduction to Architectural Science*. DOI: [10.4324/9780080878942-25](https://doi.org/10.4324/9780080878942-25)

**[XREF0259]** Unknown authors (2008). Geometrical Room Acoustics in Parallelepipeds. *Formulas of Acoustics*. DOI: [10.1007/978-3-540-76833-3_258](https://doi.org/10.1007/978-3-540-76833-3_258)

**[XREF0478]** Unknown authors (2008). Other Room Acoustical Parameters. *Formulas of Acoustics*. DOI: [10.1007/978-3-540-76833-3_263](https://doi.org/10.1007/978-3-540-76833-3_263)

**[XREF0218]** Unknown authors (2008). Statistical Room Acoustics. *Formulas of Acoustics*. DOI: [10.1007/978-3-540-76833-3_259](https://doi.org/10.1007/978-3-540-76833-3_259)

**[XREF0267]** Unknown authors (2009). International Symposium on Room Acoustics. *Building Acoustics*. DOI: [10.1260/135101009788913248](https://doi.org/10.1260/135101009788913248)

**[XREF0339]** Unknown authors (2009). Reverberation Time. *Encyclopedia of Neuroscience*. DOI: [10.1007/978-3-540-29678-2_5133](https://doi.org/10.1007/978-3-540-29678-2_5133)

**[XREF0255]** Unknown authors (2009). Room acoustics. *Acoustics and Sound Insulation*. DOI: [10.11129/detail.9783034614733.12](https://doi.org/10.11129/detail.9783034614733.12)

**[XREF0282]** Unknown authors (2009). Room Acoustics. *Building Acoustics and Vibration*. DOI: [10.1142/9789812838346_0009](https://doi.org/10.1142/9789812838346_0009)

**[XREF0547]** Unknown authors (2009). Room Acoustics. *Spon's Mechanical and Electrical Services Price Book 2010*. DOI: [10.1201/9781482285505-46](https://doi.org/10.1201/9781482285505-46)

**[XREF0551]** Unknown authors (2009). Room Acoustics. *Spon's Mechanical and Electrical Services Price Book 2010*. DOI: [10.1201/9781482285505-15](https://doi.org/10.1201/9781482285505-15)

**[XREF0293]** Unknown authors (2009). Room Acoustics, Fifth Edition. *Unspecified venue*. DOI: [10.4324/9780203876374](https://doi.org/10.4324/9780203876374)

**[XREF0348]** Unknown authors (2010). International Symposium on Room Acoustics Melbourne August 29–31 2010. *Building Acoustics*. DOI: [10.1260/1351-010x.17.1.87](https://doi.org/10.1260/1351-010x.17.1.87)

**[XREF0160]** Unknown authors (2010). Reverberation. *Underwater Acoustics*. DOI: [10.1002/9780470665244.ch8](https://doi.org/10.1002/9780470665244.ch8)

**[XREF0496]** Unknown authors (2011). Room Acoustics. *Building Science*. DOI: [10.1002/9781444392333.ch10](https://doi.org/10.1002/9781444392333.ch10)

**[XREF0352]** Unknown authors (2012). Large Room Acoustics. *Audio Engineering Explained*. DOI: [10.4324/9780240812748-28](https://doi.org/10.4324/9780240812748-28)

**[XREF0355]** Unknown authors (2012). Large Room Acoustics. *Sound System Engineering*. DOI: [10.4324/9780080959603-11](https://doi.org/10.4324/9780080959603-11)

**[XREF0477]** Unknown authors (2012). Room Acoustics and Means of Control. *Recording Studio Design*. DOI: [10.4324/9780240522418-13](https://doi.org/10.4324/9780240522418-13)

**[XREF0322]** Unknown authors (2012). Room Acoustics Measures. *Audio Metering*. DOI: [10.4324/9780240814681-35](https://doi.org/10.4324/9780240814681-35)

**[XREF0354]** Unknown authors (2012). Small Room Acoustics. *Sound System Engineering*. DOI: [10.4324/9780080959603-12](https://doi.org/10.4324/9780080959603-12)

**[XREF0382]** Unknown authors (2013). Large Room Acoustics. *Sound System Engineering 4e*. DOI: [10.4324/9780240818474-16](https://doi.org/10.4324/9780240818474-16)

**[XREF0621]** Unknown authors (2013). Microphones and Room Acoustics. *Producing Great Sound for Film and Video*. DOI: [10.4324/9780080569666-6](https://doi.org/10.4324/9780080569666-6)

**[XREF0124]** Unknown authors (2013). Noises and Reverberation. *Sonar and Underwater Acoustics*. DOI: [10.1002/9781118600580.ch3](https://doi.org/10.1002/9781118600580.ch3)

**[XREF0175]** Unknown authors (2013). Room acoustics. *Acoustics and Noise Control*. DOI: [10.4324/9781315847146-13](https://doi.org/10.4324/9781315847146-13)

**[XREF0319]** Unknown authors (2013). Room acoustics. *Building Services Engineering*. DOI: [10.4324/9780203121320-24](https://doi.org/10.4324/9780203121320-24)

**[XREF0504]** Unknown authors (2013). Room acoustics and means of control. *Recording Studio Design*. DOI: [10.4324/9780080556055-12](https://doi.org/10.4324/9780080556055-12)

**[XREF0507]** Unknown authors (2013). Room acoustics and means of control. *Recording Studio Design*. DOI: [10.4324/9780080474151-11](https://doi.org/10.4324/9780080474151-11)

**[XREF0392]** Unknown authors (2013). Small Room Acoustics. *Handbook for Sound Engineers*. DOI: [10.4324/9780080927619-13](https://doi.org/10.4324/9780080927619-13)

**[XREF0395]** Unknown authors (2013). Small Room Acoustics. *Sound System Engineering 4e*. DOI: [10.4324/9780240818474-17](https://doi.org/10.4324/9780240818474-17)

**[XREF0447]** Unknown authors (2014). 6 Requirements for good room acoustics. *Environmental Science*. DOI: [10.4324/9781315838205-13](https://doi.org/10.4324/9781315838205-13)

**[XREF0211]** Unknown authors (2014). Modeling of room acoustics. *Acoustics of Small Rooms*. DOI: [10.1201/b16866-17](https://doi.org/10.1201/b16866-17)

**[XREF0561]** Unknown authors (2014). Physics of small room sound fields. *Acoustics of Small Rooms*. DOI: [10.1201/b16866-2](https://doi.org/10.1201/b16866-2)

**[XREF0495]** Unknown authors (2014). Physics of small room soundelds. *Acoustics of Small Rooms*. DOI: [10.1201/b16866-5](https://doi.org/10.1201/b16866-5)

**[XREF0438]** Unknown authors (2014). - Room Acoustics. *Acoustic Analyses Using Matlab and Ansys*. DOI: [10.1201/b17825-13](https://doi.org/10.1201/b17825-13)

**[XREF0225]** Unknown authors (2014). Room Acoustics, Fifth Edition. *Unspecified venue*. DOI: [10.1201/9781482266450](https://doi.org/10.1201/9781482266450)

**[XREF0146]** Unknown authors (2015). Room Acoustics. *Building Acoustics*. DOI: [10.1201/b18219-10](https://doi.org/10.1201/b18219-10)

**[XREF0188]** Unknown authors (2015). The Building Block of Reverberation. *Acoustics of Multi-Use Performing Arts Centers*. DOI: [10.1201/b18997-6](https://doi.org/10.1201/b18997-6)

**[XREF0229]** Unknown authors (2015). The Building Block of Reverberation. *Acoustics of Multi-Use Performing Arts Centers*. DOI: [10.1201/b18997-4](https://doi.org/10.1201/b18997-4)

**[XREF0503]** Unknown authors (2016). Chapter 10 Electroacoustical systems in rooms. *Room Acoustics*. DOI: [10.1201/9781315372150-11](https://doi.org/10.1201/9781315372150-11)

**[XREF0526]** Unknown authors (2016). Reverberation time, dreaming and the capacity to dream. *The Work of Psychoanalysis*. DOI: [10.4324/9781315658803-10](https://doi.org/10.4324/9781315658803-10)

**[XREF0493]** Unknown authors (2016). Room acoustics. *Flooring Volume 1*. DOI: [10.11129/9783955533021-007](https://doi.org/10.11129/9783955533021-007)

**[XREF0501]** Unknown authors (2017). Room Acoustics and Means of Control. *Recording Studio Design*. DOI: [10.4324/9781315675367-4](https://doi.org/10.4324/9781315675367-4)

**[XREF0530]** Unknown authors (2019). Room Acoustics. *The SAGE Encyclopedia of Human Communication Sciences and Disorders*. DOI: [10.4135/9781483380810.n530](https://doi.org/10.4135/9781483380810.n530)

**[XREF0329]** Unknown authors (2020). Reprise: Reverberation, Circumambience, and Form-Seeking Sound (Absalom, Absalom!). *The Fact of Resonance*. DOI: [10.1515/9780823288199-009](https://doi.org/10.1515/9780823288199-009)

**[XREF0435]** Unknown authors (2020). Room Acoustics. *Fundamental Physics of Sound*. DOI: [10.1142/9789811222603_0009](https://doi.org/10.1142/9789811222603_0009)

**[XREF0304]** Unknown authors (2021). Room Acoustics. *Basics Sound Insulation*. DOI: [10.1515/9783035622010-006](https://doi.org/10.1515/9783035622010-006)

**[XREF0824]** Unknown authors (2022). Book reviews: Room Acoustics: Design and Modeling; Array Signal Processing: Concepts and Techniques; Acoustics in Building Rehabilitation; and Virtual Experiments in Mechanical Vibrations: Structural Dynamics and Signal Processing. *Unspecified venue*. DOI: [10.55753/aev.v37e54.201](https://doi.org/10.55753/aev.v37e54.201)

**[XREF0734]** Unknown authors (2023). Review: Room Acoustics of Mycelium Textiles – the Myx Sail at the Danish Design Museum — R0/PR2. *Unspecified venue*. DOI: [10.1017/btd.2024.2.pr2](https://doi.org/10.1017/btd.2024.2.pr2)

**[XREF0735]** Unknown authors (2023). Review: Room Acoustics of Mycelium Textiles – the Myx Sail at the Danish Design Museum — R0/PR3. *Unspecified venue*. DOI: [10.1017/btd.2024.2.pr3](https://doi.org/10.1017/btd.2024.2.pr3)

**[XREF0257]** Unknown authors (2024). Acoustics of Enclosures. *Acoustics of Fluid Media 1*. DOI: [10.1002/9781394325580.ch10](https://doi.org/10.1002/9781394325580.ch10)

**[XREF0548]** Unknown authors (2024). Room Acoustics. *Encyclopedia of Computer Graphics and Games*. DOI: [10.1007/978-3-031-23161-2_301022](https://doi.org/10.1007/978-3-031-23161-2_301022)

**[XREF0298]** Unknown authors (2024). Room Acoustics with Emphasis on Electroacoustics 1979. *Unspecified venue*. DOI: [10.25144/23449](https://doi.org/10.25144/23449)

**[XREF0512]** Unknown authors (2025). Reviewer #1 (Public review): Listening to the room: disrupting activity of dorsolateral prefrontal cortex impairs learning of room acoustics in human listeners. *Unspecified venue*. DOI: [10.7554/elife.107041.1.sa4](https://doi.org/10.7554/elife.107041.1.sa4)

**[XREF0518]** Unknown authors (2025). Reviewer #2 (Public review): Listening to the room: disrupting activity of dorsolateral prefrontal cortex impairs learning of room acoustics in human listeners. *Unspecified venue*. DOI: [10.7554/elife.107041.1.sa3](https://doi.org/10.7554/elife.107041.1.sa3)

**[XREF0538]** Unknown authors (2025). Reviewer #3 (Public review): Listening to the room: disrupting activity of dorsolateral prefrontal cortex impairs learning of room acoustics in human listeners. *Unspecified venue*. DOI: [10.7554/elife.107041.1.sa2](https://doi.org/10.7554/elife.107041.1.sa2)

**[XREF0519]** Unknown authors (2025). Reviewer #4 (Public review): Listening to the room: disrupting activity of dorsolateral prefrontal cortex impairs learning of room acoustics in human listeners. *Unspecified venue*. DOI: [10.7554/elife.107041.1.sa1](https://doi.org/10.7554/elife.107041.1.sa1)

**[XREF0653]** VAN DEN Berg, JH (2024). Effects of Measures to Decrease the Reverberation Time in Shipbuilding Halls. *Inter.Noise 1983*. DOI: [10.25144/22859](https://doi.org/10.25144/22859)

**[XREF0387]** Van der Harten, Arthur (2011). Customized Room Acoustics Simulations Using Scripting Interfaces. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.3602460](https://doi.org/10.1121/1.3602460)

**[XREF0806]** van der Harten, Arthur; Kahn, David (2025). Recent practice in the design of rehearsal rooms. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002192](https://doi.org/10.1121/2.0002192)

**[XREF0776]** Van Hirtum, A.; Fujiso, Y. (2012). Insulation room for aero-acoustic experiments at moderate Reynolds and low Mach numbers. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2011.06.014](https://doi.org/10.1016/j.apacoust.2011.06.014)

**[XREF0249]** van Zyl, Marise; Wright, Matthew; Fujioka, Takako (2025). Virtual acoustics, real adjustments: Leveraging virtual reality to investigate how room acoustics affect music performance. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002206](https://doi.org/10.1121/2.0002206)

**[XREF0383]** Vian, Jean-Paul; Martin, Jacques (1992). Binaural room acoustics simulation: Practical uses and applications. *Applied Acoustics*. DOI: [10.1016/0003-682x(92)90050-3](https://doi.org/10.1016/0003-682x(92)90050-3)

**[XREF0281]** Vlahou, Eleni; Ueno, Kanako; Shinn-Cunningham, Barbara G.; et al. (2020). Calibration of consonant perception to room reverberation. *Unspecified venue*. DOI: [10.1101/2020.09.01.277590](https://doi.org/10.1101/2020.09.01.277590)

**[XREF0171]** Vorlander, M (2021). News from Room Acoustics. *Acoustics 2021*. DOI: [10.25144/13755](https://doi.org/10.25144/13755)

**[XREF0308]** Vorländer, M.; Mechel, F. P. (2004). Room Acoustics. *Formulas of Acoustics*. DOI: [10.1007/978-3-662-07296-7_12](https://doi.org/10.1007/978-3-662-07296-7_12)

**[XREF0513]** Vorländer, Michael (1997). Recent Progress in Room Acoustical Computer Simulations. *Building Acoustics*. DOI: [10.1177/1351010x9700400401](https://doi.org/10.1177/1351010x9700400401)

**[XREF0451]** Walchand, Kiran P. Kamble; Walchand, Manik K. Chavan (2017). Endorsement to audio recorded in different acoustic environment with feature as reverberation time with blind reverberation time estimation method. *2017 International Conference on Big Data, IoT and Data Science (BID)*. DOI: [10.1109/bid.2017.8336573](https://doi.org/10.1109/bid.2017.8336573)

**[XREF0720]** Walker, Bruce E.; Sepmeyer, Ludwig W. (1983). Acoustics of coupled spaces relative to assisted reverberation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2021180](https://doi.org/10.1121/1.2021180)

**[XREF0648]** Walther, Klaus (1961). The Upper Limits for the Reverberation Time of Reverberation Chambers for Acoustic and Electromagnetic Waves. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.1908598](https://doi.org/10.1121/1.1908598)

**[XREF0258]** Wang, Cheng; Zhu, Guangping; Mao, Yuqing; et al. (2024). A Bayesian framework-based method for suppressing reverberation in moving target detection. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2024.110141](https://doi.org/10.1016/j.apacoust.2024.110141)

**[XREF0488]** Wang, Cheng; Zhu, Guangping; Yin, Jingwei (2024). Moving Target Detection in Reverberation Background with Iterative Reweighted Least Squares Scheme. *2024 OES China Ocean Acoustics (COA)*. DOI: [10.1109/coa58979.2024.10723421](https://doi.org/10.1109/coa58979.2024.10723421)

**[XREF0118]** Wang, DeLiang (2012). Sequential Organization and Room Reverberation for Speech Segregation. *Unspecified venue*. DOI: [10.21236/ada567198](https://doi.org/10.21236/ada567198)

**[XREF0694]** Wang, Huiqing; Hornikx, Maarten (2020). Time-domain impedance boundary condition modeling with the discontinuous Galerkin method for room acoustics simulations. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0001128](https://doi.org/10.1121/10.0001128)

**[XREF0868]** Wang, Huiqing; Sihar, Indra; Pagán Muñoz, Raúl; et al. (2019). Room acoustics modelling in the time-domain with the nodal discontinuous Galerkin method. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.5096154](https://doi.org/10.1121/1.5096154)

**[XREF0744]** Wang, J.H.; Pai, C.S. (2003). Subjective and objective verifications of the inverse functions of binaural room impulse responses. *Applied Acoustics*. DOI: [10.1016/s0003-682x(03)00105-1](https://doi.org/10.1016/s0003-682x(03)00105-1)

**[XREF0237]** Wang, Maofa; Wu, Shengjie; Guo, Shengming; et al. (2021). Study on an anti-reverberation method based on PCI-SVM. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2021.108189](https://doi.org/10.1016/j.apacoust.2021.108189)

**[XREF0389]** Wang, Shuping; Zhong, Jiaxin; Qiu, Xiaojun; et al. (2020). A note on using panel diffusers to improve sound field diffusivity in reverberation rooms below 100 Hz. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2020.107471](https://doi.org/10.1016/j.apacoust.2020.107471)

**[XREF0409]** Wang, Song-Yung; Chang, Feng-Cheng; Lin, Far-Ching (2005). The amount of wooden material in a closed room and its effect on the reverberation time. *Journal of Wood Science*. DOI: [10.1007/s10086-004-0680-9](https://doi.org/10.1007/s10086-004-0680-9)

**[XREF0428]** Ward, Joel; Lokki, Tapio (2025). Church acoustics: ease and difficulty in choral singing. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002081](https://doi.org/10.1121/2.0002081)

**[XREF0632]** Warnock, A. C. C. (1979). Sound decays in a reverberation room. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2017817](https://doi.org/10.1121/1.2017817)

**[XREF0766]** Warnock, A. C. C. (1983). Influence of reverberation room volume on measured absorption coefficients. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2021069](https://doi.org/10.1121/1.2021069)

**[XREF0637]** Warnock, Alf (2000). Qualifying a reverberation room for sound absorption measurements. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4743111](https://doi.org/10.1121/1.4743111)

**[XREF0884]** Watkins, Anthony; Raimond, Andrew; Makin, Simon (2011). Room reverberation and constancy in sparse noise-vocoded speech.. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.3587736](https://doi.org/10.1121/1.3587736)

**[XREF0471]** Way, Evelyn (2022). Evaluating the reverberation chamber as a small room: How room dimensions affect the generalization of testing results. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0015476](https://doi.org/10.1121/10.0015476)

**[XREF0880]** Webb, Craig J.; Bilbao, Stefan (2011). Computing room acoustics with CUDA - 3D FDTD schemes with boundary losses and viscosity. *2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2011.5946404](https://doi.org/10.1109/icassp.2011.5946404)

**[XREF0275]** Wen, Jimi Y.C.; Habets, Emanuel A.P.; Naylor, Patrick A. (2008). Blind estimation of reverberation time based on the distribution of signal decay rates. *2008 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2008.4517613](https://doi.org/10.1109/icassp.2008.4517613)

**[XREF0617]** Wilkie, Hannah; Harrison, Peter M. C. (2024). Reverberation Time and Musical Emotion in Recorded Music Listening. *Unspecified venue*. DOI: [10.31234/osf.io/ecm6x_v1](https://doi.org/10.31234/osf.io/ecm6x_v1)

**[XREF0640]** Wilkie, Hannah; Harrison, Peter M. C. (2024). Reverberation Time and Musical Emotion in Recorded Music Listening. *Unspecified venue*. DOI: [10.31234/osf.io/ecm6x](https://doi.org/10.31234/osf.io/ecm6x)

**[XREF0110]** Williams, Kevin L. (2011). Reverberation, Sediment Acoustics, and Targets-in-the-Environment. *Unspecified venue*. DOI: [10.21236/ada568070](https://doi.org/10.21236/ada568070)

**[XREF0122]** Williams, Kevin L. (2012). Reverberation, Sediment Acoustics, and Targets-in-the-Environment. *Unspecified venue*. DOI: [10.21236/ada575135](https://doi.org/10.21236/ada575135)

**[XREF0112]** Williams, Kevin L. (2014). Reverberation, Sediment Acoustics, and Targets-in-the-Environment. *Unspecified venue*. DOI: [10.21236/ada617897](https://doi.org/10.21236/ada617897)

**[XREF0618]** Wise, R.E.; Maling, Jr., G.C.; Masuda, K. (1976). Qualification of a 230 Cubic Metre Reverberation Room. *Noise Control Engineering*. DOI: [10.3397/1.2832042](https://doi.org/10.3397/1.2832042)

**[XREF0346]** Witew, IB; Vorlander, M (2023). Measurements in Room Acoustics How Good Are We at It. *Auditorium Acoustics 2023*. DOI: [10.25144/16009](https://doi.org/10.25144/16009)

**[XREF0400]** Witew, Ingo; D’Orazio, Dario; Martellotta, Francesco (2025). Editorial: Selected papers on Room Acoustics at Forum Acusticum 2023. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2025.110971](https://doi.org/10.1016/j.apacoust.2025.110971)

**[XREF0745]** Witew, Ingo B. (2022). Measurements in room acoustics. *Unspecified venue*. DOI: [10.30819/5529](https://doi.org/10.30819/5529)

**[XREF0277]** Wittebol, Wouter; Wang, Huiqing; Hornikx, Maarten; et al. (2024). A hybrid room acoustic modeling approach combining image source, acoustic diffusion equation, and time-domain discontinuous Galerkin methods. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2024.110068](https://doi.org/10.1016/j.apacoust.2024.110068)

**[XREF0769]** Worthmann, Brian M.; Dowling, David R. (2016). Nonlinear signal processing techniques for active sonar localization in the shallow ocean with significant environmental uncertainty and reverberation. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0000309](https://doi.org/10.1121/2.0000309)

**[XREF0366]** Wu, J. R.; Gao, T. F.; Shang, E. C. (2025). Reverberation Intensity Decaying in Range-Dependent Waveguide. *Advances in Underwater Acoustics, Structural Acoustics, and Computational Methodologies*. DOI: [10.1142/9789819803408_0026](https://doi.org/10.1142/9789819803408_0026)

**[XREF0705]** Wu, Jianglin; Chen, Yunfei; Jia, Bing; et al. (2021). Analysis of Irregular Characteristics of Reverberation Spectrum in Shallow Sea. *2021 OES China Ocean Acoustics (COA)*. DOI: [10.1109/coa50123.2021.9519986](https://doi.org/10.1109/coa50123.2021.9519986)

**[XREF0320]** Wu, Lifu; Qiu, Xiaojun; Burnett, Ian; et al. (2015). Reverberation time estimation from speech signals based on blind room impulse response identification. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4927031](https://doi.org/10.1121/1.4927031)

**[XREF0284]** Wu, Lifu; Qiu, Xiaojun; Burnett, Ian; et al. (2016). Uncertainties of reverberation time estimation via adaptively identified room impulse responses. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4943547](https://doi.org/10.1121/1.4943547)

**[XREF0590]** Wu, Xinyang; Horner, Andrew (2025). Learning to rank music mashups. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002123](https://doi.org/10.1121/2.0002123)

**[XREF0756]** Wyke, Simon Swanström; Svidt, Kjeld; Christensen, Flemming; et al. (2020). Real-Time Evaluation of Room Acoustics using IFC-Based Virtual Reality and Auralization Engines. *CIB W78 Proceedings*. DOI: [10.46421/2706-6568.37.2020.paper020](https://doi.org/10.46421/2706-6568.37.2020.paper020)

**[XREF0589]** Xiang, Ning (2015). Advanced Room-Acoustics Decay Analysis. *Acoustics, Information, and Communication*. DOI: [10.1007/978-3-319-05660-9_3](https://doi.org/10.1007/978-3-319-05660-9_3)

**[XREF0309]** Xiang, Ning; Chu, Dezhang (2008). Time-reversed maximal-length sequences for outdoor, underwater sound propagation, and room-acoustic artificial reverberation simulations.. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4782726](https://doi.org/10.1121/1.4782726)

**[XREF0576]** Xiang, Wenjie; Song, Zhongchang; Gao, Zhanyuan; et al. (2025). Application of the robust autoencoder to reduce reverberation and facilitate underwater target tracking. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2024.110303](https://doi.org/10.1016/j.apacoust.2024.110303)

**[XREF0335]** Xiang, Wenjie; Song, Zhongchang; Yang, Wuyi; et al. (2023). Reverberation suppression for detecting underwater moving target based on robust autoencoder. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2023.109301](https://doi.org/10.1016/j.apacoust.2023.109301)

**[XREF0799]** Xiaotian, Zhu; Zhemin, Zhu; Jianchun, Cheng (2004). Using optimized surface modifications to improve low frequency response in a room. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2004.03.002](https://doi.org/10.1016/j.apacoust.2004.03.002)

**[XREF0456]** Xiaotu, Liu (1988). Acoustical standards and guest room isolation in hotels. *Applied Acoustics*. DOI: [10.1016/0003-682x(88)90018-7](https://doi.org/10.1016/0003-682x(88)90018-7)

**[XREF0800]** Xie, Dingding; Wittebol, Wouter; Li, Qi; et al. (2025). A method for extracting an average scattering coefficient for room acoustic modeling. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2025.110604](https://doi.org/10.1016/j.apacoust.2025.110604)

**[XREF0643]** Xie, Yuan; Zou, Tao (2024). Adaptive Time-Frequency Blind Source Separation in High Reverberation and Echo Environments. *Unspecified venue*. DOI: [10.2139/ssrn.4952969](https://doi.org/10.2139/ssrn.4952969)

**[XREF0373]** Xiong, Feifei; Dong, Minya; Zhou, Kechenying; et al. (2023). Deep Subband Network for Joint Suppression of Echo, Noise and Reverberation in Real-Time Fullband Speech Communication. *ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp49357.2023.10096067](https://doi.org/10.1109/icassp49357.2023.10096067)

**[XREF0152]** Xiong, Feifei; Goetze, Stefan; Meyer, Bernd T. (2013). Blind estimation of reverberation time based on spectro-temporal modulation filtering. *2013 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2013.6637686](https://doi.org/10.1109/icassp.2013.6637686)

**[XREF0317]** Xu, Li-ya; Liao, Bin; Zhang, Hao; et al. (2021). Acoustic localization in ocean reverberation via matrix completion with sensor failure. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2020.107681](https://doi.org/10.1016/j.apacoust.2020.107681)

**[XREF0746]** Xu, Liya; Yang, Kunde; Yang, Qiulong (2019). Geoacoustic Inversion Using Physical–Statistical Bottom Reverberation Model in the Deep Ocean. *Acoustics Australia*. DOI: [10.1007/s40857-019-00164-3](https://doi.org/10.1007/s40857-019-00164-3)

**[XREF0288]** Xu, Shihan; Peng, Jianxin; Xiao, Yi; et al. (2021). The effect of low frequency reverberation on Chinese speech intelligibility in two classrooms. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2021.108241](https://doi.org/10.1016/j.apacoust.2021.108241)

**[XREF0528]** Xuewei Zhang; Yiye Lin; Dong Wang (2015). Lasso-based reverberation suppression in automatic speech Recognition. *2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2015.7178929](https://doi.org/10.1109/icassp.2015.7178929)

**[XREF0784]** Yadav, Manuj; Cabrera, Densil; Martens, William L. (2012). A system for simulating room acoustical environments for one’s own voice. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2011.10.001](https://doi.org/10.1016/j.apacoust.2011.10.001)

**[XREF0668]** Yahya, M.N.; Otsuru, T.; Tomiku, R.; et al. (2013). A Practical System to Predict the Absorption Coefficient, Dimension and Reverberation Time of Room using GLCM, DVP and Neural Network. *International Journal of Automotive and Mechanical Engineering*. DOI: [10.15282/ijame.8.2013.15.0103](https://doi.org/10.15282/ijame.8.2013.15.0103)

**[XREF0316]** Yan, Jiajun; Zhao, Wenlai; Wu, Yue Ivan; et al. (2023). Indoor sound source localization under reverberation by extracting the features of sample covariance. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2023.109453](https://doi.org/10.1016/j.apacoust.2023.109453)

**[XREF0025]** Yanagisawa, T.; Uemura, T. (1981). Reverberation time measuring system with correlation technique. *Applied Acoustics*. DOI: [10.1016/0003-682x(81)90054-2](https://doi.org/10.1016/0003-682x(81)90054-2)

**[XREF0814]** Yang, Mengyi; Xiang, Ning (2025). Directional perception of reverberance in an acoustically coupled volume system. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/2.0002093](https://doi.org/10.1121/2.0002093)

**[XREF0163]** Yang, Tingting; Kang, Jian (2021). Sound attenuation and reverberation in sequential spaces: An experimental study. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2021.108248](https://doi.org/10.1016/j.apacoust.2021.108248)

**[XREF0332]** Yang, Wonyoung; Hodgson, Murray (2007). Optimum Reverberation for Speech Intelligibility for Normal and Hearing-Impaired Listeners in Realistic Classrooms Using Auralization. *Building Acoustics*. DOI: [10.1260/135101007781998929](https://doi.org/10.1260/135101007781998929)

**[XREF0567]** Yang, Wonyoung; Kwak, Ki-hyun; Yang, Sihoon; et al. (2021). Reverberation times preferred by traditionally trained versus classically trained musicians for overall impression of contemporary gugak orchestras using auralisation techniques. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2021.108150](https://doi.org/10.1016/j.apacoust.2021.108150)

**[XREF0707]** Yatabe, Kohei; Sugahara, Akiko (2022). Convex-optimization-based post-processing for computing room impulse response by frequency-domain FEM. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2022.108988](https://doi.org/10.1016/j.apacoust.2022.108988)

**[XREF0500]** Yeow, K.W. (1979). Room acoustical model of external reverberation. *Journal of Sound and Vibration*. DOI: [10.1016/0022-460x(79)90485-1](https://doi.org/10.1016/0022-460x(79)90485-1)

**[XREF0331]** Ying, Yingzi; MA, LI (2010). Volume Clutter Elimination, Rough Interface Reverberation Suppression, and Target Resonance Convergence in Heterogeneous Media using an Iterative Time Reversal Mirror. *Journal of Computational Acoustics*. DOI: [10.1142/s0218396x10004140](https://doi.org/10.1142/s0218396x10004140)

**[XREF0739]** Yoshida, Takumi; Okuzono, Takeshi; Sakagami, Kimihiro (2018). Numerically stable explicit time-domain finite element method for room acoustics simulation using an equivalent impedance model. *Noise Control Engineering Journal*. DOI: [10.3397/1/376615](https://doi.org/10.3397/1/376615)

**[XREF0417]** Yoshida, Takumi; Okuzono, Takeshi; Sakagami, Kimihiro (2022). A Parallel Dissipation-Free and Dispersion-Optimized Explicit Time-Domain FEM for Large-Scale Room Acoustics Simulation. *Buildings*. DOI: [10.3390/buildings12020105](https://doi.org/10.3390/buildings12020105)

**[XREF0753]** Yoshida, Takumi; Okuzono, Takeshi; Sakagami, Kimihiro (2023). Binaural Auralization of Room Acoustics with a Highly Scalable Wave-Based Acoustics Simulation. *Applied Sciences*. DOI: [10.3390/app13052832](https://doi.org/10.3390/app13052832)

**[XREF0401]** Young, J.D. (2018). Room Acoustics. *Home Studio Mastering*. DOI: [10.4324/9781315180328-4](https://doi.org/10.4324/9781315180328-4)

**[XREF0684]** Yue, Wenrong; Yang, Juan; Xu, Feng; et al. (2021). The Analysis of Reverberation Affected by Tide Changes in Shallow Water. *2021 OES China Ocean Acoustics (COA)*. DOI: [10.1109/coa50123.2021.9520030](https://doi.org/10.1109/coa50123.2021.9520030)

**[XREF0413]** Yuxuan, Zhang; Shijin, Chen; Chengpeng, Hao (2021). A Novel Adaptive Reverberation Suppression Method for Moving Active Sonar. *2021 OES China Ocean Acoustics (COA)*. DOI: [10.1109/coa50123.2021.9520084](https://doi.org/10.1109/coa50123.2021.9520084)

**[XREF0256]** Zeng, Jiazhong; Peng, Jianxin; Zhou, Xiaoming (2021). Investigation on Chinese speech reception threshold of the elderly in noise and reverberation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2021.108129](https://doi.org/10.1016/j.apacoust.2021.108129)

**[XREF0368]** Zeng, Xiang Yang (2005). An improved broad-spectrum room acoustics model including diffuse reflections. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2005.04.001](https://doi.org/10.1016/j.apacoust.2005.04.001)

**[XREF0577]** Zeng, Xiangyang; Christensen, Claus Lynge; Rindel, Jens Holger (2006). Practical methods to define scattering coefficients in a room acoustics computer model. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2005.12.001](https://doi.org/10.1016/j.apacoust.2005.12.001)

**[XREF0560]** Zhang, Jiping; Wang, Zheming; Ma, Heng; et al. (2021). Field measurement of reverberation time and average absorption of one high-rise building room by road traffic noise penetrating facade. *INTER-NOISE and NOISE-CON Congress and Conference Proceedings*. DOI: [10.3397/in-2021-2248](https://doi.org/10.3397/in-2021-2248)

**[XREF0861]** Zhao, Yan; Wang, DeLiang; Xu, Buye; et al. (2018). Late Reverberation Suppression Using Recurrent Neural Networks with Long Short-Term Memory. *2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2018.8462275](https://doi.org/10.1109/icassp.2018.8462275)

**[XREF0835]** Zheng, Guangying; Guo, Xiaowei; Zhu, Fangwei; et al. (2024). Analysis of Bottom Reverberation Intensity Under Beam-Controlled Emission Conditions in Deep Water. *Archives of Acoustics*. DOI: [10.24425/aoa.2024.148780](https://doi.org/10.24425/aoa.2024.148780)

**[XREF0058]** Zhou, Fulin; Wang, Bin; Fan, Jun (2016). Simulation study on measuring structural surface impedance in air reverberation room. *2016 IEEE/OES China Ocean Acoustics (COA)*. DOI: [10.1109/coa.2016.7535820](https://doi.org/10.1109/coa.2016.7535820)

**[XREF0149]** Zhou, Rui; Zhu, Wenye; Li, Xiaofei (2023). Speech Dereverberation with a Reverberation Time Shortening Target. *ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp49357.2023.10096164](https://doi.org/10.1109/icassp49357.2023.10096164)

**[XREF0063]** Zhou, Xiaoru; Späh, Moritz; Hengst, Klaudius; et al. (2021). Predicting the reverberation time in rectangular rooms with non-uniform absorption distribution. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2020.107539](https://doi.org/10.1016/j.apacoust.2020.107539)

**[XREF0315]** Zhu, Yunchao; Yang, Kunde; Duan, Rui; et al. (2020). Sparse spatial spectral estimation with heavy sea bottom reverberation in the fractional fourier domain. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2019.107132](https://doi.org/10.1016/j.apacoust.2019.107132)

**[XREF0381]** Zou, Jian; Shen, Yong; Yang, Jianbin; et al. (2006). A note on the prediction method of reverberation absorption coefficient of double layer micro-perforated membrane. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2005.05.004](https://doi.org/10.1016/j.apacoust.2005.05.004)

**[XREF0168]** Zych, D. A.; Earle, Thomas (1980). The use of room reverberation time measurements in physics of music courses. *American Journal of Physics*. DOI: [10.1119/1.12245](https://doi.org/10.1119/1.12245)
