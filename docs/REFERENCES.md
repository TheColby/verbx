# verbx Academic References

Focused bibliography for extreme reverberation DSP and reverberation research in general (algorithmic reverb, FDN, convolution/IR, late-field modeling, and room reverberation metrics).

Pruned from the broader bibliography on 2026-03-01 to keep only reverb-centric papers.

Total entries: 100

---

## Where to Start

If you are new to reverb DSP and want to understand the intellectual lineage of this tool, read these seven papers in this order. Everything else in the bibliography builds on, extends, or argues with what these establish.

**1. Schroeder and Logan (1961) — "Colorless" artificial reverberation** ([entry 98](#entry-98))

This is the paper that launched a thousand reverb plugins. Schroeder identified the two fundamental primitives — the comb filter and the allpass filter — and showed how to combine them to produce a plausible diffuse tail without coloring the sound. Almost every algorithmic reverb written in the six decades since has either implemented these ideas directly or consciously departed from them. Read this before anything else.

**2. Schroeder (1961) — Further Progress with Colorless Artificial Reverberation** ([entry 97](#entry-97))

The immediate follow-up, published the same year. Refines the allpass topology and introduces the parallel-comb / series-allpass architecture that became the canonical "Schroeder reverberator." The coloration problems it leaves unsolved are exactly what motivated everything in the FDN section below.

**3. Jot and Chaigne (1997) — Maximally diffusive yet efficient FDN** ([entry 95](#entry-95))

The pivot point between classic Schroeder-style structures and modern FDN design. Jot formalized the feedback delay network as a matrix + delay system, gave a clean stability condition, and showed how to design the feedback matrix for maximum diffusion. This paper is the reason every serious reverb plugin since the late 1990s has an "FDN" somewhere inside it.

**4. Rocchesso and Smith (1997) — Circulant and elliptic FDN families** ([entry 96](#entry-96))

Published the same year as Jot-Chaigne and essential reading alongside it. Rocchesso and Smith worked out the eigenvalue structure of feedback matrices, explained why some matrix designs produce audible resonances, and gave concrete construction rules for matrices that avoid them. The circulant family they describe remains a go-to for large-order FDNs.

**5. Valimaki et al. (2012) — Fifty Years of Artificial Reverberation** ([entry 59](#entry-59))

The best survey of the field in existence. Covers plate reverb, spring reverb, Schroeder, FDN, convolution, and perceptual approaches in a single coherent narrative. If you only read one survey paper, make it this one. It is also the most frequently cross-referenced paper in this bibliography.

**6. Schlecht and Habets (2015) — Time-varying feedback matrices** ([entry 39](#entry-39))

Identifies the modal ringing problem that plagues fixed-matrix FDNs (you hear it as metallic "flutter" on transients) and proposes time-varying rotation of the feedback matrix as the solution. This is the theoretical grounding for verbx's reactive-control stream. Dense math, but the payoff is real.

**7. Schlecht and Habets (2019) — Dense Reverberation with Delay Feedback Matrices** ([entry 29](#entry-29))

The most recent of the mandatory reads. Introduces the delay-feedback matrix concept, which lets you pack feedback directly into the delay topology rather than separating the two. Yields higher echo density for a given compute budget. The paper that shapes verbx's next-generation topology expansion.

---

## Key Results Reference

Quick-lookup table of the equations you will cite most often during development. These are not derivations — follow the source links for those.

| Result | Formula | Source | Notes |
|---|---|---|---|
| **Sabine equation** | RT60 = 0.161 V / (A) where A = sum(S_i alpha_i) | Sabine (1900), summarized in entries [70](#entry-70), [71](#entry-71), [94](#entry-94) | Assumes perfectly diffuse field. Breaks down in rooms with non-uniform absorption or very low average absorption coefficient. Over-predicts RT60 in dead rooms. |
| **Eyring correction** | RT60 = 0.161 V / (-S ln(1 - alpha_mean)) | Eyring (1930), see entries [70](#entry-70), [94](#entry-94) | More accurate when average absorption is high (alpha > 0.3). Reduces to Sabine in the limit of low absorption. |
| **FDN gain calibration** | g = 10^(-3 T_d / RT60) per delay line, where T_d is delay length in seconds | Jot and Chaigne (1997), entry [95](#entry-95); Schlecht and Habets (2015), entry [39](#entry-39) | Applied per-band when using frequency-dependent absorption filters on the delay outputs. This is the central calibration formula for matching a target RT60. |
| **EDT definition** | Early Decay Time = time for first 10 dB of decay on the energy decay curve, extrapolated to 60 dB | ISO 3382-1; summarized in entry [80](#entry-80) | EDT correlates better with perceived liveness than RT60 in spaces with non-exponential decay. |
| **C80 (Clarity)** | C80 = 10 log10 [ integral_0^80ms h^2(t) dt / integral_80ms^inf h^2(t) dt ] (dB) | ISO 3382-1; see entry [80](#entry-80) | Ratio of early to late energy, 80 ms threshold. Positive values indicate clear/direct sound; negative values indicate reverberant/muddy. |
| **D50 (Definition)** | D50 = integral_0^50ms h^2(t) dt / integral_0^inf h^2(t) dt | ISO 3382-1; see entry [80](#entry-80) | Fraction of total energy arriving in first 50 ms. Ranges 0-1; higher values correlate with better speech intelligibility. Uses 50 ms threshold versus C80's 80 ms. |

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

- [Schlecht and Habets (2015) - time-varying FDN matrices](#ref-fdn-tv-matrix-2015)
- [Rocchesso and Smith (1997) - circulant/elliptic FDN families](#ref-fdn-circulant-elliptic-1997)
- [Schlecht and Habets (2019) - delay feedback matrices](#ref-fdn-delay-feedback-2019)
- [Valimaki et al. (2012) - fifty years of artificial reverberation](#ref-reverb-survey-2012)

---

## Section 1: Foundational Works

The papers every reverb developer must read. These define the vocabulary, the fundamental structures, and the problems that all subsequent work attempts to solve.

<a id="entry-98"></a>
**[F1]** Schroeder, M.R.; Logan, B.F. (1961). "Colorless" artificial reverberation. *IRE Transactions on Audio*. DOI: [10.1109/tau.1961.1166351](https://doi.org/10.1109/tau.1961.1166351)

> Annotation: The founding document of algorithmic reverberation. Introduces parallel comb filters followed by series allpass sections as a way to produce high echo density without coloring the spectrum. Every subsequent algorithmic reverb either implements this architecture or explicitly departs from it. Read this first.

<a id="entry-97"></a>
**[F2]** Schroeder, M.R. (1961). Further Progress with Colorless Artificial Reverberation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.1936895](https://doi.org/10.1121/1.1936895)

> Annotation: The companion paper published the same year, refining the allpass topology. Together with the IRE paper above, this establishes the canonical Schroeder reverberator that hardware units of the 1970s and virtually every software reverb through the 1990s was built on.

<a id="entry-96"></a><a id="ref-fdn-circulant-elliptic-1997"></a>
**[F3]** Rocchesso, D.; Smith, J.O. (1997). Circulant and elliptic feedback delay networks for artificial reverberation. *IEEE Transactions on Speech and Audio Processing*. DOI: [10.1109/89.554269](https://doi.org/10.1109/89.554269)

> Annotation: Works out the eigenvalue structure of feedback matrices in detail and explains exactly why naive matrix choices cause audible modal ringing. Introduces the circulant family of matrices, which remain a practical default for large FDN orders. Indispensable alongside Jot-Chaigne.

<a id="entry-95"></a><a id="ref-jot-chaigne-1997"></a>
**[F4]** Jot, J.-M.; Chaigne, A. (1997). Maximally diffusive yet efficient feedback delay networks for artificial reverberation. *IEEE Signal Processing Letters*. DOI: [10.1109/97.623041](https://doi.org/10.1109/97.623041)

> Annotation: The paper that unified the FDN framework and gave practitioners the tools to actually design good reverbs rather than accidentally stumbling into them. Provides the stability condition, the diffuseness criterion, and the RT60 calibration formula that verbx uses directly.

<a id="entry-59"></a><a id="ref-reverb-survey-2012"></a>
**[F5]** Valimaki, V.; Parker, J.D.; Savioja, L.; et al. (2012). Fifty Years of Artificial Reverberation. *IEEE Transactions on Audio, Speech, and Language Processing*. DOI: [10.1109/tasl.2012.2189567](https://doi.org/10.1109/tasl.2012.2189567)

> Annotation: The definitive survey of the field from Schroeder's original papers through convolution and perceptual approaches. The single most useful reference document for a reverb developer. If you understand everything in this paper you understand the intellectual foundation of the entire discipline.

**[F6]** Unknown authors (1955). Artificial reverberation. *Journal of the Institution of Electrical Engineers*. DOI: [10.1049/jiee-3.1955.0135](https://doi.org/10.1049/jiee-3.1955.0135)

> Annotation: Pre-Schroeder documentation of early tape and mechanical approaches to artificial reverberation. Useful historical context for understanding what Schroeder's 1961 papers were responding to.

**[F7]** Reiss, Joshua D.; McPherson, Andrew P. (2026). Reverberation. *Audio Effects*. DOI: [10.1201/9781003593942-10](https://doi.org/10.1201/9781003593942-10)

> Annotation: A current textbook chapter that surveys the state of practical reverb algorithm design. Useful as a pedagogical introduction before diving into the primary literature above.

**[F8]** Polack, Jean-Dominique (2025). Revisiting reverberation. *Acta Acustica*. DOI: [10.1051/aacus/2025010](https://doi.org/10.1051/aacus/2025010)

> Annotation: A senior researcher's reassessment of the theoretical assumptions behind classical reverberation theory. Worth reading for the perspective it offers on what the Sabine/Eyring/FDN frameworks do and do not capture.

**[F9]** Theodore, Michael (2004). Altiverb Reverberation Software. *Computer Music Journal*. DOI: [10.1162/comj.2004.28.2.101](https://doi.org/10.1162/comj.2004.28.2.101)

> Annotation: A practitioner's account of the first commercially successful high-quality convolution reverb plugin. Important context for understanding where the convolution approach succeeded over algorithmic approaches, and what trade-offs it brought.

**[F10]** Unknown authors (2021). The Role of Modal Excitation in Colorless Reverberation. *2021 24th International Conference on Digital Audio Effects (DAFx)*. DOI: [10.23919/dafx51585.2021.9768217](https://doi.org/10.23919/dafx51585.2021.9768217)

> Annotation: Returns to Schroeder's original "colorless" criterion and examines it through the lens of modal analysis. Explains the precise mechanism by which poorly designed delay networks produce coloration, which is the problem verbx's time-varying matrix stream addresses.

---

## Section 2: Feedback Delay Networks

FDN theory, stability conditions, matrix design, and time-varying extensions. This is the core algorithmic section for verbx.

<a id="entry-39"></a><a id="ref-fdn-tv-matrix-2015"></a>
**[FDN1]** Schlecht, S.J.; Habets, E.A.P. (2015). Time-varying feedback matrices in feedback delay networks and their application in artificial reverberation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4928394](https://doi.org/10.1121/1.4928394)

> Annotation: Proves that slowly rotating the feedback matrix breaks up the fixed modal structure that causes metallic flutter in fixed-matrix FDNs, without destabilizing the network or significantly changing the statistical RT60. The theoretical basis for Stream R1.

**[FDN2]** Schlecht, S.J.; Habets, E.A.P. (2015). Simulation of Room Reverberation Using a Feedback Delay Network. *Acoustics Australia*. DOI: [10.1007/s40857-015-0005-8](https://doi.org/10.1007/s40857-015-0005-8)

> Annotation: Companion paper to [FDN1], covering implementation details of the full FDN simulation pipeline including frequency-dependent absorption and the gain calibration workflow. More accessible than the JASA paper; read this alongside it.

<a id="entry-29"></a><a id="ref-fdn-delay-feedback-2019"></a>
**[FDN3]** Schlecht, S.J.; Habets, E.A.P. (2019). Dense Reverberation with Delay Feedback Matrices. *2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)*. DOI: [10.1109/waspaa.2019.8937284](https://doi.org/10.1109/waspaa.2019.8937284)

> Annotation: Introduces delay-feedback matrices, which embed the feedback network directly into the delay topology rather than treating the delay and feedback stages as separate. Achieves higher echo density per unit of computation. This is the architectural foundation for verbx's Stream R4 topology expansion.

**[FDN4]** Unknown authors (2024). Active Acoustics with a Phase Cancelling Modal Reverberator. *Journal of the Audio Engineering Society*. DOI: [10.17743/jaes.2022.0171](https://doi.org/10.17743/jaes.2022.0171)

> Annotation: Applies modal reverberator theory to active acoustic systems. Useful for understanding how FDN structures can be used to synthesize or suppress specific room modes, which informs edge cases in verbx's reactive control design.

**[FDN5]** Koontz, Warren L. (2013). Artificial reverberation using multi-port acoustic elements. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4854696](https://doi.org/10.1121/1.4854696)

> Annotation: Proposes multi-port acoustic element networks as an alternative FDN-like architecture. Useful as a conceptual reference for exploring topologies beyond the standard scattering-matrix formulation.

---

## Section 3: Convolution and IR Processing

Partitioned FFT convolution, impulse response measurement, and IR morphing. The theoretical backbone for verbx's Stream R3.

**[CV1]** Ducasse, Éric; Rodriguez, Samuel; Bonnet, Marc (2026). Early-reverberation imaging functions for bounded elastic domains. *Acta Acustica*. DOI: [10.1051/aacus/2025069](https://doi.org/10.1051/aacus/2025069)

> Annotation: Develops imaging functions for early reflections from bounded elastic surfaces. Relevant to physically-grounded IR construction and to understanding what the early-reflection part of a measured IR actually encodes about room geometry.

**[CV2]** Unknown authors (2008). Room Impulse Responses, Decay Curves and Reverberation Times. *Formulas of Acoustics*. DOI: [10.1007/978-3-540-76833-3_262](https://doi.org/10.1007/978-3-540-76833-3_262)

> Annotation: Reference-book treatment of the mathematical relationship between the room impulse response and the energy decay curve. Essential grounding for anyone implementing EDC analysis or RT60 estimation from measured IRs.

**[CV3]** Arcas, Kevin; Chaigne, Antoine (2010). On the quality of plate reverberation. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2009.07.013](https://doi.org/10.1016/j.apacoust.2009.07.013)

> Annotation: Analyzes plate reverb physically and identifies the spectral characteristics that distinguish high-quality plate IRs from poor ones. Useful as a reference when evaluating morphed or synthesized IRs against physical ground truth.

**[CV4]** Mechel, Fridolin (2012). Reverberation with Mirror Sources. *Room Acoustical Fields*. DOI: [10.1007/978-3-642-22356-3_17](https://doi.org/10.1007/978-3-642-22356-3_17)

> Annotation: Textbook chapter on image-source methods for computing room impulse responses. The image-source approach is the classical alternative to FDN for early reflections; understanding it illuminates the trade-offs in hybrid reverb architectures.

**[CV5]** Schneiderwind, Christian; De Sena, Enzo; Neidhardt, Annika (2026). Perceptual effects of modified late reverberation and reverberation time in auditory augmented reality in two rooms. *Acta Acustica*. DOI: [10.1051/aacus/2026012](https://doi.org/10.1051/aacus/2026012)

> Annotation: Examines what happens perceptually when you modify the late tail of a measured IR independently of the early part. Directly relevant to IR morphing workflows — establishes the perceptual independence of early and late portions and the threshold above which manipulation becomes detectable.

---

## Section 4: Room Acoustics Fundamentals

RT60, energy decay curves, Sabine/Eyring theory, and the physics of reverberant fields. These papers establish the physical quantities that verbx's algorithms attempt to match.

<a id="entry-80"></a>
**[RA1]** Unknown authors (2008). Room Impulse Responses, Decay Curves and Reverberation Times. *Formulas of Acoustics*. DOI: [10.1007/978-3-540-76833-3_262](https://doi.org/10.1007/978-3-540-76833-3_262)

> Annotation: The primary reference for the EDT, C80, and D50 definitions given in the Key Results Reference table above. The formulas here are the ones used in verbx's metric computation layer.

**[RA2]** Kuttruff, Heinrich; Vorländer, Michael (2024). Reverberation and steady-state energy density. *Room Acoustics*. DOI: [10.1201/9781003389873-5](https://doi.org/10.1201/9781003389873-5)

> Annotation: Current edition textbook chapter on statistical room acoustics theory. The definitive modern treatment of Sabine and Eyring theory, with careful attention to the diffuse-field assumption and where it fails.

**[RA3]** Unknown authors (2002). Reverberation and steady state energy density. *Room Acoustics*. DOI: [10.1201/9781482286632-9](https://doi.org/10.1201/9781482286632-9)

> Annotation: Earlier edition of the Kuttruff treatment. Included because some derivations presented here are clearer than in the updated edition.

**[RA4]** Unknown authors (2010). Deriving the Reverberation Time Equation for Different Frequencies and Surfaces. *Acoustics and Psychoacoustics*. DOI: [10.1016/b978-0-240-52175-6.00015-6](https://doi.org/10.1016/b978-0-240-52175-6.00015-6)

> Annotation: Pedagogical derivation of frequency-dependent RT60, showing explicitly how absorption coefficients vary with frequency and how this produces frequency-dependent decay. The basis for verbx's per-band RT60 calibration.

**[RA5]** Unknown authors (2010). Deriving the Reverberation Time Equation. *Acoustics and Psychoacoustics*. DOI: [10.1016/b978-0-240-52175-6.00014-4](https://doi.org/10.1016/b978-0-240-52175-6.00014-4)

> Annotation: Companion derivation to [RA4], covering the wideband case. Read before [RA4].

**[RA6]** Xiang, Ning (2020). Generalization of Sabine's reverberation theory. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0001806](https://doi.org/10.1121/10.0001806)

> Annotation: Extends the Sabine framework to non-diffuse rooms using statistical energy analysis. Useful for understanding where the simple T60 prediction fails and what a more complete physical model needs to account for.

**[RA7]** Pan, Jie (2004). Invited Review Paper: The Physics of Reverberation. *Building Acoustics*. DOI: [10.1260/1351010041494746](https://doi.org/10.1260/1351010041494746)

> Annotation: A clear physics-first treatment of how reverberant fields form, covering both the modal and geometric views. Useful for developers who want to understand why the simple RT60 model is an approximation.

**[RA8]** Tohyama, Mikio (2011). Reverberation Sound in Rooms. *Signals and Communication Technology*. DOI: [10.1007/978-3-642-20122-6_11](https://doi.org/10.1007/978-3-642-20122-6_11)

> Annotation: Signal-processing perspective on room reverberation, bridging acoustic theory and DSP formulations. Useful for practitioners who are more comfortable with z-domain analysis than with acoustic physics.

**[RA9]** Fay, Michael W. (2019). Reverberation time slope ratio. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.5136901](https://doi.org/10.1121/1.5136901)

> Annotation: Proposes the slope ratio as a diagnostic for non-exponential decay curves, which arise whenever a room has coupled sub-spaces or highly non-uniform absorption. Relevant to detecting cases where a single RT60 number is insufficient.

**[RA10]** Prato, Andrea; Casassa, Federico; Schiavi, Alessandro (2016). Reverberation time measurements in non-diffuse acoustic field by the modal reverberation time. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2016.03.041](https://doi.org/10.1016/j.apacoust.2016.03.041)

> Annotation: Addresses RT60 measurement in non-diffuse fields by separating individual modal contributions. Relevant context for understanding why blind estimation algorithms struggle in small rooms.

**[RA11]** Nowoswiat, Artur; Olechowska, Marcelina (2016). Investigation Studies on the Application of Reverberation Time. *Archives of Acoustics*. DOI: [10.1515/aoa-2016-0002](https://doi.org/10.1515/aoa-2016-0002)

> Annotation: Empirical study comparing different RT60 measurement protocols and their agreement across room types. Useful calibration context.

**[RA12]** Granzotto, Nicola; Caniato, Marco (2023). Influence of not homogeneous absorption on reverberation time. *2023 Immersive and 3D Audio: from Architecture to Automotive (I3DA)*. DOI: [10.1109/i3da57090.2023.10289246](https://doi.org/10.1109/i3da57090.2023.10289246)

> Annotation: Quantifies the error in RT60 prediction when absorption is spatially non-uniform — the regime where Sabine and Eyring both fail. Relevant to understanding the boundaries of the physical models verbx targets.

**[RA13]** Jones, Douglas R. (2011). Reverberation and Time Response. *Sound of Worship*. DOI: [10.1016/b978-0-240-81339-4.00021-1](https://doi.org/10.1016/b978-0-240-81339-4.00021-1)

> Annotation: Applied treatment of reverberation in large worship spaces. Context entry for understanding design targets at extreme RT60 values that verbx is built to reproduce.

**[RA14]** Berger, Russ (2011). From Sports Arena to Sanctuary: Taming a Texas-Sized Reverberation Time. *Acoustics Today*. DOI: [10.1121/1.3576192](https://doi.org/10.1121/1.3576192)

> Annotation: Practitioner case study of a space with exceptionally long RT60. Context entry useful for anchoring intuition about what multi-second reverberation actually sounds like in a physical space.

**[RA15]** Diaz, Cesar; Pedrero, Antonio (2005). The reverberation time of furnished rooms in dwellings. *Applied Acoustics*. DOI: [10.1016/j.apacoust.2004.12.002](https://doi.org/10.1016/j.apacoust.2004.12.002)

> Annotation: Measurement study of RT60 in typical furnished domestic rooms. Useful reference for the short-RT60 end of verbx's operating range.

**[RA16]** Schnitta, Bonnie (2013). Achieving optimal reverberation time in a room, using newly patented tuning tubes. *Proceedings of Meetings on Acoustics*. DOI: [10.1121/1.4801071](https://doi.org/10.1121/1.4801071)

> Context entry. Describes physical treatment of reverberation time; tangential to algorithmic approaches.

**[RA17]** STEPHENSON, UM (2023). ON THE INFLUENCE OF CEILING AND AUDIENCE PROFILE ON THE REVERBERATION TIME AND OTHER ROOM ACOUSTICAL PARAMETERS. *Auditorium Acoustics 2008*. DOI: [10.25144/17526](https://doi.org/10.25144/17526)

> Context entry. Geometric acoustics study of audience and ceiling geometry effects on RT60. Background context for physical room modeling.

**[RA18]** DUNHAM, J (2023). COMPARING MEASURED SOUND STRENGTH TO THEORY AS A FUNCTION OF REVERBERATION TIME AND ROOM VOLUME. *Auditorium Acoustics 2023*. DOI: [10.25144/15977](https://doi.org/10.25144/15977)

> Context entry. Empirical validation of the relationship between room volume, RT60, and sound strength. Background context.

**[RA19]** WALKER, R (2024). MICROPROCESSOR CONTROLLED EQUIPMENT FOR THE MEASUREMENT OF REVERBERATION TIME. *Acoustics '82*. DOI: [10.25144/23148](https://doi.org/10.25144/23148)

> Context entry. Historical document on early automated RT60 measurement hardware. Included for completeness.

**[RA20]** CLOW, S (2023). SOUND INSULATION TESTING - SENSITIVITY TO REVERBERATION TIME: T30 V T20. *Autumn Conference Acoustics 2003*. DOI: [10.25144/18162](https://doi.org/10.25144/18162)

> Context entry. Compares T20 and T30 evaluation windows for building acoustics measurements. Peripheral to reverb synthesis but relevant to RT60 measurement methodology.

---

## Section 5: Spatial Audio and Ambisonics

**[SP1]** COREY, J (2010). Spatial Attributes and Reverberation. *Audio Production and Critical Listening*. DOI: [10.1016/b978-0-240-81295-3.00003-4](https://doi.org/10.1016/b978-0-240-81295-3.00003-4)

> Annotation: Textbook chapter on how reverberation interacts with spatial audio perception, covering apparent source width, listener envelopment, and their relationship to early vs. late energy. Grounding for verbx's spatial output design.

**[SP2]** Betlehem, T.; Poletti, M.A. (2009). Sound field reproduction around a scatterer in reverberation. *2009 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2009.4959527](https://doi.org/10.1109/icassp.2009.4959527)

> Annotation: Addresses the problem of reproducing a reverberant sound field in the presence of a scattering object. Relevant context for spatial rendering of reverb in immersive systems.

**[SP3]** Ren Gang; Shivaswamy, S. H.; Roessner, S.; et al. (2013). An additional "Depth" of reverberation helps content stand out: Media content emphasis using audio reverberation effect. *2013 IEEE International Conference on Consumer Electronics (ICCE)*. DOI: [10.1109/icce.2013.6486811](https://doi.org/10.1109/icce.2013.6486811)

> Annotation: Explores the use of selective reverberation as a production tool for creating depth and emphasis in consumer audio. Useful framing for understanding how reverb functions as a mixing tool rather than purely a simulation.

**[SP4]** Zhan Haoke; Cai Zhiming; Yuan Bingcheng (2008). Space-time ODN suppressing reverberation method. *2008 9th International Conference on Signal Processing*. DOI: [10.1109/icosp.2008.4697671](https://doi.org/10.1109/icosp.2008.4697671)

> Context entry. Dereverberation from a spatial signal processing perspective. Included for conceptual contrast with synthesis approaches.

---

## Section 6: Perceptual Studies

How humans perceive reverberation — thresholds of detection, preference curves, and the perceptual dimensions of reverberant spaces. These papers inform which parameters verbx needs to control precisely and which it can treat as second-order.

**[PE1]** Zahorik, Pavel (2004). Perceptual scaling of room reverberation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4784535](https://doi.org/10.1121/1.4784535)

> Annotation: Maps the perceptual space of reverberation using multidimensional scaling, identifying the primary perceptual dimensions (roughly: decay time and reverberance character). Fundamental to understanding which physical parameters predict subjective quality.

**[PE2]** Javed, Hamza A.; Naylor, Patrick A. (2015). An extended reverberation decay tail metric as a measure of perceived late reverberation. *2015 23rd European Signal Processing Conference (EUSIPCO)*. DOI: [10.1109/eusipco.2015.7362546](https://doi.org/10.1109/eusipco.2015.7362546)

> Annotation: Proposes a metric that correlates more closely with the perceived extent of late reverberation than RT60 alone. Useful for evaluating whether a verbx preset sounds as long as its nominal RT60 suggests it should.

**[PE3]** Tsilfidis, Alexandros; Mourjopoulos, John (2011). Blind single-channel suppression of late reverberation based on perceptual reverberation modeling. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.3533690](https://doi.org/10.1121/1.3533690)

> Annotation: Develops a perceptual model of late reverberation and uses it to drive a suppression algorithm. The perceptual model itself — how the ear weights late energy — is the relevant contribution for verbx's design.

**[PE4]** Tsilfidis, Alexandros; Mourjopoulos, John (2009). Perceptually-motivated selective suppression of late reverberation. *2009 16th International Conference on Digital Signal Processing*. DOI: [10.1109/icdsp.2009.5201165](https://doi.org/10.1109/icdsp.2009.5201165)

> Annotation: Earlier version of the work above; contains some derivations that the 2011 paper abbreviates.

**[PE5]** Watkins, Anthony J. (2005). Listening in real-room reverberation: Effects of extrinsic context. *Auditory Signal Processing*. DOI: [10.1007/0-387-27045-0_52](https://doi.org/10.1007/0-387-27045-0_52)

> Annotation: Psychoacoustic study on how listeners adapt to reverberation over time and how prior context affects the perception of a reverberant signal. Explains why very long reverb tails can become perceptually transparent under sustained exposure.

**[PE6]** Blasinski, Lukasz; Felcyn, Jan; Kocinski, Jedrzej (2024). Perception of reverberation length in rooms with reverberation enhancement systems. *The Journal of the Acoustical Society of America*. DOI: [10.1121/10.0026485](https://doi.org/10.1121/10.0026485)

> Annotation: Studies the just-noticeable difference for RT60 changes in rooms with active enhancement systems. Directly relevant to calibrating the resolution of verbx's RT60 control.

**[PE7]** NINOUNAKIS, TJ; DAVIES, WJ (2024). THE PERCEPTION OF SMALL CHANGES IN REVERBERATION TIME WITHIN RECORDING STUDIO CONTROL ROOMS. *Reproduced Sound 1998*. DOI: [10.25144/18947](https://doi.org/10.25144/18947)

> Annotation: Measures perceptual thresholds for RT60 change in a professional monitoring environment. Complements [PE6] with studio-context data.

**[PE8]** STROTHER, TEN; DAVIES, WJ (2023). IS THERE A RELATIONSHIP BETWEEN PITCH PERCEPTION AND INDIVIDUAL PREFERENCE OF CONCERT HALL REVERBERATION TIME?. *ACOUSTICS 2023*. DOI: [10.25144/16625](https://doi.org/10.25144/16625)

> Context entry. Investigates individual differences in preferred RT60. Peripheral to algorithm design; relevant to preset design philosophy.

**[PE9]** Erkelens, Jan S.; Heusdens, Richard (2011). A statistical room impulse response model with frequency dependent reverberation time for single-microphone late reverberation suppression. *Interspeech 2011*. DOI: [10.21437/interspeech.2011-82](https://doi.org/10.21437/interspeech.2011-82)

> Annotation: Derives a statistical model of the RIR late field that explicitly captures frequency-dependent decay. The model's assumptions are directly relevant to how verbx represents the decay filter bank.

---

## Section 7: Machine Learning and Blind Estimation

CRNN-based RT60 estimation, neural dereverberation, and related machine learning approaches. Relevant to verbx's analysis pipeline.

**[ML1]** Deng, Shuwen; Mack, Wolfgang; Habets, Emanuel A.P. (2020). Online Blind Reverberation Time Estimation Using CRNNs. *Interspeech 2020*. DOI: [10.21437/interspeech.2020-2156](https://doi.org/10.21437/interspeech.2020-2156)

> Annotation: Demonstrates that a convolutional-recurrent neural network can estimate RT60 reliably from a single speech channel in real time. Sets the state of the art for the blind estimation problem that verbx's analysis mode needs to solve.

**[ML2]** Mack, Wolfgang; Deng, Shuwen; Habets, Emanuel A.P. (2020). Single-Channel Blind Direct-to-Reverberation Ratio Estimation Using Masking. *Interspeech 2020*. DOI: [10.21437/interspeech.2020-2171](https://doi.org/10.21437/interspeech.2020-2171)

> Annotation: Companion paper to [ML1], estimating the direct-to-reverberant ratio rather than RT60. DRR is the perceptually critical parameter for how "wet" a signal sounds, distinct from how long the tail is.

**[ML3]** Xiong, Feifei; Goetze, Stefan; Kollmeier, Birger; et al. (2019). Joint Estimation of Reverberation Time and Early-To-Late Reverberation Ratio From Single-Channel Speech Signals. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*. DOI: [10.1109/taslp.2018.2877894](https://doi.org/10.1109/taslp.2018.2877894)

> Annotation: Estimates RT60 and early-to-late ratio simultaneously from a single microphone. The joint approach avoids the error propagation that plagues sequential estimators and is worth evaluating for verbx's blind analysis mode.

**[ML4]** Lollmann, Heinrich W.; Brendel, Andreas; Kellermann, Walter (2018). Efficient ML-Estimator for Blind Reverberation Time Estimation. *2018 26th European Signal Processing Conference (EUSIPCO)*. DOI: [10.23919/eusipco.2018.8553001](https://doi.org/10.23919/eusipco.2018.8553001)

> Annotation: Maximum-likelihood formulation for blind RT60 estimation with a focus on computational efficiency. A strong non-neural baseline to compare CRNN approaches against.

**[ML5]** Choi, Yeonjong; Xie, Chao; Toda, Tomoki (2023). Reverberation-Controllable Voice Conversion Using Reverberation Time Estimator. *INTERSPEECH 2023*. DOI: [10.21437/interspeech.2023-1356](https://doi.org/10.21437/interspeech.2023-1356)

> Annotation: Uses an RT60 estimator as a conditioning signal inside a voice conversion network to control the reverberation of synthesized speech. Demonstrates an end-to-end neural approach to RT60-conditioned synthesis that is architecturally adjacent to verbx's generative modes.

**[ML6]** Ai, Yang; Wang, Xin; Yamagishi, Junichi; et al. (2020). Reverberation Modeling for Source-Filter-Based Neural Vocoder. *Interspeech 2020*. DOI: [10.21437/interspeech.2020-1613](https://doi.org/10.21437/interspeech.2020-1613)

> Annotation: Embeds a reverberation model inside a neural vocoder. Demonstrates how differentiable reverb models can be combined with neural speech synthesis. Relevant to any future verbx neural extensions.

**[ML7]** Kodrasi, Ina; Bourlard, Herve (2018). Single-channel Late Reverberation Power Spectral Density Estimation Using Denoising Autoencoders. *Interspeech 2018*. DOI: [10.21437/interspeech.2018-1660](https://doi.org/10.21437/interspeech.2018-1660)

> Annotation: Uses a denoising autoencoder to estimate the power spectral density of the late reverberant field. The estimated late-field PSD is then usable for suppression or analysis. Methodologically relevant to verbx's analysis pipeline.

**[ML8]** Andrijasevic, Andrea (2020). Effect of phoneme variations on blind reverberation time estimation. *Acta Acustica*. DOI: [10.1051/aacus/2020001](https://doi.org/10.1051/aacus/2020001)

> Annotation: Examines how the phonetic content of speech affects the accuracy of blind RT60 estimators. Useful for understanding failure modes and the range of conditions over which blind estimators can be trusted.

**[ML9]** Prodeus, Arkadiy (2017). Late reverberation reduction and blind reverberation time measurement for automatic speech recognition. *2017 IEEE First Ukraine Conference on Electrical and Computer Engineering (UKRCON)*. DOI: [10.1109/ukrcon.2017.8100319](https://doi.org/10.1109/ukrcon.2017.8100319)

> Context entry. Combines blind RT60 estimation with late-reverb suppression for ASR. Context for understanding the estimation-then-suppression pipeline architecture.

**[ML10]** Prodeus, Arkadiy (2015). Parameter optimization of the single channel late reverberation suppression technique. *2015 IEEE 35th International Conference on Electronics and Nanotechnology (ELNANO)*. DOI: [10.1109/elnano.2015.7146889](https://doi.org/10.1109/elnano.2015.7146889)

> Context entry. Optimization of suppression filter parameters in a statistical dereverberation framework. Earlier work by the same author as [ML9].

**[ML11]** Ma, Changxue; Shi, Guangji (2012). Reverberation time estimation based on multidelay acoustic echo cancellation. *2012 International Conference on Audio, Language and Image Processing*. DOI: [10.1109/icalip.2012.6376617](https://doi.org/10.1109/icalip.2012.6376617)

> Context entry. Exploits an existing echo cancellation framework to derive RT60 estimates. Alternative estimation strategy for completeness.

**[ML12]** Malik, Hafiz; Zhao, Hong (2012). Recording environment identification using acoustic reverberation. *2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. DOI: [10.1109/icassp.2012.6288258](https://doi.org/10.1109/icassp.2012.6288258)

> Context entry. Uses reverberation signatures to fingerprint recording environments. Tangential but provides perspective on what makes each room's impulse response distinctive.

**[ML13]** Malik, Hafiz; Farid, Hany (2010). Audio forensics from acoustic reverberation. *2010 IEEE International Conference on Acoustics, Speech and Signal Processing*. DOI: [10.1109/icassp.2010.5495479](https://doi.org/10.1109/icassp.2010.5495479)

> Context entry. Forensic identification of recording environments from reverberation. Companion study to [ML12].

**[ML14]** Zheng, Chenxi; Chan, Wai-Yip (2013). Late reverberation suppression using MMSE modulation spectral estimation. *Interspeech 2013*. DOI: [10.21437/interspeech.2013-718](https://doi.org/10.21437/interspeech.2013-718)

> Context entry. MMSE estimator operating in the modulation spectrum domain for late reverb suppression. Useful for understanding the modulation-domain perspective on reverberation.

---

## Section 8: Speech and Room Acoustics

Papers on speech intelligibility in reverberant environments, dereverberation for ASR, and vocal effects in reverberant spaces. These are more peripherally related to verbx's core synthesis and analysis functions. Included for completeness and to support any future intelligibility-aware modes.

**[SR1]** Habets, Emanuel A. P. (2010). Speech Dereverberation Using Statistical Reverberation Models. *Signals and Communication Technology*. DOI: [10.1007/978-1-84996-056-4_3](https://doi.org/10.1007/978-1-84996-056-4_3)

> Annotation: Comprehensive chapter on statistical approaches to single-channel dereverberation. The statistical models of the late reverberant field described here inform verbx's internal representations of the reverberant tail.

**[SR2]** Saijo, Kohei; Wichern, Gordon; Germain, Francois G.; et al. (2024). Enhanced Reverberation as Supervision for Unsupervised Speech Separation. *Interspeech 2024*. DOI: [10.21437/interspeech.2024-1241](https://doi.org/10.21437/interspeech.2024-1241)

> Context entry. Uses added reverberation as a data augmentation strategy for training speech separation models. Relevant for understanding how verbx might be used as a data synthesis tool.

**[SR3]** Sehr, Armin; Kellermann, Walter (2008). New Results for Feature-Domain Reverberation Modeling. *2008 Hands-Free Speech Communication and Microphone Arrays*. DOI: [10.1109/hscma.2008.4538713](https://doi.org/10.1109/hscma.2008.4538713)

> Context entry. Feature-domain reverberation model for robust ASR. Peripheral to synthesis but relevant to understanding how reverb distorts feature representations.

**[SR4]** Nishiura, Takanobu; Fukumori, Takahiro (2011). Suitable Reverberation Criteria for Distant-talking Speech Recognition. *Speech Technologies*. DOI: [10.5772/18937](https://doi.org/10.5772/18937)

> Context entry. Identifies which room acoustics metrics best predict ASR accuracy. Establishes C80 and DRR as more predictive than RT60 alone for intelligibility in ASR contexts.

**[SR5]** Leclere, Thibaud; Lavandier, Mathieu; Culling, John F. (2015). Speech intelligibility prediction in reverberation: Towards an integrated model of speech transmission, spatial unmasking, and binaural de-reverberation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4921028](https://doi.org/10.1121/1.4921028)

> Context entry. Integrated model of speech intelligibility in reverberation incorporating binaural effects. Background context for understanding the perceptual impact of long RT60 on speech.

**[SR6]** Kocinski, Jedrzej; Ozimek, Edward (2016). Speech Recognition in an Enclosure with a Long Reverberation Time. *Archives of Acoustics*. DOI: [10.1515/aoa-2016-0025](https://doi.org/10.1515/aoa-2016-0025)

> Context entry. Empirical study of intelligibility at long RT60 values. Relevant to understanding perceptual consequences of the extreme reverberation times verbx targets.

**[SR7]** Rennies, Jan; Brand, Thomas; Kollmeier, Birger (2011). Prediction of binaural intelligibility level differences in reverberation. *Interspeech 2011*. DOI: [10.21437/interspeech.2011-10](https://doi.org/10.21437/interspeech.2011-10)

> Context entry. Binaural intelligibility model predictions in reverberant conditions. Background reference for spatial dereverberation contexts.

**[SR8]** Arai, Takayuki (2006). Preprocessing speech against reverberation. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4781210](https://doi.org/10.1121/1.4781210)

> Context entry. Signal processing approaches to pre-filtering speech before transmission through reverberant environments. Inverse problem perspective on the synthesis work.

**[SR9]** Tsilfidis, Alexandros; Mourjopoulos, John (2011). Blind single-channel suppression of late reverberation based on perceptual reverberation modeling. *The Journal of the Acoustical Society of America*. (Also listed as [PE3])

**[SR10]** Hodoshima, Nao (2024). Effects of talker and playback rate of reverberation-induced speech on speech intelligibility of older adults. *Interspeech 2024*. DOI: [10.21437/interspeech.2024-1721](https://doi.org/10.21437/interspeech.2024-1721)

> Context entry. Speech intelligibility study focusing on older listeners in reverberant conditions. Peripheral to DSP design; population-specific psychoacoustic data.

**[SR11]** Hodoshima, Nao (2019). Effects of Urgent Speech and Congruent/Incongruent Text on Speech Intelligibility in Noise and Reverberation. *Interspeech 2019*. DOI: [10.21437/interspeech.2019-1902](https://doi.org/10.21437/interspeech.2019-1902)

> Context entry. Cognitive factors in speech intelligibility under reverberation. Peripheral to DSP design.

**[SR12]** Hodoshima, Nao; Arai, Takayuki; Kurisu, Kiyohiro (2012). Intelligibility of speech spoken in noise/reverberation for older adults in reverberant environments. *Interspeech 2012*. DOI: [10.21437/interspeech.2012-415](https://doi.org/10.21437/interspeech.2012-415)

> Context entry. Earlier Hodoshima et al. study on older listener intelligibility. Part of a research series.

**[SR13]** BARNETT, P (2024). STUDY OF WORD SCORE TEST RESULTS TO DETERMINE THE ROBUST COMPONENTS OF SPEECH SUBJECT TO NOISE AND REVERBERATION. *Reproduced Sound 1998*. DOI: [10.25144/18937](https://doi.org/10.25144/18937)

> Context entry. Word intelligibility scoring under combined noise and reverberation. Included for completeness.

**[SR14]** Bottalico, Pasquale; Graetzer, Simone; Hunter, Eric (2016). Effect of reverberation time on vocal fatigue. *Speech Prosody 2016*. DOI: [10.21437/speechprosody.2016-101](https://doi.org/10.21437/speechprosody.2016-101)

> Context entry. Studies how room RT60 affects talker vocal effort and fatigue. Peripheral to reverb synthesis; human factors context.

**[SR15]** Petkov, Petko N.; Stylianou, Yannis (2016). Generalizing Steady State Suppression for Enhanced Intelligibility Under Reverberation. *Interspeech 2016*. DOI: [10.21437/interspeech.2016-1026](https://doi.org/10.21437/interspeech.2016-1026)

> Context entry. Signal processing method for intelligibility enhancement in reverberant conditions. Included for completeness.

**[SR16]** Petkov, Petko N.; Braunschweiler, Norbert; Stylianou, Yannis (2016). Automated Pause Insertion for Improved Intelligibility Under Reverberation. *Interspeech 2016*. DOI: [10.21437/interspeech.2016-960](https://doi.org/10.21437/interspeech.2016-960)

> Context entry. Modifies speech rhythm to improve intelligibility in reverberant conditions. Peripheral to synthesis.

**[SR17]** Peddinti, Vijayaditya; Chen, Guoguo; Povey, Daniel; et al. (2015). Reverberation robust acoustic modeling using i-vectors with time delay neural networks. *Interspeech 2015*. DOI: [10.21437/interspeech.2015-527](https://doi.org/10.21437/interspeech.2015-527)

> Context entry. ASR robustness to reverberation using TDNN-based acoustic models. Neural ASR context; tangential to synthesis.

**[SR18]** Raut, Chandra Kant; Nishimoto, Takuya; Sagayama, Shigeki (2005). Model adaptation by state splitting of HMM for long reverberation. *Interspeech 2005*. DOI: [10.21437/interspeech.2005-157](https://doi.org/10.21437/interspeech.2005-157)

> Context entry. HMM adaptation for reverberant ASR. Historical ASR context.

**[SR19]** Choi, Jae; Kim, Jeunghun; Kang, Shin Jae; et al. (2015). Reverberation-robust acoustic indoor localization. *Interspeech 2015*. DOI: [10.21437/interspeech.2015-678](https://doi.org/10.21437/interspeech.2015-678)

> Context entry. Room localization from reverberant acoustic signals. Tangential to synthesis; relevant to analysis.

**[SR20]** Pang, Cheng; Zhang, Jie; Liu, Hong (2015). Direction of arrival estimation based on reverberation weighting and noise error estimator. *Interspeech 2015*. DOI: [10.21437/interspeech.2015-681](https://doi.org/10.21437/interspeech.2015-681)

> Context entry. DOA estimation using reverberation structure. Spatial analysis context; peripheral to verbx.

**[SR21]** Saric, Zoran; Jovicic, Slobodan (2003). Adaptive beamforming in room with reverberation. *8th European Conference on Speech Communication and Technology (Eurospeech 2003)*. DOI: [10.21437/eurospeech.2003-218](https://doi.org/10.21437/eurospeech.2003-218)

> Context entry. Beamforming in reverberant rooms. Spatial processing context.

**[SR22]** Li, Xuan; Ma, Xiaochuan; Hou, Chaohuan (2008). Broadband DOA Estimation by Beamforming with Suppression of Reverberation. *2008 Congress on Image and Signal Processing*. DOI: [10.1109/cisp.2008.550](https://doi.org/10.1109/cisp.2008.550)

> Context entry. Broadband DOA estimation compensating for reverberation. Spatial processing context.

**[SR23]** Peng, Jianxin (2015). Chinese Syllable and Phoneme Identification in Noise and Reverberation. *Archives of Acoustics*. DOI: [10.2478/aoa-2014-0052](https://doi.org/10.2478/aoa-2014-0052)

> Context entry. Language-specific phoneme intelligibility under reverberation. Peripheral to DSP design.

**[SR24]** Unknown authors (2008). REVERBERATION ASSESSMENT IN AUDIOBAND SPEECH SIGNALS FOR TELEPRESENCE SYSTEMS. *Proceedings of the International Conference on Signal Processing and Multimedia Applications*. DOI: [10.5220/0001936602570262](https://doi.org/10.5220/0001936602570262)

> Context entry. Reverberation metrics for telepresence audio quality assessment. Peripheral context.

**[SR25]** Unknown authors (1995). FICHERA, I (2020). THE ACCURACY OF REVERBERATION TIME PREDICTIONS FOR GENERAL TEACHING SPACES. *ACOUSTICS 2020*. DOI: [10.25144/13333](https://doi.org/10.25144/13333)

> Context entry. Validation of RT60 prediction formulas in classroom environments. Context for applied room acoustics.

---

## Section 9: Underwater and Non-Room Reverberation

Papers whose "reverberation" subject matter refers to underwater acoustics, seabed scattering, or other domains unrelated to architectural room acoustics. Included because they were in the source bibliography; they have no direct bearing on verbx's design. Retained for bibliographic completeness.

**[UW1]** Preston, John R.; Abraham, Douglas A.; Yang, Jie (2014). Non-stationary reverberation observations from the shallow water TREX13 reverberation experiments using the FORA triplet array. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4900079](https://doi.org/10.1121/1.4900079)

> Context entry. Shallow-water acoustic reverberation measurements. Domain is underwater acoustics; retained for completeness only.

**[UW2]** Chotiros, Nicholas (2012). Non-Rayleigh reverberation statistics. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4708991](https://doi.org/10.1121/1.4708991)

> Context entry. Statistical characterization of underwater backscatter. Retained for completeness.

**[UW3]** Katsnelson, Boris; Petnikov, Valery; Lynch, James (2011). Low-Frequency Bottom Reverberation in Shallow Water. *Fundamentals of Shallow Water Acoustics*. DOI: [10.1007/978-1-4419-9777-7_6](https://doi.org/10.1007/978-1-4419-9777-7_6)

> Context entry. Shallow water bottom reverberation. Domain is underwater acoustics; retained for completeness only.

**[UW4]** Oh (2015). Low-Frequency Normal Mode Reverberation Model. *The Journal of the Acoustical Society of Korea*. DOI: [10.7776/ask.2015.34.3.184](https://doi.org/10.7776/ask.2015.34.3.184)

> Context entry. Normal mode model for low-frequency underwater reverberation. Domain is underwater acoustics; retained for completeness only.

**[UW5]** Harrison, Chris (2009). Reverberation versus time or reverberation versus range? A definitive relationship. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.3248705](https://doi.org/10.1121/1.3248705)

> Context entry. Underwater reverberation decay modeling. Retained for completeness.

**[UW6]** LePage, Kevin D. (2009). High-frequency broadband coherent reverberation predictions for the reverberation modeling workshop. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.3248711](https://doi.org/10.1121/1.3248711)

> Context entry. Underwater high-frequency reverberation modeling workshop proceedings. Retained for completeness.

**[UW7]** Thorsos, Eric I.; Perkins, John S. (2007). Reverberation modeling issues highlighted by the first Reverberation Modeling Workshop. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.2943047](https://doi.org/10.1121/1.2943047)

> Context entry. Summary of the first underwater acoustic reverberation modeling workshop. Retained for completeness.

**[UW8]** Tang, Dajun (2013). Issues in reverberation modeling. *The Journal of the Acoustical Society of America*. DOI: [10.1121/1.4831447](https://doi.org/10.1121/1.4831447)

> Context entry. Underwater reverberation modeling issues review. Retained for completeness.

**[UW9]** Guo, Yongqiang; Che, Weiqiu (2010). Reverberation-Ray Matrix Analysis of Acoustic Waves in Multilayered Anisotropic Structures. *Acoustic Waves*. DOI: [10.5772/9788](https://doi.org/10.5772/9788)

> Context entry. Reverberation-ray matrix method for layered elastic media. Structural acoustics domain; retained for completeness.

**[UW10]** Paul, Joanna (2025). Amplification, Reverberation, and Distortion. *Music as Classical Reception*. DOI: [10.1093/9780198958024.003.0012](https://doi.org/10.1093/9780198958024.003.0012)

> Context entry. Musicological analysis of reverberation as aesthetic phenomenon. Not DSP; retained for completeness.
