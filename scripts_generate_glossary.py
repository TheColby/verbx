#!/usr/bin/env python3
"""Generate the book's alphabetized technical and musical glossary."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "docs" / "GLOSSARY.md"
MINIMUM_ENTRY_COUNT = 700

ENTRY_DATA = r"""
A-weighting :: A frequency weighting that approximates human sensitivity at moderate levels and is commonly used for environmental-noise measurements.
AB microphone technique :: A spaced-pair recording method in which arrival-time and level differences create stereo width and room impression.
Absolute threshold of hearing :: The lowest sound pressure level detectable under specified conditions, varying strongly with frequency and listener.
Absorber :: A material or structure that converts incident acoustic energy into heat rather than returning it as reflection.
Absorption :: The loss of sound energy when a boundary, object, medium, or treatment prevents a reflection of equal energy.
Absorption coefficient :: The frequency-dependent fraction of incident acoustic energy absorbed by a surface, usually expressed from zero to one.
Acoustic camera :: A microphone-array system that estimates and visualizes the direction or location of sound radiation.
Acoustic clarity :: The perceptual separation of successive sounds, influenced by direct sound, early energy, late decay, and source articulation.
Acoustic impedance :: The ratio of acoustic pressure to volume velocity or particle velocity at a specified point and frequency.
Acoustic labyrinth :: A folded passage that lengthens an acoustic path for delay, damping, loudspeaker loading, or artificial reverberation.
Acoustic mode :: A resonant pressure pattern determined by enclosure geometry, boundary conditions, and the speed of sound.
Acoustic power :: The rate at which a source emits sound energy, measured in watts and independent of measurement distance in free field.
Acoustic shadow :: A region receiving reduced direct sound because an obstacle blocks or diffracts propagation.
Acoustic treatment :: Deliberate use of absorbers, diffusers, resonators, and construction to shape a room's response.
Acoustics :: The science of sound generation, propagation, interaction, perception, measurement, and control.
Active acoustics :: Electroacoustic systems that use microphones, processing, and loudspeakers to alter a venue's apparent reverberation or spatial response.
Active noise control :: Reduction of unwanted sound through a controlled secondary field designed to interfere destructively with the disturbance.
Adaptive filter :: A filter whose coefficients update from an error criterion, often used for echo cancellation, system identification, or dereverberation.
Additive synthesis :: Construction of a signal by summing sinusoidal or otherwise elementary components with controlled amplitudes and phases.
Air absorption :: Frequency-dependent attenuation during propagation through air, strongest at high frequencies and dependent on humidity and temperature.
Algorithmic latency :: Delay caused by an algorithm's required lookahead, block size, transform, buffering, or internal state rather than hardware transport.
Algorithmic reverb :: Artificial reverberation produced by delay networks, filters, modulation, and feedback rather than direct convolution with a measured IR.
Aliasing :: Misrepresentation of frequencies above the Nyquist limit as lower frequencies after sampling or nonlinear processing.
Allpass diffuser :: An allpass filter used to increase echo density and phase dispersion while approximately preserving magnitude response.
Allpass filter :: A filter with nominally flat magnitude response and frequency-dependent phase response.
Ambience :: The audible environmental field surrounding a source, including room tone, reflections, diffuse energy, and distant activity.
Ambience extraction :: Estimation of diffuse or spatially incoherent content from a recording for remixing, upmixing, or analysis.
Ambisonics :: A scene-based spatial-audio representation using spherical-harmonic components rather than fixed loudspeaker feeds.
Amplitude :: The instantaneous or measured magnitude of a signal, wave, coefficient, or oscillation.
Amplitude envelope :: The time-varying outline of signal magnitude, commonly described by attack, decay, sustain, and release.
Amplitude modulation :: Multiplication of a signal by a varying gain, producing level motion and potentially upper and lower sidebands.
Analog-to-digital converter :: Hardware that samples an analog voltage and represents it as digital numbers at a selected rate and resolution.
Analyzer :: A tool that estimates signal, room, spectral, loudness, spatial, or decay metrics from audio.
Angular frequency :: Frequency measured in radians per second, written $\omega = 2\pi f$.
Anisotropy :: Directional dependence in a material, field, room, radiation pattern, or spatial decay.
Anti-aliasing filter :: A low-pass filter applied before sampling or rate reduction to suppress content above the new Nyquist frequency.
Antiphony :: Alternation or exchange between separated performers or groups, often using architecture as a compositional spatial parameter.
Apparent source width :: The perceived lateral extent of a sound source, strongly influenced by early lateral reflections and interaural decorrelation.
Archaeoacoustics :: Interdisciplinary study of sound, listening, architecture, and human activity at archaeological sites using acoustic measurement, modeling, and cultural evidence.
Artifact :: An unintended audible or measurable product of processing, encoding, estimation, clipping, modulation, or numerical error.
Attack :: The onset portion of a sound or the time constant with which a processor responds to increasing level.
Attenuation :: Reduction in amplitude, power, or level caused by distance, absorption, filtering, gain control, or cancellation.
Audio buffer :: A block of samples held temporarily for device transfer or processing, contributing directly to realtime latency.
Audio callback :: A host- or driver-invoked routine that must process a block before the playback deadline without blocking.
Audio unit :: Apple's plug-in architecture for audio effects, instruments, generators, and music-processing components.
Auditory masking :: Reduced audibility of one sound because another sound overlaps it in time, frequency, or both.
Auditory scene analysis :: Perceptual organization of mixed sound into sources, events, streams, and environments.
Automatic gain control :: A system that varies gain to maintain a target level or operating range over time.
Automation :: Recorded or generated time variation of parameters, routing, gain, states, or events.
Autocorrelation :: Similarity of a signal to delayed copies of itself, useful for detecting periodicity, repetition, and modal structure.
Aux send :: A mixer path that routes a controllable copy of a channel to a shared processor such as reverb.
Azimuth :: Horizontal angle around a listener or coordinate origin, usually measured in degrees.
B-format :: A conventional name for an Ambisonic component signal set before loudspeaker or binaural decoding.
Back wall :: The room boundary behind the listener, often responsible for strong delayed reflections and low-frequency modal behavior.
Backward prediction :: Estimation of a sample from future samples, used in some linear-prediction and dereverberation formulations.
Band-pass filter :: A filter that passes a selected frequency region while attenuating frequencies below and above it.
Bandlimited signal :: A signal whose spectrum is zero or negligible above a specified maximum frequency.
Bandwidth :: The frequency span occupied or passed by a signal, filter, channel, mode, or process.
Batch processing :: Noninteractive execution of the same or parameterized operation across many files or manifest entries.
Beamforming :: Combining microphone or loudspeaker array elements with delays and weights to emphasize selected directions.
Beat frequency :: The difference frequency produced when two nearby sinusoids interfere.
Bell filter :: A parametric equalizer band that boosts or cuts around a center frequency with controlled width.
Binaural audio :: Two-channel audio designed to reproduce ear-input cues for headphone listening.
Binaural room impulse response :: An impulse response measured or modeled from a source to the listener's two ears in a room.
Bit depth :: The number or format of bits used to represent sample amplitude, affecting quantization and dynamic range.
Block convolution :: Convolution performed on signal blocks, commonly accelerated with FFT methods.
Block size :: The number of samples processed or transferred at once, trading scheduling margin and efficiency against latency.
Boundary :: A surface or interface at which an acoustic wave reflects, transmits, absorbs, scatters, or changes medium.
Boundary element method :: A numerical wave-simulation method that discretizes room or object surfaces rather than the full volume.
Brightness :: Perceived emphasis of high-frequency energy, influenced by spectrum, transients, decay, and level.
Brown noise :: Noise with power decreasing approximately $6$ dB per octave, produced by integrating white noise.
Buffer underrun :: Failure to supply playback samples before the device deadline, typically heard as a click, dropout, or gap.
Bus :: A mixer signal path that combines, distributes, or processes multiple channels.
Butterworth filter :: A filter family designed for maximally flat passband magnitude response.
C-weighting :: A relatively broad frequency weighting used for high-level sound measurements and peak-oriented assessment.
Cadence :: A musical arrival or punctuation whose perceived duration can be extended or clarified by reverberation.
Calibration :: Establishing a known relationship between digital values, electrical signals, acoustic level, distance, or measurement units.
Cardioid pattern :: A first-order directional pattern with maximum sensitivity forward and a nominal null to the rear.
Causality :: The requirement that a realtime system's output at a given instant cannot depend on future input.
Cepstrum :: A transform-domain representation useful for separating source and filter periodicities, detecting echoes, or estimating pitch.
Channel :: One discrete stream or component of audio within a file, bus, device, spatial format, or processor.
Channel layout :: The ordered set of channel roles, labels, and positions associated with a multichannel signal.
Chirp :: A signal whose instantaneous frequency changes over time, often used for system identification and impulse-response measurement.
Chirped echo :: An echo whose dominant frequency changes over time because geometry, diffraction, dispersion, or a sequence of reflections reorganizes an impulsive source.
Chorus :: A modulation effect that combines a signal with slowly varying delayed copies to create width and ensemble-like motion.
Clarity index :: A room metric comparing early to late energy, such as $C_{50}$ for speech or $C_{80}$ for music.
Clipping :: Nonlinear truncation when a signal exceeds the representable or permitted range.
Close microphone :: A microphone placed near a source to emphasize direct sound and reduce room contribution.
Cloud reverb :: A dense, diffuse texture built from many delayed, filtered, modulated, or granular components.
Cochlear filterbank :: A model of frequency analysis in the inner ear using overlapping auditory bands.
Codec :: An encoder-decoder system that represents audio for storage or transmission, often with compression.
Coefficient :: A numerical multiplier controlling a filter, matrix, predictor, transform, or model term.
Comb filter :: A feedforward or feedback delay filter with regularly spaced spectral peaks and notches.
Comb-filter coloration :: Audible pitch or hollowness caused by interference between a signal and one or more delayed copies.
Common-mode signal :: A component shared by multiple channels, often contrasted with differential or side information.
Complex spectrum :: A frequency-domain representation containing both magnitude and phase information.
Compression :: Dynamic-range reduction in which gain decreases as signal level rises above a defined region.
Compressor knee :: The transition shape between uncompressed and compressed gain behavior around threshold.
Concert hall :: A performance room designed to balance clarity, blend, envelopment, loudness, and musical decay.
Condenser microphone :: A microphone using a charged capacitive element and impedance-conversion electronics.
Convolution :: The operation that applies a linear time-invariant system's impulse response to an input signal.
Convolution matrix :: A set of impulse responses mapping multiple input channels to multiple output channels.
Convolution reverb :: Reverberation created by convolving audio with measured, synthesized, or designed impulse responses.
Correlation :: Statistical similarity between signals, channels, time regions, or variables.
Correlation coefficient :: A normalized measure of linear relationship, typically ranging from minus one to one.
CPU load :: The fraction of available processor time consumed by computation, scheduling, and data movement.
Crest factor :: The difference in decibels between a signal's peak level and RMS level.
Critical band :: A frequency region within which auditory interactions such as masking are especially strong.
Critical distance :: The source distance at which direct and reverberant sound energies are approximately equal.
Cross-correlation :: Similarity between two signals as a function of relative delay.
Crossfade :: A smooth transition created by decreasing one signal while increasing another.
Crossover :: A set of complementary filters that divides audio into frequency bands for separate processing.
Crosstalk :: Unwanted leakage or coupling between channels, paths, transducers, or spatial components.
Cumulative spectral decay :: A three-dimensional display of magnitude versus frequency and time, useful for observing resonant ringing.
Cutoff frequency :: The reference frequency at which a filter transitions between passband and stopband behavior.
Damping :: Frequency-dependent dissipation that reduces resonance amplitude or shortens decay.
Damping ratio :: A dimensionless measure relating decay to oscillation in a second-order resonant system.
DAW :: Digital audio workstation software used to record, edit, route, process, automate, and mix audio.
dBFS :: Decibels relative to digital full scale, where zero denotes the maximum nominal digital peak.
Decay :: Reduction of sound energy or amplitude after excitation stops.
Decay curve :: Level or energy plotted against time to characterize reverberation, damping, or release behavior.
Decay time :: The duration required for a specified drop in level or energy under defined measurement conditions.
Decibel :: A logarithmic unit expressing a ratio of powers or amplitudes with an appropriate reference.
Decimation :: Sample-rate reduction after low-pass filtering to prevent aliasing.
Decoder :: A system that converts an encoded, scene-based, object-based, or compressed representation into playback channels or samples.
Decorrelation :: Reduction of similarity between channels or signal paths, often used to increase spaciousness and diffusion.
Deconvolution :: Estimation or recovery of an impulse response or source by mathematically undoing convolution under stated assumptions.
Delay :: A time shift measured in samples, milliseconds, seconds, or musical duration.
Delay line :: Memory that stores samples so they can be read after a controlled time interval.
Delay modulation :: Time variation of delay length, producing pitch change, phase movement, or decorrelation.
Delay network :: An interconnected system of delays, filters, gains, and matrices used for reverberation or physical modeling.
Delta function :: An idealized unit impulse with unit area and zero duration, used to define system response.
Density :: The number or perceptual concentration of reflections, grains, echoes, or events per unit time.
Dereverberation :: Estimation or recovery of a less reverberant signal from audio containing room reflections and decay.
Determinism :: Property that identical inputs, parameters, seeds, and environment produce identical outputs.
Diffuse field :: A statistical sound field with energy arriving approximately uniformly from many directions.
Diffuse-field equalization :: Spectral correction referenced to the average response of sound arriving from many directions.
Diffuser :: A surface or processor that redistributes reflected energy across directions or time.
Diffusion :: The spreading of acoustic or signal energy across directions, delays, channels, or modes.
Diffusion coefficient :: A measure of how evenly a surface distributes reflected energy over direction relative to a reference.
Digital signal processing :: Numerical representation, analysis, transformation, generation, and control of sampled signals.
Digital-to-analog converter :: Hardware that converts digital samples into a continuous electrical signal for playback.
Direct sound :: Energy traveling from source to receiver without an intervening reflection.
Direct-to-reverberant ratio :: The energy or level ratio between direct sound and reverberant sound at a receiver.
Directional microphone :: A microphone whose sensitivity varies with arrival angle.
Dispersion :: Frequency-dependent propagation speed or spreading that causes components to arrive at different times.
Distance attenuation :: Reduction in direct level as source-receiver distance increases, modified by geometry and environment.
Dither :: Low-level noise added before quantization to decorrelate error and preserve small-signal linearity.
Dolby Atmos :: An immersive-audio ecosystem combining beds, objects, metadata, rendering, and defined delivery formats.
Doppler shift :: Apparent frequency change caused by relative motion between source and receiver.
Downmix :: Reduction of a multichannel mix to fewer output channels using specified gains and routing.
Downsampling :: Conversion to a lower sample rate, normally including anti-alias filtering.
DRR :: Standard abbreviation for direct-to-reverberant ratio in room-acoustic measurement, analysis, and spatial rendering.
Dry signal :: The source signal before the reverberation or other parallel effect under discussion.
Dual-mono :: Two channels processed independently without stereo linking or cross-channel interaction.
Duck :: To reduce one signal automatically when a key signal is active.
Ducking reverb :: Reverb whose return level is reduced during source activity and allowed to bloom in gaps.
Dynamic convolution :: Convolution whose response changes with level, time, state, or an interpolated set of impulse responses.
Dynamic range :: The span between the lowest useful signal and the highest undistorted or permitted level.
Early decay time :: A reverberation estimate derived from the first 10 dB of decay and extrapolated to a 60 dB rate.
Early reflections :: The first individually structured room reflections arriving after the direct sound.
Echo :: A delayed sound perceived as a distinguishable repetition rather than fused reverberation.
Echo cancellation :: Estimation and subtraction of a known echo path, especially in communication systems.
Echo density :: The number of significant reflections or delayed events per unit time.
Echo return loss enhancement :: Additional echo reduction achieved by nonlinear post-processing after linear cancellation.
Eigenfrequency :: A natural resonant frequency associated with a system mode.
Eigenvalue :: A scalar describing how a matrix transforms a corresponding eigenvector, central to feedback-network stability and decay.
Eigenvector :: A direction in a vector space whose orientation is preserved by a matrix transformation.
Elevation :: Vertical angle of a source or loudspeaker relative to the listener or coordinate origin.
Energy decay curve :: Integrated impulse-response energy plotted backward in time, commonly used to estimate reverberation time.
Energy-time curve :: Squared impulse-response magnitude plotted against time to reveal direct sound and reflection structure.
Ensemble effect :: Perceived multiplication or widening produced by small timing, pitch, and level variations among similar signals.
Envelopment :: Perception of being surrounded by sound, strongly associated with late lateral and diffuse energy.
Equal-loudness contour :: A curve showing frequency-dependent sound pressure levels judged equally loud.
Equal-power crossfade :: A crossfade law intended to maintain approximately constant power for uncorrelated signals.
Equalization :: Frequency-selective gain adjustment used to correct, balance, color, or prepare audio.
Equivalent continuous level :: The steady level containing the same energy as a varying sound over a stated interval.
ERB :: Equivalent rectangular bandwidth, a psychoacoustic measure approximating auditory-filter width.
Excitation signal :: A known signal used to stimulate a system for measurement, modeling, or synthesis.
Exponential decay :: A decay whose amplitude changes by a constant ratio per unit time and appears linear in decibels.
External sidechain :: A detector input derived from a signal other than the processor's main input.
Fabry-Pérot resonance :: Resonance created by repeated reflections between approximately parallel boundaries.
Fade-in :: A gradual increase from silence or a lower level to the intended signal level.
Fade-out :: A gradual reduction from the current level toward silence or a lower level.
Far field :: A region sufficiently distant from a source that radiation pattern and inverse-distance behavior are approximately established.
Fast Fourier transform :: An efficient algorithm for computing the discrete Fourier transform and its inverse.
Feedback :: Routing a portion of a system's output back to its input.
Feedback delay network :: A reverberator architecture containing multiple delays coupled by a feedback matrix and associated filters.
Feedback gain :: The multiplier applied in a feedback path, controlling decay rate and stability.
Feedback matrix :: The matrix that mixes delay outputs back into delay inputs in an FDN.
FFT bin :: One discrete frequency sample in a discrete Fourier transform.
FFT convolution :: Efficient convolution using multiplication of spectra from transformed signal blocks and impulse-response partitions.
Filter :: A system that changes a signal according to frequency, time, phase, level, or statistical structure.
Filterbank :: A collection of filters that decomposes or reconstructs a signal across frequency bands.
Finite impulse response filter :: A filter with a response that becomes exactly zero after a finite number of samples.
Flanging :: A modulation effect created by mixing a signal with a very short varying delay, producing moving comb filtering.
Flat response :: A response whose magnitude remains approximately constant over a specified frequency range.
Flutter echo :: Rapid, repetitive reflections between surfaces, heard as buzzing or pitched decay.
FOA :: First-order Ambisonics, containing four spherical-harmonic components in a complete three-dimensional representation.
Folded delay line :: A long acoustic or electronic path arranged compactly by folding its geometry or memory access.
Formant :: A spectral resonance that helps define vowel identity, instrument character, or resonant coloration.
Forward masking :: Reduced audibility of a sound because a preceding sound temporarily raises its perceptual threshold.
Fractional delay :: A delay containing a noninteger number of samples, implemented by interpolation or phase methods.
Frame :: A finite group of samples analyzed or processed together, often with overlap and a window.
Free field :: An ideal or approximated environment without significant reflections.
Freeze :: A reverb state that sustains circulating energy by approaching lossless feedback while controlling stability.
Frequency :: Repetition rate measured in hertz, equal to cycles per second.
Frequency response :: Magnitude and phase behavior of a system as a function of frequency.
Frequency-dependent decay :: Reverberation in which different frequency bands decay at different rates.
Frequency-domain processing :: Signal processing performed on spectral representations rather than directly on time samples.
Frequency modulation :: Variation of instantaneous frequency by another signal, producing pitch motion and sidebands.
Full scale :: The maximum nominal magnitude representable by a digital format or signal convention.
Gain :: A multiplier or logarithmic level change applied to a signal.
Gain computer :: The part of a dynamics processor that calculates desired gain from detector level and control parameters.
Gain reduction :: The amount by which a dynamics processor attenuates a signal relative to unity gain.
Gated reverb :: Reverberation shaped by a gate or envelope so the decay terminates sooner or more abruptly than naturally.
Gated reverse reverb :: A reverse or reverse-style wet field restricted to a controlled pre-event or transition window and closed with a gate or authored gain envelope.
Gaussian noise :: Random noise whose sample amplitudes follow a normal probability distribution.
Geometric acoustics :: High-frequency room modeling using rays, image sources, or beams rather than full wave equations.
Geometric spreading :: Reduction of wave intensity as energy expands over a growing area.
Granular reverb :: Reverberant texture assembled from overlapping short grains with controlled delay, pitch, envelope, and spatial distribution.
Grain :: A short windowed segment of sound used as an elementary unit in granular processing.
Group delay :: Negative derivative of phase with respect to angular frequency, interpreted as frequency-dependent envelope delay.
GUI :: Graphical user interface containing visual controls, meters, displays, navigation, and interaction feedback.
Haas effect :: Perceptual localization dominated by the first arrival when a similar delayed sound follows within a short interval.
Hadamard matrix :: An orthogonal matrix of plus and minus entries often used for efficient energy-preserving diffusion in FDNs.
Hann window :: A raised-cosine analysis window with zero-valued endpoints and useful spectral-leakage behavior.
Harmonic distortion :: Added spectral components at integer multiples of input frequencies due to nonlinearity.
Head tracking :: Measurement of listener orientation or position used to update binaural or immersive rendering.
Head-related impulse response :: The time-domain response from a spatial source to an ear, including head, torso, and pinna effects.
Head-related transfer function :: The frequency-domain counterpart of a head-related impulse response.
Headroom :: Level margin between normal operating peaks and clipping or another hard limit.
Helmholtz resonator :: A cavity-and-neck system that resonates over a relatively narrow low-frequency region.
High-pass filter :: A filter that attenuates low frequencies while passing higher frequencies.
Higher-order Ambisonics :: Ambisonics using spherical-harmonic orders above first order for greater spatial resolution.
Hilbert transform :: A phase-shifting transform used to form analytic signals, estimate envelopes, or construct quadrature components.
Histogram :: A count or probability display showing how values are distributed across ranges.
Hop size :: The sample advance between successive overlapping analysis or synthesis frames.
Householder matrix :: An orthogonal reflection matrix useful for efficient mixing and diffusion in feedback networks.
HRTF personalization :: Adapting head-related transfer functions to an individual listener's anatomy or measured responses.
Hybrid reverb :: A reverberator combining distinct methods, commonly convolutional early reflections with an algorithmic late field.
IACC :: Interaural cross-correlation coefficient, a room and spatial metric related to similarity between ear signals.
IEC :: International Electrotechnical Commission, a standards body responsible for many audio and electroacoustic measurement standards.
Image-source method :: A geometric room model that replaces specular reflection paths with mirrored virtual sources.
Immersive audio :: Audio designed to create enveloping three-dimensional experience using channels, objects, scenes, or binaural rendering.
Impulse :: A brief excitation approximating a unit sample or Dirac delta for testing system response.
Impulse response :: The output of a system excited by an impulse, fully describing a linear time-invariant system.
In situ measurement :: Measurement performed in the actual operating environment rather than a laboratory approximation.
Infinite impulse response filter :: A recursive filter whose theoretical impulse response continues indefinitely.
Infinite reverb :: A sustained reverb mode in which decay is halted or made extremely long under explicit stability controls.
Initial time-delay gap :: The interval between direct sound and the first significant reflection.
Input gain :: Gain applied before the main processing network.
Input matrix :: A matrix that maps external channels into an internal multichannel or delay-network representation.
Integrated loudness :: Program loudness accumulated over an interval using gating and frequency weighting defined by a standard.
Interaural level difference :: Difference in level between the ears, important for horizontal localization at higher frequencies.
Interaural time difference :: Difference in arrival time between the ears, important for horizontal localization at lower frequencies.
Interchannel coherence :: Frequency-dependent similarity between channels, often used to characterize diffuseness or spatial image stability.
Interleaved audio :: Multichannel sample storage in which channel samples alternate within each frame.
Interpolation :: Estimation of values between known samples, parameters, measurements, or responses.
Inverse filter :: A filter designed to undo or compensate for another system's response within practical limits.
Inverse square law :: Free-field intensity decreases with the square of distance from a point source, corresponding to about $6$ dB per distance doubling.
IR :: Common abbreviation for impulse response in acoustics, convolution, measurement, and digital signal processing.
ISO 3382 :: A family of standards defining room-acoustic measurement procedures and parameters.
Isotropic field :: A field whose statistical properties are the same in every direction.
Jackknife estimate :: A resampling method that estimates bias or variance by systematically omitting observations.
Jitter :: Unwanted timing variation in sampling, clocks, events, callbacks, or packet arrival.
JSON report :: A machine-readable structured report using JavaScript Object Notation for metrics, settings, diagnostics, or provenance.
Just intonation :: Tuning based on simple frequency ratios rather than equal division of the octave.
Just-noticeable difference :: The smallest change in a parameter or stimulus reliably perceived under defined conditions.
K-metering :: A monitoring and metering system relating average level, peak headroom, and calibrated playback.
Kaiser window :: A parameterized analysis window whose beta value trades main-lobe width against sidelobe suppression.
Key signal :: The detector signal controlling a sidechain-dependent processor such as a ducker, gate, or compressor.
Knee :: The region around a dynamics threshold where the input-output curve changes slope.
Kurtosis :: A statistical measure of distribution tail weight or impulsiveness, sometimes used in audio-event analysis.
Lagrange interpolation :: Polynomial interpolation commonly used for fractional-delay filters with controlled time-domain behavior.
Late field :: The dense reverberant region after early reflections are no longer perceived as isolated events.
Late lateral energy fraction :: The fraction of late sound energy arriving laterally, associated with listener envelopment.
Latency :: Elapsed time from an input event or sample to the corresponding output becoming available or audible.
Latency compensation :: Host or system alignment that delays other paths to match a processor's reported latency.
Lateral energy fraction :: Ratio of early lateral energy to total early energy, used in concert-hall assessment.
Leakage :: Unwanted energy escaping between signals, channels, bands, frames, rooms, or spectral bins.
Least squares :: Optimization that minimizes the sum of squared errors between observations and a model.
Level :: A logarithmic or linear measure of signal magnitude, power, loudness proxy, or acoustic pressure.
Limiter :: A dynamics processor intended to prevent peaks from exceeding a ceiling or controlled range.
Limiter ceiling :: The maximum target output level enforced by a limiter, usually stated in dBFS or dBTP.
Linear phase :: A phase response proportional to frequency, corresponding to constant group delay.
Linear prediction :: Modeling a sample as a weighted combination of other samples, widely used in coding, analysis, and dereverberation.
Linear time-invariant system :: A system satisfying superposition whose behavior does not change over time.
Linkwitz-Riley crossover :: A crossover family formed from cascaded Butterworth sections so adjacent bands sum flat in phase-aligned conditions.
Listening position :: The specified receiver location from which spatial balance, calibration, or room response is evaluated.
Localization :: Perceptual estimation of a sound source's direction, distance, or position.
Lookahead :: Intentional delay that lets a processor inspect upcoming samples before applying gain or another decision.
Loop gain :: Product of gains around a feedback path, determining stability and decay.
Lossless matrix :: A matrix that preserves signal energy, commonly orthogonal or unitary in an FDN.
Low-frequency extension :: The lowest frequency region a transducer or system reproduces within stated limits.
Low-pass filter :: A filter that passes low frequencies while attenuating higher frequencies.
LU :: Loudness unit, representing a relative loudness difference of one decibel under the applicable standard.
LUFS :: Loudness units relative to full scale, used for standardized program-loudness measurement.
Machine learning :: Methods that infer mappings, representations, decisions, or generative behavior from data rather than only explicit rules.
Magnitude response :: Frequency-dependent amplitude gain of a system, excluding phase information.
Manifest :: A structured list describing batch inputs, outputs, parameters, labels, seeds, and provenance.
Masking threshold :: Level below which a sound becomes inaudible because of competing spectral or temporal energy.
Matrix :: A rectangular numerical array representing channel routing, mixing, transforms, or linear systems.
Matrix convolution :: Multichannel convolution in which each input-output pair can have its own impulse response.
Maximum length sequence :: A deterministic pseudorandom binary sequence used for system identification and impulse-response measurement.
Mean free path :: Average distance traveled by a sound ray between reflections in an enclosure.
Measurement microphone :: A microphone designed for known, stable, and often approximately flat response.
Median :: The middle value of an ordered dataset, robust against isolated extreme values.
Mel scale :: A perceptual frequency scale designed to approximate equal pitch-distance judgments.
Metadata :: Descriptive information about audio, channels, objects, parameters, rights, measurements, or processing history.
Metering :: Visual or machine-readable measurement of levels, loudness, peaks, dynamics, spectra, or processing state.
Mid-side processing :: Stereo processing in the sum-and-difference domain rather than directly on left and right channels.
Minimum phase :: A causal stable response whose energy is concentrated as early as possible for a given magnitude response.
Mix :: The combination of multiple sources, channels, buses, and processing paths into a program.
Modal density :: Number of acoustic modes per unit frequency, generally increasing with frequency and room volume.
Modal frequency :: Resonant frequency associated with a room or system mode.
Modal overlap :: Degree to which neighboring resonances overlap in bandwidth, contributing to a smoother statistical response.
Modulation :: Time variation of amplitude, frequency, phase, delay, filter, matrix, or another parameter.
Modulation depth :: Maximum excursion of a modulated parameter around its center value.
Modulation rate :: Frequency or speed at which a modulation source evolves.
Mono compatibility :: Degree to which a stereo or multichannel signal retains intended balance and timbre when combined to mono.
Monophonic :: Containing one audio channel or one musical line, depending on context.
Moving average :: A filter or statistic formed by averaging values over a sliding interval.
Multiband processing :: Independent or linked processing of frequency regions separated by a filterbank or crossover.
Multichannel audio :: Audio containing more than two coordinated channels.
Multirate processing :: Signal processing that uses more than one sample rate within a system.
Multitap delay :: A delay line read at multiple times, with independent gains, filters, or positions for each tap.
Music Information Retrieval :: Computational analysis of musical audio, scores, metadata, structure, similarity, or semantics.
Musical acoustics :: Study of sound production, propagation, perception, and design in musical instruments, voices, rooms, and practices.
Musical time :: Organization of duration, pulse, meter, phrase, and form, which often governs useful delay and decay choices.
Mute :: Complete or near-complete attenuation of a signal path.
N-channel :: Describing a signal or system with an arbitrary stated number $N$ of channels.
Natural reverb :: Reverberation generated by a physical acoustic environment rather than an electronic processor.
Near field :: Region close to a source where pressure and particle velocity behavior is complex and simple far-field laws may not apply.
Neural audio codec :: A learned encoder-decoder that represents audio with compact latent variables and neural synthesis.
Neural dereverberation :: Dereverberation performed partly or wholly by a learned model trained on reverberant and target audio.
Neural impulse response :: An impulse response represented, generated, estimated, or conditioned by a neural model.
Neural network :: A parameterized layered function trained to model relationships from examples or objectives.
Noise floor :: Residual background level below which measurements or low-level signal details become unreliable.
Noise gate :: A dynamics processor that attenuates signals below a threshold according to timing and range controls.
Noise shaping :: Spectral redistribution of quantization noise, usually moving energy toward less audible regions.
Nonlinear distortion :: New frequency components or waveform changes produced by a system that does not obey superposition.
Nonstationarity :: Time variation in a signal's statistical properties, spectrum, level, source, or environment.
Normalization :: Gain adjustment or numerical scaling to meet a peak, loudness, energy, or representational target.
Normalized frequency :: Frequency expressed relative to sample rate, Nyquist frequency, or another reference.
Notch filter :: A narrow stop filter used to attenuate a specific frequency or resonance.
Nyquist frequency :: Half the sample rate, the highest sinusoidal frequency uniquely representable under ideal sampling assumptions.
Object-based audio :: Spatial audio represented as audio objects plus time-varying metadata interpreted by a renderer.
Octave :: A frequency interval with a ratio of two to one.
Octave band :: A standardized or practical band whose upper edge is twice its lower edge.
Offline rendering :: Processing completed without realtime deadlines, allowing arbitrary lookahead, block sizes, and computation time.
Omnidirectional microphone :: A microphone intended to have approximately equal sensitivity in every direction.
Onset :: The beginning of an audible event, often detected from rapid changes in energy or spectrum.
Open Sound Control :: A network-oriented protocol for exchanging structured control messages among musical and media systems.
Orthogonal matrix :: A real square matrix whose transpose equals its inverse and that preserves Euclidean energy.
Oscillation :: Repeated variation around an equilibrium, periodic or quasi-periodic in time.
Output gain :: Gain applied after the principal processing network.
Output matrix :: A matrix that maps internal channels or delay states to external output channels.
Oversampling :: Processing at a multiple of the base sample rate to reduce aliasing or improve numerical behavior.
Overlap-add :: Block-processing method that reconstructs linear convolution by adding overlapping transformed output blocks.
Overlap-save :: Block-convolution method that discards corrupted circular-convolution samples and retains valid output regions.
Overload :: Operation beyond a system's level, computational, thermal, bandwidth, or scheduling capacity.
Panning :: Distribution of a source among output channels or spatial directions.
Parallel compression :: Mixing a compressed path with a less compressed path to combine density and transient definition.
Parallel reverb :: Reverb placed on a return path and blended with the dry signal rather than inserted destructively.
Parameter :: A controllable numerical, categorical, or boolean value that changes system behavior.
Parameter smoothing :: Gradual interpolation of parameter changes to avoid clicks, zipper noise, and unstable transitions.
Partitioned convolution :: FFT convolution that divides a long impulse response into partitions to manage latency and computation.
Peak hold :: A meter function that retains the maximum observed value for a defined time or until reset.
Peak level :: Maximum instantaneous or sample magnitude over an interval.
Peak normalization :: Gain scaling so the largest sample or measured peak reaches a target.
Perceptual loss :: A training objective based on features or distances intended to correlate with human hearing.
Perceptual model :: A computational approximation of auditory sensitivity, masking, localization, quality, or preference.
Phase :: Position within a periodic cycle or the angular component of a complex spectrum.
Phase cancellation :: Reduction caused by summing signals with opposing phase at some frequencies.
Phase response :: System-induced phase shift as a function of frequency.
Phaser :: An effect combining a signal with phase-shifted versions to create moving spectral notches.
Pink noise :: Noise with approximately equal energy per octave and power decreasing about $3$ dB per octave.
Pitch :: Perceptual attribute ordering sounds from low to high, related primarily but not exclusively to fundamental frequency.
Plugin :: A loadable software component that processes, generates, analyzes, or controls audio inside a host.
Plugin delay compensation :: Host alignment based on latency reported by plug-ins so parallel paths remain synchronized.
Polar pattern :: Directional sensitivity or radiation shown as magnitude versus angle.
Pole :: A complex-frequency location where a transfer function becomes unbounded, governing resonance and decay.
Pre-delay :: Time between dry sound and the onset of a reverberant response.
Pre-echo :: Energy heard before a transient, caused by symmetric filtering, transform coding, or noncausal processing.
Prediction error :: Difference between an observed sample or feature and its model prediction.
Preset :: A named stored configuration of parameters and states.
Pressure zone microphone :: A microphone designed for placement at a boundary to reduce direct-reflection comb filtering.
Psychoacoustics :: Scientific study of relationships between physical sound and auditory perception.
Pumping :: Audible gain movement in dynamics processing caused by detector behavior or excessive gain reduction.
Q factor :: Ratio of center frequency to bandwidth for a resonance or filter, indicating selectivity.
Quantization :: Mapping continuous or high-resolution values to a finite set of representable levels.
Quantization noise :: Error introduced by quantization, often modeled as noise under suitable dithering assumptions.
Quarter-inch jack :: A common audio connector format used for balanced, unbalanced, insert, instrument, and headphone signals.
Quaternion rotation :: A numerically robust representation of three-dimensional orientation used in spatial-audio head tracking.
Quiet zone :: A region designed or controlled to have reduced sound level or improved isolation.
Ray tracing :: Geometric simulation that follows many sound paths through reflections, transmission, scattering, and attenuation.
Real-time factor :: Processing time divided by signal duration; values below one indicate faster-than-realtime offline performance.
Realtime processing :: Processing that must produce each output before a fixed playback or interaction deadline.
Receiver :: The microphone, listener, ear, probe, or modeled point at which an acoustic response is evaluated.
Recursive filter :: A filter that uses previous outputs, giving an impulse response that may continue indefinitely.
Reference level :: A defined electrical, digital, acoustic, or perceptual level used for calibration and comparison.
Reflection :: Sound energy returned from a boundary or impedance discontinuity.
Reflection coefficient :: Complex or energy ratio describing how much incident sound a boundary reflects.
Reflection density :: Number of reflections arriving per unit time within an impulse response.
Reflection order :: Number of boundary interactions along a geometric propagation path.
Release :: The time behavior with which a processor returns toward unity gain or a sound decays after excitation.
Render :: To calculate and write processed audio, analysis, or an intermediate artifact.
Reproducibility :: Ability to recreate a result from documented inputs, versions, parameters, seeds, and environment.
Resampling :: Conversion between sample rates using interpolation and anti-alias filtering.
Resonance :: Enhanced response near a natural frequency due to energy storage and repeated reinforcement.
Resonant frequency :: Frequency at which a system exhibits a strong modal response.
Return channel :: Mixer channel carrying the output of a shared send effect.
Reverse reverb :: Reverb whose envelope swells toward an event, commonly created by reversing a source or impulse-response workflow.
Reverb :: Persistence and blending of sound after excitation due to many reflections or an artificial equivalent.
Reverb chamber :: A reflective room used to create acoustic reverberation or perform diffuse-field measurements.
Reverb tail :: The decaying late portion of a reverberant response.
Reverberance :: Perceived prominence and duration of reverberant energy.
Reverberation radius :: Another name for critical distance in some room-acoustic contexts.
Reverberation time :: Time required for sound energy to decay by a specified amount, conventionally 60 dB.
Ringing :: Sustained oscillation after excitation, often caused by high-Q resonance, filtering, or unstable feedback.
Room correction :: Measurement-guided equalization and sometimes timing control intended to improve playback at selected positions.
Room impulse response :: Impulse response from a source position to a receiver position within an acoustic environment.
Room mode :: Standing-wave resonance determined by room dimensions and boundary conditions.
Room tone :: Background sound of a location without the foreground performance.
Root mean square :: Square root of mean squared amplitude, used as an energy-related level measure.
Round-trip latency :: Total delay from input conversion through processing to output conversion and acoustic playback.
Routing :: Assignment and movement of signals among channels, buses, processors, devices, and outputs.
RT20 :: Reverberation estimate extrapolated from a 20 dB decay segment, often from minus 5 to minus 25 dB.
RT30 :: Reverberation estimate extrapolated from a 30 dB decay segment, often from minus 5 to minus 35 dB.
RT60 :: Time in seconds associated with a 60 dB reverberant decay, measured directly or extrapolated from a shorter reliable slope.
Sabine equation :: Statistical relation among room volume, equivalent absorption area, and reverberation time under diffuse-field assumptions.
Sample :: One discrete-time amplitude value for one channel.
Sample rate :: Number of samples represented per second, measured in hertz.
Sample-rate conversion :: Resampling between different rates with appropriate reconstruction and anti-alias filtering.
Saturation :: Gradual nonlinear limiting that adds harmonics and compresses peaks rather than clipping abruptly.
Scattering :: Redistribution of reflected sound caused by surface irregularity, geometry, or wavelength-scale structures.
Scattering coefficient :: Fraction of reflected energy redirected away from the specular direction under a defined measurement method.
Schroeder frequency :: Approximate transition above which room modes overlap sufficiently for statistical treatment.
Schroeder integration :: Backward integration of squared impulse-response energy to form an energy decay curve.
Schroeder reverberator :: Classic architecture using parallel feedback comb filters followed by serial allpass diffusers.
Score :: Symbolic representation of musical instructions, events, structure, and often spatial or performance information.
Send :: A routed copy of a channel feeding an auxiliary bus or processor.
Send level :: Gain determining how much source signal enters a send path.
Sensitivity :: Change in output, metric, or perception produced by a change in input or parameter.
Serial processing :: Processing in which one stage's output feeds the next stage's input.
Shelf filter :: A filter that boosts or cuts frequencies above or below a transition region by an approximately constant amount.
Shimmer reverb :: Reverb containing pitch-shifted feedback or parallel energy, often emphasizing intervals above the source.
Short-term loudness :: Loudness measured over a moving interval of about three seconds under common broadcast standards.
Sidechain :: A signal path used to control a processor's behavior without necessarily appearing in its audio output.
Signal :: A time-varying quantity carrying audio, control, measurement, or encoded information.
Signal flow :: Ordered routing and transformation of signals through a system.
Signal-to-noise ratio :: Ratio of desired signal power to noise power under defined measurement conditions.
Silence :: Absence of intended sound, not necessarily zero digital samples or zero acoustic energy.
Sine sweep :: A sinusoid whose frequency changes over time, used for response measurement and system identification.
Slew rate :: Maximum rate at which a value or signal can change.
Slapback delay :: A single or sparse short echo, usually long enough to be distinct but rhythmically attached to the source.
Smoothing :: Reduction of abrupt variation across time, frequency, space, or parameter values.
SOFA :: Spatially Oriented Format for Acoustics, a standardized container for spatial impulse-response data and metadata.
Soft clipper :: A nonlinear limiter with a gradual transfer curve near maximum level.
Sound absorption average :: A single-number average of absorption coefficients over defined octave bands.
Sound field :: Spatial distribution of acoustic pressure, particle velocity, phase, and energy over time.
Sound pressure :: Local deviation from ambient pressure caused by a sound wave.
Sound pressure level :: Logarithmic measure of RMS sound pressure relative to 20 micropascals in air.
Sound speed :: Propagation speed of an acoustic disturbance, dependent on medium and environmental conditions.
Source :: A physical, recorded, synthesized, or modeled origin of sound energy.
Source directivity :: Direction-dependent radiation pattern of a source as a function of frequency.
Source separation :: Estimation of individual sources or stems from a mixture.
Spatial aliasing :: Directional artifacts caused by insufficient array sampling, Ambisonic order, or loudspeaker density at a frequency.
Spatial audio :: Audio in which direction, distance, extent, environment, or motion is represented or rendered.
Spatial coherence :: Consistency of phase or waveform relationships across positions, channels, or frequency.
Spatial impression :: Combined percept of apparent source width, envelopment, distance, and room scale.
Spatialization :: Placement, motion, extent, or environmental rendering of sound in a spatial scene.
Specular reflection :: Mirror-like reflection in which outgoing angle follows the geometric law of reflection.
Spectral centroid :: Energy-weighted mean frequency, often correlated with perceived brightness.
Spectral convergence :: Degree to which an estimated spectrum approaches a target spectrum over iterations or time.
Spectral decay :: Reduction of energy over time as a function of frequency.
Spectral envelope :: Smooth curve describing broad spectral shape independent of fine harmonic detail.
Spectral flux :: Frame-to-frame spectral change, often used for onset detection and activity measurement.
Spectral leakage :: Spread of sinusoidal energy across DFT bins because the analysis frame does not contain an integer number of periods.
Spectral subtraction :: Noise- or reverberation-reduction method that subtracts an estimated unwanted magnitude spectrum.
Spectrogram :: Time-frequency image showing spectral magnitude or power across successive frames.
Spectrum :: Distribution of signal magnitude, power, phase, or energy over frequency.
Speech clarity :: Intelligibility-related perception influenced by articulation, direct sound, noise, and early-to-late energy balance.
Speech transmission index :: Standardized estimate of speech intelligibility based on modulation transfer through noise and reverberation.
Spherical harmonics :: Orthogonal angular basis functions used to represent three-dimensional sound fields in Ambisonics.
Spring reverb :: Electromechanical reverberation produced by waves propagating and dispersing along metal springs.
Standing wave :: Stationary interference pattern formed by opposing waves, producing fixed nodes and antinodes.
State :: Stored information required for a processor's future output, such as delay contents, filter memories, envelopes, and random generators.
Stem :: A grouped audio submix intended for independent routing, processing, delivery, or reconstruction.
Stereo :: Two-channel reproduction or representation conventionally associated with left and right.
Stereo width :: Perceived or measured lateral spread of a stereo image.
STFT :: Short-time Fourier transform, a sequence of windowed spectra representing time-varying frequency content.
Subwoofer :: Loudspeaker designed primarily for low-frequency reproduction.
Summing :: Combining signals by addition, with gains and routing determining the result.
Surround sound :: Playback using loudspeakers around the listener to represent direction and envelopment beyond frontal stereo.
Sustain :: Continued portion of a sound or envelope after onset, or maintenance of reverberant energy in a freeze system.
Sweet spot :: Listening region in which localization, tonality, timing, and spatial balance meet the intended design.
System identification :: Estimation of a system model or impulse response from known excitation and observed output.
T20 :: Common shorthand for a reverberation estimate based on a measured 20 dB decay range.
T30 :: Common shorthand for a reverberation estimate based on a measured 30 dB decay range.
Tail length :: Rendered or effective duration of a reverberant decay after the source ends.
Tap :: One read location and gain from a delay line.
Target curve :: Desired frequency, loudness, decay, or spatial response used to guide correction or optimization.
Temporal masking :: Reduced audibility of one sound because another occurs shortly before or after it.
Threshold :: Input level or criterion at which a processor changes state or behavior.
Time constant :: Parameter describing the rate of exponential response, smoothing, attack, release, or decay.
Time domain :: Representation of signal amplitude as a function of time or sample index.
Time-frequency mask :: A matrix of gains applied to spectral bins over time for separation, enhancement, or dereverberation.
Time stretching :: Changing signal duration without proportionally changing pitch.
Time variance :: Property of a system whose response changes with absolute time or internal modulation state.
Timbre :: Perceptual quality distinguishing sounds with similar pitch and loudness through spectrum, envelope, modulation, and noise.
Toeplitz matrix :: A matrix constant along each diagonal, arising naturally in convolution and linear prediction.
Tonality :: Degree to which a sound contains stable pitch or narrowband structure rather than noise-like energy.
Total harmonic distortion :: Ratio of harmonic-distortion energy to fundamental energy under a specified test.
Transient :: A short, rapidly changing event such as an attack, click, or percussion hit.
Transient preservation :: Retention of onset definition and peak structure through processing.
Transmission loss :: Reduction in sound power transmitted through a partition, usually frequency dependent.
True peak :: Estimated maximum of the reconstructed continuous waveform, which can exceed individual sample peaks.
Truncation :: Cutting a signal, impulse response, numerical representation, or series to a finite duration or order.
Tukey window :: A tapered-cosine window with an adjustable flat center region.
Two-port network :: A system described by input-output variables at two interfaces, useful for electroacoustic and transmission modeling.
Uncorrelated signals :: Signals with negligible linear correlation over the measurement interval.
Underflow :: Numerical result too small for normal representation, potentially becoming denormalized or zero.
Uniform partitioning :: Partitioned convolution using equal-size impulse-response blocks.
Unit impulse :: A discrete signal equal to one at one sample and zero elsewhere.
Unitary matrix :: A complex matrix whose conjugate transpose equals its inverse and that preserves energy.
Unity gain :: Gain of one, corresponding to zero decibels of level change.
Upmix :: Creation of a higher-channel-count or more spatial representation from fewer input channels.
Upsampling :: Conversion to a higher sample rate by interpolation and reconstruction filtering.
User preset :: A parameter state saved and named by the user rather than shipped as factory content.
Validation :: Checking that inputs, parameters, outputs, formats, and invariants meet defined requirements.
Variance :: Mean squared deviation from a mean, describing statistical spread.
Vector base amplitude panning :: Loudspeaker panning based on vector gains for a selected speaker basis.
Velocity microphone :: A pressure-gradient microphone whose output relates to acoustic particle velocity.
Velvet noise :: Sparse randomized impulse sequence used for efficient decorrelation, diffusion, and reverberation structures.
Venue :: A physical performance, recording, or playback space with characteristic geometry and acoustics.
Vibrato :: Periodic or quasi-periodic pitch modulation used expressively in voices, instruments, or processing.
Virtual acoustics :: Simulation or reproduction of acoustic environments through signal processing and spatial rendering.
Virtual source :: A perceived or modeled source location not occupied by a physical emitter.
Visualization :: Graphical representation of signals, spectra, decay, geometry, routing, metrics, or model state.
Vocoder :: A system that applies spectral-envelope or band-energy information from one signal to another.
Volume :: Physical enclosure size or a colloquial level control; technical writing should state which meaning is intended.
VST3 :: Steinberg's cross-platform plug-in interface supporting audio processing, instruments, automation, and host integration.
Wave equation :: Partial differential equation governing propagation of acoustic pressure or related wave quantities.
Wave field synthesis :: Spatial reproduction method using dense loudspeaker arrays to approximate desired wavefronts over an area.
Waveguide :: A physical or digital structure that constrains wave propagation along one or more dimensions.
Wavelet transform :: Multiresolution analysis using localized basis functions with scale-dependent time-frequency resolution.
Wavelength :: Distance traveled by one wave cycle, equal to sound speed divided by frequency.
WAV :: A RIFF-based audio file container commonly storing PCM or floating-point samples and metadata chunks.
Weighted prediction error :: Multichannel linear-prediction method widely used for blind speech dereverberation.
Wet signal :: The reverberated or otherwise effect-processed component in a wet-dry mixture.
Wet-dry mix :: Relative blend of processed and unprocessed signal paths.
White noise :: Noise with approximately equal power per hertz over the band of interest.
Window :: A finite weighting function applied to a frame before spectral or statistical analysis.
Window function :: A named taper such as Hann, Kaiser, or Tukey used to manage frame-edge discontinuities.
Word clock :: A digital timing reference used to synchronize audio converters and devices at sample rate.
World-space coordinate :: Position or direction expressed in a shared scene coordinate system rather than a source- or listener-local frame.
WPE :: Standard abbreviation for weighted prediction error, a widely used multichannel dereverberation method.
X-curve :: A cinema monitoring target response defined for calibrated dubbing stages and theaters.
XLR :: A locking balanced connector family widely used for microphones, line-level audio, and digital interconnection.
XML :: Extensible Markup Language, a structured text format used by some audio metadata and interchange standards.
XY microphone technique :: A coincident stereo method using angled directional microphones to encode level differences with strong mono compatibility.
Yaw :: Rotation around the vertical axis, corresponding approximately to turning the head left or right.
YIN :: A fundamental-frequency estimation algorithm based on a modified difference function and error control.
Yule-Walker equations :: Linear equations relating autocorrelation to autoregressive model coefficients.
Zero crossing :: A point where a waveform changes sign.
Zero-latency convolution :: A convolution design whose first partition is processed with no added block delay beyond the host buffer, while later partitions use FFTs.
Zero padding :: Appending zeros to a signal or frame to prevent circular wraparound or increase spectral sampling density.
Zero-phase filter :: A noncausal offline filter with no phase shift, often implemented by forward-backward processing.
Zipper noise :: Audible stepping caused by abrupt or insufficiently smoothed parameter changes.
Zobel network :: An impedance-equalization network used in loudspeaker and analog circuit design.
Zone :: A defined spatial region assigned distinct playback, measurement, routing, or acoustic behavior.
Z-transform :: Complex-frequency representation of discrete-time signals and systems used to analyze poles, zeros, and stability.
Air-loss filter :: A frequency-dependent attenuation filter placed in a propagation or feedback path to model energy lost while sound travels through air.
Artificial reverberation :: Deliberate creation or modification of reflected and decaying sound by architectural, mechanical, electromechanical, electronic, or digital means.
Backward energy integration :: Reverse cumulative summation of squared impulse-response samples used to form an energy decay curve from the response tail toward its onset.
Bass ratio :: Room-acoustic measure comparing low-frequency reverberation times with mid-frequency values to describe perceived warmth or bass persistence.
Binaural quality index :: Spatial metric derived from interaural cross-correlation and used to characterize apparent source width or listener envelopment.
Brilliance ratio :: Room-acoustic ratio comparing high-frequency reverberation with mid-frequency reverberation to characterize treble liveliness or damping.
Brute-force convolution :: Direct application of a long impulse response without structural approximation, often accurate but computationally expensive for realtime multichannel reverberation.
Center time :: Energy-weighted mean arrival time of an impulse response, commonly written $T_s$, that summarizes the balance between early and late energy.
Channel decorrelation :: Reduction of similarity between output channels so a reverberant field spreads spatially instead of collapsing toward a phantom source.
Chorus artifact :: Audible pitch wandering or ensemble-like motion caused when delay modulation in a reverberator becomes too deep, fast, or coherent.
Circulant feedback matrix :: FDN matrix whose rows are cyclic shifts of one another, enabling structured eigenanalysis and efficient implementations.
Coloration-duration factorization :: Reverberator design strategy that separates control of spectral character from control of decay time as nearly independently as possible.
Colorless reverberation :: Artificial reverberation designed to avoid conspicuous periodic resonances, metallic ringing, and comb-filter spectral coloration.
Common delay divisor :: Integer factor shared by several delay lengths that can align recurrences and reduce the effective modal richness of a delay network.
Computational room model :: Numerical representation of acoustic propagation and boundary interaction used to estimate impulse responses or sound fields in an enclosure.
Conformal damping map :: Interpretation of frequency-dependent delay-line loss as a mapping that moves lossless poles from the unit circle to frequency-dependent radii inside it.
Contractive matrix :: Matrix whose induced gain does not exceed one, useful for ensuring that a feedback network does not increase signal energy.
Coprime delay lengths :: Delay lengths whose greatest common divisor is one, chosen to reduce coincident recurrences and improve modal distribution.
Courant stability condition :: Time-step and grid-spacing constraint that keeps an explicit finite-difference wave simulation numerically stable.
Damping filter :: Filter inside a reverberant feedback path that imposes frequency-dependent loss and therefore frequency-dependent decay time.
Damping substitution :: Replacement of each unit delay by a delay combined with a propagation-loss filter to convert a lossless reverberator into a decaying one.
Decay curvature :: Departure of a decay trace from a straight line in decibels, indicating changing decay rate, coupled spaces, modes, noise, or time variance.
Decay eigenvalue :: Eigenvalue of a recursive network whose magnitude and angle determine a modal decay rate and oscillation frequency.
Decay intercept :: Level at which a fitted decay line crosses a chosen time origin, used with slope when estimating reverberation time.
Decay ridge :: Persistent narrowband feature in an energy decay relief that reveals a mode or resonance lasting longer than neighboring frequencies.
Decay slope :: Rate of level reduction over time on a decibel decay curve, usually expressed in decibels per second.
Delay density :: Number and distribution of distinct delay events available to a reverberator over a specified interval.
Delay distribution :: Statistical or designed arrangement of delay lengths that controls mode spacing, recurrence, density buildup, and temporal texture.
Delay-line damping filter :: Loss filter associated with a particular reverberator delay line and designed from a target frequency-dependent reverberation time.
Delay-line scaling :: Multiplication of nominal delay lengths by room-size, sample-rate, or tuning factors while preserving required ordering and stability constraints.
Density buildup :: Increase in the number of audible reflections per unit time as energy recirculates through a room or delay network.
Definition index :: Early-to-total energy ratio, commonly $D_{50}$ for speech, expressing how much impulse-response energy arrives during the first 50 milliseconds.
Diffuse-field assumption :: Approximation that reverberant energy is statistically uniform in position and direction, underlying many classical room-acoustic formulas.
Diffuse reflection :: Boundary interaction that redistributes incident sound across many outgoing directions rather than preserving a single mirror-like ray.
Diffusion time :: Time required for a reverberator to develop a sufficiently dense, spatially distributed response after excitation.
Distance law :: Relationship between propagation distance and sound level, such as inverse-distance pressure decay in an ideal free field.
Doppler-free modulation :: Delay or network modulation designed to reduce audible pitch shift while still breaking up static resonances.
Double-slope decay :: Energy decay containing two approximately linear decibel regions with different slopes, often caused by coupled spaces or layered processing.
Early resonance :: Low-frequency room or reverberator mode that remains individually perceptible before modal density becomes statistically high.
Early-to-late ratio :: Ratio of impulse-response energy before a selected boundary to energy after it, used to quantify clarity and distance.
Echo-density growth :: Time evolution of reflection count or normalized echo density as a reverberant response transitions from sparse echoes to a dense tail.
Eigenmode :: Characteristic oscillation associated with an eigenvector and eigenvalue of a linear acoustic or feedback system.
Elliptic feedback delay network :: Structured FDN family using elliptic or related matrix constructions to shape mode distribution and computational behavior.
Energy decay relief :: Time-frequency surface formed by backward-integrating energy within spectral bins, revealing frequency-dependent decay and resonant ridges.
Energy preservation :: Property of a lossless structure whose output or state energy equals its input or previous-state energy under the chosen norm.
Energy reflection coefficient :: Fraction of incident acoustic energy returned by a boundary, equal to one minus absorbed and transmitted fractions when accounting is complete.
Exact transfer-function model :: Point-to-point room model that represents every relevant source-receiver impulse response directly rather than approximating its perceptual structure.
Extrapolated RT60 :: Sixty-decibel decay time inferred from a shorter fitted interval such as EDT, T20, or T30 rather than observed over the full range.
FDN reverberation :: Late-reverberation synthesis using multiple delays coupled through a feedback matrix, with losses and output mixing controlling decay and spatial character.
Feedback comb bank :: Parallel or coupled collection of feedback comb filters used to create many decaying modes and a dense reverberant tail.
Finite-difference time-domain method :: Grid-based numerical method that advances discretized wave-equation variables through time while respecting stability and boundary conditions.
First-order delay filter :: One-pole or one-zero loss filter designed for a reverberator delay line to approximate a target decay at selected frequencies.
Freeverb :: Widely implemented Schroeder-Moorer-style reverberator using parallel lowpass-feedback comb filters followed by serial allpass-like diffusers.
Freeverb allpass approximation :: Freeverb diffusion section conventionally called allpass although its implementation and coefficient choices only approximate ideal allpass behavior.
Frequency-dependent energy decay curve :: Family of decay curves computed in bands or spectral bins instead of after broadband energy summation.
Frequency-dependent reverberation time :: Reverberation-time function over frequency, commonly specified in octave or fractional-octave bands and realized with feedback-path damping filters.
Grid dispersion :: Frequency- and direction-dependent wave-speed error introduced by spatial discretization in finite-difference or waveguide-mesh simulation.
Grid point :: Discrete spatial sample at which a numerical acoustic model stores and updates pressure, velocity, or traveling-wave variables.
Homogeneous feedback delay network :: FDN whose delay paths or attenuation structure follow a uniform design pattern rather than path-specific heterogeneous models.
Image-source order :: Number of boundary reflections represented by a virtual source in the image-source method.
Impulse-response smoothness :: Degree to which a late impulse response avoids isolated spikes, periodic gaps, and abrupt statistical changes.
Inhomogeneous feedback delay network :: FDN with path-dependent delays, filters, gains, or routing intended to model nonuniform propagation and decay.
Input-output room model :: Representation of a room by transfer functions or impulse responses between chosen source and receiver points without explicitly simulating the full field.
Interaural cross-correlation coefficient :: Normalized similarity of left- and right-ear signals over a specified window and frequency band, used in spatial-acoustic assessment.
Junction scattering :: Redistribution of incoming traveling waves into outgoing branches according to impedance and conservation constraints at a waveguide junction.
Late-field isotropy :: Condition in which late reverberant energy arrives with approximately equal statistics from all directions.
Late-reverberation approximation :: Efficient statistical or recursive model that replaces explicit computation of every high-order reflection after the early response.
Late-tail onset :: Time at which a reverberant response becomes dense enough to treat as a statistical late field rather than isolated reflections.
Listener envelopment :: Perception of being surrounded by reverberant sound, strongly influenced by late lateral energy and interaural decorrelation.
Lossless feedback matrix :: Feedback matrix that preserves state energy, typically orthogonal in real-valued FDNs or unitary in complex-valued FDNs.
Lossless prototype reverberator :: Recursive network configured without attenuation so its modes do not decay, used as a starting point before damping is introduced.
Lowpass-feedback comb filter :: Feedback comb filter containing a low-pass loss filter in its loop so high frequencies decay faster than low frequencies.
Matrix modulation :: Time variation of a feedback or mixing matrix used to move resonances, alter diffusion, or decorrelate channels.
Mean absorption coefficient :: Surface-area-weighted average of frequency-dependent absorption coefficients across the boundaries of a room.
Mean scattering coefficient :: Surface-area-weighted average of boundary scattering coefficients used in geometric-acoustic estimates.
Metallic ringing :: Audible pitched or bell-like persistence caused by sparse, regularly spaced, or insufficiently damped reverberator modes.
Mode-density threshold :: Frequency or time region beyond which modes or echoes are numerous enough that statistical description becomes more useful than individual tracking.
Mode-frequency distribution :: Arrangement of resonant frequencies in a room or recursive network, including spacing, degeneracy, clustering, and irregularity.
Modal decay time :: Time constant or RT60 associated with one resonant mode rather than a broadband or band-averaged response.
Modal degeneracy :: Coincidence of two or more theoretically distinct modes at the same frequency, often increasing resonance strength.
Moorer reverberator :: Artificial-reverberation architecture extending Schroeder structures with explicit early reflections and frequency-dependent damping in feedback comb filters.
Multiband delay filter :: Feedback-path filter designed from several target decay bands to approximate a detailed frequency-dependent RT60 curve.
Nested allpass filter :: Diffusion structure in which an allpass or delay network is embedded inside another allpass loop to increase echo complexity efficiently.
Noise-floor bend :: Point where a measured decay curve departs from its reverberant slope because background noise begins to dominate backward-integrated energy.
Nonexponential decay :: Reverberant decay that cannot be represented adequately by one constant exponential slope over the interval of interest.
Normalized echo density :: Echo-density measure normalized against a Gaussian or reference process so responses with different levels and durations can be compared.
Orthogonal mixing :: Energy-preserving mixing of real-valued channels or delay states using an orthogonal matrix.
Orthogonalized delay filter :: Delay-line damping design transformed to preserve a desired lossless or orthogonal network relationship while imposing frequency-dependent decay.
Output decorrelation delay :: Short channel-specific delay used at reverberator outputs to reduce interchannel correlation and broaden spatial imaging.
Parallel comb bank :: Set of comb filters driven in parallel and summed, producing a superposition of modal families with different delay periods.
Paraunitary matrix :: Frequency-dependent matrix whose conjugate-transpose product is identity on the unit circle, preserving energy across frequency.
Passivity :: Property that a system cannot generate net energy, providing a strong sufficient condition for stable physical and feedback-network models.
Perceptual reverberator :: Reverberator optimized to reproduce salient auditory cues rather than every geometrical reflection or exact point-to-point transfer function.
Point-to-point transfer function :: Acoustic transfer function from one specified source position to one specified receiver position.
Pole angle :: Angular position of a pole in the complex plane, determining its oscillation frequency in a discrete-time resonant system.
Pole radius :: Distance of a pole from the origin, determining modal decay rate and stability for a discrete-time recursive system.
Pressure reflection coefficient :: Complex ratio of reflected to incident acoustic pressure at a boundary, carrying both magnitude and phase information.
Prime delay length :: Delay length chosen as a prime integer to reduce shared periodicities with other paths in a delay network.
Prime-power delay length :: Delay length selected from powers of distinct primes to structure recurrence and mode distribution in an FDN.
Propagation-loss filter :: Filter representing distance- and frequency-dependent energy loss accumulated while sound travels through a medium.
Ray-tracing order :: Maximum number of reflections followed for each acoustic ray in a geometric room simulation.
Receiver directivity :: Direction-dependent sensitivity of a microphone, ear model, or virtual receiver used during acoustic rendering or measurement.
Rectilinear waveguide mesh :: Digital waveguide mesh arranged on an orthogonal grid, simple to implement but subject to direction-dependent dispersion.
Reflection path :: Geometric route from source to receiver containing one or more boundary interactions and associated delay, loss, and filtering.
Regression interval :: Selected decibel or time range over which a line is fitted to a decay curve for RT estimation.
Reverberant power gain :: Frequency-dependent ratio of reverberator output power to input power under specified stationary excitation and routing.
Reverberation coloration :: Audible spectral shaping caused by uneven modal amplitudes, delays, feedback, damping, or output mixing in a reverberant system.
Reverberation diffusion :: Temporal and spatial spreading that converts sparse reflections into a dense, less localized decay field.
Reverberation problem :: Engineering task of reproducing the perceptually important behavior of a space under finite computation, memory, latency, and control constraints.
Reverse cumulative integration :: Summation from the end of a sequence toward its beginning, used in Schroeder energy-decay analysis.
Room constant :: Absorption-related room quantity used in steady-state level and critical-distance formulas, often written from area and mean absorption.
Room-mode transition frequency :: Approximate frequency separating sparse individually resolvable room modes from an increasingly overlapping statistical field.
Sabine absorption area :: Equivalent perfectly absorbing area obtained by summing each surface area multiplied by its absorption coefficient.
Sample-rate delay scaling :: Adjustment of delay lengths in proportion to sampling rate so physical delay times remain approximately constant.
Scattering delay network :: Room-reverberation structure connecting delay lines through scattering nodes derived from room geometry and boundary properties.
Scattering junction :: Network node that maps incoming traveling-wave components to outgoing components while satisfying continuity and conservation relations.
Schroeder allpass section :: Recursive delay structure with nominally flat magnitude response used to increase echo density and phase dispersion.
Series allpass chain :: Cascade of allpass diffusers used to build echo density before or after a recursive reverberation stage.
Specific echo density :: Echo count per unit time under a stated amplitude or statistical criterion, used to assess reverberant texture.
Spectral coloration equalizer :: Filter used outside or around a reverberant network to correct its steady-state magnitude response independently of decay design.
Statistical late field :: Dense reverberant tail characterized through distributions, correlation, decay, and spectrum rather than individually modeled reflections.
Steady-state energy density :: Average acoustic energy per unit volume after a continuous source and room losses reach statistical equilibrium.
Surface scattering coefficient :: Frequency-dependent fraction of reflected energy redistributed away from the specular direction by boundary roughness or geometry.
Tapped-delay early-reflection model :: Finite set of delayed, scaled, filtered, and spatialized taps approximating the perceptually important first room reflections.
Time-varying delay line :: Delay whose read position changes over time, requiring interpolation and potentially producing Doppler shift or decorrelation.
Time-varying reverberator :: Reverberator whose delays, filters, matrices, or gains change over time to suppress static modes or create intentional motion.
Tonal correction filter :: Equalizer that compensates the average spectral coloration of a reverberator without redefining its modal decay times.
Transfer-function matrix :: Matrix of point-to-point transfer functions mapping multiple acoustic or electrical inputs to multiple outputs.
Triangular feedback matrix :: Upper- or lower-triangular FDN matrix whose eigenvalues are visible on its diagonal and whose coupling has a directed structure.
Triangular waveguide mesh :: Mesh using triangular spatial cells to improve angular isotropy relative to a simple rectilinear grid.
Unit-circle pole :: Pole lying exactly on the complex unit circle, corresponding to an undamped discrete-time mode in an ideal lossless system.
Unitary mixing :: Energy-preserving mixing of complex-valued states or channels using a unitary matrix.
Waveguide-mesh reverberation :: Physical-modeling approach that propagates traveling waves across an interconnected spatial mesh to approximate room acoustics.
Waveguide numerical dispersion :: Frequency- and direction-dependent propagation-speed error caused by discrete mesh geometry and sampling.
Zita-Rev1 :: Open-source FDN reverberator by Fons Adriaensen with eight delay lines, frequency-dependent decay controls, and stereo output processing.
"""


def _entries() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(ENTRY_DATA.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if " :: " not in line:
            raise ValueError(f"Glossary line {line_number} lacks ' :: ' delimiter")
        term, definition = (part.strip() for part in line.split(" :: ", 1))
        key = term.casefold()
        if key in seen:
            raise ValueError(f"Duplicate glossary term: {term}")
        if not term[0].isalpha():
            raise ValueError(f"Glossary term must begin with a letter: {term}")
        if len(definition.split()) < 6:
            raise ValueError(f"Glossary definition is too short: {term}")
        seen.add(key)
        entries.append((term, definition))

    entries.sort(key=lambda entry: entry[0].casefold())
    if len(entries) < MINIMUM_ENTRY_COUNT:
        raise ValueError(
            f"Glossary requires at least {MINIMUM_ENTRY_COUNT} entries; found {len(entries)}"
        )
    return entries


def _render(entries: list[tuple[str, str]]) -> str:
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for term, definition in entries:
        groups[term[0].upper()].append((term, definition))

    lines = [
        "# Glossary",
        "",
        (
            f"This colossal glossary defines {len(entries)} terms used throughout verbx and the "
            "wider literature of acoustics, reverberation, dereverberation, spatial audio, "
            "recording, music production, plug-in engineering, measurement, and Audio AI. "
            "Definitions are intentionally compact: each establishes the book's working meaning "
            "without pretending to replace the cited standards, textbooks, or research papers."
        ),
        "",
        (
            "A term may have narrower meanings in a particular standard or discipline. Read units, "
            "channel conventions, measurement windows, reference levels, and algorithm settings "
            "with the surrounding chapter before comparing results. Acronyms are cross-defined "
            "where they are common enough to be encountered independently."
        ),
        "",
    ]
    for letter, letter_entries in groups.items():
        lines.extend([f"## {letter}", ""])
        for term, definition in letter_entries:
            lines.extend([f"**{term}.** {definition}", ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    entries = _entries()
    OUTPUT.write_text(_render(entries), encoding="utf-8")
    print(f"Wrote {len(entries)} glossary entries to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
