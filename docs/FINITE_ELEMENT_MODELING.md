# Finite-Element Modeling for Reverb and Resonant Systems

Finite-element modeling (FEM) provides a bridge between a physical description
and an audible response. Instead of beginning with a desired RT60 or a familiar
reverb topology, an FE model begins with geometry, material properties,
constraints, sources, and receivers. The continuous wave or vibration problem
is divided into a finite set of connected elements, and the resulting system of
equations predicts how energy moves through the model. That prediction can be
sampled as an impulse response and used directly for convolution, reduced to a
modal model, or combined with a statistical late field.

This chapter distinguishes three related tasks that are easy to conflate:

- **Structural FEM** predicts vibration in solids such as plates, springs,
  shells, instrument bodies, and loudspeaker components.
- **Acoustic FEM** predicts pressure in a fluid volume such as air inside a
  room, duct, enclosure, or cavity.
- **Vibroacoustic FEM** couples structural motion to the surrounding fluid, as
  when a plate drives air or a flexible wall absorbs and reradiates energy.

verbx currently implements a bounded structural modal approximation for spring
tanks and plates. It does not yet claim to be a general CAD-to-room acoustic FE
solver. The distinction matters: a plate mesh predicts the plate's resonances,
whereas a room mesh predicts the pressure field around sources and listeners.

## From the Wave Equation to a Matrix System

For a homogeneous, lossless fluid, acoustic pressure $p(\mathbf{x},t)$ obeys
the wave equation

$$
\frac{1}{c^2}\frac{\partial^2 p}{\partial t^2}
- \nabla^2 p = s(\mathbf{x},t),
$$

where $c$ is the speed of sound, $\mathbf{x}$ is position, $t$ is time, and
$s$ represents a source. Spatially varying density and compressibility require
a more general form, but the modeling idea is unchanged. FEM multiplies the
differential equation by spatial test functions, integrates over the domain,
and uses integration by parts to obtain a weak form. The pressure field is then
approximated as a weighted sum of local basis functions:

$$
p(\mathbf{x},t) \approx \sum_{i=1}^{N} N_i(\mathbf{x})p_i(t).
$$

Here $N_i(\mathbf{x})$ is the basis function associated with degree of freedom
$i$, and $p_i(t)$ is its time-varying coefficient. Element contributions are
assembled into the semidiscrete system

$$
\mathbf{M}\ddot{\mathbf{p}}(t)
+ \mathbf{C}\dot{\mathbf{p}}(t)
+ \mathbf{K}\mathbf{p}(t)
= \mathbf{f}(t),
$$

where $\mathbf{M}$ is the mass matrix, $\mathbf{C}$ represents damping and
impedance losses, $\mathbf{K}$ is the stiffness matrix, and $\mathbf{f}(t)$ is
the source vector. The same matrix form describes many structural systems when
pressure coefficients are replaced by displacements.

The matrices are sparse because each basis function interacts only with nearby
elements. This locality makes large models tractable, but three-dimensional
audio-band room meshes can still contain millions of unknowns. Sparse storage,
iterative solvers, domain decomposition, and parallel execution are therefore
not optional refinements at realistic scale; they are the practical machinery
that makes the calculation possible.

## Boundary Conditions Are Part of the Instrument

Geometry alone does not define a room response. Every boundary must state what
happens when energy reaches it.

- A rigid wall is commonly represented by a Neumann condition with zero normal
  particle velocity.
- A pressure-release boundary uses a Dirichlet condition, often $p=0$.
- A locally reacting absorber uses an impedance or Robin condition relating
  pressure to normal velocity.
- An interface between a structure and air requires continuity of normal
  velocity and a force balance between pressure and structural stress.

Real absorbers are frequency dependent. Replacing a carpet, audience, curtain,
or microperforated panel with one broadband absorption coefficient can produce
a plausible scalar RT60 while getting modal damping, phase, and early decay
wrong. For auralization, complex impedance is often more useful than a single
absorption number because it preserves both magnitude and phase behavior.

Source and receiver definitions are equally consequential. A mathematical
point source may overexcite mesh-scale energy; a distributed monopole or a
measured directivity pattern can be more stable and more realistic. A receiver
should interpolate pressure from the containing element rather than simply use
the nearest node. Moving either endpoint by a fraction of a wavelength can
change low-frequency modal balance substantially.

## Mesh Resolution and Numerical Dispersion

An FE mesh must resolve the shortest wavelength of interest. If $h_{\max}$ is
the largest relevant element dimension and $n_{\lambda}$ is the chosen number
of nodes or elements per wavelength, a useful planning estimate is

$$
f_{\max} \approx \frac{c}{n_{\lambda}h_{\max}}.
$$

Six to ten spatial samples per wavelength is a common starting range for
low-order elements, not a guarantee. Element order, shape quality, formulation,
solver, and acceptable phase error all affect the requirement. A mesh that is
adequate for a 250 Hz room-mode study is nowhere near adequate for a 20 kHz
full-band simulation. Halving the element size in three dimensions can increase
the number of volume elements by roughly a factor of eight before time-step or
solver costs are considered.

Under-resolution produces numerical dispersion: simulated waves travel at a
frequency- and direction-dependent speed. The result can be a shifted mode,
blurred arrival, or synthetic afterglow that looks like reverberation but is a
numerical artifact. Good practice includes a mesh-convergence study. Repeat the
calculation with a finer mesh and compare eigenfrequencies, receiver response,
decay curves, and arrival times. A result that changes materially under modest
refinement is not yet a stable acoustical prediction.

Mesh quality matters as much as nominal spacing. Highly skewed or flattened
elements can degrade conditioning and interpolation. Local refinement should
follow geometric detail, impedance transitions, source and receiver regions,
and zones with high pressure gradients. Refining an entire cathedral to the
same scale as a small ornament is usually wasteful; simplifying irrelevant
geometry is part of model design.

## Frequency-Domain, Time-Domain, and Modal Solutions

Three solution strategies answer different questions.

### Frequency-domain FEM

Assuming sinusoidal steady state at angular frequency $\omega$ gives

$$
\left(\mathbf{K}+j\omega\mathbf{C}-\omega^2\mathbf{M}\right)
\mathbf{P}(\omega)=\mathbf{F}(\omega).
$$

Solving this system over a frequency grid yields transfer functions, pressure
maps, and resonance detail. It is efficient when only a limited low-frequency
band or a small set of frequencies is required. A broadband impulse response
requires enough frequency samples, consistent phase, and an inverse transform.

### Time-domain FEM

Time integration advances the matrix wave equation after an impulse, swept
source, or other excitation. It produces arrivals and decays directly, but the
time step must satisfy the formulation's stability and accuracy conditions.
Long RT60 values make this expensive because the solver must continue until the
field decays sufficiently. Absorbing boundaries and frequency-dependent losses
also require causal time-domain realizations.

### Modal reduction

For a linear system, free modes solve the generalized eigenproblem

$$
\mathbf{K}\boldsymbol{\phi}_r
= \omega_r^2\mathbf{M}\boldsymbol{\phi}_r,
$$

where $\omega_r$ and $\boldsymbol{\phi}_r$ are the angular frequency and shape
of mode $r$. Retaining a subset of modes transforms a large spatial problem
into a compact resonator bank. With source vector $\mathbf{b}$ and receiver
vector $\mathbf{g}$, the contribution of mode $r$ is weighted by

$$
a_r =
\left(\boldsymbol{\phi}_r^{\mathsf{T}}\mathbf{b}\right)
\left(\mathbf{g}^{\mathsf{T}}\boldsymbol{\phi}_r\right).
$$

A damped modal impulse response can then be written

$$
h(t)=\sum_{r=1}^{R}
\frac{a_r}{\omega_{d,r}}
e^{-\sigma_r t}\sin(\omega_{d,r}t),
$$

with damping rate $\sigma_r$ and damped angular frequency $\omega_{d,r}$.
Modal reduction is especially effective below the Schroeder frequency, where
individual room or structure modes remain perceptually and numerically
important. At higher frequencies the mode count grows rapidly, and a geometric
or statistical late-field method may be more efficient.

## From an FE Model to an Impulse Response

A practical room-acoustics pipeline is more than one matrix solve:

1. Clean and simplify geometry while retaining acoustically important volumes,
   openings, surfaces, and flexible structures.
2. Assign fluid and material parameters with units and frequency dependence.
3. Select boundary conditions, source directivity, receiver position, mesh
   order, spatial resolution, and target frequency range.
4. Solve in the frequency domain, time domain, or a reduced modal basis.
5. Convert receiver pressure to a causal impulse response with a documented
   normalization and reference distance.
6. Add high-frequency geometric or statistical energy if the FE band does not
   cover the complete audible range.
7. Validate modes, arrival times, energy decay, spectra, and spatial behavior
   against analytical cases or measurements.
8. Export the IR with geometry, mesh, solver, material, source, receiver, and
   software-version provenance.

Hybridization deserves special care. A low-frequency FE response and a
high-frequency image-source, ray-tracing, or FDN response should overlap over a
transition band. Match level, phase or group delay where meaningful, decay
slope, and spatial convention before crossfading. A hard splice can create a
spectral shelf or two unrelated rooms occupying adjacent bands.

## Structural FEM in verbx

The current verbx `modal-fe` path applies the same reduction logic to bounded
spring and plate structures. Spring tanks use lumped masses, chain stiffness,
optional inter-spring coupling, drive and pickup vectors, and mode-dependent
loss. Plates use a structured clamped grid, a mass-lumped matrix, a discrete
thin-plate bending operator, optional tension, and a bilinearly interpolated
pickup. Both systems solve normal modes offline and synthesize a causal modal
IR for the regular render pipeline.

```bash
verbx render guitar.wav spring_fe.wav \
  --engine algo --algo-model spring \
  --electromechanical-solver modal-fe \
  --spring-count 3 --spring-fe-nodes 36 \
  --spring-fe-modes 48 --spring-fe-coupling 0.14 \
  --spring-fe-loss 0.42 --rt60 2.0 --wet 0.55 --dry 0.75

verbx render vocal.wav plate_fe.wav \
  --engine algo --algo-model plate \
  --electromechanical-solver modal-fe \
  --plate-fe-nx 20 --plate-fe-ny 14 --plate-fe-modes 72 \
  --plate-fe-loss 0.18 --plate-pickup-x 0.18 \
  --plate-pickup-y 0.76 --rt60 3.4 --wet 1 --dry 0
```

These commands are deterministic sound-design and research tools. Their node
counts, mode limits, and simplified damping keep resource use bounded. They do
not include a three-dimensional room fluid mesh, detailed fixture geometry,
measured transducers, nonlinear springs, or calibrated hardware losses.

## Validation and Listening Tests

An FE render should be tested as both a numerical model and an audio object.

| Question | Numerical check | Listening check |
|---|---|---|
| Are modes in the right places? | Compare eigenfrequencies with analytical or measured peaks | Sweep a sine slowly and listen for shifted resonances |
| Is damping credible? | Fit per-band EDT, T20, and T30 | Compare low and high decay independently |
| Is the mesh converged? | Repeat at finer resolution | Level-match and null or difference the two IR renders |
| Are source and receiver positions meaningful? | Plot mode participation and transfer magnitude | Move pickup or listener and listen for expected nodes |
| Is the hybrid crossover coherent? | Inspect magnitude, phase, and energy through overlap | Listen to impulses, speech, and sustained broadband material |
| Is the model stable? | Check finite samples, passive decay, and bounded energy | Listen for growing tones, clicks, or artificial afterglow |

Use `verbx analyze` to preserve measurements beside the command and output:

```bash
verbx analyze plate_fe.wav --json-out plate_fe.analysis.json
```

The JSON report does not prove physical accuracy, but it makes comparisons
repeatable and exposes accidental changes in decay, level, or spectrum.

## Choosing FEM, Geometric Acoustics, or Statistical Reverb

FEM is strongest when wavelength-scale wave behavior matters: low-frequency
room modes, diffraction, complex boundary impedance, small cavities, and
coupled structural resonances. Image-source and ray methods are strongest when
specular path geometry dominates and the room is many wavelengths across. FDNs
and statistical tails are strongest when perceptual control, long decay, and
realtime efficiency matter more than reconstructing every boundary interaction.

The methods are complementary. A musically useful room model may use FEM for
low-frequency modes, image sources for early reflections, and an FDN for the
dense late field. The engineering goal is not to choose the most prestigious
solver; it is to choose the simplest combination that preserves the behavior
the listener and the experiment require.

## Further Reading

The bibliography includes Albert G. Prinn's 2023 review of FEM for room
acoustics, the room-IR method of Papadakis and Stavroulakis (2015), the
dispersion-reduced formulation of Okuzono and colleagues (2014), explicit
time-domain studies by Okuzono and colleagues (2015–2023), and work on
impedance boundaries, scalable wave simulation, and FEM-based room
qualification. These sources are useful companions because they show where
mesh dispersion, boundary modeling, solver choice, and validation dominate the
quality of an apparently straightforward simulation.
