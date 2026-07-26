# DXF Trace MVP

`verbx ir trace` is the first bounded physical-acoustics prototype for turning
simple CAD room outlines into convolution-ready impulse responses.

The command is experimental. It is designed to make the geometry-to-IR workflow
testable before verbx attempts robust arbitrary CAD cleanup or architectural
acoustics accuracy.

## Command

```bash
verbx ir trace room.dxf room_ir.wav \
  --source 2,3,1.5 \
  --listener 6,4,1.5 \
  --height 3 \
  --material studio \
  --rays 50000 \
  --length 4.0 \
  --target-sr 48000 \
  --json-out room_trace.json
```

Useful material names include `studio`, `hall`, `dead`, `anechoic`, `stone`,
`concrete`, `drywall`, `glass`, `wood`, `carpet`, `curtain`, `plaster`,
`brick`, `audience`, `acoustic-panel`, `diffuser`, `ceiling-tile`,
`vinyl-floor`, `water`, and `open-air`. Each preset contributes octave-band
absorption coefficients plus a scattering value to the `trace-report-v1`
material payload; the current renderer still collapses those bands to a
broadband value for first-order reflection and late-tail synthesis.

Use the generated IR with convolution render:

```bash
verbx render dry.wav in_room.wav --engine conv --ir room_ir.wav
```

## Supported Geometry

Supported in the MVP:

- ASCII DXF files.
- `LINE` entities.
- `LWPOLYLINE` entities.
- 2D room outlines with an explicit `--height`.
- DXF `$INSUNITS` values for inches, feet, millimeters, centimeters, and meters.
- Axis-aligned room-like bounding boxes derived from the detected outline.

Deferred:

- arbitrary CAD cleanup
- arcs, splines, blocks, inserts, meshes, and 3D solids
- non-rectangular acoustic diffraction
- material assignment from CAD layers
- validated architectural-acoustics simulation

## Output

`verbx ir trace` writes:

- an IR WAV with stereo output
- a normal `.ir.meta.json` sidecar unless `--silent` is used
- a `trace-report-v1` JSON report

The trace report contains:

- geometry summary
- material/absorption/scattering summary, including octave-band coefficients
- source and listener coordinates
- direct-path timing
- first-order reflection timings
- ray budget and seed
- estimated RT60
- IR metrics
- warnings

## Acoustic Model

The current model is deliberately narrow:

- direct path from source to listener
- first-order wall/floor/ceiling image-source reflections
- stochastic late tail shaped by room volume, surface area, absorption, and ray
  budget

This is useful for deterministic workflow testing and early UX, but it is not
yet a full ray tracer or architectural acoustic validator.

## Success Criteria For The Prototype

- The command fails fast on unsupported or empty geometry.
- The same DXF/options produce deterministic IR and report outputs.
- The output IR can be used by `verbx render --engine conv --ir`.
- The report gives enough geometry/reflection detail to debug CAD ingestion.
- Future geometry features update `docs/NATIVE_PARITY.md`,
  `docs/ROADMAP.md`, and tests before claiming broader scope.
