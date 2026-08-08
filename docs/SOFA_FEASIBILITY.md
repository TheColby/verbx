# SOFA Import and Extraction

_Reviewed for verbx v0.9.9 on August 5, 2026._

The Spatially Oriented Format for Acoustics (SOFA) stores measured spatial-audio data in HDF5 containers governed by named conventions. A SOFA file is not merely a multichannel impulse response with a different extension. Its arrays may describe many source positions, receivers, listeners, and emitters, and the coordinate metadata determines what those samples mean. `verbx` therefore treats SOFA as an explicit import step rather than accepting it silently wherever a WAV impulse response is expected.

## Supported Workflow

The current command-line surface is deliberately narrow. It lets an engineer inspect a container, select one FIR measurement, and export that selection as an ordinary WAV matrix for the existing convolution tools. The two commands are:

```bash
verbx ir sofa-info measurements.sofa
verbx ir sofa-extract measurements.sofa selected_ir.wav
```

`sofa-info` reports the convention and version, sample rate in hertz, `Data/IR` shape and dimension labels, and the available source, listener, receiver, and emitter position shapes. Inspect this report before extraction. A plausible channel count does not establish receiver order, coordinate units, or intended loudspeaker routing.

`sofa-extract` writes three related artifacts: the extracted WAV matrix, an IR metadata sidecar, and extraction metadata recording the selection and conversion choices. The WAV is then suitable for the same explicit matrix-routing workflow used by other verbx impulse responses.

## Selecting a Measurement

SOFA datasets often contain several measurements of the same acoustic system at different positions. The default extracts measurement zero and, for a rank-four array, emitter zero. Select other entries by index:

```bash
verbx ir sofa-extract hall.sofa hall_m012_e01.wav \
  --measurement-index 12 \
  --emitter-index 1
```

In strict mode, verbx accepts `Data/IR` rank three with dimensions `(M, R, N)` or rank four with dimensions `(M, R, E, N)`. Here, `M` is measurement index, `R` is receiver channel, `E` is emitter index, and `N` is sample index. Best-effort mode, which is the default, can accommodate less regular containers, but its output still requires listening and routing verification.

## Sample Rate and Normalization

An extracted IR retains the source sample rate unless `--target-sr` requests deterministic resampling. Normalization is explicit because changing IR gain changes the gain of every later convolution. The accepted modes are `none`, `peak`, and `rms`; the default is `peak`.

```bash
verbx ir sofa-extract room.sofa room_48k.wav \
  --target-sr 48000 \
  --normalize peak \
  --strict
```

Use `none` when calibrated level relationships must survive extraction. Use `peak` for a convenient bounded IR when absolute calibration is not part of the dataset contract. Use `rms` only when equalizing average IR energy is the intended comparison. Record the choice alongside a production master rather than relying on a filename to preserve it.

## Routing the Exported Matrix

The receiver dimension becomes the channel dimension of the exported WAV. That operation preserves samples, but it does not invent a loudspeaker layout. Before rendering program material, make a short identification render or inspect isolated impulses so that every receiver channel is mapped intentionally through `--ir-matrix-layout` or `--ir-route-map`.

A useful handoff record includes the source file hash, SOFA convention, measurement and emitter indices, sample-rate conversion, normalization mode, receiver order, and intended output layout. The extraction metadata records the computational choices; the engineer must still document the production meaning of the channels.

## Dependency and Failure Modes

SOFA support uses the optional `h5py` dependency so that the baseline verbx installation does not require an HDF5 stack. When the dependency is unavailable, the commands stop with an installation message rather than interpreting the file partially.

The most important failures are semantic rather than syntactic. A container can open successfully while using a convention, coordinate system, orientation, or receiver order inappropriate for the intended render. Strict mode catches unsupported array rank, but it cannot decide whether a chosen viewpoint is musically or geometrically correct. Treat the info report, metadata sidecars, channel-identification render, and listening test as one validation sequence.

## Current Boundary

The shipped feature extracts indexed FIR data. It does not provide coordinate queries such as nearest azimuth and elevation, automatic Ambisonic decoding, arbitrary convention conversion, or direct object-audio authoring. Those operations require convention-specific decisions that should remain visible rather than being guessed during a render.

This boundary keeps SOFA interoperability reproducible: first select and document a view, then convert it to an explicit matrix, and finally route that matrix through the established convolution engine.

## References

The formal and project references below describe the container family and its conventions. Consult the convention named by a dataset in addition to the general standard.

- AES69, *AES Standard for File Exchange: Spatial Acoustic Data File Format*: <https://www.aes.org/publications/standards/search.cfm?docID=99>
- SOFA conventions project: <https://www.sofaconventions.org/>
