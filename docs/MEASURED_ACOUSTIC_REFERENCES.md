# Measured acoustic reference validation

Two original measured room impulse responses from the **RWTH Aachen AIR
Database v1.4** are checked into `tests/fixtures/measured_air`: booth and meeting
room, each mono, 48 kHz, 24-bit PCM, 32767 frames. They are unchanged WAV members
of the archive's nested `AIR_wav_files.zip`; the fixture manifest pins both file
hashes and the upstream archive hash. The fixture directory includes the original
MIT license and its RWTH copyright/attribution notice.

Primary source: [RWTH Aachen AIR Database](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/).
The authors describe real room measurements and identify the included MIT license.
Reference: Marco Jeub, Magnus Schäfer and Peter Vary, *A Binaural Room Impulse
Response Database for the Evaluation of Dereverberation Algorithms*, DSP 2009.
Acquired November 21, 2026 from the download URL recorded in the manifest.
The 193 MiB source archive is not retained or needed for tests.

Run the offline validation:

```sh
python -m pytest tests/test_measured_convolution_reference.py -q
```

The test independently computes full **direct time-domain convolution** with
`numpy.convolve`, then compares the production partitioned FFT engine and its
streaming file path at partition sizes 256 and 1024. It disables IR normalization,
resampling and dry mixing. Assertions cover original file integrity, output frame
count, finite samples, gain, sample alignment and full flushed tail, using absolute
error tolerance `1e-10` and relative tolerance `1e-9`. The 641-frame deterministic
dry probe crosses partition boundaries and ends in a partial block. DOUBLE output
keeps PCM output quantization out of the streaming comparison.

This establishes numerical convolution parity using actual measured room data.
The dry signal is synthetic, **not an anechoic recording**. These checks do not
establish physical image-source-model accuracy, room geometry/material inference,
FDN perceptual fidelity, or agreement with independent measured decay metadata.
The roadmap's measured anechoic plus convolution corpus remains incomplete until
licensed anechoic speech/music and independently specified acoustic expectations
are added. Analytic ISM path fixtures remain a separate validation tier.

To reproduce fixture acquisition, download the pinned archive URL, verify its
SHA-256, open `AIR_1_4/AIR_wav_files.zip`, and extract only the manifest's two
WAV members plus `AIR_1_4/license.txt`. Verify the individual SHA-256 values before
using them. No download or redistribution of additional corpus material is needed.
