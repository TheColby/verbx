# Approximately one-hour RT60

The normal Python and Numba FDN paths no longer cap feedback gains at 0.995.
They retain a scalar roundoff guard immediately below unity and existing state
guards. Exact zero damping now bypasses both feedback low-pass and DC filtering;
exact zero modulation depth uses integer delays with no interpolation loss.
Positive values retain their previous processing behavior (including the minimum
one-sample modulation amplitude for positive depths). These zero-setting changes
also apply to cascade loops and the Python automation path.

This is a compatibility change: existing renders using exact zero controls will
sound different. Zero damping bypasses DC removal inside the feedback loop, so
remove unwanted input DC upstream. Changing damping automation through zero
switches the bypass and can cause a transient; do not assume a seamless transition.
Finite positive RT60 still supplies decay rather than unity feedback. A scalar
gain below unity alone is not a stability proof for all optional processors.

## Measurement

Run `.venv/bin/python scripts/validate_long_rt60.py` from the repository root.
The script renders a full 3600-second impulse response at 48 kHz in float64,
without playing or writing audio. It requires approximately 3 GB RAM plus runtime
overhead. The tested configuration is a 16-line static Hadamard FDN, 0.8 impulse,
RT60 3600 seconds, no all-pass stages, zero pre-delay control, zero damping,
zero modulation depth, modulation rate 0.1 Hz, and 4096-sample blocks. Other
configuration fields use the dataclass defaults; the script prints them all.

On September 15, 2026, the Numba CPU path produced finite samples throughout,
with absolute peak 0.0125. Ordinary least-squares fits to non-overlapping
one-second mean-square levels yielded:

| Fit interval (s) | Apparent RT60 (s) | R-squared |
| --- | ---: | ---: |
| 10-120 | 3598.363 | 0.998588 |
| 10-1200 | 3600.012 | 0.999991 |
| 1200-2400 | 3599.985 | 0.999992 |
| 2400-3600 | 3600.003 | 0.999991 |
| 10-3600 | 3600.003 | 0.999999 |

The measured source SHA-256 was
`6597a5653e3d77d97ddcd3980763df6554ee988c8b38572d62f86422e38cdc3e`;
subsequent source formatting does not affect the DSP changes measured here.
The script emits the source hash on every new run.

This is a production-kernel measurement, not an end-to-end CLI measurement or a
listening study. It establishes approximately one-hour broadband decay for the
stated neutral configuration, not for arbitrary filters, modulation, nonlinear
processing, time-varying matrices, or coupled cascades. A finite one-hour record
also does not establish infinite-horizon stability for all configurations.

For corresponding CLI rendering, use `--engine algo --rt60 3600 --damping 0
--mod-depth-ms 0 --fdn-lines 16 --fdn-matrix hadamard --allpass-stages 0` and
allow enough output tail (`--tail-limit 3600`). Use `--wet 1 --dry 0`, disable
normalization and limiting for measurement, and set the sample rate and output
subtype explicitly. The default tail budget must not be mistaken for RT60.
RF64/W64 long-file support is independent of this feedback calibration.

The JAES draft's old Table 5 and gain-saturation experiments describe the previous
implementation. They must be identified as historical, and the source audit and
paper experiments rerun before describing them as measurements of this revision.
