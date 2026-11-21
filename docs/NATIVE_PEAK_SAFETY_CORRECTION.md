# Native parity peak-safety comparison correction

The parity harness previously mapped native `--peak-safe` to Python target
peak normalization. Target normalization can amplify quiet signals; native
peak safety attenuates only when the completed output exceeds its ceiling.
The stereo fixture therefore compared differently normalized signals.

The corrected Python reference renders an unnormalized float64 intermediate,
applies one linked gain of `min(1, ceiling / peak)` after tail completion, and
then encodes the requested output subtype. It matches native ceiling clamping
at −120 to 0 dBFS. No DSP implementation or acceptance tolerance was changed.
The intermediate prevents clipping before safety attenuation when the final
format is PCM. Tests cover both below-ceiling preservation and above-ceiling
attenuation, including shared stereo gain and final PCM16 encoding.

Reproduction:

```sh
python scripts/compare_native_render_parity.py --build-native --strict-structural --report native-parity.json
```

Measured against the native renderer with the checked-in two-fixture contract:

| Scenario | Duration delta, ms | Peak delta | RMS delta, dB |
| --- | ---: | ---: | ---: |
| Mono, before and after | 105.94 | 0.061654 | 13.06 |
| Stereo, previous target normalization | 66.21 | 0.778748 | 12.62 |
| Stereo, corrected ceiling-only safety | 55.78 | 0.000000 | 5.31 |

Structural checks pass. Aggregate metric checks still fail. Matching stereo
peaks reflects the fixture's shared dry impulse and does not establish FDN DSP
parity. The remaining RMS and tail-duration discrepancies remain visible and
require separate DSP investigation. Mono has peak safety disabled and is
unchanged. Stereo duration also changes because upward normalization previously
preceded Python tail finalization, affecting the threshold crossing.
