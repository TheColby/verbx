# verbx Schema Reference

JSON and CSV format specifications for batch manifests and automation files.

_Current as of v0.7.7._

Notes for `v0.7.7`:

- Batch manifests map to the offline `verbx render` surface.
- Realtime device selection (`verbx realtime`, `--input-device`, `--output-device`)
  is CLI/session state and is not represented in these schemas.
- Render reports are produced internally by a typed `RenderReport` object, but
  the written JSON remains backward-compatible dictionary-style output.

---

## Batch Manifest (`batch.json` / `batch.jsonl`)

Used with `verbx batch render --manifest <file>`.

### JSON format

```json
{
  "version": "0.5",
  "jobs": [
    {
      "infile": "path/to/dry.wav",
      "outfile": "path/to/wet.wav",
      "options": {
        "engine": "algo",
        "rt60": 2.5,
        "wet": 0.7,
        "dry": 0.3,
        "pre_delay_ms": 20,
        "fdn_lines": 16,
        "target_sr": 192000,
        "output_subtype": "float32"
      }
    }
  ]
}
```

### JSONL format (newline-delimited)

One job object per line (no wrapping object, no `"jobs"` key):

```jsonl
{"infile": "dry/kick.wav", "outfile": "wet/kick.wav", "options": {"engine": "algo", "rt60": 0.4}}
{"infile": "dry/snare.wav", "outfile": "wet/snare.wav", "options": {"engine": "algo", "rt60": 0.8}}
```

### Job object fields

| Field | Type | Required | Description |
|---|---|---|---|
| `infile` | string | yes | Path to the dry input audio file |
| `outfile` | string | yes | Path for the processed output file |
| `options` | object | no | Any `verbx render` option as a key/value pair |

`options` keys map directly to `RenderConfig` fields. All keys are optional — unset fields use defaults. Boolean flags use JSON booleans (`true`/`false`).

**Common options:**

| Option | Type | Default | Range |
|---|---|---|---|
| `engine` | string | `"auto"` | `"algo"`, `"conv"`, `"auto"` |
| `rt60` | number | `60.0` | `0.1` – `3600.0` |
| `wet` | number | `0.8` | `0.0` – `1.0` |
| `dry` | number | `0.2` | `0.0` – `1.0` |
| `pre_delay_ms` | number | `20.0` | `0` – `500` |
| `fdn_lines` | integer | `8` | `1` – `64` |
| `fdn_matrix` | string | `"hadamard"` | `"hadamard"`, `"householder"`, `"random_orthogonal"`, `"circulant"`, `"elliptic"`, `"tv_unitary"`, `"graph"`, `"sdn_hybrid"` |
| `shimmer` | boolean | `false` | — |
| `shimmer_semitones` | number | `12` | `-24` – `24` |
| `shimmer_mix` | number | `0.25` | `0.0` – `1.0` |
| `shimmer_feedback` | number | `0.35` | `0.0` – `0.98` (safe), up to `1.25` with `unsafe_self_oscillate=true` |
| `unsafe_self_oscillate` | boolean | `false` | Enables unsafe above-unity feedback path in algorithmic mode |
| `unsafe_loop_gain` | number | `1.02` | `> 1.0` for intentional self-oscillation (`<= 1.25`) |
| `auto_fit` | string | `"none"` | `"none"`, `"speech"`, `"music"`, `"drums"`, `"ambient"` |
| `fdn_matrix_morph_to` | string/null | `null` | Optional matrix morph target family |
| `fdn_matrix_morph_seconds` | number | `0.0` | `>= 0.0` |
| `tail_stop_threshold_db` | number | `-120.0` | `-240.0` – `0.0` |
| `tail_stop_hold_ms` | number | `10.0` | `>= 0.0` |
| `tail_stop_metric` | string | `"peak"` | `"peak"`, `"rms"` |
| `algo_stream` | boolean | `false` | Enable algorithmic proxy streaming path |
| `algo_proxy_ir_max_seconds` | number | `120.0` | `> 0.0` |
| `algo_gpu_proxy` | boolean | `false` | Use CUDA convolution backend for proxy streaming |
| `output_subtype` | string | `"auto"` | `"auto"`, `"float32"`, `"float64"`, `"pcm16"`, `"pcm24"`, `"pcm32"` |
| `output_container` | string | `"auto"` | `"auto"`, `"wav"`, `"w64"`, `"rf64"` |
| `target_sr` | integer | `null` | `>= 1` (Hz; `null` uses input sample rate) |
| `lowcut` | number | `null` | Hz; `null` = disabled |
| `highcut` | number | `null` | Hz; `null` = disabled |
| `shimmer_spatial` | boolean | `false` | Enable multichannel shimmer decorrelation |
| `shimmer_spread_cents` | number | `8.0` | `>= 0.0` |
| `shimmer_decorrelation_ms` | number | `1.5` | `>= 0.0` |
| `er_geometry` | boolean | `false` | Enable first-order image-source early reflections |
| `er_room_dims_m` | array[3] | `[10.0, 7.0, 3.0]` | Positive room dimensions (meters) |
| `er_source_pos_m` | array[3] | `[2.0, 2.0, 1.5]` | Source position (meters) |
| `er_listener_pos_m` | array[3] | `[5.0, 3.5, 1.5]` | Listener position (meters) |
| `er_absorption` | number | `0.35` | `0.0` – `0.99` |
| `er_material` | string | `"studio"` | Material preset name |
| `repeat` | integer | `1` | `1` – `100` |

Generate a template with: `verbx batch template`

---

## Automation File (`.json` or `.csv`)

Used with `verbx render --automation-file <file>`.

### JSON format

```json
{
  "mode": "block",
  "block_ms": 20.0,
  "lanes": [
    {
      "target": "wet",
      "type": "breakpoints",
      "interp": "linear",
      "points": [
        {"time": 0.0, "value": 0.2},
        {"time": 5.0, "value": 0.9},
        {"time": 10.0, "value": 0.5}
      ]
    },
    {
      "target": "rt60",
      "type": "lfo",
      "shape": "sine",
      "rate_hz": 0.1,
      "depth": 0.4,
      "center": 2.0
    }
  ]
}
```

### Top-level fields

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | string | `"block"` | `"block"` (control-rate) or `"sample"` (sample-rate) |
| `block_ms` | number | `20.0` | Control block size in ms (when `mode = "block"`) |
| `lanes` | array | required | List of automation lane objects |

### Lane types

#### `breakpoints` — interpolated envelope

```json
{
  "target": "wet",
  "type": "breakpoints",
  "interp": "linear",
  "combine": "replace",
  "points": [
    {"time": 0.0, "value": 0.3},
    {"time": 4.0, "value": 0.9}
  ]
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `target` | string | required | Automation target name (see below) |
| `type` | string | required | `"breakpoints"` |
| `interp` | string | `"linear"` | Interpolation mode (see below) |
| `combine` | string | `"replace"` | How to merge with prior lanes for same target |
| `points` | array | required | Array of `{"time": seconds, "value": float}` |

**Interpolation modes:** `linear`, `hold`, `step`, `smooth`, `smoothstep`, `exp` / `exponential`

**Combine modes:** `replace`, `add`, `multiply`

#### `lfo` — low-frequency oscillator

```json
{
  "target": "damping",
  "type": "lfo",
  "shape": "sine",
  "rate_hz": 0.25,
  "depth": 0.3,
  "center": 0.5,
  "phase_deg": 0.0
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `target` | string | required | Automation target name |
| `type` | string | required | `"lfo"` |
| `shape` | string | `"sine"` | `"sine"`, `"triangle"`, `"square"`, `"sawtooth"`, `"random"` |
| `rate_hz` | number | `1.0` | Oscillation frequency in Hz |
| `depth` | number | `0.5` | Peak deviation from center (±depth) |
| `center` | number | `0.5` | Center value |
| `phase_deg` | number | `0.0` | Starting phase in degrees |

#### `segments` — piecewise constant/linear segments

```json
{
  "target": "gain-db",
  "type": "segments",
  "segments": [
    {"start": 0.0, "end": 2.0, "value": -6.0, "interp": "hold"},
    {"start": 2.0, "end": 5.0, "value": 0.0,  "interp": "linear"},
    {"start": 5.0, "end": 8.0, "value": -12.0, "interp": "smooth"}
  ]
}
```

### CSV format

```csv
target,time_s,value,interp
wet,0.0,0.2,linear
wet,5.0,0.8,linear
wet,10.0,0.5,linear
rt60,0.0,1.0,smooth
rt60,8.0,4.0,smooth
```

Required columns: `target`, `time_s`, `value`. Optional: `interp`.

### Automation targets

| Target | Domain | Range | Description |
|---|---|---|---|
| `wet` | post | 0.0 – 1.0 | Wet mix level |
| `dry` | post | 0.0 – 1.0 | Dry mix level |
| `gain-db` | post | -48.0 – 24.0 | Output gain in dB |
| `rt60` | engine | 0.1 – 3600.0 | Reverberation time in seconds |
| `damping` | engine | 0.0 – 1.0 | High-frequency damping |
| `room-size` | engine | 0.25 – 4.0 | Room size scalar |
| `room-size-macro` | engine | -1.0 – 1.0 | Room size macro (normalized) |
| `clarity-macro` | engine | -1.0 – 1.0 | Clarity macro |
| `warmth-macro` | engine | -1.0 – 1.0 | Warmth macro |
| `envelopment-macro` | engine | -1.0 – 1.0 | Envelopment macro |
| `fdn-rt60-tilt` | engine | -1.0 – 1.0 | FDN RT60 spectral tilt |
| `fdn-rt60-low` | engine | 0.1 – 3600.0 | Low-band RT60 target |
| `fdn-rt60-mid` | engine | 0.1 – 3600.0 | Mid-band RT60 target |
| `fdn-rt60-high` | engine | 0.1 – 3600.0 | High-band RT60 target |
| `fdn-xover-low-hz` | engine | 20.0 – 20000.0 | Low/mid crossover |
| `fdn-xover-high-hz` | engine | 20.0 – 20000.0 | Mid/high crossover |
| `fdn-tonal-correction-strength` | engine | 0.0 – 1.0 | Tonal correction amount |
| `ir-blend-alpha` | conv | 0.0 – 1.0 | IR morph blend (convolution engine) |

**Aliases:** Most targets accept common shorthand — e.g. `t60` → `rt60`, `gain` → `gain-db`, `room` → `room-size`, `blend-alpha` → `ir-blend-alpha`.

---

## Analysis Output (`verbx analyze --json-out`)

JSON produced by `verbx analyze --json-out <file>`.

```json
{
  "sample_rate": 48000,
  "channels": 2,
  "metrics": {
    "duration": 5.032,
    "rms": 0.1234,
    "peak": 0.876,
    "...": "..."
  }
}
```

All `metrics` values are floats unless `--room` is also passed, in which case
the room-estimate string fields (`room_class`, `room_confidence`,
`room_estimation_method`) appear as strings.

---

## Room Size Estimate (`--room` flag on `verbx analyze` / `verbx compare`)

When `--room` is passed the analysis output includes the following additional
keys (all prefixed `room_`):

### Numeric fields (float)

| Key | Unit | Description |
|---|---|---|
| `room_rt60_s` | s | Best RT60 estimate used as sizing input (mid-band preferred) |
| `room_rt60_low_s` | s | EDR RT60 estimate for the low band (20–250 Hz) |
| `room_rt60_mid_s` | s | EDR RT60 estimate for the mid band (250–2 000 Hz) |
| `room_rt60_high_s` | s | EDR RT60 estimate for the high band (2 000 Hz+) |
| `room_volume_m3` | m³ | Primary volume estimate (Sabine/Eyring blend) |
| `room_volume_m3_sabine` | m³ | Sabine-only volume estimate |
| `room_volume_m3_eyring` | m³ | Eyring-only volume estimate |
| `room_volume_m3_low` | m³ | Conservative lower bound (−30 %) |
| `room_volume_m3_high` | m³ | Conservative upper bound (+30 %) |
| `room_dim_width_m` | m | Estimated room width (shortest horizontal dimension) |
| `room_dim_depth_m` | m | Estimated room depth (1.25 × width) |
| `room_dim_height_m` | m | Estimated room height (0.62 × width) |
| `room_surface_area_m2` | m² | Total surface area of the estimated rectangular box |
| `room_mean_absorption` | — | Estimated mean absorption coefficient [0, 1] |
| `room_critical_distance_m` | m | Schroeder critical distance (direct = reverberant field) |
| `room_confidence_score` | — | Numeric confidence rating [0, 1] |

### String fields

| Key | Values | Description |
|---|---|---|
| `room_class` | `"closet"` `"small"` `"medium"` `"large"` `"very_large"` `"cathedral"` `"unknown"` | Qualitative room size label |
| `room_estimation_method` | `"sabine"` `"eyring"` `"none"` | Formula used for primary volume |
| `room_confidence` | `"high"` `"medium"` `"low"` | Qualitative confidence rating |

### Room class thresholds

| Class | Approx. volume | Approx. RT60 |
|---|---|---|
| `closet` | < 8 m³ | < 0.12 s |
| `small` | 8–50 m³ | 0.12–0.35 s |
| `medium` | 50–250 m³ | 0.35–0.80 s |
| `large` | 250–1 500 m³ | 0.80–2.0 s |
| `very_large` | 1 500–10 000 m³ | 2.0–5.0 s |
| `cathedral` | > 10 000 m³ | > 5.0 s |

### Example

```bash
verbx analyze my_reverb.wav --room --json-out room_report.json
```

```json
{
  "sample_rate": 48000,
  "channels": 2,
  "metrics": {
    "room_rt60_s": 1.42,
    "room_volume_m3": 320.5,
    "room_volume_m3_low": 224.4,
    "room_volume_m3_high": 416.7,
    "room_dim_width_m": 7.38,
    "room_dim_depth_m": 9.22,
    "room_dim_height_m": 4.58,
    "room_surface_area_m2": 271.2,
    "room_mean_absorption": 0.175,
    "room_critical_distance_m": 1.94,
    "room_class": "large",
    "room_estimation_method": "sabine",
    "room_confidence": "high",
    "room_confidence_score": 0.8,
    "...": "..."
  }
}
```

---

## Compare Report (`verbx compare --json-out`)

JSON produced by `verbx compare FILE_A FILE_B --json-out <file>`.

```json
{
  "schema": "compare-report-v1",
  "file_a": "/path/to/dry.wav",
  "file_b": "/path/to/wet.wav",
  "sample_rate_a": 48000,
  "sample_rate_b": 48000,
  "channels_a": 2,
  "channels_b": 2,
  "metrics_a": { "rms": 0.12, "..." : "..." },
  "metrics_b": { "rms": 0.18, "...": "..." },
  "delta": { "rms": 0.06, "...": "..." }
}
```

`delta` only includes keys whose values are numeric (`float`) in both files —
string-valued fields like `room_class` are present in `metrics_a`/`metrics_b`
but omitted from `delta`.

---

## Inline automation (`--automation-point`)

For simple one-off control points without a file:

```
verbx render in.wav out.wav \
  --automation-point wet:0.0:0.2:linear \
  --automation-point wet:5.0:0.9:linear \
  --automation-point rt60:0.0:1.5
```

Format: `target:time_s:value[:interp]`

- `target` — automation target name
- `time_s` — time in seconds (float)
- `value` — parameter value (float)
- `interp` — interpolation mode (optional, default `linear`)
