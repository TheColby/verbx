# verbx Schema Reference

JSON and CSV format specifications for batch manifests and automation files.

_Current as of v0.7.4._

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
| `shimmer_feedback` | number | `0.35` | `0.0` – `0.98` |
| `output_subtype` | string | `"auto"` | `"auto"`, `"float32"`, `"float64"`, `"pcm16"`, `"pcm24"`, `"pcm32"` |
| `target_sr` | integer | `null` | `>= 1` (Hz; `null` uses input sample rate) |
| `lowcut` | number | `null` | Hz; `null` = disabled |
| `highcut` | number | `null` | Hz; `null` = disabled |
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
| `fdn-tonal-correction-strength` | engine | 0.0 – 1.0 | Tonal correction amount |
| `ir-blend-alpha` | conv | 0.0 – 1.0 | IR morph blend (convolution engine) |

**Aliases:** Most targets accept common shorthand — e.g. `t60` → `rt60`, `gain` → `gain-db`, `room` → `room-size`, `blend-alpha` → `ir-blend-alpha`.

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
