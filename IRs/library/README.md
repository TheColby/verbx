# IR Library

Large folder-sorted synthetic IR collection generated with `verbx`.

## Structure

- `tiny/` (`0.5s`, `1s`, `2s`, `4s`)
- `short/` (`6s`, `8s`, `12s`, `16s`)
- `medium/` (`24s`, `32s`, `45s`, `60s`)
- `long/` (`75s`, `90s`, `120s`, `180s`)

Each bucket contains:

- `fdn/`
- `stochastic/`
- `modal/`
- `hybrid/`

Each IR file has a sidecar metadata file:

- `*.flac`
- `*.ir.meta.json`

## Manifest

Use `manifest.json` for deterministic indexing (mode, length, seed, config).

## Rebuild

```bash
uv run python scripts/generate_ir_library.py \
  --out IRs/library --sr 12000 --channels 2 --format flac --seeds-per-shape 1
```
