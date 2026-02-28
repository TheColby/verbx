#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-IRs/generated_25_cli}"
COUNT="${2:-25}"
FORMAT="${3:-flac}"

mkdir -p "$OUT_DIR"

modes=(hybrid fdn stochastic modal)
lengths=(60 75 90 120 150 180 210 240 300 360)
matrices=(hadamard householder random_orthogonal)

for ((i=0; i<COUNT; i++)); do
  idx=$((i + 1))
  mode="${modes[$((i % ${#modes[@]}))]}"
  length="${lengths[$((i % ${#lengths[@]}))]}"
  matrix="${matrices[$((i % ${#matrices[@]}))]}"
  seed=$((700 + i))

  rt60=$(python3 - <<PY
length=float(${length})
idx=int(${i})
rt60=max(8.0,min(140.0,length*(0.45 + (idx % 7)*0.08)))
print(f"{rt60:.3f}")
PY
)

  if [[ "$FORMAT" == "aiff" ]]; then
    ext="aiff"
  else
    ext="$FORMAT"
  fi
  out="$OUT_DIR/ir_$(printf "%02d" "$idx")_${mode}_${length}s.${ext}"

  hatch run verbx ir gen "$out" \
    --mode "$mode" \
    --length "$length" \
    --sr 12000 \
    --channels 2 \
    --seed "$seed" \
    --rt60 "$rt60" \
    --damping "0.$((2 + (i % 6)))" \
    --diffusion "0.$((3 + (i % 6)))" \
    --density "1.$((i % 6))" \
    --fdn-matrix "$matrix" \
    --format "$FORMAT" \
    --normalize peak \
    --peak-dbfs -1.0
done

echo "Generated $COUNT IRs in $OUT_DIR (format=${FORMAT})"
