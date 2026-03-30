#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build/native/verbx_c"
mkdir -p "${build_dir}"

cc -std=c11 -Wall -Wextra -Wpedantic \
  -I "${repo_root}/native/verbx_c/include" \
  "${repo_root}/native/verbx_c/src/audio.c" \
  "${repo_root}/native/verbx_c/src/algo_reverb.c" \
  "${repo_root}/native/verbx_c/src/render.c" \
  "${repo_root}/native/verbx_c/src/wav_io.c" \
  "${repo_root}/native/verbx_c/src/main.c" \
  "${repo_root}/native/verbx_c/src/cli.c" \
  -lm \
  -o "${build_dir}/verbx-c"

printf "Built %s\n" "${build_dir}/verbx-c"
