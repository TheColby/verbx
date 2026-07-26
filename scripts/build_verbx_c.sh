#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build/native/verbx_c"
exe="${build_dir}/verbx-c"
cc_bin="${CC:-cc}"
cflags=("-std=c11" "-Wall" "-Wextra" "-Wpedantic")
ldflags=("-lm")
clean=false
run_doctor=false
print_path=false

usage() {
  cat <<'EOF'
Usage: scripts/build_verbx_c.sh [OPTIONS]

Build the native verbx-c executable.

Options:
  --clean       Remove the native build directory before compiling.
  --doctor      Run the freshly built verbx-c doctor command after compiling.
  --print-path  Print the expected executable path and exit.
  -h, --help    Show this help.

Environment:
  CC            C compiler command (default: cc)
  CFLAGS        Extra compiler flags appended after the defaults.
  LDFLAGS       Extra linker flags appended after the defaults.
EOF
}

for arg in "$@"; do
  case "${arg}" in
    --clean)
      clean=true
      ;;
    --doctor)
      run_doctor=true
      ;;
    --print-path)
      print_path=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf "Unknown option: %s\n\n" "${arg}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${print_path}" == true ]]; then
  printf "%s\n" "${exe}"
  exit 0
fi

if [[ "${clean}" == true ]]; then
  rm -rf "${build_dir}"
fi

mkdir -p "${build_dir}"

if [[ -n "${CFLAGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_cflags=(${CFLAGS})
  cflags+=("${extra_cflags[@]}")
fi
if [[ -n "${LDFLAGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_ldflags=(${LDFLAGS})
  ldflags+=("${extra_ldflags[@]}")
fi

"${cc_bin}" "${cflags[@]}" \
  -I "${repo_root}/native/verbx_c/include" \
  "${repo_root}/native/verbx_c/src/audio.c" \
  "${repo_root}/native/verbx_c/src/algo_reverb.c" \
  "${repo_root}/native/verbx_c/src/plugin_params.c" \
  "${repo_root}/native/verbx_c/src/plugin_realtime.c" \
  "${repo_root}/native/verbx_c/src/render.c" \
  "${repo_root}/native/verbx_c/src/wav_io.c" \
  "${repo_root}/native/verbx_c/src/main.c" \
  "${repo_root}/native/verbx_c/src/cli.c" \
  "${ldflags[@]}" \
  -o "${exe}"

printf "Built %s\n" "${exe}"

if [[ "${run_doctor}" == true ]]; then
  "${exe}" doctor
fi
