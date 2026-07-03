#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/install_verbx_c.sh [OPTIONS]

Build and install the native verbx-c executable.

Options:
  --prefix PATH       Install prefix (default: ~/.local)
  --bin-dir PATH      Override binary install directory (default: <prefix>/bin)
  --man-dir PATH      Override man page root (default: <prefix>/share/man)
  --skip-build        Install the existing build/native/verbx_c/verbx-c binary.
  --no-man            Skip installing the verbx-c man page.
  --doctor            Run the installed verbx-c doctor command after install.
  -h, --help          Show this help.

Examples:
  scripts/install_verbx_c.sh --prefix "$HOME/.local"
  scripts/install_verbx_c.sh --prefix /usr/local --doctor
  scripts/install_verbx_c.sh --skip-build --no-man --bin-dir /tmp/bin
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="${HOME}/.local"
bin_dir=""
man_dir=""
skip_build=false
with_man=true
run_doctor=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      prefix="$2"
      shift 2
      ;;
    --bin-dir)
      bin_dir="$2"
      shift 2
      ;;
    --man-dir)
      man_dir="$2"
      shift 2
      ;;
    --skip-build)
      skip_build=true
      shift
      ;;
    --no-man)
      with_man=false
      shift
      ;;
    --doctor)
      run_doctor=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf "Unknown option: %s\n\n" "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${bin_dir}" ]]; then
  bin_dir="${prefix}/bin"
fi
if [[ -z "${man_dir}" ]]; then
  man_dir="${prefix}/share/man"
fi

exe_path="$("${repo_root}/scripts/build_verbx_c.sh" --print-path)"
if [[ "${skip_build}" != true ]]; then
  "${repo_root}/scripts/build_verbx_c.sh"
fi
if [[ ! -x "${exe_path}" ]]; then
  printf "Native executable not found: %s\n" "${exe_path}" >&2
  printf "Run scripts/build_verbx_c.sh first or omit --skip-build.\n" >&2
  exit 1
fi

mkdir -p "${bin_dir}"
cp "${exe_path}" "${bin_dir}/verbx-c"
chmod 755 "${bin_dir}/verbx-c"
printf "Installed %s\n" "${bin_dir}/verbx-c"

if [[ "${with_man}" == true ]]; then
  src_man="${repo_root}/man/man1/verbx-c.1"
  dest_man1="${man_dir}/man1"
  if [[ ! -f "${src_man}" ]]; then
    printf "Native man page not found: %s\n" "${src_man}" >&2
    exit 1
  fi
  mkdir -p "${dest_man1}"
  cp "${src_man}" "${dest_man1}/verbx-c.1"
  chmod 644 "${dest_man1}/verbx-c.1"
  printf "Installed %s\n" "${dest_man1}/verbx-c.1"
fi

if [[ "${run_doctor}" == true ]]; then
  "${bin_dir}/verbx-c" doctor
fi

printf "Native installation complete.\n"
printf "Try:\n"
printf "  %s/verbx-c --help\n" "${bin_dir}"
if [[ "${with_man}" == true ]]; then
  printf "  man -M %s verbx-c\n" "${man_dir}"
fi
