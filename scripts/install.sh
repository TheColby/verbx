#!/usr/bin/env bash
set -euo pipefail

usage() {
  local invocation
  invocation="$(basename "$0")"
  if [[ "${invocation}" == "install.sh" ]]; then
    invocation="./install.sh"
  else
    invocation="scripts/install.sh"
  fi

  cat <<EOF
verbx complete install helper

Installs the Python CLI, runtime extras, native CLI, man pages, and JUCE plug-ins.

Usage:
  ${invocation} [options]

General options:
  --prefix PATH       CLI/native prefix (default: ~/.local)
  --python PATH       Python executable (default: python3)
  --editable          Install Python package in editable mode (default)
  --wheel             Install Python package as a regular package
  --minimal-python    Omit realtime and SOFA Python extras
  --dev               Include Python development extras
  --skip-python-install
                      Skip the Python package
  --skip-native       Skip the native verbx-c executable
  --no-man            Skip all man pages
  --man-dir PATH      Override man-page root (default: <prefix>/share/man)

Plug-in options:
  --skip-plugins      Skip JUCE plug-in build and installation
  --skip-plugin-build Install existing Release artifacts without rebuilding
  --plugin-build-dir PATH
                      Plug-in build directory
  --juce-source PATH  Use a local JUCE source checkout
  --juce-version VER  JUCE tag downloaded when needed (default: 8.0.6)
  --no-juce-download  Require local JUCE source or an installed JUCE package
  --au-dir PATH       Audio Unit destination (macOS default: ~/Library/Audio/Plug-Ins/Components)
  --vst3-dir PATH     VST3 destination (macOS: ~/Library/Audio/Plug-Ins/VST3; Linux: ~/.vst3)
  --app-dir PATH      Standalone app destination (macOS default: ~/Applications)
  --no-standalone     Do not install the standalone plug-in host
  --reset-plugin-cache
                      Back up and clear the macOS Audio Unit cache after install
  --macos-architectures LIST
                      CMake architecture list (default: arm64;x86_64)
  --macos-deployment-target VER
                      Oldest supported macOS version (default: 12.0)
  --jobs N            Parallel build jobs (default: detected CPU count)
  --dry-run           Print the resolved installation plan and exit
  -h, --help          Show this help and exit

Examples:
  ${invocation}
  ${invocation} --juce-source /path/to/JUCE
  ${invocation} --wheel --prefix "\$HOME/.local"
  ${invocation} --skip-plugin-build --skip-python-install --skip-native --no-man
  ${invocation} --skip-plugins --dev
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_value() {
  [[ $# -ge 2 && -n "${2:-}" ]] || die "$1 requires a value"
}

install_bundle() {
  local source="$1"
  local destination_dir="$2"
  local destination temporary
  destination="${destination_dir}/$(basename "$source")"
  temporary="${destination}.verbx-install.$$"

  mkdir -p "$destination_dir"
  rm -rf "$temporary"
  cp -R "$source" "$temporary"
  rm -rf "$destination"
  mv "$temporary" "$destination"
  printf 'Installed %s\n' "$destination"
}

sign_and_verify_bundle() {
  local bundle="$1"
  [[ "$OS_NAME" == Darwin ]] || return 0
  command -v codesign >/dev/null 2>&1 || die "codesign is required on macOS"
  codesign --force --deep --sign - "$bundle"
  codesign --verify --deep --strict --verbose=2 "$bundle"
}

refresh_macos_plugin_registry() {
  local au_bundle="$1"
  local vst3_bundle="$2"
  [[ "$OS_NAME" == Darwin ]] || return 0

  touch "$au_bundle" "$vst3_bundle"
  killall AudioComponentRegistrar >/dev/null 2>&1 || true

  if [[ "$RESET_PLUGIN_CACHE" -eq 1 ]]; then
    local backup_dir cache_file
    backup_dir="${PREFIX}/share/verbx/cache-backups/au-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    for cache_file in \
      "${HOME}/Library/Caches/AudioUnitCache/com.apple.audiounits.cache" \
      "${HOME}/Library/Caches/AudioUnitCache/com.apple.audiounits.sandboxed.cache" \
      "${HOME}/Library/Preferences/com.apple.audio.AudioComponentCache.plist"
    do
      if [[ -f "$cache_file" ]]; then
        cp "$cache_file" "$backup_dir/"
        rm -f "$cache_file"
      fi
    done
    killall -9 AudioComponentRegistrar >/dev/null 2>&1 || true
    printf 'Backed up and cleared the Audio Unit cache at %s\n' "$backup_dir"
  fi
}

cpu_count() {
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.logicalcpu 2>/dev/null || true
  elif command -v nproc >/dev/null 2>&1; then
    nproc
  fi
}

PREFIX="${HOME}/.local"
PYTHON_BIN="python3"
INSTALL_MODE="editable"
WITH_RUNTIME_EXTRAS=1
WITH_DEV=0
INSTALL_PYTHON=1
WITH_NATIVE=1
WITH_MAN=1
MAN_DIR=""
WITH_PLUGINS=1
BUILD_PLUGINS=1
PLUGIN_BUILD_DIR=""
JUCE_SOURCE="${JUCE_SOURCE:-${JUCE_DIR:-}}"
JUCE_VERSION="8.0.6"
ALLOW_JUCE_DOWNLOAD=1
AU_DIR=""
VST3_DIR=""
APP_DIR=""
WITH_STANDALONE=1
RESET_PLUGIN_CACHE=0
MACOS_ARCHITECTURES="${VERBX_MACOS_ARCHITECTURES:-arm64;x86_64}"
MACOS_DEPLOYMENT_TARGET="${VERBX_MACOS_DEPLOYMENT_TARGET:-12.0}"
JOBS="$(cpu_count)"
JOBS="${JOBS:-4}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix|--python|--man-dir|--plugin-build-dir|--juce-source|--juce-version|--au-dir|--vst3-dir|--app-dir|--macos-architectures|--macos-deployment-target|--jobs)
      require_value "$1" "${2:-}"
      case "$1" in
        --prefix) PREFIX="$2" ;;
        --python) PYTHON_BIN="$2" ;;
        --man-dir) MAN_DIR="$2" ;;
        --plugin-build-dir) PLUGIN_BUILD_DIR="$2" ;;
        --juce-source) JUCE_SOURCE="$2" ;;
        --juce-version) JUCE_VERSION="$2" ;;
        --au-dir) AU_DIR="$2" ;;
        --vst3-dir) VST3_DIR="$2" ;;
        --app-dir) APP_DIR="$2" ;;
        --macos-architectures) MACOS_ARCHITECTURES="$2" ;;
        --macos-deployment-target) MACOS_DEPLOYMENT_TARGET="$2" ;;
        --jobs) JOBS="$2" ;;
      esac
      shift 2
      ;;
    --editable) INSTALL_MODE="editable"; shift ;;
    --wheel) INSTALL_MODE="wheel"; shift ;;
    --minimal-python) WITH_RUNTIME_EXTRAS=0; shift ;;
    --dev) WITH_DEV=1; shift ;;
    --skip-python-install) INSTALL_PYTHON=0; shift ;;
    --skip-native) WITH_NATIVE=0; shift ;;
    --no-man) WITH_MAN=0; shift ;;
    --skip-plugins) WITH_PLUGINS=0; shift ;;
    --skip-plugin-build) BUILD_PLUGINS=0; shift ;;
    --no-juce-download) ALLOW_JUCE_DOWNLOAD=0; shift ;;
    --no-standalone) WITH_STANDALONE=0; shift ;;
    --reset-plugin-cache) RESET_PLUGIN_CACHE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGIN_BUILD_DIR="${PLUGIN_BUILD_DIR:-${REPO_ROOT}/build/native/verbx_plugin-juce}"
MAN_DIR="${MAN_DIR:-${PREFIX}/share/man}"
OS_NAME="$(uname -s)"

case "$OS_NAME" in
  Darwin)
    AU_DIR="${AU_DIR:-${HOME}/Library/Audio/Plug-Ins/Components}"
    VST3_DIR="${VST3_DIR:-${HOME}/Library/Audio/Plug-Ins/VST3}"
    APP_DIR="${APP_DIR:-${HOME}/Applications}"
    ;;
  Linux)
    AU_DIR="${AU_DIR:-}"
    VST3_DIR="${VST3_DIR:-${HOME}/.vst3}"
    APP_DIR="${APP_DIR:-${PREFIX}/bin}"
    ;;
  *)
    die "unsupported platform: ${OS_NAME}; use macOS or Linux"
    ;;
esac

printf 'verbx installation plan\n'
printf '  repository:  %s\n' "$REPO_ROOT"
printf '  prefix:      %s\n' "$PREFIX"
printf '  Python CLI:  %s\n' "$([[ "$INSTALL_PYTHON" -eq 1 ]] && printf yes || printf no)"
printf '  native CLI:  %s\n' "$([[ "$WITH_NATIVE" -eq 1 ]] && printf yes || printf no)"
printf '  man pages:   %s\n' "$([[ "$WITH_MAN" -eq 1 ]] && printf "$MAN_DIR" || printf no)"
printf '  plug-ins:    %s\n' "$([[ "$WITH_PLUGINS" -eq 1 ]] && printf yes || printf no)"
if [[ "$WITH_PLUGINS" -eq 1 ]]; then
  [[ "$OS_NAME" != Darwin ]] || printf '  Audio Unit:  %s/VERBX.component\n' "$AU_DIR"
  printf '  VST3:        %s/VERBX.vst3\n' "$VST3_DIR"
  [[ "$WITH_STANDALONE" -eq 0 ]] || printf '  standalone:  %s\n' "$APP_DIR"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

cd "$REPO_ROOT"

if [[ "$INSTALL_PYTHON" -eq 1 ]]; then
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
  extras=()
  [[ "$WITH_RUNTIME_EXTRAS" -eq 0 ]] || extras+=(realtime sofa)
  [[ "$WITH_DEV" -eq 0 ]] || extras+=(dev)
  extra_spec=""
  if [[ "${#extras[@]}" -gt 0 ]]; then
    extra_spec="[$(IFS=,; printf '%s' "${extras[*]}")]"
  fi

  pip_args=()
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("hatchling") is not None else 1)
PY
  then
    pip_args+=(--no-build-isolation)
  fi

  printf 'Installing Python CLI and extras...\n'
  if [[ "$INSTALL_MODE" == editable ]]; then
    "$PYTHON_BIN" -m pip install "${pip_args[@]}" -e ".${extra_spec}"
  else
    "$PYTHON_BIN" -m pip install "${pip_args[@]}" ".${extra_spec}"
  fi
fi

if [[ "$WITH_NATIVE" -eq 1 ]]; then
  native_args=(--prefix "$PREFIX" --man-dir "$MAN_DIR")
  [[ "$WITH_MAN" -eq 1 ]] || native_args+=(--no-man)
  "${REPO_ROOT}/scripts/install_verbx_c.sh" "${native_args[@]}"
fi

if [[ "$WITH_MAN" -eq 1 ]]; then
  src_man_dir="${REPO_ROOT}/man/man1"
  dest_man1_dir="${MAN_DIR}/man1"
  [[ -d "$src_man_dir" ]] || die "man source directory not found: $src_man_dir"
  mkdir -p "$dest_man1_dir"
  cp "${src_man_dir}"/*.1 "$dest_man1_dir"/
  printf 'Installed man pages to %s\n' "$dest_man1_dir"
  if command -v mandb >/dev/null 2>&1; then
    mandb -q "$MAN_DIR" >/dev/null 2>&1 || true
  fi
fi

if [[ "$WITH_PLUGINS" -eq 1 ]]; then
  command -v cmake >/dev/null 2>&1 || die "CMake 3.22 or newer is required to build plug-ins"

  if [[ "$BUILD_PLUGINS" -eq 1 ]]; then
    if [[ -z "$JUCE_SOURCE" ]]; then
      cmake_cache="${PLUGIN_BUILD_DIR}/CMakeCache.txt"
      if [[ -f "$cmake_cache" ]]; then
        while IFS= read -r cache_line; do
          case "$cache_line" in
            VERBX_JUCE_SOURCE_DIR:PATH=*)
              cached_juce_source="${cache_line#*=}"
              if [[ -f "${cached_juce_source}/CMakeLists.txt" ]]; then
                JUCE_SOURCE="$cached_juce_source"
              fi
              break
              ;;
          esac
        done < "$cmake_cache"
      fi
    fi

    if [[ -z "$JUCE_SOURCE" ]]; then
      juce_candidates=(
        "${REPO_ROOT}/JUCE"
        "${REPO_ROOT}/build/deps/JUCE"
        "${REPO_ROOT}/build/_deps/juce-src"
        "${REPO_ROOT}/../JUCE"
      )
      for candidate in "${juce_candidates[@]}"; do
        if [[ -f "${candidate}/CMakeLists.txt" ]]; then
          JUCE_SOURCE="$candidate"
          break
        fi
      done
    fi

    if [[ -n "$JUCE_SOURCE" ]]; then
      [[ -f "${JUCE_SOURCE}/CMakeLists.txt" ]] || die "JUCE source checkout not found: $JUCE_SOURCE"
    elif [[ "$ALLOW_JUCE_DOWNLOAD" -eq 1 ]]; then
      command -v git >/dev/null 2>&1 || die "git is required to download JUCE"
      JUCE_SOURCE="${REPO_ROOT}/build/deps/JUCE"
      mkdir -p "$(dirname "$JUCE_SOURCE")"
      printf 'Downloading JUCE %s...\n' "$JUCE_VERSION"
      git clone --depth 1 --branch "$JUCE_VERSION" https://github.com/juce-framework/JUCE.git "$JUCE_SOURCE"
    fi

    configure_args=(
      -S "${REPO_ROOT}/native/verbx_plugin"
      -B "$PLUGIN_BUILD_DIR"
      -DVERBX_ENABLE_JUCE_PLUGIN=ON
      -DCMAKE_BUILD_TYPE=Release
    )
    if [[ "$OS_NAME" == Darwin ]]; then
      configure_args+=(
        "-DCMAKE_OSX_ARCHITECTURES=${MACOS_ARCHITECTURES}"
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOS_DEPLOYMENT_TARGET}"
      )
    fi
    [[ -z "$JUCE_SOURCE" ]] || configure_args+=("-DVERBX_JUCE_SOURCE_DIR=${JUCE_SOURCE}")
    cmake "${configure_args[@]}"
    cmake --build "$PLUGIN_BUILD_DIR" --config Release --parallel "$JOBS"
  fi

  artifact_root="${PLUGIN_BUILD_DIR}/VERBXPlugin_artefacts/Release"
  vst3_source="${artifact_root}/VST3/VERBX.vst3"
  [[ -d "$vst3_source" ]] || die "VST3 artifact not found: $vst3_source"
  install_bundle "$vst3_source" "$VST3_DIR"
  installed_vst3="${VST3_DIR}/VERBX.vst3"
  sign_and_verify_bundle "$installed_vst3"

  if [[ "$OS_NAME" == Darwin ]]; then
    au_source="${artifact_root}/AU/VERBX.component"
    [[ -d "$au_source" ]] || die "Audio Unit artifact not found: $au_source"
    install_bundle "$au_source" "$AU_DIR"
    installed_au="${AU_DIR}/VERBX.component"
    sign_and_verify_bundle "$installed_au"
  fi

  if [[ "$WITH_STANDALONE" -eq 1 ]]; then
    if [[ "$OS_NAME" == Darwin ]]; then
      standalone_source="${artifact_root}/Standalone/VERBX.app"
      [[ -d "$standalone_source" ]] || die "standalone artifact not found: $standalone_source"
      install_bundle "$standalone_source" "$APP_DIR"
      sign_and_verify_bundle "${APP_DIR}/VERBX.app"
    else
      standalone_source="${artifact_root}/Standalone/VERBX"
      [[ -x "$standalone_source" ]] || die "standalone artifact not found: $standalone_source"
      mkdir -p "$APP_DIR"
      cp "$standalone_source" "${APP_DIR}/verbx-plugin"
      chmod 755 "${APP_DIR}/verbx-plugin"
      printf 'Installed %s\n' "${APP_DIR}/verbx-plugin"
    fi
  fi

  if [[ "$OS_NAME" == Darwin ]]; then
    refresh_macos_plugin_registry "$installed_au" "$installed_vst3"
  fi
fi

printf '\nInstallation complete.\n'
printf 'Try:\n'
[[ "$INSTALL_PYTHON" -eq 0 ]] || printf '  verbx --help\n'
[[ "$WITH_NATIVE" -eq 0 ]] || printf '  %s/bin/verbx-c --help\n' "$PREFIX"
if [[ "$WITH_PLUGINS" -eq 1 ]]; then
  [[ "$OS_NAME" != Darwin || "$WITH_STANDALONE" -eq 0 ]] || printf '  open %q\n' "${APP_DIR}/VERBX.app"
  printf '  Restart or rescan your audio host to discover VERBX.\n'
  [[ "$OS_NAME" != Darwin || "$RESET_PLUGIN_CACHE" -eq 0 ]] || \
    printf '  The next DAW launch will rebuild the Audio Unit cache.\n'
fi
