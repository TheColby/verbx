#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
verbx install helper

Installs the Python package and man pages from this repository.

Usage:
  scripts/install.sh [options]

Options:
  --prefix PATH       Install prefix for man pages (default: ~/.local)
  --python PATH       Python executable to use (default: python3)
  --editable          Install package in editable mode (default)
  --wheel             Install package as regular package (non-editable)
  --dev               Include dev extras (.[dev]) during install
  --skip-python-install
                      Only install man pages; skip pip package install step
  --no-man            Skip man page installation
  --man-dir PATH      Override man page root (default: <prefix>/share/man)
  -h, --help          Show this help and exit

Examples:
  scripts/install.sh
  scripts/install.sh --dev --prefix "$HOME/.local"
  scripts/install.sh --skip-python-install --prefix "$HOME/.local"
  scripts/install.sh --wheel --no-man
EOF
}

PREFIX="${HOME}/.local"
PYTHON_BIN="python3"
INSTALL_MODE="editable"
WITH_DEV=0
INSTALL_PYTHON=1
WITH_MAN=1
MAN_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --editable)
      INSTALL_MODE="editable"
      shift
      ;;
    --wheel)
      INSTALL_MODE="wheel"
      shift
      ;;
    --dev)
      WITH_DEV=1
      shift
      ;;
    --skip-python-install)
      INSTALL_PYTHON=0
      shift
      ;;
    --no-man)
      WITH_MAN=0
      shift
      ;;
    --man-dir)
      MAN_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -z "$MAN_DIR" ]]; then
  MAN_DIR="${PREFIX}/share/man"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXTRA=""
if [[ "$WITH_DEV" -eq 1 ]]; then
  EXTRA="[dev]"
fi

if [[ "$INSTALL_PYTHON" -eq 1 ]]; then
  PIP_ARGS=()
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("hatchling") is not None else 1)
PY
  then
    # Use local backend when available to reduce network dependency.
    PIP_ARGS+=(--no-build-isolation)
  fi

  if [[ "$INSTALL_MODE" == "editable" ]]; then
    echo "Installing verbx in editable mode..."
    if [[ "${#PIP_ARGS[@]}" -gt 0 ]]; then
      "$PYTHON_BIN" -m pip install "${PIP_ARGS[@]}" -e ".${EXTRA}"
    else
      "$PYTHON_BIN" -m pip install -e ".${EXTRA}"
    fi
  else
    echo "Installing verbx in wheel mode..."
    if [[ "${#PIP_ARGS[@]}" -gt 0 ]]; then
      "$PYTHON_BIN" -m pip install "${PIP_ARGS[@]}" ".${EXTRA}"
    else
      "$PYTHON_BIN" -m pip install ".${EXTRA}"
    fi
  fi
else
  echo "Skipping Python package installation (--skip-python-install)."
fi

if [[ "$WITH_MAN" -eq 1 ]]; then
  SRC_MAN_DIR="${REPO_ROOT}/man/man1"
  DEST_MAN1_DIR="${MAN_DIR}/man1"
  if [[ ! -d "$SRC_MAN_DIR" ]]; then
    echo "Man source directory not found: $SRC_MAN_DIR" >&2
    exit 1
  fi

  mkdir -p "$DEST_MAN1_DIR"
  cp "${SRC_MAN_DIR}"/*.1 "$DEST_MAN1_DIR"/
  echo "Installed man pages to: $DEST_MAN1_DIR"

  if command -v mandb >/dev/null 2>&1; then
    mandb -q "$MAN_DIR" >/dev/null 2>&1 || true
  fi
fi

echo "Installation complete."
echo
echo "Try:"
echo "  verbx --help"
if [[ "$WITH_MAN" -eq 1 ]]; then
  echo "  man verbx"
  echo "  man verbx-render"
  echo "  man verbx-analyze"
  echo "  man verbx-dereverb"
fi
