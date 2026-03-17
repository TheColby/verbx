#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_FORMULA="${ROOT_DIR}/packaging/homebrew/verbx.rb"
TAP_NAME="${HOMEBREW_TAP:-thecolby/verbx}"
EXCLUDE_PACKAGES="${HOMEBREW_EXCLUDE_PACKAGES:-numba,llvmlite}"
EXTRA_PACKAGES="${HOMEBREW_EXTRA_PACKAGES:-numpy}"

usage() {
  cat <<'EOF'
Usage:
  scripts/refresh_homebrew_formula.sh <version>

Examples:
  scripts/refresh_homebrew_formula.sh 0.7.2
  HOMEBREW_TAP=thecolby/verbx scripts/refresh_homebrew_formula.sh 0.7.2

Notes:
  - Requires Homebrew and `brew update-python-resources`.
  - Requires an existing local tap (e.g. `brew tap-new thecolby/homebrew-verbx`).
  - Excludes numba/llvmlite to avoid fragile source builds in Homebrew environments.
EOF
}

if [[ "${1:-}" == "" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 1
fi

VERSION="$1"
if [[ ! "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "error: version must be semantic (e.g. 0.7.2)" >&2
  exit 2
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "error: brew not found in PATH" >&2
  exit 3
fi

if [[ ! -f "${WORKSPACE_FORMULA}" ]]; then
  echo "error: missing workspace formula at ${WORKSPACE_FORMULA}" >&2
  exit 4
fi

if ! TAP_REPO_DIR="$(brew --repo "${TAP_NAME}" 2>/dev/null)"; then
  echo "error: local tap '${TAP_NAME}' not found." >&2
  echo "hint: run 'brew tap-new thecolby/homebrew-verbx' once, then re-run this script." >&2
  exit 5
fi

FORMULA_PATH="${TAP_REPO_DIR}/Formula/verbx.rb"
mkdir -p "$(dirname "${FORMULA_PATH}")"

TAG="v${VERSION}"
SOURCE_URL="https://github.com/TheColby/verbx/archive/refs/tags/${TAG}.tar.gz"

TMP_TARBALL="$(mktemp "${TMPDIR:-/tmp}/verbx-${TAG}.XXXXXX.tar.gz")"
trap 'rm -f "${TMP_TARBALL}"' EXIT

echo "Downloading source tarball for ${TAG}..."
curl -fsSL "${SOURCE_URL}" -o "${TMP_TARBALL}"
SHA256="$(openssl dgst -sha256 "${TMP_TARBALL}" | sed 's/^.*= //')"

cp "${WORKSPACE_FORMULA}" "${FORMULA_PATH}"
sed -i.bak "s|^  url \".*\"$|  url \"${SOURCE_URL}\"|" "${FORMULA_PATH}"
sed -i.bak "s|^  sha256 \".*\"$|  sha256 \"${SHA256}\"|" "${FORMULA_PATH}"
rm -f "${FORMULA_PATH}.bak"

echo "Resolving Homebrew resource pins..."
brew update-python-resources \
  --ignore-non-pypi-packages \
  --exclude-packages "${EXCLUDE_PACKAGES}" \
  --extra-packages "${EXTRA_PACKAGES}" \
  "${TAP_NAME}/verbx"

cp "${FORMULA_PATH}" "${WORKSPACE_FORMULA}"

echo "Formula refreshed at:"
echo "  ${WORKSPACE_FORMULA}"
echo
echo "Next steps:"
echo "  1) Review and commit ${WORKSPACE_FORMULA}"
echo "  2) Ensure tap Formula/verbx.rb is updated and pushed"
