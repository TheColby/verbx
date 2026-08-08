#!/usr/bin/env bash
set -euo pipefail

# Generate the user guide from the repository root and forward optional flags.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
exec uv run python scripts_generate_docs_pdf.py "$@"
