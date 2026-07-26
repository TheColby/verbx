#!/usr/bin/env bash
set -euo pipefail

# Thin top-level entrypoint so users can run `./install.sh` from repo root
# without caring where the real installer lives.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${REPO_ROOT}/scripts/install.sh" "$@"
