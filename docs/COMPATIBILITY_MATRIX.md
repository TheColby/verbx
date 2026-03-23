# Compatibility Matrix (Public Alpha)

_Last updated: 2026-03-23_

This matrix tracks install/runtime channels that are expected to work in the
current alpha line (`0.7.x`), and how they are validated in CI.

## Platform + Install Channel Matrix

| Platform | Python | Install channel | Status | CI coverage |
|---|---:|---|---|---|
| Ubuntu latest | 3.11 | `uv sync` + `uv run verbx` | Supported | `install-verify.yml` |
| Ubuntu latest | 3.11 | `pip install -e .` in venv | Supported | `install-verify.yml` |
| Ubuntu latest | 3.11 | `scripts/install.sh` | Supported | `install-verify.yml` |
| Ubuntu latest | 3.11 | npm launcher (`npm/verbx.js`) | Supported | `install-verify.yml` |
| macOS latest | 3.11 | `uv sync` + `uv run verbx` | Supported | `install-verify.yml` |
| macOS latest | 3.11 | `pip install -e .` in venv | Supported | `install-verify.yml` |
| macOS latest | 3.11 | `scripts/install.sh` | Supported | `install-verify.yml` |
| macOS latest | 3.11 | npm launcher (`npm/verbx.js`) | Supported | `install-verify.yml` |

## Notes

- Primary package requires Python `>=3.11` (see `pyproject.toml`).
- npm launcher is a wrapper around the Python CLI and requires Python at runtime.
- Homebrew path is supported for users and maintainers, with release-sync
  automation in `.github/workflows/release.yml`.
- Performance regression comparison runs in CI via `perf-baseline` in
  `.github/workflows/ci.yml`.
