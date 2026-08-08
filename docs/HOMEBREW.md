# Homebrew Support

`verbx` supports Homebrew distribution through the tap formula at:

- `TheColby/homebrew-verbx`
- Formula path: `Formula/verbx.rb`

## User Install (macOS)

Tap installation is the normal Homebrew path. It lets Homebrew locate the formula by its repository name and keeps later upgrades attached to the same source.

```bash
brew tap thecolby/verbx
brew install thecolby/verbx/verbx
verbx version
```

If the tap is unavailable in your environment, you can install from the formula
in this repository:

```bash
brew install --build-from-source ./packaging/homebrew/verbx.rb
```

## Maintainer Workflow

A release maintainer refreshes the formula only after the version and release artifact are final. The workflow below keeps the formula pin, source checksum, and published tag aligned.

1. Refresh formula pins for a new release tag:

```bash
./scripts/refresh_homebrew_formula.sh 0.9.9
```

2. Commit formula changes in this repo:

```bash
git add packaging/homebrew/verbx.rb scripts/refresh_homebrew_formula.sh docs/HOMEBREW.md
git commit -m "chore(homebrew): refresh formula for v0.9.9"
```

3. Ensure tap repo formula is updated and pushed.

## Release Automation

The release workflow can synchronize the tap after a version tag is pushed. Repository secrets and policy variables determine whether a missing tap credential blocks the release.

`.github/workflows/release.yml` includes gated tap sync:

- Trigger: tag push (`v*`)
- Job: `sync-homebrew-tap`
- Requires secret: `HOMEBREW_TAP_TOKEN` (when sync required)
- Optional variable: `HOMEBREW_TAP_REPO` (default `TheColby/homebrew-verbx`)
- Policy variable: `RELEASE_REQUIRE_HOMEBREW` (`true` by default)

Behavior:

- `RELEASE_REQUIRE_HOMEBREW=true` (default): release fails if token is missing.
- `RELEASE_REQUIRE_HOMEBREW=false`: sync is skipped when token is missing.

## Compatibility Notes

The formula keeps compiler-sensitive acceleration packages outside its required resource set. This limits installation failures without changing the core command-line DSP contract.

- Formula pins Python resources using `brew update-python-resources`.
- `numba`/`llvmlite`/`scikit-learn` are intentionally excluded from Homebrew
  resources to avoid fragile compiler-bound builds in common macOS
  environments.
- Core DSP paths are unaffected; shimmer remains available with librosa and
  falls back gracefully when acceleration dependencies are absent.
