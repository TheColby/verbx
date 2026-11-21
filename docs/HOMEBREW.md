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
./scripts/refresh_homebrew_formula.sh 0.9.10
```

2. Commit formula changes in this repo:

```bash
git add packaging/homebrew/verbx.rb scripts/refresh_homebrew_formula.sh docs/HOMEBREW.md
git commit -m "chore(homebrew): refresh formula for v0.9.10"
```

3. Ensure tap repo formula is updated and pushed.

## Release Automation

The tap owns `.github/workflows/sync-release.yml`. It checks the latest published
GitHub release hourly (at minute 17) and also supports **Run workflow** for an
immediate update. It fetches the formula from the release tag, updates the source
URL and SHA-256, and commits with the tap's own `GITHUB_TOKEN`.

This setup needs no personal token or cross-repository secret. Inspect or run it
at <https://github.com/TheColby/homebrew-verbx/actions/workflows/sync-release.yml>.

The upstream release workflow retains optional direct sync for maintainers who
supply `HOMEBREW_TAP_TOKEN`. `RELEASE_REQUIRE_HOMEBREW=false` allows release
creation while the tap's scheduled workflow handles synchronization separately.

## Installation Validation

The tap's synchronization workflow passed for v0.9.10. Synchronization verifies
formula generation and checksums; it does not prove that installation succeeds.
Run the upstream `Homebrew installation smoke` workflow manually on a clean
macOS runner to install from source and exercise the installed CLI:

```bash
gh workflow run homebrew-smoke.yml --repo TheColby/verbx
```

The 2026-10-17 local upgrade attempt stopped during Homebrew dependency
resolution because the GCC bottle required Xcode Command Line Tools at
`/Library/Developer/CommandLineTools`. Xcode.app alone did not satisfy that
requirement. The verbx formula's installation stage was not reached, so a clean
installation remains unverified. Install the Command Line Tools through Apple's
normal installer before retrying on that machine, or use the clean-runner check.

## Compatibility Notes

The formula uses Python 3.14 to match Homebrew's NumPy and SciPy bottle
bindings. Those packages are supplied by formula dependencies rather than
rebuilt as duplicate Python resources. The virtualenv uses the matching system
site-packages.

Pinned resources include librosa's Numba, llvmlite, scikit-learn, joblib,
threadpoolctl, and narwhals dependencies. CMake and Ninja support source builds;
llvmlite links to versioned `llvm@22`. Formula tests run `pip check`, import the
numerical stack, and perform an actual pitch shift with finite/nonzero output.

These packaging changes require a successful hosted source installation before
being treated as validated. Homebrew's helper enables build isolation for Python
resource builds; package build requirements may still need network access.
