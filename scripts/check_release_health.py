#!/usr/bin/env python3
"""Pre-release health checks for verbx packaging.

Verifies:
1. ``pyproject.toml`` version matches the expected release tag.
2. ``CHANGELOG.md`` contains an entry for the version.
3. Homebrew formula is pinned to the release tag.
4. Built wheel and sdist metadata pass ``twine check``.

Exit 0 on all checks pass, exit 1 on any failure.

Usage::

    python scripts/check_release_health.py --tag v0.7.6
    python scripts/check_release_health.py --tag v0.7.6 --dist-dir dist/
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_pyproject_version() -> str:
    pyproject = ROOT / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError("Could not find version field in pyproject.toml")
    return m.group(1)


def _check_version_matches_tag(tag: str) -> list[str]:
    """Return list of failure messages (empty = passed)."""
    failures: list[str] = []
    # Normalise: strip leading 'v' from tag for comparison
    expected_version = tag.lstrip("v")
    actual_version = _read_pyproject_version()
    if actual_version != expected_version:
        failures.append(
            f"pyproject.toml version '{actual_version}' does not match tag '{tag}' "
            f"(expected '{expected_version}')"
        )
    return failures


def _check_changelog_entry(tag: str) -> list[str]:
    """Return list of failure messages (empty = passed)."""
    failures: list[str] = []
    changelog = ROOT / "CHANGELOG.md"
    if not changelog.exists():
        failures.append("CHANGELOG.md not found")
        return failures
    version = tag.lstrip("v")
    text = changelog.read_text(encoding="utf-8")
    # Acceptable patterns: ## v0.7.6, ## 0.7.6, ## [0.7.6], ## [v0.7.6]
    patterns = [
        rf"##\s+\[?v?{re.escape(version)}\]?",
    ]
    found = any(re.search(p, text, re.IGNORECASE) for p in patterns)
    if not found:
        failures.append(
            f"CHANGELOG.md has no entry for version '{version}'. "
            "Add a release section before tagging."
        )
    return failures


def _check_homebrew_formula_pinned(tag: str) -> list[str]:
    """Return list of failure messages (empty = passed)."""
    failures: list[str] = []
    formula = ROOT / "packaging" / "homebrew" / "verbx.rb"
    if not formula.exists():
        failures.append(f"Homebrew formula not found at {formula.relative_to(ROOT)}")
        return failures
    text = formula.read_text(encoding="utf-8")
    version = tag.lstrip("v")
    # The formula should reference the tag (e.g. refs/tags/v0.7.6.tar.gz or /v0.7.6/)
    if f"/{tag}" not in text and f"/{version}" not in text:
        failures.append(
            f"Homebrew formula at packaging/homebrew/verbx.rb is not pinned to tag '{tag}'. "
            "Run scripts/refresh_homebrew_formula.sh to update."
        )
    return failures


def _check_dist_artifacts(dist_dir: Path) -> list[str]:
    """Run twine check on built artifacts. Return failure messages."""
    failures: list[str] = []
    if not dist_dir.exists():
        failures.append(f"dist dir not found: {dist_dir}")
        return failures
    artifacts = list(dist_dir.glob("*.whl")) + list(dist_dir.glob("*.tar.gz"))
    if not artifacts:
        failures.append(f"No wheel or sdist found in {dist_dir}")
        return failures
    result = subprocess.run(
        [sys.executable, "-m", "twine", "check", "--strict", *[str(a) for a in artifacts]],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        failures.append(f"twine check failed:\n{result.stdout}\n{result.stderr}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-release packaging health checks for verbx.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tag",
        required=True,
        metavar="TAG",
        help="Release tag to validate against (e.g. v0.7.6)",
    )
    parser.add_argument(
        "--dist-dir",
        metavar="PATH",
        default=None,
        help="Directory containing built wheel and sdist to validate with twine check",
    )
    parser.add_argument(
        "--skip-twine",
        action="store_true",
        help="Skip twine check (useful when artifacts are not yet built)",
    )
    args = parser.parse_args(argv)

    all_failures: list[str] = []

    print(f"Checking release health for tag: {args.tag}")

    version_failures = _check_version_matches_tag(args.tag)
    changelog_failures = _check_changelog_entry(args.tag)
    formula_failures = _check_homebrew_formula_pinned(args.tag)

    all_failures.extend(version_failures)
    all_failures.extend(changelog_failures)
    all_failures.extend(formula_failures)

    if not args.skip_twine:
        dist_dir = Path(args.dist_dir) if args.dist_dir else ROOT / "dist"
        twine_failures = _check_dist_artifacts(dist_dir)
        all_failures.extend(twine_failures)

    if all_failures:
        print(f"\n{len(all_failures)} check(s) FAILED:", file=sys.stderr)
        for i, msg in enumerate(all_failures, 1):
            print(f"  {i}. {msg}", file=sys.stderr)
        return 1

    checks_run = 3 + (0 if args.skip_twine else 1)
    print(f"All {checks_run} release health checks PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
