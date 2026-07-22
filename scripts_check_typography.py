#!/usr/bin/env python3
"""Enforce repository-wide typography rules for textual project files."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORBIDDEN_SPACED_EM_DASH = " \N{EM DASH} "
TEXT_SUFFIXES = {
    ".cff",
    ".css",
    ".html",
    ".ipynb",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".svg",
    ".tex",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {"Dockerfile", "LICENSE", "Makefile"}
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".superpowers",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "tmp",
}


def typography_sources() -> tuple[Path, ...]:
    sources: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.name.startswith(".") or path.suffix == ".swp":
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_FILENAMES:
            sources.append(path)
    return tuple(sorted(sources))


def spaced_em_dash_violations() -> list[str]:
    violations: list[str] = []
    for path in typography_sources():
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN_SPACED_EM_DASH in line:
                relative = path.relative_to(ROOT)
                violations.append(f"{relative}:{line_number}: {line.strip()}")
    return violations


def main() -> int:
    violations = spaced_em_dash_violations()
    if violations:
        print("Spaced em dashes are forbidden; use an en dash or recast the sentence:")
        print("\n".join(violations))
        return 1
    print(
        f"Typography check passed across {len(typography_sources())} text files: "
        "no spaced em dashes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
