#!/usr/bin/env python3
"""Verify canonical launch examples stay mirrored across docs and man pages."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_FILE = ROOT / "docs" / "LAUNCH_EXAMPLES_CANONICAL.txt"
TARGET_FILES = (
    ROOT / "README.md",
    ROOT / "docs" / "EXTREME_COOKBOOK.md",
    ROOT / "man" / "man1" / "verbx-render.1",
)


def _load_canonical_commands(path: Path) -> list[str]:
    commands: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        commands.append(_normalize_text(line, is_man=False))
    return commands


def _normalize_text(text: str, *, is_man: bool) -> str:
    normalized = text.replace("\\\n", " ")
    if is_man:
        normalized_lines: list[str] = []
        for raw_line in normalized.splitlines():
            line = raw_line.rstrip()
            if line.startswith(".B "):
                line = line[3:]
            elif line.startswith(".I "):
                line = line[3:]
            elif line.startswith("."):
                line = ""
            line = line.replace("\\fB", "").replace("\\fR", "").replace("\\-", "-")
            normalized_lines.append(line)
        normalized = "\n".join(normalized_lines)
    return re.sub(r"\s+", " ", normalized).strip()


def check_launch_examples() -> tuple[bool, list[str]]:
    canonical_commands = _load_canonical_commands(CANONICAL_FILE)
    failures: list[str] = []
    for target in TARGET_FILES:
        target_text = target.read_text(encoding="utf-8")
        normalized_target = _normalize_text(target_text, is_man=target.suffix == ".1")
        missing = [cmd for cmd in canonical_commands if cmd not in normalized_target]
        if missing:
            failures.append(
                f"{target.relative_to(ROOT)} missing {len(missing)} canonical command(s):\n"
                + "\n".join(f"  - {cmd}" for cmd in missing)
            )
    return (len(failures) == 0, failures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if canonical launch examples are not mirrored.",
    )
    args = parser.parse_args()

    ok, failures = check_launch_examples()
    if ok:
        print("Launch examples are synchronized across README/cookbook/man page.")
        return 0

    print("Launch-example parity check failed:")
    for failure in failures:
        print()
        print(failure)
    if args.check:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
