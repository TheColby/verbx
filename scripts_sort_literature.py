#!/usr/bin/env python3
"""Alphabetize documentation literature lists by first author."""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCES = ROOT / "docs" / "REFERENCES.md"
README = ROOT / "README.md"
IR_SYNTHESIS = ROOT / "docs" / "IR_SYNTHESIS.md"
SOFA_FEASIBILITY = ROOT / "docs" / "SOFA_FEASIBILITY.md"

ENTRY_BLOCK_PATTERN = re.compile(
    r"(?m)^(?P<anchors>(?:(?:<a id=\"[^\"]+\"></a>)+\n)?)"
    r"(?P<entry>\*\*\[[^]]+\]\*\*\s+[^\n]+)"
    r"(?P<annotation>(?:\n\n(?:>[^\n]*(?:\n>[^\n]*)*))?)"
)
SECTION_PATTERN = re.compile(r"(?m)^## Section \d+: .+$")


def _plain(value: str) -> str:
    value = re.sub(r"[*_`]", "", value)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _author_key(value: str) -> tuple[str, ...]:
    value = re.sub(r"^\*\*\d+\.\s*", "", value.strip())
    value = re.sub(r"^\*\*", "", value)
    value = value.split(" — ", 1)[0]
    value = value.split(" (", 1)[0]
    value = value.split(":", 1)[0]
    first_author = value.split(";", 1)[0].strip()
    surname = first_author.split(",", 1)[0].strip()
    return (_plain(surname), _plain(first_author), _plain(value))


def _entry_key(block: str) -> tuple[str, ...]:
    entry = next(line for line in block.splitlines() if line.startswith("**["))
    match = re.match(
        r"\*\*\[[^]]+\]\*\*\s+(?P<authors>.+?)\s+"
        r"\((?P<year>(?:18|19|20)\d{2}[a-z]?|n\.d\.)\)\.\s+(?P<title>.+?)\.",
        entry,
    )
    if match is None:
        raise ValueError(f"Cannot parse reference entry: {entry}")
    return (
        *_author_key(match.group("authors")),
        _plain(match.group("year")),
        _plain(match.group("title")),
    )


def _sort_reference_section(section: str) -> str:
    matches = list(ENTRY_BLOCK_PATTERN.finditer(section))
    if not matches:
        return section
    prefix = section[: matches[0].start()].rstrip()
    suffix = section[matches[-1].end() :].lstrip()
    blocks = sorted((match.group(0).strip() for match in matches), key=_entry_key)
    result = prefix + "\n\n" + "\n\n".join(blocks)
    if suffix:
        result += "\n\n" + suffix
    return result.rstrip()


def sort_reference_sections(text: str) -> str:
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        raise ValueError("No numbered literature sections found")
    output = [text[: matches[0].start()].rstrip()]
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        output.append(_sort_reference_section(text[match.start() : end]))
    return "\n\n".join(output).rstrip() + "\n"


def sort_numbered_reading_list(text: str) -> str:
    start = text.find("**1. ")
    end = text.find("\n---", start)
    if start == -1 or end == -1:
        raise ValueError("Cannot find the numbered introductory reading list")
    region = text[start:end].strip()
    pattern = re.compile(r"(?ms)^\*\*\d+\. .+?(?=^\*\*\d+\. |\Z)")
    blocks = [match.group(0).strip() for match in pattern.finditer(region)]
    if len(blocks) != 7:
        raise ValueError(f"Expected 7 introductory readings, found {len(blocks)}")
    blocks.sort(key=lambda block: _author_key(block.splitlines()[0]))
    renumbered = [
        re.sub(r"^\*\*\d+\.", f"**{index}.", block) for index, block in enumerate(blocks, 1)
    ]
    return text[:start] + "\n\n".join(renumbered) + "\n\n" + text[end + 1 :]


def sort_bullet_region(text: str, start_marker: str, end_marker: str | None) -> str:
    start = text.find(start_marker)
    if start == -1:
        raise ValueError(f"Cannot find literature-list marker: {start_marker!r}")
    start += len(start_marker)
    end = text.find(end_marker, start) if end_marker is not None else len(text)
    if end == -1:
        raise ValueError(f"Cannot find literature-list end marker: {end_marker!r}")
    region = text[start:end].strip()
    pattern = re.compile(r"(?ms)^- .+?(?=^- |\Z)")
    blocks = [match.group(0).strip() for match in pattern.finditer(region)]
    if not blocks:
        raise ValueError(f"No literature bullets found after {start_marker!r}")
    blocks.sort(key=lambda block: _author_key(block.splitlines()[0][2:]))
    return text[:start] + "\n" + "\n".join(blocks) + "\n" + text[end:]


def sorted_documents() -> dict[Path, str]:
    references = REFERENCES.read_text(encoding="utf-8")
    references = sort_numbered_reading_list(references)
    references = sort_bullet_region(
        references,
        "### Core canonical links\n\n",
        "\n\n---",
    )
    references = sort_reference_sections(references)

    readme = sort_bullet_region(
        README.read_text(encoding="utf-8"),
        "Key papers:\n\n",
        "\n\nAdditional guides in `docs/`:",
    )
    ir_synthesis = sort_bullet_region(
        IR_SYNTHESIS.read_text(encoding="utf-8"),
        "For the DSP foundations underlying this implementation:\n\n",
        "\n\nSee `docs/REFERENCES.md`",
    )
    sofa = sort_bullet_region(
        SOFA_FEASIBILITY.read_text(encoding="utf-8"),
        "## References\n\n",
        None,
    )
    return {
        REFERENCES: references,
        README: readme,
        IR_SYNTHESIS: ir_synthesis,
        SOFA_FEASIBILITY: sofa,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if any list is not sorted")
    args = parser.parse_args()
    documents = sorted_documents()
    changed = [
        path for path, content in documents.items() if path.read_text(encoding="utf-8") != content
    ]
    if args.check:
        if changed:
            for path in changed:
                print(
                    f"Literature list is not alphabetized: {path.relative_to(ROOT)}",
                    file=sys.stderr,
                )
            return 1
        print("All literature lists are alphabetized by first author")
        return 0
    for path, content in documents.items():
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
