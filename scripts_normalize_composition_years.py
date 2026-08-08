#!/usr/bin/env python3
"""Keep composition years attached to musical-work titles in the documentation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATALOG_SOURCES = (
    ROOT / "docs" / "MUSICAL_PIECES_APPENDIX.md",
    ROOT / "docs" / "MUSICAL_PIECES_EXPANSION.md",
)
ENTRY_PATTERN = re.compile(
    r"(?m)^\*\*[^\n]*?, \*(?P<title>[^*]+)\* "
    r"\((?P<date>(?:c\. )?\d{4}(?:–\d{4})?(?:; revised later)?)\)\.\*\*"
)
DATE_TEXT = (
    r"(?:c\. )?\d{4}(?:–\d{4})?(?:; revised later)?"
    r"|traditional; date unknown"
)
DATE_SUFFIX = re.compile(rf"^ \((?:{DATE_TEXT})\)")

# These works occur outside, or use a spelling different from, the two formal
# listening catalogs. Dates follow composer, publisher, institutional, or
# discographic records cited in the guide's source material.
SUPPLEMENTAL_DATES = {
    "1/1": "1978",
    "ADNOS": "1973–1974",
    "Apparition de l'église éternelle": "1932",
    "Ave verum": "traditional; date unknown",
    "Choral No. 3": "1890",
    "Deep Listening": "1989",
    "Fanfare for the Common Man": "1942",
    "Fanfare pour précéder La Péri": "1912",
    "I Am Sitting in a Room": "1969",
    "La Péri": "1911",
    "Loveless": "1991",
    "Missa Papae Marcelli": "c. 1562",
    "Presque rien No. 1": "1970",
    "Répons": "1981–1984",
    "Running Up That Hill (A Deal with God)": "1985",
    "The Sacrificial Code": "2019",
    "Trois chorals pour grand orgue": "1890",
}


def composition_catalog() -> dict[str, str]:
    catalog: dict[str, str] = {}
    for source in CATALOG_SOURCES:
        for match in ENTRY_PATTERN.finditer(source.read_text(encoding="utf-8")):
            title = match.group("title")
            date = match.group("date")
            previous = catalog.setdefault(title, date)
            if previous != date:
                raise ValueError(
                    f"Conflicting dates for {title!r}: {previous!r} and {date!r}"
                )
    catalog.update(SUPPLEMENTAL_DATES)
    return catalog


def documentation_sources() -> tuple[Path, ...]:
    sources = [ROOT / "README.md"]
    sources.extend(
        path
        for path in sorted((ROOT / "docs").rglob("*.md"))
        if path.name != "USERGUIDE.md"
    )
    return tuple(sources)


def normalize_composition_years(
    markdown: str, catalog: dict[str, str] | None = None
) -> tuple[str, int]:
    catalog = catalog or composition_catalog()
    replacements = 0

    # Longest-first ordering prevents a short title from altering a longer one.
    for title in sorted(catalog, key=len, reverse=True):
        marker = f"*{title}*"
        date = catalog[title]
        offset = 0
        while True:
            index = markdown.find(marker, offset)
            if index < 0:
                break
            end = index + len(marker)
            if DATE_SUFFIX.match(markdown[end:]):
                offset = end
                continue
            insertion = f" ({date})"
            markdown = markdown[:end] + insertion + markdown[end:]
            replacements += 1
            offset = end + len(insertion)

    return markdown, replacements


def missing_composition_years(
    markdown: str, catalog: dict[str, str] | None = None
) -> list[str]:
    catalog = catalog or composition_catalog()
    missing: list[str] = []
    for title in sorted(catalog, key=str.casefold):
        marker = f"*{title}*"
        offset = 0
        while True:
            index = markdown.find(marker, offset)
            if index < 0:
                break
            end = index + len(marker)
            if not DATE_SUFFIX.match(markdown[end:]):
                missing.append(title)
            offset = end
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report missing years without modifying documentation",
    )
    args = parser.parse_args()
    catalog = composition_catalog()
    changed_files = 0
    replacements = 0

    for source in documentation_sources():
        markdown = source.read_text(encoding="utf-8")
        missing = missing_composition_years(markdown, catalog)
        if args.check:
            if missing:
                relative = source.relative_to(ROOT)
                print(f"{relative}: {', '.join(missing)}")
            replacements += len(missing)
            continue

        normalized, count = normalize_composition_years(markdown, catalog)
        if count:
            source.write_text(normalized, encoding="utf-8")
            changed_files += 1
            replacements += count

    if args.check and replacements:
        print(f"Found {replacements} composition title(s) without dates")
        return 1

    action = "Checked" if args.check else "Normalized"
    print(
        f"{action} {len(catalog)} composition titles across "
        f"{len(documentation_sources())} documentation sources; "
        f"{replacements} replacement(s) in {changed_files} file(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
