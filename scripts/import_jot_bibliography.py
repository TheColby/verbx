#!/usr/bin/env python3
"""Merge Jean-Marc Jot's public works list into the general bibliography."""

from __future__ import annotations

import argparse
import html
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "docs" / "REFERENCES.md"
OFFICIAL_URL = "https://sites.google.com/site/jmmjot/"
SECTION_END = "\n---\n\n## Section 2: Feedback Delay Networks"


class _ListItemParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.items: list[tuple[str, list[str]]] = []
        self._depth = 0
        self._text: list[str] = []
        self._links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "li":
            self._depth += 1
            self._text = []
            self._links = []
        elif self._depth and tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._links.append(href)

    def handle_data(self, data: str) -> None:
        if self._depth:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "li" or not self._depth:
            return
        text = " ".join(html.unescape("".join(self._text)).split())
        self.items.append((text, self._links.copy()))
        self._depth -= 1


def _plain(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _clean_text(value: str) -> str:
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    value = value.replace("\u2013", ":").replace("\u2014", ":")
    value = re.sub(r"\s+-\s+", ": ", value)
    value = value.replace("Muti-channel", "Multichannel")
    return " ".join(value.split())


def _authors(value: str) -> str:
    value = re.sub(r"\bet al\.?", ", et al.", value)
    value = re.sub(r"\band\b", ",", value)
    parts = [part.strip(" ,.") for part in value.split(",") if part.strip(" ,.")]
    output: list[str] = []
    for part in parts:
        if part == "et al":
            output.append("et al.")
            continue
        tokens = part.split()
        if len(tokens) < 2:
            output.append(part)
            continue
        surname = tokens[-1]
        given = " ".join(tokens[:-1])
        if part in {"J.-M. Jot", "J-.M. Jot"}:
            given = "Jean-Marc"
        output.append(f"{surname}, {given}")
    return "; ".join(output)


def _strip_resource_labels(value: str) -> str:
    value = re.sub(
        r"(?:\s+|\.)\b(?:paper|poster|slides|video|demo|report|summary|book chapter|open access)\b.*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s*\[(?:Invited|Keynote|Paper award)\]\s*", " ", value)
    return value.strip(" ,.")


def _year(value: str) -> str:
    match = re.search(r"\b((?:19|20)\d{2})\b", value)
    if not match:
        raise ValueError(f"Cannot find year in official record: {value}")
    return match.group(1)


def _venue(value: str, year: str) -> str:
    value = re.sub(rf"\([^)]*\b{year}\b[^)]*\)", "", value)
    return _strip_resource_labels(value) or "Official bibliography record"


def _unquoted_work(value: str) -> tuple[str, str, str]:
    if "E\u0301tude et re\u0301alisation" in value or "Etude et realisation" in _plain(value):
        title = value.split(",", 1)[1].split(", Doctoral dissertation", 1)[0].strip()
        return "Jot, Jean-Marc", title, "Telecom Paris doctoral dissertation"
    if "Spat: Introduction" in value:
        return (
            "Jot, Jean-Marc; Caulkins, T.; Gottfried, R.",
            "Spat: Introduction",
            "IRCAM technical report",
        )
    if "Spat: Reference Manual" in value:
        return (
            "Jot, Jean-Marc; Caulkins, T.; Gottfried, R.",
            "Spat: Reference Manual",
            "IRCAM technical report",
        )
    if "Interactive 3D Audio Rendering Guidelines" in value:
        return (
            "Jot, Jean-Marc; et al.",
            "Interactive 3D Audio Rendering Guidelines: Level 2.0 (I3DL2)",
            "Interactive Audio Special Interest Group guideline",
        )
    raise ValueError(f"Cannot parse unquoted official record: {value}")


def _publication(value: str, links: list[str]) -> dict[str, str]:
    value = _clean_text(value)
    year = _year(value)
    if '"' not in value:
        authors, title, venue = _unquoted_work(value)
    else:
        prefix, remainder = value.split('"', 1)
        title, suffix = remainder.split('"', 1)
        authors = _authors(prefix.strip(" ,."))
        venue = _venue(suffix, year)
    return {
        "authors": authors,
        "year": year,
        "title": _strip_resource_labels(title),
        "venue": venue,
        "url": links[0] if links else OFFICIAL_URL,
        "source": "Primary source" if links else "Official bibliography",
    }


def _patent_url(identifiers: str) -> str:
    match = re.search(r"\b(US|WO)\s*([0-9][0-9/,]*)", identifiers)
    if not match:
        return OFFICIAL_URL
    number = re.sub(r"[^0-9]", "", match.group(2))
    return f"https://patents.google.com/patent/{match.group(1)}{number}"


def _patent(
    value: str, links: list[str], *, pending: bool
) -> dict[str, str]:
    value = _clean_text(value)
    match = re.match(
        r"(?P<title>.+?)\.\s+(?P<identifiers>(?:US|WO)\s+.+?)\s+"
        r"\([^)]*\b(?P<year>(?:19|20)\d{2})\b[^)]*\)\.?$",
        value,
    )
    if not match:
        raise ValueError(f"Cannot parse patent record: {value}")
    identifiers = match.group("identifiers").replace(";US", "; US")
    kind = "Published patent application" if pending else "Issued patent family"
    return {
        "authors": "Jot, Jean-Marc; et al.",
        "year": match.group("year"),
        "title": match.group("title"),
        "venue": f"{kind}: {identifiers}",
        "url": links[0] if links else _patent_url(identifiers),
        "source": "Patent record",
    }


def _record_key(record: dict[str, str]) -> tuple[str, str, str]:
    return (_plain(record["title"]), record["year"], _plain(record["venue"]))


def _is_reverb_relevant(record: dict[str, str], *, patent: bool) -> bool:
    title = _plain(record["title"])
    if patent:
        return any(
            phrase in title
            for phrase in (
                "reverber",
                "acoustical quality of a room",
                "simulation of complex audio environments",
                "augmented reality headphone environment rendering",
                "direct diffuse decomposition",
                "environment acoustics persistence",
                "spatial audio encoding and reproduction of diffuse sound",
                "spatial audio for interactive audio environments",
                "spatial audio scene description and rendering",
            )
        )

    direct_phrases = (
        "reverber",
        "digital delay networks",
        "concert hall simulation",
        "distance rendering",
        "environmental audio",
        "environmental spatialization",
        "sound propagation",
        "virtual acoustic",
        "virtual audio environment",
        "complex acoustic scenes",
        "diffuse field components",
        "room adaptive",
        "spat introduction",
        "spat reference manual",
        "spat a spatial processor",
        "le spatialisateur",
        "spatialisateur de sons",
    )
    if any(phrase in title for phrase in direct_phrases):
        return True

    related_titles = {
        "advanced audio bifs environmental spatialization of audio in mpeg 4 version 2",
        "binaural simulation of complex acoustic scenes for interactive audio",
        "creating and distributing immersive audio experiences",
        "creating and distributing immersive audio from ircam spat to acoustic objects",
        "digital signal processing issues in the context of binaural and transaural stereophony",
        "perceptually motivated spatial audio scene description and rendering for 6 dof immersive music experiences",
        "real time spatial processing of sounds for music multimedia and interactive human computer interfaces",
        "rendering spatial sound for interoperable experiences in the audio metaverse",
        "spatial audio rendering for interoperable xr applications in the audio metaverse",
        "spatial sound rendering for audio augmented reality",
        "6 dof spatial audio rendering of complex scenes",
    }
    return title in related_titles


def _existing_keys(text: str) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    pattern = re.compile(
        r"(?m)^\*\*\[[^]]+\]\*\* .+? \((?P<year>(?:19|20)\d{2})\)\. "
        r"(?P<title>.+?)\. \*(?P<venue>[^*]+)\*\."
    )
    for match in pattern.finditer(text):
        keys.add(
            (
                _plain(match.group("title")),
                match.group("year"),
                _plain(match.group("venue")),
            )
        )
    return keys


def _next_foundational_id(text: str) -> int:
    values = [int(value) for value in re.findall(r"\*\*\[F(\d+)\]\*\*", text)]
    return max(values, default=0) + 1


def _format(record: dict[str, str], reference_id: int) -> str:
    return (
        f"**[F{reference_id}]** {record['authors']} ({record['year']}). "
        f"{record['title']}. *{record['venue']}*. "
        f"Source: [{record['source']}]({record['url']})"
    )


def _update_counts(text: str) -> str:
    ids = re.findall(r"(?m)^\*\*\[([^]]+)\]\*\*", text)
    extended = sum(reference_id.startswith("XREF") for reference_id in ids)
    primary = len(ids) - extended
    total = len(ids)
    text = re.sub(
        r"Total entries: .+",
        f"Total entries: {total:,} "
        f"({primary} curated and primary-source entries + {extended} extended Crossref entries)",
        text,
        count=1,
    )
    text = re.sub(
        r"This unannotated discovery index adds \d+ Crossref-derived references to the "
        r"\d+ (?:hand-curated|curated and primary-source) entries above, bringing the guide bibliography to "
        r"[\d,]+ total entries\.",
        f"This unannotated discovery index adds {extended} Crossref-derived references "
        f"to the {primary} curated and primary-source entries above, bringing the guide "
        f"bibliography to {total:,} total entries.",
        text,
        count=1,
    )
    return text


def _load_html(path: Path | None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    request = urllib.request.Request(OFFICIAL_URL, headers={"User-Agent": "verbx-docs/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-html", type=Path)
    parser.add_argument("--references", type=Path, default=REFERENCES)
    args = parser.parse_args()

    page = _ListItemParser()
    page.feed(_load_html(args.source_html))
    texts = [item[0] for item in page.items]
    publication_start = next(
        index for index, value in enumerate(texts) if value.startswith("J.-M. Jot and A. Chaigne")
    )
    patent_start = next(
        index
        for index, value in enumerate(texts)
        if value.startswith("Method and system for artificial spatialisation")
    )
    pending_start = next(
        index
        for index, value in enumerate(texts)
        if value.startswith("Adaptive environmental noise compensation")
    )

    publications = [
        _publication(value, links)
        for value, links in page.items[publication_start:patent_start]
    ]
    issued_patents = [
        _patent(value, links, pending=False)
        for value, links in page.items[patent_start:pending_start]
    ]
    pending_patents = [
        _patent(value, links, pending=True)
        for value, links in page.items[pending_start:]
    ]
    official_count = len(publications) + len(issued_patents) + len(pending_patents)
    records = [
        record
        for record in publications
        if _is_reverb_relevant(record, patent=False)
    ]
    records.extend(
        record
        for record in issued_patents + pending_patents
        if _is_reverb_relevant(record, patent=True)
    )

    references = args.references.read_text(encoding="utf-8")
    existing = _existing_keys(references)
    existing_title_years = {(title, year) for title, year, _venue_name in existing}
    unique: list[dict[str, str]] = []
    seen_source_records: set[tuple[str, str, str]] = set()
    for record in records:
        key = _record_key(record)
        title_year = (key[0], key[1])
        if title_year in existing_title_years or key in seen_source_records:
            continue
        unique.append(record)
        seen_source_records.add(key)

    next_id = _next_foundational_id(references)
    additions = "\n\n".join(
        _format(record, next_id + offset) for offset, record in enumerate(unique)
    )
    if additions:
        if SECTION_END not in references:
            raise ValueError("Cannot find the end of foundational bibliography section")
        references = references.replace(SECTION_END, f"\n\n{additions}{SECTION_END}", 1)
    references = _update_counts(references)
    args.references.write_text(references, encoding="utf-8")
    print(
        f"Merged {len(unique)} missing records from {len(records)} reverb-relevant "
        f"works ({official_count} official records reviewed); "
        f"{len(records) - len(unique)} relevant records were already present or duplicated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
