#!/usr/bin/env python3
"""Expand docs/REFERENCES.md with an unannotated Crossref literature index."""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "docs" / "REFERENCES.md"
SECTION_MARKER = "## Section 10: Extended Crossref Literature Index"
DEFAULT_TARGET_TOTAL = 1026
DEFAULT_CURATED_TOTAL = 126
GENERATED_ON = "May 22, 2026"
USER_AGENT = "verbx-reference-expander/1.0 (https://github.com/TheColby/verbx)"
TITLE_SMALL_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "from",
    "in",
    "into",
    "nor",
    "of",
    "on",
    "or",
    "over",
    "per",
    "the",
    "to",
    "using",
    "v",
    "via",
    "with",
    "within",
}
UPPERCASE_TERMS = {
    "3D",
    "AI",
    "API",
    "ASR",
    "DSP",
    "EDC",
    "FDN",
    "FFT",
    "FIR",
    "FM",
    "HRTF",
    "IEEE",
    "IR",
    "ISO",
    "JSON",
    "LUFS",
    "ML",
    "RMS",
    "RT60",
    "SIMD",
    "SNR",
    "T20",
    "T30",
    "T60",
    "TOC",
    "WAV",
    "XRUN",
}
VENUE_TERMS = {
    "ACM",
    "AES",
    "APSURSI",
    "ELNANO",
    "EUSIPCO",
    "ICASSP",
    "INTER-NOISE",
    "INTERSPEECH",
    "JAES",
    "MTS",
    "OCEANS",
    "TELFOR",
    "UKRCON",
    "WASPAA",
}

QUERIES: tuple[tuple[str, str], ...] = (
    ("Room reverberation and room acoustics", "room acoustics reverberation time RT60"),
    ("Artificial reverberation and FDNs", "artificial reverberation feedback delay network"),
    ("Impulse responses and convolution reverb", "room impulse response convolution reverberation"),
    (
        "Speech dereverberation and clarity",
        "speech dereverberation reverberation direct-to-reverberant ratio",
    ),
    ("Spatial audio and Ambisonics", "spatial audio ambisonics reverberation auralization"),
    ("Perceptual reverberation", "perception reverberation listener envelopment clarity"),
    ("Auralization and virtual acoustics", "auralization virtual acoustics reverberation"),
    (
        "Acoustic measurement and decay metrics",
        "energy decay curve reverberation time acoustic measurement",
    ),
    (
        "Acoustic simulation and image-source models",
        "image source method room acoustic simulation reverberation",
    ),
    (
        "Late reverberation modeling",
        "late reverberation statistical model acoustic signal processing",
    ),
)

RELEVANCE_TERMS: tuple[str, ...] = (
    "reverb",
    "reverber",
    "dereverb",
    "acoustic",
    "sound field",
    "impulse response",
    "room",
    "aural",
    "ambison",
    "feedback delay",
    "fdn",
    "echo",
    "clarity",
    "envelopment",
    "reflection",
    "spatial audio",
    "binaural",
    "direct-to-reverberant",
    "rt60",
    "t60",
)


def _strip_existing_extended_section(text: str) -> str:
    idx = text.find(SECTION_MARKER)
    if idx == -1:
        return text.rstrip() + "\n"
    return text[:idx].rstrip() + "\n"


def _existing_dois(text: str) -> set[str]:
    return {doi.lower() for doi in re.findall(r"https://doi\.org/([^)\s]+)", text)}


def _clean(value: str) -> str:
    value = html.unescape(value)
    value = value.replace("o\u0338", "ø").replace("O\u0338", "Ø")
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip().replace("|", "\\|")


def _has_lowercase(value: str) -> bool:
    return any(char.islower() for char in value)


def _looks_all_caps(value: str) -> bool:
    letters = [char for char in value if char.isalpha()]
    if len(letters) < 6:
        return False
    uppercase = sum(1 for char in letters if char.isupper())
    return uppercase / len(letters) > 0.85 and not _has_lowercase(value)


def _case_word(word: str, *, title_word: bool) -> str:
    if not word:
        return word
    if any(char.isdigit() for char in word) and any(char.isalpha() for char in word):
        return word.upper()
    upper = word.upper()
    if upper in UPPERCASE_TERMS or upper in VENUE_TERMS:
        return upper
    if "." in word and len(word) <= 6:
        return word.upper()
    lower = word.lower()
    if not title_word and lower in TITLE_SMALL_WORDS:
        return lower
    if "-" in lower:
        return "-".join(_case_word(part, title_word=True) for part in lower.split("-"))
    if "'" in lower:
        head, tail = lower.split("'", 1)
        return f"{head.capitalize()}'{tail}"
    return lower.capitalize()


def _title_case_text(value: str) -> str:
    if not _looks_all_caps(value):
        return value

    parts = re.split(r"(\s+)", value)
    word_indexes = [idx for idx, part in enumerate(parts) if part.strip()]
    if not word_indexes:
        return value
    first_word = word_indexes[0]
    last_word = word_indexes[-1]

    for idx in word_indexes:
        part = parts[idx]
        match = re.match(r"^([^A-Za-z0-9]*)([A-Za-z0-9][A-Za-z0-9'.-]*)([^A-Za-z0-9]*)$", part)
        if not match:
            continue
        prefix, word, suffix = match.groups()
        title_word = idx in {first_word, last_word} or prefix.endswith(
            ("(", "[", "{", ":", "—", "-")
        )
        parts[idx] = f"{prefix}{_case_word(word, title_word=title_word)}{suffix}"

    output = "".join(parts)
    output = re.sub(r"\bV\s+([A-Z0-9])", r"v \1", output)
    output = re.sub(r"\bDe-Reverberation\b", "De-Reverberation", output)
    return output


def _name_case_text(value: str) -> str:
    parts: list[str] = []
    for token in re.split(r"(\s+|;\s*)", value):
        if not token or token.isspace() or token.strip() == ";":
            parts.append(token)
            continue
        if token.lower() == "et al.":
            parts.append("et al.")
            continue
        stripped = token.strip(",")
        if len(stripped) <= 3 and stripped.isalpha():
            parts.append(token)
            continue
        parts.append(
            re.sub(
                r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:-[A-Za-zÀ-ÖØ-öø-ÿ]+)*",
                _case_name_match,
                token,
            )
        )
    return "".join(parts)


def _case_name_match(match: re.Match[str]) -> str:
    name = match.group(0)
    letters = [char for char in name if char.isalpha()]
    if len(letters) <= 3:
        return name
    if not any(char.islower() for char in letters):
        return "-".join(piece.capitalize() for piece in name.lower().split("-"))
    return name


def _normalize_entry_casing(entry: dict[str, str]) -> dict[str, str]:
    entry = dict(entry)
    entry["authors"] = _name_case_text(entry["authors"])
    entry["title"] = _title_case_text(entry["title"])
    entry["container"] = _title_case_text(entry["container"])
    return entry


def _first(value: Any) -> str:
    if isinstance(value, list) and value:
        return _clean(str(value[0]))
    if isinstance(value, str):
        return _clean(value)
    return ""


def _year(item: dict[str, Any]) -> str:
    for key in ("published-print", "published-online", "issued", "created"):
        parts = item.get(key, {}).get("date-parts")
        if parts and parts[0]:
            return str(parts[0][0])
    return "n.d."


def _authors(item: dict[str, Any]) -> str:
    authors = item.get("author") or []
    names: list[str] = []
    for author in authors[:3]:
        family = _clean(str(author.get("family", "")))
        given = _clean(str(author.get("given", "")))
        if family and given:
            names.append(f"{family}, {given}")
        elif family:
            names.append(family)
        elif given:
            names.append(given)
    if not names:
        return "Unknown authors"
    if len(authors) > 3:
        names.append("et al.")
    return "; ".join(names)


def _request(query: str, offset: int, rows: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "query.bibliographic": query,
            "filter": "from-pub-date:1950",
            "rows": str(rows),
            "offset": str(offset),
            "select": "DOI,title,author,issued,published-print,published-online,created,container-title,type",
        }
    )
    request = urllib.request.Request(
        f"https://api.crossref.org/works?{params}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return list(payload.get("message", {}).get("items", []))


def _is_relevant(title: str, container: str) -> bool:
    haystack = f"{title} {container}".lower()
    return any(term in haystack for term in RELEVANCE_TERMS)


def _collect_entries(target_count: int, existing_dois: set[str]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen = set(existing_dois)
    max_offset_per_query = 2500

    for category, query in QUERIES:
        offset = 0
        while len(entries) < target_count and offset < max_offset_per_query:
            items = _request(query, offset=offset, rows=100)
            if not items:
                break
            for item in items:
                doi = _clean(str(item.get("DOI", "")))
                if not doi:
                    continue
                doi_key = doi.lower()
                if doi_key in seen:
                    continue
                title = _first(item.get("title"))
                if not title:
                    continue
                container = _first(item.get("container-title")) or "Unspecified venue"
                if not _is_relevant(title, container):
                    continue
                seen.add(doi_key)
                entries.append(
                    _normalize_entry_casing(
                        {
                            "category": category,
                            "authors": _authors(item),
                            "year": _year(item),
                            "title": title,
                            "container": container,
                            "doi": doi,
                        }
                    )
                )
                if len(entries) >= target_count:
                    break
            offset += len(items)
            time.sleep(0.1)

    return entries[:target_count]


def _format_entry(index: int, entry: dict[str, str]) -> str:
    return (
        f"**[XREF{index:04d}]** {entry['authors']} ({entry['year']}). "
        f"{entry['title']}. *{entry['container']}*. "
        f"DOI: [{entry['doi']}](https://doi.org/{entry['doi']})"
    )


def _reference_sort_key(entry: dict[str, str]) -> tuple[str, ...]:
    first_author = entry["authors"].split(";", 1)[0].strip()
    surname = first_author.split(",", 1)[0].strip()

    def plain(value: str) -> str:
        value = unicodedata.normalize("NFKD", value)
        value = "".join(char for char in value if not unicodedata.combining(char))
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

    return (
        plain(entry["category"]),
        plain(surname),
        plain(first_author),
        plain(entry["year"]),
        plain(entry["title"]),
    )


def _extended_section(entries: list[dict[str, str]], curated_total: int) -> str:
    lines = [
        SECTION_MARKER,
        "",
        (
            f"This unannotated discovery index adds {len(entries)} Crossref-derived "
            f"references to the {curated_total} hand-curated entries above, bringing "
            f"the guide bibliography to {curated_total + len(entries):,} total entries."
        ),
        "",
        (
            "The entries below are intentionally not treated as vetted design authority. "
            "They are included to make the PDF a much broader literature map for "
            "reverberation, dereverberation, spatial audio, room acoustics, and related "
            "acoustic measurement work. Use the annotated sections above for canonical "
            "implementation guidance."
        ),
        "",
        f"Source: Crossref Works API metadata, generated {GENERATED_ON}.",
        "",
        "Discovery queries:",
        "",
    ]
    lines.extend(f"- {label}: `{query}`" for label, query in QUERIES)
    lines.extend(["", "### Extended Bibliography Entries", ""])
    current_category = ""
    for index, entry in enumerate(sorted(entries, key=_reference_sort_key), 1):
        if entry["category"] != current_category:
            current_category = entry["category"]
            lines.extend(["", f"#### {current_category}", ""])
        lines.append(_format_entry(index, entry))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _update_total(text: str, curated_total: int, extended_count: int) -> str:
    total = curated_total + extended_count
    replacement = (
        f"Total entries: {total:,} "
        f"({curated_total} curated annotated entries + {extended_count} extended Crossref entries)"
    )
    return re.sub(r"Total entries: .+", replacement, text, count=1)


REFERENCE_LINE_PATTERN = re.compile(
    r"^(\*\*\[[^\]]+\]\*\* )(.+?) \(([^)]+)\)\. (.+)\. \*([^*]+)\*\.( DOI: .+)$"
)


def _normalize_reference_line(line: str) -> str:
    match = REFERENCE_LINE_PATTERN.match(line)
    if not match:
        return line
    prefix, authors, year, title, container, doi = match.groups()
    return (
        f"{prefix}{_name_case_text(authors)} ({year}). "
        f"{_title_case_text(title)}. *{_title_case_text(container)}*.{doi}"
    )


def _normalize_existing_references(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    changed = 0
    output: list[str] = []
    for line in lines:
        normalized = _normalize_reference_line(line)
        if normalized != line:
            changed += 1
        output.append(normalized)
    path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-total", type=int, default=DEFAULT_TARGET_TOTAL)
    parser.add_argument("--curated-total", type=int, default=DEFAULT_CURATED_TOTAL)
    parser.add_argument("--out", type=Path, default=REFERENCES)
    parser.add_argument(
        "--normalize-existing",
        action="store_true",
        help="Normalize casing in the existing references file without querying Crossref.",
    )
    args = parser.parse_args()

    if args.normalize_existing:
        changed = _normalize_existing_references(args.out)
        print(f"Normalized {changed} reference lines in {args.out}")
        return 0

    source = REFERENCES.read_text(encoding="utf-8")
    base = _strip_existing_extended_section(source)
    target_extended = max(0, args.target_total - args.curated_total)
    entries = _collect_entries(target_extended, _existing_dois(base))
    if len(entries) != target_extended:
        raise RuntimeError(f"expected {target_extended} entries, collected {len(entries)}")

    output = _update_total(base, args.curated_total, len(entries))
    output = output.rstrip() + "\n\n---\n\n" + _extended_section(entries, args.curated_total)
    args.out.write_text(output, encoding="utf-8")
    print(f"Wrote {args.out} with {args.curated_total + len(entries):,} total entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
