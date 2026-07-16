#!/usr/bin/env python3
"""Cache Crossref volume, issue, and page metadata for the guide bibliography."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REFERENCES = ROOT / "docs" / "REFERENCES.md"
DEFAULT_OUT = ROOT / "docs" / "reference_metadata.json"
USER_AGENT = "verbx-reference-typesetter/1.0 (https://github.com/TheColby/verbx)"


def _dois() -> list[str]:
    text = REFERENCES.read_text(encoding="utf-8")
    return sorted(set(re.findall(r"https://doi\.org/([^)\s]+)", text)), key=str.lower)


def _clean(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _fetch(doi: str, attempts: int = 3) -> tuple[str, dict[str, str]]:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=30) as response:
                item = json.loads(response.read().decode("utf-8"))["message"]
            return doi, {
                "volume": _clean(item.get("volume")),
                "issue": _clean(item.get("issue")),
                "page": _clean(item.get("page") or item.get("article-number")),
                "publisher": _clean(item.get("publisher")),
            }
        except Exception as exc:  # Network and incomplete DOI registries are both expected.
            if attempt + 1 == attempts:
                return doi, {"error": type(exc).__name__}
            time.sleep(0.35 * (attempt + 1))
    return doi, {"error": "unknown"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    existing: dict[str, dict[str, str]] = {}
    if args.out.exists():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
    pending = [
        doi for doi in _dois()
        if doi.lower() not in existing or "error" in existing[doi.lower()]
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(_fetch, doi) for doi in pending]
        for number, future in enumerate(as_completed(futures), 1):
            doi, metadata = future.result()
            existing[doi.lower()] = metadata
            if number % 100 == 0:
                print(f"Fetched {number}/{len(pending)} uncached DOI records")

    ordered = {key: existing[key] for key in sorted(existing)}
    args.out.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    complete = sum(1 for value in ordered.values() if "error" not in value)
    print(f"Wrote {args.out} with {complete}/{len(ordered)} resolved DOI records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
