"""Tempo parsing helpers for note-synced timing controls.

The parser supports either raw seconds (``"0.125"``) or note-value syntax
(``"1/8"``, ``"1/8D"``, ``"1/8T"``) at a caller-provided BPM.
"""

from __future__ import annotations

import re
from fractions import Fraction

_NOTE_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)([DTdt]?)\s*$")


def parse_note_duration_seconds(value: str, bpm: float) -> float:
    """Parse note durations like ``1/8``, ``1/8D``, or ``1/8T`` into seconds.

    ``D`` means dotted (x1.5), ``T`` means triplet (x2/3). If the token does
    not match note syntax, it is parsed as a literal number of seconds.
    """
    beats_per_second = max(bpm, 1e-6) / 60.0
    quarter_seconds = 1.0 / beats_per_second

    match = _NOTE_RE.match(value)
    if not match:
        try:
            seconds = float(value)
        except ValueError as exc:
            msg = f"Unsupported pre-delay format: {value!r}"
            raise ValueError(msg) from exc
        if seconds < 0.0:
            msg = "Duration must be non-negative"
            raise ValueError(msg)
        return seconds

    num = int(match.group(1))
    den = int(match.group(2))
    modifier = match.group(3).upper()

    frac = Fraction(num, den)
    base_seconds = quarter_seconds * 4.0 * float(frac)

    if modifier == "D":
        base_seconds *= 1.5
    elif modifier == "T":
        base_seconds *= 2.0 / 3.0

    return float(base_seconds)


def parse_pre_delay_ms(value: str | None, bpm: float | None, fallback_ms: float) -> float:
    """Resolve pre-delay into milliseconds with safe defaults.

    ``value=None`` falls back to ``fallback_ms``. Missing BPM defaults to 120.
    """
    if value is None:
        return fallback_ms

    resolved_bpm = 120.0 if bpm is None else bpm
    seconds = parse_note_duration_seconds(value, resolved_bpm)
    return float(max(0.0, seconds * 1000.0))
