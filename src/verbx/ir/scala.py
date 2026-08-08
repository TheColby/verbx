"""Scala scale parsing and bounded frequency-target expansion."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ScalaScale:
    """Parsed Scala scale with ratios relative to its implicit unison."""

    description: str
    pitch_count: int
    ratios: tuple[float, ...]
    sha256: str

    @property
    def period_ratio(self) -> float:
        """Return the final scale entry, which defines the repeat interval."""

        return self.ratios[-1]


def parse_scala_file(path: Path) -> ScalaScale:
    """Parse one ``.scl`` file and retain a content hash for provenance."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read Scala file {path}: {exc}") from exc

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Scala file must be UTF-8 text: {path}") from exc

    lines = []
    for raw_line in text.splitlines():
        line = raw_line.split("!", 1)[0].strip()
        if line:
            lines.append(line)

    if len(lines) < 3:
        raise ValueError("Scala file must contain a description, pitch count, and pitches")

    description = lines[0]
    try:
        pitch_count = int(lines[1])
    except ValueError as exc:
        raise ValueError(f"Invalid Scala pitch count: {lines[1]!r}") from exc
    if not 1 <= pitch_count <= 4096:
        raise ValueError("Scala pitch count must be between 1 and 4096")

    pitch_lines = lines[2:]
    if len(pitch_lines) != pitch_count:
        raise ValueError(
            f"Scala file declares {pitch_count} pitches but contains {len(pitch_lines)}"
        )

    ratios = tuple(_parse_scala_pitch(line) for line in pitch_lines)
    if any(
        current <= previous
        for previous, current in zip((1.0, *ratios[:-1]), ratios, strict=True)
    ):
        raise ValueError("Scala pitches must be strictly increasing above the implicit unison")
    if ratios[-1] <= 1.0:
        raise ValueError("Scala repeat interval must be greater than unison")

    return ScalaScale(
        description=description,
        pitch_count=pitch_count,
        ratios=ratios,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def resolve_scala_frequencies(
    scale: ScalaScale,
    *,
    root_hz: float,
    root_degree: int,
    low_hz: float,
    high_hz: float,
    max_targets: int,
) -> tuple[float, ...]:
    """Expand scale degrees over repeat intervals inside one frequency range."""

    if not math.isfinite(root_hz) or root_hz <= 0.0:
        raise ValueError("Scala root frequency must be greater than 0 Hz")
    if not math.isfinite(low_hz) or not math.isfinite(high_hz):
        raise ValueError("Scala frequency limits must be finite")
    if low_hz <= 0.0 or high_hz <= low_hz:
        raise ValueError("Scala high frequency must be greater than its low frequency")
    if max_targets < 1:
        raise ValueError("Scala max target count must be at least 1")

    degrees = (1.0, *scale.ratios[:-1])
    if not 0 <= root_degree < len(degrees):
        raise ValueError(
            f"Scala root degree must be between 0 and {len(degrees) - 1} for this scale"
        )

    root_ratio = degrees[root_degree]
    period = scale.period_ratio
    lower_power = math.floor(math.log(low_hz / root_hz, period)) - 1
    upper_power = math.ceil(math.log(high_hz / root_hz, period)) + 1

    targets = {
        round(root_hz * (degree / root_ratio) * (period**power), 12)
        for power in range(lower_power, upper_power + 1)
        for degree in degrees
        if low_hz <= root_hz * (degree / root_ratio) * (period**power) <= high_hz
    }
    ordered = sorted(targets)
    if not ordered:
        raise ValueError("Scala scale produced no frequencies inside the requested range")
    if len(ordered) <= max_targets:
        return tuple(ordered)

    # Preserve logarithmic coverage when high-division scales exceed the DSP budget.
    selected_indices = (
        {
            round(index * (len(ordered) - 1) / (max_targets - 1))
            for index in range(max_targets)
        }
        if max_targets > 1
        else {len(ordered) // 2}
    )
    return tuple(ordered[index] for index in sorted(selected_indices))


def _parse_scala_pitch(value: str) -> float:
    """Convert a Scala cents, integer-ratio, or fractional-ratio entry."""

    token = value.strip()
    try:
        if "/" in token:
            numerator_text, denominator_text = token.split("/", 1)
            ratio = float(numerator_text.strip()) / float(denominator_text.strip())
        elif "." in token or "e" in token.lower():
            ratio = 2.0 ** (float(token) / 1200.0)
        else:
            ratio = float(int(token))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"Invalid Scala pitch entry: {value!r}") from exc

    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(f"Scala pitch must resolve to a positive finite ratio: {value!r}")
    return ratio
