"""Centralized control-target registry for automation and macro mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ControlTargetDomain = Literal["post", "engine", "conv"]


@dataclass(frozen=True, slots=True)
class ControlTargetSpec:
    """Definition of one controllable automation/parameter target."""

    name: str
    domain: ControlTargetDomain
    minimum: float
    maximum: float
    aliases: tuple[str, ...] = ()


def _normalize_token(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


_CONTROL_TARGET_SPECS: tuple[ControlTargetSpec, ...] = (
    ControlTargetSpec("wet", "post", 0.0, 1.0),
    ControlTargetSpec("dry", "post", 0.0, 1.0),
    ControlTargetSpec(
        "gain-db",
        "post",
        -48.0,
        24.0,
        aliases=("output-gain-db", "gain", "gaindb"),
    ),
    ControlTargetSpec("rt60", "engine", 0.1, 300.0, aliases=("t60",)),
    ControlTargetSpec("damping", "engine", 0.0, 1.0),
    ControlTargetSpec(
        "room-size",
        "engine",
        0.25,
        4.0,
        aliases=("room", "roomsize", "size"),
    ),
    ControlTargetSpec(
        "room-size-macro",
        "engine",
        -1.0,
        1.0,
        aliases=("room-macro", "roomsize-macro", "size-macro"),
    ),
    ControlTargetSpec(
        "clarity-macro",
        "engine",
        -1.0,
        1.0,
        aliases=("clarity",),
    ),
    ControlTargetSpec(
        "warmth-macro",
        "engine",
        -1.0,
        1.0,
        aliases=("warmth",),
    ),
    ControlTargetSpec(
        "envelopment-macro",
        "engine",
        -1.0,
        1.0,
        aliases=("envelopment", "enveloping", "envelope"),
    ),
    ControlTargetSpec(
        "fdn-rt60-tilt",
        "engine",
        -1.0,
        1.0,
        aliases=("rt60-tilt", "fdn-tilt"),
    ),
    ControlTargetSpec(
        "fdn-tonal-correction-strength",
        "engine",
        0.0,
        1.0,
        aliases=("tonal-correction", "tonal-correction-strength", "fdn-tonal-correction"),
    ),
    ControlTargetSpec(
        "ir-blend-alpha",
        "conv",
        0.0,
        1.0,
        aliases=("blend-alpha", "ir-alpha", "morph-alpha", "ir-morph-alpha"),
    ),
)

_CONTROL_TARGET_BY_NAME = {spec.name: spec for spec in _CONTROL_TARGET_SPECS}
_CONTROL_TARGET_ALIASES: dict[str, str] = {}
for spec in _CONTROL_TARGET_SPECS:
    _CONTROL_TARGET_ALIASES[_normalize_token(spec.name)] = spec.name
    for alias in spec.aliases:
        _CONTROL_TARGET_ALIASES[_normalize_token(alias)] = spec.name

POST_RENDER_CONTROL_TARGETS = frozenset(
    spec.name for spec in _CONTROL_TARGET_SPECS if spec.domain == "post"
)
ENGINE_CONTROL_TARGETS = frozenset(
    spec.name for spec in _CONTROL_TARGET_SPECS if spec.domain == "engine"
)
CONV_CONTROL_TARGETS = frozenset(
    spec.name for spec in _CONTROL_TARGET_SPECS if spec.domain == "conv"
)
SUPPORTED_CONTROL_TARGETS = frozenset(spec.name for spec in _CONTROL_TARGET_SPECS)
CONTROL_TARGET_LIMITS: dict[str, tuple[float, float]] = {
    spec.name: (float(spec.minimum), float(spec.maximum))
    for spec in _CONTROL_TARGET_SPECS
}


def normalize_control_target_name(value: str) -> str:
    """Normalize aliases to canonical control-target identifiers."""
    normalized = _normalize_token(value)
    return _CONTROL_TARGET_ALIASES.get(normalized, normalized)


def get_control_target_spec(value: str) -> ControlTargetSpec | None:
    """Return canonical target spec by name/alias, if known."""
    canonical = normalize_control_target_name(value)
    return _CONTROL_TARGET_BY_NAME.get(canonical)
