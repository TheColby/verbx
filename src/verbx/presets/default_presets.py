"""Built-in preset definitions."""

from __future__ import annotations

DEFAULT_PRESETS: dict[str, dict[str, float | int | bool]] = {
    "cathedral_extreme": {
        "rt60": 90.0,
        "wet": 0.9,
        "dry": 0.1,
        "repeat": 1,
        "freeze": False,
    },
    "frozen_ambient": {
        "rt60": 120.0,
        "wet": 1.0,
        "dry": 0.0,
        "repeat": 2,
        "freeze": True,
    },
    "tight_plate": {
        "rt60": 2.5,
        "wet": 0.35,
        "dry": 0.8,
        "repeat": 1,
        "freeze": False,
    },
}


def preset_names() -> list[str]:
    """Return sorted preset names."""
    return sorted(DEFAULT_PRESETS)
