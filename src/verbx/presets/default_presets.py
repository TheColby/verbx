"""Built-in preset definitions.

Presets are intentionally conservative defaults and can be treated as
starting points rather than immutable production values.
"""

from __future__ import annotations

import difflib

DEFAULT_PRESETS: dict[str, dict[str, float | int | bool | str]] = {
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
    "shimmer_wash": {
        "rt60": 80.0,
        "wet": 0.9,
        "dry": 0.1,
        "repeat": 2,
        "shimmer": True,
        "shimmer_semitones": 12.0,
        "shimmer_mix": 0.4,
        "shimmer_feedback": 0.5,
    },
    "ducked_bloom": {
        "rt60": 70.0,
        "wet": 0.75,
        "dry": 0.35,
        "repeat": 1,
        "duck": True,
        "duck_attack": 15.0,
        "duck_release": 240.0,
        "bloom": 2.0,
        "tilt": 1.5,
    },
    "targeted_air": {
        "rt60": 95.0,
        "wet": 0.85,
        "dry": 0.15,
        "repeat": 2,
        "target_lufs": -18.0,
        "target_peak_dbfs": -1.0,
        "normalize_stage": "post",
    },
    "perceptual_small_room_regression": {
        "engine": "algo",
        "rt60": 0.65,
        "pre_delay_ms": 6.0,
        "wet": 0.22,
        "dry": 0.98,
        "damping": 0.62,
        "width": 0.92,
        "fdn_matrix": "hadamard",
        "fdn_lines": 8,
        "room_size_macro": -0.65,
        "clarity_macro": 0.58,
        "warmth_macro": 0.12,
        "envelopment_macro": -0.25,
        "fdn_rt60_tilt": 0.08,
        "fdn_tonal_correction_strength": 0.28,
        "normalize_stage": "none",
    },
    "perceptual_mid_room_regression": {
        "engine": "algo",
        "rt60": 2.1,
        "pre_delay_ms": 18.0,
        "wet": 0.38,
        "dry": 0.86,
        "damping": 0.48,
        "width": 1.22,
        "fdn_matrix": "householder",
        "fdn_lines": 12,
        "room_size_macro": 0.15,
        "clarity_macro": 0.08,
        "warmth_macro": 0.18,
        "envelopment_macro": 0.22,
        "fdn_rt60_tilt": 0.0,
        "fdn_tonal_correction_strength": 0.34,
        "normalize_stage": "none",
    },
    "perceptual_long_hall_regression": {
        "engine": "algo",
        "rt60": 8.0,
        "pre_delay_ms": 42.0,
        "wet": 0.58,
        "dry": 0.68,
        "damping": 0.34,
        "width": 1.44,
        "fdn_matrix": "random_orthogonal",
        "fdn_lines": 18,
        "room_size_macro": 0.65,
        "clarity_macro": -0.22,
        "warmth_macro": 0.34,
        "envelopment_macro": 0.56,
        "fdn_rt60_tilt": -0.18,
        "fdn_tonal_correction_strength": 0.42,
        "normalize_stage": "none",
    },
    "perceptual_extreme_tail_regression": {
        "engine": "algo",
        "rt60": 45.0,
        "pre_delay_ms": 76.0,
        "wet": 0.82,
        "dry": 0.32,
        "damping": 0.24,
        "width": 1.75,
        "fdn_matrix": "tv_unitary",
        "fdn_lines": 24,
        "fdn_tv_rate_hz": 0.35,
        "fdn_tv_depth": 0.18,
        "room_size_macro": 0.95,
        "clarity_macro": -0.48,
        "warmth_macro": 0.45,
        "envelopment_macro": 0.88,
        "fdn_rt60_tilt": -0.34,
        "fdn_tonal_correction_strength": 0.52,
        "normalize_stage": "none",
    },
}


def preset_names() -> list[str]:
    """Return sorted preset names for CLI display."""
    return sorted(DEFAULT_PRESETS)


def normalize_preset_name(value: str) -> str:
    """Normalize preset token into catalog key style."""
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in token:
        token = token.replace("__", "_")
    return token


def resolve_preset(value: str) -> tuple[str, dict[str, float | int | bool | str]]:
    """Resolve preset key and payload or raise a helpful error."""
    normalized = normalize_preset_name(value)
    if normalized in DEFAULT_PRESETS:
        return normalized, dict(DEFAULT_PRESETS[normalized])

    suggestion = difflib.get_close_matches(normalized, preset_names(), n=1, cutoff=0.5)
    options = ", ".join(preset_names())
    if len(suggestion) > 0:
        raise ValueError(
            f"Unknown preset '{value}'. Did you mean '{suggestion[0]}'? "
            f"Available presets: {options}."
        )
    raise ValueError(f"Unknown preset '{value}'. Available presets: {options}.")
