"""Built-in preset definitions.

Presets are intentionally conservative defaults and can be treated as
starting points rather than immutable production values.
"""

from __future__ import annotations

import difflib

PresetValue = float | int | bool | str | tuple[float, ...]


DEFAULT_PRESETS: dict[str, dict[str, PresetValue]] = {
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
    "room_model_studio": {
        "engine": "algo",
        "rt60": 1.2,
        "pre_delay_ms": 14.0,
        "wet": 0.32,
        "dry": 0.88,
        "er_geometry": True,
        "er_room_dims_m": (6.0, 8.0, 3.0),
        "er_material": "studio",
        "room_size_macro": -0.25,
        "clarity_macro": 0.35,
        "warmth_macro": 0.12,
        "fdn_lines": 12,
        "fdn_matrix": "householder",
        "normalize_stage": "none",
    },
    "limiter_broadcast_safe": {
        "engine": "algo",
        "rt60": 2.8,
        "wet": 0.42,
        "dry": 0.82,
        "target_lufs": -18.0,
        "target_peak_dbfs": -1.0,
        "limiter": True,
        "limiter_mode": "softsign",
        "limiter_detect": "peak",
        "limiter_threshold_dbfs": -6.0,
        "limiter_ceiling_dbfs": -1.0,
        "limiter_knee_db": 4.0,
        "limiter_lookahead_ms": 2.0,
        "limiter_oversample": 4,
        "output_peak_norm": "target",
        "output_peak_target_dbfs": -1.0,
        "normalize_stage": "post",
    },
    "delivery_long_tail_safe": {
        "engine": "algo",
        "rt60": 24.0,
        "wet": 0.82,
        "dry": 0.28,
        "fdn_lines": 24,
        "fdn_matrix": "hadamard",
        "tail_limit": 12.0,
        "tail_stop_threshold_db": -120.0,
        "tail_stop_hold_ms": 20.0,
        "algo_stream": True,
        "output_container": "w64",
        "output_subtype": "float32",
        "limiter": True,
        "limiter_ceiling_dbfs": -1.0,
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


_SPACE_TEMPLATES: tuple[tuple[str, float, float, float, float, float, float, float], ...] = (
    ("closet", 0.32, 2.0, 0.12, 0.98, 0.72, 0.78, -0.95),
    ("booth", 0.45, 4.0, 0.16, 0.96, 0.68, 0.86, -0.82),
    ("control_room", 0.62, 6.0, 0.20, 0.94, 0.62, 0.94, -0.68),
    ("tracking_room", 0.85, 9.0, 0.25, 0.90, 0.58, 1.02, -0.48),
    ("drum_room", 1.10, 12.0, 0.30, 0.86, 0.54, 1.10, -0.30),
    ("live_room", 1.40, 15.0, 0.34, 0.82, 0.50, 1.18, -0.12),
    ("studio_a", 1.75, 18.0, 0.38, 0.78, 0.48, 1.24, 0.04),
    ("plate_room", 2.20, 21.0, 0.42, 0.74, 0.44, 1.30, 0.16),
    ("chamber", 2.80, 26.0, 0.48, 0.68, 0.40, 1.38, 0.28),
    ("scoring_stage", 3.60, 32.0, 0.54, 0.60, 0.36, 1.48, 0.42),
    ("small_hall", 4.80, 38.0, 0.60, 0.52, 0.32, 1.56, 0.58),
    ("concert_hall", 6.50, 48.0, 0.66, 0.44, 0.28, 1.66, 0.72),
    ("cathedral", 10.0, 68.0, 0.74, 0.34, 0.24, 1.78, 0.88),
    ("cavern", 14.0, 88.0, 0.82, 0.24, 0.20, 1.92, 1.00),
)

_STYLE_TEMPLATES: tuple[tuple[str, float, float, float, str, int, dict[str, PresetValue]], ...] = (
    ("clean", 0.30, 0.00, 0.00, "hadamard", 8, {}),
    ("dark", -0.05, 0.42, 0.08, "householder", 10, {"fdn_rt60_tilt": -0.28}),
    ("bright", 0.42, -0.16, 0.02, "hadamard", 10, {"fdn_rt60_tilt": 0.24}),
    ("warm", 0.08, 0.46, 0.18, "householder", 12, {"fdn_rt60_tilt": -0.12}),
    ("wide", 0.04, 0.14, 0.48, "random_orthogonal", 14, {}),
    ("vocal", 0.48, 0.18, -0.12, "householder", 10, {"duck": True, "duck_strength": 0.32}),
    ("drum", 0.34, -0.08, 0.08, "hadamard", 12, {"allpass_stages": 4}),
    ("guitar", 0.22, 0.26, 0.10, "householder", 10, {"lowcut": 90.0}),
    ("piano", 0.28, 0.20, 0.26, "random_orthogonal", 14, {"highcut": 14_000.0}),
    ("orchestral", 0.02, 0.22, 0.54, "random_orthogonal", 18, {}),
    (
        "ambient",
        -0.24,
        0.30,
        0.72,
        "tv_unitary",
        20,
        {"fdn_tv_rate_hz": 0.18, "fdn_tv_depth": 0.08},
    ),
    (
        "shimmer",
        -0.34,
        0.16,
        0.80,
        "tv_unitary",
        18,
        {"shimmer": True, "shimmer_mix": 0.26, "shimmer_feedback": 0.36},
    ),
    (
        "dream",
        -0.18,
        0.34,
        0.84,
        "tv_unitary",
        22,
        {"fdn_tv_rate_hz": 0.12, "fdn_tv_depth": 0.12},
    ),
    ("lofi", -0.10, 0.52, -0.08, "circulant", 8, {"highcut": 7_500.0, "lowcut": 160.0}),
    ("vintage", 0.06, 0.36, 0.16, "circulant", 10, {"allpass_gain": 0.64}),
    ("cinematic", -0.08, 0.24, 0.68, "random_orthogonal", 24, {"bloom": 0.45, "bloom_mix": 0.18}),
    ("dense", 0.12, 0.18, 0.36, "random_orthogonal", 24, {"allpass_stages": 8}),
    ("sparse", 0.16, 0.06, -0.06, "hadamard", 8, {"fdn_sparse": True, "fdn_sparse_degree": 2}),
    (
        "modulated",
        -0.06,
        0.20,
        0.50,
        "tv_unitary",
        16,
        {"fdn_tv_rate_hz": 0.28, "fdn_tv_depth": 0.14},
    ),
    ("infinite", -0.42, 0.28, 0.90, "tv_unitary", 24, {"freeze": True, "tail_limit": 30.0}),
)


def _build_expanded_reverb_presets() -> dict[str, dict[str, PresetValue]]:
    """Create a broad deterministic preset bank from curated space/style templates."""
    presets: dict[str, dict[str, PresetValue]] = {}
    for (
        space,
        base_rt60,
        base_pre_delay_ms,
        base_wet,
        base_dry,
        base_damping,
        base_width,
        room_macro,
    ) in _SPACE_TEMPLATES:
        for (
            style,
            clarity_delta,
            warmth_delta,
            envelopment_delta,
            matrix,
            lines,
            extras,
        ) in _STYLE_TEMPLATES:
            name = f"{style}_{space}"
            is_infinite = bool(extras.get("freeze", False))
            rt60_scale = 1.35 if style in {"ambient", "shimmer", "dream", "cinematic"} else 1.0
            rt60_scale = 2.1 if is_infinite else rt60_scale
            wet_offset = 0.10 if style in {"ambient", "shimmer", "dream", "infinite"} else 0.0
            dry_offset = -0.12 if style in {"ambient", "shimmer", "dream", "infinite"} else 0.0
            presets[name] = {
                "engine": "algo",
                "rt60": round(base_rt60 * rt60_scale, 3),
                "pre_delay_ms": round(base_pre_delay_ms, 3),
                "wet": round(min(1.0, max(0.05, base_wet + wet_offset)), 3),
                "dry": round(min(1.0, max(0.0, base_dry + dry_offset)), 3),
                "damping": round(min(0.95, max(0.08, base_damping + (warmth_delta * 0.08))), 3),
                "width": round(min(2.0, max(0.55, base_width + (envelopment_delta * 0.20))), 3),
                "fdn_matrix": matrix,
                "fdn_lines": max(4, lines),
                "room_size_macro": round(min(1.0, max(-1.0, room_macro)), 3),
                "clarity_macro": round(min(1.0, max(-1.0, clarity_delta)), 3),
                "warmth_macro": round(min(1.0, max(-1.0, warmth_delta)), 3),
                "envelopment_macro": round(min(1.0, max(-1.0, envelopment_delta)), 3),
                "fdn_tonal_correction_strength": 0.25,
                "normalize_stage": "none",
                **extras,
            }
    return presets


DEFAULT_PRESETS.update(_build_expanded_reverb_presets())


def preset_names() -> list[str]:
    """Return sorted preset names for CLI display."""
    return sorted(DEFAULT_PRESETS)


def normalize_preset_name(value: str) -> str:
    """Normalize preset token into catalog key style."""
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in token:
        token = token.replace("__", "_")
    return token


def resolve_preset(value: str) -> tuple[str, dict[str, PresetValue]]:
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
