from __future__ import annotations

from verbx.config import RenderConfig
from verbx.presets.default_presets import DEFAULT_PRESETS, preset_names, resolve_preset


def test_builtin_preset_bank_contains_hundreds_of_reverb_presets() -> None:
    assert len(DEFAULT_PRESETS) >= 250
    assert "warm_chamber" in DEFAULT_PRESETS
    assert "shimmer_cathedral" in DEFAULT_PRESETS
    assert "clean_drum_room" in DEFAULT_PRESETS
    assert "infinite_cavern" in DEFAULT_PRESETS


def test_all_builtin_presets_are_render_config_compatible() -> None:
    known_fields = set(RenderConfig.__dataclass_fields__)
    for name, payload in DEFAULT_PRESETS.items():
        unknown = sorted(key for key in payload if key not in known_fields)
        assert unknown == [], f"{name} contains unknown fields: {unknown}"
        RenderConfig(**payload)


def test_generated_preset_names_resolve_from_cli_style_tokens() -> None:
    resolved_name, payload = resolve_preset("Warm Chamber")

    assert resolved_name == "warm_chamber"
    assert payload["engine"] == "algo"
    assert float(payload["rt60"]) > 0.0
    assert "warm_chamber" in preset_names()
