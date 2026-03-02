from __future__ import annotations

from verbx.presets.default_presets import DEFAULT_PRESETS, get_preset, list_presets, preset_names


def test_preset_names_returns_sorted_keys():
    """Verify that preset_names returns all keys in sorted order."""
    names = preset_names()
    assert isinstance(names, list)
    assert len(names) > 0
    assert names == sorted(DEFAULT_PRESETS.keys())


def test_get_existing_preset():
    """Verify that an existing preset can be retrieved with expected keys."""
    preset = get_preset("cathedral_extreme")
    assert preset is not None
    assert isinstance(preset, dict)
    assert preset["rt60"] == 90.0
    assert preset["wet"] == 0.9


def test_get_missing_preset():
    """Verify that retrieving a non-existent preset returns None."""
    preset = get_preset("nonexistent_preset_12345")
    assert preset is None


def test_list_presets():
    """Verify that list_presets returns a list of preset names."""
    names = list_presets()
    assert isinstance(names, list)
    assert len(names) > 0
    assert names == list(DEFAULT_PRESETS.keys())


def test_all_presets_have_required_fields():
    """Verify all built-in presets have basic required configuration fields."""
    for name, preset in DEFAULT_PRESETS.items():
        assert isinstance(name, str)
        assert isinstance(preset, dict)
        assert "rt60" in preset
        assert "wet" in preset
        assert "dry" in preset
        assert "repeat" in preset
