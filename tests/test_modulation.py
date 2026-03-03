from __future__ import annotations

import numpy as np

from verbx.core.modulation import (
    apply_parameter_modulation,
    parse_mod_route_spec,
    parse_mod_sources,
)


def test_parse_mod_sources_multiple_kinds() -> None:
    sources = parse_mod_sources(
        [
            "lfo:sine:0.08:1.0*0.7",
            "env:20:300*0.5",
            "const:0.4",
        ]
    )
    assert len(sources) == 3
    assert sources[0].kind == "lfo"
    assert sources[1].kind == "env"
    assert sources[2].kind == "const"


def test_apply_parameter_modulation_gain_db_target() -> None:
    sr = 16_000
    audio = np.full((1024, 1), 0.2, dtype=np.float32)
    dry = np.zeros((1024, 1), dtype=np.float32)

    modulated, summary = apply_parameter_modulation(
        audio=audio,
        dry_reference=dry,
        sr=sr,
        target="gain-db",
        source_specs=("const:1.0",),
        value_min=-6.0,
        value_max=6.0,
        combine="sum",
        smooth_ms=0.0,
    )

    assert summary is not None
    assert summary["target"] == "gain-db"
    assert float(np.max(np.abs(modulated))) > float(np.max(np.abs(audio)))


def test_parse_mod_route_spec() -> None:
    route = parse_mod_route_spec("wet:0.1:0.9:avg:25:lfo:sine:0.08:1.0*0.7,env:20:350*0.4")
    assert route.target == "wet"
    assert route.combine == "avg"
    assert route.smooth_ms == 25.0
    assert len(route.source_specs) == 2

def test_parse_mod_sources_chaos() -> None:
    sources = parse_mod_sources(["chaos:0.1:0.5*0.8"])
    assert len(sources) == 1
    assert sources[0].kind == "chaos"
    assert sources[0].rate_hz == 0.1
    assert sources[0].depth == 0.5
    assert sources[0].weight == 0.8

def test_chaos_wave_generator() -> None:
    from verbx.core.modulation import _chaos_wave  # pyright: ignore[reportPrivateUsage]
    wave = _chaos_wave(1000, 48_000, 1.0)
    assert wave.shape == (1000,)
    assert wave.dtype == "float32"
