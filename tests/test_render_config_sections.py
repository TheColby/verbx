from dataclasses import FrozenInstanceError, asdict

import pytest

from verbx.config import (
    EngineSettings,
    ExecutionSettings,
    OutputSettings,
    RenderConfig,
    TailSettings,
)


def test_render_config_exposes_typed_section_snapshots() -> None:
    config = RenderConfig(
        engine="algo",
        algo_model="plate",
        rt60=3.5,
        block_size=2048,
        device="cpu",
        tail_limit=8.0,
        output_subtype="pcm24",
        limiter_mix=0.75,
    )

    sections = config.sections

    assert isinstance(sections.engine, EngineSettings)
    assert sections.engine.algo_model == "plate"
    assert sections.engine.rt60 == 3.5
    assert isinstance(sections.execution, ExecutionSettings)
    assert sections.execution.block_size == 2048
    assert sections.execution.device == "cpu"
    assert isinstance(sections.tail, TailSettings)
    assert sections.tail.limit == 8.0
    assert isinstance(sections.output, OutputSettings)
    assert sections.output.subtype == "pcm24"
    assert sections.output.limiter_mix == 0.75


def test_section_snapshots_are_immutable_and_do_not_go_stale() -> None:
    config = RenderConfig(wet=0.25)
    first = config.engine_settings

    with pytest.raises(FrozenInstanceError):
        first.wet = 0.5  # type: ignore[misc]

    config.wet = 0.75

    assert first.wet == 0.25
    assert config.engine_settings.wet == 0.75


def test_sections_do_not_change_flat_dataclass_serialization() -> None:
    config = RenderConfig(engine="algo", output_container="rf64")

    serialized = asdict(config)

    assert serialized["engine"] == "algo"
    assert serialized["output_container"] == "rf64"
    assert "sections" not in serialized
    assert "engine_settings" not in serialized
    assert set(serialized) == set(RenderConfig.__dataclass_fields__)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rt60": -0.1}, "rt60 must be >= 0"),
        ({"block_size": 0}, "block_size must be >= 1"),
        ({"tail_stop_hold_ms": -1.0}, "tail_stop_hold_ms must be >= 0"),
        ({"limiter_mix": 1.1}, "limiter_mix must be 0-1"),
    ],
)
def test_section_validation_preserves_flat_constructor_errors(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RenderConfig(**overrides)
