from dataclasses import dataclass


@dataclass
class ReverbPreset:
    name: str
    description: str
    engine_type: str = "algo"  # algo or conv
    rt60: float = 2.0
    wet: float = 0.5
    dry: float = 0.5
    repeat: int = 1
    freeze: bool = False


DEFAULT_PRESETS = {
    "cathedral": ReverbPreset(
        name="cathedral",
        description="Large space with long decay",
        rt60=4.5,
        wet=0.7,
        dry=0.3,
    ),
    "ambience": ReverbPreset(
        name="ambience",
        description="Short decay for presence",
        rt60=0.8,
        wet=0.3,
        dry=0.8,
    ),
    "freeze_drone": ReverbPreset(
        name="freeze_drone",
        description="Infinite sustain texture",
        rt60=10.0,
        wet=1.0,
        dry=0.0,
        freeze=True,
    ),
    "cascade": ReverbPreset(
        name="cascade",
        description="Repeated reverb for density",
        rt60=2.0,
        wet=0.6,
        dry=0.4,
        repeat=3,
    ),
}


def get_preset(name: str) -> ReverbPreset | None:
    return DEFAULT_PRESETS.get(name)


def list_presets() -> list[str]:
    return list(DEFAULT_PRESETS.keys())
