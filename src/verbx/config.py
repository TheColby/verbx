from dataclasses import dataclass


@dataclass
class Config:
    """Application Configuration."""

    # Defaults
    sample_rate: int = 44100
    chunk_size: int = 4096


def load_config() -> Config:
    """Load configuration (stub)."""
    # TODO: Load from file or env vars
    return Config()
