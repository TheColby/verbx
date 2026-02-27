"""Configuration models for CLI options and engine selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EngineName = Literal["conv", "algo", "auto"]
IRNormalize = Literal["peak", "rms", "none"]


@dataclass(slots=True)
class RenderConfig:
    """Typed render configuration used by CLI and pipeline."""

    engine: EngineName = "auto"
    rt60: float = 60.0
    pre_delay_ms: float = 20.0
    damping: float = 0.45
    width: float = 1.0
    mod_depth_ms: float = 2.0
    mod_rate_hz: float = 0.1
    wet: float = 0.8
    dry: float = 0.2
    repeat: int = 1
    freeze: bool = False
    start: float | None = None
    end: float | None = None
    block_size: int = 4096
    ir: str | None = None
    ir_normalize: IRNormalize = "peak"
    tail_limit: float | None = None
    threads: int | None = None
    partition_size: int = 16_384
    analysis_out: str | None = None
    silent: bool = False
    progress: bool = True
