"""Configuration models for CLI options and engine selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EngineName = Literal["conv", "algo", "auto"]
IRNormalize = Literal["peak", "rms", "none"]
NormalizeStage = Literal["none", "post", "per-pass"]


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
    target_lufs: float | None = None
    target_peak_dbfs: float | None = None
    use_true_peak: bool = True
    limiter: bool = True
    normalize_stage: NormalizeStage = "post"
    repeat_target_lufs: float | None = None
    repeat_target_peak_dbfs: float | None = None
    shimmer: bool = False
    shimmer_semitones: float = 12.0
    shimmer_mix: float = 0.25
    shimmer_feedback: float = 0.35
    shimmer_highcut: float | None = 10_000.0
    shimmer_lowcut: float | None = 300.0
    duck: bool = False
    duck_attack: float = 20.0
    duck_release: float = 350.0
    bloom: float = 0.0
    lowcut: float | None = None
    highcut: float | None = None
    tilt: float = 0.0
    analysis_out: str | None = None
    silent: bool = False
    progress: bool = True
