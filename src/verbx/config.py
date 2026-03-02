"""Typed configuration models shared across CLI and pipeline layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EngineName = Literal["conv", "algo", "auto"]
IRNormalize = Literal["peak", "rms", "none"]
NormalizeStage = Literal["none", "post", "per-pass"]
IRMode = Literal["fdn", "stochastic", "modal", "hybrid"]
IRMatrixLayout = Literal["output-major", "input-major"]
DeviceName = Literal["auto", "cpu", "cuda", "mps"]
OutputSubtype = Literal["auto", "float32", "float64", "pcm16", "pcm24", "pcm32"]
OutputPeakNorm = Literal["none", "input", "target", "full-scale"]
ModTarget = Literal["none", "mix", "wet", "gain-db"]
ModCombine = Literal["sum", "avg", "max"]


@dataclass(slots=True)
class RenderConfig:
    """Typed render configuration used by CLI and pipeline.

    Centralizing options in one dataclass reduces drift between CLI parsing,
    validation, and DSP pipeline behavior.
    """

    engine: EngineName = "auto"
    rt60: float = 60.0
    pre_delay_ms: float = 20.0
    damping: float = 0.45
    width: float = 1.0
    mod_depth_ms: float = 2.0
    mod_rate_hz: float = 0.1
    mod_target: ModTarget = "none"
    mod_sources: tuple[str, ...] = ()
    mod_routes: tuple[str, ...] = ()
    mod_min: float = 0.0
    mod_max: float = 1.0
    mod_combine: ModCombine = "sum"
    mod_smooth_ms: float = 20.0
    beast_mode: int = 1
    allpass_stages: int = 6
    allpass_gain: float = 0.7
    allpass_delays_ms: tuple[float, ...] = ()
    comb_delays_ms: tuple[float, ...] = ()
    fdn_lines: int = 8
    wet: float = 0.8
    dry: float = 0.2
    repeat: int = 1
    freeze: bool = False
    start: float | None = None
    end: float | None = None
    block_size: int = 4096
    ir: str | None = None
    self_convolve: bool = False
    ir_normalize: IRNormalize = "peak"
    ir_matrix_layout: IRMatrixLayout = "output-major"
    tail_limit: float | None = None
    threads: int | None = None
    device: DeviceName = "auto"
    partition_size: int = 16_384
    ir_gen: bool = False
    ir_gen_mode: IRMode = "hybrid"
    ir_gen_length: float = 60.0
    ir_gen_seed: int = 0
    ir_gen_cache_dir: str = ".verbx_cache/irs"
    target_lufs: float | None = None
    target_peak_dbfs: float | None = None
    use_true_peak: bool = True
    limiter: bool = True
    normalize_stage: NormalizeStage = "post"
    repeat_target_lufs: float | None = None
    repeat_target_peak_dbfs: float | None = None
    output_subtype: OutputSubtype = "auto"
    output_peak_norm: OutputPeakNorm = "none"
    output_peak_target_dbfs: float | None = None
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
    bpm: float | None = None
    pre_delay_note: str | None = None
    frames_out: str | None = None
    analysis_out: str | None = None
    silent: bool = False
    progress: bool = True
