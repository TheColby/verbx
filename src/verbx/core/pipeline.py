"""Render pipeline orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.analysis.framewise import write_framewise_csv
from verbx.config import NormalizeStage, RenderConfig
from verbx.core.algo_reverb import AlgoReverbConfig, AlgoReverbEngine
from verbx.core.ambient import apply_ambient_processing
from verbx.core.convolution_reverb import ConvolutionReverbConfig, ConvolutionReverbEngine
from verbx.core.engine_base import ReverbEngine
from verbx.core.freeze import freeze_segment
from verbx.core.loudness import apply_output_targets
from verbx.core.repeat import repeat_process
from verbx.io.audio import (
    peak_normalize,
    read_audio,
    soft_limiter,
    validate_audio_path,
    write_audio,
)
from verbx.io.progress import RenderProgress
from verbx.ir.generator import IRGenConfig, generate_or_load_cached_ir

AudioArray = npt.NDArray[np.float32]
PassProcessor = Callable[[AudioArray, int, int], AudioArray]


def run_render_pipeline(infile: Path, outfile: Path, config: RenderConfig) -> dict[str, Any]:
    """Run the complete render pipeline and return report metadata."""
    validate_audio_path(str(infile))

    with RenderProgress(enabled=(config.progress and not config.silent)) as progress:
        audio, sr = read_audio(str(infile))
        progress.mark_read()

        runtime_config, ir_runtime = _prepare_runtime_config(config, sr, audio.shape[1])
        engine_name, engine = _resolve_engine(runtime_config)
        progress.set_passes(max(1, config.repeat))

        input_for_engine = audio
        if runtime_config.freeze:
            input_for_engine = freeze_segment(
                audio=audio,
                sr=sr,
                start=runtime_config.start,
                end=runtime_config.end,
                mode="loop",
                xfade_ms=100.0,
            )

        tail_padding_seconds = 0.0
        if engine_name == "algo":
            tail_padding_seconds = _algo_tail_padding_seconds(runtime_config)
            input_for_engine = _append_tail_padding(
                audio=input_for_engine,
                sr=sr,
                tail_seconds=tail_padding_seconds,
            )

        repeat_post_processor = _build_per_pass_processor(runtime_config, sr)

        rendered = repeat_process(
            engine=engine,
            audio=input_for_engine,
            sr=sr,
            n=runtime_config.repeat,
            post_pass_processor=repeat_post_processor,
            progress_callback=lambda idx, total: progress.mark_process_pass(idx),
        )

        rendered = apply_ambient_processing(
            wet=rendered,
            dry_reference=input_for_engine,
            sr=sr,
            duck=runtime_config.duck,
            duck_attack=runtime_config.duck_attack,
            duck_release=runtime_config.duck_release,
            bloom=runtime_config.bloom,
            lowcut=runtime_config.lowcut,
            highcut=runtime_config.highcut,
            tilt=runtime_config.tilt,
        )

        if runtime_config.normalize_stage == "post":
            rendered = apply_output_targets(
                rendered,
                sr,
                target_lufs=runtime_config.target_lufs,
                target_peak_dbfs=runtime_config.target_peak_dbfs,
                limiter=runtime_config.limiter,
                use_true_peak=runtime_config.use_true_peak,
            )
        elif runtime_config.normalize_stage == "none" and runtime_config.limiter:
            rendered = soft_limiter(rendered, threshold_dbfs=-1.0, knee_db=6.0)
            rendered = peak_normalize(rendered, target_dbfs=-1.0)

        write_audio(str(outfile), rendered, sr)
        progress.mark_write()

        report: dict[str, Any] = {
            "engine": engine_name,
            "sample_rate": sr,
            "input_samples": int(audio.shape[0]),
            "output_samples": int(rendered.shape[0]),
            "channels": int(rendered.shape[1]),
            "config": asdict(runtime_config),
            "effective": {
                "engine_requested": config.engine,
                "engine_resolved": engine_name,
                "ir_used": runtime_config.ir,
                "tail_padding_seconds": tail_padding_seconds,
                "non_default_settings": _non_default_settings(runtime_config),
            },
        }
        if ir_runtime is not None:
            report["ir_runtime"] = ir_runtime

        if not runtime_config.silent:
            include_loudness = _should_include_loudness(runtime_config)
            analyzer = AudioAnalyzer()
            report["input"] = analyzer.analyze(audio, sr, include_loudness=include_loudness)
            report["output"] = analyzer.analyze(rendered, sr, include_loudness=include_loudness)
            analysis_path = _resolve_analysis_path(outfile, runtime_config.analysis_out)
            analysis_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            report["analysis_path"] = str(analysis_path)
            if runtime_config.frames_out is not None:
                frames_path = Path(runtime_config.frames_out)
                write_framewise_csv(frames_path, rendered, sr)
                report["frames_path"] = str(frames_path)

        progress.mark_analyze()

    return report


def _prepare_runtime_config(
    config: RenderConfig, sr: int, input_channels: int
) -> tuple[RenderConfig, dict[str, Any] | None]:
    runtime = config
    ir_runtime: dict[str, Any] | None = None

    if config.ir_gen:
        ir_cfg = IRGenConfig(
            mode=config.ir_gen_mode,
            length=config.ir_gen_length,
            sr=sr,
            channels=max(1, input_channels),
            seed=config.ir_gen_seed,
            rt60=config.rt60,
            damping=config.damping,
            lowcut=config.lowcut,
            highcut=config.highcut,
            tilt=config.tilt,
            target_lufs=config.target_lufs,
            true_peak=config.use_true_peak,
            mod_depth_ms=config.mod_depth_ms,
            mod_rate_hz=config.mod_rate_hz,
        )
        cache_dir = Path(config.ir_gen_cache_dir)
        _, _, meta, wav_path, cache_hit = generate_or_load_cached_ir(ir_cfg, cache_dir=cache_dir)
        runtime = RenderConfig(**asdict(config))
        runtime.ir = str(wav_path)
        if runtime.engine == "auto":
            runtime.engine = "conv"
        ir_runtime = {
            "mode": config.ir_gen_mode,
            "cache_hit": cache_hit,
            "ir_path": str(wav_path),
            "meta": meta,
        }

    return runtime, ir_runtime


def _resolve_engine(config: RenderConfig) -> tuple[str, ReverbEngine]:
    engine_name = config.engine
    if engine_name == "auto":
        engine_name = "conv" if config.ir is not None else "algo"

    if engine_name == "conv":
        if config.ir is None:
            msg = "Convolution engine requires --ir when --engine conv is selected"
            raise ValueError(msg)
        return "conv", ConvolutionReverbEngine(
            ConvolutionReverbConfig(
                wet=config.wet,
                dry=config.dry,
                ir_path=config.ir,
                ir_normalize=config.ir_normalize,
                partition_size=config.partition_size,
                tail_limit=config.tail_limit,
                threads=config.threads,
            )
        )

    return "algo", AlgoReverbEngine(
        AlgoReverbConfig(
            rt60=config.rt60,
            pre_delay_ms=config.pre_delay_ms,
            damping=config.damping,
            width=config.width,
            mod_depth_ms=config.mod_depth_ms,
            mod_rate_hz=config.mod_rate_hz,
            wet=config.wet,
            dry=config.dry,
            block_size=config.block_size,
            shimmer=config.shimmer,
            shimmer_semitones=config.shimmer_semitones,
            shimmer_mix=config.shimmer_mix,
            shimmer_feedback=config.shimmer_feedback,
            shimmer_highcut=config.shimmer_highcut,
            shimmer_lowcut=config.shimmer_lowcut,
        )
    )


def _resolve_analysis_path(outfile: Path, analysis_out: str | None) -> Path:
    if analysis_out is not None:
        return Path(analysis_out)
    return Path(f"{outfile}.analysis.json")


def _build_per_pass_processor(config: RenderConfig, sr: int) -> PassProcessor:
    stage: NormalizeStage = config.normalize_stage
    if stage == "per-pass":
        target_lufs = (
            config.repeat_target_lufs
            if config.repeat_target_lufs is not None
            else config.target_lufs
        )
        target_peak = (
            config.repeat_target_peak_dbfs
            if config.repeat_target_peak_dbfs is not None
            else config.target_peak_dbfs
        )

        def processor(audio: AudioArray, pass_idx: int, total_passes: int) -> AudioArray:
            _ = pass_idx, total_passes
            return apply_output_targets(
                audio,
                sr,
                target_lufs=target_lufs,
                target_peak_dbfs=target_peak,
                limiter=config.limiter,
                use_true_peak=config.use_true_peak,
            )

        return processor

    def safety_processor(audio: AudioArray, pass_idx: int, total_passes: int) -> AudioArray:
        _ = pass_idx, total_passes
        if not config.limiter:
            return audio
        limited = soft_limiter(audio, threshold_dbfs=-1.0, knee_db=6.0)
        return peak_normalize(limited, target_dbfs=-1.0)

    return safety_processor


def _should_include_loudness(config: RenderConfig) -> bool:
    return any(
        value is not None
        for value in (
            config.target_lufs,
            config.target_peak_dbfs,
            config.repeat_target_lufs,
            config.repeat_target_peak_dbfs,
        )
    )


def _algo_tail_padding_seconds(config: RenderConfig) -> float:
    """Compute explicit tail render duration for algorithmic reverbs."""
    pre_delay = max(0.0, float(config.pre_delay_ms)) / 1000.0
    return max(0.25, float(config.rt60) + pre_delay)


def _append_tail_padding(audio: AudioArray, sr: int, tail_seconds: float) -> AudioArray:
    """Append silence so stateful reverbs can decay naturally."""
    tail_samples = int(np.ceil(max(0.0, tail_seconds) * float(sr)))
    if tail_samples <= 0:
        return audio
    padding = np.zeros((tail_samples, audio.shape[1]), dtype=np.float32)
    return np.concatenate((audio, padding), axis=0)


def _non_default_settings(config: RenderConfig) -> dict[str, Any]:
    defaults = asdict(RenderConfig())
    current = asdict(config)
    return {key: value for key, value in current.items() if defaults.get(key) != value}
