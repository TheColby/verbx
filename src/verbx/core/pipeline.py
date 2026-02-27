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

AudioArray = npt.NDArray[np.float32]
PassProcessor = Callable[[AudioArray, int, int], AudioArray]


def run_render_pipeline(infile: Path, outfile: Path, config: RenderConfig) -> dict[str, Any]:
    """Run the complete render pipeline and return report metadata."""
    validate_audio_path(str(infile))

    with RenderProgress(enabled=(config.progress and not config.silent)) as progress:
        audio, sr = read_audio(str(infile))
        progress.mark_read()

        engine_name, engine = _resolve_engine(config)
        progress.set_passes(max(1, config.repeat))

        input_for_engine = audio
        if config.freeze:
            input_for_engine = freeze_segment(
                audio=audio,
                sr=sr,
                start=config.start,
                end=config.end,
                mode="loop",
                xfade_ms=100.0,
            )

        repeat_post_processor = _build_per_pass_processor(config, sr)

        rendered = repeat_process(
            engine=engine,
            audio=input_for_engine,
            sr=sr,
            n=config.repeat,
            post_pass_processor=repeat_post_processor,
            progress_callback=lambda idx, total: progress.mark_process_pass(idx),
        )

        rendered = apply_ambient_processing(
            wet=rendered,
            dry_reference=input_for_engine,
            sr=sr,
            duck=config.duck,
            duck_attack=config.duck_attack,
            duck_release=config.duck_release,
            bloom=config.bloom,
            lowcut=config.lowcut,
            highcut=config.highcut,
            tilt=config.tilt,
        )

        if config.normalize_stage == "post":
            rendered = apply_output_targets(
                rendered,
                sr,
                target_lufs=config.target_lufs,
                target_peak_dbfs=config.target_peak_dbfs,
                limiter=config.limiter,
                use_true_peak=config.use_true_peak,
            )
        elif config.normalize_stage == "none" and config.limiter:
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
            "config": asdict(config),
        }

        if not config.silent:
            include_loudness = _should_include_loudness(config)
            analyzer = AudioAnalyzer()
            report["input"] = analyzer.analyze(audio, sr, include_loudness=include_loudness)
            report["output"] = analyzer.analyze(rendered, sr, include_loudness=include_loudness)
            analysis_path = _resolve_analysis_path(outfile, config.analysis_out)
            analysis_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            report["analysis_path"] = str(analysis_path)

        progress.mark_analyze()

    return report


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
