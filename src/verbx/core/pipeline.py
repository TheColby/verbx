"""Render pipeline orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.analysis.framewise import write_framewise_csv
from verbx.config import NormalizeStage, RenderConfig
from verbx.core.accel import configure_cpu_threads, resolve_device
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
        info = sf.info(str(infile))
        input_sr = int(info.samplerate)
        input_channels = int(info.channels)

        runtime_config, ir_runtime = _prepare_runtime_config(config, input_sr, input_channels)
        resolved_device = resolve_device(runtime_config.device)
        engine_name, engine, engine_device = _resolve_engine(runtime_config, resolved_device)
        if engine_device in {"cpu", "mps"}:
            configure_cpu_threads(runtime_config.threads)

        if _can_stream_convolution(runtime_config, engine_name, engine):
            progress.set_passes(1)
            stream_engine = engine
            assert isinstance(stream_engine, ConvolutionReverbEngine)
            output_subtype = _resolve_output_subtype(runtime_config.output_subtype)
            stream_stats = stream_engine.process_streaming_file(
                infile=str(infile),
                outfile=str(outfile),
                output_subtype=output_subtype,
            )
            progress.mark_read()
            progress.mark_process_pass(1)
            progress.mark_write()

            report: dict[str, Any] = {
                "engine": engine_name,
                "sample_rate": int(stream_stats["sample_rate"]),
                "input_samples": int(stream_stats["input_samples"]),
                "output_samples": int(stream_stats["output_samples"]),
                "channels": int(stream_stats["channels"]),
                "config": asdict(runtime_config),
                "effective": {
                    "engine_requested": config.engine,
                    "engine_resolved": engine_name,
                    "device_requested": config.device,
                    "device_resolved": engine_device,
                    "device_platform_resolved": resolved_device,
                    "compute_backend": engine.backend_name(),
                    "ir_used": runtime_config.ir,
                    "tail_padding_seconds": 0.0,
                    "input_peak_linear": float(stream_stats["input_peak_linear"]),
                    "output_subtype": output_subtype if output_subtype is not None else "auto",
                    "output_peak_norm": runtime_config.output_peak_norm,
                    "output_peak_target_dbfs": runtime_config.output_peak_target_dbfs,
                    "streaming_mode": True,
                    "non_default_settings": _non_default_settings(runtime_config),
                },
            }
            if ir_runtime is not None:
                report["ir_runtime"] = ir_runtime

            if not runtime_config.silent:
                include_loudness = _should_include_loudness(runtime_config)
                analyzer = AudioAnalyzer()
                audio_in, _ = read_audio(str(infile))
                audio_out, _ = read_audio(str(outfile))
                report["input"] = analyzer.analyze(
                    audio_in,
                    input_sr,
                    include_loudness=include_loudness,
                )
                report["output"] = analyzer.analyze(
                    audio_out,
                    input_sr,
                    include_loudness=include_loudness,
                )
                analysis_path = _resolve_analysis_path(outfile, runtime_config.analysis_out)
                analysis_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                report["analysis_path"] = str(analysis_path)
                if runtime_config.frames_out is not None:
                    frames_path = Path(runtime_config.frames_out)
                    write_framewise_csv(frames_path, audio_out, input_sr)
                    report["frames_path"] = str(frames_path)

            progress.mark_analyze()
            return report

        audio, sr = read_audio(str(infile))
        input_peak_linear = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        progress.mark_read()
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

        rendered = _apply_final_peak_normalization(
            rendered,
            mode=runtime_config.output_peak_norm,
            input_peak_linear=input_peak_linear,
            target_dbfs=runtime_config.output_peak_target_dbfs,
        )

        output_subtype = _resolve_output_subtype(runtime_config.output_subtype)
        write_audio(str(outfile), rendered, sr, subtype=output_subtype)
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
                "device_requested": config.device,
                "device_resolved": engine_device,
                "device_platform_resolved": resolved_device,
                "compute_backend": engine.backend_name(),
                "ir_used": runtime_config.ir,
                "tail_padding_seconds": tail_padding_seconds,
                "input_peak_linear": input_peak_linear,
                "output_subtype": output_subtype if output_subtype is not None else "auto",
                "output_peak_norm": runtime_config.output_peak_norm,
                "output_peak_target_dbfs": runtime_config.output_peak_target_dbfs,
                "streaming_mode": False,
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


def _can_stream_convolution(config: RenderConfig, engine_name: str, engine: ReverbEngine) -> bool:
    if engine_name != "conv":
        return False
    if not isinstance(engine, ConvolutionReverbEngine):
        return False
    if engine.backend_name() == "cuda-cupy":
        return False
    if config.repeat != 1 or config.freeze:
        return False
    if config.normalize_stage != "none":
        return False
    if config.output_peak_norm != "none":
        return False
    if any(
        value is not None
        for value in (
            config.target_lufs,
            config.target_peak_dbfs,
            config.repeat_target_lufs,
            config.repeat_target_peak_dbfs,
        )
    ):
        return False
    if config.duck or config.bloom > 0.0:
        return False
    if config.lowcut is not None or config.highcut is not None or config.tilt != 0.0:
        return False
    return True


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


def _resolve_engine(config: RenderConfig, device: str) -> tuple[str, ReverbEngine, str]:
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
                ir_matrix_layout=config.ir_matrix_layout,
                partition_size=config.partition_size,
                tail_limit=config.tail_limit,
                threads=config.threads,
                device=device,
            )
        ), device

    algo_device = device if device in {"cpu", "mps"} else "cpu"
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
            device=algo_device,
        )
    ), algo_device


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


def _resolve_output_subtype(mode: str) -> str | None:
    mapping = {
        "auto": None,
        "float32": "FLOAT",
        "float64": "DOUBLE",
        "pcm16": "PCM_16",
        "pcm24": "PCM_24",
        "pcm32": "PCM_32",
    }
    if mode not in mapping:
        msg = f"Unsupported output subtype mode: {mode}"
        raise ValueError(msg)
    return mapping[mode]


def _apply_final_peak_normalization(
    audio: AudioArray,
    mode: str,
    input_peak_linear: float,
    target_dbfs: float | None,
) -> AudioArray:
    if audio.shape[0] == 0:
        return audio.copy()

    if mode == "none":
        return audio

    if mode == "full-scale":
        return peak_normalize(audio, target_dbfs=0.0)

    if mode == "target":
        if target_dbfs is None:
            msg = "output_peak_norm=target requires --output-peak-target-dbfs"
            raise ValueError(msg)
        return peak_normalize(audio, target_dbfs=float(target_dbfs))

    if mode == "input":
        current_peak = float(np.max(np.abs(audio)))
        if current_peak <= 0.0 or input_peak_linear <= 0.0:
            return audio.copy()
        gain = float(input_peak_linear / current_peak)
        return np.asarray(audio * gain, dtype=np.float32)

    msg = f"Unsupported output_peak_norm mode: {mode}"
    raise ValueError(msg)
