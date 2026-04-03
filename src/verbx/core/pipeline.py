"""Top-level render orchestration for CLI commands.

This module ties together:

- input validation and runtime configuration expansion,
- engine selection (algorithmic vs convolution),
- optional freeze/repeat/ambient/loudness stages,
- streaming fast path for long convolution renders,
- output artifact generation (audio + JSON + framewise CSV).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy.signal import resample_poly

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.analysis.framewise import write_framewise_csv
from verbx.config import NormalizeStage, RenderConfig
from verbx.core.accel import configure_cpu_threads, resolve_device_for_engine
from verbx.core.algo_proxy import render_algo_proxy_ir
from verbx.core.algo_reverb import AlgoReverbConfig, AlgoReverbEngine
from verbx.core.ambient import apply_ambient_processing, apply_tilt_eq
from verbx.core.automation import (
    CONV_AUTOMATION_TARGETS,
    ENGINE_AUTOMATION_TARGETS,
    POST_RENDER_AUTOMATION_TARGETS,
    AutomationBundle,
    apply_render_automation,
    load_automation_bundle,
    parse_automation_clamp_overrides,
    write_automation_trace,
    write_feature_vector_trace,
)
from verbx.core.convolution_reverb import ConvolutionReverbConfig, ConvolutionReverbEngine
from verbx.core.early_reflections import apply_image_source_early_reflections, material_absorption
from verbx.core.engine_base import ReverbEngine
from verbx.core.freeze import freeze_segment
from verbx.core.loudness import apply_output_targets
from verbx.core.modulation import apply_parameter_modulation, parse_mod_route_spec
from verbx.core.render_report import RenderReport
from verbx.core.repeat import repeat_process
from verbx.core.schema_versions import TRACK_C_CALIBRATION_VERSION
from verbx.core.spatial import (
    ambisonic_channel_count,
    convert_ambisonic_convention,
    decode_foa_to_stereo,
    encode_bus_to_foa,
    rotate_ambisonic_yaw,
)
from verbx.io.audio import (
    peak_normalize,
    read_audio,
    soft_limiter,
    validate_audio_path,
    write_audio,
)
from verbx.io.progress import RenderProgress
from verbx.ir.generator import IRGenConfig, generate_or_load_cached_ir
from verbx.ir.morph import (
    IRMorphConfig,
    IRMorphMode,
    generate_or_load_cached_blended_ir,
    resolve_blend_mix_values,
)

AudioArray = npt.NDArray[np.float64]
PassProcessor = Callable[[AudioArray, int, int], AudioArray]


@dataclass(slots=True)
class _PipelineContext:
    """Resolved execution context shared across pipeline stages."""

    infile: Path
    outfile: Path
    config: RenderConfig
    progress: RenderProgress
    input_sr: int
    input_channels: int
    input_duration_seconds: float
    processing_sr: int
    runtime_config: RenderConfig
    ir_runtime: dict[str, Any] | None
    perceptual_macro_summary: dict[str, Any] | None
    engine_name: str
    engine: ReverbEngine
    engine_device: str
    platform_device: str


def run_render_pipeline(infile: Path, outfile: Path, config: RenderConfig) -> RenderReport:
    """Run one end-to-end render and return a structured render report."""
    validate_audio_path(str(infile))

    with RenderProgress(enabled=(config.progress and not config.silent)) as progress:
        info = sf.info(str(infile))
        input_sr = int(info.samplerate)
        input_channels = int(info.channels)
        input_duration_seconds = float(info.frames) / float(max(1, input_sr))
        processing_sr = _resolve_processing_sample_rate(
            input_sr=input_sr,
            target_sr=config.target_sr,
        )

        runtime_config, ir_runtime = _prepare_runtime_config(
            config,
            processing_sr,
            input_channels,
            infile,
            input_duration_seconds,
        )
        perceptual_macro_summary = _build_perceptual_macro_summary(config, runtime_config)
        resolved_engine_name = _resolve_engine_name(runtime_config)
        engine_device, platform_device = resolve_device_for_engine(
            runtime_config.device,
            resolved_engine_name,
        )
        engine_name, engine, engine_device = _resolve_engine(
            runtime_config,
            engine_device,
            engine_name=resolved_engine_name,
        )
        if engine_device in {"cpu", "mps"}:
            configure_cpu_threads(runtime_config.threads)
        context = _PipelineContext(
            infile=infile,
            outfile=outfile,
            config=config,
            progress=progress,
            input_sr=input_sr,
            input_channels=input_channels,
            input_duration_seconds=input_duration_seconds,
            processing_sr=processing_sr,
            runtime_config=runtime_config,
            ir_runtime=ir_runtime,
            perceptual_macro_summary=perceptual_macro_summary,
            engine_name=engine_name,
            engine=engine,
            engine_device=engine_device,
            platform_device=platform_device,
        )
        streaming_report = _run_streaming_pipeline_if_eligible(context)
        if streaming_report is not None:
            return streaming_report
        return _run_in_memory_pipeline(context)


def _run_streaming_pipeline_if_eligible(context: _PipelineContext) -> RenderReport | None:
    """Run a conservative streaming path when the render contract allows it."""
    runtime_config = context.runtime_config
    if _can_stream_algo_proxy(runtime_config, context.engine_name, input_sr=context.input_sr):
        return _run_algo_proxy_streaming_pipeline(context)
    if _can_stream_convolution(
        runtime_config,
        context.engine_name,
        context.engine,
        input_sr=context.input_sr,
    ):
        return _run_convolution_streaming_pipeline(context)
    return None


def _run_algo_proxy_streaming_pipeline(context: _PipelineContext) -> RenderReport:
    """Execute the algorithmic proxy streaming fast path."""
    runtime_config = context.runtime_config
    progress = context.progress
    progress.set_passes(1)
    output_subtype = _resolve_output_subtype(runtime_config.output_subtype)
    output_format = _resolve_output_format(
        mode=runtime_config.output_container,
        outfile=context.outfile,
        estimated_bytes=None,
    )
    conv_device = "cuda" if runtime_config.algo_gpu_proxy else "cpu"
    proxy_tail_padding_seconds = _algo_tail_padding_seconds(runtime_config, context.processing_sr)
    proxy_hold_frames = _tail_zero_hold_samples(
        context.processing_sr,
        runtime_config.tail_stop_hold_ms,
    )

    needs_src = context.input_sr != context.processing_sr
    src_tmp: Path | None = None
    stream_infile = context.infile
    if needs_src:
        src_audio, _ = read_audio(str(context.infile))
        src_resampled = _resample_audio_polyphase(
            src_audio,
            src_sr=context.input_sr,
            dst_sr=context.processing_sr,
        )
        src_tmp = context.infile.with_name(
            f"{context.infile.stem}.proxy_src_tmp{context.infile.suffix}"
        )
        write_audio(str(src_tmp), src_resampled, context.processing_sr, subtype="DOUBLE")
        stream_infile = src_tmp

    proxy_ir_path, _ = render_algo_proxy_ir(
        config=runtime_config,
        sr=context.processing_sr,
        input_channels=context.input_channels,
    )
    stream_tmp_out = context.outfile.with_name(
        f"{context.outfile.stem}.stream_tmp{context.outfile.suffix}"
    )
    try:
        proxy_conv_config = _build_convolution_config(
            runtime_config,
            ir_path=str(proxy_ir_path),
            device=conv_device,
        )
        proxy_conv_config.ir_normalize = "none"
        conv_engine = ConvolutionReverbEngine(proxy_conv_config)
        stream_stats = conv_engine.process_streaming_file(
            infile=str(stream_infile),
            outfile=str(stream_tmp_out),
            output_subtype=output_subtype,
            output_format=output_format,
        )
    finally:
        proxy_ir_path.unlink(missing_ok=True)
        if src_tmp is not None:
            src_tmp.unlink(missing_ok=True)

    output_samples = _complete_stream_file_tail_to_zero(
        stream_tmp_out,
        threshold=_tail_threshold_linear(runtime_config.tail_stop_threshold_db),
        hold_ms=runtime_config.tail_stop_hold_ms,
        metric=runtime_config.tail_stop_metric,
        min_frames=int(stream_stats["input_samples"])
        + int(np.ceil(proxy_tail_padding_seconds * float(context.processing_sr)))
        + int(proxy_hold_frames),
    )
    stream_tmp_out.replace(context.outfile)

    needs_eq = (
        runtime_config.lowcut is not None
        or runtime_config.highcut is not None
        or abs(float(runtime_config.tilt)) > 1e-4
    )
    if needs_eq:
        eq_audio, _ = read_audio(str(context.outfile))
        eq_out = apply_tilt_eq(
            eq_audio,
            context.processing_sr,
            tilt_db=float(runtime_config.tilt),
            lowcut=runtime_config.lowcut,
            highcut=runtime_config.highcut,
        )
        write_audio(str(context.outfile), eq_out, context.processing_sr, subtype=output_subtype)
        output_samples = _complete_stream_file_tail_to_zero(
            context.outfile,
            threshold=_tail_threshold_linear(runtime_config.tail_stop_threshold_db),
            hold_ms=runtime_config.tail_stop_hold_ms,
            metric=runtime_config.tail_stop_metric,
            min_frames=int(stream_stats["input_samples"])
            + int(np.ceil(proxy_tail_padding_seconds * float(context.processing_sr)))
            + int(proxy_hold_frames),
        )

    progress.mark_read()
    progress.mark_process_pass(1)
    progress.mark_write()

    report = _build_render_report(
        engine="algo",
        sample_rate=int(stream_stats["sample_rate"]),
        input_samples=int(stream_stats["input_samples"]),
        output_samples=output_samples,
        channels=int(stream_stats["channels"]),
        runtime_config=runtime_config,
        effective={
            "engine_requested": context.config.engine,
            "engine_resolved": "algo_proxy_stream",
            "device_requested": context.config.device,
            "device_resolved": conv_device,
            "device_platform_resolved": context.platform_device,
            "compute_backend": f"algo-proxy-{conv_engine.backend_name()}",
            "ir_used": "algo_proxy_ir",
            "self_convolve": False,
            "beast_mode": runtime_config.beast_mode,
            "tail_padding_seconds": 0.0,
            "input_peak_linear": float(stream_stats["input_peak_linear"]),
            "input_sample_rate": context.input_sr,
            "processing_sample_rate": int(stream_stats["sample_rate"]),
            "sample_rate_action": _sample_rate_action(
                input_sr=context.input_sr,
                processing_sr=int(stream_stats["sample_rate"]),
            ),
            "output_subtype": output_subtype if output_subtype is not None else "auto",
            "output_container": runtime_config.output_container,
            "output_peak_norm": runtime_config.output_peak_norm,
            "output_peak_target_dbfs": runtime_config.output_peak_target_dbfs,
            "streaming_mode": True,
            "streaming_algorithmic_proxy": True,
            "perceptual_macros": context.perceptual_macro_summary,
            "non_default_settings": _non_default_settings(runtime_config),
        },
    )
    _maybe_attach_analysis_report(
        report=report,
        infile=context.infile,
        outfile=context.outfile,
        runtime_config=runtime_config,
        input_audio_path=context.infile,
        output_audio_path=context.outfile,
        input_audio=None,
        output_audio=None,
        input_sr=context.input_sr,
        output_sr=context.processing_sr,
        perceptual_macros=context.perceptual_macro_summary,
        include_ambisonic_metadata=False,
    )
    progress.mark_analyze()
    return report


def _run_convolution_streaming_pipeline(context: _PipelineContext) -> RenderReport:
    """Execute the direct convolution streaming fast path."""
    runtime_config = context.runtime_config
    progress = context.progress
    progress.set_passes(1)
    stream_engine = context.engine
    assert isinstance(stream_engine, ConvolutionReverbEngine)
    output_subtype = _resolve_output_subtype(runtime_config.output_subtype)
    output_format = _resolve_output_format(
        mode=runtime_config.output_container,
        outfile=context.outfile,
        estimated_bytes=None,
    )
    stream_tmp_out = context.outfile.with_name(
        f"{context.outfile.stem}.stream_tmp{context.outfile.suffix}"
    )
    stream_stats = stream_engine.process_streaming_file(
        infile=str(context.infile),
        outfile=str(stream_tmp_out),
        output_subtype=output_subtype,
        output_format=output_format,
    )
    output_samples = _complete_stream_file_tail_to_zero(
        stream_tmp_out,
        threshold=_tail_threshold_linear(runtime_config.tail_stop_threshold_db),
        hold_ms=runtime_config.tail_stop_hold_ms,
        metric=runtime_config.tail_stop_metric,
        min_frames=int(stream_stats["input_samples"]),
    )
    stream_tmp_out.replace(context.outfile)
    progress.mark_read()
    progress.mark_process_pass(1)
    progress.mark_write()

    report = _build_render_report(
        engine=context.engine_name,
        sample_rate=int(stream_stats["sample_rate"]),
        input_samples=int(stream_stats["input_samples"]),
        output_samples=output_samples,
        channels=int(stream_stats["channels"]),
        runtime_config=runtime_config,
        effective={
            "engine_requested": context.config.engine,
            "engine_resolved": context.engine_name,
            "device_requested": context.config.device,
            "device_resolved": context.engine_device,
            "device_platform_resolved": context.platform_device,
            "compute_backend": context.engine.backend_name(),
            "ir_used": runtime_config.ir,
            "self_convolve": runtime_config.self_convolve,
            "beast_mode": runtime_config.beast_mode,
            "tail_padding_seconds": 0.0,
            "input_peak_linear": float(stream_stats["input_peak_linear"]),
            "input_sample_rate": context.input_sr,
            "processing_sample_rate": int(stream_stats["sample_rate"]),
            "sample_rate_action": _sample_rate_action(
                input_sr=context.input_sr,
                processing_sr=int(stream_stats["sample_rate"]),
            ),
            "output_subtype": output_subtype if output_subtype is not None else "auto",
            "output_container": runtime_config.output_container,
            "output_peak_norm": runtime_config.output_peak_norm,
            "output_peak_target_dbfs": runtime_config.output_peak_target_dbfs,
            "streaming_mode": True,
            "perceptual_macros": context.perceptual_macro_summary,
            "non_default_settings": _non_default_settings(runtime_config),
        },
        ir_runtime=context.ir_runtime,
    )
    _maybe_attach_analysis_report(
        report=report,
        infile=context.infile,
        outfile=context.outfile,
        runtime_config=runtime_config,
        input_audio_path=context.infile,
        output_audio_path=context.outfile,
        input_audio=None,
        output_audio=None,
        input_sr=context.input_sr,
        output_sr=context.processing_sr,
        perceptual_macros=context.perceptual_macro_summary,
        include_ambisonic_metadata=False,
    )
    progress.mark_analyze()
    return report


def _run_in_memory_pipeline(context: _PipelineContext) -> RenderReport:
    """Execute the full in-memory render path with all post stages."""
    runtime_config = context.runtime_config
    progress = context.progress
    audio_source, sr = read_audio(str(context.infile))
    input_peak_linear = float(np.max(np.abs(audio_source))) if audio_source.size > 0 else 0.0
    if sr != context.processing_sr:
        audio = _resample_audio_polyphase(audio_source, src_sr=sr, dst_sr=context.processing_sr)
        sr = context.processing_sr
    else:
        audio = np.asarray(audio_source, dtype=np.float64)
    progress.mark_read()
    progress.set_passes(max(1, context.config.repeat))

    if runtime_config.er_geometry:
        absorption = material_absorption(
            runtime_config.er_material,
            float(runtime_config.er_absorption),
        )
        audio = apply_image_source_early_reflections(
            audio,
            sr=sr,
            room_dims_m=runtime_config.er_room_dims_m,
            source_pos_m=runtime_config.er_source_pos_m,
            listener_pos_m=runtime_config.er_listener_pos_m,
            absorption=absorption,
        )

    input_for_engine = _prepare_spatial_input(audio, runtime_config)
    if runtime_config.freeze:
        input_for_engine = freeze_segment(
            audio=input_for_engine,
            sr=sr,
            start=runtime_config.start,
            end=runtime_config.end,
            mode="loop",
            xfade_ms=100.0,
        )
    base_output_floor_samples = int(input_for_engine.shape[0])

    tail_padding_seconds = 0.0
    if context.engine_name == "algo":
        tail_padding_seconds = _algo_tail_padding_seconds(runtime_config, sr)
        input_for_engine = _append_tail_padding(
            audio=input_for_engine,
            sr=sr,
            tail_seconds=tail_padding_seconds,
        )

    has_automation_source = (
        runtime_config.automation_file is not None
        or len(runtime_config.automation_points) > 0
        or len(runtime_config.feature_vector_lanes) > 0
    )
    clamp_overrides = (
        parse_automation_clamp_overrides(runtime_config.automation_clamp)
        if has_automation_source
        else None
    )
    preloaded_automation_bundle: AutomationBundle | None = None
    if has_automation_source and context.engine_name == "algo":
        preloaded_automation_bundle = _load_runtime_automation_bundle(
            config=runtime_config,
            sr=sr,
            num_samples=int(input_for_engine.shape[0]),
            clamp_overrides=clamp_overrides,
            feature_audio=input_for_engine,
        )
        if isinstance(context.engine, AlgoReverbEngine):
            context.engine.set_parameter_automation(preloaded_automation_bundle.curves)

    repeat_post_processor = _build_per_pass_processor(runtime_config, sr)
    rendered = repeat_process(
        engine=context.engine,
        audio=input_for_engine,
        sr=sr,
        n=runtime_config.repeat,
        post_pass_processor=repeat_post_processor,
        progress_callback=lambda idx, total: progress.mark_process_pass(idx),
    )

    conv_target_summary: dict[str, Any] | None = None
    if has_automation_source:
        if (
            preloaded_automation_bundle is None
            or int(preloaded_automation_bundle.num_samples) != int(rendered.shape[0])
        ):
            preloaded_automation_bundle = _load_runtime_automation_bundle(
                config=runtime_config,
                sr=sr,
                num_samples=int(rendered.shape[0]),
                clamp_overrides=clamp_overrides,
                feature_audio=input_for_engine,
            )
        assert preloaded_automation_bundle is not None
        _validate_automation_target_domains_for_engine(
            engine_name=context.engine_name,
            bundle=preloaded_automation_bundle,
        )
        if context.engine_name == "conv":
            rendered, conv_target_summary = _apply_convolution_automation_targets(
                rendered=rendered,
                input_for_engine=input_for_engine,
                sr=sr,
                config=runtime_config,
                bundle=preloaded_automation_bundle,
                engine_device=context.engine_device,
                repeat_post_processor=repeat_post_processor,
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

    modulation_summaries: list[dict[str, Any]] = []
    if runtime_config.mod_target != "none" and len(runtime_config.mod_sources) > 0:
        rendered, modulation_summary = apply_parameter_modulation(
            audio=rendered,
            dry_reference=input_for_engine,
            sr=sr,
            target=runtime_config.mod_target,
            source_specs=runtime_config.mod_sources,
            value_min=runtime_config.mod_min,
            value_max=runtime_config.mod_max,
            combine=runtime_config.mod_combine,
            smooth_ms=runtime_config.mod_smooth_ms,
        )
        if modulation_summary is not None:
            modulation_summary["route_kind"] = "base"
            modulation_summaries.append(modulation_summary)

    for route_idx, route_spec in enumerate(runtime_config.mod_routes, start=1):
        route = parse_mod_route_spec(route_spec)
        rendered, route_summary = apply_parameter_modulation(
            audio=rendered,
            dry_reference=input_for_engine,
            sr=sr,
            target=route.target,
            source_specs=route.source_specs,
            value_min=route.value_min,
            value_max=route.value_max,
            combine=route.combine,
            smooth_ms=route.smooth_ms,
        )
        if route_summary is not None:
            route_summary["route_kind"] = "route"
            route_summary["route_index"] = route_idx
            route_summary["route_spec"] = route_spec
            modulation_summaries.append(route_summary)

    automation_summary: dict[str, Any] | None = None
    if has_automation_source:
        bundle = preloaded_automation_bundle
        if bundle is None or int(bundle.num_samples) != int(rendered.shape[0]):
            bundle = _load_runtime_automation_bundle(
                config=runtime_config,
                sr=sr,
                num_samples=int(rendered.shape[0]),
                clamp_overrides=clamp_overrides,
                feature_audio=input_for_engine,
            )
        assert bundle is not None
        _validate_automation_target_domains_for_engine(
            engine_name=context.engine_name,
            bundle=bundle,
        )
        dry_reference = _build_dry_reference_for_automation(
            engine_name=context.engine_name,
            input_for_engine=input_for_engine,
            rendered=rendered,
            config=runtime_config,
        )
        rendered, automation_summary = apply_render_automation(
            rendered=rendered,
            dry_reference=dry_reference,
            base_wet=float(runtime_config.wet),
            base_dry=float(runtime_config.dry),
            bundle=bundle,
        )
        targets = set(bundle.curves.keys())
        automation_summary["engine_targets"] = sorted(
            target for target in targets if target in ENGINE_AUTOMATION_TARGETS
        )
        automation_summary["post_targets"] = sorted(
            target for target in targets if target in POST_RENDER_AUTOMATION_TARGETS
        )
        automation_summary["conv_targets"] = sorted(
            target for target in targets if target in CONV_AUTOMATION_TARGETS
        )
        if conv_target_summary is not None:
            automation_summary["conv_summary"] = conv_target_summary
        if runtime_config.automation_trace_out is not None:
            write_automation_trace(Path(runtime_config.automation_trace_out), bundle)
        if runtime_config.feature_vector_trace_out is not None:
            write_feature_vector_trace(Path(runtime_config.feature_vector_trace_out), bundle)

    rendered = _apply_spatial_output_transform(rendered, runtime_config)
    modulation_payload: dict[str, Any] | None
    if len(modulation_summaries) == 0:
        modulation_payload = None
    elif len(modulation_summaries) == 1:
        modulation_payload = modulation_summaries[0]
    else:
        modulation_payload = {"count": len(modulation_summaries), "routes": modulation_summaries}

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
    rendered = _complete_tail_to_zero(
        rendered,
        sr,
        threshold=_tail_threshold_linear(runtime_config.tail_stop_threshold_db),
        hold_ms=runtime_config.tail_stop_hold_ms,
        metric=runtime_config.tail_stop_metric,
        min_samples=base_output_floor_samples,
    )

    output_subtype = _resolve_output_subtype(runtime_config.output_subtype)
    bytes_per_sample = {
        None: 8,
        "PCM_16": 2,
        "PCM_24": 3,
        "PCM_32": 4,
        "FLOAT": 4,
        "DOUBLE": 8,
    }.get(output_subtype, 8)
    estimated_bytes = int(rendered.shape[0]) * int(rendered.shape[1]) * int(bytes_per_sample)
    output_format = _resolve_output_format(
        mode=runtime_config.output_container,
        outfile=context.outfile,
        estimated_bytes=estimated_bytes,
    )
    write_audio(
        str(context.outfile),
        rendered,
        sr,
        subtype=output_subtype,
        format=output_format,
    )
    progress.mark_write()

    report = _build_render_report(
        engine=context.engine_name,
        sample_rate=sr,
        input_samples=int(audio.shape[0]),
        output_samples=int(rendered.shape[0]),
        channels=int(rendered.shape[1]),
        runtime_config=runtime_config,
        effective={
            "engine_requested": context.config.engine,
            "engine_resolved": context.engine_name,
            "device_requested": context.config.device,
            "device_resolved": context.engine_device,
            "device_platform_resolved": context.platform_device,
            "compute_backend": context.engine.backend_name(),
            "ir_used": runtime_config.ir,
            "self_convolve": runtime_config.self_convolve,
            "beast_mode": runtime_config.beast_mode,
            "tail_padding_seconds": tail_padding_seconds,
            "input_peak_linear": input_peak_linear,
            "input_sample_rate": context.input_sr,
            "processing_sample_rate": sr,
            "sample_rate_action": _sample_rate_action(input_sr=context.input_sr, processing_sr=sr),
            "output_subtype": output_subtype if output_subtype is not None else "auto",
            "output_container": runtime_config.output_container,
            "output_peak_norm": runtime_config.output_peak_norm,
            "output_peak_target_dbfs": runtime_config.output_peak_target_dbfs,
            "streaming_mode": False,
            "modulation": modulation_payload,
            "automation": automation_summary,
            "perceptual_macros": context.perceptual_macro_summary,
            "non_default_settings": _non_default_settings(runtime_config),
        },
        ir_runtime=context.ir_runtime,
    )
    _maybe_attach_analysis_report(
        report=report,
        infile=context.infile,
        outfile=context.outfile,
        runtime_config=runtime_config,
        input_audio_path=None,
        output_audio_path=None,
        input_audio=audio_source,
        output_audio=rendered,
        input_sr=context.input_sr,
        output_sr=sr,
        perceptual_macros=context.perceptual_macro_summary,
        include_ambisonic_metadata=True,
    )
    progress.mark_analyze()
    return report


def _build_render_report(
    *,
    engine: str,
    sample_rate: int,
    input_samples: int,
    output_samples: int,
    channels: int,
    runtime_config: RenderConfig,
    effective: dict[str, Any],
    ir_runtime: dict[str, Any] | None = None,
) -> RenderReport:
    """Create the canonical typed render report payload."""
    return RenderReport(
        engine=engine,
        sample_rate=int(sample_rate),
        input_samples=int(input_samples),
        output_samples=int(output_samples),
        channels=int(channels),
        config=asdict(runtime_config),
        effective=effective,
        ir_runtime=ir_runtime,
    )


def _maybe_attach_analysis_report(
    *,
    report: RenderReport,
    infile: Path,
    outfile: Path,
    runtime_config: RenderConfig,
    input_audio_path: Path | None,
    output_audio_path: Path | None,
    input_audio: AudioArray | None,
    output_audio: AudioArray | None,
    input_sr: int,
    output_sr: int,
    perceptual_macros: dict[str, Any] | None,
    include_ambisonic_metadata: bool,
) -> None:
    """Attach analyzer outputs and persist artifact paths when enabled."""
    if runtime_config.silent:
        return

    include_loudness = _should_include_loudness(runtime_config)
    analyzer = AudioAnalyzer()

    if input_audio is None:
        if input_audio_path is None:
            msg = "input_audio_path is required when input_audio is not provided"
            raise ValueError(msg)
        input_audio, _ = read_audio(str(input_audio_path))
    if output_audio is None:
        if output_audio_path is None:
            msg = "output_audio_path is required when output_audio is not provided"
            raise ValueError(msg)
        output_audio, _ = read_audio(str(output_audio_path))

    report.input = analyzer.analyze(
        input_audio,
        input_sr,
        include_loudness=include_loudness,
    )
    if include_ambisonic_metadata:
        report.output = analyzer.analyze(
            output_audio,
            output_sr,
            include_loudness=include_loudness,
            ambi_order=(
                runtime_config.ambi_order
                if runtime_config.ambi_order > 0 and runtime_config.ambi_decode_to == "none"
                else None
            ),
            ambi_normalization=runtime_config.ambi_normalization,
            ambi_channel_order=runtime_config.channel_order,
        )
    else:
        report.output = analyzer.analyze(
            output_audio,
            output_sr,
            include_loudness=include_loudness,
        )

    track_c_diag = _build_track_c_calibration_diagnostics(
        perceptual_macros=perceptual_macros,
        input_metrics=report.input,
        output_metrics=report.output,
    )
    if track_c_diag is not None:
        report.effective["track_c_calibration"] = track_c_diag

    analysis_path = _resolve_analysis_path(outfile, runtime_config.analysis_out)
    analysis_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    report.analysis_path = str(analysis_path)

    if runtime_config.frames_out is not None:
        frames_path = Path(runtime_config.frames_out)
        write_framewise_csv(frames_path, output_audio, output_sr)
        report.frames_path = str(frames_path)
    if (
        runtime_config.automation_trace_out is not None
        and report.effective.get("automation") is not None
    ):
        report.automation_trace_path = str(Path(runtime_config.automation_trace_out))
    if (
        runtime_config.feature_vector_trace_out is not None
        and report.effective.get("automation") is not None
    ):
        report.feature_vector_trace_path = str(Path(runtime_config.feature_vector_trace_out))


def _resolve_processing_sample_rate(*, input_sr: int, target_sr: int | None) -> int:
    """Return render/output sample-rate after optional target override."""
    if target_sr is None:
        return int(input_sr)
    return int(max(1, target_sr))


def _resample_audio_polyphase(audio: AudioArray, *, src_sr: int, dst_sr: int) -> AudioArray:
    """Deterministically resample audio using rational polyphase filtering."""
    if int(src_sr) == int(dst_sr):
        return np.asarray(audio, dtype=np.float64)
    gcd = math.gcd(int(src_sr), int(dst_sr))
    up = int(dst_sr // gcd)
    down = int(src_sr // gcd)
    resampled = resample_poly(np.asarray(audio, dtype=np.float64), up=up, down=down, axis=0)
    return np.asarray(resampled, dtype=np.float64)


def _sample_rate_action(*, input_sr: int, processing_sr: int) -> str:
    """Encode sample-rate conversion action for reproducibility metadata."""
    if int(input_sr) == int(processing_sr):
        return "none"
    return f"resample:{int(input_sr)}->{int(processing_sr)}"


def _can_stream_convolution(
    config: RenderConfig,
    engine_name: str,
    engine: ReverbEngine,
    *,
    input_sr: int,
) -> bool:
    """Return ``True`` when safe to use the low-RAM streaming convolution path."""
    if config.target_sr is not None and int(config.target_sr) != int(input_sr):
        return False
    if engine_name != "conv":
        return False
    if not isinstance(engine, ConvolutionReverbEngine):
        return False
    if engine.backend_name() == "cuda-cupy":
        return False
    if config.repeat != 1 or config.freeze:
        return False
    if config.normalize_stage == "per-pass":
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
    if config.mod_target != "none" and len(config.mod_sources) > 0:
        return False
    if len(config.mod_routes) > 0:
        return False
    if (
        config.automation_file is not None
        or len(config.automation_points) > 0
        or len(config.feature_vector_lanes) > 0
    ):
        return False
    if config.ambi_order > 0:
        return False
    return True


def _can_stream_algo_proxy(
    config: RenderConfig,
    engine_name: str,
    *,
    input_sr: int,
) -> bool:
    """Return ``True`` when algorithmic proxy streaming is allowed.

    Sample-rate conversion (``target_sr``) and post-render EQ (``lowcut``,
    ``highcut``, ``tilt``) are handled within the proxy-stream execution path
    as a pre-resample pass and a post-EQ pass respectively, so they no longer
    block eligibility.  All time-varying controls (modulation, shimmer, FDN
    time-variance, matrix morphing) remain incompatible with a static proxy IR
    and continue to force the standard in-memory path.
    """
    if engine_name != "algo":
        return False
    if not config.algo_stream:
        return False
    if config.repeat != 1 or config.freeze:
        return False
    if config.normalize_stage == "per-pass":
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
    if config.mod_target != "none" and len(config.mod_sources) > 0:
        return False
    if len(config.mod_routes) > 0:
        return False
    if (
        config.automation_file is not None
        or len(config.automation_points) > 0
        or len(config.feature_vector_lanes) > 0
    ):
        return False
    if config.shimmer:
        return False
    if config.unsafe_self_oscillate:
        return False
    if abs(float(config.mod_rate_hz)) > 1e-9 or abs(float(config.mod_depth_ms)) > 1e-9:
        return False
    if float(config.fdn_tv_rate_hz) > 0.0 or float(config.fdn_tv_depth) > 0.0:
        return False
    if config.fdn_matrix_morph_to is not None:
        return False
    if config.ambi_order > 0:
        return False
    return True


def _prepare_runtime_config(
    config: RenderConfig,
    sr: int,
    input_channels: int,
    infile: Path,
    input_duration_seconds: float,
) -> tuple[RenderConfig, dict[str, Any] | None]:
    """Build an execution config including derived IR runtime metadata."""
    runtime = _apply_beast_mode(
        RenderConfig(**asdict(config)),
        input_duration_seconds=input_duration_seconds,
    )
    runtime = _apply_perceptual_fdn_macros(runtime)
    runtime.ir_blend_base_ir = None
    runtime.ir_blend_composite_ir = None
    ir_runtime_steps: list[dict[str, Any]] = []
    ir_channels = max(1, int(input_channels))
    if runtime.ambi_order > 0:
        ir_channels = ambisonic_channel_count(int(runtime.ambi_order))

    if runtime.self_convolve:
        # Use input file as IR source and force convolution semantics.
        runtime.ir = str(infile)
        if runtime.engine == "auto":
            runtime.engine = "conv"
        ir_runtime_steps.append(
            {
                "mode": "self-convolve",
                "cache_hit": False,
                "ir_path": str(infile),
            }
        )

    if runtime.ir_gen:
        # Generate or reuse deterministic cached IR before render.
        ir_cfg = IRGenConfig(
            mode=runtime.ir_gen_mode,
            length=runtime.ir_gen_length,
            sr=sr,
            channels=ir_channels,
            seed=runtime.ir_gen_seed,
            rt60=runtime.rt60,
            damping=runtime.damping,
            lowcut=runtime.lowcut,
            highcut=runtime.highcut,
            tilt=runtime.tilt,
            target_lufs=runtime.target_lufs,
            true_peak=runtime.use_true_peak,
            mod_depth_ms=runtime.mod_depth_ms,
            mod_rate_hz=runtime.mod_rate_hz,
            fdn_lines=runtime.fdn_lines,
            fdn_matrix=runtime.fdn_matrix,
            fdn_tv_rate_hz=runtime.fdn_tv_rate_hz,
            fdn_tv_depth=runtime.fdn_tv_depth,
            fdn_tv_seed=runtime.fdn_tv_seed,
            fdn_dfm_delays_ms=runtime.fdn_dfm_delays_ms,
            fdn_sparse=runtime.fdn_sparse,
            fdn_sparse_degree=runtime.fdn_sparse_degree,
            fdn_rt60_low=runtime.fdn_rt60_low,
            fdn_rt60_mid=runtime.fdn_rt60_mid,
            fdn_rt60_high=runtime.fdn_rt60_high,
            fdn_rt60_tilt=runtime.fdn_rt60_tilt,
            fdn_tonal_correction_strength=runtime.fdn_tonal_correction_strength,
            fdn_xover_low_hz=runtime.fdn_xover_low_hz,
            fdn_xover_high_hz=runtime.fdn_xover_high_hz,
            fdn_link_filter=runtime.fdn_link_filter,
            fdn_link_filter_hz=runtime.fdn_link_filter_hz,
            fdn_link_filter_mix=runtime.fdn_link_filter_mix,
            fdn_graph_topology=runtime.fdn_graph_topology,
            fdn_graph_degree=runtime.fdn_graph_degree,
            fdn_graph_seed=runtime.fdn_graph_seed,
            fdn_spatial_coupling_mode=runtime.fdn_spatial_coupling_mode,
            fdn_spatial_coupling_strength=runtime.fdn_spatial_coupling_strength,
            fdn_nonlinearity=runtime.fdn_nonlinearity,
            fdn_nonlinearity_amount=runtime.fdn_nonlinearity_amount,
            fdn_nonlinearity_drive=runtime.fdn_nonlinearity_drive,
            room_size_macro=runtime.room_size_macro,
            clarity_macro=runtime.clarity_macro,
            warmth_macro=runtime.warmth_macro,
            envelopment_macro=runtime.envelopment_macro,
        )
        cache_dir = Path(runtime.ir_gen_cache_dir)
        _, _, meta, wav_path, cache_hit = generate_or_load_cached_ir(ir_cfg, cache_dir=cache_dir)
        runtime.ir = str(wav_path)
        if runtime.engine == "auto":
            runtime.engine = "conv"
        ir_runtime_steps.append(
            {
                "mode": runtime.ir_gen_mode,
                "cache_hit": cache_hit,
                "ir_path": str(wav_path),
                "meta": meta,
            }
        )

    if len(runtime.ir_blend) > 0:
        base_path = runtime.ir
        if base_path is None:
            msg = "IR blending requires base IR source."
            raise ValueError(msg)
        blend_paths = tuple(Path(path) for path in runtime.ir_blend)
        blend_mix = resolve_blend_mix_values(runtime.ir_blend_mix, len(blend_paths))
        blend_cfg = IRMorphConfig(
            mode=cast(IRMorphMode, runtime.ir_blend_mode),
            alpha=0.5,
            early_ms=float(runtime.ir_blend_early_ms),
            early_alpha=runtime.ir_blend_early_alpha,
            late_alpha=runtime.ir_blend_late_alpha,
            align_decay=bool(runtime.ir_blend_align_decay),
            phase_coherence=float(runtime.ir_blend_phase_coherence),
            spectral_smooth_bins=int(runtime.ir_blend_spectral_smooth_bins),
            mismatch_policy=runtime.ir_blend_mismatch_policy,
        )
        _, _, blend_meta, blended_path, blend_cache_hit = generate_or_load_cached_blended_ir(
            base_ir_path=Path(base_path),
            blend_ir_paths=blend_paths,
            blend_mix=blend_mix,
            config=blend_cfg,
            cache_dir=Path(runtime.ir_blend_cache_dir),
            target_sr=sr,
        )
        runtime.ir = str(blended_path)
        runtime.ir_blend_base_ir = str(base_path)
        runtime.ir_blend_composite_ir = str(blended_path)
        if runtime.engine == "auto":
            runtime.engine = "conv"
        ir_runtime_steps.append(
            {
                "mode": "ir-blend",
                "cache_hit": blend_cache_hit,
                "ir_path": str(blended_path),
                "meta": blend_meta,
            }
        )

    ir_runtime: dict[str, Any] | None = None
    if len(ir_runtime_steps) == 1:
        ir_runtime = ir_runtime_steps[0]
    elif len(ir_runtime_steps) > 1:
        ir_runtime = {
            "mode": "composite",
            "steps": ir_runtime_steps,
            "ir_path": str(runtime.ir) if runtime.ir is not None else None,
        }

    return runtime, ir_runtime


def _apply_beast_mode(config: RenderConfig, input_duration_seconds: float) -> RenderConfig:
    """Scale key reverb parameters for aggressive freeze-like tails."""
    factor = max(1.0, float(config.beast_mode))
    if factor <= 1.0:
        return config

    # Intentional in-place scaling of a copied config object.
    scaled = config
    scaled.rt60 = max(0.1, scaled.rt60 * factor)
    scaled.pre_delay_ms = max(0.0, scaled.pre_delay_ms * factor)
    scaled.damping = float(np.clip(scaled.damping * factor, 0.0, 1.0))
    scaled.width = float(np.clip(scaled.width * factor, 0.0, 2.0))
    scaled.mod_depth_ms = max(0.0, scaled.mod_depth_ms * factor)
    scaled.mod_rate_hz = max(0.0, scaled.mod_rate_hz * factor)
    scaled.allpass_gain = float(np.clip(scaled.allpass_gain * np.sqrt(factor), -0.99, 0.99))
    if len(scaled.allpass_gains) > 0:
        scaled.allpass_gains = tuple(
            float(np.clip(gain * np.sqrt(factor), -0.99, 0.99)) for gain in scaled.allpass_gains
        )
    if len(scaled.allpass_delays_ms) > 0:
        scaled.allpass_delays_ms = tuple(
            max(0.1, float(delay) * factor) for delay in scaled.allpass_delays_ms
        )
    if len(scaled.comb_delays_ms) > 0:
        scaled.comb_delays_ms = tuple(
            max(0.1, float(delay) * factor) for delay in scaled.comb_delays_ms
        )

    scaled.wet = float(np.clip(scaled.wet * factor, 0.0, 1.0))
    scaled.dry = float(np.clip(scaled.dry / factor, 0.0, 1.0))
    scaled.repeat = int(np.clip(round(scaled.repeat * np.sqrt(factor)), 1, 32))

    scaled.shimmer_mix = float(np.clip(scaled.shimmer_mix * factor, 0.0, 1.0))
    shimmer_feedback_max = 1.25 if scaled.unsafe_self_oscillate else 0.98
    scaled.shimmer_feedback = float(
        np.clip(scaled.shimmer_feedback * factor, 0.0, shimmer_feedback_max)
    )
    scaled.bloom = max(0.0, scaled.bloom * factor)

    if scaled.tail_limit is not None:
        scaled.tail_limit = max(0.0, scaled.tail_limit * factor)
    elif scaled.engine in {"conv", "auto"} and (
        scaled.ir is not None or scaled.ir_gen or scaled.self_convolve or len(scaled.ir_blend) > 0
    ):
        baseline = max(1.0, input_duration_seconds * 0.5)
        scaled.tail_limit = baseline * factor

    if scaled.ir_gen:
        scaled.ir_gen_length = max(0.1, scaled.ir_gen_length * factor)

    return scaled


def _apply_perceptual_fdn_macros(config: RenderConfig) -> RenderConfig:
    """Map perceptual macro controls into low-level reverb parameters."""
    mapped = config
    room_size = float(np.clip(mapped.room_size_macro, -1.0, 1.0))
    clarity = float(np.clip(mapped.clarity_macro, -1.0, 1.0))
    warmth = float(np.clip(mapped.warmth_macro, -1.0, 1.0))
    envelopment = float(np.clip(mapped.envelopment_macro, -1.0, 1.0))
    mapped.room_size_macro = room_size
    mapped.clarity_macro = clarity
    mapped.warmth_macro = warmth
    mapped.envelopment_macro = envelopment

    if max(abs(room_size), abs(clarity), abs(warmth), abs(envelopment)) <= 1e-9:
        return mapped

    calibration = _resolve_track_c_calibration_targets(
        room_size=room_size,
        clarity=clarity,
        warmth=warmth,
        envelopment=envelopment,
    )

    mapped.pre_delay_ms = float(
        max(
            0.0,
            mapped.pre_delay_ms * float(calibration["pre_delay_scale"]),
        )
    )
    mapped.width = float(
        np.clip(
            mapped.width * float(calibration["width_scale"]),
            0.0,
            2.0,
        )
    )
    mapped.wet = float(
        np.clip(
            mapped.wet * float(calibration["wet_scale"]),
            0.0,
            1.0,
        )
    )
    mapped.dry = float(
        np.clip(
            mapped.dry * float(calibration["dry_scale"]),
            0.0,
            1.0,
        )
    )
    mapped.algo_decorrelation_front = float(
        np.clip(
            mapped.algo_decorrelation_front + float(calibration["decorrelation_front_delta"]),
            0.0,
            1.0,
        )
    )
    mapped.algo_decorrelation_rear = float(
        np.clip(
            mapped.algo_decorrelation_rear + float(calibration["decorrelation_rear_delta"]),
            0.0,
            1.0,
        )
    )
    mapped.algo_decorrelation_top = float(
        np.clip(
            mapped.algo_decorrelation_top + float(calibration["decorrelation_top_delta"]),
            0.0,
            1.0,
        )
    )

    if mapped.fdn_link_filter == "none":
        if warmth > 0.25:
            mapped.fdn_link_filter = "lowpass"
            mapped.fdn_link_filter_hz = float(
                np.clip(mapped.fdn_link_filter_hz * np.power(2.0, -0.65 * warmth), 120.0, 12_000.0)
            )
            mapped.fdn_link_filter_mix = float(np.clip(0.30 + (0.50 * warmth), 0.0, 1.0))
        elif clarity > 0.35:
            mapped.fdn_link_filter = "highpass"
            mapped.fdn_link_filter_hz = float(
                np.clip(mapped.fdn_link_filter_hz * np.power(2.0, 0.45 * clarity), 80.0, 12_000.0)
            )
            mapped.fdn_link_filter_mix = float(np.clip(0.20 + (0.35 * clarity), 0.0, 1.0))

    return mapped


def _resolve_track_c_calibration_targets(
    *,
    room_size: float,
    clarity: float,
    warmth: float,
    envelopment: float,
) -> dict[str, float]:
    """Return calibrated perceptual macro transfer targets."""
    # Conservative bounds keep typo-level macros from turning into mix-destroying chaos.
    return {
        "pre_delay_scale": float(
            np.clip(np.power(2.0, (0.45 * room_size) + (0.25 * clarity)), 0.5, 2.0)
        ),
        "width_scale": float(np.clip(np.power(2.0, 0.35 * envelopment), 0.6, 2.0)),
        "wet_scale": float(
            np.clip(np.power(2.0, (0.20 * envelopment) - (0.16 * clarity)), 0.55, 1.55)
        ),
        "dry_scale": float(
            np.clip(np.power(2.0, (0.20 * clarity) - (0.15 * envelopment)), 0.55, 1.55)
        ),
        "decorrelation_front_delta": float(0.18 * envelopment),
        "decorrelation_rear_delta": float(0.28 * envelopment),
        "decorrelation_top_delta": float(0.25 * envelopment),
        # Track C calibration targets used for diagnostics.
        "banded_rt60_scale_low": float(
            np.clip(
                np.power(2.0, (0.30 * room_size) + (0.35 * warmth) - (0.22 * clarity)),
                0.5,
                2.5,
            )
        ),
        "banded_rt60_scale_mid": float(
            np.clip(
                np.power(2.0, (0.28 * room_size) + (0.10 * warmth) - (0.30 * clarity)),
                0.5,
                2.5,
            )
        ),
        "banded_rt60_scale_high": float(
            np.clip(
                np.power(2.0, (0.22 * room_size) - (0.45 * warmth) - (0.45 * clarity)),
                0.35,
                2.0,
            )
        ),
        "spectral_tilt_target": float((1.8 * warmth) - (1.2 * clarity)),
        "clarity_proxy_target": float(0.55 * clarity),
        "decay_shape_target": float((0.60 * room_size) + (0.40 * envelopment) - (0.35 * clarity)),
        "error_envelope_spectral_tilt": 0.55,
        "error_envelope_clarity_proxy": 0.35,
        "error_envelope_decay_shape": 0.45,
    }


def _build_perceptual_macro_summary(
    requested: RenderConfig,
    resolved: RenderConfig,
) -> dict[str, Any] | None:
    """Build a reproducible summary of perceptual-macro resolution."""
    input_macros = {
        "room_size_macro": float(requested.room_size_macro),
        "clarity_macro": float(requested.clarity_macro),
        "warmth_macro": float(requested.warmth_macro),
        "envelopment_macro": float(requested.envelopment_macro),
    }
    if max(abs(value) for value in input_macros.values()) <= 1e-9:
        return None
    calibration = _resolve_track_c_calibration_targets(
        room_size=float(input_macros["room_size_macro"]),
        clarity=float(input_macros["clarity_macro"]),
        warmth=float(input_macros["warmth_macro"]),
        envelopment=float(input_macros["envelopment_macro"]),
    )

    numeric_keys = (
        "rt60",
        "pre_delay_ms",
        "damping",
        "width",
        "wet",
        "dry",
        "fdn_rt60_tilt",
        "fdn_tonal_correction_strength",
        "algo_decorrelation_front",
        "algo_decorrelation_rear",
        "algo_decorrelation_top",
    )
    resolved_values: dict[str, Any] = {key: float(getattr(resolved, key)) for key in numeric_keys}
    resolved_values["fdn_link_filter"] = str(resolved.fdn_link_filter)
    resolved_values["fdn_link_filter_hz"] = float(resolved.fdn_link_filter_hz)
    resolved_values["fdn_link_filter_mix"] = float(resolved.fdn_link_filter_mix)

    delta: dict[str, float] = {}
    for key in numeric_keys:
        delta[key] = float(getattr(resolved, key) - getattr(requested, key))
    return {
        "calibration_version": TRACK_C_CALIBRATION_VERSION,
        "input": input_macros,
        "resolved": resolved_values,
        "delta_from_requested": delta,
        "calibration_targets": calibration,
    }


def _build_track_c_calibration_diagnostics(
    *,
    perceptual_macros: dict[str, Any] | None,
    input_metrics: dict[str, Any] | None,
    output_metrics: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Compute Track C calibration diagnostics from targets and measured proxies."""
    if not isinstance(perceptual_macros, dict):
        return None
    targets_raw = perceptual_macros.get("calibration_targets")
    if not isinstance(targets_raw, dict):
        return None

    targets = {
        str(key): float(value)
        for key, value in targets_raw.items()
        if isinstance(value, (int, float))
    }
    measured: dict[str, float] = {}
    errors: dict[str, float] = {}
    within_envelope = True

    if isinstance(input_metrics, dict) and isinstance(output_metrics, dict):
        slope_in = float(input_metrics.get("spectral_slope", 0.0))
        slope_out = float(output_metrics.get("spectral_slope", 0.0))
        transient_in = float(input_metrics.get("transient_density", 0.0))
        transient_out = float(output_metrics.get("transient_density", 0.0))
        dyn_in = float(input_metrics.get("dynamic_range", 0.0))
        dyn_out = float(output_metrics.get("dynamic_range", 0.0))

        measured["spectral_tilt_proxy"] = float(slope_out - slope_in)
        measured["clarity_proxy"] = float(transient_out - transient_in)
        measured["decay_shape_proxy"] = float((dyn_out - dyn_in) / max(1e-9, abs(dyn_in) + 1.0))

        target_tilt = float(targets.get("spectral_tilt_target", 0.0)) * 0.10
        target_clarity = float(targets.get("clarity_proxy_target", 0.0))
        target_decay = float(targets.get("decay_shape_target", 0.0)) * 0.25

        errors["spectral_tilt_error"] = float(measured["spectral_tilt_proxy"] - target_tilt)
        errors["clarity_proxy_error"] = float(measured["clarity_proxy"] - target_clarity)
        errors["decay_shape_error"] = float(measured["decay_shape_proxy"] - target_decay)

        if abs(errors["spectral_tilt_error"]) > float(
            targets.get("error_envelope_spectral_tilt", 0.55)
        ):
            within_envelope = False
        if abs(errors["clarity_proxy_error"]) > float(
            targets.get("error_envelope_clarity_proxy", 0.35)
        ):
            within_envelope = False
        if abs(errors["decay_shape_error"]) > float(
            targets.get("error_envelope_decay_shape", 0.45)
        ):
            within_envelope = False

    return {
        "version": TRACK_C_CALIBRATION_VERSION,
        "targets": targets,
        "measured": measured,
        "errors": errors,
        "within_envelope": bool(within_envelope),
    }


def _resolve_engine_name(config: RenderConfig) -> str:
    """Resolve concrete engine name from explicit/auto config."""
    engine_name = str(config.engine).strip().lower()
    if engine_name == "auto":
        return "conv" if config.ir is not None else "algo"
    return engine_name


def _resolve_engine(
    config: RenderConfig,
    device: str,
    *,
    engine_name: str | None = None,
) -> tuple[str, ReverbEngine, str]:
    """Resolve engine name and instantiate concrete engine instance."""
    resolved_name = _resolve_engine_name(config) if engine_name is None else str(engine_name)

    if resolved_name == "conv":
        if config.ir is None:
            msg = "Convolution engine requires --ir when --engine conv is selected"
            raise ValueError(msg)
        return (
            "conv",
            ConvolutionReverbEngine(
                _build_convolution_config(
                    config,
                    ir_path=config.ir,
                    device=device,
                )
            ),
            device,
        )

    algo_device = device if device in {"cpu", "mps"} else "cpu"
    return (
        "algo",
        AlgoReverbEngine(
            AlgoReverbConfig(
                rt60=config.rt60,
                pre_delay_ms=config.pre_delay_ms,
                damping=config.damping,
                width=config.width,
                mod_depth_ms=config.mod_depth_ms,
                mod_rate_hz=config.mod_rate_hz,
                allpass_stages=config.allpass_stages,
                allpass_gain=config.allpass_gain,
                allpass_gains=config.allpass_gains,
                allpass_delays_ms=config.allpass_delays_ms,
                comb_delays_ms=config.comb_delays_ms,
                fdn_lines=config.fdn_lines,
                fdn_matrix=config.fdn_matrix,
                fdn_tv_rate_hz=config.fdn_tv_rate_hz,
                fdn_tv_depth=config.fdn_tv_depth,
                fdn_tv_seed=config.fdn_tv_seed,
                fdn_dfm_delays_ms=config.fdn_dfm_delays_ms,
                fdn_sparse=config.fdn_sparse,
                fdn_sparse_degree=config.fdn_sparse_degree,
                fdn_cascade=config.fdn_cascade,
                fdn_cascade_mix=config.fdn_cascade_mix,
                fdn_cascade_delay_scale=config.fdn_cascade_delay_scale,
                fdn_cascade_rt60_ratio=config.fdn_cascade_rt60_ratio,
                fdn_rt60_low=config.fdn_rt60_low,
                fdn_rt60_mid=config.fdn_rt60_mid,
                fdn_rt60_high=config.fdn_rt60_high,
                fdn_rt60_tilt=config.fdn_rt60_tilt,
                fdn_tonal_correction_strength=config.fdn_tonal_correction_strength,
                fdn_xover_low_hz=config.fdn_xover_low_hz,
                fdn_xover_high_hz=config.fdn_xover_high_hz,
                fdn_link_filter=config.fdn_link_filter,
                fdn_link_filter_hz=config.fdn_link_filter_hz,
                fdn_link_filter_mix=config.fdn_link_filter_mix,
                fdn_graph_topology=config.fdn_graph_topology,
                fdn_graph_degree=config.fdn_graph_degree,
                fdn_graph_seed=config.fdn_graph_seed,
                fdn_matrix_morph_to=config.fdn_matrix_morph_to,
                fdn_matrix_morph_seconds=config.fdn_matrix_morph_seconds,
                fdn_spatial_coupling_mode=config.fdn_spatial_coupling_mode,
                fdn_spatial_coupling_strength=config.fdn_spatial_coupling_strength,
                fdn_nonlinearity=config.fdn_nonlinearity,
                fdn_nonlinearity_amount=config.fdn_nonlinearity_amount,
                fdn_nonlinearity_drive=config.fdn_nonlinearity_drive,
                room_size_macro=config.room_size_macro,
                clarity_macro=config.clarity_macro,
                warmth_macro=config.warmth_macro,
                envelopment_macro=config.envelopment_macro,
                algo_decorrelation_front=config.algo_decorrelation_front,
                algo_decorrelation_rear=config.algo_decorrelation_rear,
                algo_decorrelation_top=config.algo_decorrelation_top,
                wet=config.wet,
                dry=config.dry,
                block_size=config.block_size,
                shimmer=config.shimmer,
                shimmer_semitones=config.shimmer_semitones,
                shimmer_mix=config.shimmer_mix,
                shimmer_feedback=config.shimmer_feedback,
                shimmer_highcut=config.shimmer_highcut,
                shimmer_lowcut=config.shimmer_lowcut,
                shimmer_spatial=config.shimmer_spatial,
                shimmer_spread_cents=config.shimmer_spread_cents,
                shimmer_decorrelation_ms=config.shimmer_decorrelation_ms,
                unsafe_self_oscillate=config.unsafe_self_oscillate,
                unsafe_loop_gain=config.unsafe_loop_gain,
                output_layout=config.output_layout,
                device=algo_device,
            )
        ),
        algo_device,
    )


def _resolve_analysis_path(outfile: Path, analysis_out: str | None) -> Path:
    """Resolve analysis JSON output path."""
    if analysis_out is not None:
        return Path(analysis_out)
    return Path(f"{outfile}.analysis.json")


def _build_per_pass_processor(config: RenderConfig, sr: int) -> PassProcessor:
    """Build per-pass post processor used by repeat chaining."""
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

    def passthrough(audio: AudioArray, pass_idx: int, total_passes: int) -> AudioArray:
        _ = pass_idx, total_passes
        return np.asarray(audio, dtype=np.float64)

    return passthrough


def _should_include_loudness(config: RenderConfig) -> bool:
    """Return whether analysis should include LUFS/true-peak metrics."""
    return any(
        value is not None
        for value in (
            config.target_lufs,
            config.target_peak_dbfs,
            config.repeat_target_lufs,
            config.repeat_target_peak_dbfs,
        )
    )


def _algo_tail_padding_seconds(config: RenderConfig, sr: int) -> float:
    """Compute explicit tail render duration for algorithmic reverbs.

    When shimmer is enabled with high feedback the feedback loop outlives the
    reverb tail by a substantial margin.  The shimmer processor is called once
    per block; each call multiplies the feedback state by `feedback`.  Decay to
    threshold (1e-6) takes ``ceil(log(1e-6) / log(feedback))`` block calls.
    We add that shimmer tail on top of the RT60 tail so ``_complete_tail_to_zero``
    never needs to hard-clip an active signal.
    """
    pre_delay = max(0.0, float(config.pre_delay_ms)) / 1000.0
    base_tail = max(0.25, float(config.rt60) + pre_delay)

    if not config.shimmer:
        return base_tail

    feedback = float(np.clip(config.shimmer_feedback, 0.0, 0.98))
    if feedback < 1e-9:
        return base_tail

    # block_size/sr = seconds per shimmer process() call
    block_seconds = max(1, int(config.block_size)) / max(1.0, float(sr))
    # blocks to decay below -120 dBFS (1e-6 linear)
    blocks_to_silence = int(np.ceil(np.log(1e-6) / np.log(feedback)))
    shimmer_tail = blocks_to_silence * block_seconds
    return base_tail + shimmer_tail


def _append_tail_padding(audio: AudioArray, sr: int, tail_seconds: float) -> AudioArray:
    """Append silence so stateful reverbs can decay naturally."""
    tail_samples = int(np.ceil(max(0.0, tail_seconds) * float(sr)))
    if tail_samples <= 0:
        return audio
    padding = np.zeros((tail_samples, audio.shape[1]), dtype=np.float64)
    return np.concatenate((audio, padding), axis=0)


def _tail_zero_hold_samples(sr: int, hold_ms: float) -> int:
    """Return trailing zero-hold samples appended to finalize tail completion."""
    return max(1, round(float(sr) * (max(0.0, float(hold_ms)) / 1000.0)))


def _tail_fade_samples(sr: int, fade_ms: float = 3.0) -> int:
    """Return click-safe fade-out sample count used before hard-zero tail."""
    return max(1, round(float(sr) * (max(0.0, float(fade_ms)) / 1000.0)))


def _resolve_tail_fade_samples(sr: int, first_zero_frame: int) -> int:
    """Choose fade length that is short and cannot dominate short clips."""
    if first_zero_frame <= 1:
        return 1
    base = _tail_fade_samples(sr)
    local_cap = max(1, first_zero_frame // 4)
    return min(base, local_cap)


def _tail_fade_envelope(length: int) -> npt.NDArray[np.float64]:
    """Raised-cosine fade envelope from 1.0 to 0.0."""
    n = max(1, int(length))
    if n == 1:
        return np.zeros((1,), dtype=np.float64)
    phase = np.linspace(0.0, np.pi, n, dtype=np.float64)
    return np.asarray(0.5 * (1.0 + np.cos(phase)), dtype=np.float64)


def _tail_threshold_linear(threshold_db: float) -> float:
    """Convert tail stop threshold in dBFS to linear amplitude."""
    db = float(np.clip(threshold_db, -240.0, 0.0))
    return float(np.power(10.0, db / 20.0))


def _complete_stream_file_tail_to_zero(
    path: Path,
    *,
    threshold: float = 1e-6,
    hold_ms: float = 10.0,
    metric: str = "peak",
    min_frames: int = 0,
    scan_block_frames: int = 65536,
    write_block_frames: int = 65536,
) -> int:
    """Finalize streamed output so tail decays into an exact zero-value hold."""
    threshold_value = float(max(0.0, threshold))
    scan_frames = max(1, int(scan_block_frames))
    write_frames = max(1, int(write_block_frames))

    with sf.SoundFile(str(path), mode="r+") as stream_file:
        total_frames = int(stream_file.frames)
        channels = int(stream_file.channels)
        hold_samples = _tail_zero_hold_samples(int(stream_file.samplerate), hold_ms)

        if total_frames <= 0:
            if hold_samples > 0:
                stream_file.seek(0, whence=sf.SEEK_END)
                stream_file.write(np.zeros((hold_samples, channels), dtype=np.float64))
            return hold_samples

        last_active = -1
        cursor = total_frames
        while cursor > 0:
            start = max(0, cursor - scan_frames)
            frames = cursor - start
            stream_file.seek(start, whence=sf.SEEK_SET)
            block = np.asarray(
                stream_file.read(frames, dtype="float64", always_2d=True),
                dtype=np.float64,
            )
            if block.shape[0] > 0:
                if str(metric).strip().lower() == "rms":
                    envelope = np.sqrt(np.mean(np.square(block), axis=1))
                else:
                    envelope = np.max(np.abs(block), axis=1)
                active = np.flatnonzero(envelope > threshold_value)
                if active.size > 0:
                    last_active = start + int(active[-1])
                    break
            cursor = start

        first_zero_frame = last_active + 1 if last_active >= 0 else 0
        if last_active >= 0:
            target_frames = max(int(min_frames), first_zero_frame + hold_samples)
        else:
            target_frames = max(total_frames, int(min_frames), hold_samples)

        # Apply a short fade-out before hard-zero tail to avoid end clicks.
        if last_active >= 0 and first_zero_frame > 0:
            fade_samples = _resolve_tail_fade_samples(
                int(stream_file.samplerate),
                first_zero_frame,
            )
            fade_start = first_zero_frame - fade_samples
            stream_file.seek(fade_start, whence=sf.SEEK_SET)
            fade_region = np.asarray(
                stream_file.read(fade_samples, dtype="float64", always_2d=True),
                dtype=np.float64,
            )
            if fade_region.shape[0] > 0:
                env = _tail_fade_envelope(fade_region.shape[0])[:, np.newaxis]
                stream_file.seek(fade_start, whence=sf.SEEK_SET)
                stream_file.write(np.asarray(fade_region * env, dtype=np.float64))

        # Force exact zeros from first_zero_frame onward up to target length.
        zero_start = first_zero_frame
        zero_end = min(total_frames, target_frames)
        zero_existing = max(0, zero_end - zero_start)
        if zero_existing > 0:
            zero_block = np.zeros((write_frames, channels), dtype=np.float64)
            write_cursor = zero_start
            remaining = zero_existing
            while remaining > 0:
                frames = min(write_frames, remaining)
                stream_file.seek(write_cursor, whence=sf.SEEK_SET)
                stream_file.write(zero_block[:frames, :])
                write_cursor += frames
                remaining -= frames

        append_frames = target_frames - total_frames
        if append_frames > 0:
            zero_block = np.zeros((write_frames, channels), dtype=np.float64)
            stream_file.seek(0, whence=sf.SEEK_END)
            remaining = append_frames
            while remaining > 0:
                frames = min(write_frames, remaining)
                stream_file.write(zero_block[:frames, :])
                remaining -= frames
        elif target_frames < total_frames:
            stream_file.truncate(target_frames)

    return target_frames


def complete_stream_file_tail_to_zero(
    path: Path,
    *,
    threshold: float = 1e-6,
    hold_ms: float = 10.0,
    metric: str = "peak",
    min_frames: int = 0,
    scan_block_frames: int = 65536,
    write_block_frames: int = 65536,
) -> int:
    """Public wrapper for streamed tail completion with click-safe trimming."""
    return _complete_stream_file_tail_to_zero(
        path,
        threshold=threshold,
        hold_ms=hold_ms,
        metric=metric,
        min_frames=min_frames,
        scan_block_frames=scan_block_frames,
        write_block_frames=write_block_frames,
    )


def _complete_tail_to_zero(
    audio: AudioArray,
    sr: int,
    *,
    threshold: float = 1e-6,
    hold_ms: float = 10.0,
    metric: str = "peak",
    min_samples: int = 0,
) -> AudioArray:
    """Ensure rendered output ends with exact zeros after tail decay."""
    x = np.asarray(audio, dtype=np.float64)
    if x.shape[0] == 0:
        return x.copy()

    hold_samples = _tail_zero_hold_samples(sr, hold_ms)
    if str(metric).strip().lower() == "rms":
        envelope = np.sqrt(np.mean(np.square(x), axis=1))
    else:
        envelope = np.max(np.abs(x), axis=1)
    active = np.flatnonzero(envelope > float(max(0.0, threshold)))
    if active.size == 0:
        target_len = max(int(x.shape[0]), int(min_samples), hold_samples)
        out = np.zeros((target_len, x.shape[1]), dtype=np.float64)
        return out

    last_active = int(active[-1])
    first_zero_frame = last_active + 1
    target_len = max(1, int(min_samples), first_zero_frame + hold_samples)
    out = np.zeros((target_len, x.shape[1]), dtype=np.float64)
    fade_samples = _resolve_tail_fade_samples(sr, first_zero_frame)
    fade_start = first_zero_frame - fade_samples
    if fade_start > 0:
        out[:fade_start, :] = x[:fade_start, :]
    if fade_samples > 0:
        fade_region = x[fade_start:first_zero_frame, :]
        env = _tail_fade_envelope(fade_region.shape[0])[:, np.newaxis]
        out[fade_start:first_zero_frame, :] = np.asarray(fade_region * env, dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def complete_tail_to_zero(
    audio: AudioArray,
    sr: int,
    *,
    threshold: float = 1e-6,
    hold_ms: float = 10.0,
    metric: str = "peak",
    min_samples: int = 0,
) -> AudioArray:
    """Public wrapper for click-safe tail completion on in-memory audio."""
    return _complete_tail_to_zero(
        audio,
        sr,
        threshold=threshold,
        hold_ms=hold_ms,
        metric=metric,
        min_samples=min_samples,
    )


def _prepare_spatial_input(audio: AudioArray, config: RenderConfig) -> AudioArray:
    """Prepare input audio for Ambisonics processing when enabled."""
    if config.ambi_order <= 0:
        return np.asarray(audio, dtype=np.float64)

    prepared = np.asarray(audio, dtype=np.float64)
    source_norm = config.ambi_normalization
    source_order = config.channel_order
    if config.ambi_encode_from != "none":
        prepared = encode_bus_to_foa(prepared, source=config.ambi_encode_from)
        source_norm = "sn3d"
        source_order = "acn"
    return convert_ambisonic_convention(
        prepared,
        order=config.ambi_order,
        source_normalization=source_norm,
        source_channel_order=source_order,
        target_normalization="sn3d",
        target_channel_order="acn",
    )


def _apply_spatial_output_transform(audio: AudioArray, config: RenderConfig) -> AudioArray:
    """Apply Ambisonics output transforms (rotation and optional decode)."""
    if config.ambi_order <= 0:
        return np.asarray(audio, dtype=np.float64)

    transformed = np.asarray(audio, dtype=np.float64)
    if abs(float(config.ambi_rotate_yaw_deg)) > 1e-12:
        transformed = rotate_ambisonic_yaw(
            transformed,
            order=config.ambi_order,
            yaw_degrees=config.ambi_rotate_yaw_deg,
            normalization="sn3d",
            channel_order="acn",
        )

    if config.ambi_decode_to == "stereo":
        return decode_foa_to_stereo(
            transformed,
            order=config.ambi_order,
            normalization="sn3d",
            channel_order="acn",
        )

    return convert_ambisonic_convention(
        transformed,
        order=config.ambi_order,
        source_normalization="sn3d",
        source_channel_order="acn",
        target_normalization=config.ambi_normalization,
        target_channel_order=config.channel_order,
    )


def _build_dry_reference_for_automation(
    *,
    engine_name: str,
    input_for_engine: AudioArray,
    rendered: AudioArray,
    config: RenderConfig,
) -> AudioArray:
    """Reconstruct dry-reference signal in rendered channel topology."""
    out_len = int(rendered.shape[0])
    out_channels = int(rendered.shape[1])
    if out_len == 0:
        return np.zeros((0, out_channels), dtype=np.float64)

    if engine_name == "conv":
        return ConvolutionReverbEngine.build_dry_for_output(
            x=input_for_engine,
            out_channels=out_channels,
            out_len=out_len,
            in_layout=config.input_layout,
            out_layout=config.output_layout,
        )

    dry = np.zeros((out_len, out_channels), dtype=np.float64)
    copy_len = min(out_len, int(input_for_engine.shape[0]))
    in_channels = int(input_for_engine.shape[1])
    if in_channels == out_channels:
        dry[:copy_len, :] = input_for_engine[:copy_len, :]
        return dry
    if in_channels == 1 and out_channels > 1:
        dry[:copy_len, :] = np.repeat(input_for_engine[:copy_len, :], out_channels, axis=1)
        return dry
    mapped = min(in_channels, out_channels)
    dry[:copy_len, :mapped] = input_for_engine[:copy_len, :mapped]
    return dry


def _load_runtime_automation_bundle(
    *,
    config: RenderConfig,
    sr: int,
    num_samples: int,
    clamp_overrides: dict[str, tuple[float, float]] | None,
    feature_audio: AudioArray,
) -> AutomationBundle:
    return load_automation_bundle(
        path=None if config.automation_file is None else Path(config.automation_file),
        point_specs=config.automation_points,
        feature_lane_specs=config.feature_vector_lanes,
        feature_audio=feature_audio,
        feature_guide_path=(
            None if config.feature_guide is None else Path(config.feature_guide)
        ),
        feature_guide_policy=config.feature_guide_policy,
        sr=sr,
        num_samples=int(num_samples),
        mode=config.automation_mode,
        block_ms=config.automation_block_ms,
        smoothing_ms=config.automation_smoothing_ms,
        slew_limit_per_s=config.automation_slew_limit_per_s,
        deadband=config.automation_deadband,
        feature_frame_ms=config.feature_vector_frame_ms,
        feature_hop_ms=config.feature_vector_hop_ms,
        clamp_overrides=clamp_overrides,
    )


def _validate_automation_target_domains_for_engine(
    *,
    engine_name: str,
    bundle: AutomationBundle,
) -> None:
    targets = set(bundle.curves.keys())
    if engine_name != "algo":
        unsupported_engine = sorted(
            target for target in targets if target in ENGINE_AUTOMATION_TARGETS
        )
        if len(unsupported_engine) > 0:
            raise ValueError(
                "Automation targets require algorithmic engine: " + ", ".join(unsupported_engine)
            )
    if engine_name != "conv":
        unsupported_conv = sorted(target for target in targets if target in CONV_AUTOMATION_TARGETS)
        if len(unsupported_conv) > 0:
            raise ValueError(
                "Automation targets require convolution engine: " + ", ".join(unsupported_conv)
            )


def _build_convolution_config(
    config: RenderConfig,
    *,
    ir_path: str,
    device: str,
) -> ConvolutionReverbConfig:
    return ConvolutionReverbConfig(
        wet=config.wet,
        dry=config.dry,
        ir_path=ir_path,
        ir_normalize=config.ir_normalize,
        ir_matrix_layout=config.ir_matrix_layout,
        ir_route_map=config.ir_route_map,
        partition_size=config.partition_size,
        tail_limit=config.tail_limit,
        threads=config.threads,
        device=device,
        input_layout=config.input_layout,
        output_layout=config.output_layout,
        route_start=config.conv_route_start,
        route_end=config.conv_route_end,
        route_curve=config.conv_route_curve,
        ambi_order=config.ambi_order,
        ambi_normalization=config.ambi_normalization,
        channel_order=config.channel_order,
        ambi_rotate_yaw_deg=config.ambi_rotate_yaw_deg,
    )


def _render_convolution_variant(
    *,
    input_for_engine: AudioArray,
    sr: int,
    config: RenderConfig,
    ir_path: str,
    device: str,
    repeat_post_processor: PassProcessor,
) -> AudioArray:
    engine = ConvolutionReverbEngine(
        _build_convolution_config(
            config,
            ir_path=ir_path,
            device=device,
        )
    )
    rendered = repeat_process(
        engine=engine,
        audio=input_for_engine,
        sr=sr,
        n=config.repeat,
        post_pass_processor=repeat_post_processor,
    )
    return np.asarray(rendered, dtype=np.float64)


def _estimate_wet_component(
    *,
    rendered: AudioArray,
    dry_reference: AudioArray,
    base_wet: float,
    base_dry: float,
) -> AudioArray:
    if abs(base_wet) <= 1e-9:
        return np.asarray(rendered, dtype=np.float64)
    wet = (
        np.asarray(rendered, dtype=np.float64) - (float(base_dry) * dry_reference)
    ) / float(base_wet)
    return np.asarray(np.nan_to_num(wet, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)


def _apply_convolution_automation_targets(
    *,
    rendered: AudioArray,
    input_for_engine: AudioArray,
    sr: int,
    config: RenderConfig,
    bundle: AutomationBundle,
    engine_device: str,
    repeat_post_processor: PassProcessor,
) -> tuple[AudioArray, dict[str, Any] | None]:
    alpha_curve = bundle.curves.get("ir-blend-alpha")
    if alpha_curve is None:
        return np.asarray(rendered, dtype=np.float64), None

    base_ir = config.ir_blend_base_ir
    composite_ir = config.ir_blend_composite_ir
    if base_ir is None or composite_ir is None:
        msg = "Automation target 'ir-blend-alpha' requires --ir-blend with at least one blend IR."
        raise ValueError(msg)

    current_ir = str(config.ir) if config.ir is not None else str(composite_ir)
    base_path = str(base_ir)
    blend_path = str(composite_ir)

    base_render: AudioArray
    blend_render: AudioArray
    if Path(current_ir) == Path(blend_path):
        blend_render = np.asarray(rendered, dtype=np.float64)
        base_render = _render_convolution_variant(
            input_for_engine=input_for_engine,
            sr=sr,
            config=config,
            ir_path=base_path,
            device=engine_device,
            repeat_post_processor=repeat_post_processor,
        )
    elif Path(current_ir) == Path(base_path):
        base_render = np.asarray(rendered, dtype=np.float64)
        blend_render = _render_convolution_variant(
            input_for_engine=input_for_engine,
            sr=sr,
            config=config,
            ir_path=blend_path,
            device=engine_device,
            repeat_post_processor=repeat_post_processor,
        )
    else:
        base_render = _render_convolution_variant(
            input_for_engine=input_for_engine,
            sr=sr,
            config=config,
            ir_path=base_path,
            device=engine_device,
            repeat_post_processor=repeat_post_processor,
        )
        blend_render = _render_convolution_variant(
            input_for_engine=input_for_engine,
            sr=sr,
            config=config,
            ir_path=blend_path,
            device=engine_device,
            repeat_post_processor=repeat_post_processor,
        )

    n = int(rendered.shape[0])
    alpha = np.asarray(alpha_curve, dtype=np.float64).reshape(-1)
    if alpha.shape[0] != n:
        if alpha.shape[0] <= 1:
            fill = float(alpha[0]) if alpha.shape[0] == 1 else 0.0
            alpha = np.full((n,), fill, dtype=np.float64)
        else:
            src = np.linspace(0.0, 1.0, alpha.shape[0], dtype=np.float64)
            dst = np.linspace(0.0, 1.0, n, dtype=np.float64)
            alpha = np.asarray(np.interp(dst, src, alpha.astype(np.float64)), dtype=np.float64)
    alpha = np.asarray(np.clip(alpha, 0.0, 1.0), dtype=np.float64)

    if abs(float(config.wet)) <= 1e-9:
        mixed = ((1.0 - alpha)[:, np.newaxis] * base_render) + (alpha[:, np.newaxis] * blend_render)
        mixed = np.asarray(np.nan_to_num(mixed, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
    else:
        dry_reference = _build_dry_reference_for_automation(
            engine_name="conv",
            input_for_engine=input_for_engine,
            rendered=rendered,
            config=config,
        )
        wet_base = _estimate_wet_component(
            rendered=base_render,
            dry_reference=dry_reference,
            base_wet=float(config.wet),
            base_dry=float(config.dry),
        )
        wet_blend = _estimate_wet_component(
            rendered=blend_render,
            dry_reference=dry_reference,
            base_wet=float(config.wet),
            base_dry=float(config.dry),
        )
        wet_mix = ((1.0 - alpha)[:, np.newaxis] * wet_base) + (alpha[:, np.newaxis] * wet_blend)
        mixed = (float(config.dry) * dry_reference) + (float(config.wet) * wet_mix)
        mixed = np.asarray(np.nan_to_num(mixed, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)

    summary = {
        "ir_blend_alpha_applied": True,
        "base_ir": base_path,
        "blended_ir": blend_path,
        "alpha_min": float(np.min(alpha)) if alpha.size > 0 else 0.0,
        "alpha_max": float(np.max(alpha)) if alpha.size > 0 else 0.0,
        "alpha_mean": float(np.mean(alpha)) if alpha.size > 0 else 0.0,
    }
    return mixed, summary


def _non_default_settings(config: RenderConfig) -> dict[str, Any]:
    """Return config entries that differ from ``RenderConfig`` defaults."""
    defaults = asdict(RenderConfig())
    current = asdict(config)
    return {key: value for key, value in current.items() if defaults.get(key) != value}


def _resolve_output_subtype(mode: str) -> str | None:
    """Map CLI output subtype mode to libsndfile subtype string."""
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


def _resolve_output_format(
    *,
    mode: str,
    outfile: Path,
    estimated_bytes: int | None = None,
) -> str | None:
    """Resolve container format for long renders (WAV/W64/RF64)."""
    normalized = str(mode).strip().lower()
    if normalized not in {"auto", "wav", "w64", "rf64"}:
        msg = f"Unsupported output container mode: {mode}"
        raise ValueError(msg)
    if normalized == "wav":
        return "WAV"
    if normalized == "w64":
        return "W64"
    if normalized == "rf64":
        return "RF64"

    suffix = outfile.suffix.strip().lower()
    if suffix == ".w64":
        return "W64"
    if suffix == ".rf64":
        return "RF64"
    if (
        suffix in {".wav", ""}
        and estimated_bytes is not None
        and estimated_bytes >= 3_800_000_000
    ):
        return "W64"
    return None


def _apply_final_peak_normalization(
    audio: AudioArray,
    mode: str,
    input_peak_linear: float,
    target_dbfs: float | None,
) -> AudioArray:
    """Apply final optional peak normalization step after DSP processing."""
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
        return np.asarray(audio * gain, dtype=np.float64)

    msg = f"Unsupported output_peak_norm mode: {mode}"
    raise ValueError(msg)
