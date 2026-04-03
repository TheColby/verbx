"""Realtime command for live duplex monitoring."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import typer
from rich.console import Console
from rich.table import Table

from verbx.config import RenderConfig
from verbx.core.algo_proxy import render_algo_proxy_ir
from verbx.core.convolution_reverb import (
    ConvolutionReverbConfig,
    ConvolutionReverbEngine,
    LiveConvolutionProcessor,
)
from verbx.core.fdn_capabilities import (
    FDN_GRAPH_TOPOLOGY_CHOICES,
    FDN_LINK_FILTER_CHOICES,
    FDN_MATRIX_CHOICES,
    normalize_fdn_graph_topology_name,
    normalize_fdn_link_filter_name,
    normalize_fdn_matrix_name,
)
from verbx.io.realtime import (
    RealtimeDeviceInfo,
    RealtimeSessionSummary,
    list_audio_devices,
    resolve_audio_device,
    run_realtime_duplex,
)

console = Console()
_FDN_SPATIAL_COUPLING_CHOICES = frozenset(
    {"none", "adjacent", "front_rear", "bed_top", "all_to_all"}
)
_FDN_NONLINEARITY_CHOICES = frozenset({"none", "tanh", "softclip"})
_REALTIME_FREEZE_LOOP_GAIN = 1.002
_REALTIME_FREEZE_PROXY_SECONDS = 120.0
_ALGO_ONLY_SWITCHES: dict[str, str] = {
    "rt60": "--rt60",
    "pre_delay_ms": "--pre-delay-ms",
    "damping": "--damping",
    "width": "--width",
    "mod_depth_ms": "--mod-depth-ms",
    "mod_rate_hz": "--mod-rate-hz",
    "allpass_stages": "--allpass-stages",
    "allpass_gain": "--allpass-gain",
    "allpass_gains": "--allpass-gain",
    "allpass_delays_ms": "--allpass-delays-ms",
    "comb_delays_ms": "--comb-delays-ms",
    "fdn_lines": "--fdn-lines",
    "fdn_matrix": "--fdn-matrix",
    "fdn_tv_rate_hz": "--fdn-tv-rate-hz",
    "fdn_tv_depth": "--fdn-tv-depth",
    "fdn_dfm_delays_ms": "--fdn-dfm-delays-ms",
    "fdn_sparse": "--fdn-sparse",
    "fdn_sparse_degree": "--fdn-sparse-degree",
    "fdn_cascade": "--fdn-cascade",
    "fdn_cascade_mix": "--fdn-cascade-mix",
    "fdn_cascade_delay_scale": "--fdn-cascade-delay-scale",
    "fdn_cascade_rt60_ratio": "--fdn-cascade-rt60-ratio",
    "fdn_rt60_low": "--fdn-rt60-low",
    "fdn_rt60_mid": "--fdn-rt60-mid",
    "fdn_rt60_high": "--fdn-rt60-high",
    "fdn_rt60_tilt": "--fdn-rt60-tilt",
    "fdn_tonal_correction_strength": "--fdn-tonal-correction-strength",
    "fdn_xover_low_hz": "--fdn-xover-low-hz",
    "fdn_xover_high_hz": "--fdn-xover-high-hz",
    "fdn_link_filter": "--fdn-link-filter",
    "fdn_link_filter_hz": "--fdn-link-filter-hz",
    "fdn_link_filter_mix": "--fdn-link-filter-mix",
    "fdn_graph_topology": "--fdn-graph-topology",
    "fdn_graph_degree": "--fdn-graph-degree",
    "fdn_graph_seed": "--fdn-graph-seed",
    "fdn_matrix_morph_to": "--fdn-matrix-morph-to",
    "fdn_matrix_morph_seconds": "--fdn-matrix-morph-seconds",
    "fdn_spatial_coupling_mode": "--fdn-spatial-coupling-mode",
    "fdn_spatial_coupling_strength": "--fdn-spatial-coupling-strength",
    "fdn_nonlinearity": "--fdn-nonlinearity",
    "fdn_nonlinearity_amount": "--fdn-nonlinearity-amount",
    "fdn_nonlinearity_drive": "--fdn-nonlinearity-drive",
    "room_size_macro": "--room-size-macro",
    "clarity_macro": "--clarity-macro",
    "warmth_macro": "--warmth-macro",
    "envelopment_macro": "--envelopment-macro",
    "algo_decorrelation_front": "--algo-decorrelation-front",
    "algo_decorrelation_rear": "--algo-decorrelation-rear",
    "algo_decorrelation_top": "--algo-decorrelation-top",
    "freeze": "--freeze",
    "shimmer": "--shimmer",
    "shimmer_semitones": "--shimmer-semitones",
    "shimmer_mix": "--shimmer-mix",
    "shimmer_feedback": "--shimmer-feedback",
    "shimmer_highcut": "--shimmer-highcut",
    "shimmer_lowcut": "--shimmer-lowcut",
    "shimmer_spatial": "--shimmer-spatial",
    "shimmer_spread_cents": "--shimmer-spread-cents",
    "shimmer_decorrelation_ms": "--shimmer-decorrelation-ms",
    "unsafe_self_oscillate": "--unsafe-self-oscillate",
    "unsafe_loop_gain": "--unsafe-loop-gain",
    "algo_proxy_ir_max_seconds": "--algo-proxy-ir-max-seconds",
    "lowcut": "--lowcut",
    "highcut": "--highcut",
    "tilt": "--tilt",
}


def realtime(
    engine: Literal["auto", "conv", "algo"] = typer.Option(
        "auto",
        "--engine",
        help=(
            "Realtime engine: convolution IR, or algorithmic proxy rendered "
            "into a live convolver."
        ),
    ),
    ir: Path | None = typer.Option(
        None,
        "--ir",
        exists=True,
        readable=True,
        resolve_path=True,
        help="Impulse response path for realtime convolution mode.",
    ),
    input_device: str | None = typer.Option(
        None,
        "--input-device",
        help="Input device index or case-insensitive name substring.",
    ),
    output_device: str | None = typer.Option(
        None,
        "--output-device",
        help="Output device index or case-insensitive name substring.",
    ),
    list_devices: bool = typer.Option(
        False,
        "--list-devices",
        help="List available realtime audio devices and exit.",
    ),
    sample_rate: int = typer.Option(48_000, "--sample-rate", min=8_000),
    block_size: int = typer.Option(512, "--block-size", min=64),
    partition_size: int = typer.Option(2_048, "--partition-size", min=256),
    input_channels: int | None = typer.Option(
        None,
        "--input-channels",
        min=1,
        help="Requested live input channel count. Defaults to mono or stereo depending on device.",
    ),
    output_channels: int | None = typer.Option(
        None,
        "--output-channels",
        min=1,
        help=(
            "Requested live output channel count. Defaults to the processor's "
            "natural output width."
        ),
    ),
    duration: float | None = typer.Option(
        None,
        "--duration",
        min=0.0,
        help="Optional duration in seconds. Omit to run until Ctrl-C.",
    ),
    wet: float = typer.Option(0.8, "--wet", min=0.0, max=1.0),
    dry: float = typer.Option(0.2, "--dry", min=0.0, max=1.0),
    rt60: float = typer.Option(6.0, "--rt60", min=0.1),
    pre_delay_ms: float = typer.Option(20.0, "--pre-delay-ms", min=0.0),
    damping: float = typer.Option(0.45, "--damping", min=0.0, max=1.0),
    width: float = typer.Option(1.0, "--width", min=0.0, max=2.0),
    mod_depth_ms: float = typer.Option(2.0, "--mod-depth-ms", min=0.0),
    mod_rate_hz: float = typer.Option(0.1, "--mod-rate-hz", min=0.0),
    fdn_lines: int = typer.Option(8, "--fdn-lines", min=1, max=64),
    fdn_matrix: str = typer.Option(
        "hadamard",
        "--fdn-matrix",
        help="Algorithmic proxy matrix for realtime --engine algo.",
    ),
    fdn_tv_rate_hz: float = typer.Option(0.0, "--fdn-tv-rate-hz", min=0.0),
    fdn_tv_depth: float = typer.Option(0.0, "--fdn-tv-depth", min=0.0),
    fdn_dfm_delays_ms: str | None = typer.Option(
        None,
        "--fdn-dfm-delays-ms",
        help="Comma-separated delay-feedback modulation taps in milliseconds.",
    ),
    fdn_sparse: bool = typer.Option(False, "--fdn-sparse"),
    fdn_sparse_degree: int = typer.Option(2, "--fdn-sparse-degree", min=1, max=16),
    fdn_cascade: bool = typer.Option(False, "--fdn-cascade"),
    fdn_cascade_mix: float = typer.Option(0.35, "--fdn-cascade-mix", min=0.0, max=1.0),
    fdn_cascade_delay_scale: float = typer.Option(
        0.5,
        "--fdn-cascade-delay-scale",
        min=0.2,
        max=1.0,
    ),
    fdn_cascade_rt60_ratio: float = typer.Option(
        0.55,
        "--fdn-cascade-rt60-ratio",
        min=0.1,
        max=1.0,
    ),
    fdn_rt60_low: float | None = typer.Option(None, "--fdn-rt60-low", min=0.1),
    fdn_rt60_mid: float | None = typer.Option(None, "--fdn-rt60-mid", min=0.1),
    fdn_rt60_high: float | None = typer.Option(None, "--fdn-rt60-high", min=0.1),
    fdn_rt60_tilt: float = typer.Option(0.0, "--fdn-rt60-tilt", min=-1.0, max=1.0),
    fdn_tonal_correction_strength: float = typer.Option(
        0.0,
        "--fdn-tonal-correction-strength",
        min=0.0,
        max=1.0,
    ),
    fdn_xover_low_hz: float = typer.Option(250.0, "--fdn-xover-low-hz", min=10.0),
    fdn_xover_high_hz: float = typer.Option(4000.0, "--fdn-xover-high-hz", min=10.0),
    fdn_link_filter: Literal["none", "lowpass", "highpass"] = typer.Option(
        "none",
        "--fdn-link-filter",
    ),
    fdn_link_filter_hz: float = typer.Option(2500.0, "--fdn-link-filter-hz", min=10.0),
    fdn_link_filter_mix: float = typer.Option(1.0, "--fdn-link-filter-mix", min=0.0, max=1.0),
    fdn_graph_topology: Literal["ring", "path", "star", "random"] = typer.Option(
        "ring",
        "--fdn-graph-topology",
    ),
    fdn_graph_degree: int = typer.Option(2, "--fdn-graph-degree", min=1, max=16),
    fdn_graph_seed: int = typer.Option(2026, "--fdn-graph-seed"),
    fdn_matrix_morph_to: str | None = typer.Option(
        None,
        "--fdn-matrix-morph-to",
        help="Optional second FDN matrix family for gradual morphing.",
    ),
    fdn_matrix_morph_seconds: float = typer.Option(
        0.0,
        "--fdn-matrix-morph-seconds",
        min=0.0,
    ),
    fdn_spatial_coupling_mode: Literal[
        "none", "adjacent", "front_rear", "bed_top", "all_to_all"
    ] = typer.Option("none", "--fdn-spatial-coupling-mode"),
    fdn_spatial_coupling_strength: float = typer.Option(
        0.0,
        "--fdn-spatial-coupling-strength",
        min=0.0,
        max=1.0,
    ),
    fdn_nonlinearity: Literal["none", "tanh", "softclip"] = typer.Option(
        "none",
        "--fdn-nonlinearity",
    ),
    fdn_nonlinearity_amount: float = typer.Option(
        0.0,
        "--fdn-nonlinearity-amount",
        min=0.0,
        max=1.0,
    ),
    fdn_nonlinearity_drive: float = typer.Option(
        1.0,
        "--fdn-nonlinearity-drive",
        min=0.1,
        max=8.0,
    ),
    room_size_macro: float = typer.Option(0.0, "--room-size-macro", min=-1.0, max=1.0),
    clarity_macro: float = typer.Option(0.0, "--clarity-macro", min=-1.0, max=1.0),
    warmth_macro: float = typer.Option(0.0, "--warmth-macro", min=-1.0, max=1.0),
    envelopment_macro: float = typer.Option(0.0, "--envelopment-macro", min=-1.0, max=1.0),
    algo_decorrelation_front: float = typer.Option(
        0.0,
        "--algo-decorrelation-front",
        min=0.0,
        max=1.0,
    ),
    algo_decorrelation_rear: float = typer.Option(
        0.0,
        "--algo-decorrelation-rear",
        min=0.0,
        max=1.0,
    ),
    algo_decorrelation_top: float = typer.Option(
        0.0,
        "--algo-decorrelation-top",
        min=0.0,
        max=1.0,
    ),
    allpass_stages: int = typer.Option(6, "--allpass-stages", min=0, max=64),
    allpass_gain: str = typer.Option(
        "0.7",
        "--allpass-gain",
        help="Single allpass gain or comma-separated per-stage gains.",
    ),
    allpass_delays_ms: str | None = typer.Option(
        None,
        "--allpass-delays-ms",
        help="Comma-separated diffusion delays in milliseconds.",
    ),
    comb_delays_ms: str | None = typer.Option(
        None,
        "--comb-delays-ms",
        help="Comma-separated FDN/comb delay taps in milliseconds.",
    ),
    freeze: bool = typer.Option(
        False,
        "--freeze",
        help=(
            "Realtime algo only: approximate a frozen-space sustain by forcing "
            "a long near-infinite proxy tail."
        ),
    ),
    shimmer: bool = typer.Option(False, "--shimmer"),
    shimmer_semitones: float = typer.Option(12.0, "--shimmer-semitones"),
    shimmer_mix: float = typer.Option(0.25, "--shimmer-mix", min=0.0, max=1.0),
    shimmer_feedback: float = typer.Option(0.35, "--shimmer-feedback", min=0.0, max=1.25),
    shimmer_highcut: float | None = typer.Option(None, "--shimmer-highcut", min=10.0),
    shimmer_lowcut: float | None = typer.Option(None, "--shimmer-lowcut", min=10.0),
    shimmer_spatial: bool = typer.Option(False, "--shimmer-spatial"),
    shimmer_spread_cents: float = typer.Option(8.0, "--shimmer-spread-cents", min=0.0),
    shimmer_decorrelation_ms: float = typer.Option(
        1.5,
        "--shimmer-decorrelation-ms",
        min=0.0,
    ),
    unsafe_self_oscillate: bool = typer.Option(False, "--unsafe-self-oscillate"),
    unsafe_loop_gain: float = typer.Option(1.02, "--unsafe-loop-gain", min=0.001),
    algo_proxy_ir_max_seconds: float = typer.Option(
        120.0,
        "--algo-proxy-ir-max-seconds",
        min=0.1,
        help="Maximum rendered proxy IR duration used by realtime --engine algo.",
    ),
    lowcut: float | None = typer.Option(None, "--lowcut", min=10.0),
    highcut: float | None = typer.Option(None, "--highcut", min=10.0),
    tilt: float = typer.Option(0.0, "--tilt", min=-18.0, max=18.0),
    quiet: bool = typer.Option(False, "--quiet", help="Reduce console output."),
) -> None:
    """Run realtime duplex monitoring with selectable input/output devices."""
    try:
        devices = list_audio_devices()
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if list_devices:
        _print_device_table(devices)
        return

    try:
        input_info = resolve_audio_device(selector=input_device, kind="input", devices=devices)
        output_info = resolve_audio_device(selector=output_device, kind="output", devices=devices)
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    parsed_allpass_delays = _parse_delay_list_ms(
        allpass_delays_ms,
        option_name="--allpass-delays-ms",
    )
    parsed_allpass_gain_values = _parse_gain_list(
        allpass_gain,
        option_name="--allpass-gain",
        min_value=-0.99,
        max_value=0.99,
    )
    parsed_comb_delays = _parse_delay_list_ms(
        comb_delays_ms,
        option_name="--comb-delays-ms",
    )
    parsed_dfm_delays = _parse_delay_list_ms(
        fdn_dfm_delays_ms,
        option_name="--fdn-dfm-delays-ms",
    )
    resolved_input_channels = _resolve_input_channels(input_channels, input_info)
    realtime_config = RenderConfig(
        engine=engine,
        rt60=float(rt60),
        pre_delay_ms=float(pre_delay_ms),
        damping=float(damping),
        width=float(width),
        mod_depth_ms=float(mod_depth_ms),
        mod_rate_hz=float(mod_rate_hz),
        wet=float(wet),
        dry=float(dry),
        fdn_lines=int(fdn_lines),
        fdn_matrix=normalize_fdn_matrix_name(str(fdn_matrix)),
        fdn_tv_rate_hz=float(fdn_tv_rate_hz),
        fdn_tv_depth=float(fdn_tv_depth),
        fdn_dfm_delays_ms=parsed_dfm_delays,
        fdn_sparse=bool(fdn_sparse),
        fdn_sparse_degree=int(fdn_sparse_degree),
        fdn_cascade=bool(fdn_cascade),
        fdn_cascade_mix=float(fdn_cascade_mix),
        fdn_cascade_delay_scale=float(fdn_cascade_delay_scale),
        fdn_cascade_rt60_ratio=float(fdn_cascade_rt60_ratio),
        fdn_rt60_low=fdn_rt60_low,
        fdn_rt60_mid=fdn_rt60_mid,
        fdn_rt60_high=fdn_rt60_high,
        fdn_rt60_tilt=float(fdn_rt60_tilt),
        fdn_tonal_correction_strength=float(fdn_tonal_correction_strength),
        fdn_xover_low_hz=float(fdn_xover_low_hz),
        fdn_xover_high_hz=float(fdn_xover_high_hz),
        fdn_link_filter=normalize_fdn_link_filter_name(str(fdn_link_filter)),
        fdn_link_filter_hz=float(fdn_link_filter_hz),
        fdn_link_filter_mix=float(fdn_link_filter_mix),
        fdn_graph_topology=normalize_fdn_graph_topology_name(str(fdn_graph_topology)),
        fdn_graph_degree=int(fdn_graph_degree),
        fdn_graph_seed=int(fdn_graph_seed),
        fdn_matrix_morph_to=(
            None if fdn_matrix_morph_to is None else normalize_fdn_matrix_name(fdn_matrix_morph_to)
        ),
        fdn_matrix_morph_seconds=float(fdn_matrix_morph_seconds),
        fdn_spatial_coupling_mode=cast(
            Any,
            str(fdn_spatial_coupling_mode).strip().lower().replace("-", "_"),
        ),
        fdn_spatial_coupling_strength=float(fdn_spatial_coupling_strength),
        fdn_nonlinearity=cast(Any, str(fdn_nonlinearity).strip().lower().replace("-", "_")),
        fdn_nonlinearity_amount=float(fdn_nonlinearity_amount),
        fdn_nonlinearity_drive=float(fdn_nonlinearity_drive),
        room_size_macro=float(room_size_macro),
        clarity_macro=float(clarity_macro),
        warmth_macro=float(warmth_macro),
        envelopment_macro=float(envelopment_macro),
        algo_decorrelation_front=float(algo_decorrelation_front),
        algo_decorrelation_rear=float(algo_decorrelation_rear),
        algo_decorrelation_top=float(algo_decorrelation_top),
        allpass_stages=int(allpass_stages),
        allpass_gain=float(parsed_allpass_gain_values[0]),
        allpass_gains=parsed_allpass_gain_values if len(parsed_allpass_gain_values) > 1 else (),
        allpass_delays_ms=parsed_allpass_delays,
        comb_delays_ms=parsed_comb_delays,
        freeze=bool(freeze),
        block_size=int(block_size),
        partition_size=int(partition_size),
        ir=None if ir is None else str(ir),
        target_sr=int(sample_rate),
        output_subtype="float32",
        output_container="wav",
        shimmer=bool(shimmer),
        shimmer_semitones=float(shimmer_semitones),
        shimmer_mix=float(shimmer_mix),
        shimmer_feedback=float(shimmer_feedback),
        shimmer_highcut=shimmer_highcut,
        shimmer_lowcut=shimmer_lowcut,
        shimmer_spatial=bool(shimmer_spatial),
        shimmer_spread_cents=float(shimmer_spread_cents),
        shimmer_decorrelation_ms=float(shimmer_decorrelation_ms),
        unsafe_self_oscillate=bool(unsafe_self_oscillate),
        unsafe_loop_gain=float(unsafe_loop_gain),
        algo_proxy_ir_max_seconds=float(algo_proxy_ir_max_seconds),
        lowcut=lowcut,
        highcut=highcut,
        tilt=float(tilt),
        progress=False,
        silent=True,
    )
    _validate_realtime_config(realtime_config)

    processor, engine_summary = _build_live_processor(
        config=realtime_config,
        sample_rate=int(sample_rate),
        input_channels=resolved_input_channels,
    )
    resolved_output_channels = _resolve_output_channels(
        requested=output_channels,
        output_device=output_info,
        processor_channels=int(processor.output_channels),
    )

    if not quiet:
        _print_realtime_start_table(
            input_device=input_info,
            output_device=output_info,
            sample_rate=int(sample_rate),
            block_size=int(block_size),
            input_channels=resolved_input_channels,
            output_channels=resolved_output_channels,
            engine_summary=engine_summary,
            duration=duration,
        )

    try:
        summary = run_realtime_duplex(
            processor=processor,
            sample_rate=int(sample_rate),
            block_size=int(block_size),
            input_device=input_info,
            output_device=output_info,
            input_channels=resolved_input_channels,
            output_channels=resolved_output_channels,
            duration_seconds=duration,
        )
    finally:
        proxy_path = engine_summary.get("proxy_ir_path")
        if isinstance(proxy_path, Path):
            proxy_path.unlink(missing_ok=True)

    if not quiet:
        _print_realtime_end_table(summary)


def _build_live_processor(
    *,
    config: RenderConfig,
    sample_rate: int,
    input_channels: int,
) -> tuple[LiveConvolutionProcessor, dict[str, object]]:
    """Build the live convolution processor and reproducibility metadata."""
    resolved_engine = "conv" if config.ir is not None else "algo"
    if str(config.engine).strip().lower() == "conv" and config.ir is None:
        msg = "Realtime convolution requires --ir PATH. Use --engine algo for proxy mode."
        raise typer.BadParameter(msg)
    if str(config.engine).strip().lower() == "algo":
        resolved_engine = "algo"

    if resolved_engine == "conv":
        algo_only_overrides = _collect_algo_only_option_overrides(config)
        if len(algo_only_overrides) > 0:
            shown = ", ".join(algo_only_overrides[:8])
            if len(algo_only_overrides) > 8:
                shown = f"{shown}, ..."
            msg = (
                "These switches only apply to realtime --engine algo or auto-without --ir: "
                f"{shown}"
            )
            raise typer.BadParameter(msg)

    proxy_ir_path: Path | None = None
    ir_path: Path | None = None if config.ir is None else Path(config.ir)
    live_config = config
    if resolved_engine == "algo":
        live_config = _apply_realtime_freeze_proxy(config)
        proxy_ir_path, _ = render_algo_proxy_ir(
            config=live_config,
            sr=int(sample_rate),
            input_channels=int(input_channels),
        )
        ir_path = proxy_ir_path

    if ir_path is None:
        msg = "Realtime mode needs an IR or an algorithmic proxy configuration."
        raise typer.BadParameter(msg)

    engine = ConvolutionReverbEngine(
        ConvolutionReverbConfig(
            wet=float(config.wet),
            dry=float(config.dry),
            ir_path=str(ir_path),
            partition_size=int(config.partition_size),
            input_layout="auto",
            output_layout="auto",
            device="cpu",
        )
    )
    processor = engine.create_live_processor(
        input_channels=int(input_channels),
        sr=int(sample_rate),
    )
    summary: dict[str, object] = {
        "engine_requested": config.engine,
        "engine_resolved": "algo_proxy_live" if resolved_engine == "algo" else "conv_live",
        "compute_backend": engine.backend_name(),
        "proxy_ir_path": proxy_ir_path,
        "sample_rate": int(sample_rate),
    }
    if resolved_engine == "algo":
        summary.update(
            {
                "rt60": float(live_config.rt60),
                "fdn_matrix": str(live_config.fdn_matrix),
                "fdn_lines": int(live_config.fdn_lines),
                "freeze": bool(config.freeze),
                "shimmer": bool(live_config.shimmer),
                "tv": bool(
                    float(live_config.fdn_tv_rate_hz) > 0.0
                    and float(live_config.fdn_tv_depth) > 0.0
                ),
                "proxy_eq": _format_proxy_eq_summary(live_config),
            }
        )
    return processor, summary


def _resolve_input_channels(requested: int | None, device: RealtimeDeviceInfo) -> int:
    """Choose an input width that the selected device can actually supply."""
    available = int(device.max_input_channels)
    if available <= 0:
        raise typer.BadParameter(f"Input device '{device.name}' has no input channels.")
    if requested is None:
        return 2 if available >= 2 else 1
    if requested > available:
        raise typer.BadParameter(
            f"Input device '{device.name}' exposes {available} channels, not {requested}."
        )
    return int(requested)


def _resolve_output_channels(
    *,
    requested: int | None,
    output_device: RealtimeDeviceInfo,
    processor_channels: int,
) -> int:
    """Choose an output width that fits both the processor and hardware."""
    available = int(output_device.max_output_channels)
    if available <= 0:
        raise typer.BadParameter(f"Output device '{output_device.name}' has no output channels.")
    if requested is not None:
        if requested > available:
            raise typer.BadParameter(
                f"Output device '{output_device.name}' exposes {available} "
                f"channels, not {requested}."
            )
        return int(requested)
    return max(1, min(int(processor_channels), available))


def _print_device_table(devices: list[RealtimeDeviceInfo]) -> None:
    table = Table(title="Realtime Audio Devices")
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("Name", style="white")
    table.add_column("Host API", style="magenta")
    table.add_column("In", style="green", justify="right")
    table.add_column("Out", style="green", justify="right")
    table.add_column("Default SR", style="yellow", justify="right")
    for device in devices:
        table.add_row(
            str(device.index),
            device.name,
            device.hostapi or "",
            str(device.max_input_channels),
            str(device.max_output_channels),
            f"{device.default_samplerate:.1f}",
        )
    console.print(table)


def _print_realtime_start_table(
    *,
    input_device: RealtimeDeviceInfo,
    output_device: RealtimeDeviceInfo,
    sample_rate: int,
    block_size: int,
    input_channels: int,
    output_channels: int,
    engine_summary: dict[str, object],
    duration: float | None,
) -> None:
    table = Table(title="Realtime Session")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("engine", str(engine_summary.get("engine_resolved", "")))
    table.add_row("backend", str(engine_summary.get("compute_backend", "")))
    table.add_row("sample_rate", str(sample_rate))
    table.add_row("block_size", str(block_size))
    table.add_row("input_device", input_device.name)
    table.add_row("output_device", output_device.name)
    table.add_row("input_channels", str(input_channels))
    table.add_row("output_channels", str(output_channels))
    table.add_row("duration", "until Ctrl-C" if duration is None else f"{duration:.2f}s")
    if str(engine_summary.get("engine_resolved", "")) == "algo_proxy_live":
        rt60_value = engine_summary.get("rt60", 0.0)
        proxy_rt60 = float(rt60_value) if isinstance(rt60_value, (int, float)) else 0.0
        table.add_row("proxy_rt60", f"{proxy_rt60:.2f}s")
        table.add_row(
            "proxy_fdn",
            f"{engine_summary.get('fdn_lines', '')} x {engine_summary.get('fdn_matrix', '')}",
        )
        table.add_row("proxy_freeze", "on" if bool(engine_summary.get("freeze", False)) else "off")
        table.add_row(
            "proxy_shimmer",
            "on" if bool(engine_summary.get("shimmer", False)) else "off",
        )
        table.add_row("proxy_tv", "on" if bool(engine_summary.get("tv", False)) else "off")
        proxy_eq = str(engine_summary.get("proxy_eq", "flat"))
        if proxy_eq != "flat":
            table.add_row("proxy_eq", proxy_eq)
    console.print(table)


def _print_realtime_end_table(summary: RealtimeSessionSummary) -> None:
    table = Table(title="Realtime Session Complete")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("input_device", str(summary.input_device))
    table.add_row("output_device", str(summary.output_device))
    table.add_row("sample_rate", str(summary.sample_rate))
    table.add_row("processed_blocks", str(summary.processed_blocks))
    table.add_row("processed_seconds", f"{summary.processed_seconds:.3f}")
    table.add_row("clipped_blocks", str(summary.clipped_blocks))
    console.print(table)


def _parse_delay_list_ms(raw: str | None, *, option_name: str) -> tuple[float, ...]:
    """Parse a comma-separated millisecond delay list for realtime CLI options."""
    if raw is None:
        return ()
    cleaned = raw.strip()
    if cleaned == "":
        return ()
    values: list[float] = []
    for token in cleaned.split(","):
        part = token.strip()
        if part == "":
            continue
        try:
            delay = float(part)
        except ValueError as exc:
            msg = f"{option_name} expects a comma-separated float list in milliseconds."
            raise typer.BadParameter(msg) from exc
        if delay <= 0.0:
            msg = f"{option_name} values must be > 0 ms."
            raise typer.BadParameter(msg)
        values.append(delay)
    if len(values) == 0:
        msg = f"{option_name} must include at least one numeric value."
        raise typer.BadParameter(msg)
    return tuple(values)


def _parse_gain_list(
    raw: str,
    *,
    option_name: str,
    min_value: float,
    max_value: float,
) -> tuple[float, ...]:
    """Parse one or more comma-separated gain values for realtime CLI options."""
    cleaned = raw.strip()
    if cleaned == "":
        msg = f"{option_name} requires at least one numeric value."
        raise typer.BadParameter(msg)

    values: list[float] = []
    for token in cleaned.split(","):
        part = token.strip()
        if part == "":
            continue
        try:
            gain = float(part)
        except ValueError as exc:
            msg = f"{option_name} expects float values, optionally comma-separated."
            raise typer.BadParameter(msg) from exc
        if gain < min_value or gain > max_value:
            msg = f"{option_name} values must be in [{min_value}, {max_value}]."
            raise typer.BadParameter(msg)
        values.append(gain)

    if len(values) == 0:
        msg = f"{option_name} requires at least one numeric value."
        raise typer.BadParameter(msg)
    return tuple(values)


def _validate_realtime_config(config: RenderConfig) -> None:
    """Validate realtime options that require cross-field checks."""
    if config.wet == 0.0 and config.dry == 0.0:
        msg = "At least one of --wet or --dry must be non-zero."
        raise typer.BadParameter(msg)
    if config.lowcut is not None and config.highcut is not None and config.lowcut >= config.highcut:
        msg = "--lowcut must be lower than --highcut."
        raise typer.BadParameter(msg)
    if config.fdn_xover_low_hz >= config.fdn_xover_high_hz:
        msg = "--fdn-xover-low-hz must be lower than --fdn-xover-high-hz."
        raise typer.BadParameter(msg)
    if config.allpass_stages == 0 and len(config.allpass_delays_ms) > 0:
        msg = "--allpass-delays-ms cannot be used when --allpass-stages is 0."
        raise typer.BadParameter(msg)
    if config.allpass_stages == 0 and len(config.allpass_gains) > 0:
        msg = "--allpass-gain list cannot be used when --allpass-stages is 0."
        raise typer.BadParameter(msg)
    if len(config.allpass_gains) > 0 and len(config.allpass_gains) != config.allpass_stages:
        msg = (
            "When using comma-separated --allpass-gain values, provide exactly "
            f"{config.allpass_stages} entries (got {len(config.allpass_gains)})."
        )
        raise typer.BadParameter(msg)
    if len(config.comb_delays_ms) > 64:
        raise typer.BadParameter("--comb-delays-ms supports at most 64 entries.")
    if len(config.allpass_delays_ms) > 128:
        raise typer.BadParameter("--allpass-delays-ms supports at most 128 entries.")
    if len(config.fdn_dfm_delays_ms) > 64:
        raise typer.BadParameter("--fdn-dfm-delays-ms supports at most 64 entries.")
    if config.fdn_matrix not in FDN_MATRIX_CHOICES:
        options = ", ".join(sorted(FDN_MATRIX_CHOICES))
        raise typer.BadParameter(f"--fdn-matrix must be one of: {options}.")
    if (
        config.fdn_matrix_morph_to is not None
        and config.fdn_matrix_morph_to not in FDN_MATRIX_CHOICES
    ):
        options = ", ".join(sorted(FDN_MATRIX_CHOICES))
        raise typer.BadParameter(f"--fdn-matrix-morph-to must be one of: {options}.")
    if config.fdn_link_filter not in FDN_LINK_FILTER_CHOICES:
        options = ", ".join(sorted(FDN_LINK_FILTER_CHOICES))
        raise typer.BadParameter(f"--fdn-link-filter must be one of: {options}.")
    if config.fdn_graph_topology not in FDN_GRAPH_TOPOLOGY_CHOICES:
        options = ", ".join(sorted(FDN_GRAPH_TOPOLOGY_CHOICES))
        raise typer.BadParameter(f"--fdn-graph-topology must be one of: {options}.")
    if config.fdn_spatial_coupling_mode not in _FDN_SPATIAL_COUPLING_CHOICES:
        options = ", ".join(sorted(_FDN_SPATIAL_COUPLING_CHOICES))
        raise typer.BadParameter(f"--fdn-spatial-coupling-mode must be one of: {options}.")
    if config.fdn_nonlinearity not in _FDN_NONLINEARITY_CHOICES:
        options = ", ".join(sorted(_FDN_NONLINEARITY_CHOICES))
        raise typer.BadParameter(f"--fdn-nonlinearity must be one of: {options}.")
    resolved_fdn_lines = (
        len(config.comb_delays_ms) if len(config.comb_delays_ms) > 0 else int(config.fdn_lines)
    )
    if len(config.fdn_dfm_delays_ms) not in {0, 1, resolved_fdn_lines}:
        msg = (
            "--fdn-dfm-delays-ms must include either 1 value or exactly "
            f"{resolved_fdn_lines} values."
        )
        raise typer.BadParameter(msg)


def _collect_algo_only_option_overrides(config: RenderConfig) -> list[str]:
    """Return realtime switches that only make sense for algo-proxy mode."""
    defaults = RenderConfig()
    overrides: list[str] = []
    for field_name, switch_name in _ALGO_ONLY_SWITCHES.items():
        if getattr(config, field_name) != getattr(defaults, field_name):
            overrides.append(switch_name)
    return sorted(set(overrides))


def _apply_realtime_freeze_proxy(config: RenderConfig) -> RenderConfig:
    """Approximate a frozen live space by forcing a long self-sustaining proxy IR."""
    if not config.freeze:
        return config
    frozen = RenderConfig(**asdict(config))
    if not frozen.unsafe_self_oscillate:
        frozen.unsafe_self_oscillate = True
        frozen.unsafe_loop_gain = _REALTIME_FREEZE_LOOP_GAIN
    frozen.algo_proxy_ir_max_seconds = max(
        float(frozen.algo_proxy_ir_max_seconds),
        max(_REALTIME_FREEZE_PROXY_SECONDS, float(frozen.rt60) * 4.0),
    )
    return frozen


def _format_proxy_eq_summary(config: RenderConfig) -> str:
    """Summarize startup proxy EQ shaping in a compact human-readable form."""
    parts: list[str] = []
    if config.lowcut is not None:
        parts.append(f"lowcut={config.lowcut:.0f}Hz")
    if config.highcut is not None:
        parts.append(f"highcut={config.highcut:.0f}Hz")
    if abs(float(config.tilt)) > 1e-4:
        parts.append(f"tilt={config.tilt:+.1f}dB/oct")
    return ", ".join(parts) if len(parts) > 0 else "flat"
