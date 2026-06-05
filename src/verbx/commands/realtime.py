"""Realtime command for live duplex monitoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import typer
from rich.console import Console
from rich.table import Table

from verbx.commands.safety import (
    realtime_freeze_proxy_required_seconds,
    realtime_preflight_items,
)
from verbx.config import RenderConfig
from verbx.core.algo_proxy import render_algo_proxy_ir
from verbx.core.convolution_reverb import (
    ConvolutionReverbConfig,
    ConvolutionReverbEngine,
    LiveConvolutionProcessor,
)
from verbx.core.dereverb import (
    LiveDereverbConfig,
    create_live_dereverb_processor,
    parse_dereverb_window_weights,
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
    RealtimeBlockProcessor,
    RealtimeDeviceInfo,
    RealtimeSessionSummary,
    list_audio_devices,
    parse_channel_map,
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


@dataclass(slots=True)
class _ProcessorChain:
    """Sequentially apply two realtime processors."""

    first: RealtimeBlockProcessor
    second: RealtimeBlockProcessor
    input_channels: int
    output_channels: int

    def process_block(self, block: Any) -> Any:
        return self.second.process_block(self.first.process_block(block))


def realtime(
    live_mode: Literal["reverb", "dereverb", "dereverb-reverb"] = typer.Option(
        "reverb",
        "--live-mode",
        help=(
            "Realtime processing mode: reverb only, dereverb only, or "
            "dereverb feeding the live reverb path."
        ),
    ),
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
    input_channel_map: str | None = typer.Option(
        None,
        "--input-channel-map",
        help=(
            "Comma-separated 1-based hardware input channels to feed the processor, "
            "for example 1,3 or 1,3,5,7."
        ),
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
    output_channel_map: str | None = typer.Option(
        None,
        "--output-channel-map",
        help=(
            "Comma-separated 1-based hardware output channels that receive processor "
            "outputs, in order."
        ),
    ),
    duration: float | None = typer.Option(
        None,
        "--duration",
        min=0.0,
        help="Optional duration in seconds. Omit to run until Ctrl-C.",
    ),
    dereverb_mode: Literal["wiener", "spectral_sub"] = typer.Option(
        "wiener",
        "--dereverb-mode",
        help="Low-latency dereverb kernel used by --live-mode dereverb*.",
    ),
    dereverb_strength: float = typer.Option(
        0.7,
        "--dereverb-strength",
        min=0.0,
        max=2.0,
    ),
    dereverb_floor: float = typer.Option(
        0.08,
        "--dereverb-floor",
        min=1e-6,
        max=1.0,
    ),
    dereverb_window_ms: float = typer.Option(
        16.0,
        "--dereverb-window-ms",
        min=2.0,
    ),
    dereverb_hop_ms: float = typer.Option(
        8.0,
        "--dereverb-hop-ms",
        min=1.0,
    ),
    dereverb_tail_ms: float = typer.Option(
        120.0,
        "--dereverb-tail-ms",
        min=10.0,
    ),
    dereverb_pre_emphasis: float = typer.Option(
        0.0,
        "--dereverb-pre-emphasis",
        min=0.0,
        max=0.98,
    ),
    dereverb_mix: float = typer.Option(
        1.0,
        "--dereverb-mix",
        min=0.0,
        max=1.0,
    ),
    dereverb_max_atten_db: float = typer.Option(
        18.0,
        "--dereverb-max-atten-db",
        min=0.0,
        max=48.0,
    ),
    dereverb_stereo_link: bool = typer.Option(
        True,
        "--dereverb-stereo-link/--no-dereverb-stereo-link",
        help="Link stereo gain decisions to reduce image wobble.",
    ),
    dereverb_input_gain_db: float = typer.Option(
        0.0,
        "--dereverb-input-gain-db",
        min=-24.0,
        max=24.0,
    ),
    dereverb_output_gain_db: float = typer.Option(
        0.0,
        "--dereverb-output-gain-db",
        min=-24.0,
        max=24.0,
    ),
    dereverb_window_type: str = typer.Option(
        "hann",
        "--dereverb-window-type",
        help=(
            "Live dereverb analysis window family "
            "(hann, hamming, blackman, kaiser, dpss, tukey, chebwin, and many more)."
        ),
    ),
    dereverb_synthesis_window_type: str | None = typer.Option(
        None,
        "--dereverb-synthesis-window-type",
        help="Optional live dereverb synthesis window family. Defaults to the analysis window.",
    ),
    dereverb_window_symmetric: bool = typer.Option(
        False,
        "--dereverb-window-symmetric/--dereverb-window-periodic",
        help="Use symmetric instead of periodic live dereverb windows.",
    ),
    dereverb_window_alpha: float = typer.Option(
        0.5,
        "--dereverb-window-alpha",
        min=0.0,
    ),
    dereverb_window_beta: float = typer.Option(
        14.0,
        "--dereverb-window-beta",
        min=0.0,
    ),
    dereverb_window_std: float = typer.Option(
        2.5,
        "--dereverb-window-std",
        min=1e-6,
    ),
    dereverb_window_power: float = typer.Option(
        1.5,
        "--dereverb-window-power",
        min=1e-6,
    ),
    dereverb_window_atten_db: float = typer.Option(
        100.0,
        "--dereverb-window-atten-db",
        min=1e-3,
    ),
    dereverb_window_nbar: int = typer.Option(
        4,
        "--dereverb-window-nbar",
        min=2,
    ),
    dereverb_window_nw: float = typer.Option(
        2.5,
        "--dereverb-window-nw",
        min=1e-3,
    ),
    dereverb_window_tau: float = typer.Option(
        3.0,
        "--dereverb-window-tau",
        min=1e-6,
    ),
    dereverb_window_weights: str | None = typer.Option(
        None,
        "--dereverb-window-weights",
        help="Optional comma-separated weights for general_cosine live dereverb windows.",
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
    try:
        parsed_input_channel_map = parse_channel_map(
            input_channel_map,
            option_name="--input-channel-map",
        )
        parsed_output_channel_map = parse_channel_map(
            output_channel_map,
            option_name="--output-channel-map",
        )
    except ValueError as exc:
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
    resolved_input_channels = _resolve_input_channels(
        requested=input_channels,
        requested_map=parsed_input_channel_map,
        device=input_info,
    )
    try:
        parsed_dereverb_window_weights = parse_dereverb_window_weights(
            dereverb_window_weights
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    live_dereverb_config = LiveDereverbConfig(
        mode=dereverb_mode,
        strength=float(dereverb_strength),
        floor=float(dereverb_floor),
        window_ms=float(dereverb_window_ms),
        hop_ms=float(dereverb_hop_ms),
        tail_ms=float(dereverb_tail_ms),
        pre_emphasis=float(dereverb_pre_emphasis),
        mix=float(dereverb_mix),
        max_atten_db=float(dereverb_max_atten_db),
        stereo_link=bool(dereverb_stereo_link),
        input_gain_db=float(dereverb_input_gain_db),
        output_gain_db=float(dereverb_output_gain_db),
        analysis_window=str(dereverb_window_type),
        synthesis_window=dereverb_synthesis_window_type,
        window_symmetric=bool(dereverb_window_symmetric),
        window_alpha=float(dereverb_window_alpha),
        window_beta=float(dereverb_window_beta),
        window_std=float(dereverb_window_std),
        window_power=float(dereverb_window_power),
        window_atten_db=float(dereverb_window_atten_db),
        window_nbar=int(dereverb_window_nbar),
        window_nw=float(dereverb_window_nw),
        window_tau=float(dereverb_window_tau),
        window_weights=parsed_dereverb_window_weights,
    )
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
    if str(live_mode) != "dereverb":
        _validate_realtime_config(realtime_config)

    if str(live_mode) == "dereverb":
        processor, engine_summary = _build_live_dereverb_processor(
            config=live_dereverb_config,
            sample_rate=int(sample_rate),
            block_size=int(block_size),
            input_channels=resolved_input_channels,
        )
    elif str(live_mode) == "dereverb-reverb":
        processor, engine_summary = _build_live_dereverb_reverb_processor(
            dereverb_config=live_dereverb_config,
            reverb_config=realtime_config,
            sample_rate=int(sample_rate),
            block_size=int(block_size),
            input_channels=resolved_input_channels,
        )
    else:
        processor, engine_summary = _build_live_processor(
            config=realtime_config,
            sample_rate=int(sample_rate),
            input_channels=resolved_input_channels,
        )
    resolved_output_channels = _resolve_output_channels(
        requested=output_channels,
        requested_map=parsed_output_channel_map,
        output_device=output_info,
        processor_channels=int(processor.output_channels),
    )

    if not quiet:
        _print_realtime_preflight_table(
            realtime_preflight_items(
                config=realtime_config,
                live_mode=str(live_mode),
                sample_rate=int(sample_rate),
                block_size=int(block_size),
                duration_seconds=duration,
            )
        )
        _print_realtime_start_table(
            input_device=input_info,
            output_device=output_info,
            sample_rate=int(sample_rate),
            block_size=int(block_size),
            input_channels=resolved_input_channels,
            output_channels=resolved_output_channels,
            input_channel_map=parsed_input_channel_map,
            output_channel_map=parsed_output_channel_map,
            engine_summary=engine_summary,
            duration=duration,
        )

    def _handle_stream_started(
        input_latency_seconds: float | None,
        output_latency_seconds: float | None,
    ) -> None:
        if quiet:
            return
        _print_realtime_latency_table(
            sample_rate=int(sample_rate),
            block_size=int(block_size),
            engine_summary=engine_summary,
            reported_input_latency_seconds=input_latency_seconds,
            reported_output_latency_seconds=output_latency_seconds,
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
            input_channel_map=parsed_input_channel_map,
            output_channel_map=parsed_output_channel_map,
            duration_seconds=duration,
            on_stream_started=_handle_stream_started,
        )
    finally:
        proxy_path = engine_summary.get("proxy_ir_path")
        if isinstance(proxy_path, Path):
            proxy_path.unlink(missing_ok=True)

    if not quiet:
        _print_realtime_end_table(summary, engine_summary=engine_summary)


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


def _build_live_dereverb_processor(
    *,
    config: LiveDereverbConfig,
    sample_rate: int,
    block_size: int,
    input_channels: int,
) -> tuple[RealtimeBlockProcessor, dict[str, object]]:
    """Build a low-latency standalone live dereverb processor."""
    processor = create_live_dereverb_processor(
        sample_rate=int(sample_rate),
        input_channels=int(input_channels),
        config=config,
    )
    if int(block_size) % int(processor.hop_samples) != 0:
        msg = (
            "Low-latency live dereverb requires --block-size to be divisible by the "
            f"resolved hop size ({processor.hop_samples} samples)."
        )
        raise typer.BadParameter(msg)
    summary: dict[str, object] = {
        "engine_resolved": "dereverb_live",
        "compute_backend": "stft_cpu",
        "dereverb_mode": str(config.mode),
        "dereverb_window_type": str(config.analysis_window),
        "dereverb_synthesis_window_type": (
            str(config.analysis_window)
            if config.synthesis_window is None
            else str(config.synthesis_window)
        ),
        "dereverb_window_ms": float(config.window_ms),
        "dereverb_hop_ms": float(config.hop_ms),
        "dereverb_tail_ms": float(config.tail_ms),
        "dereverb_mix": float(config.mix),
        "dereverb_stereo_link": bool(config.stereo_link),
        "dereverb_latency_ms": (1000.0 * float(processor.latency_samples)) / float(sample_rate),
    }
    return processor, summary


def _build_live_dereverb_reverb_processor(
    *,
    dereverb_config: LiveDereverbConfig,
    reverb_config: RenderConfig,
    sample_rate: int,
    block_size: int,
    input_channels: int,
) -> tuple[RealtimeBlockProcessor, dict[str, object]]:
    """Build a live dereverb front-end feeding the existing realtime reverb path."""
    dereverb_processor, dereverb_summary = _build_live_dereverb_processor(
        config=dereverb_config,
        sample_rate=int(sample_rate),
        block_size=int(block_size),
        input_channels=int(input_channels),
    )
    reverb_processor, reverb_summary = _build_live_processor(
        config=reverb_config,
        sample_rate=int(sample_rate),
        input_channels=int(input_channels),
    )
    summary = dict(reverb_summary)
    summary.update(dereverb_summary)
    summary["engine_resolved"] = "dereverb_reverb_live"
    summary["compute_backend"] = (
        f"dereverb_stft_cpu -> {reverb_summary.get('compute_backend', 'cpu')}"
    )
    processor = _ProcessorChain(
        first=dereverb_processor,
        second=reverb_processor,
        input_channels=int(input_channels),
        output_channels=int(reverb_processor.output_channels),
    )
    return processor, summary


def _resolve_input_channels(
    *,
    requested: int | None,
    requested_map: tuple[int, ...],
    device: RealtimeDeviceInfo,
) -> int:
    """Choose an input width that the selected device can actually supply."""
    available = int(device.max_input_channels)
    if available <= 0:
        raise typer.BadParameter(f"Input device '{device.name}' has no input channels.")
    if len(requested_map) > 0:
        highest = max(int(ch) for ch in requested_map)
        if highest > available:
            raise typer.BadParameter(
                f"Input channel map references channel {highest}, but '{device.name}' "
                f"only exposes {available} input channels."
            )
        if requested is not None and requested != len(requested_map):
            raise typer.BadParameter(
                "--input-channels must match the number of entries in --input-channel-map."
            )
        return len(requested_map)
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
    requested_map: tuple[int, ...],
    output_device: RealtimeDeviceInfo,
    processor_channels: int,
) -> int:
    """Choose an output width that fits both the processor and hardware."""
    available = int(output_device.max_output_channels)
    if available <= 0:
        raise typer.BadParameter(f"Output device '{output_device.name}' has no output channels.")
    if len(requested_map) > 0:
        highest = max(int(ch) for ch in requested_map)
        if highest > available:
            raise typer.BadParameter(
                f"Output channel map references channel {highest}, but '{output_device.name}' "
                f"only exposes {available} output channels."
            )
        mapped_channels = len(requested_map)
        if requested is not None and requested != mapped_channels:
            raise typer.BadParameter(
                "--output-channels must match the number of entries in --output-channel-map."
            )
        if mapped_channels > int(processor_channels):
            raise typer.BadParameter(
                f"Realtime processor exposes {processor_channels} output channels, "
                f"not {mapped_channels}."
            )
        return mapped_channels
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


def _print_realtime_preflight_table(items: tuple[object, ...]) -> None:
    """Print quick startup risk/latency summary before opening the device."""

    table = Table(title="Realtime Preflight")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    for item in items:
        key = getattr(item, "key", "")
        value = getattr(item, "value", "")
        table.add_row(str(key), str(value))
    console.print(table)


def _print_realtime_start_table(
    *,
    input_device: RealtimeDeviceInfo,
    output_device: RealtimeDeviceInfo,
    sample_rate: int,
    block_size: int,
    input_channels: int,
    output_channels: int,
    input_channel_map: tuple[int, ...],
    output_channel_map: tuple[int, ...],
    engine_summary: dict[str, object],
    duration: float | None,
) -> None:
    table = Table(title="Realtime Session")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    resolved_engine = str(engine_summary.get("engine_resolved", ""))
    table.add_row("engine", resolved_engine)
    table.add_row("backend", str(engine_summary.get("compute_backend", "")))
    table.add_row("sample_rate", str(sample_rate))
    table.add_row("block_size", str(block_size))
    table.add_row("input_device", input_device.name)
    table.add_row("output_device", output_device.name)
    table.add_row("input_channels", str(input_channels))
    table.add_row("output_channels", str(output_channels))
    table.add_row("input_map", _format_channel_map(input_channels, input_channel_map))
    table.add_row("output_map", _format_channel_map(output_channels, output_channel_map))
    table.add_row("duration", "until Ctrl-C" if duration is None else f"{duration:.2f}s")
    table.add_row("block_latency", f"{_block_latency_ms(sample_rate, block_size):.1f}ms")
    if resolved_engine in {"dereverb_live", "dereverb_reverb_live"}:
        latency = _latency_metrics(
            sample_rate=sample_rate,
            block_size=block_size,
            engine_summary=engine_summary,
        )
        table.add_row("dereverb_mode", str(engine_summary.get("dereverb_mode", "")))
        table.add_row(
            "dereverb_windows",
            (
                f"{engine_summary.get('dereverb_window_type', 'hann')!s} -> "
                f"{engine_summary.get('dereverb_synthesis_window_type', 'hann')!s}"
            ),
        )
        table.add_row(
            "dereverb_window",
            (
                f"{_summary_float(engine_summary, 'dereverb_window_ms'):.1f}ms / "
                f"hop {_summary_float(engine_summary, 'dereverb_hop_ms'):.1f}ms"
            ),
        )
        table.add_row(
            "dereverb_tail",
            f"{_summary_float(engine_summary, 'dereverb_tail_ms'):.1f}ms",
        )
        table.add_row(
            "dereverb_mix",
            f"{_summary_float(engine_summary, 'dereverb_mix'):.2f}",
        )
        table.add_row(
            "dereverb_link",
            "on" if bool(engine_summary.get("dereverb_stereo_link", False)) else "off",
        )
        table.add_row(
            "dereverb_latency",
            f"{latency['algorithm_latency_ms']:.1f}ms",
        )
        table.add_row(
            "software_latency_est",
            f"{latency['software_latency_ms']:.1f}ms",
        )
        table.add_row(
            "end_to_end_est",
            f"{latency['software_latency_ms']:.1f}ms + backend pending",
        )
    if resolved_engine in {"algo_proxy_live", "dereverb_reverb_live"}:
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


def _print_realtime_latency_table(
    *,
    sample_rate: int,
    block_size: int,
    engine_summary: dict[str, object],
    reported_input_latency_seconds: float | None,
    reported_output_latency_seconds: float | None,
) -> None:
    """Print stream-reported latency once the backend opens the duplex device."""
    metrics = _latency_metrics(
        sample_rate=sample_rate,
        block_size=block_size,
        engine_summary=engine_summary,
        reported_input_latency_seconds=reported_input_latency_seconds,
        reported_output_latency_seconds=reported_output_latency_seconds,
    )
    table = Table(title="Realtime Stream Latency")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row(
        "reported_input_latency",
        _format_latency_ms(metrics["reported_input_latency_ms"]),
    )
    table.add_row(
        "reported_output_latency",
        _format_latency_ms(metrics["reported_output_latency_ms"]),
    )
    table.add_row(
        "reported_backend_latency",
        _format_latency_ms(metrics["reported_backend_latency_ms"]),
    )
    if str(engine_summary.get("engine_resolved", "")) in {"dereverb_live", "dereverb_reverb_live"}:
        table.add_row("software_latency_est", f"{metrics['software_latency_ms']:.1f}ms")
        table.add_row(
            "end_to_end_est",
            _format_latency_ms(metrics["end_to_end_latency_ms"]),
        )
    console.print(table)


def _print_realtime_end_table(
    summary: RealtimeSessionSummary,
    *,
    engine_summary: dict[str, object],
) -> None:
    table = Table(title="Realtime Session Complete")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("input_device", str(summary.input_device))
    table.add_row("output_device", str(summary.output_device))
    table.add_row("sample_rate", str(summary.sample_rate))
    table.add_row(
        "input_map",
        _format_channel_map(summary.input_channels, summary.input_channel_map),
    )
    table.add_row(
        "output_map",
        _format_channel_map(summary.output_channels, summary.output_channel_map),
    )
    table.add_row("processed_blocks", str(summary.processed_blocks))
    table.add_row("processed_seconds", f"{summary.processed_seconds:.3f}")
    table.add_row("clipped_blocks", str(summary.clipped_blocks))
    latency = _latency_metrics(
        sample_rate=int(summary.sample_rate),
        block_size=int(summary.block_size),
        engine_summary=engine_summary,
        reported_input_latency_seconds=summary.reported_input_latency_seconds,
        reported_output_latency_seconds=summary.reported_output_latency_seconds,
    )
    table.add_row(
        "reported_input_latency",
        _format_latency_ms(latency["reported_input_latency_ms"]),
    )
    table.add_row(
        "reported_output_latency",
        _format_latency_ms(latency["reported_output_latency_ms"]),
    )
    table.add_row(
        "reported_backend_latency",
        _format_latency_ms(latency["reported_backend_latency_ms"]),
    )
    if str(engine_summary.get("engine_resolved", "")) in {"dereverb_live", "dereverb_reverb_live"}:
        table.add_row("software_latency_est", f"{latency['software_latency_ms']:.1f}ms")
        table.add_row(
            "end_to_end_est",
            _format_latency_ms(latency["end_to_end_latency_ms"]),
        )
    console.print(table)


def _format_channel_map(channels: int, mapping: tuple[int, ...]) -> str:
    """Format a channel map for console summaries."""
    resolved = mapping if len(mapping) > 0 else tuple(range(1, int(channels) + 1))
    return ",".join(str(ch) for ch in resolved)


def _block_latency_ms(sample_rate: int, block_size: int) -> float:
    """Return the callback block duration in milliseconds."""
    return (1000.0 * float(block_size)) / float(max(1, sample_rate))


def _latency_metrics(
    *,
    sample_rate: int,
    block_size: int,
    engine_summary: dict[str, object],
    reported_input_latency_seconds: float | None = None,
    reported_output_latency_seconds: float | None = None,
) -> dict[str, float | None]:
    """Combine algorithmic, block, and backend latency into practical estimates."""
    block_latency_ms = _block_latency_ms(sample_rate, block_size)
    algorithm_latency_ms = _summary_float(engine_summary, "dereverb_latency_ms")
    reported_input_latency_ms = _seconds_to_ms(reported_input_latency_seconds)
    reported_output_latency_ms = _seconds_to_ms(reported_output_latency_seconds)
    reported_backend_latency_ms = None
    if reported_input_latency_ms is not None or reported_output_latency_ms is not None:
        reported_backend_latency_ms = (reported_input_latency_ms or 0.0) + (
            reported_output_latency_ms or 0.0
        )
    software_latency_ms = block_latency_ms + algorithm_latency_ms
    end_to_end_latency_ms = software_latency_ms
    if reported_backend_latency_ms is not None:
        end_to_end_latency_ms += reported_backend_latency_ms
    return {
        "block_latency_ms": block_latency_ms,
        "algorithm_latency_ms": algorithm_latency_ms,
        "software_latency_ms": software_latency_ms,
        "reported_input_latency_ms": reported_input_latency_ms,
        "reported_output_latency_ms": reported_output_latency_ms,
        "reported_backend_latency_ms": reported_backend_latency_ms,
        "end_to_end_latency_ms": end_to_end_latency_ms,
    }


def _seconds_to_ms(seconds: float | None) -> float | None:
    """Convert optional seconds into milliseconds."""
    if seconds is None:
        return None
    return 1000.0 * float(seconds)


def _format_latency_ms(value_ms: float | None) -> str:
    """Format optional latency values for console output."""
    if value_ms is None:
        return "unreported"
    return f"{float(value_ms):.1f}ms"


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
    if config.freeze:
        required_seconds = realtime_freeze_proxy_required_seconds(config)
        if required_seconds > float(config.algo_proxy_ir_max_seconds):
            msg = (
                "--freeze with this --rt60 needs an algorithmic proxy IR of about "
                f"{required_seconds:.1f}s. Increase --algo-proxy-ir-max-seconds "
                "or lower --rt60 to avoid a startup that looks hung."
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
        _REALTIME_FREEZE_PROXY_SECONDS,
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


def _summary_float(summary: dict[str, object], key: str, default: float = 0.0) -> float:
    """Safely coerce numeric summary fields for console display."""
    value = summary.get(key, default)
    return float(value) if isinstance(value, (int, float)) else float(default)
