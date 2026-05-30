# ruff: noqa: B008
"""Render command wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from verbx.config import (
    AmbiChannelOrder,
    AmbiDecodeTo,
    AmbiEncodeFrom,
    AmbiNormalization,
    AutoFitProfile,
    AutomationMode,
    ChannelLayout,
    DeviceName,
    EngineName,
    FDNNonlinearityMode,
    FDNSpatialCouplingMode,
    FeatureGuidePolicy,
    IRMatrixLayout,
    IRMode,
    IRMorphMismatchPolicy,
    IRNormalize,
    LimiterDetect,
    LimiterMode,
    ModCombine,
    ModTarget,
    NormalizeStage,
    OutputContainer,
    OutputPeakNorm,
    OutputSubtype,
    TailStopMetric,
)
from verbx.core.control_targets import RT60_DEFAULT_SECONDS, RT60_MAX_SECONDS, RT60_MIN_SECONDS


def _forward(name: str, params: dict[str, Any]) -> None:
    from verbx import cli as cli_module

    return cli_module.get_command_impl(name)(**params)


def render(
    ctx: typer.Context,
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    outfile: Path = typer.Argument(..., resolve_path=True),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help=(
            "Named preset baseline (see `verbx presets`) or dynamic room shorthand "
            "`room:<width>x<depth>x<height>/<material>`. Explicitly supplied CLI "
            "options override preset values."
        ),
    ),
    auto_fit: AutoFitProfile = typer.Option(
        "none",
        "--auto-fit",
        help="Apply target-oriented heuristic profile: none, speech, music, drums, ambient.",
    ),
    engine: EngineName = typer.Option("auto", "--engine", help="Engine: conv, algo, or auto."),
    rt60: float = typer.Option(
        RT60_DEFAULT_SECONDS, "--rt60", min=RT60_MIN_SECONDS, max=RT60_MAX_SECONDS
    ),
    wet: float = typer.Option(0.8, "--wet", min=0.0, max=1.0),
    dry: float = typer.Option(0.2, "--dry", min=0.0, max=1.0),
    repeat: int = typer.Option(1, "--repeat", min=1),
    freeze: bool = typer.Option(False, "--freeze", help="Enable freeze segment mode."),
    start: float | None = typer.Option(None, "--start", min=0.0),
    end: float | None = typer.Option(None, "--end", min=0.0),
    pre_delay_ms: float = typer.Option(20.0, "--pre-delay-ms", min=0.0),
    pre_delay: str | None = typer.Option(None, "--pre-delay"),
    bpm: float | None = typer.Option(None, "--bpm", min=1.0),
    damping: float = typer.Option(0.45, "--damping", min=0.0, max=1.0),
    width: float = typer.Option(1.0, "--width", min=0.0, max=2.0),
    mod_depth_ms: float = typer.Option(2.0, "--mod-depth-ms", min=0.0),
    mod_rate_hz: float = typer.Option(0.1, "--mod-rate-hz", min=0.0),
    mod_target: ModTarget = typer.Option(
        "none",
        "--mod-target",
        help="Dynamic parameter target: none, mix/wet, or gain-db.",
    ),
    mod_source: list[str] | None = typer.Option(
        None,
        "--mod-source",
        help=(
            "Repeatable modulation source spec. "
            "Examples: lfo:sine:0.08:1.0*0.7, env:20:350, "
            "audio-env:sidechain.wav:10:200, const:0.5."
        ),
    ),
    mod_route: list[str] | None = typer.Option(
        None,
        "--mod-route",
        help=(
            "Repeatable advanced route: "
            "<target>:<min>:<max>:<combine>:<smooth_ms>:<src1>,<src2>,... "
            "(target: mix|wet|gain-db)."
        ),
    ),
    mod_min: float = typer.Option(
        0.0,
        "--mod-min",
        help="Minimum mapped value for the modulation target.",
    ),
    mod_max: float = typer.Option(
        1.0,
        "--mod-max",
        help="Maximum mapped value for the modulation target.",
    ),
    mod_combine: ModCombine = typer.Option(
        "sum",
        "--mod-combine",
        help="How multiple sources are combined: sum, avg, or max.",
    ),
    mod_smooth_ms: float = typer.Option(
        20.0,
        "--mod-smooth-ms",
        min=0.0,
        help="One-pole smoothing time for modulation control signals.",
    ),
    allpass_stages: int = typer.Option(
        6,
        "--allpass-stages",
        min=0,
        max=64,
        help="Number of Schroeder allpass diffusion stages (0 disables diffusion).",
    ),
    allpass_gain: str = typer.Option(
        "0.7",
        "--allpass-gain",
        help=(
            "Allpass gain. Use one value (e.g. 0.7) for all stages, or a "
            "comma-separated list (e.g. 0.72,0.70,0.68,0.66) for per-stage gains."
        ),
    ),
    allpass_delays_ms: str | None = typer.Option(
        None,
        "--allpass-delays-ms",
        help=(
            "Optional comma-separated allpass delay list in milliseconds. Example: 5,7,11,17,23,29"
        ),
    ),
    comb_delays_ms: str | None = typer.Option(
        None,
        "--comb-delays-ms",
        help=(
            "Optional comma-separated FDN comb-like delay list in milliseconds. "
            "Example: 31,37,41,43,47,53,59,67"
        ),
    ),
    comb_cloud: bool = typer.Option(
        False,
        "--comb-cloud/--no-comb-cloud",
        help="Enable an optional pre-FDN cloud of decorrelated feedback comb filters.",
    ),
    comb_cloud_count: int = typer.Option(
        24,
        "--comb-cloud-count",
        min=1,
        max=128,
        help="Number of comb filters generated for the optional comb cloud.",
    ),
    comb_cloud_feedback: float = typer.Option(
        0.35,
        "--comb-cloud-feedback",
        min=0.0,
        max=0.95,
        help="Feedback amount used by the optional comb cloud (0..0.95).",
    ),
    comb_cloud_mix: float = typer.Option(
        0.25,
        "--comb-cloud-mix",
        min=0.0,
        max=1.0,
        help="Blend from diffusion output into comb-cloud color output (0..1).",
    ),
    comb_cloud_delays_ms: str | None = typer.Option(
        None,
        "--comb-cloud-delays-ms",
        help=(
            "Optional comma-separated delay list in milliseconds for the comb cloud. "
            "Providing this auto-enables the mode."
        ),
    ),
    comb_cloud_seed: int = typer.Option(
        2026,
        "--comb-cloud-seed",
        help="Deterministic seed used when generating the optional comb cloud.",
    ),
    fdn_lines: int = typer.Option(
        8,
        "--fdn-lines",
        min=1,
        max=64,
        help="FDN line count used when --comb-delays-ms is not provided.",
    ),
    fdn_matrix: str = typer.Option(
        "auto",
        "--fdn-matrix",
        help=(
            "FDN matrix topology: hadamard, householder, random_orthogonal, "
            "circulant, elliptic, tv_unitary, graph, or sdn_hybrid. "
            "Default resolves to hadamard."
        ),
    ),
    fdn_tv_rate_hz: float = typer.Option(
        0.0,
        "--fdn-tv-rate-hz",
        min=0.0,
        help="Block-rate update speed for --fdn-matrix tv_unitary (Hz).",
    ),
    fdn_tv_depth: float = typer.Option(
        0.0,
        "--fdn-tv-depth",
        min=0.0,
        max=1.0,
        help="Blend depth for --fdn-matrix tv_unitary (0..1).",
    ),
    fdn_dfm_delays_ms: str | None = typer.Option(
        None,
        "--fdn-dfm-delays-ms",
        help=(
            "Optional delay-feedback-matrix delays in milliseconds. "
            "Provide one value for broadcast or one per FDN line."
        ),
    ),
    fdn_sparse: bool = typer.Option(
        False,
        "--fdn-sparse/--no-fdn-sparse",
        help="Enable sparse high-order FDN pair-mixing mode.",
    ),
    fdn_sparse_degree: int = typer.Option(
        2,
        "--fdn-sparse-degree",
        min=1,
        max=16,
        help="Number of sparse pair-mixing stages used when --fdn-sparse is enabled.",
    ),
    fdn_cascade: bool = typer.Option(
        False,
        "--fdn-cascade/--no-fdn-cascade",
        help="Enable nested/cascaded FDN mode (small fast network into late network).",
    ),
    fdn_cascade_mix: float = typer.Option(
        0.35,
        "--fdn-cascade-mix",
        min=0.0,
        max=1.0,
        help="Injection amount from nested FDN into the main late-field network (0..1).",
    ),
    fdn_cascade_delay_scale: float = typer.Option(
        0.5,
        "--fdn-cascade-delay-scale",
        min=0.2,
        max=1.0,
        help="Delay scaling for nested FDN relative to primary FDN delays (0.2..1.0).",
    ),
    fdn_cascade_rt60_ratio: float = typer.Option(
        0.55,
        "--fdn-cascade-rt60-ratio",
        min=0.1,
        max=1.0,
        help="RT60 ratio for nested FDN relative to --rt60 (0.1..1.0).",
    ),
    fdn_rt60_low: float | None = typer.Option(
        None,
        "--fdn-rt60-low",
        min=RT60_MIN_SECONDS,
        max=RT60_MAX_SECONDS,
        help="Low-band RT60 target for multiband FDN decay shaping (seconds).",
    ),
    fdn_rt60_mid: float | None = typer.Option(
        None,
        "--fdn-rt60-mid",
        min=RT60_MIN_SECONDS,
        max=RT60_MAX_SECONDS,
        help="Mid-band RT60 target for multiband FDN decay shaping (seconds).",
    ),
    fdn_rt60_high: float | None = typer.Option(
        None,
        "--fdn-rt60-high",
        min=RT60_MIN_SECONDS,
        max=RT60_MAX_SECONDS,
        help="High-band RT60 target for multiband FDN decay shaping (seconds).",
    ),
    fdn_rt60_tilt: float = typer.Option(
        0.0,
        "--fdn-rt60-tilt",
        min=-1.0,
        max=1.0,
        help=(
            "Jot-style low/high RT skew around mid band (-1..1). "
            "Positive extends low-band decay and shortens highs."
        ),
    ),
    fdn_tonal_correction_strength: float = typer.Option(
        0.0,
        "--fdn-tonal-correction-strength",
        min=0.0,
        max=1.0,
        help=(
            "Track C tonal correction strength for multiband/tilted FDN response (0..1). "
            "Higher values apply stronger decay-color equalization."
        ),
    ),
    fdn_xover_low_hz: float = typer.Option(
        250.0,
        "--fdn-xover-low-hz",
        min=20.0,
        help="Low/mid crossover frequency used by multiband FDN decay shaping.",
    ),
    fdn_xover_high_hz: float = typer.Option(
        4_000.0,
        "--fdn-xover-high-hz",
        min=100.0,
        help="Mid/high crossover frequency used by multiband FDN decay shaping.",
    ),
    fdn_link_filter: str = typer.Option(
        "none",
        "--fdn-link-filter",
        help=("Feedback-link filter mode inside the FDN matrix path: none, lowpass, or highpass."),
    ),
    fdn_link_filter_hz: float = typer.Option(
        2_500.0,
        "--fdn-link-filter-hz",
        min=20.0,
        help="Cutoff frequency used by --fdn-link-filter (Hz).",
    ),
    fdn_link_filter_mix: float = typer.Option(
        1.0,
        "--fdn-link-filter-mix",
        min=0.0,
        max=1.0,
        help="Wet mix of feedback-link filter processing (0..1).",
    ),
    fdn_graph_topology: str = typer.Option(
        "ring",
        "--fdn-graph-topology",
        help="Graph topology for --fdn-matrix graph: ring, path, star, or random.",
    ),
    fdn_graph_degree: int = typer.Option(
        2,
        "--fdn-graph-degree",
        min=1,
        max=32,
        help="Graph neighborhood/connectivity degree for --fdn-matrix graph.",
    ),
    fdn_graph_seed: int = typer.Option(
        2026,
        "--fdn-graph-seed",
        help="Deterministic seed used to build graph-structured FDN pairings.",
    ),
    fdn_matrix_morph_to: str | None = typer.Option(
        None,
        "--fdn-matrix-morph-to",
        help="Optional target matrix family for gradual feedback-matrix morphing.",
    ),
    fdn_matrix_morph_seconds: float = typer.Option(
        0.0,
        "--fdn-matrix-morph-seconds",
        min=0.0,
        help="Duration (seconds) for matrix morph from --fdn-matrix to --fdn-matrix-morph-to.",
    ),
    fdn_spatial_coupling_mode: FDNSpatialCouplingMode = typer.Option(
        "none",
        "--fdn-spatial-coupling-mode",
        help=(
            "Directional wet-bus coupling mode: none, adjacent, front_rear, bed_top, all_to_all."
        ),
    ),
    fdn_spatial_coupling_strength: float = typer.Option(
        0.0,
        "--fdn-spatial-coupling-strength",
        min=0.0,
        max=1.0,
        help="Wet-bus directional coupling amount (0..1).",
    ),
    fdn_nonlinearity: FDNNonlinearityMode = typer.Option(
        "none",
        "--fdn-nonlinearity",
        help="Optional in-loop nonlinearity: none, tanh, or softclip.",
    ),
    fdn_nonlinearity_amount: float = typer.Option(
        0.0,
        "--fdn-nonlinearity-amount",
        min=0.0,
        max=1.0,
        help="Blend amount for in-loop nonlinearity shaping (0..1).",
    ),
    fdn_nonlinearity_drive: float = typer.Option(
        1.0,
        "--fdn-nonlinearity-drive",
        min=0.1,
        max=8.0,
        help="Drive multiplier for in-loop nonlinearity shaping.",
    ),
    room_size_macro: float = typer.Option(
        0.0,
        "--room-size-macro",
        min=-1.0,
        max=1.0,
        help="Perceptual room-size macro (-1..1) mapped to decay-time and spacing behavior.",
    ),
    clarity_macro: float = typer.Option(
        0.0,
        "--clarity-macro",
        min=-1.0,
        max=1.0,
        help="Perceptual clarity macro (-1..1) mapped to decay, damping, and wet balance.",
    ),
    warmth_macro: float = typer.Option(
        0.0,
        "--warmth-macro",
        min=-1.0,
        max=1.0,
        help="Perceptual warmth macro (-1..1) mapped to damping and spectral decay tilt.",
    ),
    envelopment_macro: float = typer.Option(
        0.0,
        "--envelopment-macro",
        min=-1.0,
        max=1.0,
        help="Perceptual envelopment macro (-1..1) mapped to width/decorrelation emphasis.",
    ),
    beast_mode: int = typer.Option(
        1,
        "--beast-mode",
        min=1,
        max=100,
        help=(
            "Scales core reverb parameters by an intensity multiplier (1-100) "
            "to push denser, longer, freeze-like tails."
        ),
    ),
    ir: Path | None = typer.Option(None, "--ir", exists=True, readable=True, resolve_path=True),
    ir_blend: list[Path] | None = typer.Option(
        None,
        "--ir-blend",
        exists=True,
        readable=True,
        resolve_path=True,
        help=(
            "Repeatable additional IR path for render-time convolution blending. "
            "Requires convolution render path."
        ),
    ),
    ir_blend_mix: list[float] | None = typer.Option(
        None,
        "--ir-blend-mix",
        min=0.0,
        max=1.0,
        help=(
            "Repeatable blend coefficient for each --ir-blend IR (0..1). "
            "Provide one value to broadcast to all blend IRs."
        ),
    ),
    ir_blend_mode: str = typer.Option(
        "equal-power",
        "--ir-blend-mode",
        help="IR blend morph mode: linear, equal-power, spectral, or envelope-aware.",
    ),
    ir_blend_early_ms: float = typer.Option(
        80.0,
        "--ir-blend-early-ms",
        min=0.0,
        help="Early/late split time (ms) used by envelope-aware and split blending modes.",
    ),
    ir_blend_early_alpha: float | None = typer.Option(
        None,
        "--ir-blend-early-alpha",
        min=0.0,
        max=1.0,
        help="Optional override alpha for early-reflection blend region.",
    ),
    ir_blend_late_alpha: float | None = typer.Option(
        None,
        "--ir-blend-late-alpha",
        min=0.0,
        max=1.0,
        help="Optional override alpha for late-tail blend region.",
    ),
    ir_blend_align_decay: bool = typer.Option(
        True,
        "--ir-blend-align-decay/--no-ir-blend-align-decay",
        help="Enable RT60 alignment before morphing to stabilize blend trajectories.",
    ),
    ir_blend_phase_coherence: float = typer.Option(
        0.75,
        "--ir-blend-phase-coherence",
        min=0.0,
        max=1.0,
        help="Phase-coherence safeguard strength for spectral/envelope-aware blending.",
    ),
    ir_blend_spectral_smooth_bins: int = typer.Option(
        3,
        "--ir-blend-spectral-smooth-bins",
        min=0,
        max=128,
        help="Frequency smoothing radius (FFT bins) used by spectral blend modes.",
    ),
    ir_blend_mismatch_policy: IRMorphMismatchPolicy = typer.Option(
        "coerce",
        "--ir-blend-mismatch-policy",
        help=(
            "Mismatch behavior for blend-source sample-rate/channel/duration differences: "
            "coerce (resample/align) or strict (fail)."
        ),
    ),
    ir_blend_cache_dir: str = typer.Option(
        ".verbx_cache/ir_morph",
        "--ir-blend-cache-dir",
        help="Cache directory for blended/morphed IR artifacts used by render workflow.",
    ),
    self_convolve: bool = typer.Option(
        False,
        "--self-convolve",
        help=(
            "Use INFILE as its own IR and force fast partitioned convolution "
            "(equivalent to --engine conv --ir INFILE)."
        ),
    ),
    ir_route_map: str = typer.Option(
        "auto",
        "--ir-route-map",
        help="Convolution route-map mode: auto, diagonal, broadcast, or full.",
    ),
    input_layout: ChannelLayout = typer.Option(
        "auto",
        "--input-layout",
        help=(
            "Input signal channel layout: auto, mono, stereo, LCR, 5.1, 7.1, "
            "7.1.2, 7.1.4, 7.2.4, 8.0, 16.0, 64.4"
        ),
    ),
    output_layout: ChannelLayout = typer.Option(
        "auto",
        "--output-layout",
        help=(
            "Output signal channel layout: auto, mono, stereo, LCR, 5.1, 7.1, "
            "7.1.2, 7.1.4, 7.2.4, 8.0, 16.0, 64.4"
        ),
    ),
    ir_normalize: IRNormalize = typer.Option("peak", "--ir-normalize"),
    ir_matrix_layout: IRMatrixLayout = typer.Option("output-major", "--ir-matrix-layout"),
    conv_route_start: str | None = typer.Option(
        None,
        "--conv-route-start",
        help="Convolution trajectory start position (index or alias, e.g. left, rear-left).",
    ),
    conv_route_end: str | None = typer.Option(
        None,
        "--conv-route-end",
        help="Convolution trajectory end position (index or alias).",
    ),
    conv_route_curve: str = typer.Option(
        "equal-power",
        "--conv-route-curve",
        help="Convolution trajectory curve: linear or equal-power.",
    ),
    ambi_order: int = typer.Option(
        0,
        "--ambi-order",
        min=0,
        max=7,
        help="Ambisonics order (0 disables Ambisonics-specific processing).",
    ),
    ambi_normalization: AmbiNormalization = typer.Option(
        "auto",
        "--ambi-normalization",
        help="Ambisonics normalization convention: auto, sn3d, n3d, or fuma.",
    ),
    channel_order: AmbiChannelOrder = typer.Option(
        "auto",
        "--channel-order",
        help="Ambisonics channel order convention: auto, acn, or fuma.",
    ),
    ambi_encode_from: AmbiEncodeFrom = typer.Option(
        "none",
        "--ambi-encode-from",
        help="Encode input bus into FOA before render: none, mono, or stereo.",
    ),
    ambi_decode_to: AmbiDecodeTo = typer.Option(
        "none",
        "--ambi-decode-to",
        help="Decode Ambisonics output after render: none or stereo.",
    ),
    ambi_rotate_yaw_deg: float = typer.Option(
        0.0,
        "--ambi-rotate-yaw-deg",
        help="Listener yaw rotation in degrees applied in Ambisonic domain.",
    ),
    algo_decorrelation_front: float = typer.Option(
        0.0,
        "--algo-front-variance",
        min=0.0,
        max=1.0,
        help="Algorithmic surround decorrelation variance for front channels.",
    ),
    algo_decorrelation_rear: float = typer.Option(
        0.0,
        "--algo-rear-variance",
        min=0.0,
        max=1.0,
        help="Algorithmic surround decorrelation variance for rear channels.",
    ),
    algo_decorrelation_top: float = typer.Option(
        0.0,
        "--algo-top-variance",
        min=0.0,
        max=1.0,
        help="Algorithmic surround decorrelation variance for top channels.",
    ),
    tail_limit: float | None = typer.Option(None, "--tail-limit", min=0.0),
    tail_stop_threshold_db: float = typer.Option(
        -120.0,
        "--tail-stop-threshold-db",
        min=-240.0,
        max=0.0,
        help="Tail completion threshold in dBFS used for final zero-tail writeout.",
    ),
    tail_stop_hold_ms: float = typer.Option(
        10.0,
        "--tail-stop-hold-ms",
        min=0.0,
        help="Explicit zero-hold duration appended after tail completion.",
    ),
    tail_stop_metric: TailStopMetric = typer.Option(
        "peak",
        "--tail-stop-metric",
        help="Tail stop detector metric: peak or rms.",
    ),
    threads: int | None = typer.Option(None, "--threads", min=1),
    device: DeviceName = typer.Option(
        "auto",
        "--device",
        help="Compute device preference: auto, cpu, cuda, or mps (Apple Silicon).",
    ),
    algo_stream: bool = typer.Option(
        False,
        "--algo-stream/--no-algo-stream",
        help="Use algorithmic-to-convolution proxy streaming path for long algorithmic renders.",
    ),
    algo_proxy_ir_max_seconds: float = typer.Option(
        120.0,
        "--algo-proxy-ir-max-seconds",
        min=1.0,
        help="Maximum proxy-IR duration used by --algo-stream.",
    ),
    algo_gpu_proxy: bool = typer.Option(
        False,
        "--algo-gpu-proxy/--no-algo-gpu-proxy",
        help=(
            "Route algorithmic render through proxy convolution path "
            "to leverage CUDA convolution."
        ),
    ),
    partition_size: int = typer.Option(16_384, "--partition-size", min=256),
    quality_preset: str = typer.Option(
        "hd",
        "--quality-preset",
        help=(
            "Output-definition preset: sd=44.1 kHz PCM16, md=48 kHz PCM24, "
            "hd=192 kHz float32 (default). Explicit --target-sr/--out-subtype override."
        ),
    ),
    target_sr: int | None = typer.Option(
        None,
        "--target-sr",
        min=1,
        help="Optional output/render sample rate (Hz). Input is resampled internally if needed.",
    ),
    ir_gen: bool = typer.Option(False, "--ir-gen"),
    ir_gen_mode: IRMode = typer.Option("hybrid", "--ir-gen-mode"),
    ir_gen_length: float = typer.Option(60.0, "--ir-gen-length", min=0.1),
    ir_gen_seed: int = typer.Option(0, "--ir-gen-seed"),
    ir_gen_cache_dir: str = typer.Option(".verbx_cache/irs", "--ir-gen-cache-dir"),
    block_size: int = typer.Option(4096, "--block-size", min=256),
    target_lufs: float | None = typer.Option(None, "--target-lufs"),
    target_peak_dbfs: float | None = typer.Option(None, "--target-peak-dbfs"),
    true_peak: bool = typer.Option(True, "--true-peak/--sample-peak"),
    limiter: bool = typer.Option(True, "--limiter/--no-limiter"),
    limiter_mode: LimiterMode = typer.Option(
        "tanh",
        "--limiter-mode",
        help="Limiter transfer curve: tanh, arctan, softsign, or hard.",
    ),
    limiter_detect: LimiterDetect = typer.Option(
        "peak",
        "--limiter-detect",
        help="Limiter detector mode: peak or rms.",
    ),
    limiter_threshold_dbfs: float | None = typer.Option(
        None,
        "--limiter-threshold-dbfs",
        help="Limiter onset threshold in dBFS. Defaults to the active peak target/ceiling.",
    ),
    limiter_ceiling_dbfs: float | None = typer.Option(
        None,
        "--limiter-ceiling-dbfs",
        help="Limiter output ceiling in dBFS. Defaults to the active peak target or -1 dBFS.",
    ),
    limiter_knee_db: float = typer.Option(6.0, "--limiter-knee-db", min=0.0),
    limiter_drive: float = typer.Option(1.0, "--limiter-drive", min=1e-6),
    limiter_mix: float = typer.Option(1.0, "--limiter-mix", min=0.0, max=1.0),
    limiter_attack_ms: float = typer.Option(0.5, "--limiter-attack-ms", min=0.0),
    limiter_release_ms: float = typer.Option(80.0, "--limiter-release-ms", min=0.0),
    limiter_lookahead_ms: float = typer.Option(1.5, "--limiter-lookahead-ms", min=0.0),
    limiter_stereo_link: bool = typer.Option(
        True,
        "--limiter-stereo-link/--no-limiter-stereo-link",
        help="Link channels in the limiter detector to preserve stereo image.",
    ),
    limiter_oversample: int = typer.Option(2, "--limiter-oversample", min=1, max=16),
    limiter_pre_gain_db: float = typer.Option(
        0.0,
        "--limiter-pre-gain-db",
        min=-48.0,
        max=48.0,
    ),
    limiter_post_gain_db: float = typer.Option(
        0.0,
        "--limiter-post-gain-db",
        min=-48.0,
        max=48.0,
    ),
    limiter_dc_block: bool = typer.Option(
        False,
        "--limiter-dc-block/--no-limiter-dc-block",
        help="Apply a gentle DC blocker before limiter detection.",
    ),
    normalize_stage: NormalizeStage = typer.Option("post", "--normalize-stage"),
    repeat_target_lufs: float | None = typer.Option(None, "--repeat-target-lufs"),
    repeat_target_peak_dbfs: float | None = typer.Option(None, "--repeat-target-peak-dbfs"),
    out_subtype: OutputSubtype = typer.Option(
        "auto",
        "--out-subtype",
        help=(
            "Output file subtype. Internal DSP runs in float64 regardless of container subtype; "
            "use float64/float32/PCM per delivery needs."
        ),
    ),
    output_container: OutputContainer = typer.Option(
        "auto",
        "--output-container",
        help="Output container mode: auto, wav, w64, or rf64.",
    ),
    output_peak_norm: OutputPeakNorm = typer.Option(
        "none",
        "--output-peak-norm",
        help=(
            "Final peak normalization mode: none, input peak match, explicit target, or full-scale."
        ),
    ),
    output_peak_target_dbfs: float | None = typer.Option(
        None,
        "--output-peak-target-dbfs",
        help="Target dBFS used when --output-peak-norm target is selected.",
    ),
    shimmer: bool = typer.Option(False, "--shimmer"),
    shimmer_semitones: float = typer.Option(12.0, "--shimmer-semitones"),
    shimmer_mix: float = typer.Option(0.25, "--shimmer-mix", min=0.0, max=1.0),
    shimmer_feedback: float = typer.Option(0.35, "--shimmer-feedback", min=0.0, max=1.25),
    shimmer_highcut: float | None = typer.Option(10_000.0, "--shimmer-highcut", min=10.0),
    shimmer_lowcut: float | None = typer.Option(300.0, "--shimmer-lowcut", min=10.0),
    shimmer_spatial: bool = typer.Option(
        False,
        "--shimmer-spatial/--no-shimmer-spatial",
        help="Enable multichannel shimmer spatial decorrelation.",
    ),
    shimmer_spread_cents: float = typer.Option(
        8.0,
        "--shimmer-spread-cents",
        min=0.0,
        help="Per-channel shimmer detune spread in cents (multichannel).",
    ),
    shimmer_decorrelation_ms: float = typer.Option(
        1.5,
        "--shimmer-decorrelation-ms",
        min=0.0,
        help="Per-channel shimmer delay spread in milliseconds.",
    ),
    er_geometry: bool = typer.Option(
        False,
        "--er-geometry/--no-er-geometry",
        help="Enable first-order image-source early-reflection pre-stage.",
    ),
    er_room_dims_m: str = typer.Option(
        "10,7,3",
        "--er-room-dims-m",
        help="Room dimensions in meters: L,W,H",
    ),
    er_source_pos_m: str = typer.Option(
        "2,2,1.5",
        "--er-source-pos-m",
        help="Source position in meters: x,y,z",
    ),
    er_listener_pos_m: str = typer.Option(
        "5,3.5,1.5",
        "--er-listener-pos-m",
        help="Listener position in meters: x,y,z",
    ),
    er_absorption: float = typer.Option(
        0.35,
        "--er-absorption",
        min=0.0,
        max=0.99,
        help="Wall absorption coefficient for early-reflection stage.",
    ),
    er_material: str = typer.Option(
        "studio",
        "--er-material",
        help="Early-reflection material preset: anechoic, dead, studio, hall, stone, or custom.",
    ),
    unsafe_self_oscillate: bool = typer.Option(
        False,
        "--unsafe-self-oscillate/--safe-no-self-oscillate",
        help=(
            "UNSAFE: permit feedback-path gains above unity in algorithmic mode for "
            "self-oscillating tails."
        ),
    ),
    unsafe_loop_gain: float = typer.Option(
        1.02,
        "--unsafe-loop-gain",
        min=0.01,
        max=1.25,
        help=(
            "UNSAFE loop-gain scale used with --unsafe-self-oscillate. "
            "Values >1.0 encourage self-oscillation."
        ),
    ),
    duck: bool = typer.Option(False, "--duck"),
    duck_attack: float = typer.Option(20.0, "--duck-attack", min=0.1),
    duck_release: float = typer.Option(350.0, "--duck-release", min=0.1),
    duck_strength: float = typer.Option(
        0.75,
        "--duck-strength",
        min=0.0,
        max=1.0,
        help="How strongly the wet field is attenuated when the sidechain rises.",
    ),
    duck_floor: float = typer.Option(
        0.0,
        "--duck-floor",
        min=0.0,
        max=1.0,
        help="Minimum wet gain held during ducking; useful for softer pumping.",
    ),
    bloom: float = typer.Option(0.0, "--bloom", min=0.0),
    bloom_mix: float | None = typer.Option(
        None,
        "--bloom-mix",
        min=0.0,
        max=1.0,
        help="Override bloom blend amount. Default auto-scales from --bloom.",
    ),
    lowcut: float | None = typer.Option(None, "--lowcut", min=10.0),
    lowcut_order: int = typer.Option(
        2,
        "--lowcut-order",
        min=1,
        max=8,
        help="Butterworth order used by the post-wet high-pass filter.",
    ),
    highcut: float | None = typer.Option(None, "--highcut", min=10.0),
    highcut_order: int = typer.Option(
        2,
        "--highcut-order",
        min=1,
        max=8,
        help="Butterworth order used by the post-wet low-pass filter.",
    ),
    tilt: float = typer.Option(0.0, "--tilt"),
    tilt_pivot_hz: float = typer.Option(
        1_000.0,
        "--tilt-pivot-hz",
        min=20.0,
        help="Pivot frequency used by the post-wet tilt EQ.",
    ),
    automation_file: Path | None = typer.Option(
        None,
        "--automation-file",
        exists=True,
        readable=True,
        resolve_path=True,
        help="JSON/CSV automation lanes used for time-varying render control.",
    ),
    automation_mode: AutomationMode = typer.Option(
        "auto",
        "--automation-mode",
        help="Automation evaluation mode: auto, sample, or block.",
    ),
    automation_block_ms: float = typer.Option(
        20.0,
        "--automation-block-ms",
        min=0.1,
        help="Control block size in milliseconds when automation mode is block.",
    ),
    automation_smoothing_ms: float = typer.Option(
        20.0,
        "--automation-smoothing-ms",
        min=0.0,
        help="Default smoothing time (ms) applied to automation lanes.",
    ),
    automation_slew_limit_per_s: float | None = typer.Option(
        None,
        "--automation-slew-limit-per-s",
        min=0.0,
        help=(
            "Optional max control slew as target-range fraction per second; "
            "0/None disables slew guard."
        ),
    ),
    automation_deadband: float = typer.Option(
        0.0,
        "--automation-deadband",
        min=0.0,
        max=1.0,
        help=(
            "Optional control deadband as target-range fraction; "
            "small changes below threshold are suppressed."
        ),
    ),
    automation_clamp: list[str] | None = typer.Option(
        None,
        "--automation-clamp",
        help="Clamp override in target:min:max format (repeatable).",
    ),
    automation_point: list[str] | None = typer.Option(
        None,
        "--automation-point",
        help=(
            "Inline automation control point in target:time_s:value[:interp] format (repeatable)."
        ),
    ),
    automation_trace_out: str | None = typer.Option(
        None,
        "--automation-trace-out",
        help="Optional CSV path for resolved sample-level automation curves.",
    ),
    feature_vector_lane: list[str] | None = typer.Option(
        None,
        "--feature-vector-lane",
        help=(
            "Feature-vector mapping lane (repeatable). "
            "Format: target=<target>,source=<feature>[,weight=<w>][,bias=<b>]"
            "[,curve=<linear|smoothstep|exp|log|tanh|power>][,curve_amount=<a>]"
            "[,hysteresis_up=<u>][,hysteresis_down=<d>][,combine=<replace|add|multiply>]"
            "[,smoothing_ms=<ms>]"
        ),
    ),
    feature_vector_frame_ms: float = typer.Option(
        40.0,
        "--feature-vector-frame-ms",
        min=1.0,
        help="Frame size used for feature-vector extraction (ms).",
    ),
    feature_vector_hop_ms: float = typer.Option(
        20.0,
        "--feature-vector-hop-ms",
        min=1.0,
        help="Hop size used for feature-vector extraction (ms).",
    ),
    feature_guide: Path | None = typer.Option(
        None,
        "--feature-guide",
        exists=True,
        readable=True,
        resolve_path=True,
        help=(
            "Optional external guide audio used for feature-vector extraction "
            "instead of INFILE (Track B external feature-guide ingest)."
        ),
    ),
    feature_guide_policy: FeatureGuidePolicy = typer.Option(
        "align",
        "--feature-guide-policy",
        help=(
            "Mismatch policy for --feature-guide relative to render context: "
            "align (deterministic resample + hold/trim + mixdown) or strict."
        ),
    ),
    feature_vector_trace_out: str | None = typer.Option(
        None,
        "--feature-vector-trace-out",
        help="Optional CSV path for feature+parameter trace exports.",
    ),
    frames_out: str | None = typer.Option(None, "--frames-out"),
    analysis_out: str | None = typer.Option(None, "--analysis-out"),
    lucky: int | None = typer.Option(
        None,
        "--lucky",
        min=1,
        max=500,
        help=(
            "Generate N wild random renders from one input using randomized parameters. "
            "Outputs are written to --lucky-out-dir (or OUTFILE parent by default)."
        ),
    ),
    lucky_out_dir: Path | None = typer.Option(
        None,
        "--lucky-out-dir",
        resolve_path=True,
        help="Output directory used when --lucky is enabled.",
    ),
    lucky_seed: int | None = typer.Option(
        None,
        "--lucky-seed",
        help="Optional deterministic seed for --lucky render generation.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Suppress console summary tables while still writing output and analysis artifacts.",
    ),
    verbosity: int = typer.Option(
        1,
        "--verbosity",
        min=0,
        max=2,
        help=(
            "Console detail level: 0=minimal summary, 1=summary + output features (default), "
            "2=also include input feature table."
        ),
    ),
    silent: bool = typer.Option(
        False,
        "--silent",
        help="Disable analysis JSON generation and console output.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate inputs and print resolved render plan without writing audio.",
    ),
    repro_bundle: bool = typer.Option(
        False,
        "--repro-bundle",
        help="Write a reproducibility/support JSON bundle next to OUTFILE.",
    ),
    repro_bundle_out: Path | None = typer.Option(
        None,
        "--repro-bundle-out",
        resolve_path=True,
        help="Optional explicit path for reproducibility/support JSON bundle.",
    ),
    failure_report_out: Path | None = typer.Option(
        None,
        "--failure-report-out",
        resolve_path=True,
        help="Optional JSON report path populated when render execution fails.",
    ),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
    json_out: Path | None = typer.Option(
        None,
        "--json-out",
        resolve_path=True,
        help="Optional path to write the full render report as JSON.",
    ),
) -> None:
    _forward("_render_impl", dict(locals()))
