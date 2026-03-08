"""Typer CLI entrypoint for verbx.

Commands are grouped by workflow:
- top-level render/analyze/suggest/presets
- ``ir`` synthesis/inspection
- ``cache`` management
- ``batch`` orchestration
"""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import numpy as np
import soundfile as sf
import typer
from rich.console import Console
from rich.table import Table

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.analysis.framewise import write_framewise_csv
from verbx.config import (
    AmbiChannelOrder,
    AmbiDecodeTo,
    AmbiEncodeFrom,
    AmbiNormalization,
    AutomationMode,
    ChannelLayout,
    DeviceName,
    EngineName,
    IRMatrixLayout,
    IRMode,
    IRNormalize,
    ModCombine,
    ModTarget,
    NormalizeStage,
    OutputPeakNorm,
    OutputSubtype,
    RenderConfig,
)
from verbx.core.automation import (
    ENGINE_AUTOMATION_TARGETS,
    collect_automation_targets,
    parse_automation_clamp_overrides,
    parse_automation_point_specs,
)
from verbx.core.batch_scheduler import (
    BatchJobResult,
    BatchJobSpec,
    BatchSchedulePolicy,
    estimate_job_cost,
    order_jobs,
    run_parallel_batch,
)
from verbx.core.modulation import parse_mod_route_spec, parse_mod_sources
from verbx.core.pipeline import run_render_pipeline
from verbx.core.spatial import (
    ambisonic_channel_count,
    normalize_ambisonic_metadata,
)
from verbx.core.fdn_capabilities import (
    FDN_GRAPH_TOPOLOGY_CHOICES,
    FDN_LINK_FILTER_CHOICES,
    FDN_MATRIX_CHOICES,
    normalize_fdn_graph_topology_name as _shared_normalize_fdn_graph_topology_name,
    normalize_fdn_link_filter_name as _shared_normalize_fdn_link_filter_name,
    normalize_fdn_matrix_name as _shared_normalize_fdn_matrix_name,
)
from verbx.core.tempo import parse_pre_delay_ms
from verbx.io.audio import read_audio, validate_audio_path
from verbx.ir.fitting import (
    IRFitCandidate,
    IRFitScore,
    IRFitTarget,
    build_ir_fit_candidates,
    derive_ir_fit_target,
    score_ir_candidate,
)
from verbx.ir.generator import IRGenConfig, generate_or_load_cached_ir, write_ir_artifacts
from verbx.ir.metrics import analyze_ir
from verbx.ir.shaping import apply_ir_shaping
from verbx.ir.tuning import analyze_audio_for_tuning, parse_frequency_hz
from verbx.logging import configure_logging
from verbx.presets.default_presets import preset_names

IRFileFormat = Literal["auto", "wav", "flac", "aiff", "aif", "ogg", "caf"]
_FDN_MATRIX_CHOICES = set(FDN_MATRIX_CHOICES)
_FDN_GRAPH_TOPOLOGY_CHOICES = set(FDN_GRAPH_TOPOLOGY_CHOICES)
_FDN_LINK_FILTER_CHOICES = set(FDN_LINK_FILTER_CHOICES)
_IR_ROUTE_MAP_CHOICES = {
    "auto",
    "diagonal",
    "broadcast",
    "full",
}
_CONV_ROUTE_CURVE_CHOICES = {
    "linear",
    "equal-power",
}
_AMBI_NORMALIZATION_CHOICES = {
    "auto",
    "sn3d",
    "n3d",
    "fuma",
}
_AMBI_CHANNEL_ORDER_CHOICES = {
    "auto",
    "acn",
    "fuma",
}
_AMBI_ENCODE_CHOICES = {
    "none",
    "mono",
    "stereo",
}
_AMBI_DECODE_CHOICES = {
    "none",
    "stereo",
}
_AUTOMATION_MODE_CHOICES = {
    "auto",
    "sample",
    "block",
}


class LuckyIRProcessConfig(TypedDict):
    """Typed config payload for ``ir process --lucky`` randomization."""

    damping: float
    lowcut: float | None
    highcut: float | None
    tilt: float
    normalize: Literal["none", "peak", "rms"]
    peak_dbfs: float
    target_lufs: float | None
    true_peak: bool

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    help="Extreme reverb CLI with scalable DSP architecture.",
)
ir_app = typer.Typer(help="Impulse response workflows.")
cache_app = typer.Typer(help="IR cache inspection and cleanup.")
batch_app = typer.Typer(help="Batch manifest generation and rendering.")

app.add_typer(ir_app, name="ir")
app.add_typer(cache_app, name="cache")
app.add_typer(batch_app, name="batch")

console = Console()


@app.command()
def render(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    outfile: Path = typer.Argument(..., resolve_path=True),
    engine: EngineName = typer.Option("auto", "--engine", help="Engine: conv, algo, or auto."),
    rt60: float = typer.Option(60.0, "--rt60", min=0.1),
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
            "Optional comma-separated allpass delay list in milliseconds. "
            "Example: 5,7,11,17,23,29"
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
            "circulant, elliptic, tv_unitary, or graph. Default resolves to hadamard."
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
        min=0.1,
        help="Low-band RT60 target for multiband FDN decay shaping (seconds).",
    ),
    fdn_rt60_mid: float | None = typer.Option(
        None,
        "--fdn-rt60-mid",
        min=0.1,
        help="Mid-band RT60 target for multiband FDN decay shaping (seconds).",
    ),
    fdn_rt60_high: float | None = typer.Option(
        None,
        "--fdn-rt60-high",
        min=0.1,
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
        help=(
            "Feedback-link filter mode inside the FDN matrix path: "
            "none, lowpass, or highpass."
        ),
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
        help="Input signal channel layout: auto, mono, stereo, LCR, 5.1, 7.1, 7.1.2, 7.1.4",
    ),
    output_layout: ChannelLayout = typer.Option(
        "auto",
        "--output-layout",
        help="Output signal channel layout: auto, mono, stereo, LCR, 5.1, 7.1, 7.1.2, 7.1.4",
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
    threads: int | None = typer.Option(None, "--threads", min=1),
    device: DeviceName = typer.Option(
        "auto",
        "--device",
        help="Compute device preference: auto, cpu, cuda, or mps (Apple Silicon).",
    ),
    partition_size: int = typer.Option(16_384, "--partition-size", min=256),
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
    normalize_stage: NormalizeStage = typer.Option("post", "--normalize-stage"),
    repeat_target_lufs: float | None = typer.Option(None, "--repeat-target-lufs"),
    repeat_target_peak_dbfs: float | None = typer.Option(None, "--repeat-target-peak-dbfs"),
    out_subtype: OutputSubtype = typer.Option(
        "auto",
        "--out-subtype",
        help="Output file subtype. Use float32 for 32-bit float WAV/AIFF where supported.",
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
    shimmer_feedback: float = typer.Option(0.35, "--shimmer-feedback", min=0.0, max=0.98),
    shimmer_highcut: float | None = typer.Option(10_000.0, "--shimmer-highcut", min=10.0),
    shimmer_lowcut: float | None = typer.Option(300.0, "--shimmer-lowcut", min=10.0),
    duck: bool = typer.Option(False, "--duck"),
    duck_attack: float = typer.Option(20.0, "--duck-attack", min=0.1),
    duck_release: float = typer.Option(350.0, "--duck-release", min=0.1),
    bloom: float = typer.Option(0.0, "--bloom", min=0.0),
    lowcut: float | None = typer.Option(None, "--lowcut", min=10.0),
    highcut: float | None = typer.Option(None, "--highcut", min=10.0),
    tilt: float = typer.Option(0.0, "--tilt"),
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
    automation_clamp: list[str] | None = typer.Option(
        None,
        "--automation-clamp",
        help="Clamp override in target:min:max format (repeatable).",
    ),
    automation_point: list[str] | None = typer.Option(
        None,
        "--automation-point",
        help=(
            "Inline automation control point in target:time_s:value[:interp] format "
            "(repeatable)."
        ),
    ),
    automation_trace_out: str | None = typer.Option(
        None,
        "--automation-trace-out",
        help="Optional CSV path for resolved sample-level automation curves.",
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
    silent: bool = typer.Option(False, "--silent", help="Disable analysis JSON + console output."),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    """Render input audio with algorithmic or convolution reverb."""
    resolved_pre_delay_ms = parse_pre_delay_ms(pre_delay, bpm, pre_delay_ms)
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

    config = RenderConfig(
        engine=engine,
        rt60=rt60,
        pre_delay_ms=resolved_pre_delay_ms,
        pre_delay_note=pre_delay,
        bpm=bpm,
        damping=damping,
        width=width,
        mod_depth_ms=mod_depth_ms,
        mod_rate_hz=mod_rate_hz,
        mod_target=mod_target,
        mod_sources=tuple(mod_source or []),
        mod_routes=tuple(mod_route or []),
        mod_min=mod_min,
        mod_max=mod_max,
        mod_combine=mod_combine,
        mod_smooth_ms=mod_smooth_ms,
        allpass_stages=allpass_stages,
        allpass_gain=float(parsed_allpass_gain_values[0]),
        allpass_gains=parsed_allpass_gain_values if len(parsed_allpass_gain_values) > 1 else (),
        allpass_delays_ms=parsed_allpass_delays,
        comb_delays_ms=parsed_comb_delays,
        fdn_lines=fdn_lines,
        fdn_matrix=(
            "hadamard"
            if fdn_matrix.strip().lower() == "auto"
            else _normalize_fdn_matrix_name(fdn_matrix)
        ),
        fdn_tv_rate_hz=fdn_tv_rate_hz,
        fdn_tv_depth=fdn_tv_depth,
        fdn_tv_seed=2026,
        fdn_dfm_delays_ms=parsed_dfm_delays,
        fdn_sparse=fdn_sparse,
        fdn_sparse_degree=fdn_sparse_degree,
        fdn_cascade=fdn_cascade,
        fdn_cascade_mix=fdn_cascade_mix,
        fdn_cascade_delay_scale=fdn_cascade_delay_scale,
        fdn_cascade_rt60_ratio=fdn_cascade_rt60_ratio,
        fdn_rt60_low=fdn_rt60_low,
        fdn_rt60_mid=fdn_rt60_mid,
        fdn_rt60_high=fdn_rt60_high,
        fdn_rt60_tilt=fdn_rt60_tilt,
        fdn_tonal_correction_strength=fdn_tonal_correction_strength,
        fdn_xover_low_hz=fdn_xover_low_hz,
        fdn_xover_high_hz=fdn_xover_high_hz,
        fdn_link_filter=_normalize_fdn_link_filter_name(fdn_link_filter),
        fdn_link_filter_hz=fdn_link_filter_hz,
        fdn_link_filter_mix=fdn_link_filter_mix,
        fdn_graph_topology=_normalize_fdn_graph_topology_name(fdn_graph_topology),
        fdn_graph_degree=fdn_graph_degree,
        fdn_graph_seed=fdn_graph_seed,
        room_size_macro=room_size_macro,
        clarity_macro=clarity_macro,
        warmth_macro=warmth_macro,
        envelopment_macro=envelopment_macro,
        algo_decorrelation_front=algo_decorrelation_front,
        algo_decorrelation_rear=algo_decorrelation_rear,
        algo_decorrelation_top=algo_decorrelation_top,
        beast_mode=beast_mode,
        wet=wet,
        dry=dry,
        repeat=repeat,
        freeze=freeze,
        start=start,
        end=end,
        block_size=block_size,
        ir=None if ir is None else str(ir),
        input_layout=input_layout,
        output_layout=output_layout,
        self_convolve=self_convolve,
        ir_normalize=ir_normalize,
        ir_matrix_layout=ir_matrix_layout,
        ir_route_map=_normalize_ir_route_map_name(ir_route_map),
        conv_route_start=conv_route_start,
        conv_route_end=conv_route_end,
        conv_route_curve=_normalize_conv_route_curve_name(conv_route_curve),
        ambi_order=int(ambi_order),
        ambi_normalization=cast(AmbiNormalization, str(ambi_normalization).strip().lower()),
        channel_order=cast(AmbiChannelOrder, str(channel_order).strip().lower()),
        ambi_encode_from=cast(AmbiEncodeFrom, str(ambi_encode_from).strip().lower()),
        ambi_decode_to=cast(AmbiDecodeTo, str(ambi_decode_to).strip().lower()),
        ambi_rotate_yaw_deg=float(ambi_rotate_yaw_deg),
        tail_limit=tail_limit,
        threads=threads,
        device=device,
        partition_size=partition_size,
        ir_gen=ir_gen,
        ir_gen_mode=ir_gen_mode,
        ir_gen_length=ir_gen_length,
        ir_gen_seed=ir_gen_seed,
        ir_gen_cache_dir=ir_gen_cache_dir,
        target_lufs=target_lufs,
        target_peak_dbfs=target_peak_dbfs,
        use_true_peak=true_peak,
        limiter=limiter,
        normalize_stage=normalize_stage,
        repeat_target_lufs=repeat_target_lufs,
        repeat_target_peak_dbfs=repeat_target_peak_dbfs,
        output_subtype=out_subtype,
        output_peak_norm=output_peak_norm,
        output_peak_target_dbfs=output_peak_target_dbfs,
        shimmer=shimmer,
        shimmer_semitones=shimmer_semitones,
        shimmer_mix=shimmer_mix,
        shimmer_feedback=shimmer_feedback,
        shimmer_highcut=shimmer_highcut,
        shimmer_lowcut=shimmer_lowcut,
        duck=duck,
        duck_attack=duck_attack,
        duck_release=duck_release,
        bloom=bloom,
        lowcut=lowcut,
        highcut=highcut,
        tilt=tilt,
        automation_file=None if automation_file is None else str(automation_file),
        automation_mode=cast(AutomationMode, str(automation_mode).strip().lower()),
        automation_block_ms=float(automation_block_ms),
        automation_smoothing_ms=float(automation_smoothing_ms),
        automation_clamp=tuple(automation_clamp or ()),
        automation_points=tuple(automation_point or ()),
        automation_trace_out=automation_trace_out,
        frames_out=frames_out,
        analysis_out=analysis_out,
        silent=silent,
        progress=progress,
    )

    _validate_render_call(infile, outfile, config)
    _validate_lucky_call(config, lucky, lucky_out_dir)
    configure_logging(verbose=not config.silent)

    if lucky is not None:
        try:
            info = sf.info(str(infile))
            duration_seconds = (
                float(info.frames) / float(info.samplerate) if info.samplerate > 0 else 0.0
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc

        out_dir = outfile.parent if lucky_out_dir is None else lucky_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        seed = _resolve_lucky_seed(lucky_seed)

        lucky_rows: list[dict[str, str]] = []
        for idx in range(lucky):
            rng = np.random.default_rng(seed + idx)
            lucky_config = _build_lucky_config(
                base=config,
                rng=rng,
                input_duration_seconds=duration_seconds,
            )
            lucky_config.progress = config.progress
            lucky_out = out_dir / f"{outfile.stem}.lucky_{idx + 1:03d}{outfile.suffix}"

            try:
                report = run_render_pipeline(infile=infile, outfile=lucky_out, config=lucky_config)
            except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
                raise typer.BadParameter(str(exc)) from exc

            lucky_rows.append(
                {
                    "index": str(idx + 1),
                    "outfile": str(lucky_out),
                    "engine": str(report.get("engine", "unknown")),
                    "rt60": f"{float(lucky_config.rt60):.2f}",
                    "repeat": str(int(lucky_config.repeat)),
                    "beast": str(int(lucky_config.beast_mode)),
                }
            )

        if not config.silent:
            summary = Table(title=f"Lucky Render Batch ({lucky} outputs)")
            summary.add_column("#", style="cyan", justify="right")
            summary.add_column("outfile", style="white")
            summary.add_column("engine", style="green")
            summary.add_column("rt60", justify="right")
            summary.add_column("repeat", justify="right")
            summary.add_column("beast", justify="right")
            for row in lucky_rows:
                summary.add_row(
                    row["index"],
                    row["outfile"],
                    row["engine"],
                    row["rt60"],
                    row["repeat"],
                    row["beast"],
                )
            console.print(summary)
        return

    try:
        report = run_render_pipeline(infile=infile, outfile=outfile, config=config)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if config.silent:
        return

    _print_render_summary(report)


@app.command()
def analyze(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    json_out: Path | None = typer.Option(None, "--json-out", resolve_path=True),
    lufs: bool = typer.Option(False, "--lufs", help="Include LUFS/true-peak/LRA metrics."),
    edr: bool = typer.Option(
        False,
        "--edr",
        help="Include EDR (Energy Decay Relief) summary metrics.",
    ),
    frames_out: Path | None = typer.Option(None, "--frames-out", resolve_path=True),
    ambi_order: int = typer.Option(
        0,
        "--ambi-order",
        min=0,
        max=7,
        help="Enable Ambisonics spatial metrics for the given order.",
    ),
    ambi_normalization: AmbiNormalization = typer.Option(
        "auto",
        "--ambi-normalization",
        help="Ambisonics normalization convention for analysis mode.",
    ),
    channel_order: AmbiChannelOrder = typer.Option(
        "auto",
        "--channel-order",
        help="Ambisonics channel order convention for analysis mode.",
    ),
) -> None:
    """Analyze an audio file and print a summary table."""
    _validate_analyze_call(infile, json_out, frames_out)
    try:
        validate_audio_path(str(infile))
        audio, sr = read_audio(str(infile))
        analyzer = AudioAnalyzer()
        metrics = analyzer.analyze(
            audio,
            sr,
            include_loudness=lufs,
            include_edr=edr,
            ambi_order=int(ambi_order) if int(ambi_order) > 0 else None,
            ambi_normalization=str(ambi_normalization).strip().lower(),
            ambi_channel_order=str(channel_order).strip().lower(),
        )
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title=f"Analysis: {infile.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for key in sorted(metrics):
        table.add_row(key, f"{metrics[key]:.6f}")
    console.print(table)

    if json_out is not None:
        payload = {"sample_rate": sr, "channels": audio.shape[1], "metrics": metrics}
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if frames_out is not None:
        write_framewise_csv(frames_out, audio, sr)


@app.command()
def suggest(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
) -> None:
    """Suggest practical render defaults from input analysis."""
    try:
        validate_audio_path(str(infile))
        audio, sr = read_audio(str(infile))
        analyzer = AudioAnalyzer()
        metrics = analyzer.analyze(audio, sr)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    duration = metrics["duration"]
    dynamic = metrics["dynamic_range"]
    flatness = metrics["spectral_flatness"]

    suggested_rt60 = float(np.clip(duration * 1.8, 25.0, 120.0))
    suggested_wet = float(np.clip(0.55 + (dynamic / 60.0), 0.4, 0.95))
    suggested_dry = float(np.clip(1.0 - (suggested_wet * 0.85), 0.05, 0.85))
    suggested_engine = "conv" if flatness < 0.12 else "algo"

    table = Table(title=f"Suggested Parameters: {infile.name}")
    table.add_column("Parameter", style="green")
    table.add_column("Suggested Value", style="white")
    table.add_row("engine", suggested_engine)
    table.add_row("rt60", f"{suggested_rt60:.2f}")
    table.add_row("wet", f"{suggested_wet:.3f}")
    table.add_row("dry", f"{suggested_dry:.3f}")
    table.add_row("repeat", "2" if duration < 15.0 else "1")
    table.add_row("target-lufs", "-18.0")
    table.add_row("target-peak-dbfs", "-1.0")
    table.add_row("normalize-stage", "post")
    table.add_row("shimmer", "off")
    table.add_row("duck", "off")
    console.print(table)


@app.command(name="presets")
def list_presets() -> None:
    """Print available presets."""
    names = preset_names()
    table = Table(title="Available Presets")
    table.add_column("Preset", style="green")
    for name in names:
        table.add_row(name)
    console.print(table)


@ir_app.command("gen")
def ir_gen(
    out_ir: Path = typer.Argument(..., resolve_path=True),
    out_format: IRFileFormat = typer.Option("auto", "--format"),
    mode: IRMode = typer.Option("hybrid", "--mode"),
    length: float = typer.Option(60.0, "--length", min=0.1),
    sr: int = typer.Option(48_000, "--sr", min=8_000),
    channels: int = typer.Option(2, "--channels", min=1),
    seed: int = typer.Option(0, "--seed"),
    rt60: float | None = typer.Option(None, "--rt60", min=0.1),
    rt60_low: float | None = typer.Option(None, "--rt60-low", min=0.1),
    rt60_high: float | None = typer.Option(None, "--rt60-high", min=0.1),
    damping: float = typer.Option(0.4, "--damping", min=0.0, max=1.0),
    lowcut: float | None = typer.Option(None, "--lowcut", min=10.0),
    highcut: float | None = typer.Option(None, "--highcut", min=10.0),
    tilt: float = typer.Option(0.0, "--tilt"),
    normalize: Literal["none", "peak", "rms"] = typer.Option("peak", "--normalize"),
    peak_dbfs: float = typer.Option(-1.0, "--peak-dbfs"),
    target_lufs: float | None = typer.Option(None, "--target-lufs"),
    true_peak: bool = typer.Option(True, "--true-peak/--sample-peak"),
    er_count: int = typer.Option(24, "--er-count", min=0),
    er_max_delay_ms: float = typer.Option(90.0, "--er-max-delay-ms", min=1.0),
    er_decay_shape: str = typer.Option("exp", "--er-decay-shape"),
    er_stereo_width: float = typer.Option(1.0, "--er-stereo-width", min=0.0, max=2.0),
    er_room: float = typer.Option(1.0, "--er-room", min=0.1),
    diffusion: float = typer.Option(0.5, "--diffusion", min=0.0, max=1.0),
    mod_depth_ms: float = typer.Option(1.5, "--mod-depth-ms", min=0.0),
    mod_rate_hz: float = typer.Option(0.12, "--mod-rate-hz", min=0.0),
    density: float = typer.Option(1.0, "--density", min=0.01),
    tuning: str = typer.Option("A4=440", "--tuning"),
    modal_count: int = typer.Option(48, "--modal-count", min=1),
    modal_q_min: float = typer.Option(5.0, "--modal-q-min", min=0.5),
    modal_q_max: float = typer.Option(60.0, "--modal-q-max", min=0.5),
    modal_spread_cents: float = typer.Option(5.0, "--modal-spread-cents", min=0.0),
    modal_low_hz: float = typer.Option(80.0, "--modal-low-hz", min=20.0),
    modal_high_hz: float = typer.Option(12_000.0, "--modal-high-hz", min=50.0),
    fdn_lines: int = typer.Option(8, "--fdn-lines", min=1),
    fdn_matrix: str = typer.Option(
        "hadamard",
        "--fdn-matrix",
        help=(
            "FDN matrix topology: hadamard, householder, random_orthogonal, "
            "circulant, elliptic, tv_unitary, or graph."
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
        min=0.1,
        help="Low-band RT60 target for multiband FDN decay shaping (seconds).",
    ),
    fdn_rt60_mid: float | None = typer.Option(
        None,
        "--fdn-rt60-mid",
        min=0.1,
        help="Mid-band RT60 target for multiband FDN decay shaping (seconds).",
    ),
    fdn_rt60_high: float | None = typer.Option(
        None,
        "--fdn-rt60-high",
        min=0.1,
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
        help=(
            "Feedback-link filter mode inside the FDN matrix path: "
            "none, lowpass, or highpass."
        ),
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
    fdn_stereo_inject: float = typer.Option(1.0, "--fdn-stereo-inject", min=0.0, max=1.0),
    f0: str | None = typer.Option(None, "--f0", help="e.g. 64, 64Hz, or 64 Hz"),
    analyze_input: Path | None = typer.Option(
        None,
        "--analyze-input",
        exists=True,
        readable=True,
        resolve_path=True,
        help="Input audio to estimate fundamentals/harmonics for IR tuning",
    ),
    harmonic_align_strength: float = typer.Option(
        0.75, "--harmonic-align-strength", min=0.0, max=1.0
    ),
    resonator: bool = typer.Option(
        False,
        "--resonator/--no-resonator",
        help="Enable Modalys-inspired physical modal-bank late-tail coloration.",
    ),
    resonator_mix: float = typer.Option(0.35, "--resonator-mix", min=0.0, max=1.0),
    resonator_modes: int = typer.Option(32, "--resonator-modes", min=1),
    resonator_q_min: float = typer.Option(8.0, "--resonator-q-min", min=0.5),
    resonator_q_max: float = typer.Option(90.0, "--resonator-q-max", min=0.5),
    resonator_low_hz: float = typer.Option(50.0, "--resonator-low-hz", min=20.0),
    resonator_high_hz: float = typer.Option(9000.0, "--resonator-high-hz", min=30.0),
    resonator_late_start_ms: float = typer.Option(80.0, "--resonator-late-start-ms", min=0.0),
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
    lucky: int | None = typer.Option(
        None,
        "--lucky",
        min=1,
        max=500,
        help=(
            "Generate N randomized IR files from one base setup. "
            "Outputs are written to --lucky-out-dir (or OUT_IR parent by default)."
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
        help="Optional deterministic seed for --lucky IR generation.",
    ),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    """Generate an IR file with deterministic caching."""
    _validate_ir_gen_call(
        out_ir=out_ir,
        out_format=out_format,
        rt60=rt60,
        rt60_low=rt60_low,
        rt60_high=rt60_high,
        modal_q_min=modal_q_min,
        modal_q_max=modal_q_max,
        modal_low_hz=modal_low_hz,
        modal_high_hz=modal_high_hz,
        resonator_q_min=resonator_q_min,
        resonator_q_max=resonator_q_max,
        resonator_low_hz=resonator_low_hz,
        resonator_high_hz=resonator_high_hz,
        fdn_lines=fdn_lines,
        fdn_matrix=fdn_matrix,
        fdn_tv_rate_hz=fdn_tv_rate_hz,
        fdn_tv_depth=fdn_tv_depth,
        fdn_sparse=fdn_sparse,
        fdn_sparse_degree=fdn_sparse_degree,
        fdn_cascade=fdn_cascade,
        fdn_cascade_mix=fdn_cascade_mix,
        fdn_cascade_delay_scale=fdn_cascade_delay_scale,
        fdn_cascade_rt60_ratio=fdn_cascade_rt60_ratio,
        fdn_rt60_low=fdn_rt60_low,
        fdn_rt60_mid=fdn_rt60_mid,
        fdn_rt60_high=fdn_rt60_high,
        fdn_rt60_tilt=fdn_rt60_tilt,
        fdn_tonal_correction_strength=fdn_tonal_correction_strength,
        fdn_xover_low_hz=fdn_xover_low_hz,
        fdn_xover_high_hz=fdn_xover_high_hz,
        fdn_link_filter=fdn_link_filter,
        fdn_link_filter_hz=fdn_link_filter_hz,
        fdn_link_filter_mix=fdn_link_filter_mix,
        fdn_graph_topology=fdn_graph_topology,
        fdn_graph_degree=fdn_graph_degree,
        room_size_macro=room_size_macro,
        clarity_macro=clarity_macro,
        warmth_macro=warmth_macro,
        envelopment_macro=envelopment_macro,
    )
    _validate_generic_lucky_call(lucky, lucky_out_dir)

    parsed_fdn_dfm_delays = _parse_delay_list_ms(
        fdn_dfm_delays_ms,
        option_name="--fdn-dfm-delays-ms",
    )
    if len(parsed_fdn_dfm_delays) not in {0, 1, fdn_lines}:
        msg = (
            "--fdn-dfm-delays-ms must include either 1 value or exactly "
            f"{fdn_lines} values."
        )
        raise typer.BadParameter(msg)

    f0_hz: float | None = None
    harmonic_targets_hz: tuple[float, ...] = ()

    if f0 is not None:
        try:
            f0_hz = parse_frequency_hz(f0)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    if analyze_input is not None:
        est_f0, harmonics = analyze_audio_for_tuning(analyze_input)
        if f0_hz is None:
            f0_hz = est_f0
        harmonic_targets_hz = tuple(harmonics)

    cfg = IRGenConfig(
        mode=mode,
        length=length,
        sr=sr,
        channels=channels,
        seed=seed,
        rt60=rt60,
        rt60_low=rt60_low,
        rt60_high=rt60_high,
        damping=damping,
        lowcut=lowcut,
        highcut=highcut,
        tilt=tilt,
        normalize=normalize,
        peak_dbfs=peak_dbfs,
        target_lufs=target_lufs,
        true_peak=true_peak,
        er_count=er_count,
        er_max_delay_ms=er_max_delay_ms,
        er_decay_shape=er_decay_shape,
        er_stereo_width=er_stereo_width,
        er_room=er_room,
        diffusion=diffusion,
        mod_depth_ms=mod_depth_ms,
        mod_rate_hz=mod_rate_hz,
        density=density,
        tuning=tuning,
        modal_count=modal_count,
        modal_q_min=modal_q_min,
        modal_q_max=modal_q_max,
        modal_spread_cents=modal_spread_cents,
        modal_low_hz=modal_low_hz,
        modal_high_hz=modal_high_hz,
        fdn_lines=fdn_lines,
        fdn_matrix=_normalize_fdn_matrix_name(fdn_matrix),
        fdn_tv_rate_hz=fdn_tv_rate_hz,
        fdn_tv_depth=fdn_tv_depth,
        fdn_tv_seed=seed,
        fdn_dfm_delays_ms=parsed_fdn_dfm_delays,
        fdn_sparse=fdn_sparse,
        fdn_sparse_degree=fdn_sparse_degree,
        fdn_cascade=fdn_cascade,
        fdn_cascade_mix=fdn_cascade_mix,
        fdn_cascade_delay_scale=fdn_cascade_delay_scale,
        fdn_cascade_rt60_ratio=fdn_cascade_rt60_ratio,
        fdn_rt60_low=fdn_rt60_low,
        fdn_rt60_mid=fdn_rt60_mid,
        fdn_rt60_high=fdn_rt60_high,
        fdn_rt60_tilt=fdn_rt60_tilt,
        fdn_tonal_correction_strength=fdn_tonal_correction_strength,
        fdn_xover_low_hz=fdn_xover_low_hz,
        fdn_xover_high_hz=fdn_xover_high_hz,
        fdn_link_filter=_normalize_fdn_link_filter_name(fdn_link_filter),
        fdn_link_filter_hz=fdn_link_filter_hz,
        fdn_link_filter_mix=fdn_link_filter_mix,
        fdn_graph_topology=_normalize_fdn_graph_topology_name(fdn_graph_topology),
        fdn_graph_degree=fdn_graph_degree,
        fdn_graph_seed=fdn_graph_seed,
        fdn_stereo_inject=fdn_stereo_inject,
        room_size_macro=room_size_macro,
        clarity_macro=clarity_macro,
        warmth_macro=warmth_macro,
        envelopment_macro=envelopment_macro,
        f0_hz=f0_hz,
        harmonic_targets_hz=harmonic_targets_hz,
        harmonic_align_strength=harmonic_align_strength,
        resonator=resonator,
        resonator_mix=resonator_mix,
        resonator_modes=resonator_modes,
        resonator_q_min=resonator_q_min,
        resonator_q_max=resonator_q_max,
        resonator_low_hz=resonator_low_hz,
        resonator_high_hz=resonator_high_hz,
        resonator_late_start_ms=resonator_late_start_ms,
    )

    resolved_out_ir = _resolve_ir_output_path(out_ir, out_format)

    if lucky is not None:
        out_dir = resolved_out_ir.parent if lucky_out_dir is None else lucky_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        seed_value = _resolve_lucky_seed(lucky_seed)

        rows: list[dict[str, str]] = []
        for idx in range(lucky):
            rng = np.random.default_rng(seed_value + idx)
            lucky_cfg = _build_lucky_ir_gen_config(cfg, rng=rng)
            lucky_out = (
                out_dir / f"{resolved_out_ir.stem}.lucky_{idx + 1:03d}{resolved_out_ir.suffix}"
            )
            try:
                audio, out_sr, meta, cache_path, cache_hit = generate_or_load_cached_ir(
                    lucky_cfg,
                    cache_dir=Path(cache_dir),
                )
                write_ir_artifacts(lucky_out, audio, out_sr, meta, silent=silent)
            except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
                raise typer.BadParameter(str(exc)) from exc

            rows.append(
                {
                    "index": str(idx + 1),
                    "out_ir": str(lucky_out),
                    "mode": lucky_cfg.mode,
                    "length_s": f"{float(lucky_cfg.length):.2f}",
                    "rt60": (
                        f"{float(lucky_cfg.rt60):.2f}"
                        if lucky_cfg.rt60 is not None
                        else (
                            f"{float(lucky_cfg.rt60_low or 0.0):.2f}-"
                            f"{float(lucky_cfg.rt60_high or 0.0):.2f}"
                        )
                    ),
                    "cache_hit": str(cache_hit),
                    "cache_path": str(cache_path),
                }
            )

        if not silent:
            table = Table(title=f"Lucky IR Generation Batch ({lucky} outputs)")
            table.add_column("#", style="cyan", justify="right")
            table.add_column("out_ir", style="white")
            table.add_column("mode", style="green")
            table.add_column("length_s", justify="right")
            table.add_column("rt60", justify="right")
            table.add_column("cache_hit", justify="right")
            for row in rows:
                table.add_row(
                    row["index"],
                    row["out_ir"],
                    row["mode"],
                    row["length_s"],
                    row["rt60"],
                    row["cache_hit"],
                )
            console.print(table)
        return

    try:
        audio, out_sr, meta, cache_path, cache_hit = generate_or_load_cached_ir(
            cfg,
            cache_dir=Path(cache_dir),
        )
        write_ir_artifacts(resolved_out_ir, audio, out_sr, meta, silent=silent)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if silent:
        return

    table = Table(title="IR Generation")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("mode", mode)
    table.add_row("out_ir", str(resolved_out_ir))
    table.add_row("format", out_format)
    table.add_row("cache_path", str(cache_path))
    table.add_row("cache_hit", str(cache_hit))
    table.add_row("duration_s", f"{audio.shape[0] / out_sr:.2f}")
    table.add_row("channels", str(audio.shape[1]))
    if f0_hz is not None:
        table.add_row("f0_hz", f"{f0_hz:.3f}")
    if analyze_input is not None:
        table.add_row("analyze_input", str(analyze_input))
        table.add_row("harmonics_detected", str(len(harmonic_targets_hz)))
    table.add_row("resonator", str(resonator))
    if resonator:
        table.add_row("resonator_mix", f"{resonator_mix:.3f}")
        table.add_row("resonator_modes", str(resonator_modes))
        table.add_row(
            "resonator_band_hz",
            f"{resonator_low_hz:.1f}-{resonator_high_hz:.1f}",
        )
        table.add_row("resonator_q", f"{resonator_q_min:.2f}-{resonator_q_max:.2f}")
        table.add_row("resonator_late_start_ms", f"{resonator_late_start_ms:.2f}")
    console.print(table)


@ir_app.command("analyze")
def ir_analyze(
    ir_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    json_out: Path | None = typer.Option(None, "--json-out", resolve_path=True),
) -> None:
    """Analyze an impulse response."""
    _validate_ir_analyze_call(ir_file, json_out)
    try:
        audio, sr = sf.read(str(ir_file), always_2d=True, dtype="float32")
        metrics = analyze_ir(np.asarray(audio, dtype=np.float32), int(sr))
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title=f"IR Analysis: {ir_file.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    for key in [
        "duration_seconds",
        "peak_dbfs",
        "rms_dbfs",
        "rt60_estimate_seconds",
        "early_late_ratio_db",
        "stereo_coherence",
    ]:
        value = metrics.get(key)
        if isinstance(value, float):
            table.add_row(key, f"{value:.6f}")
    decay_points = metrics.get("decay_curve_db", [])
    point_count = len(decay_points) if isinstance(decay_points, list) else 0
    table.add_row("decay_curve_points", str(point_count))
    console.print(table)

    if json_out is not None:
        payload = {"file": str(ir_file), "sample_rate": int(sr), "metrics": metrics}
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@ir_app.command("process")
def ir_process(
    in_ir: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_ir: Path = typer.Argument(..., resolve_path=True),
    damping: float = typer.Option(0.4, "--damping", min=0.0, max=1.0),
    lowcut: float | None = typer.Option(None, "--lowcut", min=10.0),
    highcut: float | None = typer.Option(None, "--highcut", min=10.0),
    tilt: float = typer.Option(0.0, "--tilt"),
    normalize: Literal["none", "peak", "rms"] = typer.Option("peak", "--normalize"),
    peak_dbfs: float = typer.Option(-1.0, "--peak-dbfs"),
    target_lufs: float | None = typer.Option(None, "--target-lufs"),
    true_peak: bool = typer.Option(True, "--true-peak/--sample-peak"),
    lucky: int | None = typer.Option(
        None,
        "--lucky",
        min=1,
        max=500,
        help=(
            "Generate N randomized processed IR files from one input IR. "
            "Outputs are written to --lucky-out-dir (or OUT_IR parent by default)."
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
        help="Optional deterministic seed for --lucky IR processing.",
    ),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    """Process an existing IR through shaping/targeting chain."""
    _validate_ir_process_call(in_ir, out_ir)
    _validate_generic_lucky_call(lucky, lucky_out_dir)
    try:
        audio, sr = sf.read(str(in_ir), always_2d=True, dtype="float32")
        base_audio = np.asarray(audio, dtype=np.float32)
        sr_i = int(sr)
        if lucky is None:
            processed = apply_ir_shaping(
                base_audio,
                sr=sr_i,
                damping=damping,
                lowcut=lowcut,
                highcut=highcut,
                tilt=tilt,
                normalize=normalize,
                peak_dbfs=peak_dbfs,
                target_lufs=target_lufs,
                use_true_peak=true_peak,
            )

            meta = {"source": str(in_ir), "metrics": analyze_ir(processed, sr_i)}
            write_ir_artifacts(out_ir, processed, sr_i, meta, silent=silent)
            return

        out_dir = out_ir.parent if lucky_out_dir is None else lucky_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        seed_value = _resolve_lucky_seed(lucky_seed)

        rows: list[dict[str, str]] = []
        for idx in range(lucky):
            rng = np.random.default_rng(seed_value + idx)
            cfg = _build_lucky_ir_process_config(
                damping=damping,
                lowcut=lowcut,
                highcut=highcut,
                tilt=tilt,
                normalize=normalize,
                peak_dbfs=peak_dbfs,
                target_lufs=target_lufs,
                true_peak=true_peak,
                rng=rng,
                sr=sr_i,
            )
            lucky_out = out_dir / f"{out_ir.stem}.lucky_{idx + 1:03d}{out_ir.suffix}"
            processed = apply_ir_shaping(
                base_audio,
                sr=sr_i,
                damping=cfg["damping"],
                lowcut=cfg["lowcut"],
                highcut=cfg["highcut"],
                tilt=cfg["tilt"],
                normalize=cfg["normalize"],
                peak_dbfs=cfg["peak_dbfs"],
                target_lufs=cfg["target_lufs"],
                use_true_peak=cfg["true_peak"],
            )

            meta = {
                "source": str(in_ir),
                "lucky": {"index": idx + 1, **cfg},
                "metrics": analyze_ir(processed, sr_i),
            }
            write_ir_artifacts(lucky_out, processed, sr_i, meta, silent=silent)
            rows.append(
                {
                    "index": str(idx + 1),
                    "out_ir": str(lucky_out),
                    "normalize": cfg["normalize"],
                    "tilt": f"{float(cfg['tilt']):.2f}",
                    "damping": f"{float(cfg['damping']):.2f}",
                    "target_lufs": (
                        f"{float(cfg['target_lufs']):.2f}"
                        if cfg["target_lufs"] is not None
                        else "none"
                    ),
                }
            )

        if not silent:
            table = Table(title=f"Lucky IR Process Batch ({lucky} outputs)")
            table.add_column("#", style="cyan", justify="right")
            table.add_column("out_ir", style="white")
            table.add_column("normalize", style="green")
            table.add_column("tilt", justify="right")
            table.add_column("damping", justify="right")
            table.add_column("target_lufs", justify="right")
            for row in rows:
                table.add_row(
                    row["index"],
                    row["out_ir"],
                    row["normalize"],
                    row["tilt"],
                    row["damping"],
                    row["target_lufs"],
                )
            console.print(table)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@ir_app.command("fit")
def ir_fit(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_ir: Path = typer.Argument(..., resolve_path=True),
    top_k: int = typer.Option(3, "--top-k", min=1),
    base_mode: IRMode = typer.Option("hybrid", "--base-mode"),
    length: float = typer.Option(60.0, "--length", min=0.1),
    seed: int = typer.Option(0, "--seed"),
    candidate_pool: int = typer.Option(12, "--candidate-pool", min=1),
    fit_workers: int = typer.Option(0, "--fit-workers", min=0, help="0 = auto"),
    analyze_tuning: bool = typer.Option(True, "--analyze-tuning/--no-analyze-tuning"),
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
) -> None:
    """Analyze source audio, score candidate IRs, and write top-k results."""
    _validate_output_audio_path(out_ir, "auto")
    try:
        audio, sr = read_audio(str(infile))
        analyzer = AudioAnalyzer()
        metrics = analyzer.analyze(audio, sr)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    pool_size = max(top_k, candidate_pool)
    target_profile = derive_ir_fit_target(metrics, sr)

    f0_hz: float | None = None
    harmonics: tuple[float, ...] = ()
    if analyze_tuning:
        try:
            f0_est, harmonic_est = analyze_audio_for_tuning(infile, max_harmonics=12)
            f0_hz = f0_est
            harmonics = tuple(harmonic_est)
        except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError):
            f0_hz = None
            harmonics = ()

    candidates = build_ir_fit_candidates(
        base_mode=base_mode,
        length=length,
        sr=sr,
        channels=max(1, min(2, audio.shape[1])),
        seed=seed,
        pool_size=pool_size,
        target=target_profile,
        f0_hz=f0_hz,
        harmonic_targets_hz=harmonics,
    )

    cache_root = Path(cache_dir)
    scored = _score_fit_candidates(
        candidates=candidates,
        target=target_profile,
        cache_dir=cache_root,
        fit_workers=fit_workers,
    )

    selected = sorted(
        scored,
        key=lambda item: item.score.score,
        reverse=True,
    )[:top_k]

    created: list[str] = []
    for rank, item in enumerate(selected, start=1):
        target_path = (
            out_ir
            if top_k == 1
            else out_ir.with_name(f"{out_ir.stem}_{rank:02d}{out_ir.suffix}")
        )
        meta = dict(item.meta)
        meta["fit"] = {
            "rank": rank,
            "score": item.score.score,
            "strategy": item.candidate.strategy,
            "target": asdict(target_profile),
            "errors": asdict(item.score),
            "detail_metrics": item.detail_metrics,
        }
        cached_audio, _ = sf.read(str(item.cache_path), always_2d=True, dtype="float32")
        write_ir_artifacts(
            target_path,
            np.asarray(cached_audio, dtype=np.float32),
            item.sr,
            meta,
            silent=False,
        )
        created.append(str(target_path))

    table = Table(title="IR Fit")
    table.add_column("Field", style="green")
    table.add_column("Value", style="white")
    table.add_row("input", str(infile))
    table.add_row("top_k", str(top_k))
    table.add_row("candidate_pool", str(pool_size))
    table.add_row("target_rt60", f"{target_profile.rt60_seconds:.2f}")
    table.add_row("target_early_late_db", f"{target_profile.early_late_ratio_db:.2f}")
    table.add_row("target_coherence", f"{target_profile.stereo_coherence:.3f}")
    if f0_hz is not None:
        table.add_row("detected_f0_hz", f"{f0_hz:.3f}")
    if selected:
        table.add_row("best_score", f"{selected[0].score.score:.5f}")
        table.add_row("best_strategy", selected[0].candidate.strategy)
    table.add_row("outputs", "\n".join(created))
    console.print(table)


@cache_app.command("info")
def cache_info(
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
) -> None:
    """Show cache statistics."""
    root = Path(cache_dir)
    if root.exists() and not root.is_dir():
        msg = f"Cache path is not a directory: {root}"
        raise typer.BadParameter(msg)
    wavs = sorted(root.glob("*.wav"))
    metas = sorted(root.glob("*.meta.json"))
    total_bytes = (
        sum(path.stat().st_size for path in root.glob("*") if path.is_file())
        if root.exists()
        else 0
    )

    table = Table(title="Cache Info")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("cache_dir", str(root))
    table.add_row("wav_files", str(len(wavs)))
    table.add_row("meta_files", str(len(metas)))
    table.add_row("size_mb", f"{total_bytes / (1024 * 1024):.3f}")
    console.print(table)


@cache_app.command("clear")
def cache_clear(
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
) -> None:
    """Clear IR cache directory."""
    root = Path(cache_dir)
    if root.exists() and not root.is_dir():
        msg = f"Cache path is not a directory: {root}"
        raise typer.BadParameter(msg)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    console.print(f"Cleared cache: {root}")


@batch_app.command("template")
def batch_template() -> None:
    """Print a batch manifest template as JSON."""
    template = {
        "version": "0.5",
        "jobs": [
            {
                "infile": "input.wav",
                "outfile": "output.wav",
                "options": {
                    "engine": "auto",
                    "rt60": 60.0,
                    "wet": 0.8,
                    "dry": 0.2,
                    "repeat": 1,
                },
            }
        ],
    }
    typer.echo(json.dumps(template, indent=2))


@batch_app.command("render")
def batch_render(
    manifest: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    jobs: int = typer.Option(0, "--jobs", min=0, help="0 = auto"),
    schedule: BatchSchedulePolicy = typer.Option("longest-first", "--schedule"),
    retries: int = typer.Option(0, "--retries", min=0),
    continue_on_error: bool = typer.Option(False, "--continue-on-error/--fail-fast"),
    checkpoint_file: Path | None = typer.Option(
        None,
        "--checkpoint-file",
        resolve_path=True,
        help="Optional checkpoint file used to persist per-job completion state.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from --checkpoint-file and skip already completed jobs.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    lucky: int | None = typer.Option(
        None,
        "--lucky",
        min=1,
        max=500,
        help=(
            "For each manifest job, generate N wild random render variants. "
            "Outputs are written to --lucky-out-dir (or each job OUTFILE parent by default)."
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
        help="Optional deterministic seed for --lucky batch generation.",
    ),
) -> None:
    """Render jobs from manifest.json."""
    if resume and checkpoint_file is None:
        msg = "--resume requires --checkpoint-file."
        raise typer.BadParameter(msg)
    _validate_generic_lucky_call(lucky, lucky_out_dir)
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON manifest: {exc}"
        raise typer.BadParameter(msg) from exc
    if not isinstance(payload, dict) or "jobs" not in payload:
        raise typer.BadParameter("Manifest must contain a top-level 'jobs' array")

    job_list = payload["jobs"]
    if not isinstance(job_list, list):
        raise typer.BadParameter("jobs must be a list")

    prepared_jobs: list[BatchJobSpec] = []
    prepared_index = 1
    lucky_seed_value = _resolve_lucky_seed(lucky_seed) if lucky is not None else 0

    for idx, job in enumerate(job_list, start=1):
        if not isinstance(job, dict):
            raise typer.BadParameter(f"jobs[{idx - 1}] must be an object")
        infile = Path(str(job.get("infile", "")))
        outfile = Path(str(job.get("outfile", "")))
        options = job.get("options", {})
        if not isinstance(options, dict):
            raise typer.BadParameter(f"jobs[{idx - 1}].options must be an object")

        try:
            render_config = _render_config_from_options(options)
        except (TypeError, ValueError) as exc:
            msg = f"jobs[{idx - 1}] has invalid options: {exc}"
            raise typer.BadParameter(msg) from exc
        _validate_batch_job_paths(infile, outfile, idx)
        if lucky is None:
            prepared_jobs.append(
                BatchJobSpec(
                    index=prepared_index,
                    infile=infile,
                    outfile=outfile,
                    config=render_config,
                    estimated_cost=estimate_job_cost(infile, render_config),
                )
            )
            prepared_index += 1
            continue

        try:
            info = sf.info(str(infile))
            duration_seconds = (
                float(info.frames) / float(info.samplerate) if info.samplerate > 0 else 0.0
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            msg = f"jobs[{idx - 1}] failed to inspect infile: {exc}"
            raise typer.BadParameter(msg) from exc

        out_dir = outfile.parent if lucky_out_dir is None else lucky_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for variant_idx in range(lucky):
            rng = np.random.default_rng(lucky_seed_value + ((idx - 1) * lucky) + variant_idx)
            lucky_config = _build_lucky_config(
                base=render_config,
                rng=rng,
                input_duration_seconds=duration_seconds,
            )
            lucky_config.progress = False
            lucky_config.analysis_out = None
            lucky_config.frames_out = None
            lucky_out = out_dir / f"{outfile.stem}.lucky_{variant_idx + 1:03d}{outfile.suffix}"
            prepared_jobs.append(
                BatchJobSpec(
                    index=prepared_index,
                    infile=infile,
                    outfile=lucky_out,
                    config=lucky_config,
                    estimated_cost=estimate_job_cost(infile, lucky_config),
                )
            )
            prepared_index += 1

    if not prepared_jobs:
        raise typer.BadParameter("Manifest has no jobs to render")

    checkpoint_payload: dict[str, Any] | None = None
    resumed_outfiles: set[str] = set()
    if checkpoint_file is not None:
        if resume:
            checkpoint_payload = _load_batch_checkpoint(checkpoint_file)
            resumed_outfiles = _checkpoint_success_outfiles(checkpoint_payload)
            if len(resumed_outfiles) > 0:
                before = len(prepared_jobs)
                prepared_jobs = [
                    job
                    for job in prepared_jobs
                    if str(job.outfile.resolve()) not in resumed_outfiles
                ]
                skipped = before - len(prepared_jobs)
                if skipped > 0:
                    console.print(f"resuming batch: skipped {skipped} completed jobs")
        else:
            checkpoint_payload = {
                "version": "0.5",
                "manifest": str(manifest.resolve()),
                "results": [],
            }

    if len(prepared_jobs) == 0:
        console.print("batch complete: no pending jobs after resume filtering")
        return

    if dry_run:
        ordered = order_jobs(prepared_jobs, schedule)
        for job in ordered:
            console.print(
                "[dry-run] job "
                f"{job.index}: {job.infile} -> {job.outfile} "
                f"(cost={job.estimated_cost:.2f}, schedule={schedule})"
            )
        return

    max_workers = int(os.cpu_count() or 1) if jobs == 0 else int(jobs)
    max_workers = max(1, min(max_workers, len(prepared_jobs)))

    def runner(job: BatchJobSpec) -> None:
        run_render_pipeline(infile=job.infile, outfile=job.outfile, config=job.config)

    def on_result(result: BatchJobResult) -> None:
        if checkpoint_file is not None and checkpoint_payload is not None:
            checkpoint_payload.setdefault("results", [])
            assert isinstance(checkpoint_payload["results"], list)
            checkpoint_payload["results"].append(
                {
                    "index": int(result.index),
                    "outfile": str(result.outfile.resolve()),
                    "success": bool(result.success),
                    "attempts": int(result.attempts),
                    "duration_seconds": float(result.duration_seconds),
                    "estimated_cost": float(result.estimated_cost),
                    "error": result.error,
                }
            )
            _write_json_atomic(checkpoint_file, checkpoint_payload)
        if result.success:
            console.print(
                f"rendered job {result.index}: {result.outfile} "
                f"(attempts={result.attempts}, {result.duration_seconds:.2f}s)"
            )
        else:
            console.print(
                f"failed job {result.index}: {result.outfile} "
                f"(attempts={result.attempts}) {result.error}"
            )

    try:
        run_parallel_batch(
            jobs=prepared_jobs,
            max_workers=max_workers,
            schedule=schedule,
            retries=retries,
            continue_on_error=continue_on_error,
            runner=runner,
            on_result=on_result,
        )
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _render_config_from_options(options: dict[str, Any]) -> RenderConfig:
    """Build ``RenderConfig`` from manifest options with safe field filtering."""
    fields = RenderConfig.__dataclass_fields__.keys()
    filtered = {key: value for key, value in options.items() if key in fields}

    pre_delay_note = filtered.get("pre_delay_note")
    bpm = filtered.get("bpm")
    if isinstance(pre_delay_note, str):
        fallback_ms = float(filtered.get("pre_delay_ms", 20.0))
        resolved_bpm = float(bpm) if isinstance(bpm, (float, int)) else None
        filtered["pre_delay_ms"] = parse_pre_delay_ms(pre_delay_note, resolved_bpm, fallback_ms)

    return RenderConfig(**filtered)


def _load_batch_checkpoint(path: Path) -> dict[str, Any]:
    """Load checkpoint payload with graceful fallback when missing/invalid."""
    if not path.exists():
        return {"version": "0.5", "results": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": "0.5", "results": []}
    if not isinstance(payload, dict):
        return {"version": "0.5", "results": []}
    results = payload.get("results")
    if not isinstance(results, list):
        payload["results"] = []
    return payload


def _checkpoint_success_outfiles(payload: dict[str, Any]) -> set[str]:
    """Return canonical outfile paths marked successful in checkpoint payload."""
    completed: set[str] = set()
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        return completed
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not bool(row.get("success", False)):
            continue
        outfile = row.get("outfile")
        if not isinstance(outfile, str):
            continue
        out_path = Path(outfile)
        if out_path.exists():
            completed.add(str(out_path.resolve()))
    return completed


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


class _ScoredFitCandidate:
    """Internal transport object for IR-fit ranking results."""

    def __init__(
        self,
        *,
        candidate: IRFitCandidate,
        score: IRFitScore,
        detail_metrics: dict[str, float],
        sr: int,
        meta: dict[str, Any],
        cache_path: Path,
    ) -> None:
        self.candidate = candidate
        self.score = score
        self.detail_metrics = detail_metrics
        self.sr = int(sr)
        self.meta = meta
        self.cache_path = cache_path


def _score_fit_candidates(
    *,
    candidates: list[IRFitCandidate],
    target: IRFitTarget,
    cache_dir: Path,
    fit_workers: int,
) -> list[_ScoredFitCandidate]:
    """Evaluate IR-fit candidates serially or in parallel."""
    worker_count = int(os.cpu_count() or 1) if fit_workers == 0 else fit_workers
    worker_count = max(1, min(worker_count, len(candidates)))

    def evaluate(candidate: IRFitCandidate) -> _ScoredFitCandidate:
        audio, sr, meta, cache_path, _ = generate_or_load_cached_ir(
            candidate.config,
            cache_dir=cache_dir,
        )
        score, detail_metrics = score_ir_candidate(ir_audio=audio, sr=sr, target=target)
        return _ScoredFitCandidate(
            candidate=candidate,
            score=score,
            detail_metrics=detail_metrics,
            sr=sr,
            meta=meta,
            cache_path=cache_path,
        )

    if worker_count == 1:
        return [evaluate(candidate) for candidate in candidates]

    scored: list[_ScoredFitCandidate] = []
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="verbx-fit") as pool:
        futures: list[Future[_ScoredFitCandidate]] = [
            pool.submit(evaluate, candidate) for candidate in candidates
        ]
        for fut in as_completed(futures):
            scored.append(fut.result())
    return scored


def _print_render_summary(report: dict[str, Any]) -> None:
    """Print the standard single-render summary table."""
    table = Table(title="Render Summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("requested_engine", str(report.get("effective", {}).get("engine_requested", "")))
    table.add_row("engine", str(report.get("engine", "unknown")))
    table.add_row("requested_device", str(report.get("effective", {}).get("device_requested", "")))
    table.add_row("device", str(report.get("effective", {}).get("device_resolved", "")))
    table.add_row(
        "device_platform",
        str(report.get("effective", {}).get("device_platform_resolved", "")),
    )
    table.add_row("compute_backend", str(report.get("effective", {}).get("compute_backend", "")))
    table.add_row("ir_used", str(report.get("effective", {}).get("ir_used")))
    table.add_row(
        "self_convolve",
        str(report.get("effective", {}).get("self_convolve", False)),
    )
    table.add_row("sample_rate", str(report.get("sample_rate", "")))
    table.add_row("channels", str(report.get("channels", "")))
    table.add_row("input_samples", str(report.get("input_samples", "")))
    table.add_row("output_samples", str(report.get("output_samples", "")))
    table.add_row(
        "tail_padding_s",
        str(report.get("effective", {}).get("tail_padding_seconds", "")),
    )
    table.add_row("beast_mode", str(report.get("effective", {}).get("beast_mode", 1)))
    table.add_row("out_subtype", str(report.get("effective", {}).get("output_subtype", "")))
    table.add_row("output_peak_norm", str(report.get("effective", {}).get("output_peak_norm", "")))
    table.add_row(
        "output_peak_target_dbfs",
        str(report.get("effective", {}).get("output_peak_target_dbfs", "")),
    )
    table.add_row("mod_target", str(report.get("config", {}).get("mod_target", "none")))
    mod_sources = report.get("config", {}).get("mod_sources", ())
    mod_routes = report.get("config", {}).get("mod_routes", ())
    if isinstance(mod_sources, list):
        table.add_row("mod_sources", str(len(mod_sources)))
    elif isinstance(mod_sources, tuple):
        table.add_row("mod_sources", str(len(mod_sources)))
    else:
        table.add_row("mod_sources", "0")
    if isinstance(mod_routes, list):
        table.add_row("mod_routes", str(len(mod_routes)))
    elif isinstance(mod_routes, tuple):
        table.add_row("mod_routes", str(len(mod_routes)))
    else:
        table.add_row("mod_routes", "0")
    table.add_row("streaming_mode", str(report.get("effective", {}).get("streaming_mode", "")))
    automation_payload = report.get("effective", {}).get("automation")
    if isinstance(automation_payload, dict):
        targets = automation_payload.get("targets", [])
        if isinstance(targets, list):
            table.add_row("automation_targets", ",".join(str(t) for t in targets))
        table.add_row("automation_mode", str(automation_payload.get("mode", "")))
    config_report = report.get("config", {})
    if isinstance(config_report, dict):
        for key in (
            "rt60",
            "pre_delay_ms",
            "beast_mode",
            "wet",
            "dry",
            "repeat",
            "ir_matrix_layout",
            "self_convolve",
            "damping",
            "width",
            "allpass_stages",
            "allpass_gain",
            "fdn_lines",
            "block_size",
        ):
            if key in config_report:
                table.add_row(key, str(config_report[key]))
        for key in ("allpass_gains", "allpass_delays_ms", "comb_delays_ms"):
            if key in config_report and isinstance(config_report[key], (list, tuple)):
                table.add_row(f"{key}_count", str(len(config_report[key])))
    table.add_row("analysis_json", str(report.get("analysis_path", "")))
    if "frames_path" in report:
        table.add_row("frames_csv", str(report.get("frames_path")))
    if "ir_runtime" in report:
        runtime = report["ir_runtime"]
        if isinstance(runtime, dict):
            table.add_row("ir_runtime_path", str(runtime.get("ir_path", "")))
            table.add_row("ir_cache_hit", str(runtime.get("cache_hit", False)))
    console.print(table)


def _build_lucky_config(
    base: RenderConfig,
    rng: np.random.Generator,
    input_duration_seconds: float,
) -> RenderConfig:
    """Build one randomized high-intensity render config for ``--lucky`` mode."""
    cfg = RenderConfig(**asdict(base))

    cfg.engine = cast(EngineName, rng.choice(np.array(["algo", "conv", "auto"], dtype=object)))
    cfg.beast_mode = int(rng.integers(1, 7))
    cfg.rt60 = float(rng.uniform(0.8, 12.0))
    cfg.pre_delay_ms = float(rng.uniform(0.0, 500.0))
    cfg.damping = float(rng.uniform(0.0, 1.0))
    cfg.width = float(rng.uniform(0.0, 2.0))
    cfg.mod_depth_ms = float(rng.uniform(0.0, 25.0))
    cfg.mod_rate_hz = float(rng.uniform(0.02, 2.0))
    cfg.wet = float(rng.uniform(0.55, 1.0))
    cfg.dry = float(rng.uniform(0.0, 0.6))
    cfg.repeat = int(rng.integers(1, 4))
    cfg.partition_size = int(rng.choice(np.array([4096, 8192, 16384, 32768], dtype=np.int32)))
    cfg.block_size = int(rng.choice(np.array([1024, 2048, 4096, 8192], dtype=np.int32)))
    cfg.progress = False
    cfg.frames_out = None
    cfg.analysis_out = None
    cfg.automation_trace_out = None

    cfg.freeze = bool(input_duration_seconds > 1.5 and rng.random() < 0.33)
    if cfg.freeze:
        max_start = max(0.0, input_duration_seconds - 0.25)
        start = float(rng.uniform(0.0, max_start))
        max_len = max(0.2, min(4.0, input_duration_seconds * 0.45))
        end = float(min(input_duration_seconds, start + rng.uniform(0.2, max_len)))
        if end <= start:
            end = min(input_duration_seconds, start + 0.2)
        cfg.start = start
        cfg.end = end
    else:
        cfg.start = None
        cfg.end = None

    cfg.target_lufs = float(rng.uniform(-30.0, -10.0)) if rng.random() < 0.45 else None
    cfg.target_peak_dbfs = float(rng.uniform(-9.0, -0.6)) if rng.random() < 0.7 else None
    cfg.use_true_peak = bool(rng.random() < 0.8)
    cfg.normalize_stage = cast(
        NormalizeStage,
        rng.choice(np.array(["none", "post", "per-pass"], dtype=object)),
    )
    if cfg.normalize_stage == "per-pass":
        cfg.repeat_target_lufs = (
            float(rng.uniform(-30.0, -12.0)) if rng.random() < 0.55 else cfg.target_lufs
        )
        cfg.repeat_target_peak_dbfs = (
            float(rng.uniform(-10.0, -0.8)) if rng.random() < 0.6 else cfg.target_peak_dbfs
        )
    else:
        cfg.repeat_target_lufs = None
        cfg.repeat_target_peak_dbfs = None

    cfg.shimmer = bool(rng.random() < 0.42)
    cfg.shimmer_semitones = float(rng.choice(np.array([7.0, 12.0, 19.0, 24.0], dtype=np.float32)))
    cfg.shimmer_mix = float(rng.uniform(0.05, 0.85))
    cfg.shimmer_feedback = float(rng.uniform(0.05, 0.95))
    cfg.shimmer_lowcut = float(rng.uniform(80.0, 600.0))
    cfg.shimmer_highcut = float(rng.uniform(2_000.0, 16_000.0))

    cfg.duck = bool(rng.random() < 0.45)
    cfg.duck_attack = float(rng.uniform(2.0, 120.0))
    cfg.duck_release = float(rng.uniform(80.0, 1200.0))
    cfg.bloom = float(rng.uniform(0.0, 6.0))
    cfg.tilt = float(rng.uniform(-8.0, 8.0))

    if rng.random() < 0.6:
        cfg.lowcut = float(rng.uniform(20.0, 500.0))
    else:
        cfg.lowcut = None
    if rng.random() < 0.8:
        min_high = (cfg.lowcut + 300.0) if cfg.lowcut is not None else 800.0
        cfg.highcut = float(rng.uniform(min_high, 20_000.0))
    else:
        cfg.highcut = None

    if rng.random() < 0.4:
        cfg.fdn_rt60_low = float(rng.uniform(6.0, 70.0))
        cfg.fdn_rt60_mid = float(rng.uniform(3.0, 40.0))
        cfg.fdn_rt60_high = float(rng.uniform(0.8, 20.0))
        cfg.fdn_tonal_correction_strength = float(rng.uniform(0.05, 0.9))
        cfg.fdn_xover_low_hz = float(rng.uniform(100.0, 700.0))
        cfg.fdn_xover_high_hz = float(
            rng.uniform(max(cfg.fdn_xover_low_hz + 250.0, 1_000.0), 8_000.0)
        )
    else:
        cfg.fdn_rt60_low = None
        cfg.fdn_rt60_mid = None
        cfg.fdn_rt60_high = None
        cfg.fdn_tonal_correction_strength = 0.0
        cfg.fdn_xover_low_hz = 250.0
        cfg.fdn_xover_high_hz = 4_000.0
    cfg.fdn_cascade = bool(rng.random() < 0.4)
    if cfg.fdn_cascade:
        cfg.fdn_cascade_mix = float(rng.uniform(0.15, 0.85))
        cfg.fdn_cascade_delay_scale = float(rng.uniform(0.25, 0.85))
        cfg.fdn_cascade_rt60_ratio = float(rng.uniform(0.2, 0.95))
    else:
        cfg.fdn_cascade_mix = 0.35
        cfg.fdn_cascade_delay_scale = 0.5
        cfg.fdn_cascade_rt60_ratio = 0.55
    if rng.random() < 0.45:
        cfg.fdn_link_filter = cast(
            str,
            rng.choice(np.array(["lowpass", "highpass"], dtype=object)),
        )
        cfg.fdn_link_filter_hz = float(rng.uniform(80.0, 12_000.0))
        cfg.fdn_link_filter_mix = float(rng.uniform(0.2, 1.0))
    else:
        cfg.fdn_link_filter = "none"
        cfg.fdn_link_filter_hz = 2_500.0
        cfg.fdn_link_filter_mix = 1.0
    cfg.algo_decorrelation_front = float(rng.uniform(0.0, 0.8)) if rng.random() < 0.45 else 0.0
    cfg.algo_decorrelation_rear = float(rng.uniform(0.0, 0.9)) if rng.random() < 0.45 else 0.0
    cfg.algo_decorrelation_top = float(rng.uniform(0.0, 0.9)) if rng.random() < 0.3 else 0.0
    cfg.ir_route_map = cast(
        str,
        rng.choice(np.array(["auto", "diagonal", "broadcast", "full"], dtype=object)),
    )
    if cfg.ir_route_map == "auto":
        cfg.conv_route_start = None
        cfg.conv_route_end = None
    elif rng.random() < 0.35:
        cfg.conv_route_start = "left"
        cfg.conv_route_end = "right"
    else:
        cfg.conv_route_start = None
        cfg.conv_route_end = None
    cfg.conv_route_curve = cast(
        str,
        rng.choice(np.array(["linear", "equal-power"], dtype=object)),
    )

    # Ensure convolution path has a valid IR source whenever selected.
    if cfg.engine in {"conv", "auto"} and rng.random() < 0.75:
        mode = int(rng.integers(0, 3))
        if mode == 0 and base.ir is not None:
            cfg.ir = base.ir
            cfg.ir_gen = False
            cfg.self_convolve = False
        elif mode == 1:
            cfg.self_convolve = True
            cfg.ir = None
            cfg.ir_gen = False
        else:
            cfg.ir_gen = True
            cfg.ir = None
            cfg.self_convolve = False
            cfg.ir_gen_mode = cast(
                IRMode,
                rng.choice(np.array(["hybrid", "fdn", "modal", "stochastic"], dtype=object)),
            )
            cfg.ir_gen_length = float(rng.uniform(5.0, 60.0))
            cfg.ir_gen_seed = int(rng.integers(0, 2_147_483_647))
    else:
        cfg.ir = None
        cfg.ir_gen = False
        cfg.self_convolve = False
        if cfg.engine == "conv":
            cfg.engine = "algo"
        cfg.ir_route_map = "auto"
        cfg.conv_route_start = None
        cfg.conv_route_end = None

    cfg.tail_limit = float(rng.uniform(2.0, 40.0)) if rng.random() < 0.4 else None
    return cfg


def _build_lucky_ir_gen_config(
    base: IRGenConfig,
    rng: np.random.Generator,
) -> IRGenConfig:
    """Build one randomized IR generation config for ``ir gen --lucky``."""
    cfg = IRGenConfig(**asdict(base))

    cfg.mode = cast(
        IRMode,
        rng.choice(np.array(["hybrid", "fdn", "stochastic", "modal"], dtype=object)),
    )
    cfg.seed = int(rng.integers(0, 2_147_483_647))
    cfg.length = float(np.clip(base.length * rng.uniform(0.5, 2.5), 0.1, 120.0))
    cfg.rt60 = float(np.clip((base.rt60 or 12.0) * rng.uniform(0.4, 2.5), 0.1, 180.0))
    cfg.rt60_low = None
    cfg.rt60_high = None

    cfg.damping = float(rng.uniform(0.0, 1.0))
    cfg.diffusion = float(rng.uniform(0.05, 1.0))
    cfg.density = float(rng.uniform(0.1, 2.2))
    cfg.mod_depth_ms = float(rng.uniform(0.0, 18.0))
    cfg.mod_rate_hz = float(rng.uniform(0.02, 1.2))

    cfg.lowcut = float(rng.uniform(20.0, 500.0)) if rng.random() < 0.6 else None
    if rng.random() < 0.8:
        min_high = (cfg.lowcut + 300.0) if cfg.lowcut is not None else 800.0
        cfg.highcut = float(rng.uniform(min_high, (cfg.sr * 0.48)))
    else:
        cfg.highcut = None
    cfg.tilt = float(rng.uniform(-8.0, 8.0))

    cfg.normalize = cast(
        Literal["none", "peak", "rms"],
        rng.choice(np.array(["none", "peak", "rms"], dtype=object)),
    )
    cfg.peak_dbfs = float(rng.uniform(-12.0, -0.5))
    cfg.target_lufs = float(rng.uniform(-32.0, -12.0)) if rng.random() < 0.45 else None
    cfg.true_peak = bool(rng.random() < 0.7)

    cfg.er_count = int(rng.integers(0, 96))
    cfg.er_max_delay_ms = float(rng.uniform(5.0, 180.0))
    cfg.er_decay_shape = cast(str, rng.choice(np.array(["exp", "linear", "sqrt"], dtype=object)))
    cfg.er_stereo_width = float(rng.uniform(0.0, 2.0))
    cfg.er_room = float(rng.uniform(0.1, 3.0))

    cfg.modal_count = int(rng.integers(8, 128))
    cfg.modal_q_min = float(rng.uniform(0.8, 20.0))
    cfg.modal_q_max = float(rng.uniform(max(cfg.modal_q_min + 0.5, 5.0), 120.0))
    cfg.modal_spread_cents = float(rng.uniform(0.0, 40.0))
    cfg.modal_low_hz = float(rng.uniform(30.0, 400.0))
    cfg.modal_high_hz = float(rng.uniform(max(cfg.modal_low_hz + 200.0, 1200.0), cfg.sr * 0.48))

    cfg.fdn_lines = int(rng.choice(np.array([4, 6, 8, 10, 12, 16], dtype=np.int32)))
    cfg.fdn_matrix = cast(
        str,
        rng.choice(
            np.array(
                [
                    "hadamard",
                    "householder",
                    "random_orthogonal",
                    "circulant",
                    "elliptic",
                    "tv_unitary",
                    "graph",
                ],
                dtype=object,
            )
        ),
    )
    if cfg.fdn_matrix == "tv_unitary":
        cfg.fdn_tv_rate_hz = float(rng.uniform(0.03, 0.45))
        cfg.fdn_tv_depth = float(rng.uniform(0.1, 0.8))
    else:
        cfg.fdn_tv_rate_hz = 0.0
        cfg.fdn_tv_depth = 0.0
    cfg.fdn_tv_seed = int(rng.integers(0, 2_147_483_647))
    cfg.fdn_sparse = bool(rng.random() < 0.45)
    cfg.fdn_sparse_degree = int(rng.integers(1, 6))
    if cfg.fdn_sparse and cfg.fdn_matrix in {"tv_unitary", "graph"}:
        cfg.fdn_sparse = False
    if cfg.fdn_matrix == "graph":
        cfg.fdn_graph_topology = cast(
            str,
            rng.choice(np.array(["ring", "path", "star", "random"], dtype=object)),
        )
        cfg.fdn_graph_degree = int(rng.integers(1, 8))
    else:
        cfg.fdn_graph_topology = "ring"
        cfg.fdn_graph_degree = 2
    cfg.fdn_graph_seed = int(rng.integers(0, 2_147_483_647))
    cfg.fdn_cascade = bool(rng.random() < 0.45)
    if cfg.fdn_cascade:
        cfg.fdn_cascade_mix = float(rng.uniform(0.1, 0.85))
        cfg.fdn_cascade_delay_scale = float(rng.uniform(0.2, 0.85))
        cfg.fdn_cascade_rt60_ratio = float(rng.uniform(0.15, 0.9))
    else:
        cfg.fdn_cascade_mix = 0.35
        cfg.fdn_cascade_delay_scale = 0.5
        cfg.fdn_cascade_rt60_ratio = 0.55
    if rng.random() < 0.55:
        cfg.fdn_rt60_low = float(rng.uniform(8.0, 90.0))
        cfg.fdn_rt60_mid = float(rng.uniform(4.0, 50.0))
        cfg.fdn_rt60_high = float(rng.uniform(1.0, 30.0))
        cfg.fdn_tonal_correction_strength = float(rng.uniform(0.05, 0.95))
        cfg.fdn_xover_low_hz = float(rng.uniform(80.0, 800.0))
        cfg.fdn_xover_high_hz = float(
            rng.uniform(
                max(1_200.0, cfg.fdn_xover_low_hz + 200.0),
                9_000.0,
            )
        )
    else:
        cfg.fdn_rt60_low = None
        cfg.fdn_rt60_mid = None
        cfg.fdn_rt60_high = None
        cfg.fdn_tonal_correction_strength = 0.0
        cfg.fdn_xover_low_hz = 250.0
        cfg.fdn_xover_high_hz = 4_000.0
    if rng.random() < 0.5:
        cfg.fdn_link_filter = cast(
            str,
            rng.choice(np.array(["lowpass", "highpass"], dtype=object)),
        )
        cfg.fdn_link_filter_hz = float(rng.uniform(80.0, cfg.sr * 0.45))
        cfg.fdn_link_filter_mix = float(rng.uniform(0.2, 1.0))
    else:
        cfg.fdn_link_filter = "none"
        cfg.fdn_link_filter_hz = 2_500.0
        cfg.fdn_link_filter_mix = 1.0
    if rng.random() < 0.4:
        dfm_count = int(rng.choice(np.array([1, cfg.fdn_lines], dtype=np.int32)))
        cfg.fdn_dfm_delays_ms = tuple(
            float(rng.uniform(0.1, 8.0))
            for _ in range(dfm_count)
        )
    else:
        cfg.fdn_dfm_delays_ms = ()
    cfg.fdn_stereo_inject = float(rng.uniform(0.0, 1.0))

    cfg.harmonic_align_strength = float(rng.uniform(0.0, 1.0))
    cfg.resonator = bool(rng.random() < 0.45)
    cfg.resonator_mix = float(rng.uniform(0.0, 0.95))
    cfg.resonator_modes = int(rng.integers(8, 80))
    cfg.resonator_q_min = float(rng.uniform(0.8, 24.0))
    cfg.resonator_q_max = float(rng.uniform(max(cfg.resonator_q_min + 0.5, 6.0), 140.0))
    cfg.resonator_low_hz = float(rng.uniform(20.0, 250.0))
    cfg.resonator_high_hz = float(
        rng.uniform(max(cfg.resonator_low_hz + 500.0, 1500.0), cfg.sr * 0.48)
    )
    cfg.resonator_late_start_ms = float(rng.uniform(0.0, 400.0))
    return cfg


def _build_lucky_ir_process_config(
    *,
    damping: float,
    lowcut: float | None,
    highcut: float | None,
    tilt: float,
    normalize: Literal["none", "peak", "rms"],
    peak_dbfs: float,
    target_lufs: float | None,
    true_peak: bool,
    rng: np.random.Generator,
    sr: int,
) -> LuckyIRProcessConfig:
    """Build one randomized shaping config for ``ir process --lucky``."""
    lucky_lowcut = lowcut if lowcut is not None else float(rng.uniform(20.0, 500.0))
    lucky_lowcut = float(np.clip(lucky_lowcut * rng.uniform(0.5, 2.0), 20.0, sr * 0.45))

    if highcut is None:
        min_high = min(sr * 0.48, lucky_lowcut + 100.0)
        lucky_highcut = float(rng.uniform(max(min_high, 400.0), sr * 0.49))
    else:
        lucky_highcut = float(np.clip(highcut * rng.uniform(0.5, 1.7), 200.0, sr * 0.49))
    if lucky_highcut <= lucky_lowcut:
        lucky_highcut = min(sr * 0.49, lucky_lowcut + 200.0)

    modes = np.array(["none", "peak", "rms"], dtype=object)
    normalize_mode = (
        cast(Literal["none", "peak", "rms"], rng.choice(modes))
        if rng.random() < 0.85
        else normalize
    )
    return {
        "damping": float(np.clip(damping * rng.uniform(0.4, 2.2), 0.0, 1.0)),
        "lowcut": lucky_lowcut if rng.random() < 0.8 else None,
        "highcut": lucky_highcut if rng.random() < 0.9 else None,
        "tilt": float(np.clip(tilt + rng.uniform(-6.0, 6.0), -12.0, 12.0)),
        "normalize": normalize_mode,
        "peak_dbfs": float(np.clip(peak_dbfs + rng.uniform(-8.0, 2.0), -18.0, -0.1)),
        "target_lufs": (
            float(np.clip((target_lufs or -22.0) + rng.uniform(-10.0, 10.0), -36.0, -8.0))
            if rng.random() < 0.6
            else None
        ),
        "true_peak": bool(rng.random() < 0.7 if true_peak else rng.random() < 0.4),
    }


def _resolve_lucky_seed(lucky_seed: int | None) -> int:
    """Resolve deterministic seed for lucky-mode batches."""
    if lucky_seed is not None:
        return int(lucky_seed)
    return int(np.random.default_rng().integers(0, 2_147_483_647))


def _resolve_ir_output_path(out_ir: Path, out_format: IRFileFormat) -> Path:
    """Resolve output IR path based on explicit format switch."""
    if out_format == "auto":
        return out_ir if out_ir.suffix else out_ir.with_suffix(".wav")

    suffix = ".aiff" if out_format == "aiff" else f".{out_format}"
    return out_ir.with_suffix(suffix)


def _parse_delay_list_ms(raw: str | None, *, option_name: str) -> tuple[float, ...]:
    """Parse a comma-separated millisecond delay list for CLI options."""
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
    """Parse one or more comma-separated gain values for CLI options."""
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


def _validate_fdn_matrix_name(fdn_matrix: str) -> None:
    """Validate FDN matrix topology identifier."""
    normalized = _normalize_fdn_matrix_name(fdn_matrix)
    if normalized not in _FDN_MATRIX_CHOICES:
        options = ", ".join(sorted(_FDN_MATRIX_CHOICES))
        msg = f"--fdn-matrix must be one of: {options}."
        raise typer.BadParameter(msg)


def _validate_fdn_tv_settings(
    *,
    fdn_matrix: str,
    fdn_tv_rate_hz: float,
    fdn_tv_depth: float,
) -> None:
    """Validate time-varying unitary matrix controls."""
    normalized = _normalize_fdn_matrix_name(fdn_matrix)
    rate = float(fdn_tv_rate_hz)
    depth = float(fdn_tv_depth)
    if normalized == "tv_unitary":
        if rate <= 0.0 or depth <= 0.0:
            msg = (
                "--fdn-matrix tv_unitary requires both --fdn-tv-rate-hz > 0 "
                "and --fdn-tv-depth > 0."
            )
            raise typer.BadParameter(msg)
        return

    if rate > 0.0 or depth > 0.0:
        msg = "--fdn-tv-rate-hz and --fdn-tv-depth are only valid with --fdn-matrix tv_unitary."
        raise typer.BadParameter(msg)


def _validate_fdn_sparse_settings(
    *,
    fdn_matrix: str,
    fdn_sparse: bool,
    fdn_sparse_degree: int,
) -> None:
    """Validate sparse high-order FDN settings."""
    normalized = _normalize_fdn_matrix_name(fdn_matrix)
    if fdn_sparse_degree < 1:
        msg = "--fdn-sparse-degree must be >= 1."
        raise typer.BadParameter(msg)
    if fdn_sparse and normalized == "tv_unitary":
        msg = "--fdn-sparse cannot be combined with --fdn-matrix tv_unitary."
        raise typer.BadParameter(msg)
    if fdn_sparse and normalized == "graph":
        msg = "--fdn-sparse cannot be combined with --fdn-matrix graph."
        raise typer.BadParameter(msg)


def _validate_fdn_graph_settings(
    *,
    fdn_matrix: str,
    fdn_graph_topology: str,
    fdn_graph_degree: int,
) -> None:
    """Validate graph-structured FDN controls."""
    normalized_matrix = _normalize_fdn_matrix_name(fdn_matrix)
    normalized_topology = _normalize_fdn_graph_topology_name(fdn_graph_topology)

    if normalized_matrix == "graph":
        if normalized_topology not in _FDN_GRAPH_TOPOLOGY_CHOICES:
            options = ", ".join(sorted(_FDN_GRAPH_TOPOLOGY_CHOICES))
            msg = f"--fdn-graph-topology must be one of: {options}."
            raise typer.BadParameter(msg)
        if int(fdn_graph_degree) < 1:
            msg = "--fdn-graph-degree must be >= 1."
            raise typer.BadParameter(msg)
        return

    # Non-default graph options are considered a configuration mismatch unless graph mode is active.
    if normalized_topology != "ring" or int(fdn_graph_degree) != 2:
        msg = (
            "--fdn-graph-topology/--fdn-graph-degree are only valid with "
            "--fdn-matrix graph."
        )
        raise typer.BadParameter(msg)


def _validate_fdn_cascade_settings(
    *,
    fdn_lines: int,
    fdn_cascade: bool,
    fdn_cascade_mix: float,
    fdn_cascade_delay_scale: float,
    fdn_cascade_rt60_ratio: float,
) -> None:
    """Validate nested/cascaded FDN controls."""
    if not fdn_cascade:
        return
    if fdn_lines < 2:
        msg = "--fdn-cascade requires at least 2 FDN lines."
        raise typer.BadParameter(msg)
    if float(fdn_cascade_mix) <= 0.0:
        msg = "--fdn-cascade-mix must be > 0 when --fdn-cascade is enabled."
        raise typer.BadParameter(msg)
    if float(fdn_cascade_delay_scale) <= 0.0:
        msg = "--fdn-cascade-delay-scale must be > 0."
        raise typer.BadParameter(msg)
    if float(fdn_cascade_rt60_ratio) <= 0.0:
        msg = "--fdn-cascade-rt60-ratio must be > 0."
        raise typer.BadParameter(msg)


def _validate_fdn_multiband_settings(
    *,
    fdn_rt60_low: float | None,
    fdn_rt60_mid: float | None,
    fdn_rt60_high: float | None,
    fdn_xover_low_hz: float,
    fdn_xover_high_hz: float,
) -> None:
    """Validate multiband FDN decay controls."""
    set_count = sum(
        value is not None for value in (fdn_rt60_low, fdn_rt60_mid, fdn_rt60_high)
    )
    if set_count not in {0, 3}:
        msg = (
            "For multiband decay use either none of --fdn-rt60-low/mid/high "
            "or provide all three values."
        )
        raise typer.BadParameter(msg)
    if float(fdn_xover_low_hz) >= float(fdn_xover_high_hz):
        msg = "--fdn-xover-low-hz must be < --fdn-xover-high-hz."
        raise typer.BadParameter(msg)


def _validate_fdn_link_filter_settings(
    *,
    fdn_link_filter: str,
    fdn_link_filter_hz: float,
    fdn_link_filter_mix: float,
) -> None:
    """Validate feedback-link filter controls used in the FDN path."""
    normalized = _normalize_fdn_link_filter_name(fdn_link_filter)
    if normalized not in _FDN_LINK_FILTER_CHOICES:
        options = ", ".join(sorted(_FDN_LINK_FILTER_CHOICES))
        msg = f"--fdn-link-filter must be one of: {options}."
        raise typer.BadParameter(msg)
    if float(fdn_link_filter_hz) <= 0.0:
        msg = "--fdn-link-filter-hz must be > 0."
        raise typer.BadParameter(msg)
    mix = float(fdn_link_filter_mix)
    if mix < 0.0 or mix > 1.0:
        msg = "--fdn-link-filter-mix must be in [0.0, 1.0]."
        raise typer.BadParameter(msg)


def _validate_perceptual_macro_settings(
    *,
    fdn_rt60_tilt: float,
    room_size_macro: float,
    clarity_macro: float,
    warmth_macro: float,
    envelopment_macro: float,
) -> None:
    """Validate perceptual macro controls and Jot-inspired RT tilt."""
    values = {
        "--fdn-rt60-tilt": float(fdn_rt60_tilt),
        "--room-size-macro": float(room_size_macro),
        "--clarity-macro": float(clarity_macro),
        "--warmth-macro": float(warmth_macro),
        "--envelopment-macro": float(envelopment_macro),
    }
    for option_name, value in values.items():
        if value < -1.0 or value > 1.0:
            msg = f"{option_name} must be in [-1.0, 1.0]."
            raise typer.BadParameter(msg)


def _validate_fdn_tonal_correction_settings(*, fdn_tonal_correction_strength: float) -> None:
    """Validate Track C tonal-correction controls."""
    strength = float(fdn_tonal_correction_strength)
    if strength < 0.0 or strength > 1.0:
        msg = "--fdn-tonal-correction-strength must be in [0.0, 1.0]."
        raise typer.BadParameter(msg)


def _normalize_fdn_matrix_name(value: str) -> str:
    """Normalize FDN matrix identifier for CLI/API compatibility."""
    return _shared_normalize_fdn_matrix_name(value)


def _normalize_fdn_link_filter_name(value: str) -> str:
    """Normalize FDN feedback-link filter identifier for CLI/API compatibility."""
    return _shared_normalize_fdn_link_filter_name(value)


def _normalize_fdn_graph_topology_name(value: str) -> str:
    """Normalize graph-structured FDN topology identifier."""
    return _shared_normalize_fdn_graph_topology_name(value)


def _normalize_ir_route_map_name(value: str) -> str:
    """Normalize convolution IR route-map identifier."""
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"diag"}:
        return "diagonal"
    if normalized in {"full_matrix", "fullmatrix"}:
        return "full"
    return normalized


def _normalize_conv_route_curve_name(value: str) -> str:
    """Normalize convolution route trajectory curve name."""
    return value.strip().lower().replace("_", "-")


def _validate_ir_route_map_name(value: str) -> None:
    """Validate named convolution route-map preset."""
    normalized = _normalize_ir_route_map_name(value)
    if normalized not in _IR_ROUTE_MAP_CHOICES:
        options = ", ".join(sorted(_IR_ROUTE_MAP_CHOICES))
        msg = f"--ir-route-map must be one of: {options}."
        raise typer.BadParameter(msg)


def _validate_conv_route_settings(
    *,
    conv_route_start: str | None,
    conv_route_end: str | None,
    conv_route_curve: str,
) -> None:
    """Validate convolution route-trajectory controls."""
    if (conv_route_start is None) != (conv_route_end is None):
        msg = "Use both --conv-route-start and --conv-route-end together."
        raise typer.BadParameter(msg)
    normalized_curve = _normalize_conv_route_curve_name(conv_route_curve)
    if normalized_curve not in _CONV_ROUTE_CURVE_CHOICES:
        options = ", ".join(sorted(_CONV_ROUTE_CURVE_CHOICES))
        msg = f"--conv-route-curve must be one of: {options}."
        raise typer.BadParameter(msg)


def _validate_ambisonic_settings(infile: Path, config: RenderConfig) -> None:
    """Validate Ambisonics render options and channel/layout compatibility."""
    order = int(config.ambi_order)
    if order < 0:
        raise typer.BadParameter("--ambi-order must be >= 0.")

    if config.ambi_normalization not in _AMBI_NORMALIZATION_CHOICES:
        options = ", ".join(sorted(_AMBI_NORMALIZATION_CHOICES))
        raise typer.BadParameter(f"--ambi-normalization must be one of: {options}.")
    if config.channel_order not in _AMBI_CHANNEL_ORDER_CHOICES:
        options = ", ".join(sorted(_AMBI_CHANNEL_ORDER_CHOICES))
        raise typer.BadParameter(f"--channel-order must be one of: {options}.")
    if config.ambi_encode_from not in _AMBI_ENCODE_CHOICES:
        options = ", ".join(sorted(_AMBI_ENCODE_CHOICES))
        raise typer.BadParameter(f"--ambi-encode-from must be one of: {options}.")
    if config.ambi_decode_to not in _AMBI_DECODE_CHOICES:
        options = ", ".join(sorted(_AMBI_DECODE_CHOICES))
        raise typer.BadParameter(f"--ambi-decode-to must be one of: {options}.")

    if order == 0:
        if config.ambi_encode_from != "none":
            raise typer.BadParameter("--ambi-encode-from requires --ambi-order 1.")
        if config.ambi_decode_to != "none":
            raise typer.BadParameter("--ambi-decode-to requires --ambi-order >= 1.")
        if abs(float(config.ambi_rotate_yaw_deg)) > 1e-12:
            raise typer.BadParameter("--ambi-rotate-yaw-deg requires --ambi-order >= 1.")
        if config.ambi_normalization != "auto" or config.channel_order != "auto":
            raise typer.BadParameter(
                "--ambi-normalization/--channel-order require --ambi-order >= 1."
            )
        return

    try:
        normalize_ambisonic_metadata(
            order=order,
            normalization=config.ambi_normalization,
            channel_order=config.channel_order,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    expected_channels = ambisonic_channel_count(order)
    if config.input_layout != "auto":
        msg = "Ambisonics mode requires --input-layout auto."
        raise typer.BadParameter(msg)
    if config.ambi_decode_to == "none" and config.output_layout != "auto":
        msg = "Ambisonics mode requires --output-layout auto unless --ambi-decode-to stereo is used."
        raise typer.BadParameter(msg)

    try:
        in_info = sf.info(str(infile))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    input_channels = int(in_info.channels)

    if config.ambi_encode_from == "none":
        if input_channels != expected_channels:
            msg = (
                f"Input channels ({input_channels}) do not match --ambi-order {order} "
                f"({expected_channels} channels expected). "
                "Use matching Ambisonic input or set --ambi-encode-from mono|stereo."
            )
            raise typer.BadParameter(msg)
    else:
        if order != 1:
            msg = "--ambi-encode-from currently supports FOA only (--ambi-order 1)."
            raise typer.BadParameter(msg)
        required_channels = 1 if config.ambi_encode_from == "mono" else 2
        if input_channels != required_channels:
            msg = (
                f"--ambi-encode-from {config.ambi_encode_from} expects "
                f"{required_channels} input channel(s), got {input_channels}."
            )
            raise typer.BadParameter(msg)

    if config.ambi_decode_to == "stereo" and config.output_layout not in {"auto", "stereo"}:
        msg = "--ambi-decode-to stereo is only valid with --output-layout auto or stereo."
        raise typer.BadParameter(msg)


def _validate_automation_settings(config: RenderConfig, outfile: Path) -> None:
    """Validate automation file/options before render dispatch."""
    if config.automation_mode not in _AUTOMATION_MODE_CHOICES:
        choices = ", ".join(sorted(_AUTOMATION_MODE_CHOICES))
        raise typer.BadParameter(f"--automation-mode must be one of: {choices}.")

    if config.automation_block_ms <= 0.0:
        raise typer.BadParameter("--automation-block-ms must be > 0.")
    if config.automation_smoothing_ms < 0.0:
        raise typer.BadParameter("--automation-smoothing-ms must be >= 0.")

    has_automation_source = (
        config.automation_file is not None
        or len(config.automation_points) > 0
    )
    has_automation_args = (
        config.automation_mode != "auto"
        or abs(float(config.automation_block_ms) - 20.0) > 1e-12
        or abs(float(config.automation_smoothing_ms) - 20.0) > 1e-12
        or len(config.automation_clamp) > 0
        or config.automation_trace_out is not None
    )
    if not has_automation_source and has_automation_args:
        msg = (
            "--automation-mode/--automation-block-ms/--automation-smoothing-ms/"
            "--automation-clamp/--automation-trace-out require --automation-file "
            "or --automation-point."
        )
        raise typer.BadParameter(msg)

    if not has_automation_source:
        return

    source_path: Path | None = None
    if config.automation_file is not None:
        source_path = Path(config.automation_file)
        if not source_path.exists():
            raise typer.BadParameter(f"Automation file not found: {config.automation_file}")
        if source_path.suffix.lower() not in {".json", ".csv"}:
            raise typer.BadParameter("--automation-file must be a .json or .csv file.")

    try:
        parse_automation_clamp_overrides(config.automation_clamp)
        parse_automation_point_specs(config.automation_points)
        targets = collect_automation_targets(
            path=source_path,
            point_specs=config.automation_points,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    conv_selected = config.engine == "conv" or (
        config.engine == "auto" and (config.ir is not None or config.ir_gen or config.self_convolve)
    )
    if conv_selected:
        engine_targets = sorted(target for target in targets if target in ENGINE_AUTOMATION_TARGETS)
        if len(engine_targets) > 0:
            joined = ", ".join(engine_targets)
            msg = (
                "Automation targets require algorithmic render path; "
                f"use --engine algo (targets: {joined})."
            )
            raise typer.BadParameter(msg)

    if config.automation_trace_out is not None:
        trace_path = Path(config.automation_trace_out)
        if trace_path.resolve() == outfile.resolve():
            raise typer.BadParameter("--automation-trace-out must differ from OUTFILE.")
        if trace_path.suffix.lower() != ".csv":
            raise typer.BadParameter("--automation-trace-out must use .csv extension.")


def _validate_lucky_call(
    config: RenderConfig,
    lucky: int | None,
    lucky_out_dir: Path | None,
) -> None:
    """Validate lucky-mode options for randomized batch rendering."""
    _validate_generic_lucky_call(lucky, lucky_out_dir)
    if lucky is None:
        return
    if config.analysis_out is not None:
        msg = "Do not use --analysis-out with --lucky (analysis files are per-output by default)."
        raise typer.BadParameter(msg)
    if config.frames_out is not None:
        msg = "Do not use --frames-out with --lucky."
        raise typer.BadParameter(msg)
    if config.automation_trace_out is not None:
        msg = "Do not use --automation-trace-out with --lucky."
        raise typer.BadParameter(msg)


def _validate_generic_lucky_call(lucky: int | None, lucky_out_dir: Path | None) -> None:
    """Validate generic lucky-mode options shared by multiple commands."""
    if lucky is None:
        return
    if lucky < 1:
        msg = "--lucky must be >= 1."
        raise typer.BadParameter(msg)
    if lucky_out_dir is not None and lucky_out_dir.exists() and not lucky_out_dir.is_dir():
        msg = f"--lucky-out-dir is not a directory: {lucky_out_dir}"
        raise typer.BadParameter(msg)


def _validate_render_call(infile: Path, outfile: Path, config: RenderConfig) -> None:
    """Validate render CLI arguments before pipeline execution."""
    _ensure_distinct_paths(infile, outfile, "INFILE", "OUTFILE")
    _validate_output_audio_path(outfile, config.output_subtype)

    if config.self_convolve:
        if config.ir is not None:
            msg = "Use either --ir or --self-convolve, not both."
            raise typer.BadParameter(msg)
        if config.ir_gen:
            msg = "Use either --ir-gen or --self-convolve, not both."
            raise typer.BadParameter(msg)
        if config.engine == "algo":
            msg = "--self-convolve is only valid with --engine conv or --engine auto."
            raise typer.BadParameter(msg)

    if (
        config.engine == "conv"
        and config.ir is None
        and not config.ir_gen
        and not config.self_convolve
    ):
        msg = "Convolution render requires --ir PATH, --ir-gen, or --self-convolve."
        raise typer.BadParameter(msg)

    if config.ir is not None and not Path(config.ir).exists():
        msg = f"IR file not found: {config.ir}"
        raise typer.BadParameter(msg)

    _validate_ir_route_map_name(config.ir_route_map)
    _validate_conv_route_settings(
        conv_route_start=config.conv_route_start,
        conv_route_end=config.conv_route_end,
        conv_route_curve=config.conv_route_curve,
    )
    _validate_ambisonic_settings(infile, config)
    _validate_automation_settings(config, outfile)

    conv_enabled = (
        config.engine == "conv"
        or (
            config.engine == "auto"
            and (config.ir is not None or config.ir_gen or config.self_convolve)
        )
    )
    if (
        config.conv_route_start is not None or config.conv_route_end is not None
    ) and not conv_enabled:
        msg = "--conv-route-start/--conv-route-end are only valid for convolution renders."
        raise typer.BadParameter(msg)

    if config.ir_route_map != "auto" and not conv_enabled:
        msg = "--ir-route-map is only valid for convolution render workflows."
        raise typer.BadParameter(msg)

    if (
        config.ir is not None
        and (config.engine == "conv" or config.engine == "auto")
        and not config.self_convolve
    ):
        try:
            in_info = sf.info(str(infile))
            ir_info = sf.info(str(config.ir))
            in_channels = int(in_info.channels)
            ir_channels = int(ir_info.channels)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc

        effective_in_channels = in_channels
        if config.ambi_order > 0:
            effective_in_channels = ambisonic_channel_count(int(config.ambi_order))
            if ir_channels not in {1, effective_in_channels} and (
                ir_channels % max(1, effective_in_channels)
            ) != 0:
                msg = (
                    f"IR channel layout ({ir_channels}) is incompatible with Ambisonics order "
                    f"{config.ambi_order} ({effective_in_channels} channels)."
                )
                raise typer.BadParameter(msg)

        if (
            config.ir_route_map == "auto"
            and config.output_layout == "auto"
            and effective_in_channels > 0
            and ir_channels > effective_in_channels
            and ir_channels % effective_in_channels == 0
        ):
            msg = (
                "Ambiguous matrix-packed IR layout detected. "
                "Set --output-layout and/or --ir-route-map (for example: --ir-route-map full)."
            )
            raise typer.BadParameter(msg)

    if config.wet == 0.0 and config.dry == 0.0:
        msg = "At least one of --wet or --dry must be non-zero."
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
        msg = "--comb-delays-ms supports at most 64 entries."
        raise typer.BadParameter(msg)
    if len(config.allpass_delays_ms) > 128:
        msg = "--allpass-delays-ms supports at most 128 entries."
        raise typer.BadParameter(msg)
    if len(config.fdn_dfm_delays_ms) > 64:
        msg = "--fdn-dfm-delays-ms supports at most 64 entries."
        raise typer.BadParameter(msg)

    _validate_fdn_matrix_name(config.fdn_matrix)
    _validate_fdn_tv_settings(
        fdn_matrix=config.fdn_matrix,
        fdn_tv_rate_hz=config.fdn_tv_rate_hz,
        fdn_tv_depth=config.fdn_tv_depth,
    )
    _validate_fdn_sparse_settings(
        fdn_matrix=config.fdn_matrix,
        fdn_sparse=config.fdn_sparse,
        fdn_sparse_degree=config.fdn_sparse_degree,
    )
    _validate_fdn_graph_settings(
        fdn_matrix=config.fdn_matrix,
        fdn_graph_topology=config.fdn_graph_topology,
        fdn_graph_degree=config.fdn_graph_degree,
    )
    _validate_fdn_multiband_settings(
        fdn_rt60_low=config.fdn_rt60_low,
        fdn_rt60_mid=config.fdn_rt60_mid,
        fdn_rt60_high=config.fdn_rt60_high,
        fdn_xover_low_hz=config.fdn_xover_low_hz,
        fdn_xover_high_hz=config.fdn_xover_high_hz,
    )
    _validate_fdn_link_filter_settings(
        fdn_link_filter=config.fdn_link_filter,
        fdn_link_filter_hz=config.fdn_link_filter_hz,
        fdn_link_filter_mix=config.fdn_link_filter_mix,
    )
    _validate_perceptual_macro_settings(
        fdn_rt60_tilt=config.fdn_rt60_tilt,
        room_size_macro=config.room_size_macro,
        clarity_macro=config.clarity_macro,
        warmth_macro=config.warmth_macro,
        envelopment_macro=config.envelopment_macro,
    )
    _validate_fdn_tonal_correction_settings(
        fdn_tonal_correction_strength=config.fdn_tonal_correction_strength,
    )
    resolved_fdn_lines = (
        len(config.comb_delays_ms) if len(config.comb_delays_ms) > 0 else int(config.fdn_lines)
    )
    _validate_fdn_cascade_settings(
        fdn_lines=resolved_fdn_lines,
        fdn_cascade=config.fdn_cascade,
        fdn_cascade_mix=config.fdn_cascade_mix,
        fdn_cascade_delay_scale=config.fdn_cascade_delay_scale,
        fdn_cascade_rt60_ratio=config.fdn_cascade_rt60_ratio,
    )
    if len(config.fdn_dfm_delays_ms) not in {0, 1, resolved_fdn_lines}:
        msg = (
            "--fdn-dfm-delays-ms must include either 1 value or exactly "
            f"{resolved_fdn_lines} values."
        )
        raise typer.BadParameter(msg)

    if config.freeze:
        if config.start is None or config.end is None:
            msg = "--freeze requires both --start and --end."
            raise typer.BadParameter(msg)
        if config.end <= config.start:
            msg = "--end must be greater than --start when --freeze is enabled."
            raise typer.BadParameter(msg)
    elif config.start is not None or config.end is not None:
        msg = "--start/--end are only valid when --freeze is enabled."
        raise typer.BadParameter(msg)

    if config.output_peak_norm == "target" and config.output_peak_target_dbfs is None:
        msg = "--output-peak-norm target requires --output-peak-target-dbfs."
        raise typer.BadParameter(msg)
    if config.output_peak_norm != "target" and config.output_peak_target_dbfs is not None:
        msg = "--output-peak-target-dbfs is only valid with --output-peak-norm target."
        raise typer.BadParameter(msg)

    if config.ir_gen and config.ir is not None:
        msg = "Use either --ir or --ir-gen, not both."
        raise typer.BadParameter(msg)

    if config.mod_min >= config.mod_max:
        msg = "--mod-min must be less than --mod-max."
        raise typer.BadParameter(msg)

    if config.mod_target == "none" and len(config.mod_sources) > 0:
        msg = "--mod-source requires --mod-target."
        raise typer.BadParameter(msg)
    if config.mod_target != "none" and len(config.mod_sources) == 0:
        msg = "--mod-target requires at least one --mod-source."
        raise typer.BadParameter(msg)

    if config.mod_target in {"mix", "wet"}:
        if config.mod_min < 0.0 or config.mod_max > 1.0:
            msg = "For --mod-target mix/wet, use --mod-min/--mod-max in [0.0, 1.0]."
            raise typer.BadParameter(msg)

    try:
        parse_mod_sources(config.mod_sources)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    for route_spec in config.mod_routes:
        try:
            parse_mod_route_spec(route_spec)
        except ValueError as exc:
            raise typer.BadParameter(f"invalid --mod-route '{route_spec}': {exc}") from exc


def _validate_analyze_call(infile: Path, json_out: Path | None, frames_out: Path | None) -> None:
    """Validate analyze command output paths."""
    if json_out is not None and infile.resolve() == json_out.resolve():
        msg = "--json-out must be different from input file."
        raise typer.BadParameter(msg)
    if frames_out is not None and infile.resolve() == frames_out.resolve():
        msg = "--frames-out must be different from input file."
        raise typer.BadParameter(msg)


def _validate_ir_gen_call(
    out_ir: Path,
    out_format: IRFileFormat,
    rt60: float | None,
    rt60_low: float | None,
    rt60_high: float | None,
    modal_q_min: float,
    modal_q_max: float,
    modal_low_hz: float,
    modal_high_hz: float,
    resonator_q_min: float,
    resonator_q_max: float,
    resonator_low_hz: float,
    resonator_high_hz: float,
    fdn_lines: int,
    fdn_matrix: str,
    fdn_tv_rate_hz: float,
    fdn_tv_depth: float,
    fdn_sparse: bool,
    fdn_sparse_degree: int,
    fdn_cascade: bool,
    fdn_cascade_mix: float,
    fdn_cascade_delay_scale: float,
    fdn_cascade_rt60_ratio: float,
    fdn_rt60_low: float | None,
    fdn_rt60_mid: float | None,
    fdn_rt60_high: float | None,
    fdn_rt60_tilt: float,
    fdn_tonal_correction_strength: float,
    fdn_xover_low_hz: float,
    fdn_xover_high_hz: float,
    fdn_link_filter: str,
    fdn_link_filter_hz: float,
    fdn_link_filter_mix: float,
    fdn_graph_topology: str,
    fdn_graph_degree: int,
    room_size_macro: float,
    clarity_macro: float,
    warmth_macro: float,
    envelopment_macro: float,
) -> None:
    """Validate IR generation options and output path constraints."""
    resolved = _resolve_ir_output_path(out_ir, out_format)
    _validate_output_audio_path(resolved, "auto")

    if rt60 is not None and (rt60_low is not None or rt60_high is not None):
        msg = "Use either --rt60 or --rt60-low/--rt60-high, not both."
        raise typer.BadParameter(msg)
    if (rt60_low is None) != (rt60_high is None):
        msg = "Both --rt60-low and --rt60-high must be provided together."
        raise typer.BadParameter(msg)
    if rt60_low is not None and rt60_high is not None and rt60_low > rt60_high:
        msg = "--rt60-low must be <= --rt60-high."
        raise typer.BadParameter(msg)
    if modal_q_min > modal_q_max:
        msg = "--modal-q-min must be <= --modal-q-max."
        raise typer.BadParameter(msg)
    if modal_low_hz >= modal_high_hz:
        msg = "--modal-low-hz must be < --modal-high-hz."
        raise typer.BadParameter(msg)
    if resonator_q_min > resonator_q_max:
        msg = "--resonator-q-min must be <= --resonator-q-max."
        raise typer.BadParameter(msg)
    if resonator_low_hz >= resonator_high_hz:
        msg = "--resonator-low-hz must be < --resonator-high-hz."
        raise typer.BadParameter(msg)

    _validate_fdn_matrix_name(fdn_matrix)
    _validate_fdn_tv_settings(
        fdn_matrix=fdn_matrix,
        fdn_tv_rate_hz=fdn_tv_rate_hz,
        fdn_tv_depth=fdn_tv_depth,
    )
    _validate_fdn_sparse_settings(
        fdn_matrix=fdn_matrix,
        fdn_sparse=fdn_sparse,
        fdn_sparse_degree=fdn_sparse_degree,
    )
    _validate_fdn_graph_settings(
        fdn_matrix=fdn_matrix,
        fdn_graph_topology=fdn_graph_topology,
        fdn_graph_degree=fdn_graph_degree,
    )
    _validate_fdn_cascade_settings(
        fdn_lines=fdn_lines,
        fdn_cascade=fdn_cascade,
        fdn_cascade_mix=fdn_cascade_mix,
        fdn_cascade_delay_scale=fdn_cascade_delay_scale,
        fdn_cascade_rt60_ratio=fdn_cascade_rt60_ratio,
    )
    _validate_fdn_multiband_settings(
        fdn_rt60_low=fdn_rt60_low,
        fdn_rt60_mid=fdn_rt60_mid,
        fdn_rt60_high=fdn_rt60_high,
        fdn_xover_low_hz=fdn_xover_low_hz,
        fdn_xover_high_hz=fdn_xover_high_hz,
    )
    _validate_fdn_link_filter_settings(
        fdn_link_filter=fdn_link_filter,
        fdn_link_filter_hz=fdn_link_filter_hz,
        fdn_link_filter_mix=fdn_link_filter_mix,
    )
    _validate_perceptual_macro_settings(
        fdn_rt60_tilt=fdn_rt60_tilt,
        room_size_macro=room_size_macro,
        clarity_macro=clarity_macro,
        warmth_macro=warmth_macro,
        envelopment_macro=envelopment_macro,
    )
    _validate_fdn_tonal_correction_settings(
        fdn_tonal_correction_strength=fdn_tonal_correction_strength,
    )


def _validate_ir_process_call(in_ir: Path, out_ir: Path) -> None:
    """Validate IR process command path arguments."""
    _ensure_distinct_paths(in_ir, out_ir, "IN_IR", "OUT_IR")
    _validate_output_audio_path(out_ir, "auto")


def _validate_ir_analyze_call(ir_file: Path, json_out: Path | None) -> None:
    """Validate IR analyze optional output path."""
    if json_out is not None and ir_file.resolve() == json_out.resolve():
        msg = "--json-out must be different from input IR file."
        raise typer.BadParameter(msg)


def _validate_batch_job_paths(infile: Path, outfile: Path, idx: int) -> None:
    """Validate one batch job's input/output paths."""
    if infile.resolve() == outfile.resolve():
        msg = f"jobs[{idx - 1}] infile and outfile must be different."
        raise typer.BadParameter(msg)
    if not infile.exists():
        msg = f"jobs[{idx - 1}] infile not found: {infile}"
        raise typer.BadParameter(msg)
    _validate_output_audio_path(outfile, "auto")


def _ensure_distinct_paths(in_path: Path, out_path: Path, in_label: str, out_label: str) -> None:
    """Ensure input and output paths are not identical."""
    if in_path.resolve() == out_path.resolve():
        msg = f"{in_label} and {out_label} must be different paths."
        raise typer.BadParameter(msg)


def _validate_output_audio_path(path: Path, out_subtype_mode: str) -> None:
    """Validate output extension and requested SoundFile subtype support."""
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "":
        msg = f"Output path must include an audio file extension: {path}"
        raise typer.BadParameter(msg)

    format_map = {
        "wav": "WAV",
        "flac": "FLAC",
        "aif": "AIFF",
        "aiff": "AIFF",
        "ogg": "OGG",
        "caf": "CAF",
        "au": "AU",
    }
    fmt = format_map.get(suffix)
    if fmt is None:
        msg = f"Unsupported output audio extension: .{suffix}"
        raise typer.BadParameter(msg)

    subtype_map = {
        "auto": None,
        "float32": "FLOAT",
        "float64": "DOUBLE",
        "pcm16": "PCM_16",
        "pcm24": "PCM_24",
        "pcm32": "PCM_32",
    }
    subtype = subtype_map.get(out_subtype_mode)
    if out_subtype_mode not in subtype_map:
        msg = f"Unsupported --out-subtype value: {out_subtype_mode}"
        raise typer.BadParameter(msg)

    if subtype is None:
        if not sf.check_format(fmt):
            msg = f"SoundFile cannot write format '{fmt}' for output path {path}"
            raise typer.BadParameter(msg)
    else:
        if not sf.check_format(fmt, subtype):
            msg = f"Subtype '{subtype}' is not supported for format '{fmt}'"
            raise typer.BadParameter(msg)


if __name__ == "__main__":
    app()
