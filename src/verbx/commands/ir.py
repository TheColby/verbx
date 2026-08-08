# ruff: noqa: B008
"""IR command wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import typer

from verbx.config import FDNNonlinearityMode, FDNSpatialCouplingMode, IRMode, IRMorphMismatchPolicy
from verbx.core.batch_scheduler import BatchSchedulePolicy
from verbx.core.control_targets import RT60_MAX_SECONDS, RT60_MIN_SECONDS

IRFileFormat = Literal["auto", "wav", "flac", "aiff", "aif", "ogg", "caf"]


def _forward(name: str, params: dict[str, Any]) -> None:
    from verbx import cli as cli_module

    return cli_module.get_command_impl(name)(**params)


def ir_gen(
    out_ir: Path = typer.Argument(..., resolve_path=True),
    out_format: IRFileFormat = typer.Option("auto", "--format"),
    mode: IRMode = typer.Option("hybrid", "--mode"),
    length: float = typer.Option(60.0, "--length", min=0.1),
    sr: int = typer.Option(48_000, "--sr", min=8_000),
    channels: int = typer.Option(2, "--channels", min=1),
    seed: int = typer.Option(0, "--seed"),
    rt60: float | None = typer.Option(None, "--rt60", min=RT60_MIN_SECONDS, max=RT60_MAX_SECONDS),
    rt60_low: float | None = typer.Option(
        None,
        "--rt60-low",
        min=RT60_MIN_SECONDS,
        max=RT60_MAX_SECONDS,
    ),
    rt60_high: float | None = typer.Option(
        None,
        "--rt60-high",
        min=RT60_MIN_SECONDS,
        max=RT60_MAX_SECONDS,
    ),
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
    scala_file: Path | None = typer.Option(
        None,
        "--scala-file",
        exists=True,
        readable=True,
        resolve_path=True,
        help="Scala .scl scale used to tune and emphasize synthetic IR resonances.",
    ),
    scala_root_hz: float = typer.Option(440.0, "--scala-root-hz", min=1.0),
    scala_root_degree: int = typer.Option(0, "--scala-root-degree", min=0),
    scala_low_hz: float | None = typer.Option(None, "--scala-low-hz", min=20.0),
    scala_high_hz: float | None = typer.Option(None, "--scala-high-hz", min=30.0),
    scala_strength: float = typer.Option(1.0, "--scala-strength", min=0.0, max=1.0),
    scala_bandwidth_cents: float = typer.Option(
        25.0, "--scala-bandwidth-cents", min=1.0, max=1_200.0
    ),
    scala_gain_db: float = typer.Option(4.0, "--scala-gain-db", min=0.0, max=24.0),
    scala_max_targets: int = typer.Option(128, "--scala-max-targets", min=1, max=512),
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
            "circulant, elliptic, tv_unitary, graph, or sdn_hybrid."
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
    _forward("_ir_gen_impl", dict(locals()))


def ir_analyze(
    ir_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    json_out: Path | None = typer.Option(None, "--json-out", resolve_path=True),
) -> None:
    _forward("_ir_analyze_impl", dict(locals()))


def ir_sofa_info(
    sofa_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    json_out: Path | None = typer.Option(None, "--json-out", resolve_path=True),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    _forward("_ir_sofa_info_impl", dict(locals()))


def ir_sofa_extract(
    sofa_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_ir: Path = typer.Argument(..., resolve_path=True),
    measurement_index: int = typer.Option(
        0,
        "--measurement-index",
        min=0,
        help="Measurement index for SOFA Data/IR extraction (first axis in strict modes).",
    ),
    emitter_index: int = typer.Option(
        0,
        "--emitter-index",
        min=0,
        help="Emitter index for rank-4 Data/IR extraction (strict mode).",
    ),
    target_sr: int | None = typer.Option(
        None,
        "--target-sr",
        min=1,
        help="Optional output sample rate target for extracted IR.",
    ),
    normalize: Literal["none", "peak", "rms"] = typer.Option(
        "peak",
        "--normalize",
        help="Normalization for extracted IR matrix.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict/--best-effort",
        help="Strict expects Data/IR rank 3 (M,R,N) or 4 (M,R,E,N).",
    ),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    _forward("_ir_sofa_extract_impl", dict(locals()))


def ir_trace(
    dxf_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_ir: Path = typer.Argument(..., resolve_path=True),
    source: str = typer.Option(
        ...,
        "--source",
        help="Source position in x,y,z meters, for example 2,3,1.5.",
    ),
    listener: str = typer.Option(
        ...,
        "--listener",
        help="Listener position in x,y,z meters, for example 6,4,1.5.",
    ),
    height: float = typer.Option(
        3.0,
        "--height",
        min=0.1,
        help="Room height in meters when the DXF is a 2D plan.",
    ),
    material: str = typer.Option(
        "studio",
        "--material",
        help=(
            "Default material/absorption preset for all room surfaces "
            "(for example studio, drywall, glass, concrete, acoustic-panel)."
        ),
    ),
    rays: int = typer.Option(
        50_000,
        "--rays",
        min=1,
        help="Stochastic ray budget used for late-tail density metadata and synthesis.",
    ),
    length: float = typer.Option(4.0, "--length", min=0.05),
    target_sr: int = typer.Option(48_000, "--target-sr", min=8_000),
    seed: int = typer.Option(0, "--seed"),
    json_out: Path | None = typer.Option(None, "--json-out", resolve_path=True),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    _forward("_ir_trace_impl", dict(locals()))


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
    _forward("_ir_process_impl", dict(locals()))


def ir_morph(
    ir_a: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    ir_b: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_ir: Path = typer.Argument(..., resolve_path=True),
    mode: str = typer.Option(
        "equal-power",
        "--mode",
        help="Morph mode: linear, equal-power, spectral, or envelope-aware.",
    ),
    alpha: float = typer.Option(0.5, "--alpha", min=0.0, max=1.0),
    early_ms: float = typer.Option(
        80.0,
        "--early-ms",
        min=0.0,
        help="Early/late split used by split/envelope-aware morphing (ms).",
    ),
    early_alpha: float | None = typer.Option(
        None,
        "--early-alpha",
        min=0.0,
        max=1.0,
        help="Optional alpha override for early-reflection region.",
    ),
    late_alpha: float | None = typer.Option(
        None,
        "--late-alpha",
        min=0.0,
        max=1.0,
        help="Optional alpha override for late-tail region.",
    ),
    align_decay: bool = typer.Option(
        True,
        "--align-decay/--no-align-decay",
        help="Align decay profiles before morphing for stable RT trajectories.",
    ),
    phase_coherence: float = typer.Option(
        0.75,
        "--phase-coherence",
        min=0.0,
        max=1.0,
        help="Phase-coherence safeguard strength for spectral morphing.",
    ),
    spectral_smooth_bins: int = typer.Option(
        3,
        "--spectral-smooth-bins",
        min=0,
        max=128,
        help="Frequency smoothing radius (FFT bins) used by spectral modes.",
    ),
    mismatch_policy: IRMorphMismatchPolicy = typer.Option(
        "coerce",
        "--mismatch-policy",
        help=(
            "Mismatch behavior for sample-rate/channel/duration differences: "
            "coerce (align) or strict (fail)."
        ),
    ),
    target_sr: int | None = typer.Option(
        None,
        "--target-sr",
        min=1,
        help="Optional target sample rate for morph processing and output.",
    ),
    cache_dir: str = typer.Option(".verbx_cache/ir_morph", "--cache-dir"),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    _forward("_ir_morph_impl", dict(locals()))


def ir_morph_sweep(
    ir_a: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    ir_b: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_dir: Path = typer.Argument(..., resolve_path=True),
    mode: str = typer.Option(
        "equal-power",
        "--mode",
        help="Morph mode: linear, equal-power, spectral, or envelope-aware.",
    ),
    alpha_points: list[float] | None = typer.Option(
        None,
        "--alpha",
        min=0.0,
        max=1.0,
        help="Explicit alpha point. Repeat to define custom sweep timeline.",
    ),
    alpha_start: float = typer.Option(0.0, "--alpha-start", min=0.0, max=1.0),
    alpha_end: float = typer.Option(1.0, "--alpha-end", min=0.0, max=1.0),
    alpha_steps: int = typer.Option(9, "--alpha-steps", min=2, max=257),
    out_prefix: str = typer.Option(
        "morph",
        "--out-prefix",
        help="Output filename prefix for generated sweep IR files.",
    ),
    early_ms: float = typer.Option(80.0, "--early-ms", min=0.0),
    early_alpha: float | None = typer.Option(None, "--early-alpha", min=0.0, max=1.0),
    late_alpha: float | None = typer.Option(None, "--late-alpha", min=0.0, max=1.0),
    align_decay: bool = typer.Option(
        True,
        "--align-decay/--no-align-decay",
        help="Align decay profiles before morphing for stable RT trajectories.",
    ),
    phase_coherence: float = typer.Option(0.75, "--phase-coherence", min=0.0, max=1.0),
    spectral_smooth_bins: int = typer.Option(3, "--spectral-smooth-bins", min=0, max=128),
    mismatch_policy: IRMorphMismatchPolicy = typer.Option(
        "coerce",
        "--mismatch-policy",
        help=(
            "Mismatch behavior for sample-rate/channel/duration differences: "
            "coerce (align) or strict (fail)."
        ),
    ),
    target_sr: int | None = typer.Option(None, "--target-sr", min=1),
    cache_dir: str = typer.Option(".verbx_cache/ir_morph", "--cache-dir"),
    workers: int = typer.Option(0, "--workers", min=0, help="0 = auto"),
    schedule: BatchSchedulePolicy = typer.Option("longest-first", "--schedule"),
    retries: int = typer.Option(0, "--retries", min=0),
    continue_on_error: bool = typer.Option(False, "--continue-on-error/--fail-fast"),
    fail_if_any_failed: bool = typer.Option(
        True,
        "--fail-if-any-failed/--allow-failed",
        help="Exit non-zero when any sweep step fails.",
    ),
    checkpoint_file: Path | None = typer.Option(
        None,
        "--checkpoint-file",
        resolve_path=True,
        help="Optional checkpoint JSON path for resume-safe sweep execution.",
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume from --checkpoint-file."),
    qa_json_out: Path | None = typer.Option(
        None,
        "--qa-json-out",
        resolve_path=True,
        help="Summary JSON output path (default: <out_dir>/morph_sweep_summary.json).",
    ),
    qa_csv_out: Path | None = typer.Option(
        None,
        "--qa-csv-out",
        resolve_path=True,
        help="Per-step QA metrics CSV path (default: <out_dir>/morph_sweep_metrics.csv).",
    ),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    _forward("_ir_morph_sweep_impl", dict(locals()))


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
    _forward("_ir_fit_impl", dict(locals()))
