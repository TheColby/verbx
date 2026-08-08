"""Typed configuration models shared across CLI and pipeline layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from verbx.core.control_targets import RT60_DEFAULT_SECONDS

EngineName = Literal["conv", "algo", "ism-fdn", "auto"]
AlgoModel = Literal["fdn", "spring", "plate"]
ElectromechanicalSolver = Literal["proxy", "modal-fe"]
IRNormalize = Literal["peak", "rms", "none"]
NormalizeStage = Literal["none", "post", "per-pass"]
IRMode = Literal["fdn", "stochastic", "modal", "hybrid"]
IRMatrixLayout = Literal["output-major", "input-major"]
DeviceName = Literal["auto", "cpu", "cuda", "mps"]
OutputSubtype = Literal["auto", "float32", "float64", "pcm16", "pcm24", "pcm32"]
OutputContainer = Literal["auto", "wav", "w64", "rf64"]
ChannelLayout = Literal[
    "auto",
    "mono",
    "stereo",
    "LCR",
    "5.1",
    "7.1",
    "7.1.2",
    "7.1.4",
    "7.2.4",
    "8.0",
    "16.0",
    "64.4",
]
OutputPeakNorm = Literal["none", "input", "target", "full-scale"]
ModTarget = Literal["none", "mix", "wet", "gain-db"]
ModCombine = Literal["sum", "avg", "max"]
AmbiNormalization = Literal["auto", "sn3d", "n3d", "fuma"]
AmbiChannelOrder = Literal["auto", "acn", "fuma"]
AmbiEncodeFrom = Literal["none", "mono", "stereo"]
AmbiDecodeTo = Literal["none", "stereo"]
AutomationMode = Literal["auto", "sample", "block"]
FeatureGuidePolicy = Literal["align", "strict"]
IRMorphMismatchPolicy = Literal["coerce", "strict"]
FDNSpatialCouplingMode = Literal["none", "adjacent", "front_rear", "bed_top", "all_to_all"]
FDNNonlinearityMode = Literal["none", "tanh", "softclip"]
TailStopMetric = Literal["peak", "rms"]
AutoFitProfile = Literal["none", "speech", "music", "drums", "ambient"]
LimiterMode = Literal["tanh", "arctan", "softsign", "hard"]
LimiterDetect = Literal["peak", "rms"]


@dataclass(frozen=True, slots=True)
class EngineSettings:
    """Core engine and mix settings extracted from :class:`RenderConfig`."""

    engine: EngineName
    algo_model: AlgoModel
    rt60: float
    pre_delay_ms: float
    damping: float
    width: float
    wet: float
    dry: float
    repeat: int
    freeze: bool

    def __post_init__(self) -> None:
        if self.algo_model not in {"fdn", "spring", "plate"}:
            raise ValueError(f"algo_model must be fdn, spring, or plate, got {self.algo_model}")
        if self.rt60 < 0.0:
            raise ValueError(f"rt60 must be >= 0, got {self.rt60}")
        if self.pre_delay_ms < 0.0:
            raise ValueError(f"pre_delay_ms must be >= 0, got {self.pre_delay_ms}")
        if not 0.0 <= self.damping <= 1.0:
            raise ValueError(f"damping must be 0-1, got {self.damping}")
        if self.wet < 0.0:
            raise ValueError(f"wet must be >= 0, got {self.wet}")
        if self.dry < 0.0:
            raise ValueError(f"dry must be >= 0, got {self.dry}")
        if self.repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {self.repeat}")


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    """Buffering and compute-backend settings for a render."""

    block_size: int
    partition_size: int
    target_sr: int | None
    threads: int | None
    device: DeviceName
    algo_stream: bool
    algo_proxy_ir_max_seconds: float
    algo_gpu_proxy: bool

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.partition_size < 1:
            raise ValueError(f"partition_size must be >= 1, got {self.partition_size}")
        if self.target_sr is not None and int(self.target_sr) < 1:
            raise ValueError(f"target_sr must be >= 1, got {self.target_sr}")
        if self.algo_proxy_ir_max_seconds <= 0.0:
            raise ValueError(
                f"algo_proxy_ir_max_seconds must be > 0, got {self.algo_proxy_ir_max_seconds}"
            )


@dataclass(frozen=True, slots=True)
class TailSettings:
    """Tail truncation and silence-detection settings."""

    limit: float | None
    stop_threshold_db: float
    stop_hold_ms: float
    stop_metric: TailStopMetric

    def __post_init__(self) -> None:
        if self.stop_hold_ms < 0.0:
            raise ValueError(f"tail_stop_hold_ms must be >= 0, got {self.stop_hold_ms}")


@dataclass(frozen=True, slots=True)
class OutputSettings:
    """Output encoding, normalization, and limiter settings."""

    subtype: OutputSubtype
    container: OutputContainer
    peak_norm: OutputPeakNorm
    peak_target_dbfs: float | None
    target_lufs: float | None
    target_peak_dbfs: float | None
    use_true_peak: bool
    limiter: bool
    limiter_mode: LimiterMode
    limiter_detect: LimiterDetect
    limiter_threshold_dbfs: float | None
    limiter_ceiling_dbfs: float | None
    limiter_knee_db: float
    limiter_drive: float
    limiter_mix: float
    limiter_attack_ms: float
    limiter_release_ms: float
    limiter_lookahead_ms: float
    limiter_stereo_link: bool
    limiter_oversample: int
    limiter_pre_gain_db: float
    limiter_post_gain_db: float
    limiter_dc_block: bool
    normalize_stage: NormalizeStage

    def __post_init__(self) -> None:
        if self.limiter_knee_db < 0.0:
            raise ValueError(f"limiter_knee_db must be >= 0, got {self.limiter_knee_db}")
        if self.limiter_drive <= 0.0:
            raise ValueError(f"limiter_drive must be > 0, got {self.limiter_drive}")
        if not 0.0 <= self.limiter_mix <= 1.0:
            raise ValueError(f"limiter_mix must be 0-1, got {self.limiter_mix}")
        if self.limiter_attack_ms < 0.0:
            raise ValueError(f"limiter_attack_ms must be >= 0, got {self.limiter_attack_ms}")
        if self.limiter_release_ms < 0.0:
            raise ValueError(f"limiter_release_ms must be >= 0, got {self.limiter_release_ms}")
        if self.limiter_lookahead_ms < 0.0:
            raise ValueError(f"limiter_lookahead_ms must be >= 0, got {self.limiter_lookahead_ms}")
        if self.limiter_oversample < 1:
            raise ValueError(f"limiter_oversample must be >= 1, got {self.limiter_oversample}")


@dataclass(frozen=True, slots=True)
class FDNConfig:
    """Feedback-delay-network controls extracted from :class:`RenderConfig`.

    The flat fields remain the public constructor and serialization format.  This
    immutable view gives DSP code a cohesive configuration boundary while the
    migration happens incrementally.
    """

    lines: int
    matrix: str
    tv_rate_hz: float
    tv_depth: float
    tv_seed: int
    dfm_delays_ms: tuple[float, ...]
    sparse: bool
    sparse_degree: int
    cascade: bool
    cascade_mix: float
    cascade_delay_scale: float
    cascade_rt60_ratio: float
    rt60_low: float | None
    rt60_mid: float | None
    rt60_high: float | None
    rt60_tilt: float
    tonal_correction_strength: float
    xover_low_hz: float
    xover_high_hz: float
    link_filter: str
    link_filter_hz: float
    link_filter_mix: float
    graph_topology: str
    graph_degree: int
    graph_seed: int
    matrix_morph_to: str | None
    matrix_morph_seconds: float
    spatial_coupling_mode: FDNSpatialCouplingMode
    spatial_coupling_strength: float
    nonlinearity: FDNNonlinearityMode
    nonlinearity_amount: float
    nonlinearity_drive: float

    def __post_init__(self) -> None:
        if self.lines < 1:
            raise ValueError(f"fdn_lines must be >= 1, got {self.lines}")
        if self.sparse_degree < 1:
            raise ValueError(f"fdn_sparse_degree must be >= 1, got {self.sparse_degree}")
        if self.matrix_morph_seconds < 0.0:
            raise ValueError(
                "fdn_matrix_morph_seconds must be >= 0, "
                f"got {self.matrix_morph_seconds}"
            )


@dataclass(frozen=True, slots=True)
class AutomationConfig:
    """Modulation, automation, and feature-guide controls."""

    mod_depth_ms: float
    mod_rate_hz: float
    mod_target: ModTarget
    mod_sources: tuple[str, ...]
    mod_routes: tuple[str, ...]
    mod_min: float
    mod_max: float
    mod_combine: ModCombine
    mod_smooth_ms: float
    file: str | None
    mode: AutomationMode
    block_ms: float
    smoothing_ms: float
    slew_limit_per_s: float | None
    deadband: float
    clamp: tuple[str, ...]
    points: tuple[str, ...]
    trace_out: str | None
    feature_vector_lanes: tuple[str, ...]
    feature_vector_frame_ms: float
    feature_vector_hop_ms: float
    feature_guide: str | None
    feature_guide_policy: FeatureGuidePolicy
    feature_vector_trace_out: str | None


@dataclass(frozen=True, slots=True)
class SpatialConfig:
    """Channel-layout, ambisonic, and geometric early-reflection controls."""

    input_layout: ChannelLayout
    output_layout: ChannelLayout
    ambi_order: int
    ambi_normalization: AmbiNormalization
    channel_order: AmbiChannelOrder
    ambi_encode_from: AmbiEncodeFrom
    ambi_decode_to: AmbiDecodeTo
    ambi_rotate_yaw_deg: float
    er_geometry: bool
    ism_order: int
    er_room_dims_m: tuple[float, float, float]
    er_source_pos_m: tuple[float, float, float]
    er_listener_pos_m: tuple[float, float, float]
    er_absorption: float
    er_material: str

    def __post_init__(self) -> None:
        if self.ambi_order < 0:
            raise ValueError(f"ambi_order must be >= 0, got {self.ambi_order}")
        if not 0 <= self.ism_order <= 6:
            raise ValueError(f"ism_order must be 0..6, got {self.ism_order}")
        if not 0.0 <= self.er_absorption <= 0.99:
            raise ValueError(f"er_absorption must be 0..0.99, got {self.er_absorption}")
        if any(dim <= 0.0 for dim in self.er_room_dims_m):
            raise ValueError(
                "er_room_dims_m must be > 0 in all dimensions, "
                f"got {self.er_room_dims_m}"
            )


@dataclass(frozen=True, slots=True)
class StreamingConfig:
    """Streaming, buffer, range, and compute-backend controls."""

    start: float | None
    end: float | None
    block_size: int
    partition_size: int
    target_sr: int | None
    threads: int | None
    device: DeviceName
    algo_stream: bool
    algo_proxy_ir_max_seconds: float
    algo_gpu_proxy: bool


@dataclass(frozen=True, slots=True)
class RenderConfigSections:
    """Coherent snapshots of the most commonly consumed render settings."""

    engine: EngineSettings
    execution: ExecutionSettings
    tail: TailSettings
    output: OutputSettings
    fdn: FDNConfig
    automation: AutomationConfig
    spatial: SpatialConfig
    streaming: StreamingConfig


@dataclass(slots=True)
class RenderConfig:
    """Typed render configuration used by CLI and pipeline.

    Centralizing options in one dataclass reduces drift between CLI parsing,
    validation, and DSP pipeline behavior.
    """

    def __post_init__(self) -> None:
        """Validate field constraints that would cause silent corruption or crashes."""
        # Constructing the typed sections delegates their local invariants while
        # retaining the flat constructor used by the CLI and saved presets.
        _ = self.sections
        if self.fdn_lines < 1:
            raise ValueError(f"fdn_lines must be >= 1, got {self.fdn_lines}")
        if self.allpass_stages < 0:
            raise ValueError(f"allpass_stages must be >= 0, got {self.allpass_stages}")
        if not 0.0 <= self.allpass_gain <= 1.0:
            raise ValueError(f"allpass_gain must be 0-1, got {self.allpass_gain}")
        if self.beast_mode < 1:
            raise ValueError(f"beast_mode must be >= 1, got {self.beast_mode}")
        if self.comb_cloud_count < 1:
            raise ValueError(f"comb_cloud_count must be >= 1, got {self.comb_cloud_count}")
        if not 0.0 <= self.comb_cloud_feedback <= 0.95:
            raise ValueError(f"comb_cloud_feedback must be 0-0.95, got {self.comb_cloud_feedback}")
        if not 0.0 <= self.comb_cloud_mix <= 1.0:
            raise ValueError(f"comb_cloud_mix must be 0-1, got {self.comb_cloud_mix}")
        if self.ambi_order < 0:
            raise ValueError(f"ambi_order must be >= 0, got {self.ambi_order}")
        if not 0.0 <= self.shimmer_mix <= 1.0:
            raise ValueError(f"shimmer_mix must be 0-1, got {self.shimmer_mix}")
        shimmer_feedback_max = 1.25 if self.unsafe_self_oscillate else 0.98
        if not 0.0 <= self.shimmer_feedback <= shimmer_feedback_max:
            raise ValueError(
                f"shimmer_feedback must be 0-{shimmer_feedback_max}, got {self.shimmer_feedback}"
            )
        if self.unsafe_loop_gain <= 0.0:
            raise ValueError(f"unsafe_loop_gain must be > 0, got {self.unsafe_loop_gain}")
        if self.fdn_matrix_morph_seconds < 0.0:
            raise ValueError(
                f"fdn_matrix_morph_seconds must be >= 0, got {self.fdn_matrix_morph_seconds}"
            )
        if self.shimmer_spread_cents < 0.0:
            raise ValueError(f"shimmer_spread_cents must be >= 0, got {self.shimmer_spread_cents}")
        if self.shimmer_decorrelation_ms < 0.0:
            raise ValueError(
                f"shimmer_decorrelation_ms must be >= 0, got {self.shimmer_decorrelation_ms}"
            )
        if not 0.0 <= self.er_absorption <= 0.99:
            raise ValueError(f"er_absorption must be 0..0.99, got {self.er_absorption}")
        if not 0 <= self.ism_order <= 6:
            raise ValueError(f"ism_order must be 0..6, got {self.ism_order}")
        if not 1 <= self.spring_count <= 8:
            raise ValueError(f"spring_count must be 1..8, got {self.spring_count}")
        if len(self.spring_specs) > 8:
            raise ValueError("at most eight per-spring specifications are supported")
        if self.electromechanical_solver not in {"proxy", "modal-fe"}:
            raise ValueError("electromechanical_solver must be proxy or modal-fe")
        if not 4 <= self.spring_fe_nodes <= 128 or not 1 <= self.spring_fe_modes <= 128:
            raise ValueError("spring FE nodes must be 4..128 and modes must be 1..128")
        if not 0.0 <= self.spring_fe_coupling <= 1.0 or not 0.0 <= self.spring_fe_loss <= 2.0:
            raise ValueError("spring FE coupling must be 0..1 and loss must be 0..2")
        if not 4 <= self.plate_fe_nx <= 32 or not 4 <= self.plate_fe_ny <= 32:
            raise ValueError("plate FE grid dimensions must be 4..32")
        if not 1 <= self.plate_fe_modes <= 128 or not 0.0 <= self.plate_fe_loss <= 2.0:
            raise ValueError("plate FE modes must be 1..128 and loss must be 0..2")
        if self.plate_width_m <= 0.0 or self.plate_height_m <= 0.0:
            raise ValueError("plate dimensions must be > 0")
        if self.plate_thickness_mm <= 0.0 or self.plate_density_kg_m3 <= 0.0:
            raise ValueError("plate thickness and density must be > 0")
        if self.plate_youngs_gpa <= 0.0:
            raise ValueError("plate_youngs_gpa must be > 0")
        if not 0.0 <= self.plate_poisson_ratio < 0.5:
            raise ValueError("plate_poisson_ratio must be 0..0.5")
        if not 0.0 <= self.plate_pickup_x <= 1.0 or not 0.0 <= self.plate_pickup_y <= 1.0:
            raise ValueError("plate pickup coordinates must be 0..1")
        if any(dim <= 0.0 for dim in self.er_room_dims_m):
            raise ValueError(
                f"er_room_dims_m must be > 0 in all dimensions, got {self.er_room_dims_m}"
            )
        if self.fdn_sparse_degree < 1:
            raise ValueError(f"fdn_sparse_degree must be >= 1, got {self.fdn_sparse_degree}")
        if not 0.0 <= self.duck_strength <= 1.0:
            raise ValueError(f"duck_strength must be 0-1, got {self.duck_strength}")
        if not 0.0 <= self.duck_floor <= 1.0:
            raise ValueError(f"duck_floor must be 0-1, got {self.duck_floor}")
        if self.bloom_mix is not None and not 0.0 <= self.bloom_mix <= 1.0:
            raise ValueError(f"bloom_mix must be 0-1, got {self.bloom_mix}")
        if self.tilt_pivot_hz <= 0.0:
            raise ValueError(f"tilt_pivot_hz must be > 0, got {self.tilt_pivot_hz}")
        if self.lowcut_order < 1:
            raise ValueError(f"lowcut_order must be >= 1, got {self.lowcut_order}")
        if self.highcut_order < 1:
            raise ValueError(f"highcut_order must be >= 1, got {self.highcut_order}")

    engine: EngineName = "auto"
    algo_model: AlgoModel = "fdn"
    spring_count: int = 1
    spring_specs: tuple[str, ...] = ()
    electromechanical_solver: ElectromechanicalSolver = "proxy"
    spring_fe_nodes: int = 24
    spring_fe_modes: int = 24
    spring_fe_coupling: float = 0.08
    spring_fe_loss: float = 0.30
    plate_width_m: float = 1.8
    plate_height_m: float = 1.2
    plate_thickness_mm: float = 0.6
    plate_density_kg_m3: float = 7_850.0
    plate_youngs_gpa: float = 200.0
    plate_poisson_ratio: float = 0.29
    plate_tension_n: float = 0.0
    plate_pickup_x: float = 0.72
    plate_pickup_y: float = 0.38
    plate_fe_nx: int = 12
    plate_fe_ny: int = 8
    plate_fe_modes: int = 32
    plate_fe_loss: float = 0.24
    rt60: float = RT60_DEFAULT_SECONDS
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
    allpass_gains: tuple[float, ...] = ()
    allpass_delays_ms: tuple[float, ...] = ()
    comb_delays_ms: tuple[float, ...] = ()
    comb_cloud: bool = False
    comb_cloud_count: int = 24
    comb_cloud_feedback: float = 0.35
    comb_cloud_mix: float = 0.25
    comb_cloud_delays_ms: tuple[float, ...] = ()
    comb_cloud_seed: int = 2026
    fdn_lines: int = 8
    fdn_matrix: str = "hadamard"
    fdn_tv_rate_hz: float = 0.0
    fdn_tv_depth: float = 0.0
    fdn_tv_seed: int = 2026
    fdn_dfm_delays_ms: tuple[float, ...] = ()
    fdn_sparse: bool = False
    fdn_sparse_degree: int = 2
    fdn_cascade: bool = False
    fdn_cascade_mix: float = 0.35
    fdn_cascade_delay_scale: float = 0.5
    fdn_cascade_rt60_ratio: float = 0.55
    fdn_rt60_low: float | None = None
    fdn_rt60_mid: float | None = None
    fdn_rt60_high: float | None = None
    fdn_rt60_tilt: float = 0.0
    fdn_tonal_correction_strength: float = 0.0
    fdn_xover_low_hz: float = 250.0
    fdn_xover_high_hz: float = 4_000.0
    fdn_link_filter: str = "none"
    fdn_link_filter_hz: float = 2_500.0
    fdn_link_filter_mix: float = 1.0
    fdn_graph_topology: str = "ring"
    fdn_graph_degree: int = 2
    fdn_graph_seed: int = 2026
    fdn_matrix_morph_to: str | None = None
    fdn_matrix_morph_seconds: float = 0.0
    fdn_spatial_coupling_mode: FDNSpatialCouplingMode = "none"
    fdn_spatial_coupling_strength: float = 0.0
    fdn_nonlinearity: FDNNonlinearityMode = "none"
    fdn_nonlinearity_amount: float = 0.0
    fdn_nonlinearity_drive: float = 1.0
    room_size_macro: float = 0.0
    clarity_macro: float = 0.0
    warmth_macro: float = 0.0
    envelopment_macro: float = 0.0
    algo_decorrelation_front: float = 0.0
    algo_decorrelation_rear: float = 0.0
    algo_decorrelation_top: float = 0.0
    wet: float = 0.8
    dry: float = 0.2
    repeat: int = 1
    freeze: bool = False
    start: float | None = None
    end: float | None = None
    block_size: int = 4096
    ir: str | None = None
    ir_blend: tuple[str, ...] = ()
    ir_blend_mix: tuple[float, ...] = ()
    ir_blend_mode: str = "equal-power"
    ir_blend_early_ms: float = 80.0
    ir_blend_early_alpha: float | None = None
    ir_blend_late_alpha: float | None = None
    ir_blend_align_decay: bool = True
    ir_blend_phase_coherence: float = 0.75
    ir_blend_spectral_smooth_bins: int = 3
    ir_blend_mismatch_policy: IRMorphMismatchPolicy = "coerce"
    ir_blend_cache_dir: str = ".verbx_cache/ir_morph"
    ir_blend_base_ir: str | None = None
    ir_blend_composite_ir: str | None = None
    input_layout: ChannelLayout = "auto"
    output_layout: ChannelLayout = "auto"
    self_convolve: bool = False
    ir_normalize: IRNormalize = "peak"
    ir_matrix_layout: IRMatrixLayout = "output-major"
    ir_route_map: str = "auto"
    conv_route_start: str | None = None
    conv_route_end: str | None = None
    conv_route_curve: str = "equal-power"
    ambi_order: int = 0
    ambi_normalization: AmbiNormalization = "auto"
    channel_order: AmbiChannelOrder = "auto"
    ambi_encode_from: AmbiEncodeFrom = "none"
    ambi_decode_to: AmbiDecodeTo = "none"
    ambi_rotate_yaw_deg: float = 0.0
    tail_limit: float | None = None
    tail_stop_threshold_db: float = -120.0
    tail_stop_hold_ms: float = 10.0
    tail_stop_metric: TailStopMetric = "peak"
    threads: int | None = None
    device: DeviceName = "auto"
    algo_stream: bool = False
    algo_proxy_ir_max_seconds: float = 120.0
    algo_gpu_proxy: bool = False
    partition_size: int = 16_384
    target_sr: int | None = None
    ir_gen: bool = False
    ir_gen_mode: IRMode = "hybrid"
    ir_gen_length: float = 60.0
    ir_gen_seed: int = 0
    ir_gen_cache_dir: str = ".verbx_cache/irs"
    target_lufs: float | None = None
    target_peak_dbfs: float | None = None
    use_true_peak: bool = True
    limiter: bool = True
    limiter_mode: LimiterMode = "tanh"
    limiter_detect: LimiterDetect = "peak"
    limiter_threshold_dbfs: float | None = None
    limiter_ceiling_dbfs: float | None = None
    limiter_knee_db: float = 6.0
    limiter_drive: float = 1.0
    limiter_mix: float = 1.0
    limiter_attack_ms: float = 0.5
    limiter_release_ms: float = 80.0
    limiter_lookahead_ms: float = 1.5
    limiter_stereo_link: bool = True
    limiter_oversample: int = 2
    limiter_pre_gain_db: float = 0.0
    limiter_post_gain_db: float = 0.0
    limiter_dc_block: bool = False
    normalize_stage: NormalizeStage = "post"
    repeat_target_lufs: float | None = None
    repeat_target_peak_dbfs: float | None = None
    output_subtype: OutputSubtype = "auto"
    output_container: OutputContainer = "auto"
    output_peak_norm: OutputPeakNorm = "none"
    output_peak_target_dbfs: float | None = None
    shimmer: bool = False
    shimmer_semitones: float = 12.0
    shimmer_mix: float = 0.25
    shimmer_feedback: float = 0.35
    shimmer_highcut: float | None = 10_000.0
    shimmer_lowcut: float | None = 300.0
    shimmer_spatial: bool = False
    shimmer_spread_cents: float = 8.0
    shimmer_decorrelation_ms: float = 1.5
    auto_fit: AutoFitProfile = "none"
    er_geometry: bool = False
    ism_order: int = 1
    er_room_dims_m: tuple[float, float, float] = (10.0, 7.0, 3.0)
    er_source_pos_m: tuple[float, float, float] = (2.0, 2.0, 1.5)
    er_listener_pos_m: tuple[float, float, float] = (5.0, 3.5, 1.5)
    er_absorption: float = 0.35
    er_material: str = "studio"
    unsafe_self_oscillate: bool = False
    unsafe_loop_gain: float = 1.02
    duck: bool = False
    duck_attack: float = 20.0
    duck_release: float = 350.0
    duck_strength: float = 0.75
    duck_floor: float = 0.0
    bloom: float = 0.0
    bloom_mix: float | None = None
    lowcut: float | None = None
    lowcut_order: int = 2
    highcut: float | None = None
    highcut_order: int = 2
    tilt: float = 0.0
    tilt_pivot_hz: float = 1_000.0
    bpm: float | None = None
    pre_delay_note: str | None = None
    frames_out: str | None = None
    analysis_out: str | None = None
    automation_file: str | None = None
    automation_mode: AutomationMode = "auto"
    automation_block_ms: float = 20.0
    automation_smoothing_ms: float = 20.0
    automation_slew_limit_per_s: float | None = None
    automation_deadband: float = 0.0
    automation_clamp: tuple[str, ...] = ()
    automation_points: tuple[str, ...] = ()
    automation_trace_out: str | None = None
    feature_vector_lanes: tuple[str, ...] = ()
    feature_vector_frame_ms: float = 40.0
    feature_vector_hop_ms: float = 20.0
    feature_guide: str | None = None
    feature_guide_policy: FeatureGuidePolicy = "align"
    feature_vector_trace_out: str | None = None
    silent: bool = False
    progress: bool = True

    @property
    def engine_settings(self) -> EngineSettings:
        """Return an immutable snapshot of core engine and mix settings."""
        return EngineSettings(
            engine=self.engine,
            algo_model=self.algo_model,
            rt60=self.rt60,
            pre_delay_ms=self.pre_delay_ms,
            damping=self.damping,
            width=self.width,
            wet=self.wet,
            dry=self.dry,
            repeat=self.repeat,
            freeze=self.freeze,
        )

    @property
    def execution_settings(self) -> ExecutionSettings:
        """Return an immutable snapshot of buffering and backend settings."""
        return ExecutionSettings(
            block_size=self.block_size,
            partition_size=self.partition_size,
            target_sr=self.target_sr,
            threads=self.threads,
            device=self.device,
            algo_stream=self.algo_stream,
            algo_proxy_ir_max_seconds=self.algo_proxy_ir_max_seconds,
            algo_gpu_proxy=self.algo_gpu_proxy,
        )

    @property
    def tail_settings(self) -> TailSettings:
        """Return an immutable snapshot of tail handling settings."""
        return TailSettings(
            limit=self.tail_limit,
            stop_threshold_db=self.tail_stop_threshold_db,
            stop_hold_ms=self.tail_stop_hold_ms,
            stop_metric=self.tail_stop_metric,
        )

    @property
    def output_settings(self) -> OutputSettings:
        """Return an immutable snapshot of encoding and limiter settings."""
        return OutputSettings(
            subtype=self.output_subtype,
            container=self.output_container,
            peak_norm=self.output_peak_norm,
            peak_target_dbfs=self.output_peak_target_dbfs,
            target_lufs=self.target_lufs,
            target_peak_dbfs=self.target_peak_dbfs,
            use_true_peak=self.use_true_peak,
            limiter=self.limiter,
            limiter_mode=self.limiter_mode,
            limiter_detect=self.limiter_detect,
            limiter_threshold_dbfs=self.limiter_threshold_dbfs,
            limiter_ceiling_dbfs=self.limiter_ceiling_dbfs,
            limiter_knee_db=self.limiter_knee_db,
            limiter_drive=self.limiter_drive,
            limiter_mix=self.limiter_mix,
            limiter_attack_ms=self.limiter_attack_ms,
            limiter_release_ms=self.limiter_release_ms,
            limiter_lookahead_ms=self.limiter_lookahead_ms,
            limiter_stereo_link=self.limiter_stereo_link,
            limiter_oversample=self.limiter_oversample,
            limiter_pre_gain_db=self.limiter_pre_gain_db,
            limiter_post_gain_db=self.limiter_post_gain_db,
            limiter_dc_block=self.limiter_dc_block,
            normalize_stage=self.normalize_stage,
        )

    @property
    def fdn_settings(self) -> FDNConfig:
        """Return an immutable snapshot of FDN-specific controls."""
        return FDNConfig(
            lines=self.fdn_lines,
            matrix=self.fdn_matrix,
            tv_rate_hz=self.fdn_tv_rate_hz,
            tv_depth=self.fdn_tv_depth,
            tv_seed=self.fdn_tv_seed,
            dfm_delays_ms=self.fdn_dfm_delays_ms,
            sparse=self.fdn_sparse,
            sparse_degree=self.fdn_sparse_degree,
            cascade=self.fdn_cascade,
            cascade_mix=self.fdn_cascade_mix,
            cascade_delay_scale=self.fdn_cascade_delay_scale,
            cascade_rt60_ratio=self.fdn_cascade_rt60_ratio,
            rt60_low=self.fdn_rt60_low,
            rt60_mid=self.fdn_rt60_mid,
            rt60_high=self.fdn_rt60_high,
            rt60_tilt=self.fdn_rt60_tilt,
            tonal_correction_strength=self.fdn_tonal_correction_strength,
            xover_low_hz=self.fdn_xover_low_hz,
            xover_high_hz=self.fdn_xover_high_hz,
            link_filter=self.fdn_link_filter,
            link_filter_hz=self.fdn_link_filter_hz,
            link_filter_mix=self.fdn_link_filter_mix,
            graph_topology=self.fdn_graph_topology,
            graph_degree=self.fdn_graph_degree,
            graph_seed=self.fdn_graph_seed,
            matrix_morph_to=self.fdn_matrix_morph_to,
            matrix_morph_seconds=self.fdn_matrix_morph_seconds,
            spatial_coupling_mode=self.fdn_spatial_coupling_mode,
            spatial_coupling_strength=self.fdn_spatial_coupling_strength,
            nonlinearity=self.fdn_nonlinearity,
            nonlinearity_amount=self.fdn_nonlinearity_amount,
            nonlinearity_drive=self.fdn_nonlinearity_drive,
        )

    @property
    def automation_settings(self) -> AutomationConfig:
        """Return an immutable snapshot of automation and modulation controls."""
        return AutomationConfig(
            mod_depth_ms=self.mod_depth_ms,
            mod_rate_hz=self.mod_rate_hz,
            mod_target=self.mod_target,
            mod_sources=self.mod_sources,
            mod_routes=self.mod_routes,
            mod_min=self.mod_min,
            mod_max=self.mod_max,
            mod_combine=self.mod_combine,
            mod_smooth_ms=self.mod_smooth_ms,
            file=self.automation_file,
            mode=self.automation_mode,
            block_ms=self.automation_block_ms,
            smoothing_ms=self.automation_smoothing_ms,
            slew_limit_per_s=self.automation_slew_limit_per_s,
            deadband=self.automation_deadband,
            clamp=self.automation_clamp,
            points=self.automation_points,
            trace_out=self.automation_trace_out,
            feature_vector_lanes=self.feature_vector_lanes,
            feature_vector_frame_ms=self.feature_vector_frame_ms,
            feature_vector_hop_ms=self.feature_vector_hop_ms,
            feature_guide=self.feature_guide,
            feature_guide_policy=self.feature_guide_policy,
            feature_vector_trace_out=self.feature_vector_trace_out,
        )

    @property
    def spatial_settings(self) -> SpatialConfig:
        """Return an immutable snapshot of spatial and geometry controls."""
        return SpatialConfig(
            input_layout=self.input_layout,
            output_layout=self.output_layout,
            ambi_order=self.ambi_order,
            ambi_normalization=self.ambi_normalization,
            channel_order=self.channel_order,
            ambi_encode_from=self.ambi_encode_from,
            ambi_decode_to=self.ambi_decode_to,
            ambi_rotate_yaw_deg=self.ambi_rotate_yaw_deg,
            er_geometry=self.er_geometry,
            ism_order=self.ism_order,
            er_room_dims_m=self.er_room_dims_m,
            er_source_pos_m=self.er_source_pos_m,
            er_listener_pos_m=self.er_listener_pos_m,
            er_absorption=self.er_absorption,
            er_material=self.er_material,
        )

    @property
    def streaming_settings(self) -> StreamingConfig:
        """Return an immutable snapshot of streaming and execution controls."""
        return StreamingConfig(
            start=self.start,
            end=self.end,
            block_size=self.block_size,
            partition_size=self.partition_size,
            target_sr=self.target_sr,
            threads=self.threads,
            device=self.device,
            algo_stream=self.algo_stream,
            algo_proxy_ir_max_seconds=self.algo_proxy_ir_max_seconds,
            algo_gpu_proxy=self.algo_gpu_proxy,
        )

    @property
    def sections(self) -> RenderConfigSections:
        """Return typed snapshots without changing the flat serialization contract."""
        return RenderConfigSections(
            engine=self.engine_settings,
            execution=self.execution_settings,
            tail=self.tail_settings,
            output=self.output_settings,
            fdn=self.fdn_settings,
            automation=self.automation_settings,
            spatial=self.spatial_settings,
            streaming=self.streaming_settings,
        )
