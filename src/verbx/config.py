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


@dataclass(slots=True)
class RenderConfig:
    """Typed render configuration used by CLI and pipeline.

    Centralizing options in one dataclass reduces drift between CLI parsing,
    validation, and DSP pipeline behavior.
    """

    def __post_init__(self) -> None:
        """Validate field constraints that would cause silent corruption or crashes."""
        if self.algo_model not in {"fdn", "spring", "plate"}:
            raise ValueError(f"algo_model must be fdn, spring, or plate, got {self.algo_model}")
        if self.rt60 < 0.0:
            raise ValueError(f"rt60 must be >= 0, got {self.rt60}")
        if self.fdn_lines < 1:
            raise ValueError(f"fdn_lines must be >= 1, got {self.fdn_lines}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.partition_size < 1:
            raise ValueError(f"partition_size must be >= 1, got {self.partition_size}")
        if self.target_sr is not None and int(self.target_sr) < 1:
            raise ValueError(f"target_sr must be >= 1, got {self.target_sr}")
        if self.tail_stop_hold_ms < 0.0:
            raise ValueError(f"tail_stop_hold_ms must be >= 0, got {self.tail_stop_hold_ms}")
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
        if self.algo_proxy_ir_max_seconds <= 0.0:
            raise ValueError(
                f"algo_proxy_ir_max_seconds must be > 0, got {self.algo_proxy_ir_max_seconds}"
            )
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
