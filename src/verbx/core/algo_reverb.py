"""Algorithmic reverb engine built around diffusion + an FDN late field.

Design goals for this implementation:

- stay stable at very long decay times (extreme RT60 settings),
- process in blocks for large files without state resets,
- remain deterministic and easy to extend with new feedback/matrix models.

Signal flow (per channel):

1. pre-delay line,
2. short all-pass diffusion network,
3. FDN late field with configurable delay-line count/lengths and:
   - RT60-calibrated per-line gains,
   - one-pole damping in each feedback path,
   - DC blocking in the loop,
   - subtle delay modulation to reduce metallic ringing.
4. optional stereo width stage (for 2ch),
5. optional shimmer stage on the wet path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, TypeVar

import numpy as np
import numpy.typing as npt

from verbx.core.control_targets import (
    ENGINE_CONTROL_TARGETS,
    RT60_MAX_SECONDS,
    RT60_MIN_SECONDS,
    normalize_control_target_name,
)
from verbx.core.engine_base import ReverbEngine
from verbx.core.fdn_capabilities import (
    FDN_LINK_FILTER_CHOICES,
    normalize_fdn_graph_topology_name,
    normalize_fdn_link_filter_name,
    normalize_fdn_matrix_name,
)
from verbx.core.shimmer import ShimmerConfig, ShimmerProcessor
from verbx.io.audio import ensure_mono_or_stereo

AudioArray = npt.NDArray[np.float64]
CurveMap = dict[str, npt.NDArray[np.float64]]

try:
    from numba import njit  # type: ignore[import-untyped]

    _numba_available = True
except Exception:  # pragma: no cover
    njit = None
    _numba_available = False

F = TypeVar("F", bound=Callable[..., object])


def _maybe_njit(func: F) -> F:
    """Wrap with numba.njit if it's installed, otherwise it's a no-op."""
    if njit is None:
        return func
    return njit(cache=True, fastmath=True)(func)  # type: ignore[return-value,no-any-return]


@dataclass(slots=True)
class AlgoReverbConfig:
    """Configuration for the algorithmic reverb engine."""

    rt60: float = 60.0
    pre_delay_ms: float = 20.0
    damping: float = 0.45
    width: float = 1.0
    mod_depth_ms: float = 2.0
    mod_rate_hz: float = 0.1
    allpass_stages: int = 6
    allpass_gain: float = 0.7
    allpass_gains: tuple[float, ...] = ()
    allpass_delays_ms: tuple[float, ...] = ()
    comb_delays_ms: tuple[float, ...] = ()
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
    fdn_spatial_coupling_mode: str = "none"
    fdn_spatial_coupling_strength: float = 0.0
    fdn_nonlinearity: str = "none"
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
    block_size: int = 4096
    shimmer: bool = False
    shimmer_semitones: float = 12.0
    shimmer_mix: float = 0.25
    shimmer_feedback: float = 0.35
    shimmer_highcut: float | None = 10_000.0
    shimmer_lowcut: float | None = 300.0
    output_layout: str = "auto"
    device: str = "cpu"


@dataclass(slots=True)
class _AllpassState:
    """Mutable state for a single Schroeder all-pass section."""

    buffer: AudioArray
    index: int = 0


class AlgoReverbEngine(ReverbEngine):
    """Block-processed Schroeder + FDN algorithmic reverb.

    The core late reverb is intentionally conservative:
    fixed-size topology, bounded gains, and explicit state safety scaling.
    That makes it robust for long tails while still sounding dense enough for
    cinematic/"frozen-time" use cases.
    """

    # primes help avoid common subharmonics between delay lines — classic Schroeder advice
    _DEFAULT_BASE_DELAY_MS = np.array(
        [31.0, 37.0, 41.0, 43.0, 47.0, 53.0, 59.0, 67.0],
        dtype=np.float64,
    )
    _DEFAULT_DIFFUSION_DELAY_MS = np.array(
        [5.0, 7.0, 11.0, 17.0, 23.0, 29.0],
        dtype=np.float64,
    )
    _AUTOMATION_TARGETS: ClassVar[frozenset[str]] = frozenset(ENGINE_CONTROL_TARGETS)
    _RT60_UPDATE_EPS = 1e-3
    _LP_ALPHA_UPDATE_EPS = 1e-5
    _TRACK_C_UPDATE_EPS = 1e-5
    _SPATIAL_COUPLING_MODES: ClassVar[frozenset[str]] = frozenset(
        {"none", "adjacent", "front_rear", "bed_top", "all_to_all"}
    )
    _NONLINEARITY_MODES: ClassVar[frozenset[str]] = frozenset({"none", "tanh", "softclip"})

    def __init__(self, config: AlgoReverbConfig) -> None:
        self._config = config
        self._base_delay_ms = self._resolve_fdn_delay_ms(config)
        self._diffusion_delay_ms = self._resolve_diffusion_delay_ms(config)
        self._dfm_delay_ms = self._resolve_dfm_delay_ms(config, int(self._base_delay_ms.shape[0]))
        self._allpass_gains = self._resolve_allpass_gains(
            config,
            self._diffusion_delay_ms.shape[0],
        )
        self._cascade_enabled = bool(config.fdn_cascade)
        # TODO: cascade path needs more thorough testing at extreme RT60 values — gains can drift
        self._cascade_mix = float(np.clip(config.fdn_cascade_mix, 0.0, 1.0))
        self._cascade_delay_scale = float(np.clip(config.fdn_cascade_delay_scale, 0.2, 1.0))
        self._cascade_rt60_ratio = float(np.clip(config.fdn_cascade_rt60_ratio, 0.1, 1.0))
        self._sparse_enabled = bool(config.fdn_sparse)
        self._sparse_degree = max(1, int(config.fdn_sparse_degree))
        self._rt60_tilt = float(np.clip(config.fdn_rt60_tilt, -1.0, 1.0))
        # This condition is a bit tangled — multiband kicks in either when all three
        # RT60 bands are set explicitly, OR when tilt/warmth/clarity macros are active.
        self._base_multiband_enabled = all(
            value is not None
            for value in (config.fdn_rt60_low, config.fdn_rt60_mid, config.fdn_rt60_high)
        ) or (
            abs(self._rt60_tilt) > 1e-6
            or abs(float(config.warmth_macro)) > 1e-6
            or abs(float(config.clarity_macro)) > 1e-6
        )
        self._multiband_enabled = self._base_multiband_enabled
        self._link_filter_mode = self._resolve_link_filter_mode(config.fdn_link_filter)
        self._link_filter_mix = float(np.clip(config.fdn_link_filter_mix, 0.0, 1.0))
        self._link_filter_enabled = (
            self._link_filter_mode != "none"
            and self._link_filter_mix > 0.0
        )
        tv_seed = int(config.fdn_tv_seed)
        self._sparse_pairings = self._build_sparse_pairings(
            size=int(self._base_delay_ms.shape[0]),
            stages=self._sparse_degree,
            seed=tv_seed,
        )
        self._matrix_kind = self._normalize_matrix_type(config.fdn_matrix)
        self._spatial_coupling_mode = self._resolve_spatial_coupling_mode(
            config.fdn_spatial_coupling_mode
        )
        self._spatial_coupling_strength = float(
            np.clip(config.fdn_spatial_coupling_strength, 0.0, 1.0)
        )
        self._nonlinearity_mode = self._resolve_nonlinearity_mode(config.fdn_nonlinearity)
        self._nonlinearity_amount = float(np.clip(config.fdn_nonlinearity_amount, 0.0, 1.0))
        self._nonlinearity_drive = float(np.clip(config.fdn_nonlinearity_drive, 0.1, 8.0))
        self._nonlinearity_enabled = (
            self._nonlinearity_mode != "none" and self._nonlinearity_amount > 0.0
        )
        self._graph_enabled = self._matrix_kind == "graph"
        self._graph_pairings = self._build_graph_pairings(
            size=int(self._base_delay_ms.shape[0]),
            topology=config.fdn_graph_topology,
            degree=max(1, int(config.fdn_graph_degree)),
            seed=int(config.fdn_graph_seed),
        )
        self._fdn_matrix = self._build_fdn_matrix(
            int(self._base_delay_ms.shape[0]),
            config.fdn_matrix,
        )
        if self._sparse_enabled:
            self._fdn_matrix = self._build_sparse_mix_matrix(
                size=int(self._base_delay_ms.shape[0]),
                pairings=self._sparse_pairings,
            )
        elif self._graph_enabled:
            self._fdn_matrix = self._build_sparse_mix_matrix(
                size=int(self._base_delay_ms.shape[0]),
                pairings=self._graph_pairings,
            )
        self._tv_matrix_base_enabled = (
            self._matrix_kind == "tv_unitary"
            and float(config.fdn_tv_rate_hz) > 0.0
            and float(config.fdn_tv_depth) > 0.0
            and not self._sparse_enabled
            and not self._graph_enabled
            and not self._base_multiband_enabled
        )
        self._tv_matrix_enabled = self._tv_matrix_base_enabled
        self._tv_phase = np.float64(0.0)
        self._tv_target_matrix = self._build_random_orthogonal_matrix(
            size=int(self._base_delay_ms.shape[0]),
            seed=tv_seed,
        )
        self._tv_rng = np.random.default_rng(tv_seed)
        cascade_lines = int(np.clip(round(float(self._base_delay_ms.shape[0]) * 0.5), 1, 32))
        self._cascade_delay_ms = np.asarray(
            np.clip(self._base_delay_ms[:cascade_lines] * self._cascade_delay_scale, 2.0, None),
            dtype=np.float64,
        )
        cascade_matrix_kind = (
            "hadamard" if self._matrix_kind in {"tv_unitary", "graph"} else config.fdn_matrix
        )
        self._cascade_matrix = self._build_fdn_matrix(cascade_lines, cascade_matrix_kind)
        self._dfm_enabled = len(self._dfm_delay_ms) > 0
        self._use_numba = (
            _numba_available
            and config.device != "cuda"
            and not self._tv_matrix_enabled
            and not self._graph_enabled
            and not self._dfm_enabled
            and not self._multiband_enabled
            and not self._link_filter_enabled
            and not self._cascade_enabled
            and not self._nonlinearity_enabled
            and abs(float(config.room_size_macro)) <= 1e-9
            and abs(float(config.clarity_macro)) <= 1e-9
            and abs(float(config.warmth_macro)) <= 1e-9
            and abs(float(config.envelopment_macro)) <= 1e-9
        )
        self._shimmer = ShimmerProcessor(
            ShimmerConfig(
                enabled=config.shimmer,
                semitones=config.shimmer_semitones,
                mix=config.shimmer_mix,
                feedback=config.shimmer_feedback,
                highcut=config.shimmer_highcut,
                lowcut=config.shimmer_lowcut,
            )
        )
        self._parameter_automation: CurveMap = {}

    def set_parameter_automation(self, curves: CurveMap | None) -> None:
        """Set sample-rate automation curves for internal FDN parameters."""
        if curves is None:
            self._parameter_automation = {}
            self._multiband_enabled = self._base_multiband_enabled
            self._tv_matrix_enabled = self._tv_matrix_base_enabled
            return
        parsed: CurveMap = {}
        for raw_target, raw_curve in curves.items():
            target = normalize_control_target_name(str(raw_target))
            if target not in self._AUTOMATION_TARGETS:
                continue
            vec = np.asarray(raw_curve, dtype=np.float64).reshape(-1)
            if vec.size == 0:
                continue
            parsed[target] = vec
        self._parameter_automation = parsed
        self._multiband_enabled = (
            self._base_multiband_enabled
            or self._automation_requires_multiband(parsed)
        )
        self._tv_matrix_enabled = self._tv_matrix_base_enabled and not self._multiband_enabled

    @staticmethod
    def _automation_requires_multiband(curves: CurveMap) -> bool:
        """Return True when automation targets imply multiband Track C processing."""
        return any(
            target in curves
            for target in (
                "fdn-rt60-tilt",
                "clarity-macro",
                "warmth-macro",
                "fdn-tonal-correction-strength",
            )
        )

    @staticmethod
    def _resample_curve(curve: npt.NDArray[np.float64], n_samples: int) -> npt.NDArray[np.float64]:
        if curve.shape[0] == n_samples:
            return np.asarray(curve, dtype=np.float64)
        if curve.shape[0] <= 1:
            value = float(curve[0]) if curve.shape[0] == 1 else 0.0
            return np.full((n_samples,), value, dtype=np.float64)
        src = np.linspace(0.0, 1.0, curve.shape[0], dtype=np.float64)
        dst = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
        return np.asarray(np.interp(dst, src, curve.astype(np.float64)), dtype=np.float64)

    def _resolve_parameter_automation(self, n_samples: int) -> CurveMap:
        resolved: CurveMap = {}
        if n_samples <= 0:
            return resolved
        for target, curve in self._parameter_automation.items():
            vec = self._resample_curve(curve, n_samples)
            if target == "rt60":
                vec = np.asarray(np.clip(vec, RT60_MIN_SECONDS, RT60_MAX_SECONDS), dtype=np.float64)
            elif target == "damping":
                vec = np.asarray(np.clip(vec, 0.0, 1.0), dtype=np.float64)
            elif target == "room-size":
                vec = np.asarray(np.clip(vec, 0.25, 4.0), dtype=np.float64)
            elif target in {
                "room-size-macro",
                "clarity-macro",
                "warmth-macro",
                "envelopment-macro",
                "fdn-rt60-tilt",
            }:
                vec = np.asarray(np.clip(vec, -1.0, 1.0), dtype=np.float64)
            elif target == "fdn-tonal-correction-strength":
                vec = np.asarray(np.clip(vec, 0.0, 1.0), dtype=np.float64)
            resolved[target] = vec
        return resolved

    def process(self, audio: AudioArray, sr: int) -> AudioArray:
        """Process audio with pre-diffusion + late FDN and wet/dry mix."""
        x = ensure_mono_or_stereo(audio)
        n_samples, n_channels = x.shape
        if n_samples == 0:
            return x.copy()

        param_automation = self._resolve_parameter_automation(n_samples)
        wet = np.zeros_like(x, dtype=np.float64)
        for channel in range(n_channels):
            wet[:, channel] = self._process_channel(
                x[:, channel],
                sr,
                parameter_automation=param_automation,
            )

        wet = self._apply_multichannel_decorrelation(wet, sr)
        wet = self._apply_spatial_coupling(wet)

        if n_channels == 2 and self._config.width != 1.0:
            wet = self._apply_stereo_width(wet, self._config.width)

        if self._config.shimmer:
            wet = self._shimmer.process(wet, sr)

        output = (self._config.dry * x) + (self._config.wet * wet)
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

        peak = float(np.max(np.abs(output)))
        if peak > 8.0:
            output *= 8.0 / peak

        return np.asarray(output, dtype=np.float64)

    def _apply_multichannel_decorrelation(self, wet: AudioArray, sr: int) -> AudioArray:
        """Apply lightweight channel-group decorrelation for surround layouts."""
        channels = int(wet.shape[1])
        if channels <= 2:
            return wet

        front_v = float(np.clip(self._config.algo_decorrelation_front, 0.0, 1.0))
        rear_v = float(np.clip(self._config.algo_decorrelation_rear, 0.0, 1.0))
        top_v = float(np.clip(self._config.algo_decorrelation_top, 0.0, 1.0))
        if max(front_v, rear_v, top_v) <= 0.0:
            return wet

        out = np.asarray(wet.copy(), dtype=np.float64)
        rng = np.random.default_rng(int(self._config.fdn_tv_seed) + 8191)

        layout = self._config.output_layout.strip().lower()
        if layout == "auto":
            layout = {
                3: "lcr",
                6: "5.1",
                8: "7.1",
                10: "7.1.2",
                12: "7.1.4",
                13: "7.2.4",
                16: "16.0",
                68: "64.4",
            }.get(channels, "auto")

        front_idx, rear_idx, top_idx = self._layout_channel_groups(layout=layout, channels=channels)
        variance_by_channel = np.zeros((channels,), dtype=np.float64)
        for idx in front_idx:
            variance_by_channel[idx] = np.float64(max(variance_by_channel[idx], front_v))
        for idx in rear_idx:
            variance_by_channel[idx] = np.float64(max(variance_by_channel[idx], rear_v))
        for idx in top_idx:
            variance_by_channel[idx] = np.float64(max(variance_by_channel[idx], top_v))

        for ch in range(channels):
            variance = float(variance_by_channel[ch])
            if variance <= 0.0:
                continue
            jitter = float(rng.uniform(-1.0, 1.0))
            max_delay_ms = 2.0 + (18.0 * variance)
            delay_samples = max(1, int((max_delay_ms / 1000.0) * float(sr)))
            delay_samples = max(1, min(delay_samples, max(1, out.shape[0] // 4)))
            mix = float(np.clip((0.1 + (0.35 * variance)) + (0.05 * abs(jitter)), 0.05, 0.55))
            gain = float(np.clip((1.0 - (0.12 * variance)) + (0.06 * jitter), 0.75, 1.1))
            delayed = np.roll(out[:, ch], delay_samples)
            out[:, ch] = np.asarray(
                ((1.0 - mix) * out[:, ch] * gain) + (mix * delayed),
                dtype=np.float64,
            )

        return np.asarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)

    def _apply_spatial_coupling(self, wet: AudioArray) -> AudioArray:
        """Apply optional directional coupling for immersive wet-channel interaction."""
        channels = int(wet.shape[1])
        if channels <= 2:
            return wet
        strength = float(np.clip(self._spatial_coupling_strength, 0.0, 1.0))
        mode = self._spatial_coupling_mode
        if strength <= 0.0 or mode == "none":
            return wet

        out = np.asarray(wet, dtype=np.float64)
        coupled = np.asarray(out.copy(), dtype=np.float64)
        layout = self._config.output_layout.strip().lower()
        if layout == "auto":
            layout = {
                3: "lcr",
                6: "5.1",
                8: "7.1",
                10: "7.1.2",
                12: "7.1.4",
                13: "7.2.4",
                16: "16.0",
                68: "64.4",
            }.get(channels, "auto")
        front_idx, rear_idx, top_idx = self._layout_channel_groups(layout=layout, channels=channels)

        if mode == "adjacent":
            coupled[:, :] = 0.5 * (np.roll(out, 1, axis=1) + np.roll(out, -1, axis=1))
        elif mode == "all_to_all":
            total = np.sum(out, axis=1, keepdims=True)
            coupled[:, :] = (total - out) / float(max(1, channels - 1))
        elif mode == "front_rear":
            if len(front_idx) == 0 or len(rear_idx) == 0:
                return out
            front_mean = np.mean(out[:, front_idx], axis=1, keepdims=True)
            rear_mean = np.mean(out[:, rear_idx], axis=1, keepdims=True)
            for idx in front_idx:
                coupled[:, idx] = rear_mean[:, 0]
            for idx in rear_idx:
                coupled[:, idx] = front_mean[:, 0]
        elif mode == "bed_top":
            bed_idx = sorted({*front_idx, *rear_idx})
            if len(bed_idx) == 0 or len(top_idx) == 0:
                return out
            bed_mean = np.mean(out[:, bed_idx], axis=1, keepdims=True)
            top_mean = np.mean(out[:, top_idx], axis=1, keepdims=True)
            for idx in bed_idx:
                coupled[:, idx] = top_mean[:, 0]
            for idx in top_idx:
                coupled[:, idx] = bed_mean[:, 0]

        mixed = ((1.0 - strength) * out) + (strength * coupled)
        return np.asarray(np.nan_to_num(mixed, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)

    @staticmethod
    def _layout_channel_groups(
        layout: str,
        channels: int,
    ) -> tuple[list[int], list[int], list[int]]:
        """Return front/rear/top channel groups for common bus layouts."""
        if layout == "lcr":
            return [0, 1, 2], [], []
        if layout == "5.1":
            return [0, 1, 2, 3], [4, 5], []
        if layout == "7.1":
            return [0, 1, 2, 3], [4, 5, 6, 7], []
        if layout == "7.1.2":
            return [0, 1, 2, 3], [4, 5, 6, 7], [8, 9]
        if layout == "7.1.4":
            return [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
        if layout == "7.2.4":
            return [0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]
        if layout == "8.0":
            return [0, 1, 2], [3, 4, 5, 6, 7], []
        if layout == "16.0":
            return list(range(0, min(8, channels))), list(range(8, channels)), []
        if layout == "64.4":
            if channels >= 68:
                return list(range(0, 32)), list(range(32, 64)), list(range(64, 68))
            return list(range(channels)), [], []
        # Fallback: all channels treated as front group.
        return list(range(channels)), [], []

    def backend_name(self) -> str:
        """Return current algorithmic backend."""
        base = "cpu-numba-fdn" if self._use_numba else "cpu-python-fdn"
        suffixes: list[str] = []
        if self._sparse_enabled:
            suffixes.append("sparse")
        if self._graph_enabled:
            suffixes.append("graph")
        if self._cascade_enabled:
            suffixes.append("cascade")
        if self._multiband_enabled:
            suffixes.append("multiband")
        tonal_curve = self._parameter_automation.get("fdn-tonal-correction-strength")
        tonal_curve_enabled = tonal_curve is not None and float(np.max(tonal_curve)) > 1e-9
        if self._multiband_enabled and (
            float(self._config.fdn_tonal_correction_strength) > 0.0
            or tonal_curve_enabled
        ):
            suffixes.append("tonalcorr")
        if self._link_filter_enabled:
            suffixes.append("linkfilter")
        if self._spatial_coupling_mode != "none" and self._spatial_coupling_strength > 0.0:
            suffixes.append("spatialcouple")
        if self._nonlinearity_enabled:
            suffixes.append("nonlinear")
        if len(suffixes) == 0:
            return base
        return f"{base}-{'-'.join(suffixes)}"

    @staticmethod
    def _orthonormalize(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project matrix to the nearest orthonormal basis with deterministic QR."""
        q, _ = np.linalg.qr(np.asarray(matrix, dtype=np.float64))
        return np.asarray(q, dtype=np.float64)

    @classmethod
    def _build_hadamard_matrix(cls, size: int) -> npt.NDArray[np.float64]:
        """Build a deterministic Hadamard-derived orthonormal matrix."""
        matrix = np.array([[1.0]], dtype=np.float64)
        while matrix.shape[0] < size:
            matrix = np.block([[matrix, matrix], [matrix, -matrix]])
        matrix = matrix[:size, :size]
        # Truncating non-power-of-two Hadamards breaks strict orthogonality.
        return cls._orthonormalize(matrix)

    @staticmethod
    def _build_random_orthogonal_matrix(
        size: int,
        seed: int = 2026,
    ) -> npt.NDArray[np.float64]:
        """Build deterministic random orthonormal matrix with QR."""
        rng = np.random.default_rng(seed)
        base = rng.standard_normal((size, size)).astype(np.float64)
        q, _ = np.linalg.qr(base)
        return np.asarray(q, dtype=np.float64)

    @staticmethod
    def _build_shift_permutation(size: int, shift: int) -> npt.NDArray[np.float64]:
        """Build cyclic permutation matrix for circulant-style constructions."""
        matrix = np.zeros((size, size), dtype=np.float64)
        for row in range(size):
            col = (row - shift) % size
            matrix[row, col] = 1.0
        return matrix

    @classmethod
    def _build_circulant_matrix(cls, size: int) -> npt.NDArray[np.float64]:
        """Build a real orthogonal circulant matrix via unit-modulus spectrum."""
        spectrum = np.ones(size, dtype=np.complex128)
        for k in range(1, (size + 1) // 2):
            angle = (2.0 * np.pi * float(k * k)) / float(max(1, size))
            value = np.exp(1j * angle)
            spectrum[k] = value
            spectrum[-k] = np.conjugate(value)
        if size % 2 == 0:
            spectrum[size // 2] = -1.0 + 0.0j

        first_column = np.fft.ifft(spectrum).real.astype(np.float64)
        matrix = np.zeros((size, size), dtype=np.float64)
        for col in range(size):
            matrix[:, col] = np.roll(first_column, col)

        gram = matrix.T @ matrix
        if np.allclose(gram, np.eye(size, dtype=np.float64), atol=1e-4):
            return np.asarray(matrix, dtype=np.float64)
        return cls._orthonormalize(matrix)

    @classmethod
    def _build_elliptic_matrix(cls, size: int) -> npt.NDArray[np.float64]:
        """Build deterministic elliptic-inspired prototype and orthonormalize it."""
        eye = np.eye(size, dtype=np.float64)
        shift_1 = cls._build_shift_permutation(size, shift=1)
        shift_2 = cls._build_shift_permutation(size, shift=2)

        proto = (
            (0.62 * eye)
            + (0.19 * (shift_1 + shift_1.T))
            + (0.05 * (shift_2 + shift_2.T))
        )
        return cls._orthonormalize(proto)

    @classmethod
    def _build_sdn_hybrid_matrix(cls, size: int) -> npt.NDArray[np.float64]:
        """Build an SDN-inspired scattering matrix from pseudo-geometry."""
        if size <= 1:
            return np.eye(size, dtype=np.float64)
        idx = np.arange(size, dtype=np.float64)
        golden = np.pi * (3.0 - np.sqrt(5.0))
        z = 1.0 - (2.0 * (idx + 0.5) / float(size))
        radius = np.sqrt(np.maximum(0.0, 1.0 - (z * z)))
        theta = golden * idx
        coords = np.stack(
            (
                radius * np.cos(theta),
                radius * np.sin(theta),
                z,
            ),
            axis=1,
        ).astype(np.float64)

        matrix = np.eye(size, dtype=np.float64) * 0.58
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                dist = float(np.linalg.norm(coords[i] - coords[j]))
                weight = 0.42 / (1.0 + (4.0 * dist))
                sign = 1.0 if ((i + j) % 2 == 0) else -1.0
                matrix[i, j] = np.float64(sign * weight)
        return cls._orthonormalize(matrix)

    @staticmethod
    def _normalize_matrix_type(matrix_type: str) -> str:
        """Normalize user matrix string to lowercase identifier."""
        return normalize_fdn_matrix_name(matrix_type)

    @classmethod
    def _resolve_spatial_coupling_mode(cls, mode: str) -> str:
        """Normalize and validate spatial-coupling mode identifier."""
        normalized = str(mode).strip().lower().replace("-", "_")
        if normalized in cls._SPATIAL_COUPLING_MODES:
            return normalized
        msg = f"Unsupported FDN spatial coupling mode: {mode}"
        raise ValueError(msg)

    @classmethod
    def _resolve_nonlinearity_mode(cls, mode: str) -> str:
        """Normalize and validate in-loop nonlinearity mode identifier."""
        normalized = str(mode).strip().lower().replace("-", "_")
        if normalized in cls._NONLINEARITY_MODES:
            return normalized
        msg = f"Unsupported FDN nonlinearity mode: {mode}"
        raise ValueError(msg)

    def _apply_feedback_nonlinearity(
        self,
        values: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Apply bounded nonlinear feedback shaping for density/color experiments."""
        if not self._nonlinearity_enabled:
            return np.asarray(values, dtype=np.float64)
        drive = float(max(1e-6, self._nonlinearity_drive))
        amount = float(np.clip(self._nonlinearity_amount, 0.0, 1.0))
        driven = np.asarray(values * drive, dtype=np.float64)
        if self._nonlinearity_mode == "tanh":
            shaped = np.tanh(driven)
        else:
            shaped = driven / (1.0 + np.abs(driven))
        normalized = np.asarray(shaped / drive, dtype=np.float64)
        blended = ((1.0 - amount) * values) + (amount * normalized)
        return np.asarray(np.clip(blended, -32.0, 32.0), dtype=np.float64)

    @staticmethod
    def _normalize_graph_topology(topology: str) -> str:
        """Normalize graph topology string to lowercase identifier."""
        return normalize_fdn_graph_topology_name(topology)

    @staticmethod
    def _resolve_link_filter_mode(mode: str) -> str:
        """Normalize and validate in-matrix feedback link filter mode."""
        normalized = normalize_fdn_link_filter_name(mode)
        if normalized in FDN_LINK_FILTER_CHOICES:
            return normalized
        msg = f"Unsupported FDN link filter mode: {mode}"
        raise ValueError(msg)

    @staticmethod
    def _build_sparse_pairings(
        size: int,
        stages: int,
        seed: int,
    ) -> npt.NDArray[np.int32]:
        """Build deterministic pairwise mixing schedules for sparse FDN mode."""
        if size <= 1:
            return np.zeros((0, 0), dtype=np.int32)

        stage_count = max(1, int(stages))
        rng = np.random.default_rng(seed)
        pairings = np.zeros((stage_count, size), dtype=np.int32)
        for stage_idx in range(stage_count):
            pairings[stage_idx, :] = rng.permutation(size).astype(np.int32)
        return pairings

    @classmethod
    def _build_graph_pairings(
        cls,
        size: int,
        topology: str,
        degree: int,
        seed: int,
    ) -> npt.NDArray[np.int32]:
        """Build deterministic graph-constrained pairwise mixing schedules."""
        if size <= 1:
            return np.zeros((0, 0), dtype=np.int32)

        edges = cls._build_graph_edges(
            size=size,
            topology=topology,
            degree=degree,
            seed=seed,
        )
        if len(edges) == 0:
            return np.zeros((0, 0), dtype=np.int32)

        stage_count = max(1, int(degree))
        rng = np.random.default_rng(seed)
        pairings = np.zeros((stage_count, size), dtype=np.int32)
        edge_array = np.asarray(edges, dtype=np.int32)

        for stage_idx in range(stage_count):
            order = np.arange(edge_array.shape[0], dtype=np.int32)
            rng.shuffle(order)
            used = np.zeros((size,), dtype=bool)
            paired: list[int] = []
            for edge_idx in order:
                a = int(edge_array[edge_idx, 0])
                b = int(edge_array[edge_idx, 1])
                if used[a] or used[b]:
                    continue
                if rng.random() < 0.5:
                    paired.extend([a, b])
                else:
                    paired.extend([b, a])
                used[a] = True
                used[b] = True

            leftovers = [idx for idx in range(size) if not used[idx]]
            rng.shuffle(leftovers)
            paired.extend(leftovers)
            pairings[stage_idx, :] = np.asarray(paired[:size], dtype=np.int32)
        return pairings

    @classmethod
    def _build_graph_edges(
        cls,
        *,
        size: int,
        topology: str,
        degree: int,
        seed: int,
    ) -> list[tuple[int, int]]:
        """Build unique undirected edges for graph-structured FDN mode."""
        normalized = cls._normalize_graph_topology(topology)
        max_degree = max(1, min(int(degree), max(1, size - 1)))
        edges: set[tuple[int, int]] = set()

        if normalized == "star":
            center = 0
            for node in range(1, size):
                edges.add((center, node))
            return sorted(edges)

        if normalized == "path":
            for step in range(1, max_degree + 1):
                for start in range(0, size - step):
                    a = start
                    b = start + step
                    edges.add((a, b) if a < b else (b, a))
            return sorted(edges)

        if normalized == "random":
            all_pairs = [(a, b) for a in range(size) for b in range(a + 1, size)]
            if len(all_pairs) == 0:
                return []
            rng = np.random.default_rng(seed)
            rng.shuffle(all_pairs)
            target = min(len(all_pairs), max(size - 1, (size * max_degree) // 2))
            return sorted(all_pairs[:target])

        # Default "ring" behavior, including unknown values.
        for step in range(1, max_degree + 1):
            for node in range(size):
                a = node
                b = (node + step) % size
                edge = (a, b) if a < b else (b, a)
                if edge[0] != edge[1]:
                    edges.add(edge)
        return sorted(edges)

    @staticmethod
    def _apply_sparse_pair_mix(
        input_vec: npt.NDArray[np.float64],
        pairings: npt.NDArray[np.int32],
        out_vec: npt.NDArray[np.float64],
        scratch: npt.NDArray[np.float64],
    ) -> None:
        """Apply sparse orthogonal pair-mixing stages to feedback vector."""
        if pairings.size == 0:
            out_vec[:] = input_vec
            return

        out_vec[:] = input_vec
        size = int(out_vec.shape[0])
        inv_sqrt2 = np.float64(1.0 / np.sqrt(2.0))

        for stage_idx in range(pairings.shape[0]):
            scratch[:] = out_vec
            perm = pairings[stage_idx]
            for idx in range(0, size - 1, 2):
                a = int(perm[idx])
                b = int(perm[idx + 1])
                va = scratch[a]
                vb = scratch[b]
                out_vec[a] = (va + vb) * inv_sqrt2
                out_vec[b] = (va - vb) * inv_sqrt2
            if size % 2 == 1:
                last = int(perm[size - 1])
                out_vec[last] = scratch[last]

    @staticmethod
    def _build_sparse_mix_matrix(
        size: int,
        pairings: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.float64]:
        """Build dense matrix equivalent of sparse pair-mixing stages."""
        if size <= 0:
            return np.zeros((0, 0), dtype=np.float64)
        if pairings.size == 0:
            return np.eye(size, dtype=np.float64)

        matrix = np.eye(size, dtype=np.float64)
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        for stage_idx in range(pairings.shape[0]):
            perm = pairings[stage_idx]
            stage_matrix = np.eye(size, dtype=np.float64)
            for idx in range(0, size - 1, 2):
                a = int(perm[idx])
                b = int(perm[idx + 1])
                stage_matrix[a, a] = inv_sqrt2
                stage_matrix[a, b] = inv_sqrt2
                stage_matrix[b, a] = inv_sqrt2
                stage_matrix[b, b] = -inv_sqrt2
            matrix = stage_matrix @ matrix
        return np.asarray(matrix, dtype=np.float64)

    @classmethod
    def _build_fdn_matrix(cls, size: int, matrix_type: str) -> npt.NDArray[np.float64]:
        """Build an orthonormal mix matrix of the requested type."""
        if size <= 0:
            return np.zeros((0, 0), dtype=np.float64)

        kind = cls._normalize_matrix_type(matrix_type)
        if kind == "householder":
            v = np.ones((size, 1), dtype=np.float64)
            matrix = np.eye(size, dtype=np.float64) - ((2.0 / size) * (v @ v.T))
            return np.asarray(matrix, dtype=np.float64)

        if kind == "random_orthogonal":
            return cls._build_random_orthogonal_matrix(size=size, seed=2026)

        if kind == "circulant":
            return cls._build_circulant_matrix(size=size)

        if kind == "elliptic":
            return cls._build_elliptic_matrix(size=size)

        if kind == "sdn_hybrid":
            return cls._build_sdn_hybrid_matrix(size=size)

        if kind == "graph":
            pairings = cls._build_graph_pairings(
                size=size,
                topology="ring",
                degree=2,
                seed=2026,
            )
            return cls._build_sparse_mix_matrix(size=size, pairings=pairings)

        # "hadamard", "tv_unitary", and unknown values default here.
        return cls._build_hadamard_matrix(size=size)

    def _current_block_matrix(self, sr: int, block_samples: int) -> npt.NDArray[np.float64]:
        """Return the active FDN matrix for one processing block."""
        if not self._tv_matrix_enabled:
            return self._fdn_matrix

        block_seconds = float(block_samples) / float(max(1, sr))
        phase_inc = np.float64((2.0 * np.pi * float(self._config.fdn_tv_rate_hz)) * block_seconds)
        self._tv_phase += phase_inc
        while self._tv_phase >= np.float64(2.0 * np.pi):
            self._tv_phase -= np.float64(2.0 * np.pi)
            seed = int(self._tv_rng.integers(0, 2_147_483_647))
            self._tv_target_matrix = self._build_random_orthogonal_matrix(
                size=int(self._fdn_matrix.shape[0]),
                seed=seed,
            )

        depth = float(np.clip(self._config.fdn_tv_depth, 0.0, 1.0))
        blend = depth * (0.5 * (1.0 + np.sin(float(self._tv_phase))))
        candidate = ((1.0 - blend) * self._fdn_matrix) + (blend * self._tv_target_matrix)
        return self._orthonormalize(candidate.astype(np.float64))

    @classmethod
    def _resolve_fdn_delay_ms(cls, config: AlgoReverbConfig) -> npt.NDArray[np.float64]:
        """Resolve user-configured comb-like FDN delay lengths in milliseconds."""
        if len(config.comb_delays_ms) > 0:
            delays = [max(0.1, float(value)) for value in config.comb_delays_ms]
            return np.asarray(delays, dtype=np.float64)

        requested = max(1, int(config.fdn_lines))
        defaults = cls._DEFAULT_BASE_DELAY_MS.astype(np.float64).tolist()
        while len(defaults) < requested:
            next_delay = (defaults[-1] * 1.11) + 1.25
            if next_delay <= defaults[-1]:
                next_delay = defaults[-1] + 0.25
            defaults.append(next_delay)
        return np.asarray(defaults[:requested], dtype=np.float64)

    @staticmethod
    def _resolve_dfm_delay_ms(
        config: AlgoReverbConfig,
        line_count: int,
    ) -> npt.NDArray[np.float64]:
        """Resolve delay-feedback-matrix (DFM) delays for each FDN line."""
        if len(config.fdn_dfm_delays_ms) == 0:
            return np.zeros((0,), dtype=np.float64)

        delays = [max(0.05, float(value)) for value in config.fdn_dfm_delays_ms]
        if len(delays) == 1 and line_count > 1:
            delays = delays * line_count

        if len(delays) != line_count:
            msg = (
                "fdn_dfm_delays_ms length must be 1 or match FDN line count "
                f"({line_count}), got {len(delays)}"
            )
            raise ValueError(msg)

        return np.asarray(delays, dtype=np.float64)

    @staticmethod
    def _resolve_multiband_rt60(
        config: AlgoReverbConfig,
        *,
        clarity_macro: float | None = None,
        warmth_macro: float | None = None,
        fdn_rt60_tilt: float | None = None,
    ) -> tuple[float, float, float]:
        """Resolve low/mid/high RT60 values with scalar fallback."""
        base_rt60 = max(0.1, float(config.rt60))
        low = max(
            0.1,
            float(config.fdn_rt60_low if config.fdn_rt60_low is not None else base_rt60),
        )
        mid = max(
            0.1,
            float(config.fdn_rt60_mid if config.fdn_rt60_mid is not None else base_rt60),
        )
        high = max(
            0.1,
            float(config.fdn_rt60_high if config.fdn_rt60_high is not None else base_rt60),
        )
        resolved_clarity = (
            float(config.clarity_macro) if clarity_macro is None else float(clarity_macro)
        )
        resolved_warmth = (
            float(config.warmth_macro) if warmth_macro is None else float(warmth_macro)
        )
        resolved_tilt = (
            float(config.fdn_rt60_tilt) if fdn_rt60_tilt is None else float(fdn_rt60_tilt)
        )
        tilt = float(
            np.clip(
                resolved_tilt + (0.45 * resolved_warmth) - (0.18 * resolved_clarity),
                -1.0,
                1.0,
            )
        )
        if abs(tilt) > 1e-9:
            ratio = float(np.power(2.0, 0.85 * tilt))
            low = float(np.clip(low * ratio, RT60_MIN_SECONDS, RT60_MAX_SECONDS))
            high = float(np.clip(high / ratio, RT60_MIN_SECONDS, RT60_MAX_SECONDS))
        return low, mid, high

    @classmethod
    def resolve_multiband_rt60(
        cls,
        config: AlgoReverbConfig,
        *,
        clarity_macro: float | None = None,
        warmth_macro: float | None = None,
        fdn_rt60_tilt: float | None = None,
    ) -> tuple[float, float, float]:
        """Public wrapper for multiband RT60 resolution logic."""
        return cls._resolve_multiband_rt60(
            config,
            clarity_macro=clarity_macro,
            warmth_macro=warmth_macro,
            fdn_rt60_tilt=fdn_rt60_tilt,
        )

    @staticmethod
    def _one_pole_alpha(cutoff_hz: float, sr: int) -> np.float64:
        """Compute stable one-pole lowpass alpha from cutoff frequency."""
        fc = max(1.0, float(cutoff_hz))
        return np.float64(1.0 - np.exp((-2.0 * np.pi * fc) / float(max(1, sr))))

    @staticmethod
    def _macro_scale(
        macro_value: float,
        *,
        sensitivity: float,
        minimum: float,
        maximum: float,
    ) -> float:
        """Convert perceptual macro in ``[-1, 1]`` to positive scale factor."""
        scale = float(np.power(2.0, sensitivity * float(np.clip(macro_value, -1.0, 1.0))))
        return float(np.clip(scale, minimum, maximum))

    @staticmethod
    def _resolve_tonal_correction_scales(
        *,
        feedback_gain_low: npt.NDArray[np.float64],
        feedback_gain_mid: npt.NDArray[np.float64],
        feedback_gain_high: npt.NDArray[np.float64],
        strength: float,
    ) -> tuple[np.float64, np.float64, np.float64]:
        """Compute bounded low/mid/high correction scales for Track C tonal balancing."""
        tonal_strength = float(np.clip(strength, 0.0, 1.0))
        if tonal_strength <= 1e-9:
            return np.float64(1.0), np.float64(1.0), np.float64(1.0)

        eps = 1e-6
        low_mean = max(eps, float(np.mean(np.asarray(feedback_gain_low, dtype=np.float64))))
        mid_mean = max(eps, float(np.mean(np.asarray(feedback_gain_mid, dtype=np.float64))))
        high_mean = max(eps, float(np.mean(np.asarray(feedback_gain_high, dtype=np.float64))))
        low_scale = float(np.clip(np.power(mid_mean / low_mean, tonal_strength), 0.5, 2.0))
        mid_scale = 1.0
        high_scale = float(np.clip(np.power(mid_mean / high_mean, tonal_strength), 0.5, 2.0))

        # Preserve overall energy tendency while rebalancing decay color.
        rms = float(
            np.sqrt(
                (
                    (low_scale * low_scale)
                    + (mid_scale * mid_scale)
                    + (high_scale * high_scale)
                )
                / 3.0
            )
        )
        if rms > eps:
            low_scale /= rms
            mid_scale /= rms
            high_scale /= rms
        return np.float64(low_scale), np.float64(mid_scale), np.float64(high_scale)

    @staticmethod
    def resolve_tonal_correction_scales(
        *,
        feedback_gain_low: npt.NDArray[np.float64],
        feedback_gain_mid: npt.NDArray[np.float64],
        feedback_gain_high: npt.NDArray[np.float64],
        strength: float,
    ) -> tuple[np.float64, np.float64, np.float64]:
        """Public wrapper for tonal-correction scale calculation."""
        return AlgoReverbEngine._resolve_tonal_correction_scales(
            feedback_gain_low=feedback_gain_low,
            feedback_gain_mid=feedback_gain_mid,
            feedback_gain_high=feedback_gain_high,
            strength=strength,
        )

    @classmethod
    def _resolve_diffusion_delay_ms(cls, config: AlgoReverbConfig) -> npt.NDArray[np.float64]:
        """Resolve user-configured allpass diffusion delay lengths in milliseconds."""
        requested = max(0, int(config.allpass_stages))
        if requested == 0:
            return np.zeros((0,), dtype=np.float64)

        if len(config.allpass_delays_ms) > 0:
            delays = [max(0.1, float(value)) for value in config.allpass_delays_ms]
        else:
            delays = cls._DEFAULT_DIFFUSION_DELAY_MS.astype(np.float64).tolist()

        while len(delays) < requested:
            next_delay = (delays[-1] * 1.28) + 0.75
            if next_delay <= delays[-1]:
                next_delay = delays[-1] + 0.2
            delays.append(next_delay)
        return np.asarray(delays[:requested], dtype=np.float64)

    @staticmethod
    def _resolve_allpass_gains(
        config: AlgoReverbConfig,
        stage_count: int,
    ) -> npt.NDArray[np.float64]:
        """Resolve one gain per allpass stage with strict count checks."""
        if stage_count <= 0:
            if len(config.allpass_gains) > 0:
                msg = "allpass_gains requires at least one diffusion stage."
                raise ValueError(msg)
            return np.zeros((0,), dtype=np.float64)

        if len(config.allpass_gains) > 0:
            if len(config.allpass_gains) != stage_count:
                msg = (
                    "allpass_gains length must match resolved allpass stage count "
                    f"({stage_count}), got {len(config.allpass_gains)}"
                )
                raise ValueError(msg)
            gains = np.asarray(config.allpass_gains, dtype=np.float64)
            return np.asarray(np.clip(gains, -0.99, 0.99), dtype=np.float64)

        return np.full(
            (stage_count,),
            np.float64(np.clip(config.allpass_gain, -0.99, 0.99)),
            dtype=np.float64,
        )

    @staticmethod
    def _apply_stereo_width(wet: AudioArray, width: float) -> AudioArray:
        """Apply a simple mid/side width transform to the wet signal."""
        w = np.clip(width, 0.0, 2.0)
        mid = 0.5 * (wet[:, 0] + wet[:, 1])
        side = 0.5 * (wet[:, 0] - wet[:, 1])
        side *= w
        out = wet.copy()
        out[:, 0] = mid + side
        out[:, 1] = mid - side
        return np.asarray(out, dtype=np.float64)

    def _process_channel(
        self,
        signal: npt.NDArray[np.float64],
        sr: int,
        *,
        parameter_automation: CurveMap | None = None,
    ) -> npt.NDArray[np.float64]:
        """Run one channel through pre-delay, diffusion, and FDN late reverb."""
        automation = parameter_automation or {}
        rt60_curve = automation.get("rt60")
        damping_curve = automation.get("damping")
        room_size_curve = automation.get("room-size")
        room_size_macro_curve = automation.get("room-size-macro")
        clarity_macro_curve = automation.get("clarity-macro")
        warmth_macro_curve = automation.get("warmth-macro")
        envelopment_macro_curve = automation.get("envelopment-macro")
        fdn_rt60_tilt_curve = automation.get("fdn-rt60-tilt")
        tonal_correction_strength_curve = automation.get("fdn-tonal-correction-strength")
        multiband_active = self._multiband_enabled or self._automation_requires_multiband(
            automation
        )
        has_dynamic_params = (
            rt60_curve is not None
            or damping_curve is not None
            or room_size_curve is not None
            or room_size_macro_curve is not None
            or clarity_macro_curve is not None
            or warmth_macro_curve is not None
            or envelopment_macro_curve is not None
            or fdn_rt60_tilt_curve is not None
            or tonal_correction_strength_curve is not None
        )
        if self._use_numba and not has_dynamic_params:
            return _process_channel_kernel(
                signal=signal,
                sr=sr,
                rt60=np.float64(self._config.rt60),
                pre_delay_ms=np.float64(self._config.pre_delay_ms),
                damping=np.float64(self._config.damping),
                mod_depth_ms=np.float64(self._config.mod_depth_ms),
                mod_rate_hz=np.float64(self._config.mod_rate_hz),
                block_size=max(256, int(self._config.block_size)),
                fdn_matrix=self._fdn_matrix,
                base_delay_ms=self._base_delay_ms,
                diffusion_delay_ms=self._diffusion_delay_ms,
                allpass_gains=self._allpass_gains,
            )

        pre_delay_samples = max(1, int((self._config.pre_delay_ms / 1000.0) * sr))
        max_mod_samples = max(1, int((self._config.mod_depth_ms / 1000.0) * sr))

        line_delays = np.maximum(
            2,
            np.asarray(np.round((self._base_delay_ms / 1000.0) * sr), dtype=np.int32),
        )
        num_lines = int(line_delays.shape[0])

        diffusion_delays = np.maximum(
            1,
            np.asarray(np.round((self._diffusion_delay_ms / 1000.0) * sr), dtype=np.int32),
        )

        allpasses = [
            _AllpassState(buffer=np.zeros(delay + 1, dtype=np.float64))
            for delay in diffusion_delays
        ]

        delay_buffers = [
            np.zeros(delay + (2 * max_mod_samples) + 4, dtype=np.float64) for delay in line_delays
        ]
        write_indices = np.zeros(num_lines, dtype=np.int32)
        lp_state = np.zeros(num_lines, dtype=np.float64)
        dc_prev_in = np.zeros(num_lines, dtype=np.float64)
        dc_prev_out = np.zeros(num_lines, dtype=np.float64)
        dfm_indices = np.zeros(num_lines, dtype=np.int32)
        dfm_buffers: list[npt.NDArray[np.float64]] = []
        if self._dfm_enabled:
            dfm_delays = np.maximum(
                1,
                np.asarray(np.round((self._dfm_delay_ms / 1000.0) * sr), dtype=np.int32),
            )
            dfm_buffers = [np.zeros(delay + 1, dtype=np.float64) for delay in dfm_delays]

        base_phase = np.linspace(0.0, 2.0 * np.pi, num_lines, endpoint=False, dtype=np.float64)
        phase = base_phase.copy()

        base_rt60 = max(self._config.rt60, 0.1)
        base_damping = float(np.clip(self._config.damping, 0.0, 1.0))
        room_size_macro_default = float(np.clip(self._config.room_size_macro, -1.0, 1.0))
        clarity_macro_default = float(np.clip(self._config.clarity_macro, -1.0, 1.0))
        warmth_macro_default = float(np.clip(self._config.warmth_macro, -1.0, 1.0))
        envelopment_macro_default = float(np.clip(self._config.envelopment_macro, -1.0, 1.0))
        delays_sec = line_delays.astype(np.float64) / float(sr)
        feedback_gain = np.power(10.0, (-3.0 * delays_sec) / base_rt60).astype(np.float64)
        feedback_gain = np.clip(feedback_gain, 0.0, 0.995)
        fdn_rt60_tilt_default = float(np.clip(self._config.fdn_rt60_tilt, -1.0, 1.0))
        fdn_rt60_tilt = fdn_rt60_tilt_default
        rt60_low, rt60_mid, rt60_high = self._resolve_multiband_rt60(
            self._config,
            clarity_macro=clarity_macro_default,
            warmth_macro=warmth_macro_default,
            fdn_rt60_tilt=fdn_rt60_tilt,
        )
        feedback_gain_low = np.power(10.0, (-3.0 * delays_sec) / rt60_low).astype(np.float64)
        feedback_gain_mid = np.power(10.0, (-3.0 * delays_sec) / rt60_mid).astype(np.float64)
        feedback_gain_high = np.power(10.0, (-3.0 * delays_sec) / rt60_high).astype(np.float64)
        feedback_gain_low = np.clip(feedback_gain_low, 0.0, 0.995)
        feedback_gain_mid = np.clip(feedback_gain_mid, 0.0, 0.995)
        feedback_gain_high = np.clip(feedback_gain_high, 0.0, 0.995)
        tonal_correction_strength_default = float(
            np.clip(self._config.fdn_tonal_correction_strength, 0.0, 1.0)
        )
        tonal_correction_strength = tonal_correction_strength_default
        tonal_correction_enabled = multiband_active and tonal_correction_strength > 0.0
        tonal_low_scale = np.float64(1.0)
        tonal_mid_scale = np.float64(1.0)
        tonal_high_scale = np.float64(1.0)
        if tonal_correction_enabled:
            tonal_low_scale, tonal_mid_scale, tonal_high_scale = (
                self._resolve_tonal_correction_scales(
                    feedback_gain_low=feedback_gain_low,
                    feedback_gain_mid=feedback_gain_mid,
                    feedback_gain_high=feedback_gain_high,
                    strength=tonal_correction_strength,
                )
            )
        xover_low = max(20.0, float(self._config.fdn_xover_low_hz))
        xover_high = max(xover_low + 10.0, float(self._config.fdn_xover_high_hz))
        nyquist_guard = max(200.0, (float(sr) * 0.5) - 50.0)
        if xover_high > nyquist_guard:
            xover_high = nyquist_guard
        if xover_low >= xover_high:
            xover_low = max(20.0, xover_high * 0.25)
        lp_alpha_low = self._one_pole_alpha(xover_low, sr)
        lp_alpha_high = self._one_pole_alpha(xover_high, sr)
        mb_lp_low_state = np.zeros(num_lines, dtype=np.float64)
        mb_lp_high_state = np.zeros(num_lines, dtype=np.float64)
        link_filter_alpha = self._one_pole_alpha(float(self._config.fdn_link_filter_hz), sr)
        link_filter_state = np.zeros(num_lines, dtype=np.float64)
        link_filter_mix = np.float64(self._link_filter_mix)
        link_filter_dry = np.float64(1.0 - self._link_filter_mix)
        link_filter_mode = self._link_filter_mode

        # Larger damping value -> stronger HF attenuation in the feedback loop.
        lp_alpha = np.float64(0.15 + (0.83 * base_damping))
        dc_alpha = np.float64(0.995)
        mod_rate = np.float64(max(self._config.mod_rate_hz, 0.0))

        pre_buffer = np.zeros(pre_delay_samples + 1, dtype=np.float64)
        pre_idx = 0

        output = np.zeros_like(signal, dtype=np.float64)
        block_size = max(256, int(self._config.block_size))
        inv_sqrt_lines = np.float64(1.0 / np.sqrt(np.float64(num_lines)))
        fdn_out = np.zeros(num_lines, dtype=np.float64)
        feedback_source = np.zeros(num_lines, dtype=np.float64)
        mixed_feedback = np.zeros(num_lines, dtype=np.float64)
        sparse_scratch = np.zeros(num_lines, dtype=np.float64)
        cascade_enabled = (
            self._cascade_enabled
            and self._cascade_mix > 0.0
            and int(self._cascade_delay_ms.shape[0]) > 0
        )
        cascade_mix = np.float64(self._cascade_mix if cascade_enabled else 0.0)
        cascade_fdn_out = np.zeros((0,), dtype=np.float64)
        cascade_feedback = np.zeros((0,), dtype=np.float64)
        cascade_base_gain = np.zeros((0,), dtype=np.float64)
        cascade_delay_buffers: list[npt.NDArray[np.float64]] = []
        cascade_write_indices = np.zeros((0,), dtype=np.int32)
        cascade_lp_state = np.zeros((0,), dtype=np.float64)
        cascade_dc_prev_in = np.zeros((0,), dtype=np.float64)
        cascade_dc_prev_out = np.zeros((0,), dtype=np.float64)
        cascade_line_delays = np.zeros((0,), dtype=np.int32)
        cascade_phase = np.zeros((0,), dtype=np.float64)
        cascade_max_mod_samples = 1
        cascade_inv_sqrt_lines = np.float64(1.0)
        cascade_delays_sec = np.zeros((0,), dtype=np.float64)
        if cascade_enabled:
            cascade_line_delays = np.maximum(
                2,
                np.asarray(np.round((self._cascade_delay_ms / 1000.0) * sr), dtype=np.int32),
            )
            cascade_num_lines = int(cascade_line_delays.shape[0])
            cascade_max_mod_samples = max(
                1,
                int(max_mod_samples * self._cascade_delay_scale),
            )
            cascade_delay_buffers = [
                np.zeros(delay + (2 * cascade_max_mod_samples) + 4, dtype=np.float64)
                for delay in cascade_line_delays
            ]
            cascade_write_indices = np.zeros(cascade_num_lines, dtype=np.int32)
            cascade_lp_state = np.zeros(cascade_num_lines, dtype=np.float64)
            cascade_dc_prev_in = np.zeros(cascade_num_lines, dtype=np.float64)
            cascade_dc_prev_out = np.zeros(cascade_num_lines, dtype=np.float64)
            cascade_phase = np.linspace(
                np.pi * 0.25,
                (2.0 * np.pi) + (np.pi * 0.25),
                cascade_num_lines,
                endpoint=False,
                dtype=np.float64,
            )
            cascade_fdn_out = np.zeros(cascade_num_lines, dtype=np.float64)
            cascade_feedback = np.zeros(cascade_num_lines, dtype=np.float64)
            cascade_delays_sec = cascade_line_delays.astype(np.float64) / float(sr)
            cascade_rt60 = max(0.1, float(base_rt60) * self._cascade_rt60_ratio)
            cascade_base_gain = np.power(10.0, (-3.0 * cascade_delays_sec) / cascade_rt60).astype(
                np.float64
            )
            cascade_base_gain = np.clip(cascade_base_gain, 0.0, 0.995)
            cascade_inv_sqrt_lines = np.float64(1.0 / np.sqrt(np.float64(cascade_num_lines)))

        macro_eps = self._TRACK_C_UPDATE_EPS
        room_size_macro = room_size_macro_default
        clarity_macro = clarity_macro_default
        warmth_macro = warmth_macro_default
        envelopment_macro = envelopment_macro_default
        macro_rt60_scale = (
            self._macro_scale(
                room_size_macro + (0.30 * envelopment_macro),
                sensitivity=0.85,
                minimum=0.4,
                maximum=3.2,
            )
            * self._macro_scale(
                clarity_macro,
                sensitivity=-0.70,
                minimum=0.45,
                maximum=1.9,
            )
            * self._macro_scale(
                warmth_macro,
                sensitivity=0.15,
                minimum=0.8,
                maximum=1.3,
            )
        )
        macro_rt60_scale = float(np.clip(macro_rt60_scale, 0.35, 4.0))
        macro_damping_delta = float(
            np.clip(
                (0.22 * warmth_macro) - (0.20 * clarity_macro) - (0.08 * room_size_macro),
                -0.45,
                0.45,
            )
        )

        # Force a first-sample gain refresh so static macro scaling is applied
        # even when no explicit automation lanes are active.
        last_rt60_effective = -1.0
        last_lp_alpha = float(lp_alpha)
        room_size_default = np.float64(1.0)

        for block_start in range(0, signal.shape[0], block_size):
            block_end = min(signal.shape[0], block_start + block_size)
            block_matrix: npt.NDArray[np.float64] | None = None
            if not self._sparse_enabled and not self._graph_enabled:
                block_matrix = self._current_block_matrix(
                    sr=sr,
                    block_samples=max(1, block_end - block_start),
                )
            for n in range(block_start, block_end):
                rt60_effective = float(base_rt60)
                damping_effective = float(base_damping)
                if rt60_curve is not None:
                    rt60_effective = float(
                        np.clip(rt60_curve[n], RT60_MIN_SECONDS, RT60_MAX_SECONDS)
                    )
                if damping_curve is not None:
                    damping_effective = float(np.clip(damping_curve[n], 0.0, 1.0))
                room_size_macro_sample = room_size_macro_default
                clarity_macro_sample = clarity_macro_default
                warmth_macro_sample = warmth_macro_default
                envelopment_macro_sample = envelopment_macro_default
                fdn_rt60_tilt_sample = fdn_rt60_tilt_default
                tonal_correction_strength_sample = tonal_correction_strength_default
                if room_size_macro_curve is not None:
                    room_size_macro_sample = float(np.clip(room_size_macro_curve[n], -1.0, 1.0))
                if clarity_macro_curve is not None:
                    clarity_macro_sample = float(np.clip(clarity_macro_curve[n], -1.0, 1.0))
                if warmth_macro_curve is not None:
                    warmth_macro_sample = float(np.clip(warmth_macro_curve[n], -1.0, 1.0))
                if envelopment_macro_curve is not None:
                    envelopment_macro_sample = float(
                        np.clip(envelopment_macro_curve[n], -1.0, 1.0)
                    )
                if fdn_rt60_tilt_curve is not None:
                    fdn_rt60_tilt_sample = float(np.clip(fdn_rt60_tilt_curve[n], -1.0, 1.0))
                if tonal_correction_strength_curve is not None:
                    tonal_correction_strength_sample = float(
                        np.clip(tonal_correction_strength_curve[n], 0.0, 1.0)
                    )

                room_size_changed = abs(room_size_macro_sample - room_size_macro) > macro_eps
                clarity_changed = abs(clarity_macro_sample - clarity_macro) > macro_eps
                warmth_changed = abs(warmth_macro_sample - warmth_macro) > macro_eps
                envelopment_changed = abs(envelopment_macro_sample - envelopment_macro) > macro_eps
                fdn_tilt_changed = abs(fdn_rt60_tilt_sample - fdn_rt60_tilt) > macro_eps
                tonal_strength_changed = (
                    abs(tonal_correction_strength_sample - tonal_correction_strength) > macro_eps
                )

                if (
                    room_size_changed
                    or clarity_changed
                    or warmth_changed
                    or envelopment_changed
                ):
                    room_size_macro = room_size_macro_sample
                    clarity_macro = clarity_macro_sample
                    warmth_macro = warmth_macro_sample
                    envelopment_macro = envelopment_macro_sample
                    macro_rt60_scale = (
                        self._macro_scale(
                            room_size_macro + (0.30 * envelopment_macro),
                            sensitivity=0.85,
                            minimum=0.4,
                            maximum=3.2,
                        )
                        * self._macro_scale(
                            clarity_macro,
                            sensitivity=-0.70,
                            minimum=0.45,
                            maximum=1.9,
                        )
                        * self._macro_scale(
                            warmth_macro,
                            sensitivity=0.15,
                            minimum=0.8,
                            maximum=1.3,
                        )
                    )
                    macro_rt60_scale = float(np.clip(macro_rt60_scale, 0.35, 4.0))
                    macro_damping_delta = float(
                        np.clip(
                            (0.22 * warmth_macro)
                            - (0.20 * clarity_macro)
                            - (0.08 * room_size_macro),
                            -0.45,
                            0.45,
                        )
                    )
                if fdn_tilt_changed:
                    fdn_rt60_tilt = fdn_rt60_tilt_sample
                if tonal_strength_changed:
                    tonal_correction_strength = tonal_correction_strength_sample
                tonal_correction_enabled = multiband_active and tonal_correction_strength > 0.0

                rt60_effective *= macro_rt60_scale
                damping_effective = float(
                    np.clip(damping_effective + macro_damping_delta, 0.0, 1.0)
                )
                room_size = float(room_size_default)
                if room_size_curve is not None:
                    room_size = float(np.clip(room_size_curve[n], 0.25, 4.0))
                    rt60_effective *= room_size
                    damping_effective = float(
                        np.clip(damping_effective - (0.15 * (room_size - 1.0)), 0.0, 1.0)
                    )

                rt60_changed = abs(rt60_effective - last_rt60_effective) > self._RT60_UPDATE_EPS
                multiband_profile_changed = (
                    multiband_active
                    and (
                        fdn_tilt_changed
                        or clarity_changed
                        or warmth_changed
                    )
                )
                if rt60_changed:
                    feedback_gain[:] = np.clip(
                        np.power(10.0, (-3.0 * delays_sec) / max(rt60_effective, 0.1)),
                        0.0,
                        0.995,
                    ).astype(np.float64)
                    if cascade_enabled and cascade_base_gain.size > 0:
                        cascade_rt60_eff = max(0.1, rt60_effective * self._cascade_rt60_ratio)
                        cascade_base_gain[:] = np.clip(
                            np.power(10.0, (-3.0 * cascade_delays_sec) / cascade_rt60_eff),
                            0.0,
                            0.995,
                        ).astype(np.float64)
                    last_rt60_effective = float(rt60_effective)

                if multiband_active and (rt60_changed or multiband_profile_changed):
                    rt60_low, rt60_mid, rt60_high = self._resolve_multiband_rt60(
                        self._config,
                        clarity_macro=clarity_macro,
                        warmth_macro=warmth_macro,
                        fdn_rt60_tilt=fdn_rt60_tilt,
                    )
                    ratio = max(0.05, rt60_effective / max(base_rt60, 0.1))
                    rt60_low_eff = max(0.1, float(rt60_low) * ratio)
                    rt60_mid_eff = max(0.1, float(rt60_mid) * ratio)
                    rt60_high_eff = max(0.1, float(rt60_high) * ratio)
                    feedback_gain_low[:] = np.clip(
                        np.power(10.0, (-3.0 * delays_sec) / rt60_low_eff),
                        0.0,
                        0.995,
                    ).astype(np.float64)
                    feedback_gain_mid[:] = np.clip(
                        np.power(10.0, (-3.0 * delays_sec) / rt60_mid_eff),
                        0.0,
                        0.995,
                    ).astype(np.float64)
                    feedback_gain_high[:] = np.clip(
                        np.power(10.0, (-3.0 * delays_sec) / rt60_high_eff),
                        0.0,
                        0.995,
                    ).astype(np.float64)

                if multiband_active and (
                    rt60_changed or multiband_profile_changed or tonal_strength_changed
                ):
                    if tonal_correction_enabled:
                        tonal_low_scale, tonal_mid_scale, tonal_high_scale = (
                            self._resolve_tonal_correction_scales(
                                feedback_gain_low=feedback_gain_low,
                                feedback_gain_mid=feedback_gain_mid,
                                feedback_gain_high=feedback_gain_high,
                                strength=tonal_correction_strength,
                            )
                        )
                    else:
                        tonal_low_scale = np.float64(1.0)
                        tonal_mid_scale = np.float64(1.0)
                        tonal_high_scale = np.float64(1.0)

                lp_alpha_sample = float(0.15 + (0.83 * np.clip(damping_effective, 0.0, 1.0)))
                if abs(lp_alpha_sample - last_lp_alpha) > self._LP_ALPHA_UPDATE_EPS:
                    lp_alpha = np.float64(lp_alpha_sample)
                    last_lp_alpha = lp_alpha_sample

                predelayed = pre_buffer[pre_idx]
                pre_buffer[pre_idx] = signal[n]
                pre_idx = (pre_idx + 1) % pre_buffer.shape[0]

                # Diffusion stage: a short all-pass cascade to smear transients
                # before they enter the long feedback network.
                diffused = predelayed
                for ap_index, ap in enumerate(allpasses):
                    diffused = self._allpass_process(
                        diffused,
                        ap,
                        gain=np.float64(self._allpass_gains[ap_index]),
                    )

                if cascade_enabled:
                    for i in range(cascade_fdn_out.shape[0]):
                        mod = cascade_max_mod_samples * np.sin(cascade_phase[i])
                        cascade_phase[i] += np.float64((2.0 * np.pi * mod_rate) / sr)
                        if cascade_phase[i] > (2.0 * np.pi):
                            cascade_phase[i] -= np.float64(2.0 * np.pi)

                        delay = float(cascade_line_delays[i]) + float(mod)
                        read_value = self._read_fractional_delay(
                            buffer=cascade_delay_buffers[i],
                            write_index=int(cascade_write_indices[i]),
                            delay_samples=delay,
                        )
                        cascade_lp_state[i] = ((1.0 - lp_alpha) * read_value) + (
                            lp_alpha * cascade_lp_state[i]
                        )
                        dc_filtered = (
                            cascade_lp_state[i]
                            - cascade_dc_prev_in[i]
                            + (dc_alpha * cascade_dc_prev_out[i])
                        )
                        cascade_dc_prev_in[i] = cascade_lp_state[i]
                        cascade_dc_prev_out[i] = dc_filtered
                        cascade_fdn_out[i] = dc_filtered

                    cascade_feedback[:] = self._cascade_matrix @ cascade_fdn_out
                    cascade_injection = np.float64(diffused * cascade_inv_sqrt_lines)
                    for i in range(cascade_fdn_out.shape[0]):
                        value = cascade_injection + (cascade_base_gain[i] * cascade_feedback[i])
                        cascade_delay_buffers[i][cascade_write_indices[i]] = value
                        cascade_write_indices[i] = (
                            cascade_write_indices[i] + 1
                        ) % cascade_delay_buffers[i].shape[0]

                for i in range(num_lines):
                    mod = max_mod_samples * np.sin(phase[i])
                    phase[i] += np.float64((2.0 * np.pi * mod_rate) / sr)
                    if phase[i] > (2.0 * np.pi):
                        phase[i] -= np.float64(2.0 * np.pi)

                    delay = float(line_delays[i]) + float(mod)
                    read_value = self._read_fractional_delay(
                        buffer=delay_buffers[i],
                        write_index=int(write_indices[i]),
                        delay_samples=delay,
                    )

                    # Damping + DC-blocking lives inside the feedback loop so
                    # high frequencies and subsonic drift decay faster.
                    lp_state[i] = ((1.0 - lp_alpha) * read_value) + (lp_alpha * lp_state[i])
                    dc_filtered = lp_state[i] - dc_prev_in[i] + (dc_alpha * dc_prev_out[i])
                    dc_prev_in[i] = lp_state[i]
                    dc_prev_out[i] = dc_filtered
                    fdn_out[i] = dc_filtered

                if self._link_filter_enabled:
                    for i in range(num_lines):
                        raw = fdn_out[i]
                        link_filter_state[i] += link_filter_alpha * (raw - link_filter_state[i])
                        if link_filter_mode == "lowpass":
                            filtered = link_filter_state[i]
                        else:
                            filtered = raw - link_filter_state[i]
                        feedback_source[i] = (link_filter_dry * raw) + (link_filter_mix * filtered)
                else:
                    feedback_source[:] = fdn_out

                if self._sparse_enabled or self._graph_enabled:
                    self._apply_sparse_pair_mix(
                        input_vec=feedback_source,
                        pairings=(
                            self._sparse_pairings if self._sparse_enabled else self._graph_pairings
                        ),
                        out_vec=mixed_feedback,
                        scratch=sparse_scratch,
                    )
                elif block_matrix is not None:
                    mixed_feedback[:] = block_matrix @ feedback_source
                else:  # pragma: no cover
                    mixed_feedback[:] = feedback_source
                if self._dfm_enabled:
                    for i in range(num_lines):
                        delayed_feedback = dfm_buffers[i][dfm_indices[i]]
                        dfm_buffers[i][dfm_indices[i]] = mixed_feedback[i]
                        dfm_indices[i] = (dfm_indices[i] + 1) % dfm_buffers[i].shape[0]
                        mixed_feedback[i] = delayed_feedback
                if cascade_enabled and cascade_feedback.size > 0:
                    cascade_size = int(cascade_feedback.shape[0])
                    for i in range(num_lines):
                        mixed_feedback[i] += cascade_mix * cascade_feedback[i % cascade_size]
                if self._nonlinearity_enabled:
                    mixed_feedback[:] = self._apply_feedback_nonlinearity(mixed_feedback)

                injection = np.float64(diffused * inv_sqrt_lines)

                for i in range(num_lines):
                    if multiband_active:
                        mb_lp_low_state[i] += lp_alpha_low * (
                            mixed_feedback[i] - mb_lp_low_state[i]
                        )
                        mb_lp_high_state[i] += lp_alpha_high * (
                            mixed_feedback[i] - mb_lp_high_state[i]
                        )
                        band_low = mb_lp_low_state[i]
                        band_mid = mb_lp_high_state[i] - mb_lp_low_state[i]
                        band_high = mixed_feedback[i] - mb_lp_high_state[i]
                        if tonal_correction_enabled:
                            shaped_feedback = (
                                (feedback_gain_low[i] * tonal_low_scale * band_low)
                                + (feedback_gain_mid[i] * tonal_mid_scale * band_mid)
                                + (feedback_gain_high[i] * tonal_high_scale * band_high)
                            )
                        else:
                            shaped_feedback = (
                                (feedback_gain_low[i] * band_low)
                                + (feedback_gain_mid[i] * band_mid)
                                + (feedback_gain_high[i] * band_high)
                            )
                        value = injection + shaped_feedback
                    else:
                        value = injection + (feedback_gain[i] * mixed_feedback[i])
                    delay_buffers[i][write_indices[i]] = value
                    write_indices[i] = (write_indices[i] + 1) % delay_buffers[i].shape[0]

                sample_out = float(np.mean(fdn_out))
                output[n] = np.float64(sample_out)

                # Soft safety guard for pathological parameter combinations.
                state_peak = float(np.max(np.abs(fdn_out)))
                if cascade_enabled and cascade_fdn_out.size > 0:
                    state_peak = max(state_peak, float(np.max(np.abs(cascade_fdn_out))))
                if state_peak > 64.0:
                    for i in range(num_lines):
                        delay_buffers[i] *= np.float64(0.5)
                        lp_state[i] *= np.float64(0.5)
                        dc_prev_in[i] *= np.float64(0.5)
                        dc_prev_out[i] *= np.float64(0.5)
                        if self._dfm_enabled:
                            dfm_buffers[i] *= np.float64(0.5)
                        if multiband_active:
                            mb_lp_low_state[i] *= np.float64(0.5)
                            mb_lp_high_state[i] *= np.float64(0.5)
                        if self._link_filter_enabled:
                            link_filter_state[i] *= np.float64(0.5)
                    if cascade_enabled:
                        for i in range(cascade_fdn_out.shape[0]):
                            cascade_delay_buffers[i] *= np.float64(0.5)
                            cascade_lp_state[i] *= np.float64(0.5)
                            cascade_dc_prev_in[i] *= np.float64(0.5)
                            cascade_dc_prev_out[i] *= np.float64(0.5)

        return output

    @staticmethod
    def _allpass_process(x: np.float64, state: _AllpassState, gain: np.float64) -> np.float64:
        """Run one sample through a Schroeder all-pass section."""
        delayed = state.buffer[state.index]
        y = (-gain * x) + delayed
        state.buffer[state.index] = x + (gain * y)
        state.index = (state.index + 1) % state.buffer.shape[0]
        return np.float64(y)

    @staticmethod
    def _read_fractional_delay(
        buffer: AudioArray, write_index: int, delay_samples: float
    ) -> np.float64:
        """Read from a circular delay line with linear interpolation."""
        size = buffer.shape[0]
        read_pos = (float(write_index) - delay_samples) % size
        idx0 = int(np.floor(read_pos))
        idx1 = (idx0 + 1) % size
        frac = np.float64(read_pos - idx0)
        sample = (np.float64(1.0) - frac) * buffer[idx0] + frac * buffer[idx1]
        return np.float64(sample)


@_maybe_njit
def _fractional_delay_read_nb(
    buffer: npt.NDArray[np.float64],
    size: int,
    write_index: int,
    delay_samples: float,
) -> np.float64:
    """Numba-compatible fractional delay read with linear interpolation."""
    read_pos = (float(write_index) - delay_samples) % size
    idx0 = int(np.floor(read_pos))
    idx1 = (idx0 + 1) % size
    frac = np.float64(read_pos - idx0)
    sample = (np.float64(1.0) - frac) * buffer[idx0] + frac * buffer[idx1]
    return np.float64(sample)


@_maybe_njit
def _process_channel_kernel(
    signal: npt.NDArray[np.float64],
    sr: int,
    rt60: np.float64,
    pre_delay_ms: np.float64,
    damping: np.float64,
    mod_depth_ms: np.float64,
    mod_rate_hz: np.float64,
    block_size: int,
    fdn_matrix: npt.NDArray[np.float64],
    base_delay_ms: npt.NDArray[np.float64],
    diffusion_delay_ms: npt.NDArray[np.float64],
    allpass_gains: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Numba kernel matching :meth:`AlgoReverbEngine._process_channel`.

    The Python and Numba paths intentionally mirror each other to keep
    behavior consistent across environments.
    """
    n_samples = signal.shape[0]
    output = np.zeros(n_samples, dtype=np.float64)
    if n_samples == 0:
        return output

    pre_delay_samples = max(1, int((float(pre_delay_ms) / 1000.0) * sr))
    max_mod_samples = max(1, int((float(mod_depth_ms) / 1000.0) * sr))

    line_delays = np.maximum(2, np.asarray(np.round((base_delay_ms / 1000.0) * sr), dtype=np.int32))
    num_lines = int(line_delays.shape[0])

    diffusion_delays = np.maximum(
        1, np.asarray(np.round((diffusion_delay_ms / 1000.0) * sr), dtype=np.int32)
    )
    num_allpasses = int(diffusion_delays.shape[0])

    max_ap_size = 1
    if num_allpasses > 0:
        max_ap_size = int(np.max(diffusion_delays)) + 1
    allpass_buffers = np.zeros((max(1, num_allpasses), max_ap_size), dtype=np.float64)
    allpass_sizes = np.ones(max(1, num_allpasses), dtype=np.int32)
    allpass_indices = np.zeros(max(1, num_allpasses), dtype=np.int32)
    for i in range(num_allpasses):
        allpass_sizes[i] = int(diffusion_delays[i]) + 1

    max_line_size = int(np.max(line_delays)) + (2 * max_mod_samples) + 4
    delay_buffers = np.zeros((num_lines, max_line_size), dtype=np.float64)
    delay_sizes = np.zeros(num_lines, dtype=np.int32)
    write_indices = np.zeros(num_lines, dtype=np.int32)
    for i in range(num_lines):
        delay_sizes[i] = int(line_delays[i]) + (2 * max_mod_samples) + 4

    lp_state = np.zeros(num_lines, dtype=np.float64)
    dc_prev_in = np.zeros(num_lines, dtype=np.float64)
    dc_prev_out = np.zeros(num_lines, dtype=np.float64)
    phase = np.zeros(num_lines, dtype=np.float64)
    for i in range(num_lines):
        phase[i] = np.float64((2.0 * np.pi * i) / max(1, num_lines))

    delays_sec = line_delays.astype(np.float64) / float(sr)
    base_gain = np.power(10.0, (-3.0 * delays_sec) / max(float(rt60), 0.1)).astype(np.float64)
    for i in range(num_lines):
        if base_gain[i] > np.float64(0.995):
            base_gain[i] = np.float64(0.995)
        elif base_gain[i] < np.float64(0.0):
            base_gain[i] = np.float64(0.0)

    damp = float(damping)
    if damp < 0.0:
        damp = 0.0
    elif damp > 1.0:
        damp = 1.0
    lp_alpha = np.float64(0.15 + (0.83 * damp))
    dc_alpha = np.float64(0.995)
    mod_rate = np.float64(max(float(mod_rate_hz), 0.0))

    pre_buffer = np.zeros(pre_delay_samples + 1, dtype=np.float64)
    pre_idx = 0

    fdn_out = np.zeros(num_lines, dtype=np.float64)
    mixed_feedback = np.zeros(num_lines, dtype=np.float64)
    two_pi = np.float64(2.0 * np.pi)
    inv_sqrt_lines = np.float64(1.0 / np.sqrt(float(num_lines)))

    for block_start in range(0, n_samples, block_size):
        block_end = min(n_samples, block_start + block_size)
        for n in range(block_start, block_end):
            predelayed = pre_buffer[pre_idx]
            pre_buffer[pre_idx] = signal[n]
            pre_idx = (pre_idx + 1) % pre_buffer.shape[0]

            diffused = predelayed
            for ap in range(num_allpasses):
                ap_size = allpass_sizes[ap]
                ap_idx = allpass_indices[ap]
                delayed = allpass_buffers[ap, ap_idx]
                gain = allpass_gains[ap]
                y = (-gain * diffused) + delayed
                allpass_buffers[ap, ap_idx] = diffused + (gain * y)
                allpass_indices[ap] = (ap_idx + 1) % ap_size
                diffused = np.float64(y)

            for i in range(num_lines):
                mod = float(max_mod_samples) * np.sin(float(phase[i]))
                phase[i] += np.float64((2.0 * np.pi * float(mod_rate)) / sr)
                if phase[i] > two_pi:
                    phase[i] -= two_pi

                delay = float(line_delays[i]) + mod
                size = int(delay_sizes[i])
                read_value = _fractional_delay_read_nb(
                    buffer=delay_buffers[i, :size],
                    size=size,
                    write_index=int(write_indices[i]),
                    delay_samples=delay,
                )

                lp_state[i] = ((1.0 - lp_alpha) * read_value) + (lp_alpha * lp_state[i])
                dc_filtered = lp_state[i] - dc_prev_in[i] + (dc_alpha * dc_prev_out[i])
                dc_prev_in[i] = lp_state[i]
                dc_prev_out[i] = dc_filtered
                fdn_out[i] = dc_filtered

            for i in range(num_lines):
                acc = np.float64(0.0)
                for j in range(num_lines):
                    acc += fdn_matrix[i, j] * fdn_out[j]
                mixed_feedback[i] = acc

            injection = np.float64(diffused * inv_sqrt_lines)
            for i in range(num_lines):
                value = injection + (base_gain[i] * mixed_feedback[i])
                size = int(delay_sizes[i])
                idx = int(write_indices[i])
                delay_buffers[i, idx] = value
                write_indices[i] = (idx + 1) % size

            output[n] = np.float64(np.mean(fdn_out))

            state_peak = np.float64(0.0)
            for i in range(num_lines):
                abs_val = np.abs(fdn_out[i])
                if abs_val > state_peak:
                    state_peak = abs_val
            if state_peak > np.float64(64.0):
                for i in range(num_lines):
                    size = int(delay_sizes[i])
                    delay_buffers[i, :size] *= np.float64(0.5)
                    lp_state[i] *= np.float64(0.5)
                    dc_prev_in[i] *= np.float64(0.5)
                    dc_prev_out[i] *= np.float64(0.5)

    return output
