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
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from verbx.core.engine_base import ReverbEngine
from verbx.core.shimmer import ShimmerConfig, ShimmerProcessor
from verbx.io.audio import ensure_mono_or_stereo

AudioArray = npt.NDArray[np.float32]

try:
    from numba import njit  # type: ignore[import-untyped]

    _numba_available = True
except Exception:  # pragma: no cover
    njit = None
    _numba_available = False

F = TypeVar("F", bound=Callable[..., object])


def _maybe_njit(func: F) -> F:
    """Decorate with ``numba.njit`` when available, else return unchanged."""
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
    allpass_delays_ms: tuple[float, ...] = ()
    comb_delays_ms: tuple[float, ...] = ()
    fdn_lines: int = 8
    wet: float = 0.8
    dry: float = 0.2
    block_size: int = 4096
    shimmer: bool = False
    shimmer_semitones: float = 12.0
    shimmer_mix: float = 0.25
    shimmer_feedback: float = 0.35
    shimmer_highcut: float | None = 10_000.0
    shimmer_lowcut: float | None = 300.0
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

    _DEFAULT_BASE_DELAY_MS = np.array(
        [31.0, 37.0, 41.0, 43.0, 47.0, 53.0, 59.0, 67.0],
        dtype=np.float32,
    )
    _DEFAULT_DIFFUSION_DELAY_MS = np.array(
        [5.0, 7.0, 11.0, 17.0, 23.0, 29.0],
        dtype=np.float32,
    )

    def __init__(self, config: AlgoReverbConfig) -> None:
        self._config = config
        self._base_delay_ms = self._resolve_fdn_delay_ms(config)
        self._diffusion_delay_ms = self._resolve_diffusion_delay_ms(config)
        self._allpass_gain = np.float32(np.clip(config.allpass_gain, -0.99, 0.99))
        self._hadamard = self._build_hadamard_matrix(int(self._base_delay_ms.shape[0]))
        self._use_numba = _numba_available and config.device != "cuda"
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

    def process(self, audio: AudioArray, sr: int) -> AudioArray:
        """Process audio with pre-diffusion + late FDN and wet/dry mix."""
        x = ensure_mono_or_stereo(audio)
        n_samples, n_channels = x.shape
        if n_samples == 0:
            return x.copy()

        wet = np.zeros_like(x, dtype=np.float32)
        for channel in range(n_channels):
            wet[:, channel] = self._process_channel(x[:, channel], sr)

        if n_channels == 2 and self._config.width != 1.0:
            wet = self._apply_stereo_width(wet, self._config.width)

        if self._config.shimmer:
            wet = self._shimmer.process(wet, sr)

        output = (self._config.dry * x) + (self._config.wet * wet)
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

        peak = float(np.max(np.abs(output)))
        if peak > 8.0:
            output *= 8.0 / peak

        return np.asarray(output, dtype=np.float32)

    def backend_name(self) -> str:
        """Return current algorithmic backend."""
        return "cpu-numba-fdn" if self._use_numba else "cpu-python-fdn"

    @staticmethod
    def _build_hadamard_matrix(size: int) -> npt.NDArray[np.float32]:
        """Build an orthonormal Hadamard-style mix matrix."""
        if size <= 0:
            return np.zeros((0, 0), dtype=np.float32)
        matrix = np.array([[1.0]], dtype=np.float32)
        while matrix.shape[0] < size:
            matrix = np.block([[matrix, matrix], [matrix, -matrix]])
        matrix = matrix[:size, :size]
        # Truncating non-power-of-two Hadamards breaks strict orthogonality;
        # QR restores an orthonormal mix while preserving the deterministic seed.
        q, _ = np.linalg.qr(matrix.astype(np.float64))
        return np.asarray(q, dtype=np.float32)

    @classmethod
    def _resolve_fdn_delay_ms(cls, config: AlgoReverbConfig) -> npt.NDArray[np.float32]:
        """Resolve user-configured comb-like FDN delay lengths in milliseconds."""
        if len(config.comb_delays_ms) > 0:
            delays = [max(0.1, float(value)) for value in config.comb_delays_ms]
            return np.asarray(delays, dtype=np.float32)

        requested = max(1, int(config.fdn_lines))
        defaults = cls._DEFAULT_BASE_DELAY_MS.astype(np.float64).tolist()
        while len(defaults) < requested:
            next_delay = (defaults[-1] * 1.11) + 1.25
            if next_delay <= defaults[-1]:
                next_delay = defaults[-1] + 0.25
            defaults.append(next_delay)
        return np.asarray(defaults[:requested], dtype=np.float32)

    @classmethod
    def _resolve_diffusion_delay_ms(cls, config: AlgoReverbConfig) -> npt.NDArray[np.float32]:
        """Resolve user-configured allpass diffusion delay lengths in milliseconds."""
        requested = max(0, int(config.allpass_stages))
        if requested == 0:
            return np.zeros((0,), dtype=np.float32)

        if len(config.allpass_delays_ms) > 0:
            delays = [max(0.1, float(value)) for value in config.allpass_delays_ms]
        else:
            delays = cls._DEFAULT_DIFFUSION_DELAY_MS.astype(np.float64).tolist()

        while len(delays) < requested:
            next_delay = (delays[-1] * 1.28) + 0.75
            if next_delay <= delays[-1]:
                next_delay = delays[-1] + 0.2
            delays.append(next_delay)
        return np.asarray(delays[:requested], dtype=np.float32)

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
        return np.asarray(out, dtype=np.float32)

    def _process_channel(self, signal: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
        """Run one channel through pre-delay, diffusion, and FDN late reverb."""
        if self._use_numba:
            return _process_channel_kernel(
                signal=signal,
                sr=sr,
                rt60=np.float32(self._config.rt60),
                pre_delay_ms=np.float32(self._config.pre_delay_ms),
                damping=np.float32(self._config.damping),
                mod_depth_ms=np.float32(self._config.mod_depth_ms),
                mod_rate_hz=np.float32(self._config.mod_rate_hz),
                block_size=max(256, int(self._config.block_size)),
                hadamard=self._hadamard,
                base_delay_ms=self._base_delay_ms,
                diffusion_delay_ms=self._diffusion_delay_ms,
                allpass_gain=self._allpass_gain,
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
            _AllpassState(buffer=np.zeros(delay + 1, dtype=np.float32))
            for delay in diffusion_delays
        ]

        delay_buffers = [
            np.zeros(delay + (2 * max_mod_samples) + 4, dtype=np.float32) for delay in line_delays
        ]
        write_indices = np.zeros(num_lines, dtype=np.int32)
        lp_state = np.zeros(num_lines, dtype=np.float32)
        dc_prev_in = np.zeros(num_lines, dtype=np.float32)
        dc_prev_out = np.zeros(num_lines, dtype=np.float32)

        base_phase = np.linspace(0.0, 2.0 * np.pi, num_lines, endpoint=False, dtype=np.float32)
        phase = base_phase.copy()

        rt60 = max(self._config.rt60, 0.1)
        delays_sec = line_delays.astype(np.float64) / float(sr)
        base_gain = np.power(10.0, (-3.0 * delays_sec) / rt60).astype(np.float32)
        base_gain = np.clip(base_gain, 0.0, 0.995)

        # Larger damping value -> stronger HF attenuation in the feedback loop.
        damping = float(np.clip(self._config.damping, 0.0, 1.0))
        lp_alpha = np.float32(0.15 + (0.83 * damping))
        dc_alpha = np.float32(0.995)
        mod_rate = np.float32(max(self._config.mod_rate_hz, 0.0))

        pre_buffer = np.zeros(pre_delay_samples + 1, dtype=np.float32)
        pre_idx = 0

        output = np.zeros_like(signal, dtype=np.float32)
        block_size = max(256, int(self._config.block_size))

        for block_start in range(0, signal.shape[0], block_size):
            block_end = min(signal.shape[0], block_start + block_size)
            for n in range(block_start, block_end):
                predelayed = pre_buffer[pre_idx]
                pre_buffer[pre_idx] = signal[n]
                pre_idx = (pre_idx + 1) % pre_buffer.shape[0]

                # Diffusion stage: a short all-pass cascade to smear transients
                # before they enter the long feedback network.
                diffused = predelayed
                for ap in allpasses:
                    diffused = self._allpass_process(diffused, ap, gain=self._allpass_gain)

                fdn_out = np.zeros(num_lines, dtype=np.float32)
                for i in range(num_lines):
                    mod = max_mod_samples * np.sin(phase[i])
                    phase[i] += np.float32((2.0 * np.pi * mod_rate) / sr)
                    if phase[i] > (2.0 * np.pi):
                        phase[i] -= np.float32(2.0 * np.pi)

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

                mixed_feedback = self._hadamard @ fdn_out
                injection = np.float32(diffused / np.sqrt(np.float32(num_lines)))

                for i in range(num_lines):
                    value = injection + (base_gain[i] * mixed_feedback[i])
                    delay_buffers[i][write_indices[i]] = value
                    write_indices[i] = (write_indices[i] + 1) % delay_buffers[i].shape[0]

                sample_out = float(np.mean(fdn_out))
                output[n] = np.float32(sample_out)

                # Soft safety guard for pathological parameter combinations.
                if np.max(np.abs(fdn_out)) > 64.0:
                    for i in range(num_lines):
                        delay_buffers[i] *= np.float32(0.5)
                        lp_state[i] *= np.float32(0.5)
                        dc_prev_in[i] *= np.float32(0.5)
                        dc_prev_out[i] *= np.float32(0.5)

        return output

    @staticmethod
    def _allpass_process(x: np.float32, state: _AllpassState, gain: np.float32) -> np.float32:
        """Run one sample through a Schroeder all-pass section."""
        delayed = state.buffer[state.index]
        y = (-gain * x) + delayed
        state.buffer[state.index] = x + (gain * y)
        state.index = (state.index + 1) % state.buffer.shape[0]
        return np.float32(y)

    @staticmethod
    def _read_fractional_delay(
        buffer: AudioArray, write_index: int, delay_samples: float
    ) -> np.float32:
        """Read from a circular delay line with linear interpolation."""
        size = buffer.shape[0]
        read_pos = (float(write_index) - delay_samples) % size
        idx0 = int(np.floor(read_pos))
        idx1 = (idx0 + 1) % size
        frac = np.float32(read_pos - idx0)
        sample = (np.float32(1.0) - frac) * buffer[idx0] + frac * buffer[idx1]
        return np.float32(sample)


@_maybe_njit
def _fractional_delay_read_nb(
    buffer: npt.NDArray[np.float32],
    size: int,
    write_index: int,
    delay_samples: float,
) -> np.float32:
    """Numba-compatible fractional delay read with linear interpolation."""
    read_pos = (float(write_index) - delay_samples) % size
    idx0 = int(np.floor(read_pos))
    idx1 = (idx0 + 1) % size
    frac = np.float32(read_pos - idx0)
    sample = (np.float32(1.0) - frac) * buffer[idx0] + frac * buffer[idx1]
    return np.float32(sample)


@_maybe_njit
def _process_channel_kernel(
    signal: npt.NDArray[np.float32],
    sr: int,
    rt60: np.float32,
    pre_delay_ms: np.float32,
    damping: np.float32,
    mod_depth_ms: np.float32,
    mod_rate_hz: np.float32,
    block_size: int,
    hadamard: npt.NDArray[np.float32],
    base_delay_ms: npt.NDArray[np.float32],
    diffusion_delay_ms: npt.NDArray[np.float32],
    allpass_gain: np.float32,
) -> npt.NDArray[np.float32]:
    """Numba kernel matching :meth:`AlgoReverbEngine._process_channel`.

    The Python and Numba paths intentionally mirror each other to keep
    behavior consistent across environments.
    """
    n_samples = signal.shape[0]
    output = np.zeros(n_samples, dtype=np.float32)
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
    allpass_buffers = np.zeros((max(1, num_allpasses), max_ap_size), dtype=np.float32)
    allpass_sizes = np.ones(max(1, num_allpasses), dtype=np.int32)
    allpass_indices = np.zeros(max(1, num_allpasses), dtype=np.int32)
    for i in range(num_allpasses):
        allpass_sizes[i] = int(diffusion_delays[i]) + 1

    max_line_size = int(np.max(line_delays)) + (2 * max_mod_samples) + 4
    delay_buffers = np.zeros((num_lines, max_line_size), dtype=np.float32)
    delay_sizes = np.zeros(num_lines, dtype=np.int32)
    write_indices = np.zeros(num_lines, dtype=np.int32)
    for i in range(num_lines):
        delay_sizes[i] = int(line_delays[i]) + (2 * max_mod_samples) + 4

    lp_state = np.zeros(num_lines, dtype=np.float32)
    dc_prev_in = np.zeros(num_lines, dtype=np.float32)
    dc_prev_out = np.zeros(num_lines, dtype=np.float32)
    phase = np.zeros(num_lines, dtype=np.float32)
    for i in range(num_lines):
        phase[i] = np.float32((2.0 * np.pi * i) / max(1, num_lines))

    delays_sec = line_delays.astype(np.float64) / float(sr)
    base_gain = np.power(10.0, (-3.0 * delays_sec) / max(float(rt60), 0.1)).astype(np.float32)
    for i in range(num_lines):
        if base_gain[i] > np.float32(0.995):
            base_gain[i] = np.float32(0.995)
        elif base_gain[i] < np.float32(0.0):
            base_gain[i] = np.float32(0.0)

    damp = float(damping)
    if damp < 0.0:
        damp = 0.0
    elif damp > 1.0:
        damp = 1.0
    lp_alpha = np.float32(0.15 + (0.83 * damp))
    dc_alpha = np.float32(0.995)
    mod_rate = np.float32(max(float(mod_rate_hz), 0.0))

    pre_buffer = np.zeros(pre_delay_samples + 1, dtype=np.float32)
    pre_idx = 0

    fdn_out = np.zeros(num_lines, dtype=np.float32)
    mixed_feedback = np.zeros(num_lines, dtype=np.float32)
    two_pi = np.float32(2.0 * np.pi)
    inv_sqrt_lines = np.float32(1.0 / np.sqrt(float(num_lines)))

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
                y = (-allpass_gain * diffused) + delayed
                allpass_buffers[ap, ap_idx] = diffused + (allpass_gain * y)
                allpass_indices[ap] = (ap_idx + 1) % ap_size
                diffused = np.float32(y)

            for i in range(num_lines):
                mod = float(max_mod_samples) * np.sin(float(phase[i]))
                phase[i] += np.float32((2.0 * np.pi * float(mod_rate)) / sr)
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
                acc = np.float32(0.0)
                for j in range(num_lines):
                    acc += hadamard[i, j] * fdn_out[j]
                mixed_feedback[i] = acc

            injection = np.float32(diffused * inv_sqrt_lines)
            for i in range(num_lines):
                value = injection + (base_gain[i] * mixed_feedback[i])
                size = int(delay_sizes[i])
                idx = int(write_indices[i])
                delay_buffers[i, idx] = value
                write_indices[i] = (idx + 1) % size

            output[n] = np.float32(np.mean(fdn_out))

            state_peak = np.float32(0.0)
            for i in range(num_lines):
                abs_val = np.abs(fdn_out[i])
                if abs_val > state_peak:
                    state_peak = abs_val
            if state_peak > np.float32(64.0):
                for i in range(num_lines):
                    size = int(delay_sizes[i])
                    delay_buffers[i, :size] *= np.float32(0.5)
                    lp_state[i] *= np.float32(0.5)
                    dc_prev_in[i] *= np.float32(0.5)
                    dc_prev_out[i] *= np.float32(0.5)

    return output
