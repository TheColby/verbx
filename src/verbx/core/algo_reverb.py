"""Algorithmic extreme reverb engine (v0.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from verbx.core.engine_base import ReverbEngine
from verbx.io.audio import ensure_mono_or_stereo

AudioArray = npt.NDArray[np.float32]


@dataclass(slots=True)
class AlgoReverbConfig:
    """Configuration for the algorithmic reverb engine."""

    rt60: float = 60.0
    pre_delay_ms: float = 20.0
    damping: float = 0.45
    width: float = 1.0
    mod_depth_ms: float = 2.0
    mod_rate_hz: float = 0.1
    wet: float = 0.8
    dry: float = 0.2
    block_size: int = 4096


@dataclass(slots=True)
class _AllpassState:
    buffer: AudioArray
    index: int = 0


class AlgoReverbEngine(ReverbEngine):
    """Block-processed Schroeder + FDN algorithmic reverb.

    This implementation focuses on stability for long RT60 settings and provides
    practical placeholders for future refinements.
    """

    _BASE_DELAY_MS = np.array([31.0, 37.0, 41.0, 43.0, 47.0, 53.0, 59.0, 67.0], dtype=np.float32)
    _DIFFUSION_DELAY_MS = np.array([5.0, 7.0, 11.0, 17.0, 23.0, 29.0], dtype=np.float32)

    def __init__(self, config: AlgoReverbConfig) -> None:
        self._config = config
        self._hadamard = self._build_hadamard_matrix(8)

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

        output = (self._config.dry * x) + (self._config.wet * wet)
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

        peak = float(np.max(np.abs(output)))
        if peak > 8.0:
            output *= 8.0 / peak

        return np.asarray(output, dtype=np.float32)

    @staticmethod
    def _build_hadamard_matrix(size: int) -> npt.NDArray[np.float32]:
        matrix = np.array([[1.0]], dtype=np.float32)
        while matrix.shape[0] < size:
            matrix = np.block([[matrix, matrix], [matrix, -matrix]])
        matrix = matrix[:size, :size]
        return matrix / np.sqrt(np.float32(size))

    @staticmethod
    def _apply_stereo_width(wet: AudioArray, width: float) -> AudioArray:
        w = np.clip(width, 0.0, 2.0)
        mid = 0.5 * (wet[:, 0] + wet[:, 1])
        side = 0.5 * (wet[:, 0] - wet[:, 1])
        side *= w
        out = wet.copy()
        out[:, 0] = mid + side
        out[:, 1] = mid - side
        return np.asarray(out, dtype=np.float32)

    def _process_channel(self, signal: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
        pre_delay_samples = max(1, int((self._config.pre_delay_ms / 1000.0) * sr))
        max_mod_samples = max(1, int((self._config.mod_depth_ms / 1000.0) * sr))

        line_delays = np.maximum(
            2,
            np.asarray(np.round((self._BASE_DELAY_MS / 1000.0) * sr), dtype=np.int32),
        )
        num_lines = int(line_delays.shape[0])

        diffusion_delays = np.maximum(
            1,
            np.asarray(np.round((self._DIFFUSION_DELAY_MS / 1000.0) * sr), dtype=np.int32),
        )

        allpasses = [
            _AllpassState(buffer=np.zeros(delay + 1, dtype=np.float32))
            for delay in diffusion_delays[:6]
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

        # Larger damping value -> stronger high-frequency damping.
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

                diffused = predelayed
                for ap in allpasses:
                    diffused = self._allpass_process(diffused, ap, gain=np.float32(0.7))

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

                if np.max(np.abs(fdn_out)) > 64.0:
                    for i in range(num_lines):
                        delay_buffers[i] *= np.float32(0.5)
                        lp_state[i] *= np.float32(0.5)
                        dc_prev_in[i] *= np.float32(0.5)
                        dc_prev_out[i] *= np.float32(0.5)

        return output

    @staticmethod
    def _allpass_process(x: np.float32, state: _AllpassState, gain: np.float32) -> np.float32:
        delayed = state.buffer[state.index]
        y = (-gain * x) + delayed
        state.buffer[state.index] = x + (gain * y)
        state.index = (state.index + 1) % state.buffer.shape[0]
        return np.float32(y)

    @staticmethod
    def _read_fractional_delay(
        buffer: AudioArray, write_index: int, delay_samples: float
    ) -> np.float32:
        size = buffer.shape[0]
        read_pos = (float(write_index) - delay_samples) % size
        idx0 = int(np.floor(read_pos))
        idx1 = (idx0 + 1) % size
        frac = np.float32(read_pos - idx0)
        sample = (np.float32(1.0) - frac) * buffer[idx0] + frac * buffer[idx1]
        return np.float32(sample)
