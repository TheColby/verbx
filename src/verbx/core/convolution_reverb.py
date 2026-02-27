"""Partitioned FFT convolution reverb (v0.1)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy.signal import resample_poly

from verbx.core.engine_base import ReverbEngine
from verbx.io.audio import ensure_mono_or_stereo

AudioArray = npt.NDArray[np.float32]


@dataclass(slots=True)
class ConvolutionReverbConfig:
    """Configuration for the convolution engine."""

    wet: float = 0.8
    dry: float = 0.2
    ir_path: str | None = None
    ir_normalize: str = "peak"
    partition_size: int = 16_384
    tail_limit: float | None = None
    threads: int | None = None


class ConvolutionReverbEngine(ReverbEngine):
    """Block-based partitioned FFT convolution supporting long IRs."""

    def __init__(self, config: ConvolutionReverbConfig) -> None:
        self._config = config

    def process(self, audio: AudioArray, sr: int) -> AudioArray:
        """Convolve input with IR using partitioned FFT overlap-add."""
        x = ensure_mono_or_stereo(audio)
        if self._config.ir_path is None:
            msg = "Convolution engine requires --ir PATH"
            raise ValueError(msg)

        ir = self._load_ir(self._config.ir_path, sr)
        if self._config.tail_limit is not None:
            max_tail = max(0, int(self._config.tail_limit * sr))
            ir = ir[: max_tail + 1, :]

        ir = self._align_ir_channels(ir, x.shape[1])
        ir = self._normalize_ir(ir, self._config.ir_normalize)

        wet = self._partitioned_convolve(x, ir, max(256, int(self._config.partition_size)))

        out_len = wet.shape[0]
        dry = np.zeros((out_len, x.shape[1]), dtype=np.float32)
        dry[: x.shape[0], :] = x

        output = (self._config.dry * dry) + (self._config.wet * wet)
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(output, dtype=np.float32)

    @staticmethod
    def _load_ir(path: str, target_sr: int) -> AudioArray:
        ir, ir_sr = sf.read(path, always_2d=True, dtype="float32")
        ir_audio = np.asarray(ir, dtype=np.float32)

        if int(ir_sr) != target_sr:
            gcd = math.gcd(int(ir_sr), int(target_sr))
            up = int(target_sr // gcd)
            down = int(ir_sr // gcd)
            ir_audio = resample_poly(ir_audio, up=up, down=down, axis=0).astype(np.float32)

        return ir_audio

    @staticmethod
    def _align_ir_channels(ir: AudioArray, input_channels: int) -> AudioArray:
        ir_channels = ir.shape[1]
        if ir_channels == input_channels:
            return ir
        if ir_channels == 1:
            return np.repeat(ir, input_channels, axis=1)

        mono = np.mean(ir, axis=1, keepdims=True, dtype=np.float32)
        return np.repeat(mono, input_channels, axis=1)

    @staticmethod
    def _normalize_ir(ir: AudioArray, mode: str) -> AudioArray:
        if mode == "none":
            return ir
        if mode == "rms":
            value = float(np.sqrt(np.mean(np.square(ir), dtype=np.float64)))
            if value > 0.0:
                return np.asarray(ir / value, dtype=np.float32)
            return ir

        # default peak normalization
        peak = float(np.max(np.abs(ir)))
        if peak > 0.0:
            return np.asarray(ir / peak, dtype=np.float32)
        return ir

    def _partitioned_convolve(
        self, x: AudioArray, ir: AudioArray, partition_size: int
    ) -> AudioArray:
        x_len, channels = x.shape
        ir_len = ir.shape[0]
        tail_samples = max(0, ir_len - 1)
        out_len = x_len + tail_samples
        output = np.zeros((out_len, channels), dtype=np.float32)

        fft_size = 1
        while fft_size < (2 * partition_size):
            fft_size <<= 1

        for channel in range(channels):
            output[:, channel] = self._partitioned_convolve_mono(
                x[:, channel], ir[:, channel], partition_size, fft_size
            )

        return output

    @staticmethod
    def _partitioned_convolve_mono(
        x: npt.NDArray[np.float32],
        ir: npt.NDArray[np.float32],
        partition_size: int,
        fft_size: int,
    ) -> npt.NDArray[np.float32]:
        x_len = x.shape[0]
        ir_len = ir.shape[0]
        tail_samples = max(0, ir_len - 1)
        output_len = x_len + tail_samples

        n_parts = int(np.ceil(ir_len / partition_size))
        ir_parts_fft = np.zeros((n_parts, fft_size // 2 + 1), dtype=np.complex64)
        for part_idx in range(n_parts):
            start = part_idx * partition_size
            end = min(ir_len, start + partition_size)
            part = np.zeros(fft_size, dtype=np.float32)
            part[: end - start] = ir[start:end]
            ir_parts_fft[part_idx, :] = np.fft.rfft(part).astype(np.complex64)

        hist = np.zeros((n_parts, fft_size // 2 + 1), dtype=np.complex64)
        hist_index = 0

        total_blocks = int(np.ceil(x_len / partition_size)) + int(
            np.ceil(tail_samples / partition_size)
        )
        overlap = np.zeros(fft_size - partition_size, dtype=np.float32)

        out_cursor = 0
        out = np.zeros(total_blocks * partition_size, dtype=np.float32)

        for block_idx in range(total_blocks):
            start = block_idx * partition_size
            end = min(x_len, start + partition_size)
            x_block = np.zeros(fft_size, dtype=np.float32)
            if start < x_len:
                x_block[: end - start] = x[start:end]

            hist[hist_index, :] = np.fft.rfft(x_block).astype(np.complex64)

            acc = np.zeros(fft_size // 2 + 1, dtype=np.complex64)
            for part_idx in range(n_parts):
                hist_slot = (hist_index - part_idx) % n_parts
                acc += ir_parts_fft[part_idx, :] * hist[hist_slot, :]

            y_block = np.fft.irfft(acc, n=fft_size).astype(np.float32)
            y_block[: overlap.shape[0]] += overlap

            out[out_cursor : out_cursor + partition_size] = y_block[:partition_size]
            out_cursor += partition_size
            overlap = y_block[partition_size:]

            hist_index = (hist_index + 1) % n_parts

        return out[:output_len]
