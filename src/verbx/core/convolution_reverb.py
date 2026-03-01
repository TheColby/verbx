"""Partitioned-FFT convolution reverb engine.

This module supports two execution styles:

- in-memory partitioned convolution for general use and CUDA path parity,
- streaming block convolution for long files when post-processing allows it.

It also handles multichannel IR routing, including matrix-packed IR layouts
(`input-major` and `output-major`) for true cross-channel convolution maps.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy import fft as sp_fft
from scipy.signal import resample_poly

from verbx.core.engine_base import ReverbEngine
from verbx.io.audio import ensure_mono_or_stereo

AudioArray = npt.NDArray[np.float32]
LOGGER = logging.getLogger(__name__)

_cupy_module: Any | None = None
_cupy_probed = False


@dataclass(slots=True)
class _StreamConvolverState:
    """State for one input->output IR route in streaming mode."""

    ir_parts_fft: npt.NDArray[np.complex64]
    hist: npt.NDArray[np.complex64]
    overlap: npt.NDArray[np.float32]
    hist_index: int
    n_parts: int
    partition_size: int
    fft_size: int


@dataclass(slots=True)
class ConvolutionReverbConfig:
    """Configuration for the convolution engine."""

    wet: float = 0.8
    dry: float = 0.2
    ir_path: str | None = None
    ir_normalize: str = "peak"
    ir_matrix_layout: str = "output-major"
    partition_size: int = 16_384
    tail_limit: float | None = None
    threads: int | None = None
    device: str = "cpu"


class ConvolutionReverbEngine(ReverbEngine):
    """Block-based partitioned FFT convolution supporting long IRs.

    Notes for contributors:

    - ``process`` is the generic in-memory path (CPU or CUDA).
    - ``process_streaming_file`` is optimized for large-file CPU workflows and
      avoids loading full input/output into RAM.
    """

    def __init__(self, config: ConvolutionReverbConfig) -> None:
        self._config = config
        self._workers = max(1, int(config.threads)) if config.threads is not None else 1
        self._backend = "cpu-scipyfft"
        self._cupy = None
        if config.device == "cuda":
            self._cupy = _get_cupy_module()
            if self._cupy is not None:
                self._backend = "cuda-cupy"
            else:
                LOGGER.warning(
                    "CUDA requested for convolution but CuPy/CUDA is unavailable; "
                    "using CPU FFT backend."
                )

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

        ir = self._normalize_ir(ir, self._config.ir_normalize)
        ir_matrix, out_channels = self._build_ir_matrix(
            ir=ir,
            input_channels=x.shape[1],
            layout=self._config.ir_matrix_layout,
        )

        part_size = max(256, int(self._config.partition_size))
        wet = self._partitioned_convolve_matrix(x, ir_matrix, part_size)
        dry = self._build_dry_for_output(x, out_channels, out_len=wet.shape[0])

        output = (self._config.dry * dry) + (self._config.wet * wet)
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(output, dtype=np.float32)

    def backend_name(self) -> str:
        """Return selected convolution backend."""
        return self._backend

    def process_streaming_file(
        self,
        infile: str,
        outfile: str,
        output_subtype: str | None = None,
    ) -> dict[str, int | float]:
        """Process file via block streaming.

        Returns a compact stats dictionary used by the pipeline/report layer.
        """
        if self._config.ir_path is None:
            msg = "Convolution engine requires --ir PATH"
            raise ValueError(msg)

        if self._cupy is not None:
            # CUDA backend currently reuses the in-memory path; streaming mode
            # is CPU-only for now.
            audio, sr = sf.read(infile, always_2d=True, dtype="float32")
            rendered = self.process(np.asarray(audio, dtype=np.float32), int(sr))
            sf.write(outfile, rendered, int(sr), subtype=output_subtype)
            peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
            return {
                "sample_rate": int(sr),
                "channels": int(rendered.shape[1]),
                "input_samples": int(audio.shape[0]),
                "output_samples": int(rendered.shape[0]),
                "input_peak_linear": peak,
            }

        with sf.SoundFile(infile, mode="r") as src:
            sr = int(src.samplerate)
            input_channels = int(src.channels)
            ir = self._load_ir(self._config.ir_path, sr)
            if self._config.tail_limit is not None:
                max_tail = max(0, int(self._config.tail_limit * sr))
                ir = ir[: max_tail + 1, :]
            ir = self._normalize_ir(ir, self._config.ir_normalize)
            ir_matrix, out_channels = self._build_ir_matrix(
                ir=ir,
                input_channels=input_channels,
                layout=self._config.ir_matrix_layout,
            )

            partition_size = max(256, int(self._config.partition_size))
            states: list[list[_StreamConvolverState | None]] = []
            for in_idx in range(input_channels):
                row: list[_StreamConvolverState | None] = []
                for out_idx in range(out_channels):
                    ir_vec = ir_matrix[in_idx, out_idx, :]
                    if np.any(np.abs(ir_vec) > 0.0):
                        row.append(
                            self._build_stream_state(ir=ir_vec, partition_size=partition_size)
                        )
                    else:
                        row.append(None)
                states.append(row)

            input_samples = 0
            output_samples = 0
            input_peak_linear = 0.0
            tail_remaining = max(0, int(ir.shape[0] - 1))

            with sf.SoundFile(
                outfile,
                mode="w",
                samplerate=sr,
                channels=out_channels,
                subtype=output_subtype,
            ) as dst:
                for block in src.blocks(
                    blocksize=partition_size,
                    dtype="float32",
                    always_2d=True,
                ):
                    in_block = np.asarray(block, dtype=np.float32)
                    samples = int(in_block.shape[0])
                    if samples == 0:
                        continue
                    input_samples += samples
                    block_peak = float(np.max(np.abs(in_block)))
                    if block_peak > input_peak_linear:
                        input_peak_linear = block_peak

                    wet_block = self._stream_accumulate_wet(states, in_block, out_channels)
                    dry_block = self._build_dry_for_output(in_block, out_channels, out_len=samples)

                    out_block = (self._config.dry * dry_block) + (self._config.wet * wet_block)
                    out_block = np.nan_to_num(out_block, nan=0.0, posinf=0.0, neginf=0.0)
                    dst.write(np.asarray(out_block, dtype=np.float32))
                    output_samples += samples

                # Flush remaining IR tail by feeding silent input blocks.
                while tail_remaining > 0:
                    tail_block_samples = min(partition_size, tail_remaining)
                    zeros_block = np.zeros((tail_block_samples, input_channels), dtype=np.float32)
                    wet_tail = self._stream_accumulate_wet(states, zeros_block, out_channels)
                    out_tail = np.nan_to_num(
                        np.asarray(self._config.wet * wet_tail, dtype=np.float32),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    dst.write(out_tail)
                    output_samples += tail_block_samples
                    tail_remaining -= tail_block_samples

        return {
            "sample_rate": sr,
            "channels": out_channels,
            "input_samples": input_samples,
            "output_samples": output_samples,
            "input_peak_linear": input_peak_linear,
        }

    @staticmethod
    def _load_ir(path: str, target_sr: int) -> AudioArray:
        """Load IR and resample to the input sample-rate when needed."""
        ir, ir_sr = sf.read(path, always_2d=True, dtype="float32")
        ir_audio = np.asarray(ir, dtype=np.float32)

        if int(ir_sr) != target_sr:
            gcd = math.gcd(int(ir_sr), int(target_sr))
            up = int(target_sr // gcd)
            down = int(ir_sr // gcd)
            ir_audio = resample_poly(ir_audio, up=up, down=down, axis=0).astype(np.float32)

        return ir_audio

    @staticmethod
    def _normalize_ir(ir: AudioArray, mode: str) -> AudioArray:
        """Normalize IR according to requested policy (`none|rms|peak`)."""
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

    @staticmethod
    def _build_dry_for_output(
        x: AudioArray,
        out_channels: int,
        out_len: int,
    ) -> AudioArray:
        """Map dry signal to output channel topology.

        Rules:
        - identical channel count -> passthrough,
        - mono output -> average input channels,
        - mono input -> replicate across output channels,
        - otherwise copy overlapping channel range.
        """
        in_channels = int(x.shape[1])
        dry = np.zeros((out_len, out_channels), dtype=np.float32)
        copy_len = min(int(x.shape[0]), out_len)
        block = x[:copy_len, :]

        if out_channels == in_channels:
            dry[:copy_len, :] = block
            return dry

        if out_channels == 1:
            dry[:copy_len, 0] = np.mean(block, axis=1, dtype=np.float32)
            return dry

        if in_channels == 1:
            dry[:copy_len, :] = np.repeat(block, out_channels, axis=1)
            return dry

        mapped = min(in_channels, out_channels)
        dry[:copy_len, :mapped] = block[:, :mapped]
        return dry

    def _build_ir_matrix(
        self,
        ir: AudioArray,
        input_channels: int,
        layout: str,
    ) -> tuple[npt.NDArray[np.float32], int]:
        """Resolve IR channel layout into an ``[in, out, taps]`` matrix."""
        ir_len = int(ir.shape[0])
        ir_channels = int(ir.shape[1])

        if input_channels < 1:
            msg = "Input must have at least one channel for convolution."
            raise ValueError(msg)

        if ir_channels == 1:
            # Mono IR is broadcast diagonally across matching in/out channels.
            out_channels = input_channels
            matrix = np.zeros((input_channels, out_channels, ir_len), dtype=np.float32)
            for ch in range(input_channels):
                matrix[ch, ch, :] = ir[:, 0]
            return matrix, out_channels

        if ir_channels == input_channels:
            # Per-channel IR behaves like channel-independent convolution.
            out_channels = input_channels
            matrix = np.zeros((input_channels, out_channels, ir_len), dtype=np.float32)
            for ch in range(input_channels):
                matrix[ch, ch, :] = ir[:, ch]
            return matrix, out_channels

        if ir_channels % input_channels != 0:
            msg = (
                "IR channel layout incompatible with input channels. "
                f"Input channels={input_channels}, IR channels={ir_channels}. "
                "Use mono IR, matching channel count IR, or matrix-packed IR with "
                "channels divisible by input channels."
            )
            raise ValueError(msg)

        out_channels = ir_channels // input_channels
        if out_channels < 1:
            msg = "Invalid IR matrix output channels resolved from IR channel layout."
            raise ValueError(msg)

        if layout not in {"output-major", "input-major"}:
            msg = f"Unsupported IR matrix layout: {layout}"
            raise ValueError(msg)

        # Matrix-packed IR route map:
        # - output-major: index = out * in_channels + in
        # - input-major:  index = in * out_channels + out
        matrix = np.zeros((input_channels, out_channels, ir_len), dtype=np.float32)
        for out_idx in range(out_channels):
            for in_idx in range(input_channels):
                if layout == "output-major":
                    ir_idx = (out_idx * input_channels) + in_idx
                else:
                    ir_idx = (in_idx * out_channels) + out_idx
                matrix[in_idx, out_idx, :] = ir[:, ir_idx]

        return matrix, out_channels

    def _partitioned_convolve_matrix(
        self,
        x: AudioArray,
        ir_matrix: npt.NDArray[np.float32],
        partition_size: int,
    ) -> AudioArray:
        """Convolve an ``M``-channel input against an ``M x N`` IR matrix."""
        if self._cupy is not None:
            return self._partitioned_convolve_cuda_matrix(x, ir_matrix, partition_size)

        x_len, in_channels = x.shape
        _, out_channels, ir_len = ir_matrix.shape
        tail_samples = max(0, ir_len - 1)
        out_len = x_len + tail_samples
        output = np.zeros((out_len, out_channels), dtype=np.float32)

        fft_size = 1
        while fft_size < (2 * partition_size):
            fft_size <<= 1

        for in_idx in range(in_channels):
            x_ch = x[:, in_idx]
            for out_idx in range(out_channels):
                ir_ch = ir_matrix[in_idx, out_idx, :]
                if not np.any(np.abs(ir_ch) > 0.0):
                    continue
                output[:, out_idx] += self._partitioned_convolve_mono(
                    x_ch,
                    ir_ch,
                    partition_size,
                    fft_size,
                )

        return output

    def _stream_accumulate_wet(
        self,
        states: list[list[_StreamConvolverState | None]],
        input_block: AudioArray,
        out_channels: int,
    ) -> AudioArray:
        """Accumulate one streamed input block across all active route states."""
        samples = int(input_block.shape[0])
        wet = np.zeros((samples, out_channels), dtype=np.float32)
        for in_idx, row in enumerate(states):
            x_ch = input_block[:, in_idx]
            for out_idx, state in enumerate(row):
                if state is None:
                    continue
                wet[:, out_idx] += self._stream_process_block(state, x_ch)
        return wet

    def _build_stream_state(
        self, ir: npt.NDArray[np.float32], partition_size: int
    ) -> _StreamConvolverState:
        """Precompute FFT partitions and initialize overlap/history buffers."""
        ir_len = int(ir.shape[0])
        n_parts = int(np.ceil(ir_len / partition_size))
        fft_size = 1
        while fft_size < (2 * partition_size):
            fft_size <<= 1

        ir_parts_fft = np.zeros((n_parts, fft_size // 2 + 1), dtype=np.complex64)
        for part_idx in range(n_parts):
            start = part_idx * partition_size
            end = min(ir_len, start + partition_size)
            part = np.zeros(fft_size, dtype=np.float32)
            part[: end - start] = ir[start:end]
            ir_parts_fft[part_idx, :] = np.asarray(
                sp_fft.rfft(part, workers=self._workers),
                dtype=np.complex64,
            )

        hist = np.zeros_like(ir_parts_fft)
        overlap = np.zeros(fft_size - partition_size, dtype=np.float32)
        return _StreamConvolverState(
            ir_parts_fft=ir_parts_fft,
            hist=hist,
            overlap=overlap,
            hist_index=0,
            n_parts=n_parts,
            partition_size=partition_size,
            fft_size=fft_size,
        )

    def _stream_process_block(
        self,
        state: _StreamConvolverState,
        x_block_in: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Process one block through one input->output partitioned route."""
        samples = int(x_block_in.shape[0])
        x_block = np.zeros(state.fft_size, dtype=np.float32)
        x_block[:samples] = x_block_in

        # Insert the newest input partition spectrum into the ring buffer.
        state.hist[state.hist_index, :] = np.asarray(
            sp_fft.rfft(x_block, workers=self._workers),
            dtype=np.complex64,
        )

        # Circular convolution in frequency domain using partition history.
        acc = np.zeros(state.fft_size // 2 + 1, dtype=np.complex64)
        for part_idx in range(state.n_parts):
            hist_slot = (state.hist_index - part_idx) % state.n_parts
            acc += state.ir_parts_fft[part_idx, :] * state.hist[hist_slot, :]

        y_block = np.asarray(
            sp_fft.irfft(acc, n=state.fft_size, workers=self._workers),
            dtype=np.float32,
        )
        # Overlap-add the carried tail from the previous block.
        y_block[: state.overlap.shape[0]] += state.overlap
        state.overlap = y_block[state.partition_size :]
        state.hist_index = (state.hist_index + 1) % state.n_parts
        return np.asarray(y_block[:samples], dtype=np.float32)

    def _partitioned_convolve_mono(
        self,
        x: npt.NDArray[np.float32],
        ir: npt.NDArray[np.float32],
        partition_size: int,
        fft_size: int,
    ) -> npt.NDArray[np.float32]:
        """Reference CPU partitioned convolution for one input/IR channel pair."""
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
            ir_parts_fft[part_idx, :] = np.asarray(
                sp_fft.rfft(part, workers=self._workers),
                dtype=np.complex64,
            )

        # ``hist`` is a circular spectrum buffer of the most recent input blocks.
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

            hist[hist_index, :] = np.asarray(
                sp_fft.rfft(x_block, workers=self._workers),
                dtype=np.complex64,
            )

            acc = np.zeros(fft_size // 2 + 1, dtype=np.complex64)
            for part_idx in range(n_parts):
                hist_slot = (hist_index - part_idx) % n_parts
                acc += ir_parts_fft[part_idx, :] * hist[hist_slot, :]

            y_block = np.asarray(
                sp_fft.irfft(acc, n=fft_size, workers=self._workers),
                dtype=np.float32,
            )
            y_block[: overlap.shape[0]] += overlap

            out[out_cursor : out_cursor + partition_size] = y_block[:partition_size]
            out_cursor += partition_size
            overlap = y_block[partition_size:]

            hist_index = (hist_index + 1) % n_parts

        return out[:output_len]

    def _partitioned_convolve_cuda_matrix(
        self,
        x: AudioArray,
        ir_matrix: npt.NDArray[np.float32],
        partition_size: int,
    ) -> AudioArray:
        """CUDA matrix convolution wrapper around per-route mono kernels."""
        cp = self._cupy
        if cp is None:
            msg = "CuPy module not loaded for CUDA convolution backend"
            raise RuntimeError(msg)

        x_len, in_channels = x.shape
        _, out_channels, ir_len = ir_matrix.shape
        tail_samples = max(0, ir_len - 1)
        out_len = x_len + tail_samples
        output = np.zeros((out_len, out_channels), dtype=np.float32)

        fft_size = 1
        while fft_size < (2 * partition_size):
            fft_size <<= 1

        x_gpu = [cp.asarray(x[:, in_idx], dtype=cp.float32) for in_idx in range(in_channels)]

        for out_idx in range(out_channels):
            acc = np.zeros(out_len, dtype=np.float32)
            for in_idx in range(in_channels):
                ir_ch = ir_matrix[in_idx, out_idx, :]
                if not np.any(np.abs(ir_ch) > 0.0):
                    continue
                out_pair = self._partitioned_convolve_mono_cuda(
                    cp=cp,
                    x=x_gpu[in_idx],
                    ir=cp.asarray(ir_ch, dtype=cp.float32),
                    partition_size=partition_size,
                    fft_size=fft_size,
                )
                acc += cp.asnumpy(out_pair)
            output[:, out_idx] = acc

        return output

    @staticmethod
    def _partitioned_convolve_mono_cuda(
        cp: Any,
        x: Any,
        ir: Any,
        partition_size: int,
        fft_size: int,
    ) -> Any:
        """CuPy implementation of mono partitioned convolution."""
        x_len = int(x.shape[0])
        ir_len = int(ir.shape[0])
        tail_samples = max(0, ir_len - 1)
        output_len = x_len + tail_samples

        n_parts = int(np.ceil(ir_len / partition_size))
        ir_parts_fft = cp.zeros((n_parts, fft_size // 2 + 1), dtype=cp.complex64)
        for part_idx in range(n_parts):
            start = part_idx * partition_size
            end = min(ir_len, start + partition_size)
            part = cp.zeros(fft_size, dtype=cp.float32)
            part[: end - start] = ir[start:end]
            ir_parts_fft[part_idx, :] = cp.fft.rfft(part).astype(cp.complex64)

        hist = cp.zeros((n_parts, fft_size // 2 + 1), dtype=cp.complex64)
        hist_index = 0

        total_blocks = int(np.ceil(x_len / partition_size)) + int(
            np.ceil(tail_samples / partition_size)
        )
        overlap = cp.zeros(fft_size - partition_size, dtype=cp.float32)

        out_cursor = 0
        out = cp.zeros(total_blocks * partition_size, dtype=cp.float32)

        for block_idx in range(total_blocks):
            start = block_idx * partition_size
            end = min(x_len, start + partition_size)
            x_block = cp.zeros(fft_size, dtype=cp.float32)
            if start < x_len:
                x_block[: end - start] = x[start:end]

            hist[hist_index, :] = cp.fft.rfft(x_block).astype(cp.complex64)

            acc = cp.zeros(fft_size // 2 + 1, dtype=cp.complex64)
            for part_idx in range(n_parts):
                hist_slot = (hist_index - part_idx) % n_parts
                acc += ir_parts_fft[part_idx, :] * hist[hist_slot, :]

            y_block = cp.fft.irfft(acc, n=fft_size).astype(cp.float32)
            y_block[: overlap.shape[0]] += overlap

            out[out_cursor : out_cursor + partition_size] = y_block[:partition_size]
            out_cursor += partition_size
            overlap = y_block[partition_size:]

            hist_index = (hist_index + 1) % n_parts

        return out[:output_len]


def _get_cupy_module() -> Any | None:
    """Probe CuPy once and cache the module handle when CUDA is available."""
    global _cupy_module, _cupy_probed
    if _cupy_probed:
        return _cupy_module

    _cupy_probed = True
    try:
        import cupy as cp  # type: ignore[import-untyped]

        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count < 1:
            return None
        _cupy_module = cp
        return _cupy_module
    except Exception:
        return None
