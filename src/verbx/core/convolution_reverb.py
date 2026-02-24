from pathlib import Path

import numpy as np
import scipy.signal

from verbx.core.engine_base import ReverbEngine
from verbx.io.audio import read_audio


class ConvolutionReverbEngine(ReverbEngine):
    """Convolution Reverb Engine (Overlap-Add)."""

    def __init__(
        self,
        impulse_response: str | Path | np.ndarray = "",
        wet: float = 0.5,
        dry: float = 0.5,
        ir_sr: int = 44100,
    ):
        self.impulse_response_path = impulse_response
        self.wet = wet
        self.dry = dry
        self.ir_sr = ir_sr

        self.ir_loaded = False
        self.ir: np.ndarray | None = None
        self.overlap_buffer = None
        self.block_size = 0

    def _load_ir(self, sr: int):
        if isinstance(self.impulse_response_path, (str, Path)) and str(
            self.impulse_response_path
        ):
            ir, ir_sr = read_audio(self.impulse_response_path)

            # Resample if needed
            if ir_sr != sr:
                # Calculate new length
                new_len = int(len(ir) * sr / ir_sr)
                # scipy.signal.resample returns ndarray or tuple.
                # If t is not provided, it returns ndarray.
                resampled = scipy.signal.resample(ir, new_len)
                if isinstance(resampled, tuple):
                    ir = resampled[0]
                else:
                    ir = resampled

            # Ensure ir is ndarray
            if not isinstance(ir, np.ndarray):
                 ir = np.array(ir)

            self.ir = ir
        elif isinstance(self.impulse_response_path, np.ndarray):
            self.ir = self.impulse_response_path
        else:
            # Default fallback (dirac delta)
            self.ir = np.zeros((1024, 1), dtype=np.float32)
            self.ir[0] = 1.0

        # Ensure IR is float32
        if self.ir is not None:
            self.ir = self.ir.astype(np.float32)

        self.overlap_buffer = None  # Will init on first process
        self.ir_loaded = True

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if not self.ir_loaded:
            self._load_ir(sr)

        if self.ir is None:
             raise RuntimeError("Failed to load Impulse Response")

        if audio.ndim == 1:
            audio = audio[:, np.newaxis]

        n_samples, n_channels = audio.shape
        ir_len = len(self.ir)
        ir_channels = self.ir.shape[1]

        # Init overlap buffer if needed
        if self.overlap_buffer is None:
            # Buffer shape: (ir_len - 1, n_channels_out)
            # Output channels = max(n_channels, ir_channels)
            out_channels = max(n_channels, ir_channels)
            self.overlap_buffer = np.zeros((ir_len - 1, out_channels), dtype=np.float32)

        # Perform convolution
        # scipy.signal.fftconvolve

        # Handle channels
        out_channels = self.overlap_buffer.shape[1]
        output = np.zeros((n_samples, out_channels), dtype=np.float32)

        # We can optimize this loop
        for c in range(out_channels):
            # Select input channel
            in_ch = audio[:, c % n_channels]
            # Select IR channel
            ir_ch = self.ir[:, c % ir_channels]

            # Convolve
            conv = scipy.signal.fftconvolve(in_ch, ir_ch, mode="full")

            # OLA
            # Add overlap
            buffer_ch = self.overlap_buffer[:, c]
            conv[: len(buffer_ch)] += buffer_ch

            # Extract output block
            output[:, c] = conv[:n_samples]

            # Update overlap buffer
            new_tail = conv[n_samples:]
            # Ensure buffer size matches
            # tail length is ir_len - 1
            self.overlap_buffer[:, c] = new_tail

        # Mix Wet/Dry
        # If dry is mono but wet is stereo, we might need to expand dry
        if n_channels < out_channels:
            dry_sig = np.tile(audio, (1, out_channels // n_channels))
        else:
            dry_sig = audio

        final_output = dry_sig * self.dry + output * self.wet

        return final_output
