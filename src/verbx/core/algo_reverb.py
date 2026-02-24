import numba
import numpy as np

from verbx.core.engine_base import ReverbEngine


# Helper functions for Numba
@numba.jit(nopython=True, cache=True)
def process_allpass_series(
    audio: np.ndarray,
    delays: np.ndarray,
    indices: np.ndarray,
    gains: np.ndarray,
    buffers: np.ndarray,
) -> np.ndarray:
    """
    Process series of allpass filters for diffusion.

    Args:
        audio: Input block (n_samples).
        delays: Delay lengths (n_stages).
        indices: Current write indices (n_stages).
        gains: Feedback gains (n_stages).
        buffers: Delay buffers (n_stages, max_delay).

    Returns:
        Processed audio block.
    """
    n_samples = len(audio)
    n_stages = len(delays)

    # Copy input to temp buffer to process in place effectively
    temp = audio.copy()

    for s in range(n_stages):
        delay_len = delays[s]
        idx = indices[s]
        g = gains[s]
        buf = buffers[s]
        buf_len = len(buf)

        for i in range(n_samples):
            in_samp = temp[i]
            # Read from delay
            read_idx = (idx - delay_len) % buf_len
            delayed = buf[read_idx]

            # Allpass structure: y[n] = -g*x[n] + x[n-D] + g*y[n-D]
            # Standard Schroeder Allpass:
            # v[n] = x[n] + g * v[n-D]  (stored in buffer?) No.
            # Usually:
            # buf_out = buffer[read_ptr]
            # buffer[write_ptr] = input + feedback * buf_out
            # output = buf_out - feedback * buffer[write_ptr] (Wait, standard AP is:)
            # y[n] = -g * x[n] + x[n-D] + g * y[n-D]
            # Implementation:
            # w[n] = x[n] + g * w[n-D]
            # y[n] = w[n-D] - g * w[n]
            # Buffer stores w.

            w_delayed = delayed
            w_new = in_samp + g * w_delayed

            # Write to buffer
            buf[idx] = w_new

            # Output
            out_samp = w_delayed - g * w_new
            temp[i] = out_samp

            idx = (idx + 1) % buf_len

        indices[s] = idx

    return temp


@numba.jit(nopython=True, cache=True)
def process_fdn_block(
    input_block: np.ndarray,
    delay_buffers: np.ndarray,
    write_indices: np.ndarray,
    delay_lengths: np.ndarray,
    feedback_matrix: np.ndarray,
    gains: np.ndarray,
    damping: float,
    damping_states: np.ndarray,
) -> np.ndarray:
    """
    Process FDN block.

    Args:
        input_block: Input audio (n_samples, n_channels_in).
                     (Assumes input is distributed to FDN lines).
        delay_buffers: (n_lines, max_len).
        write_indices: (n_lines).
        delay_lengths: (n_lines).
        feedback_matrix: (n_lines, n_lines).
        gains: (n_lines). Per-line feedback gain.
        damping: Lowpass coefficient (0..1).
        damping_states: (n_lines). Previous output of LPF.

    Returns:
        Output block (n_samples, n_lines).
    """
    n_samples = len(input_block)
    n_lines = len(delay_lengths)
    output_block = np.zeros((n_samples, n_lines), dtype=np.float32)

    # Temp array for feedback vector
    feedback_vec = np.zeros(n_lines, dtype=np.float32)

    for i in range(n_samples):
        # Read from all delay lines
        delayed_vals = np.zeros(n_lines, dtype=np.float32)
        for j in range(n_lines):
            read_idx = (write_indices[j] - delay_lengths[j]) % len(delay_buffers[j])
            delayed_vals[j] = delay_buffers[j][read_idx]

        # Apply damping (1-pole LPF)
        # y[n] = delayed * (1-d) + y[n-1] * d
        # Actually standard damping in FDN is often in the feedback path.
        # Here we damp the delayed signal before matrix.
        damped_vals = np.zeros(n_lines, dtype=np.float32)
        for j in range(n_lines):
            val = delayed_vals[j] * (1.0 - damping) + damping_states[j] * damping
            damping_states[j] = val
            damped_vals[j] = val

        # Matrix mixing
        # feedback = Matrix * damped * gains
        # We can apply gains before or after matrix. Usually per-delay gain is applied.

        # Apply gains
        for j in range(n_lines):
            damped_vals[j] *= gains[j]

        # Matrix multiply
        # feedback_vec = feedback_matrix @ damped_vals
        # Manual matrix mult for numba (or use np.dot)
        feedback_vec[:] = 0.0
        for r in range(n_lines):
            sum_val = 0.0
            for c in range(n_lines):
                sum_val += feedback_matrix[r, c] * damped_vals[c]
            feedback_vec[r] = sum_val

        # Input injection + Feedback
        # Usually input is added to the delay line input.
        # input_block[i] is (n_channels_in).
        # We need to mix input to lines.
        # Assume simple mix: input mono -> all lines, or stereo -> split.
        # Here we assume input_block is already mixed to (n_lines) or we handle it outside?
        # Let's assume input_block is (n_samples, n_lines) for simplicity in this low-level func,
        # or we pass (n_samples, n_in) and a mix matrix.
        # For efficiency, let's assume the caller mixes input to FDN inputs.

        current_input = input_block[i]  # Shape (n_lines)

        lines_input = current_input + feedback_vec

        # Write to delay lines
        for j in range(n_lines):
            idx = write_indices[j]
            # Soft clip protection
            val = lines_input[j]
            if val > 4.0:
                val = 4.0
            elif val < -4.0:
                val = -4.0

            delay_buffers[j][idx] = val
            write_indices[j] = (idx + 1) % len(delay_buffers[j])

        # Output is often the *tapped* signal (delayed).
        # We can output the damped values (late reverb).
        output_block[i, :] = damped_vals

    return output_block


class AlgoReverbEngine(ReverbEngine):
    """Algorithmic Reverb Engine (FDN + Diffusion)."""

    def __init__(
        self,
        rt60: float = 2.0,
        wet: float = 0.5,
        dry: float = 0.5,
        pre_delay_ms: float = 0.0,
        damping: float = 0.2,
        width: float = 1.0,
    ):
        self.rt60 = rt60
        self.wet = wet
        self.dry = dry
        self.width = width
        self.damping = np.clip(damping, 0.0, 0.99)

        # Parameters
        self.n_lines = 8
        self.n_allpass = 4
        self.sr = 44100  # Default, will update in process if changed?
        # Actually process takes sr, but we need to init buffers.
        # We'll re-init if SR changes or lazily init.
        self.initialized = False

    def _init_buffers(self, sr: int):
        self.sr = sr

        # FDN Delay Lines
        # Prime numbers around 50-100ms
        base_delays = np.array(
            [1117, 1361, 1613, 1987, 2399, 2851, 3343, 3889]
        )  # Primes
        scale = sr / 44100.0
        self.delay_lengths = (base_delays * scale).astype(np.int32)

        # Buffers (make them long enough for modulation/max delay)
        # 4s buffer is safe for extreme mods? No, just needs to be > max_delay.
        # But we need enough space.
        max_len = int(np.max(self.delay_lengths) + 4096)
        self.delay_buffers = np.zeros((self.n_lines, max_len), dtype=np.float32)
        self.write_indices = np.zeros(self.n_lines, dtype=np.int32)

        # Feedback Matrix (Hadamard 8x8)
        from scipy.linalg import hadamard

        self.matrix = hadamard(8).astype(np.float32)
        # Normalize matrix to be unitary (divide by sqrt(N))
        self.matrix /= np.sqrt(8.0)

        # Gains based on RT60
        # g = 10^(-3 * delay / rt60)
        self.gains = np.power(
            10.0, -3.0 * (self.delay_lengths / sr) / self.rt60
        ).astype(np.float32)

        # Damping states
        self.damping_states = np.zeros(self.n_lines, dtype=np.float32)

        # Allpass diffusion (per channel)
        # We'll support stereo input, so 2 sets of allpasses?
        # For simplicity, mono diffusion -> split to FDN -> stereo out.
        # Or stereo diffusion.
        # Let's do stereo diffusion: 2 channels * 4 stages.
        ap_delays = np.array([223, 337, 461, 599])  # Primes
        self.ap_delays = (ap_delays * scale).astype(np.int32)
        self.ap_gains = np.array([0.7, 0.7, 0.6, 0.6], dtype=np.float32)

        max_ap_len = int(np.max(self.ap_delays) + 1024)
        self.ap_buffers_L = np.zeros((self.n_allpass, max_ap_len), dtype=np.float32)
        self.ap_indices_L = np.zeros(self.n_allpass, dtype=np.int32)

        self.ap_buffers_R = np.zeros((self.n_allpass, max_ap_len), dtype=np.float32)
        self.ap_indices_R = np.zeros(self.n_allpass, dtype=np.int32)

        self.initialized = True

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if not self.initialized or self.sr != sr:
            self._init_buffers(sr)

        # Ensure input is float32
        audio = audio.astype(np.float32)
        n_samples, n_channels = audio.shape

        # 1. Diffusion (Series Allpass)
        # Split to L/R
        if n_channels >= 2:
            in_L = audio[:, 0].copy()
            in_R = audio[:, 1].copy()
        else:
            in_L = audio[:, 0].copy()
            in_R = audio[:, 0].copy()

        diffused_L = process_allpass_series(
            in_L, self.ap_delays, self.ap_indices_L, self.ap_gains, self.ap_buffers_L
        )
        diffused_R = process_allpass_series(
            in_R, self.ap_delays, self.ap_indices_R, self.ap_gains, self.ap_buffers_R
        )

        # 2. FDN Input Mixing
        # Map L->4 lines, R->4 lines
        fdn_input = np.zeros((n_samples, self.n_lines), dtype=np.float32)
        # Simple mapping: 0,1,2,3 from L; 4,5,6,7 from R
        for k in range(4):
            fdn_input[:, k] = diffused_L * 0.5
            fdn_input[:, k + 4] = diffused_R * 0.5

        # 3. FDN Processing
        fdn_output = process_fdn_block(
            fdn_input,
            self.delay_buffers,
            self.write_indices,
            self.delay_lengths,
            self.matrix,
            self.gains,
            self.damping,
            self.damping_states,
        )

        # 4. Output Mixing (Stereo)
        # Mix 8 lines down to 2
        # Decorrelate:
        # L = sum(lines 0..3) - sum(lines 4..7)?
        # Or just L = 0+2+4+6, R = 1+3+5+7?
        # Let's use a fixed mix for width.
        # If width=0 (mono), mix all.
        # If width=1 (stereo), separate.

        out_L = np.sum(fdn_output[:, 0:4], axis=1) + np.sum(
            fdn_output[:, 4:8], axis=1
        ) * (1.0 - self.width)
        out_R = np.sum(fdn_output[:, 4:8], axis=1) + np.sum(
            fdn_output[:, 0:4], axis=1
        ) * (1.0 - self.width)

        wet_sig = np.stack([out_L, out_R], axis=1)

        # 5. Wet/Dry Mix
        if n_channels == 1:
            dry_sig = np.stack([audio[:, 0], audio[:, 0]], axis=1)
        else:
            dry_sig = audio[:, :2]

        output = dry_sig * self.dry + wet_sig * self.wet

        return output
