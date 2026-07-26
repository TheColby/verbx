"""Deterministic spectral dereverberation for offline CLI processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from scipy.signal import istft, stft
from scipy.signal import windows as signal_windows

from verbx.io.audio import ensure_mono_or_stereo

AudioArray = npt.NDArray[np.float64]
DereverbMode = Literal["wiener", "spectral_sub"]

DEREVERB_WINDOW_CHOICES = frozenset(
    {
        "barthann",
        "bartlett",
        "blackman",
        "blackmanharris",
        "bohman",
        "boxcar",
        "chebwin",
        "cosine",
        "dpss",
        "exponential",
        "flattop",
        "gaussian",
        "general_cosine",
        "general_gaussian",
        "hamming",
        "hann",
        "kaiser",
        "kaiser_bessel_derived",
        "lanczos",
        "nuttall",
        "parzen",
        "taylor",
        "triang",
        "tukey",
    }
)
_DEREVERB_WINDOW_ALIASES = {
    "blackman_harris": "blackmanharris",
    "chebyshev": "chebwin",
    "general-cosine": "general_cosine",
    "general-gaussian": "general_gaussian",
    "kaiser-bessel-derived": "kaiser_bessel_derived",
    "kbd": "kaiser_bessel_derived",
    "rect": "boxcar",
    "rectangular": "boxcar",
    "triangle": "triang",
}


@dataclass(slots=True)
class DereverbConfig:
    """Configuration for deterministic spectral dereverberation."""

    mode: DereverbMode = "wiener"
    strength: float = 0.65
    floor: float = 0.08
    window_ms: float = 42.67
    hop_ms: float = 10.67
    tail_ms: float = 220.0
    pre_emphasis: float = 0.0
    mix: float = 1.0
    analysis_window: str = "hann"
    synthesis_window: str | None = None
    window_symmetric: bool = False
    window_alpha: float = 0.5
    window_beta: float = 14.0
    window_std: float = 2.5
    window_power: float = 1.5
    window_atten_db: float = 100.0
    window_nbar: int = 4
    window_nw: float = 2.5
    window_tau: float = 3.0
    window_weights: tuple[float, ...] = ()


@dataclass(slots=True)
class LiveDereverbConfig:
    """Configuration for low-latency realtime dereverberation."""

    mode: DereverbMode = "wiener"
    strength: float = 0.65
    floor: float = 0.08
    window_ms: float = 16.0
    hop_ms: float = 8.0
    tail_ms: float = 120.0
    pre_emphasis: float = 0.0
    mix: float = 1.0
    max_atten_db: float = 18.0
    stereo_link: bool = True
    input_gain_db: float = 0.0
    output_gain_db: float = 0.0
    analysis_window: str = "hann"
    synthesis_window: str | None = None
    window_symmetric: bool = False
    window_alpha: float = 0.5
    window_beta: float = 14.0
    window_std: float = 2.5
    window_power: float = 1.5
    window_atten_db: float = 100.0
    window_nbar: int = 4
    window_nw: float = 2.5
    window_tau: float = 3.0
    window_weights: tuple[float, ...] = ()


@dataclass(slots=True)
class LiveDereverbProcessor:
    """Stateful low-latency streaming dereverb processor."""

    config: LiveDereverbConfig
    sample_rate: int
    input_channels: int
    output_channels: int
    n_fft: int
    hop_samples: int
    latency_samples: int
    analysis_window: npt.NDArray[np.float64]
    synthesis_window: npt.NDArray[np.float64]
    window_energy: npt.NDArray[np.float64]
    analysis_frames: npt.NDArray[np.float64]
    overlap_add: npt.NDArray[np.float64]
    overlap_norm: npt.NDArray[np.float64]
    late_magnitude: npt.NDArray[np.float64]
    dry_delay: npt.NDArray[np.float64]
    pre_emphasis_state: npt.NDArray[np.float64]
    de_emphasis_state: npt.NDArray[np.float64]
    input_gain_linear: float
    output_gain_linear: float
    min_gain: float
    late_alpha: float
    frame_cursor: int = 0

    def process_block(self, block: AudioArray) -> AudioArray:
        """Process one realtime block with fixed-hop streaming STFT."""
        x = ensure_mono_or_stereo(block)
        samples = int(x.shape[0])
        if samples <= 0:
            return np.zeros((0, self.output_channels), dtype=np.float64)
        if int(x.shape[1]) != int(self.input_channels):
            msg = (
                "Live dereverb received a block with unexpected channel count: "
                f"{x.shape[1]} != {self.input_channels}"
            )
            raise ValueError(msg)
        if (samples % int(self.hop_samples)) != 0:
            msg = (
                "Live dereverb requires the realtime block size to be divisible by the "
                f"resolved hop size ({self.hop_samples} samples)."
            )
            raise ValueError(msg)

        dry_input = np.asarray(x * self.input_gain_linear, dtype=np.float64)
        processed_input = np.asarray(dry_input, dtype=np.float64)
        if float(self.config.pre_emphasis) > 0.0:
            processed_input = self._apply_pre_emphasis_live(processed_input)

        output = np.zeros((samples, self.output_channels), dtype=np.float64)
        hop = int(self.hop_samples)
        mix = float(np.clip(self.config.mix, 0.0, 1.0))
        stereo_link = bool(self.config.stereo_link and self.input_channels == 2)

        for start in range(0, samples, hop):
            end = start + hop
            hop_dry = np.asarray(dry_input[start:end, :].T, dtype=np.float64)
            hop_in = np.asarray(processed_input[start:end, :].T, dtype=np.float64)

            self.analysis_frames[:, :-hop] = self.analysis_frames[:, hop:]
            self.analysis_frames[:, -hop:] = hop_in

            spectra = np.fft.rfft(
                self.analysis_frames * self.analysis_window[None, :],
                axis=1,
            )
            magnitude = np.abs(spectra)
            self.late_magnitude = (
                self.late_alpha * self.late_magnitude
            ) + ((1.0 - self.late_alpha) * magnitude)

            if stereo_link:
                linked_mag = np.mean(magnitude, axis=0, keepdims=True)
                linked_late = np.mean(self.late_magnitude, axis=0, keepdims=True)
                gain = _compute_realtime_gain(
                    linked_mag,
                    linked_late,
                    mode=self.config.mode,
                    strength=float(self.config.strength),
                    min_gain=float(self.min_gain),
                )
                gain = np.repeat(gain, self.input_channels, axis=0)
            else:
                gain = _compute_realtime_gain(
                    magnitude,
                    self.late_magnitude,
                    mode=self.config.mode,
                    strength=float(self.config.strength),
                    min_gain=float(self.min_gain),
                )

            restored = np.fft.irfft(spectra * gain, n=self.n_fft, axis=1)
            restored *= self.synthesis_window[None, :]
            self.overlap_add += restored
            self.overlap_norm += self.window_energy[None, :]

            restored_hop = self.overlap_add[:, :hop] / np.maximum(
                self.overlap_norm[:, :hop],
                1e-8,
            )
            self.overlap_add[:, :-hop] = self.overlap_add[:, hop:]
            self.overlap_add[:, -hop:] = 0.0
            self.overlap_norm[:, :-hop] = self.overlap_norm[:, hop:]
            self.overlap_norm[:, -hop:] = 0.0

            if float(self.config.pre_emphasis) > 0.0:
                restored_hop = self._apply_de_emphasis_live(restored_hop)

            delayed_dry = self._delay_dry_hop(hop_dry)
            mixed_hop = (mix * restored_hop) + ((1.0 - mix) * delayed_dry)
            output[start:end, :] = np.asarray(
                mixed_hop.T * self.output_gain_linear,
                dtype=np.float64,
            )

        self.frame_cursor += samples
        return np.asarray(
            np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=np.float64,
        )

    def _delay_dry_hop(self, hop_dry: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return a latency-aligned dry hop for mix control."""
        hop = int(hop_dry.shape[1])
        if self.latency_samples <= 0:
            return np.asarray(hop_dry, dtype=np.float64)
        queued = np.concatenate((self.dry_delay, hop_dry), axis=1)
        delayed = np.asarray(queued[:, :hop], dtype=np.float64)
        self.dry_delay = np.asarray(queued[:, hop:], dtype=np.float64)
        return delayed

    def _apply_pre_emphasis_live(self, block: AudioArray) -> AudioArray:
        """Apply causal pre-emphasis to one multichannel block."""
        coef = float(np.clip(self.config.pre_emphasis, 0.0, 0.98))
        if coef <= 0.0:
            return np.asarray(block, dtype=np.float64)
        x_t = np.asarray(block.T, dtype=np.float64)
        out = np.empty_like(x_t, dtype=np.float64)
        out[:, 0] = x_t[:, 0] - (coef * self.pre_emphasis_state)
        if x_t.shape[1] > 1:
            out[:, 1:] = x_t[:, 1:] - (coef * x_t[:, :-1])
        self.pre_emphasis_state = np.asarray(x_t[:, -1], dtype=np.float64)
        return np.asarray(out.T, dtype=np.float64)

    def _apply_de_emphasis_live(
        self,
        block_t: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Apply causal de-emphasis to one channel-first hop block."""
        coef = float(np.clip(self.config.pre_emphasis, 0.0, 0.98))
        if coef <= 0.0:
            return np.asarray(block_t, dtype=np.float64)
        out = np.empty_like(block_t, dtype=np.float64)
        for ch in range(block_t.shape[0]):
            prev = float(self.de_emphasis_state[ch])
            for idx in range(block_t.shape[1]):
                sample = float(block_t[ch, idx]) + (coef * prev)
                out[ch, idx] = sample
                prev = sample
            self.de_emphasis_state[ch] = prev
        return out


def apply_dereverb(audio: AudioArray, sr: int, config: DereverbConfig) -> AudioArray:
    """Apply single-file offline dereverberation using spectral suppression.

    This is intentionally deterministic and does not require model weights.
    """
    x = ensure_mono_or_stereo(audio)
    if x.shape[0] == 0:
        return x.copy()

    n_fft, hop = _resolve_stft_window_hop(
        sr=sr,
        window_ms=float(config.window_ms),
        hop_ms=float(config.hop_ms),
    )
    strength = float(np.clip(config.strength, 0.0, 2.0))
    floor = float(np.clip(config.floor, 1e-6, 1.0))
    tail_s = max(0.001, float(config.tail_ms) / 1000.0)
    mix = float(np.clip(config.mix, 0.0, 1.0))
    pre_emphasis = float(np.clip(config.pre_emphasis, 0.0, 0.98))
    mode = str(config.mode).strip().lower()
    if mode not in {"wiener", "spectral_sub"}:
        msg = f"Unsupported dereverb mode: {config.mode}"
        raise ValueError(msg)
    analysis_window, synthesis_window = _resolve_dereverb_windows(config, n_fft)

    out = np.zeros_like(x, dtype=np.float64)
    for ch in range(x.shape[1]):
        chan = np.asarray(x[:, ch], dtype=np.float64)
        if pre_emphasis > 0.0:
            chan = _apply_pre_emphasis(chan, pre_emphasis)

        _, _, z = stft(
            chan,
            fs=float(sr),
            window=cast(Any, analysis_window),
            nperseg=n_fft,
            noverlap=n_fft - hop,
            nfft=n_fft,
            boundary="zeros",
            padded=True,
        )
        mag = np.abs(z)
        phase = np.angle(z)
        late_mag = _estimate_late_tail(mag, hop_seconds=float(hop) / float(sr), tail_seconds=tail_s)
        if mode == "spectral_sub":
            mag_hat = _apply_spectral_subtraction(mag, late_mag, strength=strength, floor=floor)
        else:
            mag_hat = _apply_wiener_gain(mag, late_mag, strength=strength, floor=floor)
        z_hat = mag_hat * np.exp(1j * phase)
        _, restored = istft(
            z_hat,
            fs=float(sr),
            window=cast(Any, synthesis_window),
            nperseg=n_fft,
            noverlap=n_fft - hop,
            nfft=n_fft,
            input_onesided=True,
            boundary=True,
        )
        restored = np.asarray(restored[: x.shape[0]], dtype=np.float64)
        if restored.shape[0] < x.shape[0]:
            restored = np.pad(restored, (0, x.shape[0] - restored.shape[0]), mode="constant")
        if pre_emphasis > 0.0:
            restored = _apply_de_emphasis(restored, pre_emphasis)
        mixed = (mix * restored) + ((1.0 - mix) * x[:, ch])
        out[:, ch] = np.asarray(mixed, dtype=np.float64)

    return np.asarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)


def create_live_dereverb_processor(
    *,
    sample_rate: int,
    input_channels: int,
    config: LiveDereverbConfig,
) -> LiveDereverbProcessor:
    """Construct a streaming dereverb processor for realtime audio I/O."""
    if int(input_channels) < 1 or int(input_channels) > 2:
        msg = "Low-latency live dereverb currently supports mono or stereo input only."
        raise ValueError(msg)
    n_fft, hop = _resolve_stft_window_hop(
        sr=int(sample_rate),
        window_ms=float(config.window_ms),
        hop_ms=float(config.hop_ms),
    )
    analysis_window, synthesis_window = _resolve_dereverb_windows(config, n_fft)
    bins = (n_fft // 2) + 1
    min_gain = max(
        float(np.clip(config.floor, 1e-6, 1.0)),
        float(10.0 ** (-max(0.0, float(config.max_atten_db)) / 20.0)),
    )
    hop_seconds = float(hop) / float(max(1, sample_rate))
    tail_seconds = max(0.001, float(config.tail_ms) / 1000.0)
    late_alpha = float(np.exp(-hop_seconds / max(hop_seconds, tail_seconds)))
    latency_samples = max(0, int(n_fft - hop))
    channels = int(input_channels)
    return LiveDereverbProcessor(
        config=config,
        sample_rate=int(sample_rate),
        input_channels=channels,
        output_channels=channels,
        n_fft=int(n_fft),
        hop_samples=int(hop),
        latency_samples=int(latency_samples),
        analysis_window=analysis_window,
        synthesis_window=synthesis_window,
        window_energy=np.asarray(analysis_window * synthesis_window, dtype=np.float64),
        analysis_frames=np.zeros((channels, n_fft), dtype=np.float64),
        overlap_add=np.zeros((channels, n_fft), dtype=np.float64),
        overlap_norm=np.zeros((channels, n_fft), dtype=np.float64),
        late_magnitude=np.zeros((channels, bins), dtype=np.float64),
        dry_delay=np.zeros((channels, latency_samples), dtype=np.float64),
        pre_emphasis_state=np.zeros(channels, dtype=np.float64),
        de_emphasis_state=np.zeros(channels, dtype=np.float64),
        input_gain_linear=float(10.0 ** (float(config.input_gain_db) / 20.0)),
        output_gain_linear=float(10.0 ** (float(config.output_gain_db) / 20.0)),
        min_gain=float(min_gain),
        late_alpha=float(late_alpha),
    )


def normalize_dereverb_window_name(name: str) -> str:
    """Normalize a user-facing window name and validate support."""
    normalized = str(name).strip().lower().replace("-", "_")
    normalized = _DEREVERB_WINDOW_ALIASES.get(normalized, normalized)
    if normalized not in DEREVERB_WINDOW_CHOICES:
        options = ", ".join(sorted(DEREVERB_WINDOW_CHOICES))
        raise ValueError(f"Unsupported dereverb window '{name}'. Supported: {options}.")
    return normalized


def parse_dereverb_window_weights(raw: str | None) -> tuple[float, ...]:
    """Parse comma-separated window weights for ``general_cosine`` windows."""
    if raw is None:
        return ()
    cleaned = raw.strip()
    if cleaned == "":
        return ()
    values: list[float] = []
    for token in cleaned.split(","):
        part = token.strip()
        if part == "":
            continue
        try:
            values.append(float(part))
        except ValueError as exc:
            raise ValueError(
                "--window-weights expects a comma-separated float list."
            ) from exc
    if len(values) == 0:
        return ()
    return tuple(values)


def _resolve_dereverb_windows(
    config: DereverbConfig | LiveDereverbConfig,
    length: int,
) -> tuple[AudioArray, AudioArray]:
    """Build analysis/synthesis windows for one dereverb configuration."""
    analysis_name = normalize_dereverb_window_name(str(config.analysis_window))
    synthesis_raw = config.synthesis_window
    synthesis_name = (
        analysis_name
        if synthesis_raw is None or str(synthesis_raw).strip() == ""
        else normalize_dereverb_window_name(str(synthesis_raw))
    )
    analysis_window = _build_dereverb_window(length=length, name=analysis_name, config=config)
    synthesis_window = _build_dereverb_window(length=length, name=synthesis_name, config=config)
    return analysis_window, synthesis_window


def _build_dereverb_window(
    *,
    length: int,
    name: str,
    config: DereverbConfig | LiveDereverbConfig,
) -> AudioArray:
    """Construct one dereverb window from the shared option set."""
    sym = bool(config.window_symmetric)
    alpha = float(config.window_alpha)
    beta = float(config.window_beta)
    std = float(config.window_std)
    power = float(config.window_power)
    atten_db = float(config.window_atten_db)
    nbar = int(config.window_nbar)
    nw = float(config.window_nw)
    tau = float(config.window_tau)
    weights = tuple(float(v) for v in config.window_weights)

    if name == "barthann":
        values = signal_windows.barthann(length, sym=sym)
    elif name == "bartlett":
        values = signal_windows.bartlett(length, sym=sym)
    elif name == "blackman":
        values = signal_windows.blackman(length, sym=sym)
    elif name == "blackmanharris":
        values = signal_windows.blackmanharris(length, sym=sym)
    elif name == "bohman":
        values = signal_windows.bohman(length, sym=sym)
    elif name == "boxcar":
        values = signal_windows.boxcar(length, sym=sym)
    elif name == "chebwin":
        values = signal_windows.chebwin(length, at=max(1e-3, atten_db), sym=sym)
    elif name == "cosine":
        values = signal_windows.cosine(length, sym=sym)
    elif name == "dpss":
        values = signal_windows.dpss(length, NW=max(1e-3, nw), sym=sym)
    elif name == "exponential":
        values = signal_windows.exponential(length, tau=max(1e-6, tau), sym=sym)
    elif name == "flattop":
        values = signal_windows.flattop(length, sym=sym)
    elif name == "gaussian":
        values = signal_windows.gaussian(length, std=max(1e-6, std), sym=sym)
    elif name == "general_cosine":
        resolved_weights = weights if len(weights) > 0 else (0.5, 0.5)
        values = signal_windows.general_cosine(length, resolved_weights, sym=sym)
    elif name == "general_gaussian":
        values = signal_windows.general_gaussian(
            length,
            p=max(1e-6, power),
            sig=max(1e-6, std),
            sym=sym,
        )
    elif name == "hamming":
        values = signal_windows.hamming(length, sym=sym)
    elif name == "hann":
        values = signal_windows.hann(length, sym=sym)
    elif name == "kaiser":
        values = signal_windows.kaiser(length, beta=max(0.0, beta), sym=sym)
    elif name == "kaiser_bessel_derived":
        values = signal_windows.kaiser_bessel_derived(length, beta=max(0.0, beta), sym=sym)
    elif name == "lanczos":
        values = signal_windows.lanczos(length, sym=sym)
    elif name == "nuttall":
        values = signal_windows.nuttall(length, sym=sym)
    elif name == "parzen":
        values = signal_windows.parzen(length, sym=sym)
    elif name == "taylor":
        values = signal_windows.taylor(
            length,
            nbar=max(2, nbar),
            sll=max(1, round(atten_db)),
            norm=True,
            sym=sym,
        )
    elif name == "triang":
        values = signal_windows.triang(length, sym=sym)
    elif name == "tukey":
        values = signal_windows.tukey(length, alpha=max(0.0, alpha), sym=sym)
    else:
        raise ValueError(f"Unsupported dereverb window '{name}'.")
    return np.asarray(values, dtype=np.float64)


def _resolve_stft_window_hop(*, sr: int, window_ms: float, hop_ms: float) -> tuple[int, int]:
    """Map millisecond window/hop settings to stable FFT-friendly samples."""
    win = max(32, round(max(2.0, window_ms) * float(sr) / 1000.0))
    hop = max(8, round(max(1.0, hop_ms) * float(sr) / 1000.0))
    if hop >= win:
        hop = max(8, win // 4)
    n_fft = 1 << max(5, int(np.ceil(np.log2(max(32, win)))))
    if n_fft < win:
        n_fft <<= 1
    return int(n_fft), int(hop)


def _compute_realtime_gain(
    magnitude: AudioArray,
    late_magnitude: AudioArray,
    *,
    mode: DereverbMode,
    strength: float,
    min_gain: float,
) -> AudioArray:
    """Return a stable spectral gain for live dereverb processing."""
    gain_floor = float(np.clip(min_gain, 1e-6, 1.0))
    if mode == "spectral_sub":
        residual = magnitude - (float(strength) * late_magnitude)
        raw_gain = residual / np.maximum(magnitude, 1e-12)
        return np.asarray(np.clip(raw_gain, gain_floor, 1.0), dtype=np.float64)
    if mode != "wiener":
        msg = f"Unsupported dereverb mode: {mode}"
        raise ValueError(msg)
    mag2 = np.square(magnitude)
    late2 = np.square(late_magnitude)
    early2 = np.maximum(mag2 - (float(strength) * late2), 0.0)
    gain = early2 / (early2 + late2 + 1e-12)
    return np.asarray(np.clip(gain, gain_floor, 1.0), dtype=np.float64)


def _estimate_late_tail(
    magnitude: AudioArray,
    *,
    hop_seconds: float,
    tail_seconds: float,
) -> AudioArray:
    """Estimate late reverberant magnitude with exponential smoothing."""
    if magnitude.ndim != 2 or magnitude.shape[1] == 0:
        return np.zeros_like(magnitude, dtype=np.float64)
    alpha = float(np.exp(-hop_seconds / max(hop_seconds, tail_seconds)))
    late = np.zeros_like(magnitude, dtype=np.float64)
    late[:, 0] = np.asarray(magnitude[:, 0], dtype=np.float64)
    blend = 1.0 - alpha
    for t in range(1, magnitude.shape[1]):
        late[:, t] = (alpha * late[:, t - 1]) + (blend * magnitude[:, t])
    return late


def _apply_spectral_subtraction(
    magnitude: AudioArray,
    late_magnitude: AudioArray,
    *,
    strength: float,
    floor: float,
) -> AudioArray:
    """Classical spectral subtraction with proportional floor safeguard."""
    residual = magnitude - (strength * late_magnitude)
    floor_mag = floor * magnitude
    return np.asarray(np.maximum(residual, floor_mag), dtype=np.float64)


def _apply_wiener_gain(
    magnitude: AudioArray,
    late_magnitude: AudioArray,
    *,
    strength: float,
    floor: float,
) -> AudioArray:
    """Apply Wiener-like gain from early-vs-late spectral energy estimate."""
    mag2 = np.square(magnitude)
    late2 = np.square(late_magnitude)
    early2 = np.maximum(mag2 - (strength * late2), 0.0)
    gain = early2 / (early2 + late2 + 1e-12)
    gain = np.clip(gain, floor, 1.0)
    return np.asarray(magnitude * gain, dtype=np.float64)


def _apply_pre_emphasis(signal: npt.NDArray[np.float64], coef: float) -> npt.NDArray[np.float64]:
    """Apply one-tap pre-emphasis before spectral processing."""
    if signal.shape[0] <= 1:
        return signal.copy()
    out = np.empty_like(signal, dtype=np.float64)
    out[0] = signal[0]
    out[1:] = signal[1:] - (coef * signal[:-1])
    return out


def _apply_de_emphasis(signal: npt.NDArray[np.float64], coef: float) -> npt.NDArray[np.float64]:
    """Invert pre-emphasis using stable one-pole recursion."""
    if signal.shape[0] <= 1:
        return signal.copy()
    out = np.empty_like(signal, dtype=np.float64)
    out[0] = signal[0]
    for idx in range(1, signal.shape[0]):
        out[idx] = signal[idx] + (coef * out[idx - 1])
    return out


# ---------------------------------------------------------------------------
# Objective quality metrics (numpy/scipy only — no external deps)
# ---------------------------------------------------------------------------

def _hz_to_bark(freq_hz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert frequency (Hz) to Bark scale (Traunmuller 1990 approximation)."""
    f = np.maximum(freq_hz, 1.0)
    return np.asarray(26.81 * f / (1960.0 + f) - 0.53, dtype=np.float64)


def _bark_weighted_snr_db(
    clean: npt.NDArray[np.float64],
    processed: npt.NDArray[np.float64],
    sample_rate: int,
) -> float:
    """Bark-scale frequency-weighted SNR — PESQ-inspired proxy metric.

    Partitions the spectrum into 24 Bark critical bands, computes signal and
    error power in each band via short-time spectral averaging, then returns a
    Bark-weighted SNR in dB.  Higher values indicate better perceptual quality.
    """
    x = clean.ravel().astype(np.float64)
    y = processed.ravel().astype(np.float64)
    n = max(len(x), len(y))
    x = np.pad(x, (0, n - len(x)))
    y = np.pad(y, (0, n - len(y)))
    error = y - x

    n_fft = int(np.clip(1 << int(np.ceil(np.log2(max(512, n)))), 512, 8192))
    hop = n_fft // 2
    window = np.hanning(n_fft).astype(np.float64)

    sig_power = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    err_power = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    frames = 0
    for start in range(0, max(1, n - n_fft + 1), hop):
        seg_x = x[start : start + n_fft] * window
        seg_e = error[start : start + n_fft] * window
        sig_power += np.abs(np.fft.rfft(seg_x)) ** 2
        err_power += np.abs(np.fft.rfft(seg_e)) ** 2
        frames += 1
    if frames == 0:
        return 0.0
    sig_power /= frames
    err_power /= frames

    freqs = np.asarray(
        np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate)),
        dtype=np.float64,
    )
    bark_bins = _hz_to_bark(freqs)

    band_snrs: list[float] = []
    for b in range(24):
        mask = (bark_bins >= float(b)) & (bark_bins < float(b + 1))
        if not mask.any():
            continue
        s = float(np.sum(sig_power[mask]))
        e = float(np.sum(err_power[mask]))
        if s < 1e-30 and e < 1e-30:
            continue
        band_snrs.append(s / (e + 1e-30))

    if not band_snrs:
        return 0.0
    return float(10.0 * np.log10(float(np.mean(band_snrs)) + 1e-30))


def _stoi_approx(
    clean: npt.NDArray[np.float64],
    processed: npt.NDArray[np.float64],
    sample_rate: int,
) -> float:
    """Short-Time Objective Intelligibility approximation (no external deps).

    Computes the mean Pearson correlation of short-time RMS envelopes between
    clean and processed signals across perceptual 1/3-octave frequency bands.
    Returns a value in ``[0, 1]`` where 1 = perfect intelligibility match.
    """
    from scipy.signal import butter, sosfilt

    x = clean.ravel().astype(np.float64)
    y = processed.ravel().astype(np.float64)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    if n < 64:
        return 0.0

    nyquist = float(sample_rate) / 2.0
    center_freqs = np.array([200.0, 315.0, 500.0, 800.0, 1250.0, 2000.0, 3150.0, 5000.0])
    center_freqs = center_freqs[center_freqs < nyquist * 0.88]
    if len(center_freqs) == 0:
        return 0.0

    frame_len = max(16, int(0.030 * float(sample_rate)))
    bw_factor = 2.0 ** (1.0 / 6.0)
    correlations: list[float] = []

    for fc in center_freqs:
        fl = fc / bw_factor
        fh = min(fc * bw_factor, nyquist * 0.98)
        if fl <= 0.0 or fl >= fh:
            continue
        try:
            sos = butter(2, [fl / nyquist, fh / nyquist], btype="bandpass", output="sos")
            x_b = sosfilt(sos, x)
            y_b = sosfilt(sos, y)
        except Exception:
            continue

        n_frames = n // frame_len
        if n_frames < 2:
            continue
        x_frame_rms = []
        y_frame_rms = []
        for i in range(n_frames):
            start = i * frame_len
            end = (i + 1) * frame_len
            x_slice = np.asarray(x_b[start:end], dtype=np.float64)
            y_slice = np.asarray(y_b[start:end], dtype=np.float64)
            x_frame_rms.append(float(np.sqrt(np.mean(np.square(x_slice)))))
            y_frame_rms.append(float(np.sqrt(np.mean(np.square(y_slice)))))
        x_env = np.array(
            x_frame_rms,
            dtype=np.float64,
        )
        y_env = np.array(
            y_frame_rms,
            dtype=np.float64,
        )
        x_std = float(np.std(x_env))
        y_std = float(np.std(y_env))
        if x_std < 1e-9 or y_std < 1e-9:
            continue
        corr = float(np.mean((x_env - float(np.mean(x_env))) * (y_env - float(np.mean(y_env))))) / (
            x_std * y_std
        )
        correlations.append(float(np.clip(corr, -1.0, 1.0)))

    if not correlations:
        return 0.0
    return float(np.clip(float(np.mean(correlations)), 0.0, 1.0))


def _mcd_db(
    clean: npt.NDArray[np.float64],
    processed: npt.NDArray[np.float64],
    sample_rate: int,
) -> float:
    """Mel-cepstral distortion (MCD) — ASR WER proxy metric.

    Measures the mean Euclidean distance between MFCC vectors (excluding C0)
    of the clean and processed signals.  Lower MCD indicates better spectral
    envelope preservation (correlates with lower ASR word-error rate).
    Returns distortion in dB; 0 dB = identical spectral envelopes.
    """
    import librosa

    hop_length = max(64, int(0.010 * float(sample_rate)))
    n_fft = max(256, 1 << int(np.ceil(np.log2(max(256, int(0.025 * float(sample_rate)))))))
    n_mfcc = 13

    c_clean = librosa.feature.mfcc(
        y=clean.ravel().astype(np.float32),
        sr=int(sample_rate),
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )[1:]
    c_proc = librosa.feature.mfcc(
        y=processed.ravel().astype(np.float32),
        sr=int(sample_rate),
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )[1:]

    n_frames = min(c_clean.shape[1], c_proc.shape[1])
    if n_frames == 0:
        return 0.0

    diff = c_clean[:, :n_frames].astype(np.float64) - c_proc[:, :n_frames].astype(np.float64)
    mcd = (10.0 / np.log(10.0)) * float(np.sqrt(2.0 * float(np.mean(np.sum(diff**2, axis=0)))))
    return float(np.clip(mcd, 0.0, 1e6))


# ---------------------------------------------------------------------------
# Benchmark harness (no external deps beyond numpy/scipy)
# ---------------------------------------------------------------------------

def run_dereverb_benchmark(
    sr: int = 24000,
    duration_s: float = 3.0,
    rt60: float = 1.2,
    configs: list[DereverbConfig] | None = None,
) -> dict[str, object]:
    """Run a synthetic benchmark of the dereverb engine against ground truth.

    Generates a clean test signal, convolves it with a simple exponential decay
    IR to create a reverberant mix, then runs each config and measures quality
    against the original clean signal.

    Parameters
    ----------
    sr:
        Sample rate for the synthetic test signal.
    duration_s:
        Duration of the test signal in seconds.
    rt60:
        Simulated reverberation time (T60) in seconds for the synthetic IR.
    configs:
        List of :class:`DereverbConfig` instances to benchmark.  When ``None``
        both default modes (wiener, spectral_sub) are run at default settings.

    Returns
    -------
    dict
        Benchmark report with keys ``sr``, ``duration_s``, ``rt60``,
        ``clean_rms``, ``reverberant_rms``, ``snr_reverberant_db`` (baseline),
        and a ``results`` list where each entry has ``mode``, ``strength``,
        ``snr_db``, ``spectral_dist_hz``, and ``rms_delta_db``.
    """
    if configs is None:
        configs = [
            DereverbConfig(mode="wiener"),
            DereverbConfig(mode="spectral_sub"),
        ]

    n = int(sr * duration_s)
    rng = np.random.default_rng(42)

    # Clean signal: band-limited noise burst (voice-like spectral shape)
    clean = rng.standard_normal(n).astype(np.float64)
    # Simple spectral shaping: emphasise 200-4000 Hz
    from scipy.signal import butter, sosfilt
    sos = butter(4, [200.0 / (sr / 2), 4000.0 / (sr / 2)], btype="bandpass", output="sos")
    clean = np.asarray(sosfilt(sos, clean), dtype=np.float64)
    peak = float(np.max(np.abs(clean)) + 1e-12)
    clean = (clean * (0.5 / peak)).astype(np.float64)
    clean_2d = clean[:, np.newaxis]  # (samples, 1)

    # Synthetic IR: exponential decay
    ir_len = min(int(sr * rt60 * 2), n)
    t_ir = np.arange(ir_len, dtype=np.float64) / sr
    decay_rate = np.log(1e-3) / rt60
    ir = np.exp(decay_rate * t_ir) * rng.standard_normal(ir_len).astype(np.float64)
    ir = ir / (np.max(np.abs(ir)) + 1e-12) * 0.3

    # Convolve to create reverberant mix
    from scipy.signal import fftconvolve
    reverberant_full = fftconvolve(clean, ir)
    reverberant = reverberant_full[:n].astype(np.float64)[:, np.newaxis]

    def _rms(x: npt.NDArray[np.float64]) -> float:
        return float(np.sqrt(np.mean(np.square(x))))

    def _snr_db(signal: npt.NDArray[np.float64], noise: npt.NDArray[np.float64]) -> float:
        sig_rms = _rms(signal)
        noise_rms = _rms(noise)
        if noise_rms < 1e-12:
            return 60.0
        return float(20.0 * np.log10(sig_rms / noise_rms + 1e-12))

    def _spectral_centroid(x: npt.NDArray[np.float64], sample_rate: int) -> float:
        s = np.abs(np.fft.rfft(x.ravel()))
        freqs = np.fft.rfftfreq(len(x.ravel()), d=1.0 / sample_rate)
        total = float(np.sum(s))
        if total < 1e-12:
            return 0.0
        return float(np.sum(freqs * s) / total)

    clean_rms = _rms(clean_2d)
    rev_rms = _rms(reverberant)
    baseline_noise = reverberant - clean_2d
    baseline_snr = _snr_db(clean_2d, baseline_noise)
    clean_centroid = _spectral_centroid(clean_2d, sr)

    results: list[dict[str, object]] = []
    for cfg in configs:
        processed = apply_dereverb(reverberant, sr, cfg)
        noise = processed - clean_2d
        snr = _snr_db(clean_2d, noise)
        centroid = _spectral_centroid(processed, sr)
        rms_delta = float(
            20.0 * np.log10((_rms(processed) + 1e-12) / (rev_rms + 1e-12))
        )
        bark_snr = _bark_weighted_snr_db(clean_2d, processed, sr)
        bark_snr_rev = _bark_weighted_snr_db(clean_2d, reverberant, sr)
        stoi = _stoi_approx(clean_2d, processed, sr)
        stoi_rev = _stoi_approx(clean_2d, reverberant, sr)
        mcd = _mcd_db(clean_2d, processed, sr)
        mcd_rev = _mcd_db(clean_2d, reverberant, sr)
        results.append(
            {
                "mode": str(cfg.mode),
                "strength": float(cfg.strength),
                "floor": float(cfg.floor),
                "snr_db": round(snr, 3),
                "snr_improvement_db": round(snr - baseline_snr, 3),
                "spectral_centroid_hz": round(centroid, 2),
                "spectral_dist_hz": round(abs(centroid - clean_centroid), 2),
                "rms_delta_db": round(rms_delta, 3),
                "bark_snr_db": round(bark_snr, 3),
                "bark_snr_improvement_db": round(bark_snr - bark_snr_rev, 3),
                "stoi_approx": round(stoi, 4),
                "stoi_improvement": round(stoi - stoi_rev, 4),
                "mcd_db": round(mcd, 3),
                "mcd_improvement_db": round(mcd_rev - mcd, 3),
            }
        )

    baseline_bark_snr = _bark_weighted_snr_db(clean_2d, reverberant, sr)
    baseline_stoi = _stoi_approx(clean_2d, reverberant, sr)
    baseline_mcd = _mcd_db(clean_2d, reverberant, sr)

    return {
        "schema": "dereverb-benchmark-v2",
        "sr": int(sr),
        "duration_s": float(duration_s),
        "rt60": float(rt60),
        "clean_rms": round(clean_rms, 6),
        "reverberant_rms": round(rev_rms, 6),
        "snr_reverberant_db": round(baseline_snr, 3),
        "bark_snr_reverberant_db": round(baseline_bark_snr, 3),
        "stoi_reverberant": round(baseline_stoi, 4),
        "mcd_reverberant_db": round(baseline_mcd, 3),
        "clean_spectral_centroid_hz": round(clean_centroid, 2),
        "results": results,
    }
