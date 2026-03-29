"""Deterministic spectral dereverberation for offline CLI processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import istft, stft

from verbx.io.audio import ensure_mono_or_stereo

AudioArray = npt.NDArray[np.float64]
DereverbMode = Literal["wiener", "spectral_sub"]


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

    out = np.zeros_like(x, dtype=np.float64)
    for ch in range(x.shape[1]):
        chan = np.asarray(x[:, ch], dtype=np.float64)
        if pre_emphasis > 0.0:
            chan = _apply_pre_emphasis(chan, pre_emphasis)

        _, _, z = stft(
            chan,
            fs=float(sr),
            window="hann",
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
            window="hann",
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

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
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
    from scipy.signal import butter, sosfilt  # noqa: PLC0415

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
        x_env = np.array(
            [float(np.sqrt(np.mean(x_b[i * frame_len : (i + 1) * frame_len] ** 2))) for i in range(n_frames)]
        )
        y_env = np.array(
            [float(np.sqrt(np.mean(y_b[i * frame_len : (i + 1) * frame_len] ** 2))) for i in range(n_frames)]
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
    import librosa  # noqa: PLC0415

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
    from scipy.signal import butter, sosfilt  # noqa: PLC0415
    sos = butter(4, [200.0 / (sr / 2), 4000.0 / (sr / 2)], btype="bandpass", output="sos")
    clean = sosfilt(sos, clean)
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
    from scipy.signal import fftconvolve  # noqa: PLC0415
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
