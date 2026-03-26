"""verbx public Python API.

Provides a stable, importable surface for using verbx as a library rather than
a CLI tool. Suitable for notebooks, pipelines, and DAW integrations.

Basic usage::

    from verbx.api import render_file, generate_ir, analyze_file
    from verbx.config import RenderConfig
    from verbx.ir import IRGenConfig

    # Render with algorithmic reverb
    report = render_file(
        infile="dry.wav",
        outfile="wet.wav",
        config=RenderConfig(engine="algo", rt60=2.5, wet=0.7),
    )

    # Generate an IR and save it
    audio, sr, meta = generate_ir(IRGenConfig(mode="fdn", duration=4.0, sr=48000))

    # Analyze an audio file
    metrics = analyze_file("wet.wav", include_loudness=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "analyze_file",
    "generate_ir",
    "read_audio",
    "render_file",
    "write_audio",
]

# ---------------------------------------------------------------------------
# Type alias (matches internal AudioArray = npt.NDArray[np.float64])
# ---------------------------------------------------------------------------
AudioArray = npt.NDArray[np.float64]


def render_file(
    infile: str | Path,
    outfile: str | Path,
    config: Any,
) -> dict[str, Any]:
    """Apply reverb processing to an audio file.

    Parameters
    ----------
    infile:
        Path to the dry input audio file. Supports WAV, FLAC, AIFF, and any
        format supported by ``soundfile``.
    outfile:
        Path for the processed output file. Parent directory must exist.
    config:
        A :class:`~verbx.config.RenderConfig` instance describing the reverb
        engine, parameters, and I/O settings.

    Returns
    -------
    dict
        Structured report containing engine name, sample counts, resolved
        runtime settings, and optional per-channel analysis metrics. Keys
        include ``"engine"``, ``"sr"``, ``"input_samples"``,
        ``"output_samples"``, ``"channels"``, ``"config"``, and
        ``"effective"``.

    Raises
    ------
    FileNotFoundError
        If ``infile`` does not exist.
    ValueError
        If ``config`` contains invalid parameter combinations.

    Examples
    --------
    >>> from verbx.api import render_file
    >>> from verbx.config import RenderConfig
    >>> report = render_file(
    ...     "dry.wav",
    ...     "wet.wav",
    ...     RenderConfig(engine="algo", rt60=4.0, wet=0.8, shimmer=True),
    ... )
    >>> report["engine"]
    'algo'
    """
    from verbx.core.pipeline import run_render_pipeline

    return run_render_pipeline(Path(infile), Path(outfile), config)


def generate_ir(
    config: Any,
    *,
    cache_dir: str | Path | None = None,
) -> tuple[AudioArray, int, dict[str, Any]]:
    """Synthesize an impulse response.

    Parameters
    ----------
    config:
        A :class:`~verbx.ir.IRGenConfig` instance describing the IR synthesis
        parameters (mode, duration, sample rate, FDN topology, etc.).
    cache_dir:
        Optional directory for caching generated IRs. When provided, verbx
        will check for a matching cached IR before synthesizing a new one.
        Cache hits are identified by a hash of the config fields.

    Returns
    -------
    audio : ndarray, shape (samples, channels)
        The synthesized impulse response as a float64 array.
    sr : int
        Sample rate of the returned audio.
    meta : dict
        Metadata dictionary including mode, duration, RT60, channel layout,
        and generation parameters.

    Examples
    --------
    >>> from verbx.api import generate_ir
    >>> from verbx.ir import IRGenConfig
    >>> audio, sr, meta = generate_ir(IRGenConfig(mode="fdn", duration=3.0))
    >>> audio.shape
    (144000, 2)
    """
    if cache_dir is not None:
        from verbx.ir.generator import generate_or_load_cached_ir

        audio, sr, meta, _path, _hit = generate_or_load_cached_ir(
            config, Path(cache_dir)
        )
        return audio, sr, meta

    from verbx.ir.generator import generate_ir as _generate_ir

    return _generate_ir(config)


def analyze_file(
    path: str | Path,
    *,
    include_loudness: bool = False,
    include_edr: bool = False,
    ambi_order: int | None = None,
    ambi_normalization: str = "auto",
    ambi_channel_order: str = "auto",
) -> dict[str, float]:
    """Compute analysis metrics for an audio file.

    Parameters
    ----------
    path:
        Path to the audio file to analyze.
    include_loudness:
        Enable EBU R128 metrics: integrated LUFS, true-peak dBTP, and
        loudness range (LRA). Adds ~20-40 % runtime overhead.
    include_edr:
        Enable frequency-dependent Energy Decay Relief (EDR) summary metrics.
    ambi_order:
        When set, compute spherical energy and directionality metrics for
        Ambisonics signals of this order. ``None`` disables spatial metrics.
    ambi_normalization:
        Normalization convention for Ambisonics (``"sn3d"``, ``"fuma"``, or
        ``"auto"``).
    ambi_channel_order:
        Channel ordering convention (``"acn"``, ``"fuma"``, or ``"auto"``).

    Returns
    -------
    dict
        Flat dictionary of float-valued metrics. Always includes: duration,
        samples, channels, rms, rms_dbfs, peak, peak_dbfs, sample_peak_dbfs,
        crest_factor, dc_offset, dynamic_range, energy, spectral_centroid,
        spectral_bandwidth, spectral_rolloff, zero_crossing_rate, and more.

    Examples
    --------
    >>> from verbx.api import analyze_file
    >>> m = analyze_file("wet.wav", include_loudness=True)
    >>> m["rms_dbfs"]
    -18.4
    """
    from verbx.analysis.analyzer import AudioAnalyzer
    from verbx.io.audio import read_audio

    audio, sr = read_audio(str(path))
    return AudioAnalyzer().analyze(
        audio,
        sr,
        include_loudness=include_loudness,
        include_edr=include_edr,
        ambi_order=ambi_order,
        ambi_normalization=ambi_normalization,
        ambi_channel_order=ambi_channel_order,
    )


def read_audio(path: str | Path) -> tuple[AudioArray, int]:
    """Read an audio file into a float64 numpy array.

    Parameters
    ----------
    path:
        Path to the audio file. Supports WAV, FLAC, AIFF, OGG, and any
        format supported by ``soundfile``.

    Returns
    -------
    audio : ndarray, shape (samples, channels)
        Audio data normalized to the range ``[-1.0, 1.0]`` in float64.
    sr : int
        Sample rate in Hz.

    Examples
    --------
    >>> from verbx.api import read_audio
    >>> audio, sr = read_audio("in.wav")
    >>> audio.shape
    (220500, 2)
    """
    from verbx.io.audio import read_audio as _read_audio

    return _read_audio(str(path))


def write_audio(
    path: str | Path,
    audio: AudioArray,
    sr: int,
    *,
    subtype: str | None = None,
) -> None:
    """Write a float64 audio array to a file.

    Parameters
    ----------
    path:
        Output file path. Format is inferred from the extension
        (``.wav``, ``.flac``, ``.aiff``, etc.).
    audio:
        Audio data with shape ``(samples, channels)`` in float64.
    sr:
        Sample rate in Hz.
    subtype:
        Optional ``soundfile`` subtype override (e.g. ``"PCM_24"``,
        ``"PCM_32"``, ``"FLOAT"``). When ``None``, verbx chooses a sensible
        default (``PCM_24`` for WAV, ``VORBIS`` for OGG, etc.).

    Examples
    --------
    >>> from verbx.api import read_audio, write_audio
    >>> audio, sr = read_audio("in.wav")
    >>> write_audio("out.flac", audio * 0.5, sr, subtype="PCM_24")
    """
    from verbx.io.audio import write_audio as _write_audio

    _write_audio(str(path), audio, sr, subtype)
