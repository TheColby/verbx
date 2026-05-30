# ruff: noqa: B008
"""Dereverb command wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import typer

from verbx.config import OutputSubtype


def _forward(name: str, params: dict[str, Any]) -> None:
    from verbx import cli as cli_module

    return cli_module.get_command_impl(name)(**params)


def dereverb(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    outfile: Path = typer.Argument(..., resolve_path=True),
    mode: Literal["wiener", "spectral_sub"] = typer.Option(
        "wiener",
        "--mode",
        help="Dereverberation mode: wiener or spectral_sub.",
    ),
    strength: float = typer.Option(
        0.65,
        "--strength",
        min=0.0,
        max=2.0,
        help="Suppression strength (higher removes more tail but can add artifacts).",
    ),
    floor: float = typer.Option(
        0.08,
        "--floor",
        min=0.0,
        max=1.0,
        help="Spectral floor to preserve ambience and avoid musical noise.",
    ),
    window_ms: float = typer.Option(
        42.67,
        "--window-ms",
        min=2.0,
        help="STFT analysis window size in milliseconds.",
    ),
    hop_ms: float = typer.Option(
        10.67,
        "--hop-ms",
        min=1.0,
        help="STFT hop size in milliseconds.",
    ),
    tail_ms: float = typer.Option(
        220.0,
        "--tail-ms",
        min=1.0,
        help="Late-field smoothing horizon in milliseconds.",
    ),
    pre_emphasis: float = typer.Option(
        0.0,
        "--pre-emphasis",
        min=0.0,
        max=0.98,
        help="Optional high-frequency emphasis before dereverberation.",
    ),
    mix: float = typer.Option(
        1.0,
        "--mix",
        min=0.0,
        max=1.0,
        help="Wet mix of dereverberated output (1.0 = fully processed).",
    ),
    window_type: str = typer.Option(
        "hann",
        "--window-type",
        help=(
            "Analysis window family "
            "(hann, hamming, blackman, kaiser, dpss, tukey, chebwin, and many more)."
        ),
    ),
    synthesis_window_type: str | None = typer.Option(
        None,
        "--synthesis-window-type",
        help="Optional synthesis window family. Defaults to --window-type.",
    ),
    window_symmetric: bool = typer.Option(
        False,
        "--window-symmetric/--window-periodic",
        help="Use symmetric windows instead of periodic STFT windows.",
    ),
    window_alpha: float = typer.Option(0.5, "--window-alpha", min=0.0),
    window_beta: float = typer.Option(14.0, "--window-beta", min=0.0),
    window_std: float = typer.Option(2.5, "--window-std", min=1e-6),
    window_power: float = typer.Option(1.5, "--window-power", min=1e-6),
    window_atten_db: float = typer.Option(100.0, "--window-atten-db", min=1e-3),
    window_nbar: int = typer.Option(4, "--window-nbar", min=2),
    window_nw: float = typer.Option(2.5, "--window-nw", min=1e-3),
    window_tau: float = typer.Option(3.0, "--window-tau", min=1e-6),
    window_weights: str | None = typer.Option(
        None,
        "--window-weights",
        help="Optional comma-separated weights for general_cosine windows.",
    ),
    out_subtype: OutputSubtype = typer.Option(
        "auto",
        "--out-subtype",
        help="Output subtype: auto, float32, float64, pcm16, pcm24, pcm32.",
    ),
    json_out: Path | None = typer.Option(
        None,
        "--json-out",
        resolve_path=True,
        help="Optional path for detailed dereverb metrics JSON.",
    ),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress summary table output."),
    benchmark: bool = typer.Option(
        False,
        "--benchmark",
        help=(
            "Run a synthetic quality benchmark (SNR, spectral distance) without processing "
            "any files. --json-out writes the benchmark report when provided."
        ),
    ),
    benchmark_rt60: float = typer.Option(
        1.2,
        "--benchmark-rt60",
        min=0.1,
        max=60.0,
        help="Simulated RT60 in seconds for the synthetic benchmark IR.",
    ),
    benchmark_sr: int = typer.Option(
        24000,
        "--benchmark-sr",
        min=8000,
        help="Sample rate for synthetic benchmark signal.",
    ),
) -> None:
    params = dict(locals())
    _forward("_dereverb_impl", params)
