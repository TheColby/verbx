"""Framewise analysis export helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import numpy.typing as npt

from verbx.analysis.features_spectral import spectral_centroid
from verbx.analysis.features_time import peak_dbfs, rms_dbfs, zero_crossing_rate

AudioArray = npt.NDArray[np.float32]


def framewise_metrics(
    audio: AudioArray,
    sr: int,
    frame_size: int = 2048,
    hop_size: int = 1024,
) -> list[dict[str, float]]:
    """Compute lightweight framewise metrics for CSV reporting."""
    frames: list[dict[str, float]] = []
    n = audio.shape[0]
    if n == 0:
        return frames

    for start in range(0, max(1, n - frame_size + 1), hop_size):
        end = min(n, start + frame_size)
        frame = audio[start:end, :]
        if frame.shape[0] < 8:
            continue

        frames.append(
            {
                "start_s": float(start / sr),
                "end_s": float(end / sr),
                "rms_dbfs": rms_dbfs(frame),
                "peak_dbfs": peak_dbfs(frame),
                "spectral_centroid": spectral_centroid(frame, sr),
                "zero_crossing_rate": zero_crossing_rate(frame),
            }
        )

    return frames


def write_framewise_csv(
    path: Path,
    audio: AudioArray,
    sr: int,
    frame_size: int = 2048,
    hop_size: int = 1024,
) -> None:
    """Write framewise analysis CSV."""
    rows = framewise_metrics(audio=audio, sr=sr, frame_size=frame_size, hop_size=hop_size)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "start_s",
                "end_s",
                "rms_dbfs",
                "peak_dbfs",
                "spectral_centroid",
                "zero_crossing_rate",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
