"""Measure a full one-hour neutral FDN impulse response (about 3 GB RAM).

Run: python scripts/validate_long_rt60.py
No audio is played or saved. This validates the stated neutral configuration,
not arbitrary damped, modulated, nonlinear, or cascaded settings.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from verbx.core.algo_reverb import AlgoReverbConfig, AlgoReverbEngine


def main() -> None:
    sr = 48_000
    duration = 3600
    config = AlgoReverbConfig(
        rt60=3600.0,
        fdn_lines=16,
        fdn_matrix="hadamard",
        allpass_stages=0,
        pre_delay_ms=0.0,
        damping=0.0,
        mod_depth_ms=0.0,
        mod_rate_hz=0.1,
        wet=1.0,
        dry=0.0,
        block_size=4096,
    )
    engine = AlgoReverbEngine(config)
    impulse = np.zeros(sr * duration, dtype=np.float64)
    impulse[0] = 0.8
    output = engine._process_channel(impulse, sr)
    del impulse
    blocks = output.reshape(duration, sr)
    powers = np.einsum("ij,ij->i", blocks, blocks) / sr
    levels = 10.0 * np.log10(np.maximum(powers, np.finfo(np.float64).tiny))
    times = np.arange(duration) + 0.5
    fits = []
    for start, end in ((10, 120), (10, 1200), (1200, 2400), (2400, 3600), (10, 3600)):
        mask = (times >= start) & (times < end)
        slope, intercept = np.polyfit(times[mask], levels[mask], 1)
        residual = levels[mask] - (intercept + slope * times[mask])
        r_squared = 1.0 - np.sum(residual**2) / np.sum((levels[mask] - np.mean(levels[mask])) ** 2)
        fits.append(
            {
                "start_s": start,
                "end_s": end,
                "rt60_s": float(-60.0 / slope),
                "r_squared": float(r_squared),
            }
        )
    source = Path(__file__).resolve().parents[1] / "src/verbx/core/algo_reverb.py"
    result = {
        "config": asdict(config),
        "sample_rate": sr,
        "duration_s": duration,
        "backend": engine.backend_name(),
        "finite": bool(np.isfinite(output).all()),
        "peak": float(max(abs(output.min()), abs(output.max()))),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "fits": fits,
    }
    print(json.dumps(result, indent=2))
    assert result["finite"]
    assert all(abs(row["rt60_s"] / config.rt60 - 1.0) < 0.1 for row in fits)


if __name__ == "__main__":
    main()
