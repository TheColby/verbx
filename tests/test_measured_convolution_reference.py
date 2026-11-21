"""Measured IR parity against independent direct convolution, without network access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from verbx.core.convolution_reverb import ConvolutionReverbConfig, ConvolutionReverbEngine

FIXTURES = Path(__file__).parent / "fixtures" / "measured_air"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


@pytest.mark.parametrize("reference", MANIFEST["references"], ids=lambda r: r["file"])
@pytest.mark.parametrize("partition_size", [256, 1024])
def test_measured_air_matches_independent_direct_convolution(
    reference: dict[str, str], partition_size: int, tmp_path: Path
) -> None:
    path = FIXTURES / reference["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
    ir, sr = sf.read(path, always_2d=True, dtype="float64")
    assert ir.shape[1] == 1
    assert np.isfinite(ir).all() and np.max(np.abs(ir)) > 0
    # Deterministic synthetic dry probe; no claim of measured anechoic speech/music.
    dry = np.random.default_rng(647).uniform(-0.2, 0.2, (641, 1))
    dry[0, 0] = 0.5
    expected = np.convolve(dry[:, 0], ir[:, 0], mode="full")[:, None]
    engine = ConvolutionReverbEngine(
        ConvolutionReverbConfig(
            wet=1.0,
            dry=0.0,
            ir_path=str(path),
            ir_normalize="none",
            partition_size=partition_size,
            device="cpu",
        )
    )
    actual = engine.process(dry, int(sr))
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-9)
    # Also check file streaming and its final flushed partition with original IR tail.
    infile, outfile = tmp_path / "dry.wav", tmp_path / "wet.wav"
    sf.write(infile, dry, sr, subtype="DOUBLE")
    engine.process_streaming_file(str(infile), str(outfile), output_subtype="DOUBLE")
    streamed, actual_sr = sf.read(outfile, always_2d=True, dtype="float64")
    assert actual_sr == sr
    assert streamed.shape == expected.shape
    np.testing.assert_allclose(streamed, expected, atol=1e-10, rtol=1e-9)
