from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from verbx.ir.scala import parse_scala_file, resolve_scala_frequencies
from verbx.ir.tuning import apply_frequency_band_emphasis

SCALE_12_EDO = """! Twelve-tone equal temperament
12-EDO test scale
12
100.0
200.0
300.0
400.0
500.0
600.0
700.0
800.0
900.0
1000.0
1100.0
2/1
"""

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bundled_scala_examples_parse_with_expected_periods() -> None:
    expected = {
        "19edo.scl": (19, 2.0),
        "5_limit_major.scl": (7, 2.0),
        "bohlen_pierce_13edo.scl": (13, 3.0),
    }

    for filename, (pitch_count, period_ratio) in expected.items():
        scale = parse_scala_file(REPO_ROOT / "examples" / "scales" / filename)
        assert scale.pitch_count == pitch_count
        assert scale.period_ratio == period_ratio


def test_parse_scala_file_supports_cents_ratios_comments_and_hash(tmp_path: Path) -> None:
    scale_path = tmp_path / "12edo.scl"
    scale_path.write_text(SCALE_12_EDO, encoding="utf-8")

    scale = parse_scala_file(scale_path)

    assert scale.description == "12-EDO test scale"
    assert scale.pitch_count == 12
    assert len(scale.ratios) == 12
    assert abs(scale.ratios[0] - (2.0 ** (100.0 / 1200.0))) < 1e-12
    assert scale.period_ratio == 2.0
    assert scale.sha256 == hashlib.sha256(scale_path.read_bytes()).hexdigest()


def test_resolve_scala_frequencies_maps_root_degree_and_bounds(tmp_path: Path) -> None:
    scale_path = tmp_path / "12edo.scl"
    scale_path.write_text(SCALE_12_EDO, encoding="utf-8")
    scale = parse_scala_file(scale_path)

    targets = resolve_scala_frequencies(
        scale,
        root_hz=440.0,
        root_degree=0,
        low_hz=400.0,
        high_hz=900.0,
        max_targets=128,
    )
    shifted = resolve_scala_frequencies(
        scale,
        root_hz=440.0,
        root_degree=1,
        low_hz=400.0,
        high_hz=900.0,
        max_targets=128,
    )

    assert any(abs(frequency - 440.0) < 1e-9 for frequency in targets)
    assert any(abs(frequency - 880.0) < 1e-9 for frequency in targets)
    assert all(400.0 <= frequency <= 900.0 for frequency in targets)
    assert any(abs(frequency - 440.0) < 1e-9 for frequency in shifted)
    assert shifted != targets


def test_resolve_scala_frequencies_respects_target_budget(tmp_path: Path) -> None:
    scale_path = tmp_path / "12edo.scl"
    scale_path.write_text(SCALE_12_EDO, encoding="utf-8")
    scale = parse_scala_file(scale_path)

    targets = resolve_scala_frequencies(
        scale,
        root_hz=440.0,
        root_degree=0,
        low_hz=20.0,
        high_hz=20_000.0,
        max_targets=16,
    )

    assert len(targets) == 16
    assert targets == tuple(sorted(targets))


def test_scala_parser_rejects_declared_pitch_count_mismatch(tmp_path: Path) -> None:
    scale_path = tmp_path / "broken.scl"
    scale_path.write_text("Broken\n3\n100.0\n2/1\n", encoding="utf-8")

    try:
        parse_scala_file(scale_path)
    except ValueError as exc:
        assert "declares 3 pitches but contains 2" in str(exc)
    else:
        raise AssertionError("Malformed Scala scale should fail")


def test_frequency_band_emphasis_raises_target_region() -> None:
    sr = 16_000
    rng = np.random.default_rng(2026)
    source = rng.standard_normal((sr * 2, 1)).astype(np.float64) * 0.05

    emphasized = apply_frequency_band_emphasis(
        source,
        sr,
        targets_hz=(440.0,),
        strength=1.0,
        bandwidth_cents=40.0,
        gain_db=12.0,
    )

    source_fft = np.abs(np.fft.rfft(source[:, 0]))
    emphasized_fft = np.abs(np.fft.rfft(emphasized[:, 0]))
    frequencies = np.fft.rfftfreq(source.shape[0], d=1.0 / sr)
    transfer = emphasized_fft / np.maximum(source_fft, 1e-12)
    target_gain = float(np.median(transfer[np.abs(frequencies - 440.0) <= 3.0]))
    control_gain = float(np.median(transfer[np.abs(frequencies - 900.0) <= 3.0]))

    assert np.all(np.isfinite(emphasized))
    assert target_gain > control_gain * 1.5
