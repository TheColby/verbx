"""Long RT60 requests must not collapse at an arbitrary feedback ceiling."""

from dataclasses import replace

import numpy as np
import pytest

from verbx.core.algo_reverb import AlgoReverbConfig, AlgoReverbEngine


def _config(rt60: float = 3600.0) -> AlgoReverbConfig:
    return AlgoReverbConfig(
        rt60=rt60,
        fdn_lines=16,
        allpass_stages=0,
        pre_delay_ms=0.0,
        damping=0.0,
        mod_depth_ms=0.0,
        mod_rate_hz=0.0,
        wet=1.0,
        dry=0.0,
    )


def test_one_hour_scalar_gain_calibration_is_not_clipped() -> None:
    engine = AlgoReverbEngine(_config())
    sr = 48_000
    delays = np.maximum(2, np.rint(engine._base_delay_ms * sr / 1000.0)) / sr
    requested = np.power(10.0, -3.0 * delays / 3600.0)
    resolved = np.clip(requested, 0.0, engine._feedback_gain_ceiling)
    assert np.all(requested > 0.995)
    np.testing.assert_array_equal(resolved, requested)
    np.testing.assert_allclose(-3.0 * delays / np.log10(resolved), 3600.0, rtol=1e-10)
    assert engine._feedback_gain_ceiling == np.nextafter(1.0, 0.0)
    assert engine._feedback_gain_scale == 1.0
    assert engine._state_guard_peak == 64.0


@pytest.mark.parametrize(
    "mode",
    ["default", "python", "automation", "cascade", "multiband", "cascade_auto", "multiband_auto"],
)
def test_long_requests_produce_distinct_finite_responses(mode: str) -> None:
    sr = 16_000
    signal = np.zeros(sr // 2, dtype=np.float64)
    signal[0] = 0.1
    rendered = []
    for rt60 in (240.0, 3600.0):
        config = _config(rt60)
        if mode in {"cascade", "cascade_auto"}:
            config = replace(config, fdn_cascade=True, fdn_cascade_mix=0.5)
        elif mode in {"multiband", "multiband_auto"}:
            config = replace(config, fdn_rt60_low=rt60, fdn_rt60_mid=rt60, fdn_rt60_high=rt60)
        engine = AlgoReverbEngine(config)
        if mode == "python":
            engine._use_numba = False
        automation = None
        if mode in {"automation", "cascade_auto", "multiband_auto"}:
            # Exercise a runtime gain update, not just the initial calibration.
            curve = np.full(signal.size, 120.0)
            curve[signal.size // 4 :] = rt60
            automation = {"rt60": curve}
            if mode == "multiband_auto":
                automation.update({f"fdn-rt60-{band}": curve for band in ("low", "mid", "high")})
        output = engine._process_channel(signal, sr, parameter_automation=automation)
        assert output.dtype == np.float64
        assert np.all(np.isfinite(output))
        rendered.append(output)
    assert np.max(np.abs(rendered[1] - rendered[0])) > 1e-8


def test_long_rt60_python_and_accelerated_paths_agree() -> None:
    sr = 16_000
    signal = np.zeros(sr // 2, dtype=np.float64)
    signal[0] = 0.1
    accelerated = AlgoReverbEngine(_config())
    python = AlgoReverbEngine(_config())
    python._use_numba = False
    np.testing.assert_allclose(
        accelerated._process_channel(signal, sr),
        python._process_channel(signal, sr),
        rtol=1e-9,
        atol=1e-12,
    )


def test_unsafe_mode_keeps_its_explicit_gain_policy() -> None:
    engine = AlgoReverbEngine(replace(_config(), unsafe_self_oscillate=True))
    assert engine._feedback_gain_ceiling == 1.25
    assert engine._feedback_gain_scale > 1.0


@pytest.mark.parametrize("python_path", [False, True])
def test_neutral_single_loop_has_requested_decay_and_no_interpolation_loss(
    python_path: bool,
) -> None:
    sr = 8000
    config = replace(_config(), fdn_lines=1, comb_delays_ms=(50.0,), mod_rate_hz=0.1)
    engine = AlgoReverbEngine(config)
    if python_path:
        engine._use_numba = False
    impulse = np.zeros(sr // 2, dtype=np.float64)
    impulse[0] = 0.1
    output = engine._process_channel(impulse, sr)
    indices = np.flatnonzero(output)
    assert indices.size >= 8
    np.testing.assert_array_equal(np.diff(indices), 400)
    gain = 10.0 ** (-3.0 * 0.05 / 3600.0)
    np.testing.assert_allclose(output[indices[1:]] / output[indices[:-1]], gain, rtol=1e-13)


def test_neutral_damping_automation_matches_static_bypass() -> None:
    sr = 8000
    impulse = np.zeros(sr // 2, dtype=np.float64)
    impulse[0] = 0.1
    engine = AlgoReverbEngine(_config())
    engine._use_numba = False
    static = engine._process_channel(impulse, sr)
    automated = engine._process_channel(
        impulse, sr, parameter_automation={"damping": np.zeros(impulse.size)}
    )
    np.testing.assert_array_equal(static, automated)
