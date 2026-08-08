from __future__ import annotations

import numpy as np
import pytest

from verbx.core.fdn_delays import (
    read_fractional_delay,
    resolve_comb_cloud_delays_ms,
    resolve_dfm_delays_ms,
    resolve_fdn_delays_ms,
)
from verbx.core.fdn_nonlinearity import (
    apply_feedback_nonlinearity,
    normalize_nonlinearity_mode,
)
from verbx.core.fdn_spatial import (
    apply_spatial_coupling,
    layout_channel_groups,
    normalize_spatial_coupling_mode,
)


def test_feedback_nonlinearity_is_bounded_and_none_is_identity() -> None:
    values = np.array([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=np.float64)
    assert apply_feedback_nonlinearity(values, mode="none", amount=1.0, drive=2.0) is values
    shaped = apply_feedback_nonlinearity(values, mode="tanh", amount=1.0, drive=2.0)
    assert np.all(np.isfinite(shaped))
    assert float(np.max(np.abs(shaped))) <= 0.5


def test_feedback_nonlinearity_normalizes_and_rejects_modes() -> None:
    assert normalize_nonlinearity_mode("softclip") == "softclip"
    with pytest.raises(ValueError, match="Unsupported FDN nonlinearity"):
        normalize_nonlinearity_mode("fold")


def test_spatial_coupling_all_to_all_excludes_source_channel() -> None:
    wet = np.zeros((4, 3), dtype=np.float64)
    wet[:, 0] = 1.0
    coupled = apply_spatial_coupling(
        wet,
        layout="lcr",
        mode="all-to-all",
        strength=1.0,
    )
    np.testing.assert_allclose(coupled[:, 0], 0.0)
    np.testing.assert_allclose(coupled[:, 1:], 0.5)


def test_spatial_layout_groups_and_mode_validation() -> None:
    assert layout_channel_groups("7.1.4", 12) == (
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    )
    assert normalize_spatial_coupling_mode("front-rear") == "front_rear"
    with pytest.raises(ValueError, match="Unsupported FDN spatial coupling"):
        normalize_spatial_coupling_mode("diagonal")


def test_delay_layout_helpers_are_deterministic_and_extend_defaults() -> None:
    defaults = np.array([31.0, 37.0], dtype=np.float64)
    resolved = resolve_fdn_delays_ms((), line_count=4, defaults_ms=defaults)
    assert resolved.shape == (4,)
    assert np.all(np.diff(resolved) > 0.0)

    first = resolve_comb_cloud_delays_ms((), enabled=True, count=6, seed=7)
    second = resolve_comb_cloud_delays_ms((), enabled=True, count=6, seed=7)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.diff(first) >= 0.35)


def test_dfm_delay_layout_and_fractional_read_contract() -> None:
    np.testing.assert_array_equal(
        resolve_dfm_delays_ms((2.5,), line_count=3),
        np.array([2.5, 2.5, 2.5], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="match FDN line count"):
        resolve_dfm_delays_ms((1.0, 2.0), line_count=3)

    buffer = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    assert read_fractional_delay(buffer, write_index=0, delay_samples=1.5) == pytest.approx(2.5)
