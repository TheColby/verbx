from __future__ import annotations

import numpy as np
import pytest

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
