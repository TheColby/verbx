from __future__ import annotations

from verbx.core.fdn_capabilities import (
    FDN_GRAPH_TOPOLOGY_CHOICES,
    FDN_LINK_FILTER_CHOICES,
    FDN_MATRIX_CHOICES,
    normalize_fdn_graph_topology_name,
    normalize_fdn_link_filter_name,
    normalize_fdn_matrix_name,
)


def test_fdn_capabilities_include_expected_defaults() -> None:
    assert "hadamard" in FDN_MATRIX_CHOICES
    assert "graph" in FDN_MATRIX_CHOICES
    assert "ring" in FDN_GRAPH_TOPOLOGY_CHOICES
    assert "none" in FDN_LINK_FILTER_CHOICES


def test_fdn_capability_normalization_helpers() -> None:
    assert normalize_fdn_matrix_name("TV-Unitary") == "tv_unitary"
    assert normalize_fdn_link_filter_name("Low-Pass") == "lowpass"
    assert normalize_fdn_graph_topology_name("line") == "path"
    assert normalize_fdn_graph_topology_name("STAR") == "star"

