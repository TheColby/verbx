"""Shared FDN option capabilities and normalization helpers."""

from __future__ import annotations

FDN_MATRIX_CHOICES = frozenset(
    {
        "hadamard",
        "householder",
        "random_orthogonal",
        "circulant",
        "elliptic",
        "tv_unitary",
        "graph",
        "sdn_hybrid",
    }
)

FDN_GRAPH_TOPOLOGY_CHOICES = frozenset(
    {
        "ring",
        "path",
        "star",
        "random",
    }
)

FDN_LINK_FILTER_CHOICES = frozenset(
    {
        "none",
        "lowpass",
        "highpass",
    }
)


def normalize_fdn_matrix_name(value: str) -> str:
    """Normalize FDN matrix identifier for CLI and engine use."""
    return str(value).strip().lower().replace("-", "_")


def normalize_fdn_link_filter_name(value: str) -> str:
    """Normalize FDN feedback-link filter identifier for CLI and engine use."""
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in {"low-pass"}:
        return "lowpass"
    if normalized in {"high-pass"}:
        return "highpass"
    return normalized


def normalize_fdn_graph_topology_name(value: str) -> str:
    """Normalize graph-topology identifier for CLI and engine use."""
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"line"}:
        return "path"
    return normalized
