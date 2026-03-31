"""FDN mix-matrix construction and graph topology utilities.

Extracted from ``algo_reverb.py`` so that matrix operations can be
imported, tested, and extended independently of the full engine.

All functions are pure (no state, no side effects) and operate only on
``numpy`` arrays.  They are intentionally kept as module-level functions
rather than static class methods so that they can be used by other
modules (e.g., future convolution engine presets, IR-shaping tools) without
importing the full ``AlgoReverbEngine``.

Public API
----------
Matrix builders
    ``build_fdn_matrix(size, matrix_type)`` — main dispatch entry point
    ``build_hadamard(size)``
    ``build_random_orthogonal(size, seed)``
    ``build_shift_permutation(size, shift)``
    ``build_circulant(size)``
    ``build_elliptic(size)``
    ``build_sdn_hybrid(size)``
    ``orthonormalize(matrix)``

Sparse / graph topology
    ``build_sparse_pairings(size, stages, seed)``
    ``build_graph_pairings(size, topology, degree, seed)``
    ``build_graph_edges(*, size, topology, degree, seed)``
    ``build_sparse_mix_matrix(size, pairings)``
    ``apply_sparse_pair_mix(input_vec, pairings, out_vec, scratch)``
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from verbx.core.fdn_capabilities import (
    normalize_fdn_graph_topology_name,
    normalize_fdn_matrix_name,
)

MatrixArray = npt.NDArray[np.float64]
PairingArray = npt.NDArray[np.int32]


# ---------------------------------------------------------------------------
# Orthogonalisation helper
# ---------------------------------------------------------------------------


def orthonormalize(matrix: MatrixArray) -> MatrixArray:
    """Project *matrix* to the nearest orthonormal basis via deterministic QR."""
    q, _ = np.linalg.qr(np.asarray(matrix, dtype=np.float64))
    return np.asarray(q, dtype=np.float64)


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------


def build_hadamard(size: int) -> MatrixArray:
    """Build a deterministic Hadamard-derived orthonormal matrix.

    Constructs the full power-of-two Hadamard matrix then truncates to
    ``size × size`` and re-orthogonalises via QR to restore strict
    orthonormality after truncation.
    """
    matrix = np.array([[1.0]], dtype=np.float64)
    while matrix.shape[0] < size:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    matrix = matrix[:size, :size]
    return orthonormalize(matrix)


def build_random_orthogonal(size: int, seed: int = 2026) -> MatrixArray:
    """Build a deterministic random orthonormal matrix via QR decomposition."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((size, size)).astype(np.float64)
    q, _ = np.linalg.qr(base)
    return np.asarray(q, dtype=np.float64)


def build_shift_permutation(size: int, shift: int) -> MatrixArray:
    """Build a cyclic permutation matrix for circulant-style constructions."""
    matrix = np.zeros((size, size), dtype=np.float64)
    for row in range(size):
        col = (row - shift) % size
        matrix[row, col] = 1.0
    return matrix


def build_circulant(size: int) -> MatrixArray:
    """Build a real orthogonal circulant matrix via unit-modulus spectrum."""
    spectrum = np.ones(size, dtype=np.complex128)
    for k in range(1, (size + 1) // 2):
        angle = (2.0 * np.pi * float(k * k)) / float(max(1, size))
        value = np.exp(1j * angle)
        spectrum[k] = value
        spectrum[-k] = np.conjugate(value)
    if size % 2 == 0:
        spectrum[size // 2] = -1.0 + 0.0j

    first_column = np.fft.ifft(spectrum).real.astype(np.float64)
    matrix = np.zeros((size, size), dtype=np.float64)
    for col in range(size):
        matrix[:, col] = np.roll(first_column, col)

    gram = matrix.T @ matrix
    if np.allclose(gram, np.eye(size, dtype=np.float64), atol=1e-4):
        return np.asarray(matrix, dtype=np.float64)
    return orthonormalize(matrix)


def build_elliptic(size: int) -> MatrixArray:
    """Build a deterministic elliptic-inspired prototype and orthonormalize it."""
    eye = np.eye(size, dtype=np.float64)
    shift_1 = build_shift_permutation(size, shift=1)
    shift_2 = build_shift_permutation(size, shift=2)
    proto = (
        (0.62 * eye)
        + (0.19 * (shift_1 + shift_1.T))
        + (0.05 * (shift_2 + shift_2.T))
    )
    return orthonormalize(proto)


def build_sdn_hybrid(size: int) -> MatrixArray:
    """Build an SDN-inspired scattering matrix from pseudo-geometry.

    Distributes delay-line nodes over a Fibonacci-lattice sphere, then
    weights coupling by inverse node distance.
    """
    if size <= 1:
        return np.eye(size, dtype=np.float64)

    idx = np.arange(size, dtype=np.float64)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - (2.0 * (idx + 0.5) / float(size))
    radius = np.sqrt(np.maximum(0.0, 1.0 - (z * z)))
    theta = golden * idx
    coords = np.stack(
        (radius * np.cos(theta), radius * np.sin(theta), z),
        axis=1,
    ).astype(np.float64)

    matrix = np.eye(size, dtype=np.float64) * 0.58
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            weight = 0.42 / (1.0 + (4.0 * dist))
            sign = 1.0 if ((i + j) % 2 == 0) else -1.0
            matrix[i, j] = np.float64(sign * weight)
    return orthonormalize(matrix)


def build_fdn_matrix(size: int, matrix_type: str) -> MatrixArray:
    """Dispatch to the correct matrix builder for *matrix_type*.

    Parameters
    ----------
    size:
        Number of FDN delay lines (matrix will be ``size × size``).
    matrix_type:
        One of ``"hadamard"``, ``"householder"``, ``"random_orthogonal"``,
        ``"circulant"``, ``"elliptic"``, ``"sdn_hybrid"``, ``"graph"``,
        ``"tv_unitary"`` (falls back to Hadamard).
    """
    if size <= 0:
        return np.zeros((0, 0), dtype=np.float64)

    kind = normalize_fdn_matrix_name(matrix_type)

    if kind == "householder":
        v = np.ones((size, 1), dtype=np.float64)
        matrix = np.eye(size, dtype=np.float64) - ((2.0 / size) * (v @ v.T))
        return np.asarray(matrix, dtype=np.float64)

    if kind == "random_orthogonal":
        return build_random_orthogonal(size=size, seed=2026)

    if kind == "circulant":
        return build_circulant(size=size)

    if kind == "elliptic":
        return build_elliptic(size=size)

    if kind == "sdn_hybrid":
        return build_sdn_hybrid(size=size)

    if kind == "graph":
        pairings = build_graph_pairings(size=size, topology="ring", degree=2, seed=2026)
        return build_sparse_mix_matrix(size=size, pairings=pairings)

    # "hadamard", "tv_unitary", and unrecognised values
    return build_hadamard(size=size)


# ---------------------------------------------------------------------------
# Sparse / graph topology
# ---------------------------------------------------------------------------


def build_sparse_pairings(size: int, stages: int, seed: int) -> PairingArray:
    """Build deterministic pairwise mixing schedules for sparse FDN mode.

    Returns an ``(stages, size)`` int32 array where each row is a
    permutation of ``[0, size)``.
    """
    if size <= 1:
        return np.zeros((0, 0), dtype=np.int32)

    stage_count = max(1, int(stages))
    rng = np.random.default_rng(seed)
    pairings = np.zeros((stage_count, size), dtype=np.int32)
    for stage_idx in range(stage_count):
        pairings[stage_idx, :] = rng.permutation(size).astype(np.int32)
    return pairings


def build_graph_edges(
    *,
    size: int,
    topology: str,
    degree: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Build unique undirected edges for graph-structured FDN mode.

    Supported topologies: ``"ring"`` (default), ``"star"``, ``"path"``,
    ``"random"``.
    """
    normalized = normalize_fdn_graph_topology_name(topology)
    max_degree = max(1, min(int(degree), max(1, size - 1)))
    edges: set[tuple[int, int]] = set()

    if normalized == "star":
        center = 0
        for node in range(1, size):
            edges.add((center, node))
        return sorted(edges)

    if normalized == "path":
        for step in range(1, max_degree + 1):
            for start in range(0, size - step):
                a, b = start, start + step
                edges.add((a, b) if a < b else (b, a))
        return sorted(edges)

    if normalized == "random":
        all_pairs = [(a, b) for a in range(size) for b in range(a + 1, size)]
        if not all_pairs:
            return []
        rng = np.random.default_rng(seed)
        rng.shuffle(all_pairs)  # type: ignore[arg-type]
        target = min(len(all_pairs), max(size - 1, (size * max_degree) // 2))
        return sorted(all_pairs[:target])

    # Default "ring" behavior
    for step in range(1, max_degree + 1):
        for node in range(size):
            a, b = node, (node + step) % size
            edge = (a, b) if a < b else (b, a)
            if edge[0] != edge[1]:
                edges.add(edge)
    return sorted(edges)


def build_graph_pairings(
    size: int,
    topology: str,
    degree: int,
    seed: int,
) -> PairingArray:
    """Build deterministic graph-constrained pairwise mixing schedules."""
    if size <= 1:
        return np.zeros((0, 0), dtype=np.int32)

    edges = build_graph_edges(size=size, topology=topology, degree=degree, seed=seed)
    if not edges:
        return np.zeros((0, 0), dtype=np.int32)

    stage_count = max(1, int(degree))
    rng = np.random.default_rng(seed)
    pairings = np.zeros((stage_count, size), dtype=np.int32)
    edge_array = np.asarray(edges, dtype=np.int32)

    for stage_idx in range(stage_count):
        order = np.arange(edge_array.shape[0], dtype=np.int32)
        rng.shuffle(order)
        used = np.zeros((size,), dtype=bool)
        paired: list[int] = []
        for edge_idx in order:
            a = int(edge_array[edge_idx, 0])
            b = int(edge_array[edge_idx, 1])
            if used[a] or used[b]:
                continue
            if rng.random() < 0.5:
                paired.extend([a, b])
            else:
                paired.extend([b, a])
            used[a] = True
            used[b] = True

        leftovers = [idx for idx in range(size) if not used[idx]]
        rng.shuffle(leftovers)  # type: ignore[arg-type]
        paired.extend(leftovers)
        pairings[stage_idx, :] = np.asarray(paired[:size], dtype=np.int32)
    return pairings


def apply_sparse_pair_mix(
    input_vec: npt.NDArray[np.float64],
    pairings: PairingArray,
    out_vec: npt.NDArray[np.float64],
    scratch: npt.NDArray[np.float64],
) -> None:
    """Apply sparse orthogonal pair-mixing stages to a feedback vector in-place.

    Modifies *out_vec* in-place; *scratch* is a same-size temporary buffer.
    """
    if pairings.size == 0:
        out_vec[:] = input_vec
        return

    out_vec[:] = input_vec
    size = int(out_vec.shape[0])
    inv_sqrt2 = np.float64(1.0 / np.sqrt(2.0))

    for stage_idx in range(pairings.shape[0]):
        scratch[:] = out_vec
        perm = pairings[stage_idx]
        for idx in range(0, size - 1, 2):
            a = int(perm[idx])
            b = int(perm[idx + 1])
            va = scratch[a]
            vb = scratch[b]
            out_vec[a] = (va + vb) * inv_sqrt2
            out_vec[b] = (va - vb) * inv_sqrt2
        if size % 2 == 1:
            last = int(perm[size - 1])
            out_vec[last] = scratch[last]


def build_sparse_mix_matrix(size: int, pairings: PairingArray) -> MatrixArray:
    """Build a dense matrix equivalent of sparse pair-mixing stages."""
    if size <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    if pairings.size == 0:
        return np.eye(size, dtype=np.float64)

    matrix = np.eye(size, dtype=np.float64)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for stage_idx in range(pairings.shape[0]):
        perm = pairings[stage_idx]
        stage_matrix = np.eye(size, dtype=np.float64)
        for idx in range(0, size - 1, 2):
            a = int(perm[idx])
            b = int(perm[idx + 1])
            stage_matrix[a, a] = inv_sqrt2
            stage_matrix[a, b] = inv_sqrt2
            stage_matrix[b, a] = inv_sqrt2
            stage_matrix[b, b] = -inv_sqrt2
        matrix = stage_matrix @ matrix
    return np.asarray(matrix, dtype=np.float64)
