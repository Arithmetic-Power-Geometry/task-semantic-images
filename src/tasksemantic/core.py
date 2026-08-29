from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Callable, Iterable, Sequence

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class SemanticImageResult:
    states: tuple
    description: str
    witness_count: int | None = None


@dataclass(frozen=True)
class SemanticProfile:
    codomain_states: int
    reachable_states: int
    description_bits: int
    realizability_checks: int
    composition_states: int

    @property
    def reachability_fraction(self) -> float:
        return self.reachable_states / self.codomain_states if self.codomain_states else 0.0


def is_vertex_cover(G: nx.Graph, cover: Iterable[int]) -> bool:
    C = set(cover)
    return all(u in C or v in C for u, v in G.edges())


def brute_vertex_covers(G: nx.Graph):
    nodes = list(G.nodes())
    for bits in product((0, 1), repeat=len(nodes)):
        cover = {nodes[i] for i, bit in enumerate(bits) if bit}
        if is_vertex_cover(G, cover):
            yield bits, cover


def exact_cardinality_image(G: nx.Graph) -> SemanticImageResult:
    sizes = sorted({len(C) for _, C in brute_vertex_covers(G)})
    if not sizes:
        return SemanticImageResult(tuple(), "empty", None)
    return SemanticImageResult(tuple(sizes), str(sizes), min(sizes))


def minimum_vertex_cover_bipartite(
    G: nx.Graph, top_nodes: Sequence[int] | set[int] | None = None
) -> set[int]:
    if not nx.is_bipartite(G):
        raise ValueError("Graph must be bipartite")
    if top_nodes is None:
        color = nx.algorithms.bipartite.color(G)
        top_nodes = {v for v, c in color.items() if c == 0}
    top_nodes = set(top_nodes)
    matching = nx.algorithms.bipartite.maximum_matching(G, top_nodes=top_nodes)
    cover = nx.algorithms.bipartite.to_vertex_cover(G, matching, top_nodes=top_nodes)
    return set(cover)


def cardinality_image_bipartite(G: nx.Graph) -> SemanticImageResult:
    cover = minimum_vertex_cover_bipartite(G)
    tau = len(cover)
    n = G.number_of_nodes()
    return SemanticImageResult(tuple(range(tau, n + 1)), f"{{{tau},...,{n}}}", tau)


def minimum_cover_partition_image_bruteforce(
    G: nx.Graph, left: set[int]
) -> SemanticImageResult:
    minimum = None
    states: set[tuple[int, int]] = set()
    for _, cover in brute_vertex_covers(G):
        size = len(cover)
        if minimum is None or size < minimum:
            minimum = size
            states = {(len(cover & left), len(cover - left))}
        elif size == minimum:
            states.add((len(cover & left), len(cover - left)))
    ordered = tuple(sorted(states))
    return SemanticImageResult(ordered, str(ordered), minimum)


def constrained_min_cover_state_bruteforce(
    G: nx.Graph, left: set[int], k_left: int, k_right: int
) -> bool:
    image = minimum_cover_partition_image_bruteforce(G, left)
    return any(a <= k_left and b <= k_right for a, b in image.states)


def proper_affine_observer(G: nx.Graph, k: int):
    """Construct the proper-affine observer used in the finite theorem checks.

    Returns node order, independent basis vertices, A, c.  k is rounded to an even
    value because the all-ones column then belongs to the even-parity subspace.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    if k % 2:
        k += 1
    basis_vertices: list[int] = []
    for v in G.nodes():
        if all(not G.has_edge(v, u) for u in basis_vertices):
            basis_vertices.append(v)
            if len(basis_vertices) == k - 1:
                break
    if len(basis_vertices) < k - 1:
        raise ValueError("No independent set large enough for requested observer")

    nodes = list(G.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    A = np.ones((k, len(nodes)), dtype=np.uint8)
    for i, v in enumerate(basis_vertices):
        A[:, pos[v]] = 0
        A[i, pos[v]] = 1
        A[k - 1, pos[v]] = 1
    c = np.zeros(k, dtype=np.uint8)
    c[0] = 1
    return nodes, basis_vertices, A, c


def affine_image_bruteforce(G: nx.Graph, A: np.ndarray, c: np.ndarray) -> SemanticImageResult:
    outputs: set[tuple[int, ...]] = set()
    for bits, _ in brute_vertex_covers(G):
        x = np.asarray(bits, dtype=np.uint8)
        y = (A @ x + c) % 2
        outputs.add(tuple(int(v) for v in y))
    states = tuple(sorted(outputs))
    return SemanticImageResult(states, f"{len(states)} affine outputs", len(states))


def affine_rank_mod2(M: np.ndarray) -> int:
    A = np.array(M, dtype=np.uint8, copy=True) % 2
    row = 0
    for col in range(A.shape[1]):
        pivot = next((r for r in range(row, A.shape[0]) if A[r, col]), None)
        if pivot is None:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        for r in range(A.shape[0]):
            if r != row and A[r, col]:
                A[r] ^= A[row]
        row += 1
        if row == A.shape[0]:
            break
    return row


def affine_span(vectors: np.ndarray) -> tuple[tuple[int, ...], ...]:
    vectors = np.asarray(vectors, dtype=np.uint8) % 2
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D matrix with basis vectors as columns")
    k, r = vectors.shape
    out = set()
    for bits in product((0, 1), repeat=r):
        coeff = np.asarray(bits, dtype=np.uint8)
        y = (vectors @ coeff) % 2
        out.add(tuple(int(v) for v in y))
    return tuple(sorted(out))


def affine_image_from_coset(offset: Sequence[int], basis_columns: np.ndarray):
    offset = np.asarray(offset, dtype=np.uint8) % 2
    span = affine_span(basis_columns)
    return tuple(sorted(tuple(int(a ^ b) for a, b in zip(offset, s)) for s in span))


def compose_affine_coset(
    offset: np.ndarray, basis_columns: np.ndarray, B: np.ndarray, d: np.ndarray
):
    offset = np.asarray(offset, dtype=np.uint8) % 2
    basis_columns = np.asarray(basis_columns, dtype=np.uint8) % 2
    B = np.asarray(B, dtype=np.uint8) % 2
    d = np.asarray(d, dtype=np.uint8) % 2
    new_offset = (B @ offset + d) % 2
    new_basis = (B @ basis_columns) % 2
    return new_offset, new_basis


def verify_proper_affine_hyperplane(G: nx.Graph, k: int) -> dict:
    nodes, basis_vertices, A, c = proper_affine_observer(G, k)
    observed = affine_image_bruteforce(G, A, c)
    k_eff = A.shape[0]
    expected = {bits for bits in product((0, 1), repeat=k_eff) if sum(bits) % 2 == 1}
    return {
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "k": k_eff,
        "basis_size": len(basis_vertices),
        "observed_states": len(observed.states),
        "expected_states": len(expected),
        "verified": set(observed.states) == expected,
        "rank": affine_rank_mod2(A),
        "dense_support_min": int((A.sum(axis=1)).min()),
    }


def semantic_profile(
    codomain_states: int,
    reachable_states: int,
    realizability_checks: int,
    composition_states: int,
) -> SemanticProfile:
    description_bits = max(1, (max(reachable_states, 1) - 1).bit_length())
    return SemanticProfile(
        codomain_states=codomain_states,
        reachable_states=reachable_states,
        description_bits=description_bits,
        realizability_checks=realizability_checks,
        composition_states=composition_states,
    )


def exact_ordered_bdd_size_graph_cnf(G: nx.Graph, order: Sequence[int] | None = None) -> int:
    """Exact reduced OBDD node count for graph CNF under one fixed variable order.

    This is a finite diagnostic only; the paper's DNNF lower bound is theorem-based.
    Terminal nodes 0 and 1 are included in the returned count.
    """
    nodes = list(order if order is not None else G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    edges = tuple(sorted((min(index[u], index[v]), max(index[u], index[v])) for u, v in G.edges()))

    unique: dict[tuple[int, int, int], int] = {}
    next_id = 2

    @lru_cache(maxsize=None)
    def rec(i: int, forced_mask: int, zero_mask: int) -> int:
        nonlocal next_id
        if i == len(nodes):
            return 1
        bit = 1 << i
        forced = bool(forced_mask & bit)

        def branch(value: int) -> int:
            new_forced = forced_mask
            new_zero = zero_mask
            if value == 0:
                new_zero |= bit
                for a, b in edges:
                    if a == i:
                        if new_zero & (1 << b):
                            return 0
                        new_forced |= 1 << b
                    elif b == i:
                        if new_zero & (1 << a):
                            return 0
                        new_forced |= 1 << a
            return rec(i + 1, new_forced, new_zero)

        if forced:
            lo = 0
            hi = rec(i + 1, forced_mask & ~bit, zero_mask)
        else:
            lo = branch(0)
            hi = branch(1)
        if lo == hi:
            return lo
        key = (i, lo, hi)
        if key not in unique:
            unique[key] = next_id
            next_id += 1
        return unique[key]

    rec(0, 0, 0)
    return next_id


def task_replacement_decision(
    tsi_states: Iterable,
    continuation: Callable[[object], bool],
) -> bool:
    return any(continuation(y) for y in tsi_states)
