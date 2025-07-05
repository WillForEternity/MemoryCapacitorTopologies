"""Small-world (Watts–Strogatz) topology generator.

Usage
-----
>>> from networks.topologies import make_topology
>>> adj = make_topology(
...     "small_world", n_nodes=500, k=6, beta=0.2, weight_scale=1.0, seed=42
... )

The module registers itself with the package-level registry when imported, so
`make_topology` can discover it dynamically.
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from . import register  # decorator


@register("small_world")
def small_world(
    n_nodes: int,
    k: int = 4,
    beta: float = 0.2,
    weight_scale: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Create a weighted adjacency matrix for a Watts–Strogatz small-world graph.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the reservoir.
    k : int, default 4
        Each node is connected to *k* nearest neighbours (must be even).
    beta : float, default 0.2
        Rewiring probability; 0 yields a regular ring lattice, 1 yields a random graph.
    weight_scale : float, default 1.0
        Standard deviation of the normal distribution used for signed edge weights.
    seed : int | None, optional
        RNG seed for reproducible graph structure and weights.

    Returns
    -------
    np.ndarray
        (N, N) float32 adjacency matrix with zero diagonal.
    """
    if k % 2 != 0:
        raise ValueError("k must be even for Watts–Strogatz model")

    rng = np.random.default_rng(seed)
    g = nx.watts_strogatz_graph(n=n_nodes, k=k, p=beta, seed=seed)
    a = nx.to_numpy_array(g, dtype=np.float32)
    # Assign random signed weights to existing edges
    # Generate symmetric random signed weights
    r = rng.normal(scale=weight_scale, size=(n_nodes, n_nodes)).astype(np.float32)
    weights = np.triu(r, 1)
    weights = weights + weights.T  # mirror to lower triangle
    return a * weights
