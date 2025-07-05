"""Verbose, self-explaining test for the Watts–Strogatz small-world topology generator.

The purpose is to sanity-check structural properties of the generated graph and
provide human-readable PASS/FAIL commentary rather than strict `assert`s.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Ensure project root is on path when executed directly
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from networks.topologies import make_topology

TOL = 1e-6


def banner(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def describe(label: str, predicate: bool, detail: str = ""):
    print(f"{label}: {'PASS' if predicate else 'FAIL'} {detail}")


def main():
    banner("Small-World Topology Generator – Verbose Behavioural Test")

    n_nodes, k, beta, seed = 200, 4, 0.2, 42
    print(
        f"Generating adjacency via make_topology('small_world', n_nodes={n_nodes}, k={k}, beta={beta}, seed={seed})\n"
    )
    adj = make_topology("small_world", n_nodes=n_nodes, k=k, beta=beta, seed=seed)

    # Basic shape checks
    describe("Adjacency shape", adj.shape == (n_nodes, n_nodes), f"-> {adj.shape}")

    # Zero diagonal
    describe("Zero diagonal", np.allclose(np.diag(adj), 0, atol=TOL))

    # Symmetry (undirected)
    describe("Symmetric matrix", np.allclose(adj, adj.T, atol=TOL))

    # Degree statistics
    deg = (adj != 0).sum(axis=0)
    avg_deg = deg.mean()
    describe(
        "Average degree ≈ k",
        math.isclose(avg_deg, k, rel_tol=0.2),
        f"-> {avg_deg:.2f} (expected ≈ {k})",
    )

    # Small-world metrics using NetworkX
    g = nx.from_numpy_array(adj)
    clustering = nx.average_clustering(g)
    path_len = nx.average_shortest_path_length(g)

    describe("Clustering coefficient > random graph (~k/n)", clustering > k / n_nodes, f"-> {clustering:.3f}")
    describe("Average path length << lattice (n/2k)", path_len < n_nodes / (2 * k), f"-> {path_len:.2f}")

    # Visual: adjacency heatmap
    print("\nGenerating adjacency heatmap…")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(adj, cmap="bwr", interpolation="nearest", vmin=-np.abs(adj).max(), vmax=np.abs(adj).max())
    ax.set_title("Small-World adjacency (red=positive, blue=negative)")
    ax.set_xlabel("Node index")
    ax.set_ylabel("Node index")
    fig.tight_layout()

    out_dir = Path(__file__).with_suffix("").with_name("outputs")
    out_dir.mkdir(exist_ok=True)
    img_path = out_dir / "small_world_adj.png"
    fig.savefig(img_path)
    print(f"Heatmap saved to {img_path.relative_to(Path.cwd())}\n")

    print("Test complete – review PASS/FAIL messages and visual to assess correctness.")


if __name__ == "__main__":
    main()
