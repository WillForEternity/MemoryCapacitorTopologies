"""Verbose behavioural test for `MemcapacitiveReservoir`.

This script validates that a reservoir built from:
    • a small-world topology (n=50)
    • default `Memcapacitor` neurons
accepts a simple sinusoidal input and produces a state tensor of the expected
shape with non-trivial dynamics.  All results are printed with PASS/FAIL tags so
humans or AI agents can judge correctness from stdout alone.
"""
from __future__ import annotations

from pathlib import Path
import math
import sys, os

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root on path when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from networks.topologies import make_topology
from networks.reservoir import MemcapacitiveReservoir

TOL = 1e-6


def banner(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def describe(label: str, predicate: bool, detail: str = ""):
    print(f"{label}: {'PASS' if predicate else 'FAIL'} {detail}")


def main():
    banner("MemcapacitiveReservoir Forward Pass – Verbose Test")

    # Build reservoir
    n_nodes = 50
    adj = make_topology("small_world", n_nodes=n_nodes, k=4, beta=0.2, seed=123)
    res = MemcapacitiveReservoir(adj, input_dim=1)

    # Create sinusoidal input sequence (batch=1, time=100, dim=1)
    timesteps = 100
    t = torch.linspace(0, 2 * math.pi, timesteps).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    x = torch.sin(t)

    # Forward pass
    states = res(x)  # (1, T, N)

    # --- Checks ---
    describe("State tensor shape", states.shape == (1, timesteps, n_nodes), f"-> {states.shape}")

    # Non-zero dynamics (some variance)
    var = states.var().item()
    describe("Non-trivial dynamics (variance > 0)", var > TOL, f"-> var={var:.3e}")

    # Determinism: rerun and compare
    res.reset_state()
    states2 = res(x)
    deterministic = torch.allclose(states, states2)
    describe("Deterministic given same input after reset_state", deterministic)

    # --- Visual ---
    print("\nGenerating state trajectory plot (first 5 nodes)…")
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(5):
        ax.plot(states[0, :, i].detach().numpy(), label=f"node {i}")
    ax.set_xlabel("t")
    ax.set_ylabel("charge q")
    ax.set_title("Reservoir node charges over time (subset)")
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_dir = Path(__file__).with_suffix("").with_name("outputs")
    out_dir.mkdir(exist_ok=True)
    img_path = out_dir / "reservoir_forward_summary.png"
    fig.savefig(img_path)
    print(f"Plot saved to {img_path.relative_to(Path.cwd())}\n")

    print("Test complete – review PASS/FAIL messages and figure to assess behaviour.")


if __name__ == "__main__":
    main()
