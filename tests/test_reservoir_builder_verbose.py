"""Verbose test for `networks.reservoir_builder.build`.

Ensures that the builder can create a single-mode reservoir from a minimal
config dict and that the resulting module produces sensible output.
"""
from __future__ import annotations

import math
import sys, os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from networks import reservoir_builder


def banner(txt):
    print("\n" + "=" * 80)
    print(txt)
    print("=" * 80)


def describe(label, ok, detail=""):
    print(f"{label}: {'PASS' if ok else 'FAIL'} {detail}")


def main():
    banner("Reservoir Builder – Verbose Test")

    cfg = {
        "topology": {"name": "small_world", "params": {"n_nodes": 30, "k": 4, "beta": 0.2, "seed": 7}},
        "reservoir": {"mode": "single", "input_dim": 1},
    }

    res = reservoir_builder.build(cfg)
    describe("Type check", res.__class__.__name__ == "MemcapacitiveReservoir")

    # Simple input sequence
    t = torch.linspace(0, 2 * math.pi, 60).unsqueeze(0).unsqueeze(-1)
    out = res(t)
    describe("Output shape", out.shape == (1, 60, 30), f"-> {out.shape}")

    # Quick dynamics sanity
    var = out.var().item()
    describe("Non-zero variance", var > 1e-5, f"var={var:.2e}")

    # Plot first node
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(out[0, :, 0].detach().numpy())
    ax.set_title("Builder test – node 0 charge vs. time")
    ax.set_xlabel("t")
    ax.set_ylabel("q")
    fig.tight_layout()
    out_dir = Path(__file__).with_suffix("").with_name("outputs")
    out_dir.mkdir(exist_ok=True)
    img_path = out_dir / "reservoir_builder_summary.png"
    fig.savefig(img_path)
    print(f"Plot saved to {img_path.relative_to(Path.cwd())}\n")

    # Unsupported mode check
    try:
        cfg_bad = {"topology": cfg["topology"], "reservoir": {"mode": "multi"}}
        _ = reservoir_builder.build(cfg_bad)
        describe("Unsupported mode raises error", False)
    except NotImplementedError:
        describe("Unsupported mode raises error", True)


if __name__ == "__main__":
    main()
