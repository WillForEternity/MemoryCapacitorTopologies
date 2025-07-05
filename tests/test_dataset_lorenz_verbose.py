"""Verbose, self-explaining test for the Lorenz attractor dataset loader.

This mirrors the Mackey–Glass verbose test but for the 3-D Lorenz sequence.  No
`assert` statements are used; instead natural-language PASS/FAIL commentary is
printed so that a human or LLM agent can judge correctness directly from the
stdout and the saved figure.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is on path for direct invocation (python tests/..)
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset

TOL = 1e-6  # numeric tolerance for checks


def banner(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def describe_result_shape(label: str, expected_shape: tuple[int, ...], actual_shape: tuple[int, ...]):
    status = "PASS" if expected_shape == actual_shape else "FAIL"
    print(f"{label}: expected {expected_shape}, got {actual_shape}  ->  {status}")


def main():
    banner("Lorenz Dataset Loader – Verbose Behavioural Test")

    length = 6000
    split = 0.7
    seed = 123

    print(
        f"Invoking load_dataset(\"lorenz\", length={length}, split={split}, seed={seed})\n"
    )
    train, test = load_dataset("lorenz", length=length, split=split, seed=seed)

    # Shape checks (3-D vectors)
    expected_train = int(length * split)
    expected_test = length - expected_train
    describe_result_shape("Train tensor shape", (expected_train, 3), tuple(train.shape))
    describe_result_shape("Test tensor shape", (expected_test, 3), tuple(test.shape))

    # Determinism check
    train2, test2 = load_dataset("lorenz", length=length, split=split, seed=seed)
    same = torch.allclose(train, train2) and torch.allclose(test, test2)
    print(f"Deterministic with fixed seed: {'PASS' if same else 'FAIL'}")

    # --- Visual summary (first 1000 points projected) ---
    print("\nGenerating visual summary (trajectory projection)…")
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(train[:1000, 0].numpy(), train[:1000, 1].numpy(), train[:1000, 2].numpy(), lw=0.5)
    ax.set_title("Lorenz attractor – first 1000 train samples")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()

    out_dir = Path(__file__).with_suffix("").with_name("outputs")
    out_dir.mkdir(exist_ok=True)
    img_path = out_dir / "lorenz_summary.png"
    fig.savefig(img_path, dpi=150)
    print(f"Visual summary saved to {img_path.relative_to(Path.cwd())}\n")

    print("Test complete – review PASS/FAIL messages and plot to assess correctness.")


if __name__ == "__main__":
    main()
