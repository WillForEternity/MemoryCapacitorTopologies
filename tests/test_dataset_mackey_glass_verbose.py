"""Verbose, self-explaining test for the Mackey–Glass dataset loader.

The goal is to let a human (or AI agent) verify the correctness of the
`datasets.load_dataset("mackey_glass")` helper by reading the natural-language
stdout produced by this script *without* needing to dig into the code.

No `assert` statements are used; instead PASS/FAIL tags are printed alongside
quantitative metrics so that tolerance thresholds can be judged externally.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is on path for direct invocation
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset

TOL = 1e-6  # tolerance for numeric checks


def banner(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def describe_result_num(label: str, expected: float, actual: float):
    status = "PASS" if math.isclose(expected, actual, rel_tol=1e-4, abs_tol=TOL) else "FAIL"
    print(f"{label}: expected {expected}, got {actual}  ->  {status}")


def describe_result_shape(label: str, expected_shape: tuple[int, ...], actual_shape: tuple[int, ...]):
    status = "PASS" if expected_shape == actual_shape else "FAIL"
    print(f"{label}: expected {expected_shape}, got {actual_shape}  ->  {status}")


def main():
    banner("Mackey–Glass Dataset Loader – Verbose Behavioural Test")

    length = 2000
    train_ratio = 0.7
    seed = 42

    print(f"Invoking load_dataset(\"mackey_glass\", length={length}, train_ratio={train_ratio}, seed={seed})\n")
    train, test = load_dataset("mackey_glass", length=length, train_ratio=train_ratio, seed=seed)

    # Shape checks
    expected_train = int(length * train_ratio)
    expected_test = length - expected_train
    describe_result_shape("Train tensor shape", (expected_train, 1), tuple(train.shape))
    describe_result_shape("Test tensor shape", (expected_test, 1), tuple(test.shape))

    # Range checks (normalised 0..1)
    train_min, train_max = train.min().item(), train.max().item()
    test_min, test_max = test.min().item(), test.max().item()

    def in_range(x):
        return 0.0 - TOL <= x <= 1.0 + TOL

    print(f"Train min/max: {train_min:.4f}/{train_max:.4f} -> {'PASS' if in_range(train_min) and in_range(train_max) else 'FAIL'}")
    print(f"Test  min/max: {test_min:.4f}/{test_max:.4f} -> {'PASS' if in_range(test_min) and in_range(test_max) else 'FAIL'}")

    # Determinism check: reload and compare
    train2, test2 = load_dataset("mackey_glass", length=length, train_ratio=train_ratio, seed=seed)
    same = torch.allclose(train, train2) and torch.allclose(test, test2)
    print(f"Deterministic with fixed seed: {'PASS' if same else 'FAIL'}")

    # --- Visual summary ---
    print("\nGenerating visual summary (first 500 points)…")
    fig, ax = plt.subplots(figsize=(10, 3))
    t = np.arange(500)
    ax.plot(t, train[:500].numpy(), label="train[0:500]")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title("Mackey–Glass time series – first 500 samples (train)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir = Path(__file__).with_suffix("").with_name("outputs")
    out_dir.mkdir(exist_ok=True)
    img_path = out_dir / "mackey_glass_summary.png"
    fig.savefig(img_path)
    print(f"Visual summary saved to {img_path.relative_to(Path.cwd())}\n")

    print("Test complete – review PASS/FAIL messages and plot to assess correctness.")


if __name__ == "__main__":
    main()
