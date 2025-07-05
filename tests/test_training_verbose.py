"""Verbose test for the minimal training pipeline.

Runs an end-to-end Mackey–Glass prediction task using ridge-regression read-out
and prints the final MSE.  Threshold is intentionally loose (≤1.0) just to
catch gross failures while keeping the test lightweight.
"""
from __future__ import annotations

import math
import sys, os
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.train import run as run_exp


def banner(txt):
    print("\n" + "=" * 80)
    print(txt)
    print("=" * 80)


def describe(label, ok, detail=""):
    print(f"{label}: {'PASS' if ok else 'FAIL'} {detail}")


def main():
    banner("Training Pipeline – Verbose Test")

    cfg = {
        "dataset": {"name": "mackey_glass", "params": {"length": 2000}},
        "topology": {"name": "small_world", "params": {"n_nodes": 100, "k": 6, "beta": 0.2, "seed": 42}},
        "reservoir_bundle": {
            "topology": {"name": "small_world", "params": {"n_nodes": 100, "k": 6, "beta": 0.2, "seed": 42}},
            "reservoir": {"mode": "single", "input_dim": 1},
        },
        "washout": 50,
    }

    metrics = run_exp(cfg)
    mse = metrics["mse"]
    describe("MSE finite", math.isfinite(mse), f"mse={mse:.4f}")
    describe("MSE <= 1.0 (sanity)", mse <= 1.0, f"mse={mse:.4f}")

    print("Test complete – review PASS/FAIL and numeric output.")


if __name__ == "__main__":
    main()
