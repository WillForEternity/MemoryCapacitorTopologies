"""Run a full training pass on the Mackey–Glass dataset and plot predictions.

The figure uses *jet* colormap so that amplitude differences are visually
encoded in colour for quick assessment.
"""
from __future__ import annotations

import math
import sys, os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.train import run as run_exp


CFG = {
    "dataset": {"name": "mackey_glass", "params": {"length": 3000}},
    "reservoir_bundle": {
        "topology": {"name": "small_world", "params": {"n_nodes": 200, "k": 6, "beta": 0.2, "seed": 123}},
        "reservoir": {"mode": "single", "input_dim": 1, "input_scale": 1.0},
    },
    "washout": 50,
}


def main():
    print("Running training pipeline…")
    metrics = run_exp(CFG)
    print(f"MSE = {metrics['mse']:.4f}")

    # Retrieve dataset and prediction from metrics? run_exp currently returns only mse.
    # We'll rerun portions to get predictions for plotting.
    from datasets import load_dataset
    from networks import reservoir_builder

    train, test = load_dataset("mackey_glass", length=3000)
    res = reservoir_builder.build(CFG["reservoir_bundle"])
    # collect states on train to fit readout
    wash = CFG["washout"]
    states_tr = res(train.unsqueeze(0))[0, wash:-1, :]
    y_tr = train[wash + 1 :]
    from training.train import _ridge_regression
    W_out = _ridge_regression(states_tr, y_tr)
    res.reset_state()
    states_te = res(test.unsqueeze(0))[0, wash:-1, :]
    y_te = test[wash + 1 :]
    y_hat = (states_te @ W_out).squeeze()

    t = np.arange(len(y_te))
    fig, ax = plt.subplots(figsize=(10, 4))
    sc = ax.scatter(t, y_hat.detach().numpy(), c=y_hat.detach().numpy(), cmap="jet", s=10, label="predicted")
    ax.plot(t, y_te.numpy(), color="black", linewidth=1, label="target")
    ax.set_title("Mackey–Glass prediction (jet colormap)")
    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="Predicted amplitude")
    out_dir = Path(__file__).with_suffix("").with_name("outputs")
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "mackey_glass_predictions_jet.png")
    print("Figure saved to tests/outputs/mackey_glass_predictions_jet.png")


if __name__ == "__main__":
    main()
