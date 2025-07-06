"""Generate plots for a trained Lorenz model.

This is a lightweight, standalone script designed to be called from the remote
experiment runner. It loads a saved model and generates both interactive HTML
and static PNG plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import ast

import numpy as np

# Ensure project root is on sys.path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train import run as run_exp

def main():
    parser = argparse.ArgumentParser(description="Plot Lorenz predictions from a saved model.")
    parser.add_argument("model_path", type=str, help="Path to the saved .npz model file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output plots.")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load the config from the .npz file
    with np.load(model_path, allow_pickle=True) as data:
        # The config is saved as a string representation of a dict
        config_str = data['config'].item()
        config = ast.literal_eval(config_str)

    # Set up the plotting configuration
    config["plot"] = True
    config["verbose"] = False
    config["fig_path"] = str(output_dir / f"{model_path.stem}_predictions.png")
    config["html_path"] = str(output_dir / f"{model_path.stem}_predictions_3d.html")

    # Load the model weights into the config
    with np.load(model_path, allow_pickle=True) as data:
        config['reservoir_bundle']['reservoir']['adjacency_matrix'] = data['adjacency']
        config['reservoir_bundle']['reservoir']['Win'] = data['Win']
        config['W_out'] = data['W_out']

    print(f"[plot_remote] Generating plots for {model_path.name}...")
    print(f"[plot_remote]   - PNG -> {config['fig_path']}")
    print(f"[plot_remote]   - HTML -> {config['html_path']}")

    # Run the experiment to generate and save plots
    run_exp(config)

    print(f"[✔] Plots saved successfully to {output_dir.resolve()}")

if __name__ == "__main__":
    main()
