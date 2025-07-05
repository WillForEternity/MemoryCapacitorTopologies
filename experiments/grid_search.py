"""Generic, configuration-driven grid search.

This script is the main entrypoint for all hyperparameter searches. It is driven
by a YAML configuration file that specifies a base configuration and a grid of
parameters to search over.

Usage
-----
    python experiments/grid_search.py [path/to/your_search.yaml]

See Also
--------
configs/lorenz_search.yaml : An example search configuration.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

# Ensure project root on sys.path so we can import sibling packages
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train import run as run_exp


def set_nested(d: dict, keys: str, value: Any):
    """Set a nested key in a dictionary.

    >>> d = {'a': {'b': 1}}
    >>> set_nested(d, 'a.b', 2)
    >>> d
    {'a': {'b': 2}}
    """
    *key_path, last_key = keys.split('.')
    for key in key_path:
        d = d.setdefault(key, {})
    d[last_key] = value


def _save_best(file: str | Path, metrics: dict):
    """Save the best performing model's weights and config."""
    file = Path(file)
    arr_dict = {
        "adjacency": metrics["adjacency"].cpu().numpy(),
        "Win": metrics["Win"].cpu().numpy(),
        "W_out": metrics["W_out"].cpu().numpy(),
    }
    np.savez(file, **arr_dict, config=str(metrics["config"]))
    print(f"\n[✔] Best network saved to {file}")


def _run_single(cfg: Dict) -> Dict:
    """Wrapper to run a single experiment configuration."""
    # Suppress verbose output from individual runs during grid search
    cfg["verbose"] = False
    cfg["plot"] = False
    metrics = run_exp(cfg)
    # Return the full metrics dict, which includes the config
    return metrics


def run_grid(config_path: str):
    """Main grid search execution function."""
    with open(config_path) as f:
        search_spec = yaml.safe_load(f)

    base_cfg = search_spec["base_config"]
    param_grid = search_spec["param_grid"]
    search_opts = search_spec["search_options"]

    # Generate all parameter combinations
    keys, values = zip(*param_grid.items())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Launching grid search with {len(param_combos)} combinations...")

    # Create a list of full configuration dictionaries for the pool
    run_configs = []
    for combo in param_combos:
        cfg = base_cfg.copy()  # Start with a fresh copy of the base config
        for key, value in combo.items():
            set_nested(cfg, key, value)
        run_configs.append(cfg)

    best_val_mse = float("inf")
    best_metrics = None

    with concurrent.futures.ProcessPoolExecutor(max_workers=search_opts.get("workers", 4)) as executor:
        future_to_cfg = {executor.submit(_run_single, cfg): cfg for cfg in run_configs}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_cfg)):
            try:
                metrics = future.result()
                val_mse = metrics["val_mse"]
                cfg = metrics["config"]
                # Create a human-readable identifier for the run's parameters
                run_params_str = ', '.join([f"{k.split('.')[-1]}={v}" for k, v in combo.items()])
                print(f"({i+1}/{len(param_combos)}) -> val_mse={val_mse:.4f} | {run_params_str}")

                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_metrics = metrics
            except Exception as exc:
                print(f"A run generated an exception: {exc}")

    if best_metrics is None:
        print("\n[✘] No runs completed successfully!")
        return

    print(f"\n--- Grid Search Complete ---")
    print(f"Best validation MSE: {best_val_mse:.5f}")
    print(f"Best parameters found:")
    best_combo = {k: best_metrics['config'] for k in param_grid.keys()}
    for key, value in best_metrics['config'].items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                 if isinstance(sub_value, dict):
                    for ssub_key, ssub_value in sub_value.items():
                        print(f'    {key}.{sub_key}.{ssub_key}: {ssub_value}')
                 else:
                    print(f'    {key}.{sub_key}: {sub_value}')
        else:
            print(f'    {key}: {value}')

    # Save the best model if a path is provided
    if (save_path := search_opts.get("save_best")):
        _save_best(save_path, best_metrics)

    # Generate a plot for the best model if requested
    if search_opts.get("plot_best", False):
        print("\nGenerating plot for best configuration...")
        plot_cfg = best_metrics["config"].copy()
        plot_cfg["plot"] = True
        run_exp(plot_cfg)
        print("[✔] Plot saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a generic grid search from a YAML config.")
    parser.add_argument("config", help="Path to the search configuration YAML file.")
    args = parser.parse_args()
    run_grid(args.config)
