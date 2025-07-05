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
import torch
import traceback
import io

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


def _run_single(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """A single grid-search evaluation. Must be a top-level function."""
    # CRITICAL: Prevent CPU thrashing when running many parallel jobs.
    # Each worker process should use a single thread.
    torch.set_num_threads(1)
    # Create a concise identifier for logging
    try:
        # Attempt to build a descriptive run ID from key parameters
        n_nodes = cfg['reservoir_bundle']['topology']['params']['n_nodes']
        sr = cfg['reservoir_bundle']['reservoir']['spectral_radius']
        lam = cfg['ridge_lam']
        seed = cfg['reservoir_bundle']['reservoir']['random_seed']
        run_id = f"n_nodes={n_nodes}, sr={sr}, lam={lam}, seed={seed}"
        print(f"[Worker] STARTING: {run_id}", flush=True)
    except KeyError:
        # Fallback if the config structure is unexpected
        print("[Worker] STARTING a new run...", flush=True)

    # Suppress verbose output from individual runs during grid search
    cfg["verbose"] = False
    cfg["plot"] = False
    try:
        metrics = run_exp(cfg)
        return metrics
    except Exception as e:
        # Capture traceback and add it to the exception to be sent back to the main process
        s = io.StringIO()
        traceback.print_exc(file=s)
        e.args = (f"{e.args[0]}\n\n--- Traceback from worker ---\n{s.getvalue()}",)
        raise


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

    print(f"Launching grid search with {len(param_combos)} combinations...", flush=True)

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
        print(f"Submitting {len(run_configs)} jobs to the process pool ({search_opts.get('workers', 4)} workers)...", flush=True)
        future_to_cfg = {executor.submit(_run_single, cfg): cfg for cfg in run_configs}
        print("All jobs submitted. Waiting for results...", flush=True)

        for i, future in enumerate(concurrent.futures.as_completed(future_to_cfg)):
            # Retrieve the original config to identify the run's parameters
            original_cfg = future_to_cfg[future]
            try:
                metrics = future.result()
                val_mse = metrics["val_mse"]

                # Create a human-readable identifier for the run's parameters for logging
                run_params = {}
                for key in param_grid.keys():
                    # Traverse the nested dict to get the value for the current param
                    value = original_cfg
                    for p in key.split('.'):
                        value = value[p]
                    run_params[key.split('.')[-1]] = value
                run_params_str = ', '.join([f"{k}={v}" for k, v in run_params.items()])

                val_mse_str = f"{val_mse:.4f}" if val_mse is not None else "N/A"
                print(f"({i+1}/{len(param_combos)}) [✔] FINISHED -> val_mse={val_mse_str} | {run_params_str}", flush=True)

                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_metrics = metrics
            except Exception as exc:
                print(f"({i+1}/{len(param_combos)}) [✘] FAILED -> {exc}", flush=True)

    if best_metrics is None:
        print("\n[✘] No runs completed successfully!", flush=True)
        return

    print(f"\n--- Grid Search Complete ---", flush=True)
    print(f"Best validation MSE: {best_val_mse:.5f}", flush=True)
    print(f"Best parameters found:", flush=True)
    best_combo = {k: best_metrics['config'] for k in param_grid.keys()}
    for key, value in best_metrics['config'].items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                 if isinstance(sub_value, dict):
                    for ssub_key, ssub_value in sub_value.items():
                        print(f'    {key}.{sub_key}.{ssub_key}: {ssub_value}', flush=True)
                 else:
                    print(f'    {key}.{sub_key}: {sub_value}', flush=True)
        else:
            print(f'    {key}: {value}', flush=True)

    # Save the best model if a path is provided
    if (save_path := search_opts.get("save_best")):
        _save_best(save_path, best_metrics)

    # Generate a plot for the best model if requested
    if search_opts.get("plot_best", False):
        print("\nGenerating plot for best configuration...", flush=True)
        plot_cfg = best_metrics["config"].copy()
        plot_cfg["plot"] = True
        run_exp(plot_cfg)
        print("[✔] Plot saved.", flush=True)


if __name__ == "__main__":
    # Set start method for PyTorch multiprocessing. 'spawn' is safer for CUDA.
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        print("[grid_search] Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        # start method can only be set once
        pass

    parser = argparse.ArgumentParser(description="Run a generic grid search from a YAML config.")
    parser.add_argument("config", help="Path to the search configuration YAML file.")
    args = parser.parse_args()
    run_grid(args.config)
