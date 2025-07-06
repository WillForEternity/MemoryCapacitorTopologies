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
from tqdm import tqdm
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
        # Logging is now handled by the main process with a progress bar.
    except KeyError:
        # Fallback if the config structure is unexpected
        pass

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

    # Create a list of full configuration dictionaries for the pool
    run_configs = []
    for combo in param_combos:
        cfg = base_cfg.copy()  # Start with a fresh copy of the base config
        for key, value in combo.items():
            set_nested(cfg, key, value)
        run_configs.append(cfg)

    best_val_mse = float("inf")
    best_metrics = None
    target_mse = search_opts.get("target_mse")

    with concurrent.futures.ProcessPoolExecutor(max_workers=search_opts.get("workers", 4)) as executor:
        future_to_cfg = {executor.submit(_run_single, cfg): cfg for cfg in run_configs}

        # Use tqdm for a clean progress bar
        pbar = tqdm(concurrent.futures.as_completed(future_to_cfg), total=len(run_configs), desc="Grid Search")
        for future in pbar:
            try:
                metrics = future.result()
                val_mse = metrics["val_mse"]

                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_metrics = metrics
                    # Update progress bar with the best MSE found so far
                    pbar.set_postfix(best_mse=f"{best_val_mse:.4f}")

                    # Early stop if target reached
                    if target_mse is not None and best_val_mse < target_mse:
                        pbar.write(f"[✓] Target MSE {target_mse} reached (val_mse={best_val_mse:.4f}). Cancelling remaining jobs…", file=sys.stdout)
                        for fut in future_to_cfg.keys():
                            fut.cancel()
                        break
            except Exception as exc:
                # Log errors to the tqdm console to avoid breaking the progress bar
                pbar.write(f"[✘] A run failed: {exc}", file=sys.stderr)

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
