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
import sys
from pathlib import Path
from typing import Any, Dict
import os
import multiprocessing
import threading
import time

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
    """Set a nested key in a dictionary."""
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


def _run_single(cfg: Dict[str, Any], worker_status: Dict[int, str]) -> Dict[str, Any]:
    """A single grid-search evaluation. Must be a top-level function."""
    pid = os.getpid()
    try:
        # CRITICAL: Prevent CPU thrashing when running many parallel jobs.
        torch.set_num_threads(1)
        # Create a concise identifier for logging
        n_nodes = cfg['reservoir_bundle']['topology']['params']['n_nodes']
        sr = cfg['reservoir_bundle']['reservoir']['spectral_radius']
        scale = cfg['reservoir_bundle']['reservoir']['input_scale']
        lam = cfg['ridge_lam']
        seed = cfg['reservoir_bundle']['reservoir']['random_seed']
        desc = f"nodes={n_nodes}, sr={sr}, scale={scale}, lam={lam}, seed={seed}"
        worker_status[pid] = desc

        # Suppress verbose output from individual runs
        cfg["verbose"] = False
        cfg["plot"] = False
        metrics = run_exp(cfg)
        return metrics
    except Exception as e:
        s = io.StringIO()
        traceback.print_exc(file=s)
        e.args = (f"{e.args[0]}\n\n--- Traceback from worker ---\n{s.getvalue()}",)
        raise
    finally:
        if pid in worker_status:
            del worker_status[pid]


def _display_status(status_dict, total, completed_ref, best_mse_ref, stop_event):
    """A thread to manage the live display of worker statuses."""
    while not stop_event.is_set():
        sys.stdout.write('\x1b[2J\x1b[H')  # ANSI codes to clear screen and move to top-left
        
        completed = completed_ref.value
        best_mse = best_mse_ref.value
        
        header = f"--- Grid Search Status ---"
        progress = f"Progress: {completed}/{total} ({completed/total:.1%}) | Best MSE: {'N/A' if best_mse == float('inf') else f'{best_mse:.5f}'}"
        
        sys.stdout.write(header + '\n')
        sys.stdout.write(progress + '\n')
        sys.stdout.write('-' * (len(progress) if len(progress) > len(header) else len(header)) + '\n')
        sys.stdout.write("\n--- Worker Activity ---\n")

        active_workers = list(status_dict.items())
        for i, (pid, desc) in enumerate(active_workers):
            sys.stdout.write(f"Worker {i+1} (PID {pid}): {desc}\n")
        
        sys.stdout.flush()
        time.sleep(0.5)


def run_grid(config_path: str):
    """Main grid search execution function."""
    with open(config_path) as f:
        search_spec = yaml.safe_load(f)

    base_cfg = search_spec["base_config"]
    param_grid = search_spec["param_grid"]
    search_opts = search_spec["search_options"]
    num_workers = search_opts.get("workers", 4)

    keys, values = zip(*param_grid.items())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    run_configs = []
    for combo in param_combos:
        cfg = base_cfg.copy()
        for key, value in combo.items():
            set_nested(cfg, key, value)
        run_configs.append(cfg)

    print(f"Launching grid search with {len(run_configs)} combinations...", flush=True)

    manager = multiprocessing.Manager()
    worker_status = manager.dict()
    completed_count = manager.Value('i', 0)
    best_val_mse_ref = manager.Value('d', float('inf'))
    stop_display_event = threading.Event()

    display_thread = threading.Thread(
        target=_display_status,
        args=(worker_status, len(run_configs), completed_count, best_val_mse_ref, stop_display_event)
    )
    display_thread.start()

    best_metrics = None
    target_mse = search_opts.get("target_mse")

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_cfg = {executor.submit(_run_single, cfg, worker_status): cfg for cfg in run_configs}
            
            for future in concurrent.futures.as_completed(future_to_cfg):
                completed_count.value += 1
                try:
                    metrics = future.result()
                    val_mse = metrics["val_mse"]

                    if val_mse < best_val_mse_ref.value:
                        best_val_mse_ref.value = val_mse
                        best_metrics = metrics

                        if target_mse is not None and best_val_mse_ref.value < target_mse:
                            print(f"\n[✓] Target MSE {target_mse} reached (val_mse={best_val_mse_ref.value:.4f}). Cancelling remaining jobs…")
                            for fut in future_to_cfg.keys():
                                fut.cancel()
                            break
                except Exception as exc:
                    # Errors will be printed to the main console after the display loop finishes
                    pass
    finally:
        stop_display_event.set()
        display_thread.join()

    if best_metrics is None:
        print("\n[✘] No runs completed successfully!", flush=True)
        return

    print(f"\n--- Grid Search Complete ---", flush=True)
    print(f"Best validation MSE: {best_val_mse_ref.value:.5f}", flush=True)
    print(f"Best parameters found:", flush=True)
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

    if (save_path := search_opts.get("save_best")):
        _save_best(save_path, best_metrics)

    if search_opts.get("plot_best", False):
        print("\nGenerating plot for best configuration...", flush=True)
        plot_cfg = best_metrics["config"].copy()
        plot_cfg["plot"] = True
        run_exp(plot_cfg)
        print("[✔] Plot saved.", flush=True)


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        print("[grid_search] Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Run a generic grid search from a YAML config.")
    parser.add_argument("config", help="Path to the search configuration YAML file.")
    args = parser.parse_args()
    run_grid(args.config)
