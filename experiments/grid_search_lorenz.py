"""Grid search for Lorenz attractor prediction.

Usage example::

    python experiments/grid_search_lorenz.py \
        --sizes 100 200 \
        --lams 1e-3 1e-4 0 \
        --scales 0.5 1 2 \
        --srs 0.7 0.9 1.1 \
        --seeds 0 1 2 \
        --val_ratio 0.2 \
        --save_best best_lorenz.npz

The script reuses the generic ``training.train.run`` pipeline so it stays
consistent with Mackey–Glass experiments.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import itertools
import statistics
from typing import List

import numpy as np
import torch

# Ensure project root on sys.path so we can import sibling packages
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train import run as run_exp

DEFAULT_SIZES = [100, 200]
DEFAULT_LAMS = [1e-3, 1e-4, 0.0]
DEFAULT_SCALES = [0.5, 1.0, 2.0]
DEFAULT_SRS = [0.7, 0.9, 1.1]
DEFAULT_SEEDS = [0, 1, 2]


def _parse_args():
    p = argparse.ArgumentParser(description="Grid search Lorenz prediction")
    p.add_argument("--sizes", type=int, nargs="*", default=DEFAULT_SIZES)
    p.add_argument("--lams", type=float, nargs="*", default=DEFAULT_LAMS)
    p.add_argument("--scales", type=float, nargs="*", default=DEFAULT_SCALES)
    p.add_argument("--srs", type=float, nargs="*", default=DEFAULT_SRS)
    p.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--washout", type=int, default=50)
    p.add_argument("--trials", type=int, default=1, help="repeats per seed")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--save_best", type=str, default="", help="npz filename to store best network")
    p.add_argument("--workers", type=int, default=4, help="Concurrent workers per node")
    return p.parse_args()


def _cfg_for(n_nodes: int, lam: float, scale: float, sr: float, seed: int, val_ratio: float, washout: int, plot: bool):
    return {
        "dataset": {
            "name": "lorenz",
            "params": {"length": 6000, "split": 0.5},
        },
        "reservoir_bundle": {
            "topology": {
                "name": "small_world",
                "params": {"n_nodes": n_nodes, "k": 6, "beta": 0.2, "seed": 42},
            },
            "reservoir": {
                "mode": "single",
                "input_dim": 3,
                "input_scale": scale,
                "spectral_radius": sr,
                "random_seed": seed,
            },
        },
        "washout": washout,
        "val_ratio": val_ratio,
        "ridge_lam": lam,
        "verbose": False,
        "plot": plot,
    }


def _save_best(file: str | Path, metrics: dict):
    file = Path(file)
    arr_dict = {
        "adjacency": metrics["adjacency"].cpu().numpy(),
        "Win": metrics["Win"].cpu().numpy(),
        "W_out": metrics["W_out"].cpu().numpy(),
    }
    np.savez(file, **arr_dict, config=str(metrics["config"]))
    print(f"Best network saved to {file}")


def _run_single(args_tuple):
    n, lam, scale, sr, seed, val_ratio, washout = args_tuple
    cfg = _cfg_for(n, lam, scale, sr, seed, val_ratio, washout, plot=False)
    m = run_exp(cfg)
    return (n, lam, scale, sr, m["val_mse"], m["mse"], m)

def run_grid(sizes: List[int], lams: List[float], scales: List[float], srs: List[float], seeds: List[int], val_ratio: float, washout: int, trials: int, do_plot: bool, save_best: str, workers: int = 4):
    results = {}
    best_val = float("inf")
    best_metrics = None

    import concurrent.futures

    combos = list(itertools.product(sizes, lams, scales, srs))
    pool_args = []
    for n, lam, scale, sr in combos:
        for seed in seeds:
            for _ in range(trials):
                pool_args.append((n, lam, scale, sr, seed, val_ratio, washout))

    val_dict = {c: [] for c in combos}
    test_dict = {c: [] for c in combos}

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as exe:
        for res in exe.map(_run_single, pool_args):
            n, lam, scale, sr, val_mse, test_mse, m = res
            key = (n, lam, scale, sr)
            val_dict[key].append(val_mse)
            test_dict[key].append(test_mse)
            print(f"Partial {key} seed done -> val={val_mse:.4f} test={test_mse:.4f}")
            if val_mse < best_val:
                best_val = val_mse
                best_metrics = m
                best_cfg = m["config"]

    results = {k: (val_dict[k], test_dict[k]) for k in combos}
    # Summary
    print("\n=== Summary ===")
    for (n, lam, scale, sr), (val_l, test_l) in results.items():
        mv, sv = statistics.fmean(val_l), statistics.pstdev(val_l)
        mt, st = statistics.fmean(test_l), statistics.pstdev(test_l)
        print(f"N={n:3d} λ={lam:>7g} scale={scale} sr={sr} -> VAL μ={mv:.4f} σ={sv:.4f} | TEST μ={mt:.4f} σ={st:.4f}")

    if best_metrics is None:
        print("No runs completed!")
        return
    print(f"\nBest configuration achieved VAL MSE={best_val:.4f}")
    if save_best:
        best_metrics["config"] = best_cfg  # insert full cfg
        _save_best(save_best, best_metrics)
    if do_plot:
        # re-run best with plotting enabled
        cfg_plot = best_cfg.copy()
        cfg_plot["plot"] = True
        run_exp(cfg_plot)


if __name__ == "__main__":
    args = _parse_args()
    run_grid(
        args.sizes,
        args.lams,
        args.scales,
        args.srs,
        args.seeds,
        args.val_ratio,
        args.washout,
        args.trials,
        args.plot,
        args.save_best,
    )
