"""Quick grid search for reservoir size (N) and ridge regularisation (lam)
for the Mackey–Glass prediction task.

Usage:
    python experiments/grid_search_mackey_glass.py

Optional CLI flags:
    --sizes 100 200 300 ...         list of reservoir node counts
    --lams 1e-2 1e-3 1e-4 0.0       list of ridge lambda values
    --trials 3                      repeats per configuration (averaged)
    --plot                          generate figure for *best* run

The script prints a table and highlights the best MSE.  Designed to be fast –
uses the built-in small-world topology and keeps training sequence short.
"""
from __future__ import annotations

import argparse
import itertools
import statistics
import sys, os
from pathlib import Path

import torch

# Allow running from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.train import run as run_exp


DEFAULT_SIZES = [100, 200, 300, 500]
DEFAULT_LAMS = [1e-2, 1e-3, 1e-4, 0.0]
DEFAULT_SCALES = [0.1, 0.5, 1.0, 2.0]
DEFAULT_SRS = [0.7, 0.9, 1.1]
DEFAULT_SEEDS = [0, 1, 2]


def parse_args():
    p = argparse.ArgumentParser(description="Grid-search reservoir size / lam")
    p.add_argument("--sizes", type=int, nargs="*", default=DEFAULT_SIZES)
    p.add_argument("--lams", type=float, nargs="*", default=DEFAULT_LAMS)
    p.add_argument("--scales", type=float, nargs="*", default=DEFAULT_SCALES)
    p.add_argument("--srs", type=float, nargs="*", default=DEFAULT_SRS)
    p.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--trials", type=int, default=1, help="repeats per config (per seed)")
    p.add_argument("--plot", action="store_true", help="plot best run")
    return p.parse_args()


def cfg_for(n_nodes: int, lam: float, scale: float, sr: float, seed: int, plot: bool, val_ratio: float):
    return {
        "dataset": {"name": "mackey_glass", "params": {"length": 3000}},
        "reservoir_bundle": {
            "topology": {
                "name": "small_world",
                "params": {"n_nodes": n_nodes, "k": 6, "beta": 0.2, "seed": 42},
            },
            "reservoir": {"mode": "single", "input_dim": 1, "input_scale": scale, "spectral_radius": sr, "random_seed": seed},
        },
        "washout": 50,
        "val_ratio": val_ratio,
        "ridge_lam": lam,
        "verbose": False,
        "plot": plot,
    }


def run_grid(sizes: list[int], lams: list[float], scales: list[float], srs: list[float], seeds: list[int], trials: int, val_ratio: float, do_plot: bool):
    # key: (n_nodes, lam, input_scale, spectral_radius)
    results: dict[tuple[int, float, float, float], list[float]] = {}
    best_mse = float("inf")
    best_cfg = None

    for n, lam, scale, sr in itertools.product(sizes, lams, scales, srs):
        test_mses: list[float] = []
        val_mses: list[float] = []
        print(f"N={n:3d} λ={lam:g} scale={scale} sr={sr} …", end="", flush=True)
        for seed in seeds:
            for t in range(trials):
                cfg = cfg_for(n, lam, scale, sr, seed, plot=False, val_ratio=val_ratio)
                metrics = run_exp(cfg)
                test_mses.append(metrics["mse"])
                val_mses.append(metrics["val_mse"])
        mean_val = statistics.fmean(val_mses)
        mean_test = statistics.fmean(test_mses)
        results[(n, lam, scale, sr)] = (test_mses, val_mses)
        print(f"  mean val={mean_val:.4f} test={mean_test:.4f}")
        if mean_val < best_mse:
            best_mse = mean_val
            best_cfg = cfg_for(n, lam, scale, sr, seeds[0], plot=do_plot, val_ratio=val_ratio)

    print("\n=== Summary ===")
    for (n, lam, scale, sr), (test_list, val_list) in results.items():
        mv = statistics.fmean(val_list)
        mt = statistics.fmean(test_list)
        sv = statistics.pstdev(val_list)
        st = statistics.pstdev(test_list)
        print(f"N={n:3d} λ={lam:>7g} scale={scale} sr={sr} -> VAL μ={mv:.4f} σ={sv:.4f} | TEST μ={mt:.4f} σ={st:.4f}")

    assert best_cfg is not None
    print(f"\nBest config: N={best_cfg['reservoir_bundle']['topology']['params']['n_nodes']}, "
          f"lam={best_cfg['ridge_lam']}  => MSE {best_mse:.4f}")

    if do_plot:
        print("Generating plot for best config …")
        best_cfg["plot"] = True
        best_cfg["verbose"] = True  # show coordinate pairs for inspection
        run_exp(best_cfg)
        print("Plot saved to", best_cfg.get("output_dir", "tests/outputs"))


if __name__ == "__main__":
    args = parse_args()
    run_grid(args.sizes, args.lams, args.scales, args.srs, args.seeds, args.trials, args.val_ratio, args.plot)
