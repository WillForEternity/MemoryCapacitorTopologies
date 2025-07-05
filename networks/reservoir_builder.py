"""Factory helpers for constructing reservoirs based on a config dict.

Designed so the GUI or training script can call a *single* function without
knowing the internal module layout.

Currently implemented modes:
    • 'single'  – returns a `MemcapacitiveReservoir`

Place-holders exist for 'multi' and 'cluster'.  Those will raise
`NotImplementedError` until implemented.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from networks.topologies import make_topology
from networks.reservoir import MemcapacitiveReservoir

__all__ = ["build"]


def build(cfg: Dict[str, Any]):
    """Build and return a reservoir according to *cfg*.

    Example cfg::
        cfg = {
            "topology": {"name": "small_world", "params": {"n_nodes": 100, "k": 6, "beta": 0.2}},
            "reservoir": {
                "mode": "single",
                "input_dim": 1,
                "memc_params": {"c0": 1.0, "k": 0.5, "dt": 1e-3},
                "input_scale": 1.0,
            },
        }
    """
    topo_cfg = cfg["topology"]
    adj: np.ndarray = make_topology(topo_cfg["name"], **topo_cfg.get("params", {}))

    # Optional spectral-radius rescaling
    res_cfg = cfg["reservoir"]
    target_sr = res_cfg.get("spectral_radius")
    if target_sr is not None:

        vals = np.linalg.eigvals(adj)
        current_sr = np.max(np.abs(vals))
        if current_sr > 0:
            adj = adj * (target_sr / current_sr)

    # re-use res_cfg from above
    mode = res_cfg.get("mode", "single")

    if mode == "single":
        # Determine device: prefer CUDA if available unless explicitly set
        device_cfg = res_cfg.get("device")
        if device_cfg is None:
            device_cfg = "cuda" if torch.cuda.is_available() else "cpu"

        return MemcapacitiveReservoir(
            adjacency=adj,
            input_dim=res_cfg["input_dim"],
            memc_params=res_cfg.get("memc_params"),
            input_scale=res_cfg.get("input_scale", 1.0),
            random_seed=res_cfg.get("random_seed", 0),
            device=device_cfg,
        )
    elif mode in {"multi", "cluster"}:
        raise NotImplementedError(f"Reservoir mode '{mode}' not implemented yet.")
    else:
        raise ValueError(f"Unknown reservoir mode: {mode}")
