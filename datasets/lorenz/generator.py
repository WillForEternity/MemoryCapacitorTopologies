"""Lorenz attractor sequence generator.

Returns training and test sequences suitable for the memcapacitive reservoir
pipeline.  By default we generate a single long trajectory, take the first part
for training and the remainder for test.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

__all__ = ["load"]

def _lorenz_step(state: np.ndarray, *, sigma: float, rho: float, beta: float, dt: float) -> np.ndarray:
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return state + dt * np.array([dx, dy, dz])

def _simulate(T: int, *, dt: float, sigma: float, rho: float, beta: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Random initial condition on the attractor basin
    state = rng.uniform(-10, 10, size=3)
    traj = np.empty((T, 3), dtype=np.float32)
    for t in range(T):
        traj[t] = state
        state = _lorenz_step(state, sigma=sigma, rho=rho, beta=beta, dt=dt)
    return traj

def load(*, length: int = 6000, dt: float = 0.01, split: float = 0.5, seed: int = 0,
         sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate Lorenz attractor sequences.

    Parameters
    ----------
    length : int
        Total timesteps to generate.
    dt : float
        Integration time-step.
    split : float
        Fraction of the trajectory to use for *training*; remainder is test.
    seed : int
        RNG seed for reproducible initial condition.
    sigma, rho, beta : float
        Lorenz system parameters.
    """
    if not (0.0 < split < 1.0):
        raise ValueError("split must be in (0,1)")

    data = _simulate(length, dt=dt, sigma=sigma, rho=rho, beta=beta, seed=seed)
    train_len = int(length * split)
    train_seq = torch.from_numpy(data[:train_len])
    test_seq = torch.from_numpy(data[train_len:])
    return train_seq, test_seq
