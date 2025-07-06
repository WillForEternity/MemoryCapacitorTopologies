"""Procedural generator for the Mackey–Glass chaotic time series.

The `load()` function returns (train, test) PyTorch tensors ready for reservoir
computing experiments.  Having the generator in its own module keeps the package
namespace clean and allows future extensions (e.g., cached files, different
parameter sets) without bloating the package `__init__`.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import torch

__all__ = ["load"]


def _generate(
    length: int,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: int = 10,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a Mackey–Glass sequence of given *length* (warm-up discarded).

    Parameters
    ----------
    length : int
        Number of points returned (after warm-up).
    tau : int, default 17
        Delay constant in time steps.
    beta, gamma : float
        Equation parameters.
    n : int, default 10
        Non-linearity order.
    seed : int | None
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    total_len = length + tau
    x = np.zeros(total_len, dtype=np.float32)
    x[0] = 1.2
    for t in range(1, total_len):
        x_tau = x[t - tau] if t - tau >= 0 else 0.0
        x[t] = x[t - 1] + beta * x_tau / (1 + x_tau ** n) - gamma * x[t - 1]
    return x[tau:]


def load(
    length: int = 10000,
    train_ratio: float = 0.7,
    seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return normalised (train, test) tensors suitable for time-series tasks."""
    series = _generate(length=length, seed=seed)

    # Safely normalize the series to [0, 1], handling constant series.
    s_min, s_max = series.min(), series.max()
    delta = s_max - s_min
    if delta > 1e-8:
        series = (series - s_min) / delta
    else:
        # If the series is constant, normalize to all zeros.
        series = np.zeros_like(series)
    split = int(train_ratio * len(series))
    train = torch.from_numpy(series[:split]).float().unsqueeze(-1)
    test = torch.from_numpy(series[split:]).float().unsqueeze(-1)
    return train, test
