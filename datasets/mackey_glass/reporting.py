"""Verbose reporting utilities specific to the Mackey-Glass dataset."""
from __future__ import annotations

import torch
from typing import Sequence

__all__ = ["verbose_pairs"]


def verbose_pairs(y_true: torch.Tensor, y_pred: torch.Tensor, step: int = 2):
    """Print coordinate pairs (ground truth, predicted) every *step* indices.

    Parameters
    ----------
    y_true, y_pred : 1-D tensors of equal length
    step : int
        Sampling stride; default prints every 2 time steps.
    """
    if y_true.ndim != 1 or y_pred.ndim != 1:
        y_true, y_pred = y_true.squeeze(), y_pred.squeeze()
    n = min(len(y_true), len(y_pred))
    print("Coordinate pairs (ground_truth, predicted):")
    for i in range(0, n, step):
        print(f"t={i:4d}: ( {y_true[i].item(): .5f} , {y_pred[i].item(): .5f} )")
