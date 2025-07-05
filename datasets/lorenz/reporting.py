"""Verbose reporting utilities specific to the Lorenz attractor dataset.

Designed to mirror the interface expected by `training.train.run`, namely a
module-level ``verbose_pairs`` callable.

For 3-D trajectories we print coordinate triples of the ground-truth and
prediction every *step* samples so that a human or LLM can eyeball error size
without external plotting.  If the prediction tensor is 1-D we fall back to the
pair-wise format used by 1-D datasets.
"""
from __future__ import annotations

import torch
from typing import Sequence

__all__ = ["verbose_pairs"]


def _fmt(t: Sequence[float]) -> str:
    """Nicely format a 1-D or 3-D point for printing."""
    if isinstance(t, torch.Tensor):
        t = t.tolist()
    if not isinstance(t, (list, tuple)):
        return f"{t: .5f}"
    if len(t) == 1:
        return f"{t[0]: .5f}"
    return "(" + ", ".join(f"{v: .4f}" for v in t) + ")"


def verbose_pairs(y_true: torch.Tensor, y_pred: torch.Tensor, step: int = 2):
    """Print coordinate tuples (ground truth, predicted) every *step* indices.

    Parameters
    ----------
    y_true, y_pred : ``torch.Tensor``
        Must have shape ``(T, D)`` where *D* is 1 or 3.
    step : int, default 2
        Sampling stride.
    """
    if y_true.ndim == 2 and y_true.shape[1] == 3:
        mode = "3d"
    else:
        # Squeeze to 1-D for scalar series
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        mode = "1d"

    n = min(len(y_true), len(y_pred))
    print("Coordinate tuples (ground_truth , predicted):")
    for i in range(0, n, step):
        gt = _fmt(y_true[i]) if mode == "3d" else f"{y_true[i].item(): .5f}"
        pr = _fmt(y_pred[i]) if mode == "3d" else f"{y_pred[i].item(): .5f}"
        print(f"t={i:4d}: ( {gt} , {pr} )")
