"""Minimal training pipeline for memcapacitive reservoirs.

Intended to be called programmatically (GUI) and from the CLI::

    python -m training.train --config configs/example.yaml

For simplicity the verbose test drives it with an in-memory config dict.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from datasets import load_dataset
from networks import reservoir_builder


def _ridge_regression(
    X: torch.Tensor,
    y: torch.Tensor,
    lam: float = 1e-2,
    bias_index: int | None = None,
):
    """Closed-form ridge regression.

    If *bias_index* is provided, that row/column is **not** regularised so the
    bias term is free while all other weights are shrink-regularised.
    """
    if y.ndim == 1:
        y = y.unsqueeze(1)
    XtX = X.T @ X  # (N,N)
    rhs = X.T @ y  # (N,1)
    reg = lam * torch.eye(XtX.shape[0], device=X.device)
    if bias_index is not None:
        reg[bias_index, bias_index] = 0.0
    W = torch.linalg.solve(XtX + reg, rhs)
    return W


def run(cfg: Dict[str, Any]):
    """Run a single experiment. Returns metrics dict."""
    # ---------------- Dataset ----------------
    ds_cfg = cfg["dataset"]
    data = load_dataset(ds_cfg["name"], **ds_cfg.get("params", {}))  # expected (train, test)
    train_seq, test_seq = data

    # Optional validation split from the tail of the training sequence
    val_ratio = cfg.get("val_ratio", 0.0)
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0,1)")
    if val_ratio > 0:
        split_idx = int(len(train_seq) * (1 - val_ratio))
        train_seq, val_seq = train_seq[:split_idx], train_seq[split_idx:]
    else:
        val_seq = None

    # ---------------- Reservoir -------------
    res = reservoir_builder.build(cfg["reservoir_bundle"])

    def _collect_states(seq: torch.Tensor, washout: int = 10):
        """Drives the reservoir with an input sequence and collects states."""
        # Ensure input is a tensor on the correct device.
        # The reservoir's forward pass also moves it, but having it on the
        # correct device here ensures the returned Y tensor is also on that device.
        seq = torch.as_tensor(seq, dtype=torch.float32).to(res.device)
        states = res(seq.unsqueeze(0))  # (1, T, N)
        # X is reservoir states, Y is the next-step prediction target
        return states[0, washout:-1, :], seq[washout + 1 :]

    Xtr, Ytr = _collect_states(train_seq, cfg.get("washout", 10))
    res.reset_state()
    if val_seq is not None:
        Xval, Yval = _collect_states(val_seq, cfg.get("washout", 10))
        res.reset_state()
    else:
        Xval = Yval = None
    Xte, Yte = _collect_states(test_seq, cfg.get("washout", 10))

    # ---------------- Bias-augmented ridge read-out ---------------
    # Center & scale features; center targets
    X_mean = Xtr.mean(0, keepdim=True)
    X_std = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Y_mean = Ytr.mean(dim=0, keepdim=True)
    Y_std = Ytr.std(dim=0, keepdim=True) + 1e-8  # avoid div-zero

    Xtr_n = (Xtr - X_mean) / X_std
    Xte_n = (Xte - X_mean) / X_std
    Ytr_z = (Ytr - Y_mean) / Y_std

    ones_tr = torch.ones(Xtr_n.shape[0], 1, device=Xtr_n.device)
    ones_te = torch.ones(Xte_n.shape[0], 1, device=Xte_n.device)
    Xtr_aug = torch.cat([Xtr_n, ones_tr], dim=1)
    Xte_aug = torch.cat([Xte_n, ones_te], dim=1)

    lam = cfg.get("ridge_lam", 1e-2)
    W_out = _ridge_regression(Xtr_aug, Ytr_z, lam=lam, bias_index=Xtr_aug.shape[1]-1)
    # Validation prediction
    if Xval is not None:
        Xval_n = (Xval - X_mean) / X_std
        ones_val = torch.ones(Xval_n.shape[0], 1, device=Xval_n.device)
        Xval_aug = torch.cat([Xval_n, ones_val], dim=1)
        Yval_hat = (Xval_aug @ W_out).squeeze() * Y_std + Y_mean
    # Test prediction
    Y_hat_z = (Xte_aug @ W_out).squeeze()
    Y_hat = Y_hat_z * Y_std + Y_mean

    # ---------------- Optional plotting -------------------------
    fig_path = None
    if cfg.get("plot", False):
        import matplotlib.pyplot as plt
        from pathlib import Path
        import numpy as np

        # If outputs are multi-dimensional, use the first dimension for y-axis,
        # but keep colour from the full norm so the scatter colouring still
        # conveys amplitude information.
        y_pred = Y_hat[:, 0] if Y_hat.ndim == 2 else Y_hat
        y_true = Yte[:, 0] if Yte.ndim == 2 else Yte

        import matplotlib
        if ds_cfg['name'] == 'lorenz' and Y_hat.ndim == 2 and Y_hat.shape[1] == 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3D projection
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(
                Yte[:, 0].cpu(),
                Yte[:, 1].cpu(),
                Yte[:, 2].cpu(),
                color='black',
                linewidth=1,
                label='target',
                alpha=0.7,
            )
            ax.scatter(
                Y_hat[:, 0].cpu(),
                Y_hat[:, 1].cpu(),
                Y_hat[:, 2].cpu(),
                c=np.linalg.norm(Y_hat.cpu(), axis=1),
                cmap='jet',
                s=6,
                label='pred',
                depthshade=True,
            )
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title('Lorenz trajectory – target (line) vs pred (dots)')
            ax.legend()
        else:
            t = np.arange(len(y_pred))
            fig, ax = plt.subplots(figsize=(8, 3))
            sc = ax.scatter(
                t,
                y_pred.detach().cpu().numpy(),
                c=y_pred.detach().cpu().numpy(),
                cmap="jet",
                s=8,
                label="pred",
            )
            ax.plot(t, y_true.cpu().numpy(), color="black", linewidth=1, label="target")
            ax.set_title(f"Prediction – {ds_cfg['name']}")
            ax.legend()
            fig.colorbar(sc, ax=ax, label="pred amplitude")
        out_dir = Path(cfg.get("output_dir", "training/outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir / f"{ds_cfg['name']}_predictions_jet.png"
        fig.savefig(fig_path)
        plt.close(fig)

        # Optional interactive plotly export for Lorenz 3-D trajectory
        if ds_cfg['name'] == 'lorenz' and Y_hat.ndim == 2 and Y_hat.shape[1] == 3:
            try:
                import plotly.graph_objects as go

                fig_int = go.Figure()
                fig_int.add_trace(
                    go.Scatter3d(
                        x=Yte[:, 0].cpu(),
                        y=Yte[:, 1].cpu(),
                        z=Yte[:, 2].cpu(),
                        mode='lines',
                        line=dict(color='black', width=2),
                        name='target',
                        opacity=0.7,
                    )
                )
                fig_int.add_trace(
                    go.Scatter3d(
                        x=Y_hat[:, 0].cpu(),
                        y=Y_hat[:, 1].cpu(),
                        z=Y_hat[:, 2].cpu(),
                        mode='markers',
                        marker=dict(size=2, color=np.linalg.norm(Y_hat.cpu(), axis=1), colorscale='Jet'),
                        name='pred',
                    )
                )
                fig_int.update_layout(title='Lorenz trajectory – interactive', scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'))
                html_path = out_dir / 'lorenz_predictions_3d.html'
                fig_int.write_html(html_path, include_plotlyjs='cdn')
            except ModuleNotFoundError:
                # Plotly not installed; skip interactive export
                pass

    # ---------------- Verbose reporting (dataset plug-in) ---------------
    if cfg.get("verbose", True):
        import importlib
        try:
            rep_mod = importlib.import_module(f"datasets.{ds_cfg['name']}.reporting")
            if hasattr(rep_mod, "verbose_pairs"):
                rep_mod.verbose_pairs(Yte, Y_hat, step=cfg.get("pair_stride", 2))
        except ModuleNotFoundError:
            pass

    test_mse = torch.mean((Y_hat - Yte) ** 2).item()
    val_mse = torch.mean((Yval_hat - Yval) ** 2).item() if Xval is not None else None
    return {
        "mse": test_mse,
        "val_mse": val_mse,
        "y_true": Yte.detach(),
        "y_pred": Y_hat.detach(),
        "W_out": W_out.detach(),
        "adjacency": res.W.detach(),
        "Win": res.Win.detach(),
        "config": cfg,
    }


# ---------------- CLI ---------------------

def _parse_cli():
    p = argparse.ArgumentParser(description="Run training experiment")
    p.add_argument("--config", type=str, required=True, help="YAML config file")
    return p.parse_args()


def _load_cfg(path: str | Path):
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


if __name__ == "__main__":
    args = _parse_cli()
    metrics = run(_load_cfg(args.config))
    print(metrics)
