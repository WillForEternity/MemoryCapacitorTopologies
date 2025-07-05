"""MemcapacitiveReservoir module

Minimal, self-contained PyTorch implementation that can already be used by the
GUI and by the forthcoming training script.  It intentionally keeps the physics
simple: each node is a `Memcapacitor` whose voltage input is the weighted sum of
previous charges (recurrent term) plus a learned input projection.

The design focuses on:
1. **Modularity** – accepts any adjacency matrix and neuron constructor.
2. **Device-agnostic** – default neuron is the provided `Memcapacitor` model but
   any `nn.Module` with `forward(v: torch.Tensor) -> torch.Tensor` can be
   injected.
3. **Batch & time support** – inputs: `(batch, time, input_dim)`; outputs
   returned states `(batch, time, n_nodes)`.
4. **Stateless wrapper** – internal state (`charges`) is stored on the module so
   `.reset_state()` enables clean sequence starts.

This is a *baseline*; future work can add spectral radius scaling, leaky
integration, and more complex inter-neuron dynamics.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from models.memcapacitor import Memcapacitor

__all__ = ["MemcapacitiveReservoir"]


class MemcapacitiveReservoir(nn.Module):
    """Reservoir of memcapacitor neurons wired by an adjacency matrix."""

    def __init__(
        self,
        adjacency: np.ndarray,
        input_dim: int,
        *,
        memc_factory: Callable[..., nn.Module] | None = None,
        memc_params: Optional[dict] = None,
        input_scale: float = 1.0,
        random_seed: int = 0,
        device: torch.device | str | None = None,
    ):
        """Parameters
        ----------
        adjacency
            `(N, N)` numpy array of float weights.
        input_dim
            Dimensionality of external inputs.
        memc_factory, memc_params
            Factory and kwargs for creating each neuron.  Defaults to
            `Memcapacitor(c0=1.0, k=0.5, dt=1e-3)`.
        input_scale
            Std-dev for the random input projection matrix.
        device
            PyTorch device.
        """
        super().__init__()
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.W = nn.Parameter(torch.from_numpy(adjacency).to(torch.float32), requires_grad=False)
        self.n_nodes = self.W.shape[0]

        rng = torch.Generator(device=self.device).manual_seed(random_seed)
        self.Win = nn.Parameter(
            torch.randn(self.n_nodes, input_dim, generator=rng, device=self.device) * input_scale,
            requires_grad=False,
        )

        # Build neurons
        if memc_factory is None:
            memc_factory = Memcapacitor
        if memc_params is None:
            memc_params = {"c0": 1.0, "k": 0.5, "dt": 1e-3}
        self.neurons = nn.ModuleList([memc_factory(**memc_params) for _ in range(self.n_nodes)])
        self._charges: Optional[torch.Tensor] = None  # (batch, n_nodes)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def reset_state(self, batch_size: int = 1):
        """Reset internal charges to zero and neuron fluxes."""
        self._charges = torch.zeros(batch_size, self.n_nodes, device=self.device)
        for n in self.neurons:
            if hasattr(n, "reset"):
                n.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # shape (B, T, input_dim)
        B, T, _ = x.shape
        if B != 1:
            raise NotImplementedError("Current implementation supports batch_size=1 (GUI pipeline will wrap batching later)")
        if self._charges is None or self._charges.shape[0] != B:
            self.reset_state(B)

        states = []
        for t in range(T):
            v_raw = torch.matmul(self._charges, self.W) + torch.matmul(x[:, t, :], self.Win.T)
            # Limit voltage magnitude to avoid numeric overflow in simple device model
            v_in = torch.tanh(v_raw)
            new_charges = []
            # process each neuron independently (could be vectorised later)
            for i, neuron in enumerate(self.neurons):
                qi = neuron(v_in[0, i])  # scalar
                new_charges.append(qi)
            self._charges = torch.stack(new_charges).unsqueeze(0)  # (1, n_nodes)
            states.append(self._charges)
        # concatenate along time then add batch dim
        states_cat = torch.cat(states, dim=0).unsqueeze(0)  # (1, T, N)
        return states_cat
