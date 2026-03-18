from __future__ import annotations

import torch
import torch.nn as nn

from .neural_pod_mode import MeanNet, NeuralPODMode


class RegimeBasis(nn.Module):
    """NeuralPOD basis for one regime

    Args:
        d_x:          spatial coordinate dimension
        d_kappa:      physical parameter dimension
        quad_weights: quadrature weights, shape (Ny,)
        hidden_dim:   hidden layer width
        n_layers:     number of hidden layers
    """

    def __init__(
        self,
        d_x: int,
        d_kappa: int,
        quad_weights: torch.Tensor,
        hidden_dim: int = 64,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_kappa = d_kappa
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.mean_net = MeanNet(d_x, d_kappa, hidden_dim, n_layers)
        self.modes = nn.ModuleList()

        self.register_buffer("quad_weights", quad_weights)  # (Ny,)

    @property
    def num_modes(self) -> int:
        """Number of trained modes P_m (no mean)"""
        return len(self.modes)

    def forward(self, x: torch.Tensor, t: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        out = self.mean_net(x, kappa)
        for mode in self.modes:
            out = out + mode(x, t, kappa)
        return out


    def _weighted_norm_sq(self, f: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return (gamma * (f ** 2 * self.quad_weights).sum(dim=-1)).sum()

    def add_mode(self) -> NeuralPODMode:
        mode = NeuralPODMode(self.d_x, self.d_kappa, self.hidden_dim, self.n_layers)
        mode = mode.to(self.quad_weights.device)
        self.modes.append(mode)
        return mode

    def prune_last_mode(self) -> None:
        self.modes = nn.ModuleList(list(self.modes)[:-1])