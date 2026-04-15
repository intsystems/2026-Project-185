from __future__ import annotations

import torch
import torch.nn as nn

from .neural_pod_mode import MeanNet, NeuralPODMode, SpatialFourierNN, FourierPODMode


class RegimeBasis(nn.Module):
    """Parametric basis: mean network + K-rank mode.

    Args:
        d_x: spatial coordinate dimension
        d_kappa: parameter dimension
        quad_weights: quadrature weights (Ny,)
        K: rank of factorization
        hidden_dim: network width
    """

    def __init__(
        self,
        d_x: int,
        d_kappa: int,
        quad_weights: torch.Tensor,
        K: int = 8,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_kappa = d_kappa
        self.K = K
        self.hidden_dim = hidden_dim

        self.mean_net = MeanNet(d_x, d_kappa, hidden_dim, n_layers=2)
        self.mode = NeuralPODMode(d_x, d_kappa, K=K, hidden_dim=hidden_dim)

        self.register_buffer("quad_weights", quad_weights)

    @property
    def num_modes(self) -> int:
        return self.K

    def forward(self, x: torch.Tensor, t: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        return self.mean_net(x, kappa) + self.mode(x, t, kappa)

    def _weighted_norm_sq(self, f: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return (gamma * (f ** 2 * self.quad_weights).sum(dim=-1)).sum()


class FourierRegimeBasis(nn.Module):
    """Sequential Fourier basis with learnable temporal coefficients.

    Args:
        d_x:             spatial dimension
        M:               number of snapshots (temporal coefficient size)
        quad_weights:    quadrature weights (Ny,)
        hidden_dim:      network width
        num_frequencies: total Fourier feature count
        scale:           single frequency scale (used when scales is None)
        scales:          multi-scale list, e.g. [1.0, 3.0, 8.0]; frequencies
                         are split evenly across scales for richer coverage
        n_layers:        depth of each phi MLP (default 2 hidden layers)
    """

    def __init__(
        self,
        d_x: int,
        M: int,
        quad_weights: torch.Tensor,
        hidden_dim: int = 128,
        num_frequencies: int = 16,
        scale: float = 10.0,
        scales: list[float] | None = None,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.M = M
        self.hidden_dim = hidden_dim
        self.num_frequencies = num_frequencies
        self.scale = scale
        self.scales = scales
        self.n_layers = n_layers

        self.mean_net = SpatialFourierNN(d_x, hidden_dim, num_frequencies, scale, scales, n_layers)
        self.modes = nn.ModuleList()
        self.register_buffer("quad_weights", quad_weights)

    @property
    def num_modes(self) -> int:
        return len(self.modes)

    def forward(self, x: torch.Tensor, t=None, kappa=None) -> torch.Tensor:
        """Reconstruct snapshots: mean plus all modes.

        Args:
            x: (Ny, d_x) spatial grid
        Returns:
            (N, Ny) full reconstruction
        """
        N = self.modes[0].lambda_ten.shape[0] if self.modes else 1
        out = self.mean_net(x).unsqueeze(0).expand(N, -1)  # (N, Ny)
        for mode in self.modes:
            phi = mode.phi(x)  # (Ny,)
            out = out + torch.outer(phi, mode.lambda_ten).T  # (N, Ny)
        return out

    def add_mode(self) -> FourierPODMode:
        mode = FourierPODMode(
            self.d_x, self.M, self.hidden_dim, self.num_frequencies,
            self.scale, self.scales, self.n_layers,
        )
        mode = mode.to(self.quad_weights.device)
        self.modes.append(mode)
        return mode
