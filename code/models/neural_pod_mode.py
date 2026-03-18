from __future__ import annotations

import math
import torch
import torch.nn as nn


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int) -> nn.Sequential:

    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class SpatialNet(nn.Module):
    """Phi_p(x; kappa): concatenates [x, kappa] -> MLP -> scal"""

    def __init__(self, d_x: int, d_kappa: int, hidden_dim: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        self.net = _mlp(d_x + d_kappa, 1, hidden_dim, n_layers)

    def forward(self, x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        N, Ny = kappa.shape[0], x.shape[0]
        x_exp = x.unsqueeze(0).expand(N, -1, -1)       # (N, Ny, d_x)
        k_exp = kappa.unsqueeze(1).expand(-1, Ny, -1)   # (N, Ny, d_kappa)
        inp = torch.cat([x_exp, k_exp], dim=-1)          # (N, Ny, d_x + d_kappa)
        return self.net(inp.reshape(N * Ny, -1)).reshape(N, Ny)


class TemporalNet(nn.Module):
    """Psi_p(t; kappa): concatenates [t, kappa] -> MLP -> scal """

    def __init__(self, d_kappa: int, hidden_dim: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        self.net = _mlp(1 + d_kappa, 1, hidden_dim, n_layers)

    def forward(self, t: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:

        inp = torch.cat([t.unsqueeze(-1), kappa], dim=-1)  # (N, 1 + d_kappa)
        return self.net(inp).squeeze(-1)                    # (N,)


class MeanNet(nn.Module):
    """Parameter-dependent mean Phi_0(x; kappa) """

    def __init__(self, d_x: int, d_kappa: int, hidden_dim: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        self.phi = SpatialNet(d_x, d_kappa, hidden_dim, n_layers)

    def forward(self, x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (Ny, d_x)
            kappa: (N, d_kappa)

        Returns:
            (N, Ny)
        """
        return self.phi(x, kappa)


class NeuralPODMode(nn.Module):
    """One separable mode: Phi_p(x; kappa) * Psi_p(t; kappa) """

    def __init__(self, d_x: int, d_kappa: int, hidden_dim: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        self.phi = SpatialNet(d_x, d_kappa, hidden_dim, n_layers)
        self.psi = TemporalNet(d_kappa, hidden_dim, n_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        phi_vals = self.phi(x, kappa)              # (N, Ny)
        psi_vals = self.psi(t, kappa)              # (N,)
        return phi_vals * psi_vals.unsqueeze(-1)    # (N, Ny)


class SpatialFourierNN(nn.Module):
    """Spatial basis: Random Fourier Features -> MLP -> scalar"""

    def __init__(self, d_x: int = 1, hidden_dim: int = 128,
                 num_frequencies: int = 16, scale: float = 10.0) -> None:
        super().__init__()
        self.B = nn.Parameter(
            torch.randn(num_frequencies, d_x) * scale, requires_grad=False
        )
        self.net = nn.Sequential(
            nn.Linear(2 * num_frequencies, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * math.pi * x @ self.B.T
        feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.net(feats).squeeze(-1)


class FourierPODMode(nn.Module):
    """FN-POD mode: spatial basis phi(x) with temporal coefficients lambda (M,)"""

    def __init__(self, d_x: int, M: int, hidden_dim: int = 128,
                 num_frequencies: int = 16, scale: float = 10.0) -> None:
        super().__init__()
        self.phi = SpatialFourierNN(d_x, hidden_dim, num_frequencies, scale)
        self.lambda_ten = nn.Parameter(torch.ones(M))

    def forward(self, x: torch.Tensor, t=None, kappa=None) -> torch.Tensor:
        return torch.outer(self.phi(x), self.lambda_ten)