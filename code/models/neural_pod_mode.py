from __future__ import annotations

import math
import torch
import torch.nn as nn


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int,
          act: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    """Build fully connected network with specified depth and width.

    Args:
        in_dim: input dimension
        out_dim: output dimension
        hidden_dim: hidden layer width
        n_layers: total layers (input + hidden + output)
        act: activation function

    Returns:
        Sequential model
    """
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), act()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), act()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)




class MeanNet(nn.Module):
    """Mean field parameterized by spatial coordinates and parameters.

    Predicts scalar mean at each spatial point and parameter value.
    """

    def __init__(self, d_x: int, d_kappa: int, hidden_dim: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        self.net = _mlp(d_x + d_kappa, 1, hidden_dim, n_layers, act=nn.Tanh)

    def forward(self, x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Forward pass computing parameter-dependent mean.

        Args:
            x: (Ny, d_x) spatial grid points
            kappa: (N, d_kappa) parameter values

        Returns:
            (N, Ny) mean values
        """
        N, Ny = kappa.shape[0], x.shape[0]
        assert x.dim() == 2, f"x must be 2D, got {x.shape}"
        assert kappa.dim() == 2, f"kappa must be 2D, got {kappa.shape}"

        x_exp = x.unsqueeze(0).expand(N, -1, -1)
        k_exp = kappa.unsqueeze(1).expand(-1, Ny, -1)
        inp = torch.cat([x_exp, k_exp], dim=-1)
        output = self.net(inp.reshape(N * Ny, -1)).reshape(N, Ny)

        assert output.shape == (N, Ny), f"Expected output shape ({N}, {Ny}), got {output.shape}"
        return output


class NeuralPODMode(nn.Module):
    """K-rank mode: spatial and temporal-parameter networks.

    Combines parameter-independent spatial basis with temporal/parameter
    coefficients via K-dimensional factorization.
    """

    def __init__(self, d_x: int, d_kappa: int, K: int = 8, hidden_dim: int = 64) -> None:
        super().__init__()
        self.param_net = _mlp(1 + d_kappa, K, hidden_dim, n_layers=2, act=nn.ReLU)
        self.basis_net = _mlp(d_x, K, hidden_dim, n_layers=2, act=nn.Tanh)

    def forward(self, x: torch.Tensor, t: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Forward pass computing K-rank mode.

        Args:
            x: (Ny, d_x) spatial grid points
            t: (N,) time values
            kappa: (N, d_kappa) parameter values

        Returns:
            (N, Ny) mode predictions via K-rank decomposition
        """
        N, Ny = kappa.shape[0], x.shape[0]
        assert x.dim() == 2, f"x must be 2D, got {x.shape}"
        assert t.dim() == 1, f"t must be 1D, got {t.shape}"
        assert kappa.dim() == 2, f"kappa must be 2D, got {kappa.shape}"

        t_inp = torch.cat([t.unsqueeze(-1), kappa], dim=-1)
        param_out = self.param_net(t_inp)

        basis_out = self.basis_net(x).unsqueeze(0).expand(N, -1, -1)

        output = (param_out.unsqueeze(1) * basis_out).sum(dim=-1)
        assert output.shape == (N, Ny), f"Expected output shape ({N}, {Ny}), got {output.shape}"
        return output


class SpatialFourierNN(nn.Module):
    """Spatial basis via Random Fourier Features.

    Encodes coordinates through sinusoids with fixed random frequencies.
    """

    def __init__(self, d_x: int = 1, hidden_dim: int = 128,
                 num_frequencies: int = 16, scale: float = 10.0) -> None:
        super().__init__()
        self.register_buffer(
            "B",
            torch.randn(num_frequencies, d_x) * scale,
            persistent=False
        )
        self.net = nn.Sequential(
            nn.Linear(2 * num_frequencies, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Fourier feature encoding.

        Args:
            x: (Ny, d_x) spatial grid points

        Returns:
            (Ny,) spatial basis values
        """
        x_proj = 2 * math.pi * x @ self.B.to(x.device).T
        feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.net(feats).squeeze(-1)


class SpatialPODNet(nn.Module):
    """Spatial basis network: coordinates to scalar basis values."""

    def __init__(self, d_x: int, hidden_dim: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        self.net = _mlp(d_x, 1, hidden_dim, n_layers, act=nn.Tanh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (Ny, d_x) spatial grid points

        Returns:
            (Ny,) spatial basis values
        """
        return self.net(x).squeeze(-1)


class TemporalPODNet(nn.Module):
    """Temporal basis network: time to scalar coefficient values."""

    def __init__(self, hidden_dim: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        self.net = _mlp(1, 1, hidden_dim, n_layers, act=nn.Tanh)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            t: (N,) time values

        Returns:
            (N,) temporal coefficient values
        """
        t_inp = t.unsqueeze(-1)
        return self.net(t_inp).squeeze(-1)


class ClassicalPODMode(nn.Module):
    """Separable mode: outer product of spatial and temporal bases.

    Mode(x, t) = Phi(x) * Psi(t), parameter-independent.
    """

    def __init__(self, d_x: int, hidden_dim: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        self.spatial_net = SpatialPODNet(d_x, hidden_dim, n_layers)
        self.temporal_net = TemporalPODNet(hidden_dim, n_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass computing separable mode.

        Args:
            x: (Ny, d_x) spatial grid points
            t: (N,) time values

        Returns:
            (N, Ny) mode values via outer product
        """
        phi = self.spatial_net(x)
        psi = self.temporal_net(t)
        return torch.outer(psi, phi)


class FourierPODMode(nn.Module):
    """Fourier-based mode with learnable temporal coefficients."""

    def __init__(self, d_x: int, M: int, hidden_dim: int = 128,
                 num_frequencies: int = 16, scale: float = 10.0) -> None:
        super().__init__()
        self.phi = SpatialFourierNN(d_x, hidden_dim, num_frequencies, scale)
        self.lambda_ten = nn.Parameter(torch.ones(M))

    def forward(self, x: torch.Tensor, t=None, kappa=None) -> torch.Tensor:

        return torch.outer(self.phi(x), self.lambda_ten)