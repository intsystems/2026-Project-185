from __future__ import annotations

import math
import torch
import torch.nn as nn


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int,
          act: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    """Fully connected network with uniform hidden width."""
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), act()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), act()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)




class MeanNet(nn.Module):
    """Parameter-dependent mean field: (x, kappa) -> scalar at each grid point."""

    def __init__(self, d_x: int, d_kappa: int, hidden_dim: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        self.net = _mlp(d_x + d_kappa, 1, hidden_dim, n_layers, act=nn.Tanh)

    def forward(self, x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Returns (N, Ny) mean values."""
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
    """K-rank separable mode: spatial basis combined with temporal/parameter coefficients."""

    def __init__(self, d_x: int, d_kappa: int, K: int = 8, hidden_dim: int = 64) -> None:
        super().__init__()
        self.param_net = _mlp(1 + d_kappa, K, hidden_dim, n_layers=2, act=nn.ReLU)
        self.basis_net = _mlp(d_x, K, hidden_dim, n_layers=2, act=nn.Tanh)

    def forward(self, x: torch.Tensor, t: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Returns (N, Ny) via K-rank outer product."""
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
    """Spatial basis via Random Fourier Features; multi-scale if `scales` is given."""

    def __init__(self, d_x: int = 1, hidden_dim: int = 128,
                 num_frequencies: int = 16, scale: float = 10.0,
                 scales: list[float] | None = None,
                 n_layers: int = 2) -> None:
        super().__init__()
        if scales is not None:
            # last scale gets remainder so total sums to num_frequencies
            k = num_frequencies // len(scales)
            Bs = [torch.randn(k, d_x) * s for s in scales[:-1]]
            Bs.append(torch.randn(num_frequencies - k * (len(scales) - 1), d_x) * scales[-1])
            B = torch.cat(Bs, dim=0)
        else:
            B = torch.randn(num_frequencies, d_x) * scale
        self.register_buffer("B", B, persistent=False)

        layers: list[nn.Module] = [nn.Linear(2 * num_frequencies, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (Ny,) basis values."""
        x_proj = 2 * math.pi * x @ self.B.to(x.device).T
        feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.net(feats).squeeze(-1)


class SpatialPODNet(nn.Module):
    """Spatial basis network: coordinates to scalar basis values."""

    def __init__(self, d_x: int, hidden_dim: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        self.net = _mlp(d_x, 1, hidden_dim, n_layers, act=nn.Tanh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (Ny,) basis values."""
        return self.net(x).squeeze(-1)


class TemporalPODNet(nn.Module):
    """Temporal basis network: time to scalar coefficient values."""

    def __init__(self, hidden_dim: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        self.net = _mlp(1, 1, hidden_dim, n_layers, act=nn.Tanh)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Returns (N,) temporal coefficients."""
        t_inp = t.unsqueeze(-1)
        return self.net(t_inp).squeeze(-1)


class ClassicalPODMode(nn.Module):
    """Separable mode: Mode(x, t) = Phi(x) * Psi(t), parameter-independent."""

    def __init__(self, d_x: int, hidden_dim: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        self.spatial_net = SpatialPODNet(d_x, hidden_dim, n_layers)
        self.temporal_net = TemporalPODNet(hidden_dim, n_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns (N, Ny) via outer product of spatial and temporal bases."""
        phi = self.spatial_net(x)
        psi = self.temporal_net(t)
        return torch.outer(psi, phi)


class FourierPODMode(nn.Module):
    """Fourier spatial basis with per-trajectory learnable amplitude coefficients."""

    def __init__(self, d_x: int, M: int, hidden_dim: int = 128,
                 num_frequencies: int = 16, scale: float = 10.0,
                 scales: list[float] | None = None,
                 n_layers: int = 2) -> None:
        super().__init__()
        self.phi = SpatialFourierNN(d_x, hidden_dim, num_frequencies, scale, scales, n_layers)
        self.lambda_ten = nn.Parameter(torch.ones(M))

    def forward(self, x: torch.Tensor, t=None, kappa=None) -> torch.Tensor:

        return torch.outer(self.phi(x), self.lambda_ten)