from __future__ import annotations

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from .neural_pod_mode import MeanNet, NeuralPODMode, SpatialFourierNN, FourierPODMode


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


class FourierRegimeBasis(nn.Module):
    """FN-POD basis with Fourier spatial features and learnable temporal coefficients.

    Args:
        d_x:             spatial dimension
        M:               number of snapshots (size of temporal coefficients)
        quad_weights:    quadrature weights (Ny,)
        hidden_dim:      hidden layer width
        num_frequencies: number of Fourier frequencies
        scale:           scale of random Fourier features
    """

    def __init__(
        self,
        d_x: int,
        M: int,
        quad_weights: torch.Tensor,
        hidden_dim: int = 128,
        num_frequencies: int = 16,
        scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.M = M
        self.hidden_dim = hidden_dim
        self.num_frequencies = num_frequencies
        self.scale = scale

        self.mean_net = SpatialFourierNN(d_x, hidden_dim, num_frequencies, scale)
        self.modes = nn.ModuleList()
        self.register_buffer("quad_weights", quad_weights)

    @property
    def num_modes(self) -> int:
        return len(self.modes)

    def forward(self, x: torch.Tensor, t=None, kappa=None) -> torch.Tensor:
        """Full reconstruction: mean + sum of modes.

        Args:
            x: (Ny, d_x) spatial grid
            t, kappa: ignored (kept for interface compatibility)

        Returns:
            (N, Ny) reconstruction for all N snapshots
        """
        N = self.modes[0].lambda_ten.shape[0] if self.modes else 1
        out = self.mean_net(x).unsqueeze(0).expand(N, -1)  # (N, Ny)
        for mode in self.modes:
            phi = mode.phi(x)  # (Ny,)
            out = out + torch.outer(phi, mode.lambda_ten).T  # (N, Ny)
        return out

    def add_mode(self) -> FourierPODMode:
        mode = FourierPODMode(
            self.d_x, self.M, self.hidden_dim, self.num_frequencies, self.scale
        )
        mode = mode.to(self.quad_weights.device)
        self.modes.append(mode)
        return mode


def plot_space_time_heatmaps(basis: FourierRegimeBasis, s_true: torch.Tensor,
                              x: torch.Tensor, t: torch.Tensor) -> None:
    """Create space-time heatmap plots: truth, prediction, and error """
    import matplotlib.pyplot as plt
    import numpy as np


    basis.eval()
    with torch.no_grad():
        s_pred = basis(x, None, None)


    s_true_np = s_true.detach().cpu().numpy()
    s_pred_np = s_pred.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy().squeeze()
    t_np = t.detach().cpu().numpy()


    error_np = np.abs(s_true_np - s_pred_np)


    fig, axes = plt.subplots(1, 3, figsize=(16, 4))


    im0 = axes[0].imshow(s_true_np, aspect='auto', origin='lower', cmap='RdBu_r', extent=[x_np.min(), x_np.max(), t_np.min(), t_np.max()])
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    axes[0].set_title('Ground Truth u(x,t)')
    plt.colorbar(im0, ax=axes[0])


    im1 = axes[1].imshow(s_pred_np, aspect='auto', origin='lower', cmap='RdBu_r', extent=[x_np.min(), x_np.max(), t_np.min(), t_np.max()])
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    axes[1].set_title(r'FN-POD Prediction $\hat{u}$(x,t)')
    plt.colorbar(im1, ax=axes[1])


    im2 = axes[2].imshow(error_np, aspect='auto', origin='lower', cmap='hot', extent=[x_np.min(), x_np.max(), t_np.min(), t_np.max()])
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('t')
    axes[2].set_title(r'Absolute Error |u -$\hat{u}$|')
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle('FN-POD Reconstruction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig