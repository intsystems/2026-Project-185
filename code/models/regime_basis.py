from __future__ import annotations

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

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
        d_x: spatial dimension
        M: number of snapshots (temporal coefficient size)
        quad_weights: quadrature weights (Ny,)
        hidden_dim: network width
        num_frequencies: Fourier feature count
        scale: Fourier frequency scale
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
        """Reconstruct snapshots: mean plus all modes.

        Args:
            x: (Ny, d_x) spatial grid
            t: unused
            kappa: unused

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
            self.d_x, self.M, self.hidden_dim, self.num_frequencies, self.scale
        )
        mode = mode.to(self.quad_weights.device)
        self.modes.append(mode)
        return mode


def plot_space_time_heatmaps(obj, s_true: torch.Tensor,
                              x: torch.Tensor, t: torch.Tensor, title: str = "NeuralPOD") -> plt.Figure:
    """Plot truth, prediction, and error in space-time.

    Args:
        obj: basis, trainer, or predictor object
        s_true: (N, Ny) snapshot matrix
        x: (Ny, d_x) spatial grid
        t: (N,) time vector
        title: plot title

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get predictions based on object type
    with torch.no_grad():
        if hasattr(obj, 'predict'):  # NeuralPODTrainer
            s_pred = obj.predict(x, t)
        elif hasattr(obj, 'basis'):  # FourierNeuralPODTrainer
            obj.basis.eval()
            s_pred = obj.basis(x, t, None)
        else:  # Direct basis object (FourierRegimeBasis, RegimeBasis)
            obj.eval()
            # For RegimeBasis, need proper kappa. Create dummy ones.
            batch_size = len(t)
            d_kappa = 1
            kappa_dummy = torch.ones(batch_size, d_kappa, device=x.device)
            s_pred = obj(x, t, kappa_dummy)

    # Convert to numpy
    s_true_np = s_true.detach().cpu().numpy()  # (N, Ny)
    s_pred_np = s_pred.detach().cpu().numpy()  # (N, Ny)
    x_np = x.detach().cpu().numpy().squeeze()   # (Ny,)
    t_np = t.detach().cpu().numpy()             # (N,)

    # Compute error
    error_np = np.abs(s_true_np - s_pred_np)

    # Get color limits (same for truth and pred)
    vmax = np.abs(s_true_np).max()
    emax = error_np.max()

    # Create 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Ground truth
    im0 = axes[0].imshow(
        s_true_np,
        aspect='auto',
        origin='lower',
        cmap='RdBu_r',
        extent=[x_np.min(), x_np.max(), t_np.min(), t_np.max()],
        vmin=-vmax,
        vmax=vmax
    )
    axes[0].set_xlabel('x', fontsize=11)
    axes[0].set_ylabel('t', fontsize=11)
    axes[0].set_title('Ground Truth u(x,t)', fontsize=12, fontweight='bold')
    cbar0 = plt.colorbar(im0, ax=axes[0])
    cbar0.set_label('u', fontsize=10)

    # 2. Prediction
    im1 = axes[1].imshow(
        s_pred_np,
        aspect='auto',
        origin='lower',
        cmap='RdBu_r',
        extent=[x_np.min(), x_np.max(), t_np.min(), t_np.max()],
        vmin=-vmax,
        vmax=vmax
    )
    axes[1].set_xlabel('x', fontsize=11)
    axes[1].set_ylabel('t', fontsize=11)
    axes[1].set_title(f'{title} Prediction û(x,t)', fontsize=12, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=axes[1])
    cbar1.set_label('û', fontsize=10)

    # 3. Absolute error
    im2 = axes[2].imshow(
        error_np,
        aspect='auto',
        origin='lower',
        cmap='hot',
        extent=[x_np.min(), x_np.max(), t_np.min(), t_np.max()],
        vmin=0,
        vmax=emax
    )
    axes[2].set_xlabel('x', fontsize=11)
    axes[2].set_ylabel('t', fontsize=11)
    axes[2].set_title('Absolute Error |u - û|', fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=axes[2])
    cbar2.set_label('error', fontsize=10)

    # Compute and display metrics
    rel_err = np.linalg.norm(s_true_np - s_pred_np) / np.linalg.norm(s_true_np)
    mean_error = error_np.mean()

    fig.suptitle(
        f'{title} Reconstruction | Relative L2 Error: {rel_err:.4f} | Mean Error: {mean_error:.4f}',
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    return fig