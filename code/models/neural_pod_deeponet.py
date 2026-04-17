from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .regime_basis import FourierRegimeBasis
from .pod_deeponet import BranchNet


@dataclass
class NeuralPODDeepONetConfig:
    """Hyperparameters for branch net training (Phase 2)."""
    lr: float = 3e-4
    n_epochs: int = 500
    batch_size: int = 64
    hidden_dim: int = 128
    n_layers: int = 4
    sensor_stride: int = 1
    log_every: int = 50


class NeuralPODDeepONet(nn.Module):
    """NeuralPOD-DeepONet for time-dependent PDEs.

    Prediction: mean_net(x,t) + branch(u_0) @ Phi(x,t).T
    where Phi are K spatiotemporal Fourier mode networks from Phase 1.

    Args:
        basis:  trained FourierRegimeBasis with d_x=2, M=N_traj
        branch: BranchNet mapping u_0 sensors to K mode coefficients
    """

    def __init__(self, basis: FourierRegimeBasis, branch: BranchNet) -> None:
        super().__init__()
        self.basis = basis
        self.branch = branch
        for p in basis.parameters():
            p.requires_grad_(False)

    @property
    def K(self) -> int:
        return len(self.basis.modes)

    def forward(self, u0: Tensor, x_flat: Tensor) -> Tensor:
        """
        Args:
            u0:     (N, m)       — initial conditions at sensor points
            x_flat: (Nt*Nx, 2)  — (x, t) coordinate pairs

        Returns:
            (N, Nt*Nx) — predicted spatiotemporal field; reshape to (N, Nt, Nx)
        """
        beta = self.branch(u0)  # (N, K)
        phi = torch.stack([m.phi(x_flat) for m in self.basis.modes], dim=1)  # (Nt*Nx, K)
        mean = self.basis.mean_net(x_flat)  # (Nt*Nx,)
        return mean.unsqueeze(0) + beta @ phi.T  # (N, Nt*Nx)


class NeuralPODDeepONetTrainer:
    """Two-stage trainer for NeuralPOD-DeepONet.

    Phase 1 (external): FourierNeuralPODTrainer.train(s_traj, x_flat)
        s_traj: (N_traj, Nt*Nx) — full spatiotemporal trajectories, flattened
        x_flat: (Nt*Nx, 2)      — (x, t) coordinate pairs
        Produces mean_net and modes[k].phi (frozen after Phase 1),
        and modes[k].lambda_ten (N_traj,) — one coefficient per trajectory.

    Phase 2 (this class): train branch net u_0 -> lambda coefficients.
    """

    def __init__(self, model: NeuralPODDeepONet, cfg: NeuralPODDeepONetConfig) -> None:
        self.model = model
        self.cfg = cfg

    def train(self, u0: Tensor) -> list[float]:
        """Train branch net on lambda coefficient regression.

        Args:
            u0: (N_traj, Nx) — initial conditions at full spatial resolution

        Returns:
            per-epoch MSE losses on lambda coefficients
        """
        basis = self.model.basis
        K = len(basis.modes)
        assert K > 0, "Run FourierNeuralPODTrainer.train(s_traj, x_flat) first."

        # stack lambda_ten from each mode into (N_traj, K)
        coeffs = torch.stack([m.lambda_ten.detach() for m in basis.modes], dim=1)

        device = next(self.model.parameters()).device
        stride = self.cfg.sensor_stride

        u0_sensors = u0[:, ::stride].to(device)
        targets = coeffs.to(device)

        N, m = u0_sensors.shape
        print(f"NeuralPOD-DeepONet Phase 2 | N={N}, m={m}, K={K}")

        dl = DataLoader(TensorDataset(u0_sensors, targets),
                        batch_size=self.cfg.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.branch.parameters(),
                                lr=self.cfg.lr, weight_decay=1e-4)

        history: list[float] = []
        for epoch in range(self.cfg.n_epochs):
            self.model.branch.train()
            total = 0.0
            for u0_b, coeff_b in dl:
                opt.zero_grad()
                loss = F.mse_loss(self.model.branch(u0_b), coeff_b)
                loss.backward()
                opt.step()
                total += loss.item()

            avg = total / len(dl)
            history.append(avg)
            if epoch % self.cfg.log_every == 0:
                print(f"  epoch {epoch:4d} | coeff_mse={avg:.4e}")

        print(f"  done: final coeff_mse={history[-1]:.4e}")
        return history

    @torch.no_grad()
    def predict(self, u0_new: Tensor, x_flat: Tensor) -> Tensor:
        """Predict full spatiotemporal trajectories for new initial conditions.

        Args:
            u0_new: (N, Nx)     — initial conditions at full spatial resolution
            x_flat: (Nt*Nx, 2) — (x, t) coordinate pairs

        Returns:
            (N, Nt*Nx) — reshape to (N, Nt, Nx) for visualization
        """
        device = next(self.model.parameters()).device
        u0_sensors = u0_new[:, ::self.cfg.sensor_stride].to(device)
        x_flat = x_flat.to(device)
        self.model.eval()
        return self.model(u0_sensors, x_flat)
