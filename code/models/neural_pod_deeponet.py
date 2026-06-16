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
    val_every: int = 50


class NeuralPODDeepONet(nn.Module):
    """NeuralPOD-DeepONet: mean_net(x,t) + branch(u0) @ Phi(x,t).T with K frozen Fourier modes."""

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
        """Returns (N, Nt*Nx) predicted field. u0: (N, m), x_flat: (Nt*Nx, 2)."""
        beta = self.branch(u0)  # (N, K)
        phi = torch.stack([m.phi(x_flat) for m in self.basis.modes], dim=1)  # (Nt*Nx, K)
        mean = self.basis.mean_net(x_flat)  # (Nt*Nx,)
        return mean.unsqueeze(0) + beta @ phi.T  # (N, Nt*Nx)


class NeuralPODDeepONetTrainer:
    """Phase 2 trainer: branch net u0 -> lambda coefficients given frozen Fourier basis from Phase 1."""

    def __init__(self, model: NeuralPODDeepONet, cfg: NeuralPODDeepONetConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.val_history: list[tuple] = []

    def train(self, u0: Tensor, val_u0: Tensor = None, val_s: Tensor = None,
              x_flat: Tensor = None) -> list[float]:
        """Train branch net on lambda coefficient regression. Returns per-epoch train MSE."""
        basis = self.model.basis
        K = len(basis.modes)
        assert K > 0, "Run FourierNeuralPODTrainer.train(s_traj, x_flat) first."

        coeffs = torch.stack([m.lambda_ten.detach() for m in basis.modes], dim=1)

        device = next(self.model.parameters()).device
        stride = self.cfg.sensor_stride

        u0_sensors = u0[:, ::stride].to(device)
        targets = coeffs.to(device)

        N, m = u0_sensors.shape
        print(f"NeuralPOD-DeepONet Phase 2 | N={N}, m={m}, K={K}")

        self.val_history = []
        do_val = val_u0 is not None and val_s is not None and x_flat is not None
        if do_val:
            val_sens  = val_u0[:, ::stride].to(device)
            val_s_dev = val_s.to(device)
            x_flat    = x_flat.to(device)

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
            if do_val and epoch % self.cfg.val_every == 0:
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(val_sens, x_flat)
                    rel_l2 = ((pred - val_s_dev).norm(dim=1) /
                              val_s_dev.norm(dim=1).clamp_min(1e-8)).mean().item()
                self.model.train()
                self.val_history.append((epoch, rel_l2))
            if epoch % self.cfg.log_every == 0:
                val_str = f"  val_rl2={self.val_history[-1][1]:.4e}" if self.val_history else ""
                print(f"  epoch {epoch:4d} | coeff_mse={avg:.4e}{val_str}")

        val_str = f"  val_rl2={self.val_history[-1][1]:.4e}" if self.val_history else ""
        print(f"  done: final coeff_mse={history[-1]:.4e}{val_str}")
        return history

    @torch.no_grad()
    def predict(self, u0_new: Tensor, x_flat: Tensor) -> Tensor:
        """Returns (N, Nt*Nx) predicted trajectories for new initial conditions."""
        device = next(self.model.parameters()).device
        u0_sensors = u0_new[:, ::self.cfg.sensor_stride].to(device)
        x_flat = x_flat.to(device)
        self.model.eval()
        return self.model(u0_sensors, x_flat)
