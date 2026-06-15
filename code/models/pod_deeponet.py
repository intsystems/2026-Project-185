from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .pod import PODBasis
from .neural_pod_mode import _mlp


@dataclass
class PODDeepONetConfig:
    """Hyperparameters for branch net training (Phase 2)."""
    lr: float = 3e-4
    n_epochs: int = 500
    batch_size: int = 64
    hidden_dim: int = 128
    n_layers: int = 4
    sensor_stride: int = 1
    log_every: int = 50
    val_every: int = 50


class BranchNet(nn.Module):
    """Maps (u0_sensors, kappa) to P mode coefficients."""

    def __init__(self, m: int, P: int, hidden_dim: int = 128, n_layers: int = 4,
                 d_kappa: int = 0) -> None:
        super().__init__()
        self.d_kappa = d_kappa
        self.net = _mlp(m + d_kappa, P, hidden_dim, n_layers, act=nn.Tanh)

    def forward(self, u0: Tensor, kappa: Tensor = None) -> Tensor:
        """Returns (N, P) mode coefficients."""
        inp = torch.cat([u0, kappa], dim=-1) if self.d_kappa > 0 else u0
        return self.net(inp)


class PODDeepONet(nn.Module):
    """POD-DeepONet: mean + branch(u0) @ Phi.T (Lu et al., 2022)."""

    def __init__(self, basis: PODBasis, branch: BranchNet) -> None:
        super().__init__()
        self.basis = basis
        self.branch = branch
        for p in basis.parameters():
            p.requires_grad_(False)

    @property
    def P(self) -> int:
        return self.basis.num_modes

    def forward(self, u0: Tensor, kappa: Tensor = None) -> Tensor:
        """Returns (N, Nt*Nx) predicted spatiotemporal field."""
        beta = self.branch(u0, kappa)                      # (N, P)
        return self.basis.mean + beta @ self.basis.modes.T # (N, Nt*Nx)


class PODDeepONetTrainer:
    """Trains the branch net (Phase 2) given a frozen POD basis from Phase 1."""

    def __init__(self, model: PODDeepONet, cfg: PODDeepONetConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.val_history: list[tuple] = []

    def train(self, u0: Tensor, val_u0: Tensor = None, val_s: Tensor = None) -> list[float]:
        """Train branch net on POD coefficient regression. Returns per-epoch train MSE."""
        basis = self.model.basis
        assert basis._initialized, "Run PODTrainer.train(s_traj) first."

        device = next(self.model.parameters()).device
        stride = self.cfg.sensor_stride

        u0_sensors = u0[:, ::stride].to(device)
        targets = basis.coeffs.to(device)

        N, m = u0_sensors.shape
        P = targets.shape[1]
        print(f"PODDeepONet Phase 2 | N={N}, m={m}, P={P}")

        self.val_history = []
        do_val = val_u0 is not None and val_s is not None
        if do_val:
            mean_d   = basis.mean.to(device)
            modes_d  = basis.modes.to(device)
            val_sens = val_u0[:, ::stride].to(device)
            val_tgt  = (val_s.to(device) - mean_d.unsqueeze(0)) @ modes_d

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
                self.model.branch.eval()
                with torch.no_grad():
                    vl = F.mse_loss(self.model.branch(val_sens), val_tgt).item()
                self.model.branch.train()
                self.val_history.append((epoch, vl))
            if epoch % self.cfg.log_every == 0:
                val_str = f"  val={self.val_history[-1][1]:.4e}" if self.val_history else ""
                print(f"  epoch {epoch:4d} | coeff_mse={avg:.4e}{val_str}")

        val_str = f"  val={self.val_history[-1][1]:.4e}" if self.val_history else ""
        print(f"  done: final coeff_mse={history[-1]:.4e}{val_str}")
        return history

    @torch.no_grad()
    def predict(self, u0_new: Tensor) -> Tensor:
        """Returns (N, Nt*Nx) predicted trajectories for new initial conditions."""
        device = next(self.model.parameters()).device
        u0_sensors = u0_new[:, ::self.cfg.sensor_stride].to(device)
        self.model.eval()
        return self.model(u0_sensors)
