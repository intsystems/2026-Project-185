from __future__ import annotations

from dataclasses import dataclass
from tqdm.auto import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_

from .regime_basis import FourierRegimeBasis
from .neural_pod_mode import FourierPODMode


def _weighted_norm_sq(r: Tensor, gamma: Tensor, w: Tensor) -> float:
    """sum_i gamma_i * sum_j w_j * r_ij^2"""
    return (gamma * (r ** 2 * w[None, :]).sum(dim=1)).sum().item()


@dataclass
class TrainHistory:
    mean_loss: list[float] = None
    mode_losses: list[list[float]] = None
    residual_norms: list[float] = None

    def __post_init__(self):
        if self.mean_loss is None:
            self.mean_loss = []
        if self.mode_losses is None:
            self.mode_losses = []
        if self.residual_norms is None:
            self.residual_norms = []


@dataclass
class FourierNeuralPODConfig:
    """Hyperparameters for Fourier basis mode extraction."""
    tol: float = 1e-3
    lr: float = 5e-4
    lr_lambda: float = 1e-2
    max_lr: float = 5e-3
    max_modes: int = 5
    n_epochs_mean: int = 165
    n_epochs_mode: int = 85
    batch_size: int = 32
    grad_clip_norm: float = 1.0
    pct_start: float = 0.1
    div_factor: float = 25.0
    log_every: int = 20
    full_batch: bool = False  # if True: skip DataLoader, run full-batch on GPU


class FourierNeuralPODTrainer:
    """Sequential mode extraction with Fourier spatial basis.

    Trains mean network, then extracts modes via Random Fourier Features.
    Each mode has temporal coefficients (learnable scalars).
    """

    def __init__(self, basis: FourierRegimeBasis, cfg: FourierNeuralPODConfig) -> None:
        self.basis = basis
        self.cfg = cfg
        self.history = TrainHistory()
        self._device = basis.quad_weights.device
        self.num_modes = 0

    def train(
        self,
        s: Tensor,
        x: Tensor,
        t: Tensor,
        kappa: Tensor = None,
        gamma: Tensor = None,
    ) -> TrainHistory:
        """Extract mean and modes from snapshot matrix.

        Args:
            s:     (N, Ny) snapshot matrix
            x:     (Ny, d_x) spatial grid
            t:     unused
            kappa: unused
            gamma: (N,) responsibility weights; uniform 1/N if None
        """
        N, Ny = s.shape
        device, dtype = s.device, s.dtype
        w = self.basis.quad_weights.to(device=device, dtype=dtype)  # (Ny,)

        if gamma is None:
            gamma = torch.ones(N, device=device, dtype=dtype) / N
        else:
            gamma = (gamma / gamma.sum()).to(device=device, dtype=dtype)

        print(f"Fourier NeuralPOD | N={N}, Ny={Ny} | max_modes={self.cfg.max_modes}")

        s_mean = (gamma[:, None] * s).sum(dim=0)                  # gamma-weighted mean (Ny,)
        s_centered = s - s_mean[None, :]
        self._tol_abs = self.cfg.tol * _weighted_norm_sq(s_centered, gamma, w)

        self._train_mean(s_mean, x, w)
        r = self._full_residual(s, x)

        self.num_modes = 0
        while (
            _weighted_norm_sq(r, gamma, w) >= self._tol_abs
            and len(self.basis.modes) < self.cfg.max_modes
        ):
            self.num_modes += 1
            mode = self.basis.add_mode()
            self._train_mode(mode, r, x, gamma, w)
            r = self._update_residual(r, mode, x)
            res = _weighted_norm_sq(r, gamma, w)
            self.history.residual_norms.append(res)
            print(f"  mode {self.num_modes}: weighted_res={res:.4e}")

        print(f"  done: {self.num_modes} modes\n")
        return self.history

    def _make_dataloader(self, *tensors) -> DataLoader:
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

    @torch.no_grad()
    def _full_residual(self, s: Tensor, x: Tensor) -> Tensor:
        return (s - self.basis.mean_net(x).unsqueeze(0)).detach()

    @torch.no_grad()
    def _update_residual(self, r: Tensor, mode: FourierPODMode, x: Tensor) -> Tensor:
        phi = mode.phi(x)
        return (r - torch.outer(phi, mode.lambda_ten).T).detach()

    def _train_mean(self, s_mean: Tensor, x: Tensor, w: Tensor) -> None:
        """Loss: sum_j w_j * (mean_net(x_j) - s_mean_j)^2"""
        opt = AdamW(self.basis.mean_net.parameters(), lr=self.cfg.lr, weight_decay=1e-2)
        pbar = tqdm(range(self.cfg.n_epochs_mean), desc="mean", leave=False)

        if self.cfg.full_batch:
            scheduler = OneCycleLR(
                opt, max_lr=self.cfg.max_lr,
                epochs=self.cfg.n_epochs_mean, steps_per_epoch=1,
                anneal_strategy="cos", pct_start=self.cfg.pct_start, div_factor=self.cfg.div_factor,
            )
            for epoch in pbar:
                opt.zero_grad()
                pred = self.basis.mean_net(x)
                loss = (w * (pred - s_mean) ** 2).sum()
                loss.backward()
                clip_grad_norm_(self.basis.mean_net.parameters(), self.cfg.grad_clip_norm)
                opt.step()
                scheduler.step()
                if epoch % self.cfg.log_every == 0:
                    self.history.mean_loss.append(loss.item())
                    pbar.set_postfix(loss=f"{loss.item():.3e}")
        else:
            dl = self._make_dataloader(x, s_mean, w)
            scheduler = OneCycleLR(
                opt, max_lr=self.cfg.max_lr,
                epochs=self.cfg.n_epochs_mean, steps_per_epoch=len(dl),
                anneal_strategy="cos", pct_start=self.cfg.pct_start, div_factor=self.cfg.div_factor,
            )
            for epoch in pbar:
                epoch_loss = 0.0
                for x_b, mean_b, w_b in dl:
                    opt.zero_grad()
                    pred = self.basis.mean_net(x_b)
                    loss = (w_b * (pred - mean_b) ** 2).sum()
                    loss.backward()
                    clip_grad_norm_(self.basis.mean_net.parameters(), self.cfg.grad_clip_norm)
                    opt.step()
                    scheduler.step()
                    epoch_loss += loss.item()
                if epoch % self.cfg.log_every == 0:
                    avg = epoch_loss / len(dl)
                    self.history.mean_loss.append(avg)
                    pbar.set_postfix(loss=f"{avg:.3e}")

    def _train_mode(
        self, mode: FourierPODMode, r: Tensor, x: Tensor, gamma: Tensor, w: Tensor
    ) -> None:
        """Loss: sum_i gamma_i * sum_j w_j * (phi(x_j) * lambda_i - r_ij)^2"""
        opt = AdamW([
            {"params": mode.phi.parameters(), "lr": self.cfg.lr, "weight_decay": 1e-2},
            {"params": [mode.lambda_ten], "lr": self.cfg.lr_lambda, "weight_decay": 0.0},
        ])
        p = len(self.history.mode_losses) + 1
        pbar = tqdm(range(self.cfg.n_epochs_mode), desc=f"mode {p}", leave=False)
        mode_history: list[float] = []

        if self.cfg.full_batch:
            scheduler = OneCycleLR(
                opt, max_lr=self.cfg.max_lr,
                epochs=self.cfg.n_epochs_mode, steps_per_epoch=1,
                anneal_strategy="cos", pct_start=self.cfg.pct_start, div_factor=self.cfg.div_factor,
            )
            for epoch in pbar:
                opt.zero_grad()
                phi  = mode.phi(x)                          # (Ny,)
                pred = torch.outer(phi, mode.lambda_ten)    # (Ny, N)
                loss = ((pred - r.T) ** 2 * gamma[None, :] * w[:, None]).sum()
                loss.backward()
                clip_grad_norm_(
                    list(mode.phi.parameters()) + [mode.lambda_ten], self.cfg.grad_clip_norm,
                )
                opt.step()
                scheduler.step()
                if epoch % self.cfg.log_every == 0:
                    mode_history.append(loss.item())
                    pbar.set_postfix(loss=f"{loss.item():.3e}")
        else:
            dl = self._make_dataloader(x, r.T.contiguous(), w)
            scheduler = OneCycleLR(
                opt, max_lr=self.cfg.max_lr,
                epochs=self.cfg.n_epochs_mode, steps_per_epoch=len(dl),
                anneal_strategy="cos", pct_start=self.cfg.pct_start, div_factor=self.cfg.div_factor,
            )
            for epoch in pbar:
                epoch_loss = 0.0
                for x_b, r_b, w_b in dl:
                    opt.zero_grad()
                    phi  = mode.phi(x_b)                          # (B,)
                    pred = torch.outer(phi, mode.lambda_ten)      # (B, N)
                    loss = ((pred - r_b) ** 2 * gamma[None, :] * w_b[:, None]).sum()
                    loss.backward()
                    clip_grad_norm_(
                        list(mode.phi.parameters()) + [mode.lambda_ten], self.cfg.grad_clip_norm,
                    )
                    opt.step()
                    scheduler.step()
                    epoch_loss += loss.item()
                if epoch % self.cfg.log_every == 0:
                    avg = epoch_loss / len(dl)
                    mode_history.append(avg)
                    pbar.set_postfix(loss=f"{avg:.3e}")

        self.history.mode_losses.append(mode_history)
