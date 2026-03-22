from __future__ import annotations

from dataclasses import dataclass
from tqdm.auto import tqdm

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_

from .regime_basis import FourierRegimeBasis
from .neural_pod_mode import FourierPODMode


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
            s: (N, Ny) snapshot matrix
            x: (Ny, d_x) spatial grid
            t: (N,) time vector
            kappa: unused
            gamma: unused

        Returns:
            TrainHistory with losses and residual norms
        """
        self._tol_abs = self.cfg.tol * s.pow(2).mean().item()
        s_mean = s.mean(dim=0)

        self._train_mean(s_mean, x)
        r = self._full_residual(s, x)

        while (
            r.pow(2).mean().item() >= self._tol_abs
            and len(self.basis.modes) < self.cfg.max_modes
        ):
            mode = self.basis.add_mode()
            self._train_mode(mode, r, x)
            r = self._update_residual(r, mode, x)
            self.history.residual_norms.append(r.pow(2).mean().item())

        return self.history

    def _make_dataloader(self, x: Tensor, target: Tensor) -> DataLoader:
        return DataLoader(
            TensorDataset(x, target),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

    @torch.no_grad()
    def _full_residual(self, s: Tensor, x: Tensor) -> Tensor:
        mean_pred = self.basis.mean_net(x)
        return (s - mean_pred.unsqueeze(0)).detach()

    @torch.no_grad()
    def _update_residual(self, r: Tensor, mode: FourierPODMode, x: Tensor) -> Tensor:
        phi = mode.phi(x)
        pred = torch.outer(phi, mode.lambda_ten).T
        return (r - pred).detach()

    def _train_mean(self, s_mean: Tensor, x: Tensor) -> None:
        dl = self._make_dataloader(x, s_mean)
        opt = AdamW(self.basis.mean_net.parameters(), lr=self.cfg.lr, weight_decay=1e-2)
        scheduler = OneCycleLR(
            opt,
            max_lr=self.cfg.max_lr,
            epochs=self.cfg.n_epochs_mean,
            steps_per_epoch=len(dl),
            anneal_strategy="cos",
            pct_start=self.cfg.pct_start,
            div_factor=self.cfg.div_factor,
        )
        pbar = tqdm(range(self.cfg.n_epochs_mean), desc="mean", leave=False)
        for epoch in pbar:
            epoch_loss = 0.0
            for x_b, mean_b in dl:
                opt.zero_grad()
                pred = self.basis.mean_net(x_b)
                loss = F.mse_loss(pred, mean_b)
                loss.backward()
                clip_grad_norm_(
                    self.basis.mean_net.parameters(), self.cfg.grad_clip_norm
                )
                opt.step()
                scheduler.step()
                epoch_loss += loss.item()
            if epoch % self.cfg.log_every == 0:
                avg = epoch_loss / len(dl)
                self.history.mean_loss.append(avg)
                pbar.set_postfix(loss=f"{avg:.3e}")

    def _train_mode(self, mode: FourierPODMode, r: Tensor, x: Tensor) -> None:
        dl = self._make_dataloader(x, r.T.contiguous())
        opt = AdamW(
            [
                {
                    "params": mode.phi.parameters(),
                    "lr": self.cfg.lr,
                    "weight_decay": 1e-2,
                },
                {
                    "params": [mode.lambda_ten],
                    "lr": self.cfg.lr_lambda,
                    "weight_decay": 0.0,
                },
            ]
        )
        scheduler = OneCycleLR(
            opt,
            max_lr=self.cfg.max_lr,
            epochs=self.cfg.n_epochs_mode,
            steps_per_epoch=len(dl),
            anneal_strategy="cos",
            pct_start=self.cfg.pct_start,
            div_factor=self.cfg.div_factor,
        )
        p = len(self.history.mode_losses) + 1
        pbar = tqdm(range(self.cfg.n_epochs_mode), desc=f"mode {p}", leave=False)
        mode_history: list[float] = []
        for epoch in pbar:
            epoch_loss = 0.0
            for x_b, r_b in dl:
                opt.zero_grad()
                phi = mode.phi(x_b)
                pred = torch.outer(phi, mode.lambda_ten)
                loss = F.mse_loss(pred, r_b)
                loss.backward()
                clip_grad_norm_(
                    list(mode.phi.parameters()) + [mode.lambda_ten],
                    self.cfg.grad_clip_norm,
                )
                opt.step()
                scheduler.step()
                epoch_loss += loss.item()
            if epoch % self.cfg.log_every == 0:
                avg = epoch_loss / len(dl)
                mode_history.append(avg)
                pbar.set_postfix(loss=f"{avg:.3e}")
        self.history.mode_losses.append(mode_history)
