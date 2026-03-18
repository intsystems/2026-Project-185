from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from tqdm.auto import tqdm

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_

from .regime_basis import RegimeBasis, FourierRegimeBasis
from .neural_pod_mode import NeuralPODMode, FourierPODMode


@dataclass
class NeuralPODConfig:
    tol:            float = 1e-3
    lr_phi:         float = 2e-3
    lr_psi:         float = 1e-2
    max_lr_phi:     float = 5e-3
    max_lr_psi:     float = 1e-1
    max_modes:      int   = 4
    n_epochs_mean:  int   = 2000
    n_epochs_mode:  int   = 5000
    batch_size:     int   = 32
    grad_clip_norm: float = 1.0
    pct_start:      float = 0.1
    log_every:      int   = 20
    lambda_bc:      float = 0.0


@dataclass
class TrainHistory:
    mean_loss:      list[float]       = field(default_factory=list)
    mode_losses:    list[list[float]] = field(default_factory=list)
    residual_norms: list[float]       = field(default_factory=list)


class WeightedNeuralPODTrainer:

    def __init__(self, basis: RegimeBasis, cfg: NeuralPODConfig) -> None:
        self.basis = basis
        self.cfg = cfg
        self.history = TrainHistory()
        self._device = basis.quad_weights.device

    def train(
        self,
        s: Tensor,
        x: Tensor,
        t: Tensor,
        kappa: Tensor,
        gamma: Tensor,
    ) -> TrainHistory:
        self._tol_abs = self.cfg.tol * torch.mean(s ** 2).item()

        self._train_mean(s, x, kappa)
        r = self._full_residual(s, x, kappa)

        while (
            torch.mean(r ** 2).item() >= self._tol_abs
            and len(self.basis.modes) < self.cfg.max_modes
        ):
            mode = self.basis.add_mode()
            self._train_mode(mode, r, x, t, kappa)
            r = self._update_residual(r, mode, x, t, kappa)
            self.history.residual_norms.append(torch.mean(r ** 2).item())

        return self.history

    def _make_dataloader(self, *tensors: Tensor) -> DataLoader:
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=self.cfg.batch_size,
            shuffle=True
        )

    @torch.no_grad()
    def _full_forward(self, fn, x: Tensor, *tensors: Tensor, chunk: int = 256) -> Tensor:
        N = tensors[0].shape[0]
        return torch.cat(
            [fn(x, *[t[i:i + chunk] for t in tensors]) for i in range(0, N, chunk)],
            dim=0,
        )

    def _full_residual(self, s: Tensor, x: Tensor, kappa: Tensor) -> Tensor:
        return (s - self._full_forward(self.basis.mean_net, x, kappa)).detach()

    def _update_residual(self, r: Tensor, mode: NeuralPODMode, x: Tensor, t: Tensor, kappa: Tensor) -> Tensor:
        return (r - self._full_forward(mode, x, t, kappa)).detach()

    def _train_mean(self, s: Tensor, x: Tensor, kappa: Tensor) -> None:
        dl = self._make_dataloader(s, kappa)
        opt = AdamW(self.basis.mean_net.parameters(), lr=self.cfg.lr_phi, weight_decay=1e-2)
        scheduler = OneCycleLR(
            opt,
            max_lr=self.cfg.max_lr_phi,
            epochs=self.cfg.n_epochs_mean,
            steps_per_epoch=len(dl),
            pct_start=self.cfg.pct_start,
            anneal_strategy='cos',
            div_factor=25.0,
        )
        pbar = tqdm(range(self.cfg.n_epochs_mean), desc="mean", leave=False)

        for epoch in pbar:
            epoch_loss = 0.0
            for s_b, kappa_b in dl:
                opt.zero_grad()
                pred = self.basis.mean_net(x, kappa_b)
                loss = F.mse_loss(pred, s_b)
                loss.backward()
                clip_grad_norm_(self.basis.mean_net.parameters(), self.cfg.grad_clip_norm)
                opt.step()
                scheduler.step()
                epoch_loss += loss.item()

            if epoch % self.cfg.log_every == 0:
                avg_loss = epoch_loss / len(dl)
                self.history.mean_loss.append(avg_loss)
                pbar.set_postfix(loss=f"{avg_loss:.3e}")

    def _train_mode(
        self,
        mode: NeuralPODMode,
        r: Tensor,
        x: Tensor,
        t: Tensor,
        kappa: Tensor,
    ) -> None:
        dl = self._make_dataloader(r, t, kappa)
        opt = AdamW([
            {"params": mode.phi.parameters(), "lr": self.cfg.lr_phi, "weight_decay": 1e-2},
            {"params": mode.psi.parameters(), "lr": self.cfg.lr_psi, "weight_decay": 0.0},
        ])
        scheduler = OneCycleLR(
            opt,
            max_lr=[self.cfg.max_lr_phi, self.cfg.max_lr_psi],
            epochs=self.cfg.n_epochs_mode,
            steps_per_epoch=len(dl),
            pct_start=self.cfg.pct_start,
            anneal_strategy='cos',
            div_factor=25.0,
        )
        p = len(self.history.mode_losses) + 1
        pbar = tqdm(range(self.cfg.n_epochs_mode), desc=f"mode {p}", leave=False)
        mode_history: list[float] = []

        for epoch in pbar:
            epoch_loss = 0.0
            for r_b, t_b, kappa_b in dl:
                opt.zero_grad()
                pred = mode(x, t_b, kappa_b)
                loss = F.mse_loss(pred, r_b)
                loss.backward()
                clip_grad_norm_(mode.parameters(), self.cfg.grad_clip_norm)
                opt.step()
                scheduler.step()
                epoch_loss += loss.item()

            if epoch % self.cfg.log_every == 0:
                avg_loss = epoch_loss / len(dl)
                mode_history.append(avg_loss)
                pbar.set_postfix(loss=f"{avg_loss:.3e}")

        self.history.mode_losses.append(mode_history)


def plot_train_history(trainer: WeightedNeuralPODTrainer) -> plt.Figure:
    h = trainer.history
    cfg = trainer.cfg
    n_modes = len(h.mode_losses)
    cmap = plt.cm.tab10

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(h.mean_loss, color="steelblue", lw=2)
    ax0.set_title("Mean network loss", fontweight="bold")
    ax0.set_xlabel(f"Epoch (x{cfg.log_every})")
    ax0.set_ylabel("MSE")
    ax0.grid(True, ls="--", alpha=0.4)

    ax1 = fig.add_subplot(gs[0, 1])
    for p, losses in enumerate(h.mode_losses):
        ax1.plot(losses, color=cmap(p % 10), lw=1.5, label=f"mode {p + 1}")
    ax1.set_title("Mode losses", fontweight="bold")
    ax1.set_xlabel(f"Epoch (x{cfg.log_every})")
    ax1.set_ylabel("MSE")
    if n_modes:
        ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, ls="--", alpha=0.4)

    ax2 = fig.add_subplot(gs[1, 0])
    if h.residual_norms:
        idx = list(range(1, len(h.residual_norms) + 1))
        ax2.plot(idx, h.residual_norms, "o-", color="darkorange", lw=2, ms=8, label="res MSE")
        ax2.axhline(trainer._tol_abs, ls="--", color="crimson", lw=1.5, label=f"tol")
        ax2.set_xticks(idx)
        ax2.legend(fontsize=8)
    ax2.set_title("Residual MSE per mode", fontweight="bold")
    ax2.set_xlabel("Mode index")
    ax2.set_ylabel("MSE")
    ax2.grid(True, ls="--", alpha=0.4)

    ax3 = fig.add_subplot(gs[1, 1])
    if h.residual_norms:
        norms = h.residual_norms
        drops = [norms[0]] + [norms[i - 1] - norms[i] for i in range(1, len(norms))]
        ax3.bar(
            range(1, len(drops) + 1), drops,
            color=[cmap(p % 10) for p in range(len(drops))],
            edgecolor="white", linewidth=0.5,
        )
        ax3.set_xticks(range(1, len(drops) + 1))
    ax3.set_title("Residual drop per mode", fontweight="bold")
    ax3.set_xlabel("Mode index")
    ax3.set_ylabel("MSE reduction")
    ax3.grid(True, ls="--", alpha=0.4, axis="y")

    return fig


@dataclass
class FourierNeuralPODConfig:
    tol:            float = 1e-3
    lr:             float = 5e-4
    lr_lambda:      float = 1e-2
    max_lr:         float = 5e-3
    max_modes:      int   = 5
    n_epochs_mean:  int   = 160+5
    n_epochs_mode:  int   = 80+5
    batch_size:     int   = 32
    grad_clip_norm: float = 1.0
    pct_start:      float = 0.1
    div_factor:     float = 25.0
    log_every:      int   = 20


class FourierNeuralPODTrainer:
    """Greedy FN-POD trainer with Fourier spatial basis and learnable temporal coefficients.

    Batches by spatial points x, trains mean then modes sequentially on residuals.
    Parameters t, kappa, gamma in train() are accepted but ignored.
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
        kappa: Tensor,
        gamma: Tensor,
    ) -> TrainHistory:
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
            TensorDataset(x, target), batch_size=self.cfg.batch_size, shuffle=True
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
