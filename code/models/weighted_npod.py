from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from tqdm.auto import tqdm
from typing import Iterator

import torch
from torch import Tensor

from .regime_basis import RegimeBasis
from .neural_pod_mode import NeuralPODMode


@dataclass
class NeuralPODConfig:
    tol:          float = 1e-4
    lr:           float = 1e-3
    max_modes:    int   = 10
    n_steps_mean: int   = 5000
    n_steps_mode: int   = 5000
    batch_size:   int   = 128
    log_every:    int   = 100


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

    def train(
        self,
        s: Tensor,
        x: Tensor,
        t: Tensor,
        kappa: Tensor,
        gamma: Tensor,
    ) -> TrainHistory:
        self._train_mean(s, x, kappa)

        r = self._full_residual(s, x, kappa)

        while (
            torch.mean(r ** 2).item() >= self.cfg.tol
            and len(self.basis.modes) < self.cfg.max_modes
        ):
            mode = self.basis.add_mode()
            self._train_mode(mode, r, x, t, kappa)
            r = self._update_residual(r, mode, x, t, kappa)
            self.history.residual_norms.append(torch.mean(r ** 2).item())

        return self.history

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

    def _batches(self, *tensors: Tensor) -> Iterator[tuple[Tensor, ...]]:
        N = tensors[0].shape[0]
        B = self.cfg.batch_size
        while True:
            idx = torch.randperm(N, device=tensors[0].device)
            for start in range(0, N, B):
                bi = idx[start:start + B]
                yield tuple(t[bi] for t in tensors)

    def _train_mean(self, s: Tensor, x: Tensor, kappa: Tensor) -> None:
        opt = torch.optim.Adam(self.basis.mean_net.parameters(), lr=self.cfg.lr)
        pbar = tqdm(range(self.cfg.n_steps_mean), desc="mean", leave=False)

        for step, (s_b, kappa_b) in zip(pbar, self._batches(s, kappa)):
            opt.zero_grad()
            loss = torch.mean((s_b - self.basis.mean_net(x, kappa_b)) ** 2)
            loss.backward()
            opt.step()
            if step % self.cfg.log_every == 0:
                self.history.mean_loss.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.3e}")

    def _train_mode(
        self,
        mode: NeuralPODMode,
        r: Tensor,
        x: Tensor,
        t: Tensor,
        kappa: Tensor,
    ) -> None:
        opt = torch.optim.Adam(mode.parameters(), lr=self.cfg.lr)
        p = len(self.history.mode_losses) + 1
        pbar = tqdm(range(self.cfg.n_steps_mode), desc=f"mode {p}", leave=False)
        mode_history: list[float] = []

        for step, (r_b, t_b, kappa_b) in zip(pbar, self._batches(r, t, kappa)):
            opt.zero_grad()
            loss = torch.mean((r_b - mode(x, t_b, kappa_b)) ** 2)
            loss.backward()
            opt.step()
            if step % self.cfg.log_every == 0:
                mode_history.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.3e}")

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
    ax0.set_xlabel(f"Step (x{cfg.log_every})")
    ax0.set_ylabel("MSE")
    ax0.grid(True, ls="--", alpha=0.4)

    ax1 = fig.add_subplot(gs[0, 1])
    for p, losses in enumerate(h.mode_losses):
        ax1.plot(losses, color=cmap(p % 10), lw=1.5, label=f"mode {p + 1}")
    ax1.set_title("Mode losses", fontweight="bold")
    ax1.set_xlabel(f"Step (x{cfg.log_every})")
    ax1.set_ylabel("MSE")
    if n_modes:
        ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, ls="--", alpha=0.4)

    ax2 = fig.add_subplot(gs[1, 0])
    if h.residual_norms:
        idx = list(range(1, len(h.residual_norms) + 1))
        ax2.plot(idx, h.residual_norms, "o-", color="darkorange", lw=2, ms=8, label="res MSE")
        ax2.axhline(cfg.tol, ls="--", color="crimson", lw=1.5, label=f"tol = {cfg.tol:.0e}")
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
