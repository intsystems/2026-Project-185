from __future__ import annotations

from dataclasses import dataclass
from tqdm.auto import tqdm

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_

from .regime_basis import FourierRegimeBasis
from .neural_pod_mode import FourierPODMode


def _weighted_norm_sq(r: Tensor, gamma: Tensor, w: Tensor) -> float:
    """Weighted squared norm: sum_i gamma_i * sum_j w_j * r_ij^2. Always on CPU."""
    r_cpu = r.cpu().float()
    gamma_cpu = gamma.cpu().float()
    w_cpu = w.cpu().float()
    return (gamma_cpu * (r_cpu ** 2 * w_cpu[None, :]).sum(dim=1)).sum().item()


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
    traj_batch_size: int = 256   # batch size over N (trajectories)
    grad_clip_norm: float = 1.0
    pct_start: float = 0.1
    div_factor: float = 25.0
    log_every: int = 20


class FourierNeuralPODTrainer:
    """Sequential mode extraction with Fourier spatial basis.

    Batching is over N (trajectories), not Ny (spatial points).
    phi(x) and mean_net(x) are computed once per epoch on the GPU (Ny,).
    Trajectory batches r[i:i+B] are moved from CPU to GPU inside the loop.
    lambda_ten lives on the GPU (N scalars per mode — always tiny).

    This keeps the GPU busy with large (Ny, B) matrix ops regardless of GPU size.
    s and r stay on CPU so datasets of arbitrary size are supported.
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
            s:     (N, Ny) snapshot matrix — may be on CPU
            x:     (Ny, d_x) spatial grid — moved to network device
            t:     unused
            kappa: unused
            gamma: (N,) responsibility weights; uniform 1/N if None
        """
        N, Ny = s.shape
        dtype = s.dtype
        dev = self._device

        x = x.to(device=dev, dtype=dtype)
        w = self.basis.quad_weights.to(device=dev, dtype=dtype)  # (Ny,)

        if gamma is None:
            gamma_cpu = torch.ones(N, dtype=dtype) / N
        else:
            gamma_cpu = (gamma / gamma.sum()).cpu().to(dtype=dtype)
        gamma_dev = gamma_cpu.to(dev)

        print(f"Fourier NeuralPOD | N={N}, Ny={Ny} | max_modes={self.cfg.max_modes} | traj_batch={self.cfg.traj_batch_size}")

        s_mean_cpu = (gamma_cpu[:, None] * s.cpu()).sum(dim=0)  # (Ny,) on CPU
        s_mean_dev = s_mean_cpu.to(dev)
        s_centered = s.cpu() - s_mean_cpu[None, :]  # (N, Ny) on CPU

        self._tol_abs = self.cfg.tol * _weighted_norm_sq(s_centered, gamma_cpu, w)

        self._train_mean(s_mean_dev, x, w)
        r = self._full_residual(s, x)  # (N, Ny) on CPU

        self.num_modes = 0
        while (
            _weighted_norm_sq(r, gamma_cpu, w) >= self._tol_abs
            and len(self.basis.modes) < self.cfg.max_modes
        ):
            self.num_modes += 1
            mode = self.basis.add_mode()
            self._train_mode(mode, r, x, gamma_dev, gamma_cpu, w)
            r = self._update_residual(r, mode, x)
            res = _weighted_norm_sq(r, gamma_cpu, w)
            self.history.residual_norms.append(res)
            print(f"  mode {self.num_modes}: weighted_res={res:.4e}")

        print(f"  done: {self.num_modes} modes\n")
        return self.history

    @torch.no_grad()
    def _full_residual(self, s: Tensor, x: Tensor) -> Tensor:
        """Returns (N, Ny) on CPU: s - mean_net(x)."""
        mean_cpu = self.basis.mean_net(x).cpu()
        return (s.cpu() - mean_cpu[None, :]).detach()

    @torch.no_grad()
    def _update_residual(self, r: Tensor, mode: FourierPODMode, x: Tensor) -> Tensor:
        """Returns (N, Ny) on CPU: r - outer(lambda, phi(x))."""
        phi_cpu = mode.phi(x).cpu()
        lambda_cpu = mode.lambda_ten.cpu()
        return (r.cpu() - torch.outer(lambda_cpu, phi_cpu)).detach()

    def _train_mean(self, s_mean: Tensor, x: Tensor, w: Tensor) -> None:
        """Loss: sum_j w_j (mean_net(x_j) - s_mean_j)^2  (full batch on GPU)."""
        opt = AdamW(self.basis.mean_net.parameters(), lr=self.cfg.lr, weight_decay=1e-2)
        scheduler = OneCycleLR(
            opt, max_lr=self.cfg.max_lr,
            epochs=self.cfg.n_epochs_mean, steps_per_epoch=1,
            anneal_strategy="cos", pct_start=self.cfg.pct_start,
            div_factor=self.cfg.div_factor,
        )
        pbar = tqdm(range(self.cfg.n_epochs_mean), desc="mean", leave=False)
        for epoch in pbar:
            opt.zero_grad()
            loss = (w * (self.basis.mean_net(x) - s_mean) ** 2).sum()
            loss.backward()
            clip_grad_norm_(self.basis.mean_net.parameters(), self.cfg.grad_clip_norm)
            opt.step()
            scheduler.step()
            if epoch % self.cfg.log_every == 0:
                self.history.mean_loss.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.3e}")

    def _train_mode(
        self, mode: FourierPODMode, r: Tensor, x: Tensor,
        gamma_dev: Tensor, gamma_cpu: Tensor, w: Tensor
    ) -> None:
        """Loss: sum_i gamma_i sum_j w_j (phi(x_j) * lambda_i - r_ij)^2."""
        B = self.cfg.traj_batch_size
        N = r.shape[0]
        dev = self._device

        opt = AdamW([
            {"params": mode.phi.parameters(), "lr": self.cfg.lr, "weight_decay": 1e-2},
            {"params": [mode.lambda_ten], "lr": self.cfg.lr_lambda, "weight_decay": 0.0},
        ])
        scheduler = OneCycleLR(
            opt, max_lr=self.cfg.max_lr,
            epochs=self.cfg.n_epochs_mode, steps_per_epoch=1,
            anneal_strategy="cos", pct_start=self.cfg.pct_start,
            div_factor=self.cfg.div_factor,
        )

        p = len(self.history.mode_losses) + 1
        pbar = tqdm(range(self.cfg.n_epochs_mode), desc=f"mode {p}", leave=False)
        mode_history: list[float] = []

        for epoch in pbar:
            opt.zero_grad()
            epoch_loss = 0.0

            phi = mode.phi(x)  # (Ny,) evaluated once per epoch

            for i in range(0, N, B):
                r_b = r[i : i + B].to(dev)  # (B, Ny)
                gamma_b = gamma_dev[i : i + B]  # (B,)

                lam_b = mode.lambda_ten[i : i + B]  # (B,)
                pred = torch.outer(phi, lam_b)  # (Ny, B)

                diff = (pred - r_b.T) ** 2  # (Ny, B)
                loss_b = (diff * w[:, None] * gamma_b[None, :]).sum()
                loss_b.backward(retain_graph=(i + B < N))

                epoch_loss += loss_b.item()

            clip_grad_norm_(
                list(mode.phi.parameters()) + [mode.lambda_ten],
                self.cfg.grad_clip_norm,
            )
            opt.step()
            scheduler.step()

            if epoch % self.cfg.log_every == 0:
                mode_history.append(epoch_loss)
                pbar.set_postfix(loss=f"{epoch_loss:.3e}")

        self.history.mode_losses.append(mode_history)
