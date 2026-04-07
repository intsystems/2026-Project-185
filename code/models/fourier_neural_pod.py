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
    """sum_i gamma_i * sum_j w_j * r_ij^2
    r, gamma, w may be on different devices — computation done on CPU.
    """
    r_cpu     = r.cpu().float()
    gamma_cpu = gamma.cpu().float()
    w_cpu     = w.cpu().float()
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
    batch_size: int = 32
    grad_clip_norm: float = 1.0
    pct_start: float = 0.1
    div_factor: float = 25.0
    log_every: int = 20
    full_batch: bool = False  # if True: skip DataLoader, run full-batch on GPU


class FourierNeuralPODTrainer:
    """Sequential mode extraction with Fourier spatial basis.

    s (N, Ny) stays on CPU throughout; only small tensors (x, w, s_mean, gamma)
    are moved to the network device. This allows training on arbitrarily large datasets.
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
        dtype  = s.dtype
        dev    = self._device

        # x and w always on network device
        x = x.to(device=dev, dtype=dtype)
        w = self.basis.quad_weights.to(device=dev, dtype=dtype)  # (Ny,)

        # gamma on CPU for weighted mean computation, then move to device
        if gamma is None:
            gamma_cpu = torch.ones(N, dtype=dtype) / N
        else:
            gamma_cpu = (gamma / gamma.sum()).cpu().to(dtype=dtype)
        gamma = gamma_cpu.to(dev)

        print(f"Fourier NeuralPOD | N={N}, Ny={Ny} | max_modes={self.cfg.max_modes}")

        # compute weighted mean on CPU (s stays on CPU)
        s_mean_cpu = (gamma_cpu[:, None] * s.cpu()).sum(dim=0)    # (Ny,) on CPU
        s_mean     = s_mean_cpu.to(dev)                            # (Ny,) on GPU
        s_centered = s.cpu() - s_mean_cpu[None, :]                 # (N, Ny) on CPU

        self._tol_abs = self.cfg.tol * _weighted_norm_sq(s_centered, gamma_cpu, w)

        self._train_mean(s_mean, x, w)
        r = self._full_residual(s, x)   # (N, Ny) on CPU

        self.num_modes = 0
        while (
            _weighted_norm_sq(r, gamma_cpu, w) >= self._tol_abs
            and len(self.basis.modes) < self.cfg.max_modes
        ):
            self.num_modes += 1
            mode = self.basis.add_mode()
            self._train_mode(mode, r, x, gamma, gamma_cpu, w)
            r = self._update_residual(r, mode, x)   # (N, Ny) on CPU
            res = _weighted_norm_sq(r, gamma_cpu, w)
            self.history.residual_norms.append(res)
            print(f"  mode {self.num_modes}: weighted_res={res:.4e}")

        print(f"  done: {self.num_modes} modes\n")
        return self.history

    def _make_dataloader(self, *tensors) -> DataLoader:
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            pin_memory=(self._device.type == "cuda"),
        )

    @torch.no_grad()
    def _full_residual(self, s: Tensor, x: Tensor) -> Tensor:
        """Returns (N, Ny) residual on CPU."""
        mean = self.basis.mean_net(x).cpu()   # (Ny,) → CPU
        return (s.cpu() - mean.unsqueeze(0)).detach()

    @torch.no_grad()
    def _update_residual(self, r: Tensor, mode: FourierPODMode, x: Tensor) -> Tensor:
        """Returns (N, Ny) updated residual on CPU."""
        phi    = mode.phi(x).cpu()                          # (Ny,) → CPU
        lambda_ = mode.lambda_ten.cpu()                     # (N,)  → CPU
        return (r.cpu() - torch.outer(phi, lambda_).T).detach()

    def _train_mean(self, s_mean: Tensor, x: Tensor, w: Tensor) -> None:
        """Loss: sum_j w_j * (mean_net(x_j) - s_mean_j)^2
        s_mean, x, w are all on network device.
        """
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
                loss = (w * (self.basis.mean_net(x) - s_mean) ** 2).sum()
                loss.backward()
                clip_grad_norm_(self.basis.mean_net.parameters(), self.cfg.grad_clip_norm)
                opt.step()
                scheduler.step()
                if epoch % self.cfg.log_every == 0:
                    self.history.mean_loss.append(loss.item())
                    pbar.set_postfix(loss=f"{loss.item():.3e}")
        else:
            # DataLoader on CPU tensors; batches moved to device inside loop
            dl = self._make_dataloader(x.cpu(), s_mean.cpu(), w.cpu())
            scheduler = OneCycleLR(
                opt, max_lr=self.cfg.max_lr,
                epochs=self.cfg.n_epochs_mean, steps_per_epoch=len(dl),
                anneal_strategy="cos", pct_start=self.cfg.pct_start, div_factor=self.cfg.div_factor,
            )
            for epoch in pbar:
                epoch_loss = 0.0
                for x_b, mean_b, w_b in dl:
                    x_b, mean_b, w_b = x_b.to(self._device), mean_b.to(self._device), w_b.to(self._device)
                    opt.zero_grad()
                    loss = (w_b * (self.basis.mean_net(x_b) - mean_b) ** 2).sum()
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
        self, mode: FourierPODMode, r: Tensor, x: Tensor,
        gamma: Tensor, gamma_cpu: Tensor, w: Tensor
    ) -> None:
        """Loss: sum_i gamma_i * sum_j w_j * (phi(x_j) * lambda_i - r_ij)^2
        r is on CPU; x, gamma, w are on network device.
        """
        opt = AdamW([
            {"params": mode.phi.parameters(), "lr": self.cfg.lr, "weight_decay": 1e-2},
            {"params": [mode.lambda_ten], "lr": self.cfg.lr_lambda, "weight_decay": 0.0},
        ])
        p = len(self.history.mode_losses) + 1
        pbar = tqdm(range(self.cfg.n_epochs_mode), desc=f"mode {p}", leave=False)
        mode_history: list[float] = []

        if self.cfg.full_batch:
            # r.T moved to GPU once — requires enough GPU memory
            r_T = r.T.to(self._device)   # (Ny, N)
            scheduler = OneCycleLR(
                opt, max_lr=self.cfg.max_lr,
                epochs=self.cfg.n_epochs_mode, steps_per_epoch=1,
                anneal_strategy="cos", pct_start=self.cfg.pct_start, div_factor=self.cfg.div_factor,
            )
            for epoch in pbar:
                opt.zero_grad()
                phi  = mode.phi(x)                           # (Ny,)
                pred = torch.outer(phi, mode.lambda_ten)     # (Ny, N)
                loss = ((pred - r_T) ** 2 * gamma[None, :] * w[:, None]).sum()
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
            # DataLoader batches spatial points; r.T stays on CPU, moved per batch
            dl = self._make_dataloader(x.cpu(), r.T.contiguous(), w.cpu())
            scheduler = OneCycleLR(
                opt, max_lr=self.cfg.max_lr,
                epochs=self.cfg.n_epochs_mode, steps_per_epoch=len(dl),
                anneal_strategy="cos", pct_start=self.cfg.pct_start, div_factor=self.cfg.div_factor,
            )
            for epoch in pbar:
                epoch_loss = 0.0
                for x_b, r_b, w_b in dl:
                    x_b = x_b.to(self._device)
                    r_b = r_b.to(self._device)   # (B, N)
                    w_b = w_b.to(self._device)
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
