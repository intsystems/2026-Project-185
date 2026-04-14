from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .pod import PODTrainer
from .pod_deeponet import BranchNet
from .fourier_neural_pod import FourierNeuralPODTrainer


@torch.no_grad()
def _eval_basis(trainer, x: Tensor) -> tuple[Tensor, Tensor]:
    """Extract frozen (mean, modes) from a regime trainer.

    Args:
        trainer: PODTrainer or FourierNeuralPODTrainer
        x:       (Ny, d_x) coordinate grid on the target device
    Returns:
        mean:  (Ny,)
        modes: (Ny, P)
    """
    if isinstance(trainer, PODTrainer):
        return (
            trainer.basis.mean.to(x.device),
            trainer.basis.modes.to(x.device),
        )
    if isinstance(trainer, FourierNeuralPODTrainer):
        mean = trainer.basis.mean_net(x)
        modes = torch.stack([mode.phi(x) for mode in trainer.basis.modes], dim=1)
        return mean, modes
    raise TypeError(f"Unsupported trainer type: {type(trainer)}")


def _num_modes(trainer) -> int:
    if isinstance(trainer, PODTrainer):
        return trainer.basis.num_modes
    if isinstance(trainer, FourierNeuralPODTrainer):
        return trainer.num_modes
    raise TypeError(f"Unsupported trainer type: {type(trainer)}")


class GatingNet(nn.Module):
    """(u0_sensors, kappa) -> M simplex weights."""

    def __init__(self, m_sensors: int, d_kappa: int, M: int,
                 hidden_dim: int = 128, n_layers: int = 4) -> None:
        super().__init__()
        self.branch = BranchNet(m_sensors, M, hidden_dim, n_layers, d_kappa=d_kappa)

    def forward(self, u0: Tensor, kappa: Tensor) -> Tensor:
        """
        Args:
            u0:    (N, m_sensors)
            kappa: (N, d_kappa)
        Returns:
            (N, M) simplex weights
        """
        return F.softmax(self.branch(u0, kappa), dim=-1)


class TEMPOOnline(nn.Module):
    """Gated operator with M frozen regime bases (Phase 2).

    Prediction (eq. 10):
        s_hat = sum_m w_m(u0, kappa) [mean_m(x) + branch_m(u0, kappa) @ Phi_m(x).T]

    Trainable: GatingNet + M BranchNets (with d_kappa > 0).
    Frozen bases (means, modes) are precomputed and passed to forward.
    """

    def __init__(self, gating: GatingNet, branches: nn.ModuleList) -> None:
        super().__init__()
        self.gating = gating
        self.branches = branches

    @property
    def M(self) -> int:
        return len(self.branches)

    def forward(
        self,
        u0: Tensor,
        kappa: Tensor,
        means: list[Tensor],
        modes: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            u0:    (N, m_sensors)
            kappa: (N, d_kappa)
            means: M x (Ny,)
            modes: M x (Ny, P_m)
        Returns:
            s_hat: (N, Ny)
            w:     (N, M) gating weights
        """
        w = self.gating(u0, kappa)
        s_hat = torch.zeros(u0.shape[0], means[0].shape[0],
                            device=u0.device, dtype=u0.dtype)
        for m, branch in enumerate(self.branches):
            b_m = branch(u0, kappa)                        # (N, P_m)
            s_hat_m = means[m].unsqueeze(0) + b_m @ modes[m].T  # (N, Ny)
            s_hat = s_hat + w[:, m : m + 1] * s_hat_m
        return s_hat, w


@dataclass
class TEMPOOnlineConfig:
    lr: float = 3e-4
    n_epochs: int = 1000
    batch_size: int = 64
    hidden_dim: int = 128
    n_layers: int = 4
    sensor_stride: int = 1
    lambda_kl: float = 0.1
    lambda_ent: float = 0.1
    log_every: int = 50


class TEMPOOnlineTrainer:
    """Phase 2 trainer: GatingNet + M BranchNets with frozen bases.

    Loss (eq. 11):
        L = L_data + lambda_kl * L_KL - lambda_ent * L_ent

        L_data = mean_i || s^(i) - s_hat^(i) ||^2
        L_KL   = mean_i sum_m  w_m^(i) log(w_m^(i) / gamma_im*)
        L_ent  = mean_i sum_m -w_m^(i) log(w_m^(i))
    """

    def __init__(self, model: TEMPOOnline, cfg: TEMPOOnlineConfig) -> None:
        self.model = model
        self.cfg = cfg

    def train(
        self,
        s: Tensor,       # (N, Ny)    full trajectories, CPU ok
        u0: Tensor,      # (N, Nx)    initial conditions, CPU ok
        kappa: Tensor,   # (N, d_kappa)
        x_flat: Tensor,  # (Ny, 2)    spatiotemporal coordinates
        gamma_star: Tensor,  # (N, M) offline EM responsibilities
        trainers: list,  # M frozen PODTrainer / FourierNeuralPODTrainer
    ) -> dict[str, list[float]]:
        device = next(self.model.parameters()).device
        stride = self.cfg.sensor_stride

        x_dev = x_flat.to(device)
        means_dev, modes_dev = [], []
        for t in trainers:
            mean, modes = _eval_basis(t, x_dev)
            means_dev.append(mean.detach())
            modes_dev.append(modes.detach())

        u0_sensors = u0[:, ::stride]

        dl = DataLoader(
            TensorDataset(u0_sensors, kappa, s, gamma_star),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )
        opt = torch.optim.AdamW(self.model.parameters(),
                                lr=self.cfg.lr, weight_decay=1e-4)

        history: dict[str, list[float]] = {'total': [], 'data': [], 'kl': [], 'ent': []}

        for epoch in range(self.cfg.n_epochs):
            self.model.train()
            epoch_loss = l_data_ep = l_kl_ep = l_ent_ep = 0.0

            for u0_b, kappa_b, s_b, gamma_b in dl:
                u0_b = u0_b.to(device)
                kappa_b = kappa_b.to(device)
                s_b = s_b.to(device)
                gamma_b = gamma_b.to(device)

                s_hat, w = self.model(u0_b, kappa_b, means_dev, modes_dev)

                l_data = F.mse_loss(s_hat, s_b)

                w_c = w.clamp(min=1e-8)
                gamma_c = gamma_b.clamp(min=1e-8)
                l_kl = (w_c * (w_c.log() - gamma_c.log())).sum(dim=1).mean()
                l_ent = -(w_c * w_c.log()).sum(dim=1).mean()

                loss = l_data + self.cfg.lambda_kl * l_kl - self.cfg.lambda_ent * l_ent

                opt.zero_grad()
                loss.backward()
                opt.step()

                n_b = u0_b.shape[0]
                epoch_loss += loss.item() * n_b
                l_data_ep += l_data.item() * n_b
                l_kl_ep += l_kl.item() * n_b
                l_ent_ep += l_ent.item() * n_b

            N = len(dl.dataset)
            history['total'].append(epoch_loss / N)
            history['data'].append(l_data_ep / N)
            history['kl'].append(l_kl_ep / N)
            history['ent'].append(l_ent_ep / N)

            if epoch % self.cfg.log_every == 0:
                print(f"  epoch {epoch:4d} | loss={epoch_loss/N:.4e} | "
                      f"data={l_data_ep/N:.4e}  kl={l_kl_ep/N:.4e}  ent={l_ent_ep/N:.4e}")

        print(f"  done: final loss={history['total'][-1]:.4e}")
        return history

    @torch.no_grad()
    def predict(
        self,
        u0_new: Tensor,    # (N, Nx)
        kappa_new: Tensor, # (N, d_kappa)
        x_flat: Tensor,    # (Ny, 2)
        trainers: list,
    ) -> tuple[Tensor, Tensor]:
        """Predict trajectories and gating weights for new inputs.

        Returns:
            s_hat: (N, Ny)
            w:     (N, M) gating weights
        """
        device = next(self.model.parameters()).device
        x_dev = x_flat.to(device)

        means_dev, modes_dev = [], []
        for t in trainers:
            mean, modes = _eval_basis(t, x_dev)
            means_dev.append(mean.detach())
            modes_dev.append(modes.detach())

        u0_sensors = u0_new[:, ::self.cfg.sensor_stride].to(device)
        kappa_dev = kappa_new.to(device)

        self.model.eval()
        return self.model(u0_sensors, kappa_dev, means_dev, modes_dev)


def build_tempo_online(
    trainers: list,
    d_kappa: int,
    Nx: int,
    cfg: TEMPOOnlineConfig,
) -> tuple[TEMPOOnline, TEMPOOnlineTrainer]:
    """Construct TEMPOOnline and trainer from frozen regime trainers.

    Args:
        trainers: M frozen PODTrainer / FourierNeuralPODTrainer
        d_kappa:  physical parameter dimension
        Nx:       number of spatial grid points
        cfg:      online config
    Returns:
        (model, trainer)
    """
    M = len(trainers)
    m_sensors = Nx // cfg.sensor_stride
    P_list = [_num_modes(t) for t in trainers]

    gating = GatingNet(m_sensors, d_kappa, M, cfg.hidden_dim, cfg.n_layers)
    branches = nn.ModuleList([
        BranchNet(m_sensors, P_list[m], cfg.hidden_dim, cfg.n_layers, d_kappa=d_kappa)
        for m in range(M)
    ])

    model = TEMPOOnline(gating, branches)
    trainer = TEMPOOnlineTrainer(model, cfg)
    return model, trainer