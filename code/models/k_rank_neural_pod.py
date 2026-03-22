from __future__ import annotations

from dataclasses import dataclass
from tqdm.auto import tqdm

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from .regime_basis import RegimeBasis


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
class NeuralPODConfig:
    """Hyperparameters for simultaneous K-rank mode training."""
    lr: float = 5e-4
    K: int = 8
    n_epochs: int = 2000
    batch_size: int = 32
    grad_clip_norm: float = 1.0
    log_every: int = 100
    lambda_basis: float = 1.0


class WeightedNeuralPODTrainer:
    """Joint mean and K-rank mode training with basis penalty.

    Loss: weighted sum of MSE and basis orthonormality constraint.
    """

    def __init__(self, basis: RegimeBasis, cfg: NeuralPODConfig) -> None:
        self.basis = basis
        self.cfg = cfg
        self.history = TrainHistory()
        self._w_data = None
        self._w_basis = None
        self.num_modes = 0

    def train(
        self,
        s: Tensor,
        x: Tensor,
        t: Tensor,
        kappa: Tensor,
        gamma: Tensor,
    ) -> TrainHistory:
        """Train mean and modes with basis orthonormality penalty.

        Args:
            s: (N, Ny) snapshot matrix
            x: (Ny, d_x) spatial grid
            t: (N,) time vector
            kappa: (N, d_kappa) parameters
            gamma: unused

        Returns:
            TrainHistory with weighted losses
        """
        print(f"\nWeighted Joint Neural POD Training")
        print(f"Data: s={tuple(s.shape)}, x={tuple(x.shape)}")
        print(f"Config: K={self.cfg.K}, n_epochs={self.cfg.n_epochs}, batch_size={self.cfg.batch_size}")
        print(f"Optimizer: lr={self.cfg.lr}, grad_clip={self.cfg.grad_clip_norm}")

        N, Ny = s.shape

        dl = DataLoader(
            TensorDataset(s, t, kappa),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        opt = AdamW(self.basis.parameters(), lr=self.cfg.lr)
        pbar = tqdm(range(self.cfg.n_epochs), desc="training", leave=True)

        # Initialize weights from first batch
        self.basis.eval()
        with torch.no_grad():
            s_init, t_init, kappa_init = next(iter(dl))
            pred_init = self.basis(x, t_init, kappa_init)
            loss_data_init = F.mse_loss(pred_init, s_init).item()

            phi_all = self.basis.mode.basis_net(x)
            rms_phi = torch.sqrt(torch.sum(phi_all ** 2) / phi_all.numel())
            loss_basis_init = (torch.abs(rms_phi - 1.0) ** 2).item()

        eps = 1e-8
        self._w_data = 1.0 / (loss_data_init + eps)
        self._w_basis = 1.0 / (loss_basis_init + eps)

        if self.cfg.lambda_basis == 0.0:
            self._w_basis = 0.0

        print(f"Weight initialization:")
        print(f"  L_data = {loss_data_init:.4e}, w_data = {self._w_data:.4e}")
        print(f"  L_basis = {loss_basis_init:.4e}, w_basis = {self._w_basis:.4e}\n")

        # Training loop
        self.basis.train()

        for epoch in pbar:
            epoch_loss_data = 0.0
            epoch_loss_basis = 0.0

            with torch.no_grad():
                phi_all = self.basis.mode.basis_net(x)
                rms_phi = torch.sqrt(torch.sum(phi_all ** 2) / phi_all.numel())
                loss_basis_epoch = (torch.abs(rms_phi - 1.0) ** 2).item()

            for s_b, t_b, kappa_b in dl:
                opt.zero_grad()

                pred_b = self.basis(x, t_b, kappa_b)
                loss_data = F.mse_loss(pred_b, s_b)
                loss_basis = loss_basis_epoch

                loss = self._w_data * loss_data + self._w_basis * loss_basis

                loss.backward()
                clip_grad_norm_(self.basis.parameters(), self.cfg.grad_clip_norm)
                opt.step()

                epoch_loss_data += loss_data.item()
                epoch_loss_basis += loss_basis

            if epoch % self.cfg.log_every == 0:
                avg_data = epoch_loss_data / len(dl)
                avg_basis = epoch_loss_basis / len(dl)
                avg_weighted = self._w_data * avg_data + self._w_basis * avg_basis
                self.history.mean_loss.append(avg_weighted)
                pbar.set_postfix(
                    loss=f"{avg_weighted:.3e}",
                    L_data=f"{avg_data:.3e}",
                    L_basis=f"{avg_basis:.3e}",
                )

        final_loss = self.history.mean_loss[-1] if self.history.mean_loss else 0.0
        print(f"Training complete: {self.cfg.n_epochs} epochs")
        print(f"Final loss: {final_loss:.4e}\n")

        return self.history
