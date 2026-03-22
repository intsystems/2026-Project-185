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

from .neural_pod_mode import ClassicalPODMode


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
class ClassicalNeuralPODConfig:
    """Hyperparameters for greedy NeuralPOD training."""
    tol: float = 1e-3
    lr: float = 5e-4
    max_lr: float = 5e-3
    max_modes: int = 10
    n_epochs: int = 100
    batch_size: int = 32
    grad_clip_norm: float = 1.0
    pct_start: float = 0.1
    div_factor: float = 25.0
    log_every: int = 10
    hidden_dim: int = 64
    n_layers: int = 2


class ClassicalNeuralPODTrainer:
    """Greedy sequential mode extraction via residual decomposition.

    Extracts parameter-independent modes with spatial network Phi(x) and
    temporal network Psi(t). Orthogonality maintained through deflation.
    """

    def __init__(self, cfg: ClassicalNeuralPODConfig) -> None:
        self.cfg = cfg
        self.history = TrainHistory()
        self.modes: list[ClassicalPODMode] = []
        self._tol_abs = None

    def train(
        self,
        s: Tensor,
        x: Tensor,
        t: Tensor,
        kappa: Tensor = None,
        gamma: Tensor = None,
    ) -> TrainHistory:
        """Extract modes from snapshot matrix via greedy decomposition.

        Args:
            s: (N, Ny) snapshot matrix
            x: (Ny, d_x) spatial grid
            t: (N,) time vector
            kappa: unused
            gamma: unused

        Returns:
            TrainHistory with epoch losses and residual norms
        """
        N, Ny = s.shape
        d_x = x.shape[1]
        device = s.device

        self._tol_abs = self.cfg.tol * s.pow(2).mean().item()
        residual = s.clone()

        print(f"\nClassical NeuralPOD (Greedy Training)")
        print(f"Data: N={N} snapshots, Ny={Ny} spatial points, d_x={d_x}")
        print(f"Tolerance (absolute): {self._tol_abs:.6e}\n")

        mode_idx = 0
        while (
            residual.pow(2).mean().item() >= self._tol_abs
            and len(self.modes) < self.cfg.max_modes
        ):
            mode_idx += 1
            residual_norm = residual.pow(2).mean().item()
            print(f"Mode {mode_idx}: residual_norm = {residual_norm:.6e}")

            mode = ClassicalPODMode(d_x, self.cfg.hidden_dim, self.cfg.n_layers).to(device)
            self._train_mode(mode, residual, x, t)
            self.modes.append(mode)

            with torch.no_grad():
                pred = mode(x, t)
                residual = residual - pred

            final_residual_norm = residual.pow(2).mean().item()
            self.history.residual_norms.append(final_residual_norm)
            print(f"  after mode: residual_norm = {final_residual_norm:.6e}\n")

        print(f"Training complete: {len(self.modes)} modes trained\n")
        return self.history

    def _train_mode(
        self,
        mode: ClassicalPODMode,
        residual: Tensor,
        x: Tensor,
        t: Tensor,
    ) -> None:
        """Train a single mode on residual data."""
        dl = DataLoader(
            TensorDataset(t, residual),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        opt = AdamW(mode.parameters(), lr=self.cfg.lr)
        scheduler = OneCycleLR(
            opt,
            max_lr=self.cfg.max_lr,
            epochs=self.cfg.n_epochs,
            steps_per_epoch=len(dl),
            anneal_strategy="cos",
            pct_start=self.cfg.pct_start,
            div_factor=self.cfg.div_factor,
        )

        pbar = tqdm(range(self.cfg.n_epochs), desc="  training", leave=False)
        mode_history: list[float] = []

        for epoch in pbar:
            epoch_loss = 0.0

            for t_b, r_b in dl:
                opt.zero_grad()
                pred = mode(x, t_b)
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

    def predict(self, x: Tensor, t: Tensor) -> Tensor:
        """Sum predictions from all extracted modes."""
        recon = torch.zeros(len(t), x.shape[0], device=x.device, dtype=x.dtype)

        for mode in self.modes:
            recon = recon + mode(x, t)

        return recon
