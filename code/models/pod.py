from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


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
class PODConfig:
    """Hyperparameters for classical POD via SVD."""
    max_modes: int = 10
    tol: float = 0.9999
    log_every: int = 1 


class PODBasis(nn.Module):
    """POD basis: mean and orthonormal modes stored as non-trainable buffers.

    Buffers (not parameters) — move with .to(device) and save with state_dict,
    but are excluded from optimizer updates.
    """

    def __init__(self) -> None:
        super().__init__()
        self._initialized = False

    def initialize(self, mean: Tensor, modes: Tensor, coeffs: Tensor) -> None:
        """Store SVD results as buffers.

        Args:
            mean:   (Ny,)   weighted mean snapshot
            modes:  (Ny, P) orthonormal POD modes (right singular vectors)
            coeffs: (N, P)  projection coefficients for training snapshots
        """
        self.register_buffer("mean", mean)
        self.register_buffer("modes", modes)
        self.register_buffer("coeffs", coeffs)
        self._initialized = True

    @property
    def num_modes(self) -> int:
        return self.modes.shape[1] if self._initialized else 0

    def forward(self, x: Tensor = None, t: Tensor = None, kappa: Tensor = None) -> Tensor:
        """Reconstruct all training snapshots.

        Args:
            x, t, kappa: unused — kept for interface compatibility

        Returns:
            (N, Ny) reconstruction:  mean + coeffs @ modes.T
        """
        return self.mean.unsqueeze(0) + self.coeffs @ self.modes.T


class PODTrainer:
    """Classical POD via SVD with optional responsibility weighting.

    Computes weighted mean, centers snapshots, runs SVD, selects modes
    by variance threshold. Stores basis for reconstruction and comparison
    """

    def __init__(self, cfg: PODConfig) -> None:
        self.cfg = cfg
        self.basis = PODBasis()
        self.history = TrainHistory()
        self.num_modes = 0

    def train(
        self,
        s: Tensor,
        x: Tensor,
        t: Tensor,
        kappa: Tensor = None,
        gamma: Tensor = None,
    ) -> TrainHistory:
        """Compute POD basis via SVD.

        Args:
            s:     (N, Ny) snapshot matrix
            x:     (Ny, d_x) spatial grid — unused, POD is grid-based
            t:     (N,) time vector — unused
            kappa: unused
            gamma: (N,) responsibility weights; uniform 1/N if None

        Returns:
            TrainHistory with residual_norms per mode
        """
        N, Ny = s.shape
        device = s.device
        dtype = s.dtype

        if gamma is None:
            gamma = torch.ones(N, device=device, dtype=dtype) / N
        else:
            gamma = (gamma / gamma.sum()).to(device=device, dtype=dtype)

        print(f"\nClassical POD (SVD)")
        print(f"Data: N={N} snapshots, Ny={Ny} spatial points")
        print(f"Config: max_modes={self.cfg.max_modes}, tol={self.cfg.tol}")

        # Weighted mean: sum_i gamma_i * s_i
        mean = (gamma.unsqueeze(1) * s).sum(dim=0)   # (Ny,)
        s_centered = s - mean.unsqueeze(0)             # (N, Ny)

        # Weighted SVD: scale rows by sqrt(gamma) so that
        # SVD of S_w = diag(sqrt(gamma)) @ S_centered gives
        # modes optimal under the gamma-weighted L2 norm
        sqrt_gamma = gamma.sqrt().unsqueeze(1)         # (N, 1)
        s_weighted = sqrt_gamma * s_centered           # (N, Ny)

        # SVD on CPU
        U, sigma, Vh = torch.linalg.svd(
            s_weighted.cpu().float(), full_matrices=False
        )
        # shapes: U (N, K), sigma (K,), Vh (K, Ny),  K = min(N, Ny)

        V = Vh.T.to(device=device, dtype=dtype)  # (Ny, K) — modes as columns
        sigma = sigma.to(device=device, dtype=dtype)

        # Select P by variance threshold, capped at max_modes
        energy = sigma ** 2
        cumvar = torch.cumsum(energy, dim=0) / energy.sum()
        above_tol = (cumvar >= self.cfg.tol).nonzero(as_tuple=True)[0]
        n_for_tol = int(above_tol[0].item()) + 1 if len(above_tol) > 0 else len(sigma)
        P = min(n_for_tol, self.cfg.max_modes, len(sigma))

        top_k = min(10, len(sigma))
        print(f"Top-{top_k} singular values: "
              f"{sigma[:top_k].cpu().numpy().round(4).tolist()}")
        print(f"Modes for {self.cfg.tol*100:.2f}% variance: {n_for_tol}  →  using P={P}\n")

        modes = V[:, :P]             # (Ny, P)
        coeffs = s_centered @ modes  # (N, P)

        self.basis.initialize(mean, modes, coeffs)
        self.num_modes = P


        self.history.mean_loss = [s_centered.pow(2).mean().item()]

        # Residual norms per mode
        residual = s_centered.clone()
        for p in range(P):
            residual = residual - torch.outer(coeffs[:, p], modes[:, p])
            res_mse = residual.pow(2).mean().item()
            self.history.residual_norms.append(res_mse)
            print(f"  Mode {p+1:2d}: σ={sigma[p].item():.4e}  "
                  f"var={cumvar[p].item()*100:.3f}%  "
                  f"residual_mse={res_mse:.4e}")

        print(f"\nPOD complete: {P} modes, "
              f"final residual MSE = {self.history.residual_norms[-1]:.4e}\n")

        return self.history

