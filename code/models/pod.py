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
    """POD basis: mean and orthonormal modes as non-trainable buffers."""

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
        """Reconstruct all training snapshots. Returns (N, Ny)."""
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
            x:     (Ny, d_x) spatial grid; unused, POD is grid-based
            t:     unused
            kappa: unused
            gamma: (N,) responsibility weights; uniform 1/N if None

        Returns:
            TrainHistory with residual_norms per mode
        """
        N, Ny = s.shape
        dtype = s.dtype
        dev = s.device
        gpu = x.device if (x is not None and x.device.type != 'cpu') else dev

        if gamma is None:
            gamma = torch.ones(N, dtype=dtype, device=dev) / N
        else:
            gamma = (gamma / gamma.sum()).to(device=dev, dtype=dtype)

        print(f"POD | N={N}, Ny={Ny} | max_modes={self.cfg.max_modes}, tol={self.cfg.tol}")

        mean = (gamma.unsqueeze(1) * s).sum(dim=0)  # (Ny,)

        sqrt_gamma = gamma.sqrt()
        if gpu != s.device:
            # move to GPU first, then center and scale in-place to avoid a large CPU copy
            s_weighted_gpu = s.to(gpu, dtype=torch.float32)
            s_weighted_gpu.sub_(mean.to(gpu)).mul_(sqrt_gamma.to(gpu).unsqueeze(1))
        else:
            # sub() yields a new tensor; mul_ avoids a second allocation
            s_weighted_gpu = s.sub(mean.unsqueeze(0)).mul_(sqrt_gamma.unsqueeze(1)).to(dtype=torch.float32)

        # randomized SVD: top-q singular vectors, O(N*q) memory
        q = min(self.cfg.max_modes + 10, min(N, Ny))
        _, sigma, V = torch.svd_lowrank(s_weighted_gpu, q=q, niter=4)
        del s_weighted_gpu

        sigma = sigma.to(dtype=dtype)
        energy = sigma ** 2
        cumvar = torch.cumsum(energy, dim=0) / energy.sum()
        above_tol = (cumvar >= self.cfg.tol).nonzero(as_tuple=True)[0]
        n_for_tol = int(above_tol[0].item()) + 1 if len(above_tol) > 0 else len(sigma)
        P = min(n_for_tol, self.cfg.max_modes, len(sigma))

        print(f"  P={P} modes ({self.cfg.tol*100:.2f}% variance, needed {n_for_tol})")
        for p in range(P):
            print(f"  mode {p+1:2d}: sigma={sigma[p].item():.4e}  cumvar={cumvar[p].item()*100:.2f}%")

        modes = V[:, :P].to(device=dev, dtype=dtype)  # (Ny, P)
        del V

        coeffs = (s @ modes - (mean @ modes).unsqueeze(0)).cpu()  # (N, P)
        modes = modes.cpu()  # (Ny, P)

        self.basis.initialize(mean, modes, coeffs)
        self.num_modes = P
        print("  done")
        return self.history

