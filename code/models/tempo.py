from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor
from sklearn.mixture import GaussianMixture

from .pod import PODTrainer, PODConfig
from .fourier_neural_pod import FourierNeuralPODTrainer
from .regime_basis import FourierRegimeBasis


def _weighted_norm_sq(r: Tensor, gamma: Tensor, w: Tensor) -> float:
    """sum_i gamma_i * sum_j w_j * r_ij^2"""
    return (gamma * (r ** 2 * w[None, :]).sum(dim=1)).sum().item()


def pod_factory(_s: Tensor, _x: Tensor, cfg: Any):
    return PODTrainer(cfg)


def fourier_pod_factory(s: Tensor, x: Tensor, cfg: Any):
    N, Ny = s.shape
    basis = FourierRegimeBasis(
        d_x=x.shape[-1],
        M=N,
        quad_weights=torch.ones(Ny, device=x.device, dtype=x.dtype) / Ny,
    ).to(x.device)
    return FourierNeuralPODTrainer(basis, cfg)


@dataclass
class TEMPOConfig:
    M: int = 2
    sigma2: float = 0.1
    P_global: int = 20
    max_em_iters: int = 10
    eps_skip: float = 0.01
    eps_large: float = 0.1
    eps_prune: float = 0.01
    eps_conv: float = 0.005
    basis_config: Any = field(default_factory=lambda: PODConfig(max_modes=8))
    basis_factory: Callable = field(default_factory=lambda: pod_factory)


class TEMPOTrainer:
    """TEMPO offline EM trainer.

    After train(): gamma (N, M), pi (M,), trainers (list of M bases)
    """

    def __init__(self, cfg: TEMPOConfig) -> None:
        self.cfg = cfg
        self.trainers: list = []
        self.gamma: Tensor = None    # (N, M)
        self.pi: Tensor = None       # (M,)
        self.alpha: Tensor = None    # (N, P_global)
        self.mu: Tensor = None       # (M, P_global)
        self.Sigma: Tensor = None    # (M, P_global, P_global)
        self._w: Tensor = None       # (Ny,) quadrature weights

    def train(self, s: Tensor, x: Tensor, t: Tensor, kappa: Tensor = None) -> None:
        self._initialize(s, x, t, kappa)
        self._em_loop(s, x, t, kappa)

    def _initialize(self, s: Tensor, x: Tensor, t: Tensor, kappa: Tensor) -> None:
        Ny = s.shape[1]
        self._w = torch.ones(Ny, device=s.device, dtype=s.dtype) / Ny
        self.alpha = self._global_pod(s, x, t)
        self.gamma, self.pi, self.mu, self.Sigma = self._init_gmm(self.alpha, s.device)
        self.trainers = self._init_bases(s, x, t, kappa)
        self._calibrate_sigma2(s, x)

    def _calibrate_sigma2(self, s: Tensor, x: Tensor) -> None:
        """Set sigma2 """
        total = 0.0
        with torch.no_grad():
            for m, trainer in enumerate(self.trainers):
                s_hat  = trainer.basis(x)
                res_sq = ((s - s_hat) ** 2 *self._w[None, :]).sum(dim=1)  # (N,)
                total += (self.gamma[:, m] * res_sq).sum().item()
        self.cfg.sigma2 = 0.5 * total / s.shape[0]
        print(f"  calibrated sigma2 = {self.cfg.sigma2:.4e}")

    def _global_pod(self, s: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """SVD on all snapshots, returns alpha (N, P_global)."""
        trainer = PODTrainer(PODConfig(max_modes=self.cfg.P_global, tol=1.0))
        trainer.train(s, x, t)
        return trainer.basis.coeffs.detach()

    def _init_gmm(self, alpha: Tensor, device: torch.device) -> tuple:
        """GMM on alpha -> initial gamma, pi, mu, Sigma."""
        alpha_np = alpha.cpu().double().numpy()
        gmm = GaussianMixture(n_components=self.cfg.M, covariance_type="full",
                              init_params='k-means++',
                              reg_covar=1e-4, max_iter=800, n_init=30)
        gmm.fit(alpha_np)

        def to_tensor(arr):
            return torch.tensor(arr, dtype=alpha.dtype, device=device)

        return (
            to_tensor(gmm.predict_proba(alpha_np)),  # (N, M)
            to_tensor(gmm.weights_),                  # (M,)
            to_tensor(gmm.means_),                    # (M, P)
            to_tensor(gmm.covariances_),              # (M, P, P)
        )

    def _init_bases(self, s: Tensor, x: Tensor, t: Tensor, kappa: Tensor) -> list:
        """Train one basis per regime with initial gamma."""
        trainers = []
        for m in range(self.cfg.M):
            trainer = self.cfg.basis_factory(s, x, self.cfg.basis_config)
            trainer.train(s, x, t, kappa, gamma=self.gamma[:, m])
            trainers.append(trainer)
        return trainers

    def _em_loop(self, s: Tensor, x: Tensor, t: Tensor, kappa: Tensor) -> None:
        """EM iterations until convergence or max_em_iters."""
        for em_iter in range(1, self.cfg.max_em_iters + 1):

            # E-step: responsibilities from reconstruction error on snapshots s
            gamma_new = self._e_step(s, x)

            # M1: mixture weights
            pi_new = gamma_new.mean(dim=0)

            # Distribution-shift criterion (uses alpha)
            mu_new, Sigma_new = self._compute_stats(gamma_new)
            delta = self._delta(mu_new, Sigma_new)

            self.gamma = gamma_new
            self.pi    = pi_new
            self.mu    = mu_new
            self.Sigma = Sigma_new

            entropy = -(gamma_new * gamma_new.clamp(min=1e-10).log()).sum(dim=1).mean().item()
            delta_str = " ".join(f"{d:.3e}" for d in delta.tolist())
            pi_str    = " ".join(f"{p:.3f}" for p in pi_new.tolist())
            print(f"EM {em_iter:3d} | delta=[{delta_str}] | pi=[{pi_str}] | H={entropy:.3f}")

            # M2: adaptive basis update per regime
            self._m2_step(s, x, t, kappa, gamma_new, delta)

            if delta.max().item() < self.cfg.eps_conv:
                print(f"  converged at iteration {em_iter}")
                break

    def _e_step(self, s: Tensor, x: Tensor) -> Tensor:
        """Posterior responsibilities gamma (N, M).

        gamma_im proportional
        Reconstruction s_hat_m_i computed in full snapshot space
        """
        log_p = torch.empty(s.shape[0], self.cfg.M, device=s.device, dtype=s.dtype)
        with torch.no_grad():
            for m, trainer in enumerate(self.trainers):
                s_hat = trainer.basis(x)                                      # (N, Ny)
                res_sq = ((s - s_hat) ** 2 * self._w[None, :]).sum(dim=1)    # (N,)
                log_p[:, m] = self.pi[m].clamp(min=1e-30).log() - res_sq / (2.0 * self.cfg.sigma2)
        log_p -= log_p.max(dim=1, keepdim=True).values
        p = log_p.exp()
        return p / p.sum(dim=1, keepdim=True)

    def _compute_stats(self, gamma: Tensor) -> tuple[Tensor, Tensor]:
        """Responsibility-weighted mean (M, P) and covariance (M, P, P) of alpha."""
        N_m = gamma.sum(dim=0).clamp(min=1e-8)                    # (M,)
        mu = (gamma.T @ self.alpha) / N_m[:, None]                 # (M, P)
        P = self.alpha.shape[1]
        Sigma = torch.zeros(self.cfg.M, P, P,
                            device=self.alpha.device, dtype=self.alpha.dtype)
        for m in range(self.cfg.M):
            diff = self.alpha - mu[m]                              # (N, P)
            A = (gamma[:, m] / N_m[m]).sqrt().unsqueeze(1) * diff  # (N, P)
            Sigma[m] = A.T @ A
        return mu, Sigma

    def _delta(self, mu_new: Tensor, Sigma_new: Tensor) -> Tensor:
        """Distribution-shift Delta_m (M,): relative change in mu and Sigma."""
        eps = 1e-8
        d_mu = (torch.norm(mu_new - self.mu, dim=1) /
                (torch.norm(self.mu, dim=1) + eps))
        d_S  = (torch.linalg.matrix_norm(Sigma_new - self.Sigma, ord='fro') /
                (torch.linalg.matrix_norm(self.Sigma, ord='fro') + eps))
        return d_mu + d_S   # (M,)

    def _m2_step(self, s: Tensor, x: Tensor, t: Tensor, kappa: Tensor,
                 gamma: Tensor, delta: Tensor) -> None:
        """M2: Skip / Incremental / Full rerun per regime based on Delta_m."""
        for m in range(self.cfg.M):
            dm      = delta[m].item()
            gamma_m = gamma[:, m]
            if dm < self.cfg.eps_skip:
                pass                                                    # Skip
            elif dm < self.cfg.eps_large:
                self._m2_incremental(m, s, x, t, kappa, gamma_m)       # Incremental
            else:
                self._m2_full_rerun(m, s, x, t, kappa, gamma_m)        # Full rerun

    def _m2_full_rerun(self, m: int, s: Tensor, x: Tensor, t: Tensor,
                       kappa: Tensor, gamma_m: Tensor) -> None:
        """Recreate basis from scratch and retrain with updated gamma."""
        trainer = self.cfg.basis_factory(s, x, self.cfg.basis_config)
        trainer.train(s, x, t, kappa, gamma=gamma_m)
        self.trainers[m] = trainer

    def _m2_incremental(self, m: int, s: Tensor, x: Tensor, t: Tensor,
                        kappa: Tensor, gamma_m: Tensor) -> None:
        """Incremental update: add/prune one mode (NeuralPOD) or retrain (POD)."""
        trainer = self.trainers[m]
        if isinstance(trainer, FourierNeuralPODTrainer):
            self._m2_incremental_fourier(trainer, s, x, gamma_m)
        else:
            # POD: SVD is globally optimal; retrain with updated gamma
            trainer.train(s, x, t, kappa, gamma=gamma_m)

    def _m2_incremental_fourier(self, trainer: FourierNeuralPODTrainer,
                                s: Tensor, x: Tensor, gamma_m: Tensor) -> None:
        """Add or prune one mode using updated responsibilities (warm start).

        Existing modes are kept. New mode added if residual norm exceeds tol.
        Last mode pruned if its weighted contribution is negligible.
        """
        gamma_m = gamma_m / gamma_m.sum()
        w = self._w

        # Recompute tolerance with updated gamma
        s_mean     = (gamma_m[:, None] * s).sum(dim=0)
        s_centered = s - s_mean[None, :]
        tol_abs    = trainer.cfg.tol * _weighted_norm_sq(s_centered, gamma_m, w)

        # Residual after all existing modes (snapshot space)
        with torch.no_grad():
            r = trainer._full_residual(s, x)
            for mode in trainer.basis.modes:
                r = trainer._update_residual(r, mode, x)

        # Mode addition
        res_norm = _weighted_norm_sq(r, gamma_m, w)
        if res_norm >= tol_abs and len(trainer.basis.modes) < trainer.cfg.max_modes:
            mode = trainer.basis.add_mode()
            trainer._train_mode(mode, r, x, gamma_m, w)
            trainer.num_modes += 1
            print(f"    added mode {trainer.num_modes} (res={res_norm:.3e})")

        # Mode pruning
        if trainer.basis.modes:
            last_mode = trainer.basis.modes[-1]
            with torch.no_grad():
                phi    = last_mode.phi(x)                                      # (Ny,)
                contrib = torch.outer(phi, last_mode.lambda_ten).T             # (N, Ny)
            if _weighted_norm_sq(contrib, gamma_m, w) < self.cfg.eps_prune:
                trainer.basis.modes = nn.ModuleList(list(trainer.basis.modes)[:-1])
                trainer.num_modes  -= 1
                print(f"    pruned last mode -> {trainer.num_modes} modes")
