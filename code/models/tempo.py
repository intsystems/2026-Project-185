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
    heteroscedastic: bool = True   # use relative error in E-step (robust to multi-scale data)
    kappa_init: bool = False       # initialize regimes from log(kappa) instead of GMM on alpha
    kappa_prior_weight: float = 1.0  # lambda: weight of beta-prior in E-step log_p
    kappa_init_noise: float = 0.05  # uniform noise added to initial gamma to break symmetry
    basis_config: Any = field(default_factory=lambda: PODConfig(max_modes=8))
    basis_factory: Callable = field(default_factory=lambda: pod_factory)


class TEMPOTrainer:
    """TEMPO offline EM trainer.

    After train(): gamma (N, M), pi (M,), trainers (list of M bases)
    """

    def __init__(self, cfg: TEMPOConfig) -> None:
        self.cfg = cfg
        self.trainers: list = []
        self.gamma: Tensor = None      # (N, M)
        self.pi: Tensor = None         # (M,)
        self.alpha: Tensor = None      # (N, P_global) — normalized when heteroscedastic
        self.mu: Tensor = None         # (M, P_global)
        self.Sigma: Tensor = None      # (M, P_global, P_global)
        self._w: Tensor = None         # (Ny,) quadrature weights
        self._s_norm_sq: Tensor = None # (N,) on CPU: ||s^(i)||^2_w per sample
        self._kappa_centers: Tensor = None  # (M,) log-kappa regime centers (CPU)
        self._kappa_T: float = None         # temperature for beta-prior in E-step
        self.history_phase1: dict = None

    def train(self, s: Tensor, x: Tensor, t: Tensor, kappa: Tensor = None) -> None:
        self._initialize(s, x, t, kappa)
        self._em_loop(s, x, t, kappa)

    def _initialize(self, s: Tensor, x: Tensor, t: Tensor, kappa: Tensor) -> None:
        Ny = s.shape[1]
        self._w = torch.ones(Ny, device=s.device, dtype=s.dtype) / Ny
        w_cpu = self._w.cpu()
        self._s_norm_sq = (s.cpu() ** 2 * w_cpu[None, :]).sum(dim=1).clamp(min=1e-10)
        print("=== Phase 1: global POD ===")
        self.alpha = self._global_pod(s, x, t)
        if self.cfg.heteroscedastic:
            self.alpha = self.alpha / self.alpha.norm(dim=1, keepdim=True).clamp(min=1e-10)
        print("=== Phase 2: GMM init ===")
        if self.cfg.kappa_init and kappa is not None:
            self.gamma, self.pi, self.mu, self.Sigma = self._init_gmm_kappa(
                self.alpha, kappa, s.device)
        else:
            self.gamma, self.pi, self.mu, self.Sigma = self._init_gmm(self.alpha, s.device)
        print(f"  pi={[f'{p:.3f}' for p in self.pi.tolist()]}")
        print("=== Phase 3: regime bases init ===")
        self.trainers = self._init_bases(s, x, t, kappa)
        print("=== Phase 4: calibrate sigma2 ===")
        self._calibrate_sigma2(s, x)

    def _calibrate_sigma2(self, s: Tensor, x: Tensor) -> None:
        """Set sigma2 from initial reconstructions (relative error when heteroscedastic)."""
        N, total = s.shape[0], 0.0
        dev = x.device
        w = self._w.to(dev)
        with torch.no_grad():
            for m, trainer in enumerate(self.trainers):
                gamma_m = self.gamma[:, m].to(dev)
                mean, modes, coeffs = self._basis_components(trainer, x, dev)
                for i in range(0, N, 256):
                    s_hat = mean + coeffs[i:i+256].to(dev) @ modes.T
                    res_sq = ((s[i:i+256].to(dev) - s_hat) ** 2 * w[None, :]).sum(dim=1)
                    if self.cfg.heteroscedastic:
                        res_sq = res_sq / self._s_norm_sq[i:i+256].to(dev)
                    total += (gamma_m[i:i+256] * res_sq).sum().item()
        self.cfg.sigma2 = 0.5 * total / N
        print(f"  calibrated sigma2 = {self.cfg.sigma2:.4e}")

    def _global_pod(self, s: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """SVD on all snapshots, returns alpha (N, P_global)."""
        trainer = PODTrainer(PODConfig(max_modes=self.cfg.P_global, tol=1.0))
        trainer.train(s, x, t)
        return trainer.basis.coeffs.detach()

    def _init_gmm(self, alpha: Tensor, device: torch.device) -> tuple:
        """GMM on alpha: returns (gamma, pi, mu, Sigma)."""
        alpha_np = alpha.cpu().double().numpy()
        gmm = GaussianMixture(n_components=self.cfg.M, covariance_type="full",
                              init_params='k-means++',
                              reg_covar=1e-4, max_iter=800, n_init=30)
        gmm.fit(alpha_np)

        def to_tensor(arr):
            return torch.tensor(arr, dtype=alpha.dtype, device=device)

        return (
            to_tensor(gmm.predict_proba(alpha_np)),  # (N, M)
            to_tensor(gmm.weights_),  # (M,)
            to_tensor(gmm.means_),  # (M, P)
            to_tensor(gmm.covariances_),  # (M, P, P)
        )

    def _init_gmm_kappa(self, alpha: Tensor, kappa: Tensor, device: torch.device) -> tuple:
        """Physics-informed init: soft-assign regimes from log(kappa) spacing.

        Regime centers are placed at M equally-spaced quantiles of log(kappa).
        Temperature T = 0.5 * (log_max - log_min) / max(M - 1, 1) so adjacent
        regimes have ~50% overlap — soft but not uniform.
        """
        log_k = kappa.float().cpu().squeeze(-1).log()    # (N,) always on CPU

        # Use unique kappa values as regime centers; fall back to linspace if counts differ
        unique_log = torch.unique(log_k).sort().values   # (K,)
        K = unique_log.shape[0]
        M = self.cfg.M
        if K == M:
            centers = unique_log
        elif K > M:
            idx = torch.linspace(0, K - 1, M).long()
            centers = unique_log[idx]
        else:
            centers = torch.linspace(unique_log[0].item(), unique_log[-1].item(), M)

        span = max(centers[-1].item() - centers[0].item(), 1e-6)
        inter = span / max(M - 1, 1)   # spacing between adjacent centers
        T = 0.2 * inter

        # Soft assignment: gamma[i,m] = softmax over -0.5*(log_k_i - c_m)^2/T^2
        dist_sq = (log_k[:, None] - centers[None, :]) ** 2    # (N, M) on CPU
        log_gamma = -0.5 * dist_sq / (T ** 2)
        log_gamma = log_gamma - log_gamma.logsumexp(dim=1, keepdim=True)
        gamma = log_gamma.exp().to(dtype=alpha.dtype, device=device)  # (N, M) -> device

        pi = gamma.mean(dim=0)

        # Weighted alpha statistics for delta tracking
        N_m = gamma.sum(dim=0).clamp(min=1e-8)
        mu = (gamma.T @ alpha.to(device)) / N_m[:, None]
        P = alpha.shape[1]
        Sigma = torch.zeros(self.cfg.M, P, P, device=device, dtype=alpha.dtype)
        for m in range(self.cfg.M):
            diff = alpha.to(device) - mu[m]
            A = (gamma[:, m] / N_m[m]).sqrt().unsqueeze(1) * diff
            Sigma[m] = A.T @ A

        self._kappa_centers = centers   # (M,) on CPU, stored for E-step prior
        self._kappa_T = T

        # small perturbation so EM runs at least a few real iterations
        if self.cfg.kappa_init_noise > 0.0:
            noise = torch.rand_like(gamma) * self.cfg.kappa_init_noise
            gamma = gamma + noise
            gamma = gamma / gamma.sum(dim=1, keepdim=True)

        print(f"  kappa_init: centers=[{', '.join(f'{c:.3g}' for c in centers.exp().tolist())}]  T={T:.3f}  noise={self.cfg.kappa_init_noise}")
        return gamma, pi, mu, Sigma

    def _init_bases(self, s: Tensor, x: Tensor, t: Tensor, kappa: Tensor) -> list:
        """Train one basis per regime with initial gamma."""
        trainers = []
        for m in range(self.cfg.M):
            print(f"  regime {m + 1}/{self.cfg.M}")
            trainer = self.cfg.basis_factory(s, x, self.cfg.basis_config)
            gamma_m = self.gamma[:, m]
            if self.cfg.heteroscedastic:
                gamma_m = gamma_m / self._s_norm_sq.to(gamma_m.device)
            trainer.train(s, x, t, kappa, gamma=gamma_m)
            trainers.append(trainer)
        return trainers

    def _em_loop(self, s: Tensor, x: Tensor, t: Tensor, kappa: Tensor) -> None:
        """EM iterations until convergence or max_em_iters."""
        N = s.shape[0]
        P = self.alpha.shape[1]
        # BIC penalty: GMM parameters (means + covariances + mixing weights)
        k_bic = self.cfg.M * P + self.cfg.M * P * (P + 1) // 2 + (self.cfg.M - 1)

        log: dict = {'iter': [], 'll': [], 'bic': [], 'entropy': [],
                     'delta': [], 'delta_max': [], 'pi': []}

        for em_iter in range(1, self.cfg.max_em_iters + 1):
            print(f"\n{'─' * 65}")
            print(f"E-step {em_iter}...")
            gamma_new, ll = self._e_step(s, x, kappa)

            # M1: mixture weights
            pi_new = gamma_new.mean(dim=0)

            # Distribution-shift criterion (uses alpha)
            mu_new, Sigma_new = self._compute_stats(gamma_new)
            delta = self._delta(mu_new, Sigma_new)

            self.gamma = gamma_new
            self.pi = pi_new
            self.mu = mu_new
            self.Sigma = Sigma_new

            entropy = -(gamma_new * gamma_new.clamp(min=1e-10).log()).sum(dim=1).mean().item()
            bic = -2.0 * ll + k_bic * torch.tensor(N).float().log().item()
            delta_str = " ".join(f"{d:.3e}" for d in delta.tolist())
            pi_str = " ".join(f"{p:.3f}" for p in pi_new.tolist())
            print(f"EM {em_iter:3d} | LL={ll:.4e} | BIC={bic:.4e} | H={entropy:.3f}")
            print(f"       | delta=[{delta_str}] | pi=[{pi_str}]")

            log['iter'].append(em_iter)
            log['ll'].append(ll)
            log['bic'].append(bic)
            log['entropy'].append(entropy)
            log['delta'].append(delta.tolist())
            log['delta_max'].append(delta.max().item())
            log['pi'].append(pi_new.tolist())

            print("M2-step: updating bases...")
            self._m2_step(s, x, t, kappa, gamma_new, delta)

            if delta.max().item() < self.cfg.eps_conv:
                print(f"\n  converged at iteration {em_iter}")
                break

        self.history_phase1 = log

    def _basis_components(self, trainer, x: Tensor, dev: torch.device):
        """Extract (mean, modes, coeffs) on dev for batched evaluation.

        Returns:
            mean:   (Ny,)   weighted mean
            modes:  (Ny, P) POD modes or stacked Fourier phis
            coeffs: (N, P)  POD coeffs or stacked lambda_ten
        """
        if isinstance(trainer, PODTrainer):
            return (
                trainer.basis.mean.to(dev),
                trainer.basis.modes.to(dev),
                trainer.basis.coeffs.to(dev),
            )
        # FourierNeuralPODTrainer
        x_dev = x.to(dev)
        mean = trainer.basis.mean_net(x_dev)
        modes = torch.stack([md.phi(x_dev) for md in trainer.basis.modes], dim=1)
        coeffs = torch.stack([md.lambda_ten for md in trainer.basis.modes], dim=1).to(dev)
        return mean, modes, coeffs

    def _e_step(self, s: Tensor, x: Tensor, kappa: Tensor = None) -> tuple[Tensor, float]:
        """Posterior responsibilities gamma (N, M) and log-likelihood, batched over N.

        Returns:
            gamma: (N, M) soft assignments on s.device
            ll:    scalar log-likelihood sum_i log sum_m pi_m p(s_i | m)
        """
        N = s.shape[0]
        dev = x.device  # run matmuls on GPU; s stays on CPU
        log_p = torch.empty(N, self.cfg.M, device=dev, dtype=s.dtype)
        w = self._w.to(dev)
        with torch.no_grad():
            for m, trainer in enumerate(self.trainers):
                mean, modes, coeffs = self._basis_components(trainer, x, dev)
                log_pi = self.pi[m].clamp(min=1e-30).log().to(dev)
                for i in range(0, N, 256):
                    s_hat = mean + coeffs[i:i+256].to(dev) @ modes.T
                    res_sq = ((s[i:i+256].to(dev) - s_hat) ** 2 * w[None, :]).sum(dim=1)
                    if self.cfg.heteroscedastic:
                        res_sq = res_sq / self._s_norm_sq[i:i+256].to(dev)
                    log_p[i:i+256, m] = log_pi - res_sq / (2.0 * self.cfg.sigma2)

            # beta-prior: log p(z=m | beta_i) added to each column
            if self.cfg.kappa_init and kappa is not None and self._kappa_centers is not None:
                log_k = kappa.float().cpu().squeeze(-1).log()  # (N,) on CPU
                centers = self._kappa_centers                  # (M,) on CPU
                T = self._kappa_T
                log_prior = -0.5 * (log_k[:, None] - centers[None, :]) ** 2 / (T ** 2)
                log_prior = log_prior - log_prior.logsumexp(dim=1, keepdim=True)  # normalize
                log_p += self.cfg.kappa_prior_weight * log_prior.to(dev)

        log_max = log_p.max(dim=1, keepdim=True).values
        p = (log_p - log_max).exp()
        ll = (log_max.squeeze(1) + p.sum(dim=1).log()).sum().item()
        gamma = p / p.sum(dim=1, keepdim=True)
        return gamma.to(s.device), ll

    def _compute_stats(self, gamma: Tensor) -> tuple[Tensor, Tensor]:
        """Responsibility-weighted mean (M, P) and covariance (M, P, P) of alpha."""
        N_m = gamma.sum(dim=0).clamp(min=1e-8)  # (M,)
        mu = (gamma.T @ self.alpha) / N_m[:, None]  # (M, P)
        P = self.alpha.shape[1]
        Sigma = torch.zeros(self.cfg.M, P, P,
                            device=self.alpha.device, dtype=self.alpha.dtype)
        for m in range(self.cfg.M):
            diff = self.alpha - mu[m]  # (N, P)
            A = (gamma[:, m] / N_m[m]).sqrt().unsqueeze(1) * diff  # (N, P)
            Sigma[m] = A.T @ A
        return mu, Sigma

    def _delta(self, mu_new: Tensor, Sigma_new: Tensor) -> Tensor:
        """Distribution-shift Delta_m (M,): relative change in mu and Sigma."""
        eps = 1e-8
        d_mu = (torch.norm(mu_new - self.mu, dim=1) /
                (torch.norm(self.mu, dim=1) + eps))
        d_S = (torch.linalg.matrix_norm(Sigma_new - self.Sigma, ord='fro') /
                (torch.linalg.matrix_norm(self.Sigma, ord='fro') + eps))
        return d_mu + d_S  # (M,)

    def _m2_step(self, s: Tensor, x: Tensor, t: Tensor, kappa: Tensor,
                 gamma: Tensor, delta: Tensor) -> None:
        """M2: Skip / Incremental / Full rerun per regime based on Delta_m."""
        s_norm_sq = self._s_norm_sq.to(gamma.device) if self.cfg.heteroscedastic else None
        for m in range(self.cfg.M):
            dm = delta[m].item()
            gamma_m = gamma[:, m]
            # heteroscedastic: weight by gamma/||s||^2 so basis fits relative error
            gamma_m_eff = gamma_m / s_norm_sq if self.cfg.heteroscedastic else gamma_m
            if dm < self.cfg.eps_skip:
                print(f"  regime {m + 1}: skip (delta={dm:.3e})")
            elif dm < self.cfg.eps_large:
                print(f"  regime {m + 1}: incremental (delta={dm:.3e})")
                self._m2_incremental(m, s, x, t, kappa, gamma_m_eff)
            else:
                print(f"  regime {m + 1}: full rerun (delta={dm:.3e})")
                self._m2_full_rerun(m, s, x, t, kappa, gamma_m_eff)

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
        """Add one mode if residual exceeds tol; prune last mode if its contribution is negligible."""
        gamma_m = gamma_m / gamma_m.sum().clamp(min=1e-10)
        gamma_cpu = gamma_m.cpu()
        w = self._w

        # Recompute tolerance with updated gamma
        s_mean = (gamma_cpu[:, None] * s.cpu()).sum(dim=0)
        s_centered = s.cpu() - s_mean[None, :]
        w_cpu = w.cpu()
        tol_abs = trainer.cfg.tol * _weighted_norm_sq(s_centered, gamma_cpu, w_cpu)

        # Residual after all existing modes, on CPU
        with torch.no_grad():
            r = trainer._full_residual(s, x)
            for mode in trainer.basis.modes:
                r = trainer._update_residual(r, mode, x)

        # Mode addition
        res_norm = _weighted_norm_sq(r, gamma_cpu, w_cpu)
        if res_norm >= tol_abs and len(trainer.basis.modes) < trainer.cfg.max_modes:
            mode = trainer.basis.add_mode()
            trainer._train_mode(mode, r, x, gamma_m.to(trainer._device), gamma_cpu, w.to(trainer._device))
            trainer.num_modes += 1
            print(f"    added mode {trainer.num_modes} (res={res_norm:.3e})")

        # Mode pruning
        if trainer.basis.modes:
            last_mode = trainer.basis.modes[-1]
            with torch.no_grad():
                phi = last_mode.phi(x)  # (Ny,)
                contrib = torch.outer(phi, last_mode.lambda_ten).T  # (N, Ny)
            if _weighted_norm_sq(contrib.cpu(), gamma_cpu, w_cpu) < self.cfg.eps_prune:
                trainer.basis.modes = nn.ModuleList(list(trainer.basis.modes)[:-1])
                trainer.num_modes -= 1
                print(f"    pruned last mode, now {trainer.num_modes} modes")
