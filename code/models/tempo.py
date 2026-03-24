from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Type

import torch
from torch import Tensor
from sklearn.mixture import GaussianMixture

from .pod import PODTrainer, PODConfig


@dataclass
class TEMPOConfig:
    M: int = 2
    sigma2: float = 0.1
    P_global: int = 20
    max_em_iters: int = 10
    eps_skip: float = 0.01
    eps_large: float = 0.1
    eps_conv: float = 0.005
    basis_config: Any = field(default_factory=lambda: PODConfig(max_modes=8))


class TEMPOTrainer:
    """TEMPO offline EM trainer.

    After train(): gamma (N, M), pi (M,), trainers (list of M bases)
    """

    def __init__(self, cfg: TEMPOConfig, basis_trainer_class: Type = PODTrainer) -> None:
        self.cfg = cfg
        self.basis_trainer_class = basis_trainer_class

        self.trainers: list = []
        self.gamma: Tensor = None    # (N, M)
        self.pi: Tensor = None       # (M,)
        self.alpha: Tensor = None    # (N, P_global), fixed after init
        self.mu: Tensor = None       # (M, P_global)
        self.Sigma: Tensor = None    # (M, P, P)




    def train(self, s: Tensor, x: Tensor, t: Tensor, kappa: Tensor = None) -> None:
        self._initialize(s, x, t, kappa)
        # self._em_loop(s, x, t, kappa)



    def _initialize(self, s: Tensor, x: Tensor, t: Tensor, kappa: Tensor) -> None:
        self.alpha = self._global_pod(s, x, t)
        self.gamma, self.pi, self.mu, self.Sigma = self._init_gmm(self.alpha, s.device)
        self.trainers = self._init_bases(s, x, t, kappa)




    def _global_pod(self, s: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """SVD on all snapshots, returns alpha (N, P_global)."""
        trainer = PODTrainer(PODConfig(max_modes=self.cfg.P_global, tol=1.0))
        trainer.train(s, x, t)
        return trainer.basis.coeffs.detach()




    def _init_gmm(self, alpha: Tensor, device: torch.device) -> tuple:
        """GMM on alpha -> initial gamma, pi, mu, Sigma."""
        alpha_np = alpha.cpu().double().numpy()
        gmm = GaussianMixture(n_components=self.cfg.M, covariance_type="full",
                              reg_covar=1e-4, random_state=0, max_iter=200)
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
        """Train one basis per regime with initial gamma"""
        trainers = []
        for m in range(self.cfg.M):
            trainer = self.basis_trainer_class(self.cfg.basis_config)
            trainer.train(s, x, t, kappa, gamma=self.gamma[:, m])
            trainers.append(trainer)
        return trainers
