"""
Empirical verification of the Regime Approximation Advantage (Proposition 1).

Proposition: For any regime assignments {γ_im} and any rank K,
    ε_TEMPO^K ≤ ε_global^K

where:
  ε_global^K = min_{rank-K P, μ}  Σ_i ||(I-P)(s_i - μ)||²    (global POD)
  ε_TEMPO^K  = Σ_m min_{rank-K P_m, μ_m}  Σ_i γ_im ||(I-P_m)(s_i - μ_m)||²

Both use exactly K modes at inference. The error is pure Phase-1 reconstruction
(offline basis quality), not online prediction.

Method:
  - Global POD: truncated SVD of centered snapshot matrix.
  - TEMPO M=1..5: k-means hard assignments on PCA-projected data, then
    per-regime truncated SVD. Hard assignments (γ_im ∈ {0,1}) are a special
    case covered by the proposition.
  - M=1 always equals Global POD (equality case, sanity check).

Usage:
    python code/scripts/verify_theorem.py
    python code/scripts/verify_theorem.py --K_max 40 --M_list 1 2 3 4 5
    python code/scripts/verify_theorem.py --out paper/figs/theorem_verification.png
"""

import argparse
import os
import sys
import pathlib

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "code"))

from utils.datasets import load_stacked


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _pod_error_curve(S_centered: np.ndarray, weights: np.ndarray, K_max: int) -> np.ndarray:
    """
    Computes weighted POD reconstruction error curve.

    Returns err: (K_max+1,) where err[k] = weighted reconstruction error
    when using k modes. Units: same as weights (normalised to sum=1 internally,
    then scaled back by weights.sum()).

    err[k] = weights.sum() * (weighted_total_var - Σ_{j=1}^k σ_j²)
    """
    W_sum = weights.sum()
    w = weights / W_sum                              # normalise
    Sw = (S_centered * np.sqrt(w[:, None])).astype(np.float32)
    q = min(K_max + 5, min(Sw.shape) - 1)
    _, sigma, _ = torch.svd_lowrank(torch.from_numpy(Sw), q=q, niter=4)
    sigma = sigma[:K_max].double().numpy()

    total_norm = float((Sw.astype(np.float64) ** 2).sum())  # normalised total var

    err_norm = np.empty(K_max + 1)
    err_norm[0] = total_norm
    for k in range(len(sigma)):
        err_norm[k + 1] = max(0.0, err_norm[k] - sigma[k] ** 2)
    # If fewer than K_max modes were computed, error stays flat
    for k in range(len(sigma), K_max):
        err_norm[k + 1] = err_norm[min(len(sigma), K_max)]

    return err_norm * W_sum   # scale back to absolute units


def global_pod_errors(S: np.ndarray, K_max: int) -> np.ndarray:
    """ε_global(K) = Σ_i ||(I - P_K)(s_i - mean)||²."""
    mean = S.mean(0)
    return _pod_error_curve(S - mean, np.ones(len(S)), K_max)


def tempo_pod_errors(S: np.ndarray, gamma: np.ndarray, K_max: int) -> np.ndarray:
    """
    ε_TEMPO(K) = Σ_m Σ_i γ_im ||(I - P_mK)(s_i - μ_m)||².

    gamma: (N, M) non-negative, rows sum to 1 (soft) or rows are one-hot (hard).
    """
    N, M = gamma.shape
    err_total = np.zeros(K_max + 1)
    for m in range(M):
        gm = gamma[:, m]           # (N,), Σ_i γ_im = N_m
        N_m = gm.sum()
        if N_m < 1e-9:
            continue
        mu_m = (gm[:, None] * S).sum(0) / N_m
        # _pod_error_curve returns N_m-scaled absolute error for this regime
        err_m = _pod_error_curve(S - mu_m, gm, K_max)
        err_total += err_m
    return err_total


def kmeans_gamma(S: np.ndarray, M: int, n_pca: int = 20, seed: int = 0) -> np.ndarray:
    """Hard (one-hot) assignments from k-means on top PCA components."""
    if M == 1:
        return np.ones((len(S), 1))
    svd = TruncatedSVD(n_components=min(n_pca, min(S.shape) - 1), random_state=seed)
    Z = svd.fit_transform(S - S.mean(0))
    labels = KMeans(n_clusters=M, random_state=seed, n_init=10).fit_predict(Z)
    gamma = np.zeros((len(S), M))
    gamma[np.arange(len(S)), labels] = 1.0
    return gamma


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PALETTE = ["#000000", "#2166ac", "#d6604d", "#4dac26", "#762a83", "#e08214"]

def make_figure(
    err_global: np.ndarray,
    tempo_errors: dict,    # {M: err_array}
    out_path: str,
):
    K_max = len(err_global) - 1
    Ks = np.arange(K_max + 1)
    total_g = err_global[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plt.rcParams.update({"font.size": 11})

    # ---- Left: normalised error curves (log scale) ----
    ax = axes[0]
    ax.semilogy(Ks, err_global / total_g, "--", color=PALETTE[0],
                lw=2.2, label="Global POD ($M{=}1$)")
    for i, (M, err) in enumerate(sorted(tempo_errors.items())):
        if M == 1:
            continue
        color = PALETTE[i % len(PALETTE)]
        ax.semilogy(Ks, err / total_g, "-", color=color, lw=2.0,
                    label=f"TEMPO $M={M}$")
        ax.fill_between(Ks, err / total_g, err_global / total_g,
                        alpha=0.08, color=color)

    ax.set_xlabel("Number of modes $K$", fontsize=12)
    ax.set_ylabel(r"$\varepsilon^K\;/\;\varepsilon^0_{\mathrm{global}}$", fontsize=12)
    ax.set_title(
        r"Prop. 1: $\varepsilon^K_{\mathrm{TEMPO}} \leq \varepsilon^K_{\mathrm{global}}$",
        fontsize=12,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- Right: relative gain (%) ----
    ax2 = axes[1]
    for i, (M, err) in enumerate(sorted(tempo_errors.items())):
        if M == 1:
            continue
        gain = (err_global - err) / (err_global + 1e-30) * 100
        color = PALETTE[i % len(PALETTE)]
        ax2.plot(Ks[1:], gain[1:], "-", color=color, lw=2.0,
                 label=f"$M={M}$")
        ax2.fill_between(Ks[1:], 0, np.maximum(gain[1:], 0),
                         alpha=0.10, color=color)

    ax2.axhline(y=0.0, color="k", lw=1.0, ls="--", alpha=0.6)
    ax2.set_xlabel("Number of modes $K$", fontsize=12)
    ax2.set_ylabel("Relative gain (%)", fontsize=12)
    ax2.set_title("Regime decomposition advantage", fontsize=12)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle(
        "1D Burgers' equation — $k$-means regime assignments",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=os.path.expanduser("~/data/1D/Burgers/Train"))
    p.add_argument("--nu_values", type=float, nargs="+", default=[0.001, 0.1, 1.0])
    p.add_argument("--n_samples", type=int, default=500,
                   help="Samples per nu value (total N = n_samples * len(nu_values))")
    p.add_argument("--M_list", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                   help="Regime counts to test. M=1 == Global POD (equality case).")
    p.add_argument("--K_max", type=int, default=32,
                   help="Maximum number of modes to evaluate.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(_ROOT / "paper" / "figs" / "theorem_verification.png"))
    return p.parse_args()


def main():
    args = parse_args()

    # --- Load data ---
    entries = [
        (nu, os.path.join(args.data_dir, f"1D_Burgers_Sols_Nu{nu}.hdf5"))
        for nu in args.nu_values
    ]
    print(f"Loading Burgers data ({args.n_samples} per nu × {len(args.nu_values)} nu)...")
    S, *_ = load_stacked(entries, n_samples=args.n_samples)
    S = S.astype(np.float64)
    N = len(S)
    print(f"  S shape: {S.shape}")

    K_max = min(args.K_max, N - 1, S.shape[1] - 1)

    # --- Global POD ---
    print("Computing global POD error curve...")
    err_g = global_pod_errors(S, K_max)

    # --- TEMPO for each M via k-means ---
    tempo_errors = {}
    for M in args.M_list:
        print(f"Computing TEMPO M={M} (k-means)...")
        gamma = kmeans_gamma(S, M, seed=args.seed)
        err = tempo_pod_errors(S, gamma, K_max)
        tempo_errors[M] = err

        gap = err_g - err
        viol = int((gap[1:] < -1e-8).sum())
        gain_at10 = float((gap / (err_g + 1e-30) * 100)[min(10, K_max)])
        print(f"  M={M}: gain@K=10: {gain_at10:+.1f}%   violations: {viol}/{K_max}")

    # --- Print summary table ---
    print(f"\n{'K':>4}", end="")
    print(f"  {'ε_global':>10}", end="")
    for M in sorted(tempo_errors):
        print(f"  {'ε_TEMPO M='+str(M):>13}", end="")
    print()

    for k in [1, 2, 5, 10, 15, 20, K_max]:
        if k > K_max:
            continue
        print(f"{k:>4}  {err_g[k]:>10.2f}", end="")
        for M in sorted(tempo_errors):
            gain = (err_g[k] - tempo_errors[M][k]) / (err_g[k] + 1e-30) * 100
            print(f"  {tempo_errors[M][k]:>10.2f} ({gain:+.1f}%)", end="")
        print()

    # --- Figure ---
    make_figure(err_g, tempo_errors, args.out)


if __name__ == "__main__":
    main()
