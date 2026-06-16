#!/usr/bin/env python
"""Generate POD and Fourier-NeuralPOD basis visualisation figures (paper appendix).

Outputs written to --out_dir:
  basis_spectrum.png       POD singular-value decay  +  FNPOD residual decay
  basis_modes.png          First K mode heatmaps, POD (top) vs FNPOD (bottom)
  basis_reconstruction.png n_viz examples: GT / POD / FNPOD / errors
"""
import argparse
import os
import pathlib
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from models.pod import PODTrainer, PODConfig
from models.regime_basis import FourierRegimeBasis
from models.fourier_neural_pod import FourierNeuralPODTrainer, FourierNeuralPODConfig

_ROOT = pathlib.Path(__file__).resolve().parents[2]

C0, C1 = plt.cm.tab10(0), plt.cm.tab10(1)


# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nu",            type=float, default=1.0)
    p.add_argument("--n_train",       type=int,   default=9000)
    p.add_argument("--max_modes",     type=int,   default=32)
    p.add_argument("--n_modes_viz",   type=int,   default=8,
                   help="modes shown per row in basis_modes.png")
    p.add_argument("--n_viz",         type=int,   default=3,
                   help="reconstruction examples")
    p.add_argument("--n_epochs_mean", type=int,   default=800)
    p.add_argument("--n_epochs_mode", type=int,   default=1200)
    p.add_argument("--hidden_dim",    type=int,   default=256)
    p.add_argument("--n_layers",      type=int,   default=3)
    p.add_argument("--num_freq",      type=int,   default=96)
    p.add_argument("--scales",        type=float, nargs="+", default=[0.5, 2.0, 6.0])
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--out_dir",       type=str,   default=str(_ROOT / "paper" / "figs"))
    return p.parse_args()


# Data

def _data_path(nu):
    local  = _ROOT / "data" / f"Burgers_Nu{nu}.hdf5"
    server = pathlib.Path(os.path.expanduser("~/data/1D/Burgers/Train")) / f"1D_Burgers_Sols_Nu{nu}.hdf5"
    return str(local) if local.exists() else str(server)


def load_data(nu, n_train):
    path = _data_path(nu)
    print(f"data: {path}")
    with h5py.File(path, "r") as f:
        raw  = f["tensor"][:n_train]
        x_np = f["x-coordinate"][:]
        t_np = f["t-coordinate"][:]
    if raw.ndim == 4:
        raw = raw[..., 0]
    N, Nt, Nx = raw.shape
    return raw, x_np, t_np[:Nt], N, Nt, Nx


# Reconstruction helpers

@torch.no_grad()
def pod_reconstruct(trainer_pod, idx):
    mean   = trainer_pod.basis.mean.cpu()               # (Ny,)
    modes  = trainer_pod.basis.modes.cpu()              # (Ny, P)
    coeffs = trainer_pod.basis.coeffs[idx].cpu()       # (n, P)
    return (mean.unsqueeze(0) + coeffs @ modes.T).numpy()  # (n, Ny)


@torch.no_grad()
def fnpod_reconstruct(trainer_fnpod, idx, x_flat, device):
    x_dev = x_flat.to(device)
    rec = trainer_fnpod.basis.mean_net(x_dev).cpu()    # (Ny,)
    rec = rec.unsqueeze(0).expand(len(idx), -1).clone()
    for mode in trainer_fnpod.basis.modes:
        phi = mode.phi(x_dev).cpu()                    # (Ny,)
        lam = mode.lambda_ten[idx].cpu()               # (n,)
        rec = rec + torch.outer(lam, phi)
    return rec.numpy()                                 # (n, Ny)


# Figures

def plot_spectrum(trainer_pod, trainer_fnpod, save_path):
    coeffs_np = trainer_pod.basis.coeffs.cpu().numpy()
    sigmas    = np.sqrt((coeffs_np ** 2).sum(axis=0))
    sigmas   /= sigmas[0]

    res_fnpod = np.array(trainer_fnpod.history.residual_norms, dtype=float)
    res_fnpod /= res_fnpod[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    ax = axes[0]
    ax.semilogy(range(1, len(sigmas) + 1), sigmas, "o-", color=C0, ms=3, lw=1.5)
    ax.set_xlabel("Mode"); ax.set_ylabel("Normalised singular value")
    ax.set_title("POD — singular value decay", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.semilogy(range(1, len(res_fnpod) + 1), res_fnpod, "o-", color=C1, ms=3, lw=1.5)
    ax.set_xlabel("Mode"); ax.set_ylabel("Normalised weighted residual")
    ax.set_title("FNPOD — residual decay", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {save_path}")


def plot_modes(trainer_pod, trainer_fnpod, x_flat, device, Nt, Nx, n_modes_viz, save_path):
    K = min(n_modes_viz, trainer_pod.basis.num_modes, len(trainer_fnpod.basis.modes))
    x_dev = x_flat.to(device)

    pod_modes  = trainer_pod.basis.modes.cpu().numpy()  # (Ny, P)

    with torch.no_grad():
        fnpod_modes = [
            trainer_fnpod.basis.modes[k].phi(x_dev).cpu().numpy()
            for k in range(K)
        ]

    fig, axes = plt.subplots(2, K, figsize=(2.2 * K, 4.2))
    for k in range(K):
        phi_pod   = pod_modes[:, k].reshape(Nt, Nx)
        phi_fnpod = fnpod_modes[k].reshape(Nt, Nx)

        for row, phi, label, color in [(0, phi_pod, "POD", C0), (1, phi_fnpod, "FNPOD", C1)]:
            ax  = axes[row, k]
            vmax = np.abs(phi).max()
            ax.imshow(phi, aspect="auto", origin="lower", cmap="RdBu",
                      vmin=-vmax, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"$\\phi_{{{k+1}}}$", fontsize=9)
            if k == 0:
                ax.set_ylabel(label, fontsize=9, color=color, fontweight="bold")

    fig.text(0.5, -0.01, "space  →", ha="center", fontsize=8)
    fig.text(-0.01, 0.5, "← time", va="center", rotation="vertical", fontsize=8)
    plt.suptitle("Basis modes (space–time, first K)", fontweight="bold", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {save_path}")


def plot_reconstruction(s_raw, trainer_pod, trainer_fnpod, x_flat, device,
                        x_np, t_np, Nt, Nx, n_viz, seed, nu, save_path):
    rng  = np.random.default_rng(seed)
    N    = s_raw.shape[0]
    idx  = torch.tensor(rng.choice(N, size=n_viz, replace=False))

    s_true    = s_raw[idx.numpy()].reshape(n_viz, Nt, Nx)
    s_pod     = pod_reconstruct(trainer_pod, idx).reshape(n_viz, Nt, Nx)
    s_fnpod   = fnpod_reconstruct(trainer_fnpod, idx, x_flat, device).reshape(n_viz, Nt, Nx)

    cols  = ["Ground truth", "POD", "|POD error|", "FNPOD", "|FNPOD error|"]
    ext   = [x_np.min(), x_np.max(), t_np.min(), t_np.max()]

    fig, axes = plt.subplots(n_viz, 5, figsize=(14, 2.8 * n_viz))
    if n_viz == 1:
        axes = axes[None, :]

    for row in range(n_viz):
        gt, rp, rf = s_true[row], s_pod[row], s_fnpod[row]
        vmax = np.abs(gt).max()
        ep, ef = np.abs(gt - rp), np.abs(gt - rf)
        emax = max(ep.max(), ef.max())

        panels = [
            (gt, "RdBu_r",  -vmax, vmax),
            (rp, "RdBu_r",  -vmax, vmax),
            (ep, "Oranges",  0,    emax),
            (rf, "RdBu_r",  -vmax, vmax),
            (ef, "Oranges",  0,    emax),
        ]
        rl2_pod  = np.linalg.norm(gt - rp) / np.linalg.norm(gt)
        rl2_fnpod = np.linalg.norm(gt - rf) / np.linalg.norm(gt)

        for col, (arr, cmap, vmin, vm) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap,
                      extent=ext, vmin=vmin, vmax=vm)
            if row == 0:
                ax.set_title(cols[col], fontsize=9, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

        axes[row, 0].set_ylabel(f"sample {idx[row].item()}", fontsize=8)
        axes[row, 1].set_xlabel(f"rl2={rl2_pod:.3f}", fontsize=8)
        axes[row, 3].set_xlabel(f"rl2={rl2_fnpod:.3f}", fontsize=8)

    plt.suptitle(f"Phase 1 reconstruction — POD vs FNPOD  ($\\nu={nu}$)",
                 fontweight="bold", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {save_path}")


# Main

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device={device}")

    raw, x_np, t_np, N, Nt, Nx = load_data(args.nu, args.n_train)
    Ny = Nt * Nx

    t_grid = torch.tensor(t_np, dtype=torch.float32)
    x_grid = torch.tensor(x_np, dtype=torch.float32)
    tt, xx = torch.meshgrid(t_grid, x_grid, indexing="ij")
    x_flat = torch.stack([xx.flatten(), tt.flatten()], dim=1)  # (Ny, 2)

    s_traj = torch.tensor(raw.reshape(N, Ny), dtype=torch.float32)

    print("=== POD ===")
    trainer_pod = PODTrainer(PODConfig(max_modes=args.max_modes))
    trainer_pod.train(s_traj.to(device), x=None, t=None)

    print("=== Fourier NeuralPOD ===")
    w     = torch.ones(Ny, dtype=torch.float32, device=device) / Ny
    basis = FourierRegimeBasis(
        d_x=2, M=N, quad_weights=w,
        hidden_dim=args.hidden_dim,
        num_frequencies=args.num_freq,
        scales=args.scales,
        n_layers=args.n_layers,
    ).to(device)
    cfg_fnpod = FourierNeuralPODConfig(
        max_modes=args.max_modes,
        n_epochs_mean=args.n_epochs_mean,
        n_epochs_mode=args.n_epochs_mode,
    )
    trainer_fnpod = FourierNeuralPODTrainer(basis, cfg_fnpod)
    trainer_fnpod.train(s_traj, x_flat.to(device), t=None)

    tag = f"nu{args.nu}"
    plot_spectrum(
        trainer_pod, trainer_fnpod,
        os.path.join(args.out_dir, f"basis_spectrum_{tag}.png"),
    )
    plot_modes(
        trainer_pod, trainer_fnpod, x_flat, device, Nt, Nx,
        args.n_modes_viz,
        os.path.join(args.out_dir, f"basis_modes_{tag}.png"),
    )
    plot_reconstruction(
        raw, trainer_pod, trainer_fnpod, x_flat, device,
        x_np, t_np, Nt, Nx, args.n_viz, args.seed, args.nu,
        os.path.join(args.out_dir, f"basis_reconstruction_{tag}.png"),
    )


if __name__ == "__main__":
    main()
