#!/usr/bin/env python
"""Train NeuralPOD-DeepONet on 1D Burgers - specialist (single nu) or joint (multiple nu).

Usage:
  Specialist: python train_neural_pod.py --nu_values 0.001
  Joint:      python train_neural_pod.py --nu_values 0.001 0.01 0.1 1.0
"""
import argparse
import json
import os
import pathlib
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from models.regime_basis import FourierRegimeBasis
from models.fourier_neural_pod import FourierNeuralPODTrainer, FourierNeuralPODConfig
from models.pod_deeponet import BranchNet
from utils.datasets import load_stacked, measure_inference_time


def rel_l2(true, pred):
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


_SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nu_values",       type=float, nargs="+", required=True,
                   help="One value = specialist; multiple = joint")
    p.add_argument("--run_name",        type=str,   default=None)
    p.add_argument("--results_dir",     type=str,   default=str(_PROJECT_ROOT / "TEMPO_results" / "burgers"))
    p.add_argument("--n_samples",       type=int,   default=9500,
                   help="Samples per nu loaded (train + test)")
    p.add_argument("--n_test_per_nu",   type=int,   default=1000)
    p.add_argument("--data_dir",        type=str,   default=os.path.expanduser("~/data/1D/Burgers/Train"))
    # Fourier basis
    p.add_argument("--max_modes",       type=int,   default=32)
    p.add_argument("--n_epochs_mean",   type=int,   default=800)
    p.add_argument("--n_epochs_mode",   type=int,   default=1200)
    p.add_argument("--hidden_dim_basis", type=int,  default=256)
    p.add_argument("--num_frequencies", type=int,   default=96)
    p.add_argument("--scales",          type=float, nargs="+", default=[0.5, 2.0, 6.0])
    p.add_argument("--n_layers_basis",  type=int,   default=3)
    # Branch network
    p.add_argument("--hidden_dim",      type=int,   default=256)
    p.add_argument("--n_layers",        type=int,   default=4)
    p.add_argument("--sensor_stride",   type=int,   default=1)
    # Training
    p.add_argument("--n_epochs",        type=int,   default=80000)
    p.add_argument("--batch_size",      type=int,   default=1024)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--log_every",       type=int,   default=1000)
    # Misc
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--n_viz",           type=int,   default=3)
    p.add_argument("--half_time",       action="store_true",
                   help="Use only first half of time steps")
    return p.parse_args()


def main():
    args = parse_args()

    joint = len(args.nu_values) > 1
    _suffix = "_half" if args.half_time else ""
    if joint:
        RUN_NAME = args.run_name or f"npod_deeponet_joint_burgers{_suffix}_v1"
    else:
        RUN_NAME = args.run_name or f"npod_deeponet_nu{args.nu_values[0]}{_suffix}_v1"
    RUN_DIR = os.path.join(args.results_dir, RUN_NAME)
    os.makedirs(RUN_DIR, exist_ok=True)

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    print(f"device={DEVICE}  joint={joint}  run_dir={os.path.abspath(RUN_DIR)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    C0, C1, C2 = plt.cm.tab10(0), plt.cm.tab10(1), plt.cm.tab10(2)

    data_dir = pathlib.Path(args.data_dir)
    entries = []
    for nu in args.nu_values:
        _local  = _PROJECT_ROOT / "data" / f"Burgers_Nu{nu}.hdf5"
        _server = data_dir / f"1D_Burgers_Sols_Nu{nu}.hdf5"
        path = str(_local) if _local.exists() else str(_server)
        entries.append((nu, path))

    s_np, kappa_np, x_np, t_np, Nx, Nt_full = load_stacked(entries, n_samples=args.n_samples)
    # s_np: (N_total, Nt_full*Nx), Nt-slow convention

    if args.half_time:
        Nt = Nt_full // 2
        s_np = s_np.reshape(-1, Nt_full, Nx)[:, :Nt, :].reshape(-1, Nt * Nx).copy()
        t_np = t_np[:Nt]
        print(f"half_time: using first {Nt} time steps")
    else:
        Nt = Nt_full
        t_np = t_np[:Nt]

    n_nu  = len(args.nu_values)
    N_per = args.n_samples
    train_idx = np.concatenate([
        np.arange(i * N_per, (i + 1) * N_per - args.n_test_per_nu)
        for i in range(n_nu)
    ])
    test_idx = np.concatenate([
        np.arange((i + 1) * N_per - args.n_test_per_nu, (i + 1) * N_per)
        for i in range(n_nu)
    ])

    # u0 = t=0 slice; Nt-slow storage means first Nx elements = t=0
    u0_all  = torch.from_numpy(s_np[:, :Nx])      # (N, Nx)
    kappa   = torch.from_numpy(kappa_np[:, None])  # (N, 1)
    s       = torch.from_numpy(s_np)               # (N, Nt*Nx)

    s_train     = s[train_idx];       s_test     = s[test_idx]
    u0_train    = u0_all[train_idx];  u0_test    = u0_all[test_idx]
    kappa_train = kappa[train_idx];   kappa_test = kappa[test_idx]

    u0_train_dev  = u0_train.to(DEVICE)
    u0_test_dev   = u0_test.to(DEVICE)
    kappa_train_d = kappa_train.to(DEVICE)
    kappa_test_d  = kappa_test.to(DEVICE)

    N_train = len(train_idx)
    N_test  = len(test_idx)
    m = len(range(0, Nx, args.sensor_stride))
    print(f"N_train={N_train}  N_test={N_test}  Nx={Nx}  Nt={Nt}  m={m}")

    # Build (x, t) coordinate pairs for mode networks: shape (Nt*Nx, 2)
    x_grid = torch.tensor(x_np, dtype=torch.float32)
    t_grid = torch.tensor(t_np, dtype=torch.float32)
    tt_g, xx_g = torch.meshgrid(t_grid, x_grid, indexing="ij")  # each (Nt, Nx)
    x_flat = torch.stack([xx_g.flatten(), tt_g.flatten()], dim=1).to(DEVICE)  # (Nt*Nx, 2)
    print(f"s_train: {tuple(s_train.shape)}, x_flat: {tuple(x_flat.shape)}")

    print("=== Phase 1: Fourier NeuralPOD ===")
    w = torch.ones(Nt * Nx, dtype=torch.float32).to(DEVICE) / (Nt * Nx)
    basis = FourierRegimeBasis(
        d_x=2, M=N_train, quad_weights=w,
        hidden_dim=args.hidden_dim_basis,
        num_frequencies=args.num_frequencies,
        scales=args.scales,
        n_layers=args.n_layers_basis,
    ).to(DEVICE)

    cfg_pod = FourierNeuralPODConfig(
        max_modes=args.max_modes,
        n_epochs_mean=args.n_epochs_mean,
        n_epochs_mode=args.n_epochs_mode,
    )
    trainer_pod = FourierNeuralPODTrainer(basis, cfg_pod)
    history_pod = trainer_pod.train(s_train, x_flat, t=None)

    K = basis.num_modes
    print(f"K={K} modes learned")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].semilogy(history_pod.residual_norms, "o-", color=C0, markersize=4, lw=1.5)
    axes[0].set_xlabel("Mode index"); axes[0].set_ylabel("Weighted residual norm")
    axes[0].set_title("Phase 1: residual per mode", fontweight="bold")
    axes[0].grid(True, ls="--", alpha=0.25); axes[0].spines[["top", "right"]].set_visible(False)
    logs = [v for sublist in history_pod.mode_losses for v in sublist]
    axes[1].semilogy(logs, color=C0, lw=0.8)
    lengths    = [len(ml) for ml in history_pod.mode_losses]
    boundaries = [sum(lengths[:i]) for i in range(1, len(lengths) + 1)]
    for b in boundaries[:-1]:
        axes[1].axvline(b, color="gray", lw=0.6, ls="--", alpha=0.5)
    axes[1].set_xlabel("Epoch (all modes)"); axes[1].set_ylabel("Mode training loss")
    axes[1].set_title("Phase 1: mode training losses", fontweight="bold")
    axes[1].grid(True, ls="--", alpha=0.25); axes[1].spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "npod_phase1.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"=== Phase 2: Branch network (d_kappa=1, {'joint' if joint else 'specialist'}) ===")
    branch = BranchNet(m=m, P=K, hidden_dim=args.hidden_dim,
                       n_layers=args.n_layers, d_kappa=1).to(DEVICE)

    # Training targets: lambda_ten from Phase 1 for each training sample, shape (N_train, K)
    coeffs = torch.stack([mode.lambda_ten.detach() for mode in basis.modes], dim=1).to(DEVICE)

    # Val targets: greedy sequential projection of s_test onto learned modes
    with torch.no_grad():
        mean_d   = basis.mean_net(x_flat)                               # (Nt*Nx,)
        residual = s_test.to(DEVICE) - mean_d.unsqueeze(0)             # (N_test, Nt*Nx)
        val_tgt_list = []
        for mode in basis.modes:
            phi_k    = mode.phi(x_flat)                                 # (Nt*Nx,)
            phi_sq_k = (phi_k ** 2).sum()
            c_k      = (residual @ phi_k) / phi_sq_k                   # (N_test,)
            val_tgt_list.append(c_k)
            residual = residual - c_k.unsqueeze(1) * phi_k.unsqueeze(0)
        val_targets = torch.stack(val_tgt_list, dim=1)                  # (N_test, K)

    u0_sensors     = u0_train_dev[:, ::args.sensor_stride]  # (N_train, m)
    u0_val_sensors = u0_test_dev[:, ::args.sensor_stride]   # (N_test,  m)
    print(f"Training | N={N_train}, m={m}, K={K}")

    dl  = DataLoader(TensorDataset(u0_sensors, kappa_train_d, coeffs),
                     batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(branch.parameters(), lr=args.lr, weight_decay=1e-4)
    VAL_EVERY = max(1, args.n_epochs // 40)

    history_branch, history_val = [], []
    for epoch in range(args.n_epochs):
        branch.train()
        total = 0.0
        for u0_b, kappa_b, coeff_b in dl:
            opt.zero_grad()
            loss = F.mse_loss(branch(u0_b, kappa_b), coeff_b)
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(dl)
        history_branch.append(avg)
        if epoch % VAL_EVERY == 0:
            branch.eval()
            with torch.no_grad():
                vl = F.mse_loss(branch(u0_val_sensors, kappa_test_d), val_targets).item()
            branch.train()
            history_val.append((epoch, vl))
        if epoch % args.log_every == 0:
            val_str = f"  val={history_val[-1][1]:.4e}" if history_val else ""
            print(f"  epoch {epoch:6d} | coeff_mse={avg:.4e}{val_str}")
    val_str = f"  val={history_val[-1][1]:.4e}" if history_val else ""
    print(f"  done: final coeff_mse={history_branch[-1]:.4e}{val_str}")

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.semilogy(history_branch, color=C0, lw=1.5, label="Train coeff MSE")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Coeff MSE", color=C0)
    ax1.tick_params(axis="y", labelcolor=C0)
    ax1.grid(True, ls="--", alpha=0.25)
    ax1.spines[["top"]].set_visible(False)
    if history_val:
        ve, vl = zip(*history_val)
        ax2 = ax1.twinx()
        ax2.semilogy(ve, vl, color=C1, lw=1.5, ls="--", label="Val coeff MSE")
        ax2.set_ylabel("Val coeff MSE", color=C1)
        ax2.tick_params(axis="y", labelcolor=C1)
        ax2.spines[["top"]].set_visible(False)
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, framealpha=0.7)
    ax1.set_title("Phase 2: branch network", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "training_dynamics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    @torch.no_grad()
    def predict_batch(u0_in, kappa_in, batch_size=128):
        branch.eval()
        basis.eval()
        phi  = torch.stack([mode.phi(x_flat) for mode in basis.modes], dim=1)  # (Nt*Nx, K)
        mean = basis.mean_net(x_flat)                                            # (Nt*Nx,)
        parts = []
        for i in range(0, len(u0_in), batch_size):
            u0_s   = u0_in[i:i + batch_size][:, ::args.sensor_stride].to(DEVICE)
            k_s    = kappa_in[i:i + batch_size].to(DEVICE)
            beta_v = branch(u0_s, k_s)                    # (B, K)
            pred   = mean.unsqueeze(0) + beta_v @ phi.T   # (B, Nt*Nx)
            parts.append(pred.cpu())
        return torch.cat(parts, dim=0).numpy()

    s_test_np  = s_test.numpy()
    pred_test  = predict_batch(u0_test, kappa_test)
    err_test   = rel_l2(s_test_np, pred_test)

    s_train_np  = s_train.numpy()
    pred_train  = predict_batch(u0_train, kappa_train)
    err_train   = rel_l2(s_train_np, pred_train)

    print(f"Train | mean={err_train.mean():.4f}  median={np.median(err_train):.4f}  std={err_train.std():.4f}")
    print(f"Test  | mean={err_test.mean():.4f}  median={np.median(err_test):.4f}  std={err_test.std():.4f}  p95={np.percentile(err_test, 95):.4f}")

    metrics = {
        "run_name":      RUN_NAME,
        "n_modes":       int(K),
        "n_train":       N_train,
        "n_test":        N_test,
        "half_time":     args.half_time,
        "Nt":            int(Nt),
        "Nx":            int(Nx),
        "residual_norms": [float(v) for v in history_pod.residual_norms],
        "mean_loss":      [float(v) for v in history_pod.mean_loss],
        "mode_losses":    [[float(v) for v in ml] for ml in history_pod.mode_losses],
        "branch_loss":    [float(v) for v in history_branch],
    }

    if not joint:
        # Specialist metrics + cross-nu eval
        metrics.update({
            "train_mean":   float(err_train.mean()),
            "train_median": float(np.median(err_train)),
            "train_std":    float(err_train.std()),
            "test_mean":    float(err_test.mean()),
            "test_median":  float(np.median(err_test)),
            "test_std":     float(err_test.std()),
            "test_p95":     float(np.percentile(err_test, 95)),
        })

        cross_nu_metrics = {}
        for nu_eval in [0.001, 0.01, 0.1, 1.0]:
            _local_nu  = _PROJECT_ROOT / "data" / f"Burgers_Nu{nu_eval}.hdf5"
            _server_nu = data_dir / f"1D_Burgers_Sols_Nu{nu_eval}.hdf5"
            fpath = _local_nu if _local_nu.exists() else _server_nu
            if not fpath.exists():
                print(f"  nu={nu_eval:.3f}: file not found, skipping")
                continue
            with h5py.File(fpath, "r") as f:
                raw_nu = f["tensor"][-args.n_test_per_nu:]
                if raw_nu.ndim == 4:
                    raw_nu = raw_nu[..., 0]
            if args.half_time:
                raw_nu = raw_nu[:, :Nt, :]
            u0_nu    = torch.from_numpy(raw_nu[:, 0, :])
            s_nu_flat = raw_nu.reshape(args.n_test_per_nu, -1).astype(np.float32)
            kappa_nu  = torch.full((args.n_test_per_nu, 1), float(nu_eval), dtype=torch.float32)
            pred_nu   = predict_batch(u0_nu, kappa_nu)
            err_nu    = rel_l2(s_nu_flat, pred_nu)
            cross_nu_metrics[nu_eval] = {
                "mean":   float(err_nu.mean()),
                "median": float(np.median(err_nu)),
                "std":    float(err_nu.std()),
                "p95":    float(np.percentile(err_nu, 95)),
            }
            tag = " (trained)" if nu_eval == args.nu_values[0] else ""
            print(f"  nu={nu_eval:.3f}{tag}: mean={err_nu.mean():.4f}  "
                  f"median={np.median(err_nu):.4f}  std={err_nu.std():.4f}")

        metrics["cross_nu"] = cross_nu_metrics

        if cross_nu_metrics:
            nu_keys   = list(cross_nu_metrics.keys())
            nu_labels = [f"nu={nu:.3f}" for nu in nu_keys]
            means     = [cross_nu_metrics[nu]["mean"]   for nu in nu_keys]
            medians   = [cross_nu_metrics[nu]["median"] for nu in nu_keys]
            x_pos     = np.arange(len(nu_labels))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x_pos - 0.2,  means,   0.35, label="Mean",   color=C0, alpha=0.85, linewidth=0)
            ax.bar(x_pos + 0.15, medians, 0.35, label="Median", color=C1, alpha=0.85, linewidth=0)
            if args.nu_values[0] in nu_keys:
                ax.axvline(nu_keys.index(args.nu_values[0]),
                           color="gray", ls="--", lw=1.2, alpha=0.7, label="Trained on")
            ax.set_xticks(x_pos); ax.set_xticklabels(nu_labels)
            ax.set_ylabel("Relative L2 error")
            ax.set_title("NeuralPOD-DeepONet - cross-nu generalization", fontweight="bold")
            ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25, axis="y")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(RUN_DIR, "cross_nu.png"), dpi=150, bbox_inches="tight")
            plt.close()

    else:
        # Joint metrics: per-nu breakdown
        metrics.update({
            "overall_mean":   float(err_test.mean()),
            "overall_median": float(np.median(err_test)),
            "overall_std":    float(err_test.std()),
        })
        print("Mean rel L2 error per nu (test):")
        for i, nu in enumerate(args.nu_values):
            sl = slice(i * args.n_test_per_nu, (i + 1) * args.n_test_per_nu)
            err_nu = err_test[sl]
            metrics[f"nu{nu:.3f}_mean"]   = float(err_nu.mean())
            metrics[f"nu{nu:.3f}_median"] = float(np.median(err_nu))
            metrics[f"nu{nu:.3f}_std"]    = float(err_nu.std())
            print(f"  nu={nu:.3f}: mean={err_nu.mean():.4f}  "
                  f"median={np.median(err_nu):.4f}  std={err_nu.std():.4f}")

        nu_unique = np.array(args.nu_values)
        x_pos   = np.arange(len(nu_unique))
        means   = [metrics[f"nu{nu:.3f}_mean"]   for nu in nu_unique]
        medians = [metrics[f"nu{nu:.3f}_median"] for nu in nu_unique]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x_pos - 0.2,  means,   0.35, label="Mean",   color=C0, alpha=0.85, linewidth=0)
        ax.bar(x_pos + 0.15, medians, 0.35, label="Median", color=C1, alpha=0.85, linewidth=0)
        ax.set_xticks(x_pos); ax.set_xticklabels([f"nu={nu:.3f}" for nu in nu_unique])
        ax.set_ylabel("Relative L2 error")
        ax.set_title("NeuralPOD-DeepONet (joint) - error per nu", fontweight="bold")
        ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, "error_per_nu.png"), dpi=150, bbox_inches="tight")
        plt.close()

    rng = np.random.default_rng(args.seed)

    if not joint:
        idxs = rng.choice(N_test, size=args.n_viz, replace=False)
        pred_viz = predict_batch(u0_test[idxs], kappa_test[idxs])

        fig, axes = plt.subplots(args.n_viz, 3, figsize=(14, 3 * args.n_viz))
        if args.n_viz == 1:
            axes = axes[None, :]
        for row, (idx, pred_flat) in enumerate(zip(idxs, pred_viz)):
            true = s_test_np[idx].reshape(Nt, Nx)
            pred = pred_flat.reshape(Nt, Nx)
            err  = np.abs(true - pred)
            vmax = np.abs(true).max()
            rl2  = float(np.linalg.norm(true - pred) / np.linalg.norm(true))
            for col, (arr, title, cmap, vmin, vm) in enumerate([
                (true, "Ground Truth",       "RdBu_r", -vmax, vmax),
                (pred, "NeuralPOD-DeepONet", "RdBu_r", -vmax, vmax),
                (err,  "Absolute Error",     "Oranges",  0,    err.max()),
            ]):
                ax = axes[row, col]
                im = ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap,
                               extent=[x_np.min(), x_np.max(), t_np.min(), t_np.max()],
                               vmin=vmin, vmax=vm)
                if row == 0:
                    ax.set_title(title, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(f"t   (rel L2={rl2:.3f})")
                if row == args.n_viz - 1:
                    ax.set_xlabel("x")
                ax.spines[["top", "right"]].set_visible(False)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.suptitle("NeuralPOD-DeepONet: reconstruction examples", fontweight="bold", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, "npod_reconstruction.png"), dpi=150, bbox_inches="tight")
        plt.close()
    else:
        n_rows = n_nu * args.n_viz
        fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3 * n_rows))
        if n_rows == 1:
            axes = axes[None, :]
        row = 0
        for i, nu in enumerate(args.nu_values):
            block_start = i * args.n_test_per_nu
            idxs_nu = rng.choice(args.n_test_per_nu, size=args.n_viz, replace=False)
            for idx_local in idxs_nu:
                abs_idx = block_start + idx_local
                pred_flat = predict_batch(u0_test[abs_idx:abs_idx + 1],
                                          kappa_test[abs_idx:abs_idx + 1])
                true = s_test_np[abs_idx].reshape(Nt, Nx)
                pred = pred_flat[0].reshape(Nt, Nx)
                err  = np.abs(true - pred)
                vmax = np.abs(true).max()
                rl2  = float(np.linalg.norm(true - pred) / np.linalg.norm(true))
                for col, (arr, title, cmap, vmin, vm) in enumerate([
                    (true, "Ground Truth",       "RdBu_r", -vmax, vmax),
                    (pred, "NeuralPOD-DeepONet", "RdBu_r", -vmax, vmax),
                    (err,  "Absolute Error",     "Oranges",  0,    err.max()),
                ]):
                    ax = axes[row, col]
                    im = ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap,
                                   extent=[x_np.min(), x_np.max(), t_np.min(), t_np.max()],
                                   vmin=vmin, vmax=vm)
                    if row == 0:
                        ax.set_title(title, fontweight="bold")
                    if col == 0:
                        ax.set_ylabel(f"nu={nu:.3f}\n(rl2={rl2:.3f})")
                    if row == n_rows - 1:
                        ax.set_xlabel("x")
                    ax.spines[["top", "right"]].set_visible(False)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                row += 1
        plt.suptitle("NeuralPOD-DeepONet (joint): reconstruction examples",
                     fontweight="bold", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, "npod_reconstruction.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Error distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.hist(err_test, bins=30, color=C0, alpha=0.8, linewidth=0)
    ax.axvline(err_test.mean(),     color=C1, ls="--", lw=1.5, label=f"Mean {err_test.mean():.4f}")
    ax.axvline(np.median(err_test), color=C2, ls="--", lw=1.5, label=f"Median {np.median(err_test):.4f}")
    ax.set_xlabel("Relative L2 error"); ax.set_ylabel("Count")
    ax.set_title("Test error distribution", fontweight="bold")
    ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.plot(np.sort(err_test), color=C0, lw=1.5)
    ax.axhline(err_test.mean(), color=C1, ls="--", lw=1.2, alpha=0.8)
    ax.set_xlabel("Trajectory rank"); ax.set_ylabel("Relative L2 error")
    ax.set_title("Sorted test errors", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("NeuralPOD-DeepONet - test set errors", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "npod_err_dist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    torch.save({
        "branch":   branch.state_dict(),
        "basis":    basis.state_dict(),
        "metrics":  metrics,
        "run_name": RUN_NAME,
    }, os.path.join(RUN_DIR, "model.pt"))

    # Inference time
    _inf_ms = measure_inference_time(
        lambda: predict_batch(u0_test, kappa_test),
        device=DEVICE
    )
    metrics["inference_ms_total"] = _inf_ms
    metrics["inference_ms_per_sample"] = _inf_ms / N_test

    metrics["hparams"] = vars(args)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {os.path.abspath(RUN_DIR)}")


if __name__ == "__main__":
    main()
