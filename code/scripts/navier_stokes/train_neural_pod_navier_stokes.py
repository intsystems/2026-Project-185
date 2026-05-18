#!/usr/bin/env python
"""Train NeuralPOD-DeepONet on 2D incompressible Navier-Stokes — specialist or joint.

Usage:
  Specialist: python train_neural_pod_navier_stokes.py --re_values 100
  Joint:      python train_neural_pod_navier_stokes.py --re_values 100 1000 3600 10000
"""
import argparse
import json
import math
import os
import pathlib
import sys

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
from utils.datasets import load_ns_stacked, measure_inference_time


def rel_l2(true, pred):
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--re_values", type=int, nargs="+", required=True,
                   help="One value = specialist; multiple = joint")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--results_dir", type=str, default=str(_PROJECT_ROOT / "TEMPO_results" / "navier_stokes"))
    p.add_argument("--n_samples", type=int, default=5000,
                   help="Samples per Re loaded (train + test)")
    p.add_argument("--n_test_per_re", type=int, default=1000)
    p.add_argument("--data_dir", type=str, default=os.path.expanduser("~/data/2D/Navier_Stokes"))
    # Fourier basis
    p.add_argument("--max_modes", type=int, default=32)
    p.add_argument("--n_epochs_mean", type=int, default=400)
    p.add_argument("--n_epochs_mode", type=int, default=400)
    p.add_argument("--hidden_dim_basis", type=int, default=256)
    p.add_argument("--num_frequencies", type=int, default=96)
    p.add_argument("--scales", type=float, nargs="+", default=[0.5, 2.0, 6.0])
    p.add_argument("--n_layers_basis", type=int, default=3)
    p.add_argument("--traj_batch_size", type=int, default=512,
                   help="Trajectory mini-batch size for NeuralPOD mode training")
    p.add_argument("--epoch_subset", type=float, default=0.5,
                   help="Fraction of N sampled per NeuralPOD epoch (< 1 saves GPU RAM)")
    # Branch network
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    # Training
    p.add_argument("--n_epochs", type=int, default=800)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=100)
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_viz", type=int, default=2)
    return p.parse_args()


def _data_path(re: int, data_dir: str) -> str:
    filename = f"2D_NavierStokes_Incomp_Re{re:05d}.npz"
    return os.path.join(data_dir, filename)


def main():
    args = parse_args()

    joint = len(args.re_values) > 1
    if joint:
        RUN_NAME = args.run_name or "npod_deeponet_joint_navier_stokes_v1"
    else:
        RUN_NAME = args.run_name or f"npod_deeponet_navier_stokes_re{args.re_values[0]}_v1"
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

    C0, C1 = plt.cm.tab10(0), plt.cm.tab10(1)

    # --- Data loading ---
    entries = []
    for re in args.re_values:
        fpath = _data_path(re, args.data_dir)
        if not os.path.exists(fpath):
            print(f"  Re={re}: file not found, skipping")
            continue
        entries.append((re, fpath))

    if not entries:
        raise RuntimeError("No data files found. Check --data_dir.")

    re_loaded = [e[0] for e in entries]
    s_np, u0_np, kappa_np, xy_np, Nx, Ny, Nt = load_ns_stacked(entries, n_samples=args.n_samples)
    Nxy = Nx * Ny

    train_idx, test_idx = [], []
    for re_val in sorted(np.unique(kappa_np)):
        idx = np.where(kappa_np == re_val)[0]
        n_test = min(args.n_test_per_re, len(idx))
        train_idx.append(idx[:-n_test])
        test_idx.append(idx[-n_test:])
    train_idx = np.concatenate(train_idx)
    test_idx  = np.concatenate(test_idx)

    s = torch.from_numpy(s_np); del s_np
    u0 = torch.from_numpy(u0_np); del u0_np
    kappa = torch.from_numpy(kappa_np[:, None])  # (N, 1)

    s_train = s[train_idx]; s_test = s[test_idx]
    u0_train = u0[train_idx]; u0_test = u0[test_idx]
    kappa_train = kappa[train_idx]; kappa_test = kappa[test_idx]

    u0_train_dev = u0_train.to(DEVICE)
    u0_test_dev = u0_test.to(DEVICE)
    kappa_train_d = kappa_train.to(DEVICE)
    kappa_test_d = kappa_test.to(DEVICE)

    N_train = len(train_idx)
    N_test = len(test_idx)
    m = u0_train.shape[1]  # Nxy * 2

    # Build full spatiotemporal-component coordinate array matching s flattening (Nt, Nx, Ny, 2)
    # s_np = vel.reshape(N, Nt*Nx*Ny*2) C-order → t slowest, x, y, c fastest
    _x1 = np.linspace(0, 1, Nx, dtype=np.float32)
    _y1 = np.linspace(0, 1, Ny, dtype=np.float32)
    _t1 = np.linspace(0, 1, Nt, dtype=np.float32)
    _c1 = np.array([0.0, 1.0], dtype=np.float32)
    _T, _X, _Y, _C = np.meshgrid(_t1, _x1, _y1, _c1, indexing='ij')  # each (Nt, Nx, Ny, 2)
    x_full = np.stack([_X.ravel(), _Y.ravel(), _T.ravel(), _C.ravel()], axis=1)  # (Nt*Nxy*2, 4)
    x_full_t = torch.tensor(x_full, dtype=torch.float32).to(DEVICE)
    Nfull = Nt * Nxy * 2  # = 172032 for 64x64 grid, 21 timesteps, 2 components
    print(f"N_train={N_train}  N_test={N_test}  Nxy={Nxy}  Nfull={Nfull}  m={m}")

    # --- Phase 1: Fourier NeuralPOD on training data ---
    print("=== Phase 1: Fourier NeuralPOD ===")
    w = torch.ones(Nfull, dtype=torch.float32).to(DEVICE) / Nfull
    basis = FourierRegimeBasis(
        d_x=4, M=N_train, quad_weights=w,
        hidden_dim=args.hidden_dim_basis,
        num_frequencies=args.num_frequencies,
        scales=args.scales,
        n_layers=args.n_layers_basis,
    ).to(DEVICE)

    cfg_pod = FourierNeuralPODConfig(
        max_modes=args.max_modes,
        n_epochs_mean=args.n_epochs_mean,
        n_epochs_mode=args.n_epochs_mode,
        traj_batch_size=args.traj_batch_size,
        epoch_subset=args.epoch_subset,
    )
    trainer_pod = FourierNeuralPODTrainer(basis, cfg_pod)
    history_pod = trainer_pod.train(s_train, x_full_t, t=None)

    K = basis.num_modes
    print(f"K={K} modes learned")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].semilogy(history_pod.residual_norms, "o-", color=C0, markersize=4, lw=1.5)
    axes[0].set_xlabel("Mode index"); axes[0].set_ylabel("Weighted residual norm")
    axes[0].set_title("Phase 1: residual per mode", fontweight="bold")
    axes[0].grid(True, ls="--", alpha=0.25); axes[0].spines[["top", "right"]].set_visible(False)
    logs = [v for sublist in history_pod.mode_losses for v in sublist]
    axes[1].semilogy(logs, color=C0, lw=0.8)
    lengths = [len(ml) for ml in history_pod.mode_losses]
    boundaries = [sum(lengths[:i]) for i in range(1, len(lengths) + 1)]
    for b in boundaries[:-1]:
        axes[1].axvline(b, color="gray", lw=0.6, ls="--", alpha=0.5)
    axes[1].set_xlabel("Epoch (all modes)"); axes[1].set_ylabel("Mode training loss")
    axes[1].set_title("Phase 1: mode training losses", fontweight="bold")
    axes[1].grid(True, ls="--", alpha=0.25); axes[1].spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "npod_phase1.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Phase 2: Branch network (always d_kappa=1) ---
    print(f"=== Phase 2: Branch network (d_kappa=1, {'joint' if joint else 'specialist'}) ===")
    branch = BranchNet(m=m, P=K, hidden_dim=args.hidden_dim,
                       n_layers=args.n_layers, d_kappa=1).to(DEVICE)

    # Training targets: lambda_ten from Phase 1, shape (N_train, K)
    coeffs = torch.stack([mode.lambda_ten.detach() for mode in basis.modes], dim=1).to(DEVICE)

    # Val targets: greedy sequential projection of s_test onto learned modes
    with torch.no_grad():
        mean_d = basis.mean_net(x_full_t)  # (Nfull,)
        residual = s_test.to(DEVICE) - mean_d.unsqueeze(0)  # (N_test, Nfull)
        val_tgt_list = []
        for mode in basis.modes:
            phi_k = mode.phi(x_full_t)  # (Nfull,)
            phi_sq_k = (phi_k ** 2).sum()
            c_k = (residual @ phi_k) / phi_sq_k  # (N_test,)
            val_tgt_list.append(c_k)
            residual = residual - c_k.unsqueeze(1) * phi_k.unsqueeze(0)
        val_targets = torch.stack(val_tgt_list, dim=1)  # (N_test, K)

    u0_sensors = u0_train_dev  # Use full initial condition (not sensor sampling)
    u0_val_sensors = u0_test_dev
    print(f"Training | N={N_train}, m={m}, K={K}")

    dl = DataLoader(TensorDataset(u0_sensors, kappa_train_d, coeffs),
                    batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(branch.parameters(), lr=args.lr, weight_decay=1e-4)
    VAL_EVERY = 50

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
            print(f"  epoch {epoch:5d} | coeff_mse={avg:.4e}{val_str}")
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

    # --- Prediction helper ---
    @torch.no_grad()
    def predict_batch(u0_in, kappa_in, batch_size=128):
        branch.eval()
        basis.eval()
        phi = torch.stack([mode.phi(x_full_t) for mode in basis.modes], dim=1)  # (Nfull, K)
        mean = basis.mean_net(x_full_t)  # (Nfull,)
        parts = []
        for i in range(0, len(u0_in), batch_size):
            u0_s = u0_in[i:i + batch_size].to(DEVICE)
            k_s = kappa_in[i:i + batch_size].to(DEVICE)
            beta_v = branch(u0_s, k_s)  # (B, K)
            pred = mean.unsqueeze(0) + beta_v @ phi.T  # (B, Nxy)
            parts.append(pred.cpu())
        return torch.cat(parts, dim=0).numpy()

    # --- Evaluation ---
    s_test_np = s_test.numpy()
    pred_test = predict_batch(u0_test, kappa_test)
    err_test = rel_l2(s_test_np, pred_test)

    err_train = rel_l2(s_train.numpy(), predict_batch(u0_train, kappa_train))

    print(f"Train | mean={err_train.mean():.4f}  median={np.median(err_train):.4f}  std={err_train.std():.4f}")
    print(f"Test  | mean={err_test.mean():.4f}  median={np.median(err_test):.4f}  std={err_test.std():.4f}  p95={np.percentile(err_test, 95):.4f}")

    metrics = {
        "run_name": RUN_NAME,
        "n_modes": int(K),
        "n_train": N_train,
        "n_test": N_test,
        "residual_norms": [float(v) for v in history_pod.residual_norms],
        "mean_loss": [float(v) for v in history_pod.mean_loss],
        "mode_losses": [[float(v) for v in ml] for ml in history_pod.mode_losses],
        "branch_loss": [float(v) for v in history_branch],
    }

    if not joint:
        metrics.update({
            "train_mean": float(err_train.mean()),
            "train_median": float(np.median(err_train)),
            "train_std": float(err_train.std()),
            "test_mean": float(err_test.mean()),
            "test_median": float(np.median(err_test)),
            "test_std": float(err_test.std()),
            "test_p95": float(np.percentile(err_test, 95)),
        })

        cross_re_metrics = {}
        for re_eval in [100, 1000, 3600, 10000]:
            fpath = _data_path(re_eval, args.data_dir)
            if not os.path.exists(fpath):
                print(f"  Re={re_eval}: file not found, skipping")
                continue
            s_re_all, u0_re_all, _, _, _, _, _ = load_ns_stacked(
                [(re_eval, fpath)], n_samples=args.n_samples
            )
            s_re = s_re_all[args.n_samples - args.n_test_per_re:]
            u0_re = u0_re_all[args.n_samples - args.n_test_per_re:]
            del s_re_all, u0_re_all
            u0_re_t = torch.from_numpy(u0_re)
            kappa_re = torch.full((len(u0_re), 1), float(re_eval), dtype=torch.float32)
            err_re = rel_l2(s_re, predict_batch(u0_re_t, kappa_re))
            cross_re_metrics[re_eval] = {
                "mean": float(err_re.mean()),
                "median": float(np.median(err_re)),
                "std": float(err_re.std()),
                "p95": float(np.percentile(err_re, 95)),
            }
            tag = " (trained)" if re_eval == args.re_values[0] else ""
            print(f"  Re={re_eval}{tag}: mean={err_re.mean():.4f}  median={np.median(err_re):.4f}")

        metrics["cross_re"] = cross_re_metrics

    else:
        metrics.update({
            "overall_mean": float(err_test.mean()),
            "overall_median": float(np.median(err_test)),
            "overall_std": float(err_test.std()),
        })
        kappa_test_np = kappa_test[:, 0].numpy()
        re_unique = np.unique(kappa_test_np)
        cross_re_metrics = {}
        print("Mean rel L2 error per Re (test):")
        for re in re_unique:
            mask = kappa_test_np == re
            m_err = float(err_test[mask].mean())
            med_err = float(np.median(err_test[mask]))
            metrics[f"re{re:.0f}_mean"] = m_err
            metrics[f"re{re:.0f}_median"] = med_err
            metrics[f"re{re:.0f}_std"] = float(err_test[mask].std())
            cross_re_metrics[re] = {
                "mean": m_err,
                "median": med_err,
                "std": float(err_test[mask].std()),
                "p95": float(np.percentile(err_test[mask], 95)),
            }
            print(f"  Re={re:.0f}: mean={m_err:.4f}  median={med_err:.4f}")

        metrics["cross_re"] = cross_re_metrics

    # --- Visualization: Error distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(err_test, bins=40, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(err_test.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {err_test.mean():.4f}")
    ax.axvline(np.median(err_test), color="orange", linestyle="--", linewidth=2, label=f"Median: {np.median(err_test):.4f}")
    ax.set_xlabel("Relative L2 Error")
    ax.set_ylabel("Count")
    ax.set_title("Test Error Distribution (2D Navier-Stokes Neural POD)", fontweight="bold")
    ax.legend(framealpha=0.7)
    ax.grid(True, ls="--", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "error_dist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Visualization: Sample reconstructions (2D spatial fields) ---
    n_viz = min(args.n_viz, len(test_idx))
    n_timesteps_show = 3
    time_indices = np.linspace(0, Nt - 1, n_timesteps_show, dtype=int)

    for sample_idx in range(n_viz):
        test_sample_idx = sample_idx
        s_true_sample = s_test_np[test_sample_idx]  # shape: (Nt * Nx * Ny * 2,)
        s_pred_sample = pred_test[test_sample_idx]

        # Reshape to (Nt, Nx, Ny, 2)
        s_true_reshaped = s_true_sample.reshape(Nt, Nx, Ny, 2)
        s_pred_reshaped = s_pred_sample.reshape(Nt, Nx, Ny, 2)

        # Get Reynolds number for this sample
        re_val = int(kappa_test[test_sample_idx, 0].item())

        # Create figure: show u and v components at select timesteps
        fig, axes = plt.subplots(2, n_timesteps_show * 2, figsize=(n_timesteps_show * 4, 6))
        fig.suptitle(f"Sample {sample_idx}: 2D Velocity Field Reconstruction (Re={re_val})",
                     fontweight="bold", fontsize=12)

        for t_i, t_idx in enumerate(time_indices):
            # U-component (velocity in x-direction)
            u_true = s_true_reshaped[t_idx, :, :, 0]
            u_pred = s_pred_reshaped[t_idx, :, :, 0]

            im0 = axes[0, t_i * 2].imshow(u_true, cmap="RdBu_r", origin="lower")
            axes[0, t_i * 2].set_title(f"u_true (t={t_idx})")
            axes[0, t_i * 2].set_xticks([])
            axes[0, t_i * 2].set_yticks([])
            plt.colorbar(im0, ax=axes[0, t_i * 2])

            im1 = axes[0, t_i * 2 + 1].imshow(u_pred, cmap="RdBu_r", origin="lower")
            axes[0, t_i * 2 + 1].set_title(f"u_pred (t={t_idx})")
            axes[0, t_i * 2 + 1].set_xticks([])
            axes[0, t_i * 2 + 1].set_yticks([])
            plt.colorbar(im1, ax=axes[0, t_i * 2 + 1])

        for t_i, t_idx in enumerate(time_indices):
            # V-component (velocity in y-direction)
            v_true = s_true_reshaped[t_idx, :, :, 1]
            v_pred = s_pred_reshaped[t_idx, :, :, 1]

            im2 = axes[1, t_i * 2].imshow(v_true, cmap="RdBu_r", origin="lower")
            axes[1, t_i * 2].set_title(f"v_true (t={t_idx})")
            axes[1, t_i * 2].set_xticks([])
            axes[1, t_i * 2].set_yticks([])
            plt.colorbar(im2, ax=axes[1, t_i * 2])

            im3 = axes[1, t_i * 2 + 1].imshow(v_pred, cmap="RdBu_r", origin="lower")
            axes[1, t_i * 2 + 1].set_title(f"v_pred (t={t_idx})")
            axes[1, t_i * 2 + 1].set_xticks([])
            axes[1, t_i * 2 + 1].set_yticks([])
            plt.colorbar(im3, ax=axes[1, t_i * 2 + 1])

        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, f"reconstruction_sample{sample_idx}.png"), dpi=100, bbox_inches="tight")
        plt.close()

    # --- 3D Visualization: Velocity magnitude surface plots ---
    from mpl_toolkits.mplot3d import Axes3D

    for sample_idx in range(n_viz):
        test_sample_idx = sample_idx
        s_true_sample = s_test_np[test_sample_idx].reshape(Nt, Nx, Ny, 2)
        s_pred_sample = pred_test[test_sample_idx].reshape(Nt, Nx, Ny, 2)

        # Compute velocity magnitude
        mag_true = np.sqrt(s_true_sample[..., 0]**2 + s_true_sample[..., 1]**2)
        mag_pred = np.sqrt(s_pred_sample[..., 0]**2 + s_pred_sample[..., 1]**2)

        re_val = int(kappa_test[test_sample_idx, 0].item())

        # Create 3D surface plots for selected timesteps
        fig = plt.figure(figsize=(15, 10))
        x_mesh = np.arange(Nx)
        y_mesh = np.arange(Ny)
        X, Y = np.meshgrid(x_mesh, y_mesh, indexing='ij')

        for t_i, t_idx in enumerate(time_indices):
            # True field
            ax_true = fig.add_subplot(2, n_timesteps_show, t_i + 1, projection='3d')
            Z_true = mag_true[t_idx]
            surf = ax_true.plot_surface(X, Y, Z_true, cmap="viridis", alpha=0.9, linewidth=0)
            ax_true.set_title(f"True Mag (t={t_idx})")
            ax_true.set_zlim(0, mag_true.max() * 1.1)
            ax_true.set_xlabel("x"); ax_true.set_ylabel("y")
            fig.colorbar(surf, ax=ax_true, shrink=0.5)

            # Predicted field
            ax_pred = fig.add_subplot(2, n_timesteps_show, n_timesteps_show + t_i + 1, projection='3d')
            Z_pred = mag_pred[t_idx]
            surf = ax_pred.plot_surface(X, Y, Z_pred, cmap="viridis", alpha=0.9, linewidth=0)
            ax_pred.set_title(f"Pred Mag (t={t_idx})")
            ax_pred.set_zlim(0, mag_true.max() * 1.1)
            ax_pred.set_xlabel("x"); ax_pred.set_ylabel("y")
            fig.colorbar(surf, ax=ax_pred, shrink=0.5)

        fig.suptitle(f"Sample {sample_idx}: 3D Velocity Magnitude (Re={re_val})", fontweight="bold", fontsize=12)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.05, hspace=0.4, wspace=0.3)
        plt.savefig(os.path.join(RUN_DIR, f"reconstruction_3d_sample{sample_idx}.png"), dpi=100, bbox_inches="tight")
        plt.close()

    print(f"Generated {n_viz} reconstruction visualization(s) + {n_viz} 3D visualization(s)")

    # --- Checkpoint ---
    torch.save({
        "branch": branch.state_dict(),
        "basis": basis.state_dict(),
        "metrics": metrics,
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
