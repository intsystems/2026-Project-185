#!/usr/bin/env python
"""Train TEMPO on 2D incompressible Navier-Stokes across multiple Reynolds numbers.

TEMPO discovers latent regimes through EM clustering, then learns per-regime operators
for velocity field prediction. Treats Reynolds number as a parameter controlling regime.

Usage:
  python train_tempo_navier_stokes.py --re_values 100 1000 3600 10000 --M 4
"""
import argparse
import json
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]

sys.path.insert(0, str(_SCRIPT_DIR.parents[1]))
from models.tempo import TEMPOTrainer, TEMPOConfig, pod_factory, fourier_pod_factory
from models.pod import PODConfig
from models.fourier_neural_pod import FourierNeuralPODConfig
from models.tempo_online import build_tempo_online, TEMPOOnlineConfig, _num_modes
from utils.datasets import load_ns_stacked, measure_inference_time


def rel_l2_vec(true, pred):
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


def parse_args():
    p = argparse.ArgumentParser()

    # Run
    p.add_argument("--run_name",      type=str,   default=None)
    p.add_argument("--results_dir",   type=str,   default=str(_PROJECT_ROOT / "TEMPO_results" / "navier_stokes"))

    # Data
    p.add_argument("--re_values",     type=int, nargs="+", default=[100, 1000, 3600, 10000])
    p.add_argument("--n_samples",     type=int,   default=5000)
    p.add_argument("--n_test_per_re", type=int,   default=1000)
    p.add_argument("--data_dir",      type=str,   default=os.path.expanduser("~/data/2D/Navier_Stokes"))

    # EM (TEMPOConfig)
    p.add_argument("--M",             type=int,   default=4,    help="Number of regimes")
    p.add_argument("--P_global",      type=int,   default=25,   help="Global POD modes for GMM init")
    p.add_argument("--sigma2",        type=float, default=0.1)
    p.add_argument("--max_em_iters",  type=int,   default=30)
    p.add_argument("--eps_skip",      type=float, default=1e-10)
    p.add_argument("--eps_large",     type=float, default=0.1)
    p.add_argument("--eps_conv",      type=float, default=0.0)
    p.add_argument("--kappa_init",    action=argparse.BooleanOptionalAction, default=True,
                   help="Init regimes from log(kappa) spacing instead of GMM on alpha")
    p.add_argument("--heteroscedastic", action="store_true", default=True,
                   help="Use relative error in EM E-step (robust for multi-scale data)")

    # Regime basis
    p.add_argument("--basis_type",    type=str,   default="pod", choices=["pod", "fourier"],
                   help="Basis type per regime")
    p.add_argument("--basis_max_modes", type=int, default=64,
                   help="Max modes per regime. Overridden by --total_modes if set.")
    p.add_argument("--total_modes",   type=int,   default=None,
                   help="If set, basis_max_modes = total_modes // M (fair comparison)")
    # Fourier basis extra args
    p.add_argument("--n_epochs_mean",        type=int,   default=500)
    p.add_argument("--n_epochs_mode",        type=int,   default=170)
    p.add_argument("--fourier_epoch_subset", type=float, default=None,
                   help="Fraction of N to use per mode epoch. Default: 1/M")

    # Online phase (TEMPOOnlineConfig)
    p.add_argument("--online_lr",         type=float, default=3e-4)
    p.add_argument("--online_epochs",     type=int,   default=200)
    p.add_argument("--online_batch",      type=int,   default=32)
    p.add_argument("--online_hidden_dim", type=int,   default=128)
    p.add_argument("--online_n_layers",   type=int,   default=4)
    p.add_argument("--sensor_stride",     type=int,   default=1)
    p.add_argument("--lambda_kl",         type=float, default=0.1)
    p.add_argument("--lambda_ent",        type=float, default=0.1)
    p.add_argument("--log_every",         type=int,   default=20)

    # Misc
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--n_viz",         type=int,   default=1)
    p.add_argument("--n_umap",        type=int,   default=5000)
    p.add_argument("--skip_umap",     action="store_true", default=False,
                   help="Skip UMAP visualization (faster, no umap-learn needed)")

    return p.parse_args()


def _data_path(re: int, data_dir: str) -> str:
    filename = f"2D_NavierStokes_Incomp_Re{re:05d}.npz"
    return os.path.join(data_dir, filename)


def main():
    args = parse_args()

    if args.total_modes is not None:
        args.basis_max_modes = max(1, args.total_modes // args.M)
        print(f"fair mode: total_modes={args.total_modes}, basis_max_modes={args.basis_max_modes} per regime")

    RUN_NAME = args.run_name or f"tempo_{args.basis_type}_navier_stokes_M{args.M}_v1"
    RUN_DIR  = os.path.join(args.results_dir, RUN_NAME)
    os.makedirs(RUN_DIR, exist_ok=True)

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    print(f"device={DEVICE}  run_dir={os.path.abspath(RUN_DIR)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    REGIME_COLORS = [plt.cm.tab10(i) for i in range(10)]
    regime_colors = REGIME_COLORS[:args.M]

    # --- Data loading ---
    entries = []
    for re in args.re_values:
        fpath = _data_path(re, args.data_dir)
        if not os.path.exists(fpath):
            print(f"  Re={re}: file not found, skipping")
            continue
        entries.append((re, fpath))

    if not entries:
        raise RuntimeError("No data files found. Check --data_dir or download the data.")

    re_loaded = [e[0] for e in entries]
    s_np, u0_np, kappa_np, xy_np, Nx, Ny, Nt = load_ns_stacked(entries, n_samples=args.n_samples)
    Nxy = Nx * Ny

    # Build full spatiotemporal-component coordinate array matching s flattening (Nt, Nx, Ny, 2).
    # Required for Fourier basis (basis_type=fourier); ignored by POD basis (basis_type=pod).
    _x1 = np.linspace(0, 1, Nx, dtype=np.float32)
    _y1 = np.linspace(0, 1, Ny, dtype=np.float32)
    _t1 = np.linspace(0, 1, Nt, dtype=np.float32)
    _c1 = np.array([0.0, 1.0], dtype=np.float32)
    _T, _X, _Y, _C = np.meshgrid(_t1, _x1, _y1, _c1, indexing='ij')  # each (Nt, Nx, Ny, 2)
    x_full = np.stack([_X.ravel(), _Y.ravel(), _T.ravel(), _C.ravel()], axis=1)  # (Nt*Nxy*2, 4)
    x_flat = torch.tensor(x_full, dtype=torch.float32)  # (Nt*Nxy*2, 4) — stays CPU for EM
    x_dev  = x_flat.to(DEVICE)

    s     = torch.from_numpy(s_np);  del s_np
    u0    = torch.from_numpy(u0_np); del u0_np
    kappa = torch.from_numpy(kappa_np[:, None])

    # Build train/test split based on kappa values (robust to NaN-dropped samples)
    train_idx, test_idx = [], []
    for re_val in sorted(np.unique(kappa_np)):
        idx = np.where(kappa_np == re_val)[0]
        n_test = min(args.n_test_per_re, len(idx))
        train_idx.append(idx[:-n_test])
        test_idx.append(idx[-n_test:])
    train_idx = torch.from_numpy(np.concatenate(train_idx))
    test_idx  = torch.from_numpy(np.concatenate(test_idx))
    del kappa_np

    print(f"s={s.shape}, u0={u0.shape}, kappa={kappa.shape}")

    # --- Phase 1: TEMPO EM ---
    print("=== Phase 1: TEMPO EM ===")

    if args.basis_type == "pod":
        basis_cfg     = PODConfig(max_modes=args.basis_max_modes)
        basis_factory = pod_factory
    else:
        fourier_epoch_subset = args.fourier_epoch_subset if args.fourier_epoch_subset is not None else 1.0 / args.M
        basis_cfg = FourierNeuralPODConfig(
            max_modes=args.basis_max_modes,
            n_epochs_mean=args.n_epochs_mean,
            n_epochs_mode=args.n_epochs_mode,
            epoch_subset=fourier_epoch_subset,
        )
        basis_factory = fourier_pod_factory

    cfg = TEMPOConfig(
        M=args.M,
        P_global=args.P_global,
        sigma2=args.sigma2,
        max_em_iters=args.max_em_iters,
        eps_skip=args.eps_skip,
        eps_large=args.eps_large,
        eps_conv=args.eps_conv,
        heteroscedastic=args.heteroscedastic,
        kappa_init=args.kappa_init,
        basis_config=basis_cfg,
        basis_factory=basis_factory,
    )

    trainer = TEMPOTrainer(cfg)
    trainer.train(s[train_idx], x_dev, t=None, kappa=kappa[train_idx])

    # EM convergence plot
    log = trainer.history_phase1
    if log:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(log["iter"], log["ll"], "o-", color=plt.cm.tab10(0), lw=1.5, markersize=4)
        axes[0].set_xlabel("EM iteration"); axes[0].set_ylabel("Log-likelihood")
        axes[0].set_title("EM: log-likelihood", fontweight="bold")
        axes[1].plot(log["iter"], log["entropy"], "o-", color=plt.cm.tab10(1), lw=1.5, markersize=4)
        axes[1].set_xlabel("EM iteration"); axes[1].set_ylabel("Entropy")
        axes[1].set_title("EM: assignment entropy", fontweight="bold")
        for m in range(args.M):
            delta_m = [d[m] for d in log["delta"]]
            axes[2].plot(log["iter"], delta_m, "o-", color=regime_colors[m],
                         lw=1.5, markersize=4, label=f"Regime {m+1}")
        axes[2].axhline(args.eps_conv, color="gray", ls="--", lw=1.2, label="eps_conv")
        axes[2].set_yscale("log")
        axes[2].set_xlabel("EM iteration"); axes[2].set_ylabel("Delta")
        axes[2].set_title("EM: distribution shift per regime", fontweight="bold")
        axes[2].legend(fontsize=8, framealpha=0.7)
        for ax in axes:
            ax.grid(True, ls="--", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, "em_convergence.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # --- Phase 2: Online gated operator ---
    print("=== Phase 2: Online gated operator ===")

    # Normalise s so Phase 2 MSE loss is O(1); rel-L2 is scale-invariant.
    # EM (Phase 1) ran on raw s with heteroscedastic relative error, so unaffected.
    # Scale POD means/modes consistently so s_hat = mean_m/c + b @ (modes_m/c).T
    s_scale = s[train_idx].std().item()
    print(f"  s_scale={s_scale:.4f}  (normalising s and POD bases for Phase 2)")
    s = s / s_scale
    from models.pod import PODTrainer
    from models.fourier_neural_pod import FourierNeuralPODTrainer
    for t in trainer.trainers:
        if isinstance(t, PODTrainer):
            t.basis.mean  = t.basis.mean  / s_scale
            t.basis.modes = t.basis.modes / s_scale

    # train_idx / test_idx computed earlier from kappa_np

    s_train     = s[train_idx];         s_test     = s[test_idx]
    u0_train    = u0[train_idx];        u0_test    = u0[test_idx]
    kappa_train = kappa[train_idx];     kappa_test = kappa[test_idx]
    gamma_train = trainer.gamma  # trainer was fitted on train_idx only

    print(f"train: {s_train.shape}, test: {s_test.shape}")
    for m, t in enumerate(trainer.trainers):
        print(f"  Regime {m+1}: P={_num_modes(t)} modes")

    cfg_online = TEMPOOnlineConfig(
        lr=args.online_lr,
        n_epochs=args.online_epochs,
        batch_size=args.online_batch,
        hidden_dim=args.online_hidden_dim,
        n_layers=args.online_n_layers,
        sensor_stride=args.sensor_stride,
        lambda_kl=args.lambda_kl,
        lambda_ent=args.lambda_ent,
        log_every=args.log_every,
    )

    model_online, online_trainer = build_tempo_online(
        trainers=trainer.trainers,
        d_kappa=kappa.shape[1],
        Nx=Nxy * 2,  # 2 velocity components
        cfg=cfg_online,
    )
    model_online = model_online.to(DEVICE)

    history = online_trainer.train(
        s=s_train, u0=u0_train, kappa=kappa_train,
        x_flat=x_dev, gamma_star=gamma_train,
        trainers=trainer.trainers,
        val_s=s_test, val_u0=u0_test, val_kappa=kappa_test,
    )

    # Training dynamics plot
    epochs = range(len(history["total"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["total"], color=plt.cm.tab10(0), label="total", lw=2)
    axes[0].plot(epochs, history["data"],  color=plt.cm.tab10(1), label="data",  lw=1.5, ls="--")
    if online_trainer.val_history:
        ve, vl = zip(*online_trainer.val_history)
        axes[0].plot(ve, vl, color=plt.cm.tab10(4), lw=1.5, ls=":", label="val data")
    axes[0].set_title("Phase 2: branch network", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(framealpha=0.7); axes[0].set_yscale("symlog")
    axes[1].plot(epochs, history["kl"],  color=plt.cm.tab10(2), label="KL",  lw=1.5)
    axes[1].plot(epochs, history["ent"], color=plt.cm.tab10(3), label="ent", lw=1.5, ls="--")
    axes[1].set_title("Regularisation terms", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(framealpha=0.7)
    for ax in axes:
        ax.grid(True, ls="--", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "training_dynamics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Evaluation ---
    s_pred, w_pred = online_trainer.predict(
        u0_new=u0_test, kappa_new=kappa_test,
        x_flat=x_dev, trainers=trainer.trainers,
    )
    s_pred_np  = s_pred.cpu().numpy()
    s_test_np  = s_test.numpy()
    w_np       = w_pred.cpu().numpy()
    kappa_t_np = kappa_test[:, 0].numpy()
    re_unique_t = np.unique(kappa_t_np)

    rel_l2 = rel_l2_vec(s_test_np, s_pred_np)
    print("Mean rel L2 error:")
    for re in re_unique_t:
        mask = kappa_t_np == re
        print(f"  Re={re:.0f}: {rel_l2[mask].mean():.4f} +/- {rel_l2[mask].std():.4f}")

    # Gating + error plots
    re_to_col = {re: plt.cm.Dark2(i / 7) for i, re in enumerate(re_unique_t)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    x_pos = np.arange(len(re_unique_t))
    width = 0.8 / args.M
    for m in range(args.M):
        w_m = [w_np[kappa_t_np == re, m].mean() for re in re_unique_t]
        axes[0].bar(x_pos + m * width, w_m, width,
                    color=regime_colors[m], label=f"Regime {m+1}", alpha=0.85, linewidth=0)
    axes[0].set_xticks(x_pos + width * (args.M - 1) / 2)
    axes[0].set_xticklabels([f"Re={re:.0f}" for re in re_unique_t])
    axes[0].set_ylabel("Mean gating weight")
    axes[0].set_title("Gating weights per Reynolds number", fontweight="bold")
    axes[0].legend(fontsize=9, framealpha=0.7); axes[0].set_ylim(0, 1)

    bp_data = [rel_l2[kappa_t_np == re] for re in re_unique_t]
    bp = axes[1].boxplot(bp_data, patch_artist=True, widths=0.45,
                         medianprops=dict(color="black", lw=1.5),
                         whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
                         flierprops=dict(marker="o", markersize=3, alpha=0.4, linestyle="none"))
    for patch, re in zip(bp["boxes"], re_unique_t):
        patch.set_facecolor(re_to_col[re]); patch.set_alpha(0.8); patch.set_linewidth(0)
    axes[1].set_xticklabels([f"Re={re:.0f}" for re in re_unique_t])
    axes[1].set_ylabel("Relative L2 error")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Error by Reynolds number", fontweight="bold")
    for ax in axes:
        ax.grid(True, ls="--", alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("TEMPO Phase 2 - test set results", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "TEMPO_phase2.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Visualization: Error distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rel_l2, bins=40, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(rel_l2.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {rel_l2.mean():.4f}")
    ax.axvline(np.median(rel_l2), color="orange", linestyle="--", linewidth=2, label=f"Median: {np.median(rel_l2):.4f}")
    ax.set_xlabel("Relative L2 Error")
    ax.set_ylabel("Count")
    ax.set_title("Test Error Distribution (2D Navier-Stokes TEMPO)", fontweight="bold")
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
        s_pred_sample = s_pred_np[test_sample_idx]

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
        s_pred_sample = s_pred_np[test_sample_idx].reshape(Nt, Nx, Ny, 2)

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

    # Save metrics
    n_params = sum(p.numel() for p in model_online.parameters())
    n_params_basis = 0
    if args.basis_type == "fourier":
        for t in trainer.trainers:
            n_params_basis += sum(p.numel() for p in t.basis.parameters())
    metrics = {
        "run_name": RUN_NAME,
        "M": args.M,
        "basis_type": args.basis_type,
        "n_params": n_params,
        "n_params_basis": n_params_basis,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }

    # Per-Re error statistics
    for re in sorted(re_unique_t):
        mask = kappa_t_np == re
        rl2 = rel_l2[mask]
        metrics[f"re{re:.0f}_mean"] = float(rl2.mean())
        metrics[f"re{re:.0f}_median"] = float(np.median(rl2))
        metrics[f"re{re:.0f}_std"] = float(rl2.std())
        metrics[f"re{re:.0f}_p95"] = float(np.percentile(rl2, 95))

    # Inference time
    _inf_ms = measure_inference_time(
        lambda: online_trainer.predict(
            u0_new=u0_test, kappa_new=kappa_test,
            x_flat=x_dev, trainers=trainer.trainers,
        ),
        device=DEVICE
    )
    metrics["inference_ms_total"] = _inf_ms
    metrics["inference_ms_per_sample"] = _inf_ms / len(u0_test)

    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save({
        "model_online": model_online.state_dict(),
        "cfg":          cfg,
        "cfg_online":   cfg_online,
        "metrics":      metrics,
        "run_name":     RUN_NAME,
    }, os.path.join(RUN_DIR, "model_online.pt"))

    torch.save(trainer, os.path.join(RUN_DIR, "trainer.pt"))

    print(f"Results saved to {RUN_DIR}")

    # UMAP visualization
    if not args.skip_umap:
        try:
            from utils.plotting import plot_umap_regimes
            rng_umap = np.random.default_rng(args.seed)
            idx_umap = rng_umap.choice(len(train_idx), min(args.n_umap, len(train_idx)), replace=False)
            plot_umap_regimes(
                alpha=trainer.alpha[idx_umap].cpu().numpy(),
                mu=trainer.mu.cpu().numpy(),
                Sigma=trainer.Sigma.cpu().numpy(),
                hard_labels=trainer.gamma[idx_umap].argmax(dim=1).cpu().numpy(),
                param_vals=kappa[train_idx][idx_umap, 0].cpu().numpy(),
                regime_colors=regime_colors,
                param_label="Re",
                title="NS trajectories - TEMPO EM regime structure",
                save_path=os.path.join(RUN_DIR, "TEMPO_phase1.png"),
                seed=args.seed,
            )
            print("UMAP saved.")
        except ImportError:
            print("umap-learn not installed, skipping UMAP plot")
        except Exception as e:
            print(f"UMAP failed: {e}")


if __name__ == "__main__":
    main()
