#!/usr/bin/env python
"""Train NeuralPOD-DeepONet on a single viscosity value."""
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from models.regime_basis import FourierRegimeBasis
from models.fourier_neural_pod import FourierNeuralPODTrainer, FourierNeuralPODConfig
from models.pod_deeponet import BranchNet
from models.neural_pod_deeponet import NeuralPODDeepONet, NeuralPODDeepONetConfig, NeuralPODDeepONetTrainer


def rel_l2(true, pred):
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


_SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nu",              type=float, required=True)
    p.add_argument("--run_name",        type=str,   default=None)
    p.add_argument("--results_dir",     type=str,   default=str(_PROJECT_ROOT / "TEMPO_results"))
    p.add_argument("--all_nu",          type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0])
    p.add_argument("--n_train",         type=int,   default=2500)
    p.add_argument("--n_test",          type=int,   default=500)
    # Fourier basis
    p.add_argument("--max_modes",       type=int,   default=20)
    p.add_argument("--n_epochs_mean",   type=int,   default=800)
    p.add_argument("--n_epochs_mode",   type=int,   default=1200)
    p.add_argument("--hidden_dim_basis",type=int,   default=256)
    p.add_argument("--num_frequencies", type=int,   default=96)
    p.add_argument("--scales",          type=float, nargs="+", default=[0.5, 2.0, 6.0])
    p.add_argument("--n_layers_basis",  type=int,   default=3)
    # Branch network
    p.add_argument("--hidden_dim",      type=int,   default=256)
    p.add_argument("--n_layers",        type=int,   default=5)
    p.add_argument("--sensor_stride",   type=int,   default=1)
    # Training
    p.add_argument("--n_epochs",        type=int,   default=80000)
    p.add_argument("--batch_size",      type=int,   default=512)
    # Misc
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--n_viz",           type=int,   default=3)
    return p.parse_args()


def main():
    args = parse_args()

    TRAIN_NU = args.nu
    RUN_NAME = args.run_name or f"npod_deeponet_nu{TRAIN_NU}_v1"
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

    C0, C1, C2 = plt.cm.tab10(0), plt.cm.tab10(1), plt.cm.tab10(2)

    # --- Data loading ---
    _local  = _SCRIPT_DIR.parents[0] / "data" / f"Burgers_Nu{TRAIN_NU}.hdf5"
    _server = pathlib.Path(os.path.expanduser("~/data/1D/Burgers/Train")) / f"1D_Burgers_Sols_Nu{TRAIN_NU}.hdf5"
    path = str(_local) if _local.exists() else str(_server)
    print(f"data: {path}")

    with h5py.File(path, "r") as f:
        raw = f["tensor"][:args.n_train + args.n_test]
        if raw.ndim == 4:
            raw = raw[..., 0]
        x_np = f["x-coordinate"][:]
        t_np = f["t-coordinate"][:]

    N_total, Nt, Nx = raw.shape
    t_np = t_np[:Nt]
    print(f"loaded: N={N_total}, Nt={Nt}, Nx={Nx}")

    tensor_train = raw[:args.n_train]
    tensor_test  = raw[args.n_train:]
    del raw

    s_traj   = torch.tensor(tensor_train.reshape(args.n_train, -1), dtype=torch.float32)
    u0_train = torch.tensor(tensor_train[:, 0, :], dtype=torch.float32).to(DEVICE)
    u0_test  = torch.tensor(tensor_test[:, 0, :],  dtype=torch.float32).to(DEVICE)

    x_grid = torch.tensor(x_np, dtype=torch.float32)
    t_grid = torch.tensor(t_np, dtype=torch.float32)
    tt, xx = torch.meshgrid(t_grid, x_grid, indexing="ij")
    x_flat = torch.stack([xx.flatten(), tt.flatten()], dim=1).to(DEVICE)

    m = len(range(0, Nx, args.sensor_stride))
    print(f"s_traj: {tuple(s_traj.shape)}, x_flat: {tuple(x_flat.shape)}, m={m}")

    # --- Phase 1: Fourier Neural POD ---
    print("=== Phase 1: Fourier Neural POD ===")
    w = torch.ones(Nt * Nx, dtype=torch.float32).to(DEVICE) / (Nt * Nx)
    basis = FourierRegimeBasis(
        d_x=2, M=args.n_train, quad_weights=w,
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
    history_pod = trainer_pod.train(s_traj, x_flat, t=None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.semilogy(history_pod.residual_norms, "o-", color=C0, markersize=4, lw=1.5)
    ax.set_xlabel("Mode index"); ax.set_ylabel("Weighted residual norm")
    ax.set_title("Phase 1: residual per mode", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    logs = [v for sublist in history_pod.mode_losses for v in sublist]
    ax.semilogy(logs, color=C0, lw=0.8)
    lengths    = [len(ml) for ml in history_pod.mode_losses]
    boundaries = [sum(lengths[:i]) for i in range(1, len(lengths) + 1)]
    for b in boundaries[:-1]:
        ax.axvline(b, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("Epoch (all modes)"); ax.set_ylabel("Mode training loss")
    ax.set_title("Phase 1: mode training losses", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "npod_phase1.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Phase 2: Branch network ---
    print("=== Phase 2: Branch network ===")
    K = basis.num_modes
    branch = BranchNet(m=m, P=K, hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(DEVICE)
    model  = NeuralPODDeepONet(basis, branch).to(DEVICE)

    cfg     = NeuralPODDeepONetConfig(n_epochs=args.n_epochs, batch_size=args.batch_size,
                                      sensor_stride=args.sensor_stride)
    trainer = NeuralPODDeepONetTrainer(model, cfg)
    history_branch = trainer.train(u0_train)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history_branch, color=C0, lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Coefficient MSE")
    ax.set_title("Phase 2: branch network training loss", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "training_dynamics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Evaluation ---
    true_train = s_traj.cpu().numpy()
    pred_train = trainer.predict(u0_train, x_flat).cpu().numpy()
    true_test  = tensor_test.reshape(args.n_test, -1)
    pred_test  = trainer.predict(u0_test, x_flat).cpu().numpy()

    err_train = rel_l2(true_train, pred_train)
    err_test  = rel_l2(true_test,  pred_test)

    print(f"Train | mean={err_train.mean():.4f}  median={np.median(err_train):.4f}  std={err_train.std():.4f}")
    print(f"Test  | mean={err_test.mean():.4f}  median={np.median(err_test):.4f}  std={err_test.std():.4f}  p95={np.percentile(err_test, 95):.4f}")

    metrics = {
        "run_name":     RUN_NAME,
        "n_modes":      int(K),
        "n_train":      args.n_train,
        "n_test":       args.n_test,
        "train_mean":   float(err_train.mean()),
        "train_median": float(np.median(err_train)),
        "train_std":    float(err_train.std()),
        "test_mean":    float(err_test.mean()),
        "test_median":  float(np.median(err_test)),
        "test_std":     float(err_test.std()),
        "test_p95":     float(np.percentile(err_test, 95)),
    }

    # --- Cross-nu generalization ---
    DATA_DIR = pathlib.Path(os.path.expanduser("~/data/1D/Burgers/Train"))
    cross_nu_metrics = {}
    for nu in args.all_nu:
        fpath = DATA_DIR / f"1D_Burgers_Sols_Nu{nu}.hdf5"
        if not fpath.exists():
            print(f"  nu={nu:.3f}: file not found, skipping")
            continue
        with h5py.File(fpath, "r") as f:
            raw_nu = f["tensor"][-args.n_test:]
            if raw_nu.ndim == 4:
                raw_nu = raw_nu[..., 0]
        u0_nu  = torch.tensor(raw_nu[:, 0, :], dtype=torch.float32).to(DEVICE)
        s_nu   = raw_nu.reshape(args.n_test, -1)
        pred_nu = trainer.predict(u0_nu, x_flat).cpu().numpy()
        err_nu  = rel_l2(s_nu, pred_nu)
        cross_nu_metrics[nu] = {
            "mean":   float(err_nu.mean()),
            "median": float(np.median(err_nu)),
            "std":    float(err_nu.std()),
            "p95":    float(np.percentile(err_nu, 95)),
        }
        tag = " (trained)" if nu == TRAIN_NU else ""
        print(f"  nu={nu:.3f}{tag}: mean={err_nu.mean():.4f}  median={np.median(err_nu):.4f}  std={err_nu.std():.4f}")

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
        if TRAIN_NU in nu_keys:
            ax.axvline(nu_keys.index(TRAIN_NU), color="gray", ls="--", lw=1.2, alpha=0.7, label="Trained on")
        ax.set_xticks(x_pos); ax.set_xticklabels(nu_labels)
        ax.set_ylabel("Relative L2 error")
        ax.set_title("NeuralPOD-DeepONet - cross-nu generalization", fontweight="bold")
        ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, "cross_nu.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # --- Reconstruction examples ---
    rng  = np.random.default_rng(args.seed)
    idxs = rng.choice(args.n_test, size=args.n_viz, replace=False)

    fig, axes = plt.subplots(args.n_viz, 3, figsize=(14, 3 * args.n_viz))
    if args.n_viz == 1:
        axes = axes[None, :]
    for row, idx in enumerate(idxs):
        pred = trainer.predict(u0_test[idx:idx + 1], x_flat).reshape(Nt, Nx).cpu().numpy()
        true = tensor_test[idx]
        err  = np.abs(true - pred)
        vmax = np.abs(true).max()
        rl2  = np.linalg.norm(true - pred) / np.linalg.norm(true)
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

    # --- Checkpoint ---
    torch.save({
        "model":    model.state_dict(),
        "basis":    basis.state_dict(),
        "cfg":      cfg,
        "cfg_pod":  cfg_pod,
        "metrics":  metrics,
        "run_name": RUN_NAME,
    }, os.path.join(RUN_DIR, "model.pt"))

    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {os.path.abspath(RUN_DIR)}")


if __name__ == "__main__":
    main()
