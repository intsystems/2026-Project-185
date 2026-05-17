#!/usr/bin/env python
"""Train POD-DeepONet on 1D Burgers — specialist (single nu) or joint (multiple nu).

Usage:
  Specialist: python train_pod.py --nu_values 0.001
  Joint:      python train_pod.py --nu_values 0.001 0.01 0.1 1.0
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
from models.pod import PODTrainer, PODConfig
from models.pod_deeponet import BranchNet, PODDeepONet
from utils.datasets import load_stacked


def rel_l2(true, pred):
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


_SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nu_values",    type=float, nargs="+", required=True,
                   help="One value = specialist; multiple = joint")
    p.add_argument("--run_name",     type=str,   default=None)
    p.add_argument("--results_dir",  type=str,   default=str(_PROJECT_ROOT / "TEMPO_results" / "burgers"))
    p.add_argument("--n_samples",    type=int,   default=9500,
                   help="Samples per nu loaded (train + test)")
    p.add_argument("--n_test_per_nu", type=int,  default=1000)
    p.add_argument("--data_dir",     type=str,   default=os.path.expanduser("~/data/1D/Burgers/Train"))
    p.add_argument("--max_modes",    type=int,   default=32)
    p.add_argument("--hidden_dim",   type=int,   default=256)
    p.add_argument("--n_layers",     type=int,   default=4)
    p.add_argument("--sensor_stride", type=int,  default=1)
    p.add_argument("--n_epochs",     type=int,   default=1200)
    p.add_argument("--batch_size",   type=int,   default=1024)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--log_every",    type=int,   default=100)
    p.add_argument("--seed",         type=int,   default=39)
    p.add_argument("--n_viz",        type=int,   default=3)
    return p.parse_args()


def main():
    args = parse_args()

    joint = len(args.nu_values) > 1
    if joint:
        RUN_NAME = args.run_name or "pod_deeponet_joint_burgers_v1"
    else:
        RUN_NAME = args.run_name or f"pod_deeponet_nu{args.nu_values[0]}_v1"
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

    # --- Data loading ---
    data_dir = pathlib.Path(args.data_dir)
    entries = []
    for nu in args.nu_values:
        _local  = _PROJECT_ROOT / "data" / f"Burgers_Nu{nu}.hdf5"
        _server = data_dir / f"1D_Burgers_Sols_Nu{nu}.hdf5"
        path = str(_local) if _local.exists() else str(_server)
        entries.append((nu, path))

    s_np, kappa_np, x_np, t_np, Nx, Nt = load_stacked(entries, n_samples=args.n_samples)
    # s_np: (N_total, Nt*Nx), Nt-slow convention; u0 = s_np[:, :Nx]

    n_nu    = len(args.nu_values)
    N_per   = args.n_samples
    train_idx = np.concatenate([
        np.arange(i * N_per, (i + 1) * N_per - args.n_test_per_nu)
        for i in range(n_nu)
    ])
    test_idx = np.concatenate([
        np.arange((i + 1) * N_per - args.n_test_per_nu, (i + 1) * N_per)
        for i in range(n_nu)
    ])

    s       = torch.from_numpy(s_np)
    u0_all  = torch.from_numpy(s_np[:, :Nx])      # (N, Nx) — initial condition
    kappa   = torch.from_numpy(kappa_np[:, None])  # (N, 1)

    s_train     = s[train_idx];       s_test     = s[test_idx]
    u0_train    = u0_all[train_idx];  u0_test    = u0_all[test_idx]
    kappa_train = kappa[train_idx];   kappa_test = kappa[test_idx]

    s_train_dev   = s_train.to(DEVICE)
    u0_train_dev  = u0_train.to(DEVICE)
    u0_test_dev   = u0_test.to(DEVICE)
    kappa_train_d = kappa_train.to(DEVICE)
    kappa_test_d  = kappa_test.to(DEVICE)

    N_train = len(train_idx)
    N_test  = len(test_idx)
    m = len(range(0, Nx, args.sensor_stride))
    print(f"N_train={N_train}  N_test={N_test}  Nx={Nx}  Nt={Nt}  m={m}")

    # --- Phase 1: POD on training data ---
    print("=== Phase 1: POD ===")
    trainer_pod = PODTrainer(PODConfig(max_modes=args.max_modes))
    trainer_pod.train(s_train_dev, x=None, t=None)
    P = trainer_pod.basis.num_modes
    print(f"P={P} modes")

    sigmas = trainer_pod.basis.coeffs.cpu().numpy().std(axis=0) * np.sqrt(N_train - 1)
    energy = sigmas ** 2
    cumvar = np.cumsum(energy) / energy.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].semilogy(range(1, len(sigmas) + 1), sigmas, "o-", color=C0, markersize=4, lw=1.5)
    axes[0].set_xlabel("Mode index"); axes[0].set_ylabel("Singular value")
    axes[0].set_title("Phase 1: singular value decay", fontweight="bold")
    axes[0].grid(True, ls="--", alpha=0.25); axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].plot(range(1, len(cumvar) + 1), cumvar, "o-", color=C0, markersize=4, lw=1.5)
    axes[1].axhline(99.99, color=C1, ls="--", lw=1.2, label="99.99%")
    axes[1].set_xlabel("Mode index"); axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title("Phase 1: cumulative variance explained", fontweight="bold")
    axes[1].legend(framealpha=0.7); axes[1].grid(True, ls="--", alpha=0.25)
    axes[1].spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "pod_phase1.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Phase 2: Branch network (always d_kappa=1) ---
    print(f"=== Phase 2: Branch network (d_kappa=1, {'joint' if joint else 'specialist'}) ===")
    mean_dev  = trainer_pod.basis.mean.to(DEVICE)
    modes_dev = trainer_pod.basis.modes.to(DEVICE)

    u0_sensors     = u0_train_dev[:, ::args.sensor_stride]  # (N_train, m)
    u0_val_sensors = u0_test_dev[:, ::args.sensor_stride]   # (N_test,  m)
    targets    = trainer_pod.basis.coeffs.to(DEVICE)         # (N_train, P)
    val_targets = (s_test.to(DEVICE) - mean_dev.unsqueeze(0)) @ modes_dev  # (N_test, P)

    branch = BranchNet(m=m, P=P, hidden_dim=args.hidden_dim,
                       n_layers=args.n_layers, d_kappa=1).to(DEVICE)
    model  = PODDeepONet(trainer_pod.basis, branch).to(DEVICE)

    dl  = DataLoader(TensorDataset(u0_sensors, kappa_train_d, targets),
                     batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(branch.parameters(), lr=args.lr, weight_decay=1e-4)
    VAL_EVERY = max(1, args.n_epochs // 20)

    history, history_val = [], []
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
        history.append(avg)
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
    print(f"  done: final coeff_mse={history[-1]:.4e}{val_str}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history, color=C0, lw=1.5, label="Train")
    if history_val:
        ve, vl = zip(*history_val)
        ax.semilogy(ve, vl, color=C1, lw=1.5, ls="--", label="Val")
        ax.legend(framealpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Coefficient MSE")
    ax.set_title("Phase 2: branch network", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "training_dynamics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Evaluation helpers ---
    def predict_batch(u0_in, kappa_in, batch_size=256):
        branch.eval()
        parts = []
        for i in range(0, len(u0_in), batch_size):
            with torch.no_grad():
                u0_s  = u0_in[i:i + batch_size, ::args.sensor_stride]
                k_s   = kappa_in[i:i + batch_size]
                beta  = branch(u0_s, k_s)
                pred  = mean_dev + beta @ modes_dev.T
                parts.append(pred.cpu())
        return torch.cat(parts, dim=0).numpy()

    # Sample-based train error (up to 2000 samples to avoid OOM)
    idx_sample = torch.randperm(N_train)[:2000]
    err_train = rel_l2(
        s_train[idx_sample].numpy(),
        predict_batch(u0_train_dev[idx_sample], kappa_train_d[idx_sample])
    )
    s_test_np  = s_test.numpy()
    pred_test  = predict_batch(u0_test_dev, kappa_test_d)
    err_test   = rel_l2(s_test_np, pred_test)

    print(f"Train | mean={err_train.mean():.4f}  median={np.median(err_train):.4f}  std={err_train.std():.4f}")
    print(f"Test  | mean={err_test.mean():.4f}  median={np.median(err_test):.4f}  std={err_test.std():.4f}  p95={np.percentile(err_test, 95):.4f}")

    metrics = {
        "run_name":  RUN_NAME,
        "n_modes":   int(P),
        "n_train":   N_train,
        "n_test":    N_test,
    }

    if not joint:
        # Specialist: standard per-split metrics + cross-nu evaluation
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
            u0_nu    = torch.from_numpy(raw_nu[:, 0, :]).to(DEVICE)
            s_nu_flat = raw_nu.reshape(args.n_test_per_nu, -1).astype(np.float32)
            # Pass actual nu so model receives correct parameter signal
            kappa_nu = torch.full((args.n_test_per_nu, 1), float(nu_eval),
                                  dtype=torch.float32, device=DEVICE)
            pred_nu  = predict_batch(u0_nu, kappa_nu)
            err_nu   = rel_l2(s_nu_flat, pred_nu)
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
            ax.set_title("POD-DeepONet - cross-nu generalization", fontweight="bold")
            ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25, axis="y")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(RUN_DIR, "cross_nu.png"), dpi=150, bbox_inches="tight")
            plt.close()

    else:
        # Joint: per-nu breakdown on test set
        metrics.update({
            "overall_mean":   float(err_test.mean()),
            "overall_median": float(np.median(err_test)),
            "overall_std":    float(err_test.std()),
        })
        kappa_test_np = kappa_test[:, 0].numpy()
        nu_unique     = np.array(args.nu_values)
        print("Mean rel L2 error per nu (test):")
        for i, nu in enumerate(nu_unique):
            sl = slice(i * args.n_test_per_nu, (i + 1) * args.n_test_per_nu)
            err_nu = err_test[sl]
            metrics[f"nu{nu:.3f}_mean"]   = float(err_nu.mean())
            metrics[f"nu{nu:.3f}_median"] = float(np.median(err_nu))
            metrics[f"nu{nu:.3f}_std"]    = float(err_nu.std())
            print(f"  nu={nu:.3f}: mean={err_nu.mean():.4f}  "
                  f"median={np.median(err_nu):.4f}  std={err_nu.std():.4f}")

        x_pos   = np.arange(len(nu_unique))
        means   = [metrics[f"nu{nu:.3f}_mean"]   for nu in nu_unique]
        medians = [metrics[f"nu{nu:.3f}_median"] for nu in nu_unique]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x_pos - 0.2,  means,   0.35, label="Mean",   color=C0, alpha=0.85, linewidth=0)
        ax.bar(x_pos + 0.15, medians, 0.35, label="Median", color=C1, alpha=0.85, linewidth=0)
        ax.set_xticks(x_pos); ax.set_xticklabels([f"nu={nu:.3f}" for nu in nu_unique])
        ax.set_ylabel("Relative L2 error")
        ax.set_title("POD-DeepONet (joint) - error per nu", fontweight="bold")
        ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, "error_per_nu.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # --- Reconstruction examples ---
    rng = np.random.default_rng(args.seed)
    t_plot = t_np[:Nt]

    if not joint:
        idxs = rng.choice(N_test, size=args.n_viz, replace=False)
        pred_viz = predict_batch(u0_test_dev[idxs], kappa_test_d[idxs])

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
                (true, "Ground Truth",   "RdBu_r", -vmax, vmax),
                (pred, "POD-DeepONet",   "RdBu_r", -vmax, vmax),
                (err,  "Absolute Error", "Oranges",  0,    err.max()),
            ]):
                ax = axes[row, col]
                im = ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap,
                               extent=[x_np.min(), x_np.max(), t_plot.min(), t_plot.max()],
                               vmin=vmin, vmax=vm)
                if row == 0:
                    ax.set_title(title, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(f"t   (rel L2={rl2:.3f})")
                if row == args.n_viz - 1:
                    ax.set_xlabel("x")
                ax.spines[["top", "right"]].set_visible(False)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.suptitle("POD-DeepONet: reconstruction examples", fontweight="bold", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, "pod_reconstruction.png"), dpi=150, bbox_inches="tight")
        plt.close()
    else:
        # Show n_viz samples per nu
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
                true = s_test_np[abs_idx].reshape(Nt, Nx)
                pred_flat = predict_batch(u0_test_dev[abs_idx:abs_idx + 1],
                                          kappa_test_d[abs_idx:abs_idx + 1])
                pred = pred_flat[0].reshape(Nt, Nx)
                err  = np.abs(true - pred)
                vmax = np.abs(true).max()
                rl2  = float(np.linalg.norm(true - pred) / np.linalg.norm(true))
                for col, (arr, title, cmap, vmin, vm) in enumerate([
                    (true, "Ground Truth",   "RdBu_r", -vmax, vmax),
                    (pred, "POD-DeepONet",   "RdBu_r", -vmax, vmax),
                    (err,  "Absolute Error", "Oranges",  0,    err.max()),
                ]):
                    ax = axes[row, col]
                    im = ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap,
                                   extent=[x_np.min(), x_np.max(), t_plot.min(), t_plot.max()],
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
        plt.suptitle("POD-DeepONet (joint): reconstruction examples", fontweight="bold", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, "pod_reconstruction.png"), dpi=150, bbox_inches="tight")
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
    plt.suptitle("POD-DeepONet - test set errors", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "pod_err_dist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Checkpoint ---
    torch.save({
        "model":    model.state_dict(),
        "metrics":  metrics,
        "run_name": RUN_NAME,
    }, os.path.join(RUN_DIR, "model.pt"))

    metrics["hparams"] = vars(args)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {os.path.abspath(RUN_DIR)}")


if __name__ == "__main__":
    main()
