#!/usr/bin/env python
"""Train joint POD-DeepONet on all viscosity values simultaneously (Burgers)."""
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from models.pod import PODTrainer, PODConfig
from models.pod_deeponet import BranchNet, PODDeepONet, PODDeepONetConfig
from utils.datasets import load_stacked


def rel_l2(true, pred):
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


_SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",      type=str,   default=None)
    p.add_argument("--results_dir",   type=str,   default=str(_PROJECT_ROOT / "TEMPO_results" / "burgers"))
    p.add_argument("--nu_values",     type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0])
    p.add_argument("--n_samples",     type=int,   default=5000, help="Samples per nu (train+test)")
    p.add_argument("--n_test_per_nu", type=int,   default=1000)
    p.add_argument("--data_dir",      type=str,   default=os.path.expanduser("~/data/1D/Burgers/Train"))
    p.add_argument("--max_modes",     type=int,   default=32)
    p.add_argument("--hidden_dim",    type=int,   default=256)
    p.add_argument("--n_layers",      type=int,   default=4)
    p.add_argument("--sensor_stride", type=int,   default=2)
    p.add_argument("--n_epochs",      type=int,   default=600)
    p.add_argument("--batch_size",    type=int,   default=1024)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--log_every",     type=int,   default=500)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    RUN_NAME = args.run_name or "pod_deeponet_joint_burgers_v1"
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

    # --- Data loading ---
    entries = [
        (nu, os.path.join(args.data_dir, f"1D_Burgers_Sols_Nu{nu}.hdf5"))
        for nu in args.nu_values
    ]
    s_np, kappa_np, x_np, t_np, Nx, Nt = load_stacked(entries, n_samples=args.n_samples)

    s     = torch.from_numpy(s_np);              del s_np
    kappa = torch.from_numpy(kappa_np[:, None])  # (N, 1)

    N_per_nu  = args.n_samples
    train_idx = torch.cat([
        torch.arange(i * N_per_nu, (i + 1) * N_per_nu - args.n_test_per_nu)
        for i in range(len(args.nu_values))
    ])
    test_idx = torch.cat([
        torch.arange((i + 1) * N_per_nu - args.n_test_per_nu, (i + 1) * N_per_nu)
        for i in range(len(args.nu_values))
    ])

    u0 = s[:, :Nx]
    s_train     = s[train_idx];      s_test     = s[test_idx]
    u0_train    = u0[train_idx];     u0_test    = u0[test_idx]
    kappa_train = kappa[train_idx];  kappa_test = kappa[test_idx]

    s_dev       = s_train.to(DEVICE)
    u0_train_d  = u0_train.to(DEVICE)
    u0_test_d   = u0_test.to(DEVICE)
    kappa_train_d = kappa_train.to(DEVICE)
    kappa_test_d  = kappa_test.to(DEVICE)

    import math
    m = math.ceil(Nx / args.sensor_stride)
    print(f"s_train={tuple(s_train.shape)}, m={m}")

    # --- Phase 1: POD on joint training data ---
    print("=== Phase 1: POD ===")
    trainer_pod = PODTrainer(PODConfig(max_modes=args.max_modes))
    trainer_pod.train(s_dev, x=None, t=None)
    P = trainer_pod.basis.num_modes
    print(f"P={P} modes")

    sigmas = trainer_pod.basis.coeffs.cpu().numpy().std(axis=0) * np.sqrt(len(s_dev) - 1)
    energy = sigmas ** 2
    cumvar = np.cumsum(energy) / energy.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].semilogy(range(1, len(sigmas) + 1), sigmas, "o-", color=plt.cm.tab10(0), markersize=4, lw=1.5)
    axes[0].set_xlabel("Mode index"); axes[0].set_ylabel("Singular value")
    axes[0].set_title("Phase 1: singular value decay", fontweight="bold")
    axes[0].grid(True, ls="--", alpha=0.25); axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].plot(range(1, len(cumvar) + 1), cumvar, "o-", color=plt.cm.tab10(0), markersize=4, lw=1.5)
    axes[1].axhline(99.99, color=plt.cm.tab10(1), ls="--", lw=1.2, label="99.99%")
    axes[1].set_xlabel("Mode index"); axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title("Phase 1: cumulative variance", fontweight="bold")
    axes[1].legend(framealpha=0.7); axes[1].grid(True, ls="--", alpha=0.25)
    axes[1].spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "pod_phase1.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Phase 2: Branch network with kappa input ---
    print("=== Phase 2: Branch network (joint, d_kappa=1) ===")
    branch = BranchNet(m=m, P=P, hidden_dim=args.hidden_dim,
                       n_layers=args.n_layers, d_kappa=1).to(DEVICE)
    model  = PODDeepONet(trainer_pod.basis, branch).to(DEVICE)

    mean_dev  = trainer_pod.basis.mean.to(DEVICE)
    modes_dev = trainer_pod.basis.modes.to(DEVICE)

    u0_sensors     = u0_train_d[:, ::args.sensor_stride]               # (N_train, m)
    targets        = trainer_pod.basis.coeffs.to(DEVICE)               # (N_train, P)
    u0_val_sensors = u0_test_d[:, ::args.sensor_stride]                # (N_test, m)
    val_targets    = (s_test.to(DEVICE) - mean_dev.unsqueeze(0)) @ modes_dev  # (N_test, P)
    print(f"Training | N={len(u0_sensors)}, m={u0_sensors.shape[1]}, P={P}")

    dl  = DataLoader(TensorDataset(u0_sensors, kappa_train_d, targets),
                     batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(branch.parameters(), lr=args.lr, weight_decay=1e-4)

    VAL_EVERY = 50
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
    ax.semilogy(history, color=plt.cm.tab10(0), lw=1.5, label="Train")
    if history_val:
        ve, vl = zip(*history_val)
        ax.semilogy(ve, vl, color=plt.cm.tab10(1), lw=1.5, ls="--", label="Val")
        ax.legend(framealpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Coefficient MSE")
    ax.set_title("Phase 2: branch network", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "training_dynamics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Evaluation ---

    def predict_batch(u0_in, kappa_in, batch_size=256):
        branch.eval()
        parts = []
        for i in range(0, len(u0_in), batch_size):
            with torch.no_grad():
                u0_s = u0_in[i:i + batch_size, ::args.sensor_stride]
                k_s  = kappa_in[i:i + batch_size]
                beta = branch(u0_s, k_s)
                pred = mean_dev + beta @ modes_dev.T
                parts.append(pred.cpu())
        return torch.cat(parts, dim=0).numpy()

    kappa_t_np = kappa_test[:, 0].numpy()
    nu_unique  = np.unique(kappa_t_np)
    s_pred     = predict_batch(u0_test_d, kappa_test_d)
    s_test_np  = s_test.numpy()
    rel_l2_all = rel_l2(s_test_np, s_pred)

    print("Mean rel L2 error (test):")
    metrics = {
        "run_name":       RUN_NAME,
        "n_modes":        int(P),
        "n_train":        int(len(s_train)),
        "n_test":         int(len(s_test)),
        "overall_mean":   float(rel_l2_all.mean()),
        "overall_median": float(np.median(rel_l2_all)),
    }
    for nu in nu_unique:
        mask = kappa_t_np == nu
        metrics[f"nu{nu:.3f}_mean"]   = float(rel_l2_all[mask].mean())
        metrics[f"nu{nu:.3f}_median"] = float(np.median(rel_l2_all[mask]))
        metrics[f"nu{nu:.3f}_std"]    = float(rel_l2_all[mask].std())
        print(f"  nu={nu:.3f}: mean={rel_l2_all[mask].mean():.4f}  "
              f"median={np.median(rel_l2_all[mask]):.4f}")

    # --- Gating weights bar (just error per nu) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos = np.arange(len(nu_unique))
    means   = [metrics[f"nu{nu:.3f}_mean"]   for nu in nu_unique]
    medians = [metrics[f"nu{nu:.3f}_median"] for nu in nu_unique]
    ax.bar(x_pos - 0.2,  means,   0.35, label="Mean",   color=plt.cm.tab10(0), alpha=0.85, linewidth=0)
    ax.bar(x_pos + 0.15, medians, 0.35, label="Median", color=plt.cm.tab10(1), alpha=0.85, linewidth=0)
    ax.set_xticks(x_pos); ax.set_xticklabels([f"nu={nu:.3f}" for nu in nu_unique])
    ax.set_ylabel("Relative L2 error")
    ax.set_title("POD-DeepONet (joint) - error per nu", fontweight="bold")
    ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "error_per_nu.png"), dpi=150, bbox_inches="tight")
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
