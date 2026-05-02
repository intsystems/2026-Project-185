#!/usr/bin/env python
"""Train joint NeuralPOD-DeepONet on all beta values simultaneously (Darcy flow)."""
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
from models.regime_basis import FourierRegimeBasis
from models.fourier_neural_pod import FourierNeuralPODTrainer, FourierNeuralPODConfig
from models.pod_deeponet import BranchNet
from utils.datasets import load_darcy_stacked
from utils.plotting import plot_cross_param_bar, plot_reconstruction_xy


def rel_l2(true, pred):
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


_SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


def _data_path(beta: float, data_dir: str) -> str:
    return os.path.join(data_dir, f"2D_DarcyFlow_beta{beta}_Train.hdf5")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",        type=str,   default=None)
    p.add_argument("--results_dir",     type=str,   default=str(_PROJECT_ROOT / "TEMPO_results" / "darcy"))
    p.add_argument("--beta_values",     type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    p.add_argument("--n_samples",       type=int,   default=8000, help="Samples per beta (train+test)")
    p.add_argument("--n_test_per_beta", type=int,   default=1000)
    p.add_argument("--data_dir",        type=str,   default=os.path.expanduser("~/data/2D/DarcyFlow"))
    # Fourier basis
    p.add_argument("--max_modes",       type=int,   default=32)
    p.add_argument("--n_epochs_mean",   type=int,   default=165)
    p.add_argument("--n_epochs_mode",   type=int,   default=50)
    p.add_argument("--hidden_dim_basis", type=int,  default=256)
    p.add_argument("--num_frequencies", type=int,   default=96)
    p.add_argument("--scales",          type=float, nargs="+", default=[0.5, 2.0, 6.0])
    p.add_argument("--n_layers_basis",  type=int,   default=3)
    # Branch network
    p.add_argument("--hidden_dim",      type=int,   default=256)
    p.add_argument("--n_layers",        type=int,   default=4)
    p.add_argument("--sensor_stride",   type=int,   default=4)
    # Training
    p.add_argument("--n_epochs",        type=int,   default=600)
    p.add_argument("--batch_size",      type=int,   default=1024)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--log_every",       type=int,   default=500)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--n_viz",           type=int,   default=2)
    return p.parse_args()


def main():
    args = parse_args()
    RUN_NAME = args.run_name or "npod_deeponet_joint_darcy_v1"
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
    entries = []
    for beta in args.beta_values:
        fpath = _data_path(beta, args.data_dir)
        if not os.path.exists(fpath):
            print(f"  beta={beta}: file not found, skipping")
            continue
        entries.append((beta, fpath))

    if not entries:
        raise RuntimeError("No data files found. Check --data_dir.")

    beta_loaded = [e[0] for e in entries]
    s_np, a_np, kappa_np, xy_np, Nx, Ny = load_darcy_stacked(entries, n_samples=args.n_samples)
    Nxy = Nx * Ny

    x_np_1d = xy_np[::Ny, 0]
    y_np_1d = xy_np[:Ny, 1]

    s     = torch.from_numpy(s_np);              del s_np
    a     = torch.from_numpy(a_np);              del a_np
    kappa = torch.from_numpy(kappa_np[:, None])  # (N, 1)

    N_per_beta = args.n_samples
    train_idx = torch.cat([
        torch.arange(i * N_per_beta, (i + 1) * N_per_beta - args.n_test_per_beta)
        for i in range(len(beta_loaded))
    ])
    test_idx = torch.cat([
        torch.arange((i + 1) * N_per_beta - args.n_test_per_beta, (i + 1) * N_per_beta)
        for i in range(len(beta_loaded))
    ])

    s_train     = s[train_idx];      s_test     = s[test_idx]
    a_train     = a[train_idx];      a_test     = a[test_idx]
    kappa_train = kappa[train_idx];  kappa_test = kappa[test_idx]

    N_train = len(s_train)
    x_flat  = torch.tensor(xy_np, dtype=torch.float32).to(DEVICE)

    import math
    m = math.ceil(Nxy / args.sensor_stride)
    print(f"s_train={tuple(s_train.shape)}, Nxy={Nxy}, m={m}")

    # --- Phase 1: Fourier NeuralPOD on joint training data ---
    print("=== Phase 1: Fourier NeuralPOD ===")
    w = torch.ones(Nxy, dtype=torch.float32).to(DEVICE) / Nxy
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
    axes[0].semilogy(history_pod.residual_norms, "o-", color=plt.cm.tab10(0), markersize=4, lw=1.5)
    axes[0].set_xlabel("Mode index"); axes[0].set_ylabel("Weighted residual norm")
    axes[0].set_title("Phase 1: residual per mode", fontweight="bold")
    axes[0].grid(True, ls="--", alpha=0.25); axes[0].spines[["top", "right"]].set_visible(False)
    logs = [v for sublist in history_pod.mode_losses for v in sublist]
    axes[1].semilogy(logs, color=plt.cm.tab10(0), lw=0.8)
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

    # --- Phase 2: Branch network with kappa input ---
    print("=== Phase 2: Branch network (joint, d_kappa=1) ===")
    branch = BranchNet(m=m, P=K, hidden_dim=args.hidden_dim,
                       n_layers=args.n_layers, d_kappa=1).to(DEVICE)

    # Extract train coefficients: (N_train, K)
    coeffs = torch.stack([mode.lambda_ten.detach() for mode in basis.modes], dim=1).to(DEVICE)

    # Precompute val targets via greedy sequential projection (matches training)
    with torch.no_grad():
        mean_d   = basis.mean_net(x_flat)                               # (Nxy,)
        residual = s_test.to(DEVICE) - mean_d.unsqueeze(0)             # (N_test, Nxy)
        val_tgt_list = []
        for mode in basis.modes:
            phi_k    = mode.phi(x_flat)                                 # (Nxy,)
            phi_sq_k = (phi_k ** 2).sum()
            c_k      = (residual @ phi_k) / phi_sq_k                   # (N_test,)
            val_tgt_list.append(c_k)
            residual = residual - c_k.unsqueeze(1) * phi_k.unsqueeze(0)
        val_targets = torch.stack(val_tgt_list, dim=1)                  # (N_test, K)

    a_sensors     = a_train.to(DEVICE)[:, ::args.sensor_stride]   # (N_train, m)
    a_val_sensors = a_test.to(DEVICE)[:, ::args.sensor_stride]    # (N_test, m)
    kappa_train_d = kappa_train.to(DEVICE)
    kappa_test_d  = kappa_test.to(DEVICE)
    print(f"Training | N={len(a_sensors)}, m={a_sensors.shape[1]}, K={K}")

    dl  = DataLoader(TensorDataset(a_sensors, kappa_train_d, coeffs),
                     batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(branch.parameters(), lr=args.lr, weight_decay=1e-4)

    VAL_EVERY = 50
    history_branch, history_val = [], []
    for epoch in range(args.n_epochs):
        branch.train()
        total = 0.0
        for a_b, kappa_b, coeff_b in dl:
            opt.zero_grad()
            loss = F.mse_loss(branch(a_b, kappa_b), coeff_b)
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(dl)
        history_branch.append(avg)
        if epoch % VAL_EVERY == 0:
            branch.eval()
            with torch.no_grad():
                vl = F.mse_loss(branch(a_val_sensors, kappa_test_d), val_targets).item()
            branch.train()
            history_val.append((epoch, vl))
        if epoch % args.log_every == 0:
            val_str = f"  val={history_val[-1][1]:.4e}" if history_val else ""
            print(f"  epoch {epoch:5d} | coeff_mse={avg:.4e}{val_str}")
    val_str = f"  val={history_val[-1][1]:.4e}" if history_val else ""
    print(f"  done: final coeff_mse={history_branch[-1]:.4e}{val_str}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history_branch, color=plt.cm.tab10(0), lw=1.5, label="Train")
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
    @torch.no_grad()
    def predict_batch(a_in, kappa_in, batch_size=128):
        branch.eval()
        basis.eval()
        phi = torch.stack([mode.phi(x_flat) for mode in basis.modes], dim=1)  # (Nxy, K)
        mean = basis.mean_net(x_flat)                                           # (Nxy,)
        parts = []
        for i in range(0, len(a_in), batch_size):
            a_s    = a_in[i:i + batch_size].to(DEVICE)[:, ::args.sensor_stride]
            k_s    = kappa_in[i:i + batch_size].to(DEVICE)
            beta_v = branch(a_s, k_s)                    # (B, K)
            pred   = mean.unsqueeze(0) + beta_v @ phi.T  # (B, Nxy)
            parts.append(pred.cpu())
        return torch.cat(parts, dim=0).numpy()

    kappa_t_np  = kappa_test[:, 0].numpy()
    beta_unique = np.unique(kappa_t_np)
    s_pred      = predict_batch(a_test, kappa_test)
    s_test_np   = s_test.numpy()
    rel_l2_all  = rel_l2(s_test_np, s_pred)

    print("Mean rel L2 error (test):")
    metrics = {
        "run_name":       RUN_NAME,
        "n_modes":        int(K),
        "n_train":        int(N_train),
        "n_test":         int(len(s_test)),
        "overall_mean":   float(rel_l2_all.mean()),
        "overall_median": float(np.median(rel_l2_all)),
    }
    cross_beta_metrics = {}
    for beta in beta_unique:
        mask    = kappa_t_np == beta
        m_err   = float(rel_l2_all[mask].mean())
        med_err = float(np.median(rel_l2_all[mask]))
        metrics[f"beta{beta:.4g}_mean"]   = m_err
        metrics[f"beta{beta:.4g}_median"] = med_err
        metrics[f"beta{beta:.4g}_std"]    = float(rel_l2_all[mask].std())
        cross_beta_metrics[beta] = {
            "mean":   m_err, "median": med_err,
            "std":    float(rel_l2_all[mask].std()),
            "p95":    float(np.percentile(rel_l2_all[mask], 95)),
        }
        print(f"  beta={beta}: mean={m_err:.4f}  median={med_err:.4f}")

    plot_cross_param_bar(
        cross_beta_metrics, None, r"$\beta$",
        "NeuralPOD-DeepONet (joint) Darcy",
        os.path.join(RUN_DIR, "cross_beta.png"),
    )

    # --- Reconstruction examples ---
    rng = np.random.default_rng(args.seed)
    true_list, pred_list, rl2_list, row_labels = [], [], [], []
    for beta in beta_unique:
        idxs = np.where(kappa_t_np == beta)[0]
        chosen = rng.choice(idxs, size=min(args.n_viz, len(idxs)), replace=False)
        for idx in chosen:
            t = s_test_np[idx].reshape(Nx, Ny)
            p = s_pred[idx].reshape(Nx, Ny)
            true_list.append(t); pred_list.append(p)
            rl2_list.append(float(np.linalg.norm(t - p) / np.linalg.norm(t)))
            row_labels.append(f"$\\beta$={beta:.4g}")

    plot_reconstruction_xy(
        true_list, pred_list, rl2_list, x_np_1d, y_np_1d,
        "NeuralPOD-DeepONet (joint)", os.path.join(RUN_DIR, "reconstruct.png"),
        row_labels=row_labels,
    )

    # --- Checkpoint ---
    torch.save({
        "model":    branch.state_dict(),
        "basis":    basis.state_dict(),
        "metrics":  metrics,
        "run_name": RUN_NAME,
    }, os.path.join(RUN_DIR, "model.pt"))

    metrics["hparams"] = vars(args)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {os.path.abspath(RUN_DIR)}")


if __name__ == "__main__":
    main()
