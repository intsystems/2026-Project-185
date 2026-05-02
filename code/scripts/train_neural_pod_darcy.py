#!/usr/bin/env python
"""Train NeuralPOD-DeepONet on 2D Darcy Flow for a single beta value."""
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from models.regime_basis import FourierRegimeBasis
from models.fourier_neural_pod import FourierNeuralPODTrainer, FourierNeuralPODConfig
from models.pod_deeponet import BranchNet
from models.neural_pod_deeponet import NeuralPODDeepONet, NeuralPODDeepONetConfig, NeuralPODDeepONetTrainer
from utils.datasets import load_darcy_stacked, DATA_DIR, DARCY_DATASETS
from utils.plotting import (
    plot_error_dist, plot_cross_param_bar, plot_reconstruction_xy,
)


def rel_l2(true, pred):
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


_SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--beta",              type=float, required=True)
    p.add_argument("--run_name",          type=str,   default=None)
    p.add_argument("--results_dir",       type=str,   default=str(_PROJECT_ROOT / "TEMPO_results" / "darcy"))
    p.add_argument("--all_beta",          type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--n_train",           type=int,   default=8000)
    p.add_argument("--n_test",            type=int,   default=1000)
    # Fourier basis
    p.add_argument("--max_modes",         type=int,   default=32)
    p.add_argument("--n_epochs_mean",     type=int,   default=800)
    p.add_argument("--n_epochs_mode",     type=int,   default=1200)
    p.add_argument("--hidden_dim_basis",  type=int,   default=256)
    p.add_argument("--num_frequencies",   type=int,   default=96)
    p.add_argument("--scales",            type=float, nargs="+", default=[0.5, 2.0, 6.0])
    p.add_argument("--n_layers_basis",    type=int,   default=3)
    # Branch network
    p.add_argument("--hidden_dim",        type=int,   default=256)
    p.add_argument("--n_layers",          type=int,   default=4)
    p.add_argument("--sensor_stride",     type=int,   default=4)
    # Training
    p.add_argument("--n_epochs",          type=int,   default=800)
    p.add_argument("--batch_size",        type=int,   default=1024)
    # Misc
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--n_viz",             type=int,   default=3)
    return p.parse_args()


def _data_path(beta: float) -> str:
    filename = f"2D_DarcyFlow_beta{beta}_Train.hdf5"
    local  = pathlib.Path(DATA_DIR) / filename
    server = pathlib.Path(os.path.expanduser("~/data/2D/DarcyFlow")) / filename
    return str(local) if local.exists() else str(server)


def main():
    args = parse_args()

    TRAIN_BETA = args.beta
    RUN_NAME   = args.run_name or f"npod_deeponet_darcy_beta{TRAIN_BETA}_v1"
    RUN_DIR    = os.path.join(args.results_dir, RUN_NAME)
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

    C0 = plt.cm.tab10(0)

    # --- Data loading ---
    path = _data_path(TRAIN_BETA)
    print(f"data: {path}")

    s_np, a_np, _, xy_np, Nx, Ny = load_darcy_stacked(
        [(TRAIN_BETA, path)], n_samples=args.n_train + args.n_test
    )
    Nxy = Nx * Ny
    print(f"loaded: N={len(s_np)}, Nx={Nx}, Ny={Ny}, Nxy={Nxy}")

    s_train = s_np[:args.n_train];  s_test = s_np[args.n_train:]
    a_train = a_np[:args.n_train];  a_test = a_np[args.n_train:]
    del s_np, a_np

    x_np = xy_np[::Ny, 0]
    y_np = xy_np[:Ny, 1]

    s_traj      = torch.tensor(s_train, dtype=torch.float32)
    a_t_train   = torch.tensor(a_train, dtype=torch.float32).to(DEVICE)
    a_t_test    = torch.tensor(a_test,  dtype=torch.float32).to(DEVICE)
    s_test_flat = torch.tensor(s_test,  dtype=torch.float32)

    x_flat = torch.tensor(xy_np, dtype=torch.float32).to(DEVICE)  # (Nxy, 2)
    m = len(range(0, Nxy, args.sensor_stride))
    print(f"s_traj: {tuple(s_traj.shape)}, x_flat: {tuple(x_flat.shape)}, m={m}")

    # --- Phase 1: Fourier Neural POD ---
    print("=== Phase 1: Fourier Neural POD ===")
    w = torch.ones(Nxy, dtype=torch.float32).to(DEVICE) / Nxy
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
    K      = basis.num_modes
    branch = BranchNet(m=m, P=K, hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(DEVICE)
    model  = NeuralPODDeepONet(basis, branch).to(DEVICE)

    cfg     = NeuralPODDeepONetConfig(n_epochs=args.n_epochs, batch_size=args.batch_size,
                                      sensor_stride=args.sensor_stride)
    trainer = NeuralPODDeepONetTrainer(model, cfg)
    history_branch = trainer.train(a_t_train, val_u0=a_t_test, val_s=s_test_flat, x_flat=x_flat)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history_branch, color=C0, lw=1.5, label="Train")
    if trainer.val_history:
        ve, vl = zip(*trainer.val_history)
        ax.semilogy(ve, vl, color=plt.cm.tab10(1), lw=1.5, ls="--", label="Val")
        ax.legend(framealpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Coefficient MSE")
    ax.set_title("Phase 2: branch network", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "training_dynamics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Evaluation ---
    def predict_batched(a_in, batch_size=256):
        parts = []
        for i in range(0, len(a_in), batch_size):
            with torch.no_grad():
                parts.append(trainer.predict(a_in[i:i + batch_size], x_flat).cpu())
        return torch.cat(parts, dim=0).numpy()

    err_train = rel_l2(s_train, predict_batched(a_t_train))
    err_test  = rel_l2(s_test,  predict_batched(a_t_test))

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

    # --- Cross-beta generalization ---
    cross_beta_metrics = {}
    for beta in args.all_beta:
        fpath = _data_path(beta)
        if not os.path.exists(fpath):
            print(f"  beta={beta}: file not found, skipping")
            continue
        s_b, a_b, _, _, _, _ = load_darcy_stacked([(beta, fpath)], n_samples=args.n_test)
        a_b_t  = torch.tensor(a_b, dtype=torch.float32).to(DEVICE)
        err_b  = rel_l2(s_b, predict_batched(a_b_t))
        cross_beta_metrics[beta] = {
            "mean":   float(err_b.mean()),
            "median": float(np.median(err_b)),
            "std":    float(err_b.std()),
            "p95":    float(np.percentile(err_b, 95)),
        }
        tag = " (trained)" if beta == TRAIN_BETA else ""
        print(f"  beta={beta}{tag}: mean={err_b.mean():.4f}  median={np.median(err_b):.4f}")

    metrics["cross_beta"] = cross_beta_metrics

    if cross_beta_metrics:
        plot_cross_param_bar(
            cross_beta_metrics, TRAIN_BETA, "beta",
            "NeuralPOD-DeepONet - cross-beta generalization",
            os.path.join(RUN_DIR, "cross_beta.png"),
        )

    # --- Reconstruction examples ---
    rng  = np.random.default_rng(args.seed)
    idxs = rng.choice(args.n_test, size=args.n_viz, replace=False)
    preds = predict_batched(a_t_test[idxs])

    true_list = [s_test[i].reshape(Nx, Ny) for i in idxs]
    pred_list = [preds[k].reshape(Nx, Ny)  for k in range(args.n_viz)]
    rl2_list  = [
        float(np.linalg.norm(s_test[i] - preds[k]) / np.linalg.norm(s_test[i]))
        for k, i in enumerate(idxs)
    ]

    plot_reconstruction_xy(
        true_list, pred_list, rl2_list, x_np, y_np,
        "NeuralPOD-DeepONet", os.path.join(RUN_DIR, "npod_reconstruction.png"),
        row_labels=[f"beta={TRAIN_BETA}  sample {i}" for i in idxs],
    )

    plot_error_dist(err_test, "NeuralPOD-DeepONet Darcy - test errors",
                    os.path.join(RUN_DIR, "npod_err_dist.png"))

    # --- Checkpoint ---
    torch.save({
        "model":    model.state_dict(),
        "basis":    basis.state_dict(),
        "cfg":      cfg,
        "cfg_pod":  cfg_pod,
        "metrics":  metrics,
        "run_name": RUN_NAME,
    }, os.path.join(RUN_DIR, "model.pt"))

    metrics["hparams"] = vars(args)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {os.path.abspath(RUN_DIR)}")


if __name__ == "__main__":
    main()
