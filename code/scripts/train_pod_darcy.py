#!/usr/bin/env python
"""Train POD-DeepONet on 2D Darcy Flow for a single beta value."""
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
from models.pod import PODTrainer, PODConfig
from models.pod_deeponet import BranchNet, PODDeepONet, PODDeepONetConfig, PODDeepONetTrainer
from utils.datasets import load_darcy_stacked, DATA_DIR, DARCY_DATASETS
from utils.plotting import (
    plot_pod_phase1, plot_error_dist, plot_cross_param_bar, plot_reconstruction_xy,
)


def rel_l2(true, pred):
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


_SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--beta",         type=float, required=True)
    p.add_argument("--run_name",     type=str,   default=None)
    p.add_argument("--results_dir",  type=str,   default=str(_PROJECT_ROOT / "TEMPO_results" / "darcy"))
    p.add_argument("--all_beta",     type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--n_train",      type=int,   default=8000)
    p.add_argument("--n_test",       type=int,   default=1000)
    p.add_argument("--max_modes",    type=int,   default=32)
    p.add_argument("--hidden_dim",   type=int,   default=256)
    p.add_argument("--n_layers",     type=int,   default=4)
    p.add_argument("--sensor_stride", type=int,  default=4)
    p.add_argument("--n_epochs",     type=int,   default=500)
    p.add_argument("--batch_size",   type=int,   default=1024)
    p.add_argument("--seed",         type=int,   default=39)
    p.add_argument("--n_viz",        type=int,   default=3)
    return p.parse_args()


def _data_path(beta: float) -> str:
    filename = f"2D_DarcyFlow_beta{beta}_Train.hdf5"
    local  = pathlib.Path(DATA_DIR) / filename
    server = pathlib.Path(os.path.expanduser("~/data/2D/DarcyFlow")) / filename
    return str(local) if local.exists() else str(server)


def main():
    args = parse_args()

    TRAIN_BETA = args.beta
    RUN_NAME   = args.run_name or f"pod_deeponet_darcy_beta{TRAIN_BETA}_v1"
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

    # --- Data loading ---
    path = _data_path(TRAIN_BETA)
    print(f"data: {path}")

    s_np, a_np, _, xy_np, Nx, Ny = load_darcy_stacked(
        [(TRAIN_BETA, path)], n_samples=args.n_train + args.n_test
    )
    N_total = len(s_np)
    print(f"loaded: N={N_total}, Nx={Nx}, Ny={Ny}")

    s_train = s_np[:args.n_train];  s_test = s_np[args.n_train:]
    a_train = a_np[:args.n_train];  a_test = a_np[args.n_train:]
    del s_np, a_np

    x_np = xy_np[::Ny, 0]  # unique x coords (Nx,): step over Ny entries per row
    y_np = xy_np[:Ny, 1]   # unique y coords (Ny,): first row of grid

    s_traj      = torch.tensor(s_train, dtype=torch.float32).to(DEVICE)
    a_t_train   = torch.tensor(a_train, dtype=torch.float32).to(DEVICE)
    a_t_test    = torch.tensor(a_test,  dtype=torch.float32).to(DEVICE)
    s_test_flat = torch.tensor(s_test,  dtype=torch.float32)

    m = len(range(0, Nx * Ny, args.sensor_stride))

    # --- Phase 1: POD ---
    print("=== Phase 1: POD ===")
    trainer_pod = PODTrainer(PODConfig(max_modes=args.max_modes))
    trainer_pod.train(s_traj, x=None, t=None)

    coeffs_np = trainer_pod.basis.coeffs.cpu().numpy()
    sigmas    = coeffs_np.std(axis=0) * np.sqrt(args.n_train - 1)

    plot_pod_phase1(sigmas, os.path.join(RUN_DIR, "pod_phase1.png"))

    # --- Phase 2: Branch network ---
    print("=== Phase 2: Branch network ===")
    P      = trainer_pod.basis.num_modes
    branch = BranchNet(m=m, P=P, hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(DEVICE)
    model  = PODDeepONet(trainer_pod.basis, branch).to(DEVICE)

    cfg     = PODDeepONetConfig(n_epochs=args.n_epochs, batch_size=args.batch_size,
                                sensor_stride=args.sensor_stride)
    trainer = PODDeepONetTrainer(model, cfg)
    history = trainer.train(a_t_train, val_u0=a_t_test, val_s=s_test_flat)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history, color=plt.cm.tab10(0), lw=1.5, label="Train")
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
                parts.append(trainer.predict(a_in[i:i + batch_size]).cpu())
        return torch.cat(parts, dim=0).numpy()

    idx_sample = torch.randperm(len(s_traj))[:2000]
    err_train  = rel_l2(s_traj[idx_sample].cpu().numpy(), predict_batched(a_t_train[idx_sample]))
    err_test   = rel_l2(s_test, predict_batched(a_t_test))

    print(f"Train | mean={err_train.mean():.4f}  median={np.median(err_train):.4f}  std={err_train.std():.4f}")
    print(f"Test  | mean={err_test.mean():.4f}  median={np.median(err_test):.4f}  std={err_test.std():.4f}  p95={np.percentile(err_test, 95):.4f}")

    metrics = {
        "run_name":     RUN_NAME,
        "n_modes":      int(P),
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
        s_b_all, a_b_all, _, _, _, _ = load_darcy_stacked([(beta, fpath)], n_samples=args.n_train + args.n_test)
        s_b = s_b_all[args.n_train:]; a_b = a_b_all[args.n_train:]
        del s_b_all, a_b_all
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
            "POD-DeepONet - cross-beta generalization",
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
        "POD-DeepONet", os.path.join(RUN_DIR, "pod_reconstruction.png"),
        row_labels=[f"beta={TRAIN_BETA}  sample {i}" for i in idxs],
    )

    plot_error_dist(err_test, "POD-DeepONet Darcy - test errors",
                    os.path.join(RUN_DIR, "pod_err_dist.png"))

    # --- Checkpoint ---
    torch.save({
        "model":    model.state_dict(),
        "cfg":      cfg,
        "metrics":  metrics,
        "run_name": RUN_NAME,
    }, os.path.join(RUN_DIR, "model.pt"))

    metrics["hparams"] = vars(args)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {os.path.abspath(RUN_DIR)}")


if __name__ == "__main__":
    main()
