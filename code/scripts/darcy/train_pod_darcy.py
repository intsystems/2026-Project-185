#!/usr/bin/env python
"""Train POD-DeepONet on 2D Darcy Flow — specialist (single beta) or joint (multiple beta).

Usage:
  Specialist: python train_pod_darcy.py --beta_values 1.0
  Joint:      python train_pod_darcy.py --beta_values 0.1 1.0 10.0 100.0
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
from models.pod import PODTrainer, PODConfig
from models.pod_deeponet import BranchNet, PODDeepONet
from utils.datasets import load_darcy_stacked, DATA_DIR, measure_inference_time
from utils.plotting import (
    plot_pod_phase1, plot_error_dist, plot_cross_param_bar, plot_reconstruction_xy,
)


def rel_l2(true, pred):
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


_SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]

# Default branch training epochs per beta (used in specialist mode with --n_epochs -1)
_EPOCHS_PER_BETA = {0.01: 100, 0.1: 100, 1.0: 180, 10.0: 1000, 100.0: 5800}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--beta_values",    type=float, nargs="+", required=True,
                   help="One value = specialist; multiple = joint")
    p.add_argument("--run_name",       type=str,   default=None)
    p.add_argument("--results_dir",    type=str,   default=str(_PROJECT_ROOT / "TEMPO_results" / "darcy"))
    p.add_argument("--n_samples",      type=int,   default=10000,
                   help="Samples per beta loaded (train + test)")
    p.add_argument("--n_test_per_beta", type=int,  default=1000)
    p.add_argument("--data_dir",       type=str,   default=os.path.expanduser("~/data/2D/DarcyFlow"))
    p.add_argument("--max_modes",      type=int,   default=32)
    p.add_argument("--hidden_dim",     type=int,   default=256)
    p.add_argument("--n_layers",       type=int,   default=4)
    p.add_argument("--sensor_stride",  type=int,   default=1)
    p.add_argument("--n_epochs",       type=int,   default=-1,
                   help="Branch epochs. -1 = auto per beta in specialist mode; 600 for joint.")
    p.add_argument("--batch_size",     type=int,   default=1024)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--log_every",      type=int,   default=500)
    p.add_argument("--seed",           type=int,   default=39)
    p.add_argument("--n_viz",          type=int,   default=3)
    return p.parse_args()


def _data_path(beta: float, data_dir: str) -> str:
    filename = f"2D_DarcyFlow_beta{beta}_Train.hdf5"
    local  = pathlib.Path(DATA_DIR) / filename
    server = pathlib.Path(data_dir) / filename
    return str(local) if local.exists() else str(server)


def main():
    args = parse_args()

    joint = len(args.beta_values) > 1

    # Resolve epoch count
    if args.n_epochs == -1:
        if not joint and args.beta_values[0] in _EPOCHS_PER_BETA:
            n_epochs = _EPOCHS_PER_BETA[args.beta_values[0]]
        else:
            n_epochs = 600
    else:
        n_epochs = args.n_epochs
    print(f"n_epochs={n_epochs}  joint={joint}")

    if joint:
        RUN_NAME = args.run_name or "pod_deeponet_joint_darcy_v1"
    else:
        RUN_NAME = args.run_name or f"pod_deeponet_darcy_beta{args.beta_values[0]}_v1"
    RUN_DIR = os.path.join(args.results_dir, RUN_NAME)
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

    x_np_1d = xy_np[::Ny, 0]   # unique x coords (Nx,)
    y_np_1d = xy_np[:Ny, 1]    # unique y coords (Ny,)

    n_beta = len(beta_loaded)
    N_per  = args.n_samples
    train_idx = np.concatenate([
        np.arange(i * N_per, (i + 1) * N_per - args.n_test_per_beta)
        for i in range(n_beta)
    ])
    test_idx = np.concatenate([
        np.arange((i + 1) * N_per - args.n_test_per_beta, (i + 1) * N_per)
        for i in range(n_beta)
    ])

    s     = torch.from_numpy(s_np);              del s_np
    a     = torch.from_numpy(a_np);              del a_np
    kappa = torch.from_numpy(kappa_np[:, None])  # (N, 1)

    s_train     = s[train_idx];      s_test     = s[test_idx]
    a_train     = a[train_idx];      a_test     = a[test_idx]
    kappa_train = kappa[train_idx];  kappa_test = kappa[test_idx]

    s_train_dev   = s_train.to(DEVICE)
    a_train_dev   = a_train.to(DEVICE)
    a_test_dev    = a_test.to(DEVICE)
    kappa_train_d = kappa_train.to(DEVICE)
    kappa_test_d  = kappa_test.to(DEVICE)

    N_train = len(train_idx)
    N_test  = len(test_idx)
    m       = math.ceil(Nxy / args.sensor_stride)
    print(f"N_train={N_train}  N_test={N_test}  Nx={Nx}  Ny={Ny}  m={m}")

    # --- Phase 1: POD on training data ---
    print("=== Phase 1: POD ===")
    trainer_pod = PODTrainer(PODConfig(max_modes=args.max_modes))
    trainer_pod.train(s_train_dev, x=None, t=None)
    P = trainer_pod.basis.num_modes
    print(f"P={P} modes")

    sigmas = trainer_pod.basis.coeffs.cpu().numpy().std(axis=0) * np.sqrt(N_train - 1)
    plot_pod_phase1(sigmas, os.path.join(RUN_DIR, "pod_phase1.png"))

    # --- Phase 2: Branch network (always d_kappa=1) ---
    print(f"=== Phase 2: Branch network (d_kappa=1, {'joint' if joint else 'specialist'}) ===")
    mean_dev  = trainer_pod.basis.mean.to(DEVICE)
    modes_dev = trainer_pod.basis.modes.to(DEVICE)

    a_sensors     = a_train_dev[:, ::args.sensor_stride]   # (N_train, m)
    a_val_sensors = a_test_dev[:, ::args.sensor_stride]    # (N_test,  m)
    targets    = trainer_pod.basis.coeffs.to(DEVICE)        # (N_train, P)
    val_targets = (s_test.to(DEVICE) - mean_dev.unsqueeze(0)) @ modes_dev  # (N_test, P)

    branch = BranchNet(m=m, P=P, hidden_dim=args.hidden_dim,
                       n_layers=args.n_layers, d_kappa=1).to(DEVICE)
    model  = PODDeepONet(trainer_pod.basis, branch).to(DEVICE)

    dl  = DataLoader(TensorDataset(a_sensors, kappa_train_d, targets),
                     batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(branch.parameters(), lr=args.lr, weight_decay=1e-4)
    VAL_EVERY = 50

    history, history_val = [], []
    for epoch in range(n_epochs):
        branch.train()
        total = 0.0
        for a_b, kappa_b, coeff_b in dl:
            opt.zero_grad()
            loss = F.mse_loss(branch(a_b, kappa_b), coeff_b)
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(dl)
        history.append(avg)
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

    # --- Evaluation helpers ---
    def predict_batch(a_in, kappa_in, batch_size=256):
        branch.eval()
        parts = []
        for i in range(0, len(a_in), batch_size):
            with torch.no_grad():
                a_s   = a_in[i:i + batch_size, ::args.sensor_stride]
                k_s   = kappa_in[i:i + batch_size]
                beta  = branch(a_s, k_s)
                pred  = mean_dev + beta @ modes_dev.T
                parts.append(pred.cpu())
        return torch.cat(parts, dim=0).numpy()

    s_test_np = s_test.numpy()
    pred_test = predict_batch(a_test_dev, kappa_test_d)
    err_test  = rel_l2(s_test_np, pred_test)

    # Sample-based train error
    idx_sample = torch.randperm(N_train)[:2000]
    err_train  = rel_l2(
        s_train[idx_sample].numpy(),
        predict_batch(a_train_dev[idx_sample], kappa_train_d[idx_sample])
    )
    print(f"Train | mean={err_train.mean():.4f}  median={np.median(err_train):.4f}  std={err_train.std():.4f}")
    print(f"Test  | mean={err_test.mean():.4f}  median={np.median(err_test):.4f}  std={err_test.std():.4f}  p95={np.percentile(err_test, 95):.4f}")

    metrics = {
        "run_name": RUN_NAME,
        "n_modes":  int(P),
        "n_train":  N_train,
        "n_test":   N_test,
    }

    if not joint:
        # Specialist metrics + cross-beta eval
        metrics.update({
            "train_mean":   float(err_train.mean()),
            "train_median": float(np.median(err_train)),
            "train_std":    float(err_train.std()),
            "test_mean":    float(err_test.mean()),
            "test_median":  float(np.median(err_test)),
            "test_std":     float(err_test.std()),
            "test_p95":     float(np.percentile(err_test, 95)),
        })

        cross_beta_metrics = {}
        for beta_eval in [0.01, 0.1, 1.0, 10.0, 100.0]:
            fpath = _data_path(beta_eval, args.data_dir)
            if not os.path.exists(fpath):
                print(f"  beta={beta_eval}: file not found, skipping")
                continue
            s_b_all, a_b_all, _, _, _, _ = load_darcy_stacked(
                [(beta_eval, fpath)], n_samples=args.n_samples
            )
            s_b = s_b_all[args.n_samples - args.n_test_per_beta:]
            a_b = a_b_all[args.n_samples - args.n_test_per_beta:]
            del s_b_all, a_b_all
            a_b_dev  = torch.from_numpy(a_b).to(DEVICE)
            kappa_b  = torch.full((len(a_b), 1), float(beta_eval),
                                  dtype=torch.float32, device=DEVICE)
            err_b    = rel_l2(s_b, predict_batch(a_b_dev, kappa_b))
            cross_beta_metrics[beta_eval] = {
                "mean":   float(err_b.mean()),
                "median": float(np.median(err_b)),
                "std":    float(err_b.std()),
                "p95":    float(np.percentile(err_b, 95)),
            }
            tag = " (trained)" if beta_eval == args.beta_values[0] else ""
            print(f"  beta={beta_eval}{tag}: mean={err_b.mean():.4f}  median={np.median(err_b):.4f}")

        metrics["cross_beta"] = cross_beta_metrics

        if cross_beta_metrics:
            plot_cross_param_bar(
                cross_beta_metrics, args.beta_values[0], "beta",
                "POD-DeepONet - cross-beta generalization",
                os.path.join(RUN_DIR, "cross_beta.png"),
            )

        # Specialist reconstruction
        rng  = np.random.default_rng(args.seed)
        idxs = rng.choice(N_test, size=args.n_viz, replace=False)
        preds_viz = predict_batch(a_test_dev[idxs], kappa_test_d[idxs])

        true_list  = [s_test_np[i].reshape(Nx, Ny) for i in idxs]
        pred_list  = [preds_viz[k].reshape(Nx, Ny) for k in range(args.n_viz)]
        rl2_list   = [
            float(np.linalg.norm(s_test_np[i] - preds_viz[k]) / np.linalg.norm(s_test_np[i]))
            for k, i in enumerate(idxs)
        ]
        row_labels = [f"beta={args.beta_values[0]}  sample {i}" for i in idxs]
        plot_reconstruction_xy(
            true_list, pred_list, rl2_list, x_np_1d, y_np_1d,
            "POD-DeepONet", os.path.join(RUN_DIR, "pod_reconstruction.png"),
            row_labels=row_labels,
        )

    else:
        # Joint metrics: per-beta breakdown
        metrics.update({
            "overall_mean":   float(err_test.mean()),
            "overall_median": float(np.median(err_test)),
            "overall_std":    float(err_test.std()),
        })
        kappa_test_np = kappa_test[:, 0].numpy()
        beta_unique   = np.unique(kappa_test_np)
        cross_beta_metrics = {}
        print("Mean rel L2 error per beta (test):")
        for beta in beta_unique:
            mask    = kappa_test_np == beta
            m_err   = float(err_test[mask].mean())
            med_err = float(np.median(err_test[mask]))
            metrics[f"beta{beta:.4g}_mean"]   = m_err
            metrics[f"beta{beta:.4g}_median"] = med_err
            metrics[f"beta{beta:.4g}_std"]    = float(err_test[mask].std())
            cross_beta_metrics[beta] = {
                "mean":   m_err,
                "median": med_err,
                "std":    float(err_test[mask].std()),
                "p95":    float(np.percentile(err_test[mask], 95)),
            }
            print(f"  beta={beta}: mean={m_err:.4f}  median={med_err:.4f}")

        plot_cross_param_bar(
            cross_beta_metrics, None, r"$\beta$",
            "POD-DeepONet (joint) Darcy",
            os.path.join(RUN_DIR, "cross_beta.png"),
        )

        # Joint reconstruction: n_viz samples per beta
        rng = np.random.default_rng(args.seed)
        true_list, pred_list, rl2_list, row_labels = [], [], [], []
        for beta in beta_unique:
            idxs = np.where(kappa_test_np == beta)[0]
            chosen = rng.choice(idxs, size=min(args.n_viz, len(idxs)), replace=False)
            for idx in chosen:
                t = s_test_np[idx].reshape(Nx, Ny)
                p = pred_test[idx].reshape(Nx, Ny)
                true_list.append(t); pred_list.append(p)
                rl2_list.append(float(np.linalg.norm(t - p) / np.linalg.norm(t)))
                row_labels.append(f"$\\beta$={beta:.4g}")
        plot_reconstruction_xy(
            true_list, pred_list, rl2_list, x_np_1d, y_np_1d,
            "POD-DeepONet (joint)", os.path.join(RUN_DIR, "pod_reconstruction.png"),
            row_labels=row_labels,
        )

    plot_error_dist(err_test, "POD-DeepONet Darcy - test errors",
                    os.path.join(RUN_DIR, "pod_err_dist.png"))

    # --- Checkpoint ---
    torch.save({
        "model":    model.state_dict(),
        "metrics":  metrics,
        "run_name": RUN_NAME,
    }, os.path.join(RUN_DIR, "model.pt"))

    # Inference time
    _inf_ms = measure_inference_time(
        lambda: predict_batch(a_test_dev, kappa_test_d),
        device=DEVICE
    )
    metrics["inference_ms_total"] = _inf_ms
    metrics["inference_ms_per_sample"] = _inf_ms / N_test

    metrics["hparams"] = vars(args)
    metrics["hparams"]["n_epochs_used"] = n_epochs
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {os.path.abspath(RUN_DIR)}")


if __name__ == "__main__":
    main()
