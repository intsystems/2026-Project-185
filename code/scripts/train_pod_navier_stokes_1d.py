#!/usr/bin/env python
"""Train POD-DeepONet on 1D Navier-Stokes — specialist or joint across Reynolds.

Usage:
  Specialist: python train_pod_navier_stokes_1d.py --re_values 100
  Joint:      python train_pod_navier_stokes_1d.py --re_values 100 1000 3600 10000
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from models.pod import PODTrainer, PODConfig
from models.pod_deeponet import BranchNet, PODDeepONet
from utils.datasets import load_ns_1d_stacked
from utils.plotting import plot_error_dist


def rel_l2(true, pred):
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

_EPOCHS_PER_RE = {100: 200, 1000: 200, 3600: 200, 10000: 200}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--re_values", type=int, nargs="+", required=True,
                   help="One value = specialist; multiple = joint")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--results_dir", type=str, default=str(_PROJECT_ROOT / "TEMPO_results" / "navier_stokes_1d"))
    p.add_argument("--n_samples", type=int, default=4000,
                   help="Samples per Re loaded (train + test)")
    p.add_argument("--n_test_per_re", type=int, default=1000)
    p.add_argument("--data_dir", type=str, default=os.path.expanduser("~/data/1D/Navier_Stokes"))
    p.add_argument("--max_modes", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--sensor_stride", type=int, default=1)
    p.add_argument("--n_epochs", type=int, default=-1,
                   help="Branch epochs. -1 = auto per Re in specialist mode; 600 for joint.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=39)
    p.add_argument("--n_viz", type=int, default=3)
    return p.parse_args()


def _data_path(re: int, data_dir: str) -> str:
    filename = f"1D_NavierStokes_Re{re:05d}.npz"
    return os.path.join(data_dir, filename)


def main():
    args = parse_args()

    joint = len(args.re_values) > 1

    # Resolve epoch count
    if args.n_epochs == -1:
        if not joint and args.re_values[0] in _EPOCHS_PER_RE:
            n_epochs = _EPOCHS_PER_RE[args.re_values[0]]
        else:
            n_epochs = 600
    else:
        n_epochs = args.n_epochs
    print(f"n_epochs={n_epochs}  joint={joint}")

    if joint:
        RUN_NAME = args.run_name or "pod_deeponet_joint_navier_stokes_1d_v1"
    else:
        RUN_NAME = args.run_name or f"pod_deeponet_navier_stokes_1d_re{args.re_values[0]}_v1"
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
    for re in args.re_values:
        fpath = _data_path(re, args.data_dir)
        if not os.path.exists(fpath):
            print(f"  Re={re}: file not found, skipping")
            continue
        entries.append((re, fpath))

    if not entries:
        raise RuntimeError("No data files found. Check --data_dir.")

    re_loaded = [e[0] for e in entries]
    s_np, u0_np, kappa_np, x_np, Nx, Nt = load_ns_1d_stacked(entries, n_samples=args.n_samples)

    n_re = len(re_loaded)
    N_per = args.n_samples
    train_idx = np.concatenate([
        np.arange(i * N_per, (i + 1) * N_per - args.n_test_per_re)
        for i in range(n_re)
    ])
    test_idx = np.concatenate([
        np.arange((i + 1) * N_per - args.n_test_per_re, (i + 1) * N_per)
        for i in range(n_re)
    ])

    s = torch.from_numpy(s_np); del s_np
    u0 = torch.from_numpy(u0_np); del u0_np
    kappa = torch.from_numpy(kappa_np[:, None])  # (N, 1)

    s_train = s[train_idx]; s_test = s[test_idx]
    u0_train = u0[train_idx]; u0_test = u0[test_idx]
    kappa_train = kappa[train_idx]; kappa_test = kappa[test_idx]

    s_train_dev = s_train.to(DEVICE)
    u0_train_dev = u0_train.to(DEVICE)
    u0_test_dev = u0_test.to(DEVICE)
    kappa_train_d = kappa_train.to(DEVICE)
    kappa_test_d = kappa_test.to(DEVICE)

    N_train = len(train_idx)
    N_test = len(test_idx)
    m = math.ceil(Nx / args.sensor_stride)
    print(f"N_train={N_train}  N_test={N_test}  Nx={Nx}  Nt={Nt}  m={m}")

    # --- Phase 1: POD on training data ---
    print("=== Phase 1: POD ===")
    trainer_pod = PODTrainer(PODConfig(max_modes=args.max_modes))
    trainer_pod.train(s_train_dev, x=None, t=None)
    P = trainer_pod.basis.num_modes
    print(f"P={P} modes")

    # --- Phase 2: Branch network ---
    print(f"=== Phase 2: Branch network (d_kappa=1, {'joint' if joint else 'specialist'}) ===")
    mean_dev = trainer_pod.basis.mean.to(DEVICE)
    modes_dev = trainer_pod.basis.modes.to(DEVICE)

    targets = trainer_pod.basis.coeffs.to(DEVICE)  # (N_train, P)
    val_targets = (s_test.to(DEVICE) - mean_dev.unsqueeze(0)) @ modes_dev  # (N_test, P)

    u0_sensors = u0_train_dev[:, ::args.sensor_stride]  # (N_train, m)
    u0_val_sensors = u0_test_dev[:, ::args.sensor_stride]  # (N_test, m)
    print(f"Training | N={N_train}, m={m}, P={P}")

    branch = BranchNet(m=m, P=P, hidden_dim=args.hidden_dim,
                       n_layers=args.n_layers, d_kappa=1).to(DEVICE)
    model = PODDeepONet(trainer_pod.basis, branch).to(DEVICE)

    dl = DataLoader(TensorDataset(u0_sensors, kappa_train_d, targets),
                    batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(branch.parameters(), lr=args.lr, weight_decay=1e-4)
    VAL_EVERY = 50

    history, history_val = [], []
    for epoch in range(n_epochs):
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

    # --- Evaluation helpers ---
    def predict_batch(u0_in, kappa_in, batch_size=256):
        branch.eval()
        parts = []
        for i in range(0, len(u0_in), batch_size):
            with torch.no_grad():
                u0_b = u0_in[i:i + batch_size, ::args.sensor_stride]
                k_b = kappa_in[i:i + batch_size]
                beta = branch(u0_b, k_b)
                pred = mean_dev + beta @ modes_dev.T
                parts.append(pred.cpu())
        return torch.cat(parts, dim=0).numpy()

    s_test_np = s_test.numpy()
    pred_test = predict_batch(u0_test_dev, kappa_test_d)
    err_test = rel_l2(s_test_np, pred_test)

    # Sample-based train error
    idx_sample = torch.randperm(N_train)[:2000]
    err_train = rel_l2(
        s_train[idx_sample].numpy(),
        predict_batch(u0_train_dev[idx_sample], kappa_train_d[idx_sample])
    )
    print(f"Train | mean={err_train.mean():.4f}  median={np.median(err_train):.4f}  std={err_train.std():.4f}")
    print(f"Test  | mean={err_test.mean():.4f}  median={np.median(err_test):.4f}  std={err_test.std():.4f}  p95={np.percentile(err_test, 95):.4f}")

    metrics = {
        "run_name": RUN_NAME,
        "n_modes": int(P),
        "n_train": N_train,
        "n_test": N_test,
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

        # Cross-Re evaluation
        cross_re_metrics = {}
        for re_eval in [100, 1000, 3600, 10000]:
            fpath = _data_path(re_eval, args.data_dir)
            if not os.path.exists(fpath):
                continue
            s_re_all, u0_re_all, _, _, _, _ = load_ns_1d_stacked(
                [(re_eval, fpath)], n_samples=args.n_samples
            )
            s_re = s_re_all[args.n_samples - args.n_test_per_re:]
            u0_re = u0_re_all[args.n_samples - args.n_test_per_re:]
            del s_re_all, u0_re_all
            u0_re_dev = torch.from_numpy(u0_re).to(DEVICE)
            kappa_re = torch.full((len(u0_re), 1), float(re_eval),
                                   dtype=torch.float32, device=DEVICE)
            err_re = rel_l2(s_re, predict_batch(u0_re_dev, kappa_re))
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

    # --- Visualizations ---
    # Error distribution
    plot_error_dist(err_test, "POD-DeepONet 1D Navier-Stokes - test errors",
                    os.path.join(RUN_DIR, "error_dist.png"))

    # Reconstruction: visualize a few test samples
    if N_test >= args.n_viz:
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(N_test, size=args.n_viz, replace=False)

        fig, axes = plt.subplots(args.n_viz, 2, figsize=(14, 3*args.n_viz))
        if args.n_viz == 1:
            axes = axes[np.newaxis, :]

        for row, idx in enumerate(idxs):
            s_true = s_test_np[idx].reshape(Nt, Nx)
            s_pred = pred_test[idx].reshape(Nt, Nx)
            err = np.linalg.norm(s_true - s_pred) / np.linalg.norm(s_true)

            # True trajectory
            im0 = axes[row, 0].imshow(s_true, aspect='auto', cmap='RdBu_r')
            axes[row, 0].set_title(f"True (sample {idx})")
            axes[row, 0].set_xlabel("x"); axes[row, 0].set_ylabel("t")
            plt.colorbar(im0, ax=axes[row, 0])

            # Predicted trajectory
            im1 = axes[row, 1].imshow(s_pred, aspect='auto', cmap='RdBu_r')
            axes[row, 1].set_title(f"Predicted (err={err:.4f})")
            axes[row, 1].set_xlabel("x"); axes[row, 1].set_ylabel("t")
            plt.colorbar(im1, ax=axes[row, 1])

        plt.suptitle("POD-DeepONet 1D NS - Trajectory Reconstruction", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, "reconstruction.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # --- Checkpoint ---
    torch.save({
        "model": model.state_dict(),
        "metrics": metrics,
        "run_name": RUN_NAME,
    }, os.path.join(RUN_DIR, "model.pt"))

    metrics["hparams"] = vars(args)
    metrics["hparams"]["n_epochs_used"] = n_epochs
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {os.path.abspath(RUN_DIR)}")


if __name__ == "__main__":
    main()
