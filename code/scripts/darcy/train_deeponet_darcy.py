#!/usr/bin/env python
"""Train vanilla DeepONet (Lu 2019) on 2D Darcy Flow — pure PyTorch.

Simple, clean implementation without DeepXDE complications.

Usage:
  Specialist: python train_deeponet_darcy.py --beta_values 1.0
  Joint:      python train_deeponet_darcy.py --beta_values 0.1 1.0 10.0 100.0
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from utils.datasets import load_darcy_stacked, DATA_DIR
from utils.plotting import (
    plot_error_dist, plot_cross_param_bar, plot_reconstruction_xy,
)


def rel_l2(true, pred):
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
_EPOCHS_PER_BETA = {0.01: 100, 0.1: 100, 1.0: 180, 10.0: 1000, 100.0: 5800}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--beta_values", type=float, nargs="+", required=True)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--results_dir", type=str, default=str(_PROJECT_ROOT / "TEMPO_results" / "darcy"))
    p.add_argument("--n_samples", type=int, default=10000)
    p.add_argument("--n_test_per_beta", type=int, default=1000)
    p.add_argument("--data_dir", type=str, default=os.path.expanduser("~/data/2D/DarcyFlow"))
    p.add_argument("--sensor_stride", type=int, default=1)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_epochs", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=39)
    p.add_argument("--n_viz", type=int, default=3)
    return p.parse_args()


def _data_path(beta: float, data_dir: str) -> str:
    filename = f"2D_DarcyFlow_beta{beta}_Train.hdf5"
    local = pathlib.Path(DATA_DIR) / filename
    server = pathlib.Path(data_dir) / filename
    return str(local) if local.exists() else str(server)


def _mlp(in_dim, out_dim, hidden_dim, n_layers, act=nn.Tanh):
    """MLP: [in_dim, hidden_dim, ..., hidden_dim, out_dim]"""
    layers = [nn.Linear(in_dim, hidden_dim), act()]
    for _ in range(n_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), act()])
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class TrunkNet(nn.Module):
    def __init__(self, d_out, hidden_dim, n_layers):
        super().__init__()
        self.net = _mlp(2, d_out, hidden_dim, n_layers, act=nn.Tanh)

    def forward(self, xy):
        return self.net(xy)


class BranchNet(nn.Module):
    def __init__(self, m, d_out, hidden_dim, n_layers):
        super().__init__()
        self.net = _mlp(m + 1, d_out, hidden_dim, n_layers, act=nn.Tanh)

    def forward(self, u0, beta):
        x = torch.cat([u0, beta], dim=-1)
        return self.net(x)


class DeepONet(nn.Module):
    def __init__(self, branch, trunk, d):
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u0, beta, xy):
        """u0: (N, m), beta: (N, 1), xy: (Nxy, 2) -> (N, Nxy)"""
        b = self.branch(u0, beta)  # (N, d)
        t = self.trunk(xy)  # (Nxy, d)
        return torch.einsum("nd,md->nm", b, t) + self.bias


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    joint = len(args.beta_values) > 1
    n_epochs = _EPOCHS_PER_BETA.get(args.beta_values[0], 600) if args.n_epochs == -1 else args.n_epochs

    RUN_NAME = args.run_name or ("deeponet_joint_darcy_v1" if joint else f"deeponet_darcy_beta{args.beta_values[0]}_v1")
    RUN_DIR = os.path.join(args.results_dir, RUN_NAME)
    os.makedirs(RUN_DIR, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={DEVICE}  epochs={n_epochs}  run={RUN_NAME}")

    # ========== DATA ==========
    entries = [(b, _data_path(b, args.data_dir)) for b in args.beta_values if os.path.exists(_data_path(b, args.data_dir))]
    if not entries:
        raise RuntimeError("No data files found")

    s_np, a_np, kappa_np, xy_np, Nx, Ny = load_darcy_stacked(entries, n_samples=args.n_samples)
    Nxy = Nx * Ny

    n_beta = len(entries)
    N_per = args.n_samples
    train_idx = np.concatenate([np.arange(i * N_per, (i + 1) * N_per - args.n_test_per_beta) for i in range(n_beta)])
    test_idx = np.concatenate([np.arange((i + 1) * N_per - args.n_test_per_beta, (i + 1) * N_per) for i in range(n_beta)])

    s_train = torch.from_numpy(s_np[train_idx].astype(np.float32)).to(DEVICE)
    s_test = torch.from_numpy(s_np[test_idx].astype(np.float32)).to(DEVICE)
    a_train = torch.from_numpy(a_np[train_idx].astype(np.float32)).to(DEVICE)
    a_test = torch.from_numpy(a_np[test_idx].astype(np.float32)).to(DEVICE)
    kappa_train = torch.from_numpy(kappa_np[train_idx].astype(np.float32)).reshape(-1, 1).to(DEVICE)
    kappa_test = torch.from_numpy(kappa_np[test_idx].astype(np.float32)).reshape(-1, 1).to(DEVICE)
    xy = torch.from_numpy(xy_np.astype(np.float32)).to(DEVICE)

    N_train, N_test = len(train_idx), len(test_idx)
    effective_stride = max(1, args.sensor_stride)
    m = math.ceil(Nxy / effective_stride)

    print(f"N_train={N_train}  N_test={N_test}  Nx={Nx}  Ny={Ny}  m={m}  Nxy={Nxy}")

    # ========== MODEL ==========
    d = args.hidden_dim
    branch = BranchNet(m, d, args.hidden_dim, args.n_layers).to(DEVICE)
    trunk = TrunkNet(d, args.hidden_dim, args.n_layers).to(DEVICE)
    model = DeepONet(branch, trunk, d).to(DEVICE)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # ========== TRAINING ==========
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(n_epochs):
        model.train()
        idx_b = np.random.choice(N_train, args.batch_size, replace=False)
        a_b = a_train[idx_b, ::effective_stride]
        k_b = kappa_train[idx_b]
        s_b = s_train[idx_b]

        pred = model(a_b, k_b, xy)
        loss = F.mse_loss(pred, s_b)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % args.log_every == 0:
            print(f"  epoch {epoch+1:5d} | loss={loss.item():.4e}")

    print(f"Done: final loss={loss.item():.4e}")

    # ========== EVALUATION ==========
    model.eval()
    with torch.no_grad():
        pred_test = model(a_test[:, ::effective_stride], kappa_test, xy).cpu().numpy()
    err_test = rel_l2(s_test.cpu().numpy(), pred_test)

    idx_sample = np.random.choice(N_train, min(2000, N_train), replace=False)
    with torch.no_grad():
        pred_train = model(a_train[idx_sample, ::effective_stride], kappa_train[idx_sample], xy).cpu().numpy()
    err_train = rel_l2(s_train[idx_sample].cpu().numpy(), pred_train)

    print(f"Train | mean={err_train.mean():.4f}  median={np.median(err_train):.4f}  std={err_train.std():.4f}")
    print(f"Test  | mean={err_test.mean():.4f}  median={np.median(err_test):.4f}  std={err_test.std():.4f}  p95={np.percentile(err_test, 95):.4f}")

    # ========== METRICS ==========
    metrics = {
        "run_name": RUN_NAME,
        "n_params": sum(p.numel() for p in model.parameters()),
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
    else:
        metrics.update({
            "overall_mean": float(err_test.mean()),
            "overall_median": float(np.median(err_test)),
        })

    cross_beta_metrics = {}
    for beta_eval in [0.01, 0.1, 1.0, 10.0, 100.0]:
        fpath = _data_path(beta_eval, args.data_dir)
        if not os.path.exists(fpath):
            continue
        s_b_all, a_b_all, k_b_all, _, _, _ = load_darcy_stacked([(beta_eval, fpath)], n_samples=args.n_samples)
        s_b = torch.from_numpy(s_b_all[args.n_samples - args.n_test_per_beta:].astype(np.float32)).to(DEVICE)
        a_b = torch.from_numpy(a_b_all[args.n_samples - args.n_test_per_beta:, ::effective_stride].astype(np.float32)).to(DEVICE)
        k_b = torch.full((len(a_b), 1), float(beta_eval), dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            pred_b = model(a_b, k_b, xy).cpu().numpy()
        err_b = rel_l2(s_b.cpu().numpy(), pred_b)
        cross_beta_metrics[beta_eval] = {
            "mean": float(err_b.mean()),
            "median": float(np.median(err_b)),
            "std": float(err_b.std()),
            "p95": float(np.percentile(err_b, 95)),
        }
        print(f"  beta={beta_eval}: mean={err_b.mean():.4f}")

    metrics["cross_beta"] = cross_beta_metrics
    if not joint:
        plot_cross_param_bar(cross_beta_metrics, args.beta_values[0], "beta", "DeepONet - cross-beta", os.path.join(RUN_DIR, "cross_beta.png"))

    # ========== VISUALIZATION ==========
    plot_error_dist(err_test, "DeepONet", os.path.join(RUN_DIR, "err_dist.png"))

    if N_test >= args.n_viz:
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(N_test, size=args.n_viz, replace=False)
        with torch.no_grad():
            preds_viz = model(a_test[idxs, ::effective_stride], kappa_test[idxs], xy).cpu().numpy()
        s_test_np = s_test.cpu().numpy()
        true_list = [s_test_np[i].reshape(Nx, Ny) for i in idxs]
        pred_list = [preds_viz[k].reshape(Nx, Ny) for k in range(args.n_viz)]
        rl2_list = [float(np.linalg.norm(s_test_np[i] - preds_viz[k]) / np.linalg.norm(s_test_np[i])) for k, i in enumerate(idxs)]
        x_np_1d = xy_np[::Ny, 0]
        y_np_1d = xy_np[:Ny, 1]
        plot_reconstruction_xy(true_list, pred_list, rl2_list, x_np_1d, y_np_1d, "DeepONet", os.path.join(RUN_DIR, "reconstruction.png"))

    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {RUN_DIR}")


if __name__ == "__main__":
    main()
