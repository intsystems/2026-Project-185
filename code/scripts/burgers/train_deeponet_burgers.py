#!/usr/bin/env python
"""Train vanilla DeepONet (Lu 2019) on 1D Burgers Flow - pure PyTorch.

Simple, clean implementation without DeepXDE complications.

Usage:
  Specialist: python train_deeponet_burgers.py --nu_values 0.001
  Joint:      python train_deeponet_burgers.py --nu_values 0.001 0.01 0.1 1.0
"""
import argparse
import h5py
import json
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
from utils.datasets import load_stacked, measure_inference_time


def rel_l2(true, pred):
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
_EPOCHS_PER_NU = {0.001: 100, 0.01: 100, 0.1: 100, 1.0: 100}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nu_values", type=float, nargs="+", required=True)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--results_dir", type=str, default=str(_PROJECT_ROOT / "TEMPO_results" / "burgers"))
    p.add_argument("--n_samples", type=int, default=9500)
    p.add_argument("--n_test_per_nu", type=int, default=1000)
    p.add_argument("--data_dir", type=str, default=os.path.expanduser("~/data/1D/Burgers/Train"))
    p.add_argument("--sensor_stride", type=int, default=1)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_epochs", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=39)
    p.add_argument("--n_viz", type=int, default=3)
    return p.parse_args()


def _data_path(nu: float, data_dir: str) -> str:
    filename = f"1D_Burgers_Sols_Nu{nu}.hdf5"
    local = pathlib.Path(_PROJECT_ROOT) / "data" / f"Burgers_Nu{nu}.hdf5"
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

    def forward(self, xt):
        return self.net(xt)


class BranchNet(nn.Module):
    def __init__(self, m, d_out, hidden_dim, n_layers):
        super().__init__()
        self.net = _mlp(m + 1, d_out, hidden_dim, n_layers, act=nn.Tanh)

    def forward(self, u0, nu):
        x = torch.cat([u0, nu], dim=-1)
        return self.net(x)


class DeepONet(nn.Module):
    def __init__(self, branch, trunk, d):
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u0, nu, xt):
        """u0: (N, m), nu: (N, 1), xt: (Nxt, 2) -> (N, Nxt)"""
        b = self.branch(u0, nu)  # (N, d)
        t = self.trunk(xt)  # (Nxt, d)
        return torch.einsum("nd,md->nm", b, t) + self.bias


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    joint = len(args.nu_values) > 1
    n_epochs = _EPOCHS_PER_NU.get(args.nu_values[0], 100) if args.n_epochs == -1 else args.n_epochs

    RUN_NAME = args.run_name or ("deeponet_joint_burgers_v1" if joint else f"deeponet_burgers_nu{args.nu_values[0]}_v1")
    RUN_DIR = os.path.join(args.results_dir, RUN_NAME)
    os.makedirs(RUN_DIR, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={DEVICE}  epochs={n_epochs}  run={RUN_NAME}")

    entries = [(nu, _data_path(nu, args.data_dir)) for nu in args.nu_values if os.path.exists(_data_path(nu, args.data_dir))]
    if not entries:
        raise RuntimeError("No data files found")

    s_np, kappa_np, x_np, t_np, Nx, Nt = load_stacked(entries, n_samples=args.n_samples)
    Nxt = Nx * Nt

    n_nu = len(entries)
    N_per = args.n_samples
    train_idx = np.concatenate([np.arange(i * N_per, (i + 1) * N_per - args.n_test_per_nu) for i in range(n_nu)])
    test_idx = np.concatenate([np.arange((i + 1) * N_per - args.n_test_per_nu, (i + 1) * N_per) for i in range(n_nu)])

    s_train = torch.from_numpy(s_np[train_idx].astype(np.float32)).to(DEVICE)
    s_test = torch.from_numpy(s_np[test_idx].astype(np.float32)).to(DEVICE)
    u0_all = torch.from_numpy(s_np[:, :Nx].astype(np.float32))
    u0_train = u0_all[train_idx].to(DEVICE)
    u0_test = u0_all[test_idx].to(DEVICE)
    kappa_train = torch.from_numpy(kappa_np[train_idx].astype(np.float32)).reshape(-1, 1).to(DEVICE)
    kappa_test = torch.from_numpy(kappa_np[test_idx].astype(np.float32)).reshape(-1, 1).to(DEVICE)

    # Create (x, t) coordinate grid
    x_1d = torch.from_numpy(x_np.astype(np.float32))
    t_1d = torch.from_numpy(t_np.astype(np.float32))
    x_grid, t_grid = torch.meshgrid(x_1d, t_1d, indexing='ij')
    xt = torch.stack([x_grid.flatten(), t_grid.flatten()], dim=-1).to(DEVICE)

    N_train, N_test = len(train_idx), len(test_idx)
    effective_stride = max(1, args.sensor_stride)
    m = len(range(0, Nx, effective_stride))

    print(f"N_train={N_train}  N_test={N_test}  Nx={Nx}  Nt={Nt}  m={m}  Nxt={Nxt}")

    d = args.hidden_dim
    branch = BranchNet(m, d, args.hidden_dim, args.n_layers).to(DEVICE)
    trunk = TrunkNet(d, args.hidden_dim, args.n_layers).to(DEVICE)
    model = DeepONet(branch, trunk, d).to(DEVICE)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(n_epochs):
        model.train()
        idx_b = np.random.choice(N_train, args.batch_size, replace=False)
        u0_b = u0_train[idx_b, ::effective_stride]
        k_b = kappa_train[idx_b]
        s_b = s_train[idx_b]

        pred = model(u0_b, k_b, xt)
        loss = F.mse_loss(pred, s_b)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % args.log_every == 0:
            print(f"  epoch {epoch+1:5d} | loss={loss.item():.4e}")

    print(f"Done: final loss={loss.item():.4e}")

    model.eval()
    with torch.no_grad():
        pred_test = model(u0_test[:, ::effective_stride], kappa_test, xt).cpu().numpy()
    err_test = rel_l2(s_test.cpu().numpy(), pred_test)

    idx_sample = np.random.choice(N_train, min(2000, N_train), replace=False)
    with torch.no_grad():
        pred_train = model(u0_train[idx_sample, ::effective_stride], kappa_train[idx_sample], xt).cpu().numpy()
    err_train = rel_l2(s_train[idx_sample].cpu().numpy(), pred_train)

    print(f"Train | mean={err_train.mean():.4f}  median={np.median(err_train):.4f}  std={err_train.std():.4f}")
    print(f"Test  | mean={err_test.mean():.4f}  median={np.median(err_test):.4f}  std={err_test.std():.4f}  p95={np.percentile(err_test, 95):.4f}")

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

    cross_nu_metrics = {}
    for nu_eval in [0.001, 0.01, 0.1, 1.0]:
        fpath = _data_path(nu_eval, args.data_dir)
        if not os.path.exists(fpath):
            continue
        s_nu_all, kappa_nu_all, _, _, _, _ = load_stacked([(nu_eval, fpath)], n_samples=args.n_samples)
        s_nu = torch.from_numpy(s_nu_all[args.n_samples - args.n_test_per_nu:].astype(np.float32)).to(DEVICE)
        u0_nu = torch.from_numpy(s_nu_all[args.n_samples - args.n_test_per_nu:, :Nx].astype(np.float32)).to(DEVICE)
        kappa_nu = torch.full((len(u0_nu), 1), float(nu_eval), dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            pred_nu = model(u0_nu[:, ::effective_stride], kappa_nu, xt).cpu().numpy()
        err_nu = rel_l2(s_nu.cpu().numpy(), pred_nu)
        cross_nu_metrics[nu_eval] = {
            "mean": float(err_nu.mean()),
            "median": float(np.median(err_nu)),
            "std": float(err_nu.std()),
            "p95": float(np.percentile(err_nu, 95)),
        }
        print(f"  nu={nu_eval}: mean={err_nu.mean():.4f}")

    metrics["cross_nu"] = cross_nu_metrics

    # Inference time
    model.eval()
    _inf_ms = measure_inference_time(
        lambda: model(u0_test[:, ::effective_stride], kappa_test, xt),
        device=DEVICE
    )
    metrics["inference_ms_total"] = _inf_ms
    metrics["inference_ms_per_sample"] = _inf_ms / N_test

    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {RUN_DIR}")


if __name__ == "__main__":
    main()
