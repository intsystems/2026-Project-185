#!/usr/bin/env python
"""Train vanilla DeepONet on 2D incompressible Navier-Stokes across Reynolds numbers.

Input: initial velocity field u0(x,y) [flattened, shape: Nx*Ny*2]
Parameter: Reynolds number Re
Output: full trajectory u(t,x,y) [flattened, shape: Nt*Nx*Ny*2]

Usage:
  Specialist: python train_deeponet_navier_stokes.py --re_values 100
  Joint:      python train_deeponet_navier_stokes.py --re_values 100 1000 3600 10000
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from utils.datasets import load_ns_stacked


def rel_l2(true, pred):
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_EPOCHS_PER_RE = {100: 500, 1000: 500, 3600: 500, 10000: 500}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--re_values", type=int, nargs="+", required=True)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--results_dir", type=str, default=str(_PROJECT_ROOT / "TEMPO_results" / "navier_stokes"))
    p.add_argument("--n_samples", type=int, default=10000)
    p.add_argument("--n_test_per_re", type=int, default=2000)
    p.add_argument("--data_dir", type=str, default=os.path.expanduser("~/data/2D/Navier_Stokes"))
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_epochs", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=39)
    p.add_argument("--n_viz", type=int, default=2)
    return p.parse_args()


def _data_path(re: int, data_dir: str) -> str:
    filename = f"2D_NavierStokes_Incomp_Re{re:05d}.npz"
    return os.path.join(data_dir, filename)


def _mlp(in_dim, out_dim, hidden_dim, n_layers, act=nn.Tanh):
    """MLP: [in_dim, hidden_dim, ..., hidden_dim, out_dim]"""
    layers = [nn.Linear(in_dim, hidden_dim), act()]
    for _ in range(n_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), act()])
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class TrunkNet(nn.Module):
    """Trunk network: (x, y, t) -> 2*d-dimensional encoding (for u and v components)"""
    def __init__(self, d_out, hidden_dim, n_layers):
        super().__init__()
        self.net = _mlp(3, 2 * d_out, hidden_dim, n_layers, act=nn.Tanh)

    def forward(self, xyt):
        return self.net(xyt)


class BranchNet(nn.Module):
    """Branch network: (u0_x, u0_y, Re) -> d-dimensional encoding"""
    def __init__(self, m, d_out, hidden_dim, n_layers):
        super().__init__()
        self.net = _mlp(m + 1, d_out, hidden_dim, n_layers, act=nn.Tanh)

    def forward(self, u0, re):
        """u0: (N, m*2) [flattened velocity], re: (N, 1)"""
        x = torch.cat([u0, re], dim=-1)
        return self.net(x)


class DeepONet(nn.Module):
    def __init__(self, branch, trunk, d):
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u0, re, xyt):
        """
        u0: (N, m*2) - flattened initial velocity
        re: (N, 1) - Reynolds number
        xyt: (Nxyt, 3) - spatial-temporal coordinates
        Output: (N, Nxyt*2) - velocity field trajectory (u and v components)
        """
        b = self.branch(u0, re)  # (N, d)
        t = self.trunk(xyt)  # (Nxyt, 2*d)
        # Reshape trunk to (Nxyt, 2, d)
        Nxyt = t.shape[0]
        d = b.shape[1]
        t = t.reshape(Nxyt, 2, d)
        # einsum: (N, d) x (Nxyt, 2, d) -> (N, Nxyt, 2)
        out = torch.einsum("nd,mcd->nmc", b, t) + self.bias
        # Flatten to (N, Nxyt*2)
        return out.reshape(out.shape[0], -1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    joint = len(args.re_values) > 1
    n_epochs = _EPOCHS_PER_RE.get(args.re_values[0], 100) if args.n_epochs == -1 else args.n_epochs

    RUN_NAME = args.run_name or ("deeponet_joint_navier_stokes_v1" if joint else f"deeponet_navier_stokes_re{args.re_values[0]}_v1")
    RUN_DIR = os.path.join(args.results_dir, RUN_NAME)
    os.makedirs(RUN_DIR, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={DEVICE}  epochs={n_epochs}  run={RUN_NAME}")

    # ========== DATA ==========
    entries = [(re, _data_path(re, args.data_dir)) for re in args.re_values if os.path.exists(_data_path(re, args.data_dir))]
    if not entries:
        raise RuntimeError("No data files found")

    s_np, u0_np, kappa_np, xy_np, Nx, Ny, Nt = load_ns_stacked(entries, n_samples=args.n_samples)
    Nxy = Nx * Ny
    Nxyt = Nt * Nxy * 2

    train_idx, test_idx = [], []
    for re_val in sorted(np.unique(kappa_np)):
        idx = np.where(kappa_np == re_val)[0]
        n_test = min(args.n_test_per_re, len(idx))
        train_idx.append(idx[:-n_test])
        test_idx.append(idx[-n_test:])
    train_idx = np.concatenate(train_idx)
    test_idx  = np.concatenate(test_idx)

    s_train = torch.from_numpy(s_np[train_idx].astype(np.float32)).to(DEVICE)
    s_test = torch.from_numpy(s_np[test_idx].astype(np.float32)).to(DEVICE)
    u0_train = torch.from_numpy(u0_np[train_idx].astype(np.float32)).to(DEVICE)
    u0_test = torch.from_numpy(u0_np[test_idx].astype(np.float32)).to(DEVICE)
    kappa_train = torch.from_numpy(kappa_np[train_idx].astype(np.float32)).reshape(-1, 1).to(DEVICE)
    kappa_test = torch.from_numpy(kappa_np[test_idx].astype(np.float32)).reshape(-1, 1).to(DEVICE)

    N_train, N_test = len(train_idx), len(test_idx)
    m = u0_np.shape[1]  # Nxy * 2 (two velocity components)

    print(f"N_train={N_train}  N_test={N_test}  Nx={Nx}  Ny={Ny}  Nt={Nt}  m={m}  Nxyt={Nxyt}")

    # Create spatial-temporal grid: (x, y, t)
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    t = np.linspace(0, 1, Nt)
    xg, yg, tg = np.meshgrid(x, y, t, indexing='ij')
    xyt = np.stack([xg.flatten(), yg.flatten(), tg.flatten()], axis=1).astype(np.float32)
    xyt = torch.from_numpy(xyt).to(DEVICE)

    # ========== MODEL ==========
    d = args.hidden_dim
    branch = BranchNet(m, d, args.hidden_dim, args.n_layers).to(DEVICE)
    trunk = TrunkNet(d, args.hidden_dim, args.n_layers).to(DEVICE)
    model = DeepONet(branch, trunk, d).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    # ========== TRAINING ==========
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(n_epochs):
        model.train()
        idx_b = np.random.choice(N_train, min(args.batch_size, N_train), replace=False)
        u0_b = u0_train[idx_b]
        k_b = kappa_train[idx_b]
        s_b = s_train[idx_b]

        pred = model(u0_b, k_b, xyt)
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
        pred_test = model(u0_test, kappa_test, xyt).cpu().numpy()
    err_test = rel_l2(s_test.cpu().numpy(), pred_test)

    idx_sample = np.random.choice(N_train, min(2000, N_train), replace=False)
    with torch.no_grad():
        pred_train = model(u0_train[idx_sample], kappa_train[idx_sample], xyt).cpu().numpy()
    err_train = rel_l2(s_train[idx_sample].cpu().numpy(), pred_train)

    print(f"Train | mean={err_train.mean():.4f}  median={np.median(err_train):.4f}  std={err_train.std():.4f}")
    print(f"Test  | mean={err_test.mean():.4f}  median={np.median(err_test):.4f}  std={err_test.std():.4f}  p95={np.percentile(err_test, 95):.4f}")

    # ========== METRICS ==========
    metrics = {
        "run_name": RUN_NAME,
        "n_params": n_params,
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

    # Cross-Re evaluation
    cross_re_metrics = {}
    for re_eval in [100, 1000, 3600, 10000]:
        fpath = _data_path(re_eval, args.data_dir)
        if not os.path.exists(fpath):
            continue
        s_re_all, u0_re_all, k_re_all, _, _, _, _ = load_ns_stacked([(re_eval, fpath)], n_samples=args.n_samples)
        s_re = torch.from_numpy(s_re_all[args.n_samples - args.n_test_per_re:].astype(np.float32)).to(DEVICE)
        u0_re = torch.from_numpy(u0_re_all[args.n_samples - args.n_test_per_re:].astype(np.float32)).to(DEVICE)
        k_re = torch.full((len(u0_re), 1), float(re_eval), dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            pred_re = model(u0_re, k_re, xyt).cpu().numpy()
        err_re = rel_l2(s_re.cpu().numpy(), pred_re)
        cross_re_metrics[re_eval] = {
            "mean": float(err_re.mean()),
            "median": float(np.median(err_re)),
            "std": float(err_re.std()),
            "p95": float(np.percentile(err_re, 95)),
        }
        print(f"  Re={re_eval}: mean={err_re.mean():.4f}")

    metrics["cross_re"] = cross_re_metrics

    # --- Visualization: Error distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(err_test, bins=40, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(err_test.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {err_test.mean():.4f}")
    ax.axvline(np.median(err_test), color="orange", linestyle="--", linewidth=2, label=f"Median: {np.median(err_test):.4f}")
    ax.set_xlabel("Relative L2 Error")
    ax.set_ylabel("Count")
    ax.set_title("Test Error Distribution (2D Navier-Stokes DeepONet)", fontweight="bold")
    ax.legend(framealpha=0.7)
    ax.grid(True, ls="--", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "error_dist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Visualization: Sample reconstructions (2D spatial fields) ---
    s_test_np = s_test.cpu().numpy()
    n_viz = min(args.n_viz, len(test_idx))
    n_timesteps_show = 3
    time_indices = np.linspace(0, Nt - 1, n_timesteps_show, dtype=int)

    for sample_idx in range(n_viz):
        test_sample_idx = sample_idx
        s_true_sample = s_test_np[test_sample_idx]  # shape: (Nt * Nx * Ny * 2,)
        s_pred_sample = pred_test[test_sample_idx]

        # Reshape to (Nt, Nx, Ny, 2)
        s_true_reshaped = s_true_sample.reshape(Nt, Nx, Ny, 2)
        s_pred_reshaped = s_pred_sample.reshape(Nt, Nx, Ny, 2)

        # Get Reynolds number for this sample
        re_val = int(kappa_test[test_sample_idx, 0].item())

        # Create figure: show u and v components at select timesteps
        fig, axes = plt.subplots(2, n_timesteps_show * 2, figsize=(n_timesteps_show * 4, 6))
        fig.suptitle(f"Sample {sample_idx}: 2D Velocity Field Reconstruction (Re={re_val})",
                     fontweight="bold", fontsize=12)

        for t_i, t_idx in enumerate(time_indices):
            # U-component (velocity in x-direction)
            u_true = s_true_reshaped[t_idx, :, :, 0]
            u_pred = s_pred_reshaped[t_idx, :, :, 0]

            im0 = axes[0, t_i * 2].imshow(u_true, cmap="RdBu_r", origin="lower")
            axes[0, t_i * 2].set_title(f"u_true (t={t_idx})")
            axes[0, t_i * 2].set_xticks([])
            axes[0, t_i * 2].set_yticks([])
            plt.colorbar(im0, ax=axes[0, t_i * 2])

            im1 = axes[0, t_i * 2 + 1].imshow(u_pred, cmap="RdBu_r", origin="lower")
            axes[0, t_i * 2 + 1].set_title(f"u_pred (t={t_idx})")
            axes[0, t_i * 2 + 1].set_xticks([])
            axes[0, t_i * 2 + 1].set_yticks([])
            plt.colorbar(im1, ax=axes[0, t_i * 2 + 1])

        for t_i, t_idx in enumerate(time_indices):
            # V-component (velocity in y-direction)
            v_true = s_true_reshaped[t_idx, :, :, 1]
            v_pred = s_pred_reshaped[t_idx, :, :, 1]

            im2 = axes[1, t_i * 2].imshow(v_true, cmap="RdBu_r", origin="lower")
            axes[1, t_i * 2].set_title(f"v_true (t={t_idx})")
            axes[1, t_i * 2].set_xticks([])
            axes[1, t_i * 2].set_yticks([])
            plt.colorbar(im2, ax=axes[1, t_i * 2])

            im3 = axes[1, t_i * 2 + 1].imshow(v_pred, cmap="RdBu_r", origin="lower")
            axes[1, t_i * 2 + 1].set_title(f"v_pred (t={t_idx})")
            axes[1, t_i * 2 + 1].set_xticks([])
            axes[1, t_i * 2 + 1].set_yticks([])
            plt.colorbar(im3, ax=axes[1, t_i * 2 + 1])

        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, f"reconstruction_sample{sample_idx}.png"), dpi=100, bbox_inches="tight")
        plt.close()

    # --- 3D Visualization: Velocity magnitude surface plots ---
    from mpl_toolkits.mplot3d import Axes3D

    for sample_idx in range(n_viz):
        test_sample_idx = sample_idx
        s_true_sample = s_test_np[test_sample_idx].reshape(Nt, Nx, Ny, 2)
        s_pred_sample = pred_test[test_sample_idx].reshape(Nt, Nx, Ny, 2)

        # Compute velocity magnitude
        mag_true = np.sqrt(s_true_sample[..., 0]**2 + s_true_sample[..., 1]**2)
        mag_pred = np.sqrt(s_pred_sample[..., 0]**2 + s_pred_sample[..., 1]**2)

        re_val = int(kappa_test[test_sample_idx, 0].item())

        # Create 3D surface plots for selected timesteps
        fig = plt.figure(figsize=(15, 10))
        x_mesh = np.arange(Nx)
        y_mesh = np.arange(Ny)
        X, Y = np.meshgrid(x_mesh, y_mesh, indexing='ij')

        for t_i, t_idx in enumerate(time_indices):
            # True field
            ax_true = fig.add_subplot(2, n_timesteps_show, t_i + 1, projection='3d')
            Z_true = mag_true[t_idx]
            surf = ax_true.plot_surface(X, Y, Z_true, cmap="viridis", alpha=0.9, linewidth=0)
            ax_true.set_title(f"True Mag (t={t_idx})")
            ax_true.set_zlim(0, mag_true.max() * 1.1)
            ax_true.set_xlabel("x"); ax_true.set_ylabel("y")
            fig.colorbar(surf, ax=ax_true, shrink=0.5)

            # Predicted field
            ax_pred = fig.add_subplot(2, n_timesteps_show, n_timesteps_show + t_i + 1, projection='3d')
            Z_pred = mag_pred[t_idx]
            surf = ax_pred.plot_surface(X, Y, Z_pred, cmap="viridis", alpha=0.9, linewidth=0)
            ax_pred.set_title(f"Pred Mag (t={t_idx})")
            ax_pred.set_zlim(0, mag_true.max() * 1.1)
            ax_pred.set_xlabel("x"); ax_pred.set_ylabel("y")
            fig.colorbar(surf, ax=ax_pred, shrink=0.5)

        fig.suptitle(f"Sample {sample_idx}: 3D Velocity Magnitude (Re={re_val})", fontweight="bold", fontsize=12)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.05, hspace=0.4, wspace=0.3)
        plt.savefig(os.path.join(RUN_DIR, f"reconstruction_3d_sample{sample_idx}.png"), dpi=100, bbox_inches="tight")
        plt.close()

    print(f"Generated {n_viz} reconstruction visualization(s) + {n_viz} 3D visualization(s)")

    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {RUN_DIR}")


if __name__ == "__main__":
    main()
