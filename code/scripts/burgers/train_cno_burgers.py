#!/usr/bin/env python
"""Train CNO2d on 1D Burgers — specialist (one nu) or joint (multiple nu).

CNO2d treats the space-time domain (x, t) as a 2D grid resized to (SIZE x SIZE).

Single --nu_values  → specialist: cross-nu generalization eval, cross_nu.png.
Multiple --nu_values → joint: per-nu error bar, error_per_nu.png.

Input:  (N, 4, SIZE, SIZE) — [u0, log10(nu) norm, x-coord, t-coord]
Output: (N, 1, SIZE, SIZE)

Setup (run once on server):
    git clone https://github.com/camlab-ethz/ConvolutionalNeuralOperator ~/CNO
    # CNO_PATH is set automatically below; override with env var CNO_PATH if needed.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.datasets import load_stacked, measure_inference_time

# --- Locate CNO2d_simplified ---
_CNO_SEARCH = [
    os.environ.get("CNO_PATH", ""),
    str(_PROJECT_ROOT.parent / "CNO" / "CNO2d_simplified"),   # project/CNO/
    str(_PROJECT_ROOT / "CNO2d_simplified"),                   # code/CNO2d_simplified/
    str(pathlib.Path.home() / "CNO" / "CNO2d_simplified"),
    str(pathlib.Path.home() / "ConvolutionalNeuralOperator" / "CNO2d_simplified"),
]
for _p in _CNO_SEARCH:
    if _p and pathlib.Path(_p).exists():
        sys.path.insert(0, _p)
        break
else:
    raise ImportError(
        "CNO2d_simplified not found.\n"
        "Run: git clone https://github.com/camlab-ethz/ConvolutionalNeuralOperator ~/CNO"
    )
from CNO2d import CNO2d


_ALL_NU = [0.001, 0.01, 0.1, 1.0]
_LOG10_NU_MIN = np.log10(min(_ALL_NU))   # -3
_LOG10_NU_MAX = np.log10(max(_ALL_NU))   #  0


def rel_l2(true, pred):
    """true, pred: (N, D) — returns (N,) per-sample relative L2."""
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nu_values",     type=float, nargs="+", required=True)
    p.add_argument("--run_name",      type=str,   default=None)
    p.add_argument("--results_dir",   type=str,   default=str(_PROJECT_ROOT / "TEMPO_results" / "burgers"))
    p.add_argument("--n_samples",     type=int,   default=9500)
    p.add_argument("--n_test_per_nu", type=int,   default=1000)
    p.add_argument("--data_dir",      type=str,   default=os.path.expanduser("~/data/1D/Burgers/Train"))
    # Grid
    p.add_argument("--size",          type=int,   default=128,
                   help="Square grid size after resizing (power of 2 recommended)")
    # CNO architecture
    p.add_argument("--n_layers",      type=int,   default=4,
                   help="Number of up/downsampling blocks")
    p.add_argument("--n_res",         type=int,   default=4,
                   help="Residual blocks per encoder/decoder level")
    p.add_argument("--n_res_neck",    type=int,   default=4,
                   help="Residual blocks at bottleneck")
    p.add_argument("--channel_multiplier", type=int, default=16)
    p.add_argument("--use_bn",        action="store_true", default=False)
    # Training
    p.add_argument("--n_epochs",      type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=5e-4)
    p.add_argument("--lr_step",       type=int,   default=40)
    p.add_argument("--lr_gamma",      type=float, default=0.5)
    p.add_argument("--log_every",     type=int,   default=5)
    # Misc
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--n_viz",         type=int,   default=3)
    return p.parse_args()


def build_cno_input(u0_np, kappa_np, x_np, t_np, Nx, Nt, size):
    """Build CNO input (N, 4, size, size) — 4 channels, resized to square.

    Channel 0: u0 broadcast along t
    Channel 1: log10(nu) normalised to [-1, 1]
    Channel 2: x-coordinate normalised to [-1, 1]
    Channel 3: t-coordinate normalised to [-1, 1]
    """
    N = len(u0_np)
    xx = np.tile(x_np[:, None], (1, Nt)).astype(np.float32)   # (Nx, Nt)
    tt = np.tile(t_np[None, :], (Nx, 1)).astype(np.float32)   # (Nx, Nt)
    xx = (xx - xx.min()) / (xx.max() - xx.min()) * 2 - 1
    tt = (tt - tt.min()) / (tt.max() - tt.min()) * 2 - 1

    u0_exp = np.repeat(u0_np[:, :, None], Nt, axis=2).astype(np.float32)         # (N, Nx, Nt)
    xx_b   = np.broadcast_to(xx[None], (N, Nx, Nt)).copy().astype(np.float32)
    tt_b   = np.broadcast_to(tt[None], (N, Nx, Nt)).copy().astype(np.float32)

    log_nu  = np.log10(kappa_np.astype(np.float64)).astype(np.float32)
    norm_nu = (log_nu - _LOG10_NU_MIN) / (_LOG10_NU_MAX - _LOG10_NU_MIN) * 2 - 1
    nu_ch   = np.broadcast_to(norm_nu[:, None, None], (N, Nx, Nt)).copy().astype(np.float32)

    X = np.stack([u0_exp, nu_ch, xx_b, tt_b], axis=1)  # (N, 4, Nx, Nt)

    # Resize (Nx, Nt) → (size, size)
    X_resized = F.interpolate(
        torch.tensor(X), size=(size, size), mode="bilinear", align_corners=False
    )
    return X_resized.numpy()   # (N, 4, size, size)


def resize_targets(s_np, Nt, Nx, size):
    """s_np: (N, Nt*Nx) → (N, 1, size, size) via bilinear resize."""
    N = len(s_np)
    # s convention: row-major (Nt, Nx) → reshape gives (N, Nt, Nx)
    # Transpose to (N, Nx, Nt) for consistent (spatial, time) layout
    s_grid = s_np.reshape(N, Nt, Nx).transpose(0, 2, 1).astype(np.float32)  # (N, Nx, Nt)
    s_t    = torch.tensor(s_grid[:, None, :, :])                              # (N, 1, Nx, Nt)
    s_resized = F.interpolate(s_t, size=(size, size), mode="bilinear", align_corners=False)
    return s_resized.numpy()   # (N, 1, size, size)


def predict_batched(model, X_tensor, device, batch_size=64):
    """X_tensor: (N, 4, size, size) → returns (N, size*size) on CPU."""
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            out = model(X_tensor[i:i + batch_size].to(device))  # (B, 1, size, size)
            parts.append(out[:, 0].reshape(len(out), -1).cpu())
    return torch.cat(parts, dim=0).numpy()


def main():
    args  = parse_args()
    SIZE  = args.size

    nu_values = args.nu_values
    is_joint  = len(nu_values) > 1

    if args.run_name:
        RUN_NAME = args.run_name
    elif is_joint:
        RUN_NAME = "cno_joint_burgers_v1"
    else:
        RUN_NAME = f"cno_burgers_nu{nu_values[0]}_v1"

    RUN_DIR = os.path.join(args.results_dir, RUN_NAME)
    os.makedirs(RUN_DIR, exist_ok=True)

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    print(f"device={DEVICE}  size={SIZE}x{SIZE}  run_dir={os.path.abspath(RUN_DIR)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    C0, C1, C2 = plt.cm.tab10(0), plt.cm.tab10(1), plt.cm.tab10(2)

    # --- Data loading ---
    data_dir  = pathlib.Path(args.data_dir)
    local_dir = _PROJECT_ROOT / "data"

    def _find_path(nu):
        local  = local_dir / f"Burgers_Nu{nu}.hdf5"
        server = data_dir  / f"1D_Burgers_Sols_Nu{nu}.hdf5"
        return str(local) if local.exists() else str(server)

    entries = [(nu, _find_path(nu)) for nu in nu_values]
    s, kappa, x_np, t_np, Nx, Nt = load_stacked(entries, n_samples=args.n_samples)

    n_nu  = len(nu_values)
    N_per = args.n_samples
    n_test = args.n_test_per_nu

    train_idx = np.concatenate([
        np.arange(i * N_per, (i + 1) * N_per - n_test) for i in range(n_nu)
    ])
    test_idx = np.concatenate([
        np.arange((i + 1) * N_per - n_test, (i + 1) * N_per) for i in range(n_nu)
    ])

    s_train     = s[train_idx]
    s_test      = s[test_idx]
    kappa_train = kappa[train_idx]
    kappa_test  = kappa[test_idx]
    del s, kappa

    u0_train = s_train[:, :Nx]
    u0_test  = s_test[:,  :Nx]

    N_train = len(train_idx)
    N_test  = len(test_idx)

    print(f"Building CNO inputs at {SIZE}x{SIZE} ...")
    X_train = torch.tensor(build_cno_input(u0_train, kappa_train, x_np, t_np, Nx, Nt, SIZE))
    X_test  = torch.tensor(build_cno_input(u0_test,  kappa_test,  x_np, t_np, Nx, Nt, SIZE))

    # Targets: (N, 1, SIZE, SIZE)
    Y_train = torch.tensor(resize_targets(s_train, Nt, Nx, SIZE))
    Y_test  = torch.tensor(resize_targets(s_test,  Nt, Nx, SIZE))

    # Flat targets for rel_l2: (N, SIZE*SIZE)
    s_train_flat = Y_train[:, 0].reshape(N_train, -1).numpy()
    s_test_flat  = Y_test[:,  0].reshape(N_test,  -1).numpy()

    print(f"CNO input: {tuple(X_train.shape)}  output: {tuple(Y_train.shape)}")

    # Pre-load to GPU if data fits
    _pin = False
    if DEVICE == "cuda":
        _data_gb = (X_train.numel() + Y_train.numel() + X_test.numel()) * 4 / 1e9
        _free_gb = torch.cuda.mem_get_info()[0] / 1e9
        if _data_gb < _free_gb * 0.45:
            print(f"Pre-loading {_data_gb:.1f} GB data to GPU")
            X_train = X_train.to(DEVICE)
            Y_train = Y_train.to(DEVICE)
            X_test  = X_test.to(DEVICE)
        else:
            print(f"Data {_data_gb:.1f} GB > {_free_gb*0.45:.1f} GB threshold, using pin_memory")
            _pin = True

    # --- Build model ---
    model = CNO2d(
        in_dim=4,
        out_dim=1,
        size=SIZE,
        N_layers=args.n_layers,
        N_res=args.n_res,
        N_res_neck=args.n_res_neck,
        channel_multiplier=args.channel_multiplier,
        use_bn=args.use_bn,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"CNO params: {n_params:,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)
    dl    = DataLoader(TensorDataset(X_train, Y_train),
                       batch_size=args.batch_size, shuffle=True,
                       pin_memory=_pin, num_workers=0)

    # --- Training ---
    mode_str = "joint" if is_joint else f"specialist nu={nu_values[0]}"
    print(f"=== Training CNO | {mode_str} | N_train={N_train} | epochs={args.n_epochs} ===")
    t0 = time.time()
    history_train, history_val = [], []

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=_pin)
            yb = yb.to(DEVICE, non_blocking=_pin)
            pred = model(xb)
            loss = F.l1_loss(pred, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item() * len(xb)
        sched.step()
        avg = total / N_train
        history_train.append(avg)

        elapsed = time.time() - t0
        if epoch % args.log_every == 0 or epoch == 1:
            pred_val = predict_batched(model, X_test, DEVICE)
            err_val  = rel_l2(s_test_flat, pred_val).mean()
            history_val.append((epoch, err_val))
            print(f"  epoch {epoch:4d} | train_l1={avg:.4e} | val_rel_l2={err_val:.4f} | {elapsed:.0f}s")
        else:
            print(f"  epoch {epoch:4d} | train_l1={avg:.4e} | {elapsed:.0f}s")

    print(f"  done: total time {time.time() - t0:.0f}s")

    # --- Training dynamics plot ---
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.semilogy(range(1, len(history_train) + 1), history_train, color=C0, lw=1.5, label="Train MSE")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE", color=C0)
    ax1.tick_params(axis="y", labelcolor=C0)
    ax1.grid(True, ls="--", alpha=0.25)
    ax1.spines[["top"]].set_visible(False)
    if history_val:
        ve, vl = zip(*history_val)
        ax2 = ax1.twinx()
        ax2.semilogy(ve, vl, color=C1, lw=1.5, ls="--", label="Val rel-L2")
        ax2.set_ylabel("Val rel-L2", color=C1)
        ax2.tick_params(axis="y", labelcolor=C1)
        ax2.spines[["top"]].set_visible(False)
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, framealpha=0.7)
    ax1.set_title("CNO training dynamics", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "training_dynamics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Final evaluation ---
    pred_train = predict_batched(model, X_train, DEVICE)
    pred_test  = predict_batched(model, X_test,  DEVICE)

    err_train = rel_l2(s_train_flat, pred_train)
    err_test  = rel_l2(s_test_flat,  pred_test)

    print(f"Train | mean={err_train.mean():.4f}  median={np.median(err_train):.4f}  std={err_train.std():.4f}")
    print(f"Test  | mean={err_test.mean():.4f}  median={np.median(err_test):.4f}  std={err_test.std():.4f}  p95={np.percentile(err_test, 95):.4f}")

    if is_joint:
        metrics: dict = {
            "run_name":       RUN_NAME,
            "n_params":       n_params,
            "n_train":        N_train,
            "n_test":         N_test,
            "overall_mean":   float(err_test.mean()),
            "overall_median": float(np.median(err_test)),
        }
        for i, nu in enumerate(nu_values):
            nu_sl = slice(i * n_test, (i + 1) * n_test)
            e_nu  = err_test[nu_sl]
            key   = f"nu{nu:.3f}"
            metrics[f"{key}_mean"]   = float(e_nu.mean())
            metrics[f"{key}_median"] = float(np.median(e_nu))
            metrics[f"{key}_std"]    = float(e_nu.std())
            print(f"  nu={nu:.3f}: mean={e_nu.mean():.4f}  median={np.median(e_nu):.4f}")

        nu_labels = [f"nu={nu:.3f}" for nu in nu_values]
        means_    = [metrics[f"nu{nu:.3f}_mean"]   for nu in nu_values]
        medians_  = [metrics[f"nu{nu:.3f}_median"] for nu in nu_values]
        x_pos     = np.arange(len(nu_labels))
        _, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x_pos - 0.2,  means_,   0.35, label="Mean",   color=C0, alpha=0.85, linewidth=0)
        ax.bar(x_pos + 0.15, medians_, 0.35, label="Median", color=C1, alpha=0.85, linewidth=0)
        ax.set_xticks(x_pos); ax.set_xticklabels(nu_labels)
        ax.set_ylabel("Relative L2 error")
        ax.set_title("CNO - per-nu test errors (joint)", fontweight="bold")
        ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, "error_per_nu.png"), dpi=150, bbox_inches="tight")
        plt.close()

    else:
        TRAIN_NU = nu_values[0]
        metrics: dict = {
            "run_name":     RUN_NAME,
            "n_params":     n_params,
            "n_train":      N_train,
            "n_test":       N_test,
            "train_mean":   float(err_train.mean()),
            "train_median": float(np.median(err_train)),
            "train_std":    float(err_train.std()),
            "test_mean":    float(err_test.mean()),
            "test_median":  float(np.median(err_test)),
            "test_std":     float(err_test.std()),
            "test_p95":     float(np.percentile(err_test, 95)),
        }

        cross_nu_metrics: dict = {}
        for nu in _ALL_NU:
            local  = local_dir / f"Burgers_Nu{nu}.hdf5"
            server = data_dir  / f"1D_Burgers_Sols_Nu{nu}.hdf5"
            fpath  = local if local.exists() else server
            if not fpath.exists():
                print(f"  nu={nu:.3f}: file not found, skipping")
                continue
            with h5py.File(fpath, "r") as f:
                raw_nu = f["tensor"][args.n_samples - n_test : args.n_samples]
                if raw_nu.ndim == 4:
                    raw_nu = raw_nu[..., 0]
            u0_nu   = raw_nu[:, 0, :].astype(np.float32)
            s_nu_np = raw_nu.reshape(n_test, Nt, Nx).transpose(0, 2, 1)  # (N, Nx, Nt)
            s_nu_r  = F.interpolate(
                torch.tensor(s_nu_np[:, None, :, :]),
                size=(SIZE, SIZE), mode="bilinear", align_corners=False
            )
            s_nu_flat = s_nu_r[:, 0].reshape(n_test, -1).numpy()
            kap_nu  = np.full(n_test, nu, dtype=np.float32)
            X_nu    = torch.tensor(build_cno_input(u0_nu, kap_nu, x_np, t_np, Nx, Nt, SIZE))
            pred_nu = predict_batched(model, X_nu, DEVICE)
            err_nu  = rel_l2(s_nu_flat, pred_nu)
            cross_nu_metrics[nu] = {
                "mean":   float(err_nu.mean()),
                "median": float(np.median(err_nu)),
                "std":    float(err_nu.std()),
                "p95":    float(np.percentile(err_nu, 95)),
            }
            tag = " (trained)" if nu == TRAIN_NU else ""
            print(f"  nu={nu:.3f}{tag}: mean={err_nu.mean():.4f}  median={np.median(err_nu):.4f}")

        metrics["cross_nu"] = cross_nu_metrics

        if cross_nu_metrics:
            nu_keys   = list(cross_nu_metrics.keys())
            nu_labels = [f"nu={nu:.3f}" for nu in nu_keys]
            means_    = [cross_nu_metrics[nu]["mean"]   for nu in nu_keys]
            medians_  = [cross_nu_metrics[nu]["median"] for nu in nu_keys]
            x_pos     = np.arange(len(nu_labels))
            _, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x_pos - 0.2,  means_,   0.35, label="Mean",   color=C0, alpha=0.85, linewidth=0)
            ax.bar(x_pos + 0.15, medians_, 0.35, label="Median", color=C1, alpha=0.85, linewidth=0)
            if TRAIN_NU in nu_keys:
                ax.axvline(nu_keys.index(TRAIN_NU), color="gray", ls="--", lw=1.2,
                           alpha=0.7, label="Trained on")
            ax.set_xticks(x_pos); ax.set_xticklabels(nu_labels)
            ax.set_ylabel("Relative L2 error")
            ax.set_title("CNO - cross-nu generalization", fontweight="bold")
            ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25, axis="y")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(RUN_DIR, "cross_nu.png"), dpi=150, bbox_inches="tight")
            plt.close()

    # --- Reconstruction examples (from test set of first nu) ---
    rng  = np.random.default_rng(args.seed)
    idxs = rng.choice(n_test, size=args.n_viz, replace=False)

    fig, axes = plt.subplots(args.n_viz, 3, figsize=(14, 3 * args.n_viz))
    if args.n_viz == 1:
        axes = axes[None, :]
    for row, idx in enumerate(idxs):
        true   = s_test_flat[idx].reshape(SIZE, SIZE)     # (SIZE, SIZE)
        pred_i = predict_batched(model, X_test[idx:idx + 1], DEVICE).reshape(SIZE, SIZE)
        err_i  = np.abs(true - pred_i)
        vmax   = np.abs(true).max()
        rl2    = float(np.linalg.norm(true - pred_i) / np.linalg.norm(true))
        for col, (arr, title, cmap, vmin, vm) in enumerate([
            (true,   "Ground Truth", "RdBu_r",  -vmax, vmax),
            (pred_i, "CNO",          "RdBu_r",  -vmax, vmax),
            (err_i,  "Abs. Error",   "Oranges",     0, err_i.max()),
        ]):
            ax = axes[row, col]
            im = ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap,
                           extent=[x_np.min(), x_np.max(), t_np.min(), t_np.max()],
                           vmin=vmin, vmax=vm)
            if row == 0:
                ax.set_title(title, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"t   (rel L2={rl2:.3f})")
            if row == args.n_viz - 1:
                ax.set_xlabel("x")
            ax.spines[["top", "right"]].set_visible(False)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("CNO: reconstruction examples", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "cno_reconstruction.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Error distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.hist(err_test, bins=30, color=C0, alpha=0.8, linewidth=0)
    ax.axvline(err_test.mean(),     color=C1, ls="--", lw=1.5,
               label=f"Mean {err_test.mean():.4f}")
    ax.axvline(np.median(err_test), color=C2, ls="--", lw=1.5,
               label=f"Median {np.median(err_test):.4f}")
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
    plt.suptitle("CNO - test set errors", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "cno_err_dist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Checkpoint ---
    torch.save({
        "model_state": model.state_dict(),
        "metrics":     metrics,
        "run_name":    RUN_NAME,
        "hparams":     vars(args),
    }, os.path.join(RUN_DIR, "model.pt"))

    # Inference time
    _inf_ms = measure_inference_time(
        lambda: predict_batched(model, X_test, DEVICE),
        device=DEVICE
    )
    metrics["inference_ms_total"] = _inf_ms
    metrics["inference_ms_per_sample"] = _inf_ms / len(X_test)

    metrics["hparams"] = vars(args)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {os.path.abspath(RUN_DIR)}")


if __name__ == "__main__":
    main()
